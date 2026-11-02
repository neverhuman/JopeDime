#!/usr/bin/env python3
"""Versioned SCQ dispatcher for the frozen Smart-v3 CPU development run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
import tomllib
from pathlib import Path
from typing import Any


if __package__:
    from .smart_v3_cpu_scq_config import (
        ACTIVE_STATES,
        CAMPAIGN_DIR,
        CHECKPOINT_SHA256,
        DATASETS,
        DATASET_OWNERSHIP,
        DEFAULT_POLL_SECONDS,
        DISPATCH_VERSION,
        EVOLUTION_JOB_COUNT,
        INFRASTRUCTURE_FAILURES,
        JOB_COUNT,
        NODES,
        PARENT_RUN_ID,
        PRIOR_RUN_ID,
        PROJECT,
        PROGRESS_SCHEMA,
        RUN_ID,
        SCQ,
        SHARDS,
        dataset_owners,
    )
    from .smart_v3_cpu_scq_inputs import (
        assert_cluster_contract, atomic_json, deterministic_tar, immutable_copy,
        marker, prepare_run, read_json, scoring_files, scq_json, sha256_file,
        source_files, training_files,
    )
    from .smart_v3_cpu_scq_specs import (
        check_spec,
        evolution_command,
        input_pin,
        job_id,
        load_progress,
        new_progress,
        progress_path,
        render_evolution_specs,
        render_score_spec,
        render_spec,
        scoring_command,
        unique_file,
        write_spec,
    )
else:
    from smart_v3_cpu_scq_config import (
        ACTIVE_STATES,
        CAMPAIGN_DIR,
        CHECKPOINT_SHA256,
        DATASETS,
        DATASET_OWNERSHIP,
        DEFAULT_POLL_SECONDS,
        DISPATCH_VERSION,
        EVOLUTION_JOB_COUNT,
        INFRASTRUCTURE_FAILURES,
        JOB_COUNT,
        NODES,
        PARENT_RUN_ID,
        PRIOR_RUN_ID,
        PROJECT,
        PROGRESS_SCHEMA,
        RUN_ID,
        SCQ,
        SHARDS,
        dataset_owners,
    )
    from smart_v3_cpu_scq_inputs import (
        assert_cluster_contract, atomic_json, deterministic_tar, immutable_copy,
        marker, prepare_run, read_json, scoring_files, scq_json, sha256_file,
        source_files, training_files,
    )
    from smart_v3_cpu_scq_specs import (
        check_spec,
        evolution_command,
        input_pin,
        job_id,
        load_progress,
        new_progress,
        progress_path,
        render_evolution_specs,
        render_score_spec,
        render_spec,
        scoring_command,
        unique_file,
        write_spec,
    )


def ensure_attempt(
    progress: dict[str, Any], logical: str, spec: Path, state: str
) -> dict[str, Any]:
    job = progress["jobs"][logical]
    if job["attempts"]:
        return job["attempts"][-1]
    attempt = {
        "job_id": job_id(logical, 0),
        "retry": 0,
        "spec_path": str(spec),
        "state": state,
    }
    job["attempts"].append(attempt)
    job["state"] = state
    return attempt


def submission_record_path(output: Path, job_id_value: str) -> Path:
    return output / "submissions" / f"{job_id_value}.json"


def spec_sha256(path: Path) -> str:
    return sha256_file(path)


def write_submission_receipt(
    output: Path,
    attempt: dict[str, Any],
    payload: dict[str, Any],
) -> Path:
    spec_path = Path(attempt["spec_path"])
    receipt = {
        "schema": "SmartV3CpuSCQSubmission.v1",
        "job_id": attempt["job_id"],
        "retry": attempt["retry"],
        "spec_sha256": spec_sha256(spec_path),
        "response_sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "accepted": True,
    }
    path = submission_record_path(output, attempt["job_id"])
    if path.exists() and read_json(path) != receipt:
        raise RuntimeError(f"submission receipt drift: {path}")
    atomic_json(path, receipt)
    return path


def validate_submission_receipt(output: Path, attempt: dict[str, Any]) -> dict[str, Any] | None:
    path = submission_record_path(output, attempt["job_id"])
    if not path.is_file():
        return None
    receipt = read_json(path)
    if (
        receipt.get("schema") != "SmartV3CpuSCQSubmission.v1"
        or receipt.get("job_id") != attempt["job_id"]
        or receipt.get("retry") != attempt["retry"]
        or receipt.get("spec_sha256") != spec_sha256(Path(attempt["spec_path"]))
        or receipt.get("accepted") is not True
    ):
        raise RuntimeError(f"invalid submission receipt: {path}")
    return receipt


def shown_job(job_id_value: str) -> dict[str, Any] | None:
    completed = subprocess.run(
        [str(SCQ), "--json", "show", job_id_value],
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"SCQ show failed for {job_id_value}: "
            f"{completed.stdout}{completed.stderr}"
        )
    payload = json.loads(completed.stdout)
    job = payload.get("Job", {}).get("job")
    if job is not None and not isinstance(job, dict):
        raise RuntimeError(f"malformed SCQ show response for {job_id_value}")
    return job


def accepted_submission_vanished(attempt: dict[str, Any]) -> bool:
    try:
        payload = scq_json("ps")
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        return False
    jobs = payload.get("Jobs", {}).get("jobs", [])
    return all(row.get("id") != attempt["job_id"] for row in jobs)


def submit_attempt(output: Path, attempt: dict[str, Any]) -> bool:
    receipt = validate_submission_receipt(output, attempt)
    existing = shown_job(attempt["job_id"])
    if existing is not None:
        if receipt is None:
            raise RuntimeError(
                "refusing to adopt an existing SCQ ID without a validated "
                f"submission receipt: {attempt['job_id']}"
            )
        expected_node = attempt.get("node")
        if (
            expected_node is not None
            and existing.get("node") is not None
            and existing.get("node") != expected_node
        ):
            raise RuntimeError(
                f"SCQ placed {attempt['job_id']} on {existing.get('node')}, "
                f"expected {expected_node}"
            )
        attempt["submission_receipt_path"] = str(
            submission_record_path(output, attempt["job_id"])
        )
        attempt["state"] = existing["state"]
        return True
    if receipt is not None:
        attempt["submission_receipt_path"] = str(
            submission_record_path(output, attempt["job_id"])
        )
        attempt["state"] = "accepted_invisible"
        return True

    if attempt["state"] not in {"rendered", "ready", "capacity_wait"}:
        raise RuntimeError(
            f"SCQ ID {attempt['job_id']} is absent without a submission receipt"
        )
    completed = subprocess.run(
        [str(SCQ), "--json", "submit", attempt["spec_path"]],
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        message = (completed.stdout + completed.stderr).lower()
        if "launch-ready-capacity" in message or "capacity" in message:
            attempt["state"] = "capacity_wait"
            return False
        raise RuntimeError(
            f"SCQ submit failed for {attempt['job_id']}: {message}"
        )
    payload = json.loads(completed.stdout)
    submitted = payload.get("Submitted", {})
    if submitted.get("id") != attempt["job_id"]:
        raise RuntimeError("SCQ submission ID mismatch")
    attempt["submission_receipt_path"] = str(
        write_submission_receipt(output, attempt, payload)
    )
    attempt["state"] = "queued"
    return True


def fetch_attempt(output: Path, logical: str, attempt: dict[str, Any]) -> None:
    destination = output / "fetched" / attempt["job_id"]
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [str(SCQ), "fetch", attempt["job_id"], "--output", str(destination)],
            check=True,
        )
    attempt["fetched_path"] = str(destination)
    attempt["state"] = "completed"
    attempt["output_sha256"] = hashlib.sha256(
        "".join(
            f"{path.relative_to(destination)}\0{sha256_file(path)}\n"
            for path in sorted(destination.rglob("*"))
            if path.is_file()
        ).encode()
    ).hexdigest()


def add_retry(
    output: Path,
    logical: str,
    progress: dict[str, Any],
    manifest: dict[str, Any],
    training: Path,
    source: Path,
    reason: str,
) -> None:
    job = progress["jobs"][logical]
    if len(job["attempts"]) != 1:
        raise RuntimeError(f"retry limit exhausted for {logical}")
    if not (
        reason.startswith("infrastructure:")
        or reason.startswith("vanished:")
    ):
        raise RuntimeError(f"retry is not infrastructure-authorized for {logical}")
    retry = 1
    if logical.startswith("evolve-"):
        node = job["node"]
        project_root = Path(progress["project_root"])
        common = [
            input_pin(
                "release_binary",
                project_root / "target/release/jope-dime-regressor",
                "jope-dime-regressor",
                True,
            ),
            input_pin("training_workspace", training, "training-workspace.tar"),
            input_pin("source_bundle", source, "source-bundle.tar"),
        ]
        text = render_spec(
            job_id=job_id(logical, retry),
            node=node,
            command=evolution_command(node, job["dataset"]),
            inputs=common,
            output_name="evolution-results",
            output_source="evolution-results",
            cpus=4,
            ram_gb=16,
            manifest=manifest,
            source_sha256=sha256_file(source),
            attempt=retry,
        )
        spec = write_spec(output, logical, retry, text)
    else:
        raise RuntimeError("score retry is rendered after evolution inputs are reloaded")
    attempt = {
        "job_id": job_id(logical, retry),
        "node": job["node"],
        "retry": retry,
        "spec_path": str(spec),
        "state": check_spec(spec),
        "predecessor_job_id": job["attempts"][0]["job_id"],
        "retry_reason": reason,
    }
    job["attempts"].append(attempt)
    job["state"] = attempt["state"]


def reconcile_job(
    output: Path,
    logical: str,
    progress: dict[str, Any],
) -> str:
    job = progress["jobs"][logical]
    attempt = job["attempts"][-1]
    if attempt["state"] == "capacity_wait":
        attempt["state"] = check_spec(Path(attempt["spec_path"]))
        job["state"] = attempt["state"]
        if attempt["state"] == "capacity_wait":
            return "capacity_wait"
    if attempt["state"] in {"ready", "rendered"}:
        if not submit_attempt(output, attempt):
            job["state"] = "capacity_wait"
            return "capacity_wait"
    elif attempt["state"] == "accepted_invisible":
        pass
    shown = shown_job(attempt["job_id"])
    if shown is None:
        if validate_submission_receipt(output, attempt) is None:
            raise RuntimeError(
                f"SCQ returned an absent job without an accepted submission receipt: "
                f"{attempt['job_id']}"
            )
        attempt["job_null_observations"] = attempt.get("job_null_observations", 0) + 1
        attempt["state"] = "accepted_invisible"
        job["state"] = "accepted_invisible"
        if (
            attempt["job_null_observations"] >= 2
            and accepted_submission_vanished(attempt)
        ):
            attempt["failure_class"] = "vanished"
            attempt["retry_reason"] = "vanished: accepted SCQ record is absent"
            return "retry"
        return job["state"]
    if shown.get("node") is not None and shown.get("node") != job["node"]:
        raise RuntimeError(
            f"SCQ placed {attempt['job_id']} on {shown.get('node')}, "
            f"expected {job['node']}"
        )
    attempt["state"] = shown["state"]
    attempt["node"] = shown.get("node")
    job["state"] = shown["state"]
    if shown["state"] == "completed":
        fetch_attempt(output, logical, attempt)
        job["fetched_path"] = attempt["fetched_path"]
        job["state"] = "completed"
    elif shown["state"] in {"failed", "cancelled", "canceled"}:
        failure = shown.get("failure_class") or "unknown"
        if (
            shown["state"] == "failed"
            and failure in INFRASTRUCTURE_FAILURES
            and len(job["attempts"]) == 1
        ):
            attempt["failure_class"] = failure
            attempt["retry_reason"] = f"infrastructure: {failure}"
            return "retry"
        raise RuntimeError(
            f"{attempt['job_id']} stopped on non-retryable failure: "
            f"{failure}: {shown.get('error_reason') or shown.get('message')}"
        )
    elif shown["state"] not in ACTIVE_STATES:
        raise RuntimeError(f"unexpected SCQ state: {shown['state']}")
    return job["state"]


def install_score_results(project_root: Path, progress: dict[str, Any]) -> None:
    fetched = Path(progress["jobs"]["score-xbabe4"]["fetched_path"])
    root = project_root / "runs/train-only-studies" / RUN_ID
    for name in (
        "smart-v3-test-results.json",
        "smart-v3-test-results.csv",
        "smart-v3-test-results.md",
    ):
        immutable_copy(unique_file(fetched, name), root / name)


def validate_evolution_archive(output: Path, logical: str, job: dict[str, Any]) -> None:
    fetched = Path(job["fetched_path"])
    archive_path = unique_file(fetched, "evolution-results.tar")
    node_file = unique_file(fetched, "node.txt")
    observed_node = node_file.read_text().strip()
    if observed_node != job["node"]:
        raise RuntimeError(
            f"{logical} output was produced on {observed_node}, expected {job['node']}"
        )
    with tarfile.open(archive_path) as archive:
        names = archive.getnames()
    selector_names = [
        name for name in names if name.endswith("smart-v3-selector-complete.v1.json")
    ]
    if len(selector_names) != 1:
        raise RuntimeError(f"{logical} did not produce exactly one selector receipt")
    if job["dataset"] not in selector_names[0]:
        raise RuntimeError(f"{logical} selector receipt is bound to another dataset")
    forbidden = (
        "test.csv",
        "baselines.v2.json",
        "g20-extension",
        "score-smart-v3-redteam",
    )
    if any(any(token in name for token in forbidden) for name in names):
        raise RuntimeError(f"{logical} evolution package contains scoring material")


def validate_score_outputs(output: Path, progress: dict[str, Any]) -> None:
    fetched = Path(progress["jobs"]["score-xbabe4"]["fetched_path"])
    results = read_json(unique_file(fetched, "smart-v3-test-results.json"))
    if (
        results.get("schema") != "TrainOnlyRedteamSmartV3TestResults.v1"
        or len(results.get("test_result_rows", [])) != EVOLUTION_JOB_COUNT
        or results.get("development_only") is not True
        or results.get("confirmatory_valid") is not False
    ):
        raise RuntimeError("Smart-v3 scorer did not produce exactly ten development rows")


def validate_completion(progress: dict[str, Any]) -> None:
    if len(progress["jobs"]) != JOB_COUNT:
        raise RuntimeError("dispatcher must manage exactly eleven fixed jobs")
    evolution_nodes = {
        progress["jobs"][f"evolve-{dataset}"]["attempts"][-1].get("node")
        for dataset in DATASETS
        if progress["jobs"][f"evolve-{dataset}"]["state"] == "completed"
    }
    if evolution_nodes != set(NODES):
        raise RuntimeError("completed SCQ records do not cover all four evolution nodes")
    for node, dataset in dataset_owners():
        job = progress["jobs"][f"evolve-{dataset}"]
        if job["state"] != "completed" or job["node"] != node:
            raise RuntimeError(f"dataset {dataset} is not complete on {node}")
        validate_evolution_archive(output, f"evolve-{dataset}", job)


def run(
    project_root: Path, output: Path, poll_seconds: int, submit: bool
) -> Path:
    assert_cluster_contract()
    manifest, training, source = prepare_run(project_root, output)
    progress = load_progress(output)
    progress["project_root"] = str(project_root)
    specs = render_evolution_specs(
        project_root, output, manifest, training, source, progress
    )
    for (_, dataset), spec in zip(dataset_owners(), specs, strict=True):
        logical = f"evolve-{dataset}"
        state = check_spec(spec)
        ensure_attempt(progress, logical, spec, state)
    atomic_json(progress_path(output), progress)
    if not submit:
        return progress_path(output)

    while True:
        done = True
        for dataset in DATASETS:
            logical = f"evolve-{dataset}"
            if progress["jobs"][logical]["state"] == "completed":
                continue
            done = False
            state = reconcile_job(output, logical, progress)
            if state == "retry":
                reason = progress["jobs"][logical]["attempts"][-1].get(
                    "retry_reason", "infrastructure: recorded failure"
                )
                add_retry(output, logical, progress, manifest, training, source, reason)
        atomic_json(progress_path(output), progress)
        if done or all(
            progress["jobs"][f"evolve-{dataset}"]["state"] == "completed"
            for dataset in DATASETS
        ):
            break
        time.sleep(poll_seconds)

    validate_completion(progress)
    score_spec = render_score_spec(
        project_root, output, manifest, training, source, progress
    )
    score_job = progress["jobs"]["score-xbabe4"]
    ensure_attempt(progress, "score-xbabe4", score_spec, check_spec(score_spec))
    atomic_json(progress_path(output), progress)
    while score_job["state"] != "completed":
        state = reconcile_job(output, "score-xbabe4", progress)
        if state == "retry":
            if len(score_job["attempts"]) != 1:
                raise RuntimeError("score retry limit exhausted")
            score_spec = render_score_spec(
                project_root,
                output,
                manifest,
                training,
                source,
                progress,
                attempt=1,
            )
            retry = {
                "job_id": job_id("score-xbabe4", 1),
                "node": "xbabe4",
                "retry": 1,
                "spec_path": str(score_spec),
                "state": check_spec(score_spec),
                "predecessor_job_id": score_job["attempts"][0]["job_id"],
                "retry_reason": score_job["attempts"][-1].get(
                    "retry_reason", "infrastructure: recorded failure"
                ),
            }
            score_job["attempts"].append(retry)
            score_job["state"] = retry["state"]
        atomic_json(progress_path(output), progress)
        if score_job["state"] != "completed":
            time.sleep(poll_seconds)
    install_score_results(project_root, progress)
    validate_score_outputs(output, progress)
    validate_completion(progress)
    return progress_path(output)


def supersede_launch1() -> list[str]:
    jobs = scq_json("ps")["Jobs"]["jobs"]
    targets = [
        job["id"]
        for job in jobs
        if job["id"].endswith("-launch1")
        and job["id"].startswith(f"redteam-{PRIOR_RUN_ID}-")
        and job["state"] in ACTIVE_STATES
    ]
    for job_id_value in targets:
        subprocess.run([str(SCQ), "cancel", job_id_value], check=True)
    return targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("plan", "run", "status", "supersede-launch1")
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path("/home/ubuntu/jope-dime")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output = (
        args.output
        or project_root
        / "runs/train-only-studies"
        / RUN_ID
        / "scq"
        / CAMPAIGN_DIR
    ).resolve()
    if args.mode == "supersede-launch1":
        print(json.dumps(supersede_launch1()))
    elif args.mode == "status":
        print(json.dumps(read_json(progress_path(output)), indent=2, sort_keys=True))
    else:
        print(
            run(
                project_root,
                output,
                args.poll_seconds,
                submit=args.mode == "run",
            )
        )


if __name__ == "__main__":
    main()
