#!/usr/bin/env python3
"""Independent perpetual supervisors for queue, reserve, ingest, and evolution."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
from pathlib import Path

from test_exact_common import atomic_json, dynamic_queue_targets, sha256_file

ROOT = Path("/home/ubuntu/jope-flywheel/test_exact_gen0")
SCRIPTS = Path(__file__).resolve().parent
SCQ = Path("/home/ubuntu/smart_cluster/target/release/scq")
SCQD = Path("/home/ubuntu/smart_cluster/scripts/scqd.sh")
SERVICES = ("queue", "reserve", "ingestion", "controller", "watchdog")
INGEST_BATCH_LIMIT = 100
_wave_process: subprocess.Popen[str] | None = None
_status_process: subprocess.Popen[str] | None = None


def loaded_bundle(root: Path = ROOT) -> str | None:
    process_bundle = os.environ.get("JOPE_LOADED_BUNDLE_SHA256")
    if process_bundle:
        return process_bundle
    pointer = root / "runtime-bundles" / "active.json"
    if not pointer.is_file():
        return None
    value = json.loads(pointer.read_text())
    manifest = Path(value["bundle"])
    if (
        not manifest.is_file()
        or sha256_file(manifest) != value["bundle_manifest_sha256"]
    ):
        raise RuntimeError("active runtime bundle pointer failed verification")
    return str(value["bundle_sha256"])


def run(args: list[str], timeout: int = 3_600) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", f"/home/ubuntu/jope-prime/src:{SCRIPTS}")
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env=env,
    )


def wave_generation_state() -> dict | None:
    global _wave_process

    if _wave_process is None:
        return None
    returncode = _wave_process.poll()
    if returncode is None:
        temporary = (
            ROOT
            / "manifests"
            / f".strict-wave-*.tmp-{_wave_process.pid}"
        )
        return {
            "pending": True,
            "phase": "generating_wave",
            "pid": _wave_process.pid,
            "manifests_written": sum(
                1
                for path in temporary.parent.glob(temporary.name)
                for _ in path.glob("*.json")
            ),
        }
    stdout, stderr = _wave_process.communicate()
    _wave_process = None
    if returncode:
        raise RuntimeError((stderr or stdout).strip()[-4_000:])
    lines = [line for line in stdout.splitlines() if line.strip()]
    return {
        "pending": False,
        "phase": "wave_committed",
        **(json.loads(lines[-1]) if lines else {}),
    }


def start_wave_generation(proposals: str, cohort: str | None = None) -> dict:
    global _wave_process

    if _wave_process is not None:
        raise RuntimeError("wave generator is already active")
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", f"/home/ubuntu/jope-prime/src:{SCRIPTS}")
    command = [
            sys.executable,
            str(SCRIPTS / "test_exact_wave.py"),
            "--candidates-per-dataset",
            "256",
            "--max-shard",
            "100",
            "--proposals",
            proposals,
        ]
    if cohort is not None:
        command.extend(["--cohort", cohort])
    _wave_process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    return {
        "pending": True,
        "phase": "generating_wave",
        "pid": _wave_process.pid,
        "manifests_written": 0,
    }


def pending_micro_wave_request() -> dict | None:
    request_root = ROOT / "ops" / "micro-wave-requests"
    state_path = ROOT / "ops" / "micro-wave-request-state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    handled = set(state.get("handled_request_sha256s") or [])
    for path in reversed(sorted(request_root.glob("*.json"))):
        digest = sha256_file(path)
        if digest not in handled:
            return {
                **json.loads(path.read_text()),
                "request": str(path),
                "request_sha256": digest,
            }
    return None


def mark_micro_wave_request(request: dict, disposition: str) -> None:
    state_path = ROOT / "ops" / "micro-wave-request-state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    handled = list(state.get("handled_request_sha256s") or [])
    digest = request["request_sha256"]
    if digest not in handled:
        handled.append(digest)
    atomic_json(
        state_path,
        {
            "schema": "JopeMicroWaveRequestState.v1",
            "handled_request_sha256s": handled,
            "latest_request": request["request"],
            "latest_request_sha256": digest,
            "latest_disposition": disposition,
            "updated_at_ms": int(time.time() * 1000),
        },
    )


def refresh_status_async(root: Path) -> None:
    global _status_process

    if _status_process is not None:
        if _status_process.poll() is None:
            return
        _status_process = None
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", f"/home/ubuntu/jope-prime/src:{SCRIPTS}")
    _status_process = subprocess.Popen(
        [
            sys.executable,
            str(SCRIPTS / "test_exact_status.py"),
            "--root",
            str(root),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )


def parse_result(result: subprocess.CompletedProcess) -> dict:
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip()[-4_000:])
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else {}


def burn_targets(root: Path = ROOT) -> tuple[int, int, float]:
    state = root / "ops" / "backlog.sqlite"
    values = []
    if state.is_file():
        database = sqlite3.connect(state)
        try:
            exists = database.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='burn_samples'"
            ).fetchone()
            if exists:
                cutoff = int(time.time() * 1000) - 60 * 60_000
                values = [
                    float(row[0])
                    for row in database.execute(
                        "SELECT jobs_per_minute FROM burn_samples "
                        "WHERE observed_at_ms>=? ORDER BY observed_at_ms",
                        (cutoff,),
                    )
                ]
        finally:
            database.close()
    from test_exact_backlog import cluster_slot_inventory

    capacity = cluster_slot_inventory()
    return dynamic_queue_targets(values, capacity["effective_cluster_slots"])


def queue_cycle() -> dict:
    result = parse_result(
        run(
            [
                sys.executable,
                str(SCRIPTS / "test_exact_backlog.py"),
                "dynamic-refill",
            ],
            timeout=3_600,
        )
    )
    if result.get("needed") and not result.get("submitted"):
        result["alert"] = "queue_below_floor_without_launch_ready_reserve"
    return result


def ensure_current_proposals(cohort: Path | None = None) -> dict:
    """Submit or harvest proposals after the micro-cohort is frozen."""

    from test_exact_controller import _fetch_job_output, _scheduler_jobs
    from test_exact_ops import render_checkpoint_proposal_spec
    from test_exact_wave import checkpoint

    checkpoint_path, checkpoint_hash = checkpoint()
    cohort_hash = sha256_file(cohort) if cohort is not None else None
    pointer_path = ROOT / "proposals" / "current.json"
    if pointer_path.is_file():
        pointer = json.loads(pointer_path.read_text())
        proposal_path = Path(pointer.get("proposals", ""))
        if (
            pointer.get("checkpoint_sha256") == checkpoint_hash
            and pointer.get("cohort_sha256") == cohort_hash
            and proposal_path.is_file()
            and sha256_file(proposal_path) == pointer.get("proposals_sha256")
        ):
            return {
                "ready": True,
                "checkpoint_sha256": checkpoint_hash,
                "proposals": str(proposal_path),
                "proposals_sha256": pointer["proposals_sha256"],
            }
    legacy = ROOT / "proposals" / "dl-proposals.jsonl"
    legacy_summary = legacy.with_suffix(".summary.json")
    if cohort is None and legacy.is_file() and legacy_summary.is_file():
        summary = json.loads(legacy_summary.read_text())
        catalog_pointer = json.loads((ROOT / "catalog" / "current.json").read_text())
        catalog = json.loads(Path(catalog_pointer["catalog"]).read_text())
        expected_counts = {
            "regression": int(catalog["regression_count"]) * 64,
            "binary": int(catalog["binary_count"]) * 64,
        }
        if (
            summary.get("model_checkpoint_sha256") == checkpoint_hash
            and summary.get("catalog_sha256") == catalog_pointer["catalog_sha256"]
            and summary.get("counts") == expected_counts
            and sha256_file(legacy) == summary.get("output_sha256")
        ):
            archive = (
                ROOT
                / "proposals"
                / "by-checkpoint"
                / f"{checkpoint_hash}.jsonl"
            )
            archive.parent.mkdir(parents=True, exist_ok=True)
            if not archive.is_file():
                shutil.copy2(legacy, archive)
            if sha256_file(archive) != summary["output_sha256"]:
                raise RuntimeError("legacy proposal archive hash drift")
            atomic_json(
                pointer_path,
                {
                    "checkpoint_sha256": checkpoint_hash,
                    "proposals": str(archive),
                    "proposals_sha256": summary["output_sha256"],
                    "summary": str(legacy_summary),
                    "summary_sha256": sha256_file(legacy_summary),
                    "legacy_adopted": True,
                },
            )
            return {
                "ready": True,
                "checkpoint_sha256": checkpoint_hash,
                "proposals": str(archive),
                "proposals_sha256": summary["output_sha256"],
                "legacy_adopted": True,
            }
    work = (
        ROOT
        / "proposals"
        / "jobs"
        / checkpoint_hash
        / (cohort_hash or "legacy-full-catalog")
    )
    work.mkdir(parents=True, exist_ok=True)
    rendered = render_checkpoint_proposal_spec(
        work / "proposal.toml",
        checkpoint=checkpoint_path,
        split=None,
        seed=20260726,
        proposals_per_dataset=64,
        cohort=cohort,
    )
    state = _scheduler_jobs().get(rendered["job_id"], {}).get("state")
    if state in {"failed", "cancelled"}:
        raise RuntimeError(f"micro-cohort proposal job failed: {state}")
    if state is None:
        checked = run([str(SCQ), "check", rendered["spec"]], timeout=900)
        if checked.returncode:
            raise RuntimeError((checked.stderr or checked.stdout).strip())
        submitted = run([str(SCQ), "submit", rendered["spec"]], timeout=900)
        detail = (submitted.stderr or submitted.stdout).strip()
        if submitted.returncode and "already exists" not in detail.lower():
            raise RuntimeError(detail)
        return {
            "pending": True,
            "phase": "submitted_micro_cohort_proposals",
            "job_id": rendered["job_id"],
            "checkpoint_sha256": checkpoint_hash,
            "cohort_sha256": cohort_hash,
        }
    if state != "completed":
        return {
            "pending": True,
            "phase": "waiting_for_micro_cohort_proposals",
            "job_id": rendered["job_id"],
            "job_state": state,
            "checkpoint_sha256": checkpoint_hash,
            "cohort_sha256": cohort_hash,
        }
    output = _fetch_job_output(rendered["job_id"], work / "fetched")
    proposals = list(output.rglob("dl-proposals.jsonl"))
    summaries = list(output.rglob("dl-proposals.summary.json"))
    if len(proposals) != 1 or len(summaries) != 1:
        raise RuntimeError("micro-cohort proposal fetch is incomplete")
    summary = json.loads(summaries[0].read_text())
    if (
        summary.get("model_checkpoint_sha256") != checkpoint_hash
        or summary.get("cohort_sha256") != cohort_hash
        or summary.get("split") is not None
        or int(summary.get("proposals_per_dataset", 0)) != 64
        or sha256_file(proposals[0]) != summary.get("output_sha256")
    ):
        raise RuntimeError("micro-cohort proposal provenance drift")
    archive = (
        ROOT
        / "proposals"
        / "by-checkpoint"
        / f"{checkpoint_hash}-{cohort_hash or 'all'}.jsonl"
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        shutil.copy2(proposals[0], archive)
    if sha256_file(archive) != summary["output_sha256"]:
        raise RuntimeError("archived micro-cohort proposal hash drift")
    atomic_json(
        pointer_path,
        {
            "checkpoint_sha256": checkpoint_hash,
            "cohort_sha256": cohort_hash,
            "proposals": str(archive),
            "proposals_sha256": sha256_file(archive),
            "summary": str(summaries[0]),
            "summary_sha256": sha256_file(summaries[0]),
        },
    )
    return {
        "ready": True,
        "checkpoint_sha256": checkpoint_hash,
        "cohort_sha256": cohort_hash,
        "proposals": str(archive),
        "proposals_sha256": sha256_file(archive),
    }


def reserve_cycle() -> dict:
    from test_exact_backlog import (
        bounded_submission_count,
        prepare,
        ready_manifest_count,
        unclaimed_manifest_count,
    )
    from test_exact_wave import checkpoint, plan_micro_cohort

    queued_target, queued_hard_cap, p95 = burn_targets()
    required = queued_target
    ready_before = ready_manifest_count()
    unclaimed_before = unclaimed_manifest_count()
    prepared = 0
    if ready_before < required and unclaimed_before:
        prepared = len(prepare(bounded_submission_count(required - ready_before)))
    ready_mid = ready_manifest_count()
    unclaimed_mid = unclaimed_manifest_count()
    generation = wave_generation_state()
    request = pending_micro_wave_request()
    if request is not None:
        _, active_checkpoint = checkpoint()
        if request.get("checkpoint_sha256") != active_checkpoint:
            mark_micro_wave_request(request, "superseded_checkpoint")
            request = None
    can_generate = ready_mid + unclaimed_mid < required
    if generation is None and can_generate:
        micro = plan_micro_cohort()
        proposal = ensure_current_proposals(Path(micro["cohort"]))
        if proposal.get("ready"):
            _, checkpoint_hash = checkpoint()
            if proposal["checkpoint_sha256"] != checkpoint_hash:
                raise RuntimeError("proposal checkpoint changed before wave generation")
            generation = start_wave_generation(
                proposal["proposals"], micro["cohort"]
            )
            if request is not None:
                mark_micro_wave_request(request, "accepted")
        else:
            generation = proposal
    result = {
        "queued_target": queued_target,
        "queued_hard_cap": queued_hard_cap,
        "p95_jobs_per_minute": p95,
        "required_launch_ready": required,
        "ready_before": ready_before,
        "unclaimed_before": unclaimed_before,
        "generated": generation,
        "micro_wave_request": (
            None
            if request is None
            else {
                "request_sha256": request["request_sha256"],
                "disposition": (
                    "accepted"
                    if generation
                    and generation.get("phase") == "generating_wave"
                    else "deferred_launch_ready_horizon"
                ),
            }
        ),
        "prepared": prepared,
        "ready_after": ready_manifest_count(),
        "unclaimed_after": unclaimed_manifest_count(),
    }
    if result["ready_after"] < queued_target:
        result["alert"] = "launch_ready_reserve_below_one_horizon"
    return result


def ingestion_cycle() -> dict:
    result = parse_result(
        run(
            [
                sys.executable,
                str(SCRIPTS / "test_exact_ingest.py"),
                "reconcile",
                "--limit",
                str(INGEST_BATCH_LIMIT),
            ],
            timeout=3_600,
        )
    )
    status = ROOT / "ops" / "last-ingest.json"
    if status.is_file():
        age = time.time() - status.stat().st_mtime
        if age > 300:
            result["alert"] = "no_labels_for_five_minutes"
    return result


def controller_cycle() -> dict:
    from test_exact_evaluator import (
        active as active_evaluator,
        candidate as candidate_evaluator,
        promote_candidate,
    )

    active = active_evaluator()
    candidate = candidate_evaluator()
    canary = None
    if candidate["evaluator_sha256"] != active["evaluator_sha256"]:
        report = (
            ROOT
            / "canary"
            / f"{candidate['evaluator_sha256']}.candidate.json"
        )
        canary = parse_result(
            run(
                [
                    sys.executable,
                    str(SCRIPTS / "test_exact_canary.py"),
                    "--evaluator",
                    candidate["evaluator"],
                    "--report",
                    str(report),
                    "--timeout",
                    "900",
                ],
                timeout=1_200,
            )
        )
        active = promote_candidate(report)
    controller = parse_result(
        run(
            [sys.executable, str(SCRIPTS / "test_exact_controller.py")],
            timeout=3_600,
        )
    )
    return {
        "evaluator": {
            "active": active["evaluator_sha256"],
            "candidate_canary": canary,
        },
        "flywheel": controller,
    }


def watchdog_cycle(state: dict) -> dict:
    ping = run([str(SCQ), "ping"], timeout=5)
    if ping.returncode == 0:
        return {
            "consecutive_failures": 0,
            "recovered": False,
            "ping": "ok",
        }
    failures = int(state.get("consecutive_failures", 0)) + 1
    if failures < 3:
        return {
            "consecutive_failures": failures,
            "recovered": False,
            "ping": "failed",
            "detail": (ping.stderr or ping.stdout).strip()[-1_000:],
        }
    started = time.monotonic()
    recovery = run([str(SCQD), "start"], timeout=65)
    if recovery.returncode:
        raise RuntimeError((recovery.stderr or recovery.stdout).strip())
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if run([str(SCQ), "ping"], timeout=5).returncode == 0:
            adopted = parse_result(
                run(
                    [
                        sys.executable,
                        str(SCRIPTS / "test_exact_backlog.py"),
                        "adopt",
                    ],
                    timeout=900,
                )
            )
            return {
                "consecutive_failures": 0,
                "recovered": True,
                "recovery_seconds": time.monotonic() - started,
                "jobs_re_adopted": adopted,
                "existing_jobs_cancelled": 0,
            }
        time.sleep(1)
    raise RuntimeError("SCQ controller did not recover within 60 seconds")


def status_path(service: str, root: Path = ROOT) -> Path:
    return root / "ops" / "services" / f"{service}.json"


def service_state(service: str, root: Path = ROOT) -> dict:
    path = status_path(service, root)
    return json.loads(path.read_text()) if path.is_file() else {}


def record_starting(service: str, root: Path = ROOT) -> dict:
    prior = service_state(service, root)
    value = {
        **prior,
        "schema": "JopeSelfplayServiceState.v1",
        "service": service,
        "pid": os.getpid(),
        "loaded_bundle_sha256": loaded_bundle(root),
        "process_started_at_ms": int(time.time() * 1000),
        "last_success_at_ms": prior.get("last_success_at_ms"),
        "lifecycle": "starting",
        "error": None,
    }
    atomic_json(status_path(service, root), value)
    return value


def execute_cycle(service: str, root: Path = ROOT) -> dict:
    prior = service_state(service, root)
    if service == "queue":
        result = queue_cycle()
    elif service == "reserve":
        result = reserve_cycle()
    elif service == "ingestion":
        result = ingestion_cycle()
    elif service == "controller":
        result = controller_cycle()
    else:
        result = watchdog_cycle(prior.get("result") or {})
    timestamp = int(time.time() * 1000)
    value = {
        "schema": "JopeSelfplayServiceState.v1",
        "service": service,
        "pid": os.getpid(),
        "loaded_bundle_sha256": loaded_bundle(root),
        "last_attempt_at_ms": timestamp,
        "last_success_at_ms": timestamp,
        "lifecycle": "running",
        "error": None,
        "result": result,
    }
    atomic_json(status_path(service, root), value)
    error_path = root / "ops" / "services" / f"{service}.error.json"
    error_path.unlink(missing_ok=True)
    return value


def record_error(service: str, error: BaseException, root: Path = ROOT) -> dict:
    prior = service_state(service, root)
    value = {
        **prior,
        "schema": "JopeSelfplayServiceState.v1",
        "service": service,
        "pid": os.getpid(),
        "loaded_bundle_sha256": loaded_bundle(root),
        "last_attempt_at_ms": int(time.time() * 1000),
        "lifecycle": "error",
        "error": str(error),
    }
    atomic_json(status_path(service, root), value)
    atomic_json(
        root / "ops" / "services" / f"{service}.error.json",
        {
            "schema": "JopeSelfplayServiceError.v1",
            "service": service,
            "at_ms": int(time.time() * 1000),
            "error": str(error),
            "traceback": traceback.format_exc(),
        },
    )
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("service", choices=SERVICES)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    intervals = {
        "queue": 5.0,
        "reserve": 5.0,
        "ingestion": 5.0,
        "controller": 30.0,
        "watchdog": 5.0,
    }
    interval = max(1.0, args.interval or intervals[args.service])
    lock_path = args.root / "ops" / "services" / f"{args.service}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit(f"{args.service} service already owns its lock") from error
        print(
            json.dumps(record_starting(args.service, args.root), sort_keys=True),
            flush=True,
        )
        while True:
            try:
                value = execute_cycle(args.service, args.root)
            except Exception as error:
                value = record_error(args.service, error, args.root)
            print(json.dumps(value, sort_keys=True), flush=True)
            if args.service == "queue":
                refresh_status_async(args.root)
            if args.once:
                break
            time.sleep(interval)


if __name__ == "__main__":
    main()
