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

SCQ = Path("/home/ubuntu/super_cluster/scq")
PROJECT = "sotarena-b42"
RUN_ID = "golden42-smart-v3-cpu-20260730-v1"
PRIOR_RUN_ID = "golden42-train-only-redteam-20260730-v1"
PARENT_RUN_ID = "golden42-transfer-evolution-20260730-v1"
CHECKPOINT_SHA256 = "fbcd5459f2c18f6cd5e4c23eef76a508a91d653a82f27c6ea41bcee8cafa2592"
NODES = ("xbabe1", "xbabe2", "xbabe3", "xbabe4")
SHARDS = {
    "xbabe1": ("55f06d147bcc2a32",),
    "xbabe2": (
        "bd49e740a79bcb6c",
        "55b144de19659467",
        "aebf5d861a8365b4",
        "019de5d76a718163",
    ),
    "xbabe3": (
        "b5beff4a53fa12c0",
        "5f91a00bd3b66742",
        "ee56a1015e344c11",
    ),
    "xbabe4": ("7af85a04de0c7018", "f8536dfeef471c2f"),
}
ACTIVE_STATES = {"queued", "held", "dispatching", "running", "preempting", "requeued"}
INFRASTRUCTURE_FAILURES = {"rpc-error", "storage", "vanished"}
DEFAULT_POLL_SECONDS = 60
DISPATCH_VERSION = "v4"
CAMPAIGN_DIR = "eleven-job-v4"
PROGRESS_SCHEMA = "SmartV3CpuSCQProgress.v2"
JOB_COUNT = 11
EVOLUTION_JOB_COUNT = 10


def dataset_owners() -> tuple[tuple[str, str], ...]:
    return tuple(
        (node, dataset)
        for node in NODES
        for dataset in SHARDS[node]
    )


DATASETS = tuple(dataset for _, dataset in dataset_owners())
DATASET_OWNERSHIP = dict(dataset_owners())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(data)
    os.replace(temporary, path)


def immutable_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.read_bytes() != destination.read_bytes():
            raise RuntimeError(f"refusing unequal overwrite: {destination}")
        return
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def scq_json(*args: str) -> Any:
    completed = subprocess.run(
        [str(SCQ), "--json", *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(completed.stdout)


def marker(node: str) -> tuple[str, str, list[str]]:
    return (
        f"hostname pin {node}",
        "/usr/bin/env",
        ["bash", "-lc", f'test "$(hostname -s)" = {shlex.quote(node)}'],
    )


def assert_cluster_contract() -> None:
    nodes = scq_json("nodes")["Nodes"]["nodes"]
    present = {row["node"]: row for row in nodes}
    if any(node not in present or not present[node]["up"] for node in NODES):
        raise RuntimeError("one or more required SCQ nodes are unavailable")
    priorities = scq_json("priority", "ls")["FamilyPriorities"]
    rows = priorities.get("families") or priorities.get("priorities") or []
    project = next((row for row in rows if row.get("family") == PROJECT), None)
    if project is None or float(project["effective_weight"]) != 10.0:
        raise RuntimeError(f"{PROJECT} effective priority is not exactly 10")


def deterministic_tar(destination: Path, files: list[Path]) -> Path:
    files = sorted(set(path.resolve() for path in files))
    for path in files:
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"cannot package missing or unsafe input: {path}")
    identity = hashlib.sha256(
        "".join(f"{path}\0{sha256_file(path)}\n" for path in files).encode()
    ).hexdigest()
    path = destination.with_name(f"{destination.stem}-{identity[:16]}{destination.suffix}")
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with tarfile.open(temporary, "w") as archive:
            for source in files:
                name = source.relative_to("/").as_posix()
                info = archive.gettarinfo(str(source), arcname=name)
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                with source.open("rb") as handle:
                    archive.addfile(info, handle)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def source_files(project_root: Path) -> list[Path]:
    files = [project_root / "Cargo.lock", project_root / "Cargo.toml"]
    for relative in ("src",):
        files.extend(
            path
            for path in (project_root / relative).rglob("*")
            if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts
        )
    return files


def training_files(project_root: Path, manifest: dict[str, Any]) -> list[Path]:
    root = project_root / "runs/train-only-studies" / RUN_ID
    files = [
        root / "manifest.redteam.v1.json",
        root / "config.v1.json",
        root / "inputs/embedding-sd.v1.json",
        root / "inputs/proposal-datasets.v1.json",
        Path(manifest["parent"]["manifest_path"]),
        Path(manifest["parent"]["validation_path"]),
        Path(manifest["proposal_bundle"]["path"]),
        Path(manifest["scaling_statistic_path"]),
    ]
    for dataset in manifest["datasets"]:
        files.extend(
            [
                Path(dataset["train_path"]),
                Path(dataset["fold_bank_path"]),
                Path(dataset["transfer_ranking_path"]),
                *(Path(path) for path in dataset["parent_generation_paths"]),
            ]
        )
    forbidden = ("/test.csv", "baselines.v2.json")
    if any(any(value in str(path) for value in forbidden) for path in files):
        raise RuntimeError("training package attempted to include test or baseline material")
    return files


def scoring_files(manifest: dict[str, Any]) -> list[Path]:
    parent_root = Path(manifest["parent"]["manifest_path"]).parent
    baseline_path = parent_root / "baselines.v2.json"
    baseline = read_json(baseline_path)
    files = [baseline_path]
    for entry in baseline["entries"]:
        files.extend(
            [
                Path(entry["prior_aspfm_report_path"]),
                Path(entry["tabpfn_v3_report_path"]),
                Path(
                    entry[
                        "tabfm_report_path"
                        if entry["stronger_tabfm_algorithm"] == "tabfm_regressor"
                        else "tabfm_ensemble_report_path"
                    ]
                ),
                Path(
                    "/home/ubuntu/sotarena/data/tabular-regression-v1"
                )
                / entry["dataset_hash"]
                / "test.csv",
            ]
        )
    return files


def prepare_run(project_root: Path, output: Path) -> tuple[dict[str, Any], Path, Path]:
    binary = project_root / "target/release/jope-dime-regressor"
    if not binary.is_file():
        raise RuntimeError("release binary is absent; build and test it locally first")
    root = project_root / "runs/train-only-studies" / RUN_ID
    manifest_path = root / "manifest.redteam.v1.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            "frozen Smart-v3 manifest is absent; refusing to regenerate proposals"
        )
    manifest = read_json(root / "manifest.redteam.v1.json")
    parent = project_root / "runs/train-only-studies" / PARENT_RUN_ID / "manifest.v2.json"
    if read_json(parent)["checkpoint"]["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise RuntimeError("accepted regression checkpoint drift")
    proposals = (
        project_root
        / "runs/train-only-studies"
        / PRIOR_RUN_ID
        / "inputs/proposals.v1.jsonl"
    )
    if (
        manifest.get("run_id") != RUN_ID
        or manifest.get("proposal_bundle", {}).get("path") != str(proposals)
        or manifest.get("parent", {}).get("manifest_path") != str(parent)
    ):
        raise RuntimeError("frozen Smart-v3 manifest identity drift")
    scq = manifest["config"]["scq"]
    expected_ownership = {
        dataset: node for node, datasets in SHARDS.items() for dataset in datasets
    }
    if (
        manifest["run_id"] != RUN_ID
        or scq["project"] != PROJECT
        or scq["effective_priority"] != 10.0
        or scq["evaluation_cpus"] != 4
        or scq["evaluation_ram_gb"] != 16
        or scq["evaluation_ownership"] != expected_ownership
        or scq["gpu_proposal_nodes"]
    ):
        raise RuntimeError("Smart-v3 CPU manifest contract drift")
    training = deterministic_tar(
        output / "inputs/training-workspace.tar",
        training_files(project_root, manifest),
    )
    source = deterministic_tar(
        output / "inputs/source-bundle.tar", source_files(project_root)
    )
    # Once an attempt has been accepted, its source bundle is part of the
    # immutable submission receipt. A controller restart must keep using that
    # exact bundle even if unrelated repository files changed meanwhile.
    pinned_sources = set()
    for spec_path in sorted((output / "specs").glob("*-r1.toml")):
        spec = tomllib.loads(spec_path.read_text())
        pinned_sources.update(
            item["source"]
            for item in spec.get("input", [])
            if item.get("name") == "source_bundle"
        )
    if pinned_sources:
        if len(pinned_sources) != 1:
            raise RuntimeError("accepted attempts use multiple source bundles")
        source = Path(next(iter(pinned_sources)))
        if not source.is_file():
            raise RuntimeError(f"pinned source bundle is missing: {source}")
    return manifest, training, source


def render_spec(
    *,
    job_id: str,
    node: str,
    command: str,
    inputs: list[dict[str, Any]],
    output_name: str,
    output_source: str,
    cpus: int,
    ram_gb: int,
    manifest: dict[str, Any],
    source_sha256: str,
    attempt: int,
) -> str:
    probe_name, probe_command, probe_args = marker(node)
    env = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "REDTEAM_SCQ_JOB_ID": job_id,
        "REDTEAM_EXPECTED_NODE": node,
        "REDTEAM_SCQ_ATTEMPT": str(attempt),
        "REDTEAM_RELEASE_BINARY_SHA256": next(
            item["sha256"] for item in inputs if item["name"] == "release_binary"
        ),
        "REDTEAM_SOURCE_BUNDLE_SHA256": source_sha256,
        "REDTEAM_MANIFEST_SHA256": manifest["receipt_sha256"],
        "REDTEAM_PROPOSAL_BUNDLE_SHA256": manifest["proposal_bundle"]["sha256"],
        "REDTEAM_NODE_MARKER_PROBE": probe_name,
        "REDTEAM_WORKERS": str(cpus),
        "REDTEAM_CPUS_RESERVED": str(cpus),
        "REDTEAM_RAM_GB": str(ram_gb),
        "REDTEAM_NON_PREEMPTIBLE": "true",
        "REDTEAM_MAX_RUNTIME_HOURS": "24",
        "REDTEAM_IMPORTED_WITHOUT_OVERWRITE": "true",
    }
    full_command = (
        'set -euo pipefail; export REDTEAM_SCHEDULER_NODE="$(hostname)"; ' + command
    )
    lines = [
        "[job]",
        f"id = {json.dumps(job_id)}",
        f"project = {json.dumps(PROJECT)}",
        'tier = "normal"',
        'command = "/usr/bin/env"',
        f"args = {json.dumps(['bash', '-lc', full_command])}",
        'workdir = "/home/ubuntu/jope-dime"',
        "",
        "[resource]",
        "gpus_per_task = 0",
        f"cpus_reserved = {cpus}",
        "omp_threads = 1",
        f"ram_mb = {ram_gb * 1024}",
        "gang_size = 1",
        'placement = "single_node"',
        "",
        "[[dependencies.smoke]]",
        f"name = {json.dumps(probe_name)}",
        f"command = {json.dumps(probe_command)}",
        f"args = {json.dumps(probe_args)}",
        "timeout_secs = 5",
        "",
        "[env]",
        *(f"{name} = {json.dumps(value)}" for name, value in env.items()),
        "",
        "[preempt]",
        "preemptible = false",
        "",
        "[limits]",
        "max_runtime_hours = 24",
        "max_restarts = 0",
        "",
    ]
    for item in inputs:
        lines.extend(
            [
                "[[input]]",
                f"name = {json.dumps(item['name'])}",
                f"source = {json.dumps(str(item['source']))}",
                f"target = {json.dumps(item['target'])}",
                f"sha256 = {json.dumps(item['sha256'])}",
            ]
        )
        if item.get("executable"):
            lines.append("executable = true")
        lines.append("")
    lines.extend(
        [
            "[[output]]",
            f"name = {json.dumps(output_name)}",
            f"source = {json.dumps(output_source)}",
            "directory = true",
            "",
        ]
    )
    return "\n".join(lines)


def input_pin(name: str, source: Path, target: str, executable: bool = False) -> dict[str, Any]:
    return {
        "name": name,
        "source": source,
        "target": target,
        "sha256": sha256_file(source),
        "executable": executable,
    }


def evolution_command(node: str, dataset: str | tuple[str, ...]) -> str:
    if isinstance(dataset, tuple):
        if len(dataset) != 1:
            raise ValueError("evolution jobs must own exactly one dataset")
        dataset = dataset[0]
    invocation = (
        "REDTEAM_PATH_ROOT=\"$mirror\" {input:release_binary} "
        f"--dataset {shlex.quote(dataset)} run-train-only-redteam "
        f"--run-id {shlex.quote(RUN_ID)} --study-root \"$studies\" "
        "--selection-mode smart-only --resume --workers 4"
    )
    member = shlex.quote(
        f"home/ubuntu/jope-dime/runs/train-only-studies/{RUN_ID}/datasets/{dataset}"
    )
    return (
        "mirror=$(mktemp -d); tar -xf {input:training_workspace} -C \"$mirror\"; "
        'studies="$mirror/home/ubuntu/jope-dime/runs/train-only-studies"; '
        + invocation
        + "; out={output:evolution-results}; mkdir -p \"$out\"; "
        + f'tar -cf "$out/evolution-results.tar" -C "$mirror" {member}; '
        + f"printf '%s\\n' \"$(hostname -s)\" > \"$out/node.txt\""
    )


def scoring_command(result_input_names: list[str]) -> str:
    extracts = " ".join(
        f"tar -xf {{input:{name}}} -C \"$mirror\";" for name in result_input_names
    )
    return (
        "mirror=$(mktemp -d); tar -xf {input:training_workspace} -C \"$mirror\"; "
        f"{extracts} "
        "tar -xf {input:score_access} -C \"$mirror\"; "
        'studies="$mirror/home/ubuntu/jope-dime/runs/train-only-studies"; '
        'sotarena="$mirror/home/ubuntu/sotarena"; '
        'REDTEAM_PATH_ROOT="$mirror" {input:release_binary} '
        f"score-smart-v3-redteam --run-id {shlex.quote(RUN_ID)} "
        '--study-root "$studies" --sotarena-root "$sotarena" --workers 2; '
        "out={output:smart-v3-results}; mkdir -p \"$out\"; "
        f'cp "$studies/{RUN_ID}/smart-v3-test-results.json" '
        f'"$studies/{RUN_ID}/smart-v3-test-results.csv" '
        f'"$studies/{RUN_ID}/smart-v3-test-results.md" "$out/"'
    )


def progress_path(output: Path) -> Path:
    return output / "progress.v1.json"


def new_progress() -> dict[str, Any]:
    jobs = {
        f"evolve-{dataset}": {
            "node": node,
            "dataset": dataset,
            "attempts": [],
            "state": "pending",
        }
        for node, dataset in dataset_owners()
    }
    jobs["score-xbabe4"] = {
        "node": "xbabe4",
        "datasets": [],
        "attempts": [],
        "state": "blocked_on_evolution",
    }
    return {
        "schema": PROGRESS_SCHEMA,
        "run_id": RUN_ID,
        "project": PROJECT,
        "jobs": jobs,
        "development_only": True,
        "confirmatory_valid": False,
    }


def load_progress(output: Path) -> dict[str, Any]:
    path = progress_path(output)
    if not path.exists():
        progress = new_progress()
        atomic_json(path, progress)
        return progress
    progress = read_json(path)
    if (
        progress.get("schema") != PROGRESS_SCHEMA
        or progress.get("run_id") != RUN_ID
        or len(progress.get("jobs", {})) != JOB_COUNT
        or set(progress.get("jobs", {}))
        != {f"evolve-{dataset}" for dataset in DATASETS} | {"score-xbabe4"}
    ):
        raise RuntimeError("Smart-v3 progress receipt drift")
    for job in progress["jobs"].values():
        for attempt in job["attempts"]:
            attempt.setdefault("node", job["node"])
    return progress


def job_id(logical: str, retry: int) -> str:
    base = f"smartv3-{RUN_ID}-{logical}-{DISPATCH_VERSION}"
    return base if retry == 0 else f"{base}-r{retry}"


def write_spec(output: Path, logical: str, attempt: int, text: str) -> Path:
    path = output / "specs" / f"{job_id(logical, attempt)}.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != text:
        raise RuntimeError(f"immutable spec drift: {path}")
    path.write_text(text)
    return path


def check_spec(path: Path) -> str:
    completed = subprocess.run(
        [str(SCQ), "--json", "check", str(path)], text=True, capture_output=True
    )
    if completed.returncode == 0:
        return "ready"
    output = completed.stdout + completed.stderr
    message = output.lower()
    if any(
        token in message
        for token in (
            "capacity",
            "launch-ready-capacity",
            "cpu target",
            "slot target",
            "insufficient ram",
            "no eligible node",
        )
    ):
        return "capacity_wait"
    try:
        payload = json.loads(output)
        error = payload.get("Error") or payload.get("error") or {}
        diagnostics = error.get("diagnostics") or []
        spec = tomllib.loads(path.read_text())
        expected_node = spec["env"]["REDTEAM_EXPECTED_NODE"]
        requirement = spec["dependencies"]["smoke"][0]["name"]
        if (
            error.get("code") == "dependency-unsatisfied"
            and diagnostics
            and all(row.get("node") != expected_node for row in diagnostics)
            and all(row.get("requirement") == requirement for row in diagnostics)
        ):
            return "capacity_wait"
    except (json.JSONDecodeError, KeyError, TypeError, tomllib.TOMLDecodeError):
        pass
    raise RuntimeError(f"SCQ check failed for {path}: {message}")


def render_evolution_specs(
    project_root: Path,
    output: Path,
    manifest: dict[str, Any],
    training: Path,
    source: Path,
    progress: dict[str, Any],
) -> list[Path]:
    binary = project_root / "target/release/jope-dime-regressor"
    inputs = [
        input_pin("release_binary", binary, "jope-dime-regressor", True),
        input_pin("training_workspace", training, "training-workspace.tar"),
        input_pin("source_bundle", source, "source-bundle.tar"),
    ]
    paths = []
    for node, dataset in dataset_owners():
        logical = f"evolve-{dataset}"
        retry = max(0, len(progress["jobs"][logical]["attempts"]) - 1)
        text = render_spec(
            job_id=job_id(logical, retry),
            node=node,
            command=evolution_command(node, dataset),
            inputs=inputs,
            output_name="evolution-results",
            output_source="evolution-results",
            cpus=4,
            ram_gb=16,
            manifest=manifest,
            source_sha256=sha256_file(source),
            attempt=retry,
        )
        paths.append(write_spec(output, logical, retry, text))
    return paths


def unique_file(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name} below {root}, found {len(matches)}")
    return matches[0]


def render_score_spec(
    project_root: Path,
    output: Path,
    manifest: dict[str, Any],
    training: Path,
    source: Path,
    progress: dict[str, Any],
    attempt: int = 0,
) -> Path:
    score_access = deterministic_tar(
        output / "inputs/scoring-access.tar", scoring_files(manifest)
    )
    inputs = [
        input_pin(
            "release_binary",
            project_root / "target/release/jope-dime-regressor",
            "jope-dime-regressor",
            True,
        ),
        input_pin("training_workspace", training, "training-workspace.tar"),
        input_pin("source_bundle", source, "source-bundle.tar"),
        input_pin("score_access", score_access, "scoring-access.tar"),
    ]
    result_names = []
    for index, dataset in enumerate(DATASETS, start=1):
        logical = f"evolve-{dataset}"
        fetched = Path(progress["jobs"][logical]["fetched_path"])
        result = unique_file(fetched, "evolution-results.tar")
        name = f"evolution_result_{index}"
        result_names.append(name)
        inputs.append(input_pin(name, result, f"{name}.tar"))
    logical = "score-xbabe4"
    text = render_spec(
        job_id=job_id(logical, attempt),
        node="xbabe4",
        command=scoring_command(result_names),
        inputs=inputs,
        output_name="smart-v3-results",
        output_source="smart-v3-results",
        cpus=2,
        ram_gb=8,
        manifest=manifest,
        source_sha256=sha256_file(source),
        attempt=attempt,
    )
    return write_spec(output, logical, attempt, text)


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
