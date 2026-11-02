from __future__ import annotations

import json
import shlex
import subprocess
import tomllib
from pathlib import Path
from typing import Any

if __package__:
    from .smart_v3_cpu_scq_config import (
        CAMPAIGN_DIR,
        DATASETS,
        DEFAULT_POLL_SECONDS,
        DISPATCH_VERSION,
        EVOLUTION_JOB_COUNT,
        INFRASTRUCTURE_FAILURES,
        JOB_COUNT,
        NODES,
        PROJECT,
        PROGRESS_SCHEMA,
        RUN_ID,
        SCQ,
        dataset_owners,
    )
    from .smart_v3_cpu_scq_inputs import (
        deterministic_tar,
        marker,
        read_json,
        scoring_files,
        sha256_file,
    )
else:
    from smart_v3_cpu_scq_config import (
        CAMPAIGN_DIR,
        DATASETS,
        DEFAULT_POLL_SECONDS,
        DISPATCH_VERSION,
        EVOLUTION_JOB_COUNT,
        INFRASTRUCTURE_FAILURES,
        JOB_COUNT,
        NODES,
        PROJECT,
        PROGRESS_SCHEMA,
        RUN_ID,
        SCQ,
        dataset_owners,
    )
    from smart_v3_cpu_scq_inputs import (
        deterministic_tar,
        marker,
        read_json,
        scoring_files,
        sha256_file,
    )

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
