#!/usr/bin/env python3
"""Resumable SCQ controller for the sealed train-only red-team campaign.

This file only renders and reconciles jobs.  Scientific configuration remains
sealed in the Rust manifest; every compute phase receives content-addressed
inputs and emits a declared package.  Placement is enforced by SCQ capability
probes, never by a scheduler-ignored host field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

SCQ = Path("/home/ubuntu/super_cluster/scq")
PROJECT = "sotarena-b42"
RUN_ID = "golden42-train-only-redteam-20260730-v1"
SMOKE_RUN_ID = f"{RUN_ID}-smoke"
VERSION = "v1"
NODES = ("xbabe1", "xbabe2", "xbabe3", "xbabe4")
GPU_NODES = ("xbabe1", "xbabe2", "xbabe3")
OWNERSHIP = {
    "xbabe1": ("5f91a00bd3b66742",),
    "xbabe2": (
        "55b144de19659467",
        "019de5d76a718163",
        "b5beff4a53fa12c0",
        "f8536dfeef471c2f",
    ),
    "xbabe3": ("bd49e740a79bcb6c", "7af85a04de0c7018"),
    "xbabe4": (
        "55f06d147bcc2a32",
        "aebf5d861a8365b4",
        "ee56a1015e344c11",
    ),
}
EXACT = (
    ("xbabe1", 202_607_300_101),
    ("xbabe1", 202_607_300_211),
    ("xbabe2", 202_607_300_307),
    ("xbabe3", 202_607_300_401),
)
JITTER = (
    ("xbabe1", 202_607_301_103, 202_607_302_101),
    ("xbabe2", 202_607_301_211, 202_607_302_203),
    ("xbabe3", 202_607_301_307, 202_607_302_307),
)
GPU_UUIDS = {
    "xbabe1": "GPU-784030c4-b1f9-d5a3-f0eb-9337f133f503",
    "xbabe2": "GPU-44c63d19-492a-ab46-cc6a-56e12a2ef51b",
    "xbabe3": "GPU-2857566c-23c5-293a-df02-7c66a7571ac4",
}
PHASE_ORDER = (
    "build",
    "tests",
    "prepare",
    "proposals",
    "aggregate-proposals",
    "smoke",
    "full",
    "validate",
    "aggregate",
    "score",
    "report",
)
MODES = ("plan", "submit-next", "status", "reconcile", "aggregate", "run")
ALL_COMMANDS = tuple(dict.fromkeys(PHASE_ORDER + MODES))
TERMINAL_STATES = {"completed", "failed", "cancelled", "canceled"}
ACTIVE_STATES = {"running", "queued", "held", "preempting", "requeued"}
INFRASTRUCTURE_FAILURE_CLASSES = {"rpc-error", "storage", "vanished"}
MAX_INFRASTRUCTURE_RETRIES = 2
DEFAULT_POLL_SECONDS = 60


def quote(value: str) -> str:
    return json.dumps(value)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def immutable_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"refusing to overwrite unequal immutable file: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def immutable_json(path: Path, value: Any) -> None:
    immutable_write(path, canonical_json(value) + b"\n")


def scq_json(*args: str) -> Any:
    completed = subprocess.run(
        [str(SCQ), "--json", *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(completed.stdout)


def study_root(project_root: Path, run_id: str = RUN_ID) -> Path:
    return project_root / "runs/train-only-studies" / run_id


def binary(project_root: Path) -> Path:
    return project_root / "target/release/jope-dime-regressor"


def marker(node: str) -> tuple[str, str, tuple[str, ...]]:
    if node in GPU_UUIDS:
        return (
            f"GPU UUID {GPU_UUIDS[node]}",
            "/usr/bin/env",
            (
                "bash",
                "-lc",
                f"nvidia-smi -i {GPU_UUIDS[node]} 2>/dev/null || "
                f"nvidia-smi --query-gpu=uuid --format=csv,noheader | "
                f"grep -Fx {GPU_UUIDS[node]}",
            ),
        )
    return (
        "xbabe4 provisioned SSH capability",
        "/usr/bin/test",
        ("-f", "/etc/ssh/sshd_config.d/xbabe4.conf"),
    )


def assert_cluster_contract() -> dict[str, Any]:
    nodes = scq_json("nodes")["Nodes"]["nodes"]
    present = {row["node"]: row for row in nodes}
    missing = [
        node
        for node in NODES
        if node not in present
        or not present[node]["up"]
        or float(present[node].get("mem_avail_gb", 0)) < 45
        or float(present[node].get("disk_avail_gb", 0)) < 100
    ]
    if missing:
        raise SystemExit(
            "required SCQ nodes lack the sealed RAM/disk admission contract: "
            + ", ".join(missing)
        )
    priorities = scq_json("priority", "ls")["FamilyPriorities"]
    rows = priorities.get("families") or priorities.get("priorities") or []
    project = next((row for row in rows if row.get("family") == PROJECT), None)
    if project is None or float(project["effective_weight"]) != 10.0:
        raise SystemExit(f"{PROJECT} effective priority is not exactly 10")
    assigned = {dataset for datasets in OWNERSHIP.values() for dataset in datasets}
    if len(assigned) != 10:
        raise SystemExit("evaluation ownership does not cover ten unique datasets")
    return {
        node: {
            "up": bool(present[node]["up"]),
            "mem_avail_gb": present[node].get("mem_avail_gb"),
            "disk_avail_gb": present[node].get("disk_avail_gb"),
        }
        for node in NODES
    }


def source_files(project_root: Path) -> list[Path]:
    roots = (
        "Cargo.lock",
        "Cargo.toml",
        "README.md",
        "manifests",
        "scripts",
        "seeds",
        "src",
        "tests",
        "vendor",
        "sotarena/sotarena",
    )
    files: list[Path] = []
    for root in roots:
        path = project_root / root
        if path.is_file():
            files.append(path)
            continue
        if not path.is_dir():
            continue
        for child in sorted(path.rglob("*")):
            relative = child.relative_to(project_root)
            if any(
                part in {"target", "__pycache__", ".pytest_cache", ".ruff_cache"}
                for part in relative.parts
            ):
                continue
            if child.is_file() and not child.is_symlink():
                files.append(child)
    return sorted(
        set(files), key=lambda path: path.relative_to(project_root).as_posix()
    )


def ensure_source_bundle(project_root: Path, root: Path) -> Path:
    files = source_files(project_root)
    identity = sha256_bytes(
        "".join(
            f"{path.relative_to(project_root).as_posix()}\0{sha256_file(path)}\n"
            for path in files
        ).encode()
    )
    output = root / f"scq/inputs/source-bundle-{identity[:16]}.tar"
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with tarfile.open(temporary, "w") as archive:
            for path in files:
                relative = path.relative_to(project_root).as_posix()
                info = archive.gettarinfo(str(path), arcname=relative)
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                with path.open("rb") as source:
                    archive.addfile(info, source)
        if output.exists():
            if output.read_bytes() != temporary.read_bytes():
                raise RuntimeError("source bundle changed during immutable creation")
        else:
            os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def tree_hash(path: Path) -> str:
    entries = []
    if not path.exists():
        return sha256_bytes(b"missing")
    for child in sorted(path.rglob("*")):
        if child.is_file():
            relative = child.relative_to(path).as_posix()
            entries.append(f"{relative}\0{sha256_file(child)}\n")
    return sha256_bytes("".join(entries).encode())


def immutable_import_tree(source: Path, destination: Path) -> None:
    """Import fetched output without ever replacing unequal controller bytes."""
    if source.is_symlink():
        raise RuntimeError(f"SCQ output contains an unsupported symlink: {source}")
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if (
                not destination.is_file()
                or destination.read_bytes() != source.read_bytes()
            ):
                raise RuntimeError(
                    f"refusing to overwrite unequal imported output: {destination}"
                )
            return
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
        return
    if not source.is_dir():
        raise RuntimeError(f"SCQ output is neither a file nor directory: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for child in sorted(source.rglob("*")):
        relative = child.relative_to(source)
        target = destination / relative
        if child.is_symlink():
            raise RuntimeError(f"SCQ output contains an unsupported symlink: {child}")
        if child.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif child.is_file():
            immutable_import_tree(child, target)
        else:
            raise RuntimeError(f"SCQ output contains an unsupported entry: {child}")


def fetch_outputs(
    project_root: Path,
    rendered: dict[str, list[Path]],
) -> dict[str, str]:
    """Fetch every completed declared package and return imported tree hashes."""
    root = study_root(project_root)
    temporary_root = root / "scq/fetch-tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    imported: dict[str, str] = {}
    for phase in PHASE_ORDER:
        for path in rendered[phase]:
            meta = meta_for(path)
            job = show_job(meta["job_id"])
            if job is None or job.get("state") != "completed":
                raise RuntimeError(
                    f"cannot fetch incomplete SCQ output: {meta['job_id']}"
                )
            with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
                fetched = Path(directory) / "output"
                subprocess.run(
                    [str(SCQ), "fetch", meta["job_id"], "--output", str(fetched)],
                    check=True,
                )
                if not fetched.exists():
                    raise RuntimeError(
                        f"SCQ fetch produced no output: {meta['job_id']}"
                    )
                destination = root / "scq/imported" / phase / meta["job_id"]
                immutable_import_tree(fetched, destination)
                imported[meta["job_id"]] = tree_hash(destination)
    return imported


def contains_forbidden_material(value: Any) -> bool:
    text = json.dumps(value, sort_keys=True).lower()
    return any(
        token in text
        for token in ("/test.csv", "\\test.csv", "baseline_path", "baseline/")
    )


def preflight(project_root: Path, node_snapshot: dict[str, Any]) -> Path:
    root = study_root(project_root)
    manifest_path = root / "manifest.redteam.v1.json"
    request_path = root / "proposal-request.v1.json"
    manifest = read_json(manifest_path)
    request = read_json(request_path)
    if manifest.get("development_only") is not True or manifest.get(
        "confirmatory_valid"
    ):
        raise RuntimeError("manifest is not development-only")
    firewall = manifest.get("firewall", {})
    if any(
        firewall.get(key) is True
        for key in (
            "contains_test_paths",
            "contains_baseline_paths",
            "test_paths_constructed",
            "baseline_paths_constructed",
        )
    ):
        raise RuntimeError("selection firewall contains baseline/test material")
    if any(
        path.exists()
        for path in (
            root / "scores.v1.json",
            root / "test-access.v1.json",
            root / "score-receipt.v1.json",
        )
    ):
        raise RuntimeError("a prior score or test-access receipt already exists")
    existing_preflight = root / "scq/preflight-receipt.v1.json"
    if existing_preflight.exists():
        return existing_preflight

    paths: dict[str, Path] = {
        "manifest": manifest_path,
        "config": root / "config.v1.json",
        "proposal_request": request_path,
        "parent_manifest": Path(manifest["parent"]["manifest_path"]),
        "parent_validation": Path(manifest["parent"]["validation_path"]),
        "proposal_bundle": Path(manifest["proposal_bundle"]["path"]),
        "scaling_statistic": Path(manifest["scaling_statistic_path"]),
        "checkpoint": Path(request["checkpoint_path"]),
        "context_catalog_pointer": Path(request["catalog_pointer_path"]),
        "proposal_dataset_manifest": Path(request["proposal_dataset_manifest_path"]),
    }
    for dataset in manifest["datasets"]:
        dataset_id = dataset["dataset_hash"]
        paths[f"{dataset_id}:train"] = Path(dataset["train_path"])
        paths[f"{dataset_id}:fold_bank"] = Path(dataset["fold_bank_path"])
        paths[f"{dataset_id}:transfer"] = Path(dataset["transfer_ranking_path"])
        for index, parent in enumerate(dataset["parent_generation_paths"]):
            paths[f"{dataset_id}:parent_generation_{index:03}"] = Path(parent)
    hashes = {}
    for name, path in sorted(paths.items()):
        if not path.is_file():
            raise RuntimeError(f"preflight input is absent: {name}: {path}")
        hashes[name] = {"path": str(path), "sha256": sha256_file(path)}
    pilot = root / "datasets/ee56a1015e344c11"
    receipt = {
        "schema": "TrainOnlyRedteamSCQPreflight.v1",
        "run_id": RUN_ID,
        "project": PROJECT,
        "effective_priority": 10.0,
        "nodes": node_snapshot,
        "required_nodes": list(NODES),
        "minimum_free_ram_gb": 45,
        "minimum_free_disk_gb": 100,
        "input_sha256": hashes,
        "selection_manifest_contains_test_or_baseline": False,
        "score_or_test_access_receipt_present": False,
        "pilot_tree_sha256": tree_hash(pilot),
        "development_only": True,
        "confirmatory_valid": False,
    }
    immutable_json(existing_preflight, receipt)
    return existing_preflight


def input_pin(
    name: str,
    source: Path,
    target: str,
    expected: str | None = None,
    *,
    executable: bool = False,
) -> dict[str, Any]:
    if not source.is_file():
        raise RuntimeError(f"SCQ input is absent: {source}")
    digest = sha256_file(source)
    if expected is not None and digest != expected:
        try:
            value = json.loads(source.read_text())
            claimed = value.pop("receipt_sha256", None)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            claimed = None
        if claimed != expected:
            raise RuntimeError(f"SCQ input hash drift for {name}: {source}")
    return {
        "name": name,
        "source": str(source),
        "target": target,
        "sha256": digest,
        "executable": executable,
    }


def input_by_name(inputs: list[dict[str, str]], name: str) -> str:
    return next(item["sha256"] for item in inputs if item["name"] == name)


def optional_input_hash(inputs: list[dict[str, str]], name: str) -> str:
    return next((item["sha256"] for item in inputs if item["name"] == name), "")


def proposal_shard_path(root: Path, kind: str, seed: int, node: str) -> Path:
    directory = root / "inputs/proposal-shards"
    expected = directory / f"{kind}-{seed}-{node}.jsonl"
    if expected.is_file():
        return expected
    matches = sorted(directory.glob(f"{kind}-{seed}-*.jsonl"))
    if len(matches) == 1:
        return matches[0]
    raise RuntimeError(f"sealed proposal shard is absent or ambiguous: {expected}")


def common_inputs(
    project_root: Path,
    source_bundle: Path,
    include_binary: bool = True,
) -> list[dict[str, str]]:
    root = study_root(project_root)
    inputs = [
        input_pin("source_bundle", source_bundle, "source-bundle.tar"),
        input_pin(
            "manifest", root / "manifest.redteam.v1.json", "manifest.redteam.v1.json"
        ),
        input_pin(
            "proposal_request",
            root / "proposal-request.v1.json",
            "proposal-request.v1.json",
        ),
    ]
    if include_binary:
        inputs.append(
            input_pin("release_binary", binary(project_root), "jope-dime-regressor")
        )
    return inputs


def render_spec(
    *,
    job_id: str,
    node: str,
    command: str,
    inputs: list[dict[str, str]],
    outputs: list[tuple[str, str, bool]],
    gpus: int = 0,
    cpus: int = 8,
    ram_gb: int = 40,
    python: bool = False,
    attempt: int = 0,
    output_package_sha256: str = "",
) -> str:
    probe_name, probe_command, probe_args = marker(node)
    parent_input_sha256 = {
        item["name"]: item["sha256"]
        for item in inputs
        if item["name"].startswith("parent_")
    }
    env = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "REDTEAM_SCQ_JOB_ID": job_id,
        "REDTEAM_EXPECTED_NODE": node,
        "REDTEAM_SCQ_ATTEMPT": str(attempt),
        "REDTEAM_RELEASE_BINARY_SHA256": input_by_name(inputs, "release_binary")
        if any(item["name"] == "release_binary" for item in inputs)
        else "",
        "REDTEAM_SOURCE_BUNDLE_SHA256": optional_input_hash(inputs, "source_bundle"),
        "REDTEAM_MANIFEST_SHA256": optional_input_hash(inputs, "manifest"),
        "REDTEAM_PROPOSAL_BUNDLE_SHA256": optional_input_hash(
            inputs, "proposal_bundle"
        ),
        "REDTEAM_FOLD_BANK_SHA256": optional_input_hash(inputs, "fold_bank"),
        "REDTEAM_PARENT_INPUT_SHA256": (
            json.dumps(
                parent_input_sha256,
                sort_keys=True,
                separators=(",", ":"),
            )
            if parent_input_sha256
            else ""
        ),
        "REDTEAM_NODE_MARKER_PROBE": probe_name,
        "REDTEAM_WORKERS": "8",
        "REDTEAM_CPUS_RESERVED": str(cpus),
        "REDTEAM_RAM_GB": str(ram_gb),
        "REDTEAM_NON_PREEMPTIBLE": "true",
        "REDTEAM_MAX_RUNTIME_HOURS": "24",
        "REDTEAM_IMPORTED_WITHOUT_OVERWRITE": "true",
        "REDTEAM_PYTHON_JOB": "true" if python else "false",
        "REDTEAM_OUTPUT_PACKAGE_SHA256": output_package_sha256,
    }
    lines = [
        "[job]",
        f"id = {quote(job_id)}",
        f"project = {quote(PROJECT)}",
        'tier = "normal"',
        'command = "/usr/bin/env"',
        f'args = ["bash", "-lc", {quote('set -euo pipefail; export REDTEAM_SCHEDULER_NODE=\\"$(hostname)\\"; ' + command)}]',
        'workdir = "/home/ubuntu/jope-dime"',
        "",
        "[resource]",
        f"gpus_per_task = {gpus}",
        f"cpus_reserved = {cpus}",
        "omp_threads = 1",
        f"ram_mb = {ram_gb * 1024}",
        "gang_size = 1",
        'placement = "single_node"',
        "",
        "[[dependencies.smoke]]",
        f"name = {quote(probe_name)}",
        f"command = {quote(probe_command)}",
        f"args = {json.dumps(list(probe_args))}",
        "timeout_secs = 5",
        "",
        "[env]",
    ]
    lines.extend(f"{key} = {quote(value)}" for key, value in env.items() if value)
    lines.extend(
        [
            "",
            "[preempt]",
            "preemptible = false",
            "",
            "[limits]",
            "max_runtime_hours = 24",
            "max_restarts = 0",
            "",
        ]
    )
    for item in inputs:
        lines.extend(
            [
                "[[input]]",
                f"name = {quote(item['name'])}",
                f"source = {quote(item['source'])}",
                f"target = {quote(item['target'])}",
                f"sha256 = {quote(item['sha256'])}",
            ]
        )
        if item.get("executable"):
            lines.append("executable = true")
        lines.append("")
    for name, source, directory in outputs:
        lines.extend(
            [
                "[[output]]",
                f"name = {quote(name)}",
                f"source = {quote(source)}",
            ]
        )
        if directory:
            lines.append("directory = true")
        lines.append("")
    return "\n".join(lines)


def prepare_command(project_root: Path, run_id: str, smoke: bool) -> str:
    suffix = " --smoke" if smoke else ""
    return (
        f"{quote(str(binary(project_root)))} prepare-train-only-redteam"
        f" --run-id {quote(run_id)}"
        f" --study-root {quote(str(project_root / 'runs/train-only-studies'))}"
        f" --sotarena-root {quote('/home/ubuntu/sotarena')}"
        f" --workers 8{suffix}; mkdir -p scq-output; cp -a"
        f" {quote(str(study_root(project_root, run_id)))} scq-output/study-skeleton"
    )


def proposal_command(
    project_root: Path,
    run_id: str,
    seed: int,
    node: str,
    perturbation_seed: int | None,
) -> str:
    root = study_root(project_root, run_id)
    request = root / "proposal-request.v1.json"
    output = Path("proposal-output")
    if perturbation_seed is None:
        name = f"exact-{seed}-{node}.jsonl"
        mode = "--proposal-origin exact_context --redteam-metadata"
    else:
        name = f"jitter-{perturbation_seed}-{node}.jsonl"
        mode = (
            "--proposal-origin jittered_context --redteam-metadata"
            f" --embedding-sd {quote(str(root / 'inputs/embedding-sd.v1.json'))}"
            " --embedding-jitter-scale 0.05"
            f" --perturbation-seed {perturbation_seed}"
        )
    return (
        f"mkdir -p {quote(str(output))}; "
        f"checkpoint=$(jq -r .checkpoint_path {quote(str(request))}); "
        f"catalog=$(jq -r .catalog_pointer_path {quote(str(request))}); "
        f"manifest=$(jq -r .proposal_dataset_manifest_path {quote(str(request))}); "
        f"PYTHONPATH=scripts:/home/ubuntu/jope-prime/src python3"
        f" scripts/test_exact_dl_generate.py"
        f' --checkpoint "$checkpoint" --catalog-pointer "$catalog"'
        f' --dataset-manifest "$manifest" --task regression'
        f" --proposals-per-dataset 64 --seed {seed} --device cuda"
        f" --output {quote(str(output / name))} {mode}"
    )


def dataset_command(project_root: Path, dataset: str, run_id: str) -> str:
    root = study_root(project_root, run_id)
    output = Path("dataset-output")
    return (
        f"mkdir -p {quote(str(output))}; "
        f"{quote(str(binary(project_root)))} --dataset {dataset}"
        f" run-train-only-redteam --run-id {quote(run_id)}"
        f" --study-root {quote(str(project_root / 'runs/train-only-studies'))}"
        " --arm g20-extension --resume --workers 8; "
        f"{quote(str(binary(project_root)))} --dataset {dataset}"
        f" run-train-only-redteam --run-id {quote(run_id)}"
        f" --study-root {quote(str(project_root / 'runs/train-only-studies'))}"
        " --arm smart-v3 --resume --workers 8; "
        f"cp -a {quote(str(root / 'datasets' / dataset))}"
        f" {quote(str(output / dataset))}"
    )


def validation_command(project_root: Path, dataset: str) -> str:
    root = study_root(project_root)
    output = Path("validation-output")
    return (
        f"mkdir -p {quote(str(output))}; "
        f"package_hash=$(find {quote(str(root / 'datasets' / dataset))} -type f -print0"
        " | sort -z | xargs -0 -r sha256sum | sha256sum | cut -d ' ' -f1); "
        'export REDTEAM_OUTPUT_PACKAGE_SHA256="$package_hash"; '
        f"{quote(str(binary(project_root)))} --dataset {dataset}"
        " validate-train-only-redteam"
        f" --run-id {quote(RUN_ID)}"
        f" --study-root {quote(str(project_root / 'runs/train-only-studies'))}"
        f" --sotarena-root {quote('/home/ubuntu/sotarena')} --workers 8; "
        f"cp {quote(str(root / 'datasets' / dataset / 'validation-shard.v1.json'))}"
        f" {quote(str(output / f'{dataset}.validation-shard.v1.json'))}"
    )


def aggregate_proposals_command(project_root: Path) -> str:
    root = study_root(project_root)
    combined = root / "inputs/proposals.v1.jsonl"
    ordered = []
    for _, seed in EXACT:
        ordered.append(root / "inputs/proposal-shards" / f"exact-{seed}-xbabe1.jsonl")
    for node, seed in EXACT:
        ordered[-len(EXACT) + EXACT.index((node, seed))] = (
            root / "inputs/proposal-shards" / f"exact-{seed}-{node}.jsonl"
        )
    ordered.extend(
        root / "inputs/proposal-shards" / f"jitter-{perturbation}-{node}.jsonl"
        for node, _, perturbation in JITTER
    )
    cat = "cat " + " ".join(quote(str(path)) for path in ordered)
    return (
        f"mkdir -p {quote(str(root / 'inputs/proposal-shards'))}; "
        f"{cat} > {quote(str(combined))}; "
        f"{quote(str(binary(project_root)))} prepare-train-only-redteam"
        f" --run-id {quote(RUN_ID)}"
        f" --study-root {quote(str(project_root / 'runs/train-only-studies'))}"
        f" --sotarena-root {quote('/home/ubuntu/sotarena')}"
        f" --proposals {quote(str(combined))} --workers 8; "
        "mkdir -p scq-output; cp "
        f"{quote(str(combined))} scq-output/proposals.v1.jsonl"
    )


def aggregate_command(project_root: Path) -> str:
    root = study_root(project_root)
    return (
        f"package_hash=$(find {quote(str(root))} -type f -print0 | sort -z"
        " | xargs -0 -r sha256sum | sha256sum | cut -d ' ' -f1); "
        'export REDTEAM_OUTPUT_PACKAGE_SHA256="$package_hash"; '
        f"{quote(str(binary(project_root)))} run-train-only-redteam"
        f" --run-id {quote(RUN_ID)}"
        f" --study-root {quote(str(project_root / 'runs/train-only-studies'))}"
        " --arm g20-extension --resume --workers 8; "
        f"{quote(str(binary(project_root)))} run-train-only-redteam"
        f" --run-id {quote(RUN_ID)}"
        f" --study-root {quote(str(project_root / 'runs/train-only-studies'))}"
        " --arm smart-v3 --resume --workers 8; "
        f"{quote(str(binary(project_root)))} validate-train-only-redteam"
        f" --run-id {quote(RUN_ID)}"
        f" --study-root {quote(str(project_root / 'runs/train-only-studies'))}"
        f" --sotarena-root {quote('/home/ubuntu/sotarena')} --workers 8; "
        f"mkdir -p scq-output; cp {quote(str(root / 'validation-success.v1.json'))}"
        " scq-output/validation-success.v1.json"
    )


def score_command(project_root: Path) -> str:
    root = study_root(project_root)
    return (
        f"{quote(str(binary(project_root)))} score-train-only-redteam"
        f" --run-id {quote(RUN_ID)}"
        f" --study-root {quote(str(project_root / 'runs/train-only-studies'))}"
        f" --sotarena-root {quote('/home/ubuntu/sotarena')} --workers 8; "
        f"mkdir -p scq-output; cp {quote(str(root / 'scores.v1.json'))}"
        " scq-output/scores.v1.json"
    )


def report_command(project_root: Path) -> str:
    root = study_root(project_root)
    return (
        f"{quote(str(binary(project_root)))} report-train-only-redteam"
        f" --run-id {quote(RUN_ID)}"
        f" --study-root {quote(str(project_root / 'runs/train-only-studies'))}"
        f" --sotarena-root {quote('/home/ubuntu/sotarena')} --workers 8; "
        f"mkdir -p scq-output; cp {quote(str(root / 'report.v1.json'))}"
        " scq-output/report.v1.json"
    )


def dataset_inputs(
    project_root: Path,
    source_bundle: Path,
    dataset: dict[str, Any],
) -> list[dict[str, str]]:
    inputs = common_inputs(project_root, source_bundle)
    root = study_root(project_root)
    inputs.extend(
        [
            input_pin(
                "proposal_bundle",
                Path(root / "inputs/proposals.v1.jsonl"),
                "proposals.v1.jsonl",
                dataset.get("proposal_bundle_sha256"),
            ),
            input_pin(
                "fold_bank",
                Path(dataset["fold_bank_path"]),
                f"fold-banks/{dataset['dataset_hash']}.v1.json",
                dataset["fold_bank_sha256"],
            ),
            input_pin(
                "train_csv",
                Path(dataset["train_path"]),
                f"train/{dataset['dataset_hash']}.csv",
                dataset["train_sha256"],
            ),
            input_pin(
                "transfer_ranking",
                Path(dataset["transfer_ranking_path"]),
                f"transfer/{dataset['dataset_hash']}.json",
                dataset["transfer_ranking_sha256"],
            ),
        ]
    )
    inputs.append(
        input_pin(
            "parent_manifest",
            Path(root / "../golden42-transfer-evolution-20260730-v1/manifest.v2.json"),
            "parent/manifest.v2.json",
        )
    )
    inputs.append(
        input_pin(
            "parent_validation",
            Path(
                root
                / "../golden42-transfer-evolution-20260730-v1/validation-success.v2.json"
            ),
            "parent/validation-success.v2.json",
        )
    )
    for index, parent in enumerate(dataset["parent_generation_paths"]):
        inputs.append(
            input_pin(
                f"parent_generation_{index:03}",
                Path(parent),
                f"parent/generation-{index:03}.v2.json",
                dataset["parent_generation_sha256"][index],
            )
        )
    return inputs


def job_id(phase: str, suffix: str, tag: str) -> str:
    return f"redteam-{RUN_ID}-scq-{phase}-{suffix}-{tag}"


def build_jobs(
    project_root: Path, sotarena_root: Path, source_bundle: Path, tag: str
) -> dict[str, list[dict[str, Any]]]:
    del sotarena_root
    root = study_root(project_root)
    manifest = read_json(root / "manifest.redteam.v1.json")
    common = common_inputs(project_root, source_bundle)
    jobs: dict[str, list[dict[str, Any]]] = {phase: [] for phase in PHASE_ORDER}
    jobs["build"].append(
        {
            "id": job_id("build", "xbabe2", tag),
            "node": "xbabe2",
            "command": "mkdir -p scq-build; tar -xf {input:source_bundle} -C scq-build; cd scq-build; cargo fmt --all -- --check; cargo test --locked train_only_redteam -- --nocapture; cargo build --release --locked; mkdir -p ../scq-output; cp target/release/jope-dime-regressor ../scq-output/jope-dime-regressor; sha256sum target/release/jope-dime-regressor > ../scq-output/release-binary.sha256",
            "inputs": [common[0]],
            "outputs": [("build-receipts", "scq-output", True)],
            "cpus": 8,
            "ram": 40,
        }
    )
    for node in NODES:
        test_command = "mkdir -p scq-tests; tar -xf {input:source_bundle} -C scq-tests; cd scq-tests; cargo test --locked train_only_redteam -- --nocapture"
        if node != "xbabe4":
            test_command += "; PYTHONPATH=scripts:/home/ubuntu/jope-prime/src python3 -m unittest scripts/test_train_only_redteam_generation.py"
        jobs["tests"].append(
            {
                "id": job_id("tests", node, tag),
                "node": node,
                "command": test_command,
                "inputs": common,
                "outputs": [("test-receipts", "scq-output", True)],
                "python": node != "xbabe4",
            }
        )
    for smoke in (False, True):
        run_id = SMOKE_RUN_ID if smoke else RUN_ID
        suffix = "smoke-xbabe4" if smoke else "xbabe4"
        jobs["prepare"].append(
            {
                "id": job_id("prepare", suffix, tag),
                "node": "xbabe4",
                "command": prepare_command(project_root, run_id, smoke),
                "inputs": common,
                "outputs": [("study-skeleton", "scq-output", True)],
            }
        )
    for node, seed in EXACT:
        inputs = common + [
            input_pin(
                "generator_script",
                project_root / "scripts/test_exact_dl_generate.py",
                "test_exact_dl_generate.py",
            ),
            input_pin(
                "checkpoint",
                Path(read_json(root / "proposal-request.v1.json")["checkpoint_path"]),
                "checkpoint.model.pt",
            ),
            input_pin(
                "context_catalog",
                Path(
                    read_json(root / "proposal-request.v1.json")["catalog_pointer_path"]
                ),
                "context-catalog-pointer.json",
            ),
            input_pin(
                "proposal_metadata",
                Path(
                    read_json(root / "proposal-request.v1.json")[
                        "proposal_dataset_manifest_path"
                    ]
                ),
                "proposal-datasets.v1.json",
            ),
            input_pin(
                "scaling_statistic",
                Path(
                    read_json(root / "proposal-request.v1.json")[
                        "scaling_statistic_path"
                    ]
                ),
                "embedding-sd.v1.json",
            ),
        ]
        jobs["proposals"].append(
            {
                "id": job_id("proposals", f"exact-{seed}-{node}", tag),
                "node": node,
                "command": proposal_command(project_root, RUN_ID, seed, node, None),
                "inputs": inputs,
                "outputs": [("proposal-package", "proposal-output", True)],
                "gpus": 1,
                "python": True,
            }
        )
    for node, seed, perturbation in JITTER:
        request = read_json(root / "proposal-request.v1.json")
        inputs = common + [
            input_pin(
                "generator_script",
                project_root / "scripts/test_exact_dl_generate.py",
                "test_exact_dl_generate.py",
            ),
            input_pin(
                "checkpoint", Path(request["checkpoint_path"]), "checkpoint.model.pt"
            ),
            input_pin(
                "context_catalog",
                Path(request["catalog_pointer_path"]),
                "context-catalog-pointer.json",
            ),
            input_pin(
                "proposal_metadata",
                Path(request["proposal_dataset_manifest_path"]),
                "proposal-datasets.v1.json",
            ),
            input_pin(
                "scaling_statistic",
                Path(request["scaling_statistic_path"]),
                "embedding-sd.v1.json",
            ),
        ]
        jobs["proposals"].append(
            {
                "id": job_id("proposals", f"jitter-{perturbation}-{node}", tag),
                "node": node,
                "command": proposal_command(
                    project_root, RUN_ID, seed, node, perturbation
                ),
                "inputs": inputs,
                "outputs": [("proposal-package", "proposal-output", True)],
                "gpus": 1,
                "python": True,
            }
        )
    proposal_inputs = common + []
    for index, (node, seed) in enumerate(EXACT):
        proposal_inputs.append(
            input_pin(
                f"proposal-shard-{index}",
                proposal_shard_path(root, "exact", seed, node),
                f"proposal-shards/exact-{seed}-{node}.jsonl",
            )
        )
    for index, (node, _, perturbation) in enumerate(JITTER, start=len(EXACT)):
        proposal_inputs.append(
            input_pin(
                f"proposal-shard-{index}",
                proposal_shard_path(root, "jitter", perturbation, node),
                f"proposal-shards/jitter-{perturbation}-{node}.jsonl",
            )
        )
    jobs["aggregate-proposals"].append(
        {
            "id": job_id("aggregate-proposals", "xbabe4", tag),
            "node": "xbabe4",
            "command": aggregate_proposals_command(project_root),
            "inputs": proposal_inputs,
            "outputs": [("proposal-bundle", "scq-output", True)],
        }
    )
    for node in NODES:
        datasets = OWNERSHIP[node]
        smoke_commands = []
        for dataset in datasets:
            smoke_commands.extend(
                [
                    f"{quote(str(binary(project_root)))} --dataset {dataset} run-train-only-redteam --run-id {quote(SMOKE_RUN_ID)} --study-root {quote(str(project_root / 'runs/train-only-studies'))} --arm g20-extension --resume --workers 8",
                    f"{quote(str(binary(project_root)))} --dataset {dataset} run-train-only-redteam --run-id {quote(SMOKE_RUN_ID)} --study-root {quote(str(project_root / 'runs/train-only-studies'))} --arm smart-v3 --resume --workers 8",
                ]
            )
        jobs["smoke"].append(
            {
                "id": job_id("smoke", node, tag),
                "node": node,
                "command": "; ".join(smoke_commands)
                + "; mkdir -p scq-output; echo smoke-complete > scq-output/smoke.receipt",
                "inputs": common,
                "outputs": [("smoke-receipts", "scq-output", True)],
            }
        )
    for node, datasets in OWNERSHIP.items():
        for dataset_id in datasets:
            dataset = next(
                row for row in manifest["datasets"] if row["dataset_hash"] == dataset_id
            )
            jobs["full"].append(
                {
                    "id": job_id("full", dataset_id, tag),
                    "node": node,
                    "command": dataset_command(project_root, dataset_id, RUN_ID),
                    "inputs": dataset_inputs(project_root, source_bundle, dataset),
                    "outputs": [("dataset-artifacts", "dataset-output", True)],
                }
            )
            jobs["validate"].append(
                {
                    "id": job_id("validate", dataset_id, tag),
                    "node": node,
                    "command": validation_command(project_root, dataset_id),
                    "inputs": dataset_inputs(project_root, source_bundle, dataset),
                    "outputs": [("validation-shard", "validation-output", True)],
                }
            )
    aggregate_inputs = common
    jobs["aggregate"].append(
        {
            "id": job_id("aggregate", "xbabe4", tag),
            "node": "xbabe4",
            "command": aggregate_command(project_root),
            "inputs": aggregate_inputs,
            "outputs": [("global-validation", "scq-output", True)],
        }
    )
    jobs["score"].append(
        {
            "id": job_id("score", "xbabe4", tag),
            "node": "xbabe4",
            "command": score_command(project_root),
            "inputs": common,
            "outputs": [("score-receipts", "scq-output", True)],
        }
    )
    jobs["report"].append(
        {
            "id": job_id("report", "xbabe4", tag),
            "node": "xbabe4",
            "command": report_command(project_root),
            "inputs": common,
            "outputs": [("final-report", "scq-output", True)],
        }
    )
    return jobs


def write_job_spec(path: Path, job: dict[str, Any], phase: str) -> None:
    body = render_spec(
        job_id=job["id"],
        node=job["node"],
        command=job["command"],
        inputs=job["inputs"],
        outputs=job["outputs"],
        gpus=job.get("gpus", 0),
        cpus=job.get("cpus", 8),
        ram_gb=job.get("ram", 40),
        python=job.get("python", False),
    )
    digest = sha256_bytes(body.encode())
    meta = {
        "schema": "TrainOnlyRedteamSCQJobMeta.v1",
        "job_id": job["id"],
        "phase": phase,
        "expected_node": job["node"],
        "spec_sha256": digest,
        "complete_spec_hash": digest,
        "input_sha256": {item["name"]: item["sha256"] for item in job["inputs"]},
        "node_marker_probe": marker(job["node"])[0],
        "workers": 8,
        "cpus_reserved": job.get("cpus", 8),
        "ram_gb": job.get("ram", 40),
        "non_preemptible": True,
        "max_runtime_hours": 24,
    }
    meta["release_binary_sha256"] = meta["input_sha256"].get("release_binary", "")
    meta["source_bundle_sha256"] = meta["input_sha256"].get("source_bundle", "")
    immutable_write(path, body.encode())
    immutable_json(path.with_suffix(".meta.json"), meta)


def emit_specs(
    project_root: Path,
    sotarena_root: Path,
    output: Path,
    tag: str = VERSION,
) -> dict[str, list[Path]]:
    del sotarena_root
    node_snapshot = assert_cluster_contract()
    root = study_root(project_root)
    source_bundle = ensure_source_bundle(project_root, root)
    preflight(project_root, node_snapshot)
    jobs = build_jobs(project_root, Path(), source_bundle, tag)
    output.mkdir(parents=True, exist_ok=True)
    rendered: dict[str, list[Path]] = {}
    for phase in PHASE_ORDER:
        rendered[phase] = []
        for job in jobs[phase]:
            path = output / f"{job['id']}.toml"
            write_job_spec(path, job, phase)
            rendered[phase].append(path)
    return rendered


def check_specs(paths: list[Path]) -> None:
    for path in paths:
        subprocess.run([str(SCQ), "check", str(path)], check=True)


def meta_for(path: Path) -> dict[str, Any]:
    return read_json(path.with_suffix(".meta.json"))


def show_job(job_id_value: str) -> dict[str, Any] | None:
    response = scq_json("show", job_id_value)
    return response.get("Job", {}).get("job")


def status(rendered: dict[str, list[Path]]) -> list[dict[str, Any]]:
    rows = []
    for phase in PHASE_ORDER:
        for path in rendered[phase]:
            meta = meta_for(path)
            job = show_job(meta["job_id"])
            rows.append(
                {
                    "phase": phase,
                    "job_id": meta["job_id"],
                    "expected_node": meta["expected_node"],
                    "state": job["state"] if job else "absent",
                    "actual_node": job.get("node") if job else None,
                    "attempt": job.get("attempt") if job else None,
                }
            )
    return rows


def reconcile(project_root: Path, rendered: dict[str, list[Path]]) -> Path:
    return reconcile_with_imports(project_root, rendered, {})


def reconcile_with_imports(
    project_root: Path,
    rendered: dict[str, list[Path]],
    imported: dict[str, str],
) -> Path:
    rows = []
    for phase in PHASE_ORDER:
        for path in rendered[phase]:
            meta = meta_for(path)
            job = show_job(meta["job_id"])
            if job is None:
                raise RuntimeError(f"SCQ ledger entry is missing: {meta['job_id']}")
            actual = job.get("node")
            if actual is not None and actual != meta["expected_node"]:
                raise RuntimeError(
                    f"SCQ placed {meta['job_id']} on {actual}, expected {meta['expected_node']}"
                )
            input_hashes = meta.get("input_sha256", {})
            rows.append(
                {
                    "phase": phase,
                    "job_id": meta["job_id"],
                    "expected_node": meta["expected_node"],
                    "scheduler_reported_node": actual,
                    "state": job["state"],
                    "attempt": job.get("attempt", 0),
                    "spec_sha256": meta["spec_sha256"],
                    "scq_show_sha256": sha256_bytes(canonical_json(job)),
                    "release_binary_sha256": meta.get("release_binary_sha256", ""),
                    "source_bundle_sha256": meta.get("source_bundle_sha256", ""),
                    "manifest_sha256": input_hashes.get("manifest", ""),
                    "proposal_bundle_sha256": input_hashes.get("proposal_bundle", ""),
                    "fold_bank_sha256": input_hashes.get("fold_bank", ""),
                    "parent_input_sha256": {
                        name: digest
                        for name, digest in input_hashes.items()
                        if name.startswith("parent_")
                    },
                    "output_package_sha256": imported.get(meta["job_id"], ""),
                    "node_marker_probe": meta["node_marker_probe"],
                    "workers": meta["workers"],
                    "cpus_reserved": meta["cpus_reserved"],
                    "ram_gb": meta["ram_gb"],
                    "non_preemptible": meta["non_preemptible"],
                    "max_runtime_hours": meta["max_runtime_hours"],
                    "imported_without_overwrite": meta["job_id"] in imported,
                }
            )
    ledger = {
        "schema": "TrainOnlyRedteamSCQExecutionLedger.v1",
        "run_id": RUN_ID,
        "project": PROJECT,
        "jobs": rows,
        "actual_nodes_match_expected": True,
        "development_only": True,
        "confirmatory_valid": False,
    }
    path = study_root(project_root) / "scq/execution-ledger.v1.json"
    immutable_json(path, ledger)
    return path


def submission_record(project_root: Path, meta: dict[str, Any]) -> Path:
    return study_root(project_root) / "scq/submissions" / f"{meta['job_id']}.json"


def verify_reusable_submission(project_root: Path, meta: dict[str, Any]) -> None:
    record = submission_record(project_root, meta)
    if not record.is_file():
        raise RuntimeError(
            f"refusing stale SCQ ID reuse without a matching submission record: {meta['job_id']}"
        )
    previous = read_json(record)
    if previous.get("complete_spec_hash") != meta["complete_spec_hash"]:
        raise RuntimeError(
            f"refusing stale SCQ ID reuse with a different complete spec: {meta['job_id']}"
        )


def submit_phase(project_root: Path, paths: list[Path]) -> None:
    if not paths:
        return
    pending = []
    for path in paths:
        check_specs([path])
        meta = meta_for(path)
        existing = show_job(meta["job_id"])
        if existing is None:
            pending.append(path)
            continue
        verify_reusable_submission(project_root, meta)
        if existing["state"] in {"failed", "cancelled", "canceled"}:
            raise RuntimeError(
                f"job {meta['job_id']} already failed; use a new versioned tag"
            )
    if pending:
        subprocess.run(
            [str(SCQ), "submit-batch", *(str(path) for path in pending)], check=True
        )
        for path in pending:
            meta = meta_for(path)
            immutable_json(
                submission_record(project_root, meta),
                {
                    "schema": "TrainOnlyRedteamSCQSubmission.v1",
                    "job_id": meta["job_id"],
                    "complete_spec_hash": meta["complete_spec_hash"],
                },
            )


def submit_next(project_root: Path, rendered: dict[str, list[Path]]) -> str:
    for phase in PHASE_ORDER:
        rows = [
            row
            for row in status(
                {
                    phase: rendered[phase],
                    **{name: [] for name in PHASE_ORDER if name != phase},
                }
            )
            if row["phase"] == phase
        ]
        if all(row["state"] == "completed" for row in rows):
            continue
        if any(row["state"] in {"failed", "cancelled", "canceled"} for row in rows):
            raise RuntimeError(
                f"{phase} has a failed job; choose a new versioned retry tag"
            )
        if any(row["state"] in {"running", "queued", "held"} for row in rows):
            return f"{phase} already has active jobs"
        submit_phase(project_root, rendered[phase])
        return f"submitted {phase}"
    return "all phases are complete"


def atomic_json(path: Path, value: Any) -> None:
    """Replace controller state atomically; scientific artifacts stay immutable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(canonical_json(value) + b"\n")
    os.replace(temporary, path)


def scq_command_json(*args: str) -> tuple[int, Any, str]:
    completed = subprocess.run(
        [str(SCQ), "--json", *args],
        check=False,
        text=True,
        capture_output=True,
    )
    output = completed.stdout.strip() or completed.stderr.strip()
    try:
        payload: Any = json.loads(output)
    except json.JSONDecodeError:
        payload = {}
    return completed.returncode, payload, output


def _diagnostic_requirements(value: Any) -> list[str]:
    requirements: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "requirement" and isinstance(child, str):
                requirements.append(child)
            requirements.extend(_diagnostic_requirements(child))
    elif isinstance(value, list):
        for child in value:
            requirements.extend(_diagnostic_requirements(child))
    return requirements


def is_capacity_failure(value: Any) -> bool:
    """Classify SCQ's dependency admission stall without weakening the spec."""
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return (
        "launch-ready-capacity" in _diagnostic_requirements(value)
        or "launch-ready-capacity" in text
    )


def is_infrastructure_failure(job: dict[str, Any]) -> bool:
    return job.get("failure_class") in INFRASTRUCTURE_FAILURE_CLASSES


def validate_fetched_tree(path: Path) -> None:
    """Reject special, symlinked, or escaping fetched content."""
    if not path.exists():
        raise RuntimeError(f"SCQ fetch produced no output: {path}")
    root = path.resolve()
    entries = [path] if path.is_file() or path.is_symlink() else path.rglob("*")
    for child in entries:
        if child.is_symlink():
            raise RuntimeError(f"SCQ output contains an unsupported symlink: {child}")
        resolved = child.resolve()
        if resolved != root and root not in resolved.parents:
            raise RuntimeError(f"SCQ output traverses its fetch root: {child}")
        if not child.is_file() and not child.is_dir():
            raise RuntimeError(f"SCQ output contains an unsupported entry: {child}")


def unique_package_file(root: Path, name: str) -> Path:
    """Resolve one fetched file, allowing only byte-identical duplicates."""
    matches = sorted(path for path in root.rglob(name) if path.is_file())
    if not matches:
        raise RuntimeError(f"fetched package is missing {name}: {root}")
    first = sha256_file(matches[0])
    unequal = [path for path in matches[1:] if sha256_file(path) != first]
    if unequal:
        raise RuntimeError(
            f"fetched package contains duplicate-unequal {name}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def tar_file_hashes(path: Path) -> dict[str, str]:
    """Validate a tar package and return hashes keyed by safe member path."""
    hashes: dict[str, str] = {}
    with tarfile.open(path, "r") as archive:
        for member in archive.getmembers():
            member_path = Path(member.name)
            if (
                member_path.is_absolute()
                or ".." in member_path.parts
                or member.issym()
                or member.islnk()
                or not (member.isfile() or member.isdir())
            ):
                raise RuntimeError(f"unsafe package member in {path}: {member.name}")
            if member.isdir():
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError(
                    f"package member is unreadable: {path}:{member.name}"
                )
            digest = sha256_bytes(handle.read())
            previous = hashes.get(member.name)
            if previous is not None and previous != digest:
                raise RuntimeError(
                    f"package contains duplicate-unequal member: {path}:{member.name}"
                )
            hashes[member.name] = digest
    if not hashes:
        raise RuntimeError(f"package contains no files: {path}")
    return hashes


def validate_package_union(paths: list[Path]) -> None:
    """Reject overlapping packages when the same path has unequal bytes."""
    seen: dict[str, tuple[str, Path]] = {}
    for path in paths:
        for name, digest in tar_file_hashes(path).items():
            previous = seen.get(name)
            if previous is not None and previous[0] != digest:
                raise RuntimeError(
                    f"duplicate-unequal package member {name}: {previous[1]} != {path}"
                )
            seen[name] = (digest, path)


def deterministic_tar(output: Path, files: list[Path]) -> Path:
    """Create one immutable tar whose paths are rooted at filesystem '/'."""
    existing = sorted(set(path.resolve() for path in files))
    for path in existing:
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"cannot package missing or unsafe input: {path}")
    identity = sha256_bytes(
        "".join(f"{path}\0{sha256_file(path)}\n" for path in existing).encode()
    )
    destination = output.with_name(f"{output.stem}-{identity[:16]}{output.suffix}")
    if destination.exists():
        tar_file_hashes(destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with tarfile.open(temporary, "w") as archive:
            for path in existing:
                arcname = path.relative_to("/").as_posix()
                info = archive.gettarinfo(str(path), arcname=arcname)
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                with path.open("rb") as source:
                    archive.addfile(info, source)
        tar_file_hashes(temporary)
        if destination.exists():
            if destination.read_bytes() != temporary.read_bytes():
                raise RuntimeError(f"immutable package drift: {destination}")
        else:
            os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def workspace_seed_files(project_root: Path, run_id: str) -> list[Path]:
    """Collect the training-only bytes needed to recreate a sealed workspace."""
    root = study_root(project_root, run_id)
    manifest = read_json(root / "manifest.redteam.v1.json")
    request = read_json(root / "proposal-request.v1.json")
    files = [
        root / "manifest.redteam.v1.json",
        root / "config.v1.json",
        root / "proposal-request.v1.json",
        root / "inputs/embedding-sd.v1.json",
        root / "inputs/proposal-datasets.v1.json",
        root / "inputs/proposals.v1.jsonl",
        Path(manifest["parent"]["manifest_path"]),
        Path(manifest["parent"]["validation_path"]),
        Path(manifest["proposal_bundle"]["path"]),
        Path(manifest["scaling_statistic_path"]),
        Path(request["checkpoint_path"]),
        Path(request["catalog_pointer_path"]),
        Path(request["proposal_dataset_manifest_path"]),
        Path(request["scaling_statistic_path"]),
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
    return sorted(set(files))


def ensure_workspace_seed(
    project_root: Path,
    output: Path,
    run_id: str,
) -> Path:
    return deterministic_tar(
        output / "inputs" / f"{run_id}-training-workspace.tar",
        workspace_seed_files(project_root, run_id),
    )


def score_access_files(project_root: Path) -> list[Path]:
    root = study_root(project_root)
    manifest = read_json(root / "manifest.redteam.v1.json")
    parent_root = Path(manifest["parent"]["manifest_path"]).parent
    sotarena = Path("/home/ubuntu/sotarena")
    files = [
        parent_root / "baselines.v2.json",
        sotarena / "results/tabular-regression-v1/jope-dime-evolution/index.v1.json",
    ]
    files.extend(
        sotarena / "data/tabular-regression-v1" / dataset["dataset_hash"] / "test.csv"
        for dataset in manifest["datasets"]
    )
    return files


def state_path(output: Path) -> Path:
    return output / "controller-state.v1.json"


def load_or_create_state(
    output: Path,
    tag: str,
    source_bundle: Path,
    full_seed: Path,
    smoke_seed: Path,
) -> dict[str, Any]:
    path = state_path(output)
    if path.exists():
        state = read_json(path)
        if state.get("tag") != tag:
            raise RuntimeError("controller output already belongs to a different tag")
        expected = {
            "source_bundle": sha256_file(source_bundle),
            "full_seed": sha256_file(full_seed),
            "smoke_seed": sha256_file(smoke_seed),
        }
        if state.get("controller_inputs_sha256") != expected:
            raise RuntimeError(
                "controller input hashes drifted after launch state creation"
            )
        return state
    state = {
        "schema": "TrainOnlyRedteamSCQControllerState.v1",
        "run_id": RUN_ID,
        "project": PROJECT,
        "tag": tag,
        "controller_inputs": {
            "source_bundle": str(source_bundle),
            "full_seed": str(full_seed),
            "smoke_seed": str(smoke_seed),
        },
        "controller_inputs_sha256": {
            "source_bundle": sha256_file(source_bundle),
            "full_seed": sha256_file(full_seed),
            "smoke_seed": sha256_file(smoke_seed),
        },
        "phases": {phase: {"jobs": {}} for phase in PHASE_ORDER},
        "development_only": True,
        "confirmatory_valid": False,
    }
    atomic_json(path, state)
    return state


def logical_job(
    state: dict[str, Any],
    phase: str,
    logical_id: str,
) -> dict[str, Any] | None:
    return state["phases"][phase]["jobs"].get(logical_id)


def completed_attempt(
    state: dict[str, Any],
    phase: str,
    logical_id: str,
) -> dict[str, Any]:
    logical = logical_job(state, phase, logical_id)
    attempts = [] if logical is None else logical.get("attempts", [])
    completed = [
        attempt
        for attempt in attempts
        if attempt.get("state") == "completed" and attempt.get("imported_output_sha256")
    ]
    if len(completed) != 1:
        raise RuntimeError(
            f"upstream output is not uniquely complete: {phase}/{logical_id}"
        )
    return completed[0]


def imported_root(
    state: dict[str, Any],
    phase: str,
    logical_id: str,
) -> Path:
    return Path(completed_attempt(state, phase, logical_id)["imported_path"])


def imported_file(
    state: dict[str, Any],
    phase: str,
    logical_id: str,
    name: str,
) -> Path:
    return unique_package_file(imported_root(state, phase, logical_id), name)


def upstream_hash(
    state: dict[str, Any],
    phase: str,
    logical_id: str,
) -> str:
    return completed_attempt(state, phase, logical_id)["imported_output_sha256"]


def phase_is_imported(
    state: dict[str, Any], phase: str, logical_ids: list[str]
) -> bool:
    return all(
        logical_job(state, phase, logical_id)
        and any(
            attempt.get("state") == "completed"
            and attempt.get("imported_output_sha256")
            for attempt in logical_job(state, phase, logical_id)["attempts"]
        )
        for logical_id in logical_ids
    )


def revalidate_preflight(
    project_root: Path,
    output: Path,
    node_snapshot: dict[str, Any],
    source_bundle: Path,
    release_binary: Path,
    full_seed: Path,
    smoke_seed: Path,
) -> Path:
    """Recompute every legacy preflight binding before the first submission."""
    root = study_root(project_root)
    receipt_path = root / "scq/preflight-receipt.v1.json"
    receipt = read_json(receipt_path)
    manifest = read_json(root / "manifest.redteam.v1.json")
    request = read_json(root / "proposal-request.v1.json")
    current: dict[str, Path] = {
        "manifest": root / "manifest.redteam.v1.json",
        "config": root / "config.v1.json",
        "proposal_request": root / "proposal-request.v1.json",
        "parent_manifest": Path(manifest["parent"]["manifest_path"]),
        "parent_validation": Path(manifest["parent"]["validation_path"]),
        "proposal_bundle": Path(manifest["proposal_bundle"]["path"]),
        "scaling_statistic": Path(manifest["scaling_statistic_path"]),
        "checkpoint": Path(request["checkpoint_path"]),
        "context_catalog_pointer": Path(request["catalog_pointer_path"]),
        "proposal_dataset_manifest": Path(request["proposal_dataset_manifest_path"]),
    }
    for dataset in manifest["datasets"]:
        dataset_id = dataset["dataset_hash"]
        current[f"{dataset_id}:train"] = Path(dataset["train_path"])
        current[f"{dataset_id}:fold_bank"] = Path(dataset["fold_bank_path"])
        current[f"{dataset_id}:transfer"] = Path(dataset["transfer_ranking_path"])
        for index, parent in enumerate(dataset["parent_generation_paths"]):
            current[f"{dataset_id}:parent_generation_{index:03}"] = Path(parent)
    recorded = receipt.get("input_sha256", {})
    if set(recorded) != set(current):
        raise RuntimeError("existing preflight receipt input inventory drift")
    for name, path in current.items():
        row = recorded[name]
        if row.get("path") != str(path) or row.get("sha256") != sha256_file(path):
            raise RuntimeError(f"existing preflight receipt drift: {name}")
    pilot_hash = tree_hash(root / "datasets/ee56a1015e344c11")
    if receipt.get("pilot_tree_sha256") != pilot_hash:
        raise RuntimeError("existing preflight pilot tree hash drift")
    if receipt.get("development_only") is not True or receipt.get("confirmatory_valid"):
        raise RuntimeError("existing preflight disclosure drift")
    revalidation = {
        "schema": "TrainOnlyRedteamSCQPreflightRevalidation.v1",
        "run_id": RUN_ID,
        "preflight_receipt_sha256": sha256_file(receipt_path),
        "source_bundle_sha256": sha256_file(source_bundle),
        "release_binary_sha256": sha256_file(release_binary),
        "full_workspace_seed_sha256": sha256_file(full_seed),
        "smoke_workspace_seed_sha256": sha256_file(smoke_seed),
        "pilot_tree_sha256": pilot_hash,
        "nodes": node_snapshot,
        "development_only": True,
        "confirmatory_valid": False,
    }
    path = output / "preflight-revalidation.v1.json"
    immutable_json(path, revalidation)
    return path


def runtime_inputs(
    project_root: Path,
    state: dict[str, Any],
    *,
    proposal_bundle: Path | None = None,
) -> list[dict[str, Any]]:
    release = imported_file(state, "build", "build-xbabe2", "jope-dime-regressor")
    inputs = [
        input_pin(
            "release_binary",
            release,
            "jope-dime-regressor",
            executable=True,
        ),
        input_pin(
            "source_bundle",
            Path(state["controller_inputs"]["source_bundle"]),
            "source-bundle.tar",
        ),
        input_pin(
            "manifest",
            study_root(project_root) / "manifest.redteam.v1.json",
            "manifest.redteam.v1.json",
        ),
    ]
    if proposal_bundle is not None:
        inputs.append(
            input_pin("proposal_bundle", proposal_bundle, "proposals.v1.jsonl")
        )
    return inputs


def mirror_setup(skeleton_input: str) -> str:
    return (
        "mirror=$(mktemp -d); "
        f'tar -xf {{input:{skeleton_input}}} -C "$mirror"; '
        'studies="$mirror/home/ubuntu/jope-dime/runs/train-only-studies"; '
        'sotarena="$mirror/home/ubuntu/sotarena"; '
        'test -d "$studies"; test -d "$sotarena"; '
    )


def mirrored_binary(command: str, *, aggregate_only: bool = False) -> str:
    relocated = command.replace(
        "/home/ubuntu/jope-dime/runs/train-only-studies", '"$studies"'
    ).replace("/home/ubuntu/sotarena", '"$sotarena"')
    aggregate = " REDTEAM_AGGREGATE_ONLY=true" if aggregate_only else ""
    return (
        f'REDTEAM_PATH_ROOT="$mirror"{aggregate} {{input:release_binary}} {relocated}'
    )


def sealed_tar_command(source: str, destination: str) -> str:
    return f'tar -cf {destination} -C "$mirror" {quote(source.lstrip("/"))}'


def logical_ids_for_phase(phase: str) -> list[str]:
    if phase == "build":
        return ["build-xbabe2"]
    if phase == "tests":
        return [f"tests-{node}" for node in NODES]
    if phase == "prepare":
        return ["prepare-full", "prepare-smoke"]
    if phase == "proposals":
        return [
            *(f"proposal-exact-{seed}-{node}" for node, seed in EXACT),
            *(
                f"proposal-jitter-{perturbation}-{node}"
                for node, _, perturbation in JITTER
            ),
        ]
    if phase == "aggregate-proposals":
        return ["aggregate-proposals"]
    if phase == "smoke":
        return [f"smoke-{node}" for node in NODES]
    if phase == "full":
        return [f"full-{dataset}" for node in NODES for dataset in OWNERSHIP[node]]
    if phase == "validate":
        return [f"validate-{dataset}" for node in NODES for dataset in OWNERSHIP[node]]
    if phase in {"aggregate", "score", "report"}:
        return [phase]
    raise RuntimeError(f"unknown phase: {phase}")


def jobs_for_phase(
    project_root: Path,
    output: Path,
    state: dict[str, Any],
    phase: str,
    tag: str,
) -> list[dict[str, Any]]:
    """Render one phase only after all of its imported dependencies exist."""
    source_bundle = Path(state["controller_inputs"]["source_bundle"])
    root = study_root(project_root)
    manifest = read_json(root / "manifest.redteam.v1.json")
    jobs: list[dict[str, Any]] = []
    if phase == "build":
        jobs.append(
            {
                "logical_id": "build-xbabe2",
                "id": job_id("build", "xbabe2", tag),
                "node": "xbabe2",
                "command": (
                    'work=$(mktemp -d); tar -xf {input:source_bundle} -C "$work"; '
                    'cd "$work"; cargo fmt --all -- --check; '
                    "cargo test --locked train_only_redteam -- --nocapture; "
                    "cargo build --release --locked; "
                    'out={output:build-receipts}; mkdir -p "$out"; '
                    'cp target/release/jope-dime-regressor "$out/jope-dime-regressor"; '
                    "sha256sum target/release/jope-dime-regressor "
                    '> "$out/release-binary.sha256"'
                ),
                "inputs": [
                    input_pin("source_bundle", source_bundle, "source-bundle.tar")
                ],
                "outputs": [("build-receipts", "build-receipts", True)],
                "upstream_package_sha256": {},
            }
        )
        return jobs

    if not phase_is_imported(state, "build", ["build-xbabe2"]):
        raise RuntimeError(f"{phase} cannot render before the build output is imported")
    release_hash = upstream_hash(state, "build", "build-xbabe2")
    base = runtime_inputs(project_root, state)

    if phase == "tests":
        for node in NODES:
            python_test = (
                "; PYTHONPATH=scripts:/home/ubuntu/jope-prime/src "
                "python3 -m unittest scripts/test_train_only_redteam_generation.py"
                if node != "xbabe4"
                else ""
            )
            jobs.append(
                {
                    "logical_id": f"tests-{node}",
                    "id": job_id("tests", node, tag),
                    "node": node,
                    "command": (
                        'work=$(mktemp -d); tar -xf {input:source_bundle} -C "$work"; '
                        'cd "$work"; {input:release_binary} --help >/dev/null; '
                        "cargo test --locked train_only_redteam -- --nocapture"
                        f"{python_test}; "
                        'out={output:test-receipts}; mkdir -p "$out"; '
                        f"printf '%s\\n' {quote(node)} > \"$out/node.txt\""
                    ),
                    "inputs": base,
                    "outputs": [("test-receipts", "test-receipts", True)],
                    "python": node != "xbabe4",
                    "upstream_package_sha256": {"build": release_hash},
                }
            )
        return jobs

    if phase == "prepare":
        if not phase_is_imported(state, "tests", logical_ids_for_phase("tests")):
            raise RuntimeError(
                "prepare cannot render before all node tests are imported"
            )
        for logical_id, seed_key in (
            ("prepare-full", "full_seed"),
            ("prepare-smoke", "smoke_seed"),
        ):
            seed = Path(state["controller_inputs"][seed_key])
            jobs.append(
                {
                    "logical_id": logical_id,
                    "id": job_id("prepare", logical_id.removeprefix("prepare-"), tag),
                    "node": "xbabe4",
                    "command": (
                        "{input:release_binary} --help >/dev/null; "
                        'out={output:study-skeleton}; mkdir -p "$out"; '
                        'cp {input:workspace_seed} "$out/study-skeleton.tar"; '
                        'sha256sum {input:workspace_seed} > "$out/study-skeleton.sha256"'
                    ),
                    "inputs": [
                        *base,
                        input_pin("workspace_seed", seed, "workspace-seed.tar"),
                    ],
                    "outputs": [("study-skeleton", "study-skeleton", True)],
                    "upstream_package_sha256": {
                        "build": release_hash,
                        **{
                            logical: upstream_hash(state, "tests", logical)
                            for logical in logical_ids_for_phase("tests")
                        },
                    },
                }
            )
        return jobs

    if phase == "proposals":
        if not phase_is_imported(state, "prepare", logical_ids_for_phase("prepare")):
            raise RuntimeError("proposals cannot render before preparation is imported")
        request = read_json(root / "proposal-request.v1.json")
        proposal_base = [
            *base,
            input_pin(
                "generator_script",
                project_root / "scripts/test_exact_dl_generate.py",
                "test_exact_dl_generate.py",
            ),
            input_pin(
                "checkpoint",
                Path(request["checkpoint_path"]),
                "checkpoint.model.pt",
            ),
            input_pin(
                "context_catalog",
                Path(request["catalog_pointer_path"]),
                "context-catalog-pointer.json",
            ),
            input_pin(
                "proposal_metadata",
                Path(request["proposal_dataset_manifest_path"]),
                "proposal-datasets.v1.json",
            ),
            input_pin(
                "scaling_statistic",
                Path(request["scaling_statistic_path"]),
                "embedding-sd.v1.json",
            ),
        ]
        for node, seed in EXACT:
            filename = f"exact-{seed}-{node}.jsonl"
            jobs.append(
                {
                    "logical_id": f"proposal-exact-{seed}-{node}",
                    "id": job_id("proposals", f"exact-{seed}-{node}", tag),
                    "node": node,
                    "command": (
                        "{input:release_binary} --help >/dev/null; "
                        'out={output:proposal-package}; mkdir -p "$out"; '
                        "PYTHONPATH=scripts:/home/ubuntu/jope-prime/src python3 "
                        "{input:generator_script} --checkpoint {input:checkpoint} "
                        "--catalog-pointer {input:context_catalog} "
                        "--dataset-manifest {input:proposal_metadata} --task regression "
                        f"--proposals-per-dataset 64 --seed {seed} --device cuda "
                        f'--output "$out/{filename}" '
                        "--proposal-origin exact_context --redteam-metadata"
                    ),
                    "inputs": proposal_base,
                    "outputs": [("proposal-package", "proposal-package", True)],
                    "gpus": 1,
                    "python": True,
                    "upstream_package_sha256": {
                        "build": release_hash,
                        "prepare-full": upstream_hash(state, "prepare", "prepare-full"),
                    },
                }
            )
        for node, seed, perturbation in JITTER:
            filename = f"jitter-{perturbation}-{node}.jsonl"
            jobs.append(
                {
                    "logical_id": f"proposal-jitter-{perturbation}-{node}",
                    "id": job_id("proposals", f"jitter-{perturbation}-{node}", tag),
                    "node": node,
                    "command": (
                        "{input:release_binary} --help >/dev/null; "
                        'out={output:proposal-package}; mkdir -p "$out"; '
                        "PYTHONPATH=scripts:/home/ubuntu/jope-prime/src python3 "
                        "{input:generator_script} --checkpoint {input:checkpoint} "
                        "--catalog-pointer {input:context_catalog} "
                        "--dataset-manifest {input:proposal_metadata} --task regression "
                        f"--proposals-per-dataset 64 --seed {seed} --device cuda "
                        f'--output "$out/{filename}" '
                        "--proposal-origin jittered_context --redteam-metadata "
                        "--embedding-sd {input:scaling_statistic} "
                        "--embedding-jitter-scale 0.05 "
                        f"--perturbation-seed {perturbation}"
                    ),
                    "inputs": proposal_base,
                    "outputs": [("proposal-package", "proposal-package", True)],
                    "gpus": 1,
                    "python": True,
                    "upstream_package_sha256": {
                        "build": release_hash,
                        "prepare-full": upstream_hash(state, "prepare", "prepare-full"),
                    },
                }
            )
        return jobs

    if phase == "aggregate-proposals":
        proposal_ids = logical_ids_for_phase("proposals")
        if not phase_is_imported(state, "proposals", proposal_ids):
            raise RuntimeError(
                "proposal aggregation cannot render before all seven shards are imported"
            )
        inputs = [
            *base,
            input_pin(
                "manifest_proposal_bundle",
                Path(manifest["proposal_bundle"]["path"]),
                "manifest-proposals.v1.jsonl",
                manifest["proposal_bundle"]["sha256"],
            ),
        ]
        ordered_names: list[str] = []
        for index, (node, seed) in enumerate(EXACT):
            name = f"proposal-shard-{index}"
            ordered_names.append(name)
            inputs.append(
                input_pin(
                    name,
                    imported_file(
                        state,
                        "proposals",
                        f"proposal-exact-{seed}-{node}",
                        f"exact-{seed}-{node}.jsonl",
                    ),
                    f"proposal-shards/{index:02}.jsonl",
                )
            )
        for index, (node, _, perturbation) in enumerate(JITTER, start=len(EXACT)):
            name = f"proposal-shard-{index}"
            ordered_names.append(name)
            inputs.append(
                input_pin(
                    name,
                    imported_file(
                        state,
                        "proposals",
                        f"proposal-jitter-{perturbation}-{node}",
                        f"jitter-{perturbation}-{node}.jsonl",
                    ),
                    f"proposal-shards/{index:02}.jsonl",
                )
            )
        cat_inputs = " ".join(f"{{input:{name}}}" for name in ordered_names)
        jobs.append(
            {
                "logical_id": "aggregate-proposals",
                "id": job_id("aggregate-proposals", "xbabe4", tag),
                "node": "xbabe4",
                "command": (
                    "{input:release_binary} --help >/dev/null; "
                    'out={output:proposal-bundle}; mkdir -p "$out"; '
                    f'cat {cat_inputs} > "$out/proposals.candidate.jsonl"; '
                    'cmp -s "$out/proposals.candidate.jsonl" '
                    "{input:manifest_proposal_bundle}; "
                    'cp "$out/proposals.candidate.jsonl" "$out/proposals.v1.jsonl"; '
                    'sha256sum "$out/proposals.v1.jsonl" > "$out/proposals.sha256"'
                ),
                "inputs": inputs,
                "outputs": [("proposal-bundle", "proposal-bundle", True)],
                "upstream_package_sha256": {
                    logical: upstream_hash(state, "proposals", logical)
                    for logical in proposal_ids
                },
            }
        )
        return jobs

    if not phase_is_imported(state, "aggregate-proposals", ["aggregate-proposals"]):
        raise RuntimeError(
            f"{phase} cannot render before proposal parity and import succeed"
        )
    authoritative_proposals = imported_file(
        state,
        "aggregate-proposals",
        "aggregate-proposals",
        "proposals.v1.jsonl",
    )
    proposal_hash = upstream_hash(state, "aggregate-proposals", "aggregate-proposals")

    if phase == "smoke":
        skeleton = imported_file(
            state, "prepare", "prepare-smoke", "study-skeleton.tar"
        )
        for node in NODES:
            invocations = []
            for dataset in OWNERSHIP[node]:
                for arm in ("g20-extension", "smart-v3"):
                    invocations.append(
                        mirrored_binary(
                            f"--dataset {dataset} run-train-only-redteam "
                            f"--run-id {quote(SMOKE_RUN_ID)} "
                            "--study-root /home/ubuntu/jope-dime/runs/train-only-studies "
                            f"--arm {arm} --resume --workers 8"
                        )
                    )
            jobs.append(
                {
                    "logical_id": f"smoke-{node}",
                    "id": job_id("smoke", node, tag),
                    "node": node,
                    "command": (
                        mirror_setup("preparation_skeleton")
                        + f'cmp -s {{input:proposal_bundle}} "$studies/{SMOKE_RUN_ID}/inputs/proposals.v1.jsonl"; '
                        + "; ".join(invocations)
                        + '; out={output:smoke-receipts}; mkdir -p "$out"; '
                        + f"printf '%s\\n' {quote(node)} > \"$out/smoke.receipt\""
                    ),
                    "inputs": [
                        *runtime_inputs(
                            project_root,
                            state,
                            proposal_bundle=authoritative_proposals,
                        ),
                        input_pin(
                            "preparation_skeleton",
                            skeleton,
                            "smoke-skeleton.tar",
                        ),
                    ],
                    "outputs": [("smoke-receipts", "smoke-receipts", True)],
                    "upstream_package_sha256": {
                        "prepare-smoke": upstream_hash(
                            state, "prepare", "prepare-smoke"
                        ),
                        "proposal-bundle": proposal_hash,
                    },
                }
            )
        return jobs

    if phase == "full":
        if not phase_is_imported(state, "smoke", logical_ids_for_phase("smoke")):
            raise RuntimeError("full execution cannot render before all smoke receipts")
        skeleton = imported_file(state, "prepare", "prepare-full", "study-skeleton.tar")
        for node in NODES:
            for dataset_id in OWNERSHIP[node]:
                dataset = next(
                    row
                    for row in manifest["datasets"]
                    if row["dataset_hash"] == dataset_id
                )
                command = (
                    mirror_setup("preparation_skeleton")
                    + f'cmp -s {{input:proposal_bundle}} "$studies/{RUN_ID}/inputs/proposals.v1.jsonl"; '
                    + mirrored_binary(
                        f"--dataset {dataset_id} run-train-only-redteam "
                        f"--run-id {quote(RUN_ID)} "
                        "--study-root /home/ubuntu/jope-dime/runs/train-only-studies "
                        "--arm g20-extension --resume --workers 8"
                    )
                    + "; "
                    + mirrored_binary(
                        f"--dataset {dataset_id} run-train-only-redteam "
                        f"--run-id {quote(RUN_ID)} "
                        "--study-root /home/ubuntu/jope-dime/runs/train-only-studies "
                        "--arm smart-v3 --resume --workers 8"
                    )
                    + '; out={output:dataset-artifacts}; mkdir -p "$out"; '
                    + sealed_tar_command(
                        f"/home/ubuntu/jope-dime/runs/train-only-studies/{RUN_ID}/datasets/{dataset_id}",
                        '"$out/dataset-package.tar"',
                    )
                )
                inputs = [
                    *runtime_inputs(
                        project_root,
                        state,
                        proposal_bundle=authoritative_proposals,
                    ),
                    input_pin(
                        "preparation_skeleton",
                        skeleton,
                        "full-skeleton.tar",
                    ),
                    input_pin(
                        "fold_bank",
                        Path(dataset["fold_bank_path"]),
                        f"fold-banks/{dataset_id}.v1.json",
                        dataset["fold_bank_sha256"],
                    ),
                ]
                for index, parent in enumerate(dataset["parent_generation_paths"]):
                    inputs.append(
                        input_pin(
                            f"parent_generation_{index:03}",
                            Path(parent),
                            f"parent/{dataset_id}/generation-{index:03}.json",
                            dataset["parent_generation_sha256"][index],
                        )
                    )
                inputs.extend(
                    [
                        input_pin(
                            "parent_manifest",
                            Path(manifest["parent"]["manifest_path"]),
                            "parent/manifest.v2.json",
                            manifest["parent"]["manifest_sha256"],
                        ),
                        input_pin(
                            "parent_validation",
                            Path(manifest["parent"]["validation_path"]),
                            "parent/validation-success.v2.json",
                            manifest["parent"]["validation_sha256"],
                        ),
                    ]
                )
                jobs.append(
                    {
                        "logical_id": f"full-{dataset_id}",
                        "id": job_id("full", dataset_id, tag),
                        "node": node,
                        "command": command,
                        "inputs": inputs,
                        "outputs": [("dataset-artifacts", "dataset-artifacts", True)],
                        "output_package_sha256": proposal_hash,
                        "upstream_package_sha256": {
                            "prepare-full": upstream_hash(
                                state, "prepare", "prepare-full"
                            ),
                            "proposal-bundle": proposal_hash,
                            **{
                                logical: upstream_hash(state, "smoke", logical)
                                for logical in logical_ids_for_phase("smoke")
                            },
                        },
                    }
                )
        return jobs

    if phase == "validate":
        full_ids = logical_ids_for_phase("full")
        if not phase_is_imported(state, "full", full_ids):
            raise RuntimeError("validation cannot render before all dataset packages")
        skeleton = imported_file(state, "prepare", "prepare-full", "study-skeleton.tar")
        for node in NODES:
            for dataset_id in OWNERSHIP[node]:
                dataset_package = imported_file(
                    state,
                    "full",
                    f"full-{dataset_id}",
                    "dataset-package.tar",
                )
                tar_file_hashes(dataset_package)
                full_hash = upstream_hash(state, "full", f"full-{dataset_id}")
                jobs.append(
                    {
                        "logical_id": f"validate-{dataset_id}",
                        "id": job_id("validate", dataset_id, tag),
                        "node": node,
                        "command": (
                            mirror_setup("preparation_skeleton")
                            + 'tar -xf {input:dataset_package} -C "$mirror"; '
                            + mirrored_binary(
                                f"--dataset {dataset_id} validate-train-only-redteam "
                                f"--run-id {quote(RUN_ID)} "
                                "--study-root /home/ubuntu/jope-dime/runs/train-only-studies "
                                "--sotarena-root /home/ubuntu/sotarena --workers 8"
                            )
                            + '; out={output:validation-shard}; mkdir -p "$out"; '
                            + f'cp "$studies/{RUN_ID}/datasets/{dataset_id}/validation-shard.v1.json" '
                            + f'"$out/{dataset_id}.validation-shard.v1.json"; '
                            + sealed_tar_command(
                                f"/home/ubuntu/jope-dime/runs/train-only-studies/{RUN_ID}/datasets/{dataset_id}",
                                '"$out/validation-package.tar"',
                            )
                        ),
                        "inputs": [
                            *runtime_inputs(
                                project_root,
                                state,
                                proposal_bundle=authoritative_proposals,
                            ),
                            input_pin(
                                "preparation_skeleton",
                                skeleton,
                                "full-skeleton.tar",
                            ),
                            input_pin(
                                "dataset_package",
                                dataset_package,
                                f"datasets/{dataset_id}.tar",
                            ),
                        ],
                        "outputs": [("validation-shard", "validation-shard", True)],
                        "output_package_sha256": full_hash,
                        "upstream_package_sha256": {f"full-{dataset_id}": full_hash},
                    }
                )
        return jobs

    if phase == "aggregate":
        validate_ids = logical_ids_for_phase("validate")
        if not phase_is_imported(state, "validate", validate_ids):
            raise RuntimeError("global aggregation requires all ten validations")
        skeleton = imported_file(state, "prepare", "prepare-full", "study-skeleton.tar")
        inputs = [
            *runtime_inputs(
                project_root, state, proposal_bundle=authoritative_proposals
            ),
            input_pin("preparation_skeleton", skeleton, "full-skeleton.tar"),
        ]
        package_names: list[str] = []
        show_names: list[str] = []
        packages: list[Path] = []
        upstream: dict[str, str] = {}
        for node in NODES:
            for dataset_id in OWNERSHIP[node]:
                full_logical = f"full-{dataset_id}"
                validation_logical = f"validate-{dataset_id}"
                for prefix, source_phase, logical, filename in (
                    ("dataset", "full", full_logical, "dataset-package.tar"),
                    (
                        "validation",
                        "validate",
                        validation_logical,
                        "validation-package.tar",
                    ),
                ):
                    name = f"{prefix}-{dataset_id}"
                    path = imported_file(state, source_phase, logical, filename)
                    packages.append(path)
                    package_names.append(name)
                    inputs.append(input_pin(name, path, f"packages/{name}.tar"))
                    upstream[logical] = upstream_hash(state, source_phase, logical)
                for source_phase, logical in (
                    ("full", full_logical),
                    ("validate", validation_logical),
                ):
                    attempt = completed_attempt(state, source_phase, logical)
                    name = f"scq-show-{logical}"
                    show_names.append(name)
                    inputs.append(
                        input_pin(
                            name,
                            Path(attempt["show_path"]),
                            f"scq-show/{logical}.json",
                        )
                    )
        validate_package_union(packages)
        extraction = " ".join(
            f'tar -xf {{input:{name}}} -C "$mirror";' for name in package_names
        )
        show_copy = " ".join(
            f'cp {{input:{name}}} "$studies/{RUN_ID}/scq-show/{name.removeprefix("scq-show-")}.json";'
            for name in show_names
        )
        jobs.append(
            {
                "logical_id": "aggregate",
                "id": job_id("aggregate", "xbabe4", tag),
                "node": "xbabe4",
                "command": (
                    mirror_setup("preparation_skeleton")
                    + extraction
                    + f'mkdir -p "$studies/{RUN_ID}/scq-show"; '
                    + show_copy
                    + mirrored_binary(
                        f"run-train-only-redteam --run-id {quote(RUN_ID)} "
                        "--study-root /home/ubuntu/jope-dime/runs/train-only-studies "
                        "--arm g20-extension --resume --workers 8",
                        aggregate_only=True,
                    )
                    + "; "
                    + mirrored_binary(
                        f"run-train-only-redteam --run-id {quote(RUN_ID)} "
                        "--study-root /home/ubuntu/jope-dime/runs/train-only-studies "
                        "--arm smart-v3 --resume --workers 8",
                        aggregate_only=True,
                    )
                    + "; "
                    + mirrored_binary(
                        f"validate-train-only-redteam --run-id {quote(RUN_ID)} "
                        "--study-root /home/ubuntu/jope-dime/runs/train-only-studies "
                        "--sotarena-root /home/ubuntu/sotarena --workers 8"
                    )
                    + '; out={output:global-validation}; mkdir -p "$out"; '
                    + f'validation="$studies/{RUN_ID}/validation-success.v1.json"; '
                    + 'validation_sha=$(sha256sum "$validation" | cut -d " " -f1); '
                    + f'freeze="$studies/{RUN_ID}/SELECTION_FREEZE.v1.json"; '
                    + "jq -n --arg run_id "
                    + quote(RUN_ID)
                    + ' --arg validation_sha256 "$validation_sha" '
                    + '\'{schema:"TrainOnlyRedteamSelectionFreeze.v1",'
                    + "run_id:$run_id,validation_sha256:$validation_sha256,"
                    + "validated_dataset_count:10,selection_frozen_before_test_access:true,"
                    + "development_only:true,confirmatory_valid:false}' "
                    + ' > "$freeze"; '
                    + 'cp "$validation" "$out/validation-success.v1.json"; '
                    + 'cp "$freeze" "$out/SELECTION_FREEZE.v1.json"; '
                    + sealed_tar_command(
                        f"/home/ubuntu/jope-dime/runs/train-only-studies/{RUN_ID}",
                        '"$out/selection-freeze.tar"',
                    )
                ),
                "inputs": inputs,
                "outputs": [("global-validation", "global-validation", True)],
                "upstream_package_sha256": upstream,
            }
        )
        return jobs

    if phase == "score":
        if not phase_is_imported(state, "aggregate", ["aggregate"]):
            raise RuntimeError("score is forbidden before global validation import")
        validate_selection_freeze(
            imported_file(
                state,
                "aggregate",
                "aggregate",
                "SELECTION_FREEZE.v1.json",
            ),
            imported_file(
                state,
                "aggregate",
                "aggregate",
                "validation-success.v1.json",
            ),
        )
        freeze_package = imported_file(
            state, "aggregate", "aggregate", "selection-freeze.tar"
        )
        access = deterministic_tar(
            output / "inputs/score-access.tar",
            score_access_files(project_root),
        )
        jobs.append(
            {
                "logical_id": "score",
                "id": job_id("score", "xbabe4", tag),
                "node": "xbabe4",
                "command": (
                    mirror_setup("selection_freeze")
                    + 'tar -xf {input:score_access} -C "$mirror"; '
                    + mirrored_binary(
                        f"score-train-only-redteam --run-id {quote(RUN_ID)} "
                        "--study-root /home/ubuntu/jope-dime/runs/train-only-studies "
                        "--sotarena-root /home/ubuntu/sotarena --workers 8"
                    )
                    + '; out={output:score-receipts}; mkdir -p "$out"; '
                    + f'cp "$studies/{RUN_ID}/scores.v1.json" "$out/scores.v1.json"; '
                    + sealed_tar_command(
                        f"/home/ubuntu/jope-dime/runs/train-only-studies/{RUN_ID}",
                        '"$out/score-package.tar"',
                    )
                ),
                "inputs": [
                    input_pin(
                        "release_binary",
                        imported_file(
                            state,
                            "build",
                            "build-xbabe2",
                            "jope-dime-regressor",
                        ),
                        "jope-dime-regressor",
                        executable=True,
                    ),
                    input_pin(
                        "selection_freeze",
                        freeze_package,
                        "selection-freeze.tar",
                    ),
                    input_pin("score_access", access, "score-access.tar"),
                ],
                "outputs": [("score-receipts", "score-receipts", True)],
                "upstream_package_sha256": {
                    "selection-freeze": upstream_hash(state, "aggregate", "aggregate")
                },
            }
        )
        return jobs

    if phase == "report":
        if not phase_is_imported(state, "score", ["score"]):
            raise RuntimeError("report cannot render before the score package")
        score_package = imported_file(state, "score", "score", "score-package.tar")
        validate_score_package(score_package)
        jobs.append(
            {
                "logical_id": "report",
                "id": job_id("report", "xbabe4", tag),
                "node": "xbabe4",
                "command": (
                    mirror_setup("score_package")
                    + mirrored_binary(
                        f"report-train-only-redteam --run-id {quote(RUN_ID)} "
                        "--study-root /home/ubuntu/jope-dime/runs/train-only-studies "
                        "--sotarena-root /home/ubuntu/sotarena --workers 8"
                    )
                    + '; out={output:final-report}; mkdir -p "$out"; '
                    + f'cp "$studies/{RUN_ID}/report.v1.json" "$out/report.v1.json"; '
                    + f'cp "$studies/{RUN_ID}/report.v1.md" "$out/report.v1.md"; '
                    + "if jq -e '.development_strategy_improved == true' "
                    + f'"$studies/{RUN_ID}/report.v1.json" >/dev/null; then '
                    + "jq -n --arg run_id "
                    + quote(RUN_ID)
                    + '\'{schema:"TrainOnlyRedteamSealedV2Trigger.v1",'
                    + "run_id:$run_id,required:true,development_only:true,"
                    + 'confirmatory_valid:false}\' > "$out/SEALED_V2_TRIGGER.v1.json"; fi'
                ),
                "inputs": [
                    input_pin(
                        "release_binary",
                        imported_file(
                            state,
                            "build",
                            "build-xbabe2",
                            "jope-dime-regressor",
                        ),
                        "jope-dime-regressor",
                        executable=True,
                    ),
                    input_pin("score_package", score_package, "score-package.tar"),
                ],
                "outputs": [("final-report", "final-report", True)],
                "upstream_package_sha256": {
                    "score": upstream_hash(state, "score", "score")
                },
            }
        )
        return jobs

    raise RuntimeError(f"phase rendering is not implemented: {phase}")


def validate_selection_freeze(freeze_path: Path, validation_path: Path) -> None:
    freeze = read_json(freeze_path)
    validation = read_json(validation_path)
    if (
        freeze.get("schema") != "TrainOnlyRedteamSelectionFreeze.v1"
        or freeze.get("run_id") != RUN_ID
        or freeze.get("validation_sha256") != sha256_file(validation_path)
        or freeze.get("validated_dataset_count") != 10
        or freeze.get("selection_frozen_before_test_access") is not True
        or freeze.get("development_only") is not True
        or freeze.get("confirmatory_valid")
        or validation.get("schema") != "TrainOnlyRedteamGlobalValidation.v1"
        or validation.get("development_only") is not True
        or validation.get("confirmatory_valid")
        or len(validation.get("dataset_validation_sha256", {})) != 10
    ):
        raise RuntimeError("score-before-freeze rejection: selection freeze is invalid")


def validate_score_package(path: Path) -> None:
    hashes = tar_file_hashes(path)
    score_members = sorted(
        name
        for name in hashes
        if re.search(r"/datasets/[0-9a-f]{16}/scores\.v1\.json$", name)
    )
    if len(score_members) != 10:
        raise RuntimeError("score package does not contain ten dataset score receipts")
    with tarfile.open(path, "r") as archive:
        variants = 0
        for name in score_members:
            handle = archive.extractfile(name)
            if handle is None:
                raise RuntimeError(f"score package member is unreadable: {name}")
            receipt = json.loads(handle.read())
            if (
                receipt.get("development_only") is not True
                or receipt.get("confirmatory_valid")
                or receipt.get("selection_frozen_before_test_access") is not True
            ):
                raise RuntimeError(f"score receipt disclosure drift: {name}")
            variants += len(receipt.get("variants", []))
    if variants != 70:
        raise RuntimeError(f"score package has {variants} frozen scores, expected 70")


def scientific_job_hash(job: dict[str, Any]) -> str:
    value = {
        key: job[key]
        for key in (
            "logical_id",
            "node",
            "command",
            "inputs",
            "outputs",
            "gpus",
            "cpus",
            "ram",
            "python",
            "output_package_sha256",
            "upstream_package_sha256",
        )
        if key in job
    }
    return sha256_bytes(canonical_json(value))


def attempt_job_id(base: str, retry_index: int) -> str:
    return base if retry_index == 0 else f"{base}-retry{retry_index}"


def write_attempt_spec(
    output: Path,
    phase: str,
    job: dict[str, Any],
    retry_index: int,
    predecessors: list[str],
    retry_reason: str | None,
) -> dict[str, Any]:
    current_id = attempt_job_id(job["id"], retry_index)
    body = render_spec(
        job_id=current_id,
        node=job["node"],
        command=job["command"],
        inputs=job["inputs"],
        outputs=job["outputs"],
        gpus=job.get("gpus", 0),
        cpus=job.get("cpus", 8),
        ram_gb=job.get("ram", 40),
        python=job.get("python", False),
        attempt=retry_index,
        output_package_sha256=job.get("output_package_sha256", ""),
    )
    spec_hash = sha256_bytes(body.encode())
    spec = output / "specs" / phase / f"{current_id}.toml"
    immutable_write(spec, body.encode())
    inputs = {item["name"]: item["sha256"] for item in job["inputs"]}
    meta = {
        "schema": "TrainOnlyRedteamSCQJobMeta.v2",
        "logical_id": job["logical_id"],
        "job_id": current_id,
        "phase": phase,
        "expected_node": job["node"],
        "attempt": retry_index,
        "spec_sha256": spec_hash,
        "complete_spec_hash": spec_hash,
        "scientific_job_sha256": scientific_job_hash(job),
        "input_sha256": inputs,
        "upstream_package_sha256": job.get("upstream_package_sha256", {}),
        "predecessor_job_ids": predecessors,
        "retry_reason": retry_reason,
        "imported_output_sha256": None,
        "node_marker_probe": marker(job["node"])[0],
        "workers": 8,
        "cpus_reserved": job.get("cpus", 8),
        "ram_gb": job.get("ram", 40),
        "non_preemptible": True,
        "max_runtime_hours": 24,
        "development_only": True,
        "confirmatory_valid": False,
    }
    immutable_json(spec.with_suffix(".meta.json"), meta)
    return {
        "job_id": current_id,
        "spec_path": str(spec),
        "meta_path": str(spec.with_suffix(".meta.json")),
        "state": "rendered",
        "retry_index": retry_index,
        "predecessor_job_ids": predecessors,
        "retry_reason": retry_reason,
        "scientific_job_sha256": meta["scientific_job_sha256"],
    }


def ensure_phase_attempts(
    output: Path,
    state: dict[str, Any],
    phase: str,
    jobs: list[dict[str, Any]],
) -> None:
    phase_jobs = state["phases"][phase]["jobs"]
    for job in jobs:
        logical = phase_jobs.get(job["logical_id"])
        if logical is None:
            attempt = write_attempt_spec(output, phase, job, 0, [], None)
            phase_jobs[job["logical_id"]] = {
                "scientific_job_sha256": scientific_job_hash(job),
                "attempts": [attempt],
            }
            continue
        if logical["scientific_job_sha256"] != scientific_job_hash(job):
            raise RuntimeError(
                f"scientific inputs drifted for retry lineage: {job['logical_id']}"
            )


def add_retry_attempt(
    output: Path,
    state: dict[str, Any],
    phase: str,
    job: dict[str, Any],
    reason: str,
) -> None:
    logical = state["phases"][phase]["jobs"][job["logical_id"]]
    retry_index = len(logical["attempts"])
    if retry_index > MAX_INFRASTRUCTURE_RETRIES:
        raise RuntimeError(
            f"infrastructure retry budget exhausted: {job['logical_id']}"
        )
    predecessors = [attempt["job_id"] for attempt in logical["attempts"]]
    attempt = write_attempt_spec(
        output,
        phase,
        job,
        retry_index,
        predecessors,
        reason,
    )
    if attempt["scientific_job_sha256"] != logical["scientific_job_sha256"]:
        raise RuntimeError("retry changed scientific inputs")
    logical["attempts"].append(attempt)


def current_attempt(logical: dict[str, Any]) -> dict[str, Any]:
    return logical["attempts"][-1]


def submission_record_path(output: Path, job_id_value: str) -> Path:
    return output / "submissions" / f"{job_id_value}.json"


def check_attempt_capacity(attempt: dict[str, Any]) -> bool:
    code, payload, output = scq_command_json("check", attempt["spec_path"])
    if code == 0:
        return True
    if is_capacity_failure(payload) or is_capacity_failure(output):
        return False
    raise RuntimeError(f"SCQ check failed for {attempt['job_id']}: {output}")


def submit_attempt(output: Path, attempt: dict[str, Any]) -> bool:
    existing = show_job(attempt["job_id"])
    record_path = submission_record_path(output, attempt["job_id"])
    meta = read_json(Path(attempt["meta_path"]))
    if existing is not None:
        if not record_path.is_file():
            raise RuntimeError(
                "refusing stale SCQ ID reuse without a launch1 submission record: "
                + attempt["job_id"]
            )
        record = read_json(record_path)
        if record.get("complete_spec_hash") != meta["complete_spec_hash"]:
            raise RuntimeError(
                f"stale SCQ ID has a different spec: {attempt['job_id']}"
            )
        attempt["state"] = existing["state"]
        return True
    if not check_attempt_capacity(attempt):
        return False
    code, payload, command_output = scq_command_json("submit", attempt["spec_path"])
    if code != 0:
        if is_capacity_failure(payload) or is_capacity_failure(command_output):
            return False
        raise RuntimeError(
            f"SCQ submit failed for {attempt['job_id']}: {command_output}"
        )
    submitted_id = payload.get("Submitted", {}).get("id")
    if submitted_id != attempt["job_id"]:
        raise RuntimeError(
            f"SCQ submission ID mismatch: {submitted_id} != {attempt['job_id']}"
        )
    immutable_json(
        record_path,
        {
            "schema": "TrainOnlyRedteamSCQSubmission.v2",
            "job_id": attempt["job_id"],
            "complete_spec_hash": meta["complete_spec_hash"],
            "scientific_job_sha256": meta["scientific_job_sha256"],
            "predecessor_job_ids": meta["predecessor_job_ids"],
            "retry_reason": meta["retry_reason"],
        },
    )
    attempt["state"] = "queued"
    return True


def capture_job_log(output: Path, phase: str, job_id_value: str) -> Path:
    completed = subprocess.run(
        [str(SCQ), "logs", job_id_value],
        check=False,
        text=True,
        capture_output=True,
    )
    data = (completed.stdout + completed.stderr).encode()
    path = output / "logs" / phase / f"{job_id_value}.log"
    immutable_write(path, data)
    return path


def fetch_completed_attempt(
    output: Path,
    phase: str,
    attempt: dict[str, Any],
    job: dict[str, Any],
    shown_payload: Any,
    shown_job: dict[str, Any],
) -> None:
    actual = shown_job.get("node")
    if actual != job["node"]:
        raise RuntimeError(
            f"SCQ placed {attempt['job_id']} on {actual}, expected {job['node']}"
        )
    show_path = output / "scq-show" / phase / f"{attempt['job_id']}.v1.json"
    immutable_json(show_path, shown_payload)
    fetch_tmp = output / "fetch-tmp"
    fetch_tmp.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=fetch_tmp) as directory:
        fetched = Path(directory) / "output"
        subprocess.run(
            [str(SCQ), "fetch", attempt["job_id"], "--output", str(fetched)],
            check=True,
        )
        validate_fetched_tree(fetched)
        destination = output / "imported" / phase / attempt["job_id"]
        immutable_import_tree(fetched, destination)
        validate_fetched_tree(destination)
    digest = tree_hash(destination)
    if attempt.get("imported_output_sha256") not in {None, "", digest}:
        raise RuntimeError(f"imported output hash drift: {attempt['job_id']}")
    attempt.update(
        {
            "state": "completed",
            "actual_node": actual,
            "scheduler_attempt": shown_job.get("attempt"),
            "show_path": str(show_path),
            "scq_show_sha256": sha256_file(show_path),
            "imported_path": str(destination),
            "imported_output_sha256": digest,
        }
    )
    meta = read_json(Path(attempt["meta_path"]))
    completion = {
        **meta,
        "schema": "TrainOnlyRedteamSCQJobCompletion.v2",
        "scheduler_reported_node": actual,
        "scheduler_attempt": shown_job.get("attempt"),
        "scq_show_sha256": sha256_file(show_path),
        "imported_path": str(destination),
        "imported_output_sha256": digest,
        "imported_without_overwrite": True,
    }
    immutable_json(
        output / "completions" / phase / f"{attempt['job_id']}.json",
        completion,
    )


def package_generations(imported: Path) -> list[int]:
    generations: set[int] = set()
    for package in imported.rglob("*.tar"):
        for name in tar_file_hashes(package):
            match = re.search(r"generation-(\d{3})", name)
            if match:
                generations.add(int(match.group(1)))
    return sorted(generations)


def dataset_from_logical(logical_id: str) -> str | None:
    match = re.search(r"(?:full|validate)-([0-9a-f]{16})$", logical_id)
    return match.group(1) if match else None


def write_progress_receipt(output: Path, state: dict[str, Any]) -> Path:
    rows = []
    for phase in PHASE_ORDER:
        for logical_id, logical in sorted(state["phases"][phase]["jobs"].items()):
            attempt = current_attempt(logical)
            imported = (
                Path(attempt["imported_path"]) if attempt.get("imported_path") else None
            )
            generations = package_generations(imported) if imported else []
            rows.append(
                {
                    "phase": phase,
                    "logical_id": logical_id,
                    "node": attempt.get("actual_node")
                    or read_json(Path(attempt["meta_path"]))["expected_node"],
                    "dataset": dataset_from_logical(logical_id),
                    "generations": generations,
                    "job_state": attempt["state"],
                    "attempt": attempt["retry_index"],
                    "job_id": attempt["job_id"],
                    "imported_package_sha256": attempt.get(
                        "imported_output_sha256", ""
                    ),
                }
            )
    receipt = {
        "schema": "TrainOnlyRedteamSCQProgress.v1",
        "run_id": RUN_ID,
        "tag": state["tag"],
        "jobs": rows,
        "development_only": True,
        "confirmatory_valid": False,
    }
    path = output / "progress.v1.json"
    atomic_json(path, receipt)
    return path


def validate_phase_imports(
    project_root: Path,
    state: dict[str, Any],
    phase: str,
) -> None:
    if phase == "build":
        binary_path = imported_file(state, phase, "build-xbabe2", "jope-dime-regressor")
        claimed = (
            unique_package_file(
                imported_root(state, phase, "build-xbabe2"),
                "release-binary.sha256",
            )
            .read_text()
            .split()[0]
        )
        if claimed != sha256_file(binary_path):
            raise RuntimeError("build output binary hash receipt drift")
    elif phase == "proposals":
        if len(logical_ids_for_phase(phase)) != 7:
            raise RuntimeError("sealed proposal cardinality drift")
        for logical_id in logical_ids_for_phase(phase):
            package = imported_root(state, phase, logical_id)
            if len(list(package.rglob("*.jsonl"))) != 1:
                raise RuntimeError(f"proposal package is incomplete: {logical_id}")
    elif phase == "aggregate-proposals":
        candidate = imported_file(
            state, phase, "aggregate-proposals", "proposals.candidate.jsonl"
        )
        authoritative = imported_file(
            state, phase, "aggregate-proposals", "proposals.v1.jsonl"
        )
        manifest = read_json(study_root(project_root) / "manifest.redteam.v1.json")
        bound = Path(manifest["proposal_bundle"]["path"])
        if (
            candidate.read_bytes() != bound.read_bytes()
            or authoritative.read_bytes() != bound.read_bytes()
        ):
            raise RuntimeError("proposal candidate parity failure")
    elif phase == "smoke":
        if len(logical_ids_for_phase(phase)) != 4:
            raise RuntimeError("smoke receipt cardinality drift")
        for logical_id in logical_ids_for_phase(phase):
            unique_package_file(
                imported_root(state, phase, logical_id), "smoke.receipt"
            )
    elif phase == "full":
        packages = [
            imported_file(state, phase, logical_id, "dataset-package.tar")
            for logical_id in logical_ids_for_phase(phase)
        ]
        if len(packages) != 10:
            raise RuntimeError("full phase does not contain ten dataset packages")
        for package in packages:
            tar_file_hashes(package)
    elif phase == "validate":
        for logical_id in logical_ids_for_phase(phase):
            package = imported_root(state, phase, logical_id)
            shard = unique_package_file(package, "*.validation-shard.v1.json")
            receipt = read_json(shard)
            if receipt.get("development_only") is not True or receipt.get(
                "confirmatory_valid"
            ):
                raise RuntimeError(f"validation shard disclosure drift: {shard}")
            tar_file_hashes(unique_package_file(package, "validation-package.tar"))
    elif phase == "aggregate":
        validate_selection_freeze(
            imported_file(state, phase, "aggregate", "SELECTION_FREEZE.v1.json"),
            imported_file(state, phase, "aggregate", "validation-success.v1.json"),
        )
        tar_file_hashes(
            imported_file(state, phase, "aggregate", "selection-freeze.tar")
        )
    elif phase == "score":
        validate_score_package(
            imported_file(state, phase, "score", "score-package.tar")
        )
    elif phase == "report":
        root = imported_root(state, phase, "report")
        report = read_json(unique_package_file(root, "report.v1.json"))
        markdown = unique_package_file(root, "report.v1.md").read_text()
        if (
            report.get("development_only") is not True
            or report.get("confirmatory_valid")
            or not report.get("fair_ladder")
            or not report.get("oracle_reference_ladder")
            or len(report.get("datasets", [])) != 10
            or "development_only=true" not in markdown
            or "confirmatory_valid=false" not in markdown
        ):
            raise RuntimeError("final report acceptance contract drift")


def supervise_phase(
    project_root: Path,
    output: Path,
    state: dict[str, Any],
    phase: str,
    tag: str,
    poll_seconds: int,
) -> None:
    jobs = jobs_for_phase(project_root, output, state, phase, tag)
    by_logical = {job["logical_id"]: job for job in jobs}
    ensure_phase_attempts(output, state, phase, jobs)
    atomic_json(state_path(output), state)
    while True:
        capacity_wait = False
        retry_jobs: list[tuple[dict[str, Any], str]] = []
        for logical_id in logical_ids_for_phase(phase):
            logical = state["phases"][phase]["jobs"][logical_id]
            if any(
                attempt.get("state") == "completed"
                and attempt.get("imported_output_sha256")
                for attempt in logical["attempts"]
            ):
                continue
            attempt = current_attempt(logical)
            if attempt["state"] == "rendered":
                if not submit_attempt(output, attempt):
                    capacity_wait = True
                    continue
            code, payload, command_output = scq_command_json("show", attempt["job_id"])
            if code != 0:
                raise RuntimeError(
                    f"SCQ show failed for {attempt['job_id']}: {command_output}"
                )
            job = payload.get("Job", {}).get("job")
            if job is None:
                raise RuntimeError(
                    f"submitted SCQ ledger entry disappeared: {attempt['job_id']}"
                )
            attempt["state"] = job["state"]
            attempt["actual_node"] = job.get("node")
            attempt["scheduler_attempt"] = job.get("attempt")
            if job["state"] == "completed":
                fetch_completed_attempt(
                    output,
                    phase,
                    attempt,
                    by_logical[logical_id],
                    payload,
                    job,
                )
            elif job["state"] in {"failed", "cancelled", "canceled"}:
                log_path = capture_job_log(output, phase, attempt["job_id"])
                attempt["log_path"] = str(log_path)
                reason = (
                    f"{job.get('failure_class') or 'unknown'}: "
                    f"{job.get('error_reason') or job.get('message') or 'no reason'}"
                )
                if (
                    job["state"] == "failed"
                    and is_infrastructure_failure(job)
                    and attempt["retry_index"] < MAX_INFRASTRUCTURE_RETRIES
                ):
                    retry_jobs.append((by_logical[logical_id], reason))
                else:
                    raise RuntimeError(
                        f"{phase} stopped on non-retryable failure "
                        f"{attempt['job_id']}: {reason}; log={log_path}"
                    )
            elif job["state"] not in ACTIVE_STATES:
                raise RuntimeError(
                    f"unexpected SCQ state for {attempt['job_id']}: {job['state']}"
                )
        for job, reason in retry_jobs:
            add_retry_attempt(output, state, phase, job, reason)
        atomic_json(state_path(output), state)
        write_progress_receipt(output, state)
        if phase_is_imported(state, phase, logical_ids_for_phase(phase)):
            validate_phase_imports(project_root, state, phase)
            return
        if poll_seconds:
            if capacity_wait:
                print(
                    f"{phase}: waiting {poll_seconds}s for exact launch-ready capacity",
                    flush=True,
                )
            time.sleep(poll_seconds)


def run_campaign(
    project_root: Path,
    output: Path,
    tag: str,
    poll_seconds: int = DEFAULT_POLL_SECONDS,
) -> Path:
    if poll_seconds < 0:
        raise RuntimeError("poll interval cannot be negative")
    output.mkdir(parents=True, exist_ok=True)
    node_snapshot = assert_cluster_contract()
    root = study_root(project_root)
    source_bundle = ensure_source_bundle(project_root, root)
    full_seed = ensure_workspace_seed(project_root, output, RUN_ID)
    smoke_seed = ensure_workspace_seed(project_root, output, SMOKE_RUN_ID)
    state = load_or_create_state(output, tag, source_bundle, full_seed, smoke_seed)
    if not any(state["phases"][phase]["jobs"] for phase in PHASE_ORDER):
        revalidate_preflight(
            project_root,
            output,
            node_snapshot,
            source_bundle,
            binary(project_root),
            full_seed,
            smoke_seed,
        )
    for phase in PHASE_ORDER:
        if phase_is_imported(state, phase, logical_ids_for_phase(phase)):
            validate_phase_imports(project_root, state, phase)
            continue
        supervise_phase(
            project_root,
            output,
            state,
            phase,
            tag,
            poll_seconds,
        )
    execution = {
        "schema": "TrainOnlyRedteamSCQExecutionLedger.v2",
        "run_id": RUN_ID,
        "project": PROJECT,
        "tag": tag,
        "phases": state["phases"],
        "development_only": True,
        "confirmatory_valid": False,
    }
    ledger = output / "execution-ledger.v2.json"
    immutable_json(ledger, execution)
    return ledger


def plan_campaign_phase(
    project_root: Path,
    output: Path,
    tag: str,
) -> list[Path]:
    """Render and check only the next dependency-eligible campaign phase."""
    output.mkdir(parents=True, exist_ok=True)
    node_snapshot = assert_cluster_contract()
    root = study_root(project_root)
    source_bundle = ensure_source_bundle(project_root, root)
    full_seed = ensure_workspace_seed(project_root, output, RUN_ID)
    smoke_seed = ensure_workspace_seed(project_root, output, SMOKE_RUN_ID)
    state = load_or_create_state(output, tag, source_bundle, full_seed, smoke_seed)
    if not any(state["phases"][phase]["jobs"] for phase in PHASE_ORDER):
        revalidate_preflight(
            project_root,
            output,
            node_snapshot,
            source_bundle,
            binary(project_root),
            full_seed,
            smoke_seed,
        )
    for phase in PHASE_ORDER:
        if phase_is_imported(state, phase, logical_ids_for_phase(phase)):
            continue
        jobs = jobs_for_phase(project_root, output, state, phase, tag)
        ensure_phase_attempts(output, state, phase, jobs)
        atomic_json(state_path(output), state)
        paths = [
            Path(current_attempt(state["phases"][phase]["jobs"][logical])["spec_path"])
            for logical in logical_ids_for_phase(phase)
        ]
        for path in paths:
            attempt = next(
                current_attempt(logical)
                for logical in state["phases"][phase]["jobs"].values()
                if current_attempt(logical)["spec_path"] == str(path)
            )
            if not check_attempt_capacity(attempt):
                print(
                    f"{phase}: {attempt['job_id']} awaits launch-ready capacity",
                    flush=True,
                )
        return paths
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=ALL_COMMANDS)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="fetch completed declared packages before aggregate reconciliation",
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path("/home/ubuntu/jope-dime")
    )
    parser.add_argument(
        "--sotarena-root", type=Path, default=Path("/home/ubuntu/sotarena")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--tag",
        default=VERSION,
        help="new version suffix for a retry; stale IDs are never reused",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=DEFAULT_POLL_SECONDS,
        help="capacity and job-state polling interval (default: 60 seconds)",
    )
    args = parser.parse_args()

    args.project_root = args.project_root.resolve()
    args.sotarena_root = args.sotarena_root.resolve()
    if args.output is None:
        args.output = (
            study_root(args.project_root)
            / "scq"
            / (args.tag if args.mode == "run" else "generated")
        )
    args.output = args.output.resolve()
    if args.mode == "run":
        print(
            run_campaign(
                args.project_root,
                args.output,
                args.tag,
                args.poll_seconds,
            )
        )
        return
    if args.mode == "plan":
        for path in plan_campaign_phase(
            args.project_root,
            args.output,
            args.tag,
        ):
            print(path)
        return

    rendered = emit_specs(args.project_root, args.sotarena_root, args.output, args.tag)
    if args.mode == "submit-next":
        print(submit_next(args.project_root, rendered))
        return
    if args.mode == "status":
        print(json.dumps(status(rendered), indent=2, sort_keys=True))
        return
    if args.mode == "reconcile":
        print(reconcile(args.project_root, rendered))
        return
    if args.mode == "aggregate":
        check_specs(rendered["aggregate"])
        if args.submit:
            submit_phase(args.project_root, rendered["aggregate"])
            print("submitted aggregate")
            return
        imported = fetch_outputs(args.project_root, rendered) if args.fetch else {}
        print(reconcile_with_imports(args.project_root, rendered, imported))
        return
    check_specs(rendered[args.mode])
    for path in rendered[args.mode]:
        print(path)
    if args.submit:
        submit_phase(args.project_root, rendered[args.mode])


if __name__ == "__main__":
    main()
