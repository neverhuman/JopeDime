from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
from pathlib import Path
from typing import Any

if __package__:
    from .smart_v3_cpu_scq_config import (
        CHECKPOINT_SHA256,
        NODES,
        PARENT_RUN_ID,
        PRIOR_RUN_ID,
        PROJECT,
        RUN_ID,
        SCQ,
        SHARDS,
    )
else:
    from smart_v3_cpu_scq_config import (
        CHECKPOINT_SHA256,
        NODES,
        PARENT_RUN_ID,
        PRIOR_RUN_ID,
        PROJECT,
        RUN_ID,
        SCQ,
        SHARDS,
    )

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
