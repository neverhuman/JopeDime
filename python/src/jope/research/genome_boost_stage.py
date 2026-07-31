#!/usr/bin/env python3
"""Stage the genome-boost-20260725-v1 campaign root layout.

Builds, under the campaign root (default
``/home/ubuntu/remote_super/quant/campaigns/genome-boost-20260725-v1``):

* ``manifests/`` — per-dataset genome selection manifests (copied verbatim).
* ``features/<hash>/`` — Rust-emitted genome feature blocks + receipts
  (copied verbatim; numeric bytes are never rewritten).
* ``worker-views/<hash>/`` — masked view: ``train.csv`` byte-copied from the
  registry, ``test.csv`` with the target column STRIPPED as text (remaining
  numeric bytes untouched), a metadata subset, the genome feature blocks, and
  a ``worker-manifest.json`` binding every artifact by sha256.
* ``--emit-node-sync-plan`` writes ``sync-plan/`` with the per-node rsync
  commands as a plan file only; nothing is executed against other nodes.

Workers never see test targets: the stripped ``test.csv`` is the only test
artifact staged into worker views.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path

CAMPAIGN_ID = "genome-boost-20260725-v1"
DEFAULT_CAMPAIGN_ROOT = Path("/home/ubuntu/remote_super/quant/campaigns") / CAMPAIGN_ID
REGISTRY_ROOT = Path("/home/ubuntu/remote_super/quant/regression")
JOPE_ROOT = Path("/home/ubuntu/jope-dime")
PANEL_FILE = JOPE_ROOT / "manifests" / "panel.txt"
VIEW_SCHEMA = "jope-dime.genome-boost-worker-view.v1"
SYNC_PLAN_SCHEMA = "jope-dime.genome-boost-node-sync-plan.v1"
FEATURE_FILES = (
    "genome-features-train.csv",
    "genome-features-test.csv",
    "genome-features-receipt.v1.json",
)
CONTROLLER_NODE = "xbabe2"
WORKER_NODES = ("xbabe1", "xbabe3")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
        temporary = Path(tmp.name)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: object) -> None:
    atomic_write_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def copy_verbatim(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(destination, source.read_bytes())


def load_datasets(argument: str) -> list[str]:
    source = PANEL_FILE if argument == "golden42" else Path(argument)
    if source.suffix == ".json":
        payload = json.loads(source.read_text())
        hashes = payload.get("datasets", [])
    else:
        hashes = [line.strip() for line in source.read_text().splitlines() if line.strip()]
    if not hashes:
        raise SystemExit(f"no dataset hashes in {source}")
    return hashes


def strip_last_column_text(csv_bytes: bytes) -> bytes:
    """Drop the final comma-separated field of every line WITHOUT rewriting
    the remaining numeric bytes."""
    lines = []
    for raw in csv_bytes.decode().splitlines():
        if not raw.strip():
            continue
        fields = raw.split(",")
        if len(fields) < 2:
            raise ValueError("cannot strip the target from a single-column row")
        lines.append(",".join(fields[:-1]))
    return ("\n".join(lines) + "\n").encode()


def stage_dataset(
    dataset: str,
    manifests_dir: Path,
    features_dir: Path,
    campaign_root: Path,
    worker_root: Path,
    copy_features: bool,
) -> dict:
    registry = REGISTRY_ROOT / dataset
    train_path = registry / "train.csv"
    test_path = registry / "test.csv"
    meta_path = registry / "meta.json"
    feature_source = features_dir / dataset
    receipt = json.loads((feature_source / "genome-features-receipt.v1.json").read_text())
    if receipt.get("dataset_hash") != dataset:
        raise SystemExit(f"{dataset}: feature receipt identity mismatch")
    if receipt.get("benchmark_valid") is not False or receipt.get("test_adaptive") is not True:
        raise SystemExit(f"{dataset}: feature receipt labels are wrong")
    if sha256_file(train_path) != receipt["train_csv_sha256"]:
        raise SystemExit(f"{dataset}: registry train.csv differs from feature receipt")
    if sha256_file(test_path) != receipt["test_csv_sha256"]:
        raise SystemExit(f"{dataset}: registry test.csv differs from feature receipt")

    # manifests/
    manifest_name = f"{dataset}.manifest.v1.json"
    manifest_source = manifests_dir / manifest_name
    if manifest_source.is_file():
        copy_verbatim(manifest_source, campaign_root / "manifests" / manifest_name)

    if copy_features:
        for name in FEATURE_FILES:
            copy_verbatim(feature_source / name, campaign_root / "features" / dataset / name)

    # worker-views/<hash>/
    view = worker_root / dataset
    view.mkdir(parents=True, exist_ok=True)
    train_bytes = train_path.read_bytes()
    atomic_write_bytes(view / "train.csv", train_bytes)
    masked_test = strip_last_column_text(test_path.read_bytes())
    atomic_write_bytes(view / "test.csv", masked_test)
    for name in ("genome-features-train.csv", "genome-features-test.csv"):
        copy_verbatim(feature_source / name, view / name)

    canonical_meta = json.loads(meta_path.read_text())
    train_rows = int(receipt["train_rows"])
    test_rows = int(receipt["test_rows"])
    n_features = int(receipt["dataset_features"])
    meta_subset = {
        "schema": "jope-dime.genome-boost-worker-meta.v1",
        "campaign_id": CAMPAIGN_ID,
        "dataset_hash": dataset,
        "task": "regression",
        "headers_removed_from_csv": bool(canonical_meta.get("headers_removed_from_csv", True)),
        "target_is_final_column_in_train": True,
        "test_target_present": False,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "n_features": n_features,
        "k_genome_features": int(receipt["emitted_columns"]),
    }
    atomic_write_json(view / "meta.json", meta_subset)

    artifacts = {
        name: sha256_file(view / name)
        for name in (
            "train.csv",
            "test.csv",
            "meta.json",
            "genome-features-train.csv",
            "genome-features-test.csv",
        )
    }
    if artifacts["train.csv"] != receipt["train_csv_sha256"]:
        raise SystemExit(f"{dataset}: staged train.csv is not byte-identical to the registry")
    if artifacts["genome-features-train.csv"] != receipt["train_features_csv_sha256"]:
        raise SystemExit(f"{dataset}: staged train feature block hash mismatch")
    if artifacts["genome-features-test.csv"] != receipt["test_features_csv_sha256"]:
        raise SystemExit(f"{dataset}: staged test feature block hash mismatch")
    manifest = {
        "schema": VIEW_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "dataset_hash": dataset,
        "test_adaptive": True,
        "benchmark_valid": False,
        "test_target_present": False,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "n_features": n_features,
        "k_genome_features": int(receipt["emitted_columns"]),
        "feature_receipt_sha256": sha256_file(
            feature_source / "genome-features-receipt.v1.json"
        ),
        "canonical_train_sha256": receipt["train_csv_sha256"],
        "canonical_test_sha256": receipt["test_csv_sha256"],
        "artifacts": artifacts,
        "created_at": utc_now(),
    }
    atomic_write_json(view / "worker-manifest.json", manifest)
    return manifest


def emit_sync_plan(campaign_root: Path) -> None:
    commands = []
    for node in WORKER_NODES:
        commands.append(
            {
                "node": node,
                "direction": "push",
                "command": [
                    "rsync",
                    "-a",
                    "--checksum",
                    f"{campaign_root}/",
                    f"{node}:{campaign_root}/",
                ],
            }
        )
        commands.append(
            {
                "node": node,
                "direction": "pull-predictions",
                "command": [
                    "rsync",
                    "-a",
                    "--checksum",
                    f"{node}:{campaign_root}/predictions/",
                    f"{campaign_root}/predictions/",
                ],
            }
        )
    plan = {
        "schema": SYNC_PLAN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "controller_node": CONTROLLER_NODE,
        "worker_nodes": list(WORKER_NODES),
        "created_at": utc_now(),
        "note": (
            "PLAN ONLY: nothing here has been executed. Run the push commands "
            "before submitting node-pinned shards, and the pull-predictions "
            "commands before finalize."
        ),
        "commands": commands,
    }
    plan_dir = campaign_root / "sync-plan"
    atomic_write_json(plan_dir / "node-sync-plan.v1.json", plan)
    script_lines = ["#!/bin/bash", "# " + plan["note"], "set -euo pipefail", ""]
    script_lines += ["# --- push campaign root to worker nodes ---"]
    script_lines += [
        " ".join(item["command"]) for item in commands if item["direction"] == "push"
    ]
    script_lines += ["", "# --- pull predictions back before finalize ---"]
    script_lines += [
        "# " + " ".join(item["command"])
        for item in commands
        if item["direction"] == "pull-predictions"
    ]
    script_path = plan_dir / "node-sync-plan.sh"
    atomic_write_bytes(script_path, ("\n".join(script_lines) + "\n").encode())
    script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    print(f"sync plan written to {plan_dir} (NOT executed)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests-dir", required=True, type=Path)
    parser.add_argument("--features-dir", required=True, type=Path)
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument(
        "--worker-root",
        type=Path,
        help="worker view destination (default: <campaign-root>/worker-views)",
    )
    parser.add_argument("--datasets", default="golden42", help='"golden42" or a file of hashes')
    parser.add_argument("--required-columns", type=int)
    parser.add_argument("--staged-datasets-file", type=Path)
    parser.add_argument("--emit-node-sync-plan", action="store_true")
    arguments = parser.parse_args()
    datasets = load_datasets(arguments.datasets)
    worker_root = arguments.worker_root or arguments.campaign_root / "worker-views"
    copy_features = arguments.worker_root is None
    staged = 0
    staged_datasets: list[str] = []
    skipped: list[str] = []
    for dataset in datasets:
        receipt_path = (
            arguments.features_dir / dataset / "genome-features-receipt.v1.json"
        )
        if not receipt_path.is_file():
            skipped.append(dataset)
            continue
        if arguments.required_columns is not None:
            receipt = json.loads(receipt_path.read_text())
            if int(receipt.get("emitted_columns", -1)) != arguments.required_columns:
                skipped.append(dataset)
                continue
        stage_dataset(
            dataset,
            arguments.manifests_dir,
            arguments.features_dir,
            arguments.campaign_root,
            worker_root,
            copy_features,
        )
        staged += 1
        staged_datasets.append(dataset)
        print(f"staged {dataset}")
    if skipped:
        print(f"skipped {len(skipped)} datasets without feature blocks: {', '.join(skipped)}")
    if arguments.emit_node_sync_plan:
        emit_sync_plan(arguments.campaign_root)
    if arguments.staged_datasets_file:
        atomic_write_bytes(
            arguments.staged_datasets_file,
            ("".join(f"{dataset}\n" for dataset in staged_datasets)).encode(),
        )
    print(f"staged {staged} worker views under {worker_root}")


if __name__ == "__main__":
    main()
