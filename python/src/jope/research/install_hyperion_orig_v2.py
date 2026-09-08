#!/usr/bin/env python3
"""Validate and exclusively install a complete 20-dataset .vs.orig.v2 run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


GENERATION_SCHEMA = "HyperionLatestModelOldSemanticsReconstruction.v1"
INSTALLATION_SCHEMA = "HyperionLatestModelOldSemanticsInstallation.v1"
DATASET_IDS = (
    "019de5d76a718163", "047e62b52a3f4ede", "13b5df533e59a626",
    "17a4542ee4fb9e44", "1cb97b1d7314d43d", "30ef3eaa6d2971a9",
    "3305a3ad973fb227", "3baa7e37fc1f7861", "3f781a51af45387e",
    "542e75bc0cfdd1bf", "693890aae9dc860b", "79e600bc5fcd5da7",
    "ab0239b9ef0ba28e", "b253fbfae8460dc1", "cf5e74b4e9594944",
    "d2daf53412ab18dc", "dee1fc8e9afd81c9", "e1c1db5260d0894e",
    "ee56a1015e344c11", "fe3cc716a05d036f",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix_shape(path: Path, *, drop_last_column: bool = False) -> list[int]:
    rows = 0
    width: int | None = None
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            if width is None:
                width = len(row)
            elif len(row) != width:
                raise ValueError(f"ragged matrix: {path}")
            for text in row:
                if not math.isfinite(float(text)):
                    raise ValueError(f"non-finite value: {path}")
            rows += 1
    if rows == 0 or width is None or width <= int(drop_last_column):
        raise ValueError(f"empty or invalid matrix: {path}")
    return [rows, width - int(drop_last_column)]


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing existing receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def copy_exclusive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as writer, source.open("rb") as reader:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-receipt", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--installed-root", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    generation = json.loads(args.generation_receipt.read_text(encoding="utf-8"))
    if generation.get("schema") != GENERATION_SCHEMA or generation.get("status") != "complete":
        raise ValueError("generation receipt is not complete v2 output")
    if generation.get("dataset_count") != 20 or generation.get("output_file_count") != 40:
        raise ValueError("generation receipt has the wrong cohort size")
    if tuple(generation.get("dataset_order", ())) != DATASET_IDS:
        raise ValueError("generation dataset order changed")
    by_id = {entry["dataset_id"]: entry for entry in generation["datasets"]}
    if tuple(by_id) != DATASET_IDS:
        raise ValueError("generation dataset entries changed order")

    planned: list[tuple[Path, Path, str, list[int], str]] = []
    for dataset_id in DATASET_IDS:
        dataset = by_id[dataset_id]
        for split in ("train", "test"):
            name = f"{split}3.vs.orig.v2"
            source = args.source_root / dataset_id / name
            destination = args.installed_root / dataset_id / name
            raw = args.installed_root / dataset_id / f"{split}.csv"
            if not source.is_file() or not raw.is_file():
                raise FileNotFoundError(source if not source.is_file() else raw)
            if destination.exists():
                raise FileExistsError(f"refusing existing destination: {destination}")
            expected = dataset["outputs"][split]
            source_sha = sha256_file(source)
            source_shape = matrix_shape(source)
            raw_shape = matrix_shape(raw, drop_last_column=True)
            if source_sha != expected["sha256"]:
                raise ValueError(f"source hash mismatch: {source}")
            if source_shape != expected["shape"] or source_shape != raw_shape:
                raise ValueError(f"source shape mismatch: {source}")
            planned.append((source, destination, source_sha, source_shape, dataset_id))

    installed: list[Path] = []
    try:
        for source, destination, expected_sha, _, _ in planned:
            copy_exclusive(source, destination)
            installed.append(destination)
            if sha256_file(destination) != expected_sha:
                raise ValueError(f"post-copy hash mismatch: {destination}")
    except BaseException:
        for destination in reversed(installed):
            destination.unlink(missing_ok=True)
        raise

    files = [
        {
            "dataset_id": dataset_id,
            "relative_path": str(destination.relative_to(args.installed_root)),
            "sha256": expected_sha,
            "shape": shape,
        }
        for _, destination, expected_sha, shape, dataset_id in planned
    ]
    receipt = {
        "schema": INSTALLATION_SCHEMA,
        "status": "complete",
        "dataset_count": 20,
        "output_file_count": 40,
        "generation_receipt": {
            "path": str(args.generation_receipt.resolve()),
            "sha256": sha256_file(args.generation_receipt),
        },
        "source_root": str(args.source_root.resolve()),
        "installed_root": str(args.installed_root.resolve()),
        "preinstall_existing_output_count": 0,
        "copy_policy": "preflight all 40; O_EXCL create; hash verify; rollback new files on failure",
        "installer_sha256": sha256_file(Path(__file__).resolve()),
        "files": files,
    }
    write_json_exclusive(args.receipt, receipt)
    print("Installed and verified 40 .vs.orig.v2 files across 20 datasets", flush=True)


if __name__ == "__main__":
    main()
