#!/usr/bin/env python3
"""Verify the installed 20-dataset ``*.vs.orig`` cohort and write a receipt."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


GENERATION_SCHEMA = "HyperionOldPythonVsReconstruction.v1"
REPLAY_SCHEMA = "HyperionOldPythonVsReplayVerification.v1"
INSTALLATION_SCHEMA = "HyperionOldPythonVsInstallation.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def inspect_matrix(path: Path) -> tuple[list[int], int]:
    rows = 0
    width: int | None = None
    values = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            if width is None:
                width = len(row)
            elif len(row) != width:
                raise ValueError(f"Ragged matrix: {path}")
            if not row:
                raise ValueError(f"Empty matrix row: {path}")
            for cell in row:
                value = float(cell)
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite matrix value: {path}")
            rows += 1
            values += len(row)
    if rows == 0 or width is None or width == 0:
        raise ValueError(f"Empty matrix: {path}")
    return [rows, width], values


def verify_installation(
    generation_manifest_path: Path,
    replay_verification_path: Path,
    installed_root: Path,
) -> dict[str, Any]:
    generation_manifest_path = generation_manifest_path.resolve()
    replay_verification_path = replay_verification_path.resolve()
    installed_root = installed_root.resolve()
    generation = load_json(generation_manifest_path)
    replay = load_json(replay_verification_path)

    if generation.get("schema") != GENERATION_SCHEMA:
        raise ValueError("Unexpected generation manifest schema")
    if generation.get("status") != "complete":
        raise ValueError("Generation manifest is incomplete")
    if generation.get("dataset_count") != 20:
        raise ValueError("Generation manifest must contain 20 datasets")
    if replay.get("schema") != REPLAY_SCHEMA or replay.get("status") != "complete":
        raise ValueError("Replay verification is incomplete")
    if replay.get("dataset_count") != 20 or replay.get("output_file_count") != 40:
        raise ValueError("Replay verification has the wrong cohort size")
    for field in (
        "all_sweep_tensor_digests_identical",
        "all_collapsed_cell_digests_identical",
        "all_output_files_byte_identical",
    ):
        if replay.get(field) is not True:
            raise ValueError(f"Replay verification did not pass: {field}")
    expected_generation_sha = replay["generation_manifests"]["first"]["sha256"]
    if sha256_file(generation_manifest_path) != expected_generation_sha:
        raise ValueError("Installed source manifest is not replay run one")

    datasets: list[dict[str, Any]] = []
    for dataset in generation["datasets"]:
        dataset_id = dataset["dataset_id"]
        dataset_root = installed_root / dataset_id
        expected_names = {"train3.vs.orig", "test3.vs.orig"}
        actual_names = {path.name for path in dataset_root.glob("*.vs.orig")}
        if actual_names != expected_names:
            raise ValueError(
                f"Unexpected .vs.orig inventory for {dataset_id}: {actual_names}"
            )

        outputs: dict[str, Any] = {}
        for split, name in (("train", "train3.vs.orig"), ("test", "test3.vs.orig")):
            path = dataset_root / name
            expected = dataset["outputs"][split]
            if not path.is_file():
                raise FileNotFoundError(path)
            actual_sha = sha256_file(path)
            if actual_sha != expected["sha256"]:
                raise ValueError(f"Installed output hash mismatch: {path}")
            shape, value_count = inspect_matrix(path)
            if shape != expected["shape"]:
                raise ValueError(f"Installed output shape mismatch: {path}")
            outputs[split] = {
                "relative_path": str(path.relative_to(installed_root)),
                "sha256": actual_sha,
                "shape": shape,
                "finite": True,
                "value_count": value_count,
            }
        datasets.append({"dataset_id": dataset_id, "outputs": outputs})

    return {
        "schema": INSTALLATION_SCHEMA,
        "status": "complete",
        "claim": "all installed outputs match the byte-identical CUDA replay",
        "installed_root": str(installed_root),
        "dataset_count": len(datasets),
        "output_file_count": len(datasets) * 2,
        "source_generation_manifest": {
            "path": str(generation_manifest_path),
            "sha256": sha256_file(generation_manifest_path),
        },
        "source_replay_verification": {
            "path": str(replay_verification_path),
            "sha256": sha256_file(replay_verification_path),
        },
        "verification_script_sha256": sha256_file(Path(__file__).resolve()),
        "preinstall_existing_output_count": 0,
        "copy_overwrite_policy": "rsync --ignore-existing",
        "datasets": datasets,
    }


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--replay-verification", type=Path, required=True)
    parser.add_argument("--installed-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = verify_installation(
        args.generation_manifest,
        args.replay_verification,
        args.installed_root,
    )
    write_json_exclusive(args.receipt.resolve(), receipt)
    print(
        f"Verified {receipt['output_file_count']} installed outputs across "
        f"{receipt['dataset_count']} datasets"
    )


if __name__ == "__main__":
    main()
