#!/usr/bin/env python3
"""Replay all confirmation train-side Hyperion cells under one pinned encoder.

Every invocation supplies the same frozen training table for both encoder input
slots.  Hyperion drops the final target column before encoding, and this harness
has no API for a test-side source.  Cached ``train3.vs`` is authorized only when
the fresh CUDA output is byte-identical; every other state predeclares raw.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import freeze_hyperion_precomputed_cell_coverage as physical


SCHEMA = "HyperionConfirmationCellReplay.v1"
EXPECTED_CONFIRMATION_DATASETS = 20
SEED = 0x4859_5045_5249_4F4E
SWEEPS = 3
SCALES = "4,7,11,15"
BATCH = 256
ENVIRONMENT = {
    "JAIN_HYPERION_CUDA_MAX_LOAD_RATIO": "1000000",
    "OMP_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "BLIS_NUM_THREADS": "2",
    "VECLIB_MAXIMUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
}
ARGV_SCHEMA = [
    "--train",
    "--test",
    "--out-cell-train",
    "--out-cell-test",
    "--out-row-train",
    "--out-row-test",
    "--weights",
    "--manifest",
    "--sweeps",
    "--scales",
    "--seed",
    "--batch",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def device_proof(stdout: str, stderr: str) -> bool:
    combined = f"{stdout}\n{stderr}"
    return combined.count("device=cuda:0") == 1 and "device=cpu" not in combined.lower()


def confirmation_entries(coverage: dict[str, Any]) -> list[dict[str, Any]]:
    if (
        coverage.get("schema") != physical.SCHEMA
        or coverage.get("dataset_count") != 42
        or coverage.get("test_artifacts_read") is not False
    ):
        raise ValueError("physical cell coverage is incomplete or foreign")
    entries = [
        entry
        for entry in coverage.get("datasets", [])
        if entry.get("cohort") == "confirmation"
    ]
    if (
        len(entries) != EXPECTED_CONFIRMATION_DATASETS
        or len({entry.get("dataset_id") for entry in entries}) != len(entries)
        or entries != sorted(entries, key=lambda entry: entry["dataset_id"])
    ):
        raise ValueError("confirmation coverage must contain 20 ordered unique entries")
    return entries


def run_one(
    entry: dict[str, Any],
    quant_root: Path,
    binary: Path,
    weights: Path,
    manifest: Path,
    cuda_visible_devices: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    dataset_id = entry["dataset_id"]
    dataset_root = quant_root / dataset_id
    train_path = physical.checked_path(dataset_root, "train.csv")
    cached_path = physical.checked_path(dataset_root, "train3.vs")
    if sha256_file(train_path) != entry["train_sha256"]:
        raise ValueError(f"{dataset_id}: training table changed after physical freeze")
    if sha256_file(cached_path) != entry["cell_sha256"]:
        raise ValueError(f"{dataset_id}: cached cell changed after physical freeze")
    environment = os.environ.copy()
    environment.update(ENVIRONMENT)
    environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    started = time.monotonic()
    with tempfile.TemporaryDirectory(
        prefix=f"jope-hyperion-replay-{dataset_id}-"
    ) as raw_root:
        root = Path(raw_root)
        fresh_cell = root / "fresh-cell-train.vs"
        command = [
            str(binary),
            "--train",
            str(train_path),
            "--test",
            str(train_path),
            "--out-cell-train",
            str(fresh_cell),
            "--out-cell-test",
            str(root / "discard-cell-hold.vs"),
            "--out-row-train",
            str(root / "discard-row-train.vs"),
            "--out-row-test",
            str(root / "discard-row-hold.vs"),
            "--weights",
            str(weights),
            "--manifest",
            str(manifest),
            "--sweeps",
            str(SWEEPS),
            "--scales",
            SCALES,
            "--seed",
            str(SEED),
            "--batch",
            str(BATCH),
        ]
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
                env=environment,
            )
        except subprocess.TimeoutExpired:
            return {
                "dataset_id": dataset_id,
                "rows": entry["rows"],
                "raw_features": entry["raw_features"],
                "cached_cell_sha256": entry["cell_sha256"],
                "state": "encoder_timeout",
                "selected_view": "raw",
                "elapsed_seconds": time.monotonic() - started,
                "test_artifacts_read": False,
            }
        elapsed = time.monotonic() - started
        proof = device_proof(completed.stdout, completed.stderr)
        if completed.returncode != 0 or not proof or not fresh_cell.is_file():
            return {
                "dataset_id": dataset_id,
                "rows": entry["rows"],
                "raw_features": entry["raw_features"],
                "cached_cell_sha256": entry["cell_sha256"],
                "state": "encoder_failure_or_invalid_cuda_proof",
                "selected_view": "raw",
                "return_code": completed.returncode,
                "device_proof_valid": proof,
                "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
                "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
                "elapsed_seconds": elapsed,
                "test_artifacts_read": False,
            }
        fresh = physical.inspect_cell(fresh_cell, entry["rows"], entry["raw_features"])
        matches = fresh["cell_sha256"] == entry["cell_sha256"]
        return {
            "dataset_id": dataset_id,
            "rows": entry["rows"],
            "raw_features": entry["raw_features"],
            "train_sha256": entry["train_sha256"],
            "cached_cell_sha256": entry["cell_sha256"],
            "fresh_cell_sha256": fresh["cell_sha256"],
            "state": "encoder_authorized" if matches else "stale_or_foreign_cache",
            "selected_view": "hyperion_cell_candidate" if matches else "raw",
            "return_code": completed.returncode,
            "device_proof_valid": proof,
            "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
            "elapsed_seconds": elapsed,
            "test_artifacts_read": False,
        }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    coverage_bytes = args.physical_coverage.read_bytes()
    coverage = json.loads(coverage_bytes)
    entries = confirmation_entries(coverage)
    for path in (args.binary, args.weights, args.manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    results: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        result = run_one(
            entry,
            args.quant_root,
            args.binary.resolve(),
            args.weights.resolve(),
            args.manifest.resolve(),
            args.cuda_visible_devices,
            args.timeout_seconds,
        )
        results.append(result)
        print(
            f"replay {index}/{len(entries)} {entry['dataset_id']} "
            f"state={result['state']}",
            flush=True,
        )
    authorized = sum(result["state"] == "encoder_authorized" for result in results)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "physical_coverage_file_sha256": hashlib.sha256(coverage_bytes).hexdigest(),
        "physical_coverage_sha256": coverage["coverage_sha256"],
        "encoder": {
            "binary_path": str(args.binary.resolve()),
            "binary_sha256": sha256_file(args.binary),
            "weights_path": str(args.weights.resolve()),
            "weights_sha256": sha256_file(args.weights),
            "manifest_path": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest),
            "seed": SEED,
            "sweeps": SWEEPS,
            "scales": [int(value) for value in SCALES.split(",")],
            "batch": BATCH,
            "cuda_visible_devices": args.cuda_visible_devices,
            "environment": ENVIRONMENT,
            "argv_schema": ARGV_SCHEMA,
        },
        "input_policy": {
            "encoder_train_source": "frozen train.csv",
            "encoder_hold_source": "the same frozen train.csv",
            "target_column_dropped_before_encoder": True,
            "test_side_source_resolvable": False,
        },
        "expected_dataset_count": EXPECTED_CONFIRMATION_DATASETS,
        "complete_dataset_count": len(results),
        "encoder_authorized_cell_count": authorized,
        "raw_fallback_count": len(results) - authorized,
        "results": results,
        "target_statistics_emitted": False,
        "test_artifacts_read": False,
    }
    report["report_sha256"] = physical.canonical_sha256(report)
    return report


def freeze(path: Path, report: dict[str, Any]) -> None:
    encoded = json.dumps(report, indent=2, sort_keys=True).encode() + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError(
                "confirmation replay report already exists with other bytes"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-coverage", required=True, type=Path)
    parser.add_argument("--quant-root", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cuda_visible_devices != "0" or args.timeout_seconds != 600:
        raise ValueError("confirmation replay execution policy drift")
    report = build_report(args)
    freeze(args.output, report)
    print(
        json.dumps(
            {
                "authorized": report["encoder_authorized_cell_count"],
                "fallback": report["raw_fallback_count"],
                "report_sha256": report["report_sha256"],
                "test_artifacts_read": report["test_artifacts_read"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
