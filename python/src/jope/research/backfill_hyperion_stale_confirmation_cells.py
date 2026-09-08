#!/usr/bin/env python3
"""Backfill only stale confirmation train cells under the pinned encoder.

Historical ``train3.vs`` files are immutable.  For each cache rejected by the
confirmation replay, this tool encodes the frozen training table twice: once
with that table as the hold input and once with a reversed-feature, zero-target
synthetic hold table.  Train outputs must be byte-identical before a new cell
matrix is written to a disjoint immutable dataset namespace.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import freeze_hyperion_precomputed_cell_coverage as physical
import replay_hyperion_confirmation_cells as replay


SCHEMA = "HyperionConfirmationCellAuthority.v2"
RECEIPT_SCHEMA = "HyperionConfirmationCellBackfillReceipt.v1"


def synthetic_hold_bytes(train_bytes: bytes, label: Path) -> bytes:
    rows = physical.source_rows(train_bytes, label)
    synthetic: list[str] = []
    for ordinal, source in enumerate(reversed(rows)):
        try:
            fields = next(csv.reader(io.StringIO(source.decode("utf-8"))))
        except (UnicodeDecodeError, csv.Error) as error:
            raise ValueError(f"{label}: malformed source row {ordinal}") from error
        if len(fields) < 2:
            raise ValueError(f"{label}: source row has no target slot")
        synthetic.append(",".join([*fields[:-1], "0"]))
    return ("\n".join(synthetic) + "\n").encode("utf-8")


def immutable_write(path: Path, value: bytes) -> None:
    if path.exists():
        if path.read_bytes() != value:
            raise ValueError(f"immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def encode(
    binary: Path,
    weights: Path,
    manifest: Path,
    train_path: Path,
    hold_path: Path,
    output_root: Path,
    cuda_visible_devices: str,
    timeout_seconds: int,
) -> tuple[bytes, dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=True)
    fresh_cell = output_root / "cell-train.vs"
    command = [
        str(binary),
        "--train",
        str(train_path),
        "--test",
        str(hold_path),
        "--out-cell-train",
        str(fresh_cell),
        "--out-cell-test",
        str(output_root / "discard-cell-hold.vs"),
        "--out-row-train",
        str(output_root / "discard-row-train.vs"),
        "--out-row-test",
        str(output_root / "discard-row-hold.vs"),
        "--weights",
        str(weights),
        "--manifest",
        str(manifest),
        "--sweeps",
        str(replay.SWEEPS),
        "--scales",
        replay.SCALES,
        "--seed",
        str(replay.SEED),
        "--batch",
        str(replay.BATCH),
    ]
    environment = os.environ.copy()
    environment.update(replay.ENVIRONMENT)
    environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    started = time.monotonic()
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
        env=environment,
    )
    elapsed = time.monotonic() - started
    proof = replay.device_proof(completed.stdout, completed.stderr)
    if completed.returncode != 0 or not proof or not fresh_cell.is_file():
        raise RuntimeError(
            "encoder failed or did not prove exactly one cuda:0 execution: "
            f"return_code={completed.returncode} proof={proof}"
        )
    return fresh_cell.read_bytes(), {
        "return_code": completed.returncode,
        "device_proof_valid": proof,
        "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
        "elapsed_seconds": elapsed,
    }


def stale_results(report: dict[str, Any]) -> list[dict[str, Any]]:
    if (
        report.get("schema") != replay.SCHEMA
        or report.get("complete_dataset_count") != replay.EXPECTED_CONFIRMATION_DATASETS
        or report.get("test_artifacts_read") is not False
    ):
        raise ValueError("confirmation replay is incomplete or foreign")
    stale = [
        result
        for result in report.get("results", [])
        if result.get("state") != "encoder_authorized"
    ]
    if len(stale) != report.get("raw_fallback_count"):
        raise ValueError("confirmation replay fallback accounting drift")
    return stale


def backfill_one(
    entry: dict[str, Any],
    quant_root: Path,
    output_root: Path,
    binary: Path,
    weights: Path,
    manifest: Path,
    cuda_visible_devices: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    dataset_id = entry["dataset_id"]
    train_path = physical.checked_path(quant_root / dataset_id, "train.csv")
    train_bytes = train_path.read_bytes()
    if hashlib.sha256(train_bytes).hexdigest() != entry["train_sha256"]:
        raise ValueError(f"{dataset_id}: training table changed before backfill")
    with tempfile.TemporaryDirectory(
        prefix=f"jope-hyperion-backfill-{dataset_id}-"
    ) as raw:
        root = Path(raw)
        synthetic_path = root / "synthetic-hold.csv"
        synthetic_path.write_bytes(synthetic_hold_bytes(train_bytes, train_path))
        first, first_proof = encode(
            binary,
            weights,
            manifest,
            train_path,
            train_path,
            root / "same-train-hold",
            cuda_visible_devices,
            timeout_seconds,
        )
        second, second_proof = encode(
            binary,
            weights,
            manifest,
            train_path,
            synthetic_path,
            root / "synthetic-hold",
            cuda_visible_devices,
            timeout_seconds,
        )
        if first != second:
            raise ValueError(f"{dataset_id}: train encoding depends on hold features")
        matrix_path = output_root / dataset_id / "train3.vs"
        immutable_write(matrix_path, first)
        inspected = physical.inspect_cell(
            matrix_path, entry["rows"], entry["raw_features"]
        )
        receipt: dict[str, Any] = {
            "schema": RECEIPT_SCHEMA,
            "dataset_id": dataset_id,
            "train_sha256": entry["train_sha256"],
            "historical_cell_sha256": entry["cached_cell_sha256"],
            "authoritative_cell_sha256": inspected["cell_sha256"],
            "authoritative_relative_path": f"{dataset_id}/train3.vs",
            "rows": entry["rows"],
            "raw_features": entry["raw_features"],
            "same_train_hold_execution": first_proof,
            "synthetic_hold_execution": second_proof,
            "train_output_hold_feature_invariant": True,
            "synthetic_hold_target": 0.0,
            "test_artifacts_read": False,
        }
        receipt["receipt_sha256"] = physical.canonical_sha256(receipt)
        immutable_write(
            output_root / dataset_id / "receipt.v1.json",
            json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n",
        )
        return receipt


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    replay_bytes = args.replay_report.read_bytes()
    replay_report = json.loads(replay_bytes)
    stale = stale_results(replay_report)
    encoder = replay_report["encoder"]
    for path, expected in [
        (args.binary, encoder["binary_sha256"]),
        (args.weights, encoder["weights_sha256"]),
        (args.manifest, encoder["manifest_sha256"]),
    ]:
        if replay.sha256_file(path) != expected:
            raise ValueError(f"encoder artifact differs from replay authority: {path}")
    namespace = args.output_root / encoder["binary_sha256"]
    receipts: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, entry in enumerate(stale, start=1):
        try:
            receipt = backfill_one(
                entry,
                args.quant_root,
                namespace,
                args.binary.resolve(),
                args.weights.resolve(),
                args.manifest.resolve(),
                args.cuda_visible_devices,
                args.timeout_seconds,
            )
            receipts.append(receipt)
            print(
                f"backfill {index}/{len(stale)} {entry['dataset_id']} "
                "state=encoder_authorized",
                flush=True,
            )
        except Exception as error:
            failures.append({"dataset_id": entry["dataset_id"], "reason": str(error)})
            print(
                f"backfill {index}/{len(stale)} {entry['dataset_id']} failed={error}",
                flush=True,
            )
    historical = [
        {
            "dataset_id": result["dataset_id"],
            "source": "historical-byte-verified",
            "cell_sha256": result["cached_cell_sha256"],
            "relative_path": f"{result['dataset_id']}/train3.vs",
            "selected_view": "hyperion_cell_candidate",
        }
        for result in replay_report["results"]
        if result["state"] == "encoder_authorized"
    ]
    backfilled = [
        {
            "dataset_id": receipt["dataset_id"],
            "source": "campaign-backfill",
            "cell_sha256": receipt["authoritative_cell_sha256"],
            "relative_path": receipt["authoritative_relative_path"],
            "receipt_sha256": receipt["receipt_sha256"],
            "selected_view": "hyperion_cell_candidate",
        }
        for receipt in receipts
    ]
    datasets = sorted([*historical, *backfilled], key=lambda entry: entry["dataset_id"])
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "confirmation_replay_file_sha256": hashlib.sha256(replay_bytes).hexdigest(),
        "confirmation_replay_report_sha256": replay_report["report_sha256"],
        "encoder": encoder,
        "backfill_namespace": str(namespace),
        "historical_files_overwritten": False,
        "expected_dataset_count": replay.EXPECTED_CONFIRMATION_DATASETS,
        "encoder_authorized_dataset_count": len(datasets),
        "backfilled_dataset_count": len(receipts),
        "raw_fallback_count": replay.EXPECTED_CONFIRMATION_DATASETS - len(datasets),
        "datasets": datasets,
        "backfill_receipts": receipts,
        "failures": failures,
        "target_statistics_emitted": False,
        "test_artifacts_read": False,
    }
    report["report_sha256"] = physical.canonical_sha256(report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-report", required=True, type=Path)
    parser.add_argument("--quant-root", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cuda_visible_devices != "0" or args.timeout_seconds != 600:
        raise ValueError("confirmation backfill execution policy drift")
    report = build_report(args)
    immutable_write(
        args.report,
        json.dumps(report, indent=2, sort_keys=True).encode() + b"\n",
    )
    print(
        json.dumps(
            {
                "authorized": report["encoder_authorized_dataset_count"],
                "backfilled": report["backfilled_dataset_count"],
                "fallback": report["raw_fallback_count"],
                "report_sha256": report["report_sha256"],
                "test_artifacts_read": report["test_artifacts_read"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if not report["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
