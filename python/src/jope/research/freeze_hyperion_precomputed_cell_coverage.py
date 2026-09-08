#!/usr/bin/env python3
"""Freeze exact train-only Hyperion cell coverage for the budget-first registry.

The freezer can name only ``train.csv`` and ``train3.vs``.  It validates the
training table and cell matrix structurally, derives stable row/group identities,
and emits no target statistic or test-side path.  ``verify`` recomputes the full
projection and requires byte-for-byte equality with the frozen authority.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import struct
from pathlib import Path
from typing import Any


SCHEMA = "HyperionPrecomputedCellCoverage.v1"
COHORTS = ("development", "confirmation")
EXPECTED_COUNTS = {"development": 22, "confirmation": 20}
ALLOWED_BASENAMES = frozenset({"train.csv", "train3.vs"})
ROW_ID_POLICY = "sha256(dataset-id-sha256 || ordinal-u64-le || source-row-bytes).v1"
ATOMIC_GROUP_POLICY = "sha256(feature-f64-le-bits).v1"
LIST_HASH_POLICY = "sha256(canonical-json-string-list).v1"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def checked_path(dataset_root: Path, basename: str) -> Path:
    if basename not in ALLOWED_BASENAMES:
        raise ValueError(f"forbidden precomputed-cell input: {basename}")
    path = dataset_root / basename
    if path.name != basename or not path.is_file():
        raise FileNotFoundError(path)
    return path


def source_rows(raw: bytes, label: Path) -> list[bytes]:
    rows = raw.split(b"\n")
    if rows and rows[-1] == b"":
        rows.pop()
    if not rows or any(row == b"" for row in rows):
        raise ValueError(f"{label}: empty table or source row")
    return [row[:-1] if row.endswith(b"\r") else row for row in rows]


def inspect_train(path: Path, dataset_id: str) -> dict[str, Any]:
    raw = path.read_bytes()
    rows = source_rows(raw, path)
    dataset_sha256 = sha256_bytes(dataset_id.encode("ascii"))
    stable_row_ids: list[str] = []
    atomic_groups: list[str] = []
    feature_count: int | None = None
    for ordinal, source in enumerate(rows):
        try:
            fields = next(csv.reader(io.StringIO(source.decode("utf-8"))))
        except (UnicodeDecodeError, csv.Error) as error:
            raise ValueError(f"{path}: malformed row {ordinal}") from error
        if len(fields) < 2:
            raise ValueError(f"{path}: row {ordinal} needs features plus target")
        if feature_count is None:
            feature_count = len(fields) - 1
        elif len(fields) != feature_count + 1:
            raise ValueError(f"{path}: ragged row {ordinal}")
        try:
            values = [float(value.strip()) for value in fields]
        except ValueError as error:
            raise ValueError(f"{path}: nonnumeric row {ordinal}") from error
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}: nonfinite row {ordinal}")
        feature_bits = b"".join(struct.pack("<d", value) for value in values[:-1])
        stable_row_ids.append(
            sha256_bytes(
                dataset_sha256.encode("ascii") + struct.pack("<Q", ordinal) + source
            )
        )
        atomic_groups.append(sha256_bytes(feature_bits))
    if feature_count is None or feature_count < 1:
        raise ValueError(f"{path}: no features")
    if len(set(stable_row_ids)) != len(stable_row_ids):
        raise ValueError(f"{path}: duplicate stable row identity")
    return {
        "dataset_sha256": dataset_sha256,
        "train_sha256": sha256_bytes(raw),
        "train_bytes": len(raw),
        "rows": len(rows),
        "raw_features": feature_count,
        "stable_row_ids_sha256": canonical_sha256(stable_row_ids),
        "atomic_groups_sha256": canonical_sha256(atomic_groups),
        "atomic_group_count": len(set(atomic_groups)),
    }


def inspect_cell(
    path: Path, expected_rows: int, expected_columns: int
) -> dict[str, Any]:
    raw = path.read_bytes()
    rows = source_rows(raw, path)
    if len(rows) != expected_rows:
        raise ValueError(f"{path}: {len(rows)} rows, expected {expected_rows}")
    for ordinal, source in enumerate(rows):
        try:
            fields = next(csv.reader(io.StringIO(source.decode("utf-8"))))
            values = [float(value.strip()) for value in fields]
        except (UnicodeDecodeError, csv.Error, ValueError) as error:
            raise ValueError(f"{path}: malformed row {ordinal}") from error
        if len(values) != expected_columns:
            raise ValueError(
                f"{path}: row {ordinal} has {len(values)} columns, "
                f"expected {expected_columns}"
            )
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"{path}: cell value outside finite [0,1] range")
    return {
        "cell_sha256": sha256_bytes(raw),
        "cell_bytes": len(raw),
        "cell_rows": expected_rows,
        "cell_columns": expected_columns,
        "cell_finite_unit_interval": True,
    }


def inspect_dataset(quant_root: Path, dataset_id: str, cohort: str) -> dict[str, Any]:
    if cohort not in COHORTS:
        raise ValueError(f"unknown cohort {cohort}")
    if not dataset_id or any(
        character not in "0123456789abcdef" for character in dataset_id
    ):
        raise ValueError(f"dataset ID is not lowercase hexadecimal: {dataset_id}")
    root = quant_root / dataset_id
    train_path = checked_path(root, "train.csv")
    cell_path = checked_path(root, "train3.vs")
    train = inspect_train(train_path, dataset_id)
    cell = inspect_cell(cell_path, train["rows"], train["raw_features"])
    return {
        "dataset_id": dataset_id,
        "cohort": cohort,
        "train_relative_path": f"{dataset_id}/train.csv",
        "cell_relative_path": f"{dataset_id}/train3.vs",
        **train,
        **cell,
        "test_artifacts_read": False,
    }


def cohort_ids(protocol: dict[str, Any]) -> list[tuple[str, str]]:
    cohorts = protocol.get("cohorts")
    if not isinstance(cohorts, dict):
        raise ValueError("protocol has no cohort mapping")
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for cohort in COHORTS:
        ids = cohorts.get(cohort)
        if not isinstance(ids, list) or len(ids) != EXPECTED_COUNTS[cohort]:
            raise ValueError(
                f"protocol {cohort} cohort must contain {EXPECTED_COUNTS[cohort]} IDs"
            )
        if ids != sorted(ids):
            raise ValueError(
                f"protocol {cohort} cohort is not deterministically ordered"
            )
        for dataset_id in ids:
            if not isinstance(dataset_id, str) or dataset_id in seen:
                raise ValueError("protocol dataset IDs are malformed or duplicated")
            seen.add(dataset_id)
            rows.append((cohort, dataset_id))
    if len(rows) != sum(EXPECTED_COUNTS.values()):
        raise ValueError("protocol registry does not contain exactly 42 datasets")
    return rows


def build_coverage(
    protocol_path: Path,
    quant_root: Path,
    encoder_replay_receipt: Path,
) -> dict[str, Any]:
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    replay_bytes = encoder_replay_receipt.read_bytes()
    replay = json.loads(replay_bytes)
    decision = replay.get("decision", {})
    if (
        replay.get("schema") != "HyperionPrecomputedCellCudaReplay.v1"
        or decision.get("precomputed_cell_matches_replay") is not True
        or decision.get("same_seed_replay_byte_identical") is not True
        or replay.get("test_artifacts_read") is not False
    ):
        raise ValueError("encoder replay receipt does not authorize precomputed cells")
    datasets = [
        inspect_dataset(quant_root, dataset_id, cohort)
        for cohort, dataset_id in cohort_ids(protocol)
    ]
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "protocol_sha256": sha256_bytes(protocol_bytes),
        "encoder_replay_receipt_sha256": sha256_bytes(replay_bytes),
        "row_id_policy": ROW_ID_POLICY,
        "atomic_group_policy": ATOMIC_GROUP_POLICY,
        "list_hash_policy": LIST_HASH_POLICY,
        "allowed_basenames": sorted(ALLOWED_BASENAMES),
        "cohort_counts": dict(EXPECTED_COUNTS),
        "dataset_count": len(datasets),
        "datasets": datasets,
        "cell_available_dataset_count": len(datasets),
        "confirmation_cell_available_dataset_count": sum(
            dataset["cohort"] == "confirmation" for dataset in datasets
        ),
        "target_statistics_emitted": False,
        "test_artifacts_read": False,
    }
    report["coverage_sha256"] = canonical_sha256(report)
    return report


def freeze(path: Path, report: dict[str, Any]) -> None:
    encoded = json.dumps(report, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError(
                "frozen precomputed-cell coverage already exists with other bytes"
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
    parser.add_argument("operation", choices=("freeze", "verify"))
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--quant-root", required=True, type=Path)
    parser.add_argument("--encoder-replay-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    current = build_coverage(
        args.protocol, args.quant_root, args.encoder_replay_receipt
    )
    if args.operation == "freeze":
        freeze(args.output, current)
    else:
        frozen = json.loads(args.output.read_text(encoding="utf-8"))
        if frozen != current:
            raise ValueError("precomputed-cell coverage changed after freezing")
    print(
        json.dumps(
            {
                "coverage_sha256": current["coverage_sha256"],
                "datasets": current["dataset_count"],
                "confirmation": current["confirmation_cell_available_dataset_count"],
                "test_artifacts_read": current["test_artifacts_read"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
