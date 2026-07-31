#!/usr/bin/env python3
"""Bridge validated primary Golden46 ASPFM winners into the required-study matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

from golden46_execution import (
    CAMPAIGN_ID,
    CANONICAL_ROOT,
    DEFAULT_ARTIFACT_ROOT,
    atomic_write_json,
    canonical_dataset,
    read_canonical_frame,
    read_json,
    sha256_file,
    utc_now,
)


RUN_ID = "golden46-new3-20260723-v4"
RUN_ROOT = Path("/home/ubuntu/jope-dime/runs") / RUN_ID
DATASETS = ("ZTS_small", "a3bd0a400ec4d7f9", "6054656956182296")
NODES = ("xbabe1", "xbabe2", "xbabe3")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def recompute_rmse(target: list[float], predictions: list[float]) -> float:
    if len(target) != len(predictions) or not target:
        raise ValueError("ASPFM prediction row count mismatch")
    squared_error = 0.0
    for expected, prediction in zip(target, predictions, strict=True):
        squared_error += (prediction - expected) ** 2
    return math.sqrt(squared_error / len(target))


def build_report(dataset_id: str) -> dict[str, Any]:
    dataset_root = RUN_ROOT / "datasets" / dataset_id
    result_path = dataset_root / "final-result.v1.json"
    confirmation_path = dataset_root / "confirmation.v1.json"
    result = read_json(result_path)
    confirmation = read_json(confirmation_path)
    if (
        result.get("schema") != "JopeGolden42AggressiveDatasetResult.v1"
        or result.get("run_id") != RUN_ID
        or result.get("dataset_hash") != dataset_id
        or result.get("outcome") != "confirmed_strict_win"
        or result.get("benchmark_valid") is not False
        or result.get("test_adaptive") is not True
        or result.get("stopping_generation", 0) < 1
        or not float(result.get("best_rmse", math.inf))
        < float(result.get("leader_rmse", -math.inf))
        or result.get("confirmation_sha256") != sha256_file(confirmation_path)
    ):
        raise ValueError(f"{dataset_id}: invalid primary Golden46 ASPFM result")
    if (
        confirmation.get("schema") != "JopeGolden42AggressiveConfirmation.v1"
        or confirmation.get("run_id") != RUN_ID
        or confirmation.get("dataset_hash") != dataset_id
        or confirmation.get("benchmark_valid") is not False
        or confirmation.get("test_adaptive") is not True
        or tuple(confirmation.get("nodes", [])) != NODES
        or len(confirmation.get("receipt_files", [])) != len(NODES)
        or len(confirmation.get("receipt_sha256", [])) != len(NODES)
    ):
        raise ValueError(f"{dataset_id}: invalid three-node confirmation")

    receipts = []
    for relative, expected_sha, node in zip(
        confirmation["receipt_files"],
        confirmation["receipt_sha256"],
        NODES,
        strict=True,
    ):
        path = dataset_root / relative
        if sha256_file(path) != expected_sha:
            raise ValueError(f"{dataset_id}: confirmation receipt hash mismatch")
        receipt = read_json(path)
        if (
            receipt.get("node") != node
            or receipt.get("purpose") != "aggressive-confirmation"
            or receipt.get("dataset_hash") != dataset_id
            or receipt.get("run_id") != RUN_ID
            or receipt.get("semantic_sha256") != result["semantic_sha256"]
            or receipt.get("canonical_genome") != result["canonical_genome"]
        ):
            raise ValueError(f"{dataset_id}: confirmation receipt identity mismatch")
        receipts.append(receipt)
    predictions = [float(value) for value in receipts[0]["predictions"]]
    prediction_bits = b"".join(struct.pack("<d", value) for value in predictions)
    if sha256_bytes(prediction_bits) != confirmation["prediction_bits_sha256"]:
        raise ValueError(f"{dataset_id}: confirmation prediction hash mismatch")
    first_bits = [struct.pack("<d", value) for value in predictions]
    for receipt in receipts[1:]:
        if [struct.pack("<d", float(value)) for value in receipt["predictions"]] != first_bits:
            raise ValueError(f"{dataset_id}: three-node predictions differ")

    source, metadata = canonical_dataset(dataset_id, CANONICAL_ROOT)
    target = (
        read_canonical_frame(source, metadata, "test")
        .iloc[:, -1]
        .to_numpy(dtype=float)
        .tolist()
    )
    rmse = recompute_rmse(target, predictions)
    if struct.pack("<d", rmse) != struct.pack("<Q", confirmation["rmse_bits"]):
        raise ValueError(f"{dataset_id}: controller RMSE bits differ from confirmation")
    if struct.pack("<d", rmse) != struct.pack("<d", float(result["best_rmse"])):
        raise ValueError(f"{dataset_id}: result RMSE differs from controller recomputation")

    return {
        "schema": "golden46.aspfm-validated-report.v1",
        "campaign_id": CAMPAIGN_ID,
        "algorithm": "jope-dime",
        "model_name": "jope-dime",
        "dataset_hash": dataset_id,
        "task_type": "regression",
        "status": "completed",
        "benchmark_valid": False,
        "test_adaptive": True,
        "run_date": utc_now(),
        "metrics": {"rmse": rmse},
        "provenance": {
            "run_id": RUN_ID,
            "arm": "primary-provenance-excluded",
            "leaderboard_default": True,
            "final_result": str(result_path),
            "final_result_sha256": sha256_file(result_path),
            "confirmation": str(confirmation_path),
            "confirmation_sha256": sha256_file(confirmation_path),
            "three_node_confirmed": True,
            "nodes": list(NODES),
            "prediction_bits_sha256": confirmation["prediction_bits_sha256"],
            "canonical_test_sha256": sha256_file(source / "test.csv"),
            "controller_computed_metrics": True,
        },
        "evolution": {
            "leader_rmse": result["leader_rmse"],
            "stopping_generation": result["stopping_generation"],
            "best_slot": result["best_slot"],
            "semantic_sha256": result["semantic_sha256"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args()
    datasets = (args.dataset,) if args.dataset else DATASETS
    reports = []
    for dataset_id in datasets:
        report = build_report(dataset_id)
        atomic_write_json(
            args.artifact_root / dataset_id / "jope-dime" / "report.json",
            report,
        )
        reports.append(report)
    print(json.dumps(reports, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
