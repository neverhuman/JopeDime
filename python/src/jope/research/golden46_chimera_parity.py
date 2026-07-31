#!/usr/bin/env python3
"""Validate the converted frozen Chimera against representative Golden42 reports."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from golden46_execution import atomic_write_json, sha256_array, sha256_file, utc_now
from golden46_special_execution import load_chimera_compatible


GOLDEN42_ROOT = Path("/home/ubuntu/remote_super/quant/regression")
DEFAULT_OUTPUT = Path(
    "/home/ubuntu/remote_super/quant/campaigns/golden46-full42-20260723-v1/"
    "manifests/chimera-converted-parity.json"
)
REPRESENTATIVE_DATASETS = (
    "cf5e74b4e9594944",
    "ab0239b9ef0ba28e",
    "542e75bc0cfdd1bf",
)
MAX_RMSE_DELTA = 0.015


def load_dataset(dataset_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = GOLDEN42_ROOT / dataset_id
    train = pd.read_csv(path / "train.csv", header=None).to_numpy(dtype=np.float32)
    test = pd.read_csv(path / "test.csv", header=None).to_numpy(dtype=np.float32)
    return train[:, :-1], train[:, -1], test


def validate_parity(device: str = "cuda") -> dict[str, Any]:
    model, checkpoint_proof = load_chimera_compatible(device)
    comparisons = []
    for dataset_id in REPRESENTATIVE_DATASETS:
        path = GOLDEN42_ROOT / dataset_id
        x_train, y_train, test = load_dataset(dataset_id)
        predictions = np.asarray(
            model.predict(x_train, y_train, test[:, :-1]),
            dtype=np.float64,
        ).reshape(-1)
        target = test[:, -1].astype(np.float64)
        converted_rmse = float(np.sqrt(np.mean(np.square(predictions - target))))
        report_path = path / "reports/chimera_regression.json"
        historical_report = json.loads(report_path.read_text())
        historical_rmse = float(historical_report["metrics"]["rmse"])
        delta = abs(converted_rmse - historical_rmse)
        comparisons.append(
            {
                "dataset_id": dataset_id,
                "train_sha256": sha256_file(path / "train.csv"),
                "test_sha256": sha256_file(path / "test.csv"),
                "historical_report_sha256": sha256_file(report_path),
                "historical_environment": {
                    "device": historical_report["environment"].get("cuda_device_name"),
                    "torch": historical_report["environment"].get("torch"),
                },
                "historical_rmse": historical_rmse,
                "converted_rmse": converted_rmse,
                "absolute_rmse_delta": delta,
                "predictions_array_sha256": sha256_array(predictions),
                "within_tolerance": delta <= MAX_RMSE_DELTA,
            }
        )
    receipt = {
        "schema": "golden46.chimera-converted-parity.v1",
        "created_at": utc_now(),
        "status": (
            "validated"
            if all(item["within_tolerance"] for item in comparisons)
            else "failed"
        ),
        "acceptance": {
            "metric": "absolute RMSE delta",
            "maximum": MAX_RMSE_DELTA,
            "rationale": (
                "The frozen reference reports ran on an NVIDIA A100; this replay "
                "uses the current CUDA device and the frozen experts use bfloat16."
            ),
        },
        "runtime": {"host": platform.node(), "device": device},
        "checkpoint_proof": checkpoint_proof,
        "comparisons": comparisons,
    }
    if receipt["status"] != "validated":
        raise ValueError("converted Chimera failed representative Golden42 parity")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    receipt = validate_parity(args.device)
    atomic_write_json(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
