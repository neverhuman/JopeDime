#!/usr/bin/env python3
"""Empirically verify Golden46 worker and HYP3 test-label isolation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from golden46_execution import (
    CAMPAIGN_ID,
    CANONICAL_ROOT,
    COHORT_PATH,
    DEFAULT_WORKER_ROOT,
    HYP3_BINARY,
    HYP3_MANIFEST,
    HYP3_OUTPUTS,
    HYP3_WEIGHTS,
    atomic_write_json,
    canonical_dataset,
    canonical_json,
    read_canonical_frame,
    regression_metrics,
    sha256_array,
    sha256_bytes,
    sha256_file,
    utc_now,
    write_headerless_csv,
)
from golden46_special_execution import (
    SPECIAL_SOURCE_PATH,
    backend_environment,
)


DEFAULT_INDEX_ROOT = COHORT_PATH.parent.parent / "indexes"
PROBE_METHOD = "xgboost_hist_trees900_depth4_lr0p02"


def run_backend(worker_path: Path, output_path: Path) -> np.ndarray:
    manifest = {
        "test_rows": int(pd.read_csv(worker_path / "test.csv", header=None).shape[0])
    }
    atomic_write_json(worker_path / "worker-manifest.json", manifest)
    command = [
        "/usr/bin/python3",
        str(SPECIAL_SOURCE_PATH),
        "backend-worker",
        "--method",
        PROBE_METHOD,
        "--worker-path",
        str(worker_path),
        "--output-path",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=backend_environment(PROBE_METHOD, "0"),
    )
    if completed.returncode != 0:
        raise RuntimeError(f"leakage probe backend failed: {completed.stdout[-4000:]}")
    return np.load(output_path / "predictions.npy", allow_pickle=False).astype(np.float64)


def run_hyp3(worker_path: Path) -> dict[str, np.ndarray]:
    command = [
        str(HYP3_BINARY),
        "--train",
        str(worker_path / "train.csv"),
        "--test",
        str(worker_path / "test.csv"),
        "--out-cell-train",
        str(worker_path / HYP3_OUTPUTS[0]),
        "--out-cell-test",
        str(worker_path / HYP3_OUTPUTS[1]),
        "--out-row-train",
        str(worker_path / HYP3_OUTPUTS[2]),
        "--out-row-test",
        str(worker_path / HYP3_OUTPUTS[3]),
        "--weights",
        str(HYP3_WEIGHTS),
        "--manifest",
        str(HYP3_MANIFEST),
        "--sweeps",
        "3",
        "--scales",
        "4,7,11,15",
    ]
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "1"
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"HYP3 leakage probe failed: {completed.stdout[-4000:]}")
    return {
        name: pd.read_csv(worker_path / name, header=None).to_numpy(dtype=np.float64)
        for name in HYP3_OUTPUTS
    }


def leakage_audit(
    dataset_id: str,
    *,
    canonical_root: Path,
    worker_root: Path,
    index_root: Path,
) -> dict[str, Any]:
    source_worker = worker_root / dataset_id
    train = pd.read_csv(source_worker / "train.csv", header=None).to_numpy(dtype=np.float64)
    test = pd.read_csv(source_worker / "test.csv", header=None).to_numpy(dtype=np.float64)
    first_test = test.copy()
    second_test = test.copy()
    first_test[:, -1] = 0.0
    second_test[:, -1] = np.linspace(-7.0, 7.0, test.shape[0], dtype=np.float64)
    if np.array_equal(first_test[:, -1], second_test[:, -1]):
        raise AssertionError("leakage probe test targets did not change")

    with tempfile.TemporaryDirectory(prefix=f"golden46-leakage-{dataset_id}-") as temp:
        temporary = Path(temp)
        predictions = []
        hyp3 = []
        for index, probe_test in enumerate((first_test, second_test)):
            worker = temporary / f"worker-{index}"
            output = temporary / f"output-{index}"
            write_headerless_csv(worker / "train.csv", train)
            write_headerless_csv(worker / "test.csv", probe_test)
            predictions.append(run_backend(worker, output))
            hyp3.append(run_hyp3(worker))

    prediction_equal = np.array_equal(predictions[0], predictions[1])
    hyp3_equal = {
        name: np.array_equal(hyp3[0][name], hyp3[1][name])
        for name in HYP3_OUTPUTS
    }
    if not prediction_equal or not all(hyp3_equal.values()):
        raise AssertionError("test-target perturbation changed worker predictions or HYP3")

    source, metadata = canonical_dataset(dataset_id, canonical_root)
    canonical_target = (
        read_canonical_frame(source, metadata, "test").iloc[:, -1].to_numpy(dtype=np.float64)
    )
    perturbed_target = np.roll(canonical_target, 1)
    canonical_metrics = regression_metrics(canonical_target, predictions[0])
    perturbed_metrics = regression_metrics(perturbed_target, predictions[0])
    metrics_changed = canonical_metrics["rmse"] != perturbed_metrics["rmse"]
    if not metrics_changed:
        raise AssertionError("controller RMSE did not respond to changed labels")

    payload = {
        "schema": "golden46.test-label-isolation-audit.v1",
        "campaign_id": CAMPAIGN_ID,
        "dataset_hash": dataset_id,
        "created_at": utc_now(),
        "probe_method": PROBE_METHOD,
        "test_features_sha256": sha256_array(first_test[:, :-1]),
        "test_target_variants": [
            sha256_array(first_test[:, -1]),
            sha256_array(second_test[:, -1]),
        ],
        "predictions": {
            "equal": prediction_equal,
            "sha256": [
                sha256_array(predictions[0]),
                sha256_array(predictions[1]),
            ],
        },
        "hyp3": {
            "equal": hyp3_equal,
            "artifacts": {
                name: [
                    sha256_array(hyp3[0][name]),
                    sha256_array(hyp3[1][name]),
                ]
                for name in HYP3_OUTPUTS
            },
            "implementation": {
                "binary_sha256": sha256_file(HYP3_BINARY),
                "weights_sha256": sha256_file(HYP3_WEIGHTS),
                "manifest_sha256": sha256_file(HYP3_MANIFEST),
            },
        },
        "controller_metrics": {
            "canonical_target_sha256": sha256_array(canonical_target),
            "perturbed_target_sha256": sha256_array(perturbed_target),
            "canonical_rmse": canonical_metrics["rmse"],
            "perturbed_rmse": perturbed_metrics["rmse"],
            "changed": metrics_changed,
        },
        "worker_received_canonical_test_labels": False,
        "passed": prediction_equal and all(hyp3_equal.values()) and metrics_changed,
    }
    payload["content_sha256"] = sha256_bytes(canonical_json(payload))
    atomic_write_json(index_root / f"{dataset_id}.leakage-audit.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--canonical-root", type=Path, default=CANONICAL_ROOT)
    parser.add_argument("--worker-root", type=Path, default=DEFAULT_WORKER_ROOT)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = leakage_audit(
        args.dataset,
        canonical_root=args.canonical_root,
        worker_root=args.worker_root,
        index_root=args.index_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
