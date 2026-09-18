#!/usr/bin/env python3
"""Genome-boost DL worker: fit(train) -> predict(test), no cross-validation.

Loads a masked genome-boost worker view (train.csv WITH target last, test.csv
WITHOUT any target column, plus the Rust-emitted genome feature blocks when
``--arm boosted``), fits one registered model-zoo regression backend on the
train split only, and writes unscored ``predictions.npy`` plus a worker
receipt. Workers never see test targets and never compute metrics; the
controller-side ``finalize`` in ``genome_boost_cluster_campaign.py`` is the
only scoring path.

The genome feature CSVs are consumed byte-for-byte as emitted by Rust: this
script hashes them and parses them, but never rewrites their numeric bytes.

Every receipt is permanently ``test_adaptive: true`` and
``benchmark_valid: false`` (the genome features were selected against this
dataset's own test RMSE and the train block is in-sample).
"""

from __future__ import annotations

import os

# Deterministic, single-threaded numeric kernels: must be exported before
# numpy/torch import anywhere in this process (mirrors
# golden46_special_execution.THREAD_LIMITS).
THREAD_LIMITS = {
    "BLIS_NUM_THREADS": "1",
    "LIGHTGBM_NUM_THREADS": "1",
    "MKL_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "TBB_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "XGBOOST_NTHREADS": "1",
    "XGBOOST_NUM_THREADS": "1",
}
for _key, _value in THREAD_LIMITS.items():
    os.environ.setdefault(_key, _value)

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


from golden46_execution import (  # noqa: E402  (env limits must precede heavy imports)
    atomic_save_npy,
    atomic_write_json,
    read_json,
    sha256_array,
    sha256_file,
    utc_now,
)

CAMPAIGN_ID = "genome-boost-20260725-v1"
QUANT_ROOT = Path("/home/ubuntu/remote_super/quant")
MODEL_ZOO_PATH = QUANT_ROOT / "models" / "tools" / "model_zoo.py"
WORKER_SCHEMA = "jope-dime.genome-boost-worker.v1"
VIEW_SCHEMA = "jope-dime.genome-boost-worker-view.v1"
TRAIN_FEATURES_FILE = "genome-features-train.csv"
TEST_FEATURES_FILE = "genome-features-test.csv"
ARMS = ("baseline", "boosted")


def stable_seed(dataset: str, backend: str, arm: str) -> int:
    digest = hashlib.sha256(f"{CAMPAIGN_ID}\0{dataset}\0{backend}\0{arm}".encode()).hexdigest()
    return int(digest[:8], 16)


def apply_seed(seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch_seeded = False
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch_seeded = True
    except Exception:
        pass
    return {"seed": seed, "torch_seeded": torch_seeded}


def validate_worker_view(worker_path: Path, dataset: str, arm: str) -> dict[str, Any]:
    manifest = read_json(worker_path / "worker-manifest.json")
    if manifest.get("schema") != VIEW_SCHEMA:
        raise ValueError(f"unexpected worker view schema {manifest.get('schema')!r}")
    if manifest.get("dataset_hash") != dataset:
        raise ValueError("worker view dataset identity mismatch")
    if manifest.get("test_target_present") is not False:
        raise ValueError("worker view must declare the stripped test target")
    required = ["train.csv", "test.csv"]
    if arm == "boosted":
        required += [TRAIN_FEATURES_FILE, TEST_FEATURES_FILE]
    for name in required:
        expected = manifest["artifacts"].get(name)
        if not expected:
            raise ValueError(f"worker view manifest lacks {name}")
        if sha256_file(worker_path / name) != expected:
            raise ValueError(f"worker artifact changed after staging: {name}")
    return manifest


def load_arrays(worker_path: Path, manifest: dict[str, Any], arm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    train = pd.read_csv(worker_path / "train.csv", header=None).to_numpy(dtype=np.float64)
    test = pd.read_csv(worker_path / "test.csv", header=None).to_numpy(dtype=np.float64)
    # The masked test.csv has NO target column: its width is exactly the
    # train feature width.
    if test.shape[1] != train.shape[1] - 1:
        raise ValueError(
            f"masked test width {test.shape[1]} != train feature width {train.shape[1] - 1}"
        )
    if train.shape[0] != int(manifest["train_rows"]) or test.shape[0] != int(manifest["test_rows"]):
        raise ValueError("worker view row counts changed")
    if not np.all(np.isfinite(train)) or not np.all(np.isfinite(test)):
        raise ValueError("worker view contains non-finite values")
    x_train = train[:, :-1].astype(np.float32, copy=False)
    y_train = train[:, -1].astype(np.float64, copy=False)
    x_test = test.astype(np.float32, copy=False)
    k_features = 0
    if arm == "boosted":
        train_block = pd.read_csv(worker_path / TRAIN_FEATURES_FILE, header=None).to_numpy(dtype=np.float32)
        test_block = pd.read_csv(worker_path / TEST_FEATURES_FILE, header=None).to_numpy(dtype=np.float32)
        if train_block.shape[0] != x_train.shape[0] or test_block.shape[0] != x_test.shape[0]:
            raise ValueError("genome feature block row counts are misaligned")
        if train_block.shape[1] != test_block.shape[1] or train_block.shape[1] == 0:
            raise ValueError("genome feature block column counts are misaligned")
        if not np.all(np.isfinite(train_block)) or not np.all(np.isfinite(test_block)):
            raise ValueError("genome feature blocks contain non-finite values")
        k_features = int(train_block.shape[1])
        # Sidecar-block augmentation, HYP3 pattern: block first, then raw x.
        x_train = np.concatenate([train_block, x_train], axis=1)
        x_test = np.concatenate([test_block, x_test], axis=1)
    return x_train, y_train, x_test, k_features


def import_queue_modules() -> tuple[Any, Any]:
    if str(QUANT_ROOT) not in sys.path:
        sys.path.insert(0, str(QUANT_ROOT))
    from models.tools import cluster_model_queue, model_zoo

    return cluster_model_queue, model_zoo


def run(worker_root: Path, dataset: str, backend: str, arm: str, output: Path) -> dict[str, Any]:
    worker_path = worker_root / dataset
    manifest = validate_worker_view(worker_path, dataset, arm)
    seed_info = apply_seed(stable_seed(dataset, backend, arm))
    queue, model_zoo = import_queue_modules()
    specs = model_zoo.specs_by_name("regression")
    spec = specs.get(backend)
    if spec is None:
        raise ValueError(f"unknown regression backend {backend!r}")
    missing = spec.missing_deps()
    if missing:
        raise RuntimeError(f"{backend}: missing dependencies: {missing}")
    if spec.factory is None:
        raise RuntimeError(f"{backend}: model factory is missing")

    started_at = utc_now()
    started = time.monotonic()
    x_train, y_train, x_test, k_features = load_arrays(worker_path, manifest, arm)
    info = model_zoo.DatasetInfo(dataset, x_train.shape[0], x_test.shape[0], x_train.shape[1])
    eligible, reason = spec.check_eligibility(info)
    if not eligible:
        raise RuntimeError(f"{backend}: ineligible for this view: {reason}")

    # fit(train) -> predict(test) ONLY. No cross-validation anywhere.
    model = spec.factory()
    fit_started = time.monotonic()
    model.fit(x_train, y_train)
    training_seconds = time.monotonic() - fit_started
    predict_started = time.monotonic()
    predictions = np.asarray(
        queue.regression_predict_outputs(model, x_test),
        dtype=np.float64,
    ).reshape(-1)
    inference_seconds = time.monotonic() - predict_started
    if predictions.shape != (x_test.shape[0],):
        raise ValueError(f"{backend}: prediction rows {predictions.shape} != {(x_test.shape[0],)}")
    if not np.all(np.isfinite(predictions)):
        raise ValueError(f"{backend}: non-finite predictions")

    prediction_path = output / "predictions.npy"
    atomic_save_npy(prediction_path, predictions)
    inputs_sha256 = {
        "train.csv": sha256_file(worker_path / "train.csv"),
        "test.csv": sha256_file(worker_path / "test.csv"),
        "worker-manifest.json": sha256_file(worker_path / "worker-manifest.json"),
        "model_zoo.py": sha256_file(MODEL_ZOO_PATH),
        "execution_source": sha256_file(Path(__file__).resolve()),
    }
    if arm == "boosted":
        inputs_sha256[TRAIN_FEATURES_FILE] = sha256_file(worker_path / TRAIN_FEATURES_FILE)
        inputs_sha256[TEST_FEATURES_FILE] = sha256_file(worker_path / TEST_FEATURES_FILE)
    receipt = {
        "schema": WORKER_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "test_adaptive": True,
        "benchmark_valid": False,
        "dataset_hash": dataset,
        "backend": backend,
        "arm": arm,
        "no_cross_validation": True,
        "in_sample_train_features": arm == "boosted",
        "seed": seed_info,
        "status": "predicted-unscored",
        "started_at": started_at,
        "ended_at": utc_now(),
        "training_time_seconds": training_seconds,
        "inference_time_seconds": inference_seconds,
        "wall_time_seconds": time.monotonic() - started,
        "train_rows": int(x_train.shape[0]),
        "test_rows": int(x_test.shape[0]),
        "n_features_base": int(x_train.shape[1]) - k_features,
        "k_genome_features": k_features,
        "n_features_used": int(x_train.shape[1]),
        "inputs_sha256": inputs_sha256,
        "predictions": {
            "path": prediction_path.name,
            "sha256": sha256_file(prediction_path),
            "array_sha256": sha256_array(predictions),
            "rows": int(predictions.shape[0]),
        },
        "environment": {
            "host": platform.node(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "pid": os.getpid(),
            "thread_limits": THREAD_LIMITS,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "test_labels_available_to_worker": False,
        "metrics_computed_by_worker": False,
    }
    atomic_write_json(output / "worker-receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-root", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    receipt = run(
        arguments.worker_root,
        arguments.dataset,
        arguments.backend,
        arguments.arm,
        arguments.output,
    )
    print(
        json.dumps(
            {
                "dataset": receipt["dataset_hash"],
                "backend": receipt["backend"],
                "arm": receipt["arm"],
                "n_features_used": receipt["n_features_used"],
                "k_genome_features": receipt["k_genome_features"],
                "rows": receipt["predictions"]["rows"],
                "wall_time_seconds": receipt["wall_time_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
