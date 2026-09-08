#!/usr/bin/env python3
"""Matched CatBoost test of raw, old .vs.orig, and latest-model .vs.orig.v2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


SEED = 42
ITERATIONS = 1000
DATASET_IDS = (
    "019de5d76a718163",
    "047e62b52a3f4ede",
    "13b5df533e59a626",
    "17a4542ee4fb9e44",
    "1cb97b1d7314d43d",
    "30ef3eaa6d2971a9",
    "3305a3ad973fb227",
    "3baa7e37fc1f7861",
    "3f781a51af45387e",
    "542e75bc0cfdd1bf",
    "693890aae9dc860b",
    "79e600bc5fcd5da7",
    "ab0239b9ef0ba28e",
    "b253fbfae8460dc1",
    "cf5e74b4e9594944",
    "d2daf53412ab18dc",
    "dee1fc8e9afd81c9",
    "e1c1db5260d0894e",
    "ee56a1015e344c11",
    "fe3cc716a05d036f",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = pd.read_csv(path, header=None).to_numpy(dtype=np.float32)
    if values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError(f"invalid split: {path}")
    features = values[:, :-1]
    target = values[:, -1].astype(np.float64)
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError(f"non-finite raw data: {path}")
    return features, target


def read_cell(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    cell = np.loadtxt(path, delimiter=",", dtype=np.float32, ndmin=2)
    if cell.shape != expected_shape:
        raise ValueError(f"cell shape mismatch: {path}: {cell.shape} != {expected_shape}")
    if not np.isfinite(cell).all():
        raise ValueError(f"non-finite cell data: {path}")
    return cell


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = target - prediction
    rmse = float(np.sqrt(np.mean(residual * residual)))
    denominator = float(np.sum((target - target.mean()) ** 2))
    r2 = float("nan") if denominator == 0 else 1.0 - float(np.sum(residual**2)) / denominator
    return {"rmse": rmse, "r2": r2}


def fit_predict(
    train: np.ndarray,
    target: np.ndarray,
    test: np.ndarray,
    threads: int,
) -> np.ndarray:
    model = CatBoostRegressor(
        iterations=ITERATIONS,
        loss_function="RMSE",
        random_seed=SEED,
        thread_count=threads,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(train, target)
    return np.asarray(model.predict(test), dtype=np.float64)


def run_dataset(root: Path, threads: int) -> dict:
    paths = {
        "train.csv": root / "train.csv",
        "test.csv": root / "test.csv",
        "train3.vs.orig": root / "train3.vs.orig",
        "test3.vs.orig": root / "test3.vs.orig",
        "train3.vs.orig.v2": root / "train3.vs.orig.v2",
        "test3.vs.orig.v2": root / "test3.vs.orig.v2",
    }
    train, train_target = read_split(paths["train.csv"])
    test, test_target = read_split(paths["test.csv"])
    if train.shape[1] != test.shape[1]:
        raise ValueError(f"raw width mismatch: {root}")
    old_train = read_cell(paths["train3.vs.orig"], train.shape)
    old_test = read_cell(paths["test3.vs.orig"], test.shape)
    v2_train = read_cell(paths["train3.vs.orig.v2"], train.shape)
    v2_test = read_cell(paths["test3.vs.orig.v2"], test.shape)

    started = time.monotonic()
    matrices = {
        "raw": (train, test),
        "old": (
            np.concatenate((train, old_train), axis=1),
            np.concatenate((test, old_test), axis=1),
        ),
        "v2": (
            np.concatenate((train, v2_train), axis=1),
            np.concatenate((test, v2_test), axis=1),
        ),
    }
    scores = {
        name: metrics(test_target, fit_predict(x_train, train_target, x_test, threads))
        for name, (x_train, x_test) in matrices.items()
    }
    raw_rmse = scores["raw"]["rmse"]
    old_rmse = scores["old"]["rmse"]
    v2_rmse = scores["v2"]["rmse"]
    return {
        "dataset_id": root.name,
        "train_rows": int(train.shape[0]),
        "test_rows": int(test.shape[0]),
        "raw_width": int(train.shape[1]),
        "append_width": int(train.shape[1] * 2),
        "raw_rmse": raw_rmse,
        "old_rmse": old_rmse,
        "v2_rmse": v2_rmse,
        "old_relative_rmse_improvement": 1.0 - old_rmse / raw_rmse,
        "v2_relative_rmse_improvement": 1.0 - v2_rmse / raw_rmse,
        "v2_relative_rmse_improvement_over_old": 1.0 - v2_rmse / old_rmse,
        "raw_r2": scores["raw"]["r2"],
        "old_r2": scores["old"]["r2"],
        "v2_r2": scores["v2"]["r2"],
        "old_win_vs_raw": old_rmse < raw_rmse,
        "v2_win_vs_raw": v2_rmse < raw_rmse,
        "v2_win_vs_old": v2_rmse < old_rmse,
        "input_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }


def summarize(results: list[dict], field: str) -> dict[str, float]:
    values = np.asarray([result[field] for result in results], dtype=np.float64)
    return {"mean": float(values.mean()), "median": float(np.median(values))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing existing output: {args.output}")

    roots = [args.data_root / dataset_id for dataset_id in DATASET_IDS]
    missing = [str(root) for root in roots if not root.is_dir()]
    if missing:
        raise FileNotFoundError(f"missing dataset roots: {missing}")

    started = time.monotonic()
    results = []
    for index, root in enumerate(roots, start=1):
        result = run_dataset(root, args.threads)
        results.append(result)
        print(
            f"[{index}/20] {root.name} raw={result['raw_rmse']:.8g} "
            f"old={result['old_rmse']:.8g} v2={result['v2_rmse']:.8g}",
            flush=True,
        )

    receipt = {
        "schema": "HyperionOrigV2CatBoostMatchedBenchmark.v1",
        "status": "complete",
        "protocol": {
            "comparison": "raw versus raw+old-cell versus raw+latest-model-old-semantics-cell",
            "metric": "existing test.csv RMSE; lower is better",
            "catboost_iterations": ITERATIONS,
            "catboost_seed": SEED,
            "catboost_threads": args.threads,
            "target_column": "last CSV column, excluded from every feature matrix",
            "test_labels": "used only after all three CatBoost fits to compute metrics",
        },
        "dataset_count": len(results),
        "old_win_count_vs_raw": sum(result["old_win_vs_raw"] for result in results),
        "v2_win_count_vs_raw": sum(result["v2_win_vs_raw"] for result in results),
        "v2_win_count_vs_old": sum(result["v2_win_vs_old"] for result in results),
        "old_relative_rmse_improvement": summarize(
            results, "old_relative_rmse_improvement"
        ),
        "v2_relative_rmse_improvement": summarize(
            results, "v2_relative_rmse_improvement"
        ),
        "v2_relative_rmse_improvement_over_old": summarize(
            results, "v2_relative_rmse_improvement_over_old"
        ),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "environment": {
            "hostname": platform.node(),
            "python": sys.version,
            "catboost": catboost.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "benchmark_source_sha256": sha256_file(Path(__file__)),
        "datasets": results,
    }
    args.output.mkdir(parents=True)
    (args.output / "results.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fields = (
        "dataset_id",
        "raw_rmse",
        "old_rmse",
        "v2_rmse",
        "old_relative_rmse_improvement",
        "v2_relative_rmse_improvement",
        "v2_relative_rmse_improvement_over_old",
        "old_win_vs_raw",
        "v2_win_vs_raw",
        "v2_win_vs_old",
    )
    with (args.output / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result[field] for field in fields})
    print(
        "SUMMARY "
        f"old={receipt['old_win_count_vs_raw']}/20 "
        f"v2={receipt['v2_win_count_vs_raw']}/20 "
        f"v2>old={receipt['v2_win_count_vs_old']}/20",
        flush=True,
    )


if __name__ == "__main__":
    main()
