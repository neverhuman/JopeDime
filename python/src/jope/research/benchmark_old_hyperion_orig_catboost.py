#!/usr/bin/env python3
"""Quick matched CatBoost test of raw versus raw + old Hyperion cells."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


SEED = 42
ITERATIONS = 1000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, header=None)
    values = frame.to_numpy(dtype=np.float32)
    if values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError(f"Invalid split: {path}")
    features = values[:, :-1]
    target = values[:, -1].astype(np.float64)
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError(f"Non-finite raw data: {path}")
    return features, target


def read_cell(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    cell = np.loadtxt(path, delimiter=",", dtype=np.float32, ndmin=2)
    if cell.shape != expected_shape:
        raise ValueError(f"Cell shape mismatch: {path}: {cell.shape} != {expected_shape}")
    if not np.isfinite(cell).all():
        raise ValueError(f"Non-finite cell data: {path}")
    return cell


def metrics(target: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    residual = target - prediction
    rmse = float(np.sqrt(np.mean(residual * residual)))
    denominator = float(np.sum((target - target.mean()) ** 2))
    r2 = float("nan") if denominator == 0 else 1.0 - float(np.sum(residual**2)) / denominator
    return rmse, r2


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


def run_dataset(dataset_root: Path, threads: int) -> dict:
    train_path = dataset_root / "train.csv"
    test_path = dataset_root / "test.csv"
    train_cell_path = dataset_root / "train3.vs.orig"
    test_cell_path = dataset_root / "test3.vs.orig"
    train, train_target = read_split(train_path)
    test, test_target = read_split(test_path)
    if train.shape[1] != test.shape[1]:
        raise ValueError(f"Raw width mismatch: {dataset_root}")
    train_cell = read_cell(train_cell_path, train.shape)
    test_cell = read_cell(test_cell_path, test.shape)

    started = time.monotonic()
    raw_prediction = fit_predict(train, train_target, test, threads)
    append_prediction = fit_predict(
        np.concatenate((train, train_cell), axis=1),
        train_target,
        np.concatenate((test, test_cell), axis=1),
        threads,
    )
    raw_rmse, raw_r2 = metrics(test_target, raw_prediction)
    append_rmse, append_r2 = metrics(test_target, append_prediction)
    return {
        "dataset_id": dataset_root.name,
        "train_rows": int(train.shape[0]),
        "test_rows": int(test.shape[0]),
        "raw_width": int(train.shape[1]),
        "append_width": int(train.shape[1] + train_cell.shape[1]),
        "raw_rmse": raw_rmse,
        "append_rmse": append_rmse,
        "relative_rmse_improvement": 1.0 - append_rmse / raw_rmse,
        "raw_r2": raw_r2,
        "append_r2": append_r2,
        "r2_lift": append_r2 - raw_r2,
        "append_win": append_rmse < raw_rmse,
        "input_sha256": {
            path.name: sha256_file(path)
            for path in (train_path, test_path, train_cell_path, test_cell_path)
        },
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing existing output: {args.output}")

    dataset_roots = sorted(
        path
        for path in args.data_root.iterdir()
        if path.is_dir()
        and (path / "train.csv").is_file()
        and (path / "test.csv").is_file()
        and (path / "train3.vs.orig").is_file()
        and (path / "test3.vs.orig").is_file()
    )
    if len(dataset_roots) != 20:
        raise ValueError(f"Expected exactly 20 complete datasets, found {len(dataset_roots)}")

    started = time.monotonic()
    results = []
    for index, dataset_root in enumerate(dataset_roots, 1):
        result = run_dataset(dataset_root, args.threads)
        results.append(result)
        print(
            f"[{index}/20] {dataset_root.name} "
            f"RMSE {result['raw_rmse']:.6g} -> {result['append_rmse']:.6g} "
            f"({'WIN' if result['append_win'] else 'LOSS'})",
            flush=True,
        )

    wins = sum(result["append_win"] for result in results)
    relative = np.asarray(
        [result["relative_rmse_improvement"] for result in results], dtype=np.float64
    )
    receipt = {
        "schema": "HyperionOldOrigCatBoostQuickBenchmark.v1",
        "status": "complete",
        "protocol": {
            "comparison": "raw versus raw plus cell columns",
            "test_source": "existing test.csv split",
            "metric": "test RMSE; lower is better",
            "catboost_iterations": ITERATIONS,
            "catboost_seed": SEED,
            "catboost_threads": args.threads,
            "target_column": "last CSV column, excluded from all features",
            "cell_files": ["train3.vs.orig", "test3.vs.orig"],
        },
        "dataset_count": len(results),
        "append_win_count": wins,
        "append_loss_count": len(results) - wins,
        "median_relative_rmse_improvement": float(np.median(relative)),
        "mean_relative_rmse_improvement": float(np.mean(relative)),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "datasets": results,
    }
    args.output.mkdir(parents=True)
    (args.output / "results.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset_id",
                "raw_rmse",
                "append_rmse",
                "relative_rmse_improvement",
                "raw_r2",
                "append_r2",
                "r2_lift",
                "append_win",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in writer.fieldnames})
    print(f"SUMMARY {wins}/20 append wins", flush=True)


if __name__ == "__main__":
    main()
