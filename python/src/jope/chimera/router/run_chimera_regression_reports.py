"""Run FINAL Chimera regression reports over real Quant regression datasets.

The runner is intentionally report-oriented: each worker loads Chimera once on a
single GPU, processes many dataset folders, and writes the same JSON shape used
by the existing Quant regression benchmarks.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import platform
import queue
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = Path("/home/ubuntu/quant/regression")
DEFAULT_CKPT = Path(
    "/home/ubuntu/fable_god/chimeria_regression_shuffle/run_FINAL_e10_trunk/FINAL_stackfusionreg.pt"
)
ALGORITHM = "chimera_regression"
MODEL_NAME = "chimera_regression"
DEFAULT_REPORT_NAME = "chimera_regression.json"
DEFAULT_SUMMARY_PREFIX = "chimera_regression"
HYP3_ALGORITHM = "hyp3_chimera_regression"
HYP3_MODEL_NAME = "hyp3_chimera_regression"
HYP3_REPORT_NAME = "hyp3_chimera_regression.json"
HYP3_SUMMARY_PREFIX = "hyp3_chimera_regression"
DEFAULT_HYP3_PAYLOAD_ROOTS = (
    Path("/tmp/hyp3_chimera_regression_payloads"),
    Path("/home/ubuntu/~quant/regression"),
)
DEFAULT_HYP3_SCRATCH_ROOT = Path("/tmp/hyp3_chimera_regression")
HYP3_REQUIRED_FILES = (
    "train3.vs",
    "test3.vs",
    "hyperion3train.vs",
    "hyperion3test.vs",
)
THREAD_LIMIT_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "TBB_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


@dataclass(frozen=True)
class WorkItem:
    dataset_hash: str
    dataset_dir: str
    report_path: str
    attempt: int = 0
    payload_dir: str | None = None


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(json_safe(payload), fh, indent=2, sort_keys=True)
            fh.write("\n")
        tmp.replace(path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def read_json_dict(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def meta_count(meta: dict[str, Any], key: str) -> Any:
    value = meta.get(key)
    if value is not None:
        return value
    counts = meta.get("counts")
    if isinstance(counts, dict):
        return counts.get(key)
    return None


def selected_real_regression(meta: dict[str, Any]) -> bool:
    if meta.get("task_type") != "regression":
        return False
    origin = meta.get("origin")
    if origin == "synthetic":
        return False
    if origin == "real":
        return True
    return origin is None and bool(meta.get("source_path"))


def unique_strings(values: list[str | None]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def find_hyp3_payload_dir(
    dataset_hash: str, dataset_name: str, payload_roots: tuple[Path, ...]
) -> Path | None:
    for root in payload_roots:
        for name in unique_strings([dataset_hash, dataset_name]):
            candidate = root / name
            if candidate.exists():
                return candidate
    return None


def expected_hyp3_paths(payload_dir: Path | None) -> dict[str, str | None]:
    return {
        file_name: str(payload_dir / file_name) if payload_dir is not None else None
        for file_name in HYP3_REQUIRED_FILES
    }


def report_status(path: Path, algorithm: str = ALGORITHM) -> str | None:
    data = read_json_dict(path)
    if not data:
        return None
    if data.get("algorithm") not in (None, algorithm):
        return None
    return data.get("status") if isinstance(data.get("status"), str) else None


def report_is_completed(path: Path, algorithm: str = ALGORITHM) -> bool:
    return report_status(path, algorithm) == "completed"


def build_worklist(
    dataset_root: Path,
    report_name: str,
    limit: int | None,
    force: bool,
    failed_only: bool,
    algorithm: str = ALGORITHM,
    feature_augmentation: str = "none",
    hyp3_payload_roots: tuple[Path, ...] = (),
) -> tuple[list[WorkItem], list[WorkItem], int, int]:
    selected: list[WorkItem] = []
    skipped_missing_files = 0
    for dataset_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        meta_path = dataset_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = read_json_dict(meta_path)
        if not selected_real_regression(meta):
            continue
        if not (dataset_dir / "train.csv").exists() or not (dataset_dir / "test.csv").exists():
            skipped_missing_files += 1
            continue
        dataset_hash = str(meta.get("dataset_hash") or dataset_dir.name)
        payload_dir = None
        if feature_augmentation == "hyp3":
            found_payload = find_hyp3_payload_dir(dataset_hash, dataset_dir.name, hyp3_payload_roots)
            payload_dir = str(found_payload) if found_payload is not None else None
        selected.append(
            WorkItem(
                dataset_hash=dataset_hash,
                dataset_dir=str(dataset_dir),
                report_path=str(dataset_dir / "reports" / report_name),
                payload_dir=payload_dir,
            )
        )

    if limit is not None and limit > 0:
        selected = selected[:limit]

    pending: list[WorkItem] = []
    skipped_completed = 0
    for item in selected:
        status = report_status(Path(item.report_path), algorithm)
        if failed_only and status != "failed":
            continue
        if not force and status == "completed":
            skipped_completed += 1
            continue
        pending.append(item)
    return selected, pending, skipped_completed, skipped_missing_files


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    from sklearn.metrics import (
        explained_variance_score,
        max_error,
        mean_absolute_error,
        mean_squared_error,
        median_absolute_error,
        r2_score,
    )

    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"prediction shape {y_pred.shape} does not match target shape {y_true.shape}")
    mse = float(mean_squared_error(y_true, y_pred))
    denom = np.maximum(np.abs(y_true), 1e-8)
    return {
        "explained_variance": float(explained_variance_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "max_error": float(max_error(y_true, y_pred)),
        "mean_absolute_percentage_error_eps": float(np.mean(np.abs((y_true - y_pred) / denom))),
        "median_absolute_error": float(median_absolute_error(y_true, y_pred)),
        "mse": mse,
        "prediction_max": float(np.max(y_pred)) if y_pred.size else None,
        "prediction_min": float(np.min(y_pred)) if y_pred.size else None,
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(math.sqrt(mse)),
        "target_max": float(np.max(y_true)) if y_true.size else None,
        "target_min": float(np.min(y_true)) if y_true.size else None,
    }


def environment_payload(ckpt: Path, gpu_device: str | int | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "checkpoint": str(ckpt),
        "repo_root": str(REPO_ROOT),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "thread_limits": {k: os.environ.get(k) for k in THREAD_LIMIT_VARS},
    }
    try:
        import sklearn

        payload["sklearn"] = sklearn.__version__
    except Exception:
        payload["sklearn"] = None
    try:
        import torch

        payload.update(
            {
                "torch": torch.__version__,
                "torch_cuda_available": torch.cuda.is_available(),
                "torch_num_threads": torch.get_num_threads(),
            }
        )
        if torch.cuda.is_available():
            payload["cuda_device_name"] = torch.cuda.get_device_name(0)
            payload["cuda_device_count_visible"] = torch.cuda.device_count()
            payload["cuda_memory_allocated_bytes"] = int(torch.cuda.memory_allocated(0))
            payload["cuda_max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(0))
    except Exception as exc:
        payload["torch_error"] = str(exc)
    if gpu_device is not None:
        payload["gpu_device"] = gpu_device
    return payload


def dataset_report_fields(
    dataset_hash: str, dataset_dir: Path, meta: dict[str, Any], ds: Any | None = None
) -> dict[str, Any]:
    if ds is not None:
        train_rows = int(ds.Xtr.shape[0])
        test_rows = int(ds.Xte.shape[0])
        n_features = int(ds.Xtr.shape[1])
        task_type = str(ds.task)
    else:
        train_rows = meta_count(meta, "train_rows")
        test_rows = meta_count(meta, "test_rows")
        n_features = meta_count(meta, "n_features")
        task_type = str(meta.get("task_type") or "regression")
    return {
        "dataset_hash": str(meta.get("dataset_hash") or dataset_hash or dataset_dir.name),
        "task_type": task_type,
        "source_path": meta.get("source_path"),
        "target_names": meta.get("target_names"),
        "train_rows": int(train_rows) if train_rows is not None else None,
        "test_rows": int(test_rows) if test_rows is not None else None,
        "n_features": int(n_features) if n_features is not None else None,
    }


def load_numeric_matrix(path: Path, label: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing {label}: {path}")
    try:
        import pandas as pd

        frame = pd.read_csv(path, header=None)
        arr = frame.to_numpy(dtype=np.float32, copy=True)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"{label} is not a numeric matrix: {path}: {exc}") from exc
    if arr.ndim != 2:
        raise ValueError(f"{label} is not a 2D numeric matrix: {path} shape={arr.shape}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{label} is empty: {path} shape={arr.shape}")
    finite = np.isfinite(arr)
    if not bool(finite.all()):
        row, col = np.argwhere(~finite)[0]
        raise ValueError(f"{label} contains a non-finite value at row={int(row)} col={int(col)}: {path}")
    return arr


def hyp3_feature_metadata_stub(
    item: WorkItem,
    dataset_dir: Path,
    payload_roots: tuple[Path, ...],
    payload_dir: Path | None = None,
) -> dict[str, Any]:
    resolved_payload = payload_dir
    if resolved_payload is None:
        resolved_payload = (
            Path(item.payload_dir)
            if item.payload_dir
            else find_hyp3_payload_dir(
                item.dataset_hash,
                dataset_dir.name,
                payload_roots,
            )
        )
    return {
        "name": HYP3_ALGORITHM,
        "mode": "prepend_feature_blocks",
        "original_feature_count": None,
        "augmented_feature_count": None,
        "block_names": ["train3", "hyperion3"],
        "block_columns": {"train3": None, "hyperion3": None},
        "source_paths": {
            "train_csv": str(dataset_dir / "train.csv"),
            "test_csv": str(dataset_dir / "test.csv"),
            **expected_hyp3_paths(resolved_payload),
        },
        "payload_dir": str(resolved_payload) if resolved_payload is not None else None,
        "payload_roots": [str(root) for root in payload_roots],
        "feature_order": ["train3", "hyperion3", "original_csv_features"],
    }


def load_hyp3_augmented_dataset(
    item: WorkItem,
    meta: dict[str, Any],
    payload_roots: tuple[Path, ...],
) -> tuple[Any, dict[str, Any]]:
    from python.fusion.data import QuantDataset

    dataset_dir = Path(item.dataset_dir)
    payload_dir = (
        Path(item.payload_dir)
        if item.payload_dir
        else find_hyp3_payload_dir(
            item.dataset_hash,
            dataset_dir.name,
            payload_roots,
        )
    )
    if payload_dir is None:
        checked = [
            str(root / name)
            for root in payload_roots
            for name in unique_strings([item.dataset_hash, dataset_dir.name])
        ]
        raise FileNotFoundError(f"missing HYP3 payload directory for {item.dataset_hash}; checked={checked}")

    train_csv = load_numeric_matrix(dataset_dir / "train.csv", "original train.csv")
    test_csv = load_numeric_matrix(dataset_dir / "test.csv", "original test.csv")
    if train_csv.shape[1] < 2 or test_csv.shape[1] < 2:
        raise ValueError(
            f"original train/test CSV must contain at least one feature column plus target: "
            f"train_shape={train_csv.shape} test_shape={test_csv.shape}"
        )
    if train_csv.shape[1] != test_csv.shape[1]:
        raise ValueError(
            f"original train/test column mismatch: train_cols={train_csv.shape[1]} test_cols={test_csv.shape[1]}"
        )

    train3 = load_numeric_matrix(payload_dir / "train3.vs", "HYP3 train3.vs")
    test3 = load_numeric_matrix(payload_dir / "test3.vs", "HYP3 test3.vs")
    hyperion3_train = load_numeric_matrix(payload_dir / "hyperion3train.vs", "HYP3 hyperion3train.vs")
    hyperion3_test = load_numeric_matrix(payload_dir / "hyperion3test.vs", "HYP3 hyperion3test.vs")

    if train3.shape[0] != train_csv.shape[0]:
        raise ValueError(
            f"HYP3 train3 row mismatch: block_rows={train3.shape[0]} train_rows={train_csv.shape[0]}"
        )
    if hyperion3_train.shape[0] != train_csv.shape[0]:
        raise ValueError(
            f"HYP3 hyperion3train row mismatch: block_rows={hyperion3_train.shape[0]} train_rows={train_csv.shape[0]}"
        )
    if test3.shape[0] != test_csv.shape[0]:
        raise ValueError(
            f"HYP3 test3 row mismatch: block_rows={test3.shape[0]} test_rows={test_csv.shape[0]}"
        )
    if hyperion3_test.shape[0] != test_csv.shape[0]:
        raise ValueError(
            f"HYP3 hyperion3test row mismatch: block_rows={hyperion3_test.shape[0]} test_rows={test_csv.shape[0]}"
        )
    if train3.shape[1] != test3.shape[1]:
        raise ValueError(
            f"HYP3 train3/test3 column mismatch: train_cols={train3.shape[1]} test_cols={test3.shape[1]}"
        )
    if hyperion3_train.shape[1] != hyperion3_test.shape[1]:
        raise ValueError(
            f"HYP3 hyperion3 train/test column mismatch: "
            f"train_cols={hyperion3_train.shape[1]} test_cols={hyperion3_test.shape[1]}"
        )

    Xtr_original = train_csv[:, :-1]
    Xte_original = test_csv[:, :-1]
    ytr = train_csv[:, -1].astype(np.float32, copy=False)
    yte = test_csv[:, -1].astype(np.float32, copy=False)
    Xtr_aug = np.concatenate([train3, hyperion3_train, Xtr_original], axis=1).astype(np.float32, copy=False)
    Xte_aug = np.concatenate([test3, hyperion3_test, Xte_original], axis=1).astype(np.float32, copy=False)

    feature_metadata = {
        "name": HYP3_ALGORITHM,
        "mode": "prepend_feature_blocks",
        "original_feature_count": int(Xtr_original.shape[1]),
        "augmented_feature_count": int(Xtr_aug.shape[1]),
        "block_names": ["train3", "hyperion3"],
        "block_columns": {
            "train3": int(train3.shape[1]),
            "hyperion3": int(hyperion3_train.shape[1]),
        },
        "source_paths": {
            "train_csv": str(dataset_dir / "train.csv"),
            "test_csv": str(dataset_dir / "test.csv"),
            "train3.vs": str(payload_dir / "train3.vs"),
            "test3.vs": str(payload_dir / "test3.vs"),
            "hyperion3train.vs": str(payload_dir / "hyperion3train.vs"),
            "hyperion3test.vs": str(payload_dir / "hyperion3test.vs"),
        },
        "payload_dir": str(payload_dir),
        "payload_roots": [str(root) for root in payload_roots],
        "feature_order": ["train3", "hyperion3", "original_csv_features"],
        "split_rows": {"train": int(Xtr_aug.shape[0]), "test": int(Xte_aug.shape[0])},
    }
    ds = QuantDataset(
        dataset_dir.name,
        "regression",
        Xtr_aug,
        ytr,
        Xte_aug,
        yte,
        int(Xtr_aug.shape[1]),
        0,
        meta,
    )
    return ds, feature_metadata


def audit_hyp3_payloads(selected: list[WorkItem], payload_roots: tuple[Path, ...]) -> dict[str, Any]:
    complete = 0
    missing: list[dict[str, Any]] = []
    for item in selected:
        dataset_dir = Path(item.dataset_dir)
        payload_dir = (
            Path(item.payload_dir)
            if item.payload_dir
            else find_hyp3_payload_dir(
                item.dataset_hash,
                dataset_dir.name,
                payload_roots,
            )
        )
        missing_files = []
        if payload_dir is None:
            missing_files = list(HYP3_REQUIRED_FILES)
        else:
            missing_files = [
                file_name for file_name in HYP3_REQUIRED_FILES if not (payload_dir / file_name).exists()
            ]
        if missing_files:
            missing.append(
                {
                    "dataset_hash": item.dataset_hash,
                    "dataset_dir": item.dataset_dir,
                    "payload_dir": str(payload_dir) if payload_dir is not None else None,
                    "missing_files": missing_files,
                }
            )
        else:
            complete += 1
    return {
        "payload_roots": [str(root) for root in payload_roots],
        "required_files": list(HYP3_REQUIRED_FILES),
        "complete_payloads": complete,
        "missing_or_incomplete_payloads": len(missing),
        "missing": missing,
    }


def failed_report(
    item: WorkItem,
    ckpt: Path,
    gpu_device: str | int | None,
    started_at: str,
    ended_at: str,
    wall_time_seconds: float,
    error: str,
    traceback_text: str | None = None,
    algorithm: str = ALGORITHM,
    model_name: str = MODEL_NAME,
    model_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_dir = Path(item.dataset_dir)
    meta = read_json_dict(dataset_dir / "meta.json")
    payload = {
        "algorithm": algorithm,
        "model_name": model_name,
        "run_date": ended_at,
        "started_at": started_at,
        "ended_at": ended_at,
        "status": "failed",
        "training_time_seconds": 0.0,
        "inference_time_seconds": None,
        "wall_time_seconds": float(wall_time_seconds),
        "inference_rows_per_second": None,
        "metrics": None,
        "model_class": "chimera.Chimera",
        "checkpoint": str(ckpt),
        "gpu_device": gpu_device,
        "error": error,
        "failure_message": error,
        "environment": environment_payload(ckpt, gpu_device),
    }
    if traceback_text:
        payload["traceback"] = traceback_text
    if model_metadata is not None:
        payload["model_metadata"] = model_metadata
    payload.update(dataset_report_fields(item.dataset_hash, dataset_dir, meta, None))
    return payload


def write_failed_report(
    item: WorkItem,
    ckpt: Path,
    gpu_device: str | int | None,
    error: str,
    traceback_text: str | None = None,
    started_at: str | None = None,
    wall_start: float | None = None,
    algorithm: str = ALGORITHM,
    model_name: str = MODEL_NAME,
    model_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = started_at or utc_now()
    ended = utc_now()
    wall = time.perf_counter() - wall_start if wall_start is not None else 0.0
    report = failed_report(
        item,
        ckpt,
        gpu_device,
        started,
        ended,
        wall,
        error,
        traceback_text,
        algorithm=algorithm,
        model_name=model_name,
        model_metadata=model_metadata,
    )
    atomic_write_json(Path(item.report_path), report)
    return report


def run_one(
    model: Any,
    item: WorkItem,
    ckpt: Path,
    gpu_device: str | int | None,
    algorithm: str = ALGORITHM,
    model_name: str = MODEL_NAME,
    feature_augmentation: str = "none",
    hyp3_payload_roots: tuple[Path, ...] = (),
) -> dict[str, Any]:
    started_at = utc_now()
    wall_start = time.perf_counter()
    dataset_dir = Path(item.dataset_dir)
    report_path = Path(item.report_path)
    meta = read_json_dict(dataset_dir / "meta.json")
    model_metadata: dict[str, Any] | None = None
    if feature_augmentation == "hyp3":
        model_metadata = {
            "feature_augmentation": hyp3_feature_metadata_stub(item, dataset_dir, hyp3_payload_roots),
        }
    try:
        if feature_augmentation == "hyp3":
            ds, feature_metadata = load_hyp3_augmented_dataset(item, meta, hyp3_payload_roots)
            model_metadata = {"feature_augmentation": feature_metadata}
        else:
            from python.fusion.data import load_dataset

            ds = load_dataset(str(dataset_dir))
        if ds.task != "regression":
            raise ValueError(f"expected regression dataset, got task={ds.task!r}")
        infer_start = time.perf_counter()
        y_pred = model.predict(ds.Xtr, ds.ytr, ds.Xte)
        infer_seconds = time.perf_counter() - infer_start
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
        metrics = regression_metrics(ds.yte, y_pred)
        ended_at = utc_now()
        wall_seconds = time.perf_counter() - wall_start
        test_rows = int(ds.Xte.shape[0])
        report = {
            "algorithm": algorithm,
            "model_name": model_name,
            "run_date": ended_at,
            "started_at": started_at,
            "ended_at": ended_at,
            "status": "completed",
            "training_time_seconds": 0.0,
            "inference_time_seconds": float(infer_seconds),
            "wall_time_seconds": float(wall_seconds),
            "inference_rows_per_second": float(test_rows / infer_seconds) if infer_seconds > 0 else None,
            "metrics": metrics,
            "model_class": "chimera.Chimera",
            "checkpoint": str(ckpt),
            "gpu_device": gpu_device,
            "environment": environment_payload(ckpt, gpu_device),
        }
        if model_metadata is not None:
            report["model_metadata"] = model_metadata
        report.update(dataset_report_fields(item.dataset_hash, dataset_dir, meta, ds))
        atomic_write_json(report_path, report)
        return {
            "status": "completed",
            "dataset_hash": item.dataset_hash,
            "report_path": str(report_path),
            "metrics": metrics,
            "inference_time_seconds": float(infer_seconds),
            "wall_time_seconds": float(wall_seconds),
        }
    except Exception as exc:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        tb = traceback.format_exc()
        report = write_failed_report(
            item,
            ckpt,
            gpu_device,
            str(exc),
            tb,
            started_at=started_at,
            wall_start=wall_start,
            algorithm=algorithm,
            model_name=model_name,
            model_metadata=model_metadata,
        )
        return {
            "status": "failed",
            "dataset_hash": item.dataset_hash,
            "report_path": str(report_path),
            "error": str(exc),
            "wall_time_seconds": report.get("wall_time_seconds"),
        }


def worker_main(
    worker_id: int,
    gpu_id: str,
    task_queue: Any,
    result_queue: Any,
    ckpt: str,
    device: str,
    thread_limit: int,
    algorithm: str,
    model_name: str,
    feature_augmentation: str,
    hyp3_payload_roots: tuple[str, ...],
) -> None:
    for var in THREAD_LIMIT_VARS:
        os.environ[var] = str(thread_limit)
    os.environ.setdefault("OMP_DYNAMIC", "FALSE")
    os.environ.setdefault("MKL_DYNAMIC", "FALSE")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    sys.path.insert(0, str(REPO_ROOT))

    ckpt_path = Path(ckpt)
    try:
        import torch

        torch.set_num_threads(thread_limit)
        torch.backends.cuda.matmul.allow_tf32 = True
        from chimera import Chimera

        load_start = time.perf_counter()
        model = Chimera(ckpt=str(ckpt_path), device=device)
        result_queue.put(
            {
                "event": "ready",
                "worker_id": worker_id,
                "gpu_device": gpu_id,
                "model_load_seconds": time.perf_counter() - load_start,
            }
        )
    except BaseException as exc:
        result_queue.put(
            {
                "event": "init_failed",
                "worker_id": worker_id,
                "gpu_device": gpu_id,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return

    while True:
        raw = task_queue.get()
        if raw is None:
            result_queue.put({"event": "stopped", "worker_id": worker_id, "gpu_device": gpu_id})
            return
        item = WorkItem(**raw)
        result_queue.put(
            {
                "event": "started",
                "worker_id": worker_id,
                "gpu_device": gpu_id,
                "item": raw,
                "dataset_hash": item.dataset_hash,
            }
        )
        try:
            result = run_one(
                model,
                item,
                ckpt_path,
                gpu_id,
                algorithm=algorithm,
                model_name=model_name,
                feature_augmentation=feature_augmentation,
                hyp3_payload_roots=tuple(Path(root) for root in hyp3_payload_roots),
            )
            result.update({"event": "done", "worker_id": worker_id, "gpu_device": gpu_id})
            result_queue.put(result)
        except BaseException as exc:
            result_queue.put(
                {
                    "event": "worker_error",
                    "worker_id": worker_id,
                    "gpu_device": gpu_id,
                    "item": raw,
                    "dataset_hash": item.dataset_hash,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            return


def parse_gpus(value: str) -> list[str]:
    if value.strip().lower() == "auto":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            gpus = [part.strip() for part in visible.split(",") if part.strip()]
            if gpus:
                return gpus
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            gpus = [line.strip() for line in out.splitlines() if line.strip()]
            if gpus:
                return gpus
        except Exception:
            pass
        try:
            import torch

            count = torch.cuda.device_count()
            if count > 0:
                return [str(i) for i in range(count)]
        except Exception:
            pass
        raise ValueError("--gpus auto could not discover any visible GPUs")
    gpus = [part.strip() for part in value.split(",") if part.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    return gpus


def terminal_report_for_item(item: WorkItem) -> tuple[str | None, dict[str, Any]]:
    path = Path(item.report_path)
    data = read_json_dict(path)
    status = data.get("status") if isinstance(data.get("status"), str) else None
    if status in {"completed", "failed"}:
        return status, data
    return None, data


def run_pending(args: argparse.Namespace, pending: list[WorkItem]) -> dict[str, Any]:
    if not pending:
        return {"attempted": 0, "completed": 0, "failed": 0, "timeouts": 0, "worker_failures": 0}

    gpus = parse_gpus(args.gpus)
    worker_specs = [(gpu, slot) for gpu in gpus for slot in range(args.workers_per_gpu)]
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    for item in pending:
        task_queue.put(asdict(item))

    workers: dict[int, dict[str, Any]] = {}
    in_progress: dict[int, dict[str, Any]] = {}
    dead_workers: set[int] = set()
    completed_hashes: set[str] = set()
    worker_seq = 0
    log_prefix = getattr(args, "log_prefix", "chimera")

    def start_worker(gpu_id: str) -> None:
        nonlocal worker_seq
        worker_id = worker_seq
        worker_seq += 1
        proc = ctx.Process(
            target=worker_main,
            args=(
                worker_id,
                gpu_id,
                task_queue,
                result_queue,
                str(args.ckpt),
                args.device,
                args.thread_limit,
                args.algorithm,
                args.model_name,
                args.feature_augmentation,
                tuple(str(root) for root in args.hyp3_payload_roots),
            ),
            daemon=False,
        )
        proc.start()
        workers[worker_id] = {
            "process": proc,
            "gpu_id": gpu_id,
            "ready": False,
            "started_at": time.monotonic(),
        }

    for gpu_id, _slot in worker_specs:
        start_worker(gpu_id)
        if args.startup_stagger_seconds > 0:
            time.sleep(args.startup_stagger_seconds)

    total = len(pending)
    counts = {
        "attempted": total,
        "completed": 0,
        "failed": 0,
        "timeouts": 0,
        "worker_failures": 0,
        "init_failures": 0,
    }
    run_started = time.monotonic()
    last_progress_print = run_started

    def mark_terminal(item: WorkItem, status: str) -> None:
        if item.dataset_hash in completed_hashes:
            return
        completed_hashes.add(item.dataset_hash)
        if status == "completed":
            counts["completed"] += 1
        else:
            counts["failed"] += 1

    def stop_worker(worker_id: int) -> None:
        info = workers.get(worker_id)
        if not info:
            return
        proc = info["process"]
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=10)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=10)
        dead_workers.add(worker_id)

    def maybe_print_progress(force: bool = False) -> None:
        nonlocal last_progress_print
        now = time.monotonic()
        done = counts["completed"] + counts["failed"]
        if not force and done % max(1, args.progress_every) != 0 and now - last_progress_print < 30:
            return
        last_progress_print = now
        elapsed = now - run_started
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = total - done
        eta = remaining / rate if rate > 0 else None
        eta_text = f"{eta:.1f}s" if eta is not None else "n/a"
        print(
            f"[{log_prefix}] done={done}/{total} completed={counts['completed']} failed={counts['failed']} "
            f"ready={sum(1 for w in workers.values() if w.get('ready'))} active={len(in_progress)} "
            f"rate={rate:.2f}/s eta={eta_text}",
            flush=True,
        )

    while counts["completed"] + counts["failed"] < total:
        try:
            msg = result_queue.get(timeout=1.0)
        except queue.Empty:
            msg = None

        if msg is not None:
            event = msg.get("event")
            worker_id = int(msg.get("worker_id", -1))
            if event == "ready":
                if worker_id in workers:
                    workers[worker_id]["ready"] = True
                    workers[worker_id]["model_load_seconds"] = msg.get("model_load_seconds")
                print(
                    f"[{log_prefix}] worker={worker_id} gpu={msg.get('gpu_device')} ready "
                    f"load={float(msg.get('model_load_seconds') or 0.0):.1f}s",
                    flush=True,
                )
            elif event == "started":
                raw_item = msg.get("item")
                if isinstance(raw_item, dict):
                    in_progress[worker_id] = {
                        "item": WorkItem(**raw_item),
                        "started": time.monotonic(),
                        "gpu_device": msg.get("gpu_device"),
                    }
            elif event == "done":
                raw = in_progress.pop(worker_id, {}).get("item")
                if raw is None:
                    raw = WorkItem(
                        dataset_hash=str(msg.get("dataset_hash")),
                        dataset_dir="",
                        report_path=str(msg.get("report_path")),
                    )
                mark_terminal(raw, str(msg.get("status") or "failed"))
                maybe_print_progress()
            elif event == "init_failed":
                counts["init_failures"] += 1
                counts["worker_failures"] += 1
                if worker_id in workers:
                    workers[worker_id]["init_failed"] = True
                print(
                    f"[{log_prefix}] worker={worker_id} gpu={msg.get('gpu_device')} init_failed: {msg.get('error')}",
                    flush=True,
                )
            elif event == "worker_error":
                counts["worker_failures"] += 1
                raw_item = msg.get("item")
                if isinstance(raw_item, dict):
                    item = WorkItem(**raw_item)
                    if item.dataset_hash not in completed_hashes:
                        write_failed_report(
                            item,
                            Path(args.ckpt),
                            msg.get("gpu_device"),
                            f"worker exited unexpectedly: {msg.get('error')}",
                            msg.get("traceback"),
                            algorithm=args.algorithm,
                            model_name=args.model_name,
                        )
                        mark_terminal(item, "failed")
                print(
                    f"[{log_prefix}] worker={worker_id} gpu={msg.get('gpu_device')} error: {msg.get('error')}",
                    flush=True,
                )

        now = time.monotonic()
        if args.dataset_timeout_seconds > 0:
            for worker_id, active in list(in_progress.items()):
                elapsed = now - float(active["started"])
                if elapsed <= args.dataset_timeout_seconds:
                    continue
                item = active["item"]
                gpu_device = active.get("gpu_device")
                counts["timeouts"] += 1
                counts["worker_failures"] += 1
                print(
                    f"[{log_prefix}] worker={worker_id} gpu={gpu_device} timeout "
                    f"dataset={item.dataset_hash} elapsed={elapsed:.1f}s",
                    flush=True,
                )
                stop_worker(worker_id)
                in_progress.pop(worker_id, None)
                write_failed_report(
                    item,
                    Path(args.ckpt),
                    gpu_device,
                    f"dataset exceeded timeout of {args.dataset_timeout_seconds:g} seconds",
                    None,
                    wall_start=active["started"],
                    algorithm=args.algorithm,
                    model_name=args.model_name,
                )
                mark_terminal(item, "failed")
                if counts["completed"] + counts["failed"] < total:
                    start_worker(str(gpu_device))

        for worker_id, info in list(workers.items()):
            proc = info["process"]
            if worker_id in dead_workers or proc.is_alive():
                continue
            proc.join(timeout=0)
            dead_workers.add(worker_id)
            active = in_progress.pop(worker_id, None)
            if active is not None:
                item = active["item"]
                status, _data = terminal_report_for_item(item)
                if status is None:
                    write_failed_report(
                        item,
                        Path(args.ckpt),
                        info.get("gpu_id"),
                        f"worker exited with code {proc.exitcode}",
                        None,
                        wall_start=active["started"],
                        algorithm=args.algorithm,
                        model_name=args.model_name,
                    )
                    status = "failed"
                mark_terminal(item, status)
            if not info.get("init_failed") and counts["completed"] + counts["failed"] < total:
                counts["worker_failures"] += 1
                start_worker(str(info["gpu_id"]))

        live_workers = [
            info for wid, info in workers.items() if wid not in dead_workers and info["process"].is_alive()
        ]
        if not live_workers and counts["completed"] + counts["failed"] < total:
            raise RuntimeError("all Chimera workers exited before the worklist completed")

    maybe_print_progress(force=True)

    live_worker_ids = [
        wid for wid, info in workers.items() if wid not in dead_workers and info["process"].is_alive()
    ]
    for _wid in live_worker_ids:
        task_queue.put(None)
    stop_deadline = time.monotonic() + 60
    while time.monotonic() < stop_deadline:
        live = [
            info["process"]
            for wid, info in workers.items()
            if wid not in dead_workers and info["process"].is_alive()
        ]
        if not live:
            break
        for proc in live:
            proc.join(timeout=0.5)
    for worker_id, info in workers.items():
        proc = info["process"]
        if proc.is_alive():
            stop_worker(worker_id)

    return counts


def summarize_reports(
    selected: list[WorkItem],
    pending: list[WorkItem],
    args: argparse.Namespace,
    run_counts: dict[str, Any],
    started_at: str,
    elapsed_seconds: float,
    skipped_completed: int,
    skipped_missing_files: int,
    payload_audit: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    completed_reports: list[dict[str, Any]] = []
    failed_reports: list[dict[str, Any]] = []
    missing: list[WorkItem] = []
    for item in selected:
        data = read_json_dict(Path(item.report_path))
        if data.get("status") == "completed" and data.get("algorithm") == args.algorithm:
            completed_reports.append(data)
        elif data.get("status") == "failed" and data.get("algorithm") == args.algorithm:
            failed_reports.append(data)
        else:
            missing.append(item)

    r2_values = [
        float(r["metrics"]["r2"])
        for r in completed_reports
        if isinstance(r.get("metrics"), dict) and r["metrics"].get("r2") is not None
    ]
    rmse_values = [
        float(r["metrics"]["rmse"])
        for r in completed_reports
        if isinstance(r.get("metrics"), dict) and r["metrics"].get("rmse") is not None
    ]

    def mean_or_none(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    def median_or_none(values: list[float]) -> float | None:
        return float(np.median(values)) if values else None

    failures = [
        {
            "dataset_hash": str(r.get("dataset_hash")),
            "report_path": str(
                Path(args.dataset_root) / str(r.get("dataset_hash")) / "reports" / args.report_name
            )
            if r.get("dataset_hash")
            else None,
            "error": r.get("error") or r.get("failure_message"),
            "gpu_device": r.get("gpu_device"),
        }
        for r in failed_reports
    ]

    ended_at = utc_now()
    summary = {
        "algorithm": args.algorithm,
        "model_name": args.model_name,
        "checkpoint": str(args.ckpt),
        "dataset_root": str(args.dataset_root),
        "report_name": args.report_name,
        "started_at": started_at,
        "ended_at": ended_at,
        "run_date": ended_at,
        "elapsed_seconds": float(elapsed_seconds),
        "selection": {
            "selected_datasets": len(selected),
            "pending_before_run": len(pending),
            "skipped_completed_before_run": int(skipped_completed),
            "skipped_missing_files": int(skipped_missing_files),
            "force": bool(args.force),
            "failed_only": bool(args.failed_only),
        },
        "feature_augmentation": args.feature_augmentation,
        "run_counts": run_counts,
        "final_report_counts": {
            "completed": len(completed_reports),
            "failed": len(failed_reports),
            "missing": len(missing),
        },
        "metrics": {
            "r2_mean": mean_or_none(r2_values),
            "r2_median": median_or_none(r2_values),
            "rmse_mean": mean_or_none(rmse_values),
            "rmse_median": median_or_none(rmse_values),
        },
        "failures": failures,
        "missing": [{"dataset_hash": item.dataset_hash, "report_path": item.report_path} for item in missing],
        "runtime": {
            "gpus": args.gpus,
            "workers_per_gpu": args.workers_per_gpu,
            "thread_limit": args.thread_limit,
            "dataset_timeout_seconds": args.dataset_timeout_seconds,
        },
    }
    if payload_audit is not None:
        summary["payload_audit"] = {
            "payload_roots": payload_audit.get("payload_roots"),
            "required_files": payload_audit.get("required_files"),
            "complete_payloads": payload_audit.get("complete_payloads"),
            "missing_or_incomplete_payloads": payload_audit.get("missing_or_incomplete_payloads"),
        }
    if args.feature_augmentation == "hyp3":
        summary["hyp3"] = {
            "payload_roots": [str(root) for root in args.hyp3_payload_roots],
            "scratch_root": str(args.scratch_root),
        }

    summary_dir = Path(args.summary_dir)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    timestamped = summary_dir / f"{args.summary_prefix}_summary_{stamp}.json"
    latest = summary_dir / f"{args.summary_prefix}_summary_latest.json"
    atomic_write_json(timestamped, summary)
    atomic_write_json(latest, summary)
    return summary, timestamped, latest


def resolve_mode_defaults(args: argparse.Namespace) -> None:
    if args.feature_augmentation == "hyp3":
        args.algorithm = args.algorithm or HYP3_ALGORITHM
        args.model_name = args.model_name or HYP3_MODEL_NAME
        args.report_name = args.report_name or HYP3_REPORT_NAME
        args.summary_prefix = args.summary_prefix or HYP3_SUMMARY_PREFIX
    else:
        args.algorithm = args.algorithm or ALGORITHM
        args.model_name = args.model_name or MODEL_NAME
        args.report_name = args.report_name or DEFAULT_REPORT_NAME
        args.summary_prefix = args.summary_prefix or DEFAULT_SUMMARY_PREFIX
    args.log_prefix = args.algorithm.replace("_", "-")


def main(argv: list[str] | None = None, default_feature_augmentation: str = "none") -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--summary-dir", type=Path, default=None)
    parser.add_argument("--report-name", default=None)
    parser.add_argument("--algorithm", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--summary-prefix", default=None)
    parser.add_argument(
        "--feature-augmentation", choices=("none", "hyp3"), default=default_feature_augmentation
    )
    parser.add_argument("--hyp3-payload-root", type=Path, action="append", default=None)
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_HYP3_SCRATCH_ROOT)
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--workers-per-gpu", type=int, default=8)
    parser.add_argument("--thread-limit", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--failed-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dataset-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--startup-stagger-seconds", type=float, default=0.1)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args(argv)
    resolve_mode_defaults(args)

    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.ckpt = args.ckpt.expanduser().resolve()
    args.summary_dir = args.summary_dir.expanduser().resolve() if args.summary_dir else args.ckpt.parent
    args.scratch_root = args.scratch_root.expanduser().resolve()
    hyp3_roots = args.hyp3_payload_root or list(DEFAULT_HYP3_PAYLOAD_ROOTS)
    args.hyp3_payload_roots = tuple(root.expanduser().resolve() for root in hyp3_roots)
    limit = args.limit if args.limit and args.limit > 0 else None

    if not args.dataset_root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {args.dataset_root}")
    if not args.ckpt.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {args.ckpt}")
    if args.workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be >= 1")
    parse_gpus(args.gpus)

    selected, pending, skipped_completed, skipped_missing_files = build_worklist(
        args.dataset_root,
        args.report_name,
        limit,
        args.force,
        args.failed_only,
        args.algorithm,
        args.feature_augmentation,
        args.hyp3_payload_roots,
    )
    payload_audit = None
    if args.feature_augmentation == "hyp3":
        args.scratch_root.mkdir(parents=True, exist_ok=True)
        payload_audit = audit_hyp3_payloads(selected, args.hyp3_payload_roots)
        atomic_write_json(args.scratch_root / "hyp3_payload_audit_latest.json", payload_audit)
    total_workers = len(parse_gpus(args.gpus)) * args.workers_per_gpu
    print(
        f"[{args.log_prefix}] selected={len(selected)} pending={len(pending)} skipped_completed={skipped_completed} "
        f"workers={total_workers} gpus={args.gpus} ckpt={args.ckpt}",
        flush=True,
    )
    if payload_audit is not None:
        print(
            f"[{args.log_prefix}] hyp3_payloads complete={payload_audit['complete_payloads']} "
            f"missing_or_incomplete={payload_audit['missing_or_incomplete_payloads']} "
            f"audit={args.scratch_root / 'hyp3_payload_audit_latest.json'}",
            flush=True,
        )
    if args.dry_run:
        for item in pending[:20]:
            print(f"[{args.log_prefix}] pending {item.dataset_hash} -> {item.report_path}", flush=True)
        if len(pending) > 20:
            print(f"[{args.log_prefix}] ... {len(pending) - 20} more", flush=True)
        return 0

    started_at = utc_now()
    t0 = time.perf_counter()
    run_counts = run_pending(args, pending)
    elapsed = time.perf_counter() - t0
    summary, timestamped, latest = summarize_reports(
        selected,
        pending,
        args,
        run_counts,
        started_at,
        elapsed,
        skipped_completed,
        skipped_missing_files,
        payload_audit,
    )
    final_counts = summary["final_report_counts"]
    print(
        f"[{args.log_prefix}] summary={timestamped} latest={latest} "
        f"completed={final_counts['completed']} failed={final_counts['failed']} missing={final_counts['missing']} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
