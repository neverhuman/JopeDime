#!/usr/bin/env python3
"""Leakage-resistant Golden46 preparation, registered execution, and scoring.

The prepare step is the only step that reads canonical test labels.  It emits a
worker view containing train features/targets and test features with a constant
dummy target.  The worker step writes predictions but never metrics.  The
controller step alone reopens canonical labels and creates the validated report.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CAMPAIGN_ID = "golden46-full42-20260723-v1"
QUANT_ROOT = Path(os.environ.get("GOLDEN46_QUANT_ROOT", "/home/ubuntu/remote_super/quant"))
CANONICAL_ROOT = QUANT_ROOT / "regression"
CAMPAIGN_ROOT = Path(
    os.environ.get(
        "GOLDEN46_CAMPAIGN_ROOT",
        str(QUANT_ROOT / "campaigns" / CAMPAIGN_ID),
    )
)
DEFAULT_WORKER_ROOT = CAMPAIGN_ROOT / "worker-views" / "regression"
DEFAULT_ARTIFACT_ROOT = CAMPAIGN_ROOT / "artifacts"
DEFAULT_CONTROLLER_ROOT = CAMPAIGN_ROOT / "controller-private"
COHORT_PATH = CAMPAIGN_ROOT / "manifests" / "cohort.json"
HYP3_BINARY = Path(
    "/home/ubuntu/remote_super/super_cluster/hyp30_vs/"
    "hyp30_vs_20260704T062139Z/bin/hyperion_vs"
)
HYP3_WEIGHTS = Path("/home/ubuntu/hyperion/weights/v5_armB/model.safetensors")
HYP3_MANIFEST = Path("/home/ubuntu/hyperion/weights/v5_armB/manifest.json")
MODEL_ZOO_PATH = QUANT_ROOT / "models" / "tools" / "model_zoo.py"
MODEL_QUEUE_PATH = QUANT_ROOT / "models" / "tools" / "cluster_model_queue.py"
EXECUTION_SOURCE_PATH = Path(__file__).resolve()
HYP3_OUTPUTS = (
    "train3.vs",
    "test3.vs",
    "hyperion3train.vs",
    "hyperion3test.vs",
)
DUMMY_TEST_TARGET = 0.0
CORE_RUNTIME_DISTRIBUTIONS = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
    ).encode()
    return sha256_bytes(header + b"\0" + array.tobytes())


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


def canonical_json(value: Any) -> bytes:
    return (json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n").encode()


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as tmp:
        temporary = Path(tmp.name)
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write(path, canonical_json(payload))


def atomic_save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as tmp:
        temporary = Path(tmp.name)
        np.save(tmp, values, allow_pickle=False)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def canonical_dataset(dataset_id: str, canonical_root: Path) -> tuple[Path, dict[str, Any]]:
    path = canonical_root / dataset_id
    metadata = read_json(path / "meta.json")
    headers_removed = metadata.get("headers_removed_from_csv")
    if not isinstance(headers_removed, bool):
        raise ValueError(f"{dataset_id}: headers_removed_from_csv must be boolean")
    return path, metadata


def read_canonical_frame(path: Path, metadata: dict[str, Any], role: str) -> pd.DataFrame:
    header = None if metadata["headers_removed_from_csv"] else 0
    frame = pd.read_csv(path / f"{role}.csv", header=header)
    if frame.shape[1] < 2:
        raise ValueError(f"{path.name}/{role}.csv lacks a target-last layout")
    numeric = frame.apply(pd.to_numeric, errors="raise")
    if not np.all(np.isfinite(numeric.to_numpy(dtype=np.float64))):
        raise ValueError(f"{path.name}/{role}.csv contains non-finite values")
    return numeric


def write_headerless_csv(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as tmp:
        temporary = Path(tmp.name)
        pd.DataFrame(values).to_csv(tmp, header=False, index=False, float_format="%.17g")
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(temporary, path)


def prepare_dataset(
    dataset_id: str,
    *,
    canonical_root: Path,
    worker_root: Path,
    controller_root: Path,
) -> dict[str, Any]:
    source, metadata = canonical_dataset(dataset_id, canonical_root)
    train = read_canonical_frame(source, metadata, "train").to_numpy(dtype=np.float64)
    test = read_canonical_frame(source, metadata, "test").to_numpy(dtype=np.float64)
    if train.shape[1] != test.shape[1]:
        raise ValueError(f"{dataset_id}: train/test column mismatch")
    train_features, train_target = train[:, :-1], train[:, -1]
    test_features, test_target = test[:, :-1], test[:, -1]
    worker_test = np.column_stack(
        [test_features, np.full(test.shape[0], DUMMY_TEST_TARGET, dtype=np.float64)]
    )

    worker_path = worker_root / dataset_id
    write_headerless_csv(worker_path / "train.csv", train)
    write_headerless_csv(worker_path / "test.csv", worker_test)
    worker_metadata = {
        "schema": "golden46.worker-dataset-view.v1",
        "campaign_id": CAMPAIGN_ID,
        "dataset_hash": dataset_id,
        "task": "regression",
        "headers_removed_from_csv": True,
        "target_is_final_column": True,
        "test_target": "fixed-dummy-never-scored",
        "train_rows": int(train.shape[0]),
        "test_rows": int(test.shape[0]),
        "n_features": int(train.shape[1] - 1),
        "provenance_group": (
            "ZTS"
            if dataset_id in {"ZTS", "ZTS_small"}
            else metadata.get("provenance_group", dataset_id)
        ),
        "worker_input_hashes": {
            "train_features": sha256_array(train_features),
            "train_target": sha256_array(train_target),
            "test_features": sha256_array(test_features),
        },
    }
    atomic_write_json(worker_path / "meta.json", worker_metadata)
    worker_manifest = {
        **worker_metadata,
        "artifacts": {
            name: sha256_file(worker_path / name)
            for name in ("train.csv", "test.csv", "meta.json")
        },
    }
    atomic_write_json(worker_path / "worker-manifest.json", worker_manifest)

    controller_manifest = {
        "schema": "golden46.controller-dataset-labels.v1",
        "campaign_id": CAMPAIGN_ID,
        "dataset_hash": dataset_id,
        "canonical_path": str(source),
        "canonical_hashes": {
            name: sha256_file(source / name)
            for name in ("train.csv", "test.csv", "meta.json")
        },
        "test_target_sha256": sha256_array(test_target),
        "test_rows": int(test.shape[0]),
        "worker_manifest_sha256": sha256_file(worker_path / "worker-manifest.json"),
    }
    atomic_write_json(controller_root / dataset_id / "labels.json", controller_manifest)
    return {
        "dataset_id": dataset_id,
        "worker_path": str(worker_path),
        "worker_manifest_sha256": controller_manifest["worker_manifest_sha256"],
        "controller_labels_path": str(controller_root / dataset_id / "labels.json"),
        "headers_removed_in_worker_view": True,
        "true_test_labels_in_worker_view": False,
    }


def import_queue_modules() -> tuple[Any, Any]:
    if str(QUANT_ROOT) not in sys.path:
        sys.path.insert(0, str(QUANT_ROOT))
    from models.tools import cluster_model_queue, model_zoo

    return cluster_model_queue, model_zoo


def runtime_distribution_versions(spec: Any) -> dict[str, str | None]:
    """Capture the exact library environment without importing optional models."""
    distributions = set(CORE_RUNTIME_DISTRIBUTIONS)
    distributions.update(
        {
            {
                "sklearn": "scikit-learn",
                "pytorch_tabnet": "pytorch-tabnet",
                "pytorch_tabr": "pytorch-tabr",
                "torch_frame": "pytorch-frame",
            }.get(dependency, dependency)
            for dependency in spec.deps
        }
    )
    versions: dict[str, str | None] = {}
    for distribution in sorted(distributions):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def method_identity(method_id: str, spec: Any) -> dict[str, Any]:
    payload = {
        "method_id": method_id,
        "model_zoo_sha256": sha256_file(MODEL_ZOO_PATH),
        "task": spec.task,
        "resource": spec.resource,
        "deps": list(spec.deps),
        "params": json_safe(spec.params),
        "priority": spec.priority,
        "factory": (
            None
            if spec.factory is None
            else f"{spec.factory.__module__}.{getattr(spec.factory, '__qualname__', spec.factory.__name__)}"
        ),
    }
    payload["identity_sha256"] = sha256_bytes(canonical_json(payload))
    return payload


def cohort_registered_methods() -> set[str]:
    cohort = read_json(COHORT_PATH)
    return set(cohort["registered_method_ids"])


def validate_worker_view(worker_path: Path, dataset_id: str) -> dict[str, Any]:
    manifest_path = worker_path / "worker-manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("dataset_hash") != dataset_id:
        raise ValueError("worker manifest dataset identity mismatch")
    for name, expected in manifest["artifacts"].items():
        if sha256_file(worker_path / name) != expected:
            raise ValueError(f"worker artifact changed after preparation: {name}")
    train = pd.read_csv(worker_path / "train.csv", header=None).to_numpy(dtype=np.float64)
    test = pd.read_csv(worker_path / "test.csv", header=None).to_numpy(dtype=np.float64)
    if train.shape[1] != test.shape[1]:
        raise ValueError("worker train/test column mismatch")
    if not np.all(test[:, -1] == DUMMY_TEST_TARGET):
        raise ValueError("worker test target column is not the fixed dummy")
    if not np.all(np.isfinite(train)) or not np.all(np.isfinite(test)):
        raise ValueError("worker view contains non-finite values")
    return manifest


def run_registered(
    dataset_id: str,
    method_id: str,
    *,
    worker_root: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    if method_id not in cohort_registered_methods():
        raise ValueError(f"{method_id} is not in the frozen 116-method registered cohort")
    queue, model_zoo = import_queue_modules()
    specs = model_zoo.specs_by_name("regression")
    spec = specs.get(method_id)
    if spec is None:
        raise ValueError(f"registered model identity disappeared: {method_id}")
    worker_path = worker_root / dataset_id
    worker_manifest = validate_worker_view(worker_path, dataset_id)
    identity = method_identity(method_id, spec)
    started_at = utc_now()
    started = time.monotonic()
    missing = spec.missing_deps()
    if missing:
        raise RuntimeError(f"{method_id}: missing dependencies: {missing}")
    info = model_zoo.DatasetInfo(
        dataset_id,
        worker_manifest["train_rows"],
        worker_manifest["test_rows"],
        worker_manifest["n_features"],
    )
    eligible, reason = spec.check_eligibility(info)
    if not eligible:
        raise RuntimeError(f"{method_id}: unexpected ineligibility: {reason}")
    if spec.factory is None:
        raise RuntimeError(f"{method_id}: model factory is missing")
    queue.apply_child_limits(None)

    train = pd.read_csv(worker_path / "train.csv", header=None).to_numpy(dtype=np.float64)
    test = pd.read_csv(worker_path / "test.csv", header=None).to_numpy(dtype=np.float64)
    x_train = train[:, :-1].astype(np.float32, copy=False)
    y_train = train[:, -1].astype(np.float64, copy=False)
    x_test = test[:, :-1].astype(np.float32, copy=False)
    augmentation = None
    if queue.feature_augmentation_for_spec(spec) is not None:
        x_train, x_test, augmentation = queue.apply_feature_augmentation_for_spec(
            worker_path,
            spec,
            x_train,
            x_test,
        )

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
    if predictions.shape != (worker_manifest["test_rows"],):
        raise ValueError(
            f"{method_id}: prediction row mismatch {predictions.shape} "
            f"!= {(worker_manifest['test_rows'],)}"
        )
    if not np.all(np.isfinite(predictions)):
        raise ValueError(f"{method_id}: non-finite predictions")

    output = artifact_root / dataset_id / method_id
    prediction_path = output / "predictions.npy"
    atomic_save_npy(prediction_path, predictions)
    receipt = {
        "schema": "golden46.prediction-worker-receipt.v1",
        "campaign_id": CAMPAIGN_ID,
        "dataset_hash": dataset_id,
        "algorithm": method_id,
        "method_identity": identity,
        "worker_implementation": {
            "execution_source_sha256": sha256_file(EXECUTION_SOURCE_PATH),
            "model_queue_sha256": sha256_file(MODEL_QUEUE_PATH),
            "model_zoo_sha256": sha256_file(MODEL_ZOO_PATH),
        },
        "status": "predicted-unscored",
        "started_at": started_at,
        "ended_at": utc_now(),
        "training_time_seconds": training_seconds,
        "inference_time_seconds": inference_seconds,
        "wall_time_seconds": time.monotonic() - started,
        "train_rows": int(x_train.shape[0]),
        "test_rows": int(x_test.shape[0]),
        "n_features": int(x_train.shape[1]),
        "worker_manifest_sha256": sha256_file(worker_path / "worker-manifest.json"),
        "predictions": {
            "path": prediction_path.name,
            "sha256": sha256_file(prediction_path),
            "array_sha256": sha256_array(predictions),
            "rows": int(predictions.shape[0]),
        },
        "feature_augmentation": augmentation,
        "environment": {
            "host": platform.node(),
            "python": platform.python_version(),
            "pid": os.getpid(),
            "resource": spec.resource,
            "distributions": runtime_distribution_versions(spec),
        },
        "test_labels_available_to_worker": False,
        "metrics_computed_by_worker": False,
    }
    atomic_write_json(output / "worker-receipt.json", receipt)
    try:
        queue.cleanup_model_resources(model)
    except Exception:
        pass
    return receipt


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - target
    mse = float(np.mean(np.square(error)))
    target_variance = float(np.var(target))
    total = float(np.sum(np.square(target - target.mean())))
    return {
        "explained_variance": (
            float(1.0 - np.var(error) / target_variance)
            if target_variance > 0
            else 0.0
        ),
        "mae": float(np.mean(np.abs(error))),
        "max_error": float(np.max(np.abs(error))),
        "median_absolute_error": float(np.median(np.abs(error))),
        "mse": mse,
        "prediction_max": float(np.max(prediction)),
        "prediction_min": float(np.min(prediction)),
        "r2": float(1.0 - np.sum(np.square(error)) / total) if total > 0 else 0.0,
        "rmse": float(math.sqrt(mse)),
        "target_max": float(np.max(target)),
        "target_min": float(np.min(target)),
    }


def finalize_report(
    dataset_id: str,
    method_id: str,
    *,
    canonical_root: Path,
    worker_root: Path,
    artifact_root: Path,
    controller_root: Path,
) -> dict[str, Any]:
    output = artifact_root / dataset_id / method_id
    receipt_path = output / "worker-receipt.json"
    receipt = read_json(receipt_path)
    if receipt.get("dataset_hash") != dataset_id or receipt.get("algorithm") != method_id:
        raise ValueError("worker receipt changed method or dataset identity")
    queue, model_zoo = import_queue_modules()
    spec = model_zoo.specs_by_name("regression").get(method_id)
    if spec is None or receipt.get("method_identity") != method_identity(method_id, spec):
        raise ValueError("worker method/checkpoint identity differs from the frozen queue identity")
    expected_implementation = {
        "execution_source_sha256": sha256_file(EXECUTION_SOURCE_PATH),
        "model_queue_sha256": sha256_file(MODEL_QUEUE_PATH),
        "model_zoo_sha256": sha256_file(MODEL_ZOO_PATH),
    }
    if receipt.get("worker_implementation") != expected_implementation:
        raise ValueError("worker implementation differs from the frozen campaign implementation")
    worker_manifest_sha = sha256_file(worker_root / dataset_id / "worker-manifest.json")
    if receipt.get("worker_manifest_sha256") != worker_manifest_sha:
        raise ValueError("worker receipt does not match the current masked worker view")
    prediction_path = output / Path(receipt["predictions"]["path"]).name
    if sha256_file(prediction_path) != receipt["predictions"]["sha256"]:
        raise ValueError("prediction artifact hash mismatch")
    predictions = np.load(prediction_path, allow_pickle=False).astype(np.float64).reshape(-1)
    if sha256_array(predictions) != receipt["predictions"]["array_sha256"]:
        raise ValueError("prediction array content mismatch")
    if not np.all(np.isfinite(predictions)):
        raise ValueError("non-finite predictions")

    source, metadata = canonical_dataset(dataset_id, canonical_root)
    target = read_canonical_frame(source, metadata, "test").iloc[:, -1].to_numpy(dtype=np.float64)
    if predictions.shape != target.shape:
        raise ValueError(f"prediction row mismatch: {predictions.shape} != {target.shape}")
    label_manifest = read_json(controller_root / dataset_id / "labels.json")
    if label_manifest["test_target_sha256"] != sha256_array(target):
        raise ValueError("canonical labels changed after controller preparation")
    metrics = regression_metrics(target, predictions)
    report = {
        "schema": "golden46.validated-report.v1",
        "campaign_id": CAMPAIGN_ID,
        "algorithm": method_id,
        "model_name": method_id,
        "dataset_hash": dataset_id,
        "task_type": "regression",
        "status": "completed",
        "benchmark_valid": True,
        "run_date": utc_now(),
        "train_rows": receipt["train_rows"],
        "test_rows": receipt["test_rows"],
        "n_features": receipt["n_features"],
        "training_time_seconds": receipt["training_time_seconds"],
        "inference_time_seconds": receipt["inference_time_seconds"],
        "metrics": metrics,
        "provenance": {
            "worker_receipt": str(receipt_path),
            "worker_receipt_sha256": sha256_file(receipt_path),
            "predictions_sha256": receipt["predictions"]["sha256"],
            "canonical_test_sha256": sha256_file(source / "test.csv"),
            "canonical_test_target_sha256": sha256_array(target),
            "controller_computed_metrics": True,
            "worker_had_test_labels": False,
        },
        "environment": {"controller_host": platform.node()},
    }
    atomic_write_json(output / "report.json", report)
    verify_report(
        dataset_id,
        method_id,
        canonical_root=canonical_root,
        artifact_root=artifact_root,
    )
    return report


def verify_report(
    dataset_id: str,
    method_id: str,
    *,
    canonical_root: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    output = artifact_root / dataset_id / method_id
    report = read_json(output / "report.json")
    receipt = read_json(output / "worker-receipt.json")
    predictions = np.load(output / "predictions.npy", allow_pickle=False).astype(np.float64).reshape(-1)
    source, metadata = canonical_dataset(dataset_id, canonical_root)
    target = read_canonical_frame(source, metadata, "test").iloc[:, -1].to_numpy(dtype=np.float64)
    recomputed = regression_metrics(target, predictions)
    if report.get("dataset_hash") != dataset_id or report.get("algorithm") != method_id:
        raise ValueError("report identity mismatch")
    if report.get("provenance", {}).get("worker_receipt_sha256") != sha256_file(
        output / "worker-receipt.json"
    ):
        raise ValueError("report worker receipt hash mismatch")
    if report.get("metrics") != recomputed:
        raise ValueError("score/report disagreement")
    if receipt["predictions"]["sha256"] != sha256_file(output / "predictions.npy"):
        raise ValueError("report prediction artifact changed")
    return {
        "dataset_id": dataset_id,
        "method_id": method_id,
        "rmse": recomputed["rmse"],
        "verified": True,
    }


def generate_hyp3(
    dataset_id: str,
    *,
    worker_root: Path,
    binary: Path,
    weights: Path,
    manifest: Path,
) -> dict[str, Any]:
    worker_path = worker_root / dataset_id
    validate_worker_view(worker_path, dataset_id)
    for path in (binary, weights, manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    command = [
        str(binary),
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
        str(weights),
        "--manifest",
        str(manifest),
        "--sweeps",
        "3",
        "--scales",
        "4,7,11,15",
    ]
    started = time.monotonic()
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        for name in HYP3_OUTPUTS:
            (worker_path / name).unlink(missing_ok=True)
        raise RuntimeError(f"HYP3 generation failed: {completed.stdout[-4000:]}")
    train_rows = read_json(worker_path / "worker-manifest.json")["train_rows"]
    test_rows = read_json(worker_path / "worker-manifest.json")["test_rows"]
    shapes = {}
    for index, name in enumerate(HYP3_OUTPUTS):
        values = pd.read_csv(worker_path / name, header=None).to_numpy(dtype=np.float64)
        expected_rows = train_rows if index in (0, 2) else test_rows
        if values.shape[0] != expected_rows or not np.all(np.isfinite(values)):
            raise ValueError(f"{name}: invalid HYP3 output shape/content {values.shape}")
        shapes[name] = list(values.shape)
    receipt = {
        "schema": "golden46.hyp3-masked-generation.v1",
        "campaign_id": CAMPAIGN_ID,
        "dataset_hash": dataset_id,
        "fit_scope": "masked worker train only",
        "test_scope": "test features plus fixed dummy target; canonical labels unavailable",
        "implementation": {
            "binary_sha256": sha256_file(binary),
            "weights_sha256": sha256_file(weights),
            "manifest_sha256": sha256_file(manifest),
            "sweeps": 3,
            "scales": [4, 7, 11, 15],
        },
        "artifacts": {
            name: {
                "sha256": sha256_file(worker_path / name),
                "shape": shapes[name],
            }
            for name in HYP3_OUTPUTS
        },
        "seconds": time.monotonic() - started,
        "test_labels_available": False,
    }
    atomic_write_json(worker_path / "hyp3-receipt.json", receipt)
    return receipt


def terminal_ineligible(
    dataset_id: str,
    method_id: str,
    reason: str,
    *,
    artifact_root: Path,
) -> dict[str, Any]:
    if dataset_id != "ZTS":
        raise ValueError("terminal size-ineligibility receipts are restricted to full ZTS")
    cohort = read_json(COHORT_PATH)
    expected = {
        row["method_id"]: row["reason"]
        for row in cohort["full_zts_terminal_ineligibilities"]
    }
    if expected.get(method_id) != reason:
        raise ValueError("terminal ineligibility does not match the frozen cohort manifest")
    receipt = {
        "schema": "golden46.terminal-receipt.v1",
        "campaign_id": CAMPAIGN_ID,
        "dataset_hash": dataset_id,
        "algorithm": method_id,
        "status": "skipped-ineligible",
        "reason": reason,
        "internal_downsampling": False,
        "method_identity_changed": False,
        "created_at": utc_now(),
    }
    atomic_write_json(artifact_root / dataset_id / method_id / "terminal-receipt.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", type=Path, default=CANONICAL_ROOT)
    parser.add_argument("--worker-root", type=Path, default=DEFAULT_WORKER_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--controller-root", type=Path, default=DEFAULT_CONTROLLER_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--dataset", required=True)

    registered = subparsers.add_parser("run-registered")
    registered.add_argument("--dataset", required=True)
    registered.add_argument("--method", required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--dataset", required=True)
    finalize.add_argument("--method", required=True)

    verify = subparsers.add_parser("verify-report")
    verify.add_argument("--dataset", required=True)
    verify.add_argument("--method", required=True)

    hyp3 = subparsers.add_parser("generate-hyp3")
    hyp3.add_argument("--dataset", required=True)
    hyp3.add_argument("--binary", type=Path, default=HYP3_BINARY)
    hyp3.add_argument("--weights", type=Path, default=HYP3_WEIGHTS)
    hyp3.add_argument("--manifest", type=Path, default=HYP3_MANIFEST)

    terminal = subparsers.add_parser("terminal-ineligible")
    terminal.add_argument("--dataset", required=True)
    terminal.add_argument("--method", required=True)
    terminal.add_argument("--reason", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        result = prepare_dataset(
            args.dataset,
            canonical_root=args.canonical_root,
            worker_root=args.worker_root,
            controller_root=args.controller_root,
        )
    elif args.command == "run-registered":
        result = run_registered(
            args.dataset,
            args.method,
            worker_root=args.worker_root,
            artifact_root=args.artifact_root,
        )
    elif args.command == "finalize":
        result = finalize_report(
            args.dataset,
            args.method,
            canonical_root=args.canonical_root,
            worker_root=args.worker_root,
            artifact_root=args.artifact_root,
            controller_root=args.controller_root,
        )
    elif args.command == "verify-report":
        result = verify_report(
            args.dataset,
            args.method,
            canonical_root=args.canonical_root,
            artifact_root=args.artifact_root,
        )
    elif args.command == "generate-hyp3":
        result = generate_hyp3(
            args.dataset,
            worker_root=args.worker_root,
            binary=args.binary,
            weights=args.weights,
            manifest=args.manifest,
        )
    elif args.command == "terminal-ineligible":
        result = terminal_ineligible(
            args.dataset,
            args.method,
            args.reason,
            artifact_root=args.artifact_root,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(json_safe(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
