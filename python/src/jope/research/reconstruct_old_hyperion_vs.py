#!/usr/bin/env python3
"""Reconstruct the January 2026 Python Hyperion cell-matrix pipeline.

This intentionally does not call JOPE's Rust Hyperion encoder.  It reuses the
forensic reproduction implementation beside the historical model on xbabe1,
pins every surviving authority by SHA-256, and emits non-overwriting
``train3.vs.orig`` / ``test3.vs.orig`` matrices.

The surviving evidence cannot recreate the January bytes exactly: the likely
legacy checkpoint is a partial load and the original writer did not record its
global Torch/CUDA RNG state.  Consequently the receipt calls these artifacts a
deterministic reconstruction of the old semantics, never an exact replay.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCHEMA = "HyperionOldPythonVsReconstruction.v1"
WINDOWS = [3, 6, 9, 12]
SHIFTS = [[0, 0], [1, 1]]
DEFAULT_DATASET_IDS = (
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

PINNED_HYPERION_FILES = {
    "process_regression_embeddings_100.py": (
        "b5f03a771c6b71650c6030294bd45bf4e833fd4102f8b54d4cc9427d91215b24"
    ),
    "Hyperion_Entanglement_v6.py": (
        "44d8653c9984ba297b0a72af99e96bd6c60d1199d49994bbcecb09b1c7d9cc96"
    ),
    "analyze_100_sweeps_autotune.py": (
        "fcd4b169b5ec4339c275cd5aba9ca7dfa35803adc8cda4c5466388620adc6989"
    ),
    "reproduce_facebook_sweep_lift.py": (
        "9974f998989a097d5f935e318c4cfbf6486714e9d61c4128ef35b846b90065e5"
    ),
    "catboost_sweep_lift_benchmark.py": (
        "ca540106693292bdb5462a8181b309e207feb6f483bdfffbb1eb36b2add975e4"
    ),
}
PINNED_CHECKPOINT_RELATIVE_PATH = (
    "runs/hyperion_v6_20260127_083701/ckpt_10.pt"
)
PINNED_CHECKPOINT_SHA256 = (
    "cce25e793f9d970bc9d6fbc4a1af9e2412a640712114f715afa95546369425c5"
)
EXPECTED_MISSING_KEYS = 30
EXPECTED_UNEXPECTED_KEYS = 33


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def require_hash(path: Path, expected: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Pinned authority is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"Pinned authority hash mismatch for {path}: {actual} != {expected}"
        )
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": actual,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def verify_authorities(hyperion_root: Path) -> dict[str, Any]:
    files = {
        name: require_hash(hyperion_root / name, expected)
        for name, expected in PINNED_HYPERION_FILES.items()
    }
    checkpoint = require_hash(
        hyperion_root / PINNED_CHECKPOINT_RELATIVE_PATH,
        PINNED_CHECKPOINT_SHA256,
    )
    return {"files": files, "checkpoint": checkpoint}


def import_reproduction_driver(hyperion_root: Path) -> ModuleType:
    path = hyperion_root / "reproduce_facebook_sweep_lift.py"
    spec = importlib.util.spec_from_file_location("hyperion_old_reproduction", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import reproduction driver: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_feature_matrix(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a headerless JOPE split and exclude its final target column."""

    if not path.is_file():
        raise FileNotFoundError(f"Input split is missing: {path}")
    frame = pd.read_csv(path, header=None)
    if frame.shape[0] == 0 or frame.shape[1] < 2:
        raise ValueError(f"Expected rows and at least one feature plus target: {path}")

    features_frame = frame.iloc[:, :-1].fillna(0)
    features = features_frame.to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(features).all():
        raise ValueError(f"Feature matrix contains non-finite values: {path}")

    return features, {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": int(frame.shape[0]),
        "columns_including_target": int(frame.shape[1]),
        "feature_columns": int(features.shape[1]),
        "target_column_index": int(frame.shape[1] - 1),
        "target_policy": "excluded before any encoder call",
    }


def load_joint_features(
    dataset_root: Path,
) -> tuple[np.ndarray, int, int, dict[str, Any]]:
    train, train_receipt = read_feature_matrix(dataset_root / "train.csv")
    test, test_receipt = read_feature_matrix(dataset_root / "test.csv")
    if train.shape[1] != test.shape[1]:
        raise ValueError(
            f"Train/test feature-width mismatch in {dataset_root}: "
            f"{train.shape[1]} != {test.shape[1]}"
        )
    joint = np.concatenate((train, test), axis=0).astype(np.float32, copy=False)
    return joint, len(train), len(test), {
        "train": train_receipt,
        "test": test_receipt,
        "joint_shape": list(joint.shape),
        "joint_order": "all train feature rows, then all test feature rows",
        "joint_feature_sha256": sha256_array(joint),
    }


def collapse_all100_mean(stack: np.ndarray, sweeps: int = 100) -> np.ndarray:
    """Match catboost_sweep_lift_benchmark.py's all-sweep mean exactly."""

    if stack.ndim != 4:
        raise ValueError(f"Expected (sweeps, scales, rows, columns), got {stack.shape}")
    if stack.shape[0] != sweeps or stack.shape[1] != len(WINDOWS):
        raise ValueError(
            f"Expected ({sweeps}, {len(WINDOWS)}, N, D), got {stack.shape}"
        )
    sequence = np.mean(stack, axis=1).transpose(1, 0, 2)
    collapsed = np.mean(sequence[:, :sweeps, :], axis=1)
    return collapsed.astype(np.float32, copy=False)


def _write_temp_matrix(path: Path, matrix: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            np.savetxt(handle, matrix, delimiter=",", fmt="%.9g")
            handle.flush()
            os.fsync(handle.fileno())
        replay = np.loadtxt(tmp, delimiter=",", dtype=np.float32, ndmin=2)
        if replay.shape != matrix.shape or not np.array_equal(replay, matrix):
            raise RuntimeError(f"Text serialization did not round-trip float32: {path}")
        return tmp
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_matrix_exclusive(path: Path, matrix: np.ndarray) -> dict[str, Any]:
    """Atomically publish a matrix while refusing to replace any existing path."""

    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    tmp = _write_temp_matrix(path, matrix)
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "finite": bool(np.isfinite(matrix).all()),
        "min": float(matrix.min()),
        "max": float(matrix.max()),
        "mean": float(matrix.mean()),
        "std": float(matrix.std()),
    }


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def parse_dataset_ids(values: Iterable[str]) -> tuple[str, ...]:
    ids = tuple(values) or DEFAULT_DATASET_IDS
    if len(ids) != len(set(ids)):
        raise ValueError("Dataset IDs must be unique")
    for dataset_id in ids:
        if len(dataset_id) != 16 or any(c not in "0123456789abcdef" for c in dataset_id):
            raise ValueError(f"Invalid dataset ID: {dataset_id}")
    return ids


def generation_environment(torch: Any) -> dict[str, Any]:
    device = torch.device("cuda:0")
    return {
        "hostname": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_cuda_device_count": torch.cuda.device_count(),
        "cuda_device": torch.cuda.get_device_name(device),
        "cuda_capability": list(torch.cuda.get_device_capability(device)),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_ids = parse_dataset_ids(args.dataset_id)
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    hyperion_root = args.hyperion_root.resolve()

    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    authorities = verify_authorities(hyperion_root)
    reproduction = import_reproduction_driver(hyperion_root)
    torch = reproduction.torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the old Hyperion reconstruction")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Exactly one CUDA device must be visible; set CUDA_VISIBLE_DEVICES explicitly"
        )
    torch.cuda.set_device(0)
    reproduction.set_reproducible_seeds(args.seed)

    device = torch.device("cuda:0")
    checkpoint = hyperion_root / PINNED_CHECKPOINT_RELATIVE_PATH
    model, cfg, load_diagnostics = reproduction.load_model(
        ckpt_path=checkpoint,
        device=device,
        load_strict=False,
    )
    if load_diagnostics["missing_key_count"] != EXPECTED_MISSING_KEYS:
        raise RuntimeError(
            "Checkpoint partial-load signature changed: missing-key count is "
            f"{load_diagnostics['missing_key_count']}"
        )
    if load_diagnostics["unexpected_key_count"] != EXPECTED_UNEXPECTED_KEYS:
        raise RuntimeError(
            "Checkpoint partial-load signature changed: unexpected-key count is "
            f"{load_diagnostics['unexpected_key_count']}"
        )

    started = time.time()
    receipts: list[dict[str, Any]] = []
    for index, dataset_id in enumerate(dataset_ids):
        dataset_started = time.time()
        joint, train_rows, test_rows, inputs = load_joint_features(
            input_root / dataset_id
        )
        stack = reproduction.generate_sweeps(
            model=model,
            cfg=cfg,
            X_raw_np=joint,
            sweeps=args.sweeps,
            device=device,
            batch_size=args.batch_size,
            use_amp=True,
            rank_scale="100",
            padding_rng="original",
        )
        expected_stack_shape = (
            args.sweeps,
            len(WINDOWS),
            train_rows + test_rows,
            joint.shape[1],
        )
        if stack.shape != expected_stack_shape:
            raise RuntimeError(
                f"Unexpected sweep shape for {dataset_id}: "
                f"{stack.shape} != {expected_stack_shape}"
            )
        if stack.dtype != np.float32 or not np.isfinite(stack).all():
            raise RuntimeError(f"Invalid sweep tensor for {dataset_id}")

        cell = collapse_all100_mean(stack, sweeps=args.sweeps)
        train_cell = cell[:train_rows]
        test_cell = cell[train_rows:]
        dataset_output = output_root / dataset_id
        train_output = write_matrix_exclusive(
            dataset_output / "train3.vs.orig", train_cell
        )
        test_output = write_matrix_exclusive(
            dataset_output / "test3.vs.orig", test_cell
        )
        receipt = {
            "dataset_id": dataset_id,
            "campaign_index": index,
            "inputs": inputs,
            "sweep_tensor": {
                "shape": list(stack.shape),
                "dtype": str(stack.dtype),
                "finite": True,
                "c_order_sha256": sha256_array(stack),
                "min": float(stack.min()),
                "max": float(stack.max()),
                "mean": float(stack.mean()),
                "std": float(stack.std()),
            },
            "collapsed_cell_c_order_sha256": sha256_array(cell),
            "outputs": {"train": train_output, "test": test_output},
            "elapsed_seconds": round(time.time() - dataset_started, 6),
        }
        receipts.append(receipt)
        write_json_exclusive(dataset_output / "receipt.json", receipt)
        print(
            f"[{index + 1}/{len(dataset_ids)}] {dataset_id}: "
            f"joint={tuple(joint.shape)} train={tuple(train_cell.shape)} "
            f"test={tuple(test_cell.shape)}",
            flush=True,
        )

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete",
        "claim": "deterministic reconstruction of old Python semantics; not byte-exact January replay",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 6),
        "dataset_count": len(dataset_ids),
        "output_file_count": len(dataset_ids) * 2,
        "dataset_order": list(dataset_ids),
        "parameters": {
            "joint_train_test_features": True,
            "joint_order": "train rows then test rows",
            "target_used_by_encoder": False,
            "nan_feature_policy": "fill with zero before float32 conversion",
            "rank_normalization": "(argsort(argsort(column)) + 1) / N * 100",
            "rank_scale": 100,
            "sweeps": args.sweeps,
            "base_sweep_seed": 42,
            "sweep_seed_stride": 777,
            "global_reconstruction_seed": args.seed,
            "windows": WINDOWS,
            "shifts": SHIFTS,
            "batch_size": args.batch_size,
            "cuda_amp": True,
            "checkpoint_load_strict": False,
            "padding_rng": "original global CUDA RNG",
            "collapse": "mean four scales per sweep, then mean all 100 sweeps",
            "output_format": "headerless comma-separated float32, 9 significant digits",
            "overwrite_policy": "exclusive creation; refuse existing output",
        },
        "authorities": authorities,
        "load_diagnostics": load_diagnostics,
        "environment": generation_environment(torch),
        "datasets": receipts,
        "known_irrecoverable_gap": {
            "exact_replay_possible": False,
            "reasons": [
                "the likely legacy checkpoint leaves live model keys randomly initialized",
                "the January writer did not seed or record global Torch/CUDA initialization RNG",
                "padding used unrecorded global CUDA RNG state",
                "the writer was edited after cache creation and no pre-edit snapshot survives",
            ],
        },
    }
    write_json_exclusive(output_root / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--hyperion-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-id",
        action="append",
        default=[],
        help="Repeat to override the frozen 20-dataset order.",
    )
    parser.add_argument("--sweeps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.sweeps != 100:
        parser.error("The .orig contract requires exactly 100 sweeps")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main() -> None:
    manifest = run(parse_args())
    print(
        f"Complete: {manifest['dataset_count']} datasets, "
        f"{manifest['output_file_count']} .vs.orig files",
        flush=True,
    )


if __name__ == "__main__":
    main()
