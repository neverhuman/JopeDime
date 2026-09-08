#!/usr/bin/env python3
"""Run the old Hyperion cell-sweep semantics with Jain's latest pinned model.

The model is Jain's governed ``hyperion_v5_arm_b`` safetensors artifact.  The
inference geometry comes from the byte-pinned historical Python V5 source on
xbabe1 because that source is the direct PyTorch counterpart of the current
Rust/Candle implementation.  Source text is fetched into memory, hash checked,
and executed without copying it into the workspace.

This is deliberately different from ``hyperion_vs``.  That CLI emits a
per-column rank of the sigmoid confidence surface after three train-only
sweeps.  Here we preserve the old ``*.vs.orig`` experiment semantics: joint
train/test feature encoding, 100 randomized sweeps, two shifted tilings,
3/6/9/12 windows, raw realness logits, and the all-100 mean collapse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file


SCHEMA = "HyperionLatestModelOldSemanticsReconstruction.v1"
WINDOWS = (3, 6, 9, 12)
SHIFTS = ((0, 0), (1, 1))
BASE_SWEEP_SEED = 42
SWEEP_SEED_STRIDE = 777
GLOBAL_SEED = 42

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

PINNED_WEIGHTS_SHA256 = (
    "88b9e8345e44d3ea609a8286215618a762f14e607fcdada6d27447b1f74668be"
)
PINNED_MANIFEST_SHA256 = (
    "ebb745f6a0aa6398b1c3d3c9e26585e0536326b65e2083401391203fa37d21ef"
)
PINNED_REMOTE_SOURCES = {
    "config.py": "c960a40dd517f4af3cd40cb6214de8dd88f66e53fcf36fcc2c1d83b34a20cd47",
    "model.py": "31e095d483013689307da7a191ed54b0e3c1ffafdf66d9779c4fd467a3fd180a",
}
PINNED_JAIN_CORE_HEAD = "a029dbe5aa7a4da7b6f5aa1c5e93f73e3e207bc6"
PINNED_JAIN_CORE_SOURCES = {
    "crates/feat-core/src/hyperion.rs": "483d8f6a75b7b47391a8ce9da13c22d37a708624bc87161434c36073af53d3cd",
    "crates/feat-core/src/hyperion_v7/backend.rs": "63a2ec015a2f9e24a538f99c61dafd1c964436d090053f12b5a0bd97f29c82b4",
    "crates/feat-core/src/hyperion_v7/model.rs": "6be397226338c9afca425e1214df912d186d2b7bbc7a84e996e043dbf53086ef",
    "crates/feat-core/src/hyperion_v7/sweep/driver.rs": "b853226cb9ffbad735c614403077645ae4aa2684309c21a8a6218107cd58c808",
}
EXPECTED_MISSING_KEYS = {
    "h_col.bias",
    "h_col.weight",
    "h_row.bias",
    "h_row.weight",
}
EXPECTED_UNEXPECTED_KEYS: set[str] = set()
EXPECTED_MANIFEST = {
    "version": "v7",
    "step": 0,
    "dim": 128,
    "heads": 4,
    "layers": 4,
    "emb_dim": 128,
    "min_patch": 2,
    "max_patch": 16,
    "n_bands": 4,
    "q_low": 1.0,
    "q_high": 99.0,
    "quant_scale": 3.0,
    "pad_u8": 128,
    "ws_list": [4, 7, 11, 15],
    "seed": 0,
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(value).tobytes(order="C"))


def require_file_hash(path: Path, expected: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"authority hash mismatch for {path}: {actual} != {expected}")
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": actual,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def fetch_remote_source(host: str, root: str, name: str) -> bytes:
    path = f"{root.rstrip('/')}/{name}"
    completed = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            host,
            "cat",
            "--",
            path,
        ],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"failed to read {host}:{path}: {completed.stderr.decode(errors='replace')}"
        )
    expected = PINNED_REMOTE_SOURCES[name]
    actual = sha256_bytes(completed.stdout)
    if actual != expected:
        raise RuntimeError(f"remote source hash mismatch for {path}: {actual} != {expected}")
    return completed.stdout


def load_pinned_python_model_source(host: str, root: str) -> tuple[type, dict[str, Any]]:
    """Load the exact V5 architecture source into synthetic in-memory modules."""

    config_bytes = fetch_remote_source(host, root, "config.py")
    model_bytes = fetch_remote_source(host, root, "model.py")
    package_name = "_hyperion_v5_pinned_authority"

    package = ModuleType(package_name)
    package.__path__ = []  # type: ignore[attr-defined]
    package.__package__ = package_name
    sys.modules[package_name] = package

    config_name = f"{package_name}.config"
    config_module = ModuleType(config_name)
    config_module.__package__ = package_name
    config_module.__file__ = f"ssh://{host}/{root.strip('/')}/config.py"
    sys.modules[config_name] = config_module
    exec(compile(config_bytes, config_module.__file__, "exec"), config_module.__dict__)

    model_name = f"{package_name}.model"
    model_module = ModuleType(model_name)
    model_module.__package__ = package_name
    model_module.__file__ = f"ssh://{host}/{root.strip('/')}/model.py"
    sys.modules[model_name] = model_module
    exec(compile(model_bytes, model_module.__file__, "exec"), model_module.__dict__)

    return model_module.MatrixGhostUnified, {
        name: {
            "authority": f"{host}:{root.rstrip('/')}/{name}",
            "sha256": PINNED_REMOTE_SOURCES[name],
            "bytes": len(value),
        }
        for name, value in (("config.py", config_bytes), ("model.py", model_bytes))
    }


def set_reproducible_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")


def load_manifest(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw != EXPECTED_MANIFEST:
        raise RuntimeError(f"latest Hyperion manifest contract changed: {raw}")
    return raw


def build_latest_model(
    model_type: type,
    manifest: dict[str, Any],
    weights_path: Path,
    device: torch.device,
) -> tuple[Any, dict[str, Any]]:
    # The pinned Python CFG has the same geometry as the governed manifest.
    config_module = sys.modules[model_type.__module__.rsplit(".", 1)[0] + ".config"]
    cfg = config_module.CFG(
        dim=manifest["dim"],
        heads=manifest["heads"],
        layers=manifest["layers"],
        emb_dim=manifest["emb_dim"],
        min_patch=manifest["min_patch"],
        max_patch=manifest["max_patch"],
        q_low=manifest["q_low"],
        q_high=manifest["q_high"],
        quant_scale=manifest["quant_scale"],
        pad_u8=manifest["pad_u8"],
        use_amp=True,
    )
    model = model_type(cfg)
    state = load_file(str(weights_path), device="cpu")
    incompatible = model.load_state_dict(state, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing != EXPECTED_MISSING_KEYS or unexpected != EXPECTED_UNEXPECTED_KEYS:
        raise RuntimeError(
            "latest weight-load signature changed: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    model = model.to(device).eval()
    return model, {
        "checkpoint_tensor_count": len(state),
        "model_tensor_count": len(model.state_dict()),
        "missing_keys": sorted(missing),
        "unexpected_keys": sorted(unexpected),
        "missing_keys_affect_real_head": False,
        "forward_used": "_rowcol + cell_mlp + h_real only",
    }


def read_feature_matrix(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, header=None)
    if frame.shape[0] == 0 or frame.shape[1] < 2:
        raise ValueError(f"invalid JOPE split: {path}")
    features = (
        frame.iloc[:, :-1]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .to_numpy(dtype=np.float32, copy=True)
    )
    if not np.isfinite(features).all():
        raise ValueError(f"non-finite features after fill: {path}")
    return features, {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": int(features.shape[0]),
        "columns_including_target": int(frame.shape[1]),
        "feature_columns": int(features.shape[1]),
        "target_column_index": int(frame.shape[1] - 1),
        "target_policy": "excluded before joint matrix construction",
    }


def load_joint_features(dataset_root: Path) -> tuple[np.ndarray, int, int, dict[str, Any]]:
    train, train_receipt = read_feature_matrix(dataset_root / "train.csv")
    test, test_receipt = read_feature_matrix(dataset_root / "test.csv")
    if train.shape[1] != test.shape[1]:
        raise ValueError(f"feature-width mismatch: {dataset_root}")
    joint = np.concatenate((train, test), axis=0).astype(np.float32, copy=False)
    return joint, len(train), len(test), {
        "train": train_receipt,
        "test": test_receipt,
        "joint_shape": list(joint.shape),
        "joint_order": "train rows then test rows",
        "joint_feature_sha256": sha256_array(joint),
    }


def rank_normalize_gpu(x_raw: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.as_tensor(x_raw, device=device, dtype=torch.float32)
    order = torch.argsort(x, dim=0)
    ranks = torch.argsort(order, dim=0).float().add_(1.0)
    return ranks.div_(x.shape[0]).mul_(100.0)


def robust_quantize_ranked(
    ranked: np.ndarray,
    q_low: float,
    q_high: float,
    quant_scale: float,
) -> np.ndarray:
    """Apply the latest model's training-time uint8 transform to old rank values."""

    if ranked.ndim != 2 or ranked.shape[0] == 0:
        raise ValueError(f"invalid ranked matrix shape: {ranked.shape}")
    lo, hi = np.percentile(ranked, [q_low, q_high], axis=0)
    lo = lo.astype(np.float32)
    hi = hi.astype(np.float32)
    mid = (lo + hi) * np.float32(0.5)
    half = (hi - lo) * np.float32(0.5)
    half = np.where(half < 1e-6, 1.0, half).astype(np.float32)
    scaled = np.clip((ranked - mid) / half, -1.0, 1.0) * np.float32(quant_scale)
    encoded = (scaled / np.float32(2.0 * quant_scale) + np.float32(0.5)) * np.float32(255.0)
    return np.clip(encoded, 0.0, 255.0).astype(np.uint8)


def normalize_u8_for_model(value: np.ndarray, quant_scale: float) -> np.ndarray:
    return (
        (value.astype(np.float32) / np.float32(255.0)) * np.float32(2.0)
        - np.float32(1.0)
    ) * np.float32(quant_scale)


def forward_real_batch(
    model: Any,
    patches: torch.Tensor,
    window: int,
    batch_size: int,
    min_patch: int,
    use_amp: bool,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for start in range(0, len(patches), batch_size):
        x = patches[start : start + batch_size]
        mask = torch.ones_like(x)
        p_id = torch.full(
            (len(x),), window - min_patch, dtype=torch.long, device=x.device
        )
        with torch.amp.autocast("cuda", enabled=use_amp):
            cell, row_h, col_h = model._rowcol(x, mask, p_id)
            size = x.shape[1]
            row_b = row_h.unsqueeze(2).expand(len(x), size, size, model.dim)
            col_b = col_h.unsqueeze(1).expand(len(x), size, size, model.dim)
            features = torch.cat(
                [cell, row_b, col_b, torch.abs(row_b - col_b), row_b * col_b],
                dim=-1,
            )
            logits = model.h_real(model.cell_mlp(features)).squeeze(-1)
        outputs.append(logits.float())
    return torch.cat(outputs, dim=0)


@torch.inference_mode()
def generate_sweep_stack(
    model: Any,
    x_raw: np.ndarray,
    manifest: dict[str, Any],
    sweeps: int,
    batch_size: int,
    device: torch.device,
    progress_label: str,
) -> np.ndarray:
    ranked = rank_normalize_gpu(x_raw, device)
    quantized = robust_quantize_ranked(
        ranked.cpu().numpy(),
        manifest["q_low"],
        manifest["q_high"],
        manifest["quant_scale"],
    )
    normalized = normalize_u8_for_model(quantized, manifest["quant_scale"])
    x_norm = torch.as_tensor(normalized, device=device, dtype=torch.float32)
    n_rows, n_cols = x_norm.shape
    outputs: list[np.ndarray] = []

    for sweep_index in range(sweeps):
        seed = BASE_SWEEP_SEED + sweep_index * SWEEP_SEED_STRIDE
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        perm_r = torch.randperm(n_rows, generator=generator, device=device)
        perm_c = torch.randperm(n_cols, generator=generator, device=device)
        inv_r = torch.argsort(perm_r)
        inv_c = torch.argsort(perm_c)
        shuffled = x_norm[perm_r][:, perm_c]

        accum = {
            window: torch.zeros((n_rows, n_cols), device=device, dtype=torch.float32)
            for window in WINDOWS
        }
        counts = {window: 0 for window in WINDOWS}

        for off_x, off_y in SHIFTS:
            for window in WINDOWS:
                pad_top = off_y
                pad_left = off_x
                pad_bottom = (window - (n_rows + pad_top) % window) % window
                pad_right = (window - (n_cols + pad_left) % window) % window
                height = n_rows + pad_top + pad_bottom
                width = n_cols + pad_left + pad_right
                canvas = torch.zeros((height, width), device=device, dtype=torch.float32)
                canvas[
                    pad_top : pad_top + n_rows, pad_left : pad_left + n_cols
                ] = shuffled

                # This intentionally uses the global CUDA RNG, as the old writer did.
                if pad_top:
                    idx = torch.randint(0, n_rows, (pad_top,), device=device)
                    canvas[:pad_top, pad_left : pad_left + n_cols] = shuffled[idx]
                if pad_bottom:
                    idx = torch.randint(0, n_rows, (pad_bottom,), device=device)
                    canvas[pad_top + n_rows :, pad_left : pad_left + n_cols] = shuffled[idx]
                if pad_left:
                    idx = torch.randint(0, n_cols, (pad_left,), device=device)
                    canvas[:, :pad_left] = canvas[:, pad_left:][:, idx]
                if pad_right:
                    idx = torch.randint(0, n_cols, (pad_right,), device=device)
                    canvas[:, pad_left + n_cols :] = canvas[
                        :, pad_left : pad_left + n_cols
                    ][:, idx]

                row_blocks = height // window
                col_blocks = width // window
                patches = (
                    canvas.view(row_blocks, window, col_blocks, window)
                    .permute(0, 2, 1, 3)
                    .reshape(-1, window, window)
                )
                logits = forward_real_batch(
                    model,
                    patches,
                    window,
                    batch_size,
                    manifest["min_patch"],
                    use_amp=True,
                )
                image = (
                    logits.view(row_blocks, col_blocks, window, window)
                    .permute(0, 2, 1, 3)
                    .reshape(height, width)
                )
                accum[window] += image[
                    pad_top : pad_top + n_rows, pad_left : pad_left + n_cols
                ]
                counts[window] += 1

        layers = []
        for window in WINDOWS:
            layer = accum[window] / float(counts[window])
            layers.append(layer[inv_r][:, inv_c])
        outputs.append(
            torch.stack(layers, dim=0).cpu().numpy().astype(np.float32, copy=False)
        )
        if sweep_index == 0 or (sweep_index + 1) % 10 == 0 or sweep_index + 1 == sweeps:
            print(
                f"[{progress_label}] sweep {sweep_index + 1}/{sweeps}",
                flush=True,
            )

    return np.stack(outputs).astype(np.float32, copy=False)


def collapse_all_sweep_mean(stack: np.ndarray, sweeps: int) -> np.ndarray:
    if stack.ndim != 4 or stack.shape[:2] != (sweeps, len(WINDOWS)):
        raise ValueError(f"unexpected sweep stack shape: {stack.shape}")
    sequence = np.mean(stack, axis=1).transpose(1, 0, 2)
    return np.mean(sequence[:, :sweeps, :], axis=1).astype(np.float32, copy=False)


def _temporary_matrix(path: Path, matrix: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            np.savetxt(handle, matrix, delimiter=",", fmt="%.9g")
            handle.flush()
            os.fsync(handle.fileno())
        replay = np.loadtxt(temporary, delimiter=",", dtype=np.float32, ndmin=2)
        if replay.shape != matrix.shape or not np.array_equal(replay, matrix):
            raise RuntimeError(f"float32 text round-trip failed: {path}")
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_matrix_exclusive(path: Path, matrix: np.ndarray) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to replace existing output: {path}")
    temporary = _temporary_matrix(path, matrix)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
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
        raise FileExistsError(f"refusing to replace existing receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_dataset_ids(values: Iterable[str]) -> tuple[str, ...]:
    ids = tuple(values) or DEFAULT_DATASET_IDS
    if len(ids) != len(set(ids)):
        raise ValueError("dataset IDs must be unique")
    for dataset_id in ids:
        if len(dataset_id) != 16 or any(c not in "0123456789abcdef" for c in dataset_id):
            raise ValueError(f"invalid dataset ID: {dataset_id}")
    return ids


def source_receipts(jain_core: Path, declared_head: str) -> dict[str, Any]:
    if declared_head != PINNED_JAIN_CORE_HEAD:
        raise RuntimeError(
            f"Jain source head changed: {declared_head} != {PINNED_JAIN_CORE_HEAD}"
        )
    return {
        "repository": str(jain_core.resolve()),
        "git_head": declared_head,
        "files": {
            relative: require_file_hash(jain_core / relative, expected)
            for relative, expected in PINNED_JAIN_CORE_SOURCES.items()
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA device is required")
    device = torch.device("cuda:0")
    dataset_ids = parse_dataset_ids(args.dataset_id)
    weights_receipt = require_file_hash(args.weights, PINNED_WEIGHTS_SHA256)
    manifest_receipt = require_file_hash(args.manifest, PINNED_MANIFEST_SHA256)
    manifest = load_manifest(args.manifest)
    model_type, python_sources = load_pinned_python_model_source(
        args.authority_host, args.authority_root
    )
    set_reproducible_seeds(GLOBAL_SEED)
    model, load_diagnostics = build_latest_model(model_type, manifest, args.weights, device)

    started = time.time()
    dataset_receipts: list[dict[str, Any]] = []
    for index, dataset_id in enumerate(dataset_ids, start=1):
        dataset_started = time.time()
        root = args.data_root / dataset_id
        joint, train_rows, test_rows, inputs = load_joint_features(root)
        stack = generate_sweep_stack(
            model,
            joint,
            manifest,
            args.sweeps,
            args.batch_size,
            device,
            f"{index}/{len(dataset_ids)} {dataset_id}",
        )
        expected = (args.sweeps, len(WINDOWS), len(joint), joint.shape[1])
        if stack.shape != expected or stack.dtype != np.float32 or not np.isfinite(stack).all():
            raise RuntimeError(f"invalid sweep stack for {dataset_id}: {stack.shape}")
        cell = collapse_all_sweep_mean(stack, args.sweeps)
        train_cell = cell[:train_rows]
        test_cell = cell[train_rows:]
        outputs = {
            "train": write_matrix_exclusive(root / "train3.vs.orig.v2", train_cell),
            "test": write_matrix_exclusive(root / "test3.vs.orig.v2", test_cell),
        }
        receipt = {
            "dataset_id": dataset_id,
            "campaign_index": index - 1,
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
            "outputs": outputs,
            "elapsed_seconds": round(time.time() - dataset_started, 6),
        }
        dataset_receipts.append(receipt)
        print(
            f"[{index}/{len(dataset_ids)}] {dataset_id} complete "
            f"train={train_cell.shape} test={test_cell.shape}",
            flush=True,
        )

    receipt = {
        "schema": SCHEMA,
        "status": "complete",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "claim": "old sweep semantics with Jain latest pinned Hyperion model; exploratory transductive feature encoding",
        "elapsed_seconds": round(time.time() - started, 6),
        "dataset_count": len(dataset_ids),
        "output_file_count": len(dataset_ids) * 2,
        "dataset_order": list(dataset_ids),
        "parameters": {
            "joint_train_test_features": True,
            "target_used_by_encoder": False,
            "rank_normalization": "torch argsort(argsort(column)) + 1, divided by joint N and multiplied by 100",
            "latest_model_preprocessing": "rank values robust-quantized at joint 1/99 percentiles to uint8, then mapped to [-3,3]",
            "sweeps": args.sweeps,
            "base_sweep_seed": BASE_SWEEP_SEED,
            "sweep_seed_stride": SWEEP_SEED_STRIDE,
            "global_cuda_padding_seed": GLOBAL_SEED,
            "global_padding_rng_scope": "one continuous CUDA RNG stream over the frozen dataset order",
            "windows": list(WINDOWS),
            "shifts": [list(value) for value in SHIFTS],
            "batch_size": args.batch_size,
            "cuda_amp": True,
            "cell_output": "raw h_real logits; no sigmoid and no percentile postprocessing",
            "collapse": "mean two shifts per scale, mean four scales per sweep, then mean all sweeps",
            "output_format": "headerless comma-separated float32, 9 significant digits",
            "overwrite_policy": "exclusive creation",
        },
        "authorities": {
            "weights": weights_receipt,
            "manifest": manifest_receipt,
            "manifest_content": manifest,
            "python_model_sources": python_sources,
            "jain_core": source_receipts(args.jain_core, args.jain_core_head),
            "generator": require_file_hash(Path(__file__), sha256_file(Path(__file__))),
        },
        "load_diagnostics": load_diagnostics,
        "environment": {
            "hostname": platform.node(),
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device": torch.cuda.get_device_name(device),
            "cuda_capability": list(torch.cuda.get_device_capability(device)),
        },
        "diagnosed_current_pipeline_differences": [
            "hyperion_vs emits an exact per-column rank grid, not raw learned cell logits",
            "hyperion_vs defaults to 3 sweeps rather than the old 100",
            "current sweep 0 is identity; the old method shuffles every sweep",
            "current train and test matrices are encoded separately; the old lift experiment encoded them jointly",
            "current tiling has one origin and fixed neutral padding; the old method uses two shifted origins and sampled row/column padding",
            "current default windows are 4/7/11/15; the old experiment used 3/6/9/12",
            "current cell artifact applies sigmoid then column percentiles; the old artifact collapsed raw h_real logits",
        ],
        "datasets": dataset_receipts,
    }
    write_json_exclusive(args.receipt, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument(
        "--jain-core", type=Path, default=Path("/home/ubuntu/jain-split/jain-core")
    )
    parser.add_argument("--jain-core-head", default=PINNED_JAIN_CORE_HEAD)
    parser.add_argument("--authority-host", default="xbabe1")
    parser.add_argument(
        "--authority-root",
        default="/home/ubuntu/seed_data_cache/Hyperion_Entanglement/hyperion_v5_modules",
    )
    parser.add_argument("--dataset-id", action="append", default=[])
    parser.add_argument("--sweeps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.sweeps != 100:
        parser.error("the .vs.orig.v2 contract requires exactly 100 sweeps")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


def main() -> None:
    result = run(parse_args())
    print(
        f"Complete: {result['dataset_count']} datasets, "
        f"{result['output_file_count']} .vs.orig.v2 files",
        flush=True,
    )


if __name__ == "__main__":
    main()
