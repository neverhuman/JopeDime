#!/usr/bin/env python3
"""H61 selection-metric and strict-margin admission runner (CPU prep & fixture-tested; GPU under qualification.lock)."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
H48_PATH = ROOT / "research/flywheel_selfplay/h48/run.py"
H49_SCORE_PATH = ROOT / "research/flywheel_selfplay/h49/score.py"
H55_PATH = ROOT / "research/flywheel_selfplay/h55/runner.py"
ELO_PATH = ROOT / "python/src/jope/north_star/elo_tracker.py"
DEFAULT_INPUT = Path.home() / "jope-data/h50/input"
DEFAULT_CHECKPOINT = Path.home() / "jope-data/h51/ckpt/round-882.pt"
DEFAULT_RUN_ROOT = Path(os.environ.get("H61_RUN_ROOT", str(Path.home() / "jope-data/h61/h61-wave")))

# Pinned H55 / H59 / H61 contract constants
DOPE_SHA = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
R882_SHA = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
MANIFEST_SHA = "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
MANIFEST_ID = "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3"
TRAINABLE_PARAMS = 29_211
R882_HARNESS_UTILITY = 1.1536152769912449

SEEDS = (12, 13, 14)
ARMS = ("S0", "S1", "S2", "S3")
FOLD_SEED = 4900
ROUNDS = 20
STEPS_PER_ROUND = 2
BATCH_SIZE = 256
SLATE_SIZE = 192
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

# H59-consistent loss and utility clipping constants
UTILITY_MIN = -2.0
UTILITY_MAX = 4.0
ROBUST_CAP = 1.0
SCALE_FLOOR = 0.05

# Unrevealed candidate mask index (0..3 are revealed anchors; 4..P-1 are unrevealed)
UNREVEALED_START_INDEX = 4
EVAL_MODE_UNREV = "unrev"
EVAL_MODE_NATIVE = "native"

# Calibrated null margin m (95th pct of single-fold 41-family null |gain| across H55/H52 receipts)
NULL_MARGIN_M = 0.001546
NULL_MARGIN_SEED_M = 0.000357
ADMISSION_LCB = 0.001
MAX_HARM_FRACTION = 0.10
HARM_THRESHOLD = -0.01
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 61_000
ELO_BOOTSTRAP_DRAWS = 2_000
ELO_PRIOR = 0.01
ELO_TIE_MARGIN = 1e-4

# Inner-CV hyperparameter grid for S2 (evaluated strictly on DL-Train fold_a inner folds)
TAU_GRID = (0.02, 0.05, 0.10, 0.20, 0.50)
M_S2_GRID = (NULL_MARGIN_M, 0.002000, 0.002500)

# Disk and resource guards (xbabe2 swap is full; start cells only when statvfs < 94.5%)
DISK_START_MAX_USED_PERCENT = 94.5
WRITER_MAX_DISK_USED_PERCENT = 95.0
MIN_DISK_FREE_BYTES = 1024**3
MIN_MEM_AVAILABLE_KIB = 16 * 1024 * 1024
MAX_HOST_CPU_PERCENT = 85.0

# H57 timing reference and H61 <= 60 min (3600 s) GPU cell wall cap
H57_MEASURED_GENERATION_SECONDS = 60.113566709915176
H57_TIMING_CELL_OVERHEAD_SECONDS = 128.61473213299178
MAX_GPU_CELL_WALL_CAP_SECONDS = 3600.0
H61_EQUIVALENT_TIMING_UNITS_PER_SEED = 8


class RetryableDiskInfraVoid(RuntimeError):
    """Raised when statvfs >= 94.5% at cell start or write time; retries the same cell without stopping the wave."""

    def __init__(self, message: str, *, stage: str, used_percent: float, available_bytes: int):
        super().__init__(message)
        self.stage = stage
        self.used_percent = float(used_percent)
        self.available_bytes = int(available_bytes)
        self.retryable = True
        self.same_cell = True
        self.stop_wave = False


@dataclass(frozen=True)
class TrainingBatch:
    """A deterministic, arm-independent batch and candidate slate selection."""

    round_index: int
    step_index: int
    seed: int
    round_rng_seed: int
    context_ids: tuple[str, ...]
    candidate_indices: tuple[np.ndarray, ...]
    candidate_pool_sha256: str


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def disk_usage_stats(path: str | Path = "/", statvfs_fn: Callable = os.statvfs) -> dict[str, float | int]:
    """Compute exact statvfs available-block usage on the target filesystem."""
    stats = statvfs_fn(str(path))
    total_bytes = int(stats.f_blocks * stats.f_frsize)
    available_bytes = int(stats.f_bavail * stats.f_frsize)
    used_percent = 100.0 * (stats.f_blocks - stats.f_bavail) / stats.f_blocks if stats.f_blocks else 100.0
    return {
        "total_bytes": total_bytes,
        "available_bytes": available_bytes,
        "used_percent": float(used_percent),
    }


def check_disk_guard(
    *,
    stage: str = "cell_start",
    path: str | Path = "/",
    statvfs_fn: Callable = os.statvfs,
    max_used_percent: float = DISK_START_MAX_USED_PERCENT,
) -> dict[str, float | int]:
    """Enforce statvfs < 94.5% and >= 1 GiB free; raise RetryableDiskInfraVoid on violation."""
    usage = disk_usage_stats(path=path, statvfs_fn=statvfs_fn)
    if int(usage["available_bytes"]) < MIN_DISK_FREE_BYTES or float(usage["used_percent"]) >= max_used_percent:
        raise RetryableDiskInfraVoid(
            f"unsafe filesystem / before H61 write ({stage}): used={usage['used_percent']:.3f}% "
            f"(limit <{max_used_percent:.1f}%), avail={usage['available_bytes']} B",
            stage=stage,
            used_percent=float(usage["used_percent"]),
            available_bytes=int(usage["available_bytes"]),
        )
    return usage


def check_writer_disk_budget(
    *,
    path: str | Path = "/",
    statvfs_fn: Callable = os.statvfs,
    max_used_percent: float = WRITER_MAX_DISK_USED_PERCENT,
) -> dict[str, float | int]:
    """Enforce the < 95.0% writer disk guard before persisting artifacts."""
    return check_disk_guard(stage="write", path=path, statvfs_fn=statvfs_fn, max_used_percent=max_used_percent)


def atomic_json(path: Path, payload: Mapping[str, object], *, statvfs_fn: Callable = os.statvfs) -> None:
    """Write JSON atomically after verifying the write-time <94.5% statvfs guard."""
    check_disk_guard(stage="write", statvfs_fn=statvfs_fn)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".new")
    try:
        tmp.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
        tmp.replace(path)
    except OSError as exc:
        usage = disk_usage_stats(statvfs_fn=statvfs_fn)
        raise RetryableDiskInfraVoid(
            f"unsafe filesystem / before H61 write: OS error ({exc}) treated as retryable infra-void",
            stage="write",
            used_percent=float(usage["used_percent"]),
            available_bytes=int(usage["available_bytes"]),
        ) from exc


def run_cells_with_disk_retry(
    cells: Iterable[tuple[str, int]],
    run_cell_fn: Callable[[str, int, int], dict],
    *,
    statvfs_fn: Callable = os.statvfs,
    max_attempts_per_cell: int = 3,
) -> dict:
    """Execute cells sequentially; a cell-start or write-time disk guard failure retries the SAME cell."""
    completed: list[dict] = []
    infra_voids: list[dict] = []
    for arm, seed in cells:
        attempt = 0
        while True:
            attempt += 1
            try:
                check_disk_guard(stage="cell_start", statvfs_fn=statvfs_fn)
                outcome = run_cell_fn(arm, seed, attempt)
                completed.append({"arm": arm, "seed": seed, "attempt": attempt, "outcome": outcome})
                break
            except RetryableDiskInfraVoid as void_err:
                infra_voids.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "attempt": attempt,
                        "stage": void_err.stage,
                        "used_percent": void_err.used_percent,
                        "retryable": void_err.retryable,
                        "same_cell": void_err.same_cell,
                        "stop_wave": void_err.stop_wave,
                    }
                )
                if attempt >= max_attempts_per_cell:
                    raise
    return {
        "wave_stopped": False,
        "completed_cells": completed,
        "infra_void_retries": infra_voids,
    }


def stage_r882_copy(source_checkpoint: Path, scratch_dir: Path, *, statvfs_fn: Callable = os.statvfs) -> Path:
    """Copy the r882 checkpoint into cell-local scratch and verify R882_SHA before loading."""
    if not source_checkpoint.exists():
        raise FileNotFoundError(f"missing r882 source checkpoint: {source_checkpoint}")
    if "flywheel-full-20261030" in str(source_checkpoint):
        raise RuntimeError("refusing to read r882 checkpoint directly from the live xbabe1 directory")
    if sha256(source_checkpoint) != R882_SHA:
        raise RuntimeError("source r882 checkpoint SHA256 mismatch")
    check_disk_guard(stage="write", statvfs_fn=statvfs_fn)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    copy_path = scratch_dir / "r882_adapter_copy.pt"
    shutil.copy2(source_checkpoint, copy_path)
    if sha256(copy_path) != R882_SHA:
        raise RuntimeError("staged r882 copy SHA256 mismatch")
    return copy_path


def prepare_training(input_dir: Path, checkpoint: Path, device: torch.device, scratch_dir: Path | None = None) -> dict:
    """Reuse H55/H59's pinned metadata, fold, staged r882 checkpoint copy, and fold-A loader contract."""
    h55 = import_module("h61_h55_runner", H55_PATH)
    if (h55.R882_SHA, h55.DOPE_SHA, h55.MANIFEST_SHA, h55.MANIFEST_ID) != (
        R882_SHA,
        DOPE_SHA,
        MANIFEST_SHA,
        MANIFEST_ID,
    ):
        raise RuntimeError("H61/H55 pinned inputs differ")
    staged_ckpt = stage_r882_copy(checkpoint, scratch_dir) if scratch_dir is not None else checkpoint
    if sha256(staged_ckpt) != R882_SHA:
        raise RuntimeError("H61 requires an exact r882 checkpoint copy")
    ctx = h55.prepare_training(input_dir, staged_ckpt, device)
    if ctx["fold_assignment"].get("seed") != FOLD_SEED:
        raise RuntimeError("H61 requires the H55 seed-4900 family fold")
    ctx["h55"] = h55
    ctx["input_dir"] = input_dir
    ctx["staged_checkpoint_path"] = str(staged_ckpt)
    return ctx


def paired_stream_seed(seed: int, round_index: int, step_index: int) -> int:
    """One fixed stream per (seed, round, step), with no arm-dependent salt."""
    if seed not in SEEDS or not 1 <= round_index <= ROUNDS or not 0 <= step_index < STEPS_PER_ROUND:
        raise ValueError("unregistered H61 seed/round/step stream")
    sampler_key = seed * 1_000_003 + 10_007
    payload = f"flywheel-batch/v1:{sampler_key}:{round_index}:{step_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def compute_candidate_pool_hash(
    context_ids: Iterable[str],
    candidate_indices: Iterable[np.ndarray],
    pools: Mapping[str, Mapping[str, object]],
) -> str:
    """Hash the ordered context IDs, candidate indices, and candidate program identities for a batch."""
    digest = hashlib.sha256()
    for cid, indices in zip(context_ids, candidate_indices, strict=True):
        arr = np.asarray(indices, dtype=np.int64)
        digest.update(str(cid).encode())
        digest.update(arr.tobytes())
        pool = pools[cid]
        if "pids" in pool:
            pids = pool["pids"]
            for idx in arr.tolist():
                digest.update(str(pids[int(idx)]).encode())
    return digest.hexdigest()


def build_paired_batches(
    train_ids: Iterable[str],
    pools: Mapping[str, dict],
    sorted_orders: Mapping[str, np.ndarray],
    cached_training_indices: Callable,
    *,
    seed: int,
    batch_seed: Callable[[int, int, int], int] | None = None,
) -> tuple[TrainingBatch, ...]:
    """Freeze every context and slate draw once so all four arms replay identical updates and pool hashes."""
    if seed not in SEEDS:
        raise ValueError("unregistered H61 seed")
    ids = tuple(sorted(train_ids))
    if not ids or set(ids) - pools.keys() or set(ids) - sorted_orders.keys():
        raise ValueError("H61 batch plan requires a nonempty fold-A-only roster")
    make_batch_seed = batch_seed or paired_stream_seed
    sampler_key = seed * 1_000_003 + 10_007
    batches: list[TrainingBatch] = []
    rng_state = np.random.get_state()
    try:
        for round_index in range(1, ROUNDS + 1):
            round_rng_seed = int((sampler_key + round_index * 104729) % (2**32))
            np.random.seed(round_rng_seed)
            rotation = round_index % len(ids)
            round_ids = ids[rotation:] + ids[:rotation]
            for step_index in range(STEPS_PER_ROUND):
                stream = make_batch_seed(seed, round_index, step_index)
                cids = tuple(random.Random(stream).choices(round_ids, k=BATCH_SIZE))
                indexes = tuple(
                    np.asarray(
                        cached_training_indices(pools[cid], sorted_orders[cid], SLATE_SIZE),
                        dtype=np.int64,
                    ).copy()
                    for cid in cids
                )
                if any(index.shape != (SLATE_SIZE,) for index in indexes):
                    raise RuntimeError("H61 sampler returned a slate with the wrong width")
                pool_hash = compute_candidate_pool_hash(cids, indexes, pools)
                batches.append(
                    TrainingBatch(
                        round_index=round_index,
                        step_index=step_index,
                        seed=stream,
                        round_rng_seed=round_rng_seed,
                        context_ids=cids,
                        candidate_indices=indexes,
                        candidate_pool_sha256=pool_hash,
                    )
                )
    finally:
        np.random.set_state(rng_state)
    if len(batches) != ROUNDS * STEPS_PER_ROUND:
        raise RuntimeError("H61 fixed round/step plan is incomplete")
    return tuple(batches)


def paired_plans_for_arms(plan: tuple[TrainingBatch, ...]) -> dict[str, tuple[TrainingBatch, ...]]:
    """Bind each arm S0..S3 to the exact same immutable batch plan and verify stream/pool-hash parity."""
    if len(plan) != ROUNDS * STEPS_PER_ROUND:
        raise ValueError("H61 paired plan has the wrong number of updates")
    plans = {arm: plan for arm in ARMS}
    assert_identical_round_streams(plans)
    return plans


def assert_identical_round_streams(plans_by_arm: Mapping[str, tuple[TrainingBatch, ...]]) -> None:
    """Assert that every arm S0..S3 has the exact same RNG stream and candidate pool hash at every round/step."""
    if set(plans_by_arm) != set(ARMS):
        raise ValueError("H61 stream parity check requires all four arms S0..S3")
    reference = plans_by_arm["S0"]
    if len(reference) != ROUNDS * STEPS_PER_ROUND:
        raise RuntimeError("reference batch plan length mismatch")
    for arm in ARMS[1:]:
        candidate = plans_by_arm[arm]
        if len(candidate) != len(reference):
            raise RuntimeError(f"arm {arm} batch count differs from S0")
        for ref_b, cand_b in zip(reference, candidate, strict=True):
            if (
                ref_b.round_index != cand_b.round_index
                or ref_b.step_index != cand_b.step_index
                or ref_b.seed != cand_b.seed
                or ref_b.round_rng_seed != cand_b.round_rng_seed
                or ref_b.context_ids != cand_b.context_ids
                or ref_b.candidate_pool_sha256 != cand_b.candidate_pool_sha256
            ):
                raise RuntimeError(
                    f"H61 stream/candidate-pool mismatch between S0 and {arm} at "
                    f"round={ref_b.round_index} step={ref_b.step_index}"
                )
            for idx_a, idx_b in zip(ref_b.candidate_indices, cand_b.candidate_indices, strict=True):
                if not np.array_equal(idx_a, idx_b):
                    raise RuntimeError(f"H61 candidate slate indices differ between S0 and {arm}")


def clip_context_utility(fitness: float | np.ndarray, gate: bool | np.ndarray) -> float | np.ndarray:
    """Clip passing utility u = -fitness to [-2.0, 4.0] and map gate failures to -2.0 (H59-consistent)."""
    fit_arr = np.asarray(fitness, dtype=np.float64)
    gate_arr = np.asarray(gate, dtype=bool)
    clipped = np.clip(-fit_arr, UTILITY_MIN, UTILITY_MAX)
    out = np.where(gate_arr, clipped, UTILITY_MIN)
    if np.ndim(fitness) == 0 and np.ndim(gate) == 0:
        return float(out)
    return out


def clip_context_gain(
    candidate_utility: float | np.ndarray,
    incumbent_utility: float | np.ndarray,
    *,
    cap: float = ROBUST_CAP,
) -> float | np.ndarray:
    """Cap per-context utility deltas to [-ROBUST_CAP, +ROBUST_CAP] so -2.0 failures cannot dominate bootstrap."""
    cand_c = np.clip(np.asarray(candidate_utility, dtype=np.float64), UTILITY_MIN, UTILITY_MAX)
    inc_c = np.clip(np.asarray(incumbent_utility, dtype=np.float64), UTILITY_MIN, UTILITY_MAX)
    delta = np.clip(cand_c - inc_c, -abs(cap), abs(cap))
    if np.ndim(candidate_utility) == 0 and np.ndim(incumbent_utility) == 0:
        return float(delta)
    return delta


def preregistered_targets(fitness: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    """Map cached cost fitness to t = -clip(u, -2, 4); failed gates retain cost 2.0 (exact H59 contract)."""
    if fitness.shape != gates.shape:
        raise ValueError("fitness and gate tensors must have matching slate shapes")
    utility = -fitness
    targets = -torch.clamp(utility, min=UTILITY_MIN, max=UTILITY_MAX)
    return torch.where(gates.to(dtype=torch.bool), targets, torch.full_like(targets, -UTILITY_MIN))


def training_slate_scale(targets: torch.Tensor) -> torch.Tensor:
    """Detached MAD scale computed only from the current training slate targets (exact H59 contract)."""
    if targets.ndim != 2 or targets.shape[1] == 0:
        raise ValueError("H61 scale requires [batch, training_slate] targets")
    with torch.no_grad():
        center = targets.detach().median(dim=1).values
        mad = (targets.detach() - center[:, None]).abs().median(dim=1).values
        return (1.4826 * mad).clamp_min(SCALE_FLOOR).detach()


def clipped_loss(prediction: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """H59-consistent per-context clipped Smooth-L1 + KL objective normalized by detached slate MAD scale."""
    if prediction.shape != targets.shape or prediction.ndim != 2:
        raise ValueError("H61 loss expects matching [batch, slate] tensors")
    smooth = F.smooth_l1_loss(prediction, targets, reduction="none", beta=1.0).clamp(max=ROBUST_CAP)
    row_smooth = smooth.mean(dim=1)
    row_kl = F.kl_div(
        F.log_softmax(-prediction / 0.2, dim=-1),
        F.softmax(-targets / 0.2, dim=-1),
        reduction="none",
    ).sum(dim=1)
    per_context = (row_smooth + 0.1 * row_kl) / training_slate_scale(targets)
    return per_context.mean()


def validate_trainable_inventory(named_parameters: Iterable[tuple[str, object]]) -> tuple[str, ...]:
    parameters = tuple(named_parameters)
    trainable = tuple((name, parameter) for name, parameter in parameters if parameter.requires_grad)
    names = tuple(sorted(name for name, _ in trainable))
    count = sum(parameter.numel() for _, parameter in trainable)
    if count != TRAINABLE_PARAMS:
        raise RuntimeError(f"H61 expected {TRAINABLE_PARAMS} trainable parameters, got {count}")
    if not names or any(
        not name.startswith("adapter_10tok.") or ".anchor." in name or name.endswith(".anchor")
        for name in names
    ):
        raise RuntimeError("H61 may train only non-anchor adapter_10tok parameters")
    return names


def validate_dope_hash(actual: str) -> None:
    if actual != DOPE_SHA:
        raise RuntimeError(f"H61 frozen DOPE SHA mismatch: {actual}")


def verify_cell_integrity(ctx: dict) -> tuple[str, ...]:
    """Verify adapter-only trainable inventory and frozen DOPE hash before/after every round."""
    names = validate_trainable_inventory(ctx["net"].named_parameters())
    actual = ctx["h48"].h46.checkpoint_guard(ctx["guard"], ctx["net"], expected=DOPE_SHA)
    validate_dope_hash(actual)
    return names


def predict_pool_scores(
    cids: Iterable[str],
    pools: Mapping[str, dict],
    predict: Callable[[list[str], list[np.ndarray]], torch.Tensor],
) -> dict[str, np.ndarray]:
    """Predict full candidate pool cost vectors for the requested contexts."""
    preds_by_cid: dict[str, np.ndarray] = {}
    by_size: dict[int, list[str]] = {}
    for cid in sorted(cids):
        by_size.setdefault(len(pools[cid]["pids"]), []).append(cid)
    for pool_size, contexts in sorted(by_size.items()):
        for start in range(0, len(contexts), 8):
            batch = contexts[start : start + 8]
            indexes = [np.arange(pool_size) for _ in batch]
            with torch.no_grad():
                values = predict(batch, indexes).detach().cpu().numpy()
            for row, cid in zip(values, batch, strict=True):
                preds_by_cid[cid] = np.asarray(row, dtype=np.float64)
    return preds_by_cid


def _batch_targets(pools: Mapping[str, dict], batch: TrainingBatch, device: torch.device) -> torch.Tensor:
    fitness = np.stack(
        [pools[cid]["fitness"][ix] for cid, ix in zip(batch.context_ids, batch.candidate_indices, strict=True)]
    )
    gates = np.stack(
        [pools[cid]["gate"][ix] for cid, ix in zip(batch.context_ids, batch.candidate_indices, strict=True)]
    )
    return preregistered_targets(
        torch.as_tensor(fitness, dtype=torch.float32, device=device),
        torch.as_tensor(gates, dtype=torch.bool, device=device),
    )


def train_arm_with_round_selection(
    ctx: dict,
    arm: str,
    seed: int,
    plan: tuple[TrainingBatch, ...],
) -> tuple[dict, dict]:
    """Run the 20-round plan, checking DOPE before/after every round and applying arm selection strictly on fold A."""
    if arm not in ARMS or seed not in SEEDS or len(plan) != ROUNDS * STEPS_PER_ROUND:
        raise ValueError("invalid H61 arm, seed or fixed update plan")
    expected_streams = tuple(
        paired_stream_seed(seed, round_index, step_index)
        for round_index in range(1, ROUNDS + 1)
        for step_index in range(STEPS_PER_ROUND)
    )
    if tuple(batch.seed for batch in plan) != expected_streams:
        raise ValueError("H61 arm received a batch plan for a different seed or stream order")
    net, h48 = ctx["net"], ctx["h48"]
    net.load_state_dict(ctx["checkpoint_state"]["state_dict"], strict=True)
    names = verify_cell_integrity(ctx)
    trainable = [parameter for _, parameter in net.named_parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    h48.base.guard.assert_optimizer_excludes_dope(optimizer, net)
    h55 = ctx.get("h55") or import_module("h61_h55_runner", H55_PATH)

    has_fold_a_selection = (
        "folds" in ctx
        and len(ctx["folds"]) >= 4
        and all("rec" in pool for pool in ctx["pools_a"].values())
    )
    selected_state = h55.adapter_state(net)
    tau_star = 0.10
    m_s2_star = NULL_MARGIN_M
    s2_inner_cv_summary: dict | None = None
    inner_sel_cids: list[str] = []
    inc_sel_preds: dict[str, np.ndarray] = {}
    r882_preds_a: dict[str, np.ndarray] = {}
    if has_fold_a_selection:
        inner_sel_families = set(ctx["folds"][3])
        inner_sel_cids = [
            cid
            for cid in sorted(ctx["pools_a"])
            if ctx["pools_a"][cid]["rec"]["family_id"] in inner_sel_families
        ]
        net.eval()
        if arm == "S2":
            r882_preds_a = predict_pool_scores(sorted(ctx["pools_a"]), ctx["pools_a"], ctx["predictor_a"])
            inc_sel_preds = {cid: r882_preds_a[cid] for cid in inner_sel_cids}
        else:
            inc_sel_preds = predict_pool_scores(inner_sel_cids, ctx["pools_a"], ctx["predictor_a"])

    grad_norms: list[float] = []
    dope_checks_by_round: list[dict] = []
    round_decisions: list[dict] = []
    accepted_rounds: list[int] = []
    for round_index in range(1, ROUNDS + 1):
        verify_cell_integrity(ctx)
        dope_before_round = h48.h46.checkpoint_guard(ctx["guard"], net, expected=DOPE_SHA)
        validate_dope_hash(dope_before_round)
        round_batches = [b for b in plan if b.round_index == round_index]
        for batch in round_batches:
            cids = list(batch.context_ids)
            indexes = list(batch.candidate_indices)
            prediction = ctx["predictor_a"](cids, indexes)
            targets = _batch_targets(ctx["pools_a"], batch, prediction.device)
            optimizer.zero_grad(set_to_none=True)
            loss = clipped_loss(prediction, targets)
            loss.backward()
            grad_norms.append(float(torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)))
            optimizer.step()
        verify_cell_integrity(ctx)
        dope_after_round = h48.h46.checkpoint_guard(ctx["guard"], net, expected=DOPE_SHA)
        validate_dope_hash(dope_after_round)
        dope_checks_by_round.append(
            {
                "round": round_index,
                "dope_sha256_before": dope_before_round,
                "dope_sha256_after": dope_after_round,
            }
        )
        if has_fold_a_selection:
            net.eval()
            if arm == "S2" and s2_inner_cv_summary is None:
                cand_preds_a = predict_pool_scores(sorted(ctx["pools_a"]), ctx["pools_a"], ctx["predictor_a"])
                inner_folds = [set(f) for f in ctx["folds"][:4]]
                forbidden_cids = [
                    r["context_sha256"]
                    for r in list(ctx.get("b_rows", ())) + list(ctx.get("dev_rows", ()))
                ]
                forbidden_fams = set(ctx.get("fold_b_families", ())) | {
                    r["family_id"] for r in ctx.get("dev_rows", ())
                }
                inc_by_fold = {i: r882_preds_a for i in range(len(inner_folds))}
                cand_by_fold = {i: cand_preds_a for i in range(len(inner_folds))}
                s2_inner_cv_summary = select_s2_hyperparameters_inner_cv(
                    inner_folds,
                    ctx["pools_a"],
                    inc_by_fold,
                    cand_by_fold,
                    forbidden_context_ids=forbidden_cids,
                    forbidden_family_ids=forbidden_fams,
                )
                tau_star = float(s2_inner_cv_summary["tau"])
                m_s2_star = float(s2_inner_cv_summary["m_s2"])
                cand_sel_preds = {cid: cand_preds_a[cid] for cid in inner_sel_cids}
            else:
                cand_sel_preds = predict_pool_scores(inner_sel_cids, ctx["pools_a"], ctx["predictor_a"])
            decision = select_round_candidate(
                arm,
                inner_sel_cids,
                ctx["pools_a"],
                inc_sel_preds,
                cand_sel_preds,
                margin=NULL_MARGIN_M,
                tau=tau_star,
                m_s2=m_s2_star,
                bootstrap_seed=BOOTSTRAP_SEED + seed * 100 + round_index,
            )
            if decision["accepted"]:
                selected_state = h55.adapter_state(net)
                inc_sel_preds = cand_sel_preds
                accepted_rounds.append(round_index)
            round_decisions.append(
                {
                    "round": round_index,
                    "accepted": bool(decision["accepted"]),
                    "reported_eval_mode": decision["reported_eval_mode"],
                    "reported_family_utility": decision["reported_family_utility"],
                    "candidate_unrev_family_utility": decision["candidate_unrev_family_utility"],
                    "incumbent_unrev_family_utility": decision["incumbent_unrev_family_utility"],
                    "unrev_mean_gain": decision["unrev_mean_gain"],
                    "unrev_lcb95": decision["unrev_lcb95"],
                    "selection_signal_gain": decision["selection_signal_gain"],
                }
            )
        else:
            selected_state = h55.adapter_state(net)

    raw_final_state = h55.adapter_state(net)
    if has_fold_a_selection:
        h55.set_adapter(net, selected_state)
    final_dope = h48.h46.checkpoint_guard(ctx["guard"], net, expected=DOPE_SHA)
    validate_dope_hash(final_dope)
    return selected_state, {
        "arm": arm,
        "seed": seed,
        "rounds_completed": ROUNDS,
        "optimizer_updates_completed": len(plan),
        "paired_streams": [[b.round_index, b.step_index, b.seed, b.candidate_pool_sha256] for b in plan],
        "trainable_parameter_names": list(names),
        "trainable_parameter_count": TRAINABLE_PARAMS,
        "dope_checks_by_round": dope_checks_by_round,
        "dope_sha256_before": dope_checks_by_round[0]["dope_sha256_before"],
        "dope_sha256_after": final_dope,
        "grad_norms": grad_norms,
        "accepted_rounds": accepted_rounds,
        "accepted_rounds_count": len(accepted_rounds),
        "rejected_rounds_count": len(round_decisions) - len(accepted_rounds),
        "round_selection_decisions": round_decisions,
        "s2_inner_cv": s2_inner_cv_summary,
        "raw_final_adapter_sha256": h55.adapter_digest(raw_final_state),
        "selected_adapter_sha256": h55.adapter_digest(selected_state),
        "fold_b_scored_during_training": False,
        "validation_payloads_opened": False,
        "holdout_payloads_opened": 0,
        "outer_rows_delivered": 0,
        "r882_checkpoint_sha256": R882_SHA,
        "source_manifest_sha256": MANIFEST_SHA,
        "manifest_identity": MANIFEST_ID,
    }


def evaluate_pool_predictions(
    cids: Iterable[str],
    pools: Mapping[str, dict],
    predictions_by_cid: Mapping[str, np.ndarray],
    *,
    eval_mode: str = EVAL_MODE_UNREV,
) -> dict:
    """Evaluate context and family utilities under leak-free unrev (4..P-1) or native (0..P-1) selection."""
    if eval_mode not in (EVAL_MODE_UNREV, EVAL_MODE_NATIVE):
        raise ValueError(f"unsupported eval_mode: {eval_mode}")
    per_context: dict[str, float] = {}
    chosen_indices: dict[str, int] = {}
    per_family_lists: dict[str, list[float]] = {}
    for cid in sorted(cids):
        preds = np.asarray(predictions_by_cid[cid], dtype=np.float64)
        pool = pools[cid]
        fitness = np.asarray(pool["fitness"], dtype=np.float64)
        gate = np.asarray(pool["gate"], dtype=bool)
        if preds.ndim != 1 or len(preds) != len(fitness) or len(preds) != len(gate):
            raise ValueError(f"prediction/pool length mismatch for context {cid}")
        if eval_mode == EVAL_MODE_UNREV:
            if len(preds) <= UNREVEALED_START_INDEX:
                raise ValueError(f"context {cid} pool size {len(preds)} has no unrevealed candidates 4..P-1")
            choice = UNREVEALED_START_INDEX + int(np.argmin(preds[UNREVEALED_START_INDEX:]))
        else:
            choice = int(np.argmin(preds))
        chosen_indices[cid] = choice
        utility = float(clip_context_utility(float(fitness[choice]), bool(gate[choice])))
        per_context[cid] = utility
        family_id = str(pool["rec"]["family_id"])
        per_family_lists.setdefault(family_id, []).append(utility)
    per_family = {fam: float(np.mean(vals)) for fam, vals in sorted(per_family_lists.items())}
    return {
        "eval_mode": eval_mode,
        "mean": float(np.mean(list(per_family.values()))) if per_family else 0.0,
        "context_mean": float(np.mean(list(per_context.values()))) if per_context else 0.0,
        "per_context": per_context,
        "per_family": per_family,
        "chosen_indices": chosen_indices,
    }


def softmin_unrev_expected_utility(
    cids: Iterable[str],
    pools: Mapping[str, dict],
    predictions_by_cid: Mapping[str, np.ndarray],
    *,
    tau: float,
) -> dict:
    """Compute smoothed expected utility under softmin(pred / tau) strictly over unrevealed indices 4..P-1."""
    if tau <= 0.0 or not np.isfinite(tau):
        raise ValueError("tau must be a positive finite float")
    per_context: dict[str, float] = {}
    per_family_lists: dict[str, list[float]] = {}
    for cid in sorted(cids):
        preds = np.asarray(predictions_by_cid[cid], dtype=np.float64)
        if len(preds) <= UNREVEALED_START_INDEX:
            raise ValueError(f"context {cid} pool size {len(preds)} has no unrevealed candidates 4..P-1")
        unrev_preds = preds[UNREVEALED_START_INDEX:]
        pool = pools[cid]
        unrev_fit = np.asarray(pool["fitness"][UNREVEALED_START_INDEX:], dtype=np.float64)
        unrev_gate = np.asarray(pool["gate"][UNREVEALED_START_INDEX:], dtype=bool)
        unrev_utilities = np.asarray(clip_context_utility(unrev_fit, unrev_gate), dtype=np.float64)
        scaled = -unrev_preds / float(tau)
        scaled = scaled - np.max(scaled)
        weights = np.exp(scaled)
        weights = weights / np.sum(weights)
        expected_u = float(np.sum(weights * unrev_utilities))
        per_context[cid] = expected_u
        family_id = str(pool["rec"]["family_id"])
        per_family_lists.setdefault(family_id, []).append(expected_u)
    per_family = {fam: float(np.mean(vals)) for fam, vals in sorted(per_family_lists.items())}
    return {
        "eval_mode": EVAL_MODE_UNREV,
        "tau": float(tau),
        "mean": float(np.mean(list(per_family.values()))) if per_family else 0.0,
        "context_mean": float(np.mean(list(per_context.values()))) if per_context else 0.0,
        "per_context": per_context,
        "per_family": per_family,
    }


def compute_clipped_family_gains(
    cids: Iterable[str],
    pools: Mapping[str, dict],
    candidate_per_context: Mapping[str, float],
    incumbent_per_context: Mapping[str, float],
    *,
    cap: float = ROBUST_CAP,
) -> dict[str, float]:
    """Compute per-family utility gains from H59-clipped per-context deltas so -2.0 failures cannot dominate."""
    per_family_deltas: dict[str, list[float]] = {}
    for cid in sorted(cids):
        delta = float(clip_context_gain(candidate_per_context[cid], incumbent_per_context[cid], cap=cap))
        fam = str(pools[cid]["rec"]["family_id"])
        per_family_deltas.setdefault(fam, []).append(delta)
    return {fam: float(np.mean(vals)) for fam, vals in sorted(per_family_deltas.items())}


def bootstrap_single_seed_family_gains(
    family_gains: Mapping[str, float],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Paired family-cluster bootstrap for a single seed's per-family gains."""
    if not family_gains:
        raise ValueError("family_gains must be nonempty")
    families = sorted(family_gains)
    values = np.asarray([float(family_gains[f]) for f in families], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("family_gains must be finite")
    nonzero_count = int(np.sum(np.abs(values) > 1e-12))
    rng = np.random.default_rng(bootstrap_seed)
    indexes = rng.integers(0, len(values), size=(draws, len(values)))
    mean_boot = values[indexes].mean(axis=1)
    median_boot = np.median(values[indexes], axis=1)
    return {
        "family_count": len(values),
        "nonzero_family_count": nonzero_count,
        "is_tie": nonzero_count == 0 or float(values.mean()) == 0.0,
        "mean_gain": float(values.mean()),
        "mean_lcb95_one_sided": float(np.quantile(mean_boot, 0.05)),
        "median_gain": float(np.median(values)),
        "median_ci95": [float(np.quantile(median_boot, 0.025)), float(np.quantile(median_boot, 0.975))],
        "harm_fraction": float(np.mean(values < HARM_THRESHOLD)),
    }


def select_round_candidate(
    arm: str,
    cids: Iterable[str],
    pools: Mapping[str, dict],
    incumbent_preds_by_cid: Mapping[str, np.ndarray],
    candidate_preds_by_cid: Mapping[str, np.ndarray],
    *,
    margin: float = NULL_MARGIN_M,
    tau: float = 0.10,
    m_s2: float = NULL_MARGIN_M,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Apply arm-specific selection rule (S0..S3) while always recording leak-free unrev (4..P-1) utility."""
    if arm not in ARMS:
        raise ValueError(f"unregistered H61 arm: {arm}")
    cids_tuple = tuple(sorted(cids))
    # Every arm's reported utility is evaluated in leak-free unrev (4..P-1) mode.
    inc_unrev = evaluate_pool_predictions(cids_tuple, pools, incumbent_preds_by_cid, eval_mode=EVAL_MODE_UNREV)
    cand_unrev = evaluate_pool_predictions(cids_tuple, pools, candidate_preds_by_cid, eval_mode=EVAL_MODE_UNREV)
    family_gains_unrev = compute_clipped_family_gains(
        cids_tuple, pools, cand_unrev["per_context"], inc_unrev["per_context"]
    )
    boot_unrev = bootstrap_single_seed_family_gains(family_gains_unrev, bootstrap_seed=bootstrap_seed)

    if arm == "S0":
        # S0 selects using the legacy native rule (0..P-1, ties accepted), but reports unrev utility!
        inc_native = evaluate_pool_predictions(cids_tuple, pools, incumbent_preds_by_cid, eval_mode=EVAL_MODE_NATIVE)
        cand_native = evaluate_pool_predictions(cids_tuple, pools, candidate_preds_by_cid, eval_mode=EVAL_MODE_NATIVE)
        native_gain = float(cand_native["context_mean"] - inc_native["context_mean"])
        accepted = bool(cand_native["context_mean"] >= inc_native["context_mean"])
        selection_signal_gain = native_gain
    elif arm == "S1":
        # S1: strict top-1 unrev-masked (4..P-1) with margin m=0.001546 AND LCB95 > +0.001; ties fail.
        selection_signal_gain = boot_unrev["mean_gain"]
        accepted = bool(
            not boot_unrev["is_tie"]
            and boot_unrev["mean_gain"] > max(margin, NULL_MARGIN_M)
            and boot_unrev["mean_lcb95_one_sided"] > ADMISSION_LCB
        )
    elif arm == "S2":
        # S2: smoothed softmin(pred / tau) over unrevealed 4..P-1; ties fail, must exceed m_s2 and LCB95 > +0.001.
        inc_soft = softmin_unrev_expected_utility(cids_tuple, pools, incumbent_preds_by_cid, tau=tau)
        cand_soft = softmin_unrev_expected_utility(cids_tuple, pools, candidate_preds_by_cid, tau=tau)
        soft_family_gains = compute_clipped_family_gains(
            cids_tuple, pools, cand_soft["per_context"], inc_soft["per_context"]
        )
        soft_boot = bootstrap_single_seed_family_gains(soft_family_gains, bootstrap_seed=bootstrap_seed)
        effective_m = max(float(m_s2), NULL_MARGIN_M)
        selection_signal_gain = soft_boot["mean_gain"]
        accepted = bool(
            not soft_boot["is_tie"]
            and not boot_unrev["is_tie"]
            and soft_boot["mean_gain"] > effective_m
            and soft_boot["mean_lcb95_one_sided"] > ADMISSION_LCB
            and boot_unrev["mean_gain"] > NULL_MARGIN_M
            and boot_unrev["mean_lcb95_one_sided"] > ADMISSION_LCB
        )
    else:  # S3
        # S3: family-cluster bootstrap LCB selection (LCB95 > m=0.001546 and median >= 0); ties fail.
        effective_m = max(float(margin), NULL_MARGIN_M, ADMISSION_LCB)
        selection_signal_gain = boot_unrev["mean_lcb95_one_sided"]
        accepted = bool(
            not boot_unrev["is_tie"]
            and boot_unrev["mean_gain"] > effective_m
            and boot_unrev["mean_lcb95_one_sided"] > effective_m
            and boot_unrev["median_gain"] >= 0.0
        )

    reported = cand_unrev if accepted else inc_unrev
    return {
        "arm": arm,
        "accepted": accepted,
        "reported_eval_mode": EVAL_MODE_UNREV,
        "reported_family_utility": reported["mean"],
        "reported_context_utility": reported["context_mean"],
        "reported_per_context": reported["per_context"],
        "reported_per_family": reported["per_family"],
        "reported_chosen_indices": reported["chosen_indices"],
        "candidate_unrev_family_utility": cand_unrev["mean"],
        "incumbent_unrev_family_utility": inc_unrev["mean"],
        "unrev_mean_gain": boot_unrev["mean_gain"],
        "unrev_lcb95": boot_unrev["mean_lcb95_one_sided"],
        "selection_signal_gain": float(selection_signal_gain),
    }


def select_s2_hyperparameters_inner_cv(
    inner_folds: Iterable[set[str]],
    pools_a: Mapping[str, dict],
    incumbent_preds_by_fold: Mapping[int, Mapping[str, np.ndarray]],
    candidate_preds_by_fold: Mapping[int, Mapping[str, np.ndarray]],
    *,
    forbidden_context_ids: Iterable[str] = (),
    forbidden_family_ids: Iterable[str] = (),
    tau_grid: Iterable[float] = TAU_GRID,
    m_grid: Iterable[float] = M_S2_GRID,
) -> dict:
    """Select (tau, m_S2) strictly on DL-Train fold-A inner CV; refuse any DL-Val or fold-B leakage."""
    folds_list = [set(f) for f in inner_folds]
    if len(folds_list) < 2 or any(not f for f in folds_list):
        raise ValueError("S2 inner CV requires at least 2 nonempty disjoint DL-Train folds")
    flat_fams = [fam for f in folds_list for fam in f]
    if len(flat_fams) != len(set(flat_fams)):
        raise ValueError("S2 inner CV folds must be family-disjoint")

    forbidden_cids = set(forbidden_context_ids)
    forbidden_fams = set(forbidden_family_ids)
    if set(flat_fams) & forbidden_fams:
        raise RuntimeError("fold-B or DL-Val family leaked into S2 inner CV")

    for cid, pool in pools_a.items():
        rec = pool.get("rec", {})
        if cid in forbidden_cids:
            raise RuntimeError(f"forbidden context {cid} (fold-B or DL-Val) entered S2 inner CV")
        if rec.get("dl_split") != "DL-Train" or rec.get("source_split") not in ("train", "inner"):
            raise RuntimeError(f"non-DL-Train row ({rec.get('dl_split')}/{rec.get('source_split')}) entered S2 inner CV")
        if rec.get("family_id") in forbidden_fams:
            raise RuntimeError(f"forbidden family {rec.get('family_id')} entered S2 inner CV")

    for fold_idx in range(len(folds_list)):
        for preds_map in (incumbent_preds_by_fold[fold_idx], candidate_preds_by_fold[fold_idx]):
            for cid in preds_map:
                if cid in forbidden_cids or cid not in pools_a:
                    raise RuntimeError(f"fold-B or DL-Val context {cid} present in S2 inner-CV predictions")

    grid_results: list[dict] = []
    best_tuple: tuple[float, float, float, float] | None = None
    best_choice: tuple[float, float] | None = None

    for tau in sorted(float(t) for t in tau_grid):
        for m_s2 in sorted(float(m) for m in m_grid):
            if m_s2 < NULL_MARGIN_M:
                raise ValueError(f"m_S2 grid value {m_s2} cannot be below calibrated null margin {NULL_MARGIN_M}")
            oof_family_gains: dict[str, float] = {}
            for fold_idx, val_families in enumerate(folds_list):
                fold_cids = [
                    cid for cid in sorted(pools_a) if pools_a[cid]["rec"]["family_id"] in val_families
                ]
                decision = select_round_candidate(
                    "S2",
                    fold_cids,
                    pools_a,
                    incumbent_preds_by_fold[fold_idx],
                    candidate_preds_by_fold[fold_idx],
                    margin=NULL_MARGIN_M,
                    tau=tau,
                    m_s2=m_s2,
                    bootstrap_seed=BOOTSTRAP_SEED + fold_idx,
                )
                inc_unrev = evaluate_pool_predictions(
                    fold_cids, pools_a, incumbent_preds_by_fold[fold_idx], eval_mode=EVAL_MODE_UNREV
                )
                fold_gains = compute_clipped_family_gains(
                    fold_cids, pools_a, decision["reported_per_context"], inc_unrev["per_context"]
                )
                oof_family_gains.update(fold_gains)

            stats = bootstrap_single_seed_family_gains(oof_family_gains, bootstrap_seed=BOOTSTRAP_SEED)
            row = {
                "tau": tau,
                "m_s2": m_s2,
                "oof_mean_gain": stats["mean_gain"],
                "oof_lcb95": stats["mean_lcb95_one_sided"],
            }
            grid_results.append(row)
            score_key = (stats["mean_gain"], stats["mean_lcb95_one_sided"], -m_s2, -tau)
            if best_tuple is None or score_key > best_tuple:
                best_tuple = score_key
                best_choice = (tau, m_s2)

    assert best_choice is not None
    return {
        "tau": best_choice[0],
        "m_s2": best_choice[1],
        "cv_fold_count": len(folds_list),
        "cv_family_count": len(flat_fams),
        "dl_val_rows_used": 0,
        "fold_b_rows_used": 0,
        "grid_results": grid_results,
    }


def paired_family_admission(
    gains_by_seed: Mapping[int, Mapping[str, float]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Family-cluster admission with H59-clipped family gains; ties fail everywhere (gain > m and LCB95 > +0.001)."""
    if tuple(sorted(gains_by_seed)) != SEEDS or draws != BOOTSTRAP_DRAWS:
        raise ValueError("H61 admission requires seeds 12/13/14 and exactly 10,000 draws")
    family_sets = [set(gains_by_seed[seed]) for seed in SEEDS]
    if not family_sets[0] or any(families != family_sets[0] for families in family_sets[1:]):
        raise ValueError("H61 paired family inventory differs across seeds")
    families = tuple(sorted(family_sets[0]))
    raw_matrix = np.asarray(
        [[float(gains_by_seed[seed][family]) for seed in SEEDS] for family in families],
        dtype=np.float64,
    )
    if not np.isfinite(raw_matrix).all():
        raise ValueError("H61 family gains must be finite")
    # Clip per-family gains to [-ROBUST_CAP, +ROBUST_CAP] so -2.0 failure contexts cannot dominate bootstrap
    matrix = np.clip(raw_matrix, -ROBUST_CAP, ROBUST_CAP)
    family_mean = matrix.mean(axis=1)
    family_seed_median = np.median(matrix, axis=1)
    nonzero_families = int(np.sum(np.abs(family_mean) > 1e-12))

    rng = np.random.default_rng(bootstrap_seed)
    indexes = rng.integers(0, len(families), size=(draws, len(families)))
    mean_boot = family_mean[indexes].mean(axis=1)
    pooled_median_boot = np.median(family_seed_median[indexes], axis=1)
    mean_gain = float(family_mean.mean())
    mean_lcb = float(np.quantile(mean_boot, 0.05))
    pooled_median = float(np.median(family_seed_median))
    harm_fraction = float(np.mean(family_seed_median < HARM_THRESHOLD))

    seed_means = np.asarray([float(matrix[:, col].mean()) for col in range(len(SEEDS))], dtype=np.float64)
    seed_boot_idx = rng.integers(0, len(SEEDS), size=(draws, len(SEEDS)))
    seed_median_boot = np.median(seed_means[seed_boot_idx], axis=1)

    per_seed = {}
    for column, seed in enumerate(SEEDS):
        values = matrix[:, column]
        boot = np.median(values[indexes], axis=1)
        per_seed[str(seed)] = {
            "family_mean_gain": float(values.mean()),
            "family_median_gain": float(np.median(values)),
            "family_median_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "harm_fraction": float(np.mean(values < HARM_THRESHOLD)),
        }

    is_tie = nonzero_families == 0 or mean_gain == 0.0
    admitted = bool(
        not is_tie
        and mean_gain > NULL_MARGIN_M
        and mean_lcb > ADMISSION_LCB
        and pooled_median >= 0.0
        and harm_fraction <= MAX_HARM_FRACTION
    )
    return {
        "seed_count": len(SEEDS),
        "family_count": len(families),
        "nonzero_family_count": nonzero_families,
        "is_tie": is_tie,
        "null_margin_m": NULL_MARGIN_M,
        "family_mean_gain": mean_gain,
        "mean_lcb95_one_sided": mean_lcb,
        "family_median_gain": pooled_median,
        "family_median_ci95": [
            float(np.quantile(pooled_median_boot, 0.025)),
            float(np.quantile(pooled_median_boot, 0.975)),
        ],
        "seed_median_gain": float(np.median(seed_means)),
        "seed_median_ci95": [
            float(np.quantile(seed_median_boot, 0.025)),
            float(np.quantile(seed_median_boot, 0.975)),
        ],
        "harm_fraction": harm_fraction,
        "bootstrap_draws": draws,
        "bootstrap_seed": bootstrap_seed,
        "primary_ranking_metric": "bradley_terry_elo_family_grouped",
        "auc_reported": False,
        "per_seed": per_seed,
        "admitted": admitted,
    }


def score_fold_b_arms(
    pools_b: Mapping[str, dict],
    r882_preds_by_cid: Mapping[str, np.ndarray],
    arm_preds_by_cid: Mapping[str, Mapping[str, np.ndarray]],
    *,
    eval_mode: str = EVAL_MODE_UNREV,
) -> dict:
    """Score r882 and all four arms S0..S3 on fold B in the SAME leak-free eval mode (unrev 4..P-1)."""
    if eval_mode != EVAL_MODE_UNREV:
        raise ValueError("H61 admission/Elo scoring MUST use leak-free eval_mode='unrev' (4..P-1) for all arms")
    if set(arm_preds_by_cid) != set(ARMS):
        raise ValueError("H61 fold-B scoring requires exactly arms S0, S1, S2, S3")
    cids_b = tuple(sorted(pools_b))
    families_b = {cid: str(pools_b[cid]["rec"]["family_id"]) for cid in cids_b}

    baseline = evaluate_pool_predictions(cids_b, pools_b, r882_preds_by_cid, eval_mode=EVAL_MODE_UNREV)
    arms_out: dict[str, dict] = {}
    for arm in ARMS:
        cand = evaluate_pool_predictions(cids_b, pools_b, arm_preds_by_cid[arm], eval_mode=EVAL_MODE_UNREV)
        if any(idx < UNREVEALED_START_INDEX for idx in cand["chosen_indices"].values()):
            raise RuntimeError(f"revealed anchor 0..3 leaked into {arm} fold-B scoring")
        family_gains = compute_clipped_family_gains(
            cids_b, pools_b, cand["per_context"], baseline["per_context"]
        )
        context_gains = {
            cid: float(clip_context_gain(cand["per_context"][cid], baseline["per_context"][cid]))
            for cid in cids_b
        }
        wins = sum(v > 0.0 for v in family_gains.values())
        ties = sum(v == 0.0 for v in family_gains.values())
        losses = sum(v < 0.0 for v in family_gains.values())
        arms_out[arm] = {
            "eval_mode": EVAL_MODE_UNREV,
            "family_utility": cand["mean"],
            "context_utility": cand["context_mean"],
            "per_family": cand["per_family"],
            "per_context": cand["per_context"],
            "chosen_indices": cand["chosen_indices"],
            "family_gains": family_gains,
            "context_gains": context_gains,
            "w_t_l_family": {"wins": wins, "ties": ties, "losses": losses},
            "elo_vs_r882": elo_vs_r882(families_b, baseline["per_context"], cand["per_context"]),
        }

    return {
        "eval_mode": EVAL_MODE_UNREV,
        "baseline": baseline,
        "families": families_b,
        "fold_b_context_ids": list(cids_b),
        "fold_b_scoring_passes": 1 + len(ARMS),
        "holdout_payloads_opened": 0,
        "outer_rows_delivered": 0,
        "arms": arms_out,
    }


def admit_arms(scores_by_seed: Mapping[int, dict]) -> dict[str, dict]:
    """Apply the preregistered strict admission rule independently to every arm S0..S3 and summarize 3-seed Elo."""
    if tuple(sorted(scores_by_seed)) != SEEDS:
        raise ValueError("H61 fold-B decision requires completed seed scores 12/13/14")
    decisions = {}
    rng = np.random.default_rng(BOOTSTRAP_SEED + 99)
    for arm in ARMS:
        gains = {seed: scores_by_seed[seed]["arms"][arm]["family_gains"] for seed in SEEDS}
        adm = paired_family_admission(gains)
        seed_elos: list[float] = []
        family_win_by_seed: list[dict[str, float]] = []
        for seed in SEEDS:
            arm_entry = scores_by_seed[seed]["arms"][arm]
            elo_obj = arm_entry.get("elo_vs_r882")
            if isinstance(elo_obj, Mapping) and "ratings" in elo_obj:
                seed_elos.append(float(elo_obj["ratings"]["H61"]))
            if isinstance(elo_obj, Mapping) and "family_pairwise_win_scores" in elo_obj:
                family_win_by_seed.append(dict(elo_obj["family_pairwise_win_scores"]))
        if len(seed_elos) == len(SEEDS):
            elo_arr = np.asarray(seed_elos, dtype=np.float64)
            boot_idx = rng.integers(0, len(SEEDS), size=(BOOTSTRAP_DRAWS, len(SEEDS)))
            elo_boot = np.median(elo_arr[boot_idx], axis=1)
            adm["elo_by_seed"] = {str(s): float(e) for s, e in zip(SEEDS, seed_elos, strict=True)}
            adm["elo_median_across_seeds"] = float(np.median(elo_arr))
            adm["elo_ci95_across_seeds"] = [
                float(np.quantile(elo_boot, 0.025)),
                float(np.quantile(elo_boot, 0.975)),
            ]
        if len(family_win_by_seed) == len(SEEDS):
            fams = sorted(family_win_by_seed[0])
            pooled_fam_scores = np.asarray(
                [np.mean([family_win_by_seed[col][f] for col in range(len(SEEDS))]) for f in fams],
                dtype=np.float64,
            )
            f_idx = rng.integers(0, len(fams), size=(ELO_BOOTSTRAP_DRAWS, len(fams)))
            w_boot = pooled_fam_scores[f_idx].sum(axis=1) + ELO_PRIOR
            l_boot = (1.0 - pooled_fam_scores[f_idx]).sum(axis=1) + ELO_PRIOR
            gamma_boot = np.clip(w_boot / np.maximum(l_boot, 1e-12), 1e-6, 1e6)
            elo_f_boot = 1500.0 + 400.0 * np.log10(np.maximum(gamma_boot, 1e-9))
            adm["elo_family_bootstrap_median"] = float(np.median(elo_f_boot))
            adm["elo_family_bootstrap_ci95"] = [
                float(np.quantile(elo_f_boot, 0.025)),
                float(np.quantile(elo_f_boot, 0.975)),
            ]
        decisions[arm] = adm
    return decisions


def terminal_read_allowed(
    arm: str,
    seed: int,
    admitted_arms: Iterable[str],
    completed_reads: Iterable[tuple[str, int]],
) -> None:
    """Refuse every DL-Val read unless that arm passed admission, and refuse duplicates."""
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("unregistered H61 terminal arm/seed")
    if arm not in set(admitted_arms):
        raise RuntimeError("H61 terminal DL-Val is forbidden unless the arm is admitted")
    if (arm, seed) in set(completed_reads):
        raise RuntimeError("duplicate H61 terminal DL-Val read refused")


def plan_terminal_reads(admitted_arms: Iterable[str]) -> tuple[tuple[str, int], ...]:
    admitted_set = set(admitted_arms)
    admitted = tuple(arm for arm in ARMS if arm in admitted_set)
    return tuple((arm, seed) for arm in admitted for seed in SEEDS)


def elo_vs_r882(
    family_by_context: Mapping[str, str],
    baseline_utility: Mapping[str, float],
    candidate_utility: Mapping[str, float],
) -> dict:
    """Compute the preregistered family-grouped Bradley-Terry Elo vs r882 (never AUC) + family bootstrap 95% CI."""
    if set(family_by_context) != set(baseline_utility) or set(family_by_context) != set(candidate_utility):
        raise ValueError("H61 Elo panels must have identical context IDs")
    elo = import_module("h61_elo_tracker", ELO_PATH)
    records = [
        elo.DatasetEvalRecord(
            dataset_id=cid,
            family_id=family_by_context[cid],
            task_type="regression",
            scores_by_contender={
                "r882": float(baseline_utility[cid]),
                "H61": float(candidate_utility[cid]),
            },
        )
        for cid in sorted(family_by_context)
    ]
    result = elo.compute_family_grouped_bradley_terry_elo(
        records, anchor_contender="r882", anchor_elo=1500.0
    )
    per_family_scores: dict[str, list[float]] = {}
    for cid in sorted(family_by_context):
        diff = float(candidate_utility[cid]) - float(baseline_utility[cid])
        if diff > ELO_TIE_MARGIN:
            s_val = 1.0
        elif diff < -ELO_TIE_MARGIN:
            s_val = 0.0
        else:
            s_val = 0.5
        per_family_scores.setdefault(str(family_by_context[cid]), []).append(s_val)
    fam_scores = {f: float(np.mean(vals)) for f, vals in sorted(per_family_scores.items())}
    fams = sorted(fam_scores)
    s_arr = np.asarray([fam_scores[f] for f in fams], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED + 77)
    idx = rng.integers(0, len(s_arr), size=(ELO_BOOTSTRAP_DRAWS, len(s_arr)))
    w_boot = s_arr[idx].sum(axis=1) + ELO_PRIOR
    l_boot = (1.0 - s_arr[idx]).sum(axis=1) + ELO_PRIOR
    gamma_boot = np.clip(w_boot / np.maximum(l_boot, 1e-12), 1e-6, 1e6)
    elo_boot = 1500.0 + 400.0 * np.log10(np.maximum(gamma_boot, 1e-9))
    result["elo_family_bootstrap_median"] = float(np.median(elo_boot))
    result["elo_family_bootstrap_ci95"] = [
        float(np.quantile(elo_boot, 0.025)),
        float(np.quantile(elo_boot, 0.975)),
    ]
    result["family_pairwise_win_scores"] = fam_scores
    result["auc_reported"] = False
    return result


def project_gpu_budget_from_h57_timing(
    *,
    generation_seconds: float = H57_MEASURED_GENERATION_SECONDS,
    cell_overhead_seconds: float = H57_TIMING_CELL_OVERHEAD_SECONDS,
    seed_cells: int = len(SEEDS),
    units_per_seed: int = H61_EQUIVALENT_TIMING_UNITS_PER_SEED,
    max_terminal_reads: int = len(SEEDS),
    contingency_fraction: float = 0.10,
    cap_seconds: float = MAX_GPU_CELL_WALL_CAP_SECONDS,
) -> dict[str, float | int | bool]:
    """Project H61 GPU cell wall from H57 timing (60.1 s/gen, 128.6 s cell overhead) and verify <= 60 min cap."""
    per_seed_cell_wall = float(cell_overhead_seconds + units_per_seed * generation_seconds)
    training_cells_wall = float(seed_cells * per_seed_cell_wall)
    fold_b_reserve = float(cell_overhead_seconds)
    terminal_reserve = float(max_terminal_reads * cell_overhead_seconds)
    subtotal = training_cells_wall + fold_b_reserve + terminal_reserve
    contingency = subtotal * float(contingency_fraction)
    projected_total = subtotal + contingency
    fits_cap = bool(projected_total <= cap_seconds)
    if not fits_cap:
        raise RuntimeError(
            f"projected H61 GPU cell wall {projected_total:.3f}s exceeds cap {cap_seconds:.3f}s"
        )
    return {
        "h57_measured_generation_seconds": float(generation_seconds),
        "h57_cell_overhead_seconds": float(cell_overhead_seconds),
        "seed_cells": int(seed_cells),
        "units_per_seed": int(units_per_seed),
        "per_seed_cell_wall_seconds": round(per_seed_cell_wall, 3),
        "training_cells_wall_seconds": round(training_cells_wall, 3),
        "fold_b_reserve_seconds": round(fold_b_reserve, 3),
        "terminal_dl_val_reserve_seconds": round(terminal_reserve, 3),
        "subtotal_seconds": round(subtotal, 3),
        "contingency_seconds": round(contingency, 3),
        "projected_total_seconds": round(projected_total, 3),
        "cap_seconds": float(cap_seconds),
        "slack_seconds": round(float(cap_seconds - projected_total), 3),
        "fits_within_60_min_cap": fits_cap,
    }


def run_dry(args) -> dict:
    """CPU-only dry-run verification without reading outcome labels or fold-B/DL-Val."""
    torch.set_num_threads(1)
    ctx = prepare_training(
        args.input,
        args.checkpoint,
        torch.device("cpu"),
        scratch_dir=args.out.parent / "dry_scratch",
    )
    names = verify_cell_integrity(ctx)
    result = {
        "schema": "jope.h61.dry-run/v1",
        "status": "pass",
        "trainable_parameter_count": TRAINABLE_PARAMS,
        "trainable_parameter_names": list(names),
        "fold_seed": FOLD_SEED,
        "fold_assignment_sha256": ctx["fold_sha"],
        "fold_b_scored": False,
        "validation_payloads_opened": False,
        "holdout_payloads_opened": 0,
        "outer_rows_delivered": 0,
        "optimizer_updates": 0,
        "dope_sha256": ctx["h48"].h46.checkpoint_guard(ctx["guard"], ctx["net"], expected=DOPE_SHA),
    }
    scratch_dir = args.out.parent / "dry_scratch"
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir, ignore_errors=True)
    atomic_json(args.out, result)
    return result


def run_train_cell(args) -> dict:
    """Execute one (arm, seed) H61 cell under qualification.lock on cuda:0."""
    if args.device != "cuda:0" or not args.lock_held:
        raise RuntimeError("H61 timing/outcome cells require cuda:0 and qualification.lock")
    if args.arm not in ARMS or args.seed not in SEEDS:
        raise RuntimeError("unregistered H61 arm or seed")
    if args.mode == "timing" and (args.arm, args.seed) != ("S0", 12):
        raise RuntimeError("only S0/seed12 is supported for timing-only mode")
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.monotonic()
    ctx = prepare_training(
        args.input,
        args.checkpoint,
        torch.device(args.device),
        scratch_dir=args.out / "scratch_ckpt",
    )
    dope_before = ctx["h48"].h46.checkpoint_guard(ctx["guard"], ctx["net"], expected=DOPE_SHA)
    validate_dope_hash(dope_before)
    plan = build_paired_batches(
        sorted(ctx["pools_a"]),
        ctx["pools_a"],
        ctx["sorted_orders"],
        ctx["h53b"].cached_training_indices,
        seed=args.seed,
    )
    adapter_state, report = train_arm_with_round_selection(ctx, args.arm, args.seed, plan)
    dope_after = ctx["h48"].h46.checkpoint_guard(ctx["guard"], ctx["net"], expected=DOPE_SHA)
    validate_dope_hash(dope_after)
    if dope_before != DOPE_SHA or dope_after != DOPE_SHA:
        raise RuntimeError("H61 DOPE checksum changed across cell")
    wall = time.monotonic() - started
    h55 = ctx["h55"]
    payload = {
        "schema": "jope.h61.timing/v1" if args.mode == "timing" else "jope.h61.cell/v1",
        "status": "complete",
        "mode": args.mode,
        "device": "cuda:0",
        "wall_seconds": wall,
        "fold_seed": FOLD_SEED,
        "fold_assignment": ctx["fold_assignment"],
        "fold_a_family_count": len(ctx["fold_a_families"]),
        "fold_b_family_count": len(ctx["fold_b_families"]),
        "fold_a_context_count": len(ctx["a_rows"]),
        "fold_b_context_count": len(ctx["b_rows"]),
        "fold_a_b_family_disjoint": not bool(set(ctx["fold_a_families"]) & set(ctx["fold_b_families"])),
        "staged_r882_copy_path": ctx["staged_checkpoint_path"],
        "adapter_sha256": h55.adapter_digest(adapter_state),
        **report,
    }
    if args.mode == "outcome":
        check_writer_disk_budget()
        out_artifact = args.out / "candidate_adapter.pt"
        ctx["h48"].save_adapter_snapshot(out_artifact, adapter_state, R882_SHA)
        payload["candidate_adapter_artifact"] = str(out_artifact)
        payload["candidate_adapter_artifact_sha256"] = sha256(out_artifact)
    scratch_ckpt_dir = args.out / "scratch_ckpt"
    if scratch_ckpt_dir.exists():
        shutil.rmtree(scratch_ckpt_dir, ignore_errors=True)
    atomic_json(args.out / ("timing.json" if args.mode == "timing" else "summary.json"), payload)
    return payload


def run_fold_b_cell(args) -> dict:
    """Score all 12 completed H61 (arm, seed) candidates and r882 on untouched fold B in leak-free unrev mode."""
    if args.device != "cuda:0" or not args.lock_held:
        raise RuntimeError("H61 fold-B scoring requires cuda:0 and qualification.lock")
    torch.set_num_threads(1)
    scores_by_seed: dict[int, dict] = {}
    for seed in SEEDS:
        ctx = prepare_training(
            args.input,
            args.checkpoint,
            torch.device(args.device),
            scratch_dir=args.out / f"scratch_seed{seed}",
        )
        h48, h55, net = ctx["h48"], ctx["h55"], ctx["net"]
        dope_before = h48.h46.checkpoint_guard(ctx["guard"], net, expected=DOPE_SHA)
        validate_dope_hash(dope_before)
        if set(ctx["fold_a_families"]) & set(ctx["fold_b_families"]):
            raise RuntimeError("H61 fold A/B overlap before admission scoring")
        # Drop fold-A payloads/tables before loading fold B to keep RSS bounded.
        del ctx["predictor_a"], ctx["pools_a"], ctx["tables_a"], ctx["tokens_a"], ctx["masks_a"], ctx["sorted_orders"]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pools_b, tables_b = h55.load_rows(ctx["input_dir"], ctx["b_rows"], h48)
        device = next(net.parameters()).device
        tokens_b, masks_b = h48.h46.make_tokens(net, tables_b, device)
        predict_b = h48.make_predictor(net, pools_b, tokens_b, masks_b, device)
        cids_b = [r["context_sha256"] for r in ctx["b_rows"]]

        net.load_state_dict(ctx["checkpoint_state"]["state_dict"], strict=True)
        validate_dope_hash(h48.h46.checkpoint_guard(ctx["guard"], net, expected=DOPE_SHA))
        net.eval()
        r882_preds_b = predict_pool_scores(cids_b, pools_b, predict_b)

        arm_preds_b: dict[str, dict[str, np.ndarray]] = {}
        for arm in ARMS:
            ckpt_path = args.run_root / f"outcomes/cells/{arm}/seed{seed}/candidate_adapter.pt"
            if not ckpt_path.exists():
                raise RuntimeError(f"missing completed H61 candidate artifact: {ckpt_path}")
            artifact = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if (
                artifact.get("base_checkpoint_sha256") != R882_SHA
                or artifact.get("dope_sha256") != DOPE_SHA
                or artifact.get("trainable_params") != TRAINABLE_PARAMS
            ):
                raise RuntimeError(f"H61 candidate artifact pin mismatch at {ckpt_path}")
            h55.set_adapter(net, artifact["adapter_trainable_state"])
            validate_dope_hash(h48.h46.checkpoint_guard(ctx["guard"], net, expected=DOPE_SHA))
            net.eval()
            arm_preds_b[arm] = predict_pool_scores(cids_b, pools_b, predict_b)

        scored = score_fold_b_arms(pools_b, r882_preds_b, arm_preds_b, eval_mode=EVAL_MODE_UNREV)
        dope_after = h48.h46.checkpoint_guard(ctx["guard"], net, expected=DOPE_SHA)
        validate_dope_hash(dope_after)
        scored["dope_sha256_before"] = dope_before
        scored["dope_sha256_after"] = dope_after
        scores_by_seed[seed] = scored
        scratch_dir = args.out / f"scratch_seed{seed}"
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir, ignore_errors=True)

    decisions = admit_arms(scores_by_seed)
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    fold_b_payload = {
        "schema": "jope.h61.fold-b/v1",
        "status": "complete",
        "eval_mode": EVAL_MODE_UNREV,
        "fold_seed": FOLD_SEED,
        "r882_checkpoint_sha256": R882_SHA,
        "source_manifest_sha256": MANIFEST_SHA,
        "manifest_identity": MANIFEST_ID,
        "dope_sha256": DOPE_SHA,
        "scores_by_seed": {str(s): scores_by_seed[s] for s in SEEDS},
        "validation_payloads_opened": False,
        "holdout_payloads_opened": 0,
        "outer_rows_delivered": 0,
        "source_commit": source_commit,
    }
    admission_payload = {
        "schema": "jope.h61.admission/v1",
        "status": "complete",
        "eval_mode": EVAL_MODE_UNREV,
        "fold_seed": FOLD_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "null_margin_m": NULL_MARGIN_M,
        "admission_lcb_threshold": ADMISSION_LCB,
        "max_harm_fraction": MAX_HARM_FRACTION,
        "harm_threshold": HARM_THRESHOLD,
        "primary_ranking_metric": "bradley_terry_elo_family_grouped",
        "auc_reported": False,
        "arms": decisions,
        "admitted_arms": [arm for arm in ARMS if decisions[arm]["admitted"]],
        "planned_terminal_reads": [
            {"arm": arm, "seed": seed}
            for arm, seed in plan_terminal_reads(
                arm for arm in ARMS if decisions[arm]["admitted"]
            )
        ],
        "validation_payloads_opened": False,
        "holdout_payloads_opened": 0,
        "outer_rows_delivered": 0,
        "source_commit": source_commit,
    }
    atomic_json(args.run_root / "outcomes/fold_b.json", fold_b_payload)
    atomic_json(args.run_root / "outcomes/admission.json", admission_payload)
    return admission_payload


def run_terminal_val_cell(args) -> dict:
    """Execute at most one terminal DL-Val read for an admitted (arm, seed) in leak-free unrev mode."""
    if args.device != "cuda:0" or not args.lock_held:
        raise RuntimeError("H61 terminal DL-Val requires cuda:0 and qualification.lock")
    if args.arm not in ARMS or args.seed not in SEEDS:
        raise RuntimeError("unregistered H61 terminal arm/seed")
    if args.out.exists():
        raise RuntimeError(f"refusing duplicate H61 terminal DL-Val receipt: {args.out}")
    admission_path = args.run_root / "outcomes/admission.json"
    admission = json.loads(admission_path.read_text())
    admitted_arms = [
        arm for arm in ARMS if admission.get("arms", {}).get(arm, {}).get("admitted") is True
    ]
    completed_reads = [
        (json.loads(p.read_text())["arm"], int(json.loads(p.read_text())["seed"]))
        for p in (args.run_root / "outcomes/terminal-val").glob("*/*.json")
        if p.exists()
    ]
    terminal_read_allowed(args.arm, args.seed, admitted_arms, completed_reads)

    h55 = import_module("h61_h55_val", H55_PATH)
    h48 = import_module("h61_h48_val", H48_PATH)
    h48.CHECKPOINT_SHA = R882_SHA
    h49 = import_module("h61_h49_val", H49_SCORE_PATH)
    staged_ckpt = stage_r882_copy(args.checkpoint, args.out.parent / f"scratch_val_{args.arm}_{args.seed}")
    net, _, guard, _ = h55.build_model(args.input, staged_ckpt, torch.device(args.device), h48)
    rows, pools, tables = h49.load_validation_packet(args.input)
    if len(rows) != 91 or len({row["family_id"] for row in rows}) != 19:
        raise RuntimeError("H61 terminal DL-Val roster mismatch")
    candidate = args.run_root / f"outcomes/cells/{args.arm}/seed{args.seed}/candidate_adapter.pt"
    artifact = torch.load(candidate, map_location="cpu", weights_only=True)
    if (
        artifact.get("base_checkpoint_sha256") != R882_SHA
        or artifact.get("dope_sha256") != DOPE_SHA
        or artifact.get("trainable_params") != TRAINABLE_PARAMS
    ):
        raise RuntimeError("H61 terminal adapter artifact pin mismatch")
    before = h48.h46.checkpoint_guard(guard, net, expected=DOPE_SHA)
    validate_dope_hash(before)
    h55.set_adapter(net, artifact["adapter_trainable_state"])
    net.eval()
    tokens, masks = h48.h46.make_tokens(net, tables, torch.device(args.device))
    predictor = h48.make_predictor(net, pools, tokens, masks, torch.device(args.device))
    val_cids = [row["context_sha256"] for row in rows]
    preds_val = predict_pool_scores(val_cids, pools, predictor)
    result = evaluate_pool_predictions(val_cids, pools, preds_val, eval_mode=EVAL_MODE_UNREV)
    after = h48.h46.checkpoint_guard(guard, net, expected=DOPE_SHA)
    validate_dope_hash(after)
    scratch_val_dir = args.out.parent / f"scratch_val_{args.arm}_{args.seed}"
    if scratch_val_dir.exists():
        shutil.rmtree(scratch_val_dir, ignore_errors=True)
    payload = {
        "schema": "jope.h61.terminal-val/v1",
        "arm": args.arm,
        "seed": args.seed,
        "eval_mode": EVAL_MODE_UNREV,
        "terminal_dl_val_score_count": 1,
        "validation_payloads_opened": True,
        "family_utility": result["mean"],
        "context_utility": result["context_mean"],
        "family_delta_vs_r882_harness": result["mean"] - R882_HARNESS_UTILITY,
        "family_utilities": result["per_family"],
        "context_utilities": result["per_context"],
        "candidate_adapter_sha256": sha256(candidate),
        "dope_sha256_before": before,
        "dope_sha256_after": after,
        "holdout_payloads_opened": 0,
        "outer_rows_delivered": 0,
        "source_manifest_sha256": MANIFEST_SHA,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    atomic_json(args.out, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dry-run", "timing", "outcome", "fold-b", "terminal-val"), required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--lock-held", action="store_true")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "dry-run":
        result = run_dry(args)
    elif args.mode == "fold-b":
        result = run_fold_b_cell(args)
    elif args.mode == "terminal-val":
        if args.arm is None or args.seed is None:
            raise RuntimeError("H61 terminal DL-Val requires arm and seed")
        result = run_terminal_val_cell(args)
    else:
        if args.arm is None or args.seed is None:
            raise RuntimeError("H61 timing/outcome needs a registered arm and seed")
        if args.out.exists():
            raise RuntimeError(f"refusing to overwrite H61 receipt {args.out}")
        args.out.mkdir(parents=True, exist_ok=False)
        result = run_train_cell(args)
    print(json.dumps({"mode": args.mode, "result": result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
