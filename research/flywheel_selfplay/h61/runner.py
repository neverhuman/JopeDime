#!/usr/bin/env python3
"""H61 selection-metric and strict-margin admission runner (CPU prep & fixture-tested; GPU execution not invoked here)."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
H55_PATH = ROOT / "research/flywheel_selfplay/h55/runner.py"
ELO_PATH = ROOT / "python/src/jope/north_star/elo_tracker.py"

# Pinned H55 / H59 / H61 contract constants
DOPE_SHA = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
R882_SHA = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
MANIFEST_SHA = "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
MANIFEST_ID = "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3"
TRAINABLE_PARAMS = 29_211

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

# Inner-CV hyperparameter grid for S2 (evaluated strictly on DL-Train fold_a inner folds)
TAU_GRID = (0.02, 0.05, 0.10, 0.20, 0.50)
M_S2_GRID = (NULL_MARGIN_M, 0.002000, 0.002500)

# Disk and resource guards (xbabe2 swap is full; start cells only when statvfs < 94.5%)
DISK_START_MAX_USED_PERCENT = 94.5
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
            f"disk guard blocked at {stage}: used={usage['used_percent']:.3f}% "
            f"(limit <{max_used_percent:.1f}%), avail={usage['available_bytes']} B",
            stage=stage,
            used_percent=float(usage["used_percent"]),
            available_bytes=int(usage["available_bytes"]),
        )
    return usage


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
            f"write-time OS error ({exc}) treated as retryable infra-void",
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
    """Run the 20-round plan, checking DOPE before/after every round. Not invoked during CPU-only prep."""
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

    grad_norms: list[float] = []
    dope_checks_by_round: list[dict] = []
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

    final_dope = h48.h46.checkpoint_guard(ctx["guard"], net, expected=DOPE_SHA)
    validate_dope_hash(final_dope)
    h55 = ctx.get("h55") or import_module("h61_h55_runner", H55_PATH)
    return h55.adapter_state(net), {
        "arm": arm,
        "seed": seed,
        "rounds_completed": ROUNDS,
        "optimizer_updates_completed": len(plan),
        "paired_streams": [[b.round_index, b.step_index, b.seed, b.candidate_pool_sha256] for b in plan],
        "trainable_parameter_names": list(names),
        "trainable_parameter_count": TRAINABLE_PARAMS,
        "dope_checks_by_round": dope_checks_by_round,
        "dope_sha256_after": final_dope,
        "grad_norms": grad_norms,
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
        arms_out[arm] = {
            "eval_mode": EVAL_MODE_UNREV,
            "family_utility": cand["mean"],
            "context_utility": cand["context_mean"],
            "per_family": cand["per_family"],
            "per_context": cand["per_context"],
            "chosen_indices": cand["chosen_indices"],
            "family_gains": family_gains,
            "context_gains": context_gains,
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
    """Apply the preregistered strict admission rule independently to every arm S0..S3."""
    if tuple(sorted(scores_by_seed)) != SEEDS:
        raise ValueError("H61 fold-B decision requires completed seed scores 12/13/14")
    decisions = {}
    for arm in ARMS:
        gains = {seed: scores_by_seed[seed]["arms"][arm]["family_gains"] for seed in SEEDS}
        decisions[arm] = paired_family_admission(gains)
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
    """Compute the preregistered family-grouped Bradley-Terry Elo vs r882 (never AUC)."""
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
    return elo.compute_family_grouped_bradley_terry_elo(
        records, anchor_contender="r882", anchor_elo=1500.0
    )


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
