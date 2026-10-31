#!/usr/bin/env python3
"""H59 paired loss-ablation runner; execution is intentionally not invoked here."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import random
import sys
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
H55_PATH = ROOT / "research/flywheel_selfplay/h55/runner.py"
ELO_PATH = ROOT / "python/src/jope/north_star/elo_tracker.py"
DOPE_SHA = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
R882_SHA = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
MANIFEST_SHA = "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
MANIFEST_ID = "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3"
TRAINABLE_PARAMS = 29_211
SEEDS = (12, 13, 14)
ARMS = ("CONTROL", "CLIP", "NORM", "CLIP+NORM")
FOLD_SEED = 4900
ROUNDS = 20
STEPS_PER_ROUND = 2
BATCH_SIZE = 256
SLATE_SIZE = 192
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
ROBUST_CAP = 1.0
SCALE_FLOOR = 0.05
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 59_000
ADMISSION_LCB = 0.001
MAX_HARM_FRACTION = 0.10
HARM_THRESHOLD = -0.01


def prepare_training(input_dir: Path, checkpoint: Path, device: torch.device) -> dict:
    """Reuse H55's pinned metadata, fold, checkpoint and fold-A loader contract."""
    h55 = import_module("h59_h55_runner", H55_PATH)
    if (h55.R882_SHA, h55.DOPE_SHA, h55.MANIFEST_SHA, h55.MANIFEST_ID) != (
        R882_SHA, DOPE_SHA, MANIFEST_SHA, MANIFEST_ID
    ):
        raise RuntimeError("H59/H55 pinned inputs differ")
    ctx = h55.prepare_training(input_dir, checkpoint, device)
    if ctx["fold_assignment"].get("seed") != FOLD_SEED:
        raise RuntimeError("H59 requires the H55 seed-4900 family fold")
    ctx["h55"] = h55
    return ctx


@dataclass(frozen=True)
class TrainingBatch:
    """A deterministic, arm-independent batch and slate selection."""

    round_index: int
    step_index: int
    seed: int
    context_ids: tuple[str, ...]
    candidate_indices: tuple[np.ndarray, ...]


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


def paired_stream_seed(seed: int, round_index: int, step_index: int) -> int:
    """One fixed stream per (seed, round, step), with no arm-dependent salt."""
    if seed not in SEEDS or not 1 <= round_index <= ROUNDS or not 0 <= step_index < STEPS_PER_ROUND:
        raise ValueError("unregistered H59 seed/round/step stream")
    sampler_key = seed * 1_000_003 + 10_007
    payload = f"flywheel-batch/v1:{sampler_key}:{round_index}:{step_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def build_paired_batches(
    train_ids: Iterable[str],
    pools: Mapping[str, dict],
    sorted_orders: Mapping[str, np.ndarray],
    cached_training_indices: Callable,
    *,
    seed: int,
    batch_seed: int | None = None,
) -> tuple[TrainingBatch, ...]:
    """Freeze every context and slate draw once so all arms replay identical inputs."""
    if seed not in SEEDS:
        raise ValueError("unregistered H59 seed")
    ids = tuple(sorted(train_ids))
    if not ids or set(ids) - pools.keys() or set(ids) - sorted_orders.keys():
        raise ValueError("H59 batch plan requires a nonempty fold-A-only roster")
    make_batch_seed = batch_seed or paired_stream_seed
    sampler_key = seed * 1_000_003 + 10_007
    batches: list[TrainingBatch] = []
    rng_state = np.random.get_state()
    try:
        for round_index in range(1, ROUNDS + 1):
            # Match H55's H53b cached sampler stream: one round stream, consumed
            # continuously across the contexts in both optimizer steps.
            np.random.seed((sampler_key + round_index * 104729) % (2**32))
            rotation = round_index % len(ids)
            round_ids = ids[rotation:] + ids[:rotation]
            for step_index in range(STEPS_PER_ROUND):
                stream = make_batch_seed(seed, round_index, step_index)
                cids = tuple(random.Random(stream).choices(round_ids, k=BATCH_SIZE))
                indexes = tuple(
                    np.asarray(cached_training_indices(pools[cid], sorted_orders[cid], SLATE_SIZE), dtype=np.int64).copy()
                    for cid in cids
                )
                if any(index.shape != (SLATE_SIZE,) for index in indexes):
                    raise RuntimeError("H59 sampler returned a slate with the wrong width")
                batches.append(TrainingBatch(round_index, step_index, stream, cids, indexes))
    finally:
        np.random.set_state(rng_state)
    if len(batches) != ROUNDS * STEPS_PER_ROUND:
        raise RuntimeError("H59 fixed round/step plan is incomplete")
    return tuple(batches)


def paired_plans_for_arms(plan: tuple[TrainingBatch, ...]) -> dict[str, tuple[TrainingBatch, ...]]:
    """Bind each arm to the very same immutable plan object."""
    if len(plan) != ROUNDS * STEPS_PER_ROUND:
        raise ValueError("H59 paired plan has the wrong number of updates")
    return {arm: plan for arm in ARMS}


def preregistered_targets(fitness: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    """Map cached cost fitness to t=-clip(u,-2,4); failed gates retain cost 2."""
    if fitness.shape != gates.shape:
        raise ValueError("fitness and gate tensors must have matching slate shapes")
    utility = -fitness
    targets = -torch.clamp(utility, min=-2.0, max=4.0)
    return torch.where(gates.to(dtype=torch.bool), targets, torch.full_like(targets, 2.0))


def training_slate_scale(targets: torch.Tensor) -> torch.Tensor:
    """Detached MAD scale computed only from the current training slate targets."""
    if targets.ndim != 2 or targets.shape[1] == 0:
        raise ValueError("H59 scale requires [batch, training_slate] targets")
    with torch.no_grad():
        center = targets.detach().median(dim=1).values
        mad = (targets.detach() - center[:, None]).abs().median(dim=1).values
        return (1.4826 * mad).clamp_min(SCALE_FLOOR).detach()


def loss_by_arm(arm: str, prediction: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Exact H59 objective: current Smooth-L1 + KL, optional cap, then MAD scaling."""
    if arm not in ARMS or prediction.shape != targets.shape or prediction.ndim != 2:
        raise ValueError("H59 loss expects a registered arm and matching [batch, slate] tensors")
    smooth = F.smooth_l1_loss(prediction, targets, reduction="none", beta=1.0)
    if arm in ("CLIP", "CLIP+NORM"):
        smooth = smooth.clamp(max=ROBUST_CAP)
    row_smooth = smooth.mean(dim=1)
    row_kl = F.kl_div(
        F.log_softmax(-prediction / 0.2, dim=-1),
        F.softmax(-targets / 0.2, dim=-1),
        reduction="none",
    ).sum(dim=1)
    per_context = row_smooth + 0.1 * row_kl
    if arm in ("NORM", "CLIP+NORM"):
        per_context = per_context / training_slate_scale(targets)
    return per_context.mean()


def validate_trainable_inventory(named_parameters: Iterable[tuple[str, object]]) -> tuple[str, ...]:
    parameters = tuple(named_parameters)
    trainable = tuple((name, parameter) for name, parameter in parameters if parameter.requires_grad)
    names = tuple(sorted(name for name, _ in trainable))
    count = sum(parameter.numel() for _, parameter in trainable)
    if count != TRAINABLE_PARAMS:
        raise RuntimeError(f"H59 expected {TRAINABLE_PARAMS} trainable parameters, got {count}")
    if not names or any(
        not name.startswith("adapter_10tok.") or ".anchor." in name or name.endswith(".anchor")
        for name in names
    ):
        raise RuntimeError("H59 may train only non-anchor adapter_10tok parameters")
    return names


def validate_dope_hash(actual: str) -> None:
    if actual != DOPE_SHA:
        raise RuntimeError(f"H59 frozen DOPE SHA mismatch: {actual}")


def verify_cell_integrity(ctx: dict) -> tuple[str, ...]:
    names = validate_trainable_inventory(ctx["net"].named_parameters())
    actual = ctx["h48"].h46.checkpoint_guard(ctx["guard"], ctx["net"], expected=DOPE_SHA)
    validate_dope_hash(actual)
    return names


def _batch_targets(pools: Mapping[str, dict], batch: TrainingBatch, device: torch.device) -> torch.Tensor:
    fitness = np.stack([pools[cid]["fitness"][ix] for cid, ix in zip(batch.context_ids, batch.candidate_indices, strict=True)])
    gates = np.stack([pools[cid]["gate"][ix] for cid, ix in zip(batch.context_ids, batch.candidate_indices, strict=True)])
    return preregistered_targets(
        torch.as_tensor(fitness, dtype=torch.float32, device=device),
        torch.as_tensor(gates, dtype=torch.bool, device=device),
    )


def train_arm(ctx: dict, arm: str, seed: int, plan: tuple[TrainingBatch, ...]) -> tuple[dict, dict]:
    """Run a fixed 40-update arm. This function is not called by CPU-only prep/tests."""
    if arm not in ARMS or seed not in SEEDS or len(plan) != ROUNDS * STEPS_PER_ROUND:
        raise ValueError("invalid H59 arm, seed or fixed update plan")
    expected_streams = tuple(
        paired_stream_seed(seed, round_index, step_index)
        for round_index in range(1, ROUNDS + 1)
        for step_index in range(STEPS_PER_ROUND)
    )
    if tuple(batch.seed for batch in plan) != expected_streams:
        raise ValueError("H59 arm received a batch plan for a different seed or stream order")
    net, h48 = ctx["net"], ctx["h48"]
    net.load_state_dict(ctx["checkpoint_state"]["state_dict"], strict=True)
    names = verify_cell_integrity(ctx)
    trainable = [parameter for _, parameter in net.named_parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    h48.base.guard.assert_optimizer_excludes_dope(optimizer, net)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=ROUNDS * STEPS_PER_ROUND, eta_min=MIN_LEARNING_RATE
    )
    grad_norms: list[float] = []
    for batch in plan:
        cids = list(batch.context_ids)
        indexes = list(batch.candidate_indices)
        prediction = ctx["predictor_a"](cids, indexes)
        targets = _batch_targets(ctx["pools_a"], batch, prediction.device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_by_arm(arm, prediction, targets)
        loss.backward()
        grad_norms.append(float(torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)))
        optimizer.step()
        scheduler.step()
        verify_cell_integrity(ctx)
    final_dope = ctx["h48"].h46.checkpoint_guard(ctx["guard"], net, expected=DOPE_SHA)
    validate_dope_hash(final_dope)
    h55 = ctx.get("h55") or import_module("h59_h55_runner", H55_PATH)
    return h55.adapter_state(net), {
        "arm": arm,
        "seed": seed,
        "rounds_completed": ROUNDS,
        "optimizer_updates_completed": len(plan),
        "paired_streams": [[b.round_index, b.step_index, b.seed] for b in plan],
        "trainable_parameter_names": list(names),
        "trainable_parameter_count": TRAINABLE_PARAMS,
        "dope_sha256_after": final_dope,
        "grad_norms": grad_norms,
        "r882_checkpoint_sha256": R882_SHA,
        "source_manifest_sha256": MANIFEST_SHA,
        "manifest_identity": MANIFEST_ID,
        "fold_assignment_sha256": ctx["fold_sha"],
        "source_sha256": ctx["source_sha256"],
        "source_commit": ctx["source_commit"],
    }


def paired_family_admission(
    gains_by_seed: Mapping[int, Mapping[str, float]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Family-cluster admission with pooled and per-seed family-median intervals."""
    if tuple(sorted(gains_by_seed)) != SEEDS or draws != BOOTSTRAP_DRAWS:
        raise ValueError("H59 admission requires seeds 12/13/14 and exactly 10,000 draws")
    family_sets = [set(gains_by_seed[seed]) for seed in SEEDS]
    if not family_sets[0] or any(families != family_sets[0] for families in family_sets[1:]):
        raise ValueError("H59 paired family inventory differs across seeds")
    families = tuple(sorted(family_sets[0]))
    matrix = np.asarray(
        [[float(gains_by_seed[seed][family]) for seed in SEEDS] for family in families],
        dtype=np.float64,
    )
    if not np.isfinite(matrix).all():
        raise ValueError("H59 family gains must be finite")
    family_mean = matrix.mean(axis=1)
    family_seed_median = np.median(matrix, axis=1)
    rng = np.random.default_rng(bootstrap_seed)
    indexes = rng.integers(0, len(families), size=(draws, len(families)))
    mean_boot = family_mean[indexes].mean(axis=1)
    pooled_median_boot = np.median(family_seed_median[indexes], axis=1)
    mean_lcb = float(np.quantile(mean_boot, 0.05))
    pooled_median = float(np.median(family_seed_median))
    harm_fraction = float(np.mean(family_seed_median < HARM_THRESHOLD))
    per_seed = {}
    for column, seed in enumerate(SEEDS):
        values = matrix[:, column]
        boot = np.median(values[indexes], axis=1)
        per_seed[str(seed)] = {
            "family_mean_gain": float(values.mean()),
            "family_median_gain": float(np.median(values)),
            "family_median_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        }
    return {
        "seed_count": len(SEEDS),
        "family_count": len(families),
        "family_mean_gain": float(family_mean.mean()),
        "mean_lcb95_one_sided": mean_lcb,
        "family_median_gain": pooled_median,
        "family_median_ci95": [float(np.quantile(pooled_median_boot, 0.025)), float(np.quantile(pooled_median_boot, 0.975))],
        "harm_fraction": harm_fraction,
        "bootstrap_draws": draws,
        "bootstrap_seed": bootstrap_seed,
        "per_seed": per_seed,
        "admitted": mean_lcb > ADMISSION_LCB and pooled_median >= 0.0 and harm_fraction <= MAX_HARM_FRACTION,
    }


def score_fold_b_arms(ctx: dict, states_by_arm: Mapping[str, dict[str, torch.Tensor]]) -> dict:
    """Score every final arm and r882 once on B; the fixed order cannot select an arm."""
    if set(states_by_arm) != set(ARMS):
        raise ValueError("H59 fold-B admission requires exactly the four preregistered arms")
    h55 = ctx.get("h55") or import_module("h59_h55_runner", H55_PATH)
    candidates = [states_by_arm[arm] for arm in ARMS[:-1]]
    scored = h55.score_fold_b(ctx, candidates, states_by_arm[ARMS[-1]])
    candidate_results = dict(zip(ARMS[:-1], scored["candidate_controls"], strict=True))
    candidate_results[ARMS[-1]] = scored["admission"]
    baseline = scored["baseline"]
    result = {
        "baseline": baseline,
        "families": scored["families"],
        "fold_b_context_ids": scored["fold_b_context_ids"],
        "fold_b_scoring_passes": scored["fold_b_scoring_passes"],
        "holdout_payloads_opened": scored["holdout_payloads_opened"],
        "outer_rows_delivered": scored["outer_rows_delivered"],
        "arms": {},
    }
    for arm, candidate in candidate_results.items():
        if set(candidate["per_family"]) != set(baseline["per_family"]):
            raise RuntimeError(f"H59 {arm} fold-B family inventory differs from r882")
        if set(candidate["per_context"]) != set(baseline["per_context"]):
            raise RuntimeError(f"H59 {arm} fold-B context inventory differs from r882")
        family_gains = {
            family: float(candidate["per_family"][family] - baseline["per_family"][family])
            for family in sorted(baseline["per_family"])
        }
        context_gains = {
            context: float(candidate["per_context"][context] - baseline["per_context"][context])
            for context in sorted(baseline["per_context"])
        }
        result["arms"][arm] = {
            "family_gains": family_gains,
            "context_gains": context_gains,
            "elo_vs_r882": elo_vs_r882(scored["families"], baseline["per_context"], candidate["per_context"]),
        }
    return result


def admit_arms(scores_by_seed: Mapping[int, dict]) -> dict[str, dict]:
    """Apply the preregistered admission rule independently to every arm."""
    if tuple(sorted(scores_by_seed)) != SEEDS:
        raise ValueError("H59 fold-B decision requires completed seed scores 12/13/14")
    decisions = {}
    for arm in ARMS:
        gains = {
            seed: scores_by_seed[seed]["arms"][arm]["family_gains"]
            for seed in SEEDS
        }
        decisions[arm] = paired_family_admission(gains)
    return decisions


def terminal_read_allowed(arm: str, seed: int, admitted_arms: Iterable[str], completed_reads: Iterable[tuple[str, int]]) -> None:
    """Refuse every DL-Val read unless that arm passed admission, and refuse duplicates."""
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("unregistered H59 terminal arm/seed")
    if arm not in set(admitted_arms):
        raise RuntimeError("H59 terminal DL-Val is forbidden unless the arm is admitted")
    if (arm, seed) in set(completed_reads):
        raise RuntimeError("duplicate H59 terminal DL-Val read refused")


def plan_terminal_reads(admitted_arms: Iterable[str]) -> tuple[tuple[str, int], ...]:
    admitted_set = set(admitted_arms)
    admitted = tuple(arm for arm in ARMS if arm in admitted_set)
    return tuple((arm, seed) for arm in admitted for seed in SEEDS)


def elo_vs_r882(
    family_by_context: Mapping[str, str],
    baseline_utility: Mapping[str, float],
    candidate_utility: Mapping[str, float],
) -> dict:
    """Compute the preregistered family-grouped Bradley-Terry Elo diagnostic."""
    if set(family_by_context) != set(baseline_utility) or set(family_by_context) != set(candidate_utility):
        raise ValueError("H59 Elo panels must have identical context IDs")
    elo = import_module("h59_elo_tracker", ELO_PATH)
    records = [
        elo.DatasetEvalRecord(
            dataset_id=cid,
            family_id=family_by_context[cid],
            task_type="regression",
            scores_by_contender={"r882": float(baseline_utility[cid]), "H59": float(candidate_utility[cid])},
        )
        for cid in sorted(family_by_context)
    ]
    return elo.compute_family_grouped_bradley_terry_elo(
        records, anchor_contender="r882", anchor_elo=1500.0
    )
