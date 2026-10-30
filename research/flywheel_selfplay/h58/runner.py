#!/usr/bin/env python3
"""
H58 Flywheel Self-Play Multi-Arm Wave Runner (t550)
===================================================
Implements four cumulative arms (each an isolated copy of the flywheel loop):
  - Arm A (CONTROL): Current flywheel harness (revealed 0..3 unmasked in loss & argmin,
    deterministic order[:44] in slots 0..43, zeroed ctx_feat/base_prior_cost/sos_prior,
    constant scale=0.4208, parent_delta_net trainable [29,211 params], HWM-only admission,
    smooth_l1 + 0.1*kl(tau=0.2) loss).
  - Arm B (FIX1): Fix Bug 1 (fixed unsorted seed anchors 0..3, permuted unrevealed slots 4..191,
    mask 0..3 from training loss and evaluation argmin) + Fix Bug 3 (real 584D ctx_feat,
    real prior10 base_prior_cost, real 2D sos_prior, per-context scale) + Fix Bug 4
    (freeze dead parent_delta_net -> 23,546 active trainable params).
  - Arm C (FIX1+2): Arm B + Fix Bug 2 (admit all evaluated GP candidates on training folds,
    including non-HWM evaluations, model-proposed false positives, and gate failures, as hard
    negatives in pools[c] and reserve hard-negative slots in training slates).
  - Arm D (FIX1+2+3): Arm C + Fix Rank 3 (shift-invariant listwise top-focused loss:
    dual-temperature Plackett-Luce / ListNet cross-entropy + pairwise gate-failure & top-choice
    hinge margins on unrevealed candidates 4..191).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/flywheel_selfplay"))
sys.path.insert(0, str(ROOT / "python/src"))
import run_fused_closed_loop as base  # noqa: E402
from jope.north_star.elo_tracker import (  # noqa: E402
    DatasetEvalRecord,
    compute_family_grouped_bradley_terry_elo,
)

MANIFEST_FILE_SHA = "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
MANIFEST_ID = "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3"
DOPE_SHA = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
FOLD_SEED = 4900
DEFAULT_SEEDS = (12, 13, 14)
ARMS = ("A", "B", "C", "D")
TRAINABLE_PARAMS_A = 29_211
TRAINABLE_PARAMS_BCD = 23_546
BOOTSTRAP_DRAWS = 10_000
GATE_MARGIN_LCB95 = 0.001


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".new")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def adapter_sha(net: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, param in sorted(net.adapter_10tok.named_parameters()):
        if not name.startswith("anchor."):
            h.update(name.encode("utf-8"))
            h.update(param.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def verify_dope_guard(net: torch.nn.Module, arm: str, expected_dope: str = DOPE_SHA) -> str:
    snap = base.guard.snapshot_dope_weights(net)
    if snap.composite_sha256 != expected_dope:
        raise RuntimeError(f"DOPE guard failed ({arm}): {snap.composite_sha256} != {expected_dope}")
    counts = base.guard.get_trainable_param_counts_by_submodule(net)
    expected_params = TRAINABLE_PARAMS_A if arm == "A" else TRAINABLE_PARAMS_BCD
    total_tr = counts.get("__global_total__", {}).get("trainable_params")
    adap_tr = counts.get("adapter_10tok", {}).get("trainable_params")
    if total_tr != expected_params or adap_tr != expected_params:
        raise RuntimeError(
            f"trainable param guard failed for arm {arm}: expected {expected_params}, got {counts}"
        )
    return snap.composite_sha256


def reset_adapter_fold_clean(
    net: torch.nn.Module,
    seed: int,
    *,
    freeze_parent_delta: bool,
    scale: float = 0.1,
) -> list[torch.nn.Parameter]:
    """
    Deterministic fold-clean initialization of `adapter_10tok` with zero `r882` weight transfer.
    Uses `init_seed = 10000 + seed` so Arms A, B, C, D start from the exact same active adapter
    weights for any given seed.
    """
    init_seed = 10_000 + int(seed)
    old = net.adapter_10tok
    cls = type(old)
    device = next(old.parameters()).device
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    fresh = cls(
        frozen_anchor=old.anchor,
        prog_mean=old.prog_mean.detach().cpu().numpy(),
        prog_std=old.prog_std.detach().cpu().numpy(),
        d_token=32,
        n_modalities=10,
        modality_dropout_p=0.10,
    ).to(device)
    for method_name in (
        "encode_multimodal_context_and_candidates",
        "predict_parent_child_delta",
    ):
        bound = getattr(old, method_name)
        setattr(fresh, method_name, MethodType(bound.__func__, fresh))
    net.adapter_10tok = fresh

    for param in net.parameters():
        param.requires_grad_(False)
    for name, param in net.adapter_10tok.named_parameters():
        if name.startswith("anchor."):
            param.requires_grad_(False)
            continue
        if scale != 1.0:
            with torch.no_grad():
                param.mul_(scale)
        if freeze_parent_delta and name.startswith("parent_delta_net."):
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

    base.guard.lock_dope_weights_and_eval_mode(net)
    trainable = [p for p in net.parameters() if p.requires_grad]
    expected = TRAINABLE_PARAMS_BCD if freeze_parent_delta else TRAINABLE_PARAMS_A
    actual = sum(p.numel() for p in trainable)
    if actual != expected:
        raise RuntimeError(f"trainable parameter count mismatch: {actual} != {expected}")
    if any(
        name.startswith("anchor.") and p.requires_grad
        for name, p in net.adapter_10tok.named_parameters()
    ):
        raise RuntimeError("frozen anchor became trainable")
    if freeze_parent_delta and any(
        name.startswith("parent_delta_net.") and p.requires_grad
        for name, p in net.adapter_10tok.named_parameters()
    ):
        raise RuntimeError("parent_delta_net remained trainable in B/C/D")
    return trainable


def clone_pools(base_pools: dict[str, dict], cids: list[str] | None = None) -> dict[str, dict]:
    """
    Create an isolated copy-on-write pool view so per-round candidate admissions never cross
    arms/folds while avoiding redundant multi-gigabyte array copies for unmutated contexts.
    `admit_gp_candidate_to_pool` replaces arrays with new `np.vstack`/`np.append` allocations.
    """
    target_cids = cids if cids is not None else list(base_pools.keys())
    out = {}
    for cid in target_cids:
        src = base_pools[cid]
        out[cid] = {
            "rec": src["rec"],
            "pids": list(src["pids"]),
            "program": src["program"],
            "proj16": src["proj16"],
            "fitness": src["fitness"],
            "logratio": src["logratio"],
            "gate": src["gate"],
            "base_prior_cost": src["base_prior_cost"],
            "sos_prior": src["sos_prior"],
            "scale": float(src["scale"]),
            "hard_negatives": set(src.get("hard_negatives", set())),
        }
    return out


def make_predictor_arm(
    net: torch.nn.Module,
    pools: dict[str, dict],
    ctx_cache: dict[str, dict[str, torch.Tensor]],
    device: torch.device,
    arm: str,
):
    """
    Build predictor for Arm A vs Arms B/C/D.
      - Arm A: zeroed ctx_feat/base_prior_cost/sos_prior, constant scale=0.4208.
      - Arms B/C/D: real 584D ctx_feat, real prior10 base_prior_cost, real 2D sos_prior,
        real per-context scale = max(std(fitness[:4]), 0.10).
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")

    def predict(cids: list[str], indexes: list[np.ndarray]) -> torch.Tensor:
        progs = np.stack([pools[cid]["program"][ix] for cid, ix in zip(cids, indexes, strict=True)])
        proj = np.stack([pools[cid]["proj16"][ix] for cid, ix in zip(cids, indexes, strict=True)])
        batch, slate = progs.shape[:2]
        n_rev = min(4, slate)
        histf = torch.as_tensor(
            np.stack([pools[cid]["fitness"][ix[:n_rev]] for cid, ix in zip(cids, indexes, strict=True)]),
            dtype=torch.float32,
            device=device,
        )
        histg = torch.as_tensor(
            np.stack([pools[cid]["gate"][ix[:n_rev]] for cid, ix in zip(cids, indexes, strict=True)]),
            dtype=torch.float32,
            device=device,
        )
        tokens_b = torch.stack([ctx_cache[cid]["tokens_10x32"] for cid in cids])
        masks_b = torch.stack([ctx_cache[cid]["mask_10"] for cid in cids])
        rev_idx = torch.arange(n_rev, device=device).reshape(1, -1).expand(batch, -1)

        if arm == "A":
            ctx_feat_t = torch.zeros((batch, 584), dtype=torch.float32, device=device)
            base_prior_t = torch.zeros((batch, slate), dtype=torch.float32, device=device)
            sos_prior_t = torch.zeros((batch, slate, 2), dtype=torch.float32, device=device)
            scale_t = torch.full((batch,), 0.4208, dtype=torch.float32, device=device)
        else:
            ctx_feat_t = torch.stack([ctx_cache[cid]["ctx_feat_584"] for cid in cids])
            base_prior_np = np.stack(
                [pools[cid]["base_prior_cost"][ix] for cid, ix in zip(cids, indexes, strict=True)]
            )
            sos_prior_np = np.stack(
                [pools[cid]["sos_prior"][ix] for cid, ix in zip(cids, indexes, strict=True)]
            )
            scales_np = np.asarray([pools[cid]["scale"] for cid in cids], dtype=np.float32)
            base_prior_t = torch.as_tensor(base_prior_np, dtype=torch.float32, device=device)
            sos_prior_t = torch.as_tensor(sos_prior_np, dtype=torch.float32, device=device)
            scale_t = torch.as_tensor(scales_np, dtype=torch.float32, device=device)

        return net.adapter_10tok.predict_costs(
            ctx_feat=ctx_feat_t,
            prog_412=torch.as_tensor(progs, dtype=torch.float32, device=device),
            proj_16=torch.as_tensor(proj, dtype=torch.float32, device=device),
            base_prior_cost=base_prior_t,
            scale=scale_t,
            sos_prior=sos_prior_t,
            mm_tokens_10x32=tokens_b,
            mm_mask_10=masks_b,
            revealed_indices=rev_idx,
            revealed_raw_f=histf,
            revealed_gate=histg,
        )["total_cost"].float()

    return predict


def sample_slate_arm(
    pool: dict,
    slate_size: int,
    rng: np.random.Generator,
    arm: str,
) -> np.ndarray:
    """
    Construct a training slate of size `slate_size` (default 192):
      - Arm A (`CONTROL`): matches `run_fused_closed_loop.py:training_indices`:
        places `order[:44]` first (so `order[0..3]` land at revealed anchor positions `0..3`!)
        and does not permute slots.
      - Arm B (`FIX1`): fixes slots `0..3` to the 4 unsorted seed anchors `[0, 1, 2, 3]`,
        samples top-44 passing from `idx >= 4` + random remaining from `idx >= 4`,
        and randomly permutes all 188 unrevealed slots `4..191`.
      - Arms C & D (`FIX1+2` / `FIX1+2+3`): fixes slots `0..3` to `[0, 1, 2, 3]`,
        samples up to 44 top passing from `idx >= 4`, up to 36 evaluated hard negatives
        (gate failures, non-HWM GP candidates, and model-proposed false positives) from `idx >= 4`,
        fills the remainder uniformly from `idx >= 4`, and randomly permutes slots `4..191`.
    """
    total = len(pool["pids"])
    if total <= 4:
        raise ValueError(f"pool must have > 4 candidates, got {total}")
    target_len = min(slate_size, total)

    if arm == "A":
        passed = np.flatnonzero(pool["gate"])
        order = (
            passed[np.argsort(pool["fitness"][passed])]
            if len(passed)
            else np.argsort(pool["fitness"])
        )
        top = order[: min(44, len(order))]
        remain = np.setdiff1d(np.arange(total), top, assume_unique=False)
        need = max(0, target_len - len(top))
        rest = (
            rng.choice(remain, size=min(need, len(remain)), replace=False)
            if len(remain) and need > 0
            else np.empty(0, dtype=int)
        )
        return np.concatenate([top, rest]).astype(np.int64)

    # Arms B, C, D: slots 0..3 are strictly the 4 seed anchors [0, 1, 2, 3]
    anchors = np.array([0, 1, 2, 3], dtype=np.int64)
    unrev_budget = target_len - 4
    unrev_all = np.arange(4, total, dtype=np.int64)
    unrev_gate = pool["gate"][4:]
    unrev_fit = pool["fitness"][4:]
    passed_u = unrev_all[np.flatnonzero(unrev_gate)]
    order_u = (
        passed_u[np.argsort(pool["fitness"][passed_u])]
        if len(passed_u)
        else unrev_all[np.argsort(unrev_fit)]
    )
    top_u = order_u[: min(44, len(order_u), unrev_budget)]

    if arm == "B":
        remain_u = np.setdiff1d(unrev_all, top_u, assume_unique=False)
        need_u = max(0, unrev_budget - len(top_u))
        rest_u = (
            rng.choice(remain_u, size=min(need_u, len(remain_u)), replace=False)
            if len(remain_u) and need_u > 0
            else np.empty(0, dtype=np.int64)
        )
        unrev_picks = np.concatenate([top_u, rest_u]).astype(np.int64)
    else:
        # Arms C & D: explicitly sample evaluated hard negatives into the slate
        hn_indices = np.asarray(
            sorted(idx for idx in pool.get("hard_negatives", set()) if 4 <= idx < total),
            dtype=np.int64,
        )
        hn_avail = np.setdiff1d(hn_indices, top_u, assume_unique=False)
        hn_quota = min(36, len(hn_avail), max(0, unrev_budget - len(top_u)))
        hn_picks = (
            rng.choice(hn_avail, size=hn_quota, replace=False)
            if hn_quota > 0
            else np.empty(0, dtype=np.int64)
        )
        chosen_so_far = np.concatenate([top_u, hn_picks]).astype(np.int64)
        remain_u = np.setdiff1d(unrev_all, chosen_so_far, assume_unique=False)
        need_u = max(0, unrev_budget - len(chosen_so_far))
        rest_u = (
            rng.choice(remain_u, size=min(need_u, len(remain_u)), replace=False)
            if len(remain_u) and need_u > 0
            else np.empty(0, dtype=np.int64)
        )
        unrev_picks = np.concatenate([chosen_so_far, rest_u]).astype(np.int64)

    # Permute unrevealed slots 4..191 so the best candidate is never at a deterministic index
    permuted_unrev = rng.permutation(unrev_picks).astype(np.int64)
    return np.concatenate([anchors, permuted_unrev])


def admit_gp_candidate_to_pool(
    pool: dict,
    rec: dict,
    *,
    arm: str,
) -> bool:
    """
    Apply arm-specific candidate admission rule (`promote()`):
      - Arms A & B: HWM-only (`cv_gate_passed == True` and `fitness < incumbent - 1e-6`).
      - Arms C & D: Admit ALL evaluated GP candidates (including non-HWM evaluations and gate failures),
        and register non-HWM / gate-failing candidates in `pool["hard_negatives"]`.
    """
    psha = str(rec["program_sha256"])
    gate_ok = bool(rec["cv_gate_passed"])
    fit = float(rec["fitness"])
    passed_mask = pool["gate"]
    incumbent = float(np.min(pool["fitness"][passed_mask])) if np.any(passed_mask) else 0.0
    beats_hwm = gate_ok and (fit < incumbent - 1e-6)

    if arm in ("A", "B"):
        if not beats_hwm or psha in pool["pids"]:
            return False
    else:
        # Arms C & D: if already in pool, still mark as hard negative if it did not beat HWM
        if psha in pool["pids"]:
            existing_idx = pool["pids"].index(psha)
            if not beats_hwm and existing_idx >= 4:
                pool["hard_negatives"].add(existing_idx)
            return False

    new_idx = len(pool["pids"])
    pool["pids"].append(psha)
    pool["program"] = np.vstack([pool["program"], np.asarray(rec["feat412"], dtype=np.float32)])
    pool["proj16"] = np.vstack([pool["proj16"], np.asarray(rec["proj16"], dtype=np.float32)])
    pool["fitness"] = np.append(pool["fitness"], fit)
    pool["logratio"] = np.append(pool["logratio"], float(rec["logratio"]))
    pool["gate"] = np.append(pool["gate"], gate_ok)
    pool["base_prior_cost"] = np.append(
        pool["base_prior_cost"], float(rec.get("base_prior_cost", 0.0))
    ).astype(np.float32)
    sos_val = rec.get("sos_prior", [0.0, 0.25])
    pool["sos_prior"] = np.vstack(
        [pool["sos_prior"], np.asarray(sos_val, dtype=np.float32).reshape(1, 2)]
    )
    if arm in ("C", "D") and not beats_hwm and new_idx >= 4:
        pool["hard_negatives"].add(new_idx)
    return True


def seed_initial_hard_negatives_for_cd(pools: dict[str, dict], fit_cids: list[str]) -> int:
    """
    For Arms C & D, initialize `pool["hard_negatives"]` on training contexts (`fit_cids`) with:
      1. All unrevealed gate failures (`idx >= 4` where `gate == False`).
      2. High-prior / high-regret decoy candidates (`idx >= 4` where `gate == True` and `fitness`
         is in the worst 35% of the pool while `base_prior_cost` is in the best 35%).
    """
    count = 0
    for cid in fit_cids:
        pool = pools[cid]
        gate = pool["gate"]
        fit = pool["fitness"]
        bprior = pool["base_prior_cost"]
        hn = pool["hard_negatives"]
        # 1. All unrevealed gate failures
        fail_idxs = np.flatnonzero(~gate[4:]) + 4
        for idx in fail_idxs.tolist():
            hn.add(int(idx))
        # 2. Prior-decoy passing candidates (good prior, poor actual fitness)
        passed_u = np.flatnonzero(gate[4:]) + 4
        if len(passed_u) >= 16:
            fit_q60 = float(np.quantile(fit[passed_u], 0.60))
            pri_q40 = float(np.quantile(bprior[passed_u], 0.40))
            decoys = passed_u[(fit[passed_u] >= fit_q60) & (bprior[passed_u] <= pri_q40)]
            for idx in decoys.tolist():
                hn.add(int(idx))
        count += len(hn)
    return count


def harvest_round_hard_negatives_cd(
    pools: dict[str, dict],
    batch_cids: list[str],
    indexes: list[np.ndarray],
    pred_tensor: torch.Tensor,
) -> int:
    """
    During training rounds in Arms C & D, inspect the model's top-8 predicted unrevealed candidates
    (`slots 4..191`) for each context in the batch. Any candidate ranked in the model's top-8 that
    either fails the gate (`gate == False`) or has actual fitness worse than the context's top-25%
    passing quantile is recorded in `pool["hard_negatives"]` for subsequent rounds.
    """
    added = 0
    pred_np = pred_tensor.detach().cpu().numpy()
    for cid, ix, row in zip(batch_cids, indexes, pred_np, strict=True):
        if len(ix) <= 4:
            continue
        pool = pools[cid]
        unrev_order = np.argsort(row[4:])[:8]
        top_pool_idxs = ix[4 + unrev_order]
        passed = np.flatnonzero(pool["gate"][4:]) + 4
        q25_fit = (
            float(np.quantile(pool["fitness"][passed], 0.25))
            if len(passed) > 0
            else 0.0
        )
        for p_idx in top_pool_idxs.tolist():
            p_idx = int(p_idx)
            if p_idx < 4:
                continue
            if (not bool(pool["gate"][p_idx])) or (float(pool["fitness"][p_idx]) > q25_fit):
                if p_idx not in pool["hard_negatives"]:
                    pool["hard_negatives"].add(p_idx)
                    added += 1
    return added


def compute_loss_arm(
    arm: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    gates: torch.Tensor,
) -> torch.Tensor:
    """
    Compute arm-specific surrogate loss:
      - Arm A: `smooth_l1_loss(pred, target, beta=0.25) + 0.1 * kl_div(tau=0.2)` across all slots `0..M-1`.
      - Arms B & C: Same functional form, but masked strictly to unrevealed slots `4..M-1` (`pred[:, 4:]`).
      - Arm D: Shift-invariant listwise top-focused loss on unrevealed slots `4..M-1`:
          1. Dual-temperature Plackett-Luce / ListNet cross-entropy (`tau_sharp=0.10`, `tau_broad=0.35`
             on within-context standardized passing regret targets; `tau_pred_1=0.15`, `tau_pred_2=0.40`
             on raw predicted costs).
          2. Pairwise gate-failure hinge margin (`max(0, 0.50 + c_best - c_fail)` and
             `0.5 * max(0, 0.25 + c_top8_mean - c_fail)`).
          3. Pairwise top-4 vs bottom-half passing hinge margin (`max(0, 0.15 + c_top4 - c_bot)`).
    """
    if arm == "A":
        reg = F.smooth_l1_loss(pred, target, beta=0.25)
        kl = F.kl_div(
            F.log_softmax(-pred / 0.2, dim=1),
            F.softmax(-target / 0.2, dim=1),
            reduction="batchmean",
        )
        return reg + 0.1 * kl

    pred_u = pred[:, 4:]
    target_u = target[:, 4:]
    gate_u = gates[:, 4:].bool()

    if arm in ("B", "C"):
        reg = F.smooth_l1_loss(pred_u, target_u, beta=0.25)
        kl = F.kl_div(
            F.log_softmax(-pred_u / 0.2, dim=1),
            F.softmax(-target_u / 0.2, dim=1),
            reduction="batchmean",
        )
        return reg + 0.1 * kl

    if arm != "D":
        raise ValueError(f"unsupported arm for loss: {arm}")

    # Arm D: Shift-invariant listwise Plackett-Luce / ListNet + Gate & Top-Choice Hinge Margins
    batch_size, m_unrev = pred_u.shape
    device = pred_u.device

    # Build within-context normalized regret target `r_u` [B, M_unrev] (detached target)
    with torch.no_grad():
        inf_pos = torch.full_like(target_u, 1e6)
        pass_costs = torch.where(gate_u, target_u, inf_pos)
        c_min = pass_costs.min(dim=1, keepdim=True).values
        has_pass = gate_u.any(dim=1, keepdim=True)
        c_min = torch.where(has_pass, c_min, target_u.min(dim=1, keepdim=True).values)

        pass_count = gate_u.sum(dim=1, keepdim=True).clamp_min(1).float()
        pass_mean = torch.where(gate_u, target_u, torch.zeros_like(target_u)).sum(dim=1, keepdim=True) / pass_count
        pass_var = (
            torch.where(gate_u, (target_u - pass_mean) ** 2, torch.zeros_like(target_u)).sum(dim=1, keepdim=True)
            / pass_count
        )
        c_std = pass_var.sqrt().clamp_min(0.05)

        regret_norm = ((target_u - c_min) / c_std).clamp(min=0.0, max=6.0)
        r_u = torch.where(gate_u, regret_norm, torch.full_like(regret_norm, 8.0))

        p_sharp = F.softmax(-r_u / 0.10, dim=1)
        p_broad = F.softmax(-r_u / 0.35, dim=1)

    log_q_sharp = F.log_softmax(-pred_u / 0.15, dim=1)
    log_q_broad = F.log_softmax(-pred_u / 0.40, dim=1)
    loss_pl = 0.7 * F.kl_div(log_q_sharp, p_sharp, reduction="batchmean") + 0.3 * F.kl_div(
        log_q_broad, p_broad, reduction="batchmean"
    )

    # Gate-failure and top-vs-bottom pairwise hinge margins (depend strictly on differences pred_i - pred_j)
    gate_losses = []
    pair_losses = []
    for b in range(batch_size):
        g_b = gate_u[b]
        p_b = pred_u[b]
        t_b = target_u[b]
        pass_idx = torch.nonzero(g_b, as_tuple=False).squeeze(-1)
        fail_idx = torch.nonzero(~g_b, as_tuple=False).squeeze(-1)

        if pass_idx.numel() > 0:
            p_pass = p_b[pass_idx]
            t_pass = t_b[pass_idx]
            order_pass = torch.argsort(t_pass)
            best_pred = p_pass[order_pass[0]]
            k8 = min(8, order_pass.numel())
            top8_mean_pred = p_pass[order_pass[:k8]].mean()

            if fail_idx.numel() > 0:
                p_fail = p_b[fail_idx]
                margin_best = F.relu(0.50 + best_pred - p_fail).mean()
                margin_top8 = F.relu(0.25 + top8_mean_pred - p_fail).mean()
                gate_losses.append(margin_best + 0.5 * margin_top8)

            if order_pass.numel() >= 8:
                k4 = min(4, order_pass.numel() // 2)
                top4_pred = p_pass[order_pass[:k4]].unsqueeze(1)  # [k4, 1]
                bot_pred = p_pass[order_pass[order_pass.numel() // 2 :]].unsqueeze(0)  # [1, M_bot]
                pair_losses.append(F.relu(0.15 + top4_pred - bot_pred).mean())

    loss_gate = torch.stack(gate_losses).mean() if gate_losses else torch.zeros((), device=device)
    loss_pair = torch.stack(pair_losses).mean() if pair_losses else torch.zeros((), device=device)
    return loss_pl + 0.50 * loss_gate + 0.25 * loss_pair


def evaluate_utility_arm(
    cids: list[str],
    pools: dict[str, dict],
    predict_fn,
    arm: str,
) -> dict:
    """
    Evaluate family-grouped utility, gate violation rate, normalized regret, and top-5% hit rate.
      - Arm A: selects `choice = int(np.argmin(row))` across all indices `0..P-1` (including revealed `0..3`).
      - Arms B, C, D: masks revealed seed anchors `0..3` from `argmin`, selecting
        `choice = 4 + int(np.argmin(row[4:]))`.
    """
    by_size: dict[int, list[str]] = {}
    for cid in cids:
        by_size.setdefault(len(pools[cid]["pids"]), []).append(cid)

    per_context: dict[str, float] = {}
    per_family: dict[str, list[float]] = {}
    viol_by_context: dict[str, float] = {}
    regret_by_context: dict[str, float] = {}
    hit5_by_context: dict[str, float] = {}

    for size, group in sorted(by_size.items()):
        for start in range(0, len(group), 64):
            batch = group[start : start + 64]
            ixs = [np.arange(size, dtype=np.int64) for _ in batch]
            with torch.no_grad():
                preds = predict_fn(batch, ixs).detach().cpu().numpy()
            for cid, row in zip(batch, preds, strict=True):
                pool = pools[cid]
                if arm == "A":
                    choice = int(np.argmin(row))
                else:
                    choice = 4 + int(np.argmin(row[4:]))

                gate_all = pool["gate"]
                fit_all = pool["fitness"]
                # Compute utility against the unrevealed passing pool (4..P-1) so all arms A/B/C/D
                # are scored on the exact same ground-truth oracle scale, while also recording
                # the standard full-pool utility!
                passed_all = np.flatnonzero(gate_all)
                if not bool(gate_all[choice]) or len(passed_all) == 0:
                    u = -1.0
                    viol = 1.0
                    reg_norm = 4.0
                    hit5 = 0.0
                else:
                    oracle = float(np.min(fit_all[passed_all]))
                    spread = max(float(np.std(fit_all[passed_all])), 0.05)
                    reg_norm = max(0.0, (float(fit_all[choice]) - oracle) / spread)
                    u = float(np.clip(1.0 - reg_norm / 2.0, -1.0, 1.0))
                    viol = 0.0
                    q05 = float(np.quantile(fit_all[passed_all], 0.05))
                    hit5 = 1.0 if float(fit_all[choice]) <= q05 + 1e-12 else 0.0

                per_context[cid] = u
                viol_by_context[cid] = viol
                regret_by_context[cid] = reg_norm
                hit5_by_context[cid] = hit5
                fam = pool["rec"]["family_id"]
                per_family.setdefault(fam, []).append(u)

    family_means = {fam: float(np.mean(vals)) for fam, vals in sorted(per_family.items())}
    return {
        "mean": float(np.mean(list(family_means.values()))),
        "median": float(np.median(list(family_means.values()))),
        "context_mean": float(np.mean(list(per_context.values()))),
        "violation_rate": float(np.mean(list(viol_by_context.values()))),
        "mean_regret_norm": float(np.mean(list(regret_by_context.values()))),
        "hit5_rate": float(np.mean(list(hit5_by_context.values()))),
        "per_family": family_means,
        "per_context": per_context,
    }


def evaluate_unrevealed_only(
    cids: list[str],
    pools: dict[str, dict],
    predict_fn,
) -> dict:
    """
    Leak-free unrevealed-only evaluation (`choice = 4 + argmin(row[4:])`) for ANY arm (including Arm A),
    showing how each arm performs when evaluated strictly on unrevealed candidates `4..P-1` without
    the Nadaraya-Watson self-distance $d(i,i)=0$ evaluation shortcut on `0..3`.
    """
    return evaluate_utility_arm(cids, pools, predict_fn, arm="B")


def bootstrap_t537(
    gains_by_family: dict[str, float],
    seed: int = 20261030,
    draws: int = BOOTSTRAP_DRAWS,
) -> dict:
    """
    Compute `t537` paired family-cluster bootstrap statistics:
      - `mean_lcb95 > +0.001` (one-sided 95% lower confidence bound, 5th percentile)
      - `family_median >= 0.0` (strictly non-negative; negative median fails)
    """
    fams = sorted(gains_by_family.keys())
    vals = np.asarray([gains_by_family[f] for f in fams], dtype=np.float64)
    if vals.size == 0:
        raise ValueError("empty family gains")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(draws, vals.size))
    boot_means = vals[idx].mean(axis=1)
    boot_medians = np.median(vals[idx], axis=1)
    mean_gain = float(vals.mean())
    median_gain = float(np.median(vals))
    mean_lcb95 = float(np.quantile(boot_means, 0.05))
    mean_ci95 = [float(np.quantile(boot_means, 0.025)), float(np.quantile(boot_means, 0.975))]
    median_ci95 = [float(np.quantile(boot_medians, 0.025)), float(np.quantile(boot_medians, 0.975))]
    pos_fams = int(np.sum(vals > 1e-9))
    zero_fams = int(np.sum(np.abs(vals) <= 1e-9))
    neg_fams = int(np.sum(vals < -1e-9))
    passed = bool((mean_lcb95 > GATE_MARGIN_LCB95) and (median_gain >= 0.0))
    return {
        "family_count": int(vals.size),
        "mean_gain": mean_gain,
        "family_median": median_gain,
        "mean_lcb95": mean_lcb95,
        "mean_ci95": mean_ci95,
        "median_ci95": median_ci95,
        "positive_families": pos_fams,
        "tied_families": zero_fams,
        "negative_families": neg_fams,
        "t537_gate_passed": passed,
    }


def compute_elo_panel(
    cids: list[str],
    family_by_cid: dict[str, str],
    scores_by_arm_and_cid: dict[str, dict[str, float]],
    anchor: str = "A",
    bootstrap_draws: int = 1000,
    seed: int = 20261030,
) -> dict:
    """Compute family-grouped Bradley-Terry Elo ratings and 95% family-bootstrap Elo CIs."""
    contenders = sorted(scores_by_arm_and_cid.keys())
    records = [
        DatasetEvalRecord(
            dataset_id=cid,
            family_id=family_by_cid[cid],
            task_type="regression",
            scores_by_contender={c: float(scores_by_arm_and_cid[c][cid]) for c in contenders},
        )
        for cid in cids
    ]
    point = compute_family_grouped_bradley_terry_elo(
        records=records,
        anchor_contender=anchor,
        anchor_elo=1500.0,
        prior=0.01,
    )
    by_fam: dict[str, list[DatasetEvalRecord]] = {}
    for rec in records:
        by_fam.setdefault(rec.family_id, []).append(rec)
    fam_names = sorted(by_fam.keys())
    rng = np.random.default_rng(seed)
    boot_ratings: dict[str, list[float]] = {c: [] for c in contenders}
    for _ in range(bootstrap_draws):
        chosen_fams = rng.choice(fam_names, size=len(fam_names), replace=True)
        boot_recs = []
        for b_idx, f_name in enumerate(chosen_fams):
            for r in by_fam[f_name]:
                boot_recs.append(
                    DatasetEvalRecord(
                        dataset_id=f"{r.dataset_id}__b{b_idx}",
                        family_id=f"{f_name}__b{b_idx}",
                        task_type="regression",
                        scores_by_contender=r.scores_by_contender,
                    )
                )
        res = compute_family_grouped_bradley_terry_elo(
            records=boot_recs,
            anchor_contender=anchor,
            anchor_elo=1500.0,
            prior=0.01,
        )
        for c in contenders:
            boot_ratings[c].append(float(res["ratings"][c]))

    elo_ci95 = {
        c: [
            round(float(np.quantile(boot_ratings[c], 0.025)), 2),
            round(float(np.quantile(boot_ratings[c], 0.975)), 2),
        ]
        for c in contenders
    }
    point["elo_ci95"] = elo_ci95
    return point


def family_folds(train_rows: list[dict], count: int = 5, seed: int = FOLD_SEED) -> list[set[str]]:
    families = sorted({row["family_id"] for row in train_rows})
    rng = np.random.default_rng(seed)
    order = list(np.asarray(families, dtype=object)[rng.permutation(len(families))])
    return [set(map(str, chunk)) for chunk in np.array_split(order, count)]


def family_balanced_batch(
    cids: list[str],
    family_by_cid: dict[str, str],
    size: int,
    rng: np.random.Generator,
) -> list[str]:
    by_fam: dict[str, list[str]] = {}
    for cid in cids:
        by_fam.setdefault(family_by_cid[cid], []).append(cid)
    names = sorted(by_fam.keys())
    chosen_fams = rng.integers(0, len(names), size=size)
    return [
        str(by_fam[names[int(fi)]][int(rng.integers(0, len(by_fam[names[int(fi)]])))])
        for fi in chosen_fams
    ]


def run_training_cell(
    *,
    arm: str,
    seed: int,
    cell_id: int,
    net: torch.nn.Module,
    base_pools: dict[str, dict],
    fit_cids: list[str],
    eval_cids: list[str],
    family_by_cid: dict[str, str],
    ctx_cache: dict[str, dict[str, torch.Tensor]],
    journal_by_cid: dict[str, list[dict]],
    rounds: int,
    steps_per_round: int,
    batch_size: int,
    slate_size: int,
    device: torch.device,
) -> dict:
    """
    Execute one self-contained flywheel training cell `(arm, seed, fold/terminal)`:
      1. Clone pools for `fit_cids` and `eval_cids` so mutations never leak across cells.
      2. Reset `net.adapter_10tok` to fold-clean init (`10000 + seed`, `scale=0.1`,
         `freeze_parent_delta=(arm != "A")`).
      3. Admit offline GP journal candidates on `fit_cids` using the arm's admission rule,
         and for Arms C/D initialize `pool["hard_negatives"]` on `fit_cids`.
      4. Run `rounds` flywheel rounds (`steps_per_round` gradient steps per round), harvesting
         model-proposed false positives / gate failures into `pool["hard_negatives"]` in Arms C/D.
      5. Evaluate once at the terminal round `R` on `eval_cids`.
    """
    t0 = time.monotonic()
    active_cids = sorted(set(fit_cids) | set(eval_cids))
    pools = clone_pools(base_pools, active_cids)

    trainable = reset_adapter_fold_clean(
        net,
        seed=seed,
        freeze_parent_delta=(arm != "A"),
        scale=0.1,
    )
    dope_before = verify_dope_guard(net, arm)
    init_sha = adapter_sha(net)
    init_snapshot = {
        name: p.detach().clone()
        for name, p in net.named_parameters()
        if p.requires_grad
    }

    # Apply journal GP candidate admissions strictly on `fit_cids`
    journal_admitted = 0
    for cid in fit_cids:
        for rec in journal_by_cid.get(cid, []):
            if admit_gp_candidate_to_pool(pools[cid], rec, arm=arm):
                journal_admitted += 1

    initial_hn_count = (
        seed_initial_hard_negatives_for_cd(pools, fit_cids)
        if arm in ("C", "D")
        else 0
    )

    predict_fn = make_predictor_arm(net, pools, ctx_cache, device, arm)
    total_steps = rounds * steps_per_round
    lr_init = 3e-4
    optimizer = torch.optim.AdamW(trainable, lr=lr_init, weight_decay=1e-4)
    base.guard.assert_optimizer_excludes_dope(optimizer, net)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1), eta_min=3e-5
    )

    # EMA shadow weights over trainable adapter parameters
    ema_decay = 0.90
    ema_state = {
        name: p.detach().clone()
        for name, p in net.adapter_10tok.named_parameters()
        if p.requires_grad
    }

    rng = np.random.default_rng(seed * 1_000_003 + cell_id * 97 + (ord(arm[0]) * 13))
    losses: list[float] = []
    grad_norms: list[float] = []
    harvested_hn_total = 0

    net.train()
    for r_idx in range(1, rounds + 1):
        for s_idx in range(steps_per_round):
            batch_cids = family_balanced_batch(fit_cids, family_by_cid, batch_size, rng)
            indexes = [sample_slate_arm(pools[cid], slate_size, rng, arm) for cid in batch_cids]

            optimizer.zero_grad(set_to_none=True)
            pred = predict_fn(batch_cids, indexes)

            gates_np = np.stack(
                [pools[cid]["gate"][ix] for cid, ix in zip(batch_cids, indexes, strict=True)]
            )
            fit_np = np.stack(
                [pools[cid]["fitness"][ix] for cid, ix in zip(batch_cids, indexes, strict=True)]
            )
            target_np = np.where(gates_np, fit_np, 10.0)
            target_t = torch.as_tensor(target_np, dtype=torch.float32, device=device)
            gates_t = torch.as_tensor(gates_np, dtype=torch.bool, device=device)

            loss = compute_loss_arm(arm, pred, target_t, gates_t)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                for name, p in net.adapter_10tok.named_parameters():
                    if p.requires_grad:
                        ema_state[name].mul_(ema_decay).add_(p.detach(), alpha=1.0 - ema_decay)

            if arm in ("C", "D") and s_idx == steps_per_round - 1:
                harvested_hn_total += harvest_round_hard_negatives_cd(
                    pools, batch_cids, indexes, pred
                )

            losses.append(float(loss.detach().cpu()))
            grad_norms.append(gnorm)

    # Load EMA weights for terminal evaluation
    with torch.no_grad():
        for name, p in net.adapter_10tok.named_parameters():
            if p.requires_grad:
                p.copy_(ema_state[name])

    net.eval()
    dope_after = verify_dope_guard(net, arm)
    final_sha = adapter_sha(net)
    update_l2 = float(
        torch.sqrt(
            sum(
                torch.sum((p.detach() - init_snapshot[name]) ** 2)
                for name, p in net.named_parameters()
                if p.requires_grad
            )
        ).cpu()
    )
    if update_l2 <= 0.0 or final_sha == init_sha:
        raise RuntimeError(f"zero parameter movement in cell ({arm}, seed={seed}, cell={cell_id})")

    # Terminal evaluation on `eval_cids`
    eval_native = evaluate_utility_arm(eval_cids, pools, predict_fn, arm=arm)
    eval_unrev = (
        evaluate_unrevealed_only(eval_cids, pools, predict_fn)
        if arm == "A"
        else eval_native
    )

    return {
        "arm": arm,
        "seed": seed,
        "cell_id": cell_id,
        "rounds": rounds,
        "steps_per_round": steps_per_round,
        "optimizer_steps": total_steps,
        "trainable_params": sum(p.numel() for p in trainable),
        "dope_sha256_before": dope_before,
        "dope_sha256_after": dope_after,
        "adapter_sha256_init": init_sha,
        "adapter_sha256_final": final_sha,
        "update_l2_from_init": update_l2,
        "journal_admitted": journal_admitted,
        "initial_hard_negatives": initial_hn_count,
        "harvested_hard_negatives": harvested_hn_total,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "eval_native": eval_native,
        "eval_unrev": eval_unrev,
        "elapsed_seconds": round(time.monotonic() - t0, 2),
    }


def load_h58_packet_and_cache(
    input_dir: Path,
    cache_dir: Path,
    checkpoint: Path,
    device: torch.device,
):
    """Load verified offline packet (`DL-Train` + `DL-Val`) and precomputed context/pool caches."""
    if sha256(input_dir / "source_manifest.json") != MANIFEST_FILE_SHA:
        raise RuntimeError("source_manifest.json file SHA mismatch")
    manifest = json.loads((input_dir / "source_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("source_manifest_sha256") != MANIFEST_ID
        or manifest.get("excluded_dl_holdout_count") != 93
        or manifest.get("excluded_payloads_opened") != 0
        or manifest.get("outer_rows_delivered") != 0
    ):
        raise RuntimeError("source_manifest isolation audit failed")

    selected = json.loads((input_dir / "selected.json").read_text(encoding="utf-8"))
    train_rows = selected["train"]
    val_rows = selected["dev"]
    if len(train_rows) != 1219 or len(val_rows) != 91:
        raise RuntimeError("cohort count mismatch")

    all_rows = train_rows + val_rows
    cids = [r["context_sha256"] for r in all_rows]
    family_by_cid = {r["context_sha256"]: r["family_id"] for r in all_rows}

    ctx_npz = np.load(cache_dir / "context_features.npz", allow_pickle=False)
    priors_npz = np.load(cache_dir / "pool_priors.npz", allow_pickle=False)
    if list(ctx_npz["cids"]) != cids or list(priors_npz["cids"]) != cids:
        raise RuntimeError("cache CID ordering mismatch")

    ctx_cache: dict[str, dict[str, torch.Tensor]] = {}
    base_pools: dict[str, dict] = {}

    tokens_all = ctx_npz["tokens_10x32"]
    masks_all = ctx_npz["mask_10"]
    ctx_feat_all = ctx_npz["ctx_feat_584"]
    offsets_all = priors_npz["offsets"]
    base_prior_all = priors_npz["base_prior_cost"]
    sos_prior_all = priors_npz["sos_prior"]
    scale_all = priors_npz["scale"]
    proj16_all = priors_npz["proj16"]

    for idx, row in enumerate(all_rows):
        cid = row["context_sha256"]
        lo, hi = int(offsets_all[idx]), int(offsets_all[idx + 1])
        ctx_cache[cid] = {
            "tokens_10x32": torch.as_tensor(tokens_all[idx], dtype=torch.float32, device=device),
            "mask_10": torch.as_tensor(masks_all[idx], dtype=torch.bool, device=device),
            "ctx_feat_584": torch.as_tensor(ctx_feat_all[idx], dtype=torch.float32, device=device),
        }
        pool_path = input_dir / "pools" / f"{cid}.npz"
        with np.load(pool_path, allow_pickle=False) as src:
            programs = src["program"].astype(np.float32)
            pids = [
                v.decode() if isinstance(v, bytes) else str(v)
                for v in src["program_sha256"].tolist()
            ]
            fitness = src["fitness"].astype(np.float64)
            logratio = src["logratio"].astype(np.float64)
            gate = src["gate"].astype(bool)
        if hi - lo != len(pids):
            raise RuntimeError(f"cached prior slice mismatch for {cid}: {hi - lo} != {len(pids)}")
        base_pools[cid] = {
            "rec": row,
            "pids": pids,
            "program": programs,
            "proj16": proj16_all[lo:hi].astype(np.float32),
            "fitness": fitness,
            "logratio": logratio,
            "gate": gate,
            "base_prior_cost": base_prior_all[lo:hi].astype(np.float32),
            "sos_prior": sos_prior_all[lo:hi].astype(np.float32),
            "scale": float(scale_all[idx]),
            "hard_negatives": set(),
        }

    journal_records = json.loads((cache_dir / "journal_candidates.json").read_text(encoding="utf-8"))
    journal_by_cid: dict[str, list[dict]] = {}
    for rec in journal_records:
        journal_by_cid.setdefault(rec["context_sha256"], []).append(rec)

    spec = importlib.util.spec_from_file_location("h58_runner_fused", input_dir / "cpu_exact_fused.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load cpu_exact_fused.py")
    fused = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fused)

    net = fused.FusedJopeNetwork(device=device, adapter_variant="h40b")
    ckpt_state = torch.load(checkpoint, map_location=device, weights_only=True)
    # Load ONLY frozen non-adapter backbone weights/buffers (zero r785/r882 adapter weight transfer)
    backbone_state = {
        k: v
        for k, v in ckpt_state["state_dict"].items()
        if not k.startswith("adapter_10tok.")
        or k.startswith("adapter_10tok.anchor.")
        or k in ("adapter_10tok.prog_mean", "adapter_10tok.prog_std")
    }
    net.load_state_dict(backbone_state, strict=False)
    for p in net.parameters():
        p.requires_grad_(False)
    base.guard.lock_dope_weights_and_eval_mode(net)
    if base.guard.snapshot_dope_weights(net).composite_sha256 != DOPE_SHA:
        raise RuntimeError("DOPE SHA mismatch after loading frozen backbone")

    return train_rows, val_rows, family_by_cid, base_pools, ctx_cache, journal_by_cid, net


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path.home() / "jope-data/h50/input")
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / "jope-data/h58/cache")
    parser.add_argument("--checkpoint", type=Path, default=Path.home() / "jope-data/h50/ckpt/round-785.pt")
    parser.add_argument("--out-dir", type=Path, default=Path.home() / "jope-data/h58/runs")
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--steps-per-round", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--slate-size", type=int, default=192)
    parser.add_argument("--lock-held", action="store_true")
    args = parser.parse_args()

    started = time.monotonic()
    if args.device == "cuda:0" and not args.lock_held:
        raise RuntimeError("GPU execution of H58 requires --lock-held under qualification.lock")
    if args.device == "cuda:0" and torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(0.90)
    torch.set_num_threads(4)

    device = torch.device(args.device)
    train_rows, val_rows, family_by_cid, base_pools, ctx_cache, journal_by_cid, net = (
        load_h58_packet_and_cache(args.input, args.cache_dir, args.checkpoint, device)
    )
    train_ids = [r["context_sha256"] for r in train_rows]
    val_ids = [r["context_sha256"] for r in val_rows]
    folds = family_folds(train_rows, count=5, seed=FOLD_SEED)

    # Evaluate fold-clean initialization (INIT_G0, 0 training steps, unrevealed-only evaluation)
    init_oof_ctx: dict[int, dict[str, float]] = {}
    init_val_ctx: dict[int, dict[str, float]] = {}
    for seed in args.seeds:
        reset_adapter_fold_clean(net, seed=seed, freeze_parent_delta=True, scale=0.1)
        net.eval()
        pred_fn = make_predictor_arm(net, base_pools, ctx_cache, device, "B")
        tr_eval = evaluate_utility_arm(train_ids, base_pools, pred_fn, arm="B")
        va_eval = evaluate_utility_arm(val_ids, base_pools, pred_fn, arm="B")
        init_oof_ctx[seed] = tr_eval["per_context"]
        init_val_ctx[seed] = va_eval["per_context"]

    # Storage for out-of-fold DL-Train and terminal DL-Val results across arms and seeds
    oof_native_ctx: dict[str, dict[int, dict[str, float]]] = {a: {} for a in ARMS}
    oof_unrev_ctx: dict[str, dict[int, dict[str, float]]] = {a: {} for a in ARMS}
    oof_native_viol: dict[str, list[float]] = {a: [] for a in ARMS}
    oof_unrev_viol: dict[str, list[float]] = {a: [] for a in ARMS}
    oof_native_hit5: dict[str, list[float]] = {a: [] for a in ARMS}
    oof_unrev_hit5: dict[str, list[float]] = {a: [] for a in ARMS}

    val_native_ctx: dict[str, dict[int, dict[str, float]]] = {a: {} for a in ARMS}
    val_unrev_ctx: dict[str, dict[int, dict[str, float]]] = {a: {} for a in ARMS}
    val_native_viol: dict[str, list[float]] = {a: [] for a in ARMS}
    val_unrev_viol: dict[str, list[float]] = {a: [] for a in ARMS}
    val_native_hit5: dict[str, list[float]] = {a: [] for a in ARMS}
    val_unrev_hit5: dict[str, list[float]] = {a: [] for a in ARMS}

    cell_receipts: list[dict] = []

    for seed in args.seeds:
        for arm in ARMS:
            seed_oof_native: dict[str, float] = {}
            seed_oof_unrev: dict[str, float] = {}
            fold_viols_native, fold_viols_unrev = [], []
            fold_hit5_native, fold_hit5_unrev = [], []

            for fold_idx, held_fams in enumerate(folds):
                fit_cids = [cid for cid in train_ids if family_by_cid[cid] not in held_fams]
                held_cids = [cid for cid in train_ids if family_by_cid[cid] in held_fams]
                res = run_training_cell(
                    arm=arm,
                    seed=seed,
                    cell_id=fold_idx + 1,
                    net=net,
                    base_pools=base_pools,
                    fit_cids=fit_cids,
                    eval_cids=held_cids,
                    family_by_cid=family_by_cid,
                    ctx_cache=ctx_cache,
                    journal_by_cid=journal_by_cid,
                    rounds=args.rounds,
                    steps_per_round=args.steps_per_round,
                    batch_size=args.batch_size,
                    slate_size=args.slate_size,
                    device=device,
                )
                seed_oof_native.update(res["eval_native"]["per_context"])
                seed_oof_unrev.update(res["eval_unrev"]["per_context"])
                fold_viols_native.append(res["eval_native"]["violation_rate"] * len(held_cids))
                fold_viols_unrev.append(res["eval_unrev"]["violation_rate"] * len(held_cids))
                fold_hit5_native.append(res["eval_native"]["hit5_rate"] * len(held_cids))
                fold_hit5_unrev.append(res["eval_unrev"]["hit5_rate"] * len(held_cids))

                receipt = {k: v for k, v in res.items() if k not in ("eval_native", "eval_unrev")}
                receipt["split"] = f"inner_fold_{fold_idx}"
                receipt["native_family_utility"] = res["eval_native"]["mean"]
                receipt["unrev_family_utility"] = res["eval_unrev"]["mean"]
                receipt["native_violation_rate"] = res["eval_native"]["violation_rate"]
                receipt["unrev_violation_rate"] = res["eval_unrev"]["violation_rate"]
                cell_receipts.append(receipt)
                print(
                    f"[H58] seed={seed} arm={arm} fold={fold_idx} "
                    f"native_u={res['eval_native']['mean']:.4f} "
                    f"unrev_u={res['eval_unrev']['mean']:.4f} "
                    f"viol={res['eval_unrev']['violation_rate']*100:.2f}% "
                    f"({res['elapsed_seconds']:.1f}s)",
                    flush=True,
                )

            if len(seed_oof_native) != len(train_ids):
                raise RuntimeError("incomplete out-of-fold predictions on DL-Train")
            oof_native_ctx[arm][seed] = seed_oof_native
            oof_unrev_ctx[arm][seed] = seed_oof_unrev
            oof_native_viol[arm].append(sum(fold_viols_native) / len(train_ids))
            oof_unrev_viol[arm].append(sum(fold_viols_unrev) / len(train_ids))
            oof_native_hit5[arm].append(sum(fold_hit5_native) / len(train_ids))
            oof_unrev_hit5[arm].append(sum(fold_hit5_unrev) / len(train_ids))

            # Single terminal DL-Val evaluation: train on all 1,219 DL-Train contexts for R rounds,
            # evaluate ONCE at terminal round R on the 91 DL-Val contexts
            term_res = run_training_cell(
                arm=arm,
                seed=seed,
                cell_id=99,
                net=net,
                base_pools=base_pools,
                fit_cids=train_ids,
                eval_cids=val_ids,
                family_by_cid=family_by_cid,
                ctx_cache=ctx_cache,
                journal_by_cid=journal_by_cid,
                rounds=args.rounds,
                steps_per_round=args.steps_per_round,
                batch_size=args.batch_size,
                slate_size=args.slate_size,
                device=device,
            )
            val_native_ctx[arm][seed] = term_res["eval_native"]["per_context"]
            val_unrev_ctx[arm][seed] = term_res["eval_unrev"]["per_context"]
            val_native_viol[arm].append(term_res["eval_native"]["violation_rate"])
            val_unrev_viol[arm].append(term_res["eval_unrev"]["violation_rate"])
            val_native_hit5[arm].append(term_res["eval_native"]["hit5_rate"])
            val_unrev_hit5[arm].append(term_res["eval_unrev"]["hit5_rate"])

            t_receipt = {k: v for k, v in term_res.items() if k not in ("eval_native", "eval_unrev")}
            t_receipt["split"] = "terminal_dl_val"
            t_receipt["terminal_dl_val_eval_count"] = 1
            t_receipt["native_family_utility"] = term_res["eval_native"]["mean"]
            t_receipt["unrev_family_utility"] = term_res["eval_unrev"]["mean"]
            t_receipt["native_violation_rate"] = term_res["eval_native"]["violation_rate"]
            t_receipt["unrev_violation_rate"] = term_res["eval_unrev"]["violation_rate"]
            cell_receipts.append(t_receipt)
            print(
                f"[H58] seed={seed} arm={arm} TERMINAL DL-Val "
                f"native_u={term_res['eval_native']['mean']:.4f} "
                f"unrev_u={term_res['eval_unrev']['mean']:.4f} "
                f"viol={term_res['eval_unrev']['violation_rate']*100:.2f}% "
                f"({term_res['elapsed_seconds']:.1f}s)",
                flush=True,
            )

    # Aggregate 3-seed mean per context and per family on DL-Train (205 families) and DL-Val (19 families)
    def aggregate_split(
        cids_split: list[str],
        ctx_by_arm_seed: dict[str, dict[int, dict[str, float]]],
        init_by_seed: dict[int, dict[str, float]],
    ):
        mean_ctx_by_arm: dict[str, dict[str, float]] = {}
        mean_fam_by_arm: dict[str, dict[str, float]] = {}
        contenders = list(ARMS) + ["INIT_G0"]
        for c in contenders:
            src = init_by_seed if c == "INIT_G0" else ctx_by_arm_seed[c]
            mean_ctx = {
                cid: float(np.mean([src[s][cid] for s in args.seeds]))
                for cid in cids_split
            }
            mean_ctx_by_arm[c] = mean_ctx
            by_f: dict[str, list[float]] = {}
            for cid, val in mean_ctx.items():
                by_f.setdefault(family_by_cid[cid], []).append(val)
            mean_fam_by_arm[c] = {f: float(np.mean(vs)) for f, vs in sorted(by_f.items())}
        return mean_ctx_by_arm, mean_fam_by_arm

    oof_unrev_mean_ctx, oof_unrev_mean_fam = aggregate_split(train_ids, oof_unrev_ctx, init_oof_ctx)
    oof_native_mean_ctx, oof_native_mean_fam = aggregate_split(train_ids, oof_native_ctx, init_oof_ctx)
    val_unrev_mean_ctx, val_unrev_mean_fam = aggregate_split(val_ids, val_unrev_ctx, init_val_ctx)
    val_native_mean_ctx, val_native_mean_fam = aggregate_split(val_ids, val_native_ctx, init_val_ctx)

    # Pairwise t537 gates on 205 DL-Train families and 19 terminal DL-Val families
    pairs = [("B", "A"), ("C", "A"), ("D", "A"), ("C", "B"), ("D", "C"), ("D", "B"), ("B", "INIT_G0"), ("C", "INIT_G0"), ("D", "INIT_G0")]

    def compute_pair_gates(mean_fam_dict: dict[str, dict[str, float]], seed_base: int) -> dict[str, dict]:
        out = {}
        for idx, (lhs, rhs) in enumerate(pairs):
            fams = sorted(mean_fam_dict[lhs].keys())
            gains = {f: mean_fam_dict[lhs][f] - mean_fam_dict[rhs][f] for f in fams}
            out[f"{lhs}_vs_{rhs}"] = bootstrap_t537(gains, seed=seed_base + idx * 101)
        return out

    oof_gates_unrev = compute_pair_gates(oof_unrev_mean_fam, 20261030)
    oof_gates_native = compute_pair_gates(oof_native_mean_fam, 20262030)
    val_gates_unrev = compute_pair_gates(val_unrev_mean_fam, 20263030)
    val_gates_native = compute_pair_gates(val_native_mean_fam, 20264030)

    # Family-grouped Bradley-Terry Elo on 205 DL-Train families and 19 DL-Val families
    oof_elo_unrev = compute_elo_panel(train_ids, family_by_cid, oof_unrev_mean_ctx, anchor="A", seed=20265001)
    oof_elo_native = compute_elo_panel(train_ids, family_by_cid, oof_native_mean_ctx, anchor="A", seed=20265002)
    val_elo_unrev = compute_elo_panel(val_ids, family_by_cid, val_unrev_mean_ctx, anchor="A", seed=20265003)
    val_elo_native = compute_elo_panel(val_ids, family_by_cid, val_native_mean_ctx, anchor="A", seed=20265004)

    summary = {
        "schema": "jope.h58.wave-summary/v1",
        "wave": "H58",
        "task_id": "t550",
        "device": str(device),
        "seeds": list(args.seeds),
        "rounds": args.rounds,
        "steps_per_round": args.steps_per_round,
        "optimizer_steps_per_cell": args.rounds * args.steps_per_round,
        "batch_size": args.batch_size,
        "slate_size": args.slate_size,
        "manifest_file_sha256": MANIFEST_FILE_SHA,
        "manifest_id": MANIFEST_ID,
        "dope_sha256": DOPE_SHA,
        "excluded_payloads_opened": 0,
        "outer_rows_delivered": 0,
        "terminal_dl_val_eval_count_per_arm_seed": 1,
        "dl_train_oof_205_families": {
            "arm_summary": {
                c: {
                    "unrev_family_utility_mean": float(np.mean(list(oof_unrev_mean_fam[c].values()))),
                    "unrev_family_utility_median": float(np.median(list(oof_unrev_mean_fam[c].values()))),
                    "native_family_utility_mean": float(np.mean(list(oof_native_mean_fam[c].values()))),
                    "native_family_utility_median": float(np.median(list(oof_native_mean_fam[c].values()))),
                    "unrev_violation_rate_mean": (
                        float(np.mean(oof_unrev_viol[c])) if c in ARMS else None
                    ),
                    "native_violation_rate_mean": (
                        float(np.mean(oof_native_viol[c])) if c in ARMS else None
                    ),
                    "unrev_hit5_rate_mean": (
                        float(np.mean(oof_unrev_hit5[c])) if c in ARMS else None
                    ),
                    "native_hit5_rate_mean": (
                        float(np.mean(oof_native_hit5[c])) if c in ARMS else None
                    ),
                    "unrev_elo": oof_elo_unrev["ratings"][c],
                    "unrev_elo_ci95": oof_elo_unrev["elo_ci95"][c],
                    "native_elo": oof_elo_native["ratings"][c],
                    "native_elo_ci95": oof_elo_native["elo_ci95"][c],
                }
                for c in list(ARMS) + ["INIT_G0"]
            },
            "t537_gates_unrev": oof_gates_unrev,
            "t537_gates_native": oof_gates_native,
            "bradley_terry_elo_unrev": oof_elo_unrev,
            "bradley_terry_elo_native": oof_elo_native,
        },
        "dl_val_terminal_19_families": {
            "arm_summary": {
                c: {
                    "unrev_family_utility_mean": float(np.mean(list(val_unrev_mean_fam[c].values()))),
                    "unrev_family_utility_median": float(np.median(list(val_unrev_mean_fam[c].values()))),
                    "native_family_utility_mean": float(np.mean(list(val_native_mean_fam[c].values()))),
                    "native_family_utility_median": float(np.median(list(val_native_mean_fam[c].values()))),
                    "unrev_violation_rate_mean": (
                        float(np.mean(val_unrev_viol[c])) if c in ARMS else None
                    ),
                    "native_violation_rate_mean": (
                        float(np.mean(val_native_viol[c])) if c in ARMS else None
                    ),
                    "unrev_hit5_rate_mean": (
                        float(np.mean(val_unrev_hit5[c])) if c in ARMS else None
                    ),
                    "native_hit5_rate_mean": (
                        float(np.mean(val_native_hit5[c])) if c in ARMS else None
                    ),
                    "unrev_elo": val_elo_unrev["ratings"][c],
                    "unrev_elo_ci95": val_elo_unrev["elo_ci95"][c],
                    "native_elo": val_elo_native["ratings"][c],
                    "native_elo_ci95": val_elo_native["elo_ci95"][c],
                }
                for c in list(ARMS) + ["INIT_G0"]
            },
            "t537_gates_unrev": val_gates_unrev,
            "t537_gates_native": val_gates_native,
            "bradley_terry_elo_unrev": val_elo_unrev,
            "bradley_terry_elo_native": val_elo_native,
        },
        "cell_count": len(cell_receipts),
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out_dir / "wave_summary.json", summary)
    atomic_json(args.out_dir / "cell_receipts.json", {"cells": cell_receipts})
    print(json.dumps(summary["dl_train_oof_205_families"]["arm_summary"], indent=2), flush=True)
    print(json.dumps(summary["dl_val_terminal_19_families"]["arm_summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
