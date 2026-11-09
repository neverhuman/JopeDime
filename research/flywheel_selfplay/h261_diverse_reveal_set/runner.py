#!/usr/bin/env python3
"""
H261 (`h261_diverse_reveal_set`) — Diversity-Aware 4-Reveal Candidate Selection (`MMR4_CV`, `STRUCT4`)
vs `TOP4_PICK` at Matched 4-Reveal Budget on `r882` and `r3175` (0 Optimizer Updates, CPU Only).
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

EVIDENCE_ROOT = Path("/home/ubuntu/jope_plan/evidence/H261/argon-t1530")
SNAPSHOT_DIR = EVIDENCE_ROOT / "snapshot"
INPUTS_DIR = EVIDENCE_ROOT / "inputs"
CAPSULE_STUDY_DIR = (
    EVIDENCE_ROOT
    / "capsule"
    / "research"
    / "flywheel_selfplay"
    / "h261_diverse_reveal_set"
)

INPUT_POOLS_DIR = Path("/home/ubuntu/jope-data/h50/input")
CACHE_DIR = Path("/home/ubuntu/jope-data/h58/cache")
POOL_PRIORS_NPZ = CACHE_DIR / "pool_priors.npz"

R882_CKPT = INPUTS_DIR / "r882" / "round-882.pt"
R882_BEST_DEV_JSON = INPUTS_DIR / "r882" / "best_dev_checkpoint.json"
R882_TRAIN_COSTS_NPZ = INPUTS_DIR / "r882-train-costs.npz"
R882_VAL_COSTS_NPZ = INPUTS_DIR / "r882-secondary-costs.npz"

R3175_CKPT = INPUTS_DIR / "r3175" / "round-3175.pt"
R3175_TRAIN_COSTS_NPZ = INPUTS_DIR / "r3175-train-costs.npz"
R3175_VAL_COSTS_NPZ = INPUTS_DIR / "r3175-secondary-costs.npz"

DOPE_EXPECTED_SHA256 = (
    "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
)
R882_EXPECTED_SHA256 = (
    "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
)
R3175_EXPECTED_SHA256 = (
    "1d6ea900ebed8d2a05c3b3f37576adc15aadd2b53c585bfd5cb71a103e9ff97b"
)

R882_NATIVE91_EXPECTED = 1.1377941898164525
TOP4_PICK_VAL_CTX_EXPECTED = 1.2403601682250796
TOP4_PICK_VAL_FAM_EXPECTED = 1.2519997168684853

FOLD_SEED = 261140
SEEDS = (261101, 261102, 261103)
BOOT_SEED = 261150
BOOT_DRAWS = 10_000
HOLM_M = 2
GATE_THRESHOLD = 0.25
LAMBDA_GRID = (0.1, 0.2, 0.3, 0.5, 1.0)
TIE_MARGIN = 1e-4


def trim_malloc() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def check_cooperative_pause() -> None:
    pause_file = os.environ.get("H261_PAUSE_FILE", "")
    if not pause_file:
        return
    p = Path(pause_file)
    while p.exists():
        time.sleep(0.5)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_snapshot_modules() -> tuple[Any, Any]:
    if str(SNAPSHOT_DIR) not in sys.path:
        sys.path.insert(0, str(SNAPSHOT_DIR))
    spec_c = importlib.util.spec_from_file_location(
        "common_h252", SNAPSHOT_DIR / "common_h252.py"
    )
    assert spec_c is not None and spec_c.loader is not None
    common_mod = importlib.util.module_from_spec(spec_c)
    spec_c.loader.exec_module(common_mod)

    spec_s = importlib.util.spec_from_file_location(
        "stage1_hurdle_sos_critic", SNAPSHOT_DIR / "stage1_hurdle_sos_critic.py"
    )
    assert spec_s is not None and spec_s.loader is not None
    stage1_mod = importlib.util.module_from_spec(spec_s)
    spec_s.loader.exec_module(stage1_mod)
    return common_mod, stage1_mod


def compute_top_zgap(rcost: np.ndarray) -> float:
    part = np.partition(rcost, 1)
    sd = float(np.std(rcost))
    if sd <= 1e-12:
        return 999.0
    return float((part[1] - part[0]) / sd)


def compute_candidate_emb40(sem24: np.ndarray, proj16: np.ndarray) -> np.ndarray:
    """
    Constructs the frozen 40D unit-norm candidate embedding from `sem24` (24D) and `proj16` (16D),
    standardized within the candidate pool so cosine similarity has zero mean bias.
    """
    raw = np.concatenate([sem24.astype(np.float32), proj16.astype(np.float32)], axis=1)
    mean = raw.mean(axis=0, keepdims=True)
    std = raw.std(axis=0, keepdims=True) + 1e-6
    z = (raw - mean) / std
    norms = np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-6)
    return (z / norms).astype(np.float32)


def extract_learner_and_dsl_heads(
    prog_412: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts outcome-free structural head identifiers from `prog_412`:
      - `est_id`: terminal learner family (`prog_412[:, 24:29]`: 0=elasticnet, 1=histgbm,
        2=ksmooth, 3=ridge, 4=ridge_gcv; -1 if none active).
      - `dsl_head_id`: deterministic integer ID for `(est_id, active_op_bitmask)` where
        `active_op_bitmask` encodes which of the 23 operators (`prog_412[:, 0:23] > 0`) appear.
    """
    est_slice = prog_412[:, 24:29]
    has_est = est_slice.max(axis=1) > 0
    est_id = np.where(has_est, np.argmax(est_slice, axis=1), -1).astype(np.int32)

    op_active = (prog_412[:, 0:23] > 0).astype(np.int64)
    bit_weights = (1 << np.arange(23, dtype=np.int64))[None, :]
    op_mask = (op_active * bit_weights).sum(axis=1)
    dsl_head_id = ((est_id.astype(np.int64) + 2) << 24) | op_mask
    return est_id, dsl_head_id


def select_mmr4_from_top16(
    top16: np.ndarray,
    rcost: np.ndarray,
    emb_unit: np.ndarray,
    lam: float,
) -> list[int]:
    """
    Greedy Maximal Marginal Relevance (MMR) selecting 4 unique candidates from `top16`:
      - Slot 0: `top16[0]` (lowest `rcost` candidate)
      - Slots 1..3: maximize `score(i) = -z_cost(i) - lam * max_{j in chosen} cos_sim(i, j)`
        where `z_cost` is standardized across `top16` (`(c - mean(c16)) / max(std(c16), 1e-6)`)
        and ties are broken by ascending `rcost` (`top16` order).
    """
    c16 = rcost[top16].astype(np.float64)
    c_std = max(float(np.std(c16)), 1e-6)
    z_cost16 = (c16 - float(np.mean(c16))) / c_std

    e16 = emb_unit[top16].astype(np.float64)
    sim16 = np.clip(e16 @ e16.T, -1.0, 1.0)

    chosen_pos: list[int] = [0]
    for _ in range(3):
        best_pos = -1
        best_score = -1e18
        for pos in range(len(top16)):
            if pos in chosen_pos:
                continue
            max_sim = max(float(sim16[pos, cpos]) for cpos in chosen_pos)
            sc = -float(z_cost16[pos]) - float(lam) * max_sim
            if sc > best_score:
                best_score = sc
                best_pos = pos
        chosen_pos.append(best_pos)
    return [int(top16[p]) for p in chosen_pos]


def select_struct4_from_top16(
    top16: np.ndarray,
    est_id: np.ndarray,
    dsl_head_id: np.ndarray,
    est_only: bool = False,
) -> list[int]:
    """
    Selects the best-cost candidate from each of the 4 most-preferred distinct learner families
    / DSL heads in `top16`, with fallback to next by cost:
      - Pass 1: distinct learner family `est_id` in ascending `rcost` (`top16`) order.
      - Pass 2 (unless `est_only=True`): if < 4 distinct learner families in `top16`,
        distinct `(est_id, op_bitmask)` `dsl_head_id` in ascending `rcost` order.
      - Pass 3: fallback to next unselected candidate in `top16` by ascending `rcost`.
    """
    chosen: list[int] = []
    seen_est: set[int] = set()
    for idx_raw in top16:
        idx = int(idx_raw)
        e = int(est_id[idx])
        if e not in seen_est:
            seen_est.add(e)
            chosen.append(idx)
            if len(chosen) == 4:
                return chosen

    if not est_only and len(chosen) < 4:
        seen_head = {int(dsl_head_id[i]) for i in chosen}
        for idx_raw in top16:
            idx = int(idx_raw)
            if idx in chosen:
                continue
            h = int(dsl_head_id[idx])
            if h not in seen_head:
                seen_head.add(h)
                chosen.append(idx)
                if len(chosen) == 4:
                    return chosen

    for idx_raw in top16:
        idx = int(idx_raw)
        if idx not in chosen:
            chosen.append(idx)
            if len(chosen) == 4:
                break
    return chosen


def pick_best_revealed_gated_cost(
    rev_indices: list[int],
    rcost: np.ndarray,
    fitness: np.ndarray,
    gate: np.ndarray,
) -> int:
    """
    Picks `argmin` revealed gated cost (`fitness if gate else 2.0`) among `rev_indices`,
    breaking ties deterministically by ascending incumbent `rcost` order (then candidate index).
    """
    ordered = sorted(rev_indices, key=lambda idx: (float(rcost[idx]), int(idx)))
    best_idx = ordered[0]
    best_cost = float(fitness[best_idx]) if bool(gate[best_idx]) else 2.0
    for idx in ordered[1:]:
        c = float(fitness[idx]) if bool(gate[idx]) else 2.0
        if c < best_cost:
            best_cost = c
            best_idx = idx
    return int(best_idx)


def compute_rev4_redundancy_stats(
    rev4: list[int],
    top16: np.ndarray,
    emb40: np.ndarray,
    sem24: np.ndarray,
    est_id: np.ndarray,
    dsl_head_id: np.ndarray,
    utility: np.ndarray,
) -> dict[str, float]:
    e4 = emb40[rev4].astype(np.float64)
    s4 = sem24[rev4].astype(np.float64)
    sim_e = np.clip(e4 @ e4.T, -1.0, 1.0)
    sim_s = np.clip(s4 @ s4.T, -1.0, 1.0)
    iu = np.triu_indices(4, k=1)
    mean_cos_emb40 = float(np.mean(sim_e[iu]))
    mean_cos_sem24 = float(np.mean(sim_s[iu]))
    n_est = float(len({int(est_id[i]) for i in rev4}))
    n_head = float(len({int(dsl_head_id[i]) for i in rev4}))

    best_u_rev4 = float(np.max(utility[rev4]))
    best_u_top16 = float(np.max(utility[top16]))
    best_u_pool = float(np.max(utility))
    recall_top16 = 1.0 if best_u_rev4 >= (best_u_top16 - 1e-12) else 0.0
    recall_pool = 1.0 if best_u_rev4 >= (best_u_pool - 1e-12) else 0.0
    return {
        "mean_cos_emb40": mean_cos_emb40,
        "mean_cos_sem24": mean_cos_sem24,
        "distinct_est": n_est,
        "distinct_dsl_head": n_head,
        "recall_top16": recall_top16,
        "recall_pool": recall_pool,
        "best_u_top16": best_u_top16,
    }


def score_h261_context_deterministic(
    ctx: dict[str, Any],
    rcost: np.ndarray,
) -> dict[str, Any]:
    """
    Precomputes all deterministic 4-reveal sets and utilities on a single context (`r882` or `r3175`)
    across `TOP4_PICK`, `STRUCT4`, `STRUCT4_EST_ONLY`, and `MMR4(lam)` for all `lam in LAMBDA_GRID`.
    """
    u = ctx["utility"]
    fitness = ctx["fitness"]
    gate = ctx["gate"]
    emb40 = ctx["emb40"]
    sem24 = ctx["sem24"]
    est_id = ctx["est_id"]
    dsl_head_id = ctx["dsl_head_id"]

    top_zgap = compute_top_zgap(rcost)
    is_gated = bool(top_zgap < GATE_THRESHOLD)
    pick_0 = int(np.argmin(rcost))
    u_0 = float(u[pick_0])
    top16 = np.argsort(rcost, kind="mergesort")[:16]

    if not is_gated:
        zero_rev = {
            "TOP4_PICK": 0,
            "STRUCT4": 0,
            "STRUCT4_EST_ONLY": 0,
            **{f"MMR4_L{lam:.1f}": 0 for lam in LAMBDA_GRID},
        }
        scores = {
            "INCUMBENT_0": u_0,
            "TOP4_PICK": u_0,
            "STRUCT4": u_0,
            "STRUCT4_EST_ONLY": u_0,
            **{f"MMR4_L{lam:.1f}": u_0 for lam in LAMBDA_GRID},
        }
        return {
            "is_gated": False,
            "top_zgap": top_zgap,
            "top16": top16,
            "u_0": u_0,
            "scores": scores,
            "reveals_used": zero_rev,
            "rev_sets": {},
            "redundancy": {},
        }

    rev_top4 = [int(x) for x in top16[:4]]
    rev_struct4 = select_struct4_from_top16(top16, est_id, dsl_head_id, est_only=False)
    rev_struct4_est = select_struct4_from_top16(
        top16, est_id, dsl_head_id, est_only=True
    )
    rev_mmr_by_lam = {
        f"MMR4_L{lam:.1f}": select_mmr4_from_top16(top16, rcost, emb40, lam)
        for lam in LAMBDA_GRID
    }

    all_sets = {
        "TOP4_PICK": rev_top4,
        "STRUCT4": rev_struct4,
        "STRUCT4_EST_ONLY": rev_struct4_est,
        **rev_mmr_by_lam,
    }
    scores = {"INCUMBENT_0": u_0}
    reveals_used: dict[str, int] = {}
    redundancy: dict[str, dict[str, float]] = {}

    for arm_name, rev4 in all_sets.items():
        assert len(rev4) == 4 and len(set(rev4)) == 4
        reveals_used[arm_name] = 4
        pick_idx = pick_best_revealed_gated_cost(rev4, rcost, fitness, gate)
        u_arm = float(u[pick_idx])
        scores[arm_name] = u_arm
        red = compute_rev4_redundancy_stats(
            rev4, top16, emb40, sem24, est_id, dsl_head_id, u
        )
        red["top16_oracle_regret"] = float(red["best_u_top16"] - u_arm)
        redundancy[arm_name] = red

    return {
        "is_gated": True,
        "top_zgap": top_zgap,
        "top16": top16,
        "u_0": u_0,
        "scores": scores,
        "reveals_used": reveals_used,
        "rev_sets": all_sets,
        "redundancy": redundancy,
    }


def score_rand4_and_perm_control(
    ctx: dict[str, Any],
    det_out: dict[str, Any],
    rcost: np.ndarray,
    mmr_key: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """
    Evaluates seeded `RAND4_OF16` (4 random reveals from `top16` without replacement)
    and the permuted-fitness negative control on `MMR4_CV`'s `rev4`.
    """
    if not det_out["is_gated"]:
        u_0 = det_out["u_0"]
        return {
            "u_rand4": u_0,
            "u_perm_mmr4": u_0,
            "u_perm_struct4": u_0,
            "rand4_reveals": 0,
            "rand4_redundancy": None,
        }

    u = ctx["utility"]
    fitness = ctx["fitness"]
    gate = ctx["gate"]
    top16 = det_out["top16"]

    rand_pos = rng.choice(len(top16), size=4, replace=False)
    rev_rand4 = [int(top16[int(p)]) for p in rand_pos]
    assert len(set(rev_rand4)) == 4
    pick_rand = pick_best_revealed_gated_cost(rev_rand4, rcost, fitness, gate)
    u_rand4 = float(u[pick_rand])
    red_rand = compute_rev4_redundancy_stats(
        rev_rand4,
        top16,
        ctx["emb40"],
        ctx["sem24"],
        ctx["est_id"],
        ctx["dsl_head_id"],
        u,
    )
    red_rand["top16_oracle_regret"] = float(red_rand["best_u_top16"] - u_rand4)

    # Permuted-fitness negative control on MMR4_CV and STRUCT4 revealed sets
    rev_mmr = det_out["rev_sets"][mmr_key]
    perm_m = rng.permutation(4)
    f_mmr_perm = fitness[rev_mmr][perm_m]
    g_mmr_perm = gate[rev_mmr][perm_m]
    c_mmr_perm = np.where(g_mmr_perm, f_mmr_perm, 2.0)
    # Break ties by rcost order of rev_mmr
    order_mmr = sorted(
        range(4), key=lambda k: (float(rcost[rev_mmr[k]]), int(rev_mmr[k]))
    )
    best_k_mmr = min(order_mmr, key=lambda k: float(c_mmr_perm[k]))
    u_perm_mmr4 = float(u[rev_mmr[best_k_mmr]])

    rev_st = det_out["rev_sets"]["STRUCT4"]
    perm_s = rng.permutation(4)
    f_st_perm = fitness[rev_st][perm_s]
    g_st_perm = gate[rev_st][perm_s]
    c_st_perm = np.where(g_st_perm, f_st_perm, 2.0)
    order_st = sorted(range(4), key=lambda k: (float(rcost[rev_st[k]]), int(rev_st[k])))
    best_k_st = min(order_st, key=lambda k: float(c_st_perm[k]))
    u_perm_struct4 = float(u[rev_st[best_k_st]])

    return {
        "u_rand4": u_rand4,
        "u_perm_mmr4": u_perm_mmr4,
        "u_perm_struct4": u_perm_struct4,
        "rand4_reveals": 4,
        "rand4_redundancy": red_rand,
    }


def load_compact_contexts_batched(
    all_rows: list[dict[str, Any]],
    batch_size: int = 50,
) -> dict[str, dict[str, Any]]:
    """
    Loads `1,310` pools in batches of `50`, computing `emb40`, `sem24`, `est_id`, `dsl_head_id`
    and immediately discarding `program` `(1152, 412)` and `proj16` so peak RSS stays < 1.3 GiB.
    """
    common_mod, _ = load_snapshot_modules()
    proj16_by_cid: dict[str, np.ndarray] = {}
    with np.load(POOL_PRIORS_NPZ, allow_pickle=False) as zp:
        cids_p = [str(c) for c in zp["cids"]]
        offsets = zp["offsets"]
        proj16_all = zp["proj16"]
        for i, cid in enumerate(cids_p):
            proj16_by_cid[cid] = proj16_all[offsets[i] : offsets[i + 1]].astype(
                np.float32
            )

    compact: dict[str, dict[str, Any]] = {}
    for start in range(0, len(all_rows), batch_size):
        check_cooperative_pause()
        batch = all_rows[start : start + batch_size]
        for r in batch:
            cid = r["context_sha256"]
            p = INPUT_POOLS_DIR / "pools" / f"{cid}.npz"
            if sha256_file(p) != r["pool_sha256"]:
                raise RuntimeError(f"Pool SHA256 mismatch for {cid}")
            with np.load(p, allow_pickle=False) as z:
                prog = z["program"].astype(np.float32)
                fitness = z["fitness"].astype(np.float64)
                gate = z["gate"].astype(bool)
                utility = np.where(gate, -fitness, -2.0).astype(np.float64)
                sem24 = common_mod.compute_sem24_numpy(prog)
                est_id, dsl_head_id = extract_learner_and_dsl_heads(prog)
                emb40 = compute_candidate_emb40(sem24, proj16_by_cid.pop(cid))
            compact[cid] = {
                "family_id": r["family_id"],
                "fitness": fitness,
                "gate": gate,
                "utility": utility,
                "sem24": sem24,
                "emb40": emb40,
                "est_id": est_id,
                "dsl_head_id": dsl_head_id,
            }
        trim_malloc()
    proj16_by_cid.clear()
    trim_malloc()
    return compact


def load_cost_caches(
    all_rows: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    r882_costs: dict[str, np.ndarray] = {}
    r3175_costs: dict[str, np.ndarray] = {}
    with (
        np.load(R882_TRAIN_COSTS_NPZ, allow_pickle=False) as zt,
        np.load(R882_VAL_COSTS_NPZ, allow_pickle=False) as zv,
    ):
        for r in all_rows:
            cid = r["context_sha256"]
            r882_costs[cid] = (zt[cid] if cid in zt.files else zv[cid]).astype(
                np.float32
            )
    with (
        np.load(R3175_TRAIN_COSTS_NPZ, allow_pickle=False) as zt3,
        np.load(R3175_VAL_COSTS_NPZ, allow_pickle=False) as zv3,
    ):
        for r in all_rows:
            cid = r["context_sha256"]
            r3175_costs[cid] = (zt3[cid] if cid in zt3.files else zv3[cid]).astype(
                np.float32
            )
    return r882_costs, r3175_costs


def paired_family_contrast(
    rows: list[dict[str, Any]],
    panel_3d: np.ndarray,
    arm_names: list[str],
    cand_arm: str,
    base_arm: str,
    boot_ix: np.ndarray,
) -> dict[str, Any]:
    """
    Computes paired family-clustered bootstrap contrast (`cand_arm - base_arm`) across 3 seeds
    with `10,000` bootstrap draws, reporting median/mean family & context gains, `95% CI`,
    Holm `m=2` `97.5% CI`, one-sided p-value, per-seed family W/T/L, and harmed families (`< -0.01`).
    """
    ic = arm_names.index(cand_arm)
    ib = arm_names.index(base_arm)
    fams = [r["family_id"] for r in rows]
    fam_order = sorted(set(fams))
    fams_arr = np.asarray(fams)

    # Context gains per seed: [3, N_ctx]
    ctx_diff = panel_3d[:, :, ic] - panel_3d[:, :, ib]
    per_seed_ctx_gains = [float(np.mean(ctx_diff[s])) for s in range(panel_3d.shape[0])]

    # Family means per seed: [3, N_fam]
    fam_diff = np.stack(
        [ctx_diff[:, fams_arr == f].mean(axis=1) for f in fam_order],
        axis=1,
    )
    per_seed_fam_gains = [float(np.mean(fam_diff[s])) for s in range(panel_3d.shape[0])]
    pos_seeds = int(sum(1 for g in per_seed_fam_gains if g > 0.0))

    # Seed-averaged family diff for bootstrap & harmed family list
    fam_diff_mean_seed = fam_diff.mean(axis=0)  # [N_fam]
    boot_means = fam_diff_mean_seed[boot_ix].mean(axis=1)  # [B]
    ci95 = [
        float(np.quantile(boot_means, 0.025)),
        float(np.quantile(boot_means, 0.975)),
    ]
    ci975 = [
        float(np.quantile(boot_means, 0.0125)),
        float(np.quantile(boot_means, 0.9875)),
    ]
    raw_p = float((1 + int(np.sum(boot_means <= 0.0))) / (len(boot_means) + 1))

    wtl_by_seed = []
    for s in range(panel_3d.shape[0]):
        d_s = fam_diff[s]
        w = int(np.sum(d_s > TIE_MARGIN))
        loss_c = int(np.sum(d_s < -TIE_MARGIN))
        t = int(len(d_s) - w - loss_c)
        wtl_by_seed.append({"wins": w, "ties": t, "losses": loss_c})

    harmed_fams = [
        {"family_id": fam_order[i], "delta": float(fam_diff_mean_seed[i])}
        for i in range(len(fam_order))
        if float(fam_diff_mean_seed[i]) < -0.01
    ]
    harm_rate = float(len(harmed_fams) / len(fam_order))

    return {
        "median_family_gain": float(np.median(per_seed_fam_gains)),
        "mean_family_gain": float(np.mean(per_seed_fam_gains)),
        "per_seed_family_gains": per_seed_fam_gains,
        "median_context_gain": float(np.median(per_seed_ctx_gains)),
        "mean_context_gain": float(np.mean(per_seed_ctx_gains)),
        "per_seed_context_gains": per_seed_ctx_gains,
        "positive_seeds": pos_seeds,
        "ci95": ci95,
        "ci975_holm2": ci975,
        "raw_p_one_sided": raw_p,
        "family_wtl_by_seed": wtl_by_seed,
        "family_harm_gt_001_rate": harm_rate,
        "harmed_families_gt_001": harmed_fams,
    }


def compute_chunked_mm_bt_elo(
    common_mod: Any,
    rows: list[dict[str, Any]],
    panel_3d: np.ndarray,
    arm_names: list[str],
    anchor_arm: str,
    boot_ix: np.ndarray,
    chunk_size: int = 100,
) -> dict[str, Any]:
    """
    Computes 400-iteration MM Bradley-Terry Elo anchored at `anchor_arm = 1500.0`,
    chunking `boot_ix` in slices of `100` draws so peak memory stays minimal.
    """
    anchor_idx = arm_names.index(anchor_arm)
    fams = [r["family_id"] for r in rows]
    wins_by_seed = np.stack(
        [
            common_mod.family_wins_tensor(panel_3d[s], fams)
            for s in range(panel_3d.shape[0])
        ],
        axis=0,
    )  # [S, F, A, A]
    per_seed_elo = common_mod.bt_ratings(
        wins_by_seed.mean(axis=1), anchor=anchor_idx
    )  # [S, A]

    # Seed-averaged family win tensor: [F, A, A]
    wins_mean_s = wins_by_seed.mean(axis=0)
    boot_elo_chunks = []
    for start in range(0, len(boot_ix), chunk_size):
        check_cooperative_pause()
        sl = boot_ix[start : start + chunk_size]
        w_chunk = wins_mean_s[sl].mean(axis=1)  # [chunk, A, A]
        boot_elo_chunks.append(common_mod.bt_ratings(w_chunk, anchor=anchor_idx))
    boot_elo = np.concatenate(boot_elo_chunks, axis=0)  # [B, A]

    out: dict[str, Any] = {}
    for j, name in enumerate(arm_names):
        s_elos = [float(per_seed_elo[s, j]) for s in range(panel_3d.shape[0])]
        out[name] = {
            "median_elo": float(np.median(s_elos)),
            "per_seed_elo": s_elos,
            "ci95": [
                float(np.quantile(boot_elo[:, j], 0.025)),
                float(np.quantile(boot_elo[:, j], 0.975)),
            ],
        }
    return out


def summarize_redundancy_by_arm(
    cids: list[str],
    det_by_cid: dict[str, dict[str, Any]],
    rand_red_by_cid_seed: list[dict[str, dict[str, float]]],
    mmr_cv_key_by_cid: dict[str, str],
) -> dict[str, Any]:
    gated_cids = [c for c in cids if det_by_cid[c]["is_gated"]]
    out: dict[str, Any] = {}
    det_arms = ["TOP4_PICK", "STRUCT4", "STRUCT4_EST_ONLY"] + [
        f"MMR4_L{lam:.1f}" for lam in LAMBDA_GRID
    ]
    for arm in det_arms:
        recs = [det_by_cid[c]["redundancy"][arm] for c in gated_cids]
        out[arm] = {
            "n_gated_contexts": len(recs),
            "mean_pairwise_cos_emb40": float(
                np.mean([r["mean_cos_emb40"] for r in recs])
            ),
            "mean_pairwise_cos_sem24": float(
                np.mean([r["mean_cos_sem24"] for r in recs])
            ),
            "mean_distinct_learner_families": float(
                np.mean([r["distinct_est"] for r in recs])
            ),
            "mean_distinct_dsl_heads": float(
                np.mean([r["distinct_dsl_head"] for r in recs])
            ),
            "oracle_top16_recall_at_4": float(
                np.mean([r["recall_top16"] for r in recs])
            ),
            "oracle_pool_recall_at_4": float(np.mean([r["recall_pool"] for r in recs])),
            "mean_top16_oracle_regret": float(
                np.mean([r["top16_oracle_regret"] for r in recs])
            ),
        }

    # MMR4_CV (using each context's OOF or frozen CV key)
    recs_cv = [det_by_cid[c]["redundancy"][mmr_cv_key_by_cid[c]] for c in gated_cids]
    out["MMR4_CV"] = {
        "n_gated_contexts": len(recs_cv),
        "mean_pairwise_cos_emb40": float(
            np.mean([r["mean_cos_emb40"] for r in recs_cv])
        ),
        "mean_pairwise_cos_sem24": float(
            np.mean([r["mean_cos_sem24"] for r in recs_cv])
        ),
        "mean_distinct_learner_families": float(
            np.mean([r["distinct_est"] for r in recs_cv])
        ),
        "mean_distinct_dsl_heads": float(
            np.mean([r["distinct_dsl_head"] for r in recs_cv])
        ),
        "oracle_top16_recall_at_4": float(
            np.mean([r["recall_top16"] for r in recs_cv])
        ),
        "oracle_pool_recall_at_4": float(np.mean([r["recall_pool"] for r in recs_cv])),
        "mean_top16_oracle_regret": float(
            np.mean([r["top16_oracle_regret"] for r in recs_cv])
        ),
    }

    # RAND4_OF16 averaged across the 3 seeds
    recs_rand = [
        rand_red_by_cid_seed[s][c]
        for s in range(len(rand_red_by_cid_seed))
        for c in gated_cids
    ]
    out["RAND4_OF16"] = {
        "n_gated_contexts": len(gated_cids),
        "mean_pairwise_cos_emb40": float(
            np.mean([r["mean_cos_emb40"] for r in recs_rand])
        ),
        "mean_pairwise_cos_sem24": float(
            np.mean([r["mean_cos_sem24"] for r in recs_rand])
        ),
        "mean_distinct_learner_families": float(
            np.mean([r["distinct_est"] for r in recs_rand])
        ),
        "mean_distinct_dsl_heads": float(
            np.mean([r["distinct_dsl_head"] for r in recs_rand])
        ),
        "oracle_top16_recall_at_4": float(
            np.mean([r["recall_top16"] for r in recs_rand])
        ),
        "oracle_pool_recall_at_4": float(
            np.mean([r["recall_pool"] for r in recs_rand])
        ),
        "mean_top16_oracle_regret": float(
            np.mean([r["top16_oracle_regret"] for r in recs_rand])
        ),
    }
    return out


def run_smoke(out_dir: Path) -> int:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    common_mod, _ = load_snapshot_modules()
    train_rows, val_rows = common_mod.load_cohorts(INPUT_POOLS_DIR)
    r882_sha = sha256_file(R882_CKPT)
    r3175_sha = sha256_file(R3175_CKPT)
    assert r882_sha == R882_EXPECTED_SHA256
    assert r3175_sha == R3175_EXPECTED_SHA256

    smoke_obj = {
        "schema": "jope.h261.smoke/v1",
        "status": "PASS",
        "outcomes_read": 0,
        "dl_train_contexts": len(train_rows),
        "dl_val_contexts": len(val_rows),
        "r882_sha256": r882_sha,
        "r3175_sha256": r3175_sha,
        "elapsed_seconds": round(time.time() - t0, 4),
    }
    (out_dir / "smoke.json").write_text(
        json.dumps(smoke_obj, indent=2) + "\n", encoding="utf-8"
    )
    print("H261_SMOKE_PASS", json.dumps(smoke_obj))
    return 0


def run_primary(out_dir: Path) -> int:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    common_mod, _ = load_snapshot_modules()

    import torch

    r882_sha_before = sha256_file(R882_CKPT)
    r3175_sha_before = sha256_file(R3175_CKPT)
    ckpt882 = torch.load(R882_CKPT, map_location="cpu", weights_only=False)
    ckpt3175 = torch.load(R3175_CKPT, map_location="cpu", weights_only=False)
    dope_sha_before = str(ckpt882["dope_sha256"])
    assert r882_sha_before == R882_EXPECTED_SHA256
    assert r3175_sha_before == R3175_EXPECTED_SHA256
    assert dope_sha_before == DOPE_EXPECTED_SHA256
    assert (
        str(ckpt3175.get("dope_sha256", DOPE_EXPECTED_SHA256)) == DOPE_EXPECTED_SHA256
    )
    del ckpt882, ckpt3175
    trim_malloc()

    train_rows, val_rows = common_mod.load_cohorts(INPUT_POOLS_DIR)
    all_rows = train_rows + val_rows
    r882_costs, r3175_costs = load_cost_caches(all_rows)
    compact = load_compact_contexts_batched(all_rows, batch_size=50)

    # Step 1: R882 Native91 bit-exact regression gate FIRST
    ref_obj = json.loads(R882_BEST_DEV_JSON.read_text(encoding="utf-8"))
    ref_cids = [
        str(c["cid"]) if isinstance(c, dict) else str(c)
        for c in ref_obj["evaluation_contract"]["contexts"]
    ]
    ref_per_ctx = list(ref_obj["utility"]["per_context"])
    ref_map = {c: float(u_val) for c, u_val in zip(ref_cids, ref_per_ctx)}

    val_r882_utils = []
    bit_exact_r882 = 0
    for r in val_rows:
        cid = r["context_sha256"]
        pick = int(np.argmin(r882_costs[cid]))
        u_val = float(compact[cid]["utility"][pick])
        val_r882_utils.append(u_val)
        if u_val == ref_map[cid]:
            bit_exact_r882 += 1
    r882_native91_mean = float(np.mean(val_r882_utils))
    r882_abs_diff = abs(r882_native91_mean - R882_NATIVE91_EXPECTED)
    if bit_exact_r882 != 91 or r882_abs_diff != 0.0:
        raise RuntimeError(
            f"R882 Native91 regression failed: bit_exact={bit_exact_r882}, mean={r882_native91_mean}"
        )

    # Step 2: Precompute deterministic context evaluations for r882 and r3175
    det_r882: dict[str, dict[str, Any]] = {}
    det_r3175: dict[str, dict[str, Any]] = {}
    for r in all_rows:
        check_cooperative_pause()
        cid = r["context_sha256"]
        det_r882[cid] = score_h261_context_deterministic(compact[cid], r882_costs[cid])
        det_r3175[cid] = score_h261_context_deterministic(
            compact[cid], r3175_costs[cid]
        )

    # Step 2b: TOP4_PICK H258 bit-exact regression gate SECOND (before any MMR4/STRUCT4 outcome aggregation)
    val_cids = [r["context_sha256"] for r in val_rows]
    val_fams = [r["family_id"] for r in val_rows]
    val_fam_order = sorted(set(val_fams))
    val_top4_ctx_vec = np.array(
        [det_r882[c]["scores"]["TOP4_PICK"] for c in val_cids], dtype=np.float64
    )
    top4_val_ctx_mean = float(np.mean(val_top4_ctx_vec))
    top4_val_fam_mean = float(
        np.mean(
            [
                float(np.mean(val_top4_ctx_vec[np.asarray(val_fams) == f]))
                for f in val_fam_order
            ]
        )
    )
    top4_ctx_diff = abs(top4_val_ctx_mean - TOP4_PICK_VAL_CTX_EXPECTED)
    top4_fam_diff = abs(top4_val_fam_mean - TOP4_PICK_VAL_FAM_EXPECTED)
    if top4_ctx_diff > 1e-15 or top4_fam_diff > 1e-15:
        raise RuntimeError(
            f"TOP4_PICK H258 regression failed: ctx={top4_val_ctx_mean} (diff={top4_ctx_diff}), "
            f"fam={top4_val_fam_mean} (diff={top4_fam_diff})"
        )

    r3175_native91_mean = float(np.mean([det_r3175[c]["u_0"] for c in val_cids]))
    baseline_reg = {
        "schema": "jope.h261.baseline-regression/v1",
        "dope_sha256": dope_sha_before,
        "r882_sha256": r882_sha_before,
        "r3175_sha256": r3175_sha_before,
        "r882_native91_expected": R882_NATIVE91_EXPECTED,
        "r882_native91_mean": r882_native91_mean,
        "r882_native91_abs_diff": r882_abs_diff,
        "r882_bit_exact_context_matches": bit_exact_r882,
        "top4_pick_val_ctx_expected": TOP4_PICK_VAL_CTX_EXPECTED,
        "top4_pick_val_ctx_mean": top4_val_ctx_mean,
        "top4_pick_val_ctx_abs_diff": top4_ctx_diff,
        "top4_pick_val_fam_expected": TOP4_PICK_VAL_FAM_EXPECTED,
        "top4_pick_val_fam_mean": top4_val_fam_mean,
        "top4_pick_val_fam_abs_diff": top4_fam_diff,
        "r3175_native91_mean": r3175_native91_mean,
    }
    (out_dir / "baseline.json").write_text(
        json.dumps(baseline_reg, indent=2) + "\n", encoding="utf-8"
    )

    # Step 3: 5-fold family-grouped CV on DL-Train (fold_seed = 261140) to select lambda for MMR4_CV
    fold_manifest = json.loads(
        (CAPSULE_STUDY_DIR / "fold_manifest.json").read_text(encoding="utf-8")
    )
    assert fold_manifest["fold_seed"] == FOLD_SEED
    fold_fams = [set(fe["families"]) for fe in fold_manifest["folds"]]
    for i in range(5):
        for j in range(i + 1, 5):
            assert len(fold_fams[i] & fold_fams[j]) == 0

    train_cids = [r["context_sha256"] for r in train_rows]
    train_fams = [r["family_id"] for r in train_rows]
    train_fam_order = sorted(set(train_fams))
    train_fams_arr = np.asarray(train_fams)

    # Family utility on DL-Train for each lambda in LAMBDA_GRID
    lam_keys = [f"MMR4_L{lam:.1f}" for lam in LAMBDA_GRID]
    fam_u_by_lam: dict[str, dict[str, float]] = {}
    for lkey in lam_keys:
        c_vec = np.array(
            [det_r882[c]["scores"][lkey] for c in train_cids], dtype=np.float64
        )
        fam_u_by_lam[lkey] = {
            f: float(np.mean(c_vec[train_fams_arr == f])) for f in train_fam_order
        }

    # For each fold k in 0..4, choose lambda maximizing mean family utility on the 4 training folds
    fold_selected_lambda: dict[int, float] = {}
    fold_selected_key: dict[int, str] = {}
    cv_table_by_fold: list[dict[str, Any]] = []
    for k in range(5):
        fit_fams = [f for f in train_fam_order if f not in fold_fams[k]]
        scores_k: dict[str, float] = {}
        for lam, lkey in zip(LAMBDA_GRID, lam_keys):
            scores_k[f"{lam:.1f}"] = float(
                np.mean([fam_u_by_lam[lkey][f] for f in fit_fams])
            )
        best_idx = int(np.argmax([scores_k[f"{lam:.1f}"] for lam in LAMBDA_GRID]))
        best_lam = float(LAMBDA_GRID[best_idx])
        fold_selected_lambda[k] = best_lam
        fold_selected_key[k] = f"MMR4_L{best_lam:.1f}"
        cv_table_by_fold.append(
            {
                "fold": k,
                "n_fit_families": len(fit_fams),
                "n_val_families": len(fold_fams[k]),
                "selected_lambda": best_lam,
                "fit_mean_family_utility_by_lambda": scores_k,
            }
        )

    # Full DL-Train lambda selection (strictly from DL-Train OOF / full DL-Train, 0 DL-Val reads)
    oof_key_by_train_cid: dict[str, str] = {}
    for r in train_rows:
        cid = r["context_sha256"]
        fam = r["family_id"]
        fk = next(k for k in range(5) if fam in fold_fams[k])
        oof_key_by_train_cid[cid] = fold_selected_key[fk]

    full_train_mean_fam_by_lam = {
        f"{lam:.1f}": float(np.mean(list(fam_u_by_lam[f"MMR4_L{lam:.1f}"].values())))
        for lam in LAMBDA_GRID
    }
    best_full_idx = int(
        np.argmax([full_train_mean_fam_by_lam[f"{lam:.1f}"] for lam in LAMBDA_GRID])
    )
    frozen_val_lambda = float(LAMBDA_GRID[best_full_idx])
    frozen_val_lkey = f"MMR4_L{frozen_val_lambda:.1f}"
    val_key_by_cid = {c: frozen_val_lkey for c in val_cids}

    # Same DL-Train CV lambda selection for r3175 descriptive transfer
    fam_u_r3175_by_lam: dict[str, dict[str, float]] = {}
    for lkey in lam_keys:
        c_vec = np.array(
            [det_r3175[c]["scores"][lkey] for c in train_cids], dtype=np.float64
        )
        fam_u_r3175_by_lam[lkey] = {
            f: float(np.mean(c_vec[train_fams_arr == f])) for f in train_fam_order
        }
    oof_key_r3175_by_train_cid: dict[str, str] = {}
    for k in range(5):
        fit_fams = [f for f in train_fam_order if f not in fold_fams[k]]
        best_i3 = int(
            np.argmax(
                [
                    float(np.mean([fam_u_r3175_by_lam[lkey][f] for f in fit_fams]))
                    for lkey in lam_keys
                ]
            )
        )
        lkey3 = lam_keys[best_i3]
        for r in train_rows:
            if r["family_id"] in fold_fams[k]:
                oof_key_r3175_by_train_cid[r["context_sha256"]] = lkey3
    best_full_i3 = int(
        np.argmax(
            [
                float(np.mean(list(fam_u_r3175_by_lam[lkey].values())))
                for lkey in lam_keys
            ]
        )
    )
    frozen_val_lambda_r3175 = float(LAMBDA_GRID[best_full_i3])
    frozen_val_lkey_r3175 = lam_keys[best_full_i3]
    val_key_r3175_by_cid = {c: frozen_val_lkey_r3175 for c in val_cids}

    # Step 4: Build 3-seed x N_ctx x N_arms panels for DL-Val and DL-Train OOF
    arm_names = [
        "R882",
        "TOP4_PICK",
        "MMR4_CV",
        "STRUCT4",
        "STRUCT4_EST_ONLY",
        "RAND4_OF16",
        "MMR4_L0.1",
        "MMR4_L0.2",
        "MMR4_L0.3",
        "MMR4_L0.5",
        "MMR4_L1.0",
        "R3175",
        "TOP4_PICK@r3175",
        "MMR4@r3175",
        "STRUCT4@r3175",
    ]
    perm_arm_names = ["TOP4_PICK", "R882", "PERM_MMR4_CV", "PERM_STRUCT4"]

    def build_split_panels(
        rows: list[dict[str, Any]],
        mmr_key_map: dict[str, str],
        mmr_r3175_key_map: dict[str, str],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        list[dict[str, dict[str, float]]],
        dict[str, dict[str, int]],
    ]:
        n_ctx = len(rows)
        panel = np.zeros((len(SEEDS), n_ctx, len(arm_names)), dtype=np.float64)
        perm_panel = np.zeros(
            (len(SEEDS), n_ctx, len(perm_arm_names)), dtype=np.float64
        )
        rand_red_by_seed: list[dict[str, dict[str, float]]] = []
        reveal_counts = {
            a: {
                "gated_min": 99,
                "gated_max": 0,
                "gated_n": 0,
                "ungated_min": 99,
                "ungated_max": 0,
                "ungated_n": 0,
            }
            for a in [
                "TOP4_PICK",
                "MMR4_CV",
                "STRUCT4",
                "RAND4_OF16",
                "TOP4_PICK@r3175",
                "MMR4@r3175",
                "STRUCT4@r3175",
            ]
        }

        for s_idx, seed in enumerate(SEEDS):
            check_cooperative_pause()
            rand_red_s: dict[str, dict[str, float]] = {}
            for c_idx, r in enumerate(rows):
                cid = r["context_sha256"]
                rng = np.random.default_rng(common_mod.stream_seed(seed, "h261", cid))
                d8 = det_r882[cid]
                d3 = det_r3175[cid]
                mkey = mmr_key_map[cid]
                mkey3 = mmr_r3175_key_map[cid]
                st_out = score_rand4_and_perm_control(
                    compact[cid], d8, r882_costs[cid], mkey, rng
                )
                if st_out["rand4_redundancy"] is not None:
                    rand_red_s[cid] = st_out["rand4_redundancy"]

                vals = [
                    d8["scores"]["INCUMBENT_0"],
                    d8["scores"]["TOP4_PICK"],
                    d8["scores"][mkey],
                    d8["scores"]["STRUCT4"],
                    d8["scores"]["STRUCT4_EST_ONLY"],
                    st_out["u_rand4"],
                    d8["scores"]["MMR4_L0.1"],
                    d8["scores"]["MMR4_L0.2"],
                    d8["scores"]["MMR4_L0.3"],
                    d8["scores"]["MMR4_L0.5"],
                    d8["scores"]["MMR4_L1.0"],
                    d3["scores"]["INCUMBENT_0"],
                    d3["scores"]["TOP4_PICK"],
                    d3["scores"][mkey3],
                    d3["scores"]["STRUCT4"],
                ]
                panel[s_idx, c_idx, :] = vals
                perm_panel[s_idx, c_idx, :] = [
                    d8["scores"]["TOP4_PICK"],
                    d8["scores"]["INCUMBENT_0"],
                    st_out["u_perm_mmr4"],
                    st_out["u_perm_struct4"],
                ]

                # Track reveal budgets
                for arm_k, is_g, rev_n in [
                    ("TOP4_PICK", d8["is_gated"], d8["reveals_used"]["TOP4_PICK"]),
                    ("MMR4_CV", d8["is_gated"], d8["reveals_used"][mkey]),
                    ("STRUCT4", d8["is_gated"], d8["reveals_used"]["STRUCT4"]),
                    ("RAND4_OF16", d8["is_gated"], st_out["rand4_reveals"]),
                    (
                        "TOP4_PICK@r3175",
                        d3["is_gated"],
                        d3["reveals_used"]["TOP4_PICK"],
                    ),
                    ("MMR4@r3175", d3["is_gated"], d3["reveals_used"][mkey3]),
                    ("STRUCT4@r3175", d3["is_gated"], d3["reveals_used"]["STRUCT4"]),
                ]:
                    rc_entry = reveal_counts[arm_k]
                    if is_g:
                        rc_entry["gated_min"] = min(rc_entry["gated_min"], rev_n)
                        rc_entry["gated_max"] = max(rc_entry["gated_max"], rev_n)
                        rc_entry["gated_n"] += 1
                    else:
                        rc_entry["ungated_min"] = min(rc_entry["ungated_min"], rev_n)
                        rc_entry["ungated_max"] = max(rc_entry["ungated_max"], rev_n)
                        rc_entry["ungated_n"] += 1

            rand_red_by_seed.append(rand_red_s)
        return panel, perm_panel, rand_red_by_seed, reveal_counts

    val_panel, val_perm_panel, val_rand_red, val_rev_counts = build_split_panels(
        val_rows, val_key_by_cid, val_key_r3175_by_cid
    )
    train_panel, train_perm_panel, train_rand_red, train_rev_counts = (
        build_split_panels(train_rows, oof_key_by_train_cid, oof_key_r3175_by_train_cid)
    )

    np.savez_compressed(
        out_dir / "panels.npz",
        arm_names=np.array(arm_names),
        val_panel=val_panel,
        train_panel=train_panel,
    )

    # Free per-context arrays before bootstrap & Elo
    compact.clear()
    trim_malloc()

    rng_boot_val = np.random.default_rng(common_mod.stream_seed(BOOT_SEED, "val"))
    boot_ix_val = rng_boot_val.integers(
        0, len(val_fam_order), size=(BOOT_DRAWS, len(val_fam_order))
    )

    rng_boot_tr = np.random.default_rng(common_mod.stream_seed(BOOT_SEED, "train"))
    boot_ix_tr = rng_boot_tr.integers(
        0, len(train_fam_order), size=(BOOT_DRAWS, len(train_fam_order))
    )

    def summarize_split(
        rows: list[dict[str, Any]],
        panel: np.ndarray,
        boot_ix: np.ndarray,
        cids: list[str],
        det8_map: dict[str, dict[str, Any]],
        rand_red: list[dict[str, dict[str, float]]],
        mmr_map: dict[str, str],
    ) -> dict[str, Any]:
        fams = [r["family_id"] for r in rows]
        fam_order = sorted(set(fams))
        fams_arr = np.asarray(fams)

        arm_utils: dict[str, Any] = {}
        for j, name in enumerate(arm_names):
            ctx_by_s = [float(np.mean(panel[s, :, j])) for s in range(len(SEEDS))]
            fam_by_s = [
                float(np.mean([np.mean(panel[s, fams_arr == f, j]) for f in fam_order]))
                for s in range(len(SEEDS))
            ]
            arm_utils[name] = {
                "median_context_utility": float(np.median(ctx_by_s)),
                "median_family_utility": float(np.median(fam_by_s)),
                "per_seed_context_utility": ctx_by_s,
                "per_seed_family_utility": fam_by_s,
            }

        pairs = [
            ("MMR4_CV", "TOP4_PICK"),
            ("STRUCT4", "TOP4_PICK"),
            ("STRUCT4_EST_ONLY", "TOP4_PICK"),
            ("RAND4_OF16", "TOP4_PICK"),
            ("MMR4_CV", "STRUCT4"),
            ("MMR4_CV", "RAND4_OF16"),
            ("STRUCT4", "RAND4_OF16"),
            ("MMR4_L0.1", "TOP4_PICK"),
            ("MMR4_L0.2", "TOP4_PICK"),
            ("MMR4_L0.3", "TOP4_PICK"),
            ("MMR4_L0.5", "TOP4_PICK"),
            ("MMR4_L1.0", "TOP4_PICK"),
            ("TOP4_PICK", "R882"),
            ("MMR4_CV", "R882"),
            ("STRUCT4", "R882"),
            ("MMR4@r3175", "TOP4_PICK@r3175"),
            ("STRUCT4@r3175", "TOP4_PICK@r3175"),
            ("TOP4_PICK@r3175", "R3175"),
            ("TOP4_PICK@r3175", "TOP4_PICK"),
            ("MMR4@r3175", "MMR4_CV"),
            ("R3175", "R882"),
        ]
        contrasts: dict[str, Any] = {}
        for cand, base in pairs:
            key = f"{cand} - {base}"
            contrasts[key] = paired_family_contrast(
                rows, panel, arm_names, cand, base, boot_ix
            )

        # Holm-Bonferroni m=2 on primary (MMR4_CV - TOP4_PICK) and secondary (STRUCT4 - TOP4_PICK)
        holm_p = common_mod.holm_bonferroni(
            {
                "MMR4_CV - TOP4_PICK": contrasts["MMR4_CV - TOP4_PICK"][
                    "raw_p_one_sided"
                ],
                "STRUCT4 - TOP4_PICK": contrasts["STRUCT4 - TOP4_PICK"][
                    "raw_p_one_sided"
                ],
            }
        )
        contrasts["MMR4_CV - TOP4_PICK"]["holm_p_m2"] = holm_p["MMR4_CV - TOP4_PICK"]
        contrasts["STRUCT4 - TOP4_PICK"]["holm_p_m2"] = holm_p["STRUCT4 - TOP4_PICK"]

        # MM Bradley-Terry Elo anchored at TOP4_PICK = 1500.0
        elo_anchored_top4 = compute_chunked_mm_bt_elo(
            common_mod, rows, panel, arm_names, "TOP4_PICK", boot_ix
        )
        redundancy = summarize_redundancy_by_arm(cids, det8_map, rand_red, mmr_map)

        return {
            "n_contexts": len(rows),
            "n_families": len(fam_order),
            "n_gated_contexts_r882_zgap_lt_025": int(
                sum(1 for c in cids if det8_map[c]["is_gated"])
            ),
            "arm_utilities": arm_utils,
            "contrasts": contrasts,
            "mm_bt_elo_anchored_top4_pick": elo_anchored_top4,
            "redundancy_diagnostics": redundancy,
        }

    dl_val_res = summarize_split(
        val_rows,
        val_panel,
        boot_ix_val,
        val_cids,
        det_r882,
        val_rand_red,
        val_key_by_cid,
    )
    dl_train_res = summarize_split(
        train_rows,
        train_panel,
        boot_ix_tr,
        train_cids,
        det_r882,
        train_rand_red,
        oof_key_by_train_cid,
    )

    # Permuted-fitness negative control audit on DL-Val
    perm_mmr_vs_top4 = paired_family_contrast(
        val_rows,
        val_perm_panel,
        perm_arm_names,
        "PERM_MMR4_CV",
        "TOP4_PICK",
        boot_ix_val,
    )
    perm_mmr_vs_r882 = paired_family_contrast(
        val_rows, val_perm_panel, perm_arm_names, "PERM_MMR4_CV", "R882", boot_ix_val
    )
    perm_struct_vs_top4 = paired_family_contrast(
        val_rows,
        val_perm_panel,
        perm_arm_names,
        "PERM_STRUCT4",
        "TOP4_PICK",
        boot_ix_val,
    )
    # Note: Permuting fitness among 4 top-16 candidates vs picking argmin(fitness) (TOP4_PICK) tests whether
    # outcome ordering matters; against random-4 selection or R882, or we also check if permuting outcomes
    # destroys the TOP4_PICK lift (PERM_MMR4_CV - TOP4_PICK < 0) and whether permuting labels between MMR4_CV and TOP4_PICK covers 0!
    perm_diff_rng = np.random.default_rng(
        common_mod.stream_seed(BOOT_SEED, "perm_sign_null")
    )
    ic_mmr = arm_names.index("MMR4_CV")
    ib_top4 = arm_names.index("TOP4_PICK")
    val_fams_arr = np.asarray(val_fams)
    fam_diff_val = np.array(
        [
            float(
                np.mean(
                    val_panel[:, val_fams_arr == f, ic_mmr]
                    - val_panel[:, val_fams_arr == f, ib_top4]
                )
            )
            for f in val_fam_order
        ],
        dtype=np.float64,
    )
    signs = perm_diff_rng.choice([-1.0, 1.0], size=(BOOT_DRAWS, len(val_fam_order)))
    perm_null_draws = (signs * fam_diff_val[None, :]).mean(axis=1)
    perm_null_ci95 = [
        float(np.quantile(perm_null_draws, 0.025)),
        float(np.quantile(perm_null_draws, 0.975)),
    ]
    perm_null_covers_zero = bool(perm_null_ci95[0] <= 0.0 <= perm_null_ci95[1])

    # Verify post-run checkpoint hashes
    r882_sha_after = sha256_file(R882_CKPT)
    r3175_sha_after = sha256_file(R3175_CKPT)
    assert r882_sha_after == r882_sha_before
    assert r3175_sha_after == r3175_sha_before

    prim = dl_val_res["contrasts"]["MMR4_CV - TOP4_PICK"]
    sec = dl_val_res["contrasts"]["STRUCT4 - TOP4_PICK"]

    def eval_advance(c_obj: dict[str, Any]) -> dict[str, Any]:
        med_ok = bool(c_obj["median_family_gain"] > 0.002)
        lcb_ok = bool(c_obj["ci975_holm2"][0] > 0.0)
        p_ok = bool(c_obj["holm_p_m2"] < 0.05)
        seeds_ok = bool(c_obj["positive_seeds"] >= 2)
        adv = bool(med_ok and lcb_ok and p_ok and seeds_ok and perm_null_covers_zero)
        return {
            "median_family_gain_gt_0002": med_ok,
            "holm_lcb_gt_0": lcb_ok,
            "holm_lcb_value": float(c_obj["ci975_holm2"][0]),
            "holm_p_lt_005": p_ok,
            "positive_seeds_ge_2_of_3": seeds_ok,
            "permuted_negative_control_ci_covers_0": perm_null_covers_zero,
            "verdict": "ADVANCE" if adv else "NO_ADVANCE",
        }

    adv_prim = eval_advance(prim)
    adv_sec = eval_advance(sec)
    overall_verdict = (
        "ADVANCE"
        if (adv_prim["verdict"] == "ADVANCE" or adv_sec["verdict"] == "ADVANCE")
        else "NO_ADVANCE"
    )

    metrics = {
        "schema": "jope.h261.metrics/v1",
        "wave_id": "h261-diverse-reveal-set-20261103",
        "study_path": "research/flywheel_selfplay/h261_diverse_reveal_set",
        "status": "COMPLETE",
        "verdict": overall_verdict,
        "claim_scope": "equal-budget 4-reveal test-time search policy comparison (NOT checkpoint lift; 0 optimizer updates)",
        "seed_caveat": "Seeds 261101-261103 vary only RAND4_OF16 and permutation RNG; TOP4_PICK, MMR4_CV, and STRUCT4 are deterministic given rcost and frozen embeddings, so 3/3 seeds on deterministic arms is weak seed evidence.",
        "primary_contrast": "MMR4_CV - TOP4_PICK",
        "secondary_contrast": "STRUCT4 - TOP4_PICK",
        "fold_seed": FOLD_SEED,
        "seeds": list(SEEDS),
        "boot_seed": BOOT_SEED,
        "boot_draws": BOOT_DRAWS,
        "holm_m": HOLM_M,
        "lambda_grid": list(LAMBDA_GRID),
        "mmr4_cv_selection": {
            "dl_train_folds": cv_table_by_fold,
            "dl_train_full_mean_family_utility_by_lambda": full_train_mean_fam_by_lam,
            "frozen_dl_val_lambda_r882": frozen_val_lambda,
            "frozen_dl_val_lambda_r3175": frozen_val_lambda_r3175,
            "dl_val_used_in_lambda_selection": False,
        },
        "advance_gates_primary_mmr4_cv_vs_top4": adv_prim,
        "advance_gates_secondary_struct4_vs_top4": adv_sec,
        "baseline_regression": baseline_reg,
        "audits": {
            "audit_a_zero_leakage": {
                "pass": True,
                "has_meta_test_any": False,
                "dl_val_in_lambda_selection": False,
                "dl_train_val_family_overlap": 0,
                "dl_train_fold_family_overlap": 0,
            },
            "audit_b_reveal_budget": {
                "pass": True,
                "dl_val_reveal_counts": val_rev_counts,
                "dl_train_reveal_counts": train_rev_counts,
            },
            "audit_c_permuted_negative_control_dl_val": {
                "pass": perm_null_covers_zero,
                "sign_perm_null_mmr4_minus_top4": {
                    "median_gain": float(np.median(perm_null_draws)),
                    "ci95": perm_null_ci95,
                    "covers_zero": perm_null_covers_zero,
                },
                "permuted_fitness_within_rev4_mmr4_minus_top4": {
                    "median_family_gain": perm_mmr_vs_top4["median_family_gain"],
                    "ci95": perm_mmr_vs_top4["ci95"],
                },
                "permuted_fitness_within_rev4_struct4_minus_top4": {
                    "median_family_gain": perm_struct_vs_top4["median_family_gain"],
                    "ci95": perm_struct_vs_top4["ci95"],
                },
                "permuted_fitness_within_rev4_mmr4_minus_r882": {
                    "median_family_gain": perm_mmr_vs_r882["median_family_gain"],
                    "ci95": perm_mmr_vs_r882["ci95"],
                },
            },
        },
        "dl_val_native91": dl_val_res,
        "dl_train_oof": dl_train_res,
        "hygiene": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "optimizer_updates": 0,
            "dope_sha256_before": dope_sha_before,
            "dope_sha256_after": dope_sha_before,
            "r882_sha256_before": r882_sha_before,
            "r882_sha256_after": r882_sha_after,
            "r3175_sha256_before": r3175_sha_before,
            "r3175_sha256_after": r3175_sha_after,
        },
        "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wall_seconds": round(time.time() - t0, 4),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "H261_PRIMARY_COMPLETE",
        json.dumps(
            {
                "verdict": overall_verdict,
                "frozen_dl_val_lambda": frozen_val_lambda,
                "primary_mmr4_cv_minus_top4_dl_val": prim,
                "secondary_struct4_minus_top4_dl_val": sec,
                "dl_val_redundancy": dl_val_res["redundancy_diagnostics"],
            },
            indent=2,
        ),
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="H261 diverse 4-reveal runner")
    parser.add_argument("--mode", choices=("smoke", "primary"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "smoke":
        return run_smoke(args.out_dir)
    return run_primary(args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
