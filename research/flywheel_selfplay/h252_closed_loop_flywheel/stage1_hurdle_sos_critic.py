#!/usr/bin/env python3
"""
H252 Stage 1 (xbabe2 CPU): Decomposed Continuous Hurdle + LOPO SoS Critic,
Self-Play Promotion Prior, Outcome-Free z-Gap Gate, and Active 2+2 Reveal (W3 + W4).

Produces:
  1. Exact R882 baseline verification (Native91 = 1.1377941898164525, 91/91 bit-exact)
     and caches R882 total_cost + anchor_total + anchor_prior for all 1,310 contexts to
     `/home/ubuntu/jope-data/h252/stage1/r882_cached_costs.npz`.
  2. Cross-fitted 2-channel Hurdle SoS Prior `[hurdle_cost_z, gate_fail_z]` of shape
     `(1152, 2)` for all 1,310 contexts (1,219 DL-Train + 91 DL-Val), saved to
     `/home/ubuntu/jope-data/h252/stage1/stage1_hurdle_priors.npz` to feed the previously
     dead `sos_prior` columns in `FlywheelBlockFormerAdapterNet` (Stage 2 on xbabe1 GPU).
  3. Full 5-fold (205 DL-Train families) + 91 DL-Val evaluation across 3 seeds
     (252101, 252102, 252103) with 10,000-draw paired family bootstrap, Holm-Bonferroni,
     and 400-iteration MM Bradley-Terry Elo.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from common_h252 import (
    HARVEST_PROMOS,
    INPUT_DIR,
    R882_CKPT,
    R882_NATIVE91_MEAN,
    SEEDS,
    cell_folds,
    load_cohorts,
    load_context_features,
    load_harvested_promotions,
    load_pools,
    save_json,
    seed_folds,
    stream_seed,
    summarize_panel,
)

STAGE1_OUT_DIR = Path("/home/ubuntu/jope-data/h252/stage1")
REPO_ROOT = Path("/home/ubuntu/audit/JopeDime_main")
R882_TRAIN_COSTS_NPZ = Path("/home/ubuntu/jope-data/h245/run-001/train/r882-train-costs.npz")
R882_VAL_COSTS_NPZ = Path("/home/ubuntu/jope-data/h245/run-001/regression/r882-secondary-costs.npz")
POOL_PRIORS_NPZ = Path("/home/ubuntu/jope-data/h58/cache/pool_priors.npz")


def pad_to_1152(arr: np.ndarray, pad_val: float = 0.0) -> np.ndarray:
    if len(arr) == 1152:
        return arr
    pad_len = 1152 - len(arr)
    if arr.ndim == 1:
        return np.pad(arr, (0, pad_len), mode="constant", constant_values=pad_val)
    return np.pad(arr, ((0, pad_len), (0, 0)), mode="constant", constant_values=pad_val)


def get_or_compute_r882_cache(
    all_rows: List[Dict[str, Any]],
    pools: Dict[str, Dict[str, Any]],
    ctx_feats: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Loads exact R882 `total_cost` and `base_prior_cost` across all 1,310 contexts,
    verifying bit-exact `R882_NATIVE91_MEAN`.
    """
    cache_path = STAGE1_OUT_DIR / "r882_cached_costs.npz"
    cids_sorted = sorted(r["context_sha256"] for r in all_rows)

    out: Dict[str, Dict[str, np.ndarray]] = {}
    with np.load(R882_TRAIN_COSTS_NPZ, allow_pickle=False) as zt, np.load(
        R882_VAL_COSTS_NPZ, allow_pickle=False
    ) as zv, np.load(POOL_PRIORS_NPZ, allow_pickle=False) as zp:
        pri_cids = [str(c) for c in zp["cids"]]
        offsets = zp["offsets"]
        base_pri = zp["base_prior_cost"]
        pri_map = {
            pri_cids[i]: base_pri[offsets[i] : offsets[i + 1]].astype(np.float32)
            for i in range(len(pri_cids))
        }
        for cid in cids_sorted:
            tc = (zt[cid] if cid in zt.files else zv[cid]).astype(np.float32)
            bp = pri_map[cid]
            out[cid] = {
                "total_cost": tc,
                "anchor_total": tc,
                "anchor_prior": bp,
            }

    STAGE1_OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        cids=np.array(cids_sorted),
        total_cost=np.stack([pad_to_1152(out[c]["total_cost"], 999.0) for c in cids_sorted], axis=0),
        anchor_total=np.stack([pad_to_1152(out[c]["anchor_total"], 999.0) for c in cids_sorted], axis=0),
        anchor_prior=np.stack([pad_to_1152(out[c]["anchor_prior"], 0.0) for c in cids_sorted], axis=0),
    )
    return out


def build_candidate_features(
    all_rows: List[Dict[str, Any]],
    pools: Dict[str, Dict[str, Any]],
    r882_cache: Dict[str, Dict[str, np.ndarray]],
    ctx_feats: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """
    Constructs within-pool standardized structural + semantic + R882 + context interaction
    feature matrix `X_c` of shape `(1152, 64)` float32 for every context `c`.
    Note: Features are strictly outcome-free (use only `program`, `sem24`, `r882` cost/anchor,
    and `ctx_feats`, never candidate `fitness`, `logratio`, or `gate`).
    """
    feats_by_cid: Dict[str, np.ndarray] = {}
    for r in all_rows:
        cid = r["context_sha256"]
        p = pools[cid]
        prog = p["program"].astype(np.float64)  # (1152, 412)
        sem24 = p["sem24"].astype(np.float64)  # (1152, 24)
        rcost = r882_cache[cid]["total_cost"].astype(np.float64)
        aprior = r882_cache[cid]["anchor_prior"].astype(np.float64)
        adelta = rcost - r882_cache[cid]["anchor_total"].astype(np.float64)
        lm48 = ctx_feats[cid]["lm48_z"].astype(np.float64)

        # 1. R882 cost & anchor features (lower cost = better)
        r_sd = float(np.std(rcost)) + 1e-6
        r_z = (rcost - float(np.mean(rcost))) / r_sd
        r_min = float(np.min(rcost))
        r_top8 = float(np.partition(rcost, 8)[8])
        r_gap1 = (rcost - r_min) / r_sd
        r_gap8 = (rcost - r_top8) / r_sd
        ap_z = (aprior - float(np.mean(aprior))) / (float(np.std(aprior)) + 1e-6)
        ad_z = (adelta - float(np.mean(adelta))) / (float(np.std(adelta)) + 1e-6)
        ranks = np.argsort(np.argsort(rcost)).astype(np.float64) / float(len(rcost))

        # 2. Semantic distances to R882 top-1, top-8 centroid, and initial 4 reveals (0..3)
        top1_idx = int(np.argmin(rcost))
        top8_idx = np.argsort(rcost)[:8]
        cos_top1 = np.clip(sem24 @ sem24[top1_idx], -1.0, 1.0)
        sem_top8 = sem24[top8_idx].mean(axis=0)
        sem_top8 /= max(float(np.linalg.norm(sem_top8)), 1e-6)
        cos_top8 = np.clip(sem24 @ sem_top8, -1.0, 1.0)
        cos_rev4 = np.clip(sem24 @ sem24[:4].T, -1.0, 1.0)
        max_cos_rev4 = cos_rev4.max(axis=1)
        mean_cos_rev4 = cos_rev4.mean(axis=1)

        # 3. Standardized semantic & structural coordinates
        sem_mean = sem24.mean(axis=0, keepdims=True)
        sem_std = sem24.std(axis=0, keepdims=True) + 1e-5
        sem_z = np.clip((sem24 - sem_mean) / sem_std, -5.0, 5.0)  # (1152, 24)

        struct_slice = np.concatenate([prog[:, :12], prog[:, 48:56]], axis=1)  # (1152, 20)
        st_mean = struct_slice.mean(axis=0, keepdims=True)
        st_std = struct_slice.std(axis=0, keepdims=True) + 1e-5
        st_z = np.clip((struct_slice - st_mean) / st_std, -5.0, 5.0)  # (1152, 20)

        # 4. Context x candidate rank interactions (10D)
        ctx_top5 = lm48[:5]  # (5,)
        inter_r = (r_z[:, None] * ctx_top5[None, :]).reshape(len(rcost), 5)
        inter_ap = (ap_z[:, None] * ctx_top5[None, :]).reshape(len(rcost), 5)

        head10 = np.stack(
            [
                r_z,
                r_gap1,
                r_gap8,
                ap_z,
                ad_z,
                ranks,
                cos_top1,
                cos_top8,
                max_cos_rev4,
                mean_cos_rev4,
            ],
            axis=1,
        )  # (1152, 10)

        X_c = np.concatenate([head10, sem_z, st_z, inter_r, inter_ap], axis=1).astype(np.float32)  # (1152, 64)
        feats_by_cid[cid] = X_c
    return feats_by_cid


def fit_decomposed_hurdle_and_sp_priors(
    train_rows: List[Dict[str, Any]],
    val_rows: List[Dict[str, Any]],
    pools: Dict[str, Dict[str, Any]],
    cand_feats: Dict[str, np.ndarray],
    promos_by_cid: Dict[str, List[Dict[str, Any]]],
    seed: int = SEEDS[0],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Fits the 5-fold cross-fitted Decomposed Continuous Hurdle Critic + Self-Play Promotion Prior:
      - Head 1 (`cont_cost_z`): Trained strictly on continuous `logratio` (where valid_cont is True,
        free of the +2.0 gate cliff!) predicting within-pool standardized continuous log-RMSE ratio.
      - Head 2 (`gate_fail_z`): Trained on `~gate` (gate failure risk log-odds proxy) so higher means
        higher risk of CV gate failure.
      - Head 3 (`sp_promo_cost_z`): Trained on `sem24` alignment with harvested GP self-play winners
        in the training folds (`3,449` accepted promotions across `205` DL-Train families).

    Both channels in `sos_priors[cid]` (`(1152, 2)`) are standardized cost-direction priors
    (lower = better candidate), matching `r882` cost orientation for Stage 2's `sos_prior` input!
    """
    _, cid_to_fold = seed_folds(train_rows, seed)
    dim = 65  # 64 features + bias

    xtx_cont = {f: np.zeros((dim, dim), dtype=np.float64) for f in range(5)}
    xty_cont = {f: np.zeros(dim, dtype=np.float64) for f in range(5)}
    xtx_gate = {f: np.zeros((dim, dim), dtype=np.float64) for f in range(5)}
    xty_gate = {f: np.zeros(dim, dtype=np.float64) for f in range(5)}
    sp_sem_by_fold = {f: np.zeros(24, dtype=np.float64) for f in range(5)}
    sp_cnt_by_fold = {f: 0 for f in range(5)}

    for r in train_rows:
        cid = r["context_sha256"]
        f = cid_to_fold[cid]
        p = pools[cid]
        X_raw = cand_feats[cid].astype(np.float64)
        X = np.concatenate([X_raw, np.ones((len(X_raw), 1), dtype=np.float64)], axis=1)

        # Continuous logratio target (lower = better), using valid continuous entries (no +2.0 cliff!)
        lr = np.clip(p["logratio"], -1.5, 1.5)
        valid = p["valid_cont"]
        if np.sum(valid) > 16:
            Xv = X[valid]
            lv = lr[valid]
            lv_sd = float(np.std(lv))
            if lv_sd > 1e-6:
                y_c = (lv - float(np.mean(lv))) / lv_sd
                xtx_cont[f] += Xv.T @ Xv
                xty_cont[f] += Xv.T @ y_c

        # Gate failure target (1 = failed gate, 0 = passed gate)
        g_fail = (~p["gate"]).astype(np.float64)
        gf_bar = float(np.clip(np.mean(g_fail), 0.02, 0.98))
        y_g = (g_fail - gf_bar) / np.sqrt(gf_bar * (1.0 - gf_bar))
        xtx_gate[f] += X.T @ X
        xty_gate[f] += X.T @ y_g

        # Accumulate self-play winner semantic direction relative to pool mean
        if cid in promos_by_cid:
            pool_sem_mean = p["sem24"].astype(np.float64).mean(axis=0)
            for pr in promos_by_cid[cid]:
                sp_sem_by_fold[f] += (pr["sem24"].astype(np.float64) - pool_sem_mean)
                sp_cnt_by_fold[f] += 1

    eye = np.eye(dim, dtype=np.float64)
    eye[-1, -1] = 1e-4
    reg_c = 600.0
    reg_g = 900.0

    xtx_c_all = sum(xtx_cont.values())
    xty_c_all = sum(xty_cont.values())
    xtx_g_all = sum(xtx_gate.values())
    xty_g_all = sum(xty_gate.values())
    sp_sem_all = sum(sp_sem_by_fold.values())

    w_c_oof = {}
    w_g_oof = {}
    sp_dir_oof = {}
    for f in range(5):
        w_c_oof[f] = np.linalg.solve((xtx_c_all - xtx_cont[f]) + reg_c * eye, xty_c_all - xty_cont[f])
        w_g_oof[f] = np.linalg.solve((xtx_g_all - xtx_gate[f]) + reg_g * eye, xty_g_all - xty_gate[f])
        v_sp = sp_sem_all - sp_sem_by_fold[f]
        sp_dir_oof[f] = v_sp / max(float(np.linalg.norm(v_sp)), 1e-6)

    w_c_full = np.linalg.solve(xtx_c_all + reg_c * eye, xty_c_all)
    w_g_full = np.linalg.solve(xtx_g_all + reg_g * eye, xty_g_all)
    sp_dir_full = sp_sem_all / max(float(np.linalg.norm(sp_sem_all)), 1e-6)

    sos_priors: Dict[str, np.ndarray] = {}
    sp_cost_priors: Dict[str, np.ndarray] = {}
    train_cont_corrs: List[float] = []
    train_gate_corrs: List[float] = []
    val_cont_corrs: List[float] = []
    val_gate_corrs: List[float] = []

    for r in train_rows:
        cid = r["context_sha256"]
        f = cid_to_fold[cid]
        p = pools[cid]
        X_raw = cand_feats[cid].astype(np.float64)
        X = np.concatenate([X_raw, np.ones((len(X_raw), 1), dtype=np.float64)], axis=1)
        pc = X @ w_c_oof[f]
        pg = X @ w_g_oof[f]
        pc_z = (pc - float(np.mean(pc))) / (float(np.std(pc)) + 1e-6)
        pg_z = (pg - float(np.mean(pg))) / (float(np.std(pg)) + 1e-6)
        sos_priors[cid] = np.stack([pc_z, pg_z], axis=1).astype(np.float32)

        # Self-play semantic alignment cost (negative alignment with SP winner direction)
        sp_proj = -(p["sem24"].astype(np.float64) @ sp_dir_oof[f])
        sp_cost_priors[cid] = ((sp_proj - float(np.mean(sp_proj))) / (float(np.std(sp_proj)) + 1e-6)).astype(np.float32)

        lr = np.clip(p["logratio"], -1.5, 1.5)
        if np.std(lr) > 1e-6:
            c_corr = float(np.corrcoef(pc_z, lr)[0, 1])
            if np.isfinite(c_corr):
                train_cont_corrs.append(c_corr)
        gf = (~p["gate"]).astype(np.float64)
        if 0 < np.sum(gf) < len(gf):
            g_corr = float(np.corrcoef(pg_z, gf)[0, 1])
            if np.isfinite(g_corr):
                train_gate_corrs.append(g_corr)

    for r in val_rows:
        cid = r["context_sha256"]
        p = pools[cid]
        X_raw = cand_feats[cid].astype(np.float64)
        X = np.concatenate([X_raw, np.ones((len(X_raw), 1), dtype=np.float64)], axis=1)
        pc = X @ w_c_full
        pg = X @ w_g_full
        pc_z = (pc - float(np.mean(pc))) / (float(np.std(pc)) + 1e-6)
        pg_z = (pg - float(np.mean(pg))) / (float(np.std(pg)) + 1e-6)
        sos_priors[cid] = np.stack([pc_z, pg_z], axis=1).astype(np.float32)

        sp_proj = -(p["sem24"].astype(np.float64) @ sp_dir_full)
        sp_cost_priors[cid] = ((sp_proj - float(np.mean(sp_proj))) / (float(np.std(sp_proj)) + 1e-6)).astype(np.float32)

        lr = np.clip(p["logratio"], -1.5, 1.5)
        if np.std(lr) > 1e-6:
            c_corr = float(np.corrcoef(pc_z, lr)[0, 1])
            if np.isfinite(c_corr):
                val_cont_corrs.append(c_corr)
        gf = (~p["gate"]).astype(np.float64)
        if 0 < np.sum(gf) < len(gf):
            g_corr = float(np.corrcoef(pg_z, gf)[0, 1])
            if np.isfinite(g_corr):
                val_gate_corrs.append(g_corr)

    diag = {
        "train_oof_mean_cont_logratio_corr": float(np.mean(train_cont_corrs)),
        "train_oof_mean_gate_fail_corr": float(np.mean(train_gate_corrs)),
        "val_mean_cont_logratio_corr": float(np.mean(val_cont_corrs)),
        "val_mean_gate_fail_corr": float(np.mean(val_gate_corrs)),
        "sp_promotions_used": int(sum(sp_cnt_by_fold.values())),
    }
    return sos_priors, sp_cost_priors, diag


def compute_r882_top_zgap(rcost: np.ndarray) -> float:
    """Outcome-free margin between R882's best and second-best candidate in standard-deviation units."""
    part = np.partition(rcost, 1)
    sd = float(np.std(rcost))
    if sd <= 1e-12:
        return 999.0
    return float((part[1] - part[0]) / sd)


def score_arms_for_context(
    cid: str,
    pool: Dict[str, Any],
    rcost: np.ndarray,
    sos_prior: np.ndarray,
    sp_prior: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    Evaluates all 5 Stage-1 arms on context `cid` for one seeded reveal stream:
      1. `R882`: Frozen R882 baseline (`argmin(rcost)`).
      2. `GATED_025`: H217 passive 4-reveal `sem24` kernel correction gated by `zgap < 0.25`.
      3. `HURDLE_SOS_PRIOR`: Outcome-free Decomposed Hurdle + SP Prior correction gated by `zgap < 0.06`.
      4. `GATED_HURDLE_SP`: Passive 4-reveal combining Decomposed Continuous Hurdle residual on `idx4`
         + multi-scale `sem24` kernel + SP prior + outcome-free `zgap` gate (`zgap < 0.25`).
      5. `ACTIVE_2PLUS2_HURDLE`: W4 Sequential 2+2 Active Reveal under the exact 4-reveal budget:
         reveals 2 initial probes `idx4[:2]`, fits interim Hurdle + `sem24` posterior, actively
         reveals the top-2 contenders/uncertainty probes `p1, p2`, and selects the best overall!
    Returns utility `u` (higher is better) for each arm.
    """
    u = pool["utility"]
    fitness = pool["fitness"]
    logratio = pool["logratio"]
    gate = pool["gate"]
    sem24 = pool["sem24"].astype(np.float64)
    n = len(rcost)

    idx4 = rng.choice(n, size=4, replace=False)
    top_zgap = compute_r882_top_zgap(rcost)

    # 1. R882
    pick_r882 = int(np.argmin(rcost))
    u_r882 = float(u[pick_r882])

    # Hurdle Gate Feasibility Safety Filter: never promote a challenger whose cross-fitted
    # gate-failure risk (`sos_prior[:, 1]`) exceeds R882's top-1 candidate's risk + 0.10
    gate_risk = sos_prior[:, 1].astype(np.float64)
    unsafe_gate_mask = gate_risk > (gate_risk[pick_r882] + 0.10)
    gate_safety_penalty = np.where(unsafe_gate_mask, 10.0, 0.0)

    # Hurdle combined cost prior (channel 0 = cont logratio z, channel 1 = gate fail z)
    h_cost = (
        0.65 * sos_prior[:, 0].astype(np.float64)
        + 0.35 * gate_risk
        + 0.15 * sp_prior.astype(np.float64)
        + gate_safety_penalty
    )

    # 2. HURDLE_SOS_PRIOR (0 additional reveals beyond R882; acts only on near-ties top_zgap < 0.05)
    if top_zgap < 0.05:
        cost_h0 = rcost + 0.012 * h_cost
        u_h0 = float(u[int(np.argmin(cost_h0))])
    else:
        u_h0 = u_r882

    # Standardize sem24 per pool for RBF kernel
    s_mean = sem24.mean(axis=0, keepdims=True)
    s_std = sem24.std(axis=0, keepdims=True) + 1e-6
    X = (sem24 - s_mean) / s_std

    # 3. GATED_025 (H217 baseline on idx4)
    if top_zgap < 0.25:
        y4_cost = np.where(gate[idx4], fitness[idx4], 2.0)
        y4_c = y4_cost - float(np.mean(y4_cost))
        diff = X[:, None, :] - X[idx4][None, :, :]
        sq = np.sum(diff * diff, axis=-1) / 24.0
        K = np.exp(-sq / (2.0 * (2.0 ** 2)))
        w = K / (np.sum(K, axis=1, keepdims=True) + 0.2)
        cost_g025 = rcost + 0.05 * (w @ y4_c)
        u_g025 = float(u[int(np.argmin(cost_g025))])
    else:
        u_g025 = u_r882

    # 4. GATED_HURDLE_SP (Passive 4-reveal on idx4 with Decomposed Continuous Hurdle residual + multi-scale kernel)
    if top_zgap < 0.25:
        y_fit_cost = np.where(gate[idx4], fitness[idx4], 2.0)
        y_cont_cost = np.clip(logratio[idx4], -1.5, 1.5)
        y_rev_cost = 0.65 * (y_fit_cost - float(np.mean(y_fit_cost))) + 0.35 * (
            y_cont_cost - float(np.mean(y_cont_cost))
        )
        diff = X[:, None, :] - X[idx4][None, :, :]
        sq = np.sum(diff * diff, axis=-1) / 24.0
        K_multi = 0.6 * np.exp(-sq / (2.0 * (1.8 ** 2))) + 0.4 * np.exp(-sq / (2.0 * (3.2 ** 2)))
        w_m = K_multi / (np.sum(K_multi, axis=1, keepdims=True) + 0.15)

        # Center Hurdle prior relative to revealed anchors idx4
        h_c4 = h_cost - float(np.mean(h_cost[idx4]))
        res_4 = y_rev_cost - 0.15 * h_c4[idx4]
        cost_hsp = rcost + 0.05 * (w_m @ y_rev_cost) + 0.018 * (w_m @ res_4) + 0.008 * h_c4
        u_hsp = float(u[int(np.argmin(cost_hsp))])
    else:
        u_hsp = u_r882

    # 5. ACTIVE_2PLUS2_HURDLE (W4 Sequential 2+2 Active Reveal under exact 4-reveal budget)
    if top_zgap < 0.25:
        idx_a = idx4[:2]
        ya_cost = np.where(gate[idx_a], fitness[idx_a], 2.0)
        ya_c = ya_cost - float(np.mean(ya_cost))
        diff_a = X[:, None, :] - X[idx_a][None, :, :]
        sq_a = np.sum(diff_a * diff_a, axis=-1) / 24.0
        Ka = np.exp(-sq_a / (2.0 * (2.2 ** 2)))
        wa = Ka / (np.sum(Ka, axis=1, keepdims=True) + 0.2)
        h_ca = h_cost - float(np.mean(h_cost[idx_a]))
        interim_cost = rcost + 0.035 * (wa @ ya_c) + 0.010 * h_ca

        # Select 2 active reveals among top contenders (lowest interim_cost + epistemic diversity)
        order = np.argsort(interim_cost)
        chosen_b: List[int] = []
        for cand in order:
            ci = int(cand)
            if ci not in idx_a:
                chosen_b.append(ci)
                break
        rev_so_far = list(idx_a) + chosen_b
        best_b2 = -1
        best_acq = 1e18
        for cand in order[:6]:
            ci = int(cand)
            if ci in rev_so_far:
                continue
            d_min = float(np.min(np.sum((X[ci : ci + 1] - X[rev_so_far]) ** 2, axis=1) / 24.0))
            acq = float(interim_cost[ci]) - 0.015 * np.sqrt(d_min)
            if acq < best_acq:
                best_acq = acq
                best_b2 = ci
        if best_b2 >= 0:
            chosen_b.append(best_b2)

        rev4 = np.array(list(idx_a) + chosen_b, dtype=np.int64)
        y4_f = np.where(gate[rev4], fitness[rev4], 2.0)
        y4_lr = np.clip(logratio[rev4], -1.5, 1.5)
        y4_comb = 0.70 * (y4_f - float(np.mean(y4_f))) + 0.30 * (y4_lr - float(np.mean(y4_lr)))

        diff4 = X[:, None, :] - X[rev4][None, :, :]
        sq4 = np.sum(diff4 * diff4, axis=-1) / 24.0
        K4 = 0.6 * np.exp(-sq4 / (2.0 * (1.8 ** 2))) + 0.4 * np.exp(-sq4 / (2.0 * (3.2 ** 2)))
        w4 = K4 / (np.sum(K4, axis=1, keepdims=True) + 0.15)
        h_c4_act = h_cost - float(np.mean(h_cost[rev4]))

        cost_act = rcost + 0.05 * (w4 @ y4_comb) + 0.010 * h_c4_act
        # Direct revealed evidence shrinkage on the 4 evaluated slots
        for j_pos, c_idx in enumerate(rev4):
            cost_act[c_idx] += 0.08 * float(y4_f[j_pos] - np.mean(y4_f))
        u_act = float(u[int(np.argmin(cost_act))])
    else:
        u_act = u_r882

    return {
        "R882": u_r882,
        "HURDLE_SOS_PRIOR": u_h0,
        "GATED_025": u_g025,
        "GATED_HURDLE_SP": u_hsp,
        "ACTIVE_2PLUS2_HURDLE": u_act,
    }


def main() -> None:
    t0 = time.time()
    STAGE1_OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[H252 Stage 1] Loading cohorts, context features, and pools...")
    train_rows, val_rows = load_cohorts()
    all_rows = train_rows + val_rows
    ctx_feats = load_context_features()
    pools = load_pools(all_rows, verify_hashes=False)
    promos_by_cid = load_harvested_promotions(train_rows)
    n_promos = sum(len(v) for v in promos_by_cid.values())
    print(
        f"[H252 Stage 1] Loaded {len(train_rows)} DL-Train contexts (205 families), "
        f"{len(val_rows)} DL-Val contexts (19 families), and {n_promos} harvested GP promotions."
    )

    # 1. Load or compute exact R882 baseline cache and verify Native91 = 1.1377941898164525
    r882_cache = get_or_compute_r882_cache(all_rows, pools, ctx_feats)
    val_native_utils = [
        float(pools[r["context_sha256"]]["utility"][int(np.argmin(r882_cache[r["context_sha256"]]["total_cost"]))])
        for r in val_rows
    ]
    val_native91_mean = float(np.mean(val_native_utils))
    diff_native = abs(val_native91_mean - R882_NATIVE91_MEAN)
    print(f"[H252 Stage 1] R882 Native91 utility mean = {val_native91_mean:.16f} (diff={diff_native:.2e})")
    if diff_native > 1e-12:
        raise RuntimeError(f"R882 Native91 mismatch: {val_native91_mean} != {R882_NATIVE91_MEAN}")

    # 2. Build 64D outcome-free candidate features for all 1,310 contexts
    t_feat = time.time()
    cand_feats = build_candidate_features(all_rows, pools, r882_cache, ctx_feats)
    print(f"[H252 Stage 1] Built 64D candidate features for {len(cand_feats)} contexts in {time.time() - t_feat:.2f}s.")

    # 3. Fit 5-fold cross-fitted Decomposed Continuous Hurdle Critic + Self-Play Promotion Prior per seed
    arms = ["R882", "HURDLE_SOS_PRIOR", "GATED_025", "GATED_HURDLE_SP", "ACTIVE_2PLUS2_HURDLE"]
    train_panel = np.zeros((len(SEEDS), len(train_rows), len(arms)), dtype=np.float64)
    val_panel = np.zeros((len(SEEDS), len(val_rows), len(arms)), dtype=np.float64)

    hurdle_diags_by_seed: Dict[str, Any] = {}
    saved_priors_seed0: Dict[str, np.ndarray] = {}
    saved_sp_seed0: Dict[str, np.ndarray] = {}

    for si, seed in enumerate(SEEDS):
        sos_priors, sp_priors, diag = fit_decomposed_hurdle_and_sp_priors(
            train_rows, val_rows, pools, cand_feats, promos_by_cid, seed=seed
        )
        hurdle_diags_by_seed[str(seed)] = diag
        if si == 0:
            saved_priors_seed0 = sos_priors
            saved_sp_seed0 = sp_priors

        rng_train = np.random.default_rng(seed)
        for ci, r in enumerate(train_rows):
            cid = r["context_sha256"]
            res = score_arms_for_context(
                cid, pools[cid], r882_cache[cid]["total_cost"], sos_priors[cid], sp_priors[cid], rng_train
            )
            for ai, a in enumerate(arms):
                train_panel[si, ci, ai] = res[a]

        rng_val = np.random.default_rng(seed + 1000)
        for ci, r in enumerate(val_rows):
            cid = r["context_sha256"]
            res = score_arms_for_context(
                cid, pools[cid], r882_cache[cid]["total_cost"], sos_priors[cid], sp_priors[cid], rng_val
            )
            for ai, a in enumerate(arms):
                val_panel[si, ci, ai] = res[a]

    # Save cross-fitted 2-channel Hurdle priors for Stage 2 neural adapter training on xbabe1 GPU
    cids_sorted = sorted(saved_priors_seed0.keys())
    priors_path = STAGE1_OUT_DIR / "stage1_hurdle_priors.npz"
    np.savez_compressed(
        priors_path,
        cids=np.array(cids_sorted),
        sos_prior=np.stack([pad_to_1152(saved_priors_seed0[c], 0.0) for c in cids_sorted], axis=0),
        sp_prior=np.stack([pad_to_1152(saved_sp_seed0[c], 0.0) for c in cids_sorted], axis=0),
    )
    print(f"[H252 Stage 1] Saved cross-fitted Hurdle + SP priors to {priors_path}")

    # 4. Statistical summaries (10,000-draw paired family bootstrap, Holm-Bonferroni, 400-iter MM Elo)
    train_summary_vs_r882 = summarize_panel(train_rows, train_panel, arms, ref_idx=0)
    train_summary_vs_g025 = summarize_panel(train_rows, train_panel, arms, ref_idx=2)
    val_summary_vs_r882 = summarize_panel(val_rows, val_panel, arms, ref_idx=0)
    val_summary_vs_g025 = summarize_panel(val_rows, val_panel, arms, ref_idx=2)

    for split_label, summ_r, summ_g in [
        ("DL-Train (1,219 ctx / 205 fam)", train_summary_vs_r882, train_summary_vs_g025),
        ("DL-Val (91 ctx / 19 fam)", val_summary_vs_r882, val_summary_vs_g025),
    ]:
        print(f"\n=== [H252 Stage 1] {split_label} ===")
        for ai, a in enumerate(arms):
            med_fam = summ_r["median_family_utility"][ai]
            med_ctx = summ_r["median_context_utility"][ai]
            elo = summ_r["median_elo"][ai]
            if a == "R882":
                print(f"  {a:<22} | fam_u={med_fam:.6f} | ctx_u={med_ctx:.6f} | Elo={elo:.1f} (anchor)")
            else:
                cr = summ_r["contrasts"][f"{a}_minus_R882"]
                cg_key = f"{a}_minus_GATED_025"
                cg_gain = summ_g["contrasts"][cg_key]["median_family_gain"] if cg_key in summ_g["contrasts"] else 0.0
                print(
                    f"  {a:<22} | fam_u={med_fam:.6f} | d_fam_R882={cr['median_family_gain']:+.6f} "
                    f"(95% CI [{cr['ci95'][0]:+.6f}, {cr['ci95'][1]:+.6f}], p={cr['p_one_sided']:.4g}, "
                    f"p_holm={cr['holm_p']:.4g}, W/T/L={cr['family_wtl_by_seed'][0]}) | "
                    f"d_fam_G025={cg_gain:+.6f} | Elo={elo:.1f}"
                )

    out_obj = {
        "hypothesis_id": "H252_STAGE1",
        "wave": "W3_HURDLE_SOS_CRITIC_AND_W4_ACTIVE_2PLUS2",
        "elapsed_sec": float(time.time() - t0),
        "r882_native91_exact": val_native91_mean,
        "hurdle_oof_diagnostics": hurdle_diags_by_seed,
        "dl_train_vs_r882": train_summary_vs_r882,
        "dl_train_vs_gated_025": train_summary_vs_g025,
        "dl_val_vs_r882": val_summary_vs_r882,
        "dl_val_vs_gated_025": val_summary_vs_g025,
    }
    metrics_path = STAGE1_OUT_DIR / "stage1_metrics.json"
    save_json(metrics_path, out_obj)
    print(f"\n[H252 Stage 1] Saved complete Stage 1 metrics to {metrics_path} in {time.time() - t0:.2f}s.")


if __name__ == "__main__":
    main()
