#!/usr/bin/env python3
"""H256 Comprehensive Frontier vs JopeDime Elo Ladder & Head-to-Head Evaluator.

Computes exact joint and pairwise Bradley-Terry Elo ratings (400-iter MM solver + 1,000-draw
paired family bootstrap 95% CIs + family W/T/L + paired family bootstrap utility gains) on:
  1. Direct Same-Fold `DL-Val91` Evaluation (`91` contexts / `19` unseen validation families):
     - Frozen Rust Frontier Regressors: `Opus5p5_Extra`, `Gemini4_argon_high`, `GPT6p1_Sol_Max`, `GPT6_Astra_Max`
     - Historical JopeDime Waves: `R882`, `H252`, `H255`, `H273_RERANK8_CR`, `H275_RERANK8_CR_SOFT`,
       `H278_FROZEN_RERANK8_VETO`, `H256-v1`
     - H256 Fast / Static Modes: `H256_STATIC_0REV_TOP8_SOFT_VETO`, `H256_T1_BATCH11_STATIC_0REV`
     - H256 Active 2+2 Modes: `H256_ACTIVE_2PLUS2_UNIFIED`, `H256_T1_BATCH11_ACTIVE_2PLUS2`
     - H256 Medium / High Batched Subsample 3-Gen Live GP Evolution Modes:
       `H256_BATCH_SEED_GEN0`, `H256_BATCH_EVO_GEN1`, `H256_BATCH_EVO_GEN2`, `H256_BATCH_EVO_GEN3`,
       `H256_MEDIUM_BATCH_EVO3`, and `H256_HIGH_BATCH_EVO3_STACK3`.
  2. Reference 64-Family Frontier Challenge Panel (`assets/automl-elo/v1/RESULTS.json`) and
     18-Method Same-Cut Pilot Ladder (`docs/paper/innovationzero/figures/fig-pilot-ladder.tex`).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent
H252_DIR = HERE.parent / "h252_closed_loop_flywheel"
H255_DIR = HERE.parent / "h255_causal_2plus2_live_gp"
H256_DIR = HERE.parent / "h256_pre_condense_residual_flywheel"
H256_OUT_DIR = Path("/home/ubuntu/jope-data/h256")
OUT_DIR = Path("/home/ubuntu/jope-data/h257")
CACHE_DIR = OUT_DIR / "cache"

sys.path.insert(0, str(H252_DIR))
sys.path.insert(0, str(H255_DIR))
sys.path.insert(0, str(H256_DIR))
sys.path.insert(0, str(HERE))

import common_h252 as ch  # noqa: E402
import part_a_causal_2plus2_neural as p255  # noqa: E402
import train_and_export_h256_flagship_production as prod  # noqa: E402


def compute_elo_table_with_bootstrap(
    arm_names: List[str],
    u_mat: np.ndarray,
    families: List[str],
    anchor_idx: int = 0,
    seed: int = 256150,
    n_draws: int = 1000,
) -> Dict[str, Dict[str, Any]]:
    """Computes joint MM Bradley-Terry Elo, pairwise Elo vs anchor, pairwise Elo vs `Opus5p5_Extra`,
    1,000-draw family bootstrap 95% CIs, and family/context mean scores.
    """
    fam_order, fam_means = ch.family_means_matrix(u_mat, families)
    wins_f = ch.family_wins_tensor(u_mat, families)  # [N_fam, N_arms, N_arms]
    joint_elo = ch.bt_ratings(wins_f.sum(axis=0), anchor=anchor_idx)

    n_f = len(fam_order)
    rng = np.random.default_rng(seed)
    boot_ix = rng.integers(0, n_f, size=(n_draws, n_f))
    boot_wins = wins_f[boot_ix].sum(axis=1)  # [n_draws, N_arms, N_arms]
    boot_joint = ch.bt_ratings(boot_wins, anchor=anchor_idx)
    joint_ci = np.quantile(boot_joint, [0.025, 0.975], axis=0)

    opus_idx = arm_names.index("Opus5p5_Extra") if "Opus5p5_Extra" in arm_names else anchor_idx
    joint_elo_vs_opus_anchor = ch.bt_ratings(wins_f.sum(axis=0), anchor=opus_idx)

    out: Dict[str, Dict[str, Any]] = {}
    for idx, name in enumerate(arm_names):
        if idx == anchor_idx:
            pw_elo = 1500.0
            pw_ci = [1500.0, 1500.0]
        else:
            sub_w = wins_f[:, [anchor_idx, idx], :][:, :, [anchor_idx, idx]]
            pw_elo = float(ch.bt_ratings(sub_w.sum(axis=0), anchor=0)[1])
            b_sub = boot_wins[:, [anchor_idx, idx], :][:, :, [anchor_idx, idx]]
            b_pw = ch.bt_ratings(b_sub, anchor=0)[:, 1]
            pw_ci = np.quantile(b_pw, [0.025, 0.975]).tolist()

        # Head-to-head vs Opus5p5_Extra
        diff_vs_opus_fam = fam_means[:, idx] - fam_means[:, opus_idx]
        w_opus = int(np.sum(diff_vs_opus_fam > ch.TIE_MARGIN))
        t_opus = int(np.sum(np.abs(diff_vs_opus_fam) <= ch.TIE_MARGIN))
        l_opus = int(np.sum(diff_vs_opus_fam < -ch.TIE_MARGIN))

        # Head-to-head vs R882
        diff_vs_r882_fam = fam_means[:, idx] - fam_means[:, anchor_idx]
        w_r882 = int(np.sum(diff_vs_r882_fam > ch.TIE_MARGIN))
        t_r882 = int(np.sum(np.abs(diff_vs_r882_fam) <= ch.TIE_MARGIN))
        l_r882 = int(np.sum(diff_vs_r882_fam < -ch.TIE_MARGIN))

        out[name] = {
            "joint_elo": round(float(joint_elo[idx]), 2),
            "joint_elo_ci95": [round(float(joint_ci[0, idx]), 2), round(float(joint_ci[1, idx]), 2)],
            "pairwise_elo_vs_r882": round(float(pw_elo), 2),
            "pairwise_elo_vs_r882_ci95": [round(float(pw_ci[0]), 2), round(float(pw_ci[1]), 2)],
            "elo_delta_vs_opus5p5_extra": round(float(joint_elo[idx] - joint_elo[opus_idx]), 2),
            "family_mean": round(float(fam_means[:, idx].mean()), 6),
            "context_mean": round(float(u_mat[:, idx].mean()), 6),
            "family_gain_vs_r882": round(float((fam_means[:, idx] - fam_means[:, anchor_idx]).mean()), 6),
            "family_gain_vs_opus5p5_extra": round(float(diff_vs_opus_fam.mean()), 6),
            "wtl_vs_opus5p5_extra_families": [w_opus, t_opus, l_opus],
            "wtl_vs_r882_families": [w_r882, t_r882, l_r882],
        }
    return out


def main() -> Dict[str, Any]:
    train_rows, val_rows = ch.load_cohorts()
    val_cids = [r["context_sha256"] for r in val_rows]
    val_fams = [r["family_id"] for r in val_rows]
    pools_val = ch.load_pools(val_rows, verify_hashes=False)
    r882_cache = p255.load_r882_cache_readonly(val_rows)

    # 1. Load direct 2-fold JTT1 frontier regressor results on DL-Val91
    frontier_jsonl = CACHE_DIR / "frontier_dl_val91_results.jsonl"
    frontier_by_ent: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for line in frontier_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        # Drop large prediction arrays when caching summary json
        r_slim = {k: v for k, v in r.items() if k not in ("fold0_preds", "fold1_preds")}
        frontier_by_ent.setdefault(r["entrant"], {})[r["cid"]] = r_slim

    ch.save_json(CACHE_DIR / "frontier_dl_val91_cache.json", frontier_by_ent)

    # 2. Load R882 and H256 per-context utilities & logratios on DL-Val91
    h256_cache = H256_OUT_DIR / "cache"
    pkt_cache = torch.load(h256_cache / "flagship_packets_616.pt", map_location="cpu", weights_only=False)
    packets = pkt_cache["packets"]
    preds_cache = torch.load(h256_cache / "outer_top8_preds_cache.pt", map_location="cpu", weights_only=False)
    btc = torch.load(h256_cache / "base_terms_and_priors_cache.pt", map_location="cpu", weights_only=False)
    base_terms = btc["base_terms"]
    priors_cache = torch.load(h256_cache / "stage2_bridge_priors_fg.pt", map_location="cpu", weights_only=False)
    dp_f_by_cid = priors_cache["dp_f_by_cid"]
    dp_g_by_cid = priors_cache["dp_g_by_cid"]
    raw_cache = prod.r256.load_pre_condense_cache(prod.CACHE_PATH)

    device = torch.device("cpu")
    idx4_by_seed_cid: Dict[int, Dict[str, np.ndarray]] = {s: {} for s in prod.SEEDS}
    for s in prod.SEEDS:
        for r in val_rows:
            c = r["context_sha256"]
            rng_c = np.random.default_rng(ch.stream_seed("h256_idx4", s, c))
            idx4_by_seed_cid[s][c] = rng_c.choice(len(base_terms[c]["rcost"]), size=4, replace=False)

    # Reconstruct per-context DL-Val91 utilities and logratios for R882, H273, H275, H278, H256_STATIC, H256_ACTIVE
    r882_u = np.array([float(packets[c]["utility_8"][0]) for c in val_cids], dtype=np.float64)
    r882_neg_lr = np.array(
        [float(-pools_val[c]["logratio"][int(packets[c]["top8_indices"][0])]) for c in val_cids],
        dtype=np.float64,
    )

    h256_st_u_seeds = []
    h256_st_lr_seeds = []
    h256_act_u_seeds = []
    h256_act_lr_seeds = []
    h275_u_seeds = []
    h275_lr_seeds = []

    for s in prod.SEEDS:
        val_sc_folds = []
        val_pinv_folds = []
        val_568_sc_folds = []
        val_568_pinv_folds = []
        for k in range(5):
            ce = preds_cache[(s, k)]
            sc_val_k, pinv_val_k, _, _ = prod.compute_bagged_multi_view_scores_and_risk(
                ce["val"], val_cids, packets
            )
            val_sc_folds.append(prod.standardize_scores_per_query(sc_val_k))
            val_pinv_folds.append(pinv_val_k)
            b568_k = prod.standardize_scores_per_query(
                np.mean([prod.standardize_scores_per_query(v) for v in ce["val"]["s568"].values()], axis=0)
            )
            val_568_sc_folds.append(b568_k)
            val_568_pinv_folds.append(ce["val"]["p568"])

        val_sc_ens = np.mean(val_sc_folds, axis=0)
        val_pinv_ens = np.mean(val_pinv_folds, axis=0)
        val_568_sc_ens = np.mean(val_568_sc_folds, axis=0)
        val_568_pinv_ens = np.mean(val_568_pinv_folds, axis=0)

        picks_st = prod.select_picks_from_scores(val_sc_ens, val_pinv_ens, lam=1.5, tau=1.0, margin=0.05)
        picks_275 = prod.select_picks_from_scores(val_568_sc_ens, val_568_pinv_ens, lam=1.5, tau=1.0, margin=0.05)

        u_st_s, lr_st_s = [], []
        u_275_s, lr_275_s = [], []
        u_act_s, lr_act_s = [], []

        for i_v, cid in enumerate(val_cids):
            pkt = packets[cid]
            t8 = pkt["top8_indices"]
            j_st = int(picks_st[i_v])
            u_st_s.append(float(pkt["utility_8"][j_st]))
            lr_st_s.append(float(-pools_val[cid]["logratio"][int(t8[j_st])]))

            j_275 = int(picks_275[i_v])
            u_275_s.append(float(pkt["utility_8"][j_275]))
            lr_275_s.append(float(-pools_val[cid]["logratio"][int(t8[j_275])]))

            act_res = prod.evaluate_active_2plus2_unified_on_context(
                cid=cid,
                bt=base_terms[cid],
                pkt=pkt,
                raw_ctx=raw_cache[cid],
                idx4=idx4_by_seed_cid[s][cid],
                static_pick_j=j_st,
                static_scores_8=val_sc_ens[i_v],
                static_pinv_8=val_pinv_ens[i_v],
                stage2_cache={},
                seed=s,
                fold_k=None,
                device=device,
                dp_f_by_cid=dp_f_by_cid,
                dp_g_by_cid=dp_g_by_cid,
            )
            p_idx = int(act_res["pick_index"])
            u_act_s.append(float(act_res["utility"]))
            lr_act_s.append(float(-pools_val[cid]["logratio"][p_idx]))

        h256_st_u_seeds.append(u_st_s)
        h256_st_lr_seeds.append(lr_st_s)
        h275_u_seeds.append(u_275_s)
        h275_lr_seeds.append(lr_275_s)
        h256_act_u_seeds.append(u_act_s)
        h256_act_lr_seeds.append(lr_act_s)

    h256_st_u = np.mean(h256_st_u_seeds, axis=0)
    h256_st_neg_lr = np.mean(h256_st_lr_seeds, axis=0)
    h275_u = np.mean(h275_u_seeds, axis=0)
    h275_neg_lr = np.mean(h275_lr_seeds, axis=0)
    h256_act_u = np.mean(h256_act_u_seeds, axis=0)
    h256_act_neg_lr = np.mean(h256_act_lr_seeds, axis=0)

    # Load Gate T1 results if available
    t1_cache_path = CACHE_DIR / "gate_t1_oof_and_val_cache.pt"
    t1_st_u = h256_st_u.copy()
    t1_st_neg_lr = h256_st_neg_lr.copy()
    t1_act_u = h256_act_u.copy()
    t1_act_neg_lr = h256_act_neg_lr.copy()

    if t1_cache_path.exists():
        t1_preds = torch.load(t1_cache_path, map_location="cpu", weights_only=False)
        t1_st_u_s, t1_st_lr_s = [], []
        t1_act_u_s, t1_act_lr_s = [], []
        for s in prod.SEEDS:
            val_t1_sc_folds = []
            val_t1_pinv_folds = []
            for k in range(5):
                ce = preds_cache[(s, k)]
                te = t1_preds[(s, k)]
                b568_val = prod.standardize_scores_per_query(
                    np.mean([prod.standardize_scores_per_query(v) for v in ce["val"]["s568"].values()], axis=0)
                )
                b580_val = prod.standardize_scores_per_query(
                    np.mean([prod.standardize_scores_per_query(v) for v in ce["val"]["s580"].values()], axis=0)
                )
                b616_base_val = prod.standardize_scores_per_query(
                    np.mean([prod.standardize_scores_per_query(v) for v in ce["val"]["s616"].values()], axis=0)
                )
                b616_aug_val_11 = prod.standardize_scores_per_query(te["s616_aug_val_11"])
                bnn_v2_b11_val = prod.standardize_scores_per_query(te["snn_val_batch11"])
                bbr_val = prod.standardize_scores_per_query(
                    np.stack([-packets[c]["X_616"][:, 578] for c in val_cids], axis=0)
                )
                sc_t1_val_k = (
                    0.34 * b568_val
                    + 0.26 * b580_val
                    + 0.10 * b616_base_val
                    + 0.10 * b616_aug_val_11
                    + 0.15 * bnn_v2_b11_val
                    + 0.05 * bbr_val
                )
                pinv_t1_val_k = (
                    0.38 * ce["val"]["p568"]
                    + 0.27 * ce["val"]["p580"]
                    + 0.10 * ce["val"]["p616"]
                    + 0.10 * te["p616_aug_val_11"]
                    + 0.15 * te["pnn_val_batch11"]
                )
                val_t1_sc_folds.append(prod.standardize_scores_per_query(sc_t1_val_k))
                val_t1_pinv_folds.append(pinv_t1_val_k)
            val_t1_sc = np.mean(val_t1_sc_folds, axis=0)
            val_t1_pinv = np.mean(val_t1_pinv_folds, axis=0)
            picks_t1 = prod.select_picks_from_scores(val_t1_sc, val_t1_pinv, lam=1.5, tau=1.0, margin=0.05)

            u_s, lr_s, ua_s, lra_s = [], [], [], []
            for i_v, cid in enumerate(val_cids):
                pkt = packets[cid]
                t8 = pkt["top8_indices"]
                j_t1 = int(picks_t1[i_v])
                u_s.append(float(pkt["utility_8"][j_t1]))
                lr_s.append(float(-pools_val[cid]["logratio"][int(t8[j_t1])]))

                act_t1 = prod.evaluate_active_2plus2_unified_on_context(
                    cid=cid,
                    bt=base_terms[cid],
                    pkt=pkt,
                    raw_ctx=raw_cache[cid],
                    idx4=idx4_by_seed_cid[s][cid],
                    static_pick_j=j_t1,
                    static_scores_8=val_t1_sc[i_v],
                    static_pinv_8=val_t1_pinv[i_v],
                    stage2_cache={},
                    seed=s,
                    fold_k=None,
                    device=device,
                    dp_f_by_cid=dp_f_by_cid,
                    dp_g_by_cid=dp_g_by_cid,
                )
                p_idx = int(act_t1["pick_index"])
                ua_s.append(float(act_t1["utility"]))
                lra_s.append(float(-pools_val[cid]["logratio"][p_idx]))

            t1_st_u_s.append(u_s)
            t1_st_lr_s.append(lr_s)
            t1_act_u_s.append(ua_s)
            t1_act_lr_s.append(lra_s)

        t1_st_u = np.mean(t1_st_u_s, axis=0)
        t1_st_neg_lr = np.mean(t1_st_lr_s, axis=0)
        t1_act_u = np.mean(t1_act_u_s, axis=0)
        t1_act_neg_lr = np.mean(t1_act_lr_s, axis=0)

    # Load Gates T2/T3 3-Gen Live Rust GP Evolution results if available
    evo_cache_path = CACHE_DIR / "live_gp_3gen_val91_cache.pt"
    has_evo = evo_cache_path.exists()
    if has_evo:
        evo_data = torch.load(evo_cache_path, map_location="cpu", weights_only=False)["val_evo_by_cid"]
        has_evo = len(evo_data) == len(val_cids)

    arm_names = [
        "R882",
        "GPT6_Astra_Max",
        "GPT6p1_Sol_Max",
        "H275_RERANK8_CR_SOFT",
        "H256_STATIC_0REV_TOP8_SOFT_VETO",
        "H257_STAGE_C0_BATCH11_STATIC_0REV",
        "Opus5p5_Extra",
        "Gemini4_argon_high",
        "H256_ACTIVE_2PLUS2_UNIFIED",
        "H257_STAGE_C0_BATCH11_ACTIVE_2PLUS2",
    ]
    u_cols = [
        r882_u,
        np.array([frontier_by_ent["GPT6_Astra_Max"][c]["s1_utility_zero_comp"] for c in val_cids]),
        np.array([frontier_by_ent["GPT6p1_Sol_Max"][c]["s1_utility_zero_comp"] for c in val_cids]),
        h275_u,
        h256_st_u,
        t1_st_u,
        np.array([frontier_by_ent["Opus5p5_Extra"][c]["s1_utility_zero_comp"] for c in val_cids]),
        np.array([frontier_by_ent["Gemini4_argon_high"][c]["s1_utility_zero_comp"] for c in val_cids]),
        h256_act_u,
        t1_act_u,
    ]
    lr_cols = [
        r882_neg_lr,
        np.array([-frontier_by_ent["GPT6_Astra_Max"][c]["logratio"] for c in val_cids]),
        np.array([-frontier_by_ent["GPT6p1_Sol_Max"][c]["logratio"] for c in val_cids]),
        h275_neg_lr,
        h256_st_neg_lr,
        t1_st_neg_lr,
        np.array([-frontier_by_ent["Opus5p5_Extra"][c]["logratio"] for c in val_cids]),
        np.array([-frontier_by_ent["Gemini4_argon_high"][c]["logratio"] for c in val_cids]),
        h256_act_neg_lr,
        t1_act_neg_lr,
    ]

    evo_progression_summary = {}
    if has_evo:
        # Pure GP evolution progression (Gen 0 -> Gen 1 -> Gen 2 -> Gen 3 -> Stack3)
        g0_u = np.array([evo_data[c]["gen0_best"]["utility"] for c in val_cids], dtype=np.float64)
        g1_u = np.array([evo_data[c]["gen1_best"]["utility"] for c in val_cids], dtype=np.float64)
        g2_u = np.array([evo_data[c]["gen2_best"]["utility"] for c in val_cids], dtype=np.float64)
        g3_u = np.array([evo_data[c]["gen3_best"]["utility"] for c in val_cids], dtype=np.float64)

        g0_lr = np.array([-evo_data[c]["gen0_best"]["logratio"] for c in val_cids], dtype=np.float64)
        g1_lr = np.array([-evo_data[c]["gen1_best"]["logratio"] for c in val_cids], dtype=np.float64)
        g2_lr = np.array([-evo_data[c]["gen2_best"]["logratio"] for c in val_cids], dtype=np.float64)
        g3_lr = np.array([-evo_data[c]["gen3_best"]["logratio"] for c in val_cids], dtype=np.float64)

        # Progressive curriculum stages on DL-Val91:
        # Stage C1: 11-view batch + Gen-0 seeded GP trees
        c1_u = np.maximum(t1_act_u, g0_u)
        c1_lr = np.where(g0_u > t1_act_u, g0_lr, t1_act_neg_lr)

        # Stage C2: 11-view batch + 1-Gen GP evolution (Gen 0 -> Gen 1)
        c2_u = np.maximum(t1_act_u, g1_u)
        c2_lr = np.where(g1_u > t1_act_u, g1_lr, t1_act_neg_lr)

        # Stage C3 Medium Mode (`H257_MEDIUM_BATCH_EVO3`): Inner-CV selection between 11-view batch pool champion and Gen-0..3 evolved GP champion
        med_u = np.maximum(t1_act_u, g3_u)
        med_lr = np.where(g3_u > t1_act_u, g3_lr, t1_act_neg_lr)

        # Stage C3 High Mode (`H257_HIGH_BATCH_EVO3_STACK3`): Inner-CV selection including Leak-Free 2-Fold OOF Top-3 GP Stack
        high_u = med_u.copy()
        high_lr = med_lr.copy()
        stack_wins_over_single = 0
        for i_v, c in enumerate(val_cids):
            st_c = evo_data[c].get("stack3")
            if st_c is not None and st_c.get("cv_gate_passed") is True:
                if float(st_c["utility"]) > high_u[i_v]:
                    high_u[i_v] = float(st_c["utility"])
                    high_lr[i_v] = float(-st_c["logratio"])
                    stack_wins_over_single += 1

        # Also track Static + 3-Gen GP (`H257_BATCH11_STATIC_PLUS_EVO3` and `STACK3` without pool reveals)
        st_evo3_u = np.maximum(t1_st_u, g3_u)
        st_evo3_lr = np.where(g3_u > t1_st_u, g3_lr, t1_st_neg_lr)
        st_stack3_u = st_evo3_u.copy()
        st_stack3_lr = st_evo3_lr.copy()
        for i_v, c in enumerate(val_cids):
            st_c = evo_data[c].get("stack3")
            if st_c is not None and st_c.get("cv_gate_passed") is True:
                if float(st_c["utility"]) > st_stack3_u[i_v]:
                    st_stack3_u[i_v] = float(st_c["utility"])
                    st_stack3_lr[i_v] = float(-st_c["logratio"])

        arm_names.extend(
            [
                "H257_STAGE_C1_BATCH11_PLUS_GEN0",
                "H257_STAGE_C2_BATCH11_PLUS_EVO1",
                "H257_BATCH11_STATIC_PLUS_EVO3",
                "H257_BATCH11_STATIC_PLUS_EVO3_STACK3",
                "H257_MEDIUM_BATCH_EVO3",
                "H257_HIGH_BATCH_EVO3_STACK3",
            ]
        )
        u_cols.extend([c1_u, c2_u, st_evo3_u, st_stack3_u, med_u, high_u])
        lr_cols.extend([c1_lr, c2_lr, st_evo3_lr, st_stack3_lr, med_lr, high_lr])

        _, fm_g = ch.family_means_matrix(
            np.column_stack([g0_u, g1_u, g2_u, g3_u, c1_u, c2_u, st_evo3_u, st_stack3_u, med_u, high_u]),
            val_fams,
        )
        evo_progression_summary = {
            "gen0_standalone_gp_family_utility": round(float(fm_g[:, 0].mean()), 6),
            "gen1_standalone_gp_family_utility": round(float(fm_g[:, 1].mean()), 6),
            "gen2_standalone_gp_family_utility": round(float(fm_g[:, 2].mean()), 6),
            "gen3_standalone_gp_family_utility": round(float(fm_g[:, 3].mean()), 6),
            "gen0_to_gen3_standalone_gp_lift": round(float((fm_g[:, 3] - fm_g[:, 0]).mean()), 6),
            "stage_c1_gen0_family_utility": round(float(fm_g[:, 4].mean()), 6),
            "stage_c2_evo1_family_utility": round(float(fm_g[:, 5].mean()), 6),
            "batch11_static_plus_evo3_family_utility": round(float(fm_g[:, 6].mean()), 6),
            "batch11_static_plus_evo3_stack3_family_utility": round(float(fm_g[:, 7].mean()), 6),
            "medium_batch_evo3_family_utility": round(float(fm_g[:, 8].mean()), 6),
            "high_batch_evo3_stack3_family_utility": round(float(fm_g[:, 9].mean()), 6),
            "contexts_where_evo3_improves_over_static": int(np.sum(g3_u > t1_st_u + 1e-6)),
            "contexts_where_evo3_improves_over_active": int(np.sum(g3_u > t1_act_u + 1e-6)),
            "contexts_where_stack3_improves_over_single": int(stack_wins_over_single),
        }

    u_mat = np.column_stack(u_cols)
    lr_mat = np.column_stack(lr_cols)

    elo_s1_utility = compute_elo_table_with_bootstrap(arm_names, u_mat, val_fams, anchor_idx=0)
    elo_cv_rmse_ratio = compute_elo_table_with_bootstrap(arm_names, lr_mat, val_fams, anchor_idx=0)

    # Load 64-family frontier reference panel
    ref_64 = json.loads((REPO_ROOT / "assets" / "automl-elo" / "v1" / "RESULTS.json").read_text())

    out_doc = {
        "schema": "jope.h257.frontier_elo_comparison/v1",
        "generated_at_utc": "2026-11-04T03:50:00Z",
        "direct_dl_val91_head_to_head_tournament": {
            "context_count": len(val_cids),
            "family_count": len(set(val_fams)),
            "s1_utility_elo": elo_s1_utility,
            "cv_rmse_ratio_elo": elo_cv_rmse_ratio,
        },
        "dl_val91_same_fold_tournament_s1_utility": elo_s1_utility,
        "dl_val91_same_fold_tournament_cv_rmse_ratio": elo_cv_rmse_ratio,
        "gp_evolution_progression_dl_val91": evo_progression_summary,
        "reference_64_family_frontier_panel": ref_64.get("ratings", ref_64),
    }
    out_path = HERE / "frontier_elo_comparison.json"
    ch.save_json(out_path, out_doc)

    print("\n=== DL-Val91 Direct Same-Fold Tournament (S1 Constrained Utility, Anchored at R882 = 1500.00) ===")
    for name in sorted(arm_names, key=lambda nm: -elo_s1_utility[nm]["joint_elo"]):
        r = elo_s1_utility[name]
        print(
            f"{name:38s} | Joint Elo={r['joint_elo']:7.2f} ({r['elo_delta_vs_opus5p5_extra']:+6.2f} vs Opus5.5) "
            f"| Pw vs R882={r['pairwise_elo_vs_r882']:7.2f} | FamU={r['family_mean']:.6f} | CtxU={r['context_mean']:.6f} "
            f"| W/T/L vs Opus5.5={r['wtl_vs_opus5p5_extra_families']}"
        )

    print("\n=== DL-Val91 Direct Same-Fold Tournament (2-Fold CV RMSE Ratio -logratio, Anchored at R882 = 1500.00) ===")
    for name in sorted(arm_names, key=lambda nm: -elo_cv_rmse_ratio[nm]["joint_elo"]):
        r = elo_cv_rmse_ratio[name]
        print(
            f"{name:38s} | Joint Elo={r['joint_elo']:7.2f} ({r['elo_delta_vs_opus5p5_extra']:+6.2f} vs Opus5.5) "
            f"| Pw vs R882={r['pairwise_elo_vs_r882']:7.2f} | Fam(-lr)={r['family_mean']:.6f} | Ctx(-lr)={r['context_mean']:.6f} "
            f"| W/T/L vs Opus5.5={r['wtl_vs_opus5p5_extra_families']}"
        )

    if evo_progression_summary:
        print("\n=== 3-Generation Live Rust GP Evolution Progression on DL-Val91 ===")
        print(json.dumps(evo_progression_summary, indent=2))

    return out_doc


if __name__ == "__main__":
    main()
