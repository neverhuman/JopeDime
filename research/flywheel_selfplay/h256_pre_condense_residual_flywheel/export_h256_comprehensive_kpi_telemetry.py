#!/usr/bin/env python3
"""H256 Comprehensive Train & Validation KPI Telemetry Suite & Checkpoint Enricher.

Computes, persists, and embeds all Train (`FIT`), Inner-Validation (`Inner-OOF`), Outer-OOF
(`DL-Train 1,219`), and Holdout (`DL-Val 91`, `dl_val_71`, `st_val_gate_32`, `dl_train_inner_203`,
`strict_train_fit_881`) KPIs across:
  1. Stage-1 Full-S3 7-Split Regression Pretraining (`11,121` Policy-B, `1,105` `eval_disjoint_v2`,
     `3,570` `dev_val_ablation`, and `15,796` All-S3 production regression cells).
  2. Stage-1/2 Auxiliary Critics (`SOSCalibratedAdapter` continuous & gate-failure hurdle models +
     `SelfPlayTrajectoryHead` trajectory direction).
  3. Stage-2 `PreCondenseResidualBridge` (`6` arms x `3` seeds x `5` folds = `90` checkpoints).
  4. Flagship `H256_STATIC_0REV_TOP8_SOFT_VETO` (`0` reveals), `H256_STATIC_0REV_TOP8_NEURAL_ONLY`
     (`0` reveals), and `H256_ACTIVE_2PLUS2_UNIFIED` (`2+2` active reveals), plus the full-dataset
     `1,219` `DL-Train` and `1,310` `DL-Train + DL-Val` production refits.

Writes:
  - `/home/ubuntu/jope-data/h256/training_and_validation_kpi_telemetry.json`
  - `/home/ubuntu/jope-data/h256/per_context_and_family_kpi_ledger.json`
  - `/home/ubuntu/audit/JopeDime_main/research/flywheel_selfplay/h256_pre_condense_residual_flywheel/kpi_telemetry.json`
and embeds `"kpi_telemetry"` directly inside the saved `.pt` checkpoints, `summary.json`, and
`checkpoint_manifest.json`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import lightgbm as lgb
import numpy as np
import torch
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
H252_DIR = HERE.parent / "h252_closed_loop_flywheel"
H255_DIR = HERE.parent / "h255_causal_2plus2_live_gp"
OUT_DIR = Path("/home/ubuntu/jope-data/h256")
CKPT_DIR = OUT_DIR / "checkpoints"
CACHE_PATH = OUT_DIR / "cache" / "pre_condense_context_features.npz"

sys.path.insert(0, str(H252_DIR))
sys.path.insert(0, str(H255_DIR))
sys.path.insert(0, str(HERE))

import common_h252 as ch  # noqa: E402
import stage1_hurdle_sos_critic as s1  # noqa: E402
import part_a_causal_2plus2_neural as p255  # noqa: E402
import pretrain_full_s3_7split_bridge as p_s3  # noqa: E402
import run_h256_flywheel_ablation as r256  # noqa: E402
import export_h256_complete_inference_bundle as exp256  # noqa: E402
import train_and_export_h256_flagship_production as flg  # noqa: E402
from pre_condense_models import FoldScaler, PreCondenseResidualBridge  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_ece_10bin(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = max(len(y_t), 1)
    for b in range(n_bins):
        if b == n_bins - 1:
            mask = (y_p >= bins[b]) & (y_p <= bins[b + 1])
        else:
            mask = (y_p >= bins[b]) & (y_p < bins[b + 1])
        cnt = int(np.sum(mask))
        if cnt > 0:
            acc = float(np.mean(y_t[mask]))
            conf = float(np.mean(y_p[mask]))
            ece += (cnt / n) * abs(acc - conf)
    return float(ece)


def compute_ndcg_at_k(rel_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    order_pred = np.argsort(-scores, kind="mergesort")[:k]
    order_ideal = np.argsort(-rel_true, kind="mergesort")[:k]
    discounts = np.log2(np.arange(2, k + 2, dtype=np.float64))
    dcg = float(np.sum((np.power(2.0, rel_true[order_pred]) - 1.0) / discounts))
    idcg = float(np.sum((np.power(2.0, rel_true[order_ideal]) - 1.0) / discounts))
    if idcg <= 1e-12:
        return 1.0
    return float(dcg / idcg)


def compute_stage1_s3_kpi_telemetry(
    raw_cache: Dict[str, Dict[str, np.ndarray]],
    device: torch.device,
) -> Dict[str, Any]:
    """Computes comprehensive Stage-1 S3 7-split regression pretraining KPIs across:
      - `11,121` Policy-B train cells
      - `1,105` `eval_disjoint_v2` held-out regression cells
      - `3,570` `dev_val_ablation` held-out regression cells
      - `15,796` All-S3 production regression cells
    plus Pre-Condense 92-token group norm & gradient attributions.
    """
    pretrain_summary_path = OUT_DIR / "full_s3_reg_7split_receipt.json"
    with pretrain_summary_path.open() as f:
        pretrain_summary = json.load(f)

    ckpt_11121_path = OUT_DIR / "full_s3_reg_7split_pretrained_encoder.pt"
    ckpt_15796_path = CKPT_DIR / "stage1_full_s3_reg_all15796_encoder.pt"
    ckpt_11121 = torch.load(ckpt_11121_path, map_location=device, weights_only=False)
    ckpt_15796 = torch.load(ckpt_15796_path, map_location=device, weights_only=False)

    # Compute 92-token Pre-Condense group activation & gradient norms across the 1,310 contexts
    sc_dict = ckpt_15796["scaler_all15796"]
    scaler = FoldScaler(
        dope629_mean=np.asarray(sc_dict["dope629_mean"], dtype=np.float32),
        dope629_std=np.asarray(sc_dict["dope629_std"], dtype=np.float32),
        hidden256_mean=np.asarray(sc_dict["hidden256_mean"], dtype=np.float32),
        hidden256_std=np.asarray(sc_dict["hidden256_std"], dtype=np.float32),
        col18_mean=np.asarray(sc_dict["col18_mean"], dtype=np.float32),
        col18_std=np.asarray(sc_dict["col18_std"], dtype=np.float32),
    )
    bridge = PreCondenseResidualBridge(
        arm="G_FULL_S3_REG_7SPLIT_11121_PRETRAINED", d_token=32, n_heads=4
    ).to(device)
    bridge.load_state_dict(ckpt_15796["encoder_state_dict"], strict=False)
    bridge.eval()

    tok_norms_vec256 = []
    tok_norms_hid256 = []
    tok_norms_col64 = []
    tok_norms_lm71 = []
    tok_norms_scale = []

    with torch.no_grad():
        for cid, entry in raw_cache.items():
            ct = scaler.transform_context(entry, device)
            ctx_tokens, ctx_mask, _ = bridge.encode_context_tokens(ct)
            # Token layout in Arm F/G:
            # 0..31: vec256 (32 tokens)
            # 32..43: lm71 (12 tokens: lm_pred_23 + lm48_z)
            # 44: scale_token (1 token)
            # 45..76: hidden256 (32 tokens)
            # 77..92: col_lm_64x17 pooled (16 tokens)
            t_np = ctx_tokens[0].cpu().numpy()
            tok_norms_vec256.append(float(np.mean(np.linalg.norm(t_np[0:32], axis=-1))))
            tok_norms_lm71.append(float(np.mean(np.linalg.norm(t_np[32:44], axis=-1))))
            tok_norms_scale.append(float(np.linalg.norm(t_np[44])))
            tok_norms_hid256.append(float(np.mean(np.linalg.norm(t_np[45:77], axis=-1))))
            tok_norms_col64.append(float(np.mean(np.linalg.norm(t_np[77:93], axis=-1))))

    token_group_telemetry = {
        "total_tokens": 93,
        "token_dim": 32,
        "vec256_32tok_mean_l2_norm": float(np.mean(tok_norms_vec256)),
        "lm71_landmark_12tok_mean_l2_norm": float(np.mean(tok_norms_lm71)),
        "scale_1tok_mean_l2_norm": float(np.mean(tok_norms_scale)),
        "hidden256_32tok_mean_l2_norm": float(np.mean(tok_norms_hid256)),
        "col_lm_64x17_16tok_mean_l2_norm": float(np.mean(tok_norms_col64)),
    }

    return {
        "model_task_filter": "regression_only (0 binary/multiclass)",
        "dope_sha256": ch.DOPE_SHA256,
        "policy_b_disjoint_11121_run": {
            "s3_regression_corpus_counts": pretrain_summary.get("s3_regression_corpus_counts", {}),
            "disjointness_checks": pretrain_summary.get("disjointness_checks", {}),
            "stage1b_bottleneck_telemetry_full_s3_regression": pretrain_summary.get(
                "stage1b_bottleneck_telemetry_full_s3_regression", {}
            ),
            "epoch_history": pretrain_summary.get("epoch_history", []),
            "out_of_sample_regression_basis_evaluation": pretrain_summary.get(
                "out_of_sample_regression_basis_evaluation", {}
            ),
        },
        "all_15796_production_run": {
            "total_regression_cells": int(ckpt_15796["total_regression_cells"]),
            "unique_regression_datasets": int(ckpt_15796["unique_regression_datasets"]),
            "fit_evaluation": ckpt_15796["fit_evaluation"],
            "epoch_history": ckpt_15796["epoch_history"],
            "elapsed_seconds": float(ckpt_15796["elapsed_seconds"]),
        },
        "pre_condense_92token_group_activation_telemetry": token_group_telemetry,
    }


def compute_critic_kpi_telemetry(
    train_rows: List[Dict[str, Any]],
    val_rows: List[Dict[str, Any]],
    pools: Dict[str, Dict[str, Any]],
    sos_priors: Dict[str, np.ndarray],
    sp_priors: Dict[str, np.ndarray],
    promos: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Computes Train-OOF (`1,219`) and Holdout (`DL-Val 91`) KPIs for the fitted
    `SOSCalibratedAdapter` (`w_c` continuous log-ratio critic + `w_g` gate-failure critic)
    and `SelfPlayTrajectoryHead` (`sp_dir`).
    """
    out: Dict[str, Any] = {}
    for split_name, rows in (("dl_train_oof_1219", train_rows), ("dl_val_91", val_rows)):
        cont_preds = []
        cont_trues = []
        gate_probs = []
        gate_trues = []
        sp_corrs = []
        promo_ranks = []

        for r in rows:
            cid = r["context_sha256"]
            p = pools[cid]
            sos = sos_priors[cid]
            sp = sp_priors[cid]

            valid = p["valid_cont"]
            lr = np.clip(p["logratio"], -1.5, 1.5)
            if np.sum(valid) > 16:
                lv = lr[valid]
                lv_sd = float(np.std(lv))
                if lv_sd > 1e-6:
                    y_c = (lv - float(np.mean(lv))) / lv_sd
                    cont_preds.append(sos[valid, 0].astype(np.float64))
                    cont_trues.append(y_c.astype(np.float64))

            g_fail = (~p["gate"]).astype(np.int32)
            gate_probs.append(np.clip(sos[:, 1].astype(np.float64), 1e-6, 1.0 - 1e-6))
            gate_trues.append(g_fail)

            u = p["utility"].astype(np.float64)
            if np.std(sp) > 1e-8 and np.std(u) > 1e-8:
                rho, _ = spearmanr(-sp, u)
                if np.isfinite(rho):
                    sp_corrs.append(float(rho))

        c_pred_all = np.concatenate(cont_preds, axis=0)
        c_true_all = np.concatenate(cont_trues, axis=0)
        g_prob_all = np.concatenate(gate_probs, axis=0)
        g_true_all = np.concatenate(gate_trues, axis=0)

        r_p, _ = pearsonr(c_pred_all, c_true_all)
        r_s, _ = spearmanr(c_pred_all, c_true_all)
        ss_res = float(np.sum((c_true_all - c_pred_all) ** 2))
        ss_tot = max(float(np.sum((c_true_all - np.mean(c_true_all)) ** 2)), 1e-12)

        out[split_name] = {
            "n_contexts": len(rows),
            "n_candidate_evaluations": int(len(g_true_all)),
            "sos_continuous_critic_w_c": {
                "pearson_r": float(r_p),
                "spearman_rho": float(r_s),
                "r2": float(1.0 - ss_res / ss_tot),
                "rmse": float(np.sqrt(np.mean((c_true_all - c_pred_all) ** 2))),
            },
            "sos_gate_failure_critic_w_g": {
                "base_invalid_rate": float(np.mean(g_true_all)),
                "auroc": float(roc_auc_score(g_true_all, g_prob_all)),
                "auprc": float(average_precision_score(g_true_all, g_prob_all)),
                "brier_score": float(brier_score_loss(g_true_all, g_prob_all)),
                "log_loss": float(log_loss(g_true_all, g_prob_all)),
                "ece_10bin": compute_ece_10bin(g_true_all, g_prob_all, n_bins=10),
            },
            "self_play_trajectory_head_sp_dir": {
                "mean_context_spearman_with_utility": float(np.mean(sp_corrs)),
                "median_context_spearman_with_utility": float(np.median(sp_corrs)),
            },
        }
    out["total_harvested_promotions"] = int(sum(len(v) for v in promos.values()))
    return out


def compute_detailed_decision_and_ranking_kpis(
    rows: List[Dict[str, Any]],
    packets: Dict[str, Dict[str, Any]],
    pools: Dict[str, Dict[str, Any]],
    u_by_seed_cid: Dict[int, Dict[str, float]],
    inv_by_seed_cid: Dict[int, Dict[str, bool]],
    pick_pool_idx_by_seed_cid: Dict[int, Dict[str, int]],
    scores8_by_seed_cid: Dict[int, Dict[str, np.ndarray]],
    pinv8_by_seed_cid: Dict[int, Dict[str, np.ndarray]],
    r882_u_by_cid: Dict[str, float],
    seeds: Tuple[int, ...] = flg.SEEDS,
) -> Dict[str, Any]:
    """Computes the complete Utility, Oracle Regret/Recovery, Family Safety, Ranking (NDCG/MRR/Spearman),
    Switch Precision, and Invalid-Gate Calibration KPIs for a partition (`rows`).
    """
    base_metrics = flg.compute_partition_metrics(
        rows, u_by_seed_cid, inv_by_seed_cid, r882_u_by_cid, seeds=seeds
    )
    cids = [r["context_sha256"] for r in rows]
    fams = sorted({r["family_id"] for r in rows})
    fam_to_cids = {
        fam: [r["context_sha256"] for r in rows if r["family_id"] == fam] for fam in fams
    }

    # Oracle utilities (Top-8 Oracle and Full-Pool 96-Candidate Oracle)
    top8_oracle_u_by_cid = {c: float(packets[c]["top8_oracle_u"]) for c in cids}
    pool_oracle_u_by_cid = {}
    r882_pool_idx_by_cid = {c: int(packets[c]["top8_indices"][0]) for c in cids}
    r882_inv_by_cid = {c: bool(packets[c]["r882_invalid"]) for c in cids}

    for c in cids:
        p = pools[c]
        valid_u = p["utility"][p["gate"]]
        pool_oracle_u_by_cid[c] = float(np.max(valid_u)) if len(valid_u) > 0 else float(np.max(p["utility"]))

    top8_oracle_ctx = float(np.mean([top8_oracle_u_by_cid[c] for c in cids]))
    top8_oracle_fam = float(
        np.mean([np.mean([top8_oracle_u_by_cid[c] for c in fam_to_cids[f]]) for f in fams])
    )
    pool_oracle_ctx = float(np.mean([pool_oracle_u_by_cid[c] for c in cids]))
    pool_oracle_fam = float(
        np.mean([np.mean([pool_oracle_u_by_cid[c] for c in fam_to_cids[f]]) for f in fams])
    )

    r882_fam = base_metrics["r882_family_utility"]
    r882_ctx = base_metrics["r882_context_utility"]
    model_fam = base_metrics["mean_family_utility"]
    model_ctx = base_metrics["mean_context_utility"]

    top8_headroom_fam = max(top8_oracle_fam - r882_fam, 1e-12)
    top8_headroom_ctx = max(top8_oracle_ctx - r882_ctx, 1e-12)
    pool_headroom_fam = max(pool_oracle_fam - r882_fam, 1e-12)
    pool_headroom_ctx = max(pool_oracle_ctx - r882_ctx, 1e-12)

    # Family win/tie/loss counts across seed-averaged family utilities
    fam_deltas_ens = []
    for f in fams:
        f_r882 = float(np.mean([r882_u_by_cid[c] for c in fam_to_cids[f]]))
        f_mod = float(
            np.mean([np.mean([u_by_seed_cid[s][c] for s in seeds]) for c in fam_to_cids[f]])
        )
        fam_deltas_ens.append(f_mod - f_r882)
    fam_deltas_arr = np.asarray(fam_deltas_ens, dtype=np.float64)

    fam_wins = int(np.sum(fam_deltas_arr > 1e-9))
    fam_losses = int(np.sum(fam_deltas_arr < -1e-9))
    fam_ties = int(len(fams) - fam_wins - fam_losses)

    # Per-seed switch, ranking, and calibration KPIs
    ndcg1_list, ndcg3_list, ndcg8_list = [], [], []
    top1_hit_list, top3_hit_list, mrr_list = [], [], []
    spearman_list, kendall_list = [], []
    sw_cnt_list, sw_win_list, sw_tie_list, sw_loss_list = [], [], [], []
    sw_gain_when_switched = []
    new_inv_sw_list = []
    inv_prevented_list = []
    auroc_list, auprc_list, brier_list, ece_list = [], [], [], []

    for s in seeds:
        sw_c = 0
        sw_w = 0
        sw_t = 0
        sw_l = 0
        new_inv = 0
        prev_inv = 0
        sw_deltas = []

        y_inv_all = []
        p_inv_all = []

        for c in cids:
            pkt = packets[c]
            rel8 = pkt["rel_8"].astype(np.float64)
            u8 = pkt["utility_8"].astype(np.float64)
            g8 = pkt["gate_8"]
            oracle_j = int(pkt["top8_oracle_j"])

            sc8 = scores8_by_seed_cid[s][c]
            pinv8 = pinv8_by_seed_cid[s][c]
            s_eff = flg.apply_soft_and_hard_veto_scores(sc8, pinv8, lam=2.0, tau=0.15)

            ndcg1_list.append(compute_ndcg_at_k(rel8, s_eff, 1))
            ndcg3_list.append(compute_ndcg_at_k(rel8, s_eff, 3))
            ndcg8_list.append(compute_ndcg_at_k(rel8, s_eff, 8))

            order_s = np.argsort(-s_eff, kind="mergesort")
            rank_oracle = int(np.where(order_s == oracle_j)[0][0]) + 1
            top1_hit_list.append(1.0 if rank_oracle == 1 else 0.0)
            top3_hit_list.append(1.0 if rank_oracle <= 3 else 0.0)
            mrr_list.append(1.0 / float(rank_oracle))

            if np.std(s_eff) > 1e-8 and np.std(u8) > 1e-8:
                rho, _ = spearmanr(s_eff, u8)
                kt, _ = kendalltau(s_eff, u8)
                if np.isfinite(rho):
                    spearman_list.append(float(rho))
                if np.isfinite(kt):
                    kendall_list.append(float(kt))

            pick_pool_idx = int(pick_pool_idx_by_seed_cid[s][c])
            r882_pool_idx = r882_pool_idx_by_cid[c]
            u_mod = float(u_by_seed_cid[s][c])
            u_r882 = float(r882_u_by_cid[c])
            inv_mod = bool(inv_by_seed_cid[s][c])
            inv_r882 = r882_inv_by_cid[c]

            if pick_pool_idx != r882_pool_idx:
                sw_c += 1
                d_u = u_mod - u_r882
                sw_deltas.append(d_u)
                if d_u > 1e-9:
                    sw_w += 1
                elif d_u < -1e-9:
                    sw_l += 1
                else:
                    sw_t += 1
                if inv_mod and not inv_r882:
                    new_inv += 1
                if inv_r882 and not inv_mod:
                    prev_inv += 1

            y_inv_all.append((~g8).astype(np.int32))
            p_inv_all.append(np.clip(pinv8, 1e-6, 1.0 - 1e-6))

        sw_cnt_list.append(sw_c)
        sw_win_list.append(sw_w)
        sw_tie_list.append(sw_t)
        sw_loss_list.append(sw_l)
        new_inv_sw_list.append(new_inv)
        inv_prevented_list.append(prev_inv)
        if sw_deltas:
            sw_gain_when_switched.append(float(np.mean(sw_deltas)))

        y_inv_vec = np.concatenate(y_inv_all, axis=0)
        p_inv_vec = np.concatenate(p_inv_all, axis=0)
        if len(np.unique(y_inv_vec)) > 1:
            auroc_list.append(float(roc_auc_score(y_inv_vec, p_inv_vec)))
            auprc_list.append(float(average_precision_score(y_inv_vec, p_inv_vec)))
        brier_list.append(float(brier_score_loss(y_inv_vec, p_inv_vec)))
        ece_list.append(compute_ece_10bin(y_inv_vec, p_inv_vec, n_bins=10))

    mean_sw = float(np.mean(sw_cnt_list))
    mean_sw_w = float(np.mean(sw_win_list))
    mean_sw_l = float(np.mean(sw_loss_list))
    r882_inv_total = int(sum(1 for c in cids if r882_inv_by_cid[c]))

    base_metrics.update(
        {
            "oracle_and_regret_kpis": {
                "top8_oracle_family_utility": top8_oracle_fam,
                "top8_oracle_context_utility": top8_oracle_ctx,
                "top8_oracle_regret_family": float(top8_oracle_fam - model_fam),
                "top8_oracle_regret_context": float(top8_oracle_ctx - model_ctx),
                "top8_oracle_recovery_fraction_family": float(
                    (model_fam - r882_fam) / top8_headroom_fam
                ),
                "top8_oracle_recovery_fraction_context": float(
                    (model_ctx - r882_ctx) / top8_headroom_ctx
                ),
                "pool_96_oracle_family_utility": pool_oracle_fam,
                "pool_96_oracle_context_utility": pool_oracle_ctx,
                "pool_96_oracle_regret_family": float(pool_oracle_fam - model_fam),
                "pool_96_oracle_regret_context": float(pool_oracle_ctx - model_ctx),
                "pool_96_oracle_recovery_fraction_family": float(
                    (model_fam - r882_fam) / pool_headroom_fam
                ),
                "pool_96_oracle_recovery_fraction_context": float(
                    (model_ctx - r882_ctx) / pool_headroom_ctx
                ),
            },
            "family_safety_and_win_loss_kpis": {
                "family_wins_vs_r882": fam_wins,
                "family_ties_vs_r882": fam_ties,
                "family_losses_vs_r882": fam_losses,
                "family_win_to_loss_ratio": float(fam_wins / max(fam_losses, 1)),
                "harmed_families_gt_001": base_metrics["mean_harmed_families_gt_001"],
                "harmed_families_gt_0": int(np.sum(fam_deltas_arr < -1e-9)),
                "worst_family_delta": float(np.min(fam_deltas_arr)),
                "p05_family_delta": float(np.percentile(fam_deltas_arr, 5)),
                "p50_family_delta": float(np.percentile(fam_deltas_arr, 50)),
                "p95_family_delta": float(np.percentile(fam_deltas_arr, 95)),
                "best_family_delta": float(np.max(fam_deltas_arr)),
            },
            "ranking_quality_kpis": {
                "ndcg_at_1": float(np.mean(ndcg1_list)),
                "ndcg_at_3": float(np.mean(ndcg3_list)),
                "ndcg_at_8": float(np.mean(ndcg8_list)),
                "top1_oracle8_hit_rate": float(np.mean(top1_hit_list)),
                "top3_oracle8_hit_rate": float(np.mean(top3_hit_list)),
                "mrr_oracle8": float(np.mean(mrr_list)),
                "spearman_rank_corr_mean": float(np.mean(spearman_list)) if spearman_list else 0.0,
                "spearman_rank_corr_median": float(np.median(spearman_list))
                if spearman_list
                else 0.0,
                "kendall_tau_mean": float(np.mean(kendall_list)) if kendall_list else 0.0,
            },
            "switch_and_override_kpis": {
                "mean_switches_count": mean_sw,
                "switch_rate_pct": float(100.0 * mean_sw / max(len(cids), 1)),
                "mean_switch_wins": mean_sw_w,
                "mean_switch_ties": float(np.mean(sw_tie_list)),
                "mean_switch_losses": mean_sw_l,
                "switch_win_loss_precision_pct": float(
                    100.0 * mean_sw_w / max(mean_sw_w + mean_sw_l, 1e-9)
                ),
                "mean_utility_gain_given_switch": float(np.mean(sw_gain_when_switched))
                if sw_gain_when_switched
                else 0.0,
            },
            "invalid_gate_and_calibration_kpis": {
                "r882_control_invalid_picks": r882_inv_total,
                "r882_control_invalid_rate_pct": float(100.0 * r882_inv_total / max(len(cids), 1)),
                "model_mean_invalid_picks": base_metrics["mean_invalid_picks"],
                "model_invalid_rate_pct": float(
                    100.0 * base_metrics["mean_invalid_picks"] / max(len(cids), 1)
                ),
                "mean_new_invalid_switches": float(np.mean(new_inv_sw_list)),
                "mean_invalid_picks_prevented_vs_r882": float(np.mean(inv_prevented_list)),
                "top8_veto_clf_auroc": float(np.mean(auroc_list)) if auroc_list else 0.0,
                "top8_veto_clf_auprc": float(np.mean(auprc_list)) if auprc_list else 0.0,
                "top8_veto_clf_brier_score": float(np.mean(brier_list)),
                "top8_veto_clf_ece_10bin": float(np.mean(ece_list)),
            },
        }
    )
    return base_metrics


def compute_feature_block_and_neural_attribution_kpis(
    prod_1310_payload: Dict[str, Any],
    packets: Dict[str, Dict[str, Any]],
    all_cids: List[str],
    device: torch.device,
) -> Dict[str, Any]:
    """Computes feature-block importance shares for both the 616D LightGBM ranker/veto-classifier
    and input-gradient saliency for the 616D `PreCondenseTop8NeuralReranker`.
    """
    ranker_booster = lgb.Booster(model_str=prod_1310_payload["top8_lgbm_ranker_616_model_str"])
    clf_booster = lgb.Booster(model_str=prod_1310_payload["top8_lgbm_veto_clf_616_model_str"])

    gain_rk = np.asarray(ranker_booster.feature_importance(importance_type="gain"), dtype=np.float64)
    split_rk = np.asarray(ranker_booster.feature_importance(importance_type="split"), dtype=np.float64)
    gain_clf = np.asarray(clf_booster.feature_importance(importance_type="gain"), dtype=np.float64)

    # Neural input-gradient saliency across all 1,310 contexts
    nn_model = flg.PreCondenseTop8NeuralReranker(in_dim=flg.ENRICHED_DIM_K8, hidden_dim=128).to(
        device
    )
    nn_model.load_state_dict(prod_1310_payload["top8_neural_reranker_state_dict"], strict=True)
    nn_model.eval()

    feat_mean = np.asarray(prod_1310_payload["top8_neural_reranker_feat_mean"], dtype=np.float32)
    feat_std = np.asarray(prod_1310_payload["top8_neural_reranker_feat_std"], dtype=np.float32)

    X_raw = np.stack([packets[c]["X_616"] for c in all_cids], axis=0).astype(np.float32)
    X_norm = np.clip((X_raw - feat_mean[None, None, :]) / feat_std[None, None, :], -10.0, 10.0)
    X_norm[:, :, 2] = X_raw[:, :, 2]

    X_t = torch.tensor(X_norm, dtype=torch.float32, device=device, requires_grad=True)
    scores_t, inv_probs_t = nn_model(X_t)
    (scores_t.abs().mean() + inv_probs_t.mean()).backward()
    nn_grad_saliency = X_t.grad.detach().abs().mean(dim=(0, 1)).cpu().numpy().astype(np.float64)

    blocks = {
        "block1_rerank8_cost_and_pairwise_gaps_28d": (0, 28),
        "block2_sem24_proj16_geometry_12d": (28, 40),
        "block3_program_412d_sem24_proj16_452d": (40, 492),
        "block4_ctx_lm48_and_dope25_76d": (492, 568),
        "block5_h256_stage2_bridge_and_critics_12d": (568, 580),
        "block6_h256_pre_condense_and_stage1_s3_encoder_36d": (580, 616),
    }

    def _summarize_imp(vec: np.ndarray) -> Dict[str, float]:
        tot = max(float(np.sum(vec)), 1e-12)
        return {
            bname: float(100.0 * np.sum(vec[lo:hi]) / tot) for bname, (lo, hi) in blocks.items()
        }

    h256_extra_names = [
        "cand_dp_F_z",
        "cand_dp_F_gap_vs_slot0",
        "cand_dp_G_s3_z",
        "cand_dp_G_s3_gap_vs_slot0",
        "cand_sos_h_cost_z",
        "cand_sos_h_cost_gap_vs_slot0",
        "cand_sos_gate_risk",
        "cand_sos_gate_risk_gap_vs_slot0",
        "cand_sp_prior_z",
        "cand_sp_prior_gap_vs_slot0",
        "cand_h256_combined_cost_z",
        "cand_h256_combined_cost_zgap_vs_slot0",
        *[f"ctx_pre_condense_log_norm_{i}" for i in range(11)],
        "ctx_hidden256_mean",
        "ctx_hidden256_std",
        "ctx_hidden256_l2norm",
        "ctx_col64_active_frac",
        "ctx_col64x17_active_mean",
        "ctx_col64x17_active_std",
        "ctx_dope_256_565_mean",
        "ctx_dope_256_565_std",
        "ctx_dope_565_629_active_mean",
        *[f"ctx_stage1_s3_reg_7split_vec16_{i}" for i in range(16)],
    ]
    assert len(h256_extra_names) == 48

    h256_48_breakdown = []
    tot_rk_gain = max(float(np.sum(gain_rk)), 1e-12)
    tot_clf_gain = max(float(np.sum(gain_clf)), 1e-12)
    tot_nn_sal = max(float(np.sum(nn_grad_saliency)), 1e-12)

    for idx_extra, fname in enumerate(h256_extra_names):
        col = 568 + idx_extra
        h256_48_breakdown.append(
            {
                "col_index": col,
                "name": fname,
                "lgbm_ranker_gain_pct": float(100.0 * gain_rk[col] / tot_rk_gain),
                "lgbm_ranker_splits": int(split_rk[col]),
                "lgbm_veto_clf_gain_pct": float(100.0 * gain_clf[col] / tot_clf_gain),
                "neural_reranker_grad_saliency_pct": float(
                    100.0 * nn_grad_saliency[col] / tot_nn_sal
                ),
            }
        )

    return {
        "lgbm_ranker_616_block_gain_share_pct": _summarize_imp(gain_rk),
        "lgbm_ranker_616_block_split_share_pct": _summarize_imp(split_rk),
        "lgbm_veto_clf_616_block_gain_share_pct": _summarize_imp(gain_clf),
        "neural_reranker_616_block_grad_saliency_share_pct": _summarize_imp(nn_grad_saliency),
        "h256_48d_features_telemetry": h256_48_breakdown,
    }


def main() -> None:
    t0 = time.time()
    device = torch.device("cpu")
    torch.set_num_threads(2)

    net, guard, dope_snap_before, w1_mat = p255.load_fused_network(device)
    train_rows, val_rows = ch.load_cohorts()
    all_rows = train_rows + val_rows
    all_cids = [r["context_sha256"] for r in all_rows]
    fam_by_cid = {r["context_sha256"]: r["family_id"] for r in all_rows}

    pools = ch.load_pools(all_rows)
    r882_cache = p255.load_r882_cache_readonly(all_rows)
    raw_cache = r256.load_pre_condense_cache(CACHE_PATH)
    static_cache = torch.load(
        OUT_DIR / "cache" / "static_adapter_terms.pt", map_location="cpu", weights_only=False
    )

    w1_f32 = w1_mat.astype(np.float32) / np.sqrt(412.0)
    for r in all_rows:
        c = r["context_sha256"]
        pools[c]["proj16"] = (pools[c]["program"] @ w1_f32).astype(np.float32)

    promos = ch.load_harvested_promotions(train_rows)
    bt_cache_path = OUT_DIR / "cache" / "base_terms_and_priors_cache.pt"
    if bt_cache_path.exists():
        btc = torch.load(bt_cache_path, map_location="cpu", weights_only=False)
        base_terms = btc["base_terms"]
        sos_priors = btc["sos_priors"]
        sp_priors = btc["sp_priors"]
        cand_feats = btc["cand_feats"]
        r882_u_by_cid = {r["context_sha256"]: base_terms[r["context_sha256"]]["u_r882"] for r in all_rows}
    else:
        cand_feats = s1.build_candidate_features(all_rows, pools, r882_cache, raw_cache)
        sos_priors, sp_priors, _ = s1.fit_decomposed_hurdle_and_sp_priors(
            train_rows, val_rows, pools, cand_feats, promos, seed=r256.FOLD_SEED
        )
        base_terms = {}
        r882_u_by_cid = {}
        for r in all_rows:
            cid = r["context_sha256"]
            rcost = r882_cache[cid]["total_cost"].astype(np.float64)
            bt = r256.compute_context_base_terms(
                cid, pools[cid], rcost, sos_priors[cid], sp_priors[cid], static_cache[cid], device
            )
            base_terms[cid] = bt
            r882_u_by_cid[cid] = bt["u_r882"]

    s3_ckpt_11121 = torch.load(
        OUT_DIR / "full_s3_reg_7split_pretrained_encoder.pt", map_location="cpu", weights_only=False
    )
    s3_scaler_policy_b = s3_ckpt_11121["scaler_policy_b"]

    idx4_by_seed_cid: Dict[int, Dict[str, np.ndarray]] = {s: {} for s in flg.SEEDS}
    for s in flg.SEEDS:
        for r in all_rows:
            c = r["context_sha256"]
            rng_c = np.random.default_rng(ch.stream_seed("h256_idx4", s, c))
            idx4_by_seed_cid[s][c] = rng_c.choice(len(base_terms[c]["rcost"]), size=4, replace=False)

    dp_f_by_cid, dp_g_by_cid, stage2_cache = flg.compute_oof_and_val_stage2_bridge_priors(
        train_rows, val_rows, raw_cache, base_terms, s3_scaler_policy_b, device
    )

    pkt_cache_path = OUT_DIR / "cache" / "flagship_packets_616.pt"
    if pkt_cache_path.exists():
        pkt_cache = torch.load(pkt_cache_path, map_location="cpu", weights_only=False)
        packets = pkt_cache["packets"]
        r882_u_by_cid = pkt_cache["r882_u_by_cid"]
    else:
        s3_vec16_by_cid = flg.extract_stage1_s3_encoder_context_vectors(
            raw_cache,
            OUT_DIR / "full_s3_reg_7split_pretrained_encoder.pt",
            "scaler_policy_b",
            device,
        )
        packets = {}
        for r in all_rows:
            cid = r["context_sha256"]
            p = pools[cid]
            rcost = r882_cache[cid]["total_cost"].astype(np.float64)
            aprior = r882_cache[cid]["anchor_prior"].astype(np.float64)
            bt = base_terms[cid]
            packets[cid] = flg.build_enriched_topk_packet_for_context(
                cid=cid,
                family_id=r["family_id"],
                rcost=rcost,
                aprior=aprior,
                prog_full=p["program"],
                sem24_full=p["sem24"],
                proj16_full=p["proj16"],
                fitness_full=p["fitness"],
                gate_full=p["gate"],
                utility_full=p["utility"],
                raw_ctx=raw_cache[cid],
                s3_ctx_vec16=s3_vec16_by_cid[cid],
                dp_f_full=dp_f_by_cid[cid],
                dp_g_full=dp_g_by_cid[cid],
                h_cost_full=bt["h_cost"],
                gate_risk_full=sos_priors[cid][:, 1],
                sp_prior_full=sp_priors[cid],
                k=flg.SLICE_K,
            )

    # 1. Stage-1 S3 Pretraining KPIs
    stage1_kpis = compute_stage1_s3_kpi_telemetry(raw_cache, device)

    # 2. Critic KPIs
    critic_kpis = compute_critic_kpi_telemetry(
        train_rows, val_rows, pools, sos_priors, sp_priors, promos
    )

    # 3. Load the 15 saved fold checkpoints & outer_top8_preds_cache, enrich fold checkpoints,
    #    and collect full train-FIT, outer-OOF, and DL-Val predictions!
    fold_to_cids = stage2_cache["fold_to_cids"]
    val_cids = [r["context_sha256"] for r in val_rows]
    preds_cache = torch.load(
        OUT_DIR / "cache" / "outer_top8_preds_cache.pt", map_location="cpu", weights_only=False
    )

    oof_static_u: Dict[int, Dict[str, float]] = {s: {} for s in flg.SEEDS}
    oof_static_inv: Dict[int, Dict[str, bool]] = {s: {} for s in flg.SEEDS}
    oof_static_pool_idx: Dict[int, Dict[str, int]] = {s: {} for s in flg.SEEDS}
    oof_static_sc8: Dict[int, Dict[str, np.ndarray]] = {s: {} for s in flg.SEEDS}
    oof_static_pinv8: Dict[int, Dict[str, np.ndarray]] = {s: {} for s in flg.SEEDS}

    oof_nn_u: Dict[int, Dict[str, float]] = {s: {} for s in flg.SEEDS}
    oof_nn_inv: Dict[int, Dict[str, bool]] = {s: {} for s in flg.SEEDS}
    oof_nn_pool_idx: Dict[int, Dict[str, int]] = {s: {} for s in flg.SEEDS}
    oof_nn_sc8: Dict[int, Dict[str, np.ndarray]] = {s: {} for s in flg.SEEDS}
    oof_nn_pinv8: Dict[int, Dict[str, np.ndarray]] = {s: {} for s in flg.SEEDS}

    oof_active_u: Dict[int, Dict[str, float]] = {s: {} for s in flg.SEEDS}
    oof_active_inv: Dict[int, Dict[str, bool]] = {s: {} for s in flg.SEEDS}
    oof_active_pool_idx: Dict[int, Dict[str, int]] = {s: {} for s in flg.SEEDS}

    val_static_u: Dict[int, Dict[str, float]] = {s: {} for s in flg.SEEDS}
    val_static_inv: Dict[int, Dict[str, bool]] = {s: {} for s in flg.SEEDS}
    val_static_pool_idx: Dict[int, Dict[str, int]] = {s: {} for s in flg.SEEDS}
    val_static_sc8: Dict[int, Dict[str, np.ndarray]] = {s: {} for s in flg.SEEDS}
    val_static_pinv8: Dict[int, Dict[str, np.ndarray]] = {s: {} for s in flg.SEEDS}

    val_nn_u: Dict[int, Dict[str, float]] = {s: {} for s in flg.SEEDS}
    val_nn_inv: Dict[int, Dict[str, bool]] = {s: {} for s in flg.SEEDS}
    val_nn_pool_idx: Dict[int, Dict[str, int]] = {s: {} for s in flg.SEEDS}
    val_nn_sc8: Dict[int, Dict[str, np.ndarray]] = {s: {} for s in flg.SEEDS}
    val_nn_pinv8: Dict[int, Dict[str, np.ndarray]] = {s: {} for s in flg.SEEDS}

    val_active_u: Dict[int, Dict[str, float]] = {s: {} for s in flg.SEEDS}
    val_active_inv: Dict[int, Dict[str, bool]] = {s: {} for s in flg.SEEDS}
    val_active_pool_idx: Dict[int, Dict[str, int]] = {s: {} for s in flg.SEEDS}

    infold_fit_gains = []
    inner_oof_fit_gains = []
    fold_cell_telemetry = []

    for s in flg.SEEDS:
        val_sc_folds = []
        val_pinv_folds = []
        val_nn_sc_folds = []
        val_nn_pinv_folds = []

        for k in range(5):
            ce = preds_cache[(s, k)]
            assess_cids = ce["assess_cids"]
            fit_folds = [f for f in range(5) if f != k]
            fit_cids = [c for f in fit_folds for c in fold_to_cids[f]]

            nn_ckpt_path = CKPT_DIR / f"stage2_top8_neural_reranker_seed{s}_fold{k}.pt"
            nn_payload = torch.load(nn_ckpt_path, map_location=device, weights_only=False)
            cfg = nn_payload["selected_cfg"]
            inner_oof_fit_gains.append(float(cfg["inner_fit_family_gain"]))

            X_fit_568 = np.concatenate([packets[c]["X_568"] for c in fit_cids], axis=0)
            X_fit_616 = np.concatenate([packets[c]["X_616"] for c in fit_cids], axis=0)
            X_fit_580 = X_fit_616[:, :580]
            y_fit = np.concatenate([packets[c]["rel_8"] for c in fit_cids], axis=0)
            y_inv_fit = np.concatenate(
                [(~packets[c]["gate_8"]).astype(np.int32) for c in fit_cids], axis=0
            )
            sel_lr = 0.05

            outer_ranker_568 = flg.fit_lgbm_ranker_200(
                X_fit_568,
                y_fit,
                len(fit_cids),
                lr=sel_lr,
                random_state=flg.stream_seed(r256.FOLD_SEED, s, k, "RERANK8_CR", "outer", sel_lr),
            )
            outer_ranker_580 = flg.fit_lgbm_ranker_200(
                X_fit_580,
                y_fit,
                len(fit_cids),
                lr=sel_lr,
                random_state=flg.stream_seed(r256.FOLD_SEED, s, k, "H256_580", "outer", sel_lr),
            )
            outer_ranker_616 = flg.fit_lgbm_ranker_200(
                X_fit_616,
                y_fit,
                len(fit_cids),
                lr=sel_lr,
                random_state=flg.stream_seed(r256.FOLD_SEED, s, k, "H256_616", "outer", sel_lr),
            )
            outer_clf_568 = flg.fit_veto_classifier(
                X_fit_568,
                y_inv_fit,
                random_state=flg.stream_seed(r256.FOLD_SEED, s, k, "RERANK8_CR_VETO", "outer_clf"),
            )
            outer_clf_580 = flg.fit_veto_classifier(
                X_fit_580,
                y_inv_fit,
                random_state=flg.stream_seed(r256.FOLD_SEED, s, k, "H256_580_VETO", "outer_clf"),
            )
            outer_clf_616 = flg.fit_veto_classifier(
                X_fit_616,
                y_inv_fit,
                random_state=flg.stream_seed(r256.FOLD_SEED, s, k, "H256_616_VETO", "outer_clf"),
            )

            # In-fold FIT evaluation KPIs
            s568_fit = flg.standardize_scores_per_query(
                np.asarray(
                    outer_ranker_568.booster_.predict(X_fit_568, num_iteration=100), dtype=np.float64
                ).reshape(len(fit_cids), flg.SLICE_K)
            )
            s580_fit = flg.standardize_scores_per_query(
                np.asarray(
                    outer_ranker_580.booster_.predict(X_fit_580, num_iteration=100), dtype=np.float64
                ).reshape(len(fit_cids), flg.SLICE_K)
            )
            s616_fit = flg.standardize_scores_per_query(
                np.asarray(
                    outer_ranker_616.booster_.predict(X_fit_616, num_iteration=100), dtype=np.float64
                ).reshape(len(fit_cids), flg.SLICE_K)
            )
            p568_fit = np.asarray(outer_clf_568.booster_.predict(X_fit_568), dtype=np.float64).reshape(
                len(fit_cids), flg.SLICE_K
            )
            p580_fit = np.asarray(outer_clf_580.booster_.predict(X_fit_580), dtype=np.float64).reshape(
                len(fit_cids), flg.SLICE_K
            )
            p616_fit = np.asarray(outer_clf_616.booster_.predict(X_fit_616), dtype=np.float64).reshape(
                len(fit_cids), flg.SLICE_K
            )
            sc_fit = 0.45 * s568_fit + 0.35 * s580_fit + 0.20 * s616_fit
            pinv_fit = 0.45 * p568_fit + 0.35 * p580_fit + 0.20 * p616_fit
            picks_fit = flg.select_picks_from_scores(sc_fit, pinv_fit, lam=1.5, tau=1.0, margin=0.05)

            fit_fams = sorted({fam_by_cid[c] for c in fit_cids})
            fit_u_diff = np.array(
                [
                    float(packets[c]["utility_8"][int(picks_fit[i])] - packets[c]["r882_u"])
                    for i, c in enumerate(fit_cids)
                ],
                dtype=np.float64,
            )
            fit_fam_gain = float(
                np.mean(
                    [
                        np.mean([fit_u_diff[i] for i, c in enumerate(fit_cids) if fam_by_cid[c] == f])
                        for f in fit_fams
                    ]
                )
            )
            fit_inv_cnt = int(
                sum(1 for i, c in enumerate(fit_cids) if not packets[c]["gate_8"][int(picks_fit[i])])
            )
            infold_fit_gains.append(fit_fam_gain)

            # Assess-fold OOF evaluation KPIs
            sc_oof, pinv_oof, sc_nn_oof, pinv_nn_oof = flg.compute_bagged_multi_view_scores_and_risk(
                ce["oof"], assess_cids, packets
            )
            picks_oof = flg.select_picks_from_scores(sc_oof, pinv_oof, lam=1.5, tau=1.0, margin=0.05)
            picks_nn_oof = flg.select_picks_from_scores(
                sc_nn_oof, pinv_nn_oof, lam=1.5, tau=1.0, margin=0.10
            )

            assess_fams = sorted({fam_by_cid[c] for c in assess_cids})
            oof_u_diff = np.array(
                [
                    float(packets[c]["utility_8"][int(picks_oof[i])] - packets[c]["r882_u"])
                    for i, c in enumerate(assess_cids)
                ],
                dtype=np.float64,
            )
            oof_fam_gain = float(
                np.mean(
                    [
                        np.mean(
                            [oof_u_diff[i] for i, c in enumerate(assess_cids) if fam_by_cid[c] == f]
                        )
                        for f in assess_fams
                    ]
                )
            )
            oof_inv_cnt = int(
                sum(
                    1
                    for i, c in enumerate(assess_cids)
                    if not packets[c]["gate_8"][int(picks_oof[i])]
                )
            )

            cell_kpi = {
                "seed": int(s),
                "fold": int(k),
                "n_fit_contexts": len(fit_cids),
                "n_assess_contexts": len(assess_cids),
                "selected_cfg": {
                    "ensemble_weights": {"w568": 0.40, "w580": 0.30, "w616": 0.15, "wnn": 0.10, "wbr": 0.05},
                    "soft_lambda": 1.5,
                    "veto_tau": 1.0,
                    "margin": 0.05,
                },
                "infold_fit_family_gain_vs_r882": fit_fam_gain,
                "infold_fit_invalid_picks": fit_inv_cnt,
                "inner_oof_fit_family_gain_vs_r882": float(cfg["inner_fit_family_gain"]),
                "inner_oof_fit_invalid_picks": int(cfg["inner_fit_invalid_count"]),
                "outer_assess_oof_family_gain_vs_r882": oof_fam_gain,
                "outer_assess_oof_invalid_picks": oof_inv_cnt,
            }
            fold_cell_telemetry.append(cell_kpi)

            # Enrich the fold checkpoint on disk with the LightGBM booster strings + cell KPI telemetry
            nn_payload["outer_ranker_568_model_str"] = outer_ranker_568.booster_.model_to_string()
            nn_payload["outer_ranker_580_model_str"] = outer_ranker_580.booster_.model_to_string()
            nn_payload["outer_ranker_616_model_str"] = outer_ranker_616.booster_.model_to_string()
            nn_payload["outer_clf_568_model_str"] = outer_clf_568.booster_.model_to_string()
            nn_payload["outer_clf_580_model_str"] = outer_clf_580.booster_.model_to_string()
            nn_payload["outer_clf_616_model_str"] = outer_clf_616.booster_.model_to_string()
            nn_payload["kpi_telemetry"] = cell_kpi
            torch.save(nn_payload, nn_ckpt_path)

            for i_c, cid in enumerate(assess_cids):
                pkt = packets[cid]
                j_st = int(picks_oof[i_c])
                oof_static_u[s][cid] = float(pkt["utility_8"][j_st])
                oof_static_inv[s][cid] = bool(not pkt["gate_8"][j_st])
                oof_static_pool_idx[s][cid] = int(pkt["top8_indices"][j_st])
                oof_static_sc8[s][cid] = sc_oof[i_c]
                oof_static_pinv8[s][cid] = pinv_oof[i_c]

                j_nn = int(picks_nn_oof[i_c])
                oof_nn_u[s][cid] = float(pkt["utility_8"][j_nn])
                oof_nn_inv[s][cid] = bool(not pkt["gate_8"][j_nn])
                oof_nn_pool_idx[s][cid] = int(pkt["top8_indices"][j_nn])
                oof_nn_sc8[s][cid] = sc_nn_oof[i_c]
                oof_nn_pinv8[s][cid] = pinv_nn_oof[i_c]

                act_res = flg.evaluate_active_2plus2_unified_on_context(
                    cid=cid,
                    bt=base_terms[cid],
                    pkt=pkt,
                    raw_ctx=raw_cache[cid],
                    idx4=idx4_by_seed_cid[s][cid],
                    static_pick_j=j_st,
                    static_scores_8=sc_oof[i_c],
                    static_pinv_8=pinv_oof[i_c],
                    stage2_cache=stage2_cache,
                    seed=s,
                    fold_k=k,
                    device=device,
                    dp_f_by_cid=dp_f_by_cid,
                    dp_g_by_cid=dp_g_by_cid,
                )
                oof_active_u[s][cid] = act_res["utility"]
                oof_active_inv[s][cid] = act_res["invalid_pick"]
                oof_active_pool_idx[s][cid] = int(act_res["pick_index"])

            sc_val_k, pinv_val_k, sc_nn_val_k, pinv_nn_val_k = (
                flg.compute_bagged_multi_view_scores_and_risk(ce["val"], val_cids, packets)
            )
            val_sc_folds.append(flg.standardize_scores_per_query(sc_val_k))
            val_pinv_folds.append(pinv_val_k)
            val_nn_sc_folds.append(flg.standardize_scores_per_query(sc_nn_val_k))
            val_nn_pinv_folds.append(pinv_nn_val_k)

        val_sc_ens = np.mean(val_sc_folds, axis=0)
        val_pinv_ens = np.mean(val_pinv_folds, axis=0)
        val_nn_sc_ens = np.mean(val_nn_sc_folds, axis=0)
        val_nn_pinv_ens = np.mean(val_nn_pinv_folds, axis=0)

        val_picks_static = flg.select_picks_from_scores(
            val_sc_ens, val_pinv_ens, lam=1.5, tau=1.0, margin=0.05
        )
        val_picks_nn = flg.select_picks_from_scores(
            val_nn_sc_ens, val_nn_pinv_ens, lam=1.5, tau=1.0, margin=0.10
        )

        for i_v, cid in enumerate(val_cids):
            pkt = packets[cid]
            j_st = int(val_picks_static[i_v])
            val_static_u[s][cid] = float(pkt["utility_8"][j_st])
            val_static_inv[s][cid] = bool(not pkt["gate_8"][j_st])
            val_static_pool_idx[s][cid] = int(pkt["top8_indices"][j_st])
            val_static_sc8[s][cid] = val_sc_ens[i_v]
            val_static_pinv8[s][cid] = val_pinv_ens[i_v]

            j_nn = int(val_picks_nn[i_v])
            val_nn_u[s][cid] = float(pkt["utility_8"][j_nn])
            val_nn_inv[s][cid] = bool(not pkt["gate_8"][j_nn])
            val_nn_pool_idx[s][cid] = int(pkt["top8_indices"][j_nn])
            val_nn_sc8[s][cid] = val_nn_sc_ens[i_v]
            val_nn_pinv8[s][cid] = val_nn_pinv_ens[i_v]

            act_res_v = flg.evaluate_active_2plus2_unified_on_context(
                cid=cid,
                bt=base_terms[cid],
                pkt=pkt,
                raw_ctx=raw_cache[cid],
                idx4=idx4_by_seed_cid[s][cid],
                static_pick_j=j_st,
                static_scores_8=val_sc_ens[i_v],
                static_pinv_8=val_pinv_ens[i_v],
                stage2_cache=stage2_cache,
                seed=s,
                fold_k=None,
                device=device,
                dp_f_by_cid=dp_f_by_cid,
                dp_g_by_cid=dp_g_by_cid,
            )
            val_active_u[s][cid] = act_res_v["utility"]
            val_active_inv[s][cid] = act_res_v["invalid_pick"]
            val_active_pool_idx[s][cid] = int(act_res_v["pick_index"])

    with p_s3.HIST_ROLES_PATH.open() as f:
        hist_roles = json.load(f)["roles"]
    row_by_cid = {r["context_sha256"]: r for r in all_rows}
    partitions = {
        "dl_train_oof_1219": train_rows,
        "dl_val_91": val_rows,
        "dl_val_71": [row_by_cid[c] for c in hist_roles["dl_val_71"] if c in row_by_cid],
        "st_val_gate_32": [row_by_cid[c] for c in hist_roles["st_val_gate_32"] if c in row_by_cid],
        "dl_train_inner_203": [
            row_by_cid[c] for c in hist_roles["dl_train_inner_203"] if c in row_by_cid
        ],
        "strict_train_fit_881": [
            row_by_cid[c] for c in hist_roles["strict_train_fit_881"] if c in row_by_cid
        ],
    }

    comb_static_u = {s: {**oof_static_u[s], **val_static_u[s]} for s in flg.SEEDS}
    comb_static_inv = {s: {**oof_static_inv[s], **val_static_inv[s]} for s in flg.SEEDS}
    comb_static_pidx = {s: {**oof_static_pool_idx[s], **val_static_pool_idx[s]} for s in flg.SEEDS}
    comb_static_sc8 = {s: {**oof_static_sc8[s], **val_static_sc8[s]} for s in flg.SEEDS}
    comb_static_pinv8 = {s: {**oof_static_pinv8[s], **val_static_pinv8[s]} for s in flg.SEEDS}

    comb_nn_u = {s: {**oof_nn_u[s], **val_nn_u[s]} for s in flg.SEEDS}
    comb_nn_inv = {s: {**oof_nn_inv[s], **val_nn_inv[s]} for s in flg.SEEDS}
    comb_nn_pidx = {s: {**oof_nn_pool_idx[s], **val_nn_pool_idx[s]} for s in flg.SEEDS}
    comb_nn_sc8 = {s: {**oof_nn_sc8[s], **val_nn_sc8[s]} for s in flg.SEEDS}
    comb_nn_pinv8 = {s: {**oof_nn_pinv8[s], **val_nn_pinv8[s]} for s in flg.SEEDS}

    comb_active_u = {s: {**oof_active_u[s], **val_active_u[s]} for s in flg.SEEDS}
    comb_active_inv = {s: {**oof_active_inv[s], **val_active_inv[s]} for s in flg.SEEDS}
    comb_active_pidx = {s: {**oof_active_pool_idx[s], **val_active_pool_idx[s]} for s in flg.SEEDS}

    flagship_detailed_kpis = {
        "H256_STATIC_0REV_TOP8_SOFT_VETO": {
            pname: compute_detailed_decision_and_ranking_kpis(
                prows,
                packets,
                pools,
                comb_static_u,
                comb_static_inv,
                comb_static_pidx,
                comb_static_sc8,
                comb_static_pinv8,
                r882_u_by_cid,
            )
            for pname, prows in partitions.items()
        },
        "H256_STATIC_0REV_TOP8_NEURAL_ONLY": {
            pname: compute_detailed_decision_and_ranking_kpis(
                prows,
                packets,
                pools,
                comb_nn_u,
                comb_nn_inv,
                comb_nn_pidx,
                comb_nn_sc8,
                comb_nn_pinv8,
                r882_u_by_cid,
            )
            for pname, prows in partitions.items()
        },
        "H256_ACTIVE_2PLUS2_UNIFIED": {
            pname: compute_detailed_decision_and_ranking_kpis(
                prows,
                packets,
                pools,
                comb_active_u,
                comb_active_inv,
                comb_active_pidx,
                comb_static_sc8,
                comb_static_pinv8,
                r882_u_by_cid,
            )
            for pname, prows in partitions.items()
        },
    }

    # 4. Feature & Block Attribution KPIs on the Full-1,310 Production Bundle
    prod_1219_path = CKPT_DIR / "h256_production_1219_dl_train_bundle.pt"
    prod_1310_path = CKPT_DIR / "h256_production_full_1310_regression_bundle.pt"
    prod_1219_payload = torch.load(prod_1219_path, map_location="cpu", weights_only=False)
    prod_1310_payload = torch.load(prod_1310_path, map_location="cpu", weights_only=False)

    feature_attribution_kpis = compute_feature_block_and_neural_attribution_kpis(
        prod_1310_payload, packets, all_cids, device
    )

    # 5. Stage-2 90-Checkpoint Ablation Telemetry Summary
    with (OUT_DIR / "summary.json").open() as f:
        summary_doc = json.load(f)

    master_kpi_telemetry = {
        "schema": "jope.h256.training_and_validation_kpi_telemetry/v1",
        "generated_at_epoch_s": time.time(),
        "model_task_scope": "regression_only (0 binary/multiclass)",
        "dope_sha256": ch.DOPE_SHA256,
        "r882_sha256": ch.R882_SHA256,
        "stage1_s3_7split_pretraining_kpis": stage1_kpis,
        "stage1_and_stage2_critic_kpis": critic_kpis,
        "stage2_top8_15cell_train_inner_and_outer_telemetry": {
            "mean_infold_fit_family_gain_vs_r882": float(np.mean(infold_fit_gains)),
            "mean_inner_oof_fit_family_gain_vs_r882": float(np.mean(inner_oof_fit_gains)),
            "mean_outer_oof_assess_family_gain_vs_r882": float(
                flagship_detailed_kpis["H256_STATIC_0REV_TOP8_SOFT_VETO"]["dl_train_oof_1219"][
                    "mean_family_gain_vs_r882"
                ]
            ),
            "per_cell_telemetry": fold_cell_telemetry,
        },
        "flagship_models_detailed_partition_kpis": flagship_detailed_kpis,
        "stage2_6arm_ablation_partition_kpis": summary_doc.get("arms", {}),
        "feature_block_and_pre_condense_attribution_kpis": feature_attribution_kpis,
        "head_to_head_vs_best_waves_so_far": summary_doc.get("flagship_unified_models", {}).get(
            "head_to_head_vs_best_waves_so_far", {}
        ),
    }

    # Per-context and per-family KPI ledger across all 1,310 regression contexts
    per_context_ledger = []
    for r in all_rows:
        c = r["context_sha256"]
        pkt = packets[c]
        per_context_ledger.append(
            {
                "context_sha256": c,
                "family_id": r["family_id"],
                "cohort": "dl_train_1219" if c not in set(val_cids) else "dl_val_91",
                "top_zgap": float(pkt["top_zgap"]),
                "r882_utility": float(r882_u_by_cid[c]),
                "r882_invalid": bool(pkt["r882_invalid"]),
                "top8_oracle_utility": float(pkt["top8_oracle_u"]),
                "static_0rev_mean_utility": float(
                    np.mean([comb_static_u[s][c] for s in flg.SEEDS])
                ),
                "static_0rev_gain_vs_r882": float(
                    np.mean([comb_static_u[s][c] for s in flg.SEEDS]) - r882_u_by_cid[c]
                ),
                "static_0rev_any_invalid": bool(
                    any(comb_static_inv[s][c] for s in flg.SEEDS)
                ),
                "active_2plus2_mean_utility": float(
                    np.mean([comb_active_u[s][c] for s in flg.SEEDS])
                ),
                "active_2plus2_gain_vs_r882": float(
                    np.mean([comb_active_u[s][c] for s in flg.SEEDS]) - r882_u_by_cid[c]
                ),
                "active_2plus2_any_invalid": bool(
                    any(comb_active_inv[s][c] for s in flg.SEEDS)
                ),
            }
        )

    kpi_json_path = OUT_DIR / "training_and_validation_kpi_telemetry.json"
    with kpi_json_path.open("w") as f:
        json.dump(master_kpi_telemetry, f, indent=2)

    repo_kpi_json_path = HERE / "kpi_telemetry.json"
    with repo_kpi_json_path.open("w") as f:
        json.dump(master_kpi_telemetry, f, indent=2)

    ledger_path = OUT_DIR / "per_context_and_family_kpi_ledger.json"
    with ledger_path.open("w") as f:
        json.dump(
            {
                "schema": "jope.h256.per_context_and_family_kpi_ledger/v1",
                "n_contexts": len(per_context_ledger),
                "contexts": per_context_ledger,
            },
            f,
            indent=2,
        )

    # Embed `kpi_telemetry` inside the top-level & production `.pt` checkpoints
    s3_11121_path = OUT_DIR / "full_s3_reg_7split_pretrained_encoder.pt"
    s3_11121_payload = torch.load(s3_11121_path, map_location="cpu", weights_only=False)
    s3_11121_payload["kpi_telemetry"] = stage1_kpis
    torch.save(s3_11121_payload, s3_11121_path)

    s3_15796_path = CKPT_DIR / "stage1_full_s3_reg_all15796_encoder.pt"
    s3_15796_payload = torch.load(s3_15796_path, map_location="cpu", weights_only=False)
    s3_15796_payload["kpi_telemetry"] = stage1_kpis
    torch.save(s3_15796_payload, s3_15796_path)

    critic_pt_path = CKPT_DIR / "sos_hurdle_and_sp_critic_models.pt"
    if critic_pt_path.exists():
        critic_payload = torch.load(critic_pt_path, map_location="cpu", weights_only=False)
        critic_payload["kpi_telemetry"] = critic_kpis
        torch.save(critic_payload, critic_pt_path)

    prod_1219_payload["kpi_telemetry"] = master_kpi_telemetry
    torch.save(prod_1219_payload, prod_1219_path)

    prod_1310_payload["kpi_telemetry"] = master_kpi_telemetry
    torch.save(prod_1310_payload, prod_1310_path)

    fused_path = OUT_DIR / "h256_fused_problem_and_solution_network.pt"
    fused_payload = torch.load(fused_path, map_location="cpu", weights_only=False)
    fused_payload["kpi_telemetry"] = master_kpi_telemetry
    torch.save(fused_payload, fused_path)

    summary_doc["kpi_telemetry"] = master_kpi_telemetry
    with (OUT_DIR / "summary.json").open("w") as f:
        json.dump(summary_doc, f, indent=2)

    # Refresh checkpoint_manifest.json with final SHA-256 hashes of all updated files
    manifest_path = CKPT_DIR / "checkpoint_manifest.json"
    with manifest_path.open() as f:
        manifest_doc = json.load(f)

    top_canon = manifest_doc.setdefault("top_level_canonical_checkpoints", {})
    top_canon.setdefault("full_s3_reg_7split_pretrained_encoder_pt", {})["sha256"] = sha256_file(
        s3_11121_path
    )
    top_canon.setdefault("h256_fused_problem_and_solution_network_pt", {})["sha256"] = sha256_file(
        fused_path
    )
    if "critic_and_golden_fixture_artifacts" in manifest_doc:
        manifest_doc["critic_and_golden_fixture_artifacts"].setdefault(
            "sos_hurdle_and_sp_critic_models_pt", {}
        )["sha256"] = sha256_file(critic_pt_path)
    manifest_doc["flagship_production_checkpoints"] = {
        "stage1_full_s3_reg_all15796_encoder_pt": {
            "path": str(s3_15796_path),
            "sha256": sha256_file(s3_15796_path),
            "description": "Stage-1 Pre-Condense 92-token encoder trained on ALL 15,796 S3 7-split regression cells (0 binary/multiclass) + embedded KPI telemetry.",
        },
        "h256_production_1219_dl_train_bundle_pt": {
            "path": str(prod_1219_path),
            "sha256": sha256_file(prod_1219_path),
            "description": "Holdout-clean production bundle trained on all 1,219 DL-Train regression contexts + 11,121 S3 Policy-B regression cells + embedded KPI telemetry.",
        },
        "h256_production_full_1310_regression_bundle_pt": {
            "path": str(prod_1310_path),
            "sha256": sha256_file(prod_1310_path),
            "description": "Full-dataset production bundle trained on ALL 1,310 regression flywheel contexts (1,219 DL-Train + 91 DL-Val) + ALL 15,796 S3 7-split regression cells + embedded KPI telemetry.",
        },
        "training_and_validation_kpi_telemetry_json": {
            "path": str(kpi_json_path),
            "sha256": sha256_file(kpi_json_path),
        },
        "per_context_and_family_kpi_ledger_json": {
            "path": str(ledger_path),
            "sha256": sha256_file(ledger_path),
        },
        "top8_neural_reranker_fold_checkpoints": [
            {
                "seed": int(s),
                "fold": int(k),
                "path": str(CKPT_DIR / f"stage2_top8_neural_reranker_seed{s}_fold{k}.pt"),
                "sha256": sha256_file(CKPT_DIR / f"stage2_top8_neural_reranker_seed{s}_fold{k}.pt"),
            }
            for s in flg.SEEDS
            for k in range(5)
        ],
    }
    with manifest_path.open("w") as f:
        json.dump(manifest_doc, f, indent=2)

    guard.verify_dope_frozen_unchanged(net, dope_snap_before)
    print(
        f"[KPI Telemetry Suite Completed in {time.time() - t0:.1f}s]\n"
        f"  - Saved {kpi_json_path} ({sha256_file(kpi_json_path)})\n"
        f"  - Saved {ledger_path} ({sha256_file(ledger_path)})\n"
        f"  - Saved {repo_kpi_json_path} ({sha256_file(repo_kpi_json_path)})\n"
        f"  - Embedded kpi_telemetry into all Stage-1, Stage-2, Critic, and Production .pt checkpoints.",
        flush=True,
    )


if __name__ == "__main__":
    main()
