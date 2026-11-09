#!/usr/bin/env python3
"""H256 Complete Inference Recipe Exporter:
1. Exports the fitted SOS/hurdle and Self-Play (SP) critic models & scalers:
   - `/home/ubuntu/jope-data/h256/checkpoints/sos_hurdle_and_sp_critic_models.pt`
   - `/home/ubuntu/jope-data/h256/checkpoints/sos_hurdle_and_sp_critic_models.npz`
2. Exports complete end-to-end golden input + expected prediction fixtures on unseen `DL-Val`:
   - `/home/ubuntu/jope-data/h256/checkpoints/golden_inference_fixture_context0.pt`
   - `/home/ubuntu/jope-data/h256/checkpoints/golden_inference_fixture_context0.json`
   (Includes both `val_rows[0]` and the first active `top_zgap < 0.25` unseen `DL-Val` context where
   both Stage-1 `delta_prior` and Stage-2 `delta_traj` are actively non-zero, evaluated under both
   the 5-fold `Arm G` ensemble and the single parameter-soup / `fold0` model.)
3. Embeds `sos_hurdle_and_sp_critic_models` and `golden_inference_fixture_summary` directly into:
   - `/home/ubuntu/jope-data/h256/h256_fused_problem_and_solution_network.pt`
   - `/home/ubuntu/jope-data/h256/stage2_full_s3_reg_flywheel_pretrained_bridge.pt`
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
H252_DIR = HERE.parent / "h252_closed_loop_flywheel"
H255_DIR = HERE.parent / "h255_causal_2plus2_live_gp"
OUT_DIR = Path("/home/ubuntu/jope-data/h256")
CKPT_DIR = OUT_DIR / "checkpoints"

sys.path.insert(0, str(H252_DIR))
sys.path.insert(0, str(H255_DIR))
sys.path.insert(0, str(HERE))

import common_h252 as ch  # noqa: E402
import stage1_hurdle_sos_critic as s1  # noqa: E402
import part_a_causal_2plus2_neural as p255  # noqa: E402
import run_h256_flywheel_ablation as r256  # noqa: E402
from pre_condense_models import FoldScaler, PreCondenseResidualBridge  # noqa: E402


def fit_and_extract_critic_weights(
    train_rows: List[Dict[str, Any]],
    pools: Dict[str, Dict[str, Any]],
    cand_feats: Dict[str, np.ndarray],
    promos_by_cid: Dict[str, List[Dict[str, Any]]],
    seed: int = r256.FOLD_SEED,
) -> Dict[str, Any]:
    """Exact replica of `s1.fit_decomposed_hurdle_and_sp_priors` returning all fitted weight vectors."""
    fams = sorted({r["family_id"] for r in train_rows})
    if len(fams) == 205:
        _, cid_to_fold = ch.seed_folds(train_rows, seed)
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(fams))
        fam_to_fold = {fams[idx]: int(i % 5) for i, idx in enumerate(perm)}
        cid_to_fold = {r["context_sha256"]: fam_to_fold[r["family_id"]] for r in train_rows}
    dim = 65  # 64 candidate features + bias 1.0

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

        g_fail = (~p["gate"]).astype(np.float64)
        gf_bar = float(np.clip(np.mean(g_fail), 0.02, 0.98))
        y_g = (g_fail - gf_bar) / np.sqrt(gf_bar * (1.0 - gf_bar))
        xtx_gate[f] += X.T @ X
        xty_gate[f] += X.T @ y_g

        if cid in promos_by_cid:
            pool_sem_mean = p["sem24"].astype(np.float64).mean(axis=0)
            for pr in promos_by_cid[cid]:
                sp_sem_by_fold[f] += pr["sem24"].astype(np.float64) - pool_sem_mean
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

    w_c_oof = np.zeros((5, dim), dtype=np.float64)
    w_g_oof = np.zeros((5, dim), dtype=np.float64)
    sp_dir_oof = np.zeros((5, 24), dtype=np.float64)
    for f in range(5):
        w_c_oof[f] = np.linalg.solve((xtx_c_all - xtx_cont[f]) + reg_c * eye, xty_c_all - xty_cont[f])
        w_g_oof[f] = np.linalg.solve((xtx_g_all - xtx_gate[f]) + reg_g * eye, xty_g_all - xty_gate[f])
        v_sp = sp_sem_all - sp_sem_by_fold[f]
        sp_dir_oof[f] = v_sp / max(float(np.linalg.norm(v_sp)), 1e-6)

    w_c_full = np.linalg.solve(xtx_c_all + reg_c * eye, xty_c_all)
    w_g_full = np.linalg.solve(xtx_g_all + reg_g * eye, xty_g_all)
    sp_dir_full = sp_sem_all / max(float(np.linalg.norm(sp_sem_all)), 1e-6)

    return {
        "w_c_full": w_c_full,
        "w_g_full": w_g_full,
        "sp_dir_full": sp_dir_full,
        "w_c_oof_5fold": w_c_oof,
        "w_g_oof_5fold": w_g_oof,
        "sp_dir_oof_5fold": sp_dir_oof,
        "reg_c": reg_c,
        "reg_g": reg_g,
        "fold_seed": seed,
        "total_promotions_used": int(sum(sp_cnt_by_fold.values())),
    }


def build_golden_context_fixture(
    row: Dict[str, Any],
    raw_ctx_entry: Dict[str, np.ndarray],
    pool: Dict[str, Any],
    r882_entry: Dict[str, np.ndarray],
    cand_feat_64: np.ndarray,
    sos_prior: np.ndarray,
    sp_prior: np.ndarray,
    st_entry: Dict[str, np.ndarray],
    bt: Dict[str, Any],
    scaler_stage2: FoldScaler,
    fold_models: List[PreCondenseResidualBridge],
    soup_model: PreCondenseResidualBridge,
    ap_ens: float,
    at_ens: float,
    idx4: np.ndarray,
    device: torch.device,
) -> Dict[str, Any]:
    cid = row["context_sha256"]
    ct_val = scaler_stage2.transform_context(raw_ctx_entry, device)

    with torch.no_grad():
        # 1. Single-model (parameter-soup) intermediate tensors
        ctx_tok_soup, ctx_mask_soup, scale8_soup = soup_model.encode_context_tokens(ct_val)
        dp_soup_t, attn_soup_t, bilin_soup_t = soup_model.forward_prior_and_context(
            ct_val,
            bt["cand_q_t"],
            bt["sem24_t"],
            bt["gate_risk_t"],
            bt["sp_prior_t"],
        )
        dp_soup_np = dp_soup_t[0].cpu().numpy().astype(np.float64)
        rev4_soup, cost_act_soup, y4_comb_soup, y4_f_soup = r256.run_2plus2_active_reveal_step(
            bt, idx4, dp_soup_np, ap_ens
        )
        base_plus_prior_soup = torch.as_tensor(
            cost_act_soup + ap_ens * dp_soup_np, dtype=torch.float32, device=device
        ).unsqueeze(0)
        dt_soup_t = soup_model.forward_causal_trajectory_residual(
            attn_soup_t,
            bilin_soup_t,
            bt["cand_q_t"],
            bt["sem24_t"],
            base_plus_prior_soup,
            torch.as_tensor(rev4_soup, dtype=torch.long, device=device).unsqueeze(0),
            torch.as_tensor(y4_comb_soup, dtype=torch.float32, device=device).unsqueeze(0),
            torch.as_tensor(y4_f_soup, dtype=torch.float32, device=device).unsqueeze(0),
            torch.as_tensor(bt["gate"][rev4_soup], dtype=torch.float32, device=device).unsqueeze(0),
        )
        dt_soup_np = dt_soup_t[0].cpu().numpy().astype(np.float64)
        cost_final_soup = (
            cost_act_soup
            + ap_ens * dp_soup_np
            + at_ens * dt_soup_np
            + 0.005 * bt["h_cost"]
        )
        order_soup = np.argsort(cost_final_soup)

        # 2. 5-Fold ensemble intermediate & final tensors
        dp_folds = []
        attn_folds = []
        bilin_folds = []
        for m in fold_models:
            dp_k, attn_k, bilin_k = m.forward_prior_and_context(
                ct_val,
                bt["cand_q_t"],
                bt["sem24_t"],
                bt["gate_risk_t"],
                bt["sp_prior_t"],
            )
            dp_folds.append(dp_k[0].cpu().numpy().astype(np.float64))
            attn_folds.append(attn_k)
            bilin_folds.append(bilin_k)
        dp_ens_np = np.mean(dp_folds, axis=0)
        rev4_ens, cost_act_ens, y4_comb_ens, y4_f_ens = r256.run_2plus2_active_reveal_step(
            bt, idx4, dp_ens_np, ap_ens
        )
        base_plus_prior_ens = torch.as_tensor(
            cost_act_ens + ap_ens * dp_ens_np, dtype=torch.float32, device=device
        ).unsqueeze(0)
        rev_idx_t = torch.as_tensor(rev4_ens, dtype=torch.long, device=device).unsqueeze(0)
        rev_y_comb_t = torch.as_tensor(y4_comb_ens, dtype=torch.float32, device=device).unsqueeze(0)
        rev_y_f_t = torch.as_tensor(y4_f_ens, dtype=torch.float32, device=device).unsqueeze(0)
        rev_g_t = torch.as_tensor(bt["gate"][rev4_ens], dtype=torch.float32, device=device).unsqueeze(0)
        dt_folds = []
        for m, attn_k, bilin_k in zip(fold_models, attn_folds, bilin_folds):
            dt_k = m.forward_causal_trajectory_residual(
                attn_k,
                bilin_k,
                bt["cand_q_t"],
                bt["sem24_t"],
                base_plus_prior_ens,
                rev_idx_t,
                rev_y_comb_t,
                rev_y_f_t,
                rev_g_t,
            )
            dt_folds.append(dt_k[0].cpu().numpy().astype(np.float64))
        dt_ens_np = np.mean(dt_folds, axis=0)
        cost_final_ens = (
            cost_act_ens
            + ap_ens * dp_ens_np
            + at_ens * dt_ens_np
            + 0.005 * bt["h_cost"]
        )
        order_ens = np.argsort(cost_final_ens)

    res_eval = r256.evaluate_context_with_model(
        bt, ct_val, fold_models, ap_ens, at_ens, idx4, device
    )

    return {
        "context_metadata": {
            "context_sha256": cid,
            "family_id": row["family_id"],
            "dataset_id": row.get("dataset_id", ""),
            "split": row.get("split", ""),
            "num_candidates": int(len(bt["rcost"])),
            "top_zgap": float(bt["top_zgap"]),
            "active_flywheel_triggered": bool(bt["top_zgap"] < 0.25),
            "r882_pick_index": int(bt["pick_r882"]),
            "r882_utility": float(bt["u_r882"]),
            "oracle_max_utility": float(bt["u_max"]),
        },
        "inputs": {
            "raw_problem_and_landmark_features": {
                "vec256": raw_ctx_entry["dope629"][:256].copy(),
                "dope629": raw_ctx_entry["dope629"].copy(),
                "hidden256": raw_ctx_entry["hidden256"].copy(),
                "col_lm_64x17": raw_ctx_entry["col_lm_64x17"].copy(),
                "col_mask_64": raw_ctx_entry["col_mask_64"].copy(),
                "lm48_z": raw_ctx_entry["lm48_z"].copy(),
                "tokens_10x32": raw_ctx_entry["tokens_10x32"].copy(),
                "mask_10": raw_ctx_entry["mask_10"].copy(),
                "ctx_feat_584": raw_ctx_entry["ctx_feat_584"].copy(),
            },
            "standardized_context_tensors": {
                k: v.detach().cpu().numpy() for k, v in ct_val.items()
            },
            "candidate_pool_features": {
                "program_412": pool["program"].copy(),
                "proj16": pool["proj16"].copy(),
                "sem24": st_entry["sem24"].copy(),
                "cand_q_32": st_entry["cand_q"].copy(),
                "eff_prior": st_entry["eff_prior"].copy(),
                "r882_total_cost": r882_entry["total_cost"].astype(np.float64).copy(),
                "r882_anchor_prior": r882_entry["anchor_prior"].astype(np.float64).copy(),
                "r882_anchor_total": r882_entry["anchor_total"].astype(np.float64).copy(),
                "cand_feats_64": cand_feat_64.copy(),
                "sos_prior_2d": sos_prior.copy(),
                "sp_prior_1d": sp_prior.copy(),
                "h_cost": bt["h_cost"].copy(),
                "initial_probe_idx4": idx4.copy(),
                "candidate_fitness": bt["fitness"].copy(),
                "candidate_logratio": bt["logratio"].copy(),
                "candidate_gate": bt["gate"].copy(),
                "candidate_utility": bt["u"].copy(),
            },
        },
        "expected_outputs_single_soup_model": {
            "alpha_prior": float(ap_ens),
            "alpha_traj": float(at_ens),
            "ctx_tokens_92x32": ctx_tok_soup.detach().cpu().numpy(),
            "ctx_mask_92": ctx_mask_soup.detach().cpu().numpy(),
            "scale_feat_8": scale8_soup.detach().cpu().numpy(),
            "attn_ctx_32": attn_soup_t.detach().cpu().numpy(),
            "bilinear_24": bilin_soup_t.detach().cpu().numpy(),
            "delta_prior_vec": dp_soup_np,
            "rev4_indices": rev4_soup,
            "cost_act": cost_act_soup,
            "delta_traj_vec": dt_soup_np,
            "cost_final": cost_final_soup,
            "pick1_index": int(order_soup[0]),
            "top4_indices": order_soup[:4].astype(np.int64),
            "pick1_utility": float(bt["u"][int(order_soup[0])]),
        },
        "expected_outputs_5fold_ensemble": {
            "alpha_prior": float(ap_ens),
            "alpha_traj": float(at_ens),
            "delta_prior_vec": dp_ens_np,
            "rev4_indices": rev4_ens,
            "cost_act": cost_act_ens,
            "delta_traj_vec": dt_ens_np,
            "cost_final": cost_final_ens,
            "pick1_index": int(order_ens[0]),
            "top4_indices": order_ens[:4].astype(np.int64),
            "pick1_utility": float(bt["u"][int(order_ens[0])]),
            "evaluate_context_with_model_return": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in res_eval.items()
            },
        },
    }


def main() -> None:
    t0 = time.time()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    torch.set_num_threads(4)

    # 1. Load locked FusedJopeNetwork, cohorts, pools, R882 cache, and pre-condense cache
    net, guard, dope_snap, w1_mat = p255.load_fused_network(device)
    train_rows, val_rows = ch.load_cohorts()
    all_rows = train_rows + val_rows
    pools = ch.load_pools(all_rows, verify_hashes=False)
    r882_cache = p255.load_r882_cache_readonly(all_rows)
    raw_ctx = r256.load_pre_condense_cache(r256.CACHE_PATH)
    promos = ch.load_harvested_promotions(train_rows)

    # 2. Build 64D candidate features and fit the exact SOS/hurdle and SP critic weights
    cand_feats = s1.build_candidate_features(all_rows, pools, r882_cache, raw_ctx)
    sos_priors, sp_priors, critic_diag = s1.fit_decomposed_hurdle_and_sp_priors(
        train_rows, val_rows, pools, cand_feats, promos, seed=r256.FOLD_SEED
    )
    critic_weights = fit_and_extract_critic_weights(
        train_rows, pools, cand_feats, promos, seed=r256.FOLD_SEED
    )

    # Verify exact match between `critic_weights` and `s1.fit_decomposed_hurdle_and_sp_priors` on `val_rows[0]`
    c0 = val_rows[0]["context_sha256"]
    X0 = np.concatenate([cand_feats[c0].astype(np.float64), np.ones((len(cand_feats[c0]), 1))], axis=1)
    pc0 = X0 @ critic_weights["w_c_full"]
    pg0 = X0 @ critic_weights["w_g_full"]
    pc0_z = ((pc0 - np.mean(pc0)) / (np.std(pc0) + 1e-6)).astype(np.float32)
    pg0_z = ((pg0 - np.mean(pg0)) / (np.std(pg0) + 1e-6)).astype(np.float32)
    sos0_check = np.stack([pc0_z, pg0_z], axis=1)
    max_sos_diff = float(np.max(np.abs(sos0_check - sos_priors[c0])))
    if max_sos_diff != 0.0:
        raise RuntimeError(f"Critic weight verification mismatch: {max_sos_diff}")

    w1_proj_412x16 = (w1_mat.astype(np.float32) / np.sqrt(412.0)).astype(np.float32)
    critic_bundle = {
        "schema": "jope.h256.sos_hurdle_and_sp_critic_models/v1",
        "dope_sha256": dope_snap.composite_sha256,
        "w_c_full": critic_weights["w_c_full"],
        "w_g_full": critic_weights["w_g_full"],
        "sp_dir_full": critic_weights["sp_dir_full"],
        "w_c_oof_5fold": critic_weights["w_c_oof_5fold"],
        "w_g_oof_5fold": critic_weights["w_g_oof_5fold"],
        "sp_dir_oof_5fold": critic_weights["sp_dir_oof_5fold"],
        "reg_c": critic_weights["reg_c"],
        "reg_g": critic_weights["reg_g"],
        "fold_seed": critic_weights["fold_seed"],
        "total_promotions_used": critic_weights["total_promotions_used"],
        "w1_proj_412x16": w1_proj_412x16,
        "critic_diagnostics": critic_diag,
        "feature_builder_spec": {
            "total_dim_with_bias": 65,
            "feature_slices": {
                "0:10": "head10 = [r_z, r_gap1, r_gap8, ap_z, ad_z, ranks, cos_top1, cos_top8, max_cos_rev4, mean_cos_rev4]",
                "10:34": "sem_z (24D within-pool standardized sem24 clipped to [-5, 5])",
                "34:54": "st_z (20D within-pool standardized program[:, :12] ++ program[:, 48:56] clipped to [-5, 5])",
                "54:59": "inter_r (5D r_z[:, None] * lm48_z[:5][None, :])",
                "59:64": "inter_ap (5D ap_z[:, None] * lm48_z[:5][None, :])",
                "64": "bias = 1.0",
            },
            "inference_formulas": {
                "cont_cost_z": "(X_65 @ w_c_full - mean) / (std + 1e-6)",
                "gate_risk": "(X_65 @ w_g_full - mean) / (std + 1e-6)",
                "sp_prior": "(-(sem24 @ sp_dir_full) - mean) / (std + 1e-6)",
                "gate_safety_penalty": "where(gate_risk > gate_risk[argmin(rcost)] + 0.10, 10.0, 0.0)",
                "h_cost": "0.65 * cont_cost_z + 0.35 * gate_risk + 0.15 * sp_prior + gate_safety_penalty",
            },
        },
    }

    critic_pt_path = CKPT_DIR / "sos_hurdle_and_sp_critic_models.pt"
    critic_npz_path = CKPT_DIR / "sos_hurdle_and_sp_critic_models.npz"
    torch.save(critic_bundle, critic_pt_path)
    np.savez_compressed(
        critic_npz_path,
        w_c_full=critic_weights["w_c_full"],
        w_g_full=critic_weights["w_g_full"],
        sp_dir_full=critic_weights["sp_dir_full"],
        w_c_oof_5fold=critic_weights["w_c_oof_5fold"],
        w_g_oof_5fold=critic_weights["w_g_oof_5fold"],
        sp_dir_oof_5fold=critic_weights["sp_dir_oof_5fold"],
        w1_proj_412x16=w1_proj_412x16,
    )

    # 3. Load trained Stage-2 `Arm G` checkpoint and build Golden Inference Fixtures on unseen `DL-Val`:
    #    - `val_rows[0]` (index 0 of unseen `DL-Val`)
    #    - First active unseen `DL-Val` context where `top_zgap < 0.25` (so both Stage-1 `delta_prior`
    #      and Stage-2 `delta_traj` are actively non-zero)
    stage2_ckpt_path = OUT_DIR / "stage2_full_s3_reg_flywheel_pretrained_bridge.pt"
    stage2_ckpt = torch.load(stage2_ckpt_path, map_location="cpu", weights_only=False)
    arm_name = stage2_ckpt["arm"]
    ap_ens = float(stage2_ckpt["ensemble_alpha_prior"])
    at_ens = float(stage2_ckpt["ensemble_alpha_traj"])
    sc_dict = stage2_ckpt["full_train_scaler"]
    scaler_stage2 = FoldScaler(
        dope629_mean=sc_dict["dope629_mean"],
        dope629_std=sc_dict["dope629_std"],
        hidden256_mean=sc_dict["hidden256_mean"],
        hidden256_std=sc_dict["hidden256_std"],
        col18_mean=sc_dict["col18_mean"],
        col18_std=sc_dict["col18_std"],
    )

    soup_model = PreCondenseResidualBridge(arm=arm_name, d_token=32, n_heads=4).to(device)
    soup_model.load_state_dict(stage2_ckpt["state_dict"], strict=True)
    soup_model.eval()

    fold_models: List[PreCondenseResidualBridge] = []
    for f_sd in stage2_ckpt["fold_state_dicts"]:
        m_f = PreCondenseResidualBridge(arm=arm_name, d_token=32, n_heads=4).to(device)
        m_f.load_state_dict(f_sd, strict=True)
        m_f.eval()
        fold_models.append(m_f)

    # Find first active unseen DL-Val context (`top_zgap < 0.25`)
    active_val_row = val_rows[0]
    for r in val_rows:
        cid_r = r["context_sha256"]
        if s1.compute_r882_top_zgap(r882_cache[cid_r]["total_cost"]) < 0.25:
            active_val_row = r
            break

    golden_rows = [active_val_row]
    if val_rows[0]["context_sha256"] != active_val_row["context_sha256"]:
        golden_rows.append(val_rows[0])

    static_sub = p255.precompute_static_adapter_terms(
        net, golden_rows, pools, raw_ctx, w1_mat, device
    )

    fixtures_by_role: Dict[str, Any] = {}
    for role_idx, r in enumerate(golden_rows):
        cid = r["context_sha256"]
        bt = r256.compute_context_base_terms(
            cid,
            pools[cid],
            r882_cache[cid]["total_cost"],
            sos_priors[cid],
            sp_priors[cid],
            static_sub[cid],
            device,
        )
        rng_c = np.random.default_rng(ch.stream_seed("h256_idx4", r256.SEEDS[0], cid))
        idx4 = rng_c.choice(len(bt["rcost"]), size=4, replace=False)
        fx = build_golden_context_fixture(
            r,
            raw_ctx[cid],
            pools[cid],
            r882_cache[cid],
            cand_feats[cid],
            sos_priors[cid],
            sp_priors[cid],
            static_sub[cid],
            bt,
            scaler_stage2,
            fold_models,
            soup_model,
            ap_ens,
            at_ens,
            idx4,
            device,
        )
        key = "primary_active_dl_val_context" if role_idx == 0 else "dl_val_row0_context"
        fixtures_by_role[key] = fx

    primary_fx = fixtures_by_role["primary_active_dl_val_context"]
    golden_payload = {
        "schema": "jope.h256.golden_inference_fixture/v1",
        "arm": arm_name,
        "seed": int(r256.SEEDS[0]),
        "dope_sha256": dope_snap.composite_sha256,
        # Top-level convenience keys for primary active DL-Val context
        **primary_fx,
        "fixtures": fixtures_by_role,
    }

    golden_pt_path = CKPT_DIR / "golden_inference_fixture_context0.pt"
    golden_json_path = CKPT_DIR / "golden_inference_fixture_context0.json"
    torch.save(golden_payload, golden_pt_path)

    golden_summary = {
        "schema": "jope.h256.golden_inference_fixture_summary/v1",
        "golden_pt_path": str(golden_pt_path),
        "golden_pt_sha256": r256.sha256_file(golden_pt_path),
        "critic_pt_path": str(critic_pt_path),
        "critic_pt_sha256": r256.sha256_file(critic_pt_path),
        "critic_npz_path": str(critic_npz_path),
        "critic_npz_sha256": r256.sha256_file(critic_npz_path),
        "primary_context_metadata": primary_fx["context_metadata"],
        "expected_single_soup_summary": {
            "alpha_prior": primary_fx["expected_outputs_single_soup_model"]["alpha_prior"],
            "alpha_traj": primary_fx["expected_outputs_single_soup_model"]["alpha_traj"],
            "delta_prior_first5": primary_fx["expected_outputs_single_soup_model"]["delta_prior_vec"][:5].tolist(),
            "delta_traj_first5": primary_fx["expected_outputs_single_soup_model"]["delta_traj_vec"][:5].tolist(),
            "rev4_indices": primary_fx["expected_outputs_single_soup_model"]["rev4_indices"].tolist(),
            "pick1_index": primary_fx["expected_outputs_single_soup_model"]["pick1_index"],
            "top4_indices": primary_fx["expected_outputs_single_soup_model"]["top4_indices"].tolist(),
            "pick1_utility": primary_fx["expected_outputs_single_soup_model"]["pick1_utility"],
        },
        "expected_5fold_ensemble_summary": {
            "alpha_prior": primary_fx["expected_outputs_5fold_ensemble"]["alpha_prior"],
            "alpha_traj": primary_fx["expected_outputs_5fold_ensemble"]["alpha_traj"],
            "delta_prior_first5": primary_fx["expected_outputs_5fold_ensemble"]["delta_prior_vec"][:5].tolist(),
            "delta_traj_first5": primary_fx["expected_outputs_5fold_ensemble"]["delta_traj_vec"][:5].tolist(),
            "rev4_indices": primary_fx["expected_outputs_5fold_ensemble"]["rev4_indices"].tolist(),
            "pick1_index": primary_fx["expected_outputs_5fold_ensemble"]["pick1_index"],
            "top4_indices": primary_fx["expected_outputs_5fold_ensemble"]["top4_indices"].tolist(),
            "pick1_utility": primary_fx["expected_outputs_5fold_ensemble"]["pick1_utility"],
        },
    }
    ch.save_json(golden_json_path, golden_summary)

    # 4. Also embed `sos_hurdle_and_sp_critic_models` and `golden_inference_fixture_summary` into
    #    the top-level Stage-2 and Unified checkpoints
    stage2_ckpt["sos_hurdle_and_sp_critic_models"] = critic_bundle
    stage2_ckpt["golden_inference_fixture_summary"] = golden_summary
    torch.save(stage2_ckpt, stage2_ckpt_path)

    unified_path = OUT_DIR / "h256_fused_problem_and_solution_network.pt"
    if unified_path.exists():
        uni_ckpt = torch.load(unified_path, map_location="cpu", weights_only=False)
        uni_ckpt["sos_hurdle_and_sp_critic_models"] = critic_bundle
        uni_ckpt["golden_inference_fixture_summary"] = golden_summary
        torch.save(uni_ckpt, unified_path)

    print(
        f"[H256 Complete Inference Bundle Exported in {time.time() - t0:.2f}s]\n"
        f"  Critic Models PT:  {critic_pt_path} (SHA-256: {golden_summary['critic_pt_sha256']})\n"
        f"  Critic Models NPZ: {critic_npz_path} (SHA-256: {golden_summary['critic_npz_sha256']})\n"
        f"  Golden Fixture PT: {golden_pt_path} (SHA-256: {golden_summary['golden_pt_sha256']})\n"
        f"  Primary Active DL-Val Context: {primary_fx['context_metadata']['context_sha256']} "
        f"(top_zgap={primary_fx['context_metadata']['top_zgap']:.4f}, "
        f"pick1_ens={golden_summary['expected_5fold_ensemble_summary']['pick1_index']}, "
        f"U_ens={golden_summary['expected_5fold_ensemble_summary']['pick1_utility']:.6f})"
    )


if __name__ == "__main__":
    main()
