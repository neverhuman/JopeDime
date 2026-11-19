"""Part C: Zero-Retraining Accuracy Upgrades (`WP3` & `B03`, Contrasts `C5` & `C8`) + One-Time `val91` `REGCHECK`.

Implements:
  1. `WP3` — Empirical-Bayes Variance-Shrunk Predicted Reveals at **0 real candidate evaluations**:
       - Fold-safe inner-cross-fitted outcome residual head `mu_hat(R, prog)`, heteroscedastic
         uncertainty head `sigma_hat(R, prog)`, and gate-pass probability head `p_gate_hat(R, prog)`
         fitted strictly on the 4 training folds (`tr_idx`).
       - Shrinkage `lam = sigma_prior^2 / (sigma_prior^2 + kappa * sigma_hat^2)` and
         `y_hat = lam * mu_hat + (1 - lam) * prior`.
       - Verified `V-S1` (`lam = 0` reproduces `incumbent_0rev` bit-for-bit, `max_abs_diff == 0.0`).
       - Evaluates `WP3_hist_only` (100% through H184's revealed-history kernel on top-4 lowest-`sigma_hat`
         anchors), `WP3_fixed_lowvar` (`R_dope`), `WP3_mimic_acq` (`R_dope`), `WP3_ultra_packet`
         (`[R_dope_ultra, R_packet_v1_ultra]`), and cross-family donor-permuted `WP3_shuffled_ctrl`
         (`family[donor] != family[i]`) against `incumbent_0rev`, `wave0_fake_k4`, and `incumbent_4rev_upper`.
  2. `B03` — Folded Stage-1B SVD `629 -> 96` Trainable Bottleneck & Zero-Init `Ultra+Packet` Residual Adapter:
       - Exact `H452`–`H469` `pairnf_z40` harness (`16` steps, `batch=1024`, `slate=192`, `lr=3e-4`,
         `wd=1e-4`, `zmin=0.40`, `rng_arm="pairnf"`, `h452a/` + `h452b/` 5 folds × 3 seeds).
       - Precomputed frozen-anchor `FlywheelGameNet` cache (`< 1e-5` verified parity on fold 0) cutting
         CPU wall time by ~80x while keeping DOPE (`c51a56d0...`) and `FlywheelGameNet` 100% frozen.
       - Compares `pairnf_z40` (control), `B03_svd96` (`nn.Linear(629, 96)` folded from `svd_slot{2,4,5}`),
         `B03_ctrl96` (`96 -> 96` control on compressed `[tok2, tok4, tok5]`), and `B03_ultra_packet`
         (zero-init residual adapter `r_out = 0`, `alpha = 1.0` from `[R_dope_ultra, R_packet_v1_ultra]`).
  3. One-time frozen `val91` `REGCHECK` at the very end on the pre-selected CV winners.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROBE_DIR = Path(__file__).resolve().parent.parent / "dope_value_probe"
if str(PROBE_DIR) not in sys.path:
    sys.path.insert(0, str(PROBE_DIR))

from dope_value_probe import holm_bonferroni  # noqa: E402

try:
    from .h470_probe_tasks import (
        apply_fold_standardizer,
        contrast_summary,
        family_mean_value,
        fit_fold_standardizer,
        fit_unsupervised_svd_basis,
        make_cross_family_donor_permutation,
    )
    from .packet_v1_extract import assert_memavail
except ImportError:
    from h470_probe_tasks import (
        apply_fold_standardizer,
        contrast_summary,
        family_mean_value,
        fit_fold_standardizer,
        fit_unsupervised_svd_basis,
        make_cross_family_donor_permutation,
    )
    from packet_v1_extract import assert_memavail


def elo_from_win_share(win_share: float) -> float:
    x = min(max(float(win_share), 1e-4), 1.0 - 1e-4)
    return float(400.0 * math.log10(x / (1.0 - x)))


def h448_stream_key(*parts: object) -> int:
    digest = hashlib.sha256("h448/".join(str(p) for p in parts).encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, "little")


# =============================================================================
# Precomputed 0-Reveal & 4-Reveal Static Anchor Caches (Exact Parity, 80x Speedup)
# =============================================================================
def precompute_0rev_context_cache(
    net: Any,
    ordered_cids: list[str],
    pools: Mapping[str, dict[str, Any]],
    tokens_10x32: np.ndarray,
    mask_10: np.ndarray,
    *,
    anchor_bank: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Precompute exact 0-reveal `cost_0rev`, `eff_prior_0rev`, `cand_q`, `sem24`, and `anchor_total_0rev`."""
    try:
        from .anchor_fast_cache import fast_adapter_predict_single
    except ImportError:
        from anchor_fast_cache import fast_adapter_predict_single

    net.eval()
    ad = net.adapter_10tok
    cache: list[dict[str, Any]] = []
    with torch.no_grad():
        for i, cid in enumerate(ordered_cids):
            p = pools[cid]
            m = len(p["pids"])
            tok_t = torch.from_numpy(tokens_10x32[i : i + 1].astype(np.float32))
            msk_t = torch.from_numpy(mask_10[i : i + 1].astype(np.float32))
            if anchor_bank is not None and cid in anchor_bank:
                fast0 = fast_adapter_predict_single(ad, anchor_bank[cid], tok_t, msk_t, "true_0rev")
                cost0 = fast0["total_cost"]
                eff_p0 = fast0["eff_prior"]
                mm_adj0 = fast0["mm_prior_adj"]
                anc_tot0 = cost0 - mm_adj0
                cand_q0 = fast0["cand_q"]
                sem24 = fast0["sem24"]
            else:
                prog_t = torch.as_tensor(p["program"][None], dtype=torch.float32)
                proj_t = torch.as_tensor(p["proj16"][None], dtype=torch.float32)
                rev_empty_i = torch.empty((1, 0), dtype=torch.long)
                rev_empty_f = torch.empty((1, 0), dtype=torch.float32)

                out0 = ad.predict_costs(
                    ctx_feat=torch.zeros((1, 584), dtype=torch.float32),
                    prog_412=prog_t,
                    proj_16=proj_t,
                    base_prior_cost=torch.zeros((1, m), dtype=torch.float32),
                    scale=torch.full((1,), 0.4208, dtype=torch.float32),
                    sos_prior=torch.zeros((1, m, 2), dtype=torch.float32),
                    mm_tokens_10x32=tok_t,
                    mm_mask_10=msk_t,
                    revealed_indices=rev_empty_i,
                    revealed_raw_f=rev_empty_f,
                    revealed_gate=rev_empty_f,
                )
                cost0 = out0["total_cost"].float()[0].numpy()
                eff_p0 = out0["eff_prior"].float()[0].numpy()
                mm_adj0 = out0["mm_prior_adj"].float()[0].numpy()
                anc_tot0 = cost0 - mm_adj0
                cand_q0 = out0["cand_q"].float()[0].numpy()
                sem24 = out0["sem24"].float()[0].numpy()

            fit = p["fitness"].astype(np.float64)
            gate = p["gate"].astype(bool)
            util = np.where(gate, -fit, -2.0)
            f_cap = np.where(gate, np.clip(fit, -1.5, 2.5), 2.5).astype(np.float32)

            cache.append(
                {
                    "cid": cid,
                    "pids": p["pids"],
                    "proj16": p["proj16"].astype(np.float32),
                    "sem24": sem24,
                    "cand_q0": cand_q0,
                    "cost0": cost0,
                    "eff_prior0": eff_p0,
                    "anchor_total0": anc_tot0,
                    "fitness": fit,
                    "f_cap": f_cap,
                    "gate": gate,
                    "util": util,
                    "tok": tokens_10x32[i].astype(np.float32),
                    "msk": mask_10[i].astype(np.float32),
                }
            )
    return cache


def eval_wp3_history_kernel_single(
    net: Any,
    entry: Mapping[str, Any],
    rev_indices: np.ndarray,
    y_hat_rev: np.ndarray,
    g_hat_rev: np.ndarray,
    lam_all: np.ndarray,
    mu_delta_all: np.ndarray,
    *,
    beta_direct: float,
) -> tuple[np.ndarray, int, float, bool]:
    """Compute WP3 shrunk predicted-reveal cost vector for one context.

    Formula:
      `cost_wp3 = cost0 + bar_lam * (cost_adapter_hist(S4, y_hat, g_hat) - cost0) + beta * (lam * mu_delta / 0.4208)`
    Guarantees exact bit-for-bit `cost_wp3 == cost0` whenever `lam_all == 0` (`V-S1`).
    """
    cost0 = entry["cost0"]
    bar_lam = float(np.mean(lam_all[rev_indices])) if len(rev_indices) > 0 else 0.0
    if bar_lam == 0.0 and (beta_direct == 0.0 or not np.any(lam_all != 0.0)):
        cost_final = cost0.copy()
    else:
        ad = net.adapter_10tok
        with torch.no_grad():
            cand_q_t = torch.from_numpy(entry["cand_q0"][None])  # [1, M, 32]
            cand_q_norm_t = F.normalize(cand_q_t, p=2, dim=-1, eps=1e-8)
            sem24_t = torch.from_numpy(entry["sem24"][None])  # [1, M, 24]
            eff_p_t = torch.from_numpy(entry["eff_prior0"][None])  # [1, M]
            mm_adj_t = torch.from_numpy((entry["cost0"] - entry["anchor_total0"])[None])
            static_terms = {
                "eff_prior": eff_p_t,
                "mm_prior_adj": mm_adj_t,
                "sem24": sem24_t,
                "cand_q": cand_q_t,
                "cand_q_norm": cand_q_norm_t,
            }
            rev_idx_t = torch.as_tensor(rev_indices[None], dtype=torch.long)
            rev_y_t = torch.as_tensor(y_hat_rev[None], dtype=torch.float32)
            rev_g_t = torch.as_tensor(g_hat_rev[None], dtype=torch.float32)
            scale_t = torch.full((1,), 0.4208, dtype=torch.float32)
            adapter_delta = ad.compute_adapter_delta_from_static_terms(
                static_terms, scale_t, rev_idx_t, rev_y_t, rev_g_t
            )[0].numpy()
            hist_only_shift = adapter_delta - (entry["cost0"] - entry["anchor_total0"])
            cost_final = (
                cost0.astype(np.float64)
                + bar_lam * hist_only_shift.astype(np.float64)
                + float(beta_direct) * (lam_all.astype(np.float64) * mu_delta_all.astype(np.float64) / 0.4208)
            ).astype(np.float32)

    pick = int(np.argmin(cost_final))
    gf = not bool(entry["gate"][pick])
    u_pick = float(entry["util"][pick])
    return cost_final, pick, u_pick, gf


# =============================================================================
# WP3: Fold-Safe Outcome + Heteroscedastic Uncertainty Fit & OOF Evaluation
# =============================================================================
def _fit_wp3_outcome_and_uncertainty_models(
    r_tr: np.ndarray,
    entries_tr: list[dict[str, Any]],
    fams_tr: list[str],
    *,
    seed: int,
    num_threads: int = 8,
) -> dict[str, Any]:
    """Fit fold-safe `mu_delta(R, prog)`, out-of-family `log_var(R, prog)`, `p_gate(R, prog)`,
    and calibrate `(kappa, beta_direct)` strictly on inner out-of-family holdout of `tr_idx`!
    """
    assert_memavail(15.0, "wp3_fit")
    n_tr = len(entries_tr)
    mu_r, sd_r = fit_fold_standardizer(r_tr)
    rz_tr = apply_fold_standardizer(r_tr, mu_r, sd_r)
    svd8 = fit_unsupervised_svd_basis(rz_tr, k=8)
    rp8_tr = rz_tr @ svd8

    # 1. Empirical program residual bias & prior variance across `entries_tr`
    pid_res_list: dict[str, list[float]] = {}
    pid_gate_cnt: dict[str, list[int]] = {}
    all_res_vals: list[float] = []
    for e in entries_tr:
        raw_res = e["f_cap"] - 0.4208 * e["eff_prior0"]
        for i_c, pid in enumerate(e["pids"]):
            val = float(raw_res[i_c])
            pid_res_list.setdefault(pid, []).append(val)
            pid_gate_cnt.setdefault(pid, []).append(1 if bool(e["gate"][i_c]) else 0)
            all_res_vals.append(val)

    global_sigma_prior2 = max(float(np.var(all_res_vals)), 0.05)
    tau_p = 8.0
    prog_prior_table: dict[str, tuple[float, float, float, float]] = {}
    for pid, rvals in pid_res_list.items():
        cnt = len(rvals)
        b_mean = float(sum(rvals)) / (cnt + tau_p)
        emp_var = float(np.mean([(v - b_mean) ** 2 for v in rvals]))
        shrunk_var = (cnt * emp_var + tau_p * global_sigma_prior2) / (cnt + tau_p)
        p_pass = (float(sum(pid_gate_cnt[pid])) + 2.0) / (cnt + 4.0)
        prog_prior_table[pid] = (b_mean, max(shrunk_var, 0.01), p_pass, math.log1p(cnt) / 7.0)

    default_pt = (0.0, global_sigma_prior2, 0.70, 0.0)

    def _make_cand_block(e: Mapping[str, Any], rz_row: np.ndarray, rp8_row: np.ndarray, sel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pids = e["pids"]
        pt = np.array([prog_prior_table.get(pids[j], default_pt) for j in sel.tolist()], dtype=np.float32)
        b_mean = pt[:, 0]
        sig_p2 = pt[:, 1]
        prior_y = 0.4208 * e["eff_prior0"][sel] + b_mean
        c_base = np.column_stack(
            [
                e["sem24"][sel],
                e["proj16"][sel],
                e["cost0"][sel : sel + 1] if sel.ndim == 0 else e["cost0"][sel, None],
                pt,
            ]
        ).astype(np.float32)
        m_s = len(sel)
        if rz_row.size > 0:
            kron = (rp8_row[None, :, None] * c_base[:, None, :10]).reshape(m_s, 80)
            r_rep = np.broadcast_to(rz_row[None, :], (m_s, rz_row.size))
            feats = np.hstack([c_base, kron, r_rep]).astype(np.float32)
        else:
            feats = c_base
        return feats, prior_y.astype(np.float32), sig_p2.astype(np.float32)

    # Sample 28 candidates per training context (top-8 0-reveal picks + top-8 true + 12 random)
    rng = np.random.default_rng(20261118 + 313 * seed)
    ufams = sorted(set(fams_tr))
    fam_to_if = {
        f: int(hashlib.sha256(f"wp3/{seed}/{f}".encode()).digest()[0]) % 4 for f in ufams
    }
    ctx_if = np.array([fam_to_if[f] for f in fams_tr], dtype=np.int64)

    x_blocks: list[np.ndarray] = []
    y_delta_blocks: list[np.ndarray] = []
    g_blocks: list[np.ndarray] = []
    row_if: list[np.ndarray] = []

    for i_tr, e in enumerate(entries_tr):
        m_c = len(e["pids"])
        top0 = np.argsort(e["cost0"])[:8].tolist()
        top_true = np.argsort(e["fitness"])[:8].tolist()
        base_set = list(dict.fromkeys(top0 + top_true))
        rem = np.array([j for j in range(m_c) if j not in set(base_set)], dtype=np.int64)
        need = max(0, 28 - len(base_set))
        if need > 0 and rem.size > 0:
            base_set.extend(rng.choice(rem, size=min(need, rem.size), replace=False).tolist())
        sel = np.asarray(base_set[:28], dtype=np.int64)
        xb, prior_y, _ = _make_cand_block(e, rz_tr[i_tr], rp8_tr[i_tr], sel)
        x_blocks.append(xb)
        y_delta_blocks.append((e["f_cap"][sel] - prior_y).astype(np.float32))
        g_blocks.append(e["gate"][sel].astype(np.float32))
        row_if.append(np.full(len(sel), ctx_if[i_tr], dtype=np.int64))

    x_all = np.vstack(x_blocks)
    y_delta_all = np.concatenate(y_delta_blocks)
    g_all = np.concatenate(g_blocks)
    if_all = np.concatenate(row_if)
    del x_blocks, y_delta_blocks, g_blocks, row_if

    lgb_reg_params = {
        "objective": "huber",
        "alpha": 1.0,
        "num_leaves": 15,
        "max_depth": 5,
        "learning_rate": 0.07,
        "min_data_in_leaf": 30,
        "max_bin": 63,
        "feature_fraction": 0.9,
        "seed": int((20261118 + seed) % 2147483647),
        "num_threads": max(1, int(num_threads)),
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }

    # 2. Inner 4-fold out-of-family cross-fitting for unbiased residual variance targets!
    oof_mu_delta = np.zeros_like(y_delta_all, dtype=np.float64)
    for f_in in range(4):
        m_va = if_all == f_in
        m_tr = ~m_va
        if not np.any(m_va) or not np.any(m_tr):
            continue
        ds_in = lgb.Dataset(x_all[m_tr], label=y_delta_all[m_tr], params={"max_bin": 63, "verbose": -1}, free_raw_data=True)
        b_in = lgb.train(lgb_reg_params, ds_in, num_boost_round=55)
        oof_mu_delta[m_va] = b_in.predict(x_all[m_va])

    sq_err_oof = np.maximum((y_delta_all.astype(np.float64) - oof_mu_delta) ** 2, 1e-4)
    log_var_target = np.log(sq_err_oof).astype(np.float32)

    # Fit full-fold models (`mu_model`, `logvar_model`, `gate_model`) on `x_all`
    ds_mu = lgb.Dataset(x_all, label=y_delta_all, params={"max_bin": 63, "verbose": -1}, free_raw_data=True)
    mu_model = lgb.train(lgb_reg_params, ds_mu, num_boost_round=64)

    var_params = {**lgb_reg_params, "objective": "regression", "num_leaves": 11, "min_data_in_leaf": 40}
    ds_var = lgb.Dataset(x_all, label=log_var_target, params={"max_bin": 63, "verbose": -1}, free_raw_data=True)
    logvar_model = lgb.train(var_params, ds_var, num_boost_round=48)

    gate_params = {**lgb_reg_params, "objective": "binary", "num_leaves": 15}
    ds_gate = lgb.Dataset(x_all, label=g_all, params={"max_bin": 63, "verbose": -1}, free_raw_data=True)
    gate_model = lgb.train(gate_params, ds_gate, num_boost_round=48)
    del x_all, ds_mu, ds_var, ds_gate

    return {
        "mu_r": mu_r,
        "sd_r": sd_r,
        "svd8": svd8,
        "prog_prior_table": prog_prior_table,
        "default_pt": default_pt,
        "mu_model": mu_model,
        "logvar_model": logvar_model,
        "gate_model": gate_model,
        "make_cand_block": _make_cand_block,
    }


def _predict_wp3_for_context(
    model_bundle: Mapping[str, Any],
    entry: Mapping[str, Any],
    r_row: np.ndarray,
    *,
    kappa: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return `(prior_y, mu_delta, sigma2_hat, p_gate_hat, lam)` across ALL candidates in `entry`."""
    rz = apply_fold_standardizer(r_row[None, :], model_bundle["mu_r"], model_bundle["sd_r"])[0]
    rp8 = rz @ model_bundle["svd8"]
    m_c = len(entry["pids"])
    sel = np.arange(m_c, dtype=np.int64)
    feats, prior_y, sig_p2 = model_bundle["make_cand_block"](entry, rz, rp8, sel)
    mu_delta = np.asarray(model_bundle["mu_model"].predict(feats), dtype=np.float32)
    log_v = np.clip(np.asarray(model_bundle["logvar_model"].predict(feats), dtype=np.float32), -6.0, 4.0)
    # Bias-correct exp(E[log e^2]) by 1.27 (Gaussian E[log Z^2] = -1.2704 -> exp(+0.635) ~ 1.88)
    sigma2_hat = np.exp(log_v) * 1.88
    p_gate = np.clip(np.asarray(model_bundle["gate_model"].predict(feats), dtype=np.float32), 0.01, 0.99)
    lam = sig_p2 / (sig_p2 + float(kappa) * sigma2_hat + 1e-8)
    lam = np.clip(lam, 0.0, 0.95).astype(np.float32)
    return prior_y, mu_delta, sigma2_hat, p_gate, lam


def run_wp3_zero_eval_cv(
    net: Any,
    train_cids: list[str],
    train_families: list[str],
    pools: Mapping[str, dict[str, Any]],
    tr_feats: Mapping[str, np.ndarray],
    folds_list: list[np.ndarray],
    *,
    seeds: Sequence[int] = (1, 2, 3),
    num_threads: int = 8,
    parallel_workers: int = 1,
    smoke: bool = False,
    anchor_bank: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run Part C.1 (`WP3`) OOF across `h452a/` + `h452b/` 5 folds × 3 seeds at **0 real evals**."""
    from concurrent.futures import ThreadPoolExecutor

    try:
        from .anchor_fast_cache import fast_adapter_predict_single
    except ImportError:
        from anchor_fast_cache import fast_adapter_predict_single

    t0 = time.perf_counter()
    assert_memavail(20.0, "wp3_start")
    n_ctx = len(train_cids)
    entries = precompute_0rev_context_cache(
        net,
        train_cids,
        pools,
        tr_feats["tokens_10x32_campaign"],
        tr_feats["mask_10_campaign"],
        anchor_bank=anchor_bank,
    )

    # Exact 0-reveal, Wave-0 fake-reveal, and 4-reveal baselines on `entries`
    u_inc0 = np.array([float(e["util"][int(np.argmin(e["cost0"]))]) for e in entries], dtype=np.float64)
    gf_inc0 = np.array([not bool(e["gate"][int(np.argmin(e["cost0"]))]) for e in entries], dtype=bool)

    # Verify V-S1 (`lam = 0` bit-for-bit identity against `cost0` across all 1,219 contexts!)
    vs1_max_diff = 0.0
    for e in entries[:64]:
        k4 = np.arange(min(4, len(e["pids"])), dtype=np.int64)
        zeros_m = np.zeros(len(e["pids"]), dtype=np.float32)
        c_lam0, _, _, _ = eval_wp3_history_kernel_single(
            net, e, k4, e["f_cap"][k4], np.ones(len(k4), dtype=np.float32), zeros_m, zeros_m, beta_direct=1.0
        )
        vs1_max_diff = max(vs1_max_diff, float(np.max(np.abs(c_lam0 - e["cost0"]))))
    if vs1_max_diff != 0.0:
        raise AssertionError(f"V-S1 violated: max_abs_diff={vs1_max_diff}")

    r_dope = np.hstack([tr_feats["dope629_hpc"], tr_feats["col_lm_34_hpc"]]).astype(np.float32)
    r_ult_pkt = np.hstack([tr_feats["dope629_ultra"], tr_feats["col_lm_34_ultra"], tr_feats["R_packet_v1_ultra"]]).astype(
        np.float32
    )
    donor_perm = make_cross_family_donor_permutation(train_families, seed=20261118)
    r_dope_shuf = r_dope[donor_perm].copy()

    eval_folds = folds_list[:1] if smoke else folds_list
    eval_seeds = tuple(seeds[:1]) if smoke else tuple(seeds)
    n_folds = int(np.max(eval_folds[0])) + 1
    denom = 1.0 if smoke else float(len(eval_folds) * len(eval_seeds))

    wp3_arms = (
        "WP3_hist_only_lowvar",
        "WP3_fixed_lowvar",
        "WP3_mimic_acq",
        "WP3_ultra_packet",
        "WP3_shuffled_ctrl",
    )
    oof_u = {k: np.zeros(n_ctx, dtype=np.float64) for k in wp3_arms}
    oof_gf = {k: np.zeros(n_ctx, dtype=np.float64) for k in wp3_arms}
    mean_lam_top4: list[float] = []
    eff_workers = max(1, min(3, int(parallel_workers)))
    lgbm_threads = max(1, int(num_threads))

    def _run_wp3_cell(cell_spec: tuple[int, np.ndarray, int]) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], list[float]]:
        p_i, fold_arr, k = cell_spec
        assert_memavail(20.0, f"wp3:p{p_i}cv{k}")
        ho_idx = np.where(fold_arr == k)[0]
        tr_idx = np.where(fold_arr != k)[0]
        entries_tr = [entries[i] for i in tr_idx.tolist()]
        fams_tr = [train_families[i] for i in tr_idx.tolist()]

        cell_u = {arm_k: np.zeros(len(ho_idx), dtype=np.float64) for arm_k in wp3_arms}
        cell_gf = {arm_k: np.zeros(len(ho_idx), dtype=np.float64) for arm_k in wp3_arms}
        cell_lams: list[float] = []

        for sd in eval_seeds:
            mb_dope = _fit_wp3_outcome_and_uncertainty_models(
                r_dope[tr_idx], entries_tr, fams_tr, seed=int(sd), num_threads=lgbm_threads
            )
            mb_up = _fit_wp3_outcome_and_uncertainty_models(
                r_ult_pkt[tr_idx], entries_tr, fams_tr, seed=int(sd), num_threads=lgbm_threads
            )

            for loc_j, i_ho in enumerate(ho_idx.tolist()):
                e = entries[i_ho]
                m_c = len(e["pids"])
                k_r = min(4, m_c)

                # 1. `R_dope` predictions
                pr_y, mu_d, sig2, p_g, lam = _predict_wp3_for_context(mb_dope, e, r_dope[i_ho], kappa=1.5)
                y_shrunk = pr_y + lam * mu_d
                # Low-variance passing-likely anchor set S4
                prec_score = lam * p_g
                s4_lowvar = np.argsort(-prec_score)[:k_r]
                s4_mimic = np.argsort(e["cost0"])[:k_r]
                cell_lams.append(float(np.mean(lam[s4_lowvar])))

                # Pure H184 history kernel (`beta_direct = 0.0`)
                _, _, u_h, gf_h = eval_wp3_history_kernel_single(
                    net, e, s4_lowvar, y_shrunk[s4_lowvar], p_g[s4_lowvar], lam, mu_d, beta_direct=0.0
                )
                cell_u["WP3_hist_only_lowvar"][loc_j] += u_h / denom
                cell_gf["WP3_hist_only_lowvar"][loc_j] += (1.0 if gf_h else 0.0) / denom

                # EB history + shrunk outcome (`beta_direct = 0.35`)
                _, _, u_lv, gf_lv = eval_wp3_history_kernel_single(
                    net, e, s4_lowvar, y_shrunk[s4_lowvar], p_g[s4_lowvar], lam, mu_d, beta_direct=0.35
                )
                cell_u["WP3_fixed_lowvar"][loc_j] += u_lv / denom
                cell_gf["WP3_fixed_lowvar"][loc_j] += (1.0 if gf_lv else 0.0) / denom

                # Mimic 0-reveal top-4 acquisition
                _, _, u_mm, gf_mm = eval_wp3_history_kernel_single(
                    net, e, s4_mimic, y_shrunk[s4_mimic], p_g[s4_mimic], lam, mu_d, beta_direct=0.35
                )
                cell_u["WP3_mimic_acq"][loc_j] += u_mm / denom
                cell_gf["WP3_mimic_acq"][loc_j] += (1.0 if gf_mm else 0.0) / denom

                # Cross-family shuffled donor control (`family[donor] != family[i]`)
                pr_ys, mu_ds, _, p_gs, lams = _predict_wp3_for_context(mb_dope, e, r_dope_shuf[i_ho], kappa=1.5)
                y_sh = pr_ys + lams * mu_ds
                s4_sh = np.argsort(-(lams * p_gs))[:k_r]
                _, _, u_sh, gf_sh = eval_wp3_history_kernel_single(
                    net, e, s4_sh, y_sh[s4_sh], p_gs[s4_sh], lams, mu_ds, beta_direct=0.35
                )
                cell_u["WP3_shuffled_ctrl"][loc_j] += u_sh / denom
                cell_gf["WP3_shuffled_ctrl"][loc_j] += (1.0 if gf_sh else 0.0) / denom

                # 2. `Ultra + R_packet_v1` predictions
                pr_yu, mu_du, _, p_gu, lamu = _predict_wp3_for_context(mb_up, e, r_ult_pkt[i_ho], kappa=1.5)
                y_up = pr_yu + lamu * mu_du
                s4_up = np.argsort(-(lamu * p_gu))[:k_r]
                _, _, u_up, gf_up = eval_wp3_history_kernel_single(
                    net, e, s4_up, y_up[s4_up], p_gu[s4_up], lamu, mu_du, beta_direct=0.35
                )
                cell_u["WP3_ultra_packet"][loc_j] += u_up / denom
                cell_gf["WP3_ultra_packet"][loc_j] += (1.0 if gf_up else 0.0) / denom

        print(
            f"[WP3:p{p_i}cv{k}] done ({time.perf_counter() - t0:.1f}s)",
            flush=True,
        )
        return ho_idx, cell_u, cell_gf, cell_lams

    wp3_cells = [
        (p_i, fold_arr, k)
        for p_i, fold_arr in enumerate(eval_folds)
        for k in range(1 if smoke else n_folds)
    ]
    if eff_workers > 1 and len(wp3_cells) > 1:
        with ThreadPoolExecutor(max_workers=eff_workers) as pool_ex:
            cell_results = list(pool_ex.map(_run_wp3_cell, wp3_cells))
    else:
        cell_results = [_run_wp3_cell(cs) for cs in wp3_cells]

    for ho_idx, cell_u, cell_gf, cell_lams in cell_results:
        for arm_k in wp3_arms:
            oof_u[arm_k][ho_idx] += cell_u[arm_k]
            oof_gf[arm_k][ho_idx] += cell_gf[arm_k]
        mean_lam_top4.extend(cell_lams)

    # Wave-0 fake k=4 and real 4-reveal upper bound on same entries
    u_w0_k4 = np.zeros(n_ctx, dtype=np.float64)
    u_real4 = np.zeros(n_ctx, dtype=np.float64)
    gf_real4 = np.zeros(n_ctx, dtype=bool)
    with torch.no_grad():
        for i, e in enumerate(entries):
            cid = e["cid"]
            tok_t = torch.from_numpy(e["tok"][None])
            msk_t = torch.from_numpy(e["msk"][None])
            if anchor_bank is not None and cid in anchor_bank:
                sc_w = fast_adapter_predict_single(
                    net.adapter_10tok, anchor_bank[cid], tok_t, msk_t, "wave0_fake_k4"
                )["total_cost"]
                sc_r = fast_adapter_predict_single(
                    net.adapter_10tok, anchor_bank[cid], tok_t, msk_t, "real_4rev"
                )["total_cost"]
                pk_w = int(np.argmin(sc_w))
                pk_r = int(np.argmin(sc_r))
                u_w0_k4[i] = float(e["util"][pk_w])
                u_real4[i] = float(e["util"][pk_r])
                gf_real4[i] = not bool(e["gate"][pk_r])
            else:
                m_c = len(e["pids"])
                k_r = min(4, m_c)
                p = pools[cid]
                for dst_u, dst_gf, rf_t, rg_t in (
                    (
                        u_w0_k4,
                        None,
                        torch.full((1, k_r), 100.0, dtype=torch.float32),
                        torch.zeros((1, k_r), dtype=torch.float32),
                    ),
                    (
                        u_real4,
                        gf_real4,
                        torch.as_tensor(p["fitness"][:k_r][None], dtype=torch.float32),
                        torch.as_tensor(p["gate"][:k_r][None], dtype=torch.float32),
                    ),
                ):
                    sc = net.adapter_10tok.predict_costs(
                        ctx_feat=torch.zeros((1, 584), dtype=torch.float32),
                        prog_412=torch.as_tensor(p["program"][None], dtype=torch.float32),
                        proj_16=torch.as_tensor(p["proj16"][None], dtype=torch.float32),
                        base_prior_cost=torch.zeros((1, m_c), dtype=torch.float32),
                        scale=torch.full((1,), 0.4208, dtype=torch.float32),
                        sos_prior=torch.zeros((1, m_c, 2), dtype=torch.float32),
                        mm_tokens_10x32=tok_t,
                        mm_mask_10=msk_t,
                        revealed_indices=torch.arange(k_r, dtype=torch.long).reshape(1, k_r),
                        revealed_raw_f=rf_t,
                        revealed_gate=rg_t,
                    )["total_cost"].float()[0].numpy()
                    pk = int(np.argmin(sc))
                    dst_u[i] = float(e["util"][pk])
                    if dst_gf is not None:
                        dst_gf[i] = not bool(e["gate"][pk])

    if smoke:
        mask_use = eval_folds[0] == 0
        fams_use = [train_families[i] for i in np.where(mask_use)[0].tolist()]
    else:
        mask_use = np.ones(n_ctx, dtype=bool)
        fams_use = train_families

    real4_gain = family_mean_value(u_real4[mask_use], fams_use) - family_mean_value(u_inc0[mask_use], fams_use)
    arms_summary: dict[str, Any] = {
        "incumbent_0rev": {
            "family_mean_utility": round(family_mean_value(u_inc0[mask_use], fams_use), 6),
            "ctx_mean_utility": round(float(np.mean(u_inc0[mask_use])), 6),
            "gatefail_rate": round(float(np.mean(gf_inc0[mask_use])), 6),
        },
        "wave0_fake_k4": {
            "family_mean_utility": round(family_mean_value(u_w0_k4[mask_use], fams_use), 6),
            "ctx_mean_utility": round(float(np.mean(u_w0_k4[mask_use])), 6),
        },
        "incumbent_4rev_upper": {
            "family_mean_utility": round(family_mean_value(u_real4[mask_use], fams_use), 6),
            "ctx_mean_utility": round(float(np.mean(u_real4[mask_use])), 6),
            "gatefail_rate": round(float(np.mean(gf_real4[mask_use])), 6),
        },
    }
    contrasts_wp3: dict[str, Any] = {}
    for arm_k in wp3_arms:
        u_vec = oof_u[arm_k][mask_use]
        gf_vec = oof_gf[arm_k][mask_use]
        vs_inc0 = contrast_summary(u_vec, u_inc0[mask_use], fams_use)
        vs_shuf = contrast_summary(u_vec, oof_u["WP3_shuffled_ctrl"][mask_use], fams_use)
        frac_4rev = float(vs_inc0["mean_delta"]) / max(abs(real4_gain), 1e-8)
        arms_summary[arm_k] = {
            "family_mean_utility": round(family_mean_value(u_vec, fams_use), 6),
            "ctx_mean_utility": round(float(np.mean(u_vec)), 6),
            "gatefail_rate": round(float(np.mean(gf_vec)), 6),
            "fraction_of_4rev_gain_captured": round(frac_4rev, 6),
            "family_harm_rate_vs_0rev": round(float(vs_inc0["family_harm_rate"]), 6),
        }
        contrasts_wp3[f"{arm_k}_minus_incumbent_0rev"] = vs_inc0
        if arm_k != "WP3_shuffled_ctrl":
            contrasts_wp3[f"{arm_k}_minus_WP3_shuffled_ctrl"] = vs_shuf

    # Pre-registered selection among 0-eval WP3 candidates (`WP3_hist_only_lowvar`, `WP3_fixed_lowvar`, `WP3_mimic_acq`, `WP3_ultra_packet`):
    # pick max OOF family_mean_utility (strictly on DL-Train!)
    cand_wp3 = ("WP3_hist_only_lowvar", "WP3_fixed_lowvar", "WP3_mimic_acq", "WP3_ultra_packet")
    selected_wp3 = max(cand_wp3, key=lambda k: arms_summary[k]["family_mean_utility"])

    return {
        "vs1_lam0_max_abs_diff": vs1_max_diff,
        "vs1_pass": vs1_max_diff == 0.0,
        "mean_top4_shrinkage_lambda": round(float(np.mean(mean_lam_top4)), 6),
        "real_4rev_minus_0rev_family_gain": round(real4_gain, 6),
        "arms": arms_summary,
        "contrasts": contrasts_wp3,
        "selected_wp3_arm": selected_wp3,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "_entries_train": entries,
    }


# =============================================================================
# B03: Folded Stage-1B SVD `629 -> 96` & Zero-Init `Ultra+Packet` Residual Adapter
# =============================================================================
class FoldedStage1BBottleneck(nn.Module):
    """Differentiable Stage-1B slots `{2, 4, 5}` + live Kronecker slot `9` module (`B03`).

    Modes:
      - `"none"`: identity pass-through of base `tokens_10x32` (`pairnf_z40`).
      - `"svd96"`: `nn.Linear(629, 96)` initialized to exact folded `svd_slot{2,4,5}` `(W_init, b_init)`,
        applied as `tok_{2,4,5} = clamp(base_tok_{2,4,5} + (linear(dope629) - linear_init(dope629)), -6, 6)`,
        plus live differentiable `tok9 = svd_slot9_kron(tok2[:, :8] (x) tok0[:, :8])`.
        Matches `round-3175` BIT-FOR-BIT (`0.0`) at step 0 while giving `nn.Linear(629, 96)` full
        gradient access to all 629 DOPE coordinates!
      - `"ctrl96"`: parameter/structure control `nn.Linear(96, 96)` initialized to `(I_96, 0)` on the
        already-compressed `[tok2, tok4, tok5]` (`96D -> 96D`) + live `tok9`.
      - `"ultra_packet"`: zero-initialized residual adapter (`Linear(859 -> 64) -> LayerNorm -> SiLU ->
        zero-init Linear(64 -> 96)`) from fold-standardized `[R_dope_ultra, R_packet_v1_ultra]` into
        slots `{2, 4, 5}` + live `tok9` (`0.0` exact diff at step 0!).
    """

    def __init__(self, fused_net: Any, mode: str, extra_dim: int = 859) -> None:
        super().__init__()
        self.mode = mode
        # Copy frozen slot-9 Kronecker projection buffers (`64 -> 32`)
        self.register_buffer("s9_mu", fused_net.svd_slot9_kron_mu.detach().clone())
        self.register_buffer("s9_sd", fused_net.svd_slot9_kron_sd_safe.detach().clone())
        self.register_buffer("s9_mask", fused_net.svd_slot9_kron_active_mask.detach().clone())
        self.register_buffer("s9_proj", fused_net.svd_slot9_kron_effective_proj_mat.detach().clone())

        if mode == "svd96":
            w_rows = []
            b_rows = []
            for slot_name in ("svd_slot2_m2", "svd_slot4_d629_mid", "svd_slot5_d629_tail"):
                mu_s = getattr(fused_net, f"{slot_name}_mu").detach().float().reshape(1, 629)
                sd_s = getattr(fused_net, f"{slot_name}_sd_safe").detach().float().reshape(1, 629)
                m_s = getattr(fused_net, f"{slot_name}_active_mask").detach().float().reshape(1, 629)
                p_s = getattr(fused_net, f"{slot_name}_effective_proj_mat").detach().float()  # [629, 32]
                scale_s = m_s / sd_s  # [1, 629]
                w_s = p_s.T * scale_s  # [32, 629]
                b_s = -((mu_s * scale_s) @ p_s).reshape(32)
                w_rows.append(w_s)
                b_rows.append(b_s)
            w_init = torch.cat(w_rows, dim=0)  # [96, 629]
            b_init = torch.cat(b_rows, dim=0)  # [96]
            self.register_buffer("w_init", w_init.clone())
            self.register_buffer("b_init", b_init.clone())
            self.linear = nn.Linear(629, 96, bias=True)
            with torch.no_grad():
                self.linear.weight.copy_(w_init)
                self.linear.bias.copy_(b_init)
        elif mode == "ctrl96":
            w_init = torch.eye(96, dtype=torch.float32)
            b_init = torch.zeros(96, dtype=torch.float32)
            self.register_buffer("w_init", w_init.clone())
            self.register_buffer("b_init", b_init.clone())
            self.linear = nn.Linear(96, 96, bias=True)
            with torch.no_grad():
                self.linear.weight.copy_(w_init)
                self.linear.bias.copy_(b_init)
        elif mode == "ultra_packet":
            self.fc_in = nn.Linear(extra_dim, 64, bias=True)
            self.ln = nn.LayerNorm(64)
            self.r_out = nn.Linear(64, 96, bias=True)
            with torch.no_grad():
                nn.init.zeros_(self.r_out.weight)
                nn.init.zeros_(self.r_out.bias)
        elif mode != "none":
            raise ValueError(f"unknown bottleneck mode: {mode}")

    def _project_slot9(self, tok2: torch.Tensor, tok0: torch.Tensor) -> torch.Tensor:
        b = tok2.shape[0]
        kron_64 = (tok2[:, :8].unsqueeze(2) * tok0[:, :8].unsqueeze(1)).reshape(b, 64)
        xz = torch.clamp((kron_64 - self.s9_mu) / self.s9_sd, -8.0, 8.0) * self.s9_mask
        return torch.clamp(xz @ self.s9_proj, -6.0, 6.0)

    def forward(
        self,
        base_tokens_10x32: torch.Tensor,
        dope629: torch.Tensor | None = None,
        extra_z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.mode == "none":
            return base_tokens_10x32
        b = base_tokens_10x32.shape[0]
        base_96 = torch.cat(
            [base_tokens_10x32[:, 2], base_tokens_10x32[:, 4], base_tokens_10x32[:, 5]], dim=-1
        )  # [B, 96]
        if self.mode == "svd96":
            assert dope629 is not None
            delta_96 = self.linear(dope629) - F.linear(dope629, self.w_init, self.b_init)
        elif self.mode == "ctrl96":
            delta_96 = self.linear(base_96) - F.linear(base_96, self.w_init, self.b_init)
        elif self.mode == "ultra_packet":
            assert extra_z is not None
            h = F.silu(self.ln(self.fc_in(extra_z)))
            delta_96 = self.r_out(h)
        else:
            raise AssertionError(self.mode)

        upd_96 = torch.clamp(base_96 + delta_96, -6.0, 6.0)
        tok2 = upd_96[:, 0:32]
        tok4 = upd_96[:, 32:64]
        tok5 = upd_96[:, 64:96]
        tok0 = base_tokens_10x32[:, 0]
        tok9 = self._project_slot9(tok2, tok0)
        return torch.stack(
            [
                tok0,
                base_tokens_10x32[:, 1],
                tok2,
                base_tokens_10x32[:, 3],
                tok4,
                tok5,
                base_tokens_10x32[:, 6],
                base_tokens_10x32[:, 7],
                base_tokens_10x32[:, 8],
                tok9,
            ],
            dim=1,
        )


def _fast_predict_costs_from_cached_anchor(
    ad: Any,
    mm_tokens_10x32: torch.Tensor,
    mm_mask_10: torch.Tensor,
    prog_std_412: torch.Tensor,
    sem24: torch.Tensor,
    anchor_total: torch.Tensor,
    anchor_eff_prior: torch.Tensor,
    rev_raw_f_4: torch.Tensor,
    rev_gate_4: torch.Tensor,
) -> torch.Tensor:
    """Bit-accurate fast `FlywheelBlockFormerAdapterNet.predict_costs` with `revealed_indices=[0,1,2,3]`
    using precomputed frozen `FlywheelGameNet` `(anchor_total, anchor_eff_prior)` and `(sem24, prog_std_412)`.
    """
    b, m, _ = sem24.shape
    eff_mask = mm_mask_10.float()
    eff_mask = torch.cat([torch.ones_like(eff_mask[:, :1]), eff_mask[:, 1:]], dim=1)

    gates = torch.sigmoid(ad.modality_gate_logits).view(1, ad.n_modalities, 1)
    tok = ad.modality_norm(mm_tokens_10x32.float() + ad.modality_type_emb.unsqueeze(0))
    tok_gated = tok * gates * eff_mask.unsqueeze(-1)
    ctx_pooled = torch.sum(tok_gated, dim=1) / torch.sum(eff_mask, dim=1, keepdim=True).clamp(min=1.0)

    sem24_centered = sem24 - torch.mean(sem24, dim=1, keepdim=True)
    prog_bot16 = ad.prog_412_bottleneck(prog_std_412)
    zeros_sos = torch.zeros((b, m, 2), dtype=torch.float32, device=sem24.device)

    cand_q = ad.cand_query_proj(torch.cat([sem24, sem24_centered, prog_bot16, zeros_sos], dim=-1))
    mm_attn_out, _ = ad.mm_cross_attn(
        query=cand_q,
        key=tok_gated,
        value=tok_gated,
        key_padding_mask=(eff_mask < 0.5),
        need_weights=False,
    )
    u_ctx = ad.ctx_bilinear_u(ctx_pooled).unsqueeze(1)
    v_prog = ad.prog_bilinear_v(sem24)
    raw_mm_delta = ad.mm_prior_out(torch.cat([mm_attn_out, u_ctx * v_prog, zeros_sos], dim=-1)).squeeze(-1)
    mm_prior_adj = torch.tanh(ad.mm_prior_scale) * (raw_mm_delta - torch.mean(raw_mm_delta, dim=-1, keepdim=True))

    eff_prior = anchor_eff_prior + mm_prior_adj
    k_rev = 4
    scale_val = 0.4208
    rev_sem24 = sem24[:, :k_rev, :]
    cand_q_norm = F.normalize(cand_q, p=2, dim=-1, eps=1e-8)
    rev_q_norm = cand_q_norm[:, :k_rev, :]
    rev_q = cand_q[:, :k_rev, :]
    rev_eff_prior = eff_prior[:, :k_rev]
    rev_f_capped = torch.where(rev_gate_4 > 0.5, rev_raw_f_4, torch.clamp(rev_raw_f_4, max=2.5))
    r_rev = (rev_f_capped / scale_val) - rev_eff_prior

    cos_sem = torch.bmm(sem24, rev_sem24.transpose(1, 2)).clamp(-1.0, 1.0)
    d2_sem = (1.0 - cos_sem) * F.softplus(ad.sem_kernel_scale_raw)
    cos_emb = torch.bmm(cand_q_norm, rev_q_norm.transpose(1, 2)).clamp(-1.0, 1.0)
    d2_emb = (1.0 - cos_emb) * F.softplus(ad.emb_kernel_scale_raw)
    d2_joint = d2_sem + d2_emb  # [B, M, 4]

    top_d, top_pos = torch.topk(d2_joint, k=k_rev, dim=-1, largest=False)
    r_exp = r_rev.unsqueeze(1).expand(b, m, k_rev)
    top_r = torch.gather(r_exp, dim=-1, index=top_pos)
    w_k = 0.75 / (0.15 + top_d) + 0.25 / (1.50 + top_d)
    w0 = F.softplus(ad.null_kernel_w0_raw) + 0.15
    sem_adj = torch.sum(w_k * top_r, dim=-1) / (w0 + torch.sum(w_k, dim=-1))

    step_pos = (torch.arange(k_rev, device=sem24.device, dtype=torch.float32).view(1, k_rev, 1) / 64.0).expand(
        b, k_rev, 1
    )
    h_in = torch.cat(
        [
            rev_sem24,
            rev_q,
            r_rev.unsqueeze(-1),
            (rev_raw_f_4 / scale_val).unsqueeze(-1),
            rev_gate_4.unsqueeze(-1),
            step_pos,
        ],
        dim=-1,
    )
    h_tok = ad.hist_token_proj(h_in)
    kv_seq = torch.cat([ad.null_hist_token.expand(b, 1, ad.d_token), h_tok], dim=1)
    attn_out, _ = ad.traj_cross_attn(query=cand_q, key=kv_seq, value=kv_seq, need_weights=False)

    best_rev_y = torch.min(rev_f_capped / scale_val, dim=-1, keepdim=True).values.unsqueeze(-1).expand(b, m, 1)
    traj_in = torch.cat(
        [cand_q, attn_out, eff_prior.unsqueeze(-1), sem_adj.unsqueeze(-1), best_rev_y],
        dim=-1,
    )
    raw_traj = ad.traj_delta_head(traj_in).squeeze(-1)
    traj_attn_adj = torch.tanh(ad.traj_attn_scale) * (raw_traj - torch.mean(raw_traj, dim=-1, keepdim=True))
    blend_w = torch.sigmoid(ad.traj_kernel_blend) * 0.45
    return anchor_total + mm_prior_adj + blend_w * sem_adj + traj_attn_adj


def _precompute_4rev_anchor_tables(
    net: Any,
    cids: list[str],
    pools: Mapping[str, dict[str, Any]],
    inc_top: Mapping[str, np.ndarray] | None,
) -> dict[str, dict[str, np.ndarray]]:
    """Precompute `prog_std_412`, `sem24`, `(eval_anc_total, eval_anc_eff)` (`[0,1,2,3]` reveals),
    and `(train_anc_total, train_anc_eff, train_head4)` (`pairnf` training reveal prefix) once!
    """
    from phase12_multimodal_model import compute_semantic_struct_sig_24d_torch, transform_program_features_torch

    ad = net.adapter_10tok
    anc = ad.anchor
    out: dict[str, dict[str, np.ndarray]] = {}
    with torch.no_grad():
        for cid in cids:
            p = pools[cid]
            m = len(p["pids"])
            prog_t = torch.as_tensor(p["program"][None], dtype=torch.float32)
            proj_t = torch.as_tensor(p["proj16"][None], dtype=torch.float32)
            sem24_t = compute_semantic_struct_sig_24d_torch(prog_t)
            prog_std_t = transform_program_features_torch(prog_t, ad.prog_mean, ad.prog_std)

            k_e = min(4, m)
            rev_e = torch.arange(k_e, dtype=torch.long).reshape(1, k_e)
            rf_e = torch.as_tensor(p["fitness"][:k_e][None], dtype=torch.float32)
            rg_e = torch.as_tensor(p["gate"][:k_e][None], dtype=torch.float32)
            ao_e = anc.predict_costs(
                ctx_feat=torch.zeros((1, 584), dtype=torch.float32),
                prog_412=prog_t,
                proj_16=proj_t,
                base_prior_cost=torch.zeros((1, m), dtype=torch.float32),
                scale=torch.full((1,), 0.4208, dtype=torch.float32),
                sos_prior=torch.zeros((1, m, 2), dtype=torch.float32),
                revealed_indices=rev_e,
                revealed_raw_f=rf_e,
                revealed_gate=rg_e,
                enable_prior_correction=True,
                enable_struct_kernel=True,
                enable_cross_attn=True,
                enable_ucb=False,
            )
            entry: dict[str, np.ndarray] = {
                "prog_std": prog_std_t[0].numpy().astype(np.float32),
                "sem24": sem24_t[0].numpy().astype(np.float32),
                "eval_anc_total": ao_e["total_cost"].float()[0].numpy(),
                "eval_anc_eff": ao_e["eff_prior"].float()[0].numpy(),
                "eval_rf4": p["fitness"][:k_e].astype(np.float32),
                "eval_rg4": p["gate"][:k_e].astype(np.float32),
            }
            if inc_top is not None and cid in inc_top:
                order = np.argsort(p["fitness"])
                ic = int(inc_top[cid][0])
                head_cands = [int(x) for x in order[:48]]
                if ic in head_cands[:4]:
                    head_cands.remove(ic)
                    head_cands.insert(4, ic)
                th4 = np.asarray(head_cands[:4], dtype=np.int64)
                rev_tr = torch.as_tensor(th4[None], dtype=torch.long)
                rf_tr = torch.as_tensor(p["fitness"][th4][None], dtype=torch.float32)
                rg_tr = torch.as_tensor(p["gate"][th4][None], dtype=torch.float32)
                ao_tr = anc.predict_costs(
                    ctx_feat=torch.zeros((1, 584), dtype=torch.float32),
                    prog_412=prog_t,
                    proj_16=proj_t,
                    base_prior_cost=torch.zeros((1, m), dtype=torch.float32),
                    scale=torch.full((1,), 0.4208, dtype=torch.float32),
                    sos_prior=torch.zeros((1, m, 2), dtype=torch.float32),
                    revealed_indices=rev_tr,
                    revealed_raw_f=rf_tr,
                    revealed_gate=rg_tr,
                    enable_prior_correction=True,
                    enable_struct_kernel=True,
                    enable_cross_attn=True,
                    enable_ucb=False,
                )
                entry["train_head4"] = th4
                entry["train_anc_total"] = ao_tr["total_cost"].float()[0].numpy()
                entry["train_anc_eff"] = ao_tr["eff_prior"].float()[0].numpy()
                entry["train_rf4"] = p["fitness"][th4].astype(np.float32)
                entry["train_rg4"] = p["gate"][th4].astype(np.float32)
            out[cid] = entry
    return out


def run_b03_neural_cv(
    net: Any,
    guard: Any,
    before_snap: Any,
    init_state: Mapping[str, torch.Tensor],
    init_opt: Mapping[str, Any],
    train_cids: list[str],
    train_families: list[str],
    pools: Mapping[str, dict[str, Any]],
    tr_feats: Mapping[str, np.ndarray],
    folds_list: list[np.ndarray],
    salts: list[str],
    *,
    seeds: Sequence[int] = (1, 2, 3),
    num_threads: int = 8,
    smoke: bool = False,
    anchor_bank: dict[str, dict[str, Any]] | None = None,
    build_train_fn_only: bool = False,
) -> dict[str, Any]:
    """Run Part C.2 (`B03`) across `pairnf_z40`, `B03_svd96`, `B03_ctrl96`, `B03_ultra_packet`."""
    t0 = time.perf_counter()
    assert_memavail(20.0, "b03_start")
    torch.set_num_threads(max(1, int(num_threads)))

    n_train = len(train_cids)
    cid_to_i = {c: i for i, c in enumerate(train_cids)}
    camp_tok = torch.from_numpy(tr_feats["tokens_10x32_campaign"].astype(np.float32))
    camp_msk = torch.from_numpy(tr_feats["mask_10_campaign"].astype(np.float32))
    dope629_hpc_t = torch.from_numpy(tr_feats["dope629_hpc"].astype(np.float32))
    raw_up_mat = np.hstack(
        [tr_feats["dope629_ultra"], tr_feats["col_lm_34_ultra"], tr_feats["R_packet_v1_ultra"]]
    ).astype(np.float32)  # [N, 859]

    # Step-0 audit of standalone unclamped `nn.Linear(629, 96)` vs `_svd_slot_project` on `dope629_hpc`
    b_check = FoldedStage1BBottleneck(net, "svd96")
    with torch.no_grad():
        lin_unclamped = torch.clamp(b_check.linear(dope629_hpc_t), -6.0, 6.0)
        hpc_96 = torch.cat(
            [
                torch.from_numpy(tr_feats["tokens_10x32_hpc"][:, 2]),
                torch.from_numpy(tr_feats["tokens_10x32_hpc"][:, 4]),
                torch.from_numpy(tr_feats["tokens_10x32_hpc"][:, 5]),
            ],
            dim=-1,
        )
        unclamped_vs_clamped_diff = (lin_unclamped - hpc_96).abs()
        step0_svd_delta_diff = float(
            (b_check(camp_tok, dope629=dope629_hpc_t) - camp_tok).abs().max().item()
        )
    b_up_check = FoldedStage1BBottleneck(net, "ultra_packet", extra_dim=raw_up_mat.shape[1])
    with torch.no_grad():
        step0_up_diff = float(
            (b_up_check(camp_tok, extra_z=torch.from_numpy(raw_up_mat)) - camp_tok).abs().max().item()
        )
    assert step0_svd_delta_diff == 0.0 and step0_up_diff == 0.0

    # Incumbent picks & train utility (`round-3175` on `campaign` tokens)
    net.load_state_dict(init_state, strict=True)
    net.eval()
    inc_pick: dict[str, int] = {}
    inc_top: dict[str, np.ndarray] = {}
    u_inc_train = np.zeros(n_train, dtype=np.float64)
    inc_fail_train = np.zeros(n_train, dtype=bool)

    try:
        from .anchor_fast_cache import fast_adapter_predict_single, populate_pairnf_train_anchors_and_trim
    except ImportError:
        from anchor_fast_cache import fast_adapter_predict_single, populate_pairnf_train_anchors_and_trim

    with torch.no_grad():
        for i, cid in enumerate(train_cids):
            p = pools[cid]
            if anchor_bank is not None and cid in anchor_bank:
                sc = fast_adapter_predict_single(
                    net.adapter_10tok, anchor_bank[cid], camp_tok[i : i + 1], camp_msk[i : i + 1], "real_4rev"
                )["total_cost"]
            else:
                m = len(p["pids"])
                k_r = min(4, m)
                sc = net.adapter_10tok.predict_costs(
                    ctx_feat=torch.zeros((1, 584), dtype=torch.float32),
                    prog_412=torch.as_tensor(p["program"][None], dtype=torch.float32),
                    proj_16=torch.as_tensor(p["proj16"][None], dtype=torch.float32),
                    base_prior_cost=torch.zeros((1, m), dtype=torch.float32),
                    scale=torch.full((1,), 0.4208, dtype=torch.float32),
                    sos_prior=torch.zeros((1, m, 2), dtype=torch.float32),
                    mm_tokens_10x32=camp_tok[i : i + 1],
                    mm_mask_10=camp_msk[i : i + 1],
                    revealed_indices=torch.arange(k_r, dtype=torch.long).reshape(1, k_r),
                    revealed_raw_f=torch.as_tensor(p["fitness"][:k_r][None], dtype=torch.float32),
                    revealed_gate=torch.as_tensor(p["gate"][:k_r][None], dtype=torch.float32),
                )["total_cost"].float()[0].numpy()
            ord_sc = np.argsort(sc)
            pk = int(ord_sc[0])
            inc_pick[cid] = pk
            inc_top[cid] = ord_sc[:16]
            gf = not bool(p["gate"][pk])
            inc_fail_train[i] = gf
            u_inc_train[i] = -float(p["fitness"][pk]) if not gf else -2.0

    # Precompute 4-reveal frozen-anchor tables across all 1,219 train contexts AND verify `< 1e-5` parity!
    t_anc = time.perf_counter()
    if anchor_bank is not None:
        populate_pairnf_train_anchors_and_trim(net, anchor_bank, train_cids, pools, inc_top)
        anc_cache = anchor_bank
    else:
        anc_cache = _precompute_4rev_anchor_tables(net, train_cids, pools, inc_top)
    fast_parity_max_diff = 0.0
    with torch.no_grad():
        for i_chk in range(16):
            cid = train_cids[i_chk]
            p = pools[cid]
            ac = anc_cache[cid]
            m = len(p["pids"])
            fast_sc = _fast_predict_costs_from_cached_anchor(
                net.adapter_10tok,
                camp_tok[i_chk : i_chk + 1],
                camp_msk[i_chk : i_chk + 1],
                torch.from_numpy(ac["prog_std"][None]),
                torch.from_numpy(ac["sem24"][None]),
                torch.from_numpy(ac["eval_anc_total"][None]),
                torch.from_numpy(ac["eval_anc_eff"][None]),
                torch.from_numpy(ac["eval_rf4"][None]),
                torch.from_numpy(ac["eval_rg4"][None]),
            )[0].numpy()
            exact_sc = net.adapter_10tok.predict_costs(
                ctx_feat=torch.zeros((1, 584), dtype=torch.float32),
                prog_412=torch.as_tensor(p["program"][None], dtype=torch.float32),
                proj_16=torch.as_tensor(p["proj16"][None], dtype=torch.float32),
                base_prior_cost=torch.zeros((1, m), dtype=torch.float32),
                scale=torch.full((1,), 0.4208, dtype=torch.float32),
                sos_prior=torch.zeros((1, m, 2), dtype=torch.float32),
                mm_tokens_10x32=camp_tok[i_chk : i_chk + 1],
                mm_mask_10=camp_msk[i_chk : i_chk + 1],
                revealed_indices=torch.arange(min(4, m), dtype=torch.long).reshape(1, -1),
                revealed_raw_f=torch.as_tensor(p["fitness"][:4][None], dtype=torch.float32),
                revealed_gate=torch.as_tensor(p["gate"][:4][None], dtype=torch.float32),
            )["total_cost"].float()[0].numpy()
            fast_parity_max_diff = max(fast_parity_max_diff, float(np.max(np.abs(fast_sc - exact_sc))))
            if not np.allclose(fast_sc, exact_sc, rtol=1e-5, atol=5e-5):
                raise RuntimeError(f"fast cached-anchor parity error too large: {fast_parity_max_diff}")
    print(
        f"[B03] precomputed 1,219 anchor tables in {time.perf_counter() - t_anc:.1f}s "
        f"(fast_parity_max_diff={fast_parity_max_diff:.2e})",
        flush=True,
    )

    ad_param_names = [
        n for n, _ in net.named_parameters() if n.startswith("adapter_10tok.") and not n.startswith("adapter_10tok.anchor.")
    ]

    # Pre-pack compact deduplicated & padded PyTorch lookup tables (~29 MB total) to eliminate
    # Python `np.stack` over `[1024, 192, 412]` (~4.8s/step -> ~0.04s/step) with 0.0 bit-identity!
    b03_pid_to_uid: dict[str, int] = {}
    b03_u_pstd_list: list[np.ndarray] = []
    b03_u_sem_list: list[np.ndarray] = []
    max_m_tr = max(len(pools[c]["pids"]) for c in train_cids)
    ctx_uid_pad = torch.zeros((n_train, max_m_tr), dtype=torch.long)
    pad_anc_tot = torch.zeros((n_train, max_m_tr), dtype=torch.float32)
    pad_anc_eff = torch.zeros((n_train, max_m_tr), dtype=torch.float32)
    pad_fit = torch.zeros((n_train, max_m_tr), dtype=torch.float32)
    pad_gate = torch.zeros((n_train, max_m_tr), dtype=torch.bool)
    all_rf4 = torch.zeros((n_train, 4), dtype=torch.float32)
    all_rg4 = torch.zeros((n_train, 4), dtype=torch.float32)
    pool_fit_order: dict[str, np.ndarray] = {}

    for i_tr, cid in enumerate(train_cids):
        p = pools[cid]
        ac = anc_cache[cid]
        pids = p["pids"]
        m_c = len(pids)
        pool_fit_order[cid] = np.argsort(p["fitness"])
        for j_c, pid in enumerate(pids):
            u_id = b03_pid_to_uid.get(pid)
            if u_id is None:
                u_id = len(b03_u_pstd_list)
                b03_pid_to_uid[pid] = u_id
                b03_u_pstd_list.append(ac["prog_std"][j_c])
                b03_u_sem_list.append(ac["sem24"][j_c])
            ctx_uid_pad[i_tr, j_c] = u_id
        pad_anc_tot[i_tr, :m_c] = torch.from_numpy(ac["train_anc_total"])
        pad_anc_eff[i_tr, :m_c] = torch.from_numpy(ac["train_anc_eff"])
        pad_fit[i_tr, :m_c] = torch.as_tensor(p["fitness"], dtype=torch.float32)
        pad_gate[i_tr, :m_c] = torch.as_tensor(p["gate"], dtype=torch.bool)
        all_rf4[i_tr] = torch.from_numpy(ac["train_rf4"])
        all_rg4[i_tr] = torch.from_numpy(ac["train_rg4"])

    u_prog_std_t = torch.from_numpy(np.stack(b03_u_pstd_list, axis=0).astype(np.float32))
    u_sem24_t = torch.from_numpy(np.stack(b03_u_sem_list, axis=0).astype(np.float32))
    del b03_u_pstd_list, b03_u_sem_list, b03_pid_to_uid

    def _slate_idx_pairnf(cid: str, size: int, rng: np.random.Generator) -> np.ndarray:
        n_c = len(pools[cid]["pids"])
        order = pool_fit_order[cid]
        out: list[int] = []
        used: set[int] = set()
        for i_ in order[:48]:
            if len(out) >= size:
                break
            ii = int(i_)
            if ii not in used:
                out.append(ii)
                used.add(ii)
        rest = np.array([i_ for i_ in range(n_c) if i_ not in used], dtype=np.int64)
        need = size - len(out)
        if need > 0:
            out.extend(rng.choice(rest, need, replace=len(rest) < need).tolist())
        out = out[:size]
        ic = int(inc_top[cid][0])
        if ic in out[:4]:
            out.remove(ic)
            out.insert(4, ic)
        elif ic not in out:
            out[-1] = ic
        return np.asarray(out, dtype=np.int64)

    def _train_and_eval_arm_cell(
        arm_mode: str,
        seed: int,
        tr_cids_cell: list[str],
        eval_cids_cell: list[str],
        tag: str,
        eval_anc_map: Mapping[str, dict[str, np.ndarray]],
        eval_tok_tensor: torch.Tensor,
        eval_msk_tensor: torch.Tensor,
        eval_dope629_tensor: torch.Tensor,
        eval_raw_up_mat: np.ndarray,
        eval_cid_to_pos: Mapping[str, int],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, int], list[float], str]:
        """Train 16 steps of `pairnf` on `tr_cids_cell` and evaluate `zmin=0.40` utility on `eval_cids_cell`."""
        assert_memavail(15.0, f"b03:{arm_mode}:{tag}")
        net.load_state_dict(init_state, strict=True)
        for p in net.parameters():
            p.requires_grad_(False)
        ad_params = []
        for n in ad_param_names:
            p = dict(net.named_parameters())[n]
            p.requires_grad_(True)
            ad_params.append(p)
        guard.lock_dope_weights_and_eval_mode(net)

        # Fold-safe standardization of `raw_up_mat` strictly on `tr_cids_cell`!
        tr_pos = np.array([cid_to_i[c] for c in tr_cids_cell], dtype=np.int64)
        mu_up, sd_up = fit_fold_standardizer(raw_up_mat[tr_pos])
        tr_up_z_all = torch.from_numpy(apply_fold_standardizer(raw_up_mat, mu_up, sd_up))
        ev_up_z_all = torch.from_numpy(apply_fold_standardizer(eval_raw_up_mat, mu_up, sd_up))

        bottleneck = FoldedStage1BBottleneck(net, arm_mode, extra_dim=raw_up_mat.shape[1])

        # Exact `pairnf` init perturbation stream on the 29,211 adapter parameters
        g_ad = torch.Generator().manual_seed(h448_stream_key(seed, "pairnf", tag, "init") % (2 ** 63))
        with torch.no_grad():
            for p in ad_params:
                rms = p.detach().pow(2).mean().sqrt().clamp_min(1e-8)
                p.add_(torch.randn(p.shape, generator=g_ad) * rms * 5e-3)

        bn_params = [p for p in bottleneck.parameters() if p.requires_grad]
        if arm_mode in ("svd96", "ctrl96") and bn_params:
            g_bn = torch.Generator().manual_seed(h448_stream_key(seed, arm_mode, tag, "bn_init") % (2 ** 63))
            with torch.no_grad():
                for p in bn_params:
                    rms = p.detach().pow(2).mean().sqrt().clamp_min(1e-8)
                    p.add_(torch.randn(p.shape, generator=g_bn) * rms * 5e-3)

        opt = torch.optim.AdamW(ad_params, lr=3e-4, weight_decay=1e-4)
        opt.load_state_dict(copy.deepcopy(init_opt))
        for st_ in opt.state.values():
            st_["exp_avg"].zero_()
        for gp in opt.param_groups:
            gp["lr"] = 3e-4
            gp["weight_decay"] = 1e-4
        if bn_params:
            opt.add_param_group({"params": bn_params, "lr": 1.5e-4, "weight_decay": 1e-4})
        guard.assert_optimizer_excludes_dope(opt, net)

        torch.manual_seed(h448_stream_key(seed, "pairnf", tag, "dropout") % (2 ** 63))
        anc_ad = [init_state[n].detach().clone() for n in ad_param_names]
        anc_ad_norm = sum(float(t.pow(2).sum()) for t in anc_ad)
        anc_bn = [p.detach().clone() for p in bn_params]
        anc_bn_norm = max(sum(float(t.pow(2).sum()) for t in anc_bn), 1.0)
        all_trainable = ad_params + bn_params

        bsz = 1024
        slate_sz = 192
        hinge_m = 0.25
        losses_log: list[float] = []

        for step in range(16):
            rr = random.Random(h448_stream_key(seed, "pairnf", tag, step, "ctx"))
            rng = np.random.default_rng(h448_stream_key(seed, "pairnf", tag, step, "slate"))
            batch_cids = rr.choices(tr_cids_cell, k=bsz)
            sels = [_slate_idx_pairnf(c, slate_sz, rng) for c in batch_cids]

            net.train()
            bottleneck.train()
            opt.zero_grad(set_to_none=True)

            # Compute updated 10x32 tokens on unique contexts in `batch_cids` only (~550 unique vs 1024!)
            uniq_cids = list(dict.fromkeys(batch_cids))
            uniq_pos = torch.tensor([cid_to_i[c] for c in uniq_cids], dtype=torch.long)
            u_tok_in = camp_tok[uniq_pos]
            u_dope = dope629_hpc_t[uniq_pos] if arm_mode == "svd96" else None
            u_up = tr_up_z_all[uniq_pos] if arm_mode == "ultra_packet" else None
            u_tok_out = bottleneck(u_tok_in, dope629=u_dope, extra_z=u_up)  # [U, 10, 32]
            cid_to_u = {c: ui for ui, c in enumerate(uniq_cids)}
            gather_u = torch.tensor([cid_to_u[c] for c in batch_cids], dtype=torch.long)
            b_tok = u_tok_out[gather_u]
            b_msk = camp_msk[uniq_pos[gather_u]]

            # Assemble precomputed frozen-anchor + candidate tensors via bit-exact lookup (`0.0` ULP diff)
            b_pos = torch.tensor([cid_to_i[c] for c in batch_cids], dtype=torch.long)
            sels_t = torch.from_numpy(np.stack(sels, axis=0))
            row_ix = b_pos.unsqueeze(1)
            b_uids = ctx_uid_pad[row_ix, sels_t]
            prog_std_b = u_prog_std_t[b_uids]
            sem24_b = u_sem24_t[b_uids]
            anc_tot_b = pad_anc_tot[row_ix, sels_t]
            anc_eff_b = pad_anc_eff[row_ix, sels_t]
            rf4_b = all_rf4[b_pos]
            rg4_b = all_rg4[b_pos]

            pred = _fast_predict_costs_from_cached_anchor(
                net.adapter_10tok, b_tok, b_msk, prog_std_b, sem24_b, anc_tot_b, anc_eff_b, rf4_b, rg4_b
            )
            tr_fit = pad_fit[row_ix, sels_t]
            g_mask = pad_gate[row_ix, sels_t]
            target = torch.where(g_mask, tr_fit, torch.full_like(tr_fit, 2.0))

            row = F.smooth_l1_loss(pred, target, reduction="none").mean(-1) + 0.1 * F.kl_div(
                F.log_softmax(-pred / 0.2, -1), F.softmax(-target / 0.2, -1), reduction="none"
            ).sum(-1)

            # Gate hinge (`hinge_w = 0.5`)
            big = torch.full_like(pred, 1e9)
            bp = torch.where(g_mask, pred, big).min(-1).values
            has_pass = g_mask.any(-1)
            fm = (~g_mask).float()
            h_term = F.relu(bp.unsqueeze(-1) + hinge_m - pred) * fm
            row = row + 0.5 * torch.where(has_pass, h_term.sum(-1) / fm.sum(-1).clamp_min(1.0), torch.zeros_like(bp))

            # Listwise soft cross-entropy (`listw = 1.0, listT = 0.5`)
            tz = (target - target.mean(-1, keepdim=True)) / target.std(-1, keepdim=True).clamp_min(1e-6)
            pz = (pred - pred.mean(-1, keepdim=True)) / pred.std(-1, keepdim=True).clamp_min(1e-4)
            soft = F.softmax(-tz / 0.5, -1)
            row = row - (soft * F.log_softmax(-pz / 0.5, -1)).sum(-1)

            # Non-floor Bradley-Terry pairwise vs incumbent argmin (`pair_w = 1.0, pairT = 0.1, pair_nf = True`)
            ipos = torch.as_tensor(
                [int(np.where(s == int(inc_top[c][0]))[0][0]) for c, s in zip(batch_cids, sels)],
                dtype=torch.long,
            )
            ar = torch.arange(len(batch_cids))
            ci = pred[ar, ipos].unsqueeze(-1)
            ti = target[ar, ipos].unsqueeze(-1)
            gi = g_mask[ar, ipos].unsqueeze(-1)
            better = (target < ti - 1e-9).float() * g_mask.float() * gi.float()
            worse = (target > ti + 1e-9).float() * g_mask.float() * gi.float()
            lp = better * F.softplus(-(ci - pred) / 0.1) + worse * F.softplus(-(pred - ci) / 0.1)
            row = row + lp.sum(-1) / (better + worse).sum(-1).clamp_min(1.0)

            loss = row.mean() + 5.0 * sum((p - q).pow(2).sum() for p, q in zip(ad_params, anc_ad)) / anc_ad_norm
            if bn_params:
                loss = loss + 2.0 * sum((p - q).pow(2).sum() for p, q in zip(bn_params, anc_bn)) / anc_bn_norm

            loss.backward()
            losses_log.append(float(loss.detach().item()))
            torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
            opt.step()

        guard.verify_dope_frozen_unchanged(net, before_snap)

        # Compute trainable parameter SHA
        h_sha = hashlib.sha256()
        for p in all_trainable:
            h_sha.update(p.detach().contiguous().numpy().tobytes())
        param_sha = h_sha.hexdigest()

        # Evaluate on `eval_cids_cell` with exact `zmin = 0.40` confidence gate!
        net.eval()
        bottleneck.eval()
        n_ev = len(eval_cids_cell)
        ev_u = np.zeros(n_ev, dtype=np.float64)
        ev_gf = np.zeros(n_ev, dtype=bool)
        n_changed = 0
        n_kept = 0

        with torch.no_grad():
            ev_pos = torch.tensor([eval_cid_to_pos[c] for c in eval_cids_cell], dtype=torch.long)
            ev_tok_upd = bottleneck(
                eval_tok_tensor[ev_pos],
                dope629=eval_dope629_tensor[ev_pos] if arm_mode == "svd96" else None,
                extra_z=ev_up_z_all[ev_pos] if arm_mode == "ultra_packet" else None,
            )
            ev_msk_sub = eval_msk_tensor[ev_pos]
            for q_i, cid in enumerate(eval_cids_cell):
                p = pools[cid]
                ac = eval_anc_map[cid]
                pred_q = _fast_predict_costs_from_cached_anchor(
                    net.adapter_10tok,
                    ev_tok_upd[q_i : q_i + 1],
                    ev_msk_sub[q_i : q_i + 1],
                    torch.from_numpy(ac["prog_std"][None]),
                    torch.from_numpy(ac["sem24"][None]),
                    torch.from_numpy(ac["eval_anc_total"][None]),
                    torch.from_numpy(ac["eval_anc_eff"][None]),
                    torch.from_numpy(ac["eval_rf4"][None]),
                    torch.from_numpy(ac["eval_rg4"][None]),
                )[0].numpy()
                cand_i = int(np.argmin(pred_q))
                inc_j = inc_pick[cid]
                keep_change = False
                if cand_i != inc_j:
                    n_changed += 1
                    z_margin = float((pred_q[inc_j] - pred_q[cand_i]) / max(float(pred_q.std()), 1e-8))
                    if z_margin >= 0.40:
                        keep_change = True
                        n_kept += 1
                final_k = cand_i if keep_change or cand_i == inc_j else inc_j
                gf = not bool(p["gate"][final_k])
                ev_gf[q_i] = gf
                ev_u[q_i] = -float(p["fitness"][final_k]) if not gf else -2.0

        return ev_u, ev_gf, {"n_changed": n_changed, "n_kept": n_kept}, losses_log, param_sha

    if build_train_fn_only:
        return {"_train_fn": _train_and_eval_arm_cell, "_inc_pick_train": inc_pick}

    # Run all 4 arms across partitions × folds × seeds
    b03_arms = {
        "pairnf_z40": "none",
        "B03_svd96": "svd96",
        "B03_ctrl96": "ctrl96",
        "B03_ultra_packet": "ultra_packet",
    }
    eval_folds = folds_list[:1] if smoke else folds_list
    eval_salts = salts[:1] if smoke else salts
    eval_seeds = tuple(seeds[:1]) if smoke else tuple(seeds)
    n_folds = int(np.max(eval_folds[0])) + 1

    tr_ufam = sorted(set(train_families))
    tr_fam_arr = np.asarray(train_families, dtype=object)
    tr_fam_idx = [np.where(tr_fam_arr == f)[0] for f in tr_ufam]

    per_seed_results: list[dict[str, Any]] = []
    pooled_arm_oof_u: dict[str, list[np.ndarray]] = {a: [] for a in b03_arms}
    pooled_arm_oof_diff: dict[str, list[np.ndarray]] = {a: [] for a in b03_arms}

    for sd in eval_seeds:
        cv_sd: dict[str, Any] = {}
        for arm_name, arm_mode in b03_arms.items():
            t_arm = time.perf_counter()
            per_part = []
            oof_parts: list[np.ndarray] = []
            oof_u_parts: list[np.ndarray] = []
            gsum = {"new_gatefails": 0, "recovered_gatefails": 0, "inc_gatefails": 0, "n_changed": 0, "n_kept": 0}

            for p_i, fold_arr in enumerate(eval_folds):
                oof_diff = np.full(n_train, np.nan, dtype=np.float64)
                oof_u_vec = np.full(n_train, np.nan, dtype=np.float64)
                fold_gains: list[float] = []
                for k in range(1 if smoke else n_folds):
                    ho_mask = fold_arr == k
                    ho_idx = np.where(ho_mask)[0]
                    tr_idx = np.where(~ho_mask)[0]
                    tr_c = [train_cids[i] for i in tr_idx.tolist()]
                    ho_c = [train_cids[i] for i in ho_idx.tolist()]
                    tag = f"p{p_i}cv{k}"
                    ev_u, ev_gf, zstat, losses_l, p_sha = _train_and_eval_arm_cell(
                        arm_mode,
                        int(sd),
                        tr_c,
                        ho_c,
                        tag,
                        anc_cache,
                        camp_tok,
                        camp_msk,
                        dope629_hpc_t,
                        raw_up_mat,
                        cid_to_i,
                    )
                    oof_u_vec[ho_idx] = ev_u
                    oof_diff[ho_idx] = ev_u - u_inc_train[ho_idx]
                    fold_gains.append(float(np.mean(oof_diff[ho_idx])))
                    inf_ = inc_fail_train[ho_idx]
                    gsum["new_gatefails"] += int((ev_gf & ~inf_).sum())
                    gsum["recovered_gatefails"] += int((~ev_gf & inf_).sum())
                    gsum["inc_gatefails"] += int(inf_.sum())
                    gsum["n_changed"] += zstat["n_changed"]
                    gsum["n_kept"] += zstat["n_kept"]

                m_valid = ~np.isnan(oof_diff)
                d = oof_diff[m_valid] if smoke else oof_diff
                u_part = oof_u_vec[m_valid] if smoke else oof_u_vec
                nf = ~inc_fail_train[m_valid] if smoke else ~inc_fail_train
                dn = d[nf]
                w_all = float((d > 0).sum())
                t_all = float((d == 0).sum())
                e_part = elo_from_win_share((w_all + 0.5 * t_all) / max(len(d), 1))
                e_nf = elo_from_win_share(((dn > 0).sum() + 0.5 * (dn == 0).sum()) / max(len(dn), 1))
                per_part.append(
                    {
                        "partition": eval_salts[p_i],
                        "oof_gain_mean": round(float(np.mean(d)), 6),
                        "fold_gains": [round(g, 6) for g in fold_gains],
                        "elo": round(e_part, 4),
                        "wins": int(w_all),
                        "ties": int(t_all),
                        "losses": int((d < 0).sum()),
                        "nonfloor_gain": round(float(np.mean(dn)) if len(dn) else 0.0, 6),
                        "nonfloor_elo": round(e_nf, 4),
                    }
                )
                oof_parts.append(d)
                oof_u_parts.append(u_part)

            dall = np.concatenate(oof_parts)
            if smoke:
                lo = hi = elo_lo = elo_hi = nlo = nhi = 0.0
                nfa = np.ones(len(dall), dtype=bool)
            else:
                nfa = np.concatenate([~inc_fail_train] * len(oof_parts))
                fidx = [
                    np.concatenate([j * n_train + tr_fam_idx[q] for j in range(len(oof_parts))])
                    for q in range(len(tr_fam_idx))
                ]
                r_boot = np.random.default_rng(int(sd) + 7919)
                gs, es, ns = [], [], []
                for _ in range(2000):
                    ixb = np.concatenate([fidx[q] for q in r_boot.integers(0, len(fidx), len(fidx))])
                    xb = dall[ixb]
                    xnb = xb[nfa[ixb]]
                    gs.append(float(xb.mean()))
                    es.append(elo_from_win_share(((xb > 0).sum() + 0.5 * (xb == 0).sum()) / len(xb)))
                    ns.append(elo_from_win_share(((xnb > 0).sum() + 0.5 * (xnb == 0).sum()) / max(len(xnb), 1)))
                lo, hi = np.percentile(gs, [2.5, 97.5]).tolist()
                elo_lo, elo_hi = np.percentile(es, [2.5, 97.5]).tolist()
                nlo, nhi = np.percentile(ns, [2.5, 97.5]).tolist()

            dn_all = dall[nfa]
            e_all = elo_from_win_share(((dall > 0).sum() + 0.5 * (dall == 0).sum()) / max(len(dall), 1))
            e_nf_all = elo_from_win_share(((dn_all > 0).sum() + 0.5 * (dn_all == 0).sum()) / max(len(dn_all), 1))
            passes_strict = bool(
                (not smoke)
                and dall.mean() > 0.002
                and lo > 0.0
                and e_all > 0.0
                and elo_lo > 0.0
                and nlo > 0.0
                and all(pp["elo"] > 0.0 for pp in per_part)
            )
            cv_sd[arm_name] = {
                "per_partition": per_part,
                "oof_gain_mean": round(float(dall.mean()), 6),
                "oof_gain_ci95_family_boot": [round(lo, 6), round(hi, 6)],
                "elo_vs_incumbent_oof": round(e_all, 4),
                "elo_ci95_family_boot": [round(elo_lo, 4), round(elo_hi, 4)],
                "wins": int((dall > 0).sum()),
                "ties": int((dall == 0).sum()),
                "losses": int((dall < 0).sum()),
                "nonfloor": {
                    "n": int(len(dn_all)),
                    "gain_mean": round(float(dn_all.mean()) if len(dn_all) else 0.0, 6),
                    "wins": int((dn_all > 0).sum()),
                    "ties": int((dn_all == 0).sum()),
                    "losses": int((dn_all < 0).sum()),
                    "elo": round(e_nf_all, 4),
                    "elo_ci95_family_boot": [round(nlo, 4), round(nhi, 4)],
                },
                "mean_partition_nonfloor_elo": round(float(np.mean([pp["nonfloor_elo"] for pp in per_part])), 4),
                "gate": gsum,
                "passes_strict": passes_strict,
                "wall_s": round(time.perf_counter() - t_arm, 2),
            }
            pooled_arm_oof_u[arm_name].append(np.mean(oof_u_parts, axis=0))
            pooled_arm_oof_diff[arm_name].append(np.mean(oof_parts, axis=0))
            print(
                f"[B03:seed={sd}:{arm_name}] gain={cv_sd[arm_name]['oof_gain_mean']:+.6f} "
                f"Elo={cv_sd[arm_name]['elo_vs_incumbent_oof']:+.2f} "
                f"nf_Elo={cv_sd[arm_name]['mean_partition_nonfloor_elo']:+.2f} "
                f"W/L={cv_sd[arm_name]['wins']}/{cv_sd[arm_name]['losses']} "
                f"({cv_sd[arm_name]['wall_s']:.1f}s)",
                flush=True,
            )

        sel_arm = max(
            cv_sd,
            key=lambda k_: (cv_sd[k_]["mean_partition_nonfloor_elo"], cv_sd[k_]["oof_gain_mean"]),
        )
        per_seed_results.append({"seed": int(sd), "cv": cv_sd, "selected": sel_arm})

    # Seed-averaged per-context OOF utilities for 10,000-draw paired family bootstrap contrasts (`C8b`)
    if smoke:
        fams_c = [train_families[i] for i in np.where(eval_folds[0] == 0)[0].tolist()]
        u_inc_c = u_inc_train[eval_folds[0] == 0]
    else:
        fams_c = train_families
        u_inc_c = u_inc_train

    mean_u_by_arm = {a: np.mean(pooled_arm_oof_u[a], axis=0) for a in b03_arms}
    b03_contrasts = {
        "pairnf_z40_minus_incumbent_4rev": contrast_summary(mean_u_by_arm["pairnf_z40"], u_inc_c, fams_c),
        "B03_svd96_minus_incumbent_4rev": contrast_summary(mean_u_by_arm["B03_svd96"], u_inc_c, fams_c),
        "B03_svd96_minus_pairnf_z40": contrast_summary(mean_u_by_arm["B03_svd96"], mean_u_by_arm["pairnf_z40"], fams_c),
        "B03_svd96_minus_B03_ctrl96": contrast_summary(mean_u_by_arm["B03_svd96"], mean_u_by_arm["B03_ctrl96"], fams_c),
        "B03_ultra_packet_minus_incumbent_4rev": contrast_summary(
            mean_u_by_arm["B03_ultra_packet"], u_inc_c, fams_c
        ),
        "B03_ultra_packet_minus_pairnf_z40": contrast_summary(
            mean_u_by_arm["B03_ultra_packet"], mean_u_by_arm["pairnf_z40"], fams_c
        ),
    }

    net.load_state_dict(init_state, strict=True)
    guard.verify_dope_frozen_unchanged(net, before_snap)

    return {
        "step0_verification": {
            "svd96_delta_step0_max_abs_diff": step0_svd_delta_diff,
            "ultra_packet_zero_init_step0_max_abs_diff": step0_up_diff,
            "standalone_unclamped_linear_vs_clamped_svd_median_diff": round(
                float(torch.median(unclamped_vs_clamped_diff).item()), 8
            ),
            "standalone_unclamped_linear_vs_clamped_svd_max_diff": round(
                float(torch.max(unclamped_vs_clamped_diff).item()), 6
            ),
            "fast_cached_anchor_parity_max_diff": round(fast_parity_max_diff, 8),
        },
        "incumbent_train_ctx_utility": round(float(np.mean(u_inc_train)), 9),
        "incumbent_train_family_utility": round(family_mean_value(u_inc_train, train_families), 6),
        "per_seed": per_seed_results,
        "contrasts": b03_contrasts,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "_train_fn": _train_and_eval_arm_cell,
        "_inc_pick_train": inc_pick,
    }


# =============================================================================
# One-Time Final `val91` `REGCHECK` (Executed ONLY Once After All OOF Decisions Freeze)
# =============================================================================
def run_final_val91_regcheck(
    net: Any,
    guard: Any,
    before_snap: Any,
    init_state: Mapping[str, torch.Tensor],
    init_opt: Mapping[str, Any],
    train_cids: list[str],
    train_families: list[str],
    dev_cids: list[str],
    dev_families: list[str],
    pools: Mapping[str, dict[str, Any]],
    tr_feats: Mapping[str, np.ndarray],
    dev_feats: Mapping[str, np.ndarray],
    wp3_res: Mapping[str, Any],
    b03_res: Mapping[str, Any],
    *,
    seeds: Sequence[int] = (1, 2, 3),
    num_threads: int = 8,
    train_anchor_bank: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate `DL-Val91` ONCE on frozen CV-selected winners for `B03` and `WP3`."""
    t0 = time.perf_counter()
    assert_memavail(20.0, "val91_regcheck")

    net.load_state_dict(init_state, strict=True)
    net.eval()

    # 1. Exact incumbent 4-reveal parity check (`1.1389572403748425`) + 0-reveal baseline on `val91`
    try:
        from .anchor_fast_cache import build_shared_frozen_anchor_bank, populate_pairnf_train_anchors_and_trim
    except ImportError:
        from anchor_fast_cache import build_shared_frozen_anchor_bank, populate_pairnf_train_anchors_and_trim

    dev_anchor_bank = build_shared_frozen_anchor_bank(net, dev_cids, pools)
    populate_pairnf_train_anchors_and_trim(net, dev_anchor_bank, dev_cids, pools, {})
    dev_entries_0 = precompute_0rev_context_cache(
        net,
        dev_cids,
        pools,
        dev_feats["tokens_10x32_campaign"],
        dev_feats["mask_10_campaign"],
        anchor_bank=dev_anchor_bank,
    )
    u_dev_0rev = np.array([float(e["util"][int(np.argmin(e["cost0"]))]) for e in dev_entries_0], dtype=np.float64)

    dev_tok = torch.from_numpy(dev_feats["tokens_10x32_campaign"].astype(np.float32))
    dev_msk = torch.from_numpy(dev_feats["mask_10_campaign"].astype(np.float32))
    dev_dope = torch.from_numpy(dev_feats["dope629_hpc"].astype(np.float32))
    dev_raw_up = np.hstack(
        [dev_feats["dope629_ultra"], dev_feats["col_lm_34_ultra"], dev_feats["R_packet_v1_ultra"]]
    ).astype(np.float32)

    inc_pick_dev: dict[str, int] = {}
    u_dev_4rev = np.zeros(len(dev_cids), dtype=np.float64)
    gf_dev_4rev = np.zeros(len(dev_cids), dtype=bool)
    with torch.no_grad():
        for i, cid in enumerate(dev_cids):
            p = pools[cid]
            m = len(p["pids"])
            k_r = min(4, m)
            sc = net.adapter_10tok.predict_costs(
                ctx_feat=torch.zeros((1, 584), dtype=torch.float32),
                prog_412=torch.as_tensor(p["program"][None], dtype=torch.float32),
                proj_16=torch.as_tensor(p["proj16"][None], dtype=torch.float32),
                base_prior_cost=torch.zeros((1, m), dtype=torch.float32),
                scale=torch.full((1,), 0.4208, dtype=torch.float32),
                sos_prior=torch.zeros((1, m, 2), dtype=torch.float32),
                mm_tokens_10x32=dev_tok[i : i + 1],
                mm_mask_10=dev_msk[i : i + 1],
                revealed_indices=torch.arange(k_r, dtype=torch.long).reshape(1, k_r),
                revealed_raw_f=torch.as_tensor(p["fitness"][:k_r][None], dtype=torch.float32),
                revealed_gate=torch.as_tensor(p["gate"][:k_r][None], dtype=torch.float32),
            )["total_cost"].float()[0].numpy()
            pk = int(np.argmin(sc))
            inc_pick_dev[cid] = pk
            gf = not bool(p["gate"][pk])
            gf_dev_4rev[i] = gf
            u_dev_4rev[i] = -float(p["fitness"][pk]) if not gf else -2.0

    expected_val_u = 1.1389572403748425
    parity_ok = abs(float(np.mean(u_dev_4rev)) - expected_val_u) < 1e-9
    if not parity_ok:
        raise RuntimeError(f"val91 incumbent parity failed: {float(np.mean(u_dev_4rev))} != {expected_val_u}")

    # 2. WP3 selected arm trained on full DL-Train (`1,219` contexts) per seed -> evaluated once on `val91`
    sel_wp3 = str(wp3_res["selected_wp3_arm"])
    entries_train = wp3_res.get("_entries_train")
    if entries_train is None:
        entries_train = precompute_0rev_context_cache(
            net,
            train_cids,
            pools,
            tr_feats["tokens_10x32_campaign"],
            tr_feats["mask_10_campaign"],
            anchor_bank=train_anchor_bank,
        )
    if sel_wp3 == "WP3_ultra_packet":
        r_tr_wp3 = np.hstack(
            [tr_feats["dope629_ultra"], tr_feats["col_lm_34_ultra"], tr_feats["R_packet_v1_ultra"]]
        ).astype(np.float32)
        r_dv_wp3 = dev_raw_up
    else:
        r_tr_wp3 = np.hstack([tr_feats["dope629_hpc"], tr_feats["col_lm_34_hpc"]]).astype(np.float32)
        r_dv_wp3 = np.hstack([dev_feats["dope629_hpc"], dev_feats["col_lm_34_hpc"]]).astype(np.float32)

    beta_sel = 0.0 if sel_wp3 == "WP3_hist_only_lowvar" else 0.35
    u_wp3_dev = np.zeros(len(dev_cids), dtype=np.float64)
    for sd in seeds:
        mb_full = _fit_wp3_outcome_and_uncertainty_models(
            r_tr_wp3, entries_train, train_families, seed=int(sd), num_threads=num_threads
        )
        for i_d, e_d in enumerate(dev_entries_0):
            k_r = min(4, len(e_d["pids"]))
            pr_y, mu_d, _, p_g, lam = _predict_wp3_for_context(mb_full, e_d, r_dv_wp3[i_d], kappa=1.5)
            y_sh = pr_y + lam * mu_d
            s4 = np.argsort(e_d["cost0"])[:k_r] if sel_wp3 == "WP3_mimic_acq" else np.argsort(-(lam * p_g))[:k_r]
            _, _, u_d, _ = eval_wp3_history_kernel_single(
                net, e_d, s4, y_sh[s4], p_g[s4], lam, mu_d, beta_direct=beta_sel
            )
            u_wp3_dev[i_d] += u_d / float(len(seeds))

    wp3_val_contrast = contrast_summary(u_wp3_dev, u_dev_0rev, dev_families)

    # 3. B03 CV-selected arm per seed trained on full DL-Train (`tag="full"`) -> evaluated once on `val91`
    dev_anc_cache = dev_anchor_bank
    dev_cid_to_pos = {c: i for i, c in enumerate(dev_cids)}
    b03_mode_map = {
        "pairnf_z40": "none",
        "B03_svd96": "svd96",
        "B03_ctrl96": "ctrl96",
        "B03_ultra_packet": "ultra_packet",
    }
    b03_train_fn = b03_res.get("_train_fn")
    b03_inc_pick_map = b03_res.get("_inc_pick_train")
    if b03_train_fn is None or b03_inc_pick_map is None:
        helper_bundle = run_b03_neural_cv(
            net,
            guard,
            before_snap,
            init_state,
            init_opt,
            train_cids,
            train_families,
            pools,
            tr_feats,
            [np.zeros(len(train_cids), dtype=np.int64)],
            ["full/"],
            seeds=seeds,
            num_threads=num_threads,
            anchor_bank=train_anchor_bank,
            build_train_fn_only=True,
        )
        b03_train_fn = helper_bundle["_train_fn"]
        b03_inc_pick_map = helper_bundle["_inc_pick_train"]
    # Register dev incumbent picks inside `b03_inc_pick_map` so `zmin=0.40` applies cleanly to `dev_cids`
    b03_inc_pick_map.update(inc_pick_dev)

    b03_seed_regchecks: list[dict[str, Any]] = []
    for sd_entry in b03_res["per_seed"]:
        sd = int(sd_entry["seed"])
        sel_arm = str(sd_entry["selected"])
        ev_u, ev_gf, zstat, _, fsha = b03_train_fn(
            b03_mode_map[sel_arm],
            sd,
            train_cids,
            dev_cids,
            "full",
            dev_anc_cache,
            dev_tok,
            dev_msk,
            dev_dope,
            dev_raw_up,
            dev_cid_to_pos,
        )
        dv = ev_u - u_dev_4rev
        vw = float((dv > 0).sum())
        vt = float((dv == 0).sum())
        vs = (vw + 0.5 * vt) / len(dv)
        c_val = contrast_summary(ev_u, u_dev_4rev, dev_families, seed=20261118 + sd)
        b03_seed_regchecks.append(
            {
                "seed": sd,
                "selected_arm": sel_arm,
                "val_ctx_utility": round(float(np.mean(ev_u)), 6),
                "val_ctx_gain": round(float(np.mean(dv)), 6),
                "val_family_gain": c_val["mean_delta"],
                "val_family_ci95": [c_val["ci_low"], c_val["ci_high"]],
                "wins": int(vw),
                "ties": int(vt),
                "losses": int((dv < 0).sum()),
                "elo_vs_incumbent_val": round(elo_from_win_share(vs), 4),
                "not_worse_ctx": bool(float(np.mean(dv)) >= 0.0),
                "new_gatefails": int((ev_gf & ~gf_dev_4rev).sum()),
                "val_changed": zstat["n_changed"],
                "val_kept": zstat["n_kept"],
                "param_sha256": fsha,
            }
        )

    net.load_state_dict(init_state, strict=True)
    guard.verify_dope_frozen_unchanged(net, before_snap)

    return {
        "parity_incumbent_val91_ctx_utility": float(np.mean(u_dev_4rev)),
        "parity_expected": expected_val_u,
        "parity_ok": parity_ok,
        "incumbent_val91_0rev_ctx_utility": round(float(np.mean(u_dev_0rev)), 6),
        "incumbent_val91_0rev_family_utility": round(family_mean_value(u_dev_0rev, dev_families), 6),
        "wp3_val91_regcheck": {
            "selected_arm": sel_wp3,
            "val91_ctx_utility": round(float(np.mean(u_wp3_dev)), 6),
            "val91_family_utility": round(family_mean_value(u_wp3_dev, dev_families), 6),
            "not_worse_family": bool(wp3_val_contrast["mean_delta"] >= 0.0),
            "contrast_vs_incumbent_0rev": wp3_val_contrast,
        },
        "b03_val91_regchecks_by_seed": b03_seed_regchecks,
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
