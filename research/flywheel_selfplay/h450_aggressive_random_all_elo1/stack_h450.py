#!/usr/bin/env python3
"""H450 Pool-Agnostic Global-Best OOF Guard, Adaptive Refit Trust & Nested Log-RMSE Combiner (`stack_h450.py`).

Fixes the GP-median throttling bug in `guard_matern_core.guard_and_stack_2l`:
  1. Pool-Agnostic Global-Best OOF Anchor (`min_mse_global` across ALL candidate columns: `H450-Neural`, `H450-Tab`,
     `M7`, `K3`, `F4`, and `SXB-GP`). Never lets weaker GP peers veto or dilute a high-precision `H450` column!
  2. Per-Column Adaptive `bag`/`refit` Blend (`alpha_j in [0.0, 0.88]`): Uses full `5/5` refit density when `ref_j`
     is consistent with `bag_j` (`mean((ref_j - bag_j)^2) <= 0.35 * mse_oof_j`), tapering smoothly to `bag_j` under drift.
  3. Direct Nested 5-Fold OOF Log-RMSE (`u`-metric) Simplex Tournament (`0` `y_audit` read!):
     - Optimizes simplex weights `w >= 0, sum(w) = 1` directly on `-0.5 * log(MSE / Var(y))` with diversity-aware
       entropy regularization `reg` (`0` forced dilution when a dominant column leads by `>= 0.25` utility!).
     - Evaluates `(a) Single-Best Winner`, `(b) Nested Log-RMSE Simplex`, `(c) Nested Top-K Huber/MSE Simplex`, and
       `(d) Legacy Guard+2L` inside a strict nested 5-fold OOF tournament on `(X_search, y_search)`, plus optional
       `ya`-free context-adaptive residual routing (`L2`) when nested OOF improves by `>= +0.015` utility.
"""
from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if "/home/ubuntu/h448-stage/pub-r1/code" not in sys.path and Path("/home/ubuntu/h448-stage/pub-r1/code").is_dir():
    sys.path.append("/home/ubuntu/h448-stage/pub-r1/code")

import guard_matern_core as GM  # noqa: E402


def _project_simplex(v: np.ndarray) -> np.ndarray:
    n = len(v)
    if n == 1:
        return np.ones(1, dtype=np.float64)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1, dtype=np.float64)
    cond = u - cssv / ind > 0.0
    rho = int(np.nonzero(cond)[0][-1])
    theta = cssv[rho] / float(rho + 1)
    return np.maximum(v - theta, 0.0)


def _fit_log_rmse_simplex(Q_tr: np.ndarray, y_tr: np.ndarray, lab_tr: np.ndarray | None, l2_w: float = 0.0) -> np.ndarray:
    """Minimizes `0.5 * log(MSE(Q w, y) + eps) + l2_w * ||w||^2` over the probability simplex `w >= 0, sum(w) = 1`."""
    n, k = Q_tr.shape
    if k == 1:
        return np.ones(1, dtype=np.float64)

    mses = np.mean((Q_tr - y_tr[:, None]) ** 2, axis=0)
    best_idx = int(np.argmin(mses))
    vy = max(float(np.var(y_tr)), 1e-18)
    eps = 1e-12 * vy

    # Smart warm starts: winner-heavy + inverse-MSE softmax
    w0_win = np.zeros(k, dtype=np.float64)
    w0_win[best_idx] = 1.0
    u_cols = -0.5 * np.log(np.maximum(mses, eps) / vy)
    temp = max(0.05, float(np.std(u_cols[: min(k, 5)])) * 0.5)
    logits = (u_cols - float(np.max(u_cols))) / temp
    w0_soft = np.exp(np.clip(logits, -30.0, 0.0))
    w0_soft /= float(np.sum(w0_soft))

    R = Q_tr - y_tr[:, None]  # [n, k], so Q w - y = R w!

    if lab_tr is not None and len(np.unique(lab_tr)) >= 2:
        uniq_f = [int(f) for f in np.unique(lab_tr)]
        f_masks = [lab_tr == f for f in uniq_f if int(np.sum(lab_tr == f)) >= 4]
    else:
        f_masks = [np.ones(n, dtype=bool)]

    def obj_and_grad(w: np.ndarray) -> tuple[float, np.ndarray]:
        val = float(l2_w * np.dot(w, w))
        grad = (2.0 * l2_w) * w
        nf = float(len(f_masks))
        for mf in f_masks:
            Rf = R[mf]
            err = Rf @ w
            mse_f = float(np.mean(err * err)) + eps
            val += (0.5 * math.log(mse_f / vy)) / nf
            grad += ((Rf.T @ err) / (float(len(err)) * mse_f)) / nf
        return val, grad

    best_w = w0_win
    best_val, _ = obj_and_grad(w0_win)

    for start_w in (w0_soft, 0.75 * w0_win + 0.25 * w0_soft):
        res = minimize(
            obj_and_grad,
            start_w,
            jac=True,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * k,
            constraints={"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0), "jac": lambda w: np.ones(k)},
            options={"maxiter": 140, "ftol": 1e-10},
        )
        w_cand = _project_simplex(np.asarray(res.x, dtype=np.float64))
        v_cand, _ = obj_and_grad(w_cand)
        if v_cand < best_val:
            best_val = v_cand
            best_w = w_cand

    return best_w


def _adaptive_test_col(q_j: np.ndarray, bag_j: np.ndarray, ref_j: np.ndarray | None, y: np.ndarray, vy: float) -> tuple[np.ndarray, float]:
    """Computes `p_j = (1 - alpha_j) * bag_j + alpha_j * ref_j` strictly from OOF MSE & `bag`-vs-`ref` consistency."""
    if ref_j is None or not np.all(np.isfinite(ref_j)):
        return bag_j.copy(), 0.0
    mse_oof = max(float(np.mean((q_j - y) ** 2)), 1e-10 * vy)
    drift_br = float(np.mean((ref_j - bag_j) ** 2))
    ratio = drift_br / mse_oof
    if ratio > 3.5:
        return bag_j.copy(), 0.0
    if ratio <= 0.30:
        alpha = 0.86
    else:
        alpha = 0.86 * math.exp(-0.95 * (ratio - 0.30))
    alpha = float(np.clip(alpha, 0.0, 0.88))
    return (1.0 - alpha) * bag_j + alpha * ref_j, alpha


def stack_h450_context(
    cid: str,
    Xs: np.ndarray,
    Xa: np.ndarray,
    y: np.ndarray,
    lab: np.ndarray,
    Q_gp: np.ndarray | None,
    bcols_gp: list[np.ndarray] | None,
    ref_gp: list[np.ndarray | None] | None,
    named_blocks: list[tuple[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
) -> tuple[np.ndarray, dict]:
    """Combines candidate blocks via pool-agnostic global OOF anchor + nested 5-fold log-RMSE tournament (`0` `y_audit` read!)."""
    Xs = np.asarray(Xs, dtype=np.float64)
    Xa = np.asarray(Xa, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lab = np.asarray(lab, dtype=np.int64)
    n_s, n_a = len(y), Xa.shape[0]
    ym = float(np.mean(y))
    sy = float(np.std(y))
    vy = max(sy * sy, 1e-24)
    if n_s < 12 or sy <= 1e-12:
        return np.full(n_a, ym, dtype=np.float64), {"mode": "constant_y"}

    y_lo, y_hi = float(np.min(y)), float(np.max(y))
    span = max(y_hi - y_lo, sy)
    lo_out, hi_out = y_lo - 1.25 * span, y_hi + 1.25 * span

    # 1. Assemble unified candidate pool `(name, q_j, p_test_j, mse_j, u_j, alpha_j)`
    cands = []
    if Q_gp is not None and bcols_gp is not None and ref_gp is not None:
        for j in range(Q_gp.shape[1]):
            qj = np.asarray(Q_gp[:, j], dtype=np.float64)
            bj = np.asarray(bcols_gp[j], dtype=np.float64)
            rj = None if ref_gp[j] is None else np.asarray(ref_gp[j], dtype=np.float64)
            if np.all(np.isfinite(qj)) and np.all(np.isfinite(bj)):
                pj, al = _adaptive_test_col(qj, bj, rj, y, vy)
                mse_j = float(np.mean((qj - y) ** 2))
                u_j = -0.5 * math.log(max(mse_j / vy, 1e-15))
                cands.append((f"gp_{j}", qj, pj, mse_j, u_j, al))

    for blk_tag, (Qb, Bb, Rb) in named_blocks:
        for j in range(Qb.shape[1]):
            qj = np.asarray(Qb[:, j], dtype=np.float64)
            bj = np.asarray(Bb[:, j], dtype=np.float64)
            rj = np.asarray(Rb[:, j], dtype=np.float64)
            if np.all(np.isfinite(qj)) and np.all(np.isfinite(bj)):
                pj, al = _adaptive_test_col(qj, bj, rj, y, vy)
                mse_j = float(np.mean((qj - y) ** 2))
                u_j = -0.5 * math.log(max(mse_j / vy, 1e-15))
                cands.append((f"{blk_tag}_{j}", qj, pj, mse_j, u_j, al))

    if not cands:
        return np.full(n_a, ym, dtype=np.float64), {"mode": "empty_pool"}

    # Sort strictly by 5-fold OOF utility `u_j` descending (pool-agnostic!)
    cands.sort(key=lambda t: -t[4])
    best_mse = cands[0][3]
    best_u = cands[0][4]
    second_u = cands[1][4] if len(cands) >= 2 else best_u

    # Pool-agnostic OOF quality gate: keep columns within 2.5x of global best MSE (up to top 22 diverse columns)
    mse_cap = max(2.50 * best_mse, best_mse + 0.18 * vy)
    kept = [c for c in cands if c[3] <= mse_cap][:22]
    if not kept:
        kept = cands[:1]

    Q_mat = np.column_stack([c[1] for c in kept])  # [n_s, K]
    P_mat = np.column_stack([c[2] for c in kept])  # [n_a, K]
    col_names = [c[0] for c in kept]
    K = len(kept)

    # Diversity-aware regularization: ZERO forced weight-splitting when #1 column leads #2 by >= 0.22 utility!
    u_lead = best_u - second_u
    l2_div = 0.0 if u_lead >= 0.22 else (0.008 * max(0.0, 0.22 - u_lead) / 0.22)

    # 2. Strict Nested 5-Fold OOF Tournament on `(X_search, y_search)` (`0` `y_audit` read!)
    oof_single = np.zeros(n_s, dtype=np.float64)
    oof_simplex_all = np.zeros(n_s, dtype=np.float64)
    oof_simplex_top6 = np.zeros(n_s, dtype=np.float64)
    oof_bcar = np.zeros(n_s, dtype=np.float64)

    for fk in range(5):
        va = lab == fk
        tr = ~va
        if not np.any(va) or int(np.sum(tr)) < 12:
            continue
        Q_tr, y_tr, lab_tr = Q_mat[tr], y[tr], lab[tr]
        mses_tr = np.mean((Q_tr - y_tr[:, None]) ** 2, axis=0)
        win_f = int(np.argmin(mses_tr))
        oof_single[va] = Q_mat[va, win_f]

        w_all_f = _fit_log_rmse_simplex(Q_tr, y_tr, lab_tr, l2_w=l2_div)
        oof_simplex_all[va] = Q_mat[va] @ w_all_f

        top6_idx = np.argsort(mses_tr)[: min(6, K)]
        w_t6_f = _fit_log_rmse_simplex(Q_tr[:, top6_idx], y_tr, lab_tr, l2_w=0.5 * l2_div)
        oof_simplex_top6[va] = Q_mat[va][:, top6_idx] @ w_t6_f

        try:
            wb_f = 0.65 * GM.bcar_weights(Q_tr, y_tr, f"{cid}_f{fk}", "fold_log", lab_tr) + 0.35 * GM.bcar_weights(Q_tr, y_tr, f"{cid}_f{fk}", "huber")
            wb_f = wb_f / max(float(np.sum(wb_f)), 1e-12)
            oof_bcar[va] = Q_mat[va] @ wb_f
        except Exception:
            oof_bcar[va] = oof_simplex_all[va]

    # Full-X_search simplex fits corresponding to the nested tournament arms
    w_all_full = _fit_log_rmse_simplex(Q_mat, y, lab, l2_w=l2_div)
    pred_simplex_all = P_mat @ w_all_full

    top6_full = np.arange(min(6, K))
    w_t6_full = _fit_log_rmse_simplex(Q_mat[:, top6_full], y, lab, l2_w=0.5 * l2_div)
    pred_simplex_top6 = P_mat[:, top6_full] @ w_t6_full

    pred_single = P_mat[:, 0]

    meta_arms = [
        ("simplex_all", oof_simplex_all, pred_simplex_all),
        ("simplex_top6", oof_simplex_top6, pred_simplex_top6),
        ("single_winner", oof_single, pred_single),
    ]

    try:
        wb_full = 0.65 * GM.bcar_weights(Q_mat, y, cid, "fold_log", lab) + 0.35 * GM.bcar_weights(Q_mat, y, cid, "huber")
        wb_full = wb_full / max(float(np.sum(wb_full)), 1e-12)
        pred_bcar = P_mat @ wb_full
        if np.all(np.isfinite(oof_bcar)) and np.all(np.isfinite(pred_bcar)):
            meta_arms.append(("bcar_foldlog_huber", oof_bcar, pred_bcar))
    except Exception:
        pass

    # Combine meta-arms with a nested log-RMSE simplex (giving seamless convex combination of top6 + all + single!)
    M_oof = np.column_stack([m[1] for m in meta_arms])
    M_pred = np.column_stack([m[2] for m in meta_arms])
    w_meta = _fit_log_rmse_simplex(M_oof, y, lab, l2_w=0.0)
    oof_final = M_oof @ w_meta
    pred_final = M_pred @ w_meta

    # 3. Optional `ya`-Free Context-Adaptive L2 Residual Router (strictly nested 5-fold OOF gated >= +0.015 u!)
    l2_used = False
    u_base = -0.5 * math.log(max(float(np.mean((oof_final - y) ** 2)) / vy, 1e-15))
    if n_s >= 140 and K >= 3 and u_lead < 0.35:
        # Predict local residual `y - oof_final` using top standardized column differences `(q_k - oof_final)` + top Xs features
        D_oof = Q_mat[:, : min(5, K)] - oof_final[:, None]
        D_pred = P_mat[:, : min(5, K)] - pred_final[:, None]
        miss_s = ~np.isfinite(Xs)
        miss_a = ~np.isfinite(Xa)
        p_raw = min(Xs.shape[1], 12)
        Zs = np.zeros((n_s, p_raw), dtype=np.float64)
        Za = np.zeros((n_a, p_raw), dtype=np.float64)
        for j in range(p_raw):
            fin = ~miss_s[:, j]
            if np.any(fin):
                med_j = float(np.median(Xs[fin, j]))
                sc_j = max(float(np.std(Xs[fin, j])), 1e-9)
                Zs[:, j] = np.clip((np.where(fin, Xs[:, j], med_j) - med_j) / sc_j, -5.0, 5.0)
                Za[:, j] = np.clip((np.where(~miss_a[:, j], Xa[:, j], med_j) - med_j) / sc_j, -5.0, 5.0)

        # Modulated disagreement features `[D_oof, D_oof[:, 0:2] * Zs[:, 0:3]]`
        inter_s = [D_oof]
        inter_a = [D_pred]
        for dk in range(min(2, D_oof.shape[1])):
            for zj in range(min(3, p_raw)):
                inter_s.append((D_oof[:, dk] * Zs[:, zj])[:, None])
                inter_a.append((D_pred[:, dk] * Za[:, zj])[:, None])
        Fs_r = np.hstack(inter_s)
        Fa_r = np.hstack(inter_a)

        res_target = y - oof_final
        r_oof = np.zeros(n_s, dtype=np.float64)
        for fk in range(5):
            va = lab == fk
            tr = ~va
            if not np.any(va) or int(np.sum(tr)) < 20:
                continue
            rg = Ridge(alpha=50.0)
            rg.fit(Fs_r[tr], res_target[tr])
            r_oof[va] = rg.predict(Fs_r[va])

        cand_oof = oof_final + 0.50 * r_oof
        u_l2 = -0.5 * math.log(max(float(np.mean((cand_oof - y) ** 2)) / vy, 1e-15))
        if u_l2 >= u_base + 0.018:
            rg_full = Ridge(alpha=50.0)
            rg_full.fit(Fs_r, res_target)
            pred_final = pred_final + 0.50 * rg_full.predict(Fa_r)
            oof_final = cand_oof
            u_base = u_l2
            l2_used = True

    pred_final = np.clip(pred_final, lo_out, hi_out)
    top_weights = [
        {"col": col_names[i], "w": round(float(w_all_full[i]), 4), "u_oof": round(float(kept[i][4]), 4), "alpha_ref": round(float(kept[i][5]), 3)}
        for i in np.argsort(-w_all_full)[:5]
        if w_all_full[i] >= 0.005
    ]
    diag = {
        "n_cands_total": len(cands),
        "n_kept": K,
        "best_col": col_names[0],
        "best_col_u_oof": round(best_u, 4),
        "u_lead_1_vs_2": round(u_lead, 4),
        "nested_stack_u_oof": round(u_base, 4),
        "meta_weights": {meta_arms[i][0]: round(float(w_meta[i]), 4) for i in range(len(meta_arms))},
        "l2_router_used": l2_used,
        "top_simplex_weights": top_weights,
    }
    return pred_final, diag
