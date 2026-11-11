#!/usr/bin/env python3
"""H431 (CW-3 Part A): Generate the 12-family ICL Capability Battery (ICB_v1, 340 tasks).

  icb_gen.py --out ICB_v1.npz [--seed 431001]

Grid (CW3_CURRICULUM.md section 3):
  - Families B1..B11: n_ctx in (256, 1024), n_q = 256, noise in (0.0, 0.05, 0.20), 5 tasks/cell -> 330 tasks
  - Family B12 (heavy tails t(2) at 0.20): n_ctx in (256, 1024), n_q = 256, noise = 0.20, 5 tasks/cell -> 10 tasks
  Total = 340 tasks. Every reference predictor (B1..B11) is fitted strictly on context rows [:n_ctx] without reading
  query rows or query targets; B12's reference is the true noise-free linear function X_q @ beta (CW-3 section 3).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

FAMILIES = (
    ("B1", "linear_dense"),
    ("B2", "linear_sparse"),
    ("B3", "single_key"),
    ("B4", "pair_key"),
    ("B5", "key_plus_slope"),
    ("B6", "arithmetic_pair"),
    ("B7", "smooth_1d"),
    ("B8", "additive_smooth"),
    ("B9", "unordered_factors"),
    ("B10", "multiplicative"),
    ("B11", "thresholds_tree"),
    ("B12", "heavy_tails_t2"),
)


def u8_np(pred: np.ndarray, yq: np.ndarray, ym: float) -> float:
    r = float(np.sqrt(np.mean((pred - yq) ** 2)))
    r0 = max(float(np.sqrt(np.mean((ym - yq) ** 2))), 1e-30)
    return float(np.clip(-math.log(max(r / r0, 1e-30)), -2.0, 8.0))


def _ols_predict(xc: np.ndarray, yc: np.ndarray, xq: np.ndarray, lam: float = 1e-8) -> np.ndarray:
    mu = xc.mean(axis=0, keepdims=True)
    sd = np.maximum(xc.std(axis=0, keepdims=True), 1e-9)
    zc = (xc - mu) / sd
    zq = (xq - mu) / sd
    ym = float(yc.mean())
    a = zc.T @ zc + lam * np.eye(zc.shape[1], dtype=np.float64)
    w = np.linalg.solve(a, zc.T @ (yc - ym))
    return (zq @ w) + ym


def _spline_basis(z: np.ndarray, knots: np.ndarray, z_mu: float, z_sd: float) -> np.ndarray:
    zs = (z - z_mu) / z_sd
    ks = (knots - z_mu) / z_sd
    diff = np.maximum(0.0, zs[:, None] - ks[None, :]) ** 3
    return np.column_stack([zs, zs ** 2, zs ** 3, diff])


def _cv_spline_1d(zc: np.ndarray, yc: np.ndarray, zq: np.ndarray, n_knots: int = 48) -> np.ndarray:
    """64-feature penalized cubic-spline smoother fitted strictly on context rows [:n_ctx] via 5-fold CV over lambda."""
    z_mu = float(zc.mean())
    z_sd = max(float(zc.std()), 1e-9)
    qs = np.linspace(0.02, 0.98, n_knots)
    knots = np.unique(np.quantile(zc, qs))
    bc = _spline_basis(zc, knots, z_mu, z_sd)
    bq = _spline_basis(zq, knots, z_mu, z_sd)
    ym = float(yc.mean())
    y0 = yc - ym
    n = len(zc)
    folds = np.arange(n) % 5
    best_lam, best_mse = 1e-6, float("inf")
    eye = np.eye(bc.shape[1], dtype=np.float64)
    for lam in (1e-9, 1e-6, 1e-4, 1e-2, 1e-1, 1.0):
        oof = np.empty(n, dtype=np.float64)
        for k in range(5):
            tr, va = folds != k, folds == k
            b_tr = bc[tr]
            ym_k = float(yc[tr].mean())
            w = np.linalg.solve(b_tr.T @ b_tr + lam * eye, b_tr.T @ (yc[tr] - ym_k))
            oof[va] = bc[va] @ w + ym_k
        mse = float(np.mean((oof - yc) ** 2))
        if mse < best_mse:
            best_mse, best_lam = mse, lam
    w = np.linalg.solve(bc.T @ bc + best_lam * eye, bc.T @ y0)
    return bq @ w + ym


def _cv_additive_spline(xc: np.ndarray, yc: np.ndarray, xq: np.ndarray, cols: list[int], n_knots: int = 24) -> np.ndarray:
    """Joint penalized spline smoother on the true additive columns (converged backfit), fitted on context only."""
    parts_c, parts_q = [], []
    qs = np.linspace(0.04, 0.96, n_knots)
    for j in cols:
        zc, zq = xc[:, j], xq[:, j]
        z_mu, z_sd = float(zc.mean()), max(float(zc.std()), 1e-9)
        knots = np.unique(np.quantile(zc, qs))
        parts_c.append(_spline_basis(zc, knots, z_mu, z_sd))
        parts_q.append(_spline_basis(zq, knots, z_mu, z_sd))
    bc = np.hstack(parts_c)
    bq = np.hstack(parts_q)
    n = len(yc)
    folds = np.arange(n) % 5
    eye = np.eye(bc.shape[1], dtype=np.float64)
    best_lam, best_mse = 1e-6, float("inf")
    for lam in (1e-9, 1e-6, 1e-4, 1e-2, 1e-1, 1.0):
        oof = np.empty(n, dtype=np.float64)
        for k in range(5):
            tr, va = folds != k, folds == k
            b_tr = bc[tr]
            ym_k = float(yc[tr].mean())
            w = np.linalg.solve(b_tr.T @ b_tr + lam * eye, b_tr.T @ (yc[tr] - ym_k))
            oof[va] = bc[va] @ w + ym_k
        mse = float(np.mean((oof - yc) ** 2))
        if mse < best_mse:
            best_mse, best_lam = mse, lam
    ym = float(yc.mean())
    w = np.linalg.solve(bc.T @ bc + best_lam * eye, bc.T @ (yc - ym))
    return bq @ w + ym


def _onehot_encode(xc_cols: np.ndarray, xq_cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build exact-level one-hot matrices from context levels only; unseen query levels map to all-zero row."""
    blocks_c, blocks_q = [], []
    for j in range(xc_cols.shape[1]):
        vals = np.rint(xc_cols[:, j]).astype(np.int64)
        qvals = np.rint(xq_cols[:, j]).astype(np.int64)
        levs = np.unique(vals)
        blocks_c.append((vals[:, None] == levs[None, :]).astype(np.float64))
        blocks_q.append((qvals[:, None] == levs[None, :]).astype(np.float64))
    return np.hstack(blocks_c), np.hstack(blocks_q)


def fit_reference_predictor(meta: dict, xc: np.ndarray, yc: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Context-only reference predictor for a task (never sees yq; query rows xq are predicted independently)."""
    fid = meta["family"]
    if fid == "B1":
        return _ols_predict(xc, yc, xq, lam=1e-8)
    if fid == "B2":
        idx = meta["active_cols"]
        return _ols_predict(xc[:, idx], yc, xq[:, idx], lam=1e-8)
    if fid == "B3":
        jk = meta["key_col"]
        kc = np.rint(xc[:, jk]).astype(np.int64)
        kq = np.rint(xq[:, jk]).astype(np.int64)
        ym = float(yc.mean())
        sums: dict[int, float] = {}
        cnts: dict[int, int] = {}
        for g, val in zip(kc.tolist(), yc.tolist()):
            sums[g] = sums.get(g, 0.0) + val
            cnts[g] = cnts.get(g, 0) + 1
        return np.array([sums[g] / cnts[g] if g in cnts else ym for g in kq.tolist()], dtype=np.float64)
    if fid == "B4":
        ja, jb = meta["key_cols"]
        ac, bc = np.rint(xc[:, ja]).astype(np.int64), np.rint(xc[:, jb]).astype(np.int64)
        aq, bq = np.rint(xq[:, ja]).astype(np.int64), np.rint(xq[:, jb]).astype(np.int64)
        ym = float(yc.mean())
        pair_sum, pair_cnt = {}, {}
        ma_sum, ma_cnt = {}, {}
        mb_sum, mb_cnt = {}, {}
        for a, b, val in zip(ac.tolist(), bc.tolist(), yc.tolist()):
            pair_sum[(a, b)] = pair_sum.get((a, b), 0.0) + val
            pair_cnt[(a, b)] = pair_cnt.get((a, b), 0) + 1
            ma_sum[a] = ma_sum.get(a, 0.0) + val
            ma_cnt[a] = ma_cnt.get(a, 0) + 1
            mb_sum[b] = mb_sum.get(b, 0.0) + val
            mb_cnt[b] = mb_cnt.get(b, 0) + 1
        out = np.empty(len(xq), dtype=np.float64)
        for i, (a, b) in enumerate(zip(aq.tolist(), bq.tolist())):
            if (a, b) in pair_cnt:
                out[i] = pair_sum[(a, b)] / pair_cnt[(a, b)]
            else:
                ea = (ma_sum[a] / ma_cnt[a] - ym) if a in ma_cnt else 0.0
                eb = (mb_sum[b] / mb_cnt[b] - ym) if b in mb_cnt else 0.0
                out[i] = ym + ea + eb
        return out
    if fid == "B5":
        jk, js = meta["key_col"], meta["slope_col"]
        kc = np.rint(xc[:, jk]).astype(np.int64)
        kq = np.rint(xq[:, jk]).astype(np.int64)
        sc, sq = xc[:, js], xq[:, js]
        ym = float(yc.mean())
        models = {}
        for g in np.unique(kc):
            m = kc == g
            if int(m.sum()) >= 3 and float(sc[m].std()) > 1e-9:
                a_mat = np.column_stack([np.ones(int(m.sum())), sc[m]])
                w = np.linalg.solve(a_mat.T @ a_mat + 1e-8 * np.eye(2), a_mat.T @ yc[m])
                models[int(g)] = (float(w[0]), float(w[1]))
            else:
                models[int(g)] = (float(yc[m].mean()), 0.0)
        return np.array([models[g][0] + models[g][1] * s if g in models else ym
                         for g, s in zip(kq.tolist(), sq.tolist())], dtype=np.float64)
    if fid == "B6":
        ja, jb, op = meta["pair_cols"][0], meta["pair_cols"][1], meta["op"]
        if op == "sub":
            dc, dq = xc[:, ja] - xc[:, jb], xq[:, ja] - xq[:, jb]
        elif op == "mul":
            dc, dq = xc[:, ja] * xc[:, jb], xq[:, ja] * xq[:, jb]
        else:
            dc = xc[:, ja] / np.where(np.abs(xc[:, jb]) > 1e-9, xc[:, jb], 1e-9)
            dq = xq[:, ja] / np.where(np.abs(xq[:, jb]) > 1e-9, xq[:, jb], 1e-9)
        return _cv_spline_1d(dc, yc, dq, n_knots=48)
    if fid == "B7":
        j = meta["smooth_col"]
        return _cv_spline_1d(xc[:, j], yc, xq[:, j], n_knots=48)
    if fid == "B8":
        return _cv_additive_spline(xc, yc, xq, meta["smooth_cols"], n_knots=24)
    if fid == "B9":
        fcols = meta["factor_cols"]
        oh_c, oh_q = _onehot_encode(xc[:, fcols], xq[:, fcols])
        ym = float(yc.mean())
        eye = np.eye(oh_c.shape[1], dtype=np.float64)
        n = len(yc)
        folds = np.arange(n) % 5
        best_lam, best_mse = 1e-6, float("inf")
        for lam in (1e-8, 1e-4, 1e-2, 0.1, 1.0):
            oof = np.empty(n, dtype=np.float64)
            for k in range(5):
                tr, va = folds != k, folds == k
                ym_k = float(yc[tr].mean())
                w = np.linalg.solve(oh_c[tr].T @ oh_c[tr] + lam * eye, oh_c[tr].T @ (yc[tr] - ym_k))
                oof[va] = oh_c[va] @ w + ym_k
            mse = float(np.mean((oof - yc) ** 2))
            if mse < best_mse:
                best_mse, best_lam = mse, lam
        w = np.linalg.solve(oh_c.T @ oh_c + best_lam * eye, oh_c.T @ (yc - ym))
        return oh_q @ w + ym
    if fid == "B10":
        fcols, j_num = meta["factor_cols"], meta["num_col"]
        oh_c, oh_q = _onehot_encode(xc[:, fcols], xq[:, fcols])
        s_mu, s_sd = float(xc[:, j_num].mean()), max(float(xc[:, j_num].std()), 1e-9)
        bc = np.column_stack([oh_c, (xc[:, j_num] - s_mu) / s_sd])
        bq = np.column_stack([oh_q, (xq[:, j_num] - s_mu) / s_sd])
        ly = np.log(np.maximum(yc, 1e-30))
        lym = float(ly.mean())
        eye = np.eye(bc.shape[1], dtype=np.float64)
        w = np.linalg.solve(bc.T @ bc + 1e-6 * eye, bc.T @ (ly - lym))
        pred_log_c = bc @ w + lym
        pred_log_q = bq @ w + lym
        smear = float(np.mean(np.exp(np.clip(ly - pred_log_c, -10.0, 10.0))))
        return np.exp(np.clip(pred_log_q, -20.0, 20.0)) * smear
    if fid == "B11":
        tcols, thrs = meta["tree_cols"], meta["tree_thrs"]
        lc = ((xc[:, tcols[0]] > thrs[0]).astype(np.int64)
              + 2 * (xc[:, tcols[1]] > thrs[1]).astype(np.int64)
              + 4 * (xc[:, tcols[2]] > thrs[2]).astype(np.int64))
        lq = ((xq[:, tcols[0]] > thrs[0]).astype(np.int64)
              + 2 * (xq[:, tcols[1]] > thrs[1]).astype(np.int64)
              + 4 * (xq[:, tcols[2]] > thrs[2]).astype(np.int64))
        ym = float(yc.mean())
        leaf_means = {leaf: float(yc[lc == leaf].mean()) for leaf in range(8) if np.any(lc == leaf)}
        return np.array([leaf_means.get(int(leaf), ym) for leaf in lq], dtype=np.float64)
    if fid == "B12":
        # CW-3 section 3: reference predictor for B12 is the true noise-free linear function X_q @ beta_true.
        return xq @ np.asarray(meta["beta_true"], dtype=np.float64) + float(meta["intercept_true"])
    raise ValueError(f"unknown family {fid}")


def _draw_group_ids(rng: np.random.Generator, n_ctx: int, n_q: int, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Ensure every group appears at least twice in context so noise-free lookup is well-posed, then random codes."""
    base_ctx = np.repeat(np.arange(n_groups), 2)
    rem_ctx = rng.integers(0, n_groups, size=n_ctx - len(base_ctx))
    g_ctx = np.concatenate([base_ctx, rem_ctx])
    rng.shuffle(g_ctx)
    g_q = rng.integers(0, n_groups, size=n_q)
    g_all = np.concatenate([g_ctx, g_q])
    codes = rng.choice(np.arange(100, 9900), size=n_groups, replace=False)
    return g_all, codes[g_all].astype(np.float64)


def make_single_task(rng: np.random.Generator, fid: str, n_ctx: int, n_q: int, noise: float, rep: int) -> dict:
    n = n_ctx + n_q
    fn = 64 if fid == "B2" else 16
    # Continuous covariates with realistic affine column scales
    x = rng.standard_normal((n, fn))
    scales = np.exp(rng.uniform(-1.5, 2.5, size=(1, fn)))
    shifts = rng.standard_normal((1, fn)) * scales
    x_aff = x * scales + shifts
    perm = rng.permutation(fn)
    meta: dict = {"family": fid, "n_ctx": n_ctx, "n_q": n_q, "noise": noise, "rep": rep, "F": fn}

    if fid == "B1":
        w = rng.standard_normal(fn) * np.exp(-0.12 * np.arange(fn))
        f_clean = x @ w
        x_out = x_aff[:, perm]
    elif fid == "B2":
        act = perm[:5].tolist()
        w = rng.standard_normal(5) * np.array([1.5, 1.2, 1.0, 0.8, 0.6])
        x_out = x_aff
        # Standardize active columns for clean signal construction
        z_act = (x_out[:, act] - x_out[:, act].mean(0)) / np.maximum(x_out[:, act].std(0), 1e-9)
        f_clean = z_act @ w
        meta["active_cols"] = act
    elif fid == "B3":
        n_groups = 16 if (rep < 3 or n_ctx == 256 and rep < 4) else min(128, n_ctx // 2)
        g_idx, g_codes = _draw_group_ids(rng, n_ctx, n_q, n_groups)
        eff = rng.standard_normal(n_groups) * 2.0
        f_clean = eff[g_idx]
        x_out = x_aff.copy()
        jk = int(perm[0])
        x_out[:, jk] = g_codes
        meta["key_col"] = jk
        meta["n_groups"] = int(n_groups)
    elif fid == "B4":
        lev = 8 if n_ctx == 256 else 12
        # Guarantee every pair appears in context at least once when n_ctx >= 2 * lev^2
        n_pairs = lev * lev
        if n_ctx >= 2 * n_pairs:
            p_ctx = np.concatenate([np.repeat(np.arange(n_pairs), 2), rng.integers(0, n_pairs, size=n_ctx - 2 * n_pairs)])
        else:
            p_ctx = np.concatenate([np.arange(n_pairs), rng.integers(0, n_pairs, size=n_ctx - n_pairs)])
        rng.shuffle(p_ctx)
        p_q = rng.integers(0, n_pairs, size=n_q)
        p_all = np.concatenate([p_ctx, p_q])
        ga, gb = p_all // lev, p_all % lev
        codes_a = rng.choice(np.arange(10, 500), size=lev, replace=False)
        codes_b = rng.choice(np.arange(500, 990), size=lev, replace=False)
        table = rng.standard_normal((lev, lev)) * 1.5
        f_clean = table[ga, gb]
        x_out = x_aff.copy()
        ja, jb = int(perm[0]), int(perm[1])
        x_out[:, ja] = codes_a[ga].astype(np.float64)
        x_out[:, jb] = codes_b[gb].astype(np.float64)
        meta["key_cols"] = [ja, jb]
        meta["levels"] = int(lev)
    elif fid == "B5":
        n_groups = 16
        # At least 6 rows per group in context so per-group OLS has exact degrees of freedom
        base_ctx = np.repeat(np.arange(n_groups), 6)
        g_ctx = np.concatenate([base_ctx, rng.integers(0, n_groups, size=n_ctx - len(base_ctx))])
        rng.shuffle(g_ctx)
        g_all = np.concatenate([g_ctx, rng.integers(0, n_groups, size=n_q)])
        codes = rng.choice(np.arange(20, 2000), size=n_groups, replace=False)
        jk, js = int(perm[0]), int(perm[1])
        x_out = x_aff.copy()
        x_out[:, jk] = codes[g_all].astype(np.float64)
        a_g = rng.standard_normal(n_groups) * 2.0
        b_g = rng.standard_normal(n_groups) * 1.5
        s_std = (x_out[:, js] - x_out[:, js].mean()) / max(float(x_out[:, js].std()), 1e-9)
        f_clean = a_g[g_all] + b_g[g_all] * s_std
        meta["key_col"] = jk
        meta["slope_col"] = js
    elif fid == "B6":
        ops = ("sub", "mul", "div")
        hs = ("id", "tanh", "sq")
        op = ops[rep % 3]
        hname = hs[(rep + (0 if n_ctx == 256 else 1)) % 3]
        ja, jb = int(perm[0]), int(perm[1])
        x_out = x_aff.copy()
        sc = float(np.exp(rng.uniform(-0.5, 2.0)))
        if op == "div":
            xa_raw = rng.uniform(0.5, 5.0, size=n) * sc
            xb_raw = rng.uniform(0.8, 4.5, size=n) * float(np.exp(rng.uniform(-0.5, 1.5)))
            d = xa_raw / xb_raw
        elif op == "sub":
            shift_shared = rng.uniform(10.0, 200.0)
            xa_raw = shift_shared + rng.standard_normal(n) * sc
            xb_raw = shift_shared + rng.standard_normal(n) * sc
            d = xa_raw - xb_raw
        else:
            xa_raw = rng.uniform(-2.5, 2.5, size=n) * sc + rng.uniform(1.0, 8.0)
            xb_raw = rng.uniform(-2.5, 2.5, size=n) + rng.uniform(1.0, 8.0)
            d = xa_raw * xb_raw
        x_out[:, ja] = xa_raw
        x_out[:, jb] = xb_raw
        dz = (d - d.mean()) / max(float(d.std()), 1e-9)
        f_clean = dz if hname == "id" else (np.tanh(1.4 * dz) if hname == "tanh" else (dz ** 2 - 1.0))
        meta["pair_cols"] = [ja, jb]
        meta["op"] = op
        meta["h"] = hname
    elif fid == "B7":
        j = int(perm[0])
        x_out = x_aff.copy()
        u_raw = rng.uniform(-2.5, 2.5, size=n)
        x_out[:, j] = u_raw * float(scales[0, j]) + float(shifts[0, j])
        freqs = np.array([0.7, 1.3, 2.1, 3.0])
        phases = rng.uniform(0.0, 2.0 * math.pi, size=4)
        amps = rng.standard_normal(4) / np.array([1.0, 1.4, 2.0, 2.8])
        f_clean = sum(amps[m] * np.sin(freqs[m] * u_raw + phases[m]) for m in range(4))
        meta["smooth_col"] = j
    elif fid == "B8":
        cols = perm[:4].tolist()
        x_out = x_aff.copy()
        f_clean = np.zeros(n, dtype=np.float64)
        for m, j in enumerate(cols):
            u_raw = rng.uniform(-2.5, 2.5, size=n)
            x_out[:, j] = u_raw * float(scales[0, j]) + float(shifts[0, j])
            if m == 0:
                term = np.sin(1.5 * u_raw) + 0.4 * u_raw
            elif m == 1:
                term = 0.7 * (u_raw ** 2 - 2.0)
            elif m == 2:
                term = np.tanh(1.6 * (u_raw - 0.3))
            else:
                term = np.cos(1.8 * u_raw) - 0.3 * u_raw
            f_clean += term / max(float(term.std()), 1e-9)
        meta["smooth_cols"] = cols
    elif fid == "B9":
        fcols = perm[:6].tolist()
        x_out = x_aff.copy()
        f_clean = np.zeros(n, dtype=np.float64)
        for j in fcols:
            lev = int(rng.integers(3, 13))
            g_idx, g_codes = _draw_group_ids(rng, n_ctx, n_q, lev)
            x_out[:, j] = g_codes
            eff = rng.standard_normal(lev)
            f_clean += eff[g_idx]
        meta["factor_cols"] = fcols
    elif fid == "B10":
        fcols = perm[:4].tolist()
        j_num = int(perm[4])
        x_out = x_aff.copy()
        z_num = np.clip(rng.standard_normal(n), -2.5, 2.5)
        x_out[:, j_num] = z_num * float(scales[0, j_num]) + float(shifts[0, j_num])
        log_sig = 0.5 * z_num
        for j in fcols:
            lev = int(rng.integers(3, 9))
            g_idx, g_codes = _draw_group_ids(rng, n_ctx, n_q, lev)
            x_out[:, j] = g_codes
            eff = rng.standard_normal(lev) * 0.45
            log_sig = log_sig + eff[g_idx]
        log_sig = (log_sig - log_sig.mean()) / max(float(log_sig.std()), 1e-9) * 0.85
        f_clean = np.exp(log_sig) * float(np.exp(rng.uniform(0.5, 4.0)))
        meta["factor_cols"] = fcols
        meta["num_col"] = j_num
        if noise > 0.0:
            # Multiplicative log-normal noise calibrated so std(y - f_clean) / std(f_clean) == noise
            eps = rng.standard_normal(n)
            lo_s, hi_s = 0.0, 1.5
            sig_sd = float(f_clean.std())
            for _ in range(28):
                mid = 0.5 * (lo_s + hi_s)
                cand = f_clean * np.exp(mid * eps - 0.5 * mid * mid)
                if float((cand - f_clean).std()) < noise * sig_sd:
                    lo_s = mid
                else:
                    hi_s = mid
            s_log = 0.5 * (lo_s + hi_s)
            y = f_clean * np.exp(s_log * eps - 0.5 * s_log * s_log)
        else:
            y = f_clean.copy()
        ref_q = fit_reference_predictor(meta, x_out[:n_ctx], y[:n_ctx], x_out[n_ctx:])
        return {"x": x_out.astype(np.float32), "y": y.astype(np.float32),
                "f_clean": f_clean.astype(np.float64), "ref_q": ref_q.astype(np.float64), "meta": meta}
    elif fid == "B11":
        tcols = perm[:3].tolist()
        x_out = x_aff.copy()
        # Use context medians as thresholds so all 8 leaves are balanced
        thrs = [float(np.median(x_out[:n_ctx, c])) for c in tcols]
        leaf = ((x_out[:, tcols[0]] > thrs[0]).astype(np.int64)
                + 2 * (x_out[:, tcols[1]] > thrs[1]).astype(np.int64)
                + 4 * (x_out[:, tcols[2]] > thrs[2]).astype(np.int64))
        vals = rng.standard_normal(8) * 2.0
        f_clean = vals[leaf]
        meta["tree_cols"] = tcols
        meta["tree_thrs"] = thrs
    elif fid == "B12":
        x_out = x_aff[:, perm]
        mu_true = x_out.mean(0)
        sd_true = np.maximum(x_out.std(0), 1e-9)
        w_std = rng.standard_normal(fn) * np.exp(-0.10 * np.arange(fn))
        beta_raw = w_std / sd_true
        intercept_raw = float(-mu_true @ beta_raw + 25.0)
        f_clean = x_out @ beta_raw + intercept_raw
        meta["beta_true"] = beta_raw.tolist()
        meta["intercept_true"] = intercept_raw
        # Student-t(2) heavy-tailed noise scaled to 0.20 * std(f_clean) via robust clip-calibrated or sample sd
        t2 = rng.standard_normal(n) / np.sqrt(np.maximum(rng.chisquare(df=2.0, size=n) / 2.0, 1e-4))
        # Scale by IQR-equivalent (IQR of t(2) = 2.0 -> unit-IQR scale, times signal std * noise)
        iqr_t2 = max(float(np.quantile(t2, 0.75) - np.quantile(t2, 0.25)) / 1.349, 1e-6)
        y = f_clean + (t2 / iqr_t2) * (noise * float(f_clean.std()))
        ref_q = fit_reference_predictor(meta, x_out[:n_ctx], y[:n_ctx], x_out[n_ctx:])
        return {"x": x_out.astype(np.float32), "y": y.astype(np.float32),
                "f_clean": f_clean.astype(np.float64), "ref_q": ref_q.astype(np.float64), "meta": meta}
    else:
        raise ValueError(fid)

    # Scale clean target to realistic spread and add Gaussian noise at `noise * std(f_clean)`
    t_scale = float(np.exp(rng.uniform(-0.5, 3.0)))
    t_shift = float(rng.standard_normal() * t_scale * 2.0)
    f_scaled = (f_clean - f_clean.mean()) / max(float(f_clean.std()), 1e-9) * t_scale + t_shift
    sig_sd = float(f_scaled.std())
    y = f_scaled + (rng.standard_normal(n) * (noise * sig_sd) if noise > 0.0 else 0.0)
    ref_q = fit_reference_predictor(meta, x_out[:n_ctx], y[:n_ctx], x_out[n_ctx:])
    return {"x": x_out.astype(np.float32), "y": y.astype(np.float32),
            "f_clean": f_scaled.astype(np.float64), "ref_q": ref_q.astype(np.float64), "meta": meta}


def generate_battery(seed: int = 431_001) -> list[dict]:
    rng = np.random.default_rng(seed)
    tasks: list[dict] = []
    for fid, fname in FAMILIES:
        noises = (0.20,) if fid == "B12" else (0.0, 0.05, 0.20)
        for n_ctx in (256, 1024):
            for nz in noises:
                for rep in range(5):
                    t = make_single_task(rng, fid, n_ctx=n_ctx, n_q=256, noise=nz, rep=rep)
                    t["meta"]["family_name"] = fname
                    t["meta"]["task_id"] = len(tasks)
                    yq = t["y"][n_ctx:].astype(np.float64)
                    ym = float(t["y"][:n_ctx].astype(np.float64).mean())
                    t["meta"]["u8_ref"] = u8_np(t["ref_q"], yq, ym)
                    tasks.append(t)
    return tasks


def save_battery(tasks: list[dict], out_path: Path, seed: int) -> dict:
    if out_path.exists():
        raise SystemExit(f"{out_path} exists (create-once)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xs = [t["x"].reshape(-1) for t in tasks]
    ys = [t["y"] for t in tasks]
    refs = [t["ref_q"] for t in tasks]
    shapes = np.array([[t["x"].shape[0], t["x"].shape[1], t["meta"]["n_ctx"], t["meta"]["n_q"]] for t in tasks], dtype=np.int64)
    x_off = np.concatenate([[0], np.cumsum([a.size for a in xs[:-1]], dtype=np.int64)])
    y_off = np.concatenate([[0], np.cumsum([a.size for a in ys[:-1]], dtype=np.int64)])
    r_off = np.concatenate([[0], np.cumsum([a.size for a in refs[:-1]], dtype=np.int64)])
    families = np.array([t["meta"]["family"] for t in tasks])
    noises = np.array([t["meta"]["noise"] for t in tasks], dtype=np.float64)
    u8_refs = np.array([t["meta"]["u8_ref"] for t in tasks], dtype=np.float64)
    meta_json = json.dumps([t["meta"] for t in tasks], sort_keys=True)

    tmp = out_path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        x_data=np.concatenate(xs),
        y_data=np.concatenate(ys),
        ref_q=np.concatenate(refs),
        shapes=shapes,
        x_off=x_off,
        y_off=y_off,
        r_off=r_off,
        families=families,
        noises=noises,
        u8_refs=u8_refs,
        meta_json=np.array(meta_json),
        seed=np.int64(seed),
    )
    tmp.rename(out_path)
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    by_fam = {}
    for fid, fname in FAMILIES:
        nf = [t["meta"]["u8_ref"] for t in tasks if t["meta"]["family"] == fid and t["meta"]["noise"] == 0.0]
        all_u = [t["meta"]["u8_ref"] for t in tasks if t["meta"]["family"] == fid]
        by_fam[fid] = {
            "name": fname,
            "tasks": len(all_u),
            "mean_u8_ref_all": round(float(np.mean(all_u)), 4),
            "mean_u8_ref_noisefree": round(float(np.mean(nf)), 4) if nf else None,
            "min_u8_ref_noisefree": round(float(np.min(nf)), 4) if nf else None,
        }
    manifest = {
        "schema": "jope.h431.icb/v1",
        "seed": seed,
        "tasks": len(tasks),
        "sha256": sha,
        "bytes": out_path.stat().st_size,
        "families": by_fam,
    }
    mpath = out_path.with_suffix(".npz.json")
    with open(mpath, "x") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=431_001)
    a = ap.parse_args()
    tasks = generate_battery(a.seed)
    manifest = save_battery(tasks, Path(a.out), a.seed)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
