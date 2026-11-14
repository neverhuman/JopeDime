#!/usr/bin/env python3
"""Wave H456 Ultra-Entropy Operators (`ultra_entropy_ops_h456.py`).

Implements the four core H456 entropy expansions under a strict `0.0` query-leakage contract
(all fitted statistics, rank knots, ridge/stump parameters, quantizers, and noise scales depend
exclusively on `(X[:n_ctx], y[:n_ctx])`, and every query row `n_ctx + q` is transformed strictly
pointwise with zero cross-query coupling):

  1. Continuous multi-regime varied row sampling (`N_use in [32..2048]`, `p_ctx in [0.05..0.96]`,
     4 fresh full-table row-sampling regimes, OOM-exact microbatch sizing).
  2. Mandatory simultaneous feature removal (`drop in [0.05..0.90]`, including wide tables) +
     reserved synthetic/distractor replacement (`k_inj in [10%..70%]` of `F_use in [4..80]`).
  3. 28 strictly context-fitted (`Xc, yc`) synthetic & adversarial distractor families (adding
     OOF Ridge-GCV, depth-2 oblivious decision-stump ensembles, context quantile splines,
     copula rank-differences, spurious context-overfit / pseudo-ID distractors, and MNAR twins).
  4. Unbiased column subsetting, full column & `FID_SLOTS` permutations, `50%` sign flips,
     `exp(U(-4, 4))` scale + `3*sd` shift + signed-power `[0.35, 2.5]` + `K`-level ordinal
     quantization + Cauchy outlier cells + heteroscedastic/correlated context-IQR noise +
     wider post-prenorm affine jitter (`exp(+-0.45)`) + monotone target warps.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.special import ndtri
import torch

N_MIN_H456 = 32
F_MIN_H456 = 4
F_MAX_DEFAULT_H456 = 80
NUM_SYNTH_FAMILIES_H456 = 28
CLEAN_FAMILIES_H456 = tuple(range(0, 22))
DISTRACTOR_FAMILIES_H456 = tuple(range(22, 28))


def _pointwise_clean(arr: np.ndarray, cap: float = 25.0) -> np.ndarray:
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=cap, neginf=-cap), -cap, cap)


def _ctx_standardize_col(col: np.ndarray, n_ctx: int) -> np.ndarray:
    """Standardizes `col` strictly using mean/std of `col[:n_ctx]` (`0.0` query leakage)."""
    cc = col[:n_ctx]
    fin = np.isfinite(cc)
    med = float(np.median(cc[fin])) if np.any(fin) else 0.0
    clean = np.where(np.isfinite(col), col, med)
    c_slice = clean[:n_ctx]
    mu = float(np.mean(c_slice))
    sd = float(np.std(c_slice))
    if sd < 1e-9 or not math.isfinite(sd):
        return np.zeros_like(clean, dtype=np.float64)
    return _pointwise_clean((clean - mu) / sd, cap=25.0)


def _ctx_impute_matrix(X_all: np.ndarray, n_ctx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Imputes `X_all` using medians of `X_all[:n_ctx]` and returns `(X_imp, med_ctx, iqr_ctx)`."""
    _, p = X_all.shape
    med = np.zeros(p, dtype=np.float64)
    iqr = np.ones(p, dtype=np.float64)
    Xc = X_all[:n_ctx]
    for j in range(p):
        cj = Xc[:, j]
        fin = np.isfinite(cj)
        if np.any(fin):
            vals = cj[fin]
            med[j] = float(np.median(vals))
            q25, q75 = np.percentile(vals, [25.0, 75.0])
            span = float(q75 - q25)
            if span < 1e-9:
                span = float(np.std(vals))
            iqr[j] = span if (span >= 1e-9 and math.isfinite(span)) else 1.0
    X_imp = np.where(np.isfinite(X_all), X_all, med[None, :])
    return X_imp, med, iqr


def compute_oom_safe_batch_size(
    n_use: int,
    f_use: int,
    tokens_cap: int,
    row_tokens_cap: int,
    attn_cap: int,
    bmax: int,
) -> int:
    by_cells = max(1, int(tokens_cap) // max(1, int(n_use) * max(int(f_use), 1)))
    by_rows = max(1, int(row_tokens_cap) // max(1, int(n_use)))
    by_attn = max(1, int(attn_cap) // max(1, int(n_use) * int(n_use)))
    return max(1, min(int(bmax), by_cells, by_rows, by_attn))


def sample_episode_shape_h456(
    rng: np.random.Generator,
    max_rows: int = 2048,
    max_cols: int = F_MAX_DEFAULT_H456,
    tokens_cap: int = 120000,
    row_tokens_cap: int = 8192,
    attn_cap: int = 2800000,
    bmax: int = 12,
    p_ctx_min: float = 0.05,
    p_ctx_max: float = 0.96,
    is_teacher: bool = False,
) -> tuple[int, int, int, int]:
    """Samples continuous `(B, N_use, F_use, n_ctx)` with `60%` log-uniform + `40%` uniform `N_use`."""
    hi_n = max(N_MIN_H456, min(int(max_rows), 768 if is_teacher else int(max_rows)))
    if float(rng.random()) < 0.60:
        ln_n = float(rng.uniform(math.log(float(N_MIN_H456)), math.log(float(hi_n))))
        n_use = int(round(math.exp(ln_n)))
    else:
        n_use = int(rng.integers(N_MIN_H456, hi_n + 1))
    n_use = max(N_MIN_H456, min(hi_n, n_use))

    hi_f = max(F_MIN_H456, int(max_cols))
    if float(rng.random()) < 0.50:
        ln_f = float(rng.uniform(math.log(float(F_MIN_H456)), math.log(float(hi_f))))
        f_use = int(round(math.exp(ln_f)))
    else:
        f_use = int(rng.integers(F_MIN_H456, hi_f + 1))
    f_use = max(F_MIN_H456, min(hi_f, f_use))

    u_mode = float(rng.random())
    lo_p = max(0.04, float(p_ctx_min))
    hi_p = min(0.97, max(lo_p + 0.05, float(p_ctx_max)))
    if u_mode < 0.25:
        p_ctx = float(rng.uniform(lo_p, min(0.25, hi_p)))
    elif u_mode < 0.70:
        p_ctx = float(rng.uniform(min(0.25, hi_p), min(0.80, hi_p)))
    else:
        p_ctx = float(rng.uniform(min(0.80, hi_p), hi_p))
    n_ctx = max(8, min(n_use - 4, int(round(p_ctx * float(n_use)))))

    bsz = compute_oom_safe_batch_size(n_use, f_use, tokens_cap, row_tokens_cap, attn_cap, bmax)
    return bsz, n_use, f_use, n_ctx


def sample_rows_multiregime_h456(
    tab_f64: np.ndarray,
    n_use: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Draws `n_use` rows fresh from full `tab_f64` via 4 regimes (returns `(sub_tab, regime_id)`)."""
    n_raw = int(tab_f64.shape[0])
    p_raw = max(1, int(tab_f64.shape[1]) - 1)
    if n_raw <= n_use:
        perm = rng.permutation(n_raw)
        if n_raw == n_use:
            return np.array(tab_f64[perm], dtype=np.float64, copy=True), 0
        extra = rng.integers(0, n_raw, size=n_use - n_raw)
        idx = rng.permutation(np.concatenate([perm, extra]))
        return np.array(tab_f64[idx], dtype=np.float64, copy=True), 0

    u_reg = float(rng.random())
    if u_reg < 0.40:
        idx = rng.choice(n_raw, size=n_use, replace=False)
        return np.array(tab_f64[rng.permutation(idx)], dtype=np.float64, copy=True), 0

    if u_reg < 0.70:
        # Stratified quantile draw along a random raw feature column (unsupervised before split)
        col_j = int(rng.integers(0, p_raw))
        vals = np.nan_to_num(tab_f64[:, col_j], nan=0.0, posinf=0.0, neginf=0.0)
        order = np.argsort(vals + 1e-12 * rng.standard_normal(n_raw), kind="mergesort")
        n_strata = int(rng.integers(4, 17))
        splits = np.array_split(order, min(n_strata, n_use))
        chosen: list[np.ndarray] = []
        rem = n_use
        for si, bucket in enumerate(splits):
            take = rem if si == len(splits) - 1 else max(1, int(round(rem / float(len(splits) - si))))
            take = min(take, int(bucket.size), rem)
            if take > 0:
                chosen.append(rng.choice(bucket, size=take, replace=False))
                rem -= take
        idx = np.concatenate(chosen) if chosen else rng.choice(n_raw, size=n_use, replace=False)
        if idx.size < n_use:
            pool = np.setdiff1d(np.arange(n_raw, dtype=np.int64), idx, assume_unique=False)
            idx = np.concatenate([idx, rng.choice(pool, size=n_use - idx.size, replace=False)])
        return np.array(tab_f64[rng.permutation(idx[:n_use])], dtype=np.float64, copy=True), 1

    if u_reg < 0.85:
        # Covariate-projection window/tail slice (`u = X @ w_rand`), then exchangeably permuted
        k_proj = min(p_raw, 8)
        cols = rng.choice(p_raw, size=k_proj, replace=False)
        X_sub = np.nan_to_num(tab_f64[:, cols], nan=0.0, posinf=0.0, neginf=0.0)
        med = np.median(X_sub, axis=0)
        mad = np.maximum(np.median(np.abs(X_sub - med), axis=0), 1e-6)
        proj = ((X_sub - med) / mad) @ rng.standard_normal(k_proj)
        order = np.argsort(proj, kind="mergesort")
        win = min(n_raw, max(n_use, int(round(float(rng.uniform(1.05, 1.55)) * float(n_use)))))
        start = int(rng.integers(0, max(1, n_raw - win + 1)))
        window_idx = order[start : start + win]
        idx = rng.choice(window_idx, size=n_use, replace=False)
        return np.array(tab_f64[rng.permutation(idx)], dtype=np.float64, copy=True), 2

    # Soft leverage / density-tilted subsample (`w_i propto (1 + mean(z_i^2))^alpha`)
    k_lev = min(p_raw, 8)
    cols = rng.choice(p_raw, size=k_lev, replace=False)
    X_sub = np.nan_to_num(tab_f64[:, cols], nan=0.0, posinf=0.0, neginf=0.0)
    med = np.median(X_sub, axis=0)
    mad = np.maximum(np.median(np.abs(X_sub - med), axis=0), 1e-6)
    z2 = np.mean(np.clip(((X_sub - med) / mad) ** 2, 0.0, 64.0), axis=1)
    alpha = float(rng.uniform(-0.50, 0.50))
    weights = np.power(1.0 + z2, alpha)
    weights /= float(np.sum(weights))
    idx = rng.choice(n_raw, size=n_use, replace=False, p=weights)
    return np.array(tab_f64[rng.permutation(idx)], dtype=np.float64, copy=True), 3


def _synth_oof_ridge_gcv(Z_std: np.ndarray, yc_std: np.ndarray, n_ctx: int, rng: np.random.Generator) -> np.ndarray:
    """Family 18: K=4 internal OOF multi-feature Ridge-GCV on `(Xc, yc)`, out-of-sample on `Xq`."""
    p = Z_std.shape[1]
    k_use = min(p, int(rng.integers(2, min(10, p) + 1))) if p >= 2 else 1
    cols = rng.choice(p, size=k_use, replace=False)
    Phi = Z_std[:, cols]
    Phic = Phi[:n_ctx]
    perm = rng.permutation(n_ctx)
    n_folds = min(4, max(2, n_ctx // 4))
    folds = np.array_split(perm, n_folds)
    alphas = (0.1, 1.0, 10.0, 50.0)
    eye = np.eye(k_use, dtype=np.float64)
    best_a = 1.0
    best_mse = float("inf")
    for a_val in alphas:
        oof_try = np.zeros(n_ctx, dtype=np.float64)
        for va in folds:
            mask = np.ones(n_ctx, dtype=bool)
            mask[va] = False
            Xt = Phic[mask]
            yt = yc_std[mask]
            w = np.linalg.solve(Xt.T @ Xt + a_val * eye, Xt.T @ yt)
            oof_try[va] = Phic[va] @ w
        mse = float(np.mean((yc_std - oof_try) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_a = a_val
    feat = np.zeros(Z_std.shape[0], dtype=np.float64)
    for va in folds:
        mask = np.ones(n_ctx, dtype=bool)
        mask[va] = False
        Xt = Phic[mask]
        yt = yc_std[mask]
        w = np.linalg.solve(Xt.T @ Xt + best_a * eye, Xt.T @ yt)
        feat[va] = Phic[va] @ w
    w_full = np.linalg.solve(Phic.T @ Phic + best_a * eye, Phic.T @ yc_std)
    feat[n_ctx:] = Phi[n_ctx:] @ w_full
    return feat


def _synth_oof_oblivious_stumps(Z_std: np.ndarray, yc_std: np.ndarray, n_ctx: int, rng: np.random.Generator) -> np.ndarray:
    """Family 19: Depth-2 random oblivious decision-stump ensemble (`K=4` OOF on `Xc, yc`)."""
    p = Z_std.shape[1]
    n_tot = Z_std.shape[0]
    n_trees = 4
    perm = rng.permutation(n_ctx)
    folds = np.array_split(perm, min(4, max(2, n_ctx // 4)))
    acc = np.zeros(n_tot, dtype=np.float64)
    for _ in range(n_trees):
        c0 = int(rng.integers(0, p))
        c1 = int(rng.integers(0, p))
        t0 = float(np.quantile(Z_std[:n_ctx, c0], float(rng.uniform(0.20, 0.80))))
        t1 = float(np.quantile(Z_std[:n_ctx, c1], float(rng.uniform(0.20, 0.80))))
        leaf = ((Z_std[:, c0] > t0).astype(np.int64) << 1) | (Z_std[:, c1] > t1).astype(np.int64)
        leaf_c = leaf[:n_ctx]
        tree_out = np.zeros(n_tot, dtype=np.float64)
        smooth = 4.0
        for va in folds:
            mask = np.ones(n_ctx, dtype=bool)
            mask[va] = False
            lt = leaf_c[mask]
            yt = yc_std[mask]
            g_mu = float(np.mean(yt)) if yt.size > 0 else 0.0
            means = np.array(
                [
                    (float(np.sum(yt[lt == lv])) + smooth * g_mu) / (float(np.sum(lt == lv)) + smooth)
                    for lv in range(4)
                ],
                dtype=np.float64,
            )
            tree_out[va] = means[leaf_c[va]]
        g_all = float(np.mean(yc_std))
        means_full = np.array(
            [
                (float(np.sum(yc_std[leaf_c == lv])) + smooth * g_all) / (float(np.sum(leaf_c == lv)) + smooth)
                for lv in range(4)
            ],
            dtype=np.float64,
        )
        tree_out[n_ctx:] = means_full[leaf[n_ctx:]]
        acc += tree_out / float(n_trees)
    return acc


def _eval_single_synth_family_h456(
    fam: int,
    Z_std: np.ndarray,
    X_imp: np.ndarray,
    yc_std: np.ndarray,
    n_ctx: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Computes one synthetic/distractor column (`fam in 0..27`) with `0.0` query leakage."""
    n_total, p = Z_std.shape
    ja = int(rng.integers(0, p))
    jb = int(rng.integers(0, p))
    jc = int(rng.integers(0, p))
    za, zb, zc = Z_std[:, ja], Z_std[:, jb], Z_std[:, jc]

    if fam == 0:
        return za * zb
    if fam == 1:
        tau = float(rng.uniform(0.15, 1.0))
        return za / (np.abs(zb) + tau)
    if fam == 2:
        return za - zb
    if fam == 3:
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        return math.cos(theta) * za + math.sin(theta) * zb
    if fam == 4:
        return np.sign(za) * np.sqrt(np.abs(za) + 1e-12)
    if fam == 5:
        return np.sign(za) * np.log1p(np.abs(za))
    if fam == 6:
        z2 = za * za
        return z2 - float(np.mean(z2[:n_ctx]))
    if fam == 7:
        return np.sqrt(za * za + zb * zb + 1e-12)
    if fam == 8:
        return np.maximum(za, zb) if float(rng.random()) < 0.5 else np.minimum(za, zb)
    if fam == 9:
        return za / (1.0 + np.exp(-np.clip(zb, -15.0, 15.0)))
    if fam in (10, 11, 21):
        csa = np.sort(X_imp[:n_ctx, ja])
        rk_a = (np.searchsorted(csa, X_imp[:, ja], side="left") + 0.5) / float(n_ctx + 1)
        if fam == 10:
            return rk_a
        ga = ndtri(np.clip(rk_a, 1e-5, 1.0 - 1e-5))
        if fam == 11:
            return ga
        csb = np.sort(X_imp[:n_ctx, jb])
        rk_b = (np.searchsorted(csb, X_imp[:, jb], side="left") + 0.5) / float(n_ctx + 1)
        return ga - ndtri(np.clip(rk_b, 1e-5, 1.0 - 1e-5))
    if fam == 12:
        n_bins = int(rng.integers(6, 20))
        qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
        edges = np.unique(np.quantile(X_imp[:n_ctx, ja], qs))
        b_idx = np.searchsorted(edges, X_imp[:, ja], side="right")
        cnts = np.bincount(b_idx[:n_ctx], minlength=len(edges) + 2)
        return np.log1p(cnts[b_idx].astype(np.float64))
    if fam == 13:
        return (za + zb) * zc
    if fam == 14:
        n_bins = int(rng.integers(4, 12))
        edges = np.unique(np.quantile(X_imp[:n_ctx, ja], np.linspace(0.0, 1.0, n_bins + 1)[1:-1]))
        nb = len(edges) + 1
        b_all = np.searchsorted(edges, X_imp[:, ja], side="right")
        b_ctx = b_all[:n_ctx]
        feat = np.zeros(n_total, dtype=np.float64)
        perm = rng.permutation(n_ctx)
        folds = np.array_split(perm, min(5, max(2, n_ctx // 4)))
        sm = 5.0
        for va in folds:
            tr_m = np.ones(n_ctx, dtype=bool)
            tr_m[va] = False
            bt, yt = b_ctx[tr_m], yc_std[tr_m]
            g_mu = float(np.mean(yt)) if yt.size else 0.0
            b_sum = np.bincount(bt, weights=yt, minlength=nb)
            b_cnt = np.bincount(bt, minlength=nb).astype(np.float64)
            feat[va] = ((b_sum + sm * g_mu) / (b_cnt + sm))[b_ctx[va]]
        b_sum_f = np.bincount(b_ctx, weights=yc_std, minlength=nb)
        b_cnt_f = np.bincount(b_ctx, minlength=nb).astype(np.float64)
        feat[n_ctx:] = ((b_sum_f + sm * float(np.mean(yc_std))) / (b_cnt_f + sm))[b_all[n_ctx:]]
        return feat
    if fam == 15:
        Phi = np.column_stack([za, za * za - 1.0])
        Phic = Phi[:n_ctx]
        w = np.linalg.solve(Phic.T @ Phic + 2.0 * np.eye(2), Phic.T @ yc_std)
        return Phi @ w
    if fam == 16:
        w1 = float(rng.normal(0.0, 1.35))
        w2 = float(rng.normal(0.0, 1.35))
        ph = float(rng.uniform(0.0, 2.0 * math.pi))
        return np.cos(w1 * za + w2 * zb + ph)
    if fam == 17:
        k_sub = min(p, int(rng.integers(2, 7))) if p >= 2 else 1
        sub_c = rng.choice(p, size=k_sub, replace=False)
        mu_c = np.mean(Z_std[:n_ctx, sub_c], axis=0, keepdims=True)
        return np.sqrt(np.mean((Z_std[:, sub_c] - mu_c) ** 2, axis=1) + 1e-12)
    if fam == 18:
        return _synth_oof_ridge_gcv(Z_std, yc_std, n_ctx, rng)
    if fam == 19:
        return _synth_oof_oblivious_stumps(Z_std, yc_std, n_ctx, rng)
    if fam == 20:
        # Piecewise-linear context quantile hinge spline basis
        knots = np.quantile(za[:n_ctx], [0.20, 0.50, 0.80])
        wts = rng.normal(0.0, 1.0, size=3)
        return wts[0] * np.maximum(0.0, za - knots[0]) + wts[1] * np.maximum(0.0, za - knots[1]) + wts[2] * np.maximum(0.0, knots[2] - za)
    if fam == 22:
        return rng.standard_normal(n_total)
    if fam == 23:
        df = int(rng.choice([2, 3, 5]))
        return np.clip(rng.standard_t(df, size=n_total), -18.0, 18.0)
    if fam == 24:
        k_cat = int(rng.integers(2, 16))
        return rng.integers(0, k_cat, size=n_total).astype(np.float64)
    if fam == 25:
        rho = float(rng.uniform(0.30, 0.90))
        return rho * za + math.sqrt(max(0.0, 1.0 - rho * rho)) * rng.standard_normal(n_total)
    if fam == 26:
        # Adversarial spurious context-overfit / pseudo-ID distractor (`0.0` query `y` touch!)
        base_noise = rng.standard_normal(n_total)
        if float(rng.random()) < 0.65:
            # Leak tiny slice of yc ONLY on 15% of context rows -> zero population signal on queries
            k_spur = max(2, int(round(0.15 * float(n_ctx))))
            spur_idx = rng.choice(n_ctx, size=k_spur, replace=False)
            amp = float(rng.uniform(0.35, 0.85))
            base_noise[spur_idx] += amp * yc_std[spur_idx]
            return base_noise
        # High-cardinality pseudo-ID distractor
        return ((rng.integers(0, 1_000_003, size=n_total) % 997).astype(np.float64) - 498.0) / 288.0
    # Family 27: MNAR censored twin + missingness indicator (thresholds strictly from Xc)
    q_cut = float(np.quantile(za[:n_ctx], float(rng.uniform(0.55, 0.90))))
    upper = float(rng.random()) < 0.5
    mask = (za > q_cut) if upper else (za < -abs(q_cut))
    if float(rng.random()) < 0.45:
        return mask.astype(np.float64)
    med_c = float(np.median(za[:n_ctx]))
    noise = 0.15 * rng.standard_normal(n_total)
    return np.where(mask, med_c + noise, za)


def generate_28_context_synth_h456(
    X_all: np.ndarray,
    y_ctx: np.ndarray,
    n_ctx: int,
    k_inj: int,
    rng: np.random.Generator,
    distractor_prob: float = 0.35,
    forced_families: list[int] | None = None,
) -> np.ndarray:
    """Generates `k_inj` synthetic/distractor columns strictly fitted on `(X_all[:n_ctx], y_ctx[:n_ctx])`."""
    n_total, p = X_all.shape
    if k_inj <= 0 or p <= 0 or n_ctx < 8:
        return np.zeros((n_total, 0), dtype=np.float64)

    X_imp, _, _ = _ctx_impute_matrix(X_all, n_ctx)
    Z_std = np.column_stack([_ctx_standardize_col(X_imp[:, j], n_ctx) for j in range(p)])
    yc = _pointwise_clean(np.asarray(y_ctx[:n_ctx], dtype=np.float64), cap=1e6)
    yc_sd = float(np.std(yc))
    yc_std = (yc - float(np.mean(yc))) / yc_sd if yc_sd > 1e-9 else np.zeros(n_ctx, dtype=np.float64)

    cols_out: list[np.ndarray] = []
    for idx_k in range(k_inj):
        if forced_families is not None and idx_k < len(forced_families):
            fam = int(forced_families[idx_k])
        elif float(rng.random()) < float(distractor_prob):
            fam = int(rng.choice(DISTRACTOR_FAMILIES_H456))
        else:
            fam = int(rng.choice(CLEAN_FAMILIES_H456))
        raw_col = _eval_single_synth_family_h456(fam, Z_std, X_imp, yc_std, n_ctx, rng)
        cols_out.append(_ctx_standardize_col(_pointwise_clean(raw_col, cap=50.0), n_ctx))

    return np.column_stack(cols_out)


def _select_unbiased_or_rank_cols(
    X_all: np.ndarray,
    y_ctx: np.ndarray,
    n_ctx: int,
    k_keep: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """70% pure uniform random column subsetting + 30% noisy Spearman-like rank screen on `(Xc, yc)`."""
    p = int(X_all.shape[1])
    if p <= k_keep:
        return np.arange(p, dtype=np.int64)
    if float(rng.random()) < 0.70:
        return np.sort(rng.choice(p, size=k_keep, replace=False)).astype(np.int64)
    Xc = X_all[:n_ctx]
    yc = np.asarray(y_ctx[:n_ctx], dtype=np.float64)
    yr = np.argsort(np.argsort(yc)).astype(np.float64)
    yr -= float(np.mean(yr))
    yn = max(float(np.linalg.norm(yr)), 1e-12)
    scores = np.zeros(p, dtype=np.float64)
    for j in range(p):
        cj = np.nan_to_num(Xc[:, j], nan=0.0, posinf=0.0, neginf=0.0)
        cr = np.argsort(np.argsort(cj)).astype(np.float64)
        cr -= float(np.mean(cr))
        cn = float(np.linalg.norm(cr))
        corr = abs(float(np.dot(cr, yr)) / (cn * yn)) if cn > 1e-12 else 0.0
        scores[j] = corr + float(rng.uniform(0.0, 0.35))
    return np.sort(np.argsort(-scores)[:k_keep]).astype(np.int64)


def remove_and_inject_features_h456(
    X_raw: np.ndarray,
    y_ctx: np.ndarray,
    n_ctx: int,
    f_target: int,
    rng: np.random.Generator,
    drop_min_frac: float = 0.05,
    drop_max_frac: float = 0.90,
    inj_min_frac: float = 0.10,
    inj_max_frac: float = 0.70,
    distractor_prob: float = 0.35,
    anchor_full_prob: float = 0.10,
    zero_inj_prob: float = 0.15,
) -> tuple[np.ndarray, dict[str, int]]:
    """Simultaneous raw feature removal + mandatory reserved synthetic/distractor injection (including wide tables)."""
    n_tot, p_raw = X_raw.shape
    f_target = max(1, int(f_target))

    # 1. Reserve `k_inj` BEFORE raw column selection so wide tables (`p_raw >= f_target`) also get synth/distractors!
    if float(rng.random()) < float(zero_inj_prob) or f_target <= 1:
        k_reserve = 0
    else:
        lo_i = max(0.0, min(0.85, float(inj_min_frac)))
        hi_i = max(lo_i, min(0.85, float(inj_max_frac)))
        frac_inj = float(rng.uniform(lo_i, hi_i))
        k_reserve = max(1, min(f_target - 1, int(round(frac_inj * float(f_target)))))

    raw_budget = max(1, f_target - k_reserve)

    # 2. Apply raw feature removal (`drop_frac in [drop_min_frac, drop_max_frac]`)
    min_keep = 1 if float(rng.random()) < 0.20 else min(2, p_raw)
    if p_raw <= min_keep or float(rng.random()) < float(anchor_full_prob):
        n_survive = min(p_raw, raw_budget)
    else:
        lo_d = max(0.0, min(0.92, float(drop_min_frac)))
        hi_d = max(lo_d, min(0.92, float(drop_max_frac)))
        drop_f = float(rng.uniform(lo_d, hi_d))
        n_after_drop = max(min_keep, int(round(float(p_raw) * (1.0 - drop_f))))
        n_survive = min(raw_budget, n_after_drop)

    kept_idx = _select_unbiased_or_rank_cols(X_raw, y_ctx, n_ctx, n_survive, rng)
    X_kept = np.array(X_raw[:, kept_idx], dtype=np.float64, copy=True)

    # 3. Fill remaining `f_target - n_survive` columns from the 28-family zero-leakage context bank
    k_inj = max(0, f_target - int(X_kept.shape[1]))
    if k_inj > 0:
        X_syn = generate_28_context_synth_h456(
            X_kept,
            y_ctx[:n_ctx],
            n_ctx,
            k_inj=k_inj,
            rng=rng,
            distractor_prob=distractor_prob,
        )
        X_comb = np.hstack([X_kept, X_syn])
    else:
        X_comb = X_kept

    perm = rng.permutation(f_target)
    return X_comb[:, perm], {"p_raw": p_raw, "n_kept_raw": int(X_kept.shape[1]), "k_injected": int(k_inj), "f_out": f_target}


def apply_permute_scale_noise_warps_h456(
    X_in: np.ndarray,
    n_ctx: int,
    rng: np.random.Generator,
    noise_max: float = 0.12,
    scale_log_max: float = 4.0,
) -> np.ndarray:
    """Applies context-fitted (`X[:n_ctx]`) sign flips, heavy-scale/power/quantize warps, Cauchy spikes & correlated noise."""
    X_imp, med_c, iqr_c = _ctx_impute_matrix(X_in, n_ctx)
    n_tot, p = X_imp.shape
    Z = _pointwise_clean((X_imp - med_c[None, :]) / iqr_c[None, :], cap=30.0)

    # 1. Monotone signed-power warp (`|z|^gamma * sign(z)`) on ~35% of columns
    pow_mask = rng.random(p) < 0.35
    if np.any(pow_mask):
        gammas = np.exp(rng.uniform(math.log(0.35), math.log(2.50), size=p))
        for j in np.flatnonzero(pow_mask).tolist():
            Z[:, j] = np.sign(Z[:, j]) * np.power(np.abs(Z[:, j]) + 1e-12, float(gammas[j]))

    # 2. Random ordinal quantization into K in [3..32] `Xc`-quantile levels on ~20% of continuous columns
    quant_mask = rng.random(p) < 0.20
    if np.any(quant_mask):
        for j in np.flatnonzero(quant_mask).tolist():
            k_lv = int(rng.integers(3, 33))
            edges = np.unique(np.quantile(Z[:n_ctx, j], np.linspace(0.0, 1.0, k_lv + 1)[1:-1]))
            if edges.size >= 2:
                bins = np.searchsorted(edges, Z[:, j], side="right").astype(np.float64)
                Z[:, j] = (bins - 0.5 * float(edges.size)) / max(float(edges.size), 1.0) * 2.5

    # 3. Sparse 2% Cauchy cell outlier injections bounded at +-12 context IQR units
    if float(rng.random()) < 0.45:
        spike_mask = rng.random((n_tot, p)) < 0.02
        spikes = np.clip(rng.standard_cauchy(size=(n_tot, p)), -12.0, 12.0)
        Z = np.where(spike_mask, Z + 0.50 * spikes, Z)

    # 4. Heteroscedastic + rank-1 correlated feature noise calibrated to `Z[:n_ctx]` IQR (`0.0` query leakage)
    if float(noise_max) > 0.0:
        z_iqr = np.ones(p, dtype=np.float64)
        for j in range(p):
            q25, q75 = np.percentile(Z[:n_ctx, j], [25.0, 75.0])
            sp = float(q75 - q25)
            z_iqr[j] = sp if (sp >= 1e-6 and math.isfinite(sp)) else 1.0
        active = (rng.random(p) < 0.55).astype(np.float64)
        sigmas = rng.uniform(0.005, float(noise_max), size=p) * z_iqr * active
        rho = float(rng.uniform(0.0, 0.55))
        eps_ind = rng.standard_normal((n_tot, p))
        z_shared = rng.standard_normal((n_tot, 1))
        v_load = rng.choice([-1.0, 1.0], size=(1, p))
        corr_eps = math.sqrt(max(0.0, 1.0 - rho * rho)) * eps_ind + rho * (z_shared @ v_load)
        Z = Z + corr_eps * sigmas[None, :]

    # 5. Random 50% sign flips + heavy raw affine scale `exp(U(-scale_log_max, +scale_log_max))` & shift
    signs = rng.choice([-1.0, 1.0], size=p)
    scales = np.exp(rng.uniform(-float(scale_log_max), float(scale_log_max), size=p))
    shifts = rng.normal(0.0, 3.0, size=p) * scales
    out = _pointwise_clean(Z, cap=40.0) * (signs * scales)[None, :] + shifts[None, :]
    perm = rng.permutation(p)
    return out[:, perm]


def warp_target_monotone_h456(
    y_all: np.ndarray,
    n_ctx: int,
    rng: np.random.Generator,
    warp_prob: float = 0.45,
) -> np.ndarray:
    """Applies a strictly monotone target warp anchored ONLY on `y_all[:n_ctx]` (`0.0` query leakage)."""
    y_clean = _pointwise_clean(np.asarray(y_all, dtype=np.float64), cap=1e6)
    if float(rng.random()) >= float(warp_prob):
        return y_clean.copy()
    yc = y_clean[:n_ctx]
    med = float(np.median(yc))
    q25, q75 = np.percentile(yc, [25.0, 75.0])
    scale = float(q75 - q25)
    if scale < 1e-9:
        scale = float(np.std(yc))
    if scale < 1e-9 or not math.isfinite(scale):
        return y_clean.copy()

    z = _pointwise_clean((y_clean - med) / scale, cap=50.0)
    mode = int(rng.integers(0, 4))
    if mode == 0:
        # Log-modulus monotone warp
        alpha = float(rng.uniform(0.35, 2.50))
        out = np.sign(z) * np.log1p(alpha * np.abs(z)) / alpha
    elif mode == 1:
        # Signed power monotone warp
        gamma = float(math.exp(rng.uniform(math.log(0.45), math.log(1.85))))
        out = np.sign(z) * np.power(np.abs(z) + 1e-9, gamma)
    elif mode == 2:
        # Asymmetric Yeo-Johnson-style monotone warp (`pos_g` for z>=0, `neg_g` for z<0)
        pos_g = float(rng.uniform(0.50, 1.75))
        neg_g = float(rng.uniform(0.50, 1.75))
        z_pos = np.maximum(z, 0.0)
        z_neg = np.maximum(-z, 0.0)
        out = np.where(z >= 0.0, np.power(1.0 + z_pos, pos_g) - 1.0, -(np.power(1.0 + z_neg, neg_g) - 1.0))
    else:
        # Context-knot piecewise-linear rank-logit warp (`interp` strictly against sorted `z[:n_ctx]`!)
        zc_sorted = np.unique(np.sort(z[:n_ctx]))
        if zc_sorted.size < 4:
            return z
        ranks_u = (np.arange(zc_sorted.size, dtype=np.float64) + 0.5) / float(zc_sorted.size)
        yk = ndtri(np.clip(ranks_u, 1e-3, 1.0 - 1e-3))
        out = np.interp(z, zc_sorted, yk)
        lo_s = float((yk[1] - yk[0]) / max(zc_sorted[1] - zc_sorted[0], 1e-6))
        hi_s = float((yk[-1] - yk[-2]) / max(zc_sorted[-1] - zc_sorted[-2], 1e-6))
        out = np.where(z < zc_sorted[0], yk[0] + max(lo_s, 0.05) * (z - zc_sorted[0]), out)
        out = np.where(z > zc_sorted[-1], yk[-1] + max(hi_s, 0.05) * (z - zc_sorted[-1]), out)
    if float(rng.random()) < 0.50:
        out = -out
    return _pointwise_clean(out, cap=50.0)


def sample_fid_slots_h456(f_use: int, fid_slots: int, seed: int, device: torch.device) -> torch.Tensor:
    """3-mode `FID_SLOTS` permutation (uniform `randperm`, strided modular, block-shifted)."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed) & 0x7FFFFFFF)
    mode = int(torch.randint(0, 3, (1,), generator=gen).item())
    if mode == 0 or f_use >= fid_slots:
        return torch.randperm(fid_slots, generator=gen)[:f_use].to(device=device)
    if mode == 1:
        start = int(torch.randint(0, fid_slots, (1,), generator=gen).item())
        stride = int(torch.randint(1, max(2, fid_slots // max(f_use, 1) + 1), (1,), generator=gen).item())
        while math.gcd(stride, fid_slots) != 1:
            stride += 1
        idx = (start + stride * torch.arange(f_use, dtype=torch.long)) % fid_slots
        return idx[torch.randperm(f_use, generator=gen)].to(device=device)
    offset = int(torch.randint(0, max(1, fid_slots - f_use + 1), (1,), generator=gen).item())
    blk = torch.arange(offset, offset + f_use, dtype=torch.long)
    return blk[torch.randperm(f_use, generator=gen)].to(device=device)


def apply_postnorm_affine_h456(
    x_norm: torch.Tensor,
    seed: int,
    affine_max: float = 0.45,
    shift_max: float = 0.35,
) -> torch.Tensor:
    """Applies wider post-prenorm channelwise affine jitter (`g ~ exp(U(-0.45, 0.45))`, `d ~ U(-0.35, 0.35)`)."""
    if affine_max <= 0.0 or x_norm.ndim != 4:
        return x_norm
    bsz, _, f_cnt, _ = x_norm.shape
    gen = torch.Generator(device="cpu").manual_seed(int(seed) & 0x7FFFFFFF)
    log_g = (torch.rand((bsz, 1, f_cnt, 1), generator=gen, dtype=torch.float32) * 2.0 - 1.0) * float(affine_max)
    sh = (torch.rand((bsz, 1, f_cnt, 1), generator=gen, dtype=torch.float32) * 2.0 - 1.0) * float(shift_max)
    g_dev = torch.exp(log_g).to(device=x_norm.device, dtype=x_norm.dtype)
    d_dev = sh.to(device=x_norm.device, dtype=x_norm.dtype)
    out = x_norm.clone()
    out[..., 0:1] = out[..., 0:1] * g_dev + d_dev
    return out
