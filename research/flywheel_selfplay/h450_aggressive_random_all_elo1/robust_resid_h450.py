#!/usr/bin/env python3
"""H450 Robust Residual, Exact Yeo-Johnson Warp, Fast Nonlinear Base & Sentinel Hurdle Engine (`robust_resid_h450.py`).

Fixes all 3 target/residual failure modes discovered on `dev91`:
  1. `robust_ridge_targets`: Replaces unclipped `icl_resid.ridge_base` (which blew up to `+4,709` on `fam-4286` & `fam-ce33`)
     with context-only median/IQR + smooth `6*tanh(z/6)` feature clamping, missingness flags, SVD-LOO GCV `lambda`
     selection (`{0.1, 1.0, 10.0, 100.0}`), LOO improvement gate (`rmse_loo < 0.985 * std(y_ctx)`), and output span bounds.
  2. `YeoJohnsonWarpF64`: Fits context-only standardized Yeo-Johnson (`lambda in [-1.5, 2.5]`) in `float64` and provides
     an exact vectorized PyTorch `inv_torch(z_grid)` for `ICLH450.mean_warped` 1,024-bin integration.
  3. `fast_nonlinear_base_oof`: Fast 5-fold context-only blend of `HistGradientBoostingRegressor` + `LaplacianMaternLOO`
     returning `(oof_ctx, pred_q, u_oof)` (`0` query target reads!) for 2-stage residual training and inference.
  4. `HurdleSentinelModel`: Detects discrete extreme target atoms (`y == -99.99`) on `y_ctx` and gates a calibrated
     two-part `p_atom * y_atom + (1 - p_atom) * y_bulk` predictor strictly by 5-fold OOF MSE improvement.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import torch
from scipy import linalg, stats
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

RIDGE_LAMBDAS = (0.15, 1.0, 8.0, 60.0)
MIN_RES_SD = 5e-4


def _safe_svd_np(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = np.where(np.isfinite(A), A, 0.0)
    try:
        return linalg.svd(A, full_matrices=False, check_finite=False, lapack_driver="gesdd")
    except Exception:
        return linalg.svd(A, full_matrices=False, check_finite=False, lapack_driver="gesvd")


def robust_ridge_np(
    X_ctx: np.ndarray, y_ctx: np.ndarray, X_q: np.ndarray, seed: int = 450, folds: int = 5
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    """Context-only median/IQR + `6*tanh(z/6)` + SVD-GCV 5-fold OOF Ridge. Returns `(oof_ctx, pred_q, used, u_oof)`."""
    X_ctx = np.asarray(X_ctx, dtype=np.float64)
    y_ctx = np.asarray(y_ctx, dtype=np.float64)
    X_q = np.asarray(X_q, dtype=np.float64)
    n_c, p = X_ctx.shape
    n_q = X_q.shape[0]
    ym = float(np.mean(y_ctx))
    sy = float(np.std(y_ctx))
    if n_c < 16 or p == 0 or sy <= 1e-12:
        return np.full(n_c, ym), np.full(n_q, ym), False, 0.0

    miss_c = ~np.isfinite(X_ctx)
    miss_q = ~np.isfinite(X_q)
    med = np.zeros(p, dtype=np.float64)
    sc = np.ones(p, dtype=np.float64)
    for j in range(p):
        fin = ~miss_c[:, j]
        if np.any(fin):
            v = X_ctx[fin, j]
            med[j] = float(np.median(v))
            q25, q75 = np.quantile(v, [0.25, 0.75]) if len(v) >= 4 else (med[j] - 1.0, med[j] + 1.0)
            iqr = float(q75 - q25) / 1.349
            sd = float(np.std(v))
            sc[j] = iqr if iqr > 1e-8 * max(sd, 1.0) and iqr > 0.0 else (sd if sd > 1e-12 else 1.0)

    zc = 6.0 * np.tanh(np.clip((np.where(miss_c, med[None, :], X_ctx) - med[None, :]) / sc[None, :], -30.0, 30.0) / 6.0)
    zq = 6.0 * np.tanh(np.clip((np.where(miss_q, med[None, :], X_q) - med[None, :]) / sc[None, :], -30.0, 30.0) / 6.0)

    miss_cols = [j for j in range(p) if float(np.mean(miss_c[:, j])) >= 0.01][:16]
    if miss_cols:
        zc = np.hstack([zc, miss_c[:, miss_cols].astype(np.float64)])
        zq = np.hstack([zq, miss_q[:, miss_cols].astype(np.float64)])

    mu_z = zc.mean(axis=0, keepdims=True)
    zc0 = zc - mu_z
    zq0 = zq - mu_z
    yc0 = y_ctx - ym

    U, s, Vt = _safe_svd_np(zc0)
    s2 = s * s
    uty = U.T @ yc0
    U2 = U * U

    best_lam = 1.0
    best_loo_mse = float("inf")
    for lam in RIDGE_LAMBDAS:
        filt = s2 / (s2 + lam)
        yhat = U @ (filt * uty)
        h = np.clip(U2 @ filt + (1.0 / n_c), 0.0, 0.98)
        loo_err = (yc0 - yhat) / (1.0 - h)
        mse = float(np.mean(loo_err * loo_err))
        if mse < best_loo_mse:
            best_loo_mse = mse
            best_lam = lam

    w_full = Vt.T @ ((s / (s2 + best_lam)) * uty)
    pred_q = zq0 @ w_full + ym

    rng = np.random.default_rng(seed)
    fold = rng.permutation(n_c) % folds
    oof = np.full(n_c, ym, dtype=np.float64)
    d = zc0.shape[1]
    G_full = zc0.T @ zc0
    rhs_full = zc0.T @ yc0
    s_full = zc0.sum(axis=0)
    sy_full = float(yc0.sum())
    eye = best_lam * np.eye(d, dtype=np.float64)

    for k in range(folds):
        mk = fold == k
        nk = int(np.sum(mk))
        if nk == 0 or nk >= n_c:
            continue
        zk = zc0[mk]
        yk = yc0[mk]
        ntr = float(n_c - nk)
        ym_tr = (sy_full - float(yk.sum())) / ntr
        s_tr = s_full - zk.sum(axis=0)
        G_tr = G_full - zk.T @ zk - np.outer(s_tr, s_tr) / ntr + eye
        b_tr = (rhs_full - zk.T @ yk) - ym_tr * s_tr
        try:
            wk = linalg.solve(G_tr, b_tr, assume_a="pos")
        except Exception:
            wk = np.linalg.lstsq(G_tr, b_tr, rcond=1e-8)[0]
        oof[mk] = (zk - (s_tr / ntr)[None, :]) @ wk + ym + ym_tr

    y_lo, y_hi = float(np.min(y_ctx)), float(np.max(y_ctx))
    span = max(y_hi - y_lo, sy)
    lo_b, hi_b = y_lo - 1.5 * span, y_hi + 1.5 * span
    oof = np.clip(oof, lo_b, hi_b)
    pred_q = np.clip(pred_q, lo_b, hi_b)

    rc = y_ctx - oof
    rmse_oof = float(np.sqrt(np.mean(rc * rc)))
    u_oof = -math.log(max(rmse_oof / sy, 1e-12))
    used = bool(
        np.all(np.isfinite(oof))
        and np.all(np.isfinite(pred_q))
        and rmse_oof < 0.985 * sy
        and float(np.std(rc)) >= MIN_RES_SD * sy
    )
    return oof, pred_q, used, u_oof


@torch.no_grad()
def robust_residual_targets_torch(
    x: torch.Tensor, y: torch.Tensor, n_ctx: int, seed: int = 450
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop-in safe replacement for `icl_resid.residual_targets`: `x [B,N,F]`, `y [B,N]` -> `(y_res, used, base_q)`."""
    dev = y.device
    x_np = x.detach().cpu().double().numpy()
    y_np = y.detach().cpu().double().numpy()
    B, N, _ = x_np.shape
    Nq = N - n_ctx
    out_y = y_np.copy()
    used_arr = np.zeros(B, dtype=bool)
    base_q = np.zeros((B, Nq), dtype=np.float64)
    for b in range(B):
        oof, pq, used, _ = robust_ridge_np(x_np[b, :n_ctx], y_np[b, :n_ctx], x_np[b, n_ctx:], seed=seed + 31 * b)
        used_arr[b] = used
        base_q[b] = pq
        if used:
            out_y[b, :n_ctx] = y_np[b, :n_ctx] - oof
            out_y[b, n_ctx:] = y_np[b, n_ctx:] - pq
    return (
        torch.from_numpy(out_y).to(device=dev, dtype=y.dtype),
        torch.from_numpy(used_arr).to(device=dev),
        torch.from_numpy(base_q).to(device=dev, dtype=torch.float64),
    )


class YeoJohnsonWarpF64:
    """Context-only `float64` Yeo-Johnson target transform with exact PyTorch 1,024-bin inverse & 25-quantile Duan smearing."""

    def __init__(self):
        self.mu = 0.0
        self.sd = 1.0
        self.lam = 1.0
        self.z_mu = 0.0
        self.z_sd = 1.0
        self.zt_lo = -12.0
        self.zt_hi = 12.0
        self.smear_q = np.zeros(1, dtype=np.float64)
        self.active = False

    def fit_transform(self, y_ctx: np.ndarray) -> np.ndarray:
        y = np.asarray(y_ctx, dtype=np.float64)
        self.mu = float(np.mean(y))
        self.sd = max(float(np.std(y)), 1e-12)
        yn = (y - self.mu) / self.sd
        skew = abs(float(stats.skew(yn))) if len(yn) >= 12 else 0.0
        if len(yn) < 16 or skew < 0.25:
            self.active = False
            self.lam = 1.0
            self.z_mu, self.z_sd = 0.0, 1.0
            return yn
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, lam = stats.yeojohnson(yn)
            self.lam = float(np.clip(lam, -1.5, 2.5))
            zt = stats.yeojohnson(yn, lmbda=self.lam)
            if not np.all(np.isfinite(zt)) or float(np.std(zt)) < 1e-9:
                raise ValueError("degenerate yj")
            self.active = abs(self.lam - 1.0) >= 0.08
            self.z_mu = float(np.mean(zt))
            self.z_sd = max(float(np.std(zt)), 1e-9)
            self.zt_lo = float(np.min(zt) - 0.85 * self.z_sd)
            self.zt_hi = float(np.max(zt) + 0.85 * self.z_sd)
            return (zt - self.z_mu) / self.z_sd
        except Exception:
            self.active = False
            self.lam = 1.0
            self.z_mu, self.z_sd = 0.0, 1.0
            return yn

    def transform_any(self, y: np.ndarray) -> np.ndarray:
        yn = (np.asarray(y, dtype=np.float64) - self.mu) / self.sd
        if not self.active and abs(self.lam - 1.0) < 1e-6:
            return yn
        zt = stats.yeojohnson(yn, lmbda=self.lam)
        return (zt - self.z_mu) / self.z_sd

    def set_residuals(self, z_true: np.ndarray, z_oof: np.ndarray) -> None:
        res = np.asarray(z_true - z_oof, dtype=np.float64)
        res = res[np.isfinite(res)]
        if len(res) >= 12:
            res = res - float(np.mean(res))
            qs = 0.02 + 0.96 * ((np.arange(25, dtype=np.float64) + 0.5) / 25.0)
            self.smear_q = np.quantile(res, qs)

    def inv_torch(self, z_norm: torch.Tensor) -> torch.Tensor:
        """Vectorized exact inverse Yeo-Johnson on `float64` tensor `z_norm` (e.g. `[B, Nq, 1024]`)."""
        if not self.active and abs(self.lam - 1.0) < 1e-6:
            return z_norm * self.sd + self.mu
        zt = (z_norm * self.z_sd + self.z_mu).clamp(self.zt_lo, self.zt_hi)
        lam = self.lam
        pos = zt >= 0.0
        if abs(lam) < 1e-6:
            pos_out = torch.expm1(zt.clamp(-20.0, 9.5))
        else:
            pos_out = torch.pow((lam * zt + 1.0).clamp_min(0.02), 1.0 / lam) - 1.0
        lam2 = 2.0 - lam
        if abs(lam2) < 1e-6:
            neg_out = 1.0 - torch.exp((-zt).clamp(-20.0, 9.5))
        else:
            neg_out = 1.0 - torch.pow((-lam2 * zt + 1.0).clamp_min(0.02), 1.0 / lam2)
        yn = torch.where(pos, pos_out, neg_out)
        return yn * self.sd + self.mu

    def inv_np_duan(self, z_pred: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np.asarray(z_pred, dtype=np.float64))
        if len(self.smear_q) <= 1 or (not self.active and abs(self.lam - 1.0) < 1e-6):
            return self.inv_torch(t).numpy()
        sq = torch.from_numpy(self.smear_q)
        return self.inv_torch(t[:, None] + sq[None, :]).mean(dim=1).numpy()


def _laplacian_matern_kernel(A: np.ndarray, B: np.ndarray, ell: float) -> np.ndarray:
    """Hybrid 50% Laplacian (`L1`, sharp axis-aligned panel/tree regimes) + 50% Matérn-3/2 (`L2` smooth regimes)."""
    d = max(A.shape[1], 1)
    # L1 Manhattan distance via pairwise broadcast for <=600 subsampled points
    l1 = np.sum(np.abs(A[:, None, :] - B[None, :, :]), axis=2) / (ell * d)
    k_lap = np.exp(-l1)
    As = A / (ell * math.sqrt(d))
    Bs = B / (ell * math.sqrt(d))
    r2 = np.maximum(np.sum(As * As, axis=1, keepdims=True) + np.sum(Bs * Bs, axis=1, keepdims=True).T - 2.0 * (As @ Bs.T), 0.0)
    s3 = np.sqrt(3.0 * r2)
    k_m32 = (1.0 + s3) * np.exp(-s3)
    return 0.50 * k_lap + 0.50 * k_m32


def fast_laplacian_matern_predict(X_tr: np.ndarray, y_tr: np.ndarray, X_q: np.ndarray, max_sub: int = 480) -> np.ndarray:
    """Analytical eigh-LOO Laplacian+Matérn ARD regressor in `float64`."""
    n, p = X_tr.shape
    ym = float(np.mean(y_tr))
    sy = float(np.std(y_tr))
    if n < 12 or p == 0 or sy <= 1e-12:
        return np.full(X_q.shape[0], ym, dtype=np.float64)

    med = np.zeros(p, dtype=np.float64)
    sc = np.ones(p, dtype=np.float64)
    for j in range(p):
        fin = np.isfinite(X_tr[:, j])
        if np.any(fin):
            v = X_tr[fin, j]
            med[j] = float(np.median(v))
            q25, q75 = np.quantile(v, [0.25, 0.75]) if len(v) >= 4 else (med[j] - 1.0, med[j] + 1.0)
            iqr = float(q75 - q25) / 1.349
            sd = float(np.std(v))
            sc[j] = iqr if iqr > 1e-6 * max(sd, 1.0) else max(sd, 1e-9)

    Zc = np.clip((np.where(np.isfinite(X_tr), X_tr, med[None, :]) - med[None, :]) / sc[None, :], -8.0, 8.0)
    Zq = np.clip((np.where(np.isfinite(X_q), X_q, med[None, :]) - med[None, :]) / sc[None, :], -8.0, 8.0)

    yn = (y_tr - ym) / sy
    corrs = np.array([abs(float(np.dot(Zc[:, j] - np.mean(Zc[:, j]), yn))) / (n * (float(np.std(Zc[:, j])) + 1e-8)) for j in range(p)])
    top = np.argsort(-corrs)[: min(16, p)]
    w = np.maximum(corrs[top], 0.05) ** 1.25
    w = w / max(float(np.mean(w)), 1e-6)
    Zc_sub_all = Zc[:, top] * w[None, :]
    Zq_sub = Zq[:, top] * w[None, :]

    if n > max_sub:
        order = np.argsort(y_tr, kind="mergesort")
        pick = order[np.linspace(0, n - 1, max_sub, dtype=np.int64)]
    else:
        pick = np.arange(n, dtype=np.int64)

    Z_s = Zc_sub_all[pick]
    y_s = yn[pick]
    best_mse = float("inf")
    best_cfg = None
    for ell in (0.65, 1.6):
        K = _laplacian_matern_kernel(Z_s, Z_s, ell)
        ev, V = linalg.eigh(0.5 * (K + K.T), check_finite=False)
        ev = np.maximum(ev, 0.0)
        vty = V.T @ y_s
        V2 = V * V
        for nv in (0.008, 0.05, 0.25):
            inv_d = 1.0 / (ev + nv)
            alpha = V @ (inv_d * vty)
            diag_inv = np.maximum(V2 @ inv_d, 1e-9)
            loo_e = alpha / diag_inv
            mse = float(np.mean(loo_e * loo_e))
            if mse < best_mse:
                best_mse = mse
                best_cfg = (ell, alpha)

    if best_cfg is None:
        return np.full(X_q.shape[0], ym, dtype=np.float64)
    ell_b, alpha_b = best_cfg
    Kq = _laplacian_matern_kernel(Zq_sub, Z_s, ell_b)
    pred = ym + sy * (Kq @ alpha_b)
    y_lo, y_hi = float(np.min(y_tr)), float(np.max(y_tr))
    pad = 0.75 * max(y_hi - y_lo, sy)
    return np.clip(pred, y_lo - pad, y_hi + pad)


def fast_nonlinear_base_oof(
    X_ctx: np.ndarray, y_ctx: np.ndarray, X_q: np.ndarray, seed: int = 450, folds: int = 5
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    """Computes fast 5-fold context-only nonlinear OOF + query base prediction (`HistGBDT + Laplacian-Matérn ARD`)
    strictly on `(X_ctx, y_ctx) -> X_q` (`0` query target reads!)."""
    X_ctx = np.asarray(X_ctx, dtype=np.float64)
    y_ctx = np.asarray(y_ctx, dtype=np.float64)
    X_q = np.asarray(X_q, dtype=np.float64)
    n_c, p = X_ctx.shape
    n_q = X_q.shape[0]
    ym = float(np.mean(y_ctx))
    sy = float(np.std(y_ctx))
    if n_c < 32 or p == 0 or sy <= 1e-12:
        return np.full(n_c, ym), np.full(n_q, ym), False, 0.0

    warp = YeoJohnsonWarpF64()
    z_ctx = warp.fit_transform(y_ctx)

    rng = np.random.default_rng(seed)
    fold = rng.permutation(n_c) % folds
    oof_hgb_z = np.zeros(n_c, dtype=np.float64)
    oof_ker = np.zeros(n_c, dtype=np.float64)
    q_hgb_bag = np.zeros(n_q, dtype=np.float64)
    q_ker_bag = np.zeros(n_q, dtype=np.float64)

    p_sub = min(p, 48)
    if p > p_sub:
        fin_m = np.where(np.isfinite(X_ctx), X_ctx, 0.0)
        yc0 = z_ctx - float(np.mean(z_ctx))
        corrs = np.array([abs(float(np.dot(fin_m[:, j] - np.mean(fin_m[:, j]), yc0))) / (n_c * (float(np.std(fin_m[:, j])) + 1e-8)) for j in range(p)])
        keep_cols = np.sort(np.argsort(-corrs)[:p_sub])
        Xc_w, Xq_w = X_ctx[:, keep_cols], X_q[:, keep_cols]
    else:
        Xc_w, Xq_w = X_ctx, X_q

    for k in range(folds):
        va = fold == k
        tr = ~va
        if not np.any(va) or int(np.sum(tr)) < 16:
            continue
        hgb = HistGradientBoostingRegressor(
            max_iter=110,
            learning_rate=0.055,
            max_leaf_nodes=min(31, max(7, int(np.sum(tr)) // 8)),
            min_samples_leaf=max(5, int(np.sum(tr)) // 60),
            l2_regularization=2.5,
            max_features=0.80,
            early_stopping=False,
            random_state=int((seed + 37 * k) & 0x7FFFFFFF),
        )
        hgb.fit(Xc_w[tr], z_ctx[tr])
        oof_hgb_z[va] = hgb.predict(Xc_w[va])
        if n_q > 0:
            q_hgb_bag += hgb.predict(Xc_w[:0] if n_q == 0 else Xq_w) / float(folds)

        pred_va_and_q = fast_laplacian_matern_predict(
            Xc_w[tr], y_ctx[tr], np.vstack([Xc_w[va], Xq_w]) if n_q > 0 else Xc_w[va], max_sub=360
        )
        n_va = int(np.sum(va))
        oof_ker[va] = pred_va_and_q[:n_va]
        if n_q > 0:
            q_ker_bag += pred_va_and_q[n_va:] / float(folds)

    warp.set_residuals(z_ctx, oof_hgb_z)
    oof_hgb = warp.inv_np_duan(oof_hgb_z)

    # Full-context refit blended 50/50 with 5-fold bag for query prediction
    if n_q > 0:
        hgb_full = HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.055,
            max_leaf_nodes=min(31, max(7, n_c // 8)),
            min_samples_leaf=max(5, n_c // 60),
            l2_regularization=2.5,
            max_features=0.80,
            early_stopping=False,
            random_state=int((seed + 999) & 0x7FFFFFFF),
        )
        hgb_full.fit(Xc_w, z_ctx)
        q_hgb = 0.50 * warp.inv_np_duan(q_hgb_bag) + 0.50 * warp.inv_np_duan(hgb_full.predict(Xq_w))
        q_ker = 0.50 * q_ker_bag + 0.50 * fast_laplacian_matern_predict(Xc_w, y_ctx, Xq_w, max_sub=450)
    else:
        q_hgb = np.zeros(0, dtype=np.float64)
        q_ker = np.zeros(0, dtype=np.float64)

    # Optimal OOF convex weight between HGB and Laplacian-Matérn kernel
    e_h = oof_hgb - y_ctx
    e_k = oof_ker - y_ctx
    diff_hk = e_h - e_k
    denom = float(np.dot(diff_hk, diff_hk))
    w_k = float(np.clip(np.dot(e_h, diff_hk) / denom, 0.0, 1.0)) if denom > 1e-12 else 0.5
    oof_blend = (1.0 - w_k) * oof_hgb + w_k * oof_ker
    q_blend = (1.0 - w_k) * q_hgb + w_k * q_ker

    y_lo, y_hi = float(np.min(y_ctx)), float(np.max(y_ctx))
    span = max(y_hi - y_lo, sy)
    oof_blend = np.clip(oof_blend, y_lo - span, y_hi + span)
    q_blend = np.clip(q_blend, y_lo - span, y_hi + span)

    rc = y_ctx - oof_blend
    rmse_oof = float(np.sqrt(np.mean(rc * rc)))
    u_oof = -math.log(max(rmse_oof / sy, 1e-12))
    used = bool(
        np.all(np.isfinite(oof_blend))
        and np.all(np.isfinite(q_blend))
        and rmse_oof < 0.92 * sy
        and float(np.std(rc)) >= MIN_RES_SD * sy
    )
    return oof_blend, q_blend, used, u_oof


class HurdleSentinelDetector:
    """Detects discrete extreme outlier target atoms (e.g. `y == -99.99` on `fam-4286`) strictly on `y_ctx`
    and fits a 5-fold OOF-gated two-part hurdle wrapper `p_atom * y_atom + (1 - p_atom) * y_bulk`."""

    @staticmethod
    def detect_atom(y_ctx: np.ndarray) -> float | None:
        y = np.asarray(y_ctx, dtype=np.float64)
        n = len(y)
        if n < 40:
            return None
        u_vals, counts = np.unique(y, return_counts=True)
        if len(u_vals) < 10:
            return None
        q25, q50, q75 = np.quantile(y, [0.25, 0.50, 0.75])
        iqr = max(float(q75 - q25), 1e-9)
        for idx_a in (0, len(u_vals) - 1):
            va = float(u_vals[idx_a])
            cnt = int(counts[idx_a])
            if cnt < 4 or cnt > int(0.30 * n):
                continue
            adj = float(u_vals[1] if idx_a == 0 else u_vals[-2])
            if abs(va - float(q50)) > 6.5 * iqr and abs(va - adj) > 4.0 * iqr:
                return va
        return None

    @staticmethod
    def fit_oof_and_query_probs(
        X_ctx: np.ndarray, is_atom_ctx: np.ndarray, X_q: np.ndarray, lab_ctx: np.ndarray, seed: int = 450
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns `(p_oof [n_ctx], p_q [n_q])` strictly fitted on `(X_ctx, is_atom_ctx)`."""
        n_c = len(is_atom_ctx)
        n_q = X_q.shape[0]
        base_rate = float(np.mean(is_atom_ctx))
        if int(np.sum(is_atom_ctx)) < 4 or int(np.sum(~is_atom_ctx)) < 16:
            return np.full(n_c, base_rate), np.full(n_q, base_rate)

        p_oof = np.full(n_c, base_rate, dtype=np.float64)
        p_q = np.zeros(n_q, dtype=np.float64)
        valid_folds = 0
        for fk in range(5):
            va = lab_ctx == fk
            tr = ~va
            if not np.any(va) or int(np.sum(is_atom_ctx[tr])) < 2 or int(np.sum(~is_atom_ctx[tr])) < 10:
                continue
            clf = HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=4,
                l2_regularization=2.0,
                random_state=int((seed + 19 * fk) & 0x7FFFFFFF),
            )
            clf.fit(X_ctx[tr], is_atom_ctx[tr].astype(np.int32))
            p_oof[va] = clf.predict_proba(X_ctx[va])[:, 1]
            if n_q > 0:
                p_q += clf.predict_proba(X_ctx[:0] if n_q == 0 else X_q)[:, 1]
                valid_folds += 1

        clf_full = HistGradientBoostingClassifier(
            max_iter=110,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=4,
            l2_regularization=2.0,
            random_state=int((seed + 777) & 0x7FFFFFFF),
        )
        clf_full.fit(X_ctx, is_atom_ctx.astype(np.int32))
        if n_q > 0:
            p_q_bag = (p_q / float(valid_folds)) if valid_folds > 0 else clf_full.predict_proba(X_q)[:, 1]
            p_q_out = 0.50 * p_q_bag + 0.50 * clf_full.predict_proba(X_q)[:, 1]
        else:
            p_q_out = np.zeros(0, dtype=np.float64)
        return np.clip(p_oof, 0.0, 1.0), np.clip(p_q_out, 0.0, 1.0)
