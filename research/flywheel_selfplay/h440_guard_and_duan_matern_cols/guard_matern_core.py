#!/usr/bin/env python3
"""Wave B Core (`guard_and_duan_matern_cols`):
  1. 100% ya-Free Peer-Consensus Extrapolation Guard + Smooth Refit-Trust + Multi-Loss BCAR + eigh L2 Router.
  2. Wave 2-4 (`W2-W4`) Duan-Smeared Entity-Hierarchy CatBoost, Spectral-LOO Active-Matern-RQ GP,
     Rank-Gauss Sharp-ARD Subspace GP, SVD-GCV Spline-Ratio Ridge, and Compositional Residual-DAG / RegimeRoute.
  3. Plain non-Duan, non-entity ablation controls (`cb_plain`, `gp_plain`).

Strict Purity & Hygiene:
  - 100% ours-only primitives (NumPy, SciPy, scikit-learn, CatBoost thread_count=1).
  - Zero external foundation-model weights or distillation.
  - Strict fold-only fitting (`tr -> va, Xa`), zero test-batch (`Xa`) cross-row leakage, zero `ya` access.
"""
from __future__ import annotations

import hashlib
import math
import numpy as np
from catboost import CatBoostRegressor
from scipy import linalg, special, stats
from sklearn.ensemble import HistGradientBoostingRegressor

PRIMARY_NAMES = (
    "rel_cb_yjduan",
    "matern_rq_duan",
    "subgp_rgduan",
    "spline_ratio_yjduan",
    "dag_gp_on_cb",
    "dag_hgb_on_spline",
    "regime_route_gp",
)
CONTROL_NAMES = (
    "cb_plain",
    "gp_plain",
)
NAMES = PRIMARY_NAMES + CONTROL_NAMES

BAG_SEED, N_BAGS = 416_201, 32
W_MIN, MAX_MEMBERS = 0.005, 32


def _seed32(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:4], "little") & 0x7FFFFFFF


def _stream_seed(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


def _safe_svd(M: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust thin SVD: gesdd -> gesvd -> Gram eigh fallback (prevents LAPACK LinAlgError)."""
    A = np.where(np.isfinite(M), M, 0.0)
    try:
        return linalg.svd(A, full_matrices=False, check_finite=False, lapack_driver="gesdd")
    except Exception:
        pass
    try:
        return linalg.svd(A, full_matrices=False, check_finite=False, lapack_driver="gesvd")
    except Exception:
        n, p = A.shape
        if n >= p:
            ev, V = linalg.eigh(A.T @ A, check_finite=False)
            order = np.argsort(ev)[::-1]
            s = np.sqrt(np.maximum(ev[order], 0.0))
            Vt = V[:, order].T
            U = (A @ Vt.T) / np.where(s > 1e-12, s, 1.0)[None, :]
            return U, s, Vt
        ev, U = linalg.eigh(A @ A.T, check_finite=False)
        order = np.argsort(ev)[::-1]
        s = np.sqrt(np.maximum(ev[order], 0.0))
        U = U[:, order]
        Vt = ((U.T @ A).T / np.where(s > 1e-12, s, 1.0)[None, :]).T
        return U, s, Vt


def _safe_eigh(K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = np.where(np.isfinite(K), K, 0.0)
    A = 0.5 * (A + A.T)
    try:
        ev, V = np.linalg.eigh(A)
        return np.maximum(ev, 0.0), V
    except Exception:
        try:
            ev, V = linalg.eigh(A, check_finite=False, driver="evr")
            return np.maximum(ev, 0.0), V
        except Exception:
            n = A.shape[0]
            return np.maximum(np.diag(A), 0.0), np.eye(n, dtype=np.float64)


# ==================================================================================================
# PART 1: 15-Quantile Duan Smearing Target Transforms (Yeo-Johnson & Rank-Gauss with Linear Tails)
# ==================================================================================================
class TargetWarpDuan:
    def __init__(self, kind: str = "identity"):
        self.kind = kind
        self.mu = 0.0
        self.sd = 1.0
        self.lam = 1.0
        self.rg_scores = None
        self.rg_vals = None
        self.slope_lo = 0.0
        self.slope_hi = 0.0
        self.smear_q = np.zeros(1, dtype=np.float64)

    def fit(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        self.mu = float(np.mean(y))
        self.sd = float(np.std(y))
        if self.sd < 1e-12:
            self.sd = 1.0
            self.kind = "identity"
            return np.zeros_like(y)

        if self.kind == "identity":
            return (y - self.mu) / self.sd

        if self.kind == "yeo_johnson":
            yn = (y - self.mu) / self.sd
            try:
                _, lam = stats.yeojohnson(yn)
                self.lam = float(np.clip(lam, -1.5, 2.5))
                zt = stats.yeojohnson(yn, lmbda=self.lam)
            except Exception:
                self.lam = 1.0
                zt = yn
            self.z_mu = float(np.mean(zt))
            self.z_sd = max(float(np.std(zt)), 1e-9)
            self.zt_lo = float(np.min(zt) - 0.65 * self.z_sd)
            self.zt_hi = float(np.max(zt) + 0.65 * self.z_sd)
            return (zt - self.z_mu) / self.z_sd

        if self.kind == "rank_gauss":
            n = len(y)
            u_vals, inv_idx, counts = np.unique(y, return_inverse=True, return_counts=True)
            if len(u_vals) <= 3:
                self.kind = "identity"
                return (y - self.mu) / self.sd
            cum = np.cumsum(counts)
            mid = (cum - 0.5 * counts) / float(n)
            mid = np.clip(mid, 1.0 / (2.0 * n + 2.0), 1.0 - 1.0 / (2.0 * n + 2.0))
            scores = special.ndtri(mid)
            self.rg_scores = scores
            self.rg_vals = u_vals
            k_tail = max(1, min(len(u_vals) - 1, max(2, len(u_vals) // 10)))
            ds_lo = max(float(scores[k_tail] - scores[0]), 1e-6)
            ds_hi = max(float(scores[-1] - scores[-1 - k_tail]), 1e-6)
            self.slope_lo = max(0.0, float(u_vals[k_tail] - u_vals[0]) / ds_lo) * 0.5
            self.slope_hi = max(0.0, float(u_vals[-1] - u_vals[-1 - k_tail]) / ds_hi) * 0.5
            return scores[inv_idx]

        raise ValueError(self.kind)

    def set_residuals(self, z_true: np.ndarray, z_oof: np.ndarray) -> None:
        if self.kind == "identity":
            self.smear_q = np.zeros(1, dtype=np.float64)
            return
        res = np.asarray(z_true - z_oof, dtype=np.float64)
        res = res[np.isfinite(res)]
        if len(res) < 8:
            self.smear_q = np.zeros(1, dtype=np.float64)
            return
        res = res - np.mean(res)
        qs = 0.02 + 0.96 * ((np.arange(15, dtype=np.float64) + 0.5) / 15.0)
        self.smear_q = np.quantile(res, qs)

    def _inv_raw(self, z: np.ndarray) -> np.ndarray:
        if self.kind == "identity":
            return z * self.sd + self.mu
        if self.kind == "yeo_johnson":
            zt = np.clip(z * self.z_sd + self.z_mu, getattr(self, "zt_lo", -8.0), getattr(self, "zt_hi", 8.0))
            lam = self.lam
            out = np.empty_like(zt)
            pos = zt >= 0
            if abs(lam) < 1e-6:
                out[pos] = np.expm1(np.clip(zt[pos], -20.0, 9.0))
            else:
                out[pos] = np.power(np.maximum(lam * zt[pos] + 1.0, 0.05), 1.0 / lam) - 1.0
            lam2 = 2.0 - lam
            if abs(lam2) < 1e-6:
                out[~pos] = 1.0 - np.exp(np.clip(-zt[~pos], -20.0, 9.0))
            else:
                out[~pos] = 1.0 - np.power(np.maximum(-lam2 * zt[~pos], -0.95) * 0.0 + np.maximum(-lam2 * zt[~pos] + 1.0, 0.05), 1.0 / lam2)
            return out * self.sd + self.mu
        if self.kind == "rank_gauss":
            sc, vl = self.rg_scores, self.rg_vals
            interp = np.interp(z, sc, vl)
            below = z < sc[0]
            above = z > sc[-1]
            if np.any(below):
                interp[below] = vl[0] + self.slope_lo * (z[below] - sc[0])
            if np.any(above):
                interp[above] = vl[-1] + self.slope_hi * (z[above] - sc[-1])
            return interp
        raise ValueError(self.kind)

    def inverse(self, z_pred: np.ndarray) -> np.ndarray:
        z_pred = np.asarray(z_pred, dtype=np.float64)
        if self.kind == "identity" or len(self.smear_q) <= 1:
            return self._inv_raw(z_pred)
        shifted = z_pred[:, None] + self.smear_q[None, :]
        inv_mat = self._inv_raw(shifted.ravel()).reshape(shifted.shape)
        return np.mean(inv_mat, axis=1)


# ==================================================================================================
# PART 2: MultiView Preprocessor & Entity-Hierarchy Relational Feature Builder
# ==================================================================================================
class MultiViewPreprocessor:
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultiViewPreprocessor":
        X = np.asarray(X, dtype=np.float64)
        n, p = X.shape
        self.p = p
        self.med = np.zeros(p)
        self.scale = np.ones(p)
        self.q_grids = []
        self.keep_cols = []

        X_imp = np.empty_like(X)
        for j in range(p):
            col = X[:, j]
            fin = np.isfinite(col)
            med = float(np.median(col[fin])) if np.any(fin) else 0.0
            self.med[j] = med
            c_clean = np.where(fin, col, med)
            X_imp[:, j] = c_clean
            sd = float(np.std(c_clean))
            if sd < 1e-12:
                continue
            self.keep_cols.append(j)
            q25, q75 = np.quantile(c_clean, [0.25, 0.75])
            iqr = float(q75 - q25) / 1.349
            self.scale[j] = iqr if iqr > 1e-9 else sd
            qs = np.linspace(0.0, 1.0, min(129, n))
            grid = np.quantile(c_clean, qs)
            u_grid, u_idx = np.unique(grid, return_index=True)
            self.q_grids.append((j, u_grid, qs[u_idx]))

        if not self.keep_cols:
            self.keep_cols = [0]
            self.scale[0] = 1.0
            self.q_grids = [(0, np.array([0.0]), np.array([0.5]))]

        Xk = X_imp[:, self.keep_cols]
        yk = y - np.mean(y)
        yn = yk / max(float(np.std(yk)), 1e-9)
        Xz = (Xk - np.mean(Xk, axis=0)) / np.maximum(np.std(Xk, axis=0), 1e-9)
        corr_lin = np.abs((Xz.T @ yn) / float(n))
        rx = np.argsort(np.argsort(Xk, axis=0), axis=0).astype(np.float64)
        rx = (rx - np.mean(rx, axis=0)) / np.maximum(np.std(rx, axis=0), 1e-9)
        ry = np.argsort(np.argsort(yn)).astype(np.float64)
        ry = (ry - np.mean(ry)) / max(float(np.std(ry)), 1e-9)
        corr_rk = np.abs((rx.T @ ry) / float(n))
        rel = np.maximum(corr_lin, corr_rk)
        self.ard = 0.25 + 1.75 * np.power(rel / max(float(np.max(rel)), 1e-6), 0.75)
        return self

    def transform(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        X_imp = np.where(np.isfinite(X), X, self.med[None, :])
        Xk = X_imp[:, self.keep_cols]
        Z = (Xk - self.med[self.keep_cols]) / self.scale[self.keep_cols]
        X_smooth = 3.0 * Z / np.sqrt(1.0 + (Z / 3.0) ** 2)
        X_blom = np.empty_like(Xk)
        for idx, (j, u_grid, u_qs) in enumerate(self.q_grids):
            if len(u_grid) <= 1:
                X_blom[:, idx] = 0.0
            else:
                p_est = np.clip(np.interp(X_imp[:, j], u_grid, u_qs), 1e-3, 1.0 - 1e-3)
                X_blom[:, idx] = special.ndtri(p_est) * 0.5
        return X_smooth, X_blom


def _nonlinear_relevance(Xs: np.ndarray, Xb: np.ndarray, y: np.ndarray) -> np.ndarray:
    n, p = Xs.shape
    yc = y - np.mean(y)
    vy = max(float(np.dot(yc, yc)), 1e-12)
    xs_c = Xs - Xs.mean(axis=0, keepdims=True)
    xb_c = Xb - Xb.mean(axis=0, keepdims=True)
    r_lin = np.abs(xs_c.T @ yc) / np.sqrt(np.maximum(np.sum(xs_c * xs_c, axis=0) * vy, 1e-12))
    r_rnk = np.abs(xb_c.T @ yc) / np.sqrt(np.maximum(np.sum(xb_c * xb_c, axis=0) * vy, 1e-12))
    r_bin = np.zeros(p, dtype=np.float64)
    edges = np.array([0.2, 0.4, 0.6, 0.8])
    for j in range(p):
        col = Xs[:, j]
        qs = np.unique(np.quantile(col, edges))
        if len(qs) == 0:
            continue
        b = np.digitize(col, qs)
        cnt = np.bincount(b, minlength=len(qs) + 1).astype(np.float64)
        sy = np.bincount(b, weights=yc, minlength=len(qs) + 1)
        mu_b = sy / (cnt + 8.0)
        ss_exp = float(np.sum(cnt * (mu_b * mu_b)))
        r_bin[j] = math.sqrt(max(0.0, min(1.0, ss_exp / vy)))
    score = np.maximum.reduce([r_lin, r_rnk, r_bin])
    if float(np.max(score)) < 1e-9:
        return np.ones(p, dtype=np.float64)
    return score / float(np.max(score))


class EntityHierarchyEngine:
    def __init__(self, max_single_keys: int = 5, max_pair_keys: int = 2, max_val_cols: int = 6, shrink_m: float = 5.0):
        self.max_single_keys = max_single_keys
        self.max_pair_keys = max_pair_keys
        self.max_val_cols = max_val_cols
        self.shrink_m = shrink_m

    def _detect_missing_matrix(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        n, p = X.shape
        isnan = ~np.isfinite(X)
        if fit:
            self.zero_sentinel_cols_ = []
            for j in range(p):
                col = X[~isnan[:, j], j]
                if len(col) < 20:
                    continue
                z_frac = float(np.mean(col == 0.0))
                nz = col[col != 0.0]
                if 0.01 <= z_frac <= 0.60 and len(nz) >= 10:
                    q05, q95 = np.quantile(nz, [0.05, 0.95])
                    if q05 > 0.0 or q95 < 0.0:
                        self.zero_sentinel_cols_.append(j)
            self.miss_indicator_cols_ = [
                j
                for j in range(p)
                if float(np.mean(isnan[:, j] | ((X[:, j] == 0.0) if j in self.zero_sentinel_cols_ else False))) >= 0.01
            ][:16]

        M_full = isnan.copy()
        for j in self.zero_sentinel_cols_:
            if j < p:
                M_full[:, j] |= X[:, j] == 0.0
        return M_full

    @staticmethod
    def _encode_key_col(col: np.ndarray) -> np.ndarray:
        out = np.full(len(col), -999999, dtype=np.int64)
        fin = np.isfinite(col)
        if np.any(fin):
            sc = max(float(np.max(np.abs(col[fin]))), 1.0)
            out[fin] = np.round((col[fin] / sc) * 1e7).astype(np.int64)
        return out

    def _encode_key_col_fit(self, col: np.ndarray, fit: bool, j: int) -> np.ndarray:
        out = np.full(len(col), -999999, dtype=np.int64)
        fin = np.isfinite(col)
        if fit:
            self.key_scale_[j] = max(float(np.max(np.abs(col[fin]))), 1.0) if np.any(fin) else 1.0
        sc = self.key_scale_.get(j, 1.0)
        if np.any(fin):
            out[fin] = np.round(np.clip(col[fin] / sc, -1e9, 1e9) * 1e7).astype(np.int64)
        return out

    def fit_transform(self, X_tr: np.ndarray, y_tr: np.ndarray, seed: int) -> np.ndarray:
        n, p = X_tr.shape
        self.key_scale_ = {}
        M_tr = self._detect_missing_matrix(X_tr, fit=True)
        self.med_ = np.zeros(p, dtype=np.float64)
        self.iqr_ = np.ones(p, dtype=np.float64)
        X_clean = np.zeros((n, p), dtype=np.float64)
        for j in range(p):
            valid = ~M_tr[:, j]
            vals = X_tr[valid, j] if np.any(valid) else np.zeros(1)
            med = float(np.median(vals))
            q25, q75 = np.quantile(vals, [0.25, 0.75]) if len(vals) >= 4 else (med - 1.0, med + 1.0)
            sc = max(float(q75 - q25), float(np.std(vals)), 1e-6)
            self.med_[j] = med
            self.iqr_[j] = sc
            c = np.where(M_tr[:, j], med, X_tr[:, j])
            X_clean[:, j] = np.clip((c - med) / sc, -10.0, 10.0)

        prep = MultiViewPreprocessor().fit(X_tr, y_tr)
        Xs_tr, Xb_tr = prep.transform(X_tr)
        rel_sub = _nonlinear_relevance(Xs_tr, Xb_tr, y_tr)
        rel = np.zeros(p, dtype=np.float64)
        for loc_i, orig_j in enumerate(prep.keep_cols):
            if orig_j < p and loc_i < len(rel_sub):
                rel[orig_j] = float(rel_sub[loc_i])
        self.rel_ = rel

        max_card = max(3, min(n // 3, 300))
        key_cands = []
        for j in range(p):
            codes_j = self._encode_key_col_fit(np.where(M_tr[:, j], np.nan, X_tr[:, j]), fit=True, j=j)
            u = np.unique(codes_j[codes_j != -999999])
            if 2 <= len(u) <= max_card:
                key_cands.append((float(rel[j]), len(u), j))
        key_cands.sort(key=lambda t: -t[0])
        self.single_keys_ = [j for _, _, j in key_cands[: self.max_single_keys]]

        self.pair_keys_ = []
        if len(self.single_keys_) >= 2:
            for a in range(min(3, len(self.single_keys_))):
                for b in range(a + 1, min(3, len(self.single_keys_))):
                    if len(self.pair_keys_) < self.max_pair_keys:
                        self.pair_keys_.append((self.single_keys_[a], self.single_keys_[b]))

        cont_cands = [j for j in np.argsort(-rel) if j not in self.single_keys_[:2]]
        self.val_cols_ = [int(j) for j in cont_cands[: min(self.max_val_cols, p)]]
        self.keep_raw_cols_ = np.sort(np.argsort(-rel)[: min(40, p)])

        self.y_mean_ = float(np.mean(y_tr))
        self.y_std_ = max(float(np.std(y_tr)), 1e-6)
        y_norm = (y_tr - self.y_mean_) / self.y_std_

        self.key_specs_ = []
        feats_tr = self._build_miss_feats(M_tr)

        rng = np.random.default_rng(seed)
        inner_fold = rng.integers(0, 5, size=n)

        all_key_defs = [("single", (j,)) for j in self.single_keys_] + [("pair", ab) for ab in self.pair_keys_]
        if len(self.miss_indicator_cols_) >= 2:
            all_key_defs.append(("miss_pat", tuple(self.miss_indicator_cols_[:8])))

        for ktype, cols in all_key_defs:
            codes = self._hash_key(X_tr, M_tr, ktype, cols)
            uniq, inv, counts = np.unique(codes, return_inverse=True, return_counts=True)
            if len(uniq) < 2 or int(np.max(counts)) < 3:
                continue

            val_mat = X_clean[:, self.val_cols_]
            v_dim = val_mat.shape[1]
            g_sum = np.zeros((len(uniq), v_dim), dtype=np.float64)
            g_sum2 = np.zeros((len(uniq), v_dim), dtype=np.float64)
            if v_dim > 0:
                np.add.at(g_sum, inv, val_mat)
                np.add.at(g_sum2, inv, val_mat * val_mat)

            cnt_f = counts[:, None].astype(np.float64)
            glob_mean = np.mean(val_mat, axis=0, keepdims=True) if v_dim > 0 else np.zeros((1, 0))
            glob_var = (np.var(val_mat, axis=0, keepdims=True) + 1e-4) if v_dim > 0 else np.ones((1, 0))
            m = self.shrink_m
            shrunk_mean = (g_sum + m * glob_mean) / (cnt_f + m)
            raw_var = np.maximum(g_sum2 / np.maximum(cnt_f, 1.0) - (g_sum / np.maximum(cnt_f, 1.0)) ** 2, 0.0)
            shrunk_std = np.sqrt((cnt_f * raw_var + m * glob_var) / (cnt_f + m)) + 0.1

            gy_sum = np.bincount(inv, weights=y_norm, minlength=len(uniq))
            gy_sum2 = np.bincount(inv, weights=y_norm * y_norm, minlength=len(uniq))
            gy_mean_full = gy_sum / (counts + m)
            gy_var_full = np.maximum(gy_sum2 / np.maximum(counts, 1) - (gy_sum / np.maximum(counts, 1)) ** 2, 0.0)
            gy_std_full = np.sqrt((counts * gy_var_full + m * 1.0) / (counts + m))

            oof_y_mean = np.zeros(n, dtype=np.float64)
            oof_y_std = np.ones(n, dtype=np.float64)
            for f in range(5):
                tr_f = inner_fold != f
                va_f = inner_fold == f
                if not np.any(va_f):
                    continue
                c_tr = np.bincount(inv[tr_f], minlength=len(uniq))
                s_tr = np.bincount(inv[tr_f], weights=y_norm[tr_f], minlength=len(uniq))
                s2_tr = np.bincount(inv[tr_f], weights=y_norm[tr_f] ** 2, minlength=len(uniq))
                m_f = s_tr / (c_tr + m)
                v_f = np.maximum(s2_tr / np.maximum(c_tr, 1) - (s_tr / np.maximum(c_tr, 1)) ** 2, 0.0)
                sd_f = np.sqrt((c_tr * v_f + m * 1.0) / (c_tr + m))
                oof_y_mean[va_f] = m_f[inv[va_f]]
                oof_y_std[va_f] = sd_f[inv[va_f]]

            lut = {
                int(u_code): (
                    float(np.log1p(counts[idx])),
                    shrunk_mean[idx],
                    shrunk_std[idx],
                    float(gy_mean_full[idx]),
                    float(gy_std_full[idx]),
                )
                for idx, u_code in enumerate(uniq)
            }
            self.key_specs_.append((ktype, cols, lut, glob_mean[0], np.sqrt(glob_var[0])))

            log_deg = np.log1p(counts[inv])[:, None]
            row_gmean = shrunk_mean[inv]
            row_z = np.clip((val_mat - row_gmean) / shrunk_std[inv], -8.0, 8.0) if v_dim > 0 else np.zeros((n, 0))
            feats_tr.extend([log_deg, row_gmean[:, :3], row_z[:, :3], oof_y_mean[:, None], oof_y_std[:, None]])

        H_tr = np.hstack(feats_tr) if feats_tr else np.zeros((n, 0), dtype=np.float64)
        return np.hstack([X_clean[:, self.keep_raw_cols_], H_tr])

    def _build_miss_feats(self, M: np.ndarray) -> list[np.ndarray]:
        out = [np.mean(M, axis=1, keepdims=True)]
        if self.miss_indicator_cols_:
            out.append(M[:, self.miss_indicator_cols_].astype(np.float64))
        return out

    def _hash_key(self, X: np.ndarray, M: np.ndarray, ktype: str, cols: tuple[int, ...]) -> np.ndarray:
        if ktype == "single":
            return self._encode_key_col_fit(np.where(M[:, cols[0]], np.nan, X[:, cols[0]]), fit=False, j=cols[0])
        if ktype == "pair":
            a = self._encode_key_col_fit(np.where(M[:, cols[0]], np.nan, X[:, cols[0]]), fit=False, j=cols[0])
            b = self._encode_key_col_fit(np.where(M[:, cols[1]], np.nan, X[:, cols[1]]), fit=False, j=cols[1])
            return a * 1000003 + b
        h = np.zeros(X.shape[0], dtype=np.int64)
        for bit, j in enumerate(cols):
            h |= M[:, j].astype(np.int64) << bit
        return h

    def transform(self, X: np.ndarray) -> np.ndarray:
        n, p = X.shape
        M = self._detect_missing_matrix(X, fit=False)
        X_clean = np.zeros((n, p), dtype=np.float64)
        for j in range(p):
            c = np.where(M[:, j], self.med_[j], X[:, j])
            X_clean[:, j] = np.clip((c - self.med_[j]) / self.iqr_[j], -10.0, 10.0)

        feats = self._build_miss_feats(M)
        val_mat = X_clean[:, self.val_cols_]
        v_dim = val_mat.shape[1]

        for ktype, cols, lut, glob_m, glob_sd in self.key_specs_:
            codes = self._hash_key(X, M, ktype, cols)
            log_deg = np.zeros((n, 1), dtype=np.float64)
            row_gmean = np.broadcast_to(glob_m, (n, v_dim)).copy() if v_dim > 0 else np.zeros((n, 0))
            row_gsd = np.broadcast_to(glob_sd + 0.1, (n, v_dim)).copy() if v_dim > 0 else np.ones((n, 0))
            ym = np.zeros((n, 1), dtype=np.float64)
            ysd = np.ones((n, 1), dtype=np.float64)
            for i, code in enumerate(codes):
                hit = lut.get(int(code))
                if hit is not None:
                    log_deg[i, 0], row_gmean[i], row_gsd[i], ym[i, 0], ysd[i, 0] = hit
            row_z = np.clip((val_mat - row_gmean) / row_gsd, -8.0, 8.0) if v_dim > 0 else np.zeros((n, 0))
            feats.extend([log_deg, row_gmean[:, :3], row_z[:, :3], ym, ysd])

        H = np.hstack(feats) if feats else np.zeros((n, 0), dtype=np.float64)
        return np.hstack([X_clean[:, self.keep_raw_cols_], H])


# ==================================================================================================
# PART 3: Spectral-LOO Active-Matern-RQ GP, Sharp-ARD RankGauss GP, SplineRatio, and Residual DAGs
# ==================================================================================================
def _matern_rq_kernel(A: np.ndarray, B: np.ndarray, ell: float) -> np.ndarray:
    d = max(A.shape[1], 1)
    sc = ell * math.sqrt(d)
    As = A / sc
    Bs = B / sc
    r2 = np.maximum(np.sum(As * As, axis=1, keepdims=True) + np.sum(Bs * Bs, axis=1, keepdims=True).T - 2.0 * (As @ Bs.T), 0.0)
    r = np.sqrt(r2)
    s3 = math.sqrt(3.0) * r
    s5 = math.sqrt(5.0) * r
    k32 = (1.0 + s3) * np.exp(-s3)
    k52 = (1.0 + s5 + (5.0 / 3.0) * r2) * np.exp(-s5)
    krq = (1.0 + r2) ** (-0.5)
    klin = 0.10 * np.clip(As @ Bs.T, -10.0, 10.0)
    return 0.45 * k32 + 0.35 * k52 + 0.20 * krq + klin


class ActiveMaternRQSubGPDuan:
    def __init__(self, top_k: int = 16, max_exact: int = 550, kinds: tuple[str, ...] = ("identity", "yeo_johnson")):
        self.top_k = top_k
        self.max_exact = max_exact
        self.kinds = kinds

    def fit(self, X_feat: np.ndarray, y: np.ndarray, seed: int) -> "ActiveMaternRQSubGPDuan":
        n, p = X_feat.shape
        yc = y - np.mean(y)
        ysd = max(float(np.std(yc)), 1e-8)
        yn = yc / ysd
        corrs = np.zeros(p, dtype=np.float64)
        for j in range(p):
            cj = X_feat[:, j] - np.mean(X_feat[:, j])
            s = float(np.std(cj))
            if s > 1e-8:
                corrs[j] = abs(float(np.dot(cj, yn)) / (n * s))
        k = min(self.top_k, p)
        self.idx_ = np.argsort(-corrs)[:k]
        w_rel = np.maximum(corrs[self.idx_], 0.05) ** 1.35
        self.w_rel_ = w_rel / max(float(np.mean(w_rel)), 1e-6)

        Z = X_feat[:, self.idx_] * self.w_rel_[None, :]
        if n > self.max_exact:
            order = np.argsort(y, kind="stable")
            pick = np.linspace(0, n - 1, self.max_exact, dtype=np.int64)
            sub = order[pick]
        else:
            sub = np.arange(n, dtype=np.int64)
        self.Z_sub_ = Z[sub]
        y_sub = y[sub]

        spec_by_ell = []
        for ell in (0.85, 1.85):
            K_base = _matern_rq_kernel(self.Z_sub_, self.Z_sub_, ell)
            eigvals, V = _safe_eigh(K_base)
            V2 = V * V
            spec_by_ell.append((ell, eigvals, V, V2))

        best_loo_mse = float("inf")
        best_cfg = None
        for kind in self.kinds:
            warp_base = TargetWarpDuan(kind)
            z_sub = warp_base.fit(y_sub)
            z_mean = float(np.mean(z_sub))
            zc = z_sub - z_mean
            for ell, eigvals, V, V2 in spec_by_ell:
                vt_zc = V.T @ zc
                for noise in (0.015, 0.06, 0.22):
                    inv_d = 1.0 / (eigvals + noise)
                    alpha = V @ (inv_d * vt_zc)
                    diag_Kinv = np.maximum(V2 @ inv_d, 1e-9)
                    z_loo = z_sub - alpha / diag_Kinv
                    warp_cand = TargetWarpDuan(kind)
                    warp_cand.fit(y_sub)
                    warp_cand.set_residuals(z_sub, z_loo)
                    y_loo = warp_cand.inverse(z_loo)
                    loo_mse = float(np.mean((y_loo - y_sub) ** 2))
                    if np.isfinite(loo_mse) and loo_mse < best_loo_mse:
                        best_loo_mse = loo_mse
                        best_cfg = (warp_cand, ell, z_mean, alpha)

        if best_cfg is None:
            self.fallback_mean_ = float(np.mean(y))
            self.cfg_ = None
        else:
            self.cfg_ = best_cfg
        return self

    def predict(self, X_feat: np.ndarray) -> np.ndarray:
        if self.cfg_ is None:
            return np.full(X_feat.shape[0], self.fallback_mean_, dtype=np.float64)
        warp, ell, z_mean, alpha = self.cfg_
        Z = X_feat[:, self.idx_] * self.w_rel_[None, :]
        K_star = _matern_rq_kernel(Z, self.Z_sub_, ell)
        z_pred = z_mean + K_star @ alpha
        return warp.inverse(z_pred)


class SharpARDRankGaussSubGP:
    """MultiView [Xs, 0.55*Xb] Sharp-ARD GP in Rank-Gauss + Linear-Tails + 15-Quantile Duan Smearing space."""

    def __init__(self, max_exact: int = 650):
        self.max_exact = max_exact

    def fit(self, Xs: np.ndarray, Xb: np.ndarray, rel: np.ndarray, y: np.ndarray) -> "SharpARDRankGaussSubGP":
        n, p = Xs.shape
        top = np.sort(np.argsort(-rel)[: min(14, p)])
        w = np.power(np.maximum(rel[top], 0.03), 1.35)
        w = w / math.sqrt(max(float(np.sum(w * w)), 1e-12))
        self.top_ = top
        self.w_ = w
        Phi_all = np.hstack([Xs[:, top] * w[None, :], Xb[:, top] * (0.55 * w[None, :])])
        if n > self.max_exact:
            order = np.argsort(y, kind="stable")
            sub = order[np.linspace(0, n - 1, self.max_exact, dtype=np.int64)]
        else:
            sub = np.arange(n, dtype=np.int64)
        self.Phi_tr_ = Phi_all[sub]
        y_sub = y[sub]

        self.warp_ = TargetWarpDuan("rank_gauss")
        z = self.warp_.fit(y_sub)
        z_mean = float(np.mean(z))
        zc = z - z_mean

        best_mse = float("inf")
        best_alpha = np.zeros(len(sub), dtype=np.float64)
        best_oof = np.full(len(sub), z_mean, dtype=np.float64)
        best_ell = 1.0

        for ell in (0.45, 0.95, 2.0):
            K = _matern_rq_kernel(self.Phi_tr_, self.Phi_tr_, ell)
            evals, Q = _safe_eigh(K)
            Qtz = Q.T @ zc
            Q2 = Q * Q
            for nv in (1e-3, 1.5e-2, 8e-2, 0.28):
                inv_d = 1.0 / (evals + nv)
                alpha = Q @ (Qtz * inv_d)
                diag_inv = np.maximum(Q2 @ inv_d, 1e-12)
                loo_err = alpha / diag_inv
                mse = float(np.mean(loo_err * loo_err))
                if mse < best_mse:
                    best_mse = mse
                    best_alpha = alpha
                    best_oof = z - loo_err
                    best_ell = ell

        self.z_mean_ = z_mean
        self.alpha_ = best_alpha
        self.ell_ = best_ell
        self.warp_.set_residuals(z, best_oof)
        return self

    def predict(self, Xs: np.ndarray, Xb: np.ndarray) -> np.ndarray:
        Phi_q = np.hstack([Xs[:, self.top_] * self.w_[None, :], Xb[:, self.top_] * (0.55 * self.w_[None, :])])
        Kq = _matern_rq_kernel(Phi_q, self.Phi_tr_, self.ell_)
        return self.warp_.inverse(self.z_mean_ + Kq @ self.alpha_)


class SplineRatioYJDuan:
    """Exact SVD-LOO Ridge over 7-knot quantile hinges + top-10 bilinear products + log-ratios in YJ-Duan space."""

    def _build_design(self, Xs: np.ndarray, Xb: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            p = Xs.shape[1]
            self.top_s = np.sort(np.argsort(-self.rel)[: min(24, p)])
            self.top_i = np.sort(np.argsort(-self.rel)[: min(10, p)])
            qs = np.linspace(0.12, 0.88, 7)
            self.knots = np.quantile(Xs[:, self.top_s], qs, axis=0)
        cols = [Xs[:, self.top_s], Xb[:, self.top_s]]
        Sub = Xs[:, self.top_s]
        for k in range(self.knots.shape[0]):
            cols.append(np.maximum(0.0, Sub - self.knots[k][None, :]))
        Ti = Xs[:, self.top_i]
        Log_i = np.sign(Ti) * np.log1p(np.abs(Ti))
        ni = Ti.shape[1]
        prods, ratios = [], []
        for a in range(ni):
            prods.append(np.clip(Ti[:, a] * Ti[:, a], -9.0, 9.0))
            for b in range(a + 1, ni):
                prods.append(np.clip(Ti[:, a] * Ti[:, b], -9.0, 9.0))
                ratios.append(Log_i[:, a] - Log_i[:, b])
        if prods:
            cols.append(np.column_stack(prods))
        if ratios:
            cols.append(np.column_stack(ratios))
        Phi = np.hstack(cols)
        if fit:
            self.mu = Phi.mean(axis=0)
            self.sd = np.maximum(Phi.std(axis=0), 1e-6)
        return np.clip((Phi - self.mu) / self.sd, -8.0, 8.0)

    def fit(self, Xs: np.ndarray, Xb: np.ndarray, rel: np.ndarray, y: np.ndarray) -> "SplineRatioYJDuan":
        self.rel = rel
        Phi = self._build_design(Xs, Xb, fit=True)
        n = Phi.shape[0]
        self.warp_ = TargetWarpDuan("yeo_johnson")
        z = self.warp_.fit(y)
        z_mean = float(np.mean(z))
        zc = z - z_mean

        U, s, Vt = _safe_svd(Phi)
        s2 = s * s
        Utz = U.T @ zc
        U_sq = U * U

        best_mse = float("inf")
        best_w = np.zeros(Phi.shape[1], dtype=np.float64)
        best_oof = np.full(n, z_mean, dtype=np.float64)
        for lam in np.geomspace(0.5, 5e4, 20):
            shrink = s / (s2 + lam)
            h_diag = U_sq @ (s2 / (s2 + lam)) + (1.0 / n)
            denom = np.maximum(1.0 - h_diag, 1e-4)
            z_hat = U @ (Utz * (s2 / (s2 + lam))) + z_mean
            loo_err = (z - z_hat) / denom
            mse = float(np.mean(loo_err * loo_err))
            if mse < best_mse:
                best_mse = mse
                best_w = Vt.T @ (Utz * shrink)
                best_oof = z - loo_err

        self.w_ = best_w
        self.b_ = z_mean
        self.warp_.set_residuals(z, best_oof)
        return self

    def predict(self, Xs: np.ndarray, Xb: np.ndarray) -> np.ndarray:
        Phi = self._build_design(Xs, Xb, fit=False)
        return self.warp_.inverse(Phi @ self.w_ + self.b_)


class DuanCatBoostBag:
    def __init__(self, kind: str = "identity", depth: int = 6, n_bags: int = 2):
        self.kind = kind
        self.depth = depth
        self.n_bags = n_bags

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int) -> "DuanCatBoostBag":
        self.warp = TargetWarpDuan(self.kind)
        z = self.warp.fit(y)
        n = len(z)
        rng = np.random.default_rng(seed)
        self.models = []
        oof_sum = np.zeros(n, dtype=np.float64)
        oof_cnt = np.zeros(n, dtype=np.float64)
        iters = 210 if n <= 2500 else 160
        for b in range(self.n_bags):
            order = np.argsort(z, kind="stable")
            oob = np.zeros(n, dtype=bool)
            for s in range(0, n, 5):
                blk = order[s : s + 5]
                oob[blk[rng.integers(0, len(blk))]] = True
            tr = ~oob
            cb = CatBoostRegressor(
                iterations=iters,
                learning_rate=0.055,
                depth=min(self.depth, 6),
                rsm=0.50,
                border_count=64,
                l2_leaf_reg=4.0,
                random_seed=int((seed + 131 * b) & 0x7FFFFFFF),
                thread_count=1,
                verbose=False,
                allow_writing_files=False,
            )
            cb.fit(X[tr], z[tr], eval_set=(X[oob], z[oob]), early_stopping_rounds=30, verbose=False)
            self.models.append(cb)
            oof_sum[oob] += cb.predict(X[oob])
            oof_cnt[oob] += 1.0
        valid = oof_cnt > 0
        if self.kind != "identity" and int(np.sum(valid)) >= 12:
            self.warp.set_residuals(z[valid], oof_sum[valid] / oof_cnt[valid])
        self.oof_raw_pred_ = np.where(valid, self.warp.inverse(oof_sum / np.maximum(oof_cnt, 1.0)), float(np.mean(y)))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.mean([self.warp.inverse(m.predict(X)) for m in self.models], axis=0)


class DAGSubGPOnCatBoostResid:
    def fit(self, X_feat: np.ndarray, base_cb: DuanCatBoostBag, y: np.ndarray, seed: int) -> "DAGSubGPOnCatBoostResid":
        self.base_ = base_cb
        r_oof = y - base_cb.oof_raw_pred_
        p_norm = (base_cb.oof_raw_pred_ - float(np.mean(y))) / max(float(np.std(y)), 1e-8)
        X_aug = np.column_stack([X_feat, p_norm[:, None]])
        self.gp_ = ActiveMaternRQSubGPDuan(top_k=14, max_exact=450, kinds=("identity",)).fit(X_aug, r_oof, seed + 17)
        n = len(y)
        rng = np.random.default_rng(seed + 23)
        perm = rng.permutation(n)
        h = n // 2
        gp_a = ActiveMaternRQSubGPDuan(top_k=12, max_exact=350, kinds=("identity",)).fit(X_aug[perm[:h]], r_oof[perm[:h]], seed + 31)
        gp_b = ActiveMaternRQSubGPDuan(top_k=12, max_exact=350, kinds=("identity",)).fit(X_aug[perm[h:]], r_oof[perm[h:]], seed + 37)
        r_hat = np.zeros(n, dtype=np.float64)
        r_hat[perm[h:]] = gp_a.predict(X_aug[perm[h:]])
        r_hat[perm[:h]] = gp_b.predict(X_aug[perm[:h]])
        eta = float(np.dot(r_oof, r_hat)) / (float(np.dot(r_hat, r_hat)) + 1e-9)
        base_mse = float(np.mean(r_oof ** 2))
        comb_mse = float(np.mean((r_oof - np.clip(eta, 0.0, 0.85) * r_hat) ** 2))
        self.eta_ = float(np.clip(eta, 0.0, 0.85)) if comb_mse < 0.995 * base_mse else 0.0
        self.y_mean_ = float(np.mean(y))
        self.y_std_ = max(float(np.std(y)), 1e-8)
        return self

    def predict(self, X_feat: np.ndarray) -> np.ndarray:
        p1 = self.base_.predict(X_feat)
        if self.eta_ <= 1e-4:
            return p1
        p_norm = (p1 - self.y_mean_) / self.y_std_
        X_aug = np.column_stack([X_feat, p_norm[:, None]])
        return p1 + self.eta_ * self.gp_.predict(X_aug)


class DAGHGBOnSplineRidgeResid:
    def fit(self, X_feat: np.ndarray, y: np.ndarray, seed: int) -> "DAGHGBOnSplineRidgeResid":
        n, p = X_feat.shape
        yc = y - float(np.mean(y))
        self.y_mean_ = float(np.mean(y))
        corrs = np.array([abs(float(np.dot(X_feat[:, j] - np.mean(X_feat[:, j]), yc))) / (n * (float(np.std(X_feat[:, j])) + 1e-8)) for j in range(p)])
        self.top_idx_ = np.argsort(-corrs)[: min(12, p)]
        Xt = X_feat[:, self.top_idx_]
        self.knots_ = np.quantile(Xt, [0.25, 0.50, 0.75], axis=0)
        B = self._basis(X_feat)

        U, s, Vt = _safe_svd(B)
        best_lam, best_loo_pred, best_mse = 10.0, np.zeros(n), float("inf")
        uty = U.T @ yc
        for lam in (0.5, 2.5, 12.0, 60.0):
            filt = (s * s) / (s * s + lam)
            y_hat = U @ (filt * uty)
            h_diag = np.clip(np.sum((U * U) * filt[None, :], axis=1), 0.0, 0.98)
            loo_err = (yc - y_hat) / (1.0 - h_diag)
            mse = float(np.mean(loo_err * loo_err))
            if mse < best_mse:
                best_mse = mse
                best_lam = lam
                best_loo_pred = yc - loo_err
        filt = s / (s * s + best_lam)
        self.w_ridge_ = Vt.T @ (filt * uty)

        r_loo = yc - best_loo_pred
        rng = np.random.default_rng(seed)
        self.trees_ = []
        oob_sum = np.zeros(n, dtype=np.float64)
        oob_cnt = np.zeros(n, dtype=np.float64)
        for b in range(3):
            oob = rng.random(n) < 0.22
            if int(np.sum(oob)) < 8 or int(np.sum(~oob)) < 20:
                oob = np.zeros(n, dtype=bool)
                oob[::5] = True
            tr = ~oob
            hgb = HistGradientBoostingRegressor(
                max_iter=220,
                learning_rate=0.04,
                max_leaf_nodes=31,
                min_samples_leaf=max(8, n // 50),
                l2_regularization=3.0,
                max_features=0.75,
                early_stopping=True,
                validation_fraction=0.12,
                n_iter_no_change=20,
                random_state=int((seed + 59 * b) & 0x7FFFFFFF),
            )
            hgb.fit(X_feat[tr], r_loo[tr])
            self.trees_.append(hgb)
            oob_sum[oob] += hgb.predict(X_feat[oob])
            oob_cnt[oob] += 1.0
        v = oob_cnt > 0
        r_oob = oob_sum[v] / oob_cnt[v]
        eta = float(np.dot(r_loo[v], r_oob)) / (float(np.dot(r_oob, r_oob)) + 1e-9)
        self.eta_ = float(np.clip(eta, 0.0, 0.95))
        return self

    def _basis(self, X_feat: np.ndarray) -> np.ndarray:
        Xt = np.clip(X_feat[:, self.top_idx_], -8.0, 8.0)
        cols = [X_feat]
        for q_i in range(3):
            cols.append(np.maximum(0.0, Xt - self.knots_[q_i][None, :]))
        return np.hstack(cols)

    def predict(self, X_feat: np.ndarray) -> np.ndarray:
        p_lin = self.y_mean_ + self._basis(X_feat) @ self.w_ridge_
        p_tree = np.mean([t.predict(X_feat) for t in self.trees_], axis=0)
        return p_lin + self.eta_ * p_tree


class RegimeRouteSubGPTree:
    def fit(self, X_feat: np.ndarray, y: np.ndarray, seed: int) -> "RegimeRouteSubGPTree":
        n, p = X_feat.shape
        self.global_gp_ = ActiveMaternRQSubGPDuan(top_k=14, max_exact=550).fit(X_feat, y, seed + 101)
        if n < 140 or p < 2:
            self.active_ = False
            return self
        yc = y - float(np.mean(y))
        best_score, best_col, best_thr, best_tau = -1.0, 0, 0.0, 0.5
        for j in range(min(p, 24)):
            col = X_feat[:, j]
            q50 = float(np.median(col))
            left = col <= q50
            right = ~left
            if int(np.sum(left)) < 45 or int(np.sum(right)) < 45:
                continue
            diff = abs(float(np.mean(yc[left])) - float(np.mean(yc[right]))) / (float(np.std(yc)) + 1e-8)
            if diff > best_score:
                best_score = diff
                best_col = j
                best_thr = q50
                best_tau = max(0.25 * float(np.std(col)), 1e-4)
        if best_score < 0.05:
            self.active_ = False
            return self
        self.active_ = True
        self.col_ = best_col
        self.thr_ = best_thr
        self.tau_ = best_tau
        left = X_feat[:, best_col] <= best_thr
        self.left_m_ = ActiveMaternRQSubGPDuan(top_k=12, max_exact=450).fit(X_feat[left], y[left], seed + 103)
        self.right_m_ = ActiveMaternRQSubGPDuan(top_k=12, max_exact=450).fit(X_feat[~left], y[~left], seed + 107)
        return self

    def predict(self, X_feat: np.ndarray) -> np.ndarray:
        p_glob = self.global_gp_.predict(X_feat)
        if not self.active_:
            return p_glob
        z = np.clip((X_feat[:, self.col_] - self.thr_) / self.tau_, -20.0, 20.0)
        w_r = 1.0 / (1.0 + np.exp(-z))
        p_loc = (1.0 - w_r) * self.left_m_.predict(X_feat) + w_r * self.right_m_.predict(X_feat)
        return 0.50 * p_glob + 0.50 * p_loc


def heads(X_tr: np.ndarray, y_tr: np.ndarray, X_q: np.ndarray, seed: int) -> tuple[dict[str, np.ndarray], dict]:
    """Fit all 9 heads (7 primary W2-W4 + 2 plain controls) strictly on (X_tr, y_tr) and predict on X_q."""
    eng = EntityHierarchyEngine()
    F_tr = eng.fit_transform(X_tr, y_tr, seed)
    F_q = eng.transform(X_q)

    prep = MultiViewPreprocessor().fit(X_tr, y_tr)
    Xs_tr, Xb_tr = prep.transform(X_tr)
    Xs_q, Xb_q = prep.transform(X_q)
    rel = _nonlinear_relevance(Xs_tr, Xb_tr, y_tr)

    # Plain median/IQR raw view (no EntityHierarchy, no Duan smearing) for controls
    med_raw = np.zeros(X_tr.shape[1], dtype=np.float64)
    iqr_raw = np.ones(X_tr.shape[1], dtype=np.float64)
    X_raw_tr = np.empty_like(X_tr, dtype=np.float64)
    X_raw_q = np.empty_like(X_q, dtype=np.float64)
    for j in range(X_tr.shape[1]):
        fin_tr = np.isfinite(X_tr[:, j])
        vals = X_tr[fin_tr, j] if np.any(fin_tr) else np.zeros(1)
        med = float(np.median(vals))
        q25, q75 = np.quantile(vals, [0.25, 0.75]) if len(vals) >= 4 else (med - 1.0, med + 1.0)
        sc = max(float(q75 - q25), float(np.std(vals)), 1e-6)
        med_raw[j], iqr_raw[j] = med, sc
        X_raw_tr[:, j] = np.clip((np.where(fin_tr, X_tr[:, j], med) - med) / sc, -10.0, 10.0)
        fin_q = np.isfinite(X_q[:, j])
        X_raw_q[:, j] = np.clip((np.where(fin_q, X_q[:, j], med) - med) / sc, -10.0, 10.0)

    # Primary heads
    m_cb_duan = DuanCatBoostBag(kind="yeo_johnson", depth=6, n_bags=2).fit(F_tr, y_tr, seed + 1)
    m_cb_rel_z = DuanCatBoostBag(kind="identity", depth=6, n_bags=2).fit(F_tr, y_tr, seed + 2)
    m_gp_duan = ActiveMaternRQSubGPDuan(top_k=16, max_exact=550, kinds=("identity", "yeo_johnson")).fit(F_tr, y_tr, seed + 3)
    m_rg_gp = SharpARDRankGaussSubGP(max_exact=650).fit(Xs_tr, Xb_tr, rel, y_tr)
    m_sp_duan = SplineRatioYJDuan().fit(Xs_tr, Xb_tr, rel, y_tr)
    m_dag_gp = DAGSubGPOnCatBoostResid().fit(F_tr, m_cb_rel_z, y_tr, seed + 4)
    m_dag_hgb = DAGHGBOnSplineRidgeResid().fit(F_tr, y_tr, seed + 5)
    m_route = RegimeRouteSubGPTree().fit(F_tr, y_tr, seed + 6)

    # Plain ablation controls (no EntityHierarchy, no Yeo-Johnson / Rank-Gauss Duan smearing, no DAG residuals)
    c_cb_plain = DuanCatBoostBag(kind="identity", depth=6, n_bags=2).fit(X_raw_tr, y_tr, seed + 7)
    c_gp_plain = ActiveMaternRQSubGPDuan(top_k=16, max_exact=550, kinds=("identity",)).fit(X_raw_tr, y_tr, seed + 8)

    preds = {
        "rel_cb_yjduan": m_cb_duan.predict(F_q),
        "matern_rq_duan": m_gp_duan.predict(F_q),
        "subgp_rgduan": m_rg_gp.predict(Xs_q, Xb_q),
        "spline_ratio_yjduan": m_sp_duan.predict(Xs_q, Xb_q),
        "dag_gp_on_cb": m_dag_gp.predict(F_q),
        "dag_hgb_on_spline": m_dag_hgb.predict(F_q),
        "regime_route_gp": m_route.predict(F_q),
        "cb_plain": c_cb_plain.predict(X_raw_q),
        "gp_plain": c_gp_plain.predict(X_raw_q),
    }
    info = {
        "entity_keys": len(eng.single_keys_),
        "pair_keys": len(eng.pair_keys_),
        "miss_cols": len(eng.miss_indicator_cols_),
        "dag_gp_eta": round(float(m_dag_gp.eta_), 4),
        "dag_hgb_eta": round(float(m_dag_hgb.eta_), 4),
        "route_active": bool(m_route.active_),
    }
    return preds, info


# ==================================================================================================
# PART 4: 100% ya-Free Peer-Consensus Extrapolation Guard + Multi-Loss BCAR + eigh L2 Combiner
# ==================================================================================================
def guard_candidate_pool_ya_free(
    ys: np.ndarray,
    lab: np.ndarray,
    Q: np.ndarray,
    bcols: list[np.ndarray | None],
    refit: list[np.ndarray | None],
    max_peer_sy: float = 10.0,
    rms_peer_sy: float = 3.5,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray | None], np.ndarray]:
    """Filters out candidates whose Xa predictions blow up relative to bounded top-7 OOF peer median.
    Uses ZERO `ya` information.
    """
    ys = np.asarray(ys, dtype=np.float64)
    ym = float(np.mean(ys))
    sy = max(float(np.std(ys)), 1e-9)
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    y_span = max(y_max - y_min, sy)
    na = len(bcols[0]) if bcols and bcols[0] is not None else 0
    mean_a = np.full(na, ym, dtype=np.float64)

    m = Q.shape[1]
    b_resolved: list[np.ndarray] = []
    r_resolved: list[np.ndarray | None] = []
    finite_mask = np.zeros(m, dtype=bool)
    s_oof = np.full(m, np.inf, dtype=np.float64)
    med_oof = np.full(m, np.inf, dtype=np.float64)

    for j in range(m):
        qj = Q[:, j]
        bj = bcols[j]
        rj = refit[j]
        if bj is None or not np.all(np.isfinite(bj)):
            bj = rj if (rj is not None and np.all(np.isfinite(rj))) else None
        if bj is None or not np.all(np.isfinite(qj)):
            b_resolved.append(mean_a)
            r_resolved.append(None)
            continue
        b_arr = np.asarray(bj, dtype=np.float64)
        r_arr = np.asarray(rj, dtype=np.float64) if (rj is not None and np.all(np.isfinite(rj))) else None
        b_resolved.append(b_arr)
        r_resolved.append(r_arr)
        finite_mask[j] = True
        s_oof[j] = float(np.sqrt(np.mean((qj - ys) ** 2)))
        f_rm = [float(np.sqrt(np.mean((qj[lab == k] - ys[lab == k]) ** 2))) for k in range(5) if np.any(lab == k)]
        med_oof[j] = float(np.median(f_rm)) if f_rm else s_oof[j]

    lo_bound = y_min - 4.0 * y_span
    hi_bound = y_max + 4.0 * y_span
    bounded_peers = [
        j
        for j in np.argsort(med_oof)
        if finite_mask[j] and (na == 0 or (float(np.min(b_resolved[j])) >= lo_bound and float(np.max(b_resolved[j])) <= hi_bound))
    ][:7]

    p_peer = np.median(np.column_stack([b_resolved[j] for j in bounded_peers]), axis=1) if (bounded_peers and na > 0) else mean_a

    kept: list[int] = []
    b_clean: list[np.ndarray] = []
    r_clean: list[np.ndarray | None] = []

    for j in range(m):
        if not finite_mask[j]:
            continue
        bj = b_resolved[j]
        rj = r_resolved[j]
        dev_b = np.abs(bj - p_peer) / sy if na > 0 else np.zeros(1)
        max_dev_b = float(np.max(dev_b)) if na > 0 else 0.0
        rms_dev_b = float(np.sqrt(np.mean(dev_b * dev_b))) if na > 0 else 0.0

        sc = max(s_oof[j], 0.03 * sy, 1e-9)
        if rj is not None and na > 0:
            drift_br = float(np.sqrt(np.mean((rj - bj) ** 2))) / sc
            max_dev_r = float(np.max(np.abs(rj - p_peer) / sy))
            rms_dev_r = float(np.sqrt(np.mean(((rj - p_peer) / sy) ** 2)))
            if max_dev_r > max_peer_sy or rms_dev_r > rms_peer_sy or (drift_br > 4.0 and max_dev_r > 4.0):
                rj = None
                drift_br = 0.0
        else:
            drift_br = 0.0

        if max_dev_b > max_peer_sy or rms_dev_b > rms_peer_sy or (drift_br > 4.0 and max_dev_b > 4.0):
            continue

        kept.append(j)
        b_clean.append(bj)
        r_clean.append(rj)

    if not kept:
        best_j = int(np.argmin(med_oof)) if np.any(finite_mask) else 0
        kept = [best_j]
        b_clean = [np.clip(b_resolved[best_j], lo_bound, hi_bound)]
        r_clean = [None]

    kept_arr = np.array(kept, dtype=np.int64)
    return Q[:, kept_arr], b_clean, r_clean, kept_arr


def _kcar_fast(R: np.ndarray, steps: int = 100) -> np.ndarray:
    r_norm2 = np.einsum("ij,ij->j", R, R)
    s = np.zeros(R.shape[0], dtype=np.float64)
    cnt = np.zeros(R.shape[1], dtype=np.float64)
    for _ in range(steps):
        dots = s @ R
        j = int(np.argmin(2.0 * dots + r_norm2))
        cnt[j] += 1.0
        s += R[:, j]
    return cnt


def _kcar_fold_logrmse_fast(R: np.ndarray, lab: np.ndarray, steps: int = 100, eps: float = 1e-12) -> np.ndarray:
    folds = [np.flatnonzero(lab == k) for k in range(5)]
    folds = [ix for ix in folds if len(ix) > 0]
    K = len(folds)
    Rf = [R[ix] for ix in folds]
    nf = [float(len(ix)) for ix in folds]
    r_norm2_f = [np.einsum("ij,ij->j", rf, rf) for rf in Rf]
    sf = [np.zeros(len(ix), dtype=np.float64) for ix in folds]
    s_norm2_f = [0.0] * K
    cnt = np.zeros(R.shape[1], dtype=np.float64)
    for t in range(1, steps + 1):
        inv_t2 = 1.0 / (t * t)
        score = np.zeros(R.shape[1], dtype=np.float64)
        for k in range(K):
            dots = sf[k] @ Rf[k]
            mse_k = np.maximum((s_norm2_f[k] + 2.0 * dots + r_norm2_f[k]) * (inv_t2 / nf[k]), 0.0)
            score += 0.5 * np.log(mse_k + eps)
        j = int(np.argmin(score))
        cnt[j] += 1.0
        for k in range(K):
            sf[k] += Rf[k][:, j]
            s_norm2_f[k] = float(sf[k] @ sf[k])
    return cnt


def _prune(w: np.ndarray, w_min: float = W_MIN, max_m: int = MAX_MEMBERS) -> np.ndarray:
    w = np.where(w >= w_min, w, 0.0)
    nz = np.flatnonzero(w > 0)
    if len(nz) == 0:
        w2 = np.zeros_like(w)
        w2[0] = 1.0
        return w2
    if len(nz) > max_m:
        keep = sorted(nz, key=lambda i: (-w[i], i))[:max_m]
        w2 = np.zeros_like(w)
        w2[keep] = w[keep]
        w = w2
    return w / w.sum()


def bcar_weights(Q: np.ndarray, y: np.ndarray, cid: str, mode: str = "mse", lab: np.ndarray | None = None) -> np.ndarray:
    n, m = Q.shape
    R_full = Q - y[:, None]
    sy = max(float(np.std(y)), 1e-12)
    col_mse_full = np.einsum("ij,ij->j", R_full, R_full) / max(n, 1)
    s_col = np.sqrt(np.maximum(col_mse_full, 0.0))
    min_s = float(np.min(s_col))

    # Precision-dilution guard (e.g. near-exact tables min_s < 0.03*sy, u_oof > 3.5):
    # prevent column-subsampling (br_mse) or finite-step Caruana noise from putting 0.01-0.03 weight
    # on orders-of-magnitude noisier columns alongside a near-exact candidate.
    if min_s < 0.03 * sy and m > 1:
        active = np.flatnonzero(s_col <= 3.0 * max(min_s, 1e-12 * sy))
        if len(active) == 0:
            active = np.array([int(np.argmin(s_col))], dtype=np.int64)
        if len(active) == 1:
            w_one = np.zeros(m, dtype=np.float64)
            w_one[int(active[0])] = 1.0
            return w_one
    else:
        active = np.arange(m, dtype=np.int64)

    R = R_full[:, active]
    ma = len(active)
    min_col_mse = float(np.min(col_mse_full[active]))
    var_y = max(float(np.var(y)), 1e-30)
    eps = max(1e-30, min(1e-12 * var_y, 1e-4 * max(min_col_mse, 1e-30)))
    row_w = None
    if mode == "huber":
        col_mse = col_mse_full[active]
        top8 = np.argsort(col_mse)[: min(8, ma)]
        r_med = np.median(R[:, top8], axis=1)
        scale = max(1.4826 * float(np.median(np.abs(r_med))), 1e-12 * sy)
        delta = 2.5 * scale
        abs_r = np.abs(r_med)
        row_w = np.where(abs_r <= delta, 1.0, delta / np.maximum(abs_r, 1e-30))
    acc = np.zeros(ma, dtype=np.float64)
    for b_idx in range(N_BAGS):
        rng = np.random.default_rng(_stream_seed(BAG_SEED, "BCAR", cid, b_idx))
        rows = rng.integers(0, n, size=n)
        if mode == "mse":
            acc += _kcar_fast(R[rows], 100) / 100.0
        elif mode == "huber":
            acc += _kcar_fast(R[rows] * np.sqrt(row_w[rows])[:, None], 100) / 100.0
        elif mode == "fold_log":
            acc += _kcar_fold_logrmse_fast(R[rows], lab[rows], 100, eps) / 100.0
        elif mode == "br_mse":
            cols = np.sort(rng.choice(ma, size=int(math.ceil(ma * 0.65)), replace=False)) if ma >= 2 else np.arange(ma)
            acc[cols] += _kcar_fast(R[np.ix_(rows, cols)], 100) / 100.0
    w_act = _prune(acc / N_BAGS)
    w_out = np.zeros(m, dtype=np.float64)
    w_out[active] = w_act
    return w_out


def _safe_deploy_cols(w: np.ndarray, Q: np.ndarray, y: np.ndarray, bcols: list[np.ndarray], refit: list[np.ndarray | None]):
    keep = np.flatnonzero(w > 0)
    ww = w[keep] / w[keep].sum()
    s_oof = np.sqrt(np.mean((Q - y[:, None]) ** 2, axis=0))
    sy = max(float(np.std(y)), 1e-9)
    cols_aud = []
    for idx in keep:
        b, r = bcols[idx], refit[idx]
        if r is None or not np.all(np.isfinite(r)):
            cols_aud.append(b)
            continue
        sc = max(float(s_oof[idx]), 1e-12 * sy, 1e-15)
        drift = float(np.sqrt(np.mean((r - b) ** 2))) / sc
        alpha = 0.50 / (1.0 + max(0.0, drift - 1.25) ** 2)
        cols_aud.append((1.0 - alpha) * b + alpha * r)
    p_oof = Q[:, keep] @ ww
    p_aud = np.column_stack(cols_aud) @ ww
    return p_oof, p_aud


def _matern32_gram(Za: np.ndarray, Zb: np.ndarray, ell: float) -> np.ndarray:
    d = max(Za.shape[1], 1)
    sc = ell * math.sqrt(d)
    A = Za / sc
    B = Zb / sc
    an2 = np.sum(A * A, axis=1, keepdims=True)
    bn2 = np.sum(B * B, axis=1, keepdims=True).T
    dist = np.sqrt(np.maximum(an2 + bn2 - 2.0 * (A @ B.T), 0.0))
    s = math.sqrt(3.0) * dist
    return (1.0 + s) * np.exp(-s) + 0.20 * (A @ B.T)


def fit_l2_residual_gate(
    Xs: np.ndarray,
    Xa: np.ndarray,
    y: np.ndarray,
    lab: np.ndarray,
    p1_oof: np.ndarray,
    p1_aud: np.ndarray,
    Q_all: np.ndarray,
    aud_cols_all: list[np.ndarray],
    n_gp: int,
) -> tuple[np.ndarray, float]:
    """Strictly SEARCH-only OOF-ratcheted Level-2 Input-Aware Residual Router."""
    r_raw = y - p1_oof
    s_base = float(np.sqrt(np.mean(r_raw * r_raw)))
    sy = max(float(np.std(y)), 1e-9)
    if s_base < 1e-5 * sy or len(y) > 3000:
        return p1_aud, 0.0

    base_f = np.array([float(np.sqrt(np.mean(r_raw[lab == k] ** 2))) for k in range(5)])
    if np.any(base_f < 1e-12):
        return p1_aud, 0.0

    prep = MultiViewPreprocessor().fit(Xs, y)
    Xs_s, _ = prep.transform(Xs)
    Xs_a, _ = prep.transform(Xa)
    top8 = np.sort(np.argsort(-prep.ard)[: min(8, Xs_s.shape[1])])

    f_rm = np.array([[np.sqrt(np.mean((Q_all[lab == k, j] - y[lab == k]) ** 2)) for k in range(5)] for j in range(Q_all.shape[1])])
    med_rm = np.median(f_rm, axis=1)
    gp_top = np.argsort(med_rm[:n_gp])[: min(4, n_gp)]
    aud_mat = np.column_stack(aud_cols_all)

    dev_s = [(Q_all[:, j] - p1_oof) / max(s_base, 1e-9) for j in gp_top]
    dev_a = [(aud_mat[:, j] - p1_aud) / max(s_base, 1e-9) for j in gp_top]

    if Q_all.shape[1] > n_gp:
        or_top = n_gp + np.argsort(med_rm[n_gp:])[: min(3, Q_all.shape[1] - n_gp)]
        for j in or_top:
            dev_s.append((Q_all[:, j] - p1_oof) / max(s_base, 1e-9))
            dev_a.append((aud_mat[:, j] - p1_aud) / max(s_base, 1e-9))
        diff_s = np.clip((Q_all[:, or_top[0]] - Q_all[:, gp_top[0]]) / max(s_base, 1e-9), -4.0, 4.0)
        diff_a = np.clip((aud_mat[:, or_top[0]] - aud_mat[:, gp_top[0]]) / max(s_base, 1e-9), -4.0, 4.0)
        for c in top8[: min(4, len(top8))]:
            dev_s.append(Xs_s[:, c] * diff_s)
            dev_a.append(Xs_a[:, c] * diff_a)

    spread_s = np.log1p(np.std(np.column_stack(dev_s[: min(6, len(dev_s))]), axis=1))
    spread_a = np.log1p(np.std(np.column_stack(dev_a[: min(6, len(dev_a))]), axis=1))

    # NOTE: Normalize strictly by SEARCH statistics (z_mu, z_sd) so query rows Xa have zero batch interaction!
    Zs_raw = np.clip(np.column_stack([Xs_s[:, top8]] + dev_s + [spread_s]), -5.0, 5.0)
    Za_raw = np.clip(np.column_stack([Xs_a[:, top8]] + dev_a + [spread_a]), -5.0, 5.0)
    z_mu = Zs_raw.mean(axis=0)
    z_sd = np.maximum(Zs_raw.std(axis=0), 1e-6)
    Zs = (Zs_raw - z_mu) / z_sd
    Za = np.clip((Za_raw - z_mu) / z_sd, -5.0, 5.0)

    mad_r = max(1.4826 * float(np.median(np.abs(r_raw - np.median(r_raw)))), 1e-6 * sy)
    r_train = np.clip(r_raw, -3.0 * mad_r, 3.0 * mad_r)

    best_gain = 0.0
    best_cfg = None
    for ell in (1.2, 2.5, 50.0):
        K_ss = _matern32_gram(Zs, Zs, ell)
        fold_eigh = []
        for k in range(5):
            va = lab == k
            tr = ~va
            ev, V = _safe_eigh(K_ss[np.ix_(tr, tr)])
            vt_r = V.T @ r_train[tr]
            kv_va = K_ss[np.ix_(va, tr)] @ V
            fold_eigh.append((va, ev, vt_r, kv_va))
        for lam in (2.0, 8.0, 25.0):
            oof_r = np.zeros_like(r_raw)
            for va, ev, vt_r, kv_va in fold_eigh:
                oof_r[va] = kv_va @ (vt_r / (ev + lam))
            for eta in (0.35, 0.65, 1.0):
                cand_oof = p1_oof + eta * oof_r
                new_f = np.array([float(np.sqrt(np.mean((y[lab == k] - cand_oof[lab == k]) ** 2))) for k in range(5)])
                folds_won = int(np.sum(new_f < base_f * 0.9995))
                gain = float(np.mean(np.log(base_f / np.maximum(new_f, 1e-15))))
                if folds_won >= 4 and gain > best_gain:
                    best_gain = gain
                    best_cfg = (ell, lam, eta)

    if best_cfg is None or best_gain < 0.004:
        return p1_aud, 0.0

    ell, lam, eta = best_cfg
    K_ss = _matern32_gram(Zs, Zs, ell)
    K_as = _matern32_gram(Za, Zs, ell)
    r_bag = np.zeros(len(p1_aud), dtype=np.float64)
    for k in range(5):
        tr = lab != k
        ev_k, V_k = _safe_eigh(K_ss[np.ix_(tr, tr)])
        alpha_k = V_k @ ((V_k.T @ r_train[tr]) / (ev_k + lam))
        r_bag += (K_as[:, tr] @ alpha_k) / 5.0
    ev_f, V_f = _safe_eigh(K_ss)
    alpha_f = V_f @ ((V_f.T @ r_train) / (ev_f + lam))
    r_full = K_as @ alpha_f
    r_aud = np.clip(0.5 * r_bag + 0.5 * r_full, -2.5 * mad_r, 2.5 * mad_r)
    return p1_aud + eta * r_aud, best_gain


def guard_and_stack_2l(
    cid: str,
    Xs: np.ndarray,
    Xa: np.ndarray,
    ys: np.ndarray,
    lab: np.ndarray,
    Q_gp: np.ndarray,
    bcols_gp: list[np.ndarray | None],
    refit_gp: list[np.ndarray | None],
    extra_blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    med_gate: float = 1.30,
    drift_gate: float = 2.50,
) -> tuple[np.ndarray, dict]:
    """Strictly SEARCH-only + ya-free Peer-Consensus Extrapolation Guard + Precision-Dilution Guard
    + H434 Diversity Bound + Multi-Loss BCAR + eigh L2 Ratchet."""
    sy = max(float(np.std(ys)), 1e-9)
    Q_gp_g, bcols_gp_g, refit_gp_g, kept_gp = guard_candidate_pool_ya_free(ys, lab, Q_gp, bcols_gp, refit_gp)
    n_gp_clean = Q_gp_g.shape[1]

    s_oof0 = np.sqrt(np.mean((Q_gp_g - ys[:, None]) ** 2, axis=0))
    min_oof0 = float(np.min(s_oof0))
    f_rm0 = np.array(
        [[np.sqrt(np.mean((Q_gp_g[lab == k, j] - ys[lab == k]) ** 2)) for k in range(5)] for j in range(n_gp_clean)]
    )
    min_med0 = float(np.median(f_rm0, axis=1).min())

    # Base OOF blend reference for ya-free H434 (overlap_rho, r_rat) diversity check
    w_ref0 = bcar_weights(Q_gp_g, ys, cid, "mse")
    e_base0 = Q_gp_g @ w_ref0 - ys
    rm_base0 = max(float(np.sqrt(np.mean(e_base0 * e_base0))), 1e-15)
    norm_base0 = max(float(np.linalg.norm(e_base0)), 1e-15)

    n_ext_admitted = 0
    if extra_blocks:
        Qo_raw = np.hstack([x[0] for x in extra_blocks])
        AKFo_raw = np.hstack([x[1] for x in extra_blocks])
        PAo_raw = np.hstack([x[2] for x in extra_blocks])

        Q_joint = np.hstack([Q_gp_g, Qo_raw])
        b_joint = bcols_gp_g + [AKFo_raw[:, j] for j in range(Qo_raw.shape[1])]
        r_joint = refit_gp_g + [PAo_raw[:, j] for j in range(Qo_raw.shape[1])]
        _, b_jg, r_jg, kept_j = guard_candidate_pool_ya_free(ys, lab, Q_joint, b_joint, r_joint)

        Q_ext_list: list[np.ndarray] = []
        b_ext_list: list[np.ndarray] = []
        r_ext_list: list[np.ndarray | None] = []
        for pos_k, orig_idx in enumerate(kept_j):
            if orig_idx < n_gp_clean:
                continue
            j = int(orig_idx - n_gp_clean)
            qj = Qo_raw[:, j]
            bj = b_jg[pos_k]
            rj = r_jg[pos_k]
            rm_j = float(np.sqrt(np.mean((qj - ys) ** 2)))
            if not math.isfinite(rm_j):
                continue
            med_j = float(np.median([np.sqrt(np.mean((qj[lab == k] - ys[lab == k]) ** 2)) for k in range(5)]))

            # Precision-dilution veto on high-precision / near-exact tables (e.g. cid 7d3e30f33f, u_oof > 3.5):
            if min_oof0 < 0.03 * sy:
                if rm_j > 3.0 * max(min_oof0, 1e-12 * sy) or med_j > 3.0 * max(min_med0, 1e-12 * sy):
                    continue

            denom_j = max(rm_j, 0.25 * min_med0, 1e-12 * sy, 1e-15)
            drift_j = float(np.sqrt(np.mean((rj - bj) ** 2))) / denom_j if rj is not None else 0.0
            if drift_j > drift_gate:
                rj = None  # demote unstable refit to clean fold-bag rather than dropping a strong column

            ej = qj - ys
            r_rat = rm_j / rm_base0
            rho = float(np.dot(e_base0, ej) / max(norm_base0 * float(np.linalg.norm(ej)), 1e-15))
            h434_ok = bool(
                min_oof0 >= 0.03 * sy
                and r_rat <= 2.0
                and rho <= (0.90 - 0.10 * (r_rat ** 2)) / max(0.80 * r_rat, 1e-12)
            )
            if med_j <= med_gate * min_med0 or rm_j <= med_gate * min_oof0 or h434_ok:
                Q_ext_list.append(qj[:, None])
                b_ext_list.append(bj)
                r_ext_list.append(rj)

        n_ext_admitted = len(Q_ext_list)
        if Q_ext_list:
            Q_all = np.hstack([Q_gp_g] + Q_ext_list)
            bcols_all = bcols_gp_g + b_ext_list
            refit_all = refit_gp_g + r_ext_list
        else:
            Q_all, bcols_all, refit_all = Q_gp_g, bcols_gp_g, refit_gp_g
    else:
        Q_all, bcols_all, refit_all = Q_gp_g, bcols_gp_g, refit_gp_g

    s_oof_all = np.sqrt(np.mean((Q_all - ys[:, None]) ** 2, axis=0))
    aud_cols_all = []
    for idx in range(Q_all.shape[1]):
        b, r = bcols_all[idx], refit_all[idx]
        if r is None or not np.all(np.isfinite(r)):
            aud_cols_all.append(b)
        else:
            sc = max(float(s_oof_all[idx]), 1e-12 * sy, 1e-15)
            drift = float(np.sqrt(np.mean((r - b) ** 2))) / sc
            alpha = 0.50 / (1.0 + max(0.0, drift - 1.25) ** 2)
            aud_cols_all.append((1.0 - alpha) * b + alpha * r)

    w_l1 = _prune(
        0.50 * bcar_weights(Q_all, ys, cid, "fold_log", lab)
        + 0.35 * bcar_weights(Q_all, ys, cid, "huber")
        + 0.15 * bcar_weights(Q_all, ys, cid, "br_mse")
    )
    p1_oof, p1_aud = _safe_deploy_cols(w_l1, Q_all, ys, bcols_all, refit_all)
    p2_aud, l2_gain = fit_l2_residual_gate(Xs, Xa, ys, lab, p1_oof, p1_aud, Q_all, aud_cols_all, n_gp_clean)
    meta = {
        "gp_kept": int(n_gp_clean),
        "gp_dropped": int(Q_gp.shape[1] - n_gp_clean),
        "ext_admitted": int(n_ext_admitted),
        "ext_weight_share": float(w_l1[n_gp_clean:].sum()) if Q_all.shape[1] > n_gp_clean else 0.0,
        "l2_gain": float(l2_gain),
    }
    return p2_aud, meta
