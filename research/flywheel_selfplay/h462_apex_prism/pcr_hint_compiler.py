"""Protocol Consistency Rule (`PCR`) Inner-OOF Hint & Reliability Compiler (`pcr_hint_compiler.py`).

Implements `SYMBIONT §7` + `Solver-Trained §6` strictly without label leakage or `Yeo-Johnson` warps:
  - **PCR Invariant (`T8`):** Every label-dependent hint (`Class B/C`: inner-OOF uncapped Matérn-5/2 GP
    mean & SD under `identity` and rank-Gauss `rg` warps ONLY, GCV linear ridge, ARD-kNN, and smoothed
    key target encoding) is computed **strictly out-of-fold** across `K_inner = 5` deterministic target-
    stratified inner folds on context rows `(x_context, y_context)`.
  - Query rows `x_query` receive fold-averaged predictions from those identical `5` inner-fold fits with
    **zero cross-query interaction (`T1 == 0.0`)**.
  - **Reliability Channels:** Every `Class B/C` hint is paired with a constant-per-context fold-stability
    skill channel `rel_j = max(0, 1 - rmse_oof_j / std(y_context))` plus per-row GP predictive SD / fold
    disagreement so `PRISM` knows when to trust or ignore each hint.
  - **Counterfactual Corrupted-Hint Training (`corrupt_pcr_hints`):** Corrupts `20%` of training-episode
    hint channels (sign flip, noise injection, or zeroing while keeping raw `X` intact) so `PRISM` never
    collapses to copying a single hint column.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import linalg as sla
from scipy import stats as sstats

from .assay_regime import _matern52_gram


@dataclasses.dataclass(frozen=True)
class PCRHintBundle:
    """Compiled context & query hint matrices plus `hint_protocol_id` SHA-256."""

    context_hints: np.ndarray
    query_hints: np.ndarray
    context_reliability: np.ndarray
    query_reliability: np.ndarray
    hint_names: Tuple[str, ...]
    hint_protocol_id: str


def _target_stratified_inner_folds(y_ctx: np.ndarray, k_inner: int = 5, seed: int = 462201) -> np.ndarray:
    y = np.asarray(y_ctx, dtype=np.float64).reshape(-1)
    order = np.argsort(y, kind="mergesort")
    rng = np.random.default_rng(int(seed))
    lab = np.zeros(y.shape[0], dtype=np.int64)
    for start in range(0, y.shape[0], int(k_inner)):
        blk = order[start : min(start + int(k_inner), y.shape[0])]
        lab[blk] = rng.permutation(int(k_inner))[: blk.size]
    return lab


class RankGaussTargetWarp:
    """Monotone empirical rank-Gauss (`rg`) target warp (**`Yeo-Johnson` banned**)."""

    def __init__(self) -> None:
        self.y_sorted = np.zeros(1, dtype=np.float64)
        self.z_grid = np.zeros(1, dtype=np.float64)

    def fit(self, y_tr: np.ndarray) -> "RankGaussTargetWarp":
        y = np.asarray(y_tr, dtype=np.float64).reshape(-1)
        self.y_sorted = np.sort(y)
        n = max(self.y_sorted.size, 1)
        u = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
        self.z_grid = sstats.norm.ppf(np.clip(u, 1e-4, 1.0 - 1e-4))
        return self

    def transform(self, y_in: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(y_in, dtype=np.float64), self.y_sorted, self.z_grid)

    def inverse_transform(self, z_in: np.ndarray) -> np.ndarray:
        z = np.asarray(z_in, dtype=np.float64)
        # Linear tail continuation using robust outer deciles (never explodes super-linearly!)
        lo_z, hi_z = float(self.z_grid[0]), float(self.z_grid[-1])
        lo_y, hi_y = float(self.y_sorted[0]), float(self.y_sorted[-1])
        q10_i = max(0, int(0.10 * (self.y_sorted.size - 1)))
        q90_i = min(self.y_sorted.size - 1, int(0.90 * (self.y_sorted.size - 1)))
        slope_r = (float(self.y_sorted[q90_i]) - float(self.y_sorted[q10_i])) / max(
            float(self.z_grid[q90_i] - self.z_grid[q10_i]), 1e-6
        )
        interp = np.interp(z, self.z_grid, self.y_sorted)
        below = z < lo_z
        above = z > hi_z
        out = interp.copy()
        if np.any(below):
            out[below] = lo_y + slope_r * (z[below] - lo_z)
        if np.any(above):
            out[above] = hi_y + slope_r * (z[above] - hi_z)
        return out


def _build_class_a_formula_hints(
    x_ctx: np.ndarray,
    y_ctx: np.ndarray,
    x_qu: np.ndarray,
    max_formulas: int = 6,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Score-screen `Class A` algebraic ratios, products, differences, and log-ratios on `(x_ctx, y_ctx)`."""
    n_c, n_cols = x_ctx.shape
    n_q = x_qu.shape[0]
    if n_cols < 2 or max_formulas <= 0:
        return np.zeros((n_c, 0), dtype=np.float64), np.zeros((n_q, 0), dtype=np.float64), []

    med = np.median(x_ctx, axis=0)
    q25, q75 = np.percentile(x_ctx, [25.0, 75.0], axis=0)
    sc = np.where((q75 - q25) > 1e-9, q75 - q25, np.maximum(np.std(x_ctx, axis=0), 1e-9))
    zc = (x_ctx - med) / sc
    zq = (x_qu - med) / sc

    yc = y_ctx - float(np.mean(y_ctx))
    y_norm = max(float(np.linalg.norm(yc)), 1e-12)
    col_scores = [abs(float(np.dot(zc[:, j], yc))) for j in range(n_cols)]
    top_cols = [int(i) for i in np.argsort(-np.asarray(col_scores))[: min(4, n_cols)]]

    cands: List[Tuple[float, str, np.ndarray, np.ndarray]] = []
    for a_pos in range(len(top_cols)):
        for b_pos in range(a_pos + 1, len(top_cols)):
            i, j = top_cols[a_pos], top_cols[b_pos]
            ops = [
                (f"prod_c{i}_c{j}", zc[:, i] * zc[:, j], zq[:, i] * zq[:, j]),
                (f"diff_c{i}_c{j}", zc[:, i] - zc[:, j], zq[:, i] - zq[:, j]),
                (
                    f"rat_c{i}_over_c{j}",
                    zc[:, i] / (np.abs(zc[:, j]) + 0.5),
                    zq[:, i] / (np.abs(zq[:, j]) + 0.5),
                ),
                (
                    f"lograt_c{i}_c{j}",
                    np.log((np.abs(zc[:, i]) + 0.25) / (np.abs(zc[:, j]) + 0.25)),
                    np.log((np.abs(zq[:, i]) + 0.25) / (np.abs(zq[:, j]) + 0.25)),
                ),
            ]
            for nm, vc, vq in ops:
                vc_c = vc - float(np.mean(vc))
                nrm = float(np.linalg.norm(vc_c))
                if nrm <= 1e-9:
                    continue
                score = abs(float(np.dot(vc_c, yc))) / (nrm * y_norm)
                mu_v = float(np.median(vc))
                iqr_v = float(np.percentile(vc, 75.0) - np.percentile(vc, 25.0))
                sd_v = iqr_v if iqr_v > 1e-9 else max(float(np.std(vc)), 1e-9)
                cands.append((score, nm, (vc - mu_v) / sd_v, (vq - mu_v) / sd_v))

    cands.sort(key=lambda item: -item[0])
    sel = cands[: int(max_formulas)]
    if not sel:
        return np.zeros((n_c, 0), dtype=np.float64), np.zeros((n_q, 0), dtype=np.float64), []
    return (
        np.column_stack([s[2] for s in sel]),
        np.column_stack([s[3] for s in sel]),
        [s[1] for s in sel],
    )


def _fit_predict_fold_priors(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_eval: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit inner-fold `{ridge, gp_id_mean, gp_id_sd, gp_rg_mean, ard_knn, key_te}` on `(x_tr, y_tr)`."""
    n_tr, n_cols = x_tr.shape
    n_ev = x_eval.shape[0]
    y_mu = float(np.mean(y_tr))
    yc = y_tr - y_mu
    y_var = max(float(np.var(yc)), 1e-9)

    med = np.median(x_tr, axis=0)
    q25, q75 = np.percentile(x_tr, [25.0, 75.0], axis=0)
    sc = np.where((q75 - q25) > 1e-9, q75 - q25, np.maximum(np.std(x_tr, axis=0), 1e-9))
    z_tr = np.clip((x_tr - med) / sc, -20.0, 20.0)
    z_ev = np.clip((x_eval - med) / sc, -20.0, 20.0)

    # 1. GCV linear ridge
    u_svd, s_svd, vt_svd = sla.svd(z_tr, full_matrices=False, check_finite=False)
    uty = u_svd.T @ yc
    best_gcv = float("inf")
    best_lam = 10.0
    for lam in (0.1, 1.0, 10.0, 50.0, 250.0):
        filt = (s_svd * s_svd) / (s_svd * s_svd + lam)
        res_r = yc - u_svd @ (filt * uty)
        denom_gcv = max(1.0 - float(np.sum(filt)) / float(n_tr), 1e-3) ** 2
        gcv = float(np.mean(res_r * res_r)) / denom_gcv
        if gcv < best_gcv:
            best_gcv = gcv
            best_lam = lam
    w_ridge = vt_svd.T @ ((s_svd / (s_svd * s_svd + best_lam)) * uty)
    pred_ridge = y_mu + z_ev @ w_ridge

    # 2. Uncapped Matérn-5/2 GP (identity target) + predictive SD
    best_loo = float("inf")
    best_ell = 1.4
    best_rho = 1e-2
    best_ev_vecs: Optional[Tuple[np.ndarray, np.ndarray]] = None
    for ell in (0.7, 1.8):
        k_tt = _matern52_gram(z_tr, z_tr, ell)
        ev_k, v_k = sla.eigh(0.5 * (k_tt + k_tt.T), check_finite=False)
        ev_k = np.maximum(ev_k, 0.0)
        vty = v_k.T @ yc
        vsq = v_k * v_k
        for rho in (1e-6, 1e-3, 2e-2, 0.2):
            inv_d = 1.0 / (ev_k + rho)
            alpha = v_k @ (vty * inv_d)
            e_loo = alpha / np.maximum(vsq @ inv_d, 1e-15)
            rmse = float(math.sqrt(np.mean(e_loo * e_loo)))
            if rmse < best_loo:
                best_loo = rmse
                best_ell = ell
                best_rho = rho
                best_ev_vecs = (ev_k, v_k)

    assert best_ev_vecs is not None
    ev_g, v_g = best_ev_vecs
    inv_dg = 1.0 / (ev_g + best_rho)
    alpha_id = v_g @ ((v_g.T @ yc) * inv_dg)
    k_et = _matern52_gram(z_ev, z_tr, best_ell)
    pred_gp_id = y_mu + k_et @ alpha_id

    # Predictive variance k(x*, x*) - k_*^T (K + rho I)^{-1} k_*
    d_cols = max(int(z_ev.shape[1]), 1)
    k_diag_ev = 1.0 + 0.15 * np.sum((z_ev / (best_ell * math.sqrt(float(d_cols)))) ** 2, axis=1)
    proj_v = k_et @ v_g
    var_ratio = np.maximum(k_diag_ev - (proj_v * proj_v) @ inv_dg, 1e-9)
    pred_gp_sd = np.sqrt(y_var * var_ratio)

    # 3. Rank-Gauss (`rg`) Matérn-5/2 GP (monotone, outlier-resistant, NO Yeo-Johnson!)
    rg = RankGaussTargetWarp().fit(y_tr)
    z_y = rg.transform(y_tr)
    alpha_rg = v_g @ ((v_g.T @ z_y) / (ev_g + max(best_rho, 1e-2)))
    pred_gp_rg = rg.inverse_transform(k_et @ alpha_rg)

    # 4. ARD distance-weighted kNN (k = min(12, n_tr - 1)) with strictly train-only bandwidth (`T1 == 0.0`)
    ard_w = np.abs(w_ridge)
    ard_w = ard_w / max(float(np.mean(ard_w)), 1e-6)
    ard_w = np.clip(ard_w, 0.2, 3.0)
    zw_tr = z_tr * ard_w[None, :]
    zw_ev = z_ev * ard_w[None, :]
    k_nn = min(12, max(1, n_tr - 1))
    d2_tr = (
        np.sum(zw_tr * zw_tr, axis=1, keepdims=True)
        + np.sum(zw_tr * zw_tr, axis=1, keepdims=True).T
        - 2.0 * (zw_tr @ zw_tr.T)
    )
    d2_tr = np.maximum(d2_tr, 0.0)
    np.fill_diagonal(d2_tr, np.inf)
    nn_tr_d2 = np.partition(d2_tr, kth=k_nn - 1, axis=1)[:, :k_nn]
    med_d2 = max(float(np.median(nn_tr_d2)), 1e-6)

    d2 = (
        np.sum(zw_ev * zw_ev, axis=1, keepdims=True)
        + np.sum(zw_tr * zw_tr, axis=1, keepdims=True).T
        - 2.0 * (zw_ev @ zw_tr.T)
    )
    d2 = np.maximum(d2, 0.0)
    nn_idx = np.argpartition(d2, kth=k_nn - 1, axis=1)[:, :k_nn]
    nn_d2 = np.take_along_axis(d2, nn_idx, axis=1)
    w_nn = np.exp(-nn_d2 / (2.0 * med_d2))
    w_nn = w_nn / np.maximum(np.sum(w_nn, axis=1, keepdims=True), 1e-12)
    pred_knn = np.sum(w_nn * y_tr[nn_idx], axis=1)

    # 5. Smoothed key target encoding on highest-multiplicity discrete column (or binned top ARD col)
    best_j = int(np.argmax(ard_w))
    mult_best = 0.0
    for j in range(n_cols):
        u_c = max(int(np.unique(np.round(x_tr[:, j], 8)).size), 1)
        mult = float(n_tr) / float(u_c)
        if mult > mult_best and u_c >= 2:
            mult_best = mult
            best_j = j
    if mult_best >= 3.0:
        keys_tr = np.round(x_tr[:, best_j], 8)
        keys_ev = np.round(x_eval[:, best_j], 8)
    else:
        edges = np.quantile(x_tr[:, best_j], np.linspace(0.0, 1.0, 9)[1:-1])
        keys_tr = np.searchsorted(edges, x_tr[:, best_j]).astype(np.float64)
        keys_ev = np.searchsorted(edges, x_eval[:, best_j]).astype(np.float64)

    uniq_k, inv_tr, counts = np.unique(keys_tr, return_inverse=True, return_counts=True)
    sums = np.bincount(inv_tr, weights=y_tr, minlength=uniq_k.size)
    tau_te = 8.0
    te_vals = (sums + tau_te * y_mu) / (counts.astype(np.float64) + tau_te)
    key_map = {float(k_v): float(v_v) for k_v, v_v in zip(uniq_k, te_vals)}
    pred_te = np.array([key_map.get(float(kv), y_mu) for kv in keys_ev], dtype=np.float64)

    return pred_ridge, pred_gp_id, pred_gp_sd, pred_gp_rg, pred_knn, pred_te


def compile_pcr_hints(
    x_context: np.ndarray,
    y_context: np.ndarray,
    x_query: np.ndarray,
    k_inner: int = 5,
    max_formulas: int = 6,
    seed: int = 462201,
) -> PCRHintBundle:
    """Compile strictly inner-OOF `PCR` hints + reliability channels (`T1 == 0.0`, `T8` leak-free)."""
    x_c = np.ascontiguousarray(x_context, dtype=np.float64)
    y_c = np.ascontiguousarray(y_context, dtype=np.float64).reshape(-1)
    x_q = np.ascontiguousarray(x_query, dtype=np.float64)
    n_c = x_c.shape[0]
    n_q = x_q.shape[0]
    y_med = float(np.median(y_c))
    y_iqr = float(np.percentile(y_c, 75.0) - np.percentile(y_c, 25.0))
    y_scale = y_iqr if y_iqr > 1e-9 else max(float(np.std(y_c)), 1e-9)
    y_std = max(float(np.std(y_c)), 1e-12)

    fa_c, fa_q, fa_names = _build_class_a_formula_hints(x_c, y_c, x_q, max_formulas=max_formulas)

    k_eff = min(max(2, int(k_inner)), max(2, n_c // 4))
    lab = _target_stratified_inner_folds(y_c, k_inner=k_eff, seed=seed)

    n_priors = 6  # ridge, gp_id, gp_sd, gp_rg, knn, key_te
    oof_priors = np.zeros((n_c, n_priors), dtype=np.float64)
    fold_q_preds = np.zeros((k_eff, n_q, n_priors), dtype=np.float64)

    for k in range(k_eff):
        va = lab == k
        tr = ~va
        n_va = int(np.sum(va))
        both_ev = np.vstack([x_c[va], x_q])
        p_ridge, p_gpid, p_gpsd, p_gprg, p_knn, p_te = _fit_predict_fold_priors(x_c[tr], y_c[tr], both_ev)
        stacked = np.column_stack([p_ridge, p_gpid, p_gpsd, p_gprg, p_knn, p_te])
        oof_priors[va] = stacked[:n_va]
        fold_q_preds[k] = stacked[n_va:]

    q_priors = np.mean(fold_q_preds, axis=0)
    q_fold_std = np.std(fold_q_preds[:, :, 1], axis=0)  # GP fold disagreement on query

    # Normalize location-valued hints (`0, 1, 3, 4, 5`) in `(y - y_med) / y_scale` units; SD (`2`) in `/ y_scale`
    loc_cols = [0, 1, 3, 4, 5]
    c_norm = np.zeros_like(oof_priors)
    q_norm = np.zeros_like(q_priors)
    for idx in loc_cols:
        c_norm[:, idx] = (oof_priors[:, idx] - y_med) / y_scale
        q_norm[:, idx] = (q_priors[:, idx] - y_med) / y_scale
    c_norm[:, 2] = oof_priors[:, 2] / y_scale
    q_norm[:, 2] = q_priors[:, 2] / y_scale

    prior_names = [
        "pcr_ridge_oof",
        "pcr_gp_id_oof",
        "pcr_gp_sd_oof",
        "pcr_gp_rg_oof",
        "pcr_ard_knn_oof",
        "pcr_key_te_oof",
    ]
    skills = []
    for idx in loc_cols:
        rmse_j = float(math.sqrt(np.mean((y_c - oof_priors[:, idx]) ** 2)))
        skills.append(max(0.0, 1.0 - rmse_j / y_std))
    rel_vec = np.array(skills + [float(np.median(c_norm[:, 2]))], dtype=np.float64)
    c_rel = np.column_stack([np.tile(rel_vec, (n_c, 1)), c_norm[:, 2]])
    q_rel = np.column_stack([np.tile(rel_vec, (n_q, 1)), q_fold_std / y_scale])

    c_all = np.hstack([fa_c, c_norm]) if fa_c.shape[1] > 0 else c_norm
    q_all = np.hstack([fa_q, q_norm]) if fa_q.shape[1] > 0 else q_norm
    all_names = tuple(fa_names + prior_names)

    proto_spec = json.dumps(
        {
            "ban_yeo_johnson": True,
            "hint_names": list(prior_names),
            "k_inner": int(k_eff),
            "max_formulas": int(max_formulas),
            "schema": "jope.h462.pcr_hints/v1",
            "warps": ["identity", "rank_gauss"],
        },
        sort_keys=True,
    ).encode("utf-8")
    proto_id = hashlib.sha256(proto_spec).hexdigest()[:16]

    return PCRHintBundle(
        context_hints=c_all,
        query_hints=q_all,
        context_reliability=c_rel,
        query_reliability=q_rel,
        hint_names=all_names,
        hint_protocol_id=f"pcr_v1_{proto_id}",
    )


def corrupt_pcr_hints_counterfactual(
    bundle: PCRHintBundle,
    corrupt_prob: float = 0.20,
    seed: int = 462299,
) -> Tuple[PCRHintBundle, np.ndarray]:
    """Corrupt a subset of hint channels identically on context & query (`SYMBIONT §10.4` repair training)."""
    rng = np.random.default_rng(int(seed))
    n_h = bundle.context_hints.shape[1]
    mask = rng.uniform(size=n_h) < float(corrupt_prob)
    c_h = bundle.context_hints.copy()
    q_h = bundle.query_hints.copy()
    c_r = bundle.context_reliability.copy()
    q_r = bundle.query_reliability.copy()

    for j in np.flatnonzero(mask):
        mode = int(rng.integers(0, 3))
        if mode == 0:
            c_h[:, j] = -1.5 * c_h[:, j]
            q_h[:, j] = -1.5 * q_h[:, j]
        elif mode == 1:
            c_h[:, j] = rng.normal(scale=2.0, size=c_h.shape[0])
            q_h[:, j] = rng.normal(scale=2.0, size=q_h.shape[0])
        else:
            c_h[:, j] = 0.0
            q_h[:, j] = 0.0
    if np.any(mask) and c_r.shape[1] > 0:
        c_r[:, 0] *= 0.25
        q_r[:, 0] *= 0.25

    return (
        PCRHintBundle(
            context_hints=c_h,
            query_hints=q_h,
            context_reliability=c_r,
            query_reliability=q_r,
            hint_names=bundle.hint_names,
            hint_protocol_id=bundle.hint_protocol_id + "_corrupted",
        ),
        mask,
    )
