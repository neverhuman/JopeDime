"""Protocol Consistency Rule (`PCR`) Inner-OOF Hint & Reliability Compiler (`pcr_hint_compiler.py`).

Implements `HNC-1` + `Correction 1` (`cl_fwriter`, `2026-11-15T14:11Z` / `14:13Z`):
  - **H1 Self-Exclusion:** Label-free deterministic inner fold assignment (`_label_free_inner_folds`) +
    strictly per-inner-fold train-only target normalization `(y_med_tr, y_scale_tr)` and hyperparameter
    selection. Adding `+1000 * sd(y)` to `y_i` leaves every non-selected channel at row `i` unchanged to
    `0.0` (`< 1e-12` relative).
  - **H1 Exception + Finding 2 (`Class A` Cross-Fitted Select-Then-Score Reliability):** `Class A` algebraic
    formula definitions `(op, col_i, col_j)` are selected once per context on `C`, applied identically to `C`
    and `Q` via inner-fold 1D ridge location fits, and paired with cross-fitted select-then-score reliability
    channels (`select on C \\ fold_k, fit on C \\ fold_k, score on fold_k`).
  - **H2 Query Exclusion (`T1 == 0.0`):** Query hints depend only on context labels and each query row's own `x`.
  - **H3 Row-Type Parity (Finding 1 & Correction 1 Finding 7):**
    * Both context and query reliability matrices carry `gp_sd` in the same slot; query-only `gp_fold_disagreement`
      is placed in its own dedicated channel paired with an explicit `gp_fold_disagreement_missing` indicator
      (`1.0` on context rows, `0.0` on query rows).
    * Query hints come from a single inner-fold model `k_q` chosen deterministically at random per episode
      (`Correction 1` Fix `(a)`), matching the `(k_eff - 1) / k_eff` single-fold variance of context OOF hints.
  - **H4 Train/Deploy Parity (Finding 3):** `corrupt_pcr_hints_counterfactual` recomputes ONLY the corrupted
    channels' reliability from the corrupted values via the exact deploy `L350` skill formula, leaving
    uncorrupted channels untouched (no training-only `* 0.25` flag).
  - **H7 Protocol Identity (Finding 4):** `hint_protocol_id` is the 64-hex SHA-256 of the compiler module
    source bytes plus every compiler parameter; data-dependent `k_eff` is recorded separately on `PCRHintBundle`.
  - **H8 Calibration Canary:** Over 200 synthetic episodes (`max_formulas = 6`), context-side OOF skill and
    query-side skill match within `<= 0.05` on average across every location hint channel.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import linalg as sla
from scipy import stats as sstats

from .assay_regime import _matern52_gram

_COMPILER_SOURCE_SHA256: str = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

PRIOR_HINT_NAMES: Tuple[str, ...] = (
    "pcr_ridge_oof",
    "pcr_gp_id_oof",
    "pcr_gp_sd_oof",
    "pcr_gp_rg_oof",
    "pcr_ard_knn_oof",
    "pcr_key_te_oof",
)
PRIOR_LOC_INDICES: Tuple[int, ...] = (0, 1, 3, 4, 5)
PRIOR_GPSD_INDEX: int = 2


@dataclasses.dataclass(frozen=True)
class PCRHintBundle:
    """Compiled context & query hint matrices, reliability matrices, and `HNC-1` metadata."""

    context_hints: np.ndarray
    query_hints: np.ndarray
    context_reliability: np.ndarray
    query_reliability: np.ndarray
    hint_names: Tuple[str, ...]
    reliability_names: Tuple[str, ...]
    hint_protocol_id: str
    k_eff: int
    query_fold_index: int
    selected_formula_channels: Tuple[int, ...]
    non_selected_channels: Tuple[int, ...]
    skill_col_for_hint: Tuple[Optional[int], ...]
    gp_sd_hint_col: int
    gp_sd_median_rel_col: int
    gp_sd_row_rel_col: int
    context_target_norm: np.ndarray
    oof_predictions_raw: np.ndarray
    query_predictions_raw: np.ndarray


def compute_l350_skill(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Canonical `L350` truncated RMSE skill `max(0, 1 - rmse(y, y_hat) / std(y))`."""
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if yt.size == 0:
        return 0.0
    y_std = max(float(np.std(yt)), 1e-12)
    rmse = float(math.sqrt(np.mean((yt - yp) ** 2)))
    return float(max(0.0, 1.0 - rmse / y_std))


def _label_free_inner_folds(n_c: int, k_inner: int = 5, seed: int = 462201) -> np.ndarray:
    """Deterministic `y`-independent inner fold assignment (`H1` self-exclusion)."""
    rng = np.random.default_rng(int(seed) ^ 0x517CC1B7)
    perm = rng.permutation(int(n_c))
    lab = np.zeros(int(n_c), dtype=np.int64)
    lab[perm] = np.arange(int(n_c), dtype=np.int64) % int(k_inner)
    return lab


def _robust_loc_scale(v: np.ndarray) -> Tuple[float, float]:
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    med = float(np.median(arr))
    q25, q75 = np.percentile(arr, [25.0, 75.0])
    iqr = float(q75 - q25)
    sc = iqr if iqr > 1e-9 else max(float(np.std(arr)), 1e-9)
    return med, sc


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


def _eval_formula_op(z_mat: np.ndarray, op: str, i: int, j: int) -> np.ndarray:
    zi = z_mat[:, i]
    zj = z_mat[:, j]
    if op == "prod":
        return zi * zj
    if op == "diff":
        return zi - zj
    if op == "rat":
        return zi / (np.abs(zj) + 0.5)
    if op == "lograt":
        return np.log((np.abs(zi) + 0.25) / (np.abs(zj) + 0.25))
    raise ValueError(f"Unknown Class A formula op: {op!r}")


def _select_class_a_definitions(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    max_formulas: int = 6,
) -> List[Tuple[str, int, int, str]]:
    """Select up to `max_formulas` algebraic formula definitions `(op, col_i, col_j, name)` on `(x_fit, y_fit)`."""
    n_r, n_cols = x_fit.shape
    if n_cols < 2 or max_formulas <= 0 or n_r < 4:
        return []

    med = np.median(x_fit, axis=0)
    q25, q75 = np.percentile(x_fit, [25.0, 75.0], axis=0)
    sc = np.where((q75 - q25) > 1e-9, q75 - q25, np.maximum(np.std(x_fit, axis=0), 1e-9))
    zf = np.clip((x_fit - med) / sc, -20.0, 20.0)

    yc = y_fit - float(np.mean(y_fit))
    y_norm = max(float(np.linalg.norm(yc)), 1e-12)
    col_scores = [abs(float(np.dot(zf[:, j], yc))) for j in range(n_cols)]
    top_cols = [int(i) for i in np.argsort(-np.asarray(col_scores))[: min(4, n_cols)]]

    cands: List[Tuple[float, str, int, int, str]] = []
    for a_pos in range(len(top_cols)):
        for b_pos in range(a_pos + 1, len(top_cols)):
            i, j = top_cols[a_pos], top_cols[b_pos]
            for op, nm in (
                ("prod", f"prod_c{i}_c{j}"),
                ("diff", f"diff_c{i}_c{j}"),
                ("rat", f"rat_c{i}_over_c{j}"),
                ("lograt", f"lograt_c{i}_c{j}"),
            ):
                vf = _eval_formula_op(zf, op, i, j)
                vf_c = vf - float(np.mean(vf))
                nrm = float(np.linalg.norm(vf_c))
                if nrm <= 1e-9:
                    score = -1.0
                else:
                    score = abs(float(np.dot(vf_c, yc))) / (nrm * y_norm)
                cands.append((score, op, i, j, nm))

    cands.sort(key=lambda item: (-item[0], item[4]))
    return [(c[1], c[2], c[3], c[4]) for c in cands[: int(max_formulas)]]


def _fit_predict_1d_formulas(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_eval: np.ndarray,
    defs: Sequence[Tuple[str, int, int, str]],
    target_slots: int,
) -> np.ndarray:
    """Fit 1D damped linear predictors on `(x_tr, y_tr)` for `defs`, padding missing slots with `mean(y_tr)`."""
    n_ev = x_eval.shape[0]
    y_mu = float(np.mean(y_tr))
    out = np.full((n_ev, int(target_slots)), y_mu, dtype=np.float64)
    if target_slots <= 0 or not defs:
        return out

    med = np.median(x_tr, axis=0)
    q25, q75 = np.percentile(x_tr, [25.0, 75.0], axis=0)
    sc = np.where((q75 - q25) > 1e-9, q75 - q25, np.maximum(np.std(x_tr, axis=0), 1e-9))
    z_tr = np.clip((x_tr - med) / sc, -20.0, 20.0)
    z_ev = np.clip((x_eval - med) / sc, -20.0, 20.0)
    yc = y_tr - y_mu

    for r_idx, (op, col_i, col_j, _nm) in enumerate(defs[: int(target_slots)]):
        f_tr = _eval_formula_op(z_tr, op, col_i, col_j)
        f_ev = _eval_formula_op(z_ev, op, col_i, col_j)
        f_mu = float(np.mean(f_tr))
        fc_tr = f_tr - f_mu
        fc_ev = np.clip(f_ev - f_mu, -25.0, 25.0)
        denom = float(np.dot(fc_tr, fc_tr)) + 2.0
        slope = float(np.dot(fc_tr, yc)) / max(denom, 1e-9)
        out[:, r_idx] = y_mu + slope * fc_ev
    return out


def _fit_predict_fold_priors(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_eval: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit inner-fold `{ridge, gp_id_mean, gp_id_sd, gp_rg_mean, ard_knn, key_te}` strictly on `(x_tr, y_tr)`."""
    n_tr, n_cols = x_tr.shape
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

    d_cols = max(int(z_ev.shape[1]), 1)
    k_diag_ev = 1.0 + 0.15 * np.sum((z_ev / (best_ell * math.sqrt(float(d_cols)))) ** 2, axis=1)
    proj_v = k_et @ v_g
    var_ratio = np.maximum(k_diag_ev - (proj_v * proj_v) @ inv_dg, 1e-9)
    pred_gp_sd = np.sqrt(y_var * var_ratio)

    # 3. Rank-Gauss (`rg`) Matérn-5/2 GP (`Yeo-Johnson` banned)
    rg = RankGaussTargetWarp().fit(y_tr)
    z_y = rg.transform(y_tr)
    alpha_rg = v_g @ ((v_g.T @ z_y) / (ev_g + max(best_rho, 1e-2)))
    pred_gp_rg = rg.inverse_transform(k_et @ alpha_rg)

    # 4. ARD distance-weighted kNN with strictly train-only bandwidth (`T1 == 0.0`)
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

    # 5. Smoothed key target encoding
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
    query_fold_mode: str = "random_single_fold",
) -> PCRHintBundle:
    """Compile strictly inner-OOF `PCR` hints + `HNC-1` compliant reliability channels."""
    if query_fold_mode not in ("random_single_fold", "fold_mean_legacy"):
        raise ValueError(f"Unsupported query_fold_mode: {query_fold_mode!r}")

    x_c = np.ascontiguousarray(x_context, dtype=np.float64)
    y_c = np.ascontiguousarray(y_context, dtype=np.float64).reshape(-1)
    x_q = np.ascontiguousarray(x_query, dtype=np.float64)
    n_c = x_c.shape[0]
    n_q = x_q.shape[0]

    # Once-per-context Class A formula definition selection (H1 exception conditions 1 & 2)
    fa_defs = _select_class_a_definitions(x_c, y_c, max_formulas=int(max_formulas))
    m_fa = len(fa_defs)
    fa_names = [f"pcr_fa_{nm}" for (_op, _i, _j, nm) in fa_defs]

    k_eff = min(max(2, int(k_inner)), max(2, n_c // 4))
    lab = _label_free_inner_folds(n_c, k_inner=k_eff, seed=int(seed))
    rng_q = np.random.default_rng(int(seed) ^ 0x2C1B3F91)
    k_q = int(rng_q.integers(0, k_eff))

    n_priors = len(PRIOR_HINT_NAMES)
    oof_priors_raw = np.zeros((n_c, n_priors), dtype=np.float64)
    oof_priors_norm = np.zeros((n_c, n_priors), dtype=np.float64)
    fold_q_priors_raw = np.zeros((k_eff, n_q, n_priors), dtype=np.float64)
    fold_q_priors_norm = np.zeros((k_eff, n_q, n_priors), dtype=np.float64)

    oof_fa_fixed_raw = np.zeros((n_c, m_fa), dtype=np.float64)
    oof_fa_cf_raw = np.zeros((n_c, m_fa), dtype=np.float64)
    oof_fa_norm = np.zeros((n_c, m_fa), dtype=np.float64)
    fold_q_fa_raw = np.zeros((k_eff, n_q, m_fa), dtype=np.float64)
    fold_q_fa_norm = np.zeros((k_eff, n_q, m_fa), dtype=np.float64)

    for k in range(k_eff):
        va = lab == k
        tr = ~va
        n_va = int(np.sum(va))
        y_med_tr, y_scale_tr = _robust_loc_scale(y_c[tr])

        both_ev = np.vstack([x_c[va], x_q])
        p_ridge, p_gpid, p_gpsd, p_gprg, p_knn, p_te = _fit_predict_fold_priors(x_c[tr], y_c[tr], both_ev)
        stacked_raw = np.column_stack([p_ridge, p_gpid, p_gpsd, p_gprg, p_knn, p_te])
        stacked_norm = np.zeros_like(stacked_raw)
        for idx in PRIOR_LOC_INDICES:
            stacked_norm[:, idx] = (stacked_raw[:, idx] - y_med_tr) / y_scale_tr
        stacked_norm[:, PRIOR_GPSD_INDEX] = stacked_raw[:, PRIOR_GPSD_INDEX] / y_scale_tr

        oof_priors_raw[va] = stacked_raw[:n_va]
        oof_priors_norm[va] = stacked_norm[:n_va]
        fold_q_priors_raw[k] = stacked_raw[n_va:]
        fold_q_priors_norm[k] = stacked_norm[n_va:]

        if m_fa > 0:
            # Same-definition 1D location fits for context/query channel values (H1 cond 2 + H3 parity)
            fa_both_raw = _fit_predict_1d_formulas(x_c[tr], y_c[tr], both_ev, fa_defs, m_fa)
            fa_both_norm = (fa_both_raw - y_med_tr) / y_scale_tr
            oof_fa_fixed_raw[va] = fa_both_raw[:n_va]
            oof_fa_norm[va] = fa_both_norm[:n_va]
            fold_q_fa_raw[k] = fa_both_raw[n_va:]
            fold_q_fa_norm[k] = fa_both_norm[n_va:]

            # Cross-fitted select-then-score on C \ fold_k -> score on fold_k (H1 cond 3 + Finding 2)
            defs_tr = _select_class_a_definitions(x_c[tr], y_c[tr], max_formulas=m_fa)
            oof_fa_cf_raw[va] = _fit_predict_1d_formulas(x_c[tr], y_c[tr], x_c[va], defs_tr, m_fa)

    if query_fold_mode == "random_single_fold":
        q_priors_raw = fold_q_priors_raw[k_q]
        q_priors_norm = fold_q_priors_norm[k_q]
        q_fa_raw = fold_q_fa_raw[k_q]
        q_fa_norm = fold_q_fa_norm[k_q]
    else:
        q_priors_raw = np.mean(fold_q_priors_raw, axis=0)
        q_priors_norm = np.mean(fold_q_priors_norm, axis=0)
        q_fa_raw = np.mean(fold_q_fa_raw, axis=0)
        q_fa_norm = np.mean(fold_q_fa_norm, axis=0)

    # Dedicated query GP fold-disagreement channel (Finding 1: separate from gp_sd + missing flag on C)
    q_fold_std_norm = np.std(fold_q_priors_norm[:, :, 1], axis=0)

    c_all = np.hstack([oof_fa_norm, oof_priors_norm]) if m_fa > 0 else oof_priors_norm
    q_all = np.hstack([q_fa_norm, q_priors_norm]) if m_fa > 0 else q_priors_norm
    oof_all_raw = np.hstack([oof_fa_cf_raw, oof_priors_raw]) if m_fa > 0 else oof_priors_raw
    q_all_raw = np.hstack([q_fa_raw, q_priors_raw]) if m_fa > 0 else q_priors_raw
    all_names = tuple(fa_names + list(PRIOR_HINT_NAMES))

    # Build skill & uncertainty reliability channels with strict H3 row-type parity
    skill_vals: List[float] = []
    rel_names: List[str] = []
    skill_col_map: List[Optional[int]] = []

    for r_idx in range(m_fa):
        skill_col_map.append(len(skill_vals))
        skill_vals.append(compute_l350_skill(y_c, oof_fa_cf_raw[:, r_idx]))
        rel_names.append(f"rel_skill_{fa_names[r_idx]}")

    for p_idx, p_name in enumerate(PRIOR_HINT_NAMES):
        if p_idx == PRIOR_GPSD_INDEX:
            skill_col_map.append(None)
        else:
            skill_col_map.append(len(skill_vals))
            skill_vals.append(compute_l350_skill(y_c, oof_priors_raw[:, p_idx]))
            rel_names.append(f"rel_skill_{p_name}")

    gp_sd_hint_col = m_fa + PRIOR_GPSD_INDEX
    gp_sd_med_val = float(np.median(c_all[:, gp_sd_hint_col]))
    gp_sd_median_rel_col = len(skill_vals)
    rel_vec = np.array(skill_vals + [gp_sd_med_val], dtype=np.float64)
    rel_names.append("rel_median_gp_sd")

    gp_sd_row_rel_col = gp_sd_median_rel_col + 1
    rel_names.extend(["rel_row_gp_sd", "rel_row_gp_fold_disagreement", "rel_row_gp_fold_disagreement_missing"])

    c_rel = np.column_stack(
        [
            np.tile(rel_vec, (n_c, 1)),
            c_all[:, gp_sd_hint_col],
            np.zeros(n_c, dtype=np.float64),
            np.ones(n_c, dtype=np.float64),
        ]
    )
    q_rel = np.column_stack(
        [
            np.tile(rel_vec, (n_q, 1)),
            q_all[:, gp_sd_hint_col],
            q_fold_std_norm,
            np.zeros(n_q, dtype=np.float64),
        ]
    )

    y_med_all, y_scale_all = _robust_loc_scale(y_c)
    y_c_norm = (y_c - y_med_all) / y_scale_all

    # H7 Identity: hash compiler module source bytes + every compiler parameter; record k_eff separately
    proto_spec = json.dumps(
        {
            "ban_yeo_johnson": True,
            "compiler_source_sha256": _COMPILER_SOURCE_SHA256,
            "k_inner_param": int(k_inner),
            "max_formulas": int(max_formulas),
            "prior_names": list(PRIOR_HINT_NAMES),
            "query_fold_mode": str(query_fold_mode),
            "schema": "jope.h462.pcr_hints/v2_hnc1",
            "seed": int(seed),
            "warps": ["identity", "rank_gauss"],
        },
        sort_keys=True,
    ).encode("utf-8")
    proto_id = "pcr_v2_" + hashlib.sha256(proto_spec).hexdigest()

    return PCRHintBundle(
        context_hints=c_all,
        query_hints=q_all,
        context_reliability=c_rel,
        query_reliability=q_rel,
        hint_names=all_names,
        reliability_names=tuple(rel_names),
        hint_protocol_id=proto_id,
        k_eff=int(k_eff),
        query_fold_index=int(k_q),
        selected_formula_channels=tuple(range(m_fa)),
        non_selected_channels=tuple(range(m_fa, m_fa + n_priors)),
        skill_col_for_hint=tuple(skill_col_map),
        gp_sd_hint_col=int(gp_sd_hint_col),
        gp_sd_median_rel_col=int(gp_sd_median_rel_col),
        gp_sd_row_rel_col=int(gp_sd_row_rel_col),
        context_target_norm=y_c_norm,
        oof_predictions_raw=oof_all_raw,
        query_predictions_raw=q_all_raw,
    )


def evaluate_h8_episode_skills(bundle: PCRHintBundle, y_query: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """Return `{hint_name: (ctx_skill, qry_skill)}` using the identical `L350` formula on every location channel."""
    yq = np.asarray(y_query, dtype=np.float64).reshape(-1)
    out: Dict[str, Tuple[float, float]] = {}
    for j, h_name in enumerate(bundle.hint_names):
        s_col = bundle.skill_col_for_hint[j]
        if s_col is None:
            continue
        ctx_sk = float(bundle.context_reliability[0, s_col])
        qry_sk = compute_l350_skill(yq, bundle.query_predictions_raw[:, j])
        out[h_name] = (ctx_sk, qry_sk)
    return out


def corrupt_pcr_hints_counterfactual(
    bundle: PCRHintBundle,
    corrupt_prob: float = 0.20,
    seed: int = 462299,
) -> Tuple[PCRHintBundle, np.ndarray]:
    """Corrupt a subset of hint channels and recompute ONLY their reliability by the deploy `L350` formula (`H4`)."""
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

        # H4 Finding 3: recompute ONLY the corrupted channel's deploy reliability; leave uncorrupted untouched!
        s_col = bundle.skill_col_for_hint[int(j)]
        if s_col is not None:
            new_skill = compute_l350_skill(bundle.context_target_norm, c_h[:, int(j)])
            c_r[:, s_col] = new_skill
            q_r[:, s_col] = new_skill
        elif int(j) == bundle.gp_sd_hint_col:
            c_h[:, j] = np.abs(c_h[:, j])
            q_h[:, j] = np.abs(q_h[:, j])
            new_med = float(np.median(c_h[:, j]))
            c_r[:, bundle.gp_sd_median_rel_col] = new_med
            q_r[:, bundle.gp_sd_median_rel_col] = new_med
            c_r[:, bundle.gp_sd_row_rel_col] = c_h[:, j]
            q_r[:, bundle.gp_sd_row_rel_col] = q_h[:, j]

    return (
        dataclasses.replace(
            bundle,
            context_hints=c_h,
            query_hints=q_h,
            context_reliability=c_r,
            query_reliability=q_r,
        ),
        mask,
    )
