"""Mathematical Building-Block Landmarks and Inter-Landmark Summary Statistics.

Mines first-principles scalar primitives (asinh, atan, erf, cbrt, expm1, sin/cos,
fract, hypot, atan2, step, hinge, clamp) and row-level summary statistics (row_mean,
row_std, row_l1, row_l2, row_min, row_max, row_skew, row_missing_frac) strictly on
D_train to compute:
1. 16 fast mathematical building-block landmark probes (per-table & per-column).
2. 48 inter-landmark summary statistics (5 functional family moments, 12 cross-landmark
   ratios/synergies, 8 precision-weighted gain/SE confidence stats, and 8 2D column x
   landmark alignment correlations) to maximize Deep Learning conditioning lift.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple
import numpy as np

LEVEL_RANK = {"low": 0, "small": 0, "medium": 1, "high": 2, "ultra": 3}


def _squash(x: np.ndarray | float) -> np.ndarray | float:
    if isinstance(x, (int, float, np.floating)):
        v = float(x)
        return 0.0 if not math.isfinite(v) else v / (1.0 + abs(v))
    arr = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return arr / (1.0 + np.abs(arr))


def _erf_vec(x: np.ndarray) -> np.ndarray:
    sign, ax = np.sign(x), np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return sign * (1.0 - poly * np.exp(-(ax ** 2)))


def _gini(values: np.ndarray) -> float:
    arr = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1 or float(np.sum(arr)) <= 1e-12:
        return 0.0
    s = np.sort(arr)
    n = s.size
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float(np.clip((2.0 * np.sum(idx * s)) / (n * float(np.sum(s))) - (n + 1.0) / n, 0.0, 1.0))


def _safe_corr_vec(A: np.ndarray, y_z: np.ndarray) -> np.ndarray:
    """Fast vectorized absolute Pearson correlation between columns of A (n, k) and standardized y_z (n,)."""
    if A.size == 0 or y_z.size == 0:
        return np.zeros(A.shape[1] if A.ndim == 2 else 1, dtype=np.float64)
    A_c = A - np.mean(A, axis=0, keepdims=True)
    A_s = np.std(A_c, axis=0)
    valid = A_s > 1e-9
    out = np.zeros(A.shape[1], dtype=np.float64)
    if np.any(valid):
        out[valid] = np.abs(np.mean((A_c[:, valid] / A_s[valid]) * y_z[:, None], axis=0))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_pair_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_f, b_f = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
    if a_f.size < 2 or b_f.size != a_f.size:
        return 0.0
    sa, sb = float(np.std(a_f)), float(np.std(b_f))
    if sa <= 1e-9 or sb <= 1e-9:
        return 0.0
    c = float(np.mean(((a_f - float(np.mean(a_f))) / sa) * ((b_f - float(np.mean(b_f))) / sb)))
    return float(np.clip(c, -1.0, 1.0)) if math.isfinite(c) else 0.0


def compute_row_summary_features(
    X: np.ndarray, missing_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute 8 invariant row-level summary statistics across feature columns (never target).

    Returns array of shape (n, 8):
    [row_mean, row_std, row_l1, row_l2, row_min, row_max, row_skew, row_missing_frac]
    """
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n, p = arr.shape
    if n == 0 or p == 0:
        return np.zeros((n, 8), dtype=np.float64)
    finite = np.isfinite(arr)
    miss_frac = (
        np.mean(np.asarray(missing_mask, dtype=np.float64), axis=1)
        if missing_mask is not None and missing_mask.shape == arr.shape
        else 1.0 - np.mean(finite, axis=1)
    )
    clean = np.where(finite, arr, 0.0)
    r_mean = np.mean(clean, axis=1)
    centered = clean - r_mean[:, None]
    r_std = np.sqrt(np.mean(centered ** 2, axis=1) + 1e-8)
    r_l1 = np.mean(np.abs(clean), axis=1)
    r_l2 = np.sqrt(np.mean(clean ** 2, axis=1) + 1e-8)
    r_min = np.min(clean, axis=1)
    r_max = np.max(clean, axis=1)
    r_skew = np.clip(np.mean((centered / r_std[:, None]) ** 3, axis=1), -10.0, 10.0)
    return np.column_stack([r_mean, r_std, r_l1, r_l2, r_min, r_max, r_skew, miss_frac])


def compute_building_block_landmarks(
    X_train_scaled: np.ndarray,
    y_train_scaled: np.ndarray,
    *,
    level: str = "ultra",
) -> Dict[str, np.ndarray]:
    """Compute 16 mathematical building-block landmarks, per-column probes, and multiclass/tail stats."""
    lvl_r = LEVEL_RANK.get(level, 3)
    X = np.asarray(X_train_scaled, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y_train_scaled, dtype=np.float64).reshape(-1)
    n, p = X.shape
    if n < 6 or p == 0:
        return {
            "bb_gains16": np.zeros(16, dtype=np.float32),
            "row_summary_corrs8": np.zeros(8, dtype=np.float32),
            "col_bb_probes7": np.zeros((max(p, 1), 7), dtype=np.float32),
            "het_and_tail8": np.zeros(8, dtype=np.float32),
            "multiclass_simplex20": np.zeros(20, dtype=np.float32),
            "building_block_vec68": np.zeros(68, dtype=np.float32),
        }

    # Standardize columns on D_train
    c_mean, c_std = np.mean(X, axis=0), np.std(X, axis=0)
    valid_c = c_std > 1e-9
    Z = np.zeros_like(X, dtype=np.float64)
    Z[:, valid_c] = np.clip((X[:, valid_c] - c_mean[valid_c]) / c_std[valid_c], -8.0, 8.0)

    y_std = float(np.std(y))
    y_z = (y - float(np.mean(y))) / y_std if y_std > 1e-9 else np.zeros(n, dtype=np.float64)

    # 1. Per-column mathematical building-block probes (p, 7):
    # [asinh, atan, erf, step_med, hinge_iqr, fract_mod, xi_rank_jump]
    c_asinh = _safe_corr_vec(np.arcsinh(Z), y_z)
    c_atan = _safe_corr_vec(np.arctan(Z), y_z)
    c_erf = _safe_corr_vec(_erf_vec(Z / math.sqrt(2.0)), y_z)
    c_cbrt = _safe_corr_vec(np.sign(Z) * (np.abs(Z) ** (1.0 / 3.0)), y_z)
    c_log1p = _safe_corr_vec(np.sign(Z) * np.log1p(np.abs(Z)), y_z)
    c_tanh = _safe_corr_vec(np.tanh(0.75 * Z), y_z) if lvl_r >= 1 else np.zeros(p)
    c_laplace = _safe_corr_vec(np.expm1(-np.abs(Z)), y_z) if lvl_r >= 1 else np.zeros(p)

    q10, q25, q50, q75, q90 = np.quantile(Z, (0.10, 0.25, 0.50, 0.75, 0.90), axis=0)
    c_clamp = _safe_corr_vec(np.clip(Z, q10[None, :], q90[None, :]), y_z) if lvl_r >= 1 else np.zeros(p)
    c_step = _safe_corr_vec((Z >= q50[None, :]).astype(np.float64), y_z) if lvl_r >= 2 else np.zeros(p)
    c_hinge = (
        _safe_corr_vec(np.maximum(0.0, Z - q75[None, :]) - np.maximum(0.0, q25[None, :] - Z), y_z)
        if lvl_r >= 2 else np.zeros(p)
    )
    c_sincos = (
        np.maximum(
            _safe_corr_vec(np.sin(0.5 * math.pi * Z), y_z),
            _safe_corr_vec(np.cos(0.5 * math.pi * Z), y_z),
        )
        if lvl_r >= 2 else np.zeros(p)
    )
    c_fract = _safe_corr_vec(Z - np.floor(Z), y_z) if lvl_r >= 2 else np.zeros(p)

    # Fast vectorized Chatterjee xi rank-jump dependence proxy at Ultra (or High)
    c_xi = np.zeros(p, dtype=np.float64)
    if lvl_r >= 2 and y_std > 1e-9:
        y_ranks = np.argsort(np.argsort(y)).astype(np.float64)
        order_x = np.argsort(Z, axis=0)
        y_sorted = y_ranks[order_x]
        diffs = np.mean(np.abs(np.diff(y_sorted, axis=0)), axis=0)
        c_xi = np.clip(1.0 - (3.0 * diffs) / float(max(n + 1, 2)), 0.0, 1.0)

    # 2. Row-level summary features & their target correlations
    row_feats = compute_row_summary_features(Z)
    row_corrs = _safe_corr_vec(row_feats, y_z)
    row_gain = float(np.max(row_corrs)) if row_corrs.size > 0 else 0.0

    # 3. Bivariate polar & normalized-ratio probes on top driver pairs (High & Ultra)
    hypot_gain, atan2_gain, norm_diff_gain = 0.0, 0.0, 0.0
    if lvl_r >= 2 and p >= 2 and y_std > 1e-9:
        base_score = np.maximum(c_asinh, np.maximum(c_step, c_xi))
        top_idx = np.argsort(base_score)[::-1][: min(p, 6)]
        h_cols, a_cols, d_cols = [], [], []
        for i_pos in range(len(top_idx)):
            for j_pos in range(i_pos + 1, len(top_idx)):
                ia, ib = int(top_idx[i_pos]), int(top_idx[j_pos])
                zi, zj = Z[:, ia], Z[:, ib]
                h_cols.append(np.hypot(zi, zj))
                a_cols.append(np.arctan2(zi, zj))
                d_cols.append((zi - zj) / (1.0 + np.abs(zi) + np.abs(zj)))
        if h_cols:
            hypot_gain = float(np.max(_safe_corr_vec(np.column_stack(h_cols), y_z)))
            atan2_gain = float(np.max(_safe_corr_vec(np.column_stack(a_cols), y_z)))
            norm_diff_gain = float(np.max(_safe_corr_vec(np.column_stack(d_cols), y_z)))

    # Assemble the 16 building-block landmark gains (respecting level visibility)
    bb_gains16 = np.array([
        float(np.max(c_asinh)), float(np.max(c_atan)), float(np.max(c_erf)),
        float(np.max(c_cbrt)), float(np.max(c_log1p)), row_gain,
        float(np.max(c_tanh)) if lvl_r >= 1 else 0.0,
        float(np.max(c_laplace)) if lvl_r >= 1 else 0.0,
        float(np.max(c_clamp)) if lvl_r >= 1 else 0.0,
        float(np.max(c_step)) if lvl_r >= 2 else 0.0,
        float(np.max(c_hinge)) if lvl_r >= 2 else 0.0,
        float(np.max(c_sincos)) if lvl_r >= 2 else 0.0,
        float(np.max(c_fract)) if lvl_r >= 2 else 0.0,
        hypot_gain if lvl_r >= 2 else 0.0,
        atan2_gain if lvl_r >= 3 else 0.0,
        norm_diff_gain if lvl_r >= 3 else 0.0,
    ], dtype=np.float64)

    # 4. Conditional heteroscedasticity & upper/lower tail co-dependence (8 dims)
    het_and_tail8 = np.zeros(8, dtype=np.float64)
    if lvl_r >= 1 and y_std > 1e-9:
        best_j = int(np.argmax(c_asinh))
        res_abs = np.abs(y_z - float(c_asinh[best_j]) * Z[:, best_j])
        res_std = float(np.std(res_abs))
        het_corrs = _safe_corr_vec(Z, (res_abs - float(np.mean(res_abs))) / max(res_std, 1e-9)) if res_std > 1e-9 else np.zeros(p)
        s_het = np.sort(het_corrs)[::-1]
        het_and_tail8[:4] = [float(s_het[0]) if s_het.size else 0.0, float(np.mean(s_het[: min(3, p)])), float(np.mean(het_corrs)), _gini(het_corrs)]
        if lvl_r >= 2:
            y_hi, y_lo = (y_z > 1.0).astype(np.float64), (y_z < -1.0).astype(np.float64)
            u_corrs = _safe_corr_vec((Z > 1.0).astype(np.float64), y_hi - float(np.mean(y_hi)))
            l_corrs = _safe_corr_vec((Z < -1.0).astype(np.float64), y_lo - float(np.mean(y_lo)))
            het_and_tail8[4:] = [float(np.max(u_corrs)), float(np.mean(u_corrs)), float(np.max(l_corrs)), float(np.mean(l_corrs))]

    # 5. Multiclass simplex, Fisher discriminant & target modality profile (20 dims)
    mc_vec20 = _compute_multiclass_and_target_simplex(Z, y, lvl_r)

    # 6. 16 contrasts among the building-block probes themselves
    c_lin = _safe_corr_vec(Z, y_z)
    lin_max = float(np.max(c_lin)) if c_lin.size > 0 else 0.0
    denom = max(float(np.max(bb_gains16)), lin_max, 0.02)
    bb_contrasts16 = np.clip(np.array([
        (bb_gains16[0] - lin_max) / denom, (bb_gains16[1] - lin_max) / denom,
        (bb_gains16[2] - lin_max) / denom, (bb_gains16[3] - lin_max) / denom,
        (bb_gains16[5] - lin_max) / denom, float(np.mean(row_corrs)) / denom, _gini(row_corrs),
        (bb_gains16[7] - bb_gains16[6]) / denom if lvl_r >= 1 else 0.0,
        (bb_gains16[8] - lin_max) / denom if lvl_r >= 1 else 0.0,
        (bb_gains16[9] - bb_gains16[2]) / denom if lvl_r >= 2 else 0.0,
        (bb_gains16[10] - lin_max) / denom if lvl_r >= 2 else 0.0,
        (bb_gains16[11] - lin_max) / denom if lvl_r >= 2 else 0.0,
        (bb_gains16[12] - lin_max) / denom if lvl_r >= 2 else 0.0,
        (bb_gains16[13] - lin_max) / denom if lvl_r >= 2 else 0.0,
        (bb_gains16[14] - bb_gains16[13]) / denom if lvl_r >= 3 else 0.0,
        (bb_gains16[15] - lin_max) / denom if lvl_r >= 3 else 0.0,
    ], dtype=np.float64), -1.0, 1.0)

    col_bb_probes7 = np.column_stack([c_asinh, c_atan, c_erf, c_step, c_hinge, c_fract, c_xi]).astype(np.float32)
    building_block_vec68 = np.concatenate([
        _squash(bb_gains16), _squash(row_corrs), bb_contrasts16, _squash(het_and_tail8), mc_vec20,
    ]).astype(np.float32)

    return {
        "bb_gains16": _squash(bb_gains16).astype(np.float32),
        "row_summary_corrs8": _squash(row_corrs).astype(np.float32),
        "col_bb_probes7": np.nan_to_num(col_bb_probes7, nan=0.0, posinf=0.0, neginf=0.0),
        "het_and_tail8": _squash(het_and_tail8).astype(np.float32),
        "multiclass_simplex20": mc_vec20.astype(np.float32),
        "building_block_vec68": np.nan_to_num(building_block_vec68, nan=0.0, posinf=0.0, neginf=0.0),
    }


def _compute_multiclass_and_target_simplex(Z: np.ndarray, y: np.ndarray, lvl_r: int) -> np.ndarray:
    """Compute 20 target simplex, Fisher discriminant, and class-centroid separation features."""
    out = np.zeros(20, dtype=np.float64)
    n, p = Z.shape
    u_vals, counts = np.unique(np.round(y, 5), return_counts=True)
    k = u_vals.size
    if k <= 1:
        return out
    is_discrete_target = 2 <= k <= min(32, max(n // 3, 2))
    out[0] = math.log2(min(k, 64)) / 6.0 if is_discrete_target else 0.0
    if is_discrete_target:
        priors = counts.astype(np.float64) / float(n)
        ent = -float(np.sum(priors * np.log(np.maximum(priors, 1e-12)))) / max(math.log(k), 1e-9)
        out[1] = float(np.clip(ent, 0.0, 1.0))
        out[2] = float(np.min(priors) / max(float(np.max(priors)), 1e-9))
        out[3] = _gini(priors)
        if lvl_r >= 1:
            # Between-class vs within-class Fisher variance ratio per feature
            centroids = np.zeros((k, p), dtype=np.float64)
            for idx_c, val_c in enumerate(u_vals):
                mask_c = np.abs(y - val_c) <= 1e-4
                if np.any(mask_c):
                    centroids[idx_c] = np.mean(Z[mask_c], axis=0)
            between_var = np.sum(priors[:, None] * (centroids ** 2), axis=0)
            within_var = np.maximum(1.0 - between_var, 1e-4)
            fisher = _squash(between_var / within_var)
            s_fish = np.sort(fisher)[::-1]
            for m in range(min(4, s_fish.size)):
                out[4 + m] = float(s_fish[m])
            out[8] = float(np.mean(fisher))
            out[9] = _gini(fisher)
            if lvl_r >= 2 and k >= 2:
                sv = np.linalg.svd(np.sqrt(priors)[:, None] * centroids, compute_uv=False)
                sv_sq = _squash(sv)
                for m in range(min(6, sv_sq.size)):
                    out[10 + m] = float(sv_sq[m])
    # Target distribution moments (always populated for regression & classification)
    y_std = max(float(np.std(y)), 1e-9)
    y_z = (y - float(np.mean(y))) / y_std
    out[16] = float(_squash(float(np.mean(y_z ** 3))))
    out[17] = float(_squash(float(np.mean(y_z ** 4) - 3.0) / 4.0))
    out[18] = float(_squash(float(np.mean(np.arcsinh(y_z) ** 2))))
    q10, q25, q75, q90 = np.quantile(y_z, (0.10, 0.25, 0.75, 0.90))
    out[19] = float(_squash((q90 - q10) / max(q75 - q25, 1e-4) / 3.0))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def compute_inter_landmark_summary_stats(
    rep_desc: Dict[str, Any],
    tab_desc: Dict[str, np.ndarray],
    bb_desc: Dict[str, np.ndarray],
    *,
    level: str = "ultra",
) -> np.ndarray:
    """Compute 48 rich inter-landmark summary features (relationships BETWEEN landmarks)."""
    lvl_r = LEVEL_RANK.get(level, 3)
    g23 = np.asarray(rep_desc["gains23"], dtype=np.float64)
    m23 = np.asarray(rep_desc["mask23"], dtype=bool)
    se23 = np.asarray(rep_desc.get("ses23", np.zeros(23)), dtype=np.float64)
    lam23 = np.asarray(rep_desc.get("lambdas23", np.zeros(23)), dtype=np.float64)
    bb16 = np.asarray(bb_desc["bb_gains16"], dtype=np.float64)

    # 1. Category A: 5 Functional Landmark Family Moments (5 families x 4 stats = 20 dims)
    families = [
        np.concatenate([g23[[0, 1, 2, 4]], bb16[[0, 1, 2, 3, 4]]]),                      # F0: Monotone/Tail
        np.concatenate([g23[[3, 5, 9]], bb16[[6, 7, 13]]]),                              # F1: Symmetric/Radial
        np.concatenate([g23[[6, 7, 15, 16, 17]], bb16[[9, 10]]]) if lvl_r >= 1 else np.zeros(1),  # F2: Piecewise/Tree
        np.concatenate([g23[[8, 12]], bb16[[11, 12, 14]]]) if lvl_r >= 2 else np.zeros(1),        # F3: Periodic/Wave
        np.concatenate([g23[[10, 11, 18, 19, 20]], bb16[[5, 15]]]) if lvl_r >= 1 else bb16[[5]],  # F4: Relational/Row
    ]
    all_active = np.concatenate([g23[m23], bb16[bb16 > 0]])
    s_max = max(float(np.max(all_active)) if all_active.size > 0 else 0.0, 0.02)

    fam_vec = np.zeros(20, dtype=np.float64)
    for f_idx, f_vals in enumerate(families):
        pos = f_vals[np.isfinite(f_vals) & (np.abs(f_vals) > 1e-9)]
        if pos.size > 0:
            f_max, f_mean, f_std = float(np.max(pos)), float(np.mean(pos)), float(np.std(pos))
            fam_vec[4 * f_idx + 0] = float(np.clip(f_max, -1.0, 1.0))
            fam_vec[4 * f_idx + 1] = float(np.clip(f_mean, -1.0, 1.0))
            fam_vec[4 * f_idx + 2] = float(np.clip(f_std / s_max, 0.0, 1.0))
            fam_vec[4 * f_idx + 3] = float(np.clip(f_max / s_max, -1.0, 1.0))

    # 2. Category B: 12 Cross-Family Ratios, Log-Odds & Uplifts BETWEEN Landmarks
    lin_g = max(float(g23[0]), float(bb16[4]) * 0.5, 0.0)
    best1_g = max(float(g23[14]), lin_g, 0.01)
    tail_best = max(float(g23[1]), float(g23[2]), float(bb16[0]), float(bb16[3]))
    sharp_best = max(float(g23[7]), float(g23[15]), float(bb16[9]))
    smooth_best = max(float(g23[4]), float(bb16[1]), float(bb16[2]), float(bb16[6]))
    pair_polar_best = max(float(g23[11]), float(bb16[13]), float(bb16[14]), float(bb16[15]))
    row_comb = max(float(g23[10]), float(bb16[5]))
    per_best = max(float(g23[8]), float(bb16[11]), float(bb16[12]))
    d_lin = float(g23[0] - g23[21]) if m23[0] and m23[21] else 0.0
    d_gbdt = float(g23[17] - g23[22]) if lvl_r >= 3 and m23[17] and m23[22] else 0.0

    # Global landmark softmax entropy
    if all_active.size > 1:
        logits = 8.0 * all_active / s_max
        logits = logits - np.max(logits)
        probs = np.exp(logits) / max(float(np.sum(np.exp(logits))), 1e-12)
        lm_entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12)))) / max(math.log(probs.size), 1e-9)
    else:
        lm_entropy = 0.0

    ratios_vec = np.array([
        np.clip((tail_best - lin_g) / s_max, -1.0, 1.0),
        np.clip((sharp_best - smooth_best) / s_max, -1.0, 1.0) if lvl_r >= 2 else 0.0,
        np.clip(float(g23[16] - g23[15]) / s_max, -1.0, 1.0) if lvl_r >= 2 else 0.0,
        np.clip(float(g23[17] - g23[16]) / s_max, -1.0, 1.0) if lvl_r >= 3 else 0.0,
        np.clip((pair_polar_best - best1_g) / s_max, -1.0, 1.0) if lvl_r >= 2 else 0.0,
        np.clip((row_comb - best1_g) / s_max, -1.0, 1.0),
        float(_squash(row_comb / (lin_g + 0.02))),
        np.clip((per_best - fam_vec[0]) / s_max, -1.0, 1.0) if lvl_r >= 2 else 0.0,
        math.tanh(math.asinh(max(float(g23[20]), 0.0) / (max(float(g23[17]), 0.0) + 0.02)) - math.asinh(1.0)) if lvl_r >= 3 else 0.0,
        np.clip(float(g23[18] - g23[19]) / s_max, -1.0, 1.0) if lvl_r >= 3 else 0.0,
        np.clip((d_gbdt - d_lin) / (max(abs(d_gbdt), abs(d_lin)) + 0.02), -1.0, 1.0) if lvl_r >= 3 else np.clip(d_lin / s_max, -1.0, 1.0),
        float(np.clip(lm_entropy, 0.0, 1.0)),
    ], dtype=np.float64)

    # 3. Category C: 8 Confidence (Gain/SE t-score) & Regularization Summary Stats
    conf_vec = np.zeros(8, dtype=np.float64)
    if np.any(m23):
        g_act, se_act, lam_act = g23[m23], np.maximum(se23[m23], 1e-3), lam23[m23]
        weights = 1.0 / (se_act ** 2 + 1e-4)
        t_stats = g_act / se_act
        conf_vec[0] = float(np.clip(np.sum(weights * g_act) / max(float(np.sum(weights)), 1e-9), -1.0, 1.0))
        conf_vec[1] = float(_squash(float(np.max(t_stats)) / 10.0))
        conf_vec[2] = float(_squash(float(np.mean(t_stats)) / 10.0))
        conf_vec[3] = float(_squash(float(np.std(t_stats)) / 10.0))
        conf_vec[4] = float(np.mean(t_stats > 2.0))
        conf_vec[5] = float(np.clip(np.mean(lam_act), 0.0, 1.0))
        conf_vec[6] = float(np.clip(np.std(lam_act), 0.0, 1.0))
        conf_vec[7] = _safe_pair_corr(g_act, lam_act)

    # 4. Category D: 8 2D Column x Landmark Spectrum Alignment Summary Stats
    align_vec = np.zeros(8, dtype=np.float64)
    col_spec = np.asarray(rep_desc.get("column_spectrum", np.zeros((1, 9))), dtype=np.float64)
    col_bb = np.asarray(bb_desc["col_bb_probes7"], dtype=np.float64)
    p = col_bb.shape[0]
    if p >= 1:
        s_lin = col_spec[:p, 0] if col_spec.shape[0] >= p else col_bb[:, 0]
        s_step = np.maximum(col_bb[:, 3], col_spec[:p, 1] if col_spec.shape[0] >= p else 0.0)
        s_xi = np.maximum(col_bb[:, 6], col_spec[:p, 7] if col_spec.shape[0] >= p else 0.0)
        s_asinh = col_bb[:, 0]
        s_nonlin = np.max(col_bb[:, 1:], axis=1)
        align_vec[0] = _safe_pair_corr(s_lin, s_step) if lvl_r >= 2 else 0.0
        align_vec[1] = _safe_pair_corr(s_lin, s_xi) if lvl_r >= 2 else 0.0
        align_vec[2] = _safe_pair_corr(s_lin, s_asinh)
        orth_diff = np.maximum(s_nonlin - s_lin, 0.0)
        s_orth = np.sort(orth_diff)[::-1]
        align_vec[3] = float(_squash(float(s_orth[0])))
        align_vec[4] = float(_squash(float(np.mean(s_orth[: min(3, p)]))))
        # Per-column primitive entropy
        mat_pos = np.maximum(col_bb, 1e-6)
        mat_prob = mat_pos / np.sum(mat_pos, axis=1, keepdims=True)
        col_ent = -np.sum(mat_prob * np.log(mat_prob), axis=1) / math.log(col_bb.shape[1])
        align_vec[5] = float(np.clip(np.mean(col_ent), 0.0, 1.0))
        align_vec[6] = float(_squash(_gini(s_nonlin) / (_gini(s_lin) + 0.05)))
        align_vec[7] = float(_squash(float(bb16[5]) / (max(float(np.max(s_nonlin)), float(np.max(s_lin)), 0.02))))

    out48 = np.concatenate([fam_vec, ratios_vec, conf_vec, align_vec]).astype(np.float32)
    return np.nan_to_num(out48, nan=0.0, posinf=0.0, neginf=0.0)


def build_v3_conditioning_bundle(
    report: Dict[str, Any],
    X_train_scaled: np.ndarray,
    y_train_scaled: np.ndarray,
    *,
    level: str = "ultra",
) -> Dict[str, Any]:
    """Assemble full 256-dim hierarchical landmark + inter-landmark summary vector & (p, 24) column matrix."""
    from .descriptors import compute_table_descriptors, extract_report_descriptors

    rep_desc = extract_report_descriptors(report, level=level)
    tab_desc = compute_table_descriptors(X_train_scaled, y_train_scaled, level=level)
    bb_desc = compute_building_block_landmarks(X_train_scaled, y_train_scaled, level=level)
    inter_vec48 = compute_inter_landmark_summary_stats(rep_desc, tab_desc, bb_desc, level=level)

    vec140 = np.concatenate([rep_desc["vector116"], tab_desc["table_vec24"]]).astype(np.float32)
    vec256 = np.concatenate([vec140, inter_vec48, bb_desc["building_block_vec68"]]).astype(np.float32)

    col_desc8 = np.asarray(tab_desc["column_desc"], dtype=np.float32)
    p = col_desc8.shape[0]
    col_spec_raw = np.asarray(rep_desc["column_spectrum"], dtype=np.float32)
    col_spec9 = np.zeros((p, 9), dtype=np.float32)
    take_p = min(p, col_spec_raw.shape[0])
    if take_p > 0:
        col_spec9[:take_p] = col_spec_raw[:take_p]
    col_bb7 = np.asarray(bb_desc["col_bb_probes7"], dtype=np.float32)[:p]
    col_landmarks24 = np.column_stack([col_spec9, col_desc8, col_bb7]).astype(np.float32)

    # Upgraded multi-primitive column driver weights (combining linear, rank, step/hinge, asinh, fract & xi)
    emp_base = np.asarray(tab_desc["empirical_driver_weights"], dtype=np.float64)[:p]
    rep_prior = np.asarray(rep_desc["column_prior_importance"], dtype=np.float64)
    rep_p = np.zeros(p, dtype=np.float64)
    if take_p > 0:
        rep_p[:take_p] = rep_prior[:take_p]
    bb_col_max = np.max(np.abs(col_bb7.astype(np.float64)), axis=1)
    if float(np.sum(bb_col_max)) > 1e-9:
        bb_col_w = bb_col_max / float(np.sum(bb_col_max))
    else:
        bb_col_w = np.full(p, 1.0 / max(p, 1), dtype=np.float64)
    v3_drivers = 0.35 * emp_base + 0.35 * rep_p + 0.30 * bb_col_w
    v3_drivers = v3_drivers / max(float(np.sum(v3_drivers)), 1e-9)

    return {
        "vec140": np.nan_to_num(vec140, nan=0.0, posinf=0.0, neginf=0.0),
        "vec256": np.nan_to_num(vec256, nan=0.0, posinf=0.0, neginf=0.0),
        "inter_landmark_vec48": inter_vec48,
        "building_block_vec68": bb_desc["building_block_vec68"],
        "col_landmarks24": np.nan_to_num(col_landmarks24, nan=0.0, posinf=0.0, neginf=0.0),
        "v3_driver_weights": v3_drivers.astype(np.float32),
        "rep_desc": rep_desc,
        "tab_desc": tab_desc,
        "bb_desc": bb_desc,
    }


def _synthesize_report_from_train(X_fit: np.ndarray, y_fit: np.ndarray, task: str) -> Dict[str, Any]:
    """Synthesize a leak-free train-only landmark report record from (X_fit, y_fit)."""
    n, p = X_fit.shape
    bb = compute_building_block_landmarks(X_fit, y_fit, level="ultra")
    bb16 = np.asarray(bb["bb_gains16"], dtype=np.float64)
    col7 = np.asarray(bb["col_bb_probes7"], dtype=np.float64)
    cells = []
    probe_map = [
        bb16[4], bb16[0], bb16[3], bb16[6], bb16[1], bb16[7],
        bb16[10], bb16[9], bb16[11], bb16[7], bb16[5], bb16[13],
        bb16[11], 0.01, max(float(bb16[0]), float(bb16[4])), bb16[9],
        max(float(bb16[9]), float(bb16[13])), max(float(bb16[9]), float(bb16[13])) * 1.05,
        bb16[13] * 0.9, bb16[13] * 0.95, max(float(bb16[0]), float(bb16[6])) * 1.05,
        bb16[4] * 0.85, max(float(bb16[9]), float(bb16[13])) * 0.85,
    ]
    for idx in range(23):
        cells.append({
            "id": idx + 1,
            "state": "complete",
            "gain": round(float(probe_map[idx]), 5),
            "se": round(1.0 / math.sqrt(max(n, 16)), 5),
            "stable": True,
            "lambda_indices": [2, 2, 3, 2, 3],
        })
    spectrum = []
    for j in range(p):
        c = col7[j] if j < col7.shape[0] else np.zeros(7)
        spectrum.append({
            "slot": j,
            "values": [
                float(c[0]), float(c[3]), float(c[4]), float(c[0]), float(c[1]),
                float(c[5]), float(c[2]), float(c[6]), float(c[3]),
            ],
        })
    u_cls = len(np.unique(np.rint(y_fit))) if task in ("classification", "multiclass") else 0
    return {
        "panel": "landmarks-native-v1",
        "kind": "real",
        "privacy": "private",
        "task": "classification" if task in ("classification", "multiclass") else "regression",
        "classes": u_cls,
        "rows": n,
        "features": p,
        "cells": cells,
        "target_stats": [float(np.mean(y_fit)), float(np.std(y_fit)), None],
        "spectrum": spectrum,
    }


