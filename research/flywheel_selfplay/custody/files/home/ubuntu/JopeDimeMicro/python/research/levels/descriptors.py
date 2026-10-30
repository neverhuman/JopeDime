"""Hierarchical 3-level landmark and data descriptive feature extraction.

Partitions pre-calculated S3 FIT-CV landmark features and rich data descriptors into:
- Medium (Macro tier): 9 macro additive/monotone/dominance/missingness landmarks + coarse table descriptors
- High (Larger tier): +9 piecewise/interaction/kernel/shallow-tree landmarks (18 total) + spectral/collinearity descriptors
- Ultra (Fine tier): +5 local/GBDT/neural/learning-curve landmarks (23 total) + full per-column Chatterjee xi & functional spectra
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

LANDMARK_NAMES: Tuple[str, ...] = (
    "lin",           # 0 (id 1)  - Macro
    "log",           # 1 (id 2)  - Macro
    "sqrt",          # 2 (id 3)  - Macro
    "pow",           # 3 (id 4)  - Macro
    "sat",           # 4 (id 5)  - Macro
    "abs",           # 5 (id 6)  - Macro
    "hinge",         # 6 (id 7)  - Larger
    "step",          # 7 (id 8)  - Larger
    "periodic",      # 8 (id 9)  - Larger
    "bump",          # 9 (id 10) - Larger
    "rows",          # 10 (id 11) - Larger
    "pairs",         # 11 (id 12) - Larger
    "rff",           # 12 (id 13) - Larger
    "missing",       # 13 (id 14) - Macro
    "best1",         # 14 (id 15) - Macro
    "stump",         # 15 (id 16) - Larger
    "otree3",        # 16 (id 17) - Larger
    "gbdt",          # 17 (id 18) - Fine
    "knn1",          # 18 (id 19) - Fine
    "knn16",         # 19 (id 20) - Fine
    "net",           # 20 (id 21) - Fine
    "lin_quarter",   # 21 (id 22) - Macro
    "gbdt_quarter",  # 22 (id 23) - Fine
)

MACRO_INDICES: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 13, 14, 21)
LARGER_ADDED_INDICES: Tuple[int, ...] = (6, 7, 8, 9, 10, 11, 12, 15, 16)
FINE_ADDED_INDICES: Tuple[int, ...] = (17, 18, 19, 20, 22)

LEVEL_LANDMARK_INDICES: Dict[str, Tuple[int, ...]] = {
    "low": MACRO_INDICES[:6],
    "small": MACRO_INDICES[:6],
    "medium": MACRO_INDICES,
    "high": tuple(sorted(MACRO_INDICES + LARGER_ADDED_INDICES)),
    "ultra": tuple(range(23)),
}


def squash(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return arr / (1.0 + np.abs(arr))


def gini_coefficient(values: np.ndarray) -> float:
    arr = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    total = float(np.sum(arr))
    if total <= 1e-12:
        return 0.0
    sorted_arr = np.sort(arr)
    n = sorted_arr.size
    idx = np.arange(1, n + 1, dtype=np.float64)
    gini = float((2.0 * np.sum(idx * sorted_arr)) / (n * total) - (n + 1.0) / n)
    return float(np.clip(gini, 0.0, 1.0))


def level_landmark_mask(level: str) -> np.ndarray:
    if level not in LEVEL_LANDMARK_INDICES:
        raise ValueError(f"unknown_level_{level}")
    mask = np.zeros(23, dtype=bool)
    for idx in LEVEL_LANDMARK_INDICES[level]:
        mask[idx] = True
    return mask


def compute_level_contrasts(gains: np.ndarray, mask: np.ndarray, level: str = "ultra") -> np.ndarray:
    """Compute 16 structural contrasts strictly from landmarks admitted at `level`."""
    allowed = level_landmark_mask(level) & np.asarray(mask, dtype=bool)
    g = np.where(allowed, np.asarray(gains, dtype=np.float64), np.nan)

    def _get(idx: int) -> float:
        return float(g[idx]) if allowed[idx] and math.isfinite(g[idx]) else float("nan")

    def _max(idxs: Sequence[int]) -> float:
        vals = [float(g[i]) for i in idxs if allowed[i] and math.isfinite(g[i])]
        return max(vals) if vals else float("nan")

    def _sub(a: float, b: float, denom: float) -> float:
        if math.isfinite(a) and math.isfinite(b) and denom >= 0.01:
            return float(np.clip((a - b) / denom, -1.0, 1.0))
        return 0.0

    s = _max(range(23))
    if not math.isfinite(s) or s < 0.01:
        return np.zeros(16, dtype=np.float64)

    lin = _get(0)
    add_macro = _max((0, 1, 2, 3, 4, 5, 14))
    add_all = _max((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14))
    mono_excess = _sub(_max((1, 2, 4)), lin, s)
    sym_excess = _sub(_max((3, 5, 9)), lin, s)
    best1_share = float(np.clip(_get(14) / s, -1.0, 1.0)) if math.isfinite(_get(14)) else 0.0
    miss_share = float(np.clip(_get(13) / s, -1.0, 1.0)) if math.isfinite(_get(13)) else 0.0
    lin_lc = _sub(lin, _get(21), s)

    # Larger contrasts (zeroed automatically if level=="medium")
    hinge_step = _sub(_get(7), _get(6), s)
    periodic_ex = _sub(_get(8), _max((0, 4)), s)
    pairs_ex = _sub(_get(11), lin, s)
    rff_ex = _sub(_get(12), add_all, s)
    otree_ex = _sub(_max((15, 16)), add_all, s)

    # Fine contrasts (zeroed automatically if level in ("medium", "high"))
    nonadd_ex = _sub(_max((17, 12, 20)), add_all, s)
    knn_ratio = 0.0
    if math.isfinite(_get(18)) and math.isfinite(_get(19)) and abs(_get(19)) > 1e-6:
        knn_ratio = float(np.clip(_get(18) / max(_get(19), 1e-6), -1.0, 1.0))
    gbdt_lc = _sub(_get(17), _get(22), s)
    net_vs_gbdt = _sub(_get(20), _max((17, 12)), s)

    out = np.array([
        float(np.clip(s, -1.0, 1.0)),
        float(np.clip(lin / s, -1.0, 1.0)) if math.isfinite(lin) else 0.0,
        float(np.clip(add_macro / s, -1.0, 1.0)) if math.isfinite(add_macro) else 0.0,
        mono_excess,
        sym_excess,
        best1_share,
        miss_share,
        lin_lc,
        hinge_step,
        periodic_ex,
        pairs_ex,
        rff_ex,
        otree_ex,
        nonadd_ex,
        knn_ratio,
        gbdt_lc + 0.5 * net_vs_gbdt,
    ], dtype=np.float64)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def extract_report_descriptors(
    report: Dict[str, Any],
    *,
    level: str = "ultra",
) -> Dict[str, Any]:
    """Extract level-filtered landmark & data descriptive features from S3 report.json."""
    lvl_mask = level_landmark_mask(level)
    cells = report.get("cells", [])

    gains = np.zeros(23, dtype=np.float64)
    ses = np.zeros(23, dtype=np.float64)
    present = np.zeros(23, dtype=bool)
    stable = np.zeros(23, dtype=bool)
    lambdas = np.zeros(23, dtype=np.float64)

    for c in cells:
        idx = int(c["id"]) - 1
        if not (0 <= idx < 23) or not lvl_mask[idx]:
            continue
        state = c.get("state", "")
        g = c.get("gain")
        if state in ("complete", "uninformative") and g is not None and math.isfinite(float(g)):
            gains[idx] = float(g)
            present[idx] = True
            se_val = c.get("se")
            if se_val is not None and math.isfinite(float(se_val)):
                ses[idx] = float(se_val)
            stable[idx] = bool(c.get("stable", False))
            l_idx = c.get("lambda_indices") or []
            if l_idx:
                lambdas[idx] = float(np.mean(l_idx)) / 8.0

    # Within-level midranks among present landmarks
    midranks = np.zeros(23, dtype=np.float64)
    valid_idxs = np.where(present)[0]
    if valid_idxs.size > 1:
        for i in valid_idxs:
            gi = gains[i]
            less = float(np.sum(gains[valid_idxs] < gi))
            eq = float(np.sum(gains[valid_idxs] == gi))
            midranks[i] = (less + 0.5 * (eq - 1.0)) / float(valid_idxs.size - 1)
    elif valid_idxs.size == 1:
        midranks[valid_idxs[0]] = 0.5

    contrasts = compute_level_contrasts(gains, present, level=level)

    # Target statistics & global geometry
    t_stats_raw = report.get("target_stats") or [None, None, None]
    t_stats = np.array([
        float(t_stats_raw[0]) if t_stats_raw[0] is not None and math.isfinite(float(t_stats_raw[0])) else 0.0,
        float(t_stats_raw[1]) if t_stats_raw[1] is not None and math.isfinite(float(t_stats_raw[1])) else 0.0,
    ], dtype=np.float64)
    t_stats = squash(t_stats)

    rows = max(int(report.get("rows", 1)), 1)
    feats = max(int(report.get("features", 1)), 1)
    classes = int(report.get("classes", 0))
    is_clf = 1.0 if classes >= 2 or report.get("task") == "classification" else 0.0
    shape_vec = np.array([
        is_clf,
        math.log10(rows) / 6.0,
        math.log2(feats) / 12.0,
        math.log10(feats / float(rows)) / 4.0,
    ], dtype=np.float64)

    # Per-feature spectrum (9 slots per feature: lin, step, hinge, log, sat, periodic, abs, xi, stump)
    spec_list = report.get("spectrum") or []
    spec_matrix = np.zeros((feats, 9), dtype=np.float64)
    spec_mask = np.zeros((feats, 9), dtype=bool)
    for item in spec_list:
        slot = int(item.get("slot", -1))
        if 0 <= slot < feats:
            vals = item.get("values") or [None] * 9
            for k in range(min(len(vals), 9)):
                # Respect level visibility:
                # Small/Low/Medium sees lin(0), log(3), sat(4), abs(6)
                # High adds step(1), hinge(2), periodic(5), stump(8)
                # Ultra adds Chatterjee xi(7)
                if level in ("low", "small", "medium") and k in (1, 2, 5, 7, 8):
                    continue
                if level == "high" and k == 7:
                    continue
                v = vals[k]
                if v is not None and math.isfinite(float(v)):
                    spec_matrix[slot, k] = float(v)
                    spec_mask[slot, k] = True

    spec_summary = _summarize_feature_spectrum(spec_matrix, spec_mask)
    squashed_gains = squash(gains)
    squashed_ses = squash(ses)

    # Assemble flat vector of length 116:
    # 23 squashed gains + 23 present + 23 midranks + 16 contrasts + 2 target_stats + 4 shape + 16 spec_summary + 9 group summaries = 116
    macro_gain_max = float(np.max(squashed_gains[list(MACRO_INDICES)])) if np.any(present[list(MACRO_INDICES)]) else 0.0
    larger_gain_max = (
        float(np.max(squashed_gains[list(LARGER_ADDED_INDICES)]))
        if level in ("high", "ultra") and np.any(present[list(LARGER_ADDED_INDICES)])
        else 0.0
    )
    fine_gain_max = (
        float(np.max(squashed_gains[list(FINE_ADDED_INDICES)]))
        if level == "ultra" and np.any(present[list(FINE_ADDED_INDICES)])
        else 0.0
    )
    group_summary = np.array([
        macro_gain_max,
        larger_gain_max,
        fine_gain_max,
        float(np.mean(squashed_gains[list(MACRO_INDICES)])),
        float(np.mean(squashed_gains[list(LARGER_ADDED_INDICES)])) if level in ("high", "ultra") else 0.0,
        float(np.mean(squashed_gains[list(FINE_ADDED_INDICES)])) if level == "ultra" else 0.0,
        float(np.mean(squashed_ses[present])) if np.any(present) else 0.0,
        float(np.mean(stable[present])) if np.any(present) else 0.0,
        float(np.mean(lambdas[present])) if np.any(present) else 0.0,
    ], dtype=np.float64)

    vector = np.concatenate([
        squashed_gains,               # 23
        present.astype(np.float64),   # 23
        midranks,                     # 23
        contrasts,                    # 16
        t_stats,                      # 2
        shape_vec,                    # 4
        spec_summary,                 # 16
        group_summary,                # 9
    ])  # Total = 116 coordinates

    # Prior column importance weights from spectrum
    col_prior = np.max(np.abs(spec_matrix), axis=1)
    if np.sum(col_prior) > 1e-9:
        col_prior = col_prior / np.sum(col_prior)
    else:
        col_prior = np.full(feats, 1.0 / float(feats), dtype=np.float64)

    return {
        "level": level,
        "gains23": squashed_gains,
        "ses23": squashed_ses,
        "lambdas23": lambdas,
        "mask23": present,
        "midranks23": midranks,
        "contrasts16": contrasts,
        "target_stats2": t_stats,
        "shape4": shape_vec,
        "spectrum_summary16": spec_summary,
        "group_summary9": group_summary,
        "vector116": vector.astype(np.float32),
        "column_spectrum": squash(spec_matrix).astype(np.float32),
        "column_prior_importance": col_prior.astype(np.float32),
    }


def _summarize_feature_spectrum(spec_matrix: np.ndarray, spec_mask: np.ndarray) -> np.ndarray:
    """Summarize per-column (p, 9) univariate functional & Chatterjee xi spectrum into 16 table scalars."""
    if spec_matrix.size == 0 or not np.any(spec_mask):
        return np.zeros(16, dtype=np.float64)
    s_mat = squash(spec_matrix)
    lin_col = np.maximum(s_mat[:, 0], 0.0)
    nonlin_macro = np.max(np.maximum(s_mat[:, [3, 4, 6]], 0.0), axis=1)
    nonlin_larger = np.max(np.maximum(s_mat[:, [1, 2, 5, 8]], 0.0), axis=1)
    xi_col = np.maximum(s_mat[:, 7], 0.0)
    best_per_col = np.max(np.maximum(s_mat, 0.0), axis=1)

    sorted_best = np.sort(best_per_col)[::-1]
    top1 = float(sorted_best[0]) if sorted_best.size >= 1 else 0.0
    top3 = float(np.mean(sorted_best[:3])) if sorted_best.size >= 1 else 0.0

    return np.array([
        float(np.max(lin_col)),
        float(np.mean(lin_col)),
        gini_coefficient(lin_col),
        float(np.max(nonlin_macro)),
        float(np.mean(nonlin_macro)),
        float(np.max(nonlin_larger)),
        float(np.mean(nonlin_larger)),
        float(np.max(xi_col)),
        float(np.mean(xi_col)),
        gini_coefficient(xi_col),
        top1,
        top3,
        gini_coefficient(best_per_col),
        float(np.max(nonlin_macro - lin_col)),
        float(np.max(nonlin_larger - lin_col)),
        float(np.max(xi_col - lin_col)),
    ], dtype=np.float64)


def compute_table_descriptors(
    X_train_scaled: np.ndarray,
    y_train_scaled: np.ndarray,
    *,
    level: str = "ultra",
) -> Dict[str, np.ndarray]:
    """Compute fast train-only data descriptive features on scaled D_train in <3ms.

    Returns:
    - table_vec (24 floats): covariance eigenspectrum, tail/kurtosis profile, target attribution Gini
    - column_desc (p, 8 floats): per-column distributional and target-correlation descriptors
    - empirical_driver_weights (p floats): normalized empirical feature relevance on D_train
    """
    X = np.asarray(X_train_scaled, dtype=np.float64)
    y = np.asarray(y_train_scaled, dtype=np.float64).reshape(-1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if n < 4 or p == 0:
        return {
            "table_vec24": np.zeros(24, dtype=np.float32),
            "column_desc": np.zeros((max(p, 1), 8), dtype=np.float32),
            "empirical_driver_weights": np.full(max(p, 1), 1.0 / max(p, 1), dtype=np.float32),
        }

    # Standardize columns safely for correlation / spectral analysis
    col_mean = np.mean(X, axis=0)
    col_std = np.std(X, axis=0)
    valid_col = col_std > 1e-9
    Z = np.zeros_like(X, dtype=np.float64)
    Z[:, valid_col] = (X[:, valid_col] - col_mean[valid_col]) / col_std[valid_col]

    # 1. Per-column distributional & target correlation descriptors
    skew = np.mean(Z ** 3, axis=0)
    kurt = np.clip(np.mean(Z ** 4, axis=0) - 3.0, -3.0, 50.0)
    q05, q25, q75, q95 = np.quantile(Z, (0.05, 0.25, 0.75, 0.95), axis=0)
    iqr = np.maximum(q75 - q25, 1e-6)
    tail_ratio = np.clip((q95 - q05) / iqr, 0.0, 20.0) / 10.0

    # Sparsity / zero-or-mode concentration proxy
    med = np.median(Z, axis=0)
    mode_conc = np.mean(np.abs(Z - med) < 0.05, axis=0)

    # Target Pearson & Spearman rank correlation on D_train
    y_std = float(np.std(y))
    if y_std > 1e-9:
        y_z = (y - float(np.mean(y))) / y_std
        pearson_r = np.abs(np.mean(Z * y_z[:, None], axis=0))
        y_rank = np.argsort(np.argsort(y)).astype(np.float64)
        y_rank = (y_rank - np.mean(y_rank)) / max(float(np.std(y_rank)), 1e-9)
        X_ranks = np.argsort(np.argsort(Z, axis=0), axis=0).astype(np.float64)
        X_ranks = (X_ranks - np.mean(X_ranks, axis=0)) / np.maximum(np.std(X_ranks, axis=0), 1e-9)
        spearman_r = np.abs(np.mean(X_ranks * y_rank[:, None], axis=0))
    else:
        pearson_r = np.zeros(p, dtype=np.float64)
        spearman_r = np.zeros(p, dtype=np.float64)

    # 2. Covariance / Correlation matrix & eigenspectrum (for High and Ultra; coarse only for Medium)
    if p > 1:
        corr = (Z.T @ Z) / float(n)
        np.fill_diagonal(corr, 1.0)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        off_diag = np.abs(corr - np.eye(p))
        col_max_corr = np.max(off_diag, axis=1)
        col_mean_corr = np.sum(off_diag, axis=1) / float(p - 1)
        eigvals = np.linalg.eigvalsh(corr)
        eigvals = np.maximum(eigvals[::-1], 0.0)
        eig_sum = max(float(np.sum(eigvals)), 1e-9)
        eig_shares = eigvals / eig_sum
        eff_rank = float((eig_sum ** 2) / max(float(np.sum(eigvals ** 2)), 1e-9)) / float(p)
        spec_entropy = float(
            -np.sum(eig_shares[eig_shares > 1e-12] * np.log(eig_shares[eig_shares > 1e-12]))
            / max(math.log(p), 1e-9)
        )
        top1_eig = float(eig_shares[0])
        top3_eig = float(np.sum(eig_shares[: min(3, p)]))
        top5_eig = float(np.sum(eig_shares[: min(5, p)]))
    else:
        col_max_corr = np.zeros(1, dtype=np.float64)
        col_mean_corr = np.zeros(1, dtype=np.float64)
        eff_rank = 1.0
        spec_entropy = 1.0
        top1_eig = 1.0
        top3_eig = 1.0
        top5_eig = 1.0

    # Empirical driver weights combining linear, rank, and non-linear quadratic correlation on D_train
    if y_std > 1e-9 and level == "ultra":
        Z2 = Z ** 2 - 1.0
        Z2_std = np.maximum(np.std(Z2, axis=0), 1e-6)
        quad_r = np.abs(np.mean((Z2 / Z2_std) * y_z[:, None], axis=0))
    else:
        quad_r = np.zeros(p, dtype=np.float64)

    raw_driver = 0.45 * pearson_r + 0.35 * spearman_r + 0.20 * quad_r
    if float(np.sum(raw_driver)) > 1e-9:
        driver_weights = raw_driver / float(np.sum(raw_driver))
    else:
        driver_weights = np.full(p, 1.0 / float(p), dtype=np.float64)

    sorted_r = np.sort(pearson_r)[::-1]
    sorted_sp = np.sort(spearman_r)[::-1]

    # Mask fine/larger descriptors appropriately by level
    include_larger = 1.0 if level in ("high", "ultra") else 0.0
    include_fine = 1.0 if level == "ultra" else 0.0

    table_vec = np.array([
        # Macro (always active: 8 features)
        eff_rank,
        spec_entropy,
        float(np.mean(np.abs(skew))) / (1.0 + float(np.mean(np.abs(skew)))),
        float(np.mean(mode_conc)),
        float(sorted_r[0]) if sorted_r.size >= 1 else 0.0,
        float(np.mean(pearson_r)),
        gini_coefficient(pearson_r),
        math.log10(max(n, 1)) / 6.0,
        # Larger (active at High & Ultra: 8 features)
        include_larger * top1_eig,
        include_larger * top3_eig,
        include_larger * top5_eig,
        include_larger * float(np.max(col_max_corr)),
        include_larger * float(np.mean(col_mean_corr)),
        include_larger * float(np.mean(tail_ratio)),
        include_larger * (float(sorted_sp[0]) if sorted_sp.size >= 1 else 0.0),
        include_larger * float(np.max(spearman_r - pearson_r)),
        # Fine (active at Ultra: 8 features)
        include_fine * float(np.max(np.abs(skew)) / (1.0 + np.max(np.abs(skew)))),
        include_fine * float(np.mean(np.maximum(kurt, 0.0)) / (1.0 + np.mean(np.maximum(kurt, 0.0)))),
        include_fine * float(np.max(tail_ratio)),
        include_fine * float(np.mean( sorted_r[: min(3, p)] )),
        include_fine * float(np.mean( sorted_sp[: min(3, p)] )),
        include_fine * float(np.max(quad_r)),
        include_fine * gini_coefficient(raw_driver),
        include_fine * float(np.max(raw_driver)),
    ], dtype=np.float32)

    col_desc = np.column_stack([
        squash(skew),
        squash(kurt) * include_larger,
        tail_ratio * include_larger,
        mode_conc,
        col_max_corr * include_larger,
        pearson_r,
        spearman_r * include_larger,
        quad_r * include_fine,
    ]).astype(np.float32)

    return {
        "table_vec24": np.nan_to_num(table_vec, nan=0.0, posinf=0.0, neginf=0.0),
        "column_desc": np.nan_to_num(col_desc, nan=0.0, posinf=0.0, neginf=0.0),
        "empirical_driver_weights": np.nan_to_num(driver_weights.astype(np.float32), nan=1.0 / p),
    }
