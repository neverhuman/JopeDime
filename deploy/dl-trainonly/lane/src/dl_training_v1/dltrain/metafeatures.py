"""PLAN §5 P6 / §6.4 arm A0: the train-only context meta-features m, version ``jope.dl-a0-meta/v1``.

The P6 spec calls this producer ``dltrain/a0.py``; in this lane it is ``dltrain/metafeatures.py`` (module name in
``MODULE``). The API is the spec's: VERSION, FEATURES, FEATURE_LIST_SHA256, A0Refused, A0Error, features, to_f32le,
from_raw and source_tree_sha256.

One A0 vector holds 88 float32 little-endian values ('<f4', 352 bytes) in the fixed order of ``FEATURES``. It is
computed in float64 from the outer-train rows only:
- ``from_raw`` passes the raw JTT1 bytes to exactly two functions, ``jtt1.train_view`` (which decodes only the train
  range) and ``jtt1.train_rows_sha256``, and discards the header that train_view returns;
- ``features`` receives (X, y, inner_labels) and nothing else.

Invariance. The CSVs that JTT1 is built from are min-max scaled over ALL rows (train and test), so absolute location
and scale in JTT1 units encode outer-test extremes. Every feature is therefore invariant, up to float rounding, to a
positive affine map of each X column and of y; robust scale enters only as ratios (spec §2 deviation from the §6.4
wording). Statistics are evaluated in unit coordinates: each active column, and a non-degenerate y, is mapped onto
[0, 1] by an exact power-of-two prescale followed by its own min and range. That equals the spec's formulas
mathematically (every feature is affine-invariant) and keeps inputs at 1e+-300 free of overflow and underflow.
Counts and equalities (distinct values, modes, minima, duplicates, ranks) use the raw values. The 1-NN landmarker
(feature 84) treats squared distances within 1e-6 relative (+1e-9 absolute) of the minimum as ties and keeps the
spec's lowest-position rule inside that band: on discrete data, rounding otherwise splits exact distance ties
differently in each coordinate system (KNN_TIE_RTOL documents the measurement).

Determinism. There is no BLAS matmul: Gram matrices and projections use einsum(optimize=False), the power-iteration
matvec is an elementwise product and a row sum, and LAPACK is used only for the k <= 8 ridge solve. The bytes are
reproducible under one runtime identity and do not depend on the OpenMP/OpenBLAS thread count.

Refusals: A0Refused('too_few_rows') when n < 8 and A0Refused('nonfinite_target') when any y is non-finite. Nothing
is imputed. A0Error marks a bug (a non-finite or out-of-bound feature, or an unexpected floating-point exception)
and means STOP.
"""
import ast
import functools
import math
from pathlib import Path

import numpy as np

from dltrain import common, jtt1, labels

VERSION = "jope.dl-a0-meta/v1"
MODULE = "dltrain.metafeatures"
FEATURE_LIST_SCHEMA = "jope.dl-a0-feature-list/v1"
DTYPE = "<f4"
MIN_ROWS = 8
FEATURE_ABS_BOUND = 1e4
LOWCARD_MAX = 16
LANDMARK_K = 8
RIDGE = 1e-6
KNN_CHUNK = 256
# 1-NN ties (feature 84): fold-train rows whose squared distance is within KNN_TIE_RTOL * dmin + KNN_TIE_ATOL of the
# row minimum are tied, and ties go to the lowest position. An exact tie in exact arithmetic is split by float rounding
# once the columns are mapped to unit coordinates and standardized; measured on real JTT1 ctx 6ebcb49c66f39192 under
# the verify affine maps, rounding moved squared distances by <= 2.6e-12 relative (4.3e-13 absolute), while genuine
# near-minimum gaps started at 1e-9 relative. A plain argmin flipped 27 neighbours over 16 maps (|delta| 1.9e-3 in
# feature 84); with this band the flips are 0. The band is ~3e5 x the measured rounding, so a genuine gap sitting
# within rounding of its edge (the only remaining way to flip) is rare; candidates closer than one part per million
# count as equidistant.
KNN_TIE_RTOL = 1e-6
KNN_TIE_ATOL = 1e-9
POWER_ITERATIONS = 64
DECILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
Y_QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 0.99)

FEATURES = (
    # S: size (0-3)
    "s_log_n_train", "s_log_p", "s_p_over_n", "s_inv_sqrt_n",
    # M: missingness and data quality (4-14)
    "m_x_nonfinite_cell_frac", "m_x_invalid_col_frac", "m_x_const_col_frac", "m_x_binary_col_frac",
    "m_x_lowcard_col_frac", "m_x_mode_share_mean", "m_x_mode_share_max", "m_x_dominant_col_frac",
    "m_x_at_min_frac_mean", "m_x_dup_row_frac", "m_xy_dup_conflict_frac",
    # Y: target, robust-unit and shape statistics (15-39)
    "y_robust_over_std", "y_mean_med_rz", "y_skew_slog", "y_kurt_ex_slog",
    "y_rq_p000", "y_rq_p001", "y_rq_p005", "y_rq_p010", "y_rq_p025",
    "y_rq_p075", "y_rq_p090", "y_rq_p095", "y_rq_p099", "y_rq_p100",
    "y_log1p_range_rz", "y_tail3_frac", "y_tail5_frac", "y_unique_frac", "y_mode_share",
    "y_at_min_frac", "y_at_max_frac", "y_bimodality", "y_logshift_skew_slog", "y_lag1_autocorr",
    "y_iqr_over_range",
    # X: feature marginals over the active columns (40-53)
    "x_active_col_frac", "x_abs_skew_q50", "x_abs_skew_q90", "x_kurt_ex_q50", "x_kurt_ex_q90",
    "x_uniq_frac_q10", "x_uniq_frac_q50", "x_uniq_frac_q90", "x_out5_frac_mean", "x_out5_frac_max",
    "x_log1p_range_rz_q50", "x_log1p_range_rz_q90", "x_iqr_over_range_q10", "x_iqr_over_range_q50",
    # R: feature-target relationships (54-73)
    "r_abs_max", "r_abs_q90", "r_abs_q50", "r_abs_mean", "r_top3_abs_mean", "r_sq_sum_log1p",
    "r_gt01_frac", "r_gt03_frac", "r_pos_frac", "rho_abs_max", "rho_abs_q90", "rho_abs_q50",
    "rho_abs_mean", "rho_minus_r_max", "rho_minus_r_mean", "eta2_max", "eta2_q50", "eta2_minus_r2_max",
    "eta2_minus_r2_mean", "r_best_uniq_frac",
    # C: redundancy (74-81)
    "c_abs_corr_q50", "c_abs_corr_q90", "c_abs_corr_max", "c_abs_corr_mean", "c_gt09_pair_frac",
    "c_twin_col_frac", "c_pr_frac", "c_top_eig_share",
    # L: landmarkers (82-87)
    "lin_top8_r2_in", "lin_top8_r2_cv", "knn1_top8_r2_cv", "lin_resid_absfit_corr", "lin_resid_skew_slog",
    "lin_resid_kurt_ex_slog",
)
N_FEATURES = len(FEATURES)
VECTOR_BYTES = 4 * N_FEATURES
INDEX = {name: i for i, name in enumerate(FEATURES)}
FEATURE_LIST_SHA256 = common.sha256_bytes(common.canonical(
    {"schema": FEATURE_LIST_SCHEMA, "version": VERSION, "dtype": DTYPE, "features": list(FEATURES)}))
GROUPS = {"S": range(0, 4), "M": range(4, 15), "Y": range(15, 40), "X": range(40, 54), "R": range(54, 74),
          "C": range(74, 82), "L": range(82, 88)}

if N_FEATURES != 88 or len(set(FEATURES)) != N_FEATURES:
    raise RuntimeError("the A0 feature list must hold 88 unique names")


class A0Refused(ValueError):
    """The context gets no A0 vector; the manifest lists it as missing:<reason>. Nothing is imputed."""

    REASONS = ("nonfinite_target", "too_few_rows")

    def __init__(self, reason: str, detail: str = ""):
        if reason not in self.REASONS:
            raise ValueError(f"unknown A0 refusal reason {reason!r}")
        super().__init__(f"A0 refused: {reason}" + (f" ({detail})" if detail else ""))
        self.reason = reason


class A0Error(RuntimeError):
    """A bug in the A0 computation (a non-finite or out-of-bound feature, an unexpected floating-point exception,
    a broken shape contract). It means STOP, never a missing context."""


# --------------------------------------------------------------------------------------------- scalar helpers
def _slog(v: float) -> float:
    """sign(v) * ln(1 + |v|)."""
    v = float(v)
    return math.copysign(math.log1p(abs(v)), v) if v != 0.0 else 0.0


def _q(a, q) -> float:
    return float(np.quantile(a, q, method="linear"))


def _clip(v: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, float(v)))


def _unit_cols(a):
    """Map each column of a finite 2-d array (every column non-constant) onto [0, 1]: an exact power-of-two prescale
    to |values| < 1, then (b - min) / (max - min). The column minimum maps to exactly 0 and the maximum to exactly 1;
    the map is weakly order-preserving, so equal raw values stay equal."""
    amin = a.min(axis=0)
    amax = a.max(axis=0)
    _, e = np.frexp(np.maximum(np.abs(amin), np.abs(amax)))
    b = np.ldexp(a, -e[None, :])
    bmin = np.ldexp(amin, -e)
    bmax = np.ldexp(amax, -e)
    return (b - bmin[None, :]) / (bmax - bmin)[None, :]


def _unit(a):
    """The 1-d form of _unit_cols; the caller guarantees max(a) != min(a)."""
    return _unit_cols(np.asarray(a, dtype=np.float64).reshape(-1, 1))[:, 0]


def _moments(a, is_unit=False):
    """(skew, excess kurtosis) with population moments; (0, 0) for a constant array. Affine-invariant, so it is
    evaluated on the unit-coordinate copy (`is_unit` skips the copy for an array that is already in unit coordinates;
    the map is exactly idempotent there, so the result is the same)."""
    if a.max() == a.min():
        return 0.0, 0.0
    u = a if is_unit else _unit(a)
    d = u - u.mean()
    d2 = d * d
    m2 = float(np.mean(d2))
    m3 = float(np.mean(d2 * d))
    m4 = float(np.mean(d2 * d2))
    return m3 / (m2 * math.sqrt(m2)), m4 / (m2 * m2) - 3.0


def _pearson(a, b, a_unit=False, b_unit=False) -> float:
    """0 if either input is constant; else the clipped Pearson correlation (evaluated in unit coordinates; the flags
    mark inputs that already are)."""
    if a.max() == a.min() or b.max() == b.min():
        return 0.0
    u = a if a_unit else _unit(a)
    w = b if b_unit else _unit(b)
    du = u - u.mean()
    dw = w - w.mean()
    num = float(np.sum(du * dw))
    den = math.sqrt(float(np.sum(du * du)) * float(np.sum(dw * dw)))
    return _clip(num / den, -1.0, 1.0)


def _avg_rank(a):
    """1-based ranks with ties averaged (mergesort order)."""
    n = a.shape[0]
    order = np.argsort(a, kind="mergesort")
    s = a[order]
    new = np.empty(n, dtype=bool)
    new[0] = True
    np.not_equal(s[1:], s[:-1], out=new[1:])
    starts = np.flatnonzero(new)
    ends = np.append(starts[1:], n) - 1
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.repeat((starts + ends) / 2.0 + 1.0, ends - starts + 1)
    return ranks


def _r2(y_eval, yhat) -> float:
    """1 - SSE/SST with SST about the evaluation rows' own mean; 0 when SST is 0."""
    if y_eval.max() == y_eval.min():
        return 0.0
    dev = y_eval - y_eval.mean()
    sst = float(np.sum(dev * dev))
    if not sst > 0.0:
        return 0.0
    res = y_eval - yhat
    return 1.0 - float(np.sum(res * res)) / sst


# ------------------------------------------------------------------------------------------------ group M
def _duplicates(xv, y):
    """(m_x_dup_row_frac, m_xy_dup_conflict_frac). Rows are compared as the raw bytes of (X + 0.0) over the valid
    columns (which turns -0.0 into +0.0); a duplicate-X group of size >= 2 conflicts when its (y + 0.0) differ."""
    n = xv.shape[0]
    rows = np.ascontiguousarray(xv + 0.0)
    groups: dict[bytes, list[int]] = {}
    for i in range(n):
        groups.setdefault(rows[i].tobytes(), []).append(i)
    yk = y + 0.0
    conflict = 0
    for members in groups.values():
        if len(members) >= 2:
            vals = yk[members]
            if vals.max() != vals.min():
                conflict += len(members)
    return (n - len(groups)) / n, conflict / n


# ------------------------------------------------------------------------------------------------ group R
def _eta2(x_raw, x_unit, distinct: int, yu, mu: float, total: float) -> float:
    """Correlation ratio: one bin per distinct value when distinct <= 16, else decile bins
    (searchsorted(Q(x, .1..9), x, side='right'))."""
    if distinct <= LOWCARD_MAX:
        _, codes = np.unique(x_raw, return_inverse=True)
        codes = codes.reshape(-1)
    else:
        codes = np.searchsorted(np.quantile(x_unit, DECILES, method="linear"), x_unit, side="right")
    counts = np.bincount(codes)
    sums = np.bincount(codes, weights=yu)
    nz = counts > 0
    means = sums[nz] / counts[nz]
    between = float(np.sum(counts[nz] * (means - mu) ** 2))
    return _clip(between / total, 0.0, 1.0)


# ------------------------------------------------------------------------------------------------ group L
def _select(xa, xu, y, rows, k: int) -> list[int]:
    """The top-k active columns by |rho| computed on `rows`; columns constant on those rows are excluded; ties go to
    the lower index. (A column whose unit-coordinate spread underflows to 0 on the rows counts as constant.)"""
    ry = _avg_rank(y[rows])
    cand, score = [], []
    for j in range(xa.shape[1]):
        xr = xa[rows, j]
        if xr.max() == xr.min() or not float(np.std(xu[rows, j])) > 0.0:
            continue
        cand.append(j)
        score.append(abs(_pearson(_avg_rank(xr), ry)))
    ranked = sorted(range(len(cand)), key=lambda i: (-score[i], cand[i]))
    return [cand[i] for i in ranked[:k]]


def _fit(xu, yu, rows, cols):
    """Ridge on the rows' standardized columns: beta = solve(Z'Z + 1e-6*n_rows*I, Z'(y - ybar))."""
    yr = yu[rows]
    ybar = float(np.mean(yr))
    if not cols:
        return ybar, None, None, None
    s = xu[np.ix_(rows, cols)]
    mu = s.mean(axis=0)
    sd = s.std(axis=0)
    z = (s - mu) / sd
    gram = np.einsum("ni,nj->ij", z, z, optimize=False)
    gram[np.diag_indices(len(cols))] += RIDGE * s.shape[0]
    rhs = np.einsum("ni,n->i", z, yr - ybar, optimize=False)
    return ybar, mu, sd, np.linalg.solve(gram, rhs)


def _predict(xu, model, rows, cols):
    ybar, mu, sd, beta = model
    if beta is None:
        return np.full(len(rows), ybar)
    z = (xu[np.ix_(rows, cols)] - mu) / sd
    return ybar + np.einsum("nk,k->n", z, beta, optimize=False)


def _knn1_predict(xu, yu, model, train, evals, cols):
    """y of the nearest fold-train row (squared Euclidean distance on the fold-train standardization); ties go to the
    lowest position, where a tie is any squared distance <= dmin + KNN_TIE_RTOL*dmin + KNN_TIE_ATOL (so float
    rounding cannot split an exact tie; see KNN_TIE_RTOL). Distances are accumulated per column in chunks of <= 256
    evaluation rows, without BLAS."""
    ytr = yu[train]
    if not cols:
        return np.full(evals.size, ytr[0])
    _, mu, sd, _ = model
    zt = (xu[np.ix_(train, cols)] - mu) / sd
    ze = (xu[np.ix_(evals, cols)] - mu) / sd
    out = np.empty(evals.size, dtype=np.float64)
    for start in range(0, evals.size, KNN_CHUNK):
        e = ze[start:start + KNN_CHUNK]
        dist = np.zeros((e.shape[0], zt.shape[0]), dtype=np.float64)
        for f in range(len(cols)):
            diff = e[:, f][:, None] - zt[:, f][None, :]
            dist += diff * diff
        dmin = dist.min(axis=1)
        tied = dist <= (dmin + (KNN_TIE_RTOL * dmin + KNN_TIE_ATOL))[:, None]
        out[start:start + e.shape[0]] = ytr[np.argmax(tied, axis=1)]     # first True = lowest tied position
    return out


def _cross_validated(xa, xu, y, yu, g, k: int):
    """(lin_top8_r2_cv, knn1_top8_r2_cv): mean over the two inner folds of the clipped R2 on g == f, with selection,
    standardization and fit on g != f. Both are 0 unless set(g) == {0, 1} and each fold has >= 2k + 2 rows."""
    present = np.unique(g)
    if present.size != 2 or int(present[0]) != 0 or int(present[1]) != 1:
        return 0.0, 0.0
    folds = (np.flatnonzero(g == 0), np.flatnonzero(g == 1))
    if min(folds[0].size, folds[1].size) < 2 * k + 2:
        return 0.0, 0.0
    lin = knn = 0.0
    for f in (0, 1):
        evals, train = folds[f], folds[1 - f]
        cols = _select(xa, xu, y, train, k)
        model = _fit(xu, yu, train, cols)
        y_eval = yu[evals]
        lin += _clip(_r2(y_eval, _predict(xu, model, evals, cols)), -1.0, 1.0)
        knn += _clip(_r2(y_eval, _knn1_predict(xu, yu, model, train, evals, cols)), -1.0, 1.0)
    return lin / 2.0, knn / 2.0


# ------------------------------------------------------------------------------------------------ assembly
def _compute(x, y, g):
    n, p = x.shape
    v = np.zeros(N_FEATURES, dtype=np.float64)

    # S: size
    v[0] = math.log(n)
    v[1] = math.log(p)
    v[2] = p / n
    v[3] = 1.0 / math.sqrt(n)

    # M: missingness and data quality. Invalid columns (any non-finite cell) are only counted.
    finite = np.isfinite(x)
    valid = finite.all(axis=0)
    valid_idx = np.flatnonzero(valid)
    v[4] = (n * p - int(np.count_nonzero(finite))) / (n * p)
    v[5] = (p - valid_idx.size) / p
    distinct = np.zeros(p, dtype=np.int64)
    mode_share = np.zeros(p, dtype=np.float64)
    for j in valid_idx:
        _, counts = np.unique(x[:, j], return_counts=True)
        distinct[j] = counts.size
        mode_share[j] = counts.max() / n
    active = valid & (distinct >= 2)
    v[6] = int(np.count_nonzero(valid & (distinct == 1))) / p
    v[7] = int(np.count_nonzero(valid & (distinct == 2))) / p
    v[8] = int(np.count_nonzero(valid & (distinct >= 3) & (distinct <= LOWCARD_MAX))) / p
    if valid_idx.size:
        v[9] = float(np.mean(mode_share[valid_idx]))
        v[10] = float(np.max(mode_share[valid_idx]))
    v[11] = int(np.count_nonzero(valid & (mode_share > 0.5))) / p
    a_idx = np.flatnonzero(active)
    p_a = int(a_idx.size)
    xa = np.ascontiguousarray(x[:, a_idx])
    if p_a:
        v[12] = float(np.mean([np.mean(xa[:, j] == xa[:, j].min()) for j in range(p_a)]))
    v[13], v[14] = _duplicates(x[:, valid_idx], y)

    # Y: target statistics in robust units (unit coordinates: ymin -> 0, ymax -> 1)
    deg_y = bool(y.max() == y.min())
    yu = mu = None
    if deg_y:
        v[32] = 1.0 / n
        v[33] = v[34] = v[35] = 1.0
    else:
        yu = _unit(y)
        med = float(np.median(yu))
        mu = float(np.mean(yu))
        sd = float(np.std(yu))
        s = labels.robust_scale(yu)
        skew_y, kurt_y = _moments(yu, is_unit=True)
        v[15] = s / sd
        v[16] = _slog((mu - med) / s)
        v[17] = _slog(skew_y)
        v[18] = _slog(kurt_y)
        v[19] = _slog((0.0 - med) / s)
        for i, q in enumerate(Y_QUANTILES):
            v[20 + i] = _slog((_q(yu, q) - med) / s)
        v[28] = _slog((1.0 - med) / s)
        v[29] = math.log1p(1.0 / s)
        absdev = np.abs(yu - med)
        v[30] = float(np.mean(absdev > 3.0 * s))
        v[31] = float(np.mean(absdev > 5.0 * s))
        _, ycounts = np.unique(y, return_counts=True)
        v[32] = ycounts.size / n
        v[33] = ycounts.max() / n
        v[34] = float(np.mean(y == y.min()))
        v[35] = float(np.mean(y == y.max()))
        v[36] = _clip((skew_y * skew_y + 1.0) / (kurt_y + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))), 0.0, 1.0)
        v[37] = _slog(_moments(np.log1p(yu / s))[0])
        v[38] = _pearson(yu[:-1], yu[1:])
        v[39] = _q(yu, 0.75) - _q(yu, 0.25)

    if p_a == 0:
        v[80] = v[81] = 1.0                       # C rule for p_a < 2; X, R and L stay 0
        return v

    # X: marginals of the active columns (unit coordinates: min -> 0, max -> 1)
    xu = np.ascontiguousarray(_unit_cols(xa))
    skew_x = np.empty(p_a)
    kurt_x = np.empty(p_a)
    out5 = np.empty(p_a)
    range_rz = np.empty(p_a)
    iqr_ratio = np.empty(p_a)
    uniq = distinct[a_idx] / n
    for j in range(p_a):
        col = xu[:, j]
        skew_x[j], kurt_x[j] = _moments(col, is_unit=True)
        rs = labels.robust_scale(col)
        out5[j] = float(np.mean(np.abs(col - float(np.median(col))) > 5.0 * rs))
        range_rz[j] = math.log1p(1.0 / rs)
        iqr_ratio[j] = _q(col, 0.75) - _q(col, 0.25)
    v[40] = p_a / p
    v[41] = _slog(_q(np.abs(skew_x), 0.5))
    v[42] = _slog(_q(np.abs(skew_x), 0.9))
    v[43] = _slog(_q(kurt_x, 0.5))
    v[44] = _slog(_q(kurt_x, 0.9))
    v[45] = _q(uniq, 0.1)
    v[46] = _q(uniq, 0.5)
    v[47] = _q(uniq, 0.9)
    v[48] = float(np.mean(out5))
    v[49] = float(np.max(out5))
    v[50] = _q(range_rz, 0.5)
    v[51] = _q(range_rz, 0.9)
    v[52] = _q(iqr_ratio, 0.1)
    v[53] = _q(iqr_ratio, 0.5)

    # C: redundancy of the active columns
    if p_a < 2:
        v[80] = v[81] = 1.0
    else:
        z = (xu - xu.mean(axis=0)) / xu.std(axis=0)
        c = np.einsum("ni,nj->ij", z, z, optimize=False) / n
        pairs = np.abs(c[np.triu_indices(p_a, 1)])
        v[74] = _q(pairs, 0.5)
        v[75] = _q(pairs, 0.9)
        v[76] = float(np.max(pairs))
        v[77] = float(np.mean(pairs))
        v[78] = float(np.mean(pairs > 0.9))
        off = np.abs(c)
        np.fill_diagonal(off, -1.0)
        v[79] = float(np.mean(off.max(axis=1) > 0.99))
        np.multiply(c, c, out=off)
        v[80] = p_a / float(np.sum(off))
        # 64 power iterations; matvec = (C * v[None, :]).sum(1) into a reused buffer (no BLAS, no per-step allocation)
        buf = off
        vec = np.full(p_a, 1.0 / math.sqrt(p_a))
        share = None
        for _ in range(POWER_ITERATIONS):
            cv = np.multiply(c, vec[None, :], out=buf).sum(axis=1)
            norm = math.sqrt(float(np.sum(cv * cv)))
            if norm == 0.0:
                share = 0.0
                break
            vec = cv / norm
        if share is None:
            share = float(np.sum(vec * np.multiply(c, vec[None, :], out=buf).sum(axis=1))) / p_a
        v[81] = share

    if deg_y:
        return v                                  # R and L are 0 for a constant target

    # R: feature-target relationships
    ry = _avg_rank(y)
    total = float(np.sum((yu - mu) ** 2))
    r = np.empty(p_a)
    rho = np.empty(p_a)
    eta = np.empty(p_a)
    for j in range(p_a):
        r[j] = _pearson(xu[:, j], yu, a_unit=True, b_unit=True)
        rho[j] = _pearson(_avg_rank(xa[:, j]), ry)
        eta[j] = _eta2(xa[:, j], xu[:, j], int(distinct[a_idx[j]]), yu, mu, total)
    ar = np.abs(r)
    arho = np.abs(rho)
    r2 = r * r
    v[54] = float(np.max(ar))
    v[55] = _q(ar, 0.9)
    v[56] = _q(ar, 0.5)
    v[57] = float(np.mean(ar))
    v[58] = float(np.mean(np.sort(ar)[::-1][:min(3, p_a)]))
    v[59] = math.log1p(float(np.sum(r2)))
    v[60] = float(np.mean(ar > 0.1))
    v[61] = float(np.mean(ar > 0.3))
    v[62] = float(np.mean(r > 0.0))
    v[63] = float(np.max(arho))
    v[64] = _q(arho, 0.9)
    v[65] = _q(arho, 0.5)
    v[66] = float(np.mean(arho))
    v[67] = float(np.max(arho - ar))
    v[68] = float(np.mean(arho - ar))
    v[69] = float(np.max(eta))
    v[70] = _q(eta, 0.5)
    v[71] = float(np.max(eta - r2))
    v[72] = float(np.mean(eta - r2))
    v[73] = float(uniq[int(np.argmax(ar))])

    # L: landmarkers on the top-k columns by |rho|
    k = min(LANDMARK_K, p_a)
    all_rows = np.arange(n)
    cols = sorted(range(p_a), key=lambda j: (-arho[j], j))[:k]
    model = _fit(xu, yu, all_rows, cols)
    yhat = _predict(xu, model, all_rows, cols)
    v[82] = _clip(_r2(yu, yhat), 0.0, 1.0)
    v[83], v[84] = _cross_validated(xa, xu, y, yu, g, k)
    resid = yu - yhat
    v[85] = _pearson(np.abs(resid), yhat)
    skew_e, kurt_e = _moments(resid)
    v[86] = _slog(skew_e)
    v[87] = _slog(kurt_e)
    return v


def features(X, y, inner_labels):
    """The 88 A0 features (float64) of one context's outer-train rows: X f64[n, p], y f64[n], inner_labels u8[n]."""
    x = np.asarray(X, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    g = np.asarray(inner_labels)
    if x.ndim != 2 or yy.ndim != 1 or g.ndim != 1 or x.shape[0] != yy.shape[0] or g.shape[0] != yy.shape[0] \
            or x.shape[1] < 1:
        raise A0Error(f"shape contract broken: X {x.shape}, y {yy.shape}, inner_labels {g.shape}")
    n = int(x.shape[0])
    if n < MIN_ROWS:
        raise A0Refused("too_few_rows", f"{n} train rows < {MIN_ROWS}")
    if not bool(np.all(np.isfinite(yy))):
        raise A0Refused("nonfinite_target", f"{int(np.count_nonzero(~np.isfinite(yy)))} non-finite y values")
    try:
        with np.errstate(divide="raise", over="raise", invalid="raise", under="ignore"):
            v = _compute(x, yy, g)
    except (FloatingPointError, ZeroDivisionError, OverflowError, labels.LabelError, np.linalg.LinAlgError) as exc:
        raise A0Error(f"A0 computation failed: {type(exc).__name__}: {exc}") from exc
    bad = [FEATURES[i] for i in np.flatnonzero(~np.isfinite(v))]
    if bad:
        raise A0Error(f"non-finite A0 features: {bad}")
    over = [FEATURES[i] for i in np.flatnonzero(np.abs(v) > FEATURE_ABS_BOUND)]
    if over:
        raise A0Error(f"A0 features outside [-{FEATURE_ABS_BOUND:g}, {FEATURE_ABS_BOUND:g}]: {over}")
    return v


def to_f32le(v) -> bytes:
    """Cast the float64 features to '<f4' (352 bytes); -0.0 becomes +0.0; every value must stay finite."""
    a = np.asarray(v, dtype=np.float64)
    if a.shape != (N_FEATURES,) or not bool(np.all(np.isfinite(a))):
        raise A0Error(f"to_f32le needs {N_FEATURES} finite values, got shape {a.shape}")
    f = a.astype(DTYPE)
    f[f == 0] = 0.0
    if not bool(np.all(np.isfinite(f))):
        raise A0Error("an A0 feature is not finite in float32")
    out = f.tobytes()
    if len(out) != VECTOR_BYTES:
        raise A0Error(f"A0 vector has {len(out)} bytes, expected {VECTOR_BYTES}")
    return out


def from_f32le(data: bytes):
    """Decode one stored A0 vector (read-only float32[88])."""
    if len(data) != VECTOR_BYTES:
        raise ValueError(f"an A0 vector has {VECTOR_BYTES} bytes, got {len(data)}")
    return np.frombuffer(data, dtype=DTYPE)


def from_raw(raw):
    """(A0 bytes, {n_train, p, train_row_sha256}) of one raw JTT1 input. Only jtt1.train_view decodes, and only the
    train range; the header it returns is discarded."""
    _, inner_labels, x, y = jtt1.train_view(raw)
    vector = to_f32le(features(x, y, inner_labels))
    return vector, {"n_train": int(x.shape[0]), "p": int(x.shape[1]), "train_row_sha256": jtt1.train_rows_sha256(raw)}


# ------------------------------------------------------------------------------------------ source identity
_PACKAGE = "dltrain"
_THIS_MODULE = "metafeatures"


def _package_imports(tree: ast.AST, package_dir: Path) -> set[str]:
    """The dltrain modules a parsed module imports anywhere in its body (pattern copied from train.py/holdout.py)."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if (node.level == 0 and module == _PACKAGE) or (node.level == 1 and not module):
                found.update(alias.name for alias in node.names)
            elif node.level == 0 and module.startswith(_PACKAGE + "."):
                found.add(module.split(".")[1])
            elif node.level == 1:
                found.add(module.split(".")[0])
        elif isinstance(node, ast.Import):
            found.update(alias.name.split(".")[1] for alias in node.names if alias.name.startswith(_PACKAGE + "."))
    return {name for name in found if (package_dir / f"{name}.py").is_file()}


@functools.cache
def _source_tree() -> tuple[tuple[str, str], ...]:
    package_dir = Path(__file__).resolve().parent
    digests: dict[str, str] = {}
    pending_modules = [_THIS_MODULE]
    while pending_modules:
        name = pending_modules.pop()
        if f"{name}.py" in digests:
            continue
        raw_src = (package_dir / f"{name}.py").read_bytes()
        digests[f"{name}.py"] = common.sha256_bytes(raw_src)
        pending_modules.extend(sorted(_package_imports(ast.parse(raw_src, filename=f"{name}.py"), package_dir)))
    digests["__init__.py"] = common.sha256_file(package_dir / "__init__.py")
    return tuple((f"{_PACKAGE}/{name}", digest) for name, digest in sorted(digests.items()))


def source_tree_files() -> list[tuple[str, str]]:
    """(relative path, sha256) of the A0 producer's static dltrain import closure, sorted by path."""
    return list(_source_tree())


def source_tree_sha256() -> str:
    """sha256 over 'path NUL sha256 LF' lines of this module's static dltrain import closure (holdout.py/train.py
    convention)."""
    return common.sha256_bytes("".join(f"{path}\0{digest}\n" for path, digest in _source_tree()).encode("utf-8"))
