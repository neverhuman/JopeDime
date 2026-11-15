"""`SEARCH`-Only `float64` Regime & Tail Assay (`assay_regime.py`).

Computes strictly inductive `SEARCH`-only context diagnostics (`ARENA §5`, `cl_h458` `05:52Z` tail stat):
  - `tail_robust_sd`: `max_i |y_s - median(y_s)| / (1.4826 * MAD(y_s))` (`> 12.0` marks classic-floor tail zone).
  - `loo_u`: closed-form uncapped Matérn-5/2 Leave-One-Out log-RMSE utility vs `std(y_s)` across noise fractions.
  - `slope`: log-log LOO RMSE decay slope across nested subsamples `n in {140, 280, 560, min(1120, n_s)}`.
  - `top1_err_share`: `max_i e_{loo,i}^2 / sum_i e_{loo,i}^2` (single-row error concentration).
  - `is_lattice`: discrete/combinatorial multiplicity indicator.
  - `regime`: `"PRECISION"`, `"HEAVY_TAIL"`, `"LATTICE"`, or `"ORDINARY"`.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Sequence, Tuple

import numpy as np
from scipy import linalg as sla


@dataclasses.dataclass(frozen=True)
class AssayReport:
    """Strictly `SEARCH`-derived context regime diagnostics."""

    n_rows: int
    n_features: int
    loo_u: float
    slope: float
    top1_err_share: float
    tail_robust_sd: float
    max_discrete_ratio: float
    is_precision: bool
    is_heavy_tail: bool
    is_lattice: bool
    regime: str


def compute_robust_tail_stat(y_search: np.ndarray) -> float:
    """Compute `max_i |y_i - med(y)| / max(1.4826 * MAD(y), 1e-6 * std(y))` (`cl_h458` `05:52Z`)."""
    y = np.asarray(y_search, dtype=np.float64).reshape(-1)
    if y.size < 4:
        return 0.0
    med = float(np.median(y))
    mad = 1.4826 * float(np.median(np.abs(y - med)))
    std = max(float(np.std(y)), 1e-12)
    scale = max(mad, 1e-3 * std)
    return float(np.max(np.abs(y - med)) / scale)


def _matern52_gram(xa: np.ndarray, xb: np.ndarray, ell: float) -> np.ndarray:
    """Isotropic standardized Matérn-5/2 + weak linear Gram in `float64`."""
    d_cols = max(int(xa.shape[1]), 1)
    sc = float(ell) * math.sqrt(float(d_cols))
    a = xa / sc
    b = xb / sc
    an2 = np.sum(a * a, axis=1, keepdims=True)
    bn2 = np.sum(b * b, axis=1, keepdims=True).T
    dist = np.sqrt(np.maximum(an2 + bn2 - 2.0 * (a @ b.T), 0.0))
    s = math.sqrt(5.0) * dist
    return (1.0 + s + (s * s) / 3.0) * np.exp(-s) + 0.15 * (a @ b.T)


def uncapped_matern52_loo_fit(
    x_search: np.ndarray,
    y_search: np.ndarray,
    ell_grid: Sequence[float] = (0.6, 1.4, 3.2),
    rho_grid: Sequence[float] = (1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 0.5),
) -> Tuple[np.ndarray, float, float, float]:
    """Closed-form `e_loo = alpha / diag(C)` uncapped Matérn-5/2 fit (`float64`, zero noise floor cap).

    Returns `(best_e_loo, best_loo_rmse, best_ell, best_rho)`.
    """
    x = np.ascontiguousarray(x_search, dtype=np.float64)
    y = np.ascontiguousarray(y_search, dtype=np.float64).reshape(-1)
    y_mu = float(np.mean(y))
    yc = y - y_mu
    y_std = max(float(np.std(yc)), 1e-12)

    x_med = np.median(x, axis=0)
    q25, q75 = np.percentile(x, [25.0, 75.0], axis=0)
    scale_x = np.where((q75 - q25) > 1e-9, q75 - q25, np.maximum(np.std(x, axis=0), 1e-9))
    zx = np.clip((x - x_med) / scale_x, -25.0, 25.0)

    best_rmse = y_std
    best_e_loo = yc.copy()
    best_ell = float(ell_grid[0])
    best_rho = float(rho_grid[-1])

    for ell in ell_grid:
        k_mat = _matern52_gram(zx, zx, float(ell))
        ev, vecs = sla.eigh(0.5 * (k_mat + k_mat.T), check_finite=False)
        ev = np.maximum(ev, 0.0)
        vt_y = vecs.T @ yc
        v_sq = vecs * vecs
        for rho in rho_grid:
            lam = max(float(rho), 1e-10)
            inv_d = 1.0 / (ev + lam)
            alpha = vecs @ (vt_y * inv_d)
            c_diag = np.maximum(v_sq @ inv_d, 1e-15)
            e_loo = alpha / c_diag
            rmse = float(math.sqrt(np.mean(e_loo * e_loo)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_e_loo = e_loo
                best_ell = float(ell)
                best_rho = float(rho)

    return best_e_loo, best_rmse, best_ell, best_rho


def run_assay(
    x_search: np.ndarray,
    y_search: np.ndarray,
    seed: int = 462001,
) -> AssayReport:
    """Run `SEARCH`-only `ASSAY` regime classification (`< 1s` at `n_s <= 1120`)."""
    x = np.ascontiguousarray(x_search, dtype=np.float64)
    y = np.ascontiguousarray(y_search, dtype=np.float64).reshape(-1)
    n_rows, n_cols = x.shape
    y_std = max(float(np.std(y)), 1e-12)
    tail_sd = compute_robust_tail_stat(y)

    rng = np.random.default_rng(int(seed))
    max_n = min(1120, n_rows)
    perm = rng.permutation(n_rows)[:max_n]
    sizes = sorted(set([s for s in (70, 140, 280, 560, max_n) if 24 <= s <= max_n]))
    if not sizes:
        sizes = [max_n]

    log_ns = []
    log_rmses = []
    last_e_loo = y[perm] - float(np.mean(y[perm]))
    last_rmse = y_std
    for s_n in sizes:
        sub = perm[:s_n]
        e_loo, rmse_s, _, _ = uncapped_matern52_loo_fit(x[sub], y[sub])
        log_ns.append(math.log(float(s_n)))
        log_rmses.append(math.log(max(rmse_s, 1e-15)))
        if s_n == sizes[-1]:
            last_e_loo = e_loo
            last_rmse = rmse_s

    if len(log_ns) >= 2:
        x_ln = np.asarray(log_ns, dtype=np.float64)
        y_lr = np.asarray(log_rmses, dtype=np.float64)
        x_c = x_ln - float(np.mean(x_ln))
        denom = max(float(np.dot(x_c, x_c)), 1e-12)
        slope = -float(np.dot(x_c, y_lr - float(np.mean(y_lr))) / denom)
    else:
        slope = 0.0

    loo_u = float(math.log(y_std / max(last_rmse, 1e-15)))
    sq_err = last_e_loo * last_e_loo
    top1_share = float(np.max(sq_err) / max(float(np.sum(sq_err)), 1e-30))

    disc_ratios = []
    for j in range(n_cols):
        u_cnt = max(int(np.unique(np.round(x[:, j], 8)).size), 1)
        disc_ratios.append(float(n_rows) / float(u_cnt))
    max_disc = max(disc_ratios) if disc_ratios else 1.0

    is_precision = bool(loo_u >= 1.85 or (slope >= 0.45 and loo_u >= 1.25))
    is_heavy_tail = bool(tail_sd > 12.0 or top1_share > 0.28)
    is_lattice = bool(max_disc >= 8.0 and not is_precision)

    if is_precision:
        regime = "PRECISION"
    elif is_heavy_tail:
        regime = "HEAVY_TAIL"
    elif is_lattice:
        regime = "LATTICE"
    else:
        regime = "ORDINARY"

    return AssayReport(
        n_rows=int(n_rows),
        n_features=int(n_cols),
        loo_u=loo_u,
        slope=slope,
        top1_err_share=top1_share,
        tail_robust_sd=tail_sd,
        max_discrete_ratio=max_disc,
        is_precision=is_precision,
        is_heavy_tail=is_heavy_tail,
        is_lattice=is_lattice,
        regime=regime,
    )
