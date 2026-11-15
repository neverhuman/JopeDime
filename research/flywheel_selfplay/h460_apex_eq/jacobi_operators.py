"""Multi-Regime Discontinuous, Relational, and Rational Jacobi Operators (`jacobi_operators.py`).

Pure-Python `float64` implementations of the non-smooth and algebraic operator families
prioritized by `JACOBI`, `GP²`, and `cl_h458` (`04:08Z`, `280b` smooth-kernel floor finding):
  - `P2`: Exact O(n log n + n * top_cuts) sorted prefix-sum threshold scan `step(x_j - c)`
          conditioned on the precision residual `alpha = Sigma^{-1} r` with min-leaf support.
  - `B1`: Label-free exact-equality / partition PSD Gram kernel `K_part(x, x')`.
  - `B2`: Label-free relational Laplace / L1-delta PSD Gram kernel `K_delta(x, x')`.
  - `B4`: Gated-smooth regime-split basis `[1(x_j > c) * z_k, 1(x_j <= c) * z_k]`.
  - `P1/P3`: Pole-guarded rational law `x_u / (1 + exp(a)*|x_v| + b*x_v^2)` with 5-step
             damped Levenberg-Marquardt alignment against `alpha` (train-only).
"""
from __future__ import annotations

import dataclasses
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclasses.dataclass(frozen=True)
class StepFeatureCandidate:
    """Discovered threshold indicator `1(x[:, col_idx] > threshold)`."""

    name: str
    col_idx: int
    threshold: float
    proxy_score: float
    exact_score_z2: float
    phi_train: np.ndarray
    phi_val: Optional[np.ndarray]


def scan_best_step_features(
    x_train: np.ndarray,
    alpha: np.ndarray,
    inv_cov: np.ndarray,
    x_val: Optional[np.ndarray] = None,
    min_leaf: Optional[int] = None,
    top_per_col: int = 3,
    max_return: int = 8,
) -> List[StepFeatureCandidate]:
    """Exact prefix-sum O(n log n) scan per feature + exact `phi^T C phi` verification on top cuts."""
    x_tr = np.ascontiguousarray(x_train, dtype=np.float64)
    n_rows, n_cols = x_tr.shape
    x_va = None if x_val is None else np.ascontiguousarray(x_val, dtype=np.float64)
    leaf_floor = max(8, int(np.ceil(0.02 * n_rows))) if min_leaf is None else max(2, int(min_leaf))
    if n_rows < 2 * leaf_floor + 1:
        return []

    alpha_centered = np.asarray(alpha, dtype=np.float64).reshape(n_rows)
    candidates: List[StepFeatureCandidate] = []

    for j in range(n_cols):
        col = x_tr[:, j]
        order = np.argsort(col, kind="mergesort")
        s_col = col[order]
        s_alpha = alpha_centered[order]

        # Suffix sum of alpha for 1(x_j > c_k) where right side is indices [k+1 : n_rows]
        rev_cum = np.cumsum(s_alpha[::-1])[::-1]
        valid_cuts = np.flatnonzero(np.diff(s_col) > 1e-12)
        valid_cuts = valid_cuts[(valid_cuts >= leaf_floor - 1) & (valid_cuts <= n_rows - leaf_floor - 1)]
        if valid_cuts.size == 0:
            continue

        n_right = (n_rows - 1 - valid_cuts).astype(np.float64)
        p_right = n_right / float(n_rows)
        var_step = np.maximum(p_right * (1.0 - p_right) * float(n_rows), 1e-12)
        g_right = rev_cum[valid_cuts + 1]
        proxy_z2 = (g_right * g_right) / var_step

        best_local = valid_cuts[np.argsort(-proxy_z2)[: max(1, int(top_per_col))]]
        for cut_idx in best_local:
            k_int = int(cut_idx)
            thr = 0.5 * float(s_col[k_int] + s_col[k_int + 1])
            raw_tr = (col > thr).astype(np.float64)
            mean_tr = float(np.mean(raw_tr))
            std_tr = max(float(np.std(raw_tr)), 1e-9)
            phi_tr = (raw_tr - mean_tr) / std_tr
            phi_va = None if x_va is None else ((x_va[:, j] > thr).astype(np.float64) - mean_tr) / std_tr

            denom = max(float(phi_tr @ (inv_cov @ phi_tr)), 1e-15)
            z2_exact = float(np.dot(phi_tr, alpha_centered)) ** 2 / denom
            candidates.append(
                StepFeatureCandidate(
                    name=f"step(c{j}>{thr:.6g})",
                    col_idx=j,
                    threshold=thr,
                    proxy_score=float((g_right[valid_cuts == k_int][0] ** 2) / var_step[valid_cuts == k_int][0]),
                    exact_score_z2=z2_exact,
                    phi_train=phi_tr,
                    phi_val=phi_va,
                )
            )

    candidates.sort(key=lambda item: -item.exact_score_z2)
    return candidates[: max(1, int(max_return))]


def build_gated_smooth_features(
    step_cands: Sequence[StepFeatureCandidate],
    x_train: np.ndarray,
    x_val: Optional[np.ndarray] = None,
    max_basis_cols: int = 4,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    """Construct `B4` regime-gated linear interaction columns `[gate * z_k, (1-gate) * z_k]`."""
    x_tr = np.ascontiguousarray(x_train, dtype=np.float64)
    n_rows, n_cols = x_tr.shape
    x_va = None if x_val is None else np.ascontiguousarray(x_val, dtype=np.float64)
    if not step_cands or n_cols == 0:
        empty_va = None if x_va is None else np.zeros((x_va.shape[0], 0), dtype=np.float64)
        return np.zeros((n_rows, 0), dtype=np.float64), empty_va, []

    col_vars = np.var(x_tr, axis=0)
    top_cols = np.argsort(-col_vars)[: min(int(max_basis_cols), n_cols)]
    tr_list: List[np.ndarray] = []
    va_list: List[np.ndarray] = []
    names: List[str] = []

    for sc in step_cands[:3]:
        gate_tr = (x_tr[:, sc.col_idx] > sc.threshold).astype(np.float64)
        gate_va = None if x_va is None else (x_va[:, sc.col_idx] > sc.threshold).astype(np.float64)
        for k_idx in top_cols:
            k = int(k_idx)
            mu_k = float(np.mean(x_tr[:, k]))
            sd_k = max(float(np.std(x_tr[:, k])), 1e-9)
            z_tr = (x_tr[:, k] - mu_k) / sd_k
            z_va = None if x_va is None else (x_va[:, k] - mu_k) / sd_k
            for tag, g_tr, g_va in (
                ("hi", gate_tr, gate_va),
                ("lo", 1.0 - gate_tr, None if gate_va is None else 1.0 - gate_va),
            ):
                prod_tr = g_tr * z_tr
                p_mu = float(np.mean(prod_tr))
                p_sd = float(np.std(prod_tr))
                if p_sd <= 1e-8:
                    continue
                tr_list.append((prod_tr - p_mu) / p_sd)
                if z_va is not None and g_va is not None:
                    va_list.append((g_va * z_va - p_mu) / p_sd)
                names.append(f"gated_{tag}({sc.name}*c{k})")

    if not tr_list:
        empty_va = None if x_va is None else np.zeros((x_va.shape[0], 0), dtype=np.float64)
        return np.zeros((n_rows, 0), dtype=np.float64), empty_va, []
    mat_tr = np.column_stack(tr_list)
    mat_va = None if x_va is None else np.column_stack(va_list)
    return mat_tr, mat_va, names


def _quantize_codes_train_only(
    x_train: np.ndarray,
    x_val: Optional[np.ndarray],
    n_bins: int = 16,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Map low-cardinality columns to exact values and continuous columns to train-quantile bins."""
    x_tr = np.ascontiguousarray(x_train, dtype=np.float64)
    n_rows, n_cols = x_tr.shape
    codes_tr = np.zeros((n_rows, n_cols), dtype=np.int64)
    codes_va = None if x_val is None else np.zeros((x_val.shape[0], n_cols), dtype=np.int64)
    qs = np.linspace(0.0, 1.0, int(n_bins) + 1)[1:-1]

    for j in range(n_cols):
        col = np.round(x_tr[:, j], decimals=8)
        uniq = np.unique(col)
        if uniq.size <= max(32, 2 * int(n_bins)):
            codes_tr[:, j] = np.searchsorted(uniq, col)
            if codes_va is not None and x_val is not None:
                va_round = np.round(x_val[:, j], decimals=8)
                idx = np.searchsorted(uniq, va_round)
                idx_clip = np.clip(idx, 0, uniq.size - 1)
                exact = uniq[idx_clip] == va_round
                codes_va[:, j] = np.where(exact, idx_clip, -1 - j)
        else:
            edges = np.unique(np.quantile(x_tr[:, j], qs))
            codes_tr[:, j] = np.digitize(x_tr[:, j], edges)
            if codes_va is not None and x_val is not None:
                codes_va[:, j] = np.digitize(x_val[:, j], edges)
    return codes_tr, codes_va


def build_partition_kernel(
    x_train: np.ndarray,
    x_val: Optional[np.ndarray] = None,
    n_bins: int = 16,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """`B1` Label-free exact-equality / partition PSD Gram kernel `K_part(x, x')`."""
    codes_tr, codes_va = _quantize_codes_train_only(x_train, x_val, n_bins=n_bins)
    n_cols = max(1, codes_tr.shape[1])
    k_tr = np.zeros((codes_tr.shape[0], codes_tr.shape[0]), dtype=np.float64)
    for j in range(codes_tr.shape[1]):
        c_j = codes_tr[:, j]
        k_tr += (c_j[:, None] == c_j[None, :]).astype(np.float64)
    k_tr /= float(n_cols)

    if codes_va is None:
        return k_tr, None
    k_va = np.zeros((codes_va.shape[0], codes_tr.shape[0]), dtype=np.float64)
    for j in range(codes_tr.shape[1]):
        k_va += (codes_va[:, j][:, None] == codes_tr[:, j][None, :]).astype(np.float64)
    k_va /= float(n_cols)
    return k_tr, k_va


def build_delta_kernel(
    x_train: np.ndarray,
    x_val: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """`B2` Label-free relational Laplace (`exp(-sum_j |x_j - x'_j| / (d * scale_j))`) PSD kernel."""
    x_tr = np.ascontiguousarray(x_train, dtype=np.float64)
    n_cols = max(1, x_tr.shape[1])
    med = np.median(x_tr, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(x_tr - med), axis=0), 1e-6) * float(n_cols)
    z_tr = (x_tr - med) / scale
    dist_tr = np.sum(np.abs(z_tr[:, None, :] - z_tr[None, :, :]), axis=2)
    k_tr = np.exp(-dist_tr)

    if x_val is None:
        return k_tr, None
    z_va = (np.ascontiguousarray(x_val, dtype=np.float64) - med) / scale
    dist_va = np.sum(np.abs(z_va[:, None, :] - z_tr[None, :, :]), axis=2)
    return k_tr, np.exp(-dist_va)


def fit_rational_law_vp(
    x_train: np.ndarray,
    alpha: np.ndarray,
    u_col: int,
    v_col: int,
    x_val: Optional[np.ndarray] = None,
    n_steps: int = 5,
    damping: float = 1e-2,
) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    """`P1/P3` Pole-safe rational law `z_u / (1 + exp(a)*|z_v| + exp(b)*z_v^2)` via 5-step LM."""
    x_tr = np.ascontiguousarray(x_train, dtype=np.float64)
    mu_u, sd_u = float(np.mean(x_tr[:, u_col])), max(float(np.std(x_tr[:, u_col])), 1e-9)
    mu_v, sd_v = float(np.mean(x_tr[:, v_col])), max(float(np.std(x_tr[:, v_col])), 1e-9)
    zu_tr = (x_tr[:, u_col] - mu_u) / sd_u
    zv_tr = (x_tr[:, v_col] - mu_v) / sd_v
    target_dir = np.asarray(alpha, dtype=np.float64).reshape(-1)
    t_norm = max(float(np.linalg.norm(target_dir)), 1e-12)
    t_unit = target_dir / t_norm

    theta = np.array([-0.5, -1.5], dtype=np.float64)  # log-rates keep denominator >= 1.0 everywhere

    def _eval_basis(zu: np.ndarray, zv: np.ndarray, th: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ea = math.exp(float(np.clip(th[0], -8.0, 4.0)))
        eb = math.exp(float(np.clip(th[1], -8.0, 4.0)))
        den = 1.0 + ea * np.abs(zv) + eb * (zv * zv)
        raw = zu / den
        return raw, den

    for _ in range(max(1, int(n_steps))):
        raw, den = _eval_basis(zu_tr, zv_tr, theta)
        r_sd = max(float(np.std(raw)), 1e-9)
        phi = (raw - float(np.mean(raw))) / r_sd
        sign = 1.0 if float(np.dot(phi, t_unit)) >= 0.0 else -1.0
        err = sign * t_unit - phi
        ea = math.exp(float(np.clip(theta[0], -8.0, 4.0)))
        eb = math.exp(float(np.clip(theta[1], -8.0, 4.0)))
        d_raw_da = -raw * (ea * np.abs(zv_tr)) / den
        d_raw_db = -raw * (eb * (zv_tr * zv_tr)) / den
        jac = np.column_stack([(d_raw_da - np.mean(d_raw_da)) / r_sd, (d_raw_db - np.mean(d_raw_db)) / r_sd])
        jtj = jac.T @ jac + float(damping) * np.eye(2, dtype=np.float64)
        step = np.linalg.solve(jtj, jac.T @ err)
        theta = np.clip(theta + 0.5 * step, -8.0, 4.0)

    raw_tr, _ = _eval_basis(zu_tr, zv_tr, theta)
    rmu, rsd = float(np.mean(raw_tr)), max(float(np.std(raw_tr)), 1e-9)
    phi_tr = (raw_tr - rmu) / rsd
    phi_va = None
    if x_val is not None:
        x_va = np.ascontiguousarray(x_val, dtype=np.float64)
        zu_va = (x_va[:, u_col] - mu_u) / sd_u
        zv_va = (x_va[:, v_col] - mu_v) / sd_v
        raw_va, _ = _eval_basis(zu_va, zv_va, theta)
        phi_va = (raw_va - rmu) / rsd
    ea_f = math.exp(float(theta[0]))
    eb_f = math.exp(float(theta[1]))
    return phi_tr, phi_va, f"rat(c{u_col}/(1+{ea_f:.3g}|c{v_col}|+{eb_f:.3g}c{v_col}^2))"
