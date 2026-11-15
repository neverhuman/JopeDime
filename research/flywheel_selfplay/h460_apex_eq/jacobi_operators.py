"""Multi-Regime Discontinuous, Relational, and Rational Jacobi Operators (`jacobi_operators.py`).

Pure-Python `float64` implementations of the non-smooth and algebraic operator families
prioritized by `JACOBI`, `GP^2`, `cl_mtc` (`04:32Z`/`04:37Z` router dictionary), `cl_h458`
(`04:08Z`, `280b` smooth-kernel floor finding), and `cl_fwriter` (`04:36Z` `G0_8` + `N1-N6`):
  - `P2`: Exact O(n log n + n * top_cuts) sorted prefix-sum threshold scan `step(x_j - c)`
          conditioned on the precision residual `alpha = Sigma^{-1} r` with min-leaf support.
  - `B1`: Label-free exact-equality / partition PSD Gram kernel `K_part(x, x')` + LOO projection.
  - `B2`: Label-free relational Laplace / L1-delta PSD Gram kernel `K_delta(x, x')` + LOO projection.
  - `B4`: Gated-smooth regime-split basis `[1(x_j > c) * z_k, 1(x_j <= c) * z_k]`.
  - `P1/P3`: Pole-guarded rational law `z_u / (1 + exp(a)*|z_v| + exp(b)*z_v^2)` with 5-step
             damped Levenberg-Marquardt alignment against `alpha` (train-only).
  - `build_production_router_dictionary`: Unified production dictionary builder called identically
             on real fits, inner router folds (`N1`), and nested placebo null draws (`G0_8`).
"""
from __future__ import annotations

import dataclasses
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import linalg as sla


def _spectral_kernel_features(
    k_tr: np.ndarray,
    k_va: Optional[np.ndarray],
    tag: str,
    max_eigs: int = 4,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    """Label-free (`X`-only) centered Nyström spectral features of PSD Gram matrix `k_tr`."""
    n_tr = k_tr.shape[0]
    if n_tr < 4:
        empty_va = None if k_va is None else np.zeros((k_va.shape[0], 0), dtype=np.float64)
        return np.zeros((n_tr, 0), dtype=np.float64), empty_va, []
    row_mu = np.mean(k_tr, axis=1, keepdims=True)
    col_mu = np.mean(k_tr, axis=0, keepdims=True)
    grand_mu = float(np.mean(k_tr))
    k_c = 0.5 * ((k_tr - row_mu - col_mu + grand_mu) + (k_tr - row_mu - col_mu + grand_mu).T)
    m_take = min(int(max_eigs), n_tr - 1)
    vals, vecs = sla.eigh(k_c, subset_by_index=[n_tr - m_take, n_tr - 1], check_finite=False)
    order = np.argsort(-vals)
    vals = vals[order]
    vecs = vecs[:, order]
    keep = vals > max(1e-6, 1e-5 * float(vals[0]) if vals.size else 1e-6)
    if not np.any(keep):
        empty_va = None if k_va is None else np.zeros((k_va.shape[0], 0), dtype=np.float64)
        return np.zeros((n_tr, 0), dtype=np.float64), empty_va, []
    vals_k = vals[keep]
    vecs_k = vecs[:, keep]
    scale_tr = math.sqrt(float(n_tr))
    phi_tr = vecs_k * scale_tr
    names = [f"{tag}_eig{i}" for i in range(phi_tr.shape[1])]
    if k_va is None:
        return phi_tr, None, names
    va_row_mu = np.mean(k_va, axis=1, keepdims=True)
    k_va_c = k_va - va_row_mu - col_mu + grand_mu
    phi_va = (k_va_c @ vecs_k) * (scale_tr / vals_k[None, :])
    return phi_tr, phi_va, names


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


@dataclasses.dataclass(frozen=True)
class PrecomputedXContext:
    """Label-free `x_train`/`x_val` caches (sort orders & label-free PSD Gram matrices).

    Depends strictly on `x_train` and `x_val` (zero label dependence), so reusing it across
    permutation draws of `r` is 100% IEEE-754 bit-identical to recomputing from scratch.
    """

    x_train: np.ndarray
    x_val: Optional[np.ndarray]
    col_orders: Tuple[np.ndarray, ...]
    z_raw_train: np.ndarray
    z_raw_val: Optional[np.ndarray]
    k_part_train: np.ndarray
    k_part_val: Optional[np.ndarray]
    k_delta_train: np.ndarray
    k_delta_val: Optional[np.ndarray]
    b1_b2_spec_train: np.ndarray
    b1_b2_spec_val: Optional[np.ndarray]
    b1_b2_spec_names: Tuple[str, ...]

    def without_val(self) -> "PrecomputedXContext":
        return PrecomputedXContext(
            x_train=self.x_train,
            x_val=None,
            col_orders=self.col_orders,
            z_raw_train=self.z_raw_train,
            z_raw_val=None,
            k_part_train=self.k_part_train,
            k_part_val=None,
            k_delta_train=self.k_delta_train,
            k_delta_val=None,
            b1_b2_spec_train=self.b1_b2_spec_train,
            b1_b2_spec_val=None,
            b1_b2_spec_names=self.b1_b2_spec_names,
        )

    @classmethod
    def from_x(
        cls,
        x_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        n_bins: int = 16,
    ) -> "PrecomputedXContext":
        x_tr = np.ascontiguousarray(x_train, dtype=np.float64)
        x_va = None if x_val is None else np.ascontiguousarray(x_val, dtype=np.float64)
        orders = tuple(np.argsort(x_tr[:, j], kind="mergesort") for j in range(x_tr.shape[1]))
        mu = np.mean(x_tr, axis=0)
        sd = np.maximum(np.std(x_tr, axis=0), 1e-9)
        z_tr = (x_tr - mu) / sd
        z_va = None if x_va is None else (x_va - mu) / sd
        kp_tr, kp_va = build_partition_kernel(x_tr, x_va, n_bins=n_bins)
        kd_tr, kd_va = build_delta_kernel(x_tr, x_va)
        sp_p_tr, sp_p_va, sp_p_nm = _spectral_kernel_features(kp_tr, kp_va, "b1_part", max_eigs=4)
        sp_d_tr, sp_d_va, sp_d_nm = _spectral_kernel_features(kd_tr, kd_va, "b2_delta", max_eigs=4)
        spec_tr = np.hstack([sp_p_tr, sp_d_tr])
        spec_va = None if x_va is None else np.hstack([sp_p_va, sp_d_va])  # type: ignore[list-item]
        return cls(
            x_train=x_tr,
            x_val=x_va,
            col_orders=orders,
            z_raw_train=z_tr,
            z_raw_val=z_va,
            k_part_train=kp_tr,
            k_part_val=kp_va,
            k_delta_train=kd_tr,
            k_delta_val=kd_va,
            b1_b2_spec_train=spec_tr,
            b1_b2_spec_val=spec_va,
            b1_b2_spec_names=tuple(sp_p_nm + sp_d_nm),
        )


def scan_best_step_features(
    x_train: np.ndarray,
    alpha: np.ndarray,
    inv_cov: np.ndarray,
    x_val: Optional[np.ndarray] = None,
    min_leaf: Optional[int] = None,
    top_per_col: int = 3,
    max_return: int = 8,
    col_orders: Optional[Sequence[np.ndarray]] = None,
) -> List[StepFeatureCandidate]:
    """Exact prefix-sum O(n log n) scan per feature + exact batched `phi^T C phi` verification."""
    x_tr = np.ascontiguousarray(x_train, dtype=np.float64)
    n_rows, n_cols = x_tr.shape
    x_va = None if x_val is None else np.ascontiguousarray(x_val, dtype=np.float64)
    leaf_floor = max(8, int(np.ceil(0.02 * n_rows))) if min_leaf is None else max(2, int(min_leaf))
    if n_rows < 2 * leaf_floor + 1 or n_cols == 0:
        return []

    alpha_centered = np.asarray(alpha, dtype=np.float64).reshape(n_rows)
    pre_meta: List[Tuple[int, float, float, np.ndarray, Optional[np.ndarray]]] = []

    for j in range(n_cols):
        col = x_tr[:, j]
        order = np.argsort(col, kind="mergesort") if col_orders is None else col_orders[j]
        s_col = col[order]
        s_alpha = alpha_centered[order]

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

        top_pos = np.argsort(-proxy_z2)[: max(1, int(top_per_col))]
        for pos in top_pos:
            pos_i = int(pos)
            k_int = int(valid_cuts[pos_i])
            thr = 0.5 * float(s_col[k_int] + s_col[k_int + 1])
            raw_tr = (col > thr).astype(np.float64)
            mean_tr = float(np.mean(raw_tr))
            std_tr = max(float(np.std(raw_tr)), 1e-9)
            phi_tr = (raw_tr - mean_tr) / std_tr
            phi_va = None if x_va is None else ((x_va[:, j] > thr).astype(np.float64) - mean_tr) / std_tr
            pre_meta.append((j, thr, float(proxy_z2[pos_i]), phi_tr, phi_va))

    if not pre_meta:
        return []

    phi_mat = np.column_stack([m[3] for m in pre_meta])
    c_phi = inv_cov @ phi_mat
    denoms = np.maximum(np.einsum("ij,ij->j", phi_mat, c_phi), 1e-15)
    numer = (phi_mat.T @ alpha_centered) ** 2
    z2_all = numer / denoms

    candidates = [
        StepFeatureCandidate(
            name=f"step(c{m[0]}>{m[1]:.6g})",
            col_idx=m[0],
            threshold=m[1],
            proxy_score=m[2],
            exact_score_z2=float(z2_all[idx]),
            phi_train=m[3],
            phi_val=m[4],
        )
        for idx, m in enumerate(pre_meta)
    ]
    candidates.sort(key=lambda item: -item.exact_score_z2)
    return candidates[: max(1, int(max_return))]


def scan_best_equality_features(
    x_train: np.ndarray,
    alpha: np.ndarray,
    inv_cov: np.ndarray,
    x_val: Optional[np.ndarray] = None,
    min_leaf: Optional[int] = None,
    top_per_col: int = 2,
    max_return: int = 6,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    """Scan discrete/repeated-value equality indicators `1(x_j == v)` (`PREREG_G1` §2)."""
    x_tr = np.ascontiguousarray(x_train, dtype=np.float64)
    n_rows, n_cols = x_tr.shape
    x_va = None if x_val is None else np.ascontiguousarray(x_val, dtype=np.float64)
    leaf_floor = max(8, int(np.ceil(0.02 * n_rows))) if min_leaf is None else max(2, int(min_leaf))
    alpha_vec = np.asarray(alpha, dtype=np.float64).reshape(n_rows)

    pre_cols: List[Tuple[str, np.ndarray, Optional[np.ndarray], float]] = []
    for j in range(n_cols):
        col_r = np.round(x_tr[:, j], decimals=8)
        vals, counts = np.unique(col_r, return_counts=True)
        keep = (counts >= leaf_floor) & (counts <= n_rows - leaf_floor)
        if not np.any(keep):
            continue
        vals_k = vals[keep]
        local: List[Tuple[float, float, np.ndarray, Optional[np.ndarray]]] = []
        for v in vals_k:
            raw_tr = (col_r == v).astype(np.float64)
            mu = float(np.mean(raw_tr))
            sd = max(float(np.std(raw_tr)), 1e-9)
            phi_tr = (raw_tr - mu) / sd
            proxy = float(np.dot(phi_tr, alpha_vec)) ** 2 / float(n_rows)
            phi_va = (
                None
                if x_va is None
                else ((np.round(x_va[:, j], decimals=8) == v).astype(np.float64) - mu) / sd
            )
            local.append((proxy, float(v), phi_tr, phi_va))
        local.sort(key=lambda t: -t[0])
        for proxy, v_f, p_tr, p_va in local[: max(1, int(top_per_col))]:
            pre_cols.append((f"eq(c{j}=={v_f:.6g})", p_tr, p_va, proxy))

    if not pre_cols:
        empty_va = None if x_va is None else np.zeros((x_va.shape[0], 0), dtype=np.float64)
        return np.zeros((n_rows, 0), dtype=np.float64), empty_va, []

    phi_mat = np.column_stack([c[1] for c in pre_cols])
    denoms = np.maximum(np.einsum("ij,ij->j", phi_mat, inv_cov @ phi_mat), 1e-15)
    z2_all = ((phi_mat.T @ alpha_vec) ** 2) / denoms
    order = np.argsort(-z2_all)[: max(1, int(max_return))]
    tr_sel = np.column_stack([pre_cols[int(i)][1] for i in order])
    va_sel = (
        None
        if x_va is None
        else np.column_stack([pre_cols[int(i)][2] for i in order if pre_cols[int(i)][2] is not None])
    )
    names_sel = [pre_cols[int(i)][0] for i in order]
    return tr_sel, va_sel, names_sel


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
    mu_all = np.mean(x_tr[:, top_cols], axis=0)
    sd_all = np.maximum(np.std(x_tr[:, top_cols], axis=0), 1e-9)
    z_top_tr = (x_tr[:, top_cols] - mu_all) / sd_all
    z_top_va = None if x_va is None else (x_va[:, top_cols] - mu_all) / sd_all

    tr_list: List[np.ndarray] = []
    va_list: List[np.ndarray] = []
    names: List[str] = []

    for sc in step_cands[:3]:
        gate_tr = (x_tr[:, sc.col_idx] > sc.threshold).astype(np.float64)
        gate_va = None if x_va is None else (x_va[:, sc.col_idx] > sc.threshold).astype(np.float64)
        for pos, k_idx in enumerate(top_cols):
            k = int(k_idx)
            z_tr = z_top_tr[:, pos]
            z_va = None if z_top_va is None else z_top_va[:, pos]
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
    abs_zv = np.abs(zv_tr)
    sq_zv = zv_tr * zv_tr

    target_dir = np.asarray(alpha, dtype=np.float64).reshape(-1)
    t_norm = max(float(np.linalg.norm(target_dir)), 1e-12)
    t_unit = target_dir / t_norm

    th0, th1 = -0.5, -1.5  # log-rates keep denominator >= 1.0 everywhere
    damp = float(damping)

    for _ in range(max(1, int(n_steps))):
        ea = math.exp(min(4.0, max(-8.0, th0)))
        eb = math.exp(min(4.0, max(-8.0, th1)))
        den = 1.0 + ea * abs_zv + eb * sq_zv
        raw = zu_tr / den
        rmu = float(np.mean(raw))
        rsd = max(float(np.std(raw)), 1e-9)
        phi = (raw - rmu) / rsd
        sign = 1.0 if float(np.dot(phi, t_unit)) >= 0.0 else -1.0
        err = sign * t_unit - phi

        j0 = -raw * (ea * abs_zv) / den
        j1 = -raw * (eb * sq_zv) / den
        j0 = (j0 - float(np.mean(j0))) / rsd
        j1 = (j1 - float(np.mean(j1))) / rsd

        a00 = float(np.dot(j0, j0)) + damp
        a01 = float(np.dot(j0, j1))
        a11 = float(np.dot(j1, j1)) + damp
        b0 = float(np.dot(j0, err))
        b1 = float(np.dot(j1, err))
        det = max(a00 * a11 - a01 * a01, 1e-18)
        s0 = (a11 * b0 - a01 * b1) / det
        s1 = (a00 * b1 - a01 * b0) / det
        th0 = min(4.0, max(-8.0, th0 + 0.5 * s0))
        th1 = min(4.0, max(-8.0, th1 + 0.5 * s1))

    ea_f = math.exp(th0)
    eb_f = math.exp(th1)
    raw_tr = zu_tr / (1.0 + ea_f * abs_zv + eb_f * sq_zv)
    rmu_f, rsd_f = float(np.mean(raw_tr)), max(float(np.std(raw_tr)), 1e-9)
    phi_tr = (raw_tr - rmu_f) / rsd_f
    phi_va = None
    if x_val is not None:
        x_va = np.ascontiguousarray(x_val, dtype=np.float64)
        zu_va = (x_va[:, u_col] - mu_u) / sd_u
        zv_va = (x_va[:, v_col] - mu_v) / sd_v
        raw_va = zu_va / (1.0 + ea_f * np.abs(zv_va) + eb_f * (zv_va * zv_va))
        phi_va = (raw_va - rmu_f) / rsd_f
    return phi_tr, phi_va, f"rat(c{u_col}/(1+{ea_f:.3g}|c{v_col}|+{eb_f:.3g}c{v_col}^2))"


def fit_rational_law_varpro_full(
    x_train: np.ndarray,
    target_vec: np.ndarray,
    u_col: int,
    v_col: int,
    active_bank: Optional[np.ndarray] = None,
    row_weights: Optional[np.ndarray] = None,
    x_val: Optional[np.ndarray] = None,
    active_bank_val: Optional[np.ndarray] = None,
    anchor_train: Optional[np.ndarray] = None,
    anchor_val: Optional[np.ndarray] = None,
    allow_anchor_cancel: bool = False,
    lam_ridge: float = 1e-6,
    n_steps: int = 8,
    damping: float = 1e-2,
) -> Dict[str, object]:
    """Full-bank Golub-Pereyra Variable Projection with optional anchor cancellation (`F10`, `Learner Compiler §6.4`).

    Solves the reduced objective `min_{theta} 0.5 ||W^{1/2}(t - Z(theta) beta*(theta))||_2^2 + 0.5 lam ||P beta*||_2^2`
    where `Z(theta) = [1, anchor_col?, active_bank?, 1/Q, z_u/Q, z_u*z_v/Q]`, `Q(v; theta) = 1 + e^{th0}|z_v| + e^{th1}z_v^2`,
    and computes exact reduced gradients via `dbeta* = A^{-1}[(dZ)^T W r - Z^T W (dZ) beta*]`.
    When `allow_anchor_cancel=True`, `delta_a = clip(beta*_anchor, -1.0, 0.0)` cancels corrupted smooth incumbents.
    """
    x_tr = np.ascontiguousarray(x_train, dtype=np.float64)
    n_rows = x_tr.shape[0]
    t_vec = np.ascontiguousarray(target_vec, dtype=np.float64).reshape(n_rows)
    w_vec = (
        np.ones(n_rows, dtype=np.float64)
        if row_weights is None
        else np.maximum(np.asarray(row_weights, dtype=np.float64).reshape(n_rows), 1e-12)
    )
    mu_u, sd_u = float(np.mean(x_tr[:, u_col])), max(float(np.std(x_tr[:, u_col])), 1e-9)
    mu_v, sd_v = float(np.mean(x_tr[:, v_col])), max(float(np.std(x_tr[:, v_col])), 1e-9)
    zu_tr = (x_tr[:, u_col] - mu_u) / sd_u
    zv_tr = (x_tr[:, v_col] - mu_v) / sd_v
    abs_zv = np.abs(zv_tr)
    sq_zv = zv_tr * zv_tr

    fixed_cols = [np.ones((n_rows, 1), dtype=np.float64)]
    has_anchor = anchor_train is not None and allow_anchor_cancel
    if has_anchor:
        fixed_cols.append(np.ascontiguousarray(anchor_train, dtype=np.float64).reshape(n_rows, 1))
    if active_bank is not None and active_bank.size > 0:
        fixed_cols.append(np.ascontiguousarray(active_bank, dtype=np.float64).reshape(n_rows, -1))
    z_fixed = np.hstack(fixed_cols)
    n_fixed = z_fixed.shape[1]

    def _eval_varpro(th0_v: float, th1_v: float) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        ea = math.exp(min(6.0, max(-10.0, th0_v)))
        eb = math.exp(min(6.0, max(-10.0, th1_v)))
        den = 1.0 + ea * abs_zv + eb * sq_zv
        inv_q = 1.0 / den
        z_rat = np.column_stack([inv_q, zu_tr * inv_q, (zu_tr * zv_tr) * inv_q])
        z_all = np.hstack([z_fixed, z_rat])
        p_tot = z_all.shape[1]
        pen = np.full(p_tot, float(lam_ridge), dtype=np.float64)
        pen[0] = 1e-12  # unpenalized intercept
        zw = z_all * w_vec[:, None]
        a_mat = zw.T @ z_all + np.diag(pen)
        rhs = zw.T @ t_vec
        beta = sla.solve(a_mat, rhs, assume_a="pos", check_finite=False)
        if has_anchor and (beta[1] < -1.0 or beta[1] > 0.0):
            delta_a_c = min(0.0, max(-1.0, float(beta[1])))
            keep = [i for i in range(p_tot) if i != 1]
            t_adj = t_vec - delta_a_c * z_all[:, 1]
            a_sub = a_mat[np.ix_(keep, keep)]
            beta_sub = sla.solve(a_sub, zw[:, keep].T @ t_adj, assume_a="pos", check_finite=False)
            beta = np.zeros(p_tot, dtype=np.float64)
            beta[1] = delta_a_c
            beta[keep] = beta_sub
        res = t_vec - z_all @ beta
        loss = 0.5 * float(np.dot(w_vec * res, res)) + 0.5 * float(np.dot(pen * beta, beta))

        # Exact reduced gradient dL/dtheta = -r^T W (dZ beta) when interior (envelope theorem!),
        # and exact Golub-Pereyra Jacobian J_vp = -(I - Z A^{-1} Z^T W)(dZ beta) for Gauss-Newton!
        d_invq_0 = -(inv_q * inv_q) * (ea * abs_zv)
        d_invq_1 = -(inv_q * inv_q) * (eb * sq_zv)
        dz0_beta = (
            d_invq_0 * beta[n_fixed]
            + (zu_tr * d_invq_0) * beta[n_fixed + 1]
            + (zu_tr * zv_tr * d_invq_0) * beta[n_fixed + 2]
        )
        dz1_beta = (
            d_invq_1 * beta[n_fixed]
            + (zu_tr * d_invq_1) * beta[n_fixed + 1]
            + (zu_tr * zv_tr * d_invq_1) * beta[n_fixed + 2]
        )
        grad = np.array(
            [-float(np.dot(w_vec * res, dz0_beta)), -float(np.dot(w_vec * res, dz1_beta))],
            dtype=np.float64,
        )
        g_cols = np.column_stack([dz0_beta, dz1_beta])
        proj = z_all @ sla.solve(a_mat, zw.T @ g_cols, assume_a="pos", check_finite=False)
        j_vp = g_cols - proj
        return loss, grad, j_vp, beta

    th = np.array([-0.5, -1.5], dtype=np.float64)
    loss_0, grad_0, _, _ = _eval_varpro(float(th[0]), float(th[1]))
    cur_loss = loss_0
    for _ in range(max(1, int(n_steps))):
        cur_loss, grad, j_vp, beta = _eval_varpro(float(th[0]), float(th[1]))
        jt_wj = (j_vp * w_vec[:, None]).T @ j_vp + float(damping) * np.eye(2, dtype=np.float64)
        step = -sla.solve(jt_wj, grad, assume_a="pos", check_finite=False)
        for ls in (1.0, 0.5, 0.25, 0.1):
            cand_th = np.clip(th + ls * step, -10.0, 6.0)
            cand_loss, _, _, _ = _eval_varpro(float(cand_th[0]), float(cand_th[1]))
            if cand_loss <= cur_loss + 1e-12:
                th = cand_th
                cur_loss = cand_loss
                break

    final_loss, final_grad, _, beta_star = _eval_varpro(float(th[0]), float(th[1]))
    ea_f = math.exp(float(th[0]))
    eb_f = math.exp(float(th[1]))
    den_tr = 1.0 + ea_f * abs_zv + eb_f * sq_zv
    inv_q_tr = 1.0 / den_tr
    z_rat_tr = np.column_stack([inv_q_tr, zu_tr * inv_q_tr, (zu_tr * zv_tr) * inv_q_tr])
    pred_tr = np.hstack([z_fixed, z_rat_tr]) @ beta_star
    delta_a = float(beta_star[1]) if has_anchor else 0.0

    pred_va = None
    if x_val is not None:
        x_va = np.ascontiguousarray(x_val, dtype=np.float64)
        n_va = x_va.shape[0]
        zu_va = (x_va[:, u_col] - mu_u) / sd_u
        zv_va = (x_va[:, v_col] - mu_v) / sd_v
        inv_q_va = 1.0 / (1.0 + ea_f * np.abs(zv_va) + eb_f * (zv_va * zv_va))
        v_cols = [np.ones((n_va, 1), dtype=np.float64)]
        if has_anchor:
            assert anchor_val is not None
            v_cols.append(np.ascontiguousarray(anchor_val, dtype=np.float64).reshape(n_va, 1))
        if active_bank is not None and active_bank.size > 0:
            assert active_bank_val is not None
            v_cols.append(np.ascontiguousarray(active_bank_val, dtype=np.float64).reshape(n_va, -1))
        v_cols.append(np.column_stack([inv_q_va, zu_va * inv_q_va, (zu_va * zv_va) * inv_q_va]))
        pred_va = np.hstack(v_cols) @ beta_star

    return {
        "beta_star": beta_star,
        "delta_anchor": delta_a,
        "eval_fn": lambda th0_q, th1_q: _eval_varpro(float(th0_q), float(th1_q))[0],
        "final_grad": final_grad,
        "final_loss": final_loss,
        "initial_grad": grad_0,
        "initial_loss": loss_0,
        "pred_train": pred_tr,
        "pred_val": pred_va,
        "theta": th.copy(),
    }


def _loo_kernel_feature(
    k_tr: np.ndarray,
    k_va: Optional[np.ndarray],
    alpha: np.ndarray,
    tag: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """Build zero-diagonal LOO kernel smoother column `(K_tr - diag(K_tr)) @ alpha`."""
    k_zero = k_tr.copy()
    np.fill_diagonal(k_zero, 0.0)
    raw_tr = k_zero @ alpha
    sd = float(np.std(raw_tr))
    if sd <= 1e-9:
        return None, None, tag
    mu = float(np.mean(raw_tr))
    phi_tr = (raw_tr - mu) / sd
    phi_va = None if k_va is None else ((k_va @ alpha) - mu) / sd
    return phi_tr, phi_va, tag


def build_production_router_dictionary(
    x_train: np.ndarray,
    alpha: np.ndarray,
    inv_cov: np.ndarray,
    x_val: Optional[np.ndarray] = None,
    top_per_col: int = 3,
    max_step_return: int = 12,
    min_leaf: Optional[int] = None,
    max_gated_basis_cols: int = 4,
    max_rational_pairs: int = 6,
    include_b1_b2_kernels: bool = True,
    precomputed: Optional[PrecomputedXContext] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    """Production G1/router dictionary builder (`G0_8` §2 & `N1`).

    Constructs strictly on `(x_train, alpha, inv_cov)` (own residual):
      1. Raw standardized features (`p` cols);
      2. `scan_best_step_features` (`P2` step discontinuities);
      3. `build_gated_smooth_features` (`B4` regime-gated smooth interactions);
      4. `fit_rational_law_vp` (`P1` LM rational laws on residual-ranked `(u, v)` pairs);
      5. `B1/B2` LOO partition & relational Laplace kernel smoothers.
    """
    ctx = precomputed or PrecomputedXContext.from_x(x_train, x_val)
    x_tr = ctx.x_train
    x_va = ctx.x_val
    n_rows, n_cols = x_tr.shape
    alpha_vec = np.asarray(alpha, dtype=np.float64).reshape(n_rows)

    tr_blocks: List[np.ndarray] = []
    va_blocks: List[np.ndarray] = []
    names: List[str] = []

    if n_cols > 0:
        tr_blocks.append(ctx.z_raw_train)
        if x_va is not None and ctx.z_raw_val is not None:
            va_blocks.append(ctx.z_raw_val)
        names.extend([f"raw_c{j}" for j in range(n_cols)])

    steps = scan_best_step_features(
        x_tr,
        alpha_vec,
        inv_cov,
        x_val=x_va,
        min_leaf=min_leaf,
        top_per_col=top_per_col,
        max_return=max_step_return,
        col_orders=ctx.col_orders,
    )
    if steps:
        tr_blocks.append(np.column_stack([s.phi_train for s in steps]))
        if x_va is not None:
            va_blocks.append(np.column_stack([s.phi_val for s in steps if s.phi_val is not None]))
        names.extend([s.name for s in steps])

        gated_tr, gated_va, gated_names = build_gated_smooth_features(
            steps, x_tr, x_val=x_va, max_basis_cols=max_gated_basis_cols
        )
        if gated_tr.shape[1] > 0:
            tr_blocks.append(gated_tr)
            if x_va is not None and gated_va is not None:
                va_blocks.append(gated_va)
            names.extend(gated_names)

    eq_tr, eq_va, eq_names = scan_best_equality_features(
        x_tr, alpha_vec, inv_cov, x_val=x_va, min_leaf=min_leaf
    )
    if eq_tr.shape[1] > 0:
        tr_blocks.append(eq_tr)
        if x_va is not None and eq_va is not None:
            va_blocks.append(eq_va)
        names.extend(eq_names)

    if n_cols >= 2 and int(max_rational_pairs) > 0:
        corrs = np.abs(ctx.z_raw_train.T @ alpha_vec)
        u_rank = np.argsort(-corrs)
        v_rank = np.argsort(-np.var(x_tr, axis=0))
        pairs: List[Tuple[int, int]] = []
        for u_i in u_rank:
            for v_i in v_rank:
                u_c, v_c = int(u_i), int(v_i)
                if u_c != v_c:
                    pairs.append((u_c, v_c))
                if len(pairs) >= int(max_rational_pairs):
                    break
            if len(pairs) >= int(max_rational_pairs):
                break
        rat_tr_cols: List[np.ndarray] = []
        rat_va_cols: List[np.ndarray] = []
        for u_c, v_c in pairs:
            r_tr, r_va, r_name = fit_rational_law_vp(x_tr, alpha_vec, u_col=u_c, v_col=v_c, x_val=x_va)
            rat_tr_cols.append(r_tr)
            if x_va is not None and r_va is not None:
                rat_va_cols.append(r_va)
            names.append(r_name)
        if rat_tr_cols:
            tr_blocks.append(np.column_stack(rat_tr_cols))
            if x_va is not None and rat_va_cols:
                va_blocks.append(np.column_stack(rat_va_cols))

    if include_b1_b2_kernels and ctx.b1_b2_spec_train.shape[1] > 0:
        tr_blocks.append(ctx.b1_b2_spec_train)
        if x_va is not None and ctx.b1_b2_spec_val is not None:
            va_blocks.append(ctx.b1_b2_spec_val)
        names.extend(ctx.b1_b2_spec_names)

    if not tr_blocks:
        empty_va = None if x_va is None else np.zeros((x_va.shape[0], 0), dtype=np.float64)
        return np.zeros((n_rows, 0), dtype=np.float64), empty_va, []

    bank_tr = np.column_stack(tr_blocks)
    bank_va = None if x_va is None else np.column_stack(va_blocks)
    return bank_tr, bank_va, names
