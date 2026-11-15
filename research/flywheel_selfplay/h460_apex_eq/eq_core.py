"""Joint Residual-Covariance Equilibrium Core (`eq_core.py`).

Unifies the six `tips/core` formulations under one exact `float64` state:
  Sigma = tau_H^2 H H^T + sum_g w_g Phi_g Phi_g^T + sum_c sigma_c^2 K_c + diag(sigma_n^2 + d_robust)

Incorporates all 2026-11-15 peer-review hardening (`cl_fwriter` `04:34Z` red-team + `cl_mtc` + `cl_h458`):
- Scale-covariant base noise variance `sigma_n2` initialized from residual scale (`Finding #8`).
- Full-rank scan over all ranked candidates without truncating at top-24 (`Finding #11`).
- Both fixed-pool (`calibrate_placebo_c0`) and nested-discovery (`calibrate_nested_discovery_placebo_c0`)
  Romano-Wolf permutation/Gaussian placebos (`Finding #2`).
- Unregularized QR span-novelty guard (`Compiler §18.7`), closed-form LOO/LFO, Huber/IRLS
  `d_robust`, single-row influence cap, and SEARCH target-range clamp helper.
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import linalg as sla


@dataclasses.dataclass(frozen=True)
class EquilibriumConfig:
    """Hyperparameters and safety guards for `EquilibriumState`."""

    sigma_n2: float = 0.05
    auto_scale_sigma_n2: bool = True
    sigma_n2_var_frac: float = 0.10
    default_prior_w: float = 1.0
    max_active_groups: int = 12
    span_novelty_eps: float = 1e-4
    foster_george_c0: float = 2.5
    robust_irls: bool = True
    huber_k: float = 4.0
    irls_passes: int = 1
    min_effective_support: float = 4.0
    max_single_row_score_share: float = 0.55
    winsor_alpha_k: float = 4.5
    range_clamp_iqr_mult: float = 1.5
    fold_floor_rel_tol: float = 0.05
    enforce_hard_fold_floor: bool = True
    min_improved_folds: int = 0
    min_rel_rmse_gain: float = 0.0
    damping_grid: Tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)
    loo_sigma_n2_grid: Optional[Tuple[float, ...]] = None
    affine_center_incumbent: bool = False
    jitter: float = 1e-10

    @classmethod
    def for_regime(cls, regime: str) -> "EquilibriumConfig":
        """Regime-aware preset (`ARENA §6.1` `D1, D3, D4`): disables tail-crushing clamps on exact/tail laws."""
        reg = str(regime).strip().upper()
        rho_grid = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 0.2, 0.5, 1.0)
        if reg in ("DETERMINISTIC", "EXACT"):
            return cls(
                sigma_n2=1e-8,
                auto_scale_sigma_n2=True,
                sigma_n2_var_frac=1e-6,
                loo_sigma_n2_grid=rho_grid,
                affine_center_incumbent=True,
                robust_irls=False,
                range_clamp_iqr_mult=1e9,
                enforce_hard_fold_floor=True,
            )
        if reg == "TAIL":
            return cls(
                sigma_n2=1e-4,
                auto_scale_sigma_n2=True,
                sigma_n2_var_frac=0.02,
                loo_sigma_n2_grid=rho_grid,
                affine_center_incumbent=True,
                robust_irls=False,
                range_clamp_iqr_mult=1e9,
                enforce_hard_fold_floor=True,
            )
        return cls(
            auto_scale_sigma_n2=True,
            loo_sigma_n2_grid=rho_grid,
            affine_center_incumbent=True,
            robust_irls=True,
            enforce_hard_fold_floor=True,
        )


@dataclasses.dataclass
class CandidateGroup:
    """Active feature block `Phi_g` (`n x p_g`) on training rows plus optional query features."""

    name: str
    phi_train: np.ndarray
    phi_val: Optional[np.ndarray]
    weight: float
    family_tag: str = "feature"


@dataclasses.dataclass
class KernelBlock:
    """Active PSD Gram block `K_c` (`n x n`) on training rows plus cross-Gram `K_val_train`."""

    name: str
    k_train: np.ndarray
    k_val_train: Optional[np.ndarray]
    variance: float
    family_tag: str = "kernel"


@dataclasses.dataclass(frozen=True)
class AdmissionDecision:
    """Outcome of screening + span-novelty + Foster-George threshold evaluation."""

    action: str
    candidate_index: int
    candidate_name: str
    raw_score_z2: float
    robust_score_z2: float
    delta_log_evidence: float
    optimal_weight: float
    span_innovation: float
    effective_support: float
    max_row_share: float
    threshold_z2: float
    vetoed_indices: Tuple[int, ...] = ()
    replaced_group_index: Optional[int] = None
    frozen_gaussian_evidence_delta: float = 0.0
    robust_acquisition_gain: float = 0.0


def foster_george_threshold(n_candidates: int, c0: float = 2.5) -> float:
    """Multiplicity-calibrated score threshold `tau(N) = 2 ln(max(N, 1)) + c0`."""
    return 2.0 * math.log(max(int(n_candidates), 1)) + float(c0)


def compute_search_clamp_bounds(y_train: np.ndarray, iqr_mult: float = 1.5) -> Tuple[float, float]:
    """Compute `[y_min - kappa*IQR, y_max + kappa*IQR]` strictly from training targets."""
    y_tr = np.asarray(y_train, dtype=np.float64).reshape(-1)
    q25, q75 = np.percentile(y_tr, [25.0, 75.0])
    iqr = float(q75 - q25)
    pad = float(iqr_mult) * (iqr if iqr > 1e-9 else max(float(np.std(y_tr)), 1e-6))
    return float(np.min(y_tr)) - pad, float(np.max(y_tr)) + pad


def apply_search_range_clamp(preds: np.ndarray, y_train: np.ndarray, iqr_mult: float = 1.5) -> np.ndarray:
    """Clamp predictions to training target support `[lo, hi]` (`cl_fwriter` Finding #4 / `N4`)."""
    lo, hi = compute_search_clamp_bounds(y_train, iqr_mult=iqr_mult)
    return np.clip(np.asarray(preds, dtype=np.float64), lo, hi)


def apply_mad3_clamp(preds: np.ndarray, y_train: np.ndarray, k_mad: float = 3.0) -> np.ndarray:
    """Router-compatible symmetric MAD output clamp around training target median (`N4`)."""
    y_tr = np.asarray(y_train, dtype=np.float64).reshape(-1)
    med = float(np.median(y_tr))
    mad = 1.4826 * float(np.median(np.abs(y_tr - med)))
    scale = mad if mad > 1e-9 else max(float(np.std(y_tr)), 1e-6)
    half = float(k_mad) * max(scale, float(np.std(y_tr)))
    return np.clip(np.asarray(preds, dtype=np.float64), med - half, med + half)


def _conformal_order_index(n_b: int, alpha_f: float) -> int:
    """Exact Romano-Wolf / Phipson-Smyth order index ensuring `(n_b - idx) / (n_b + 1) <= alpha_f`."""
    return min(n_b - 1, max(0, int(math.ceil((1.0 - alpha_f) * (n_b + 1))) - 1))


def _winsorize_vector(vec: np.ndarray, k_mad: float) -> np.ndarray:
    med = float(np.median(vec))
    mad = 1.4826 * float(np.median(np.abs(vec - med)))
    if mad <= 1e-12:
        std = float(np.std(vec))
        if std <= 1e-12:
            return vec.copy()
        mad = std
    bound = float(k_mad) * mad
    return np.clip(vec, med - bound, med + bound)


def compute_effective_support(col: np.ndarray) -> float:
    """Centered fourth-moment effective row count `||u||_2^4 / ||u||_4^4` in `[1, n]`."""
    centered = np.asarray(col, dtype=np.float64).reshape(-1) - float(np.mean(col))
    l2_sq = float(np.dot(centered, centered))
    if l2_sq <= 1e-18:
        return 0.0
    l4_sum = float(np.sum((centered * centered) ** 2))
    return (l2_sq * l2_sq) / max(l4_sum, 1e-30)


def _max_unvetoed_pool_z2(
    state: "EquilibriumState",
    pool: np.ndarray,
    alpha_vec: np.ndarray,
) -> float:
    """Exact `ADD`-eligible score `z^2` matching `screen_dictionary` rank + spike + QR guards."""
    if pool.size == 0 or pool.shape[1] == 0:
        return 0.0
    n_rows = state.n_rows
    c_pool = state.inv_cov @ pool
    m_vec = np.maximum(np.einsum("ij,ij->j", pool, c_pool), 1e-15)
    chi2_raw = max(float(np.dot(alpha_vec, state.cov @ alpha_vec)) / float(n_rows), 1e-15)
    g_raw = pool.T @ alpha_vec
    z2_raw = (g_raw * g_raw) / (m_vec * chi2_raw)

    if state.cfg.robust_irls:
        alpha_rob = _winsorize_vector(alpha_vec, state.cfg.winsor_alpha_k)
        chi2_rob = max(float(np.dot(alpha_rob, state.cov @ alpha_rob)) / float(n_rows), 1e-15)
        g_rob = pool.T @ alpha_rob
        z2_eff = np.minimum(z2_raw, (g_rob * g_rob) / (m_vec * chi2_rob))
    else:
        z2_eff = z2_raw

    order = np.argsort(-z2_eff)
    for idx_i in order:
        idx = int(idx_i)
        col = pool[:, idx]
        if state.cfg.robust_irls:
            n_eff = compute_effective_support(col)
            contrib_abs = np.abs((col - float(np.mean(col))) * alpha_vec)
            max_share = float(np.max(contrib_abs)) / max(float(np.sum(contrib_abs)), 1e-15)
            if n_eff < state.cfg.min_effective_support or max_share > state.cfg.max_single_row_score_share:
                continue
        innov = state.span_innovation(col)
        if innov < state.cfg.span_novelty_eps:
            if float(z2_eff[idx]) > 1.0:
                return 0.0
            continue
        return float(z2_eff[idx])
    return 0.0


def calibrate_placebo_c0(
    state: "EquilibriumState",
    x_pool: np.ndarray,
    n_trials: int = 200,
    target_false_admissions_per_fit: float = 0.05,
    seed: int = 460001,
    legacy_38ff3faf_fudge: float = 0.0,
) -> float:
    """Calibrate `c0` in `tau(N) = 2 ln N + c0` for a genuinely FIXED, label-free OOF pool.

    WARNING (`G0_8`): Do NOT use this function on residual-adaptive banks; use
    `calibrate_nested_discovery_placebo_c0`, which rebuilds the bank on every null draw.
    """
    rng = np.random.default_rng(seed)
    pool = np.ascontiguousarray(x_pool, dtype=np.float64)
    n_rows, n_cands = pool.shape
    if n_cands == 0:
        return float(state.cfg.foster_george_c0)
    base_2ln = 2.0 * math.log(max(n_cands, 1))
    alpha_f = max(1e-4, min(0.5, float(target_false_admissions_per_fit)))
    min_b = max(39, int(math.ceil(1.0 / alpha_f)) - 1)
    n_b = max(min_b, int(n_trials))
    max_perm = np.zeros(n_b, dtype=np.float64)
    max_gauss = np.zeros(n_b, dtype=np.float64)

    for t in range(n_b):
        perm = rng.permutation(n_rows)
        r_perm = state.residual[perm]
        max_perm[t] = _max_unvetoed_pool_z2(state, pool, state.inv_cov @ r_perm)

        xi = rng.normal(size=n_rows)
        alpha_g = sla.solve_triangular(state.chol_lower.T, xi, lower=False, check_finite=False)
        max_gauss[t] = _max_unvetoed_pool_z2(state, pool, alpha_g)

    rank_idx = _conformal_order_index(n_b, alpha_f)
    q_val = max(float(np.sort(max_perm)[rank_idx]), float(np.sort(max_gauss)[rank_idx])) + float(
        legacy_38ff3faf_fudge
    )
    return max(float(state.cfg.foster_george_c0), q_val - base_2ln + 1e-9)


def calibrate_nested_discovery_placebo_c0(
    state: "EquilibriumState",
    discovery_fn: Callable[["EquilibriumState"], np.ndarray],
    n_trials: int = 49,
    target_false_admissions_per_fit: float = 0.025,
    seed: int = 460009,
) -> float:
    """Nested placebo calibration (`cl_fwriter` `G0_8` & Finding #2): rebuilds state + bank per draw.

    On each null draw `t`, constructs a fresh `EquilibriumState` on the permuted residual (when
    no non-constant groups/kernels are active yet, preserving exact finite-sample row
    exchangeability) or covariance-matched Gaussian draw `r = L xi` (when active groups/kernels
    exist), runs IRLS outlier noise estimation on that null draw, rebuilds the ENTIRE candidate
    bank `discovery_fn(null_state)`, standardizes it, and records `max_z2 - 2 ln(N_draw)`.
    """
    rng = np.random.default_rng(seed)
    n_rows = state.n_rows
    alpha_f = max(1e-4, min(0.5, float(target_false_admissions_per_fit)))
    min_b = max(39, int(math.ceil(1.0 / alpha_f)) - 1)
    n_b = max(min_b, int(n_trials))
    has_nonexchangeable_cov = bool(state.groups or state.kernels)
    excess = np.zeros(n_b, dtype=np.float64)

    for t in range(n_b):
        if not has_nonexchangeable_cov:
            perm = rng.permutation(n_rows)
            r_draw = state.residual[perm]
        else:
            xi = rng.normal(size=n_rows)
            r_draw = state.chol_lower @ xi

        st_draw = EquilibriumState(
            y_train=state.inc_train + r_draw,
            incumbent_train=state.inc_train,
            anchor_h_train=state.anchor_h_train,
            tau_h2=state.tau_h2,
            cfg=state.cfg,
        )
        for grp in state.groups:
            st_draw.groups.append(
                CandidateGroup(grp.name, grp.phi_train, None, grp.weight, grp.family_tag)
            )
        for ker in state.kernels:
            st_draw.kernels.append(
                KernelBlock(ker.name, ker.k_train, None, ker.variance, ker.family_tag)
            )
        if state.groups or state.kernels:
            st_draw.refactorize()
        st_draw.update_robust_noise_irls()

        raw_pool = np.ascontiguousarray(discovery_fn(st_draw), dtype=np.float64)
        if raw_pool.ndim == 2 and raw_pool.shape[1] > 0:
            mu = np.mean(raw_pool, axis=0)
            sd = np.maximum(np.std(raw_pool, axis=0), 1e-9)
            z_pool = (raw_pool - mu) / sd
            z2_top = _max_unvetoed_pool_z2(st_draw, z_pool, st_draw.alpha)
            excess[t] = z2_top - 2.0 * math.log(max(z_pool.shape[1], 1))

    rank_idx = _conformal_order_index(n_b, alpha_f)
    c0_emp = float(np.sort(excess)[rank_idx]) + 1e-9
    return max(float(state.cfg.foster_george_c0), c0_emp)


def exact_ridge_duplicate_gain(
    x_active: Optional[np.ndarray],
    x_new: np.ndarray,
    y_vec: np.ndarray,
    lam: float,
) -> float:
    """Exact unhalved ridge objective reduction `2*(J(X) - J([X, x_new]))` (`Compiler §18.7`)."""
    n_rows = y_vec.shape[0]
    cov_u = float(lam) * np.eye(n_rows, dtype=np.float64)
    if x_active is not None and x_active.size > 0:
        xa = np.ascontiguousarray(x_active, dtype=np.float64).reshape(n_rows, -1)
        cov_u = cov_u + (xa @ xa.T)
    chol = sla.cho_factor(cov_u, lower=True, check_finite=False)
    xn = np.ascontiguousarray(x_new, dtype=np.float64).reshape(n_rows)
    cu_y = sla.cho_solve(chol, y_vec, check_finite=False)
    cu_x = sla.cho_solve(chol, xn, check_finite=False)
    num = float(np.dot(xn, cu_y)) ** 2
    den = 1.0 + float(np.dot(xn, cu_x))
    return float(lam) * num / max(den, 1e-15)


class EquilibriumState:
    """Exact `float64` covariance equilibrium over feature groups, PSD kernels, and robust noise."""

    def __init__(
        self,
        y_train: np.ndarray,
        incumbent_train: Optional[np.ndarray] = None,
        incumbent_val: Optional[np.ndarray] = None,
        anchor_h_train: Optional[np.ndarray] = None,
        anchor_h_val: Optional[np.ndarray] = None,
        tau_h2: float = 0.0,
        cfg: Optional[EquilibriumConfig] = None,
    ) -> None:
        self.cfg = cfg or EquilibriumConfig()
        self.y_train = np.ascontiguousarray(y_train, dtype=np.float64).reshape(-1)
        self.n_rows = int(self.y_train.shape[0])
        self._affine_offset = (
            float(np.mean(self.y_train))
            if (incumbent_train is None and self.cfg.affine_center_incumbent)
            else 0.0
        )
        self.inc_train = (
            np.full(self.n_rows, self._affine_offset, dtype=np.float64)
            if incumbent_train is None
            else np.ascontiguousarray(incumbent_train, dtype=np.float64).reshape(-1)
        )
        self.inc_val = (
            None
            if incumbent_val is None
            else np.ascontiguousarray(incumbent_val, dtype=np.float64).reshape(-1)
        )
        self.residual = self.y_train - self.inc_train
        res_var = float(np.var(self.residual))
        self._scale_jitter = (
            self.cfg.jitter * res_var
            if (self.cfg.auto_scale_sigma_n2 and res_var > 0.0)
            else self.cfg.jitter
        )
        self.base_sigma_n2 = (
            max(self.cfg.sigma_n2_var_frac * res_var, self._scale_jitter)
            if self.cfg.auto_scale_sigma_n2 and res_var > 0.0
            else max(float(self.cfg.sigma_n2), self.cfg.jitter)
        )
        self.anchor_h_train = (
            None
            if anchor_h_train is None
            else np.ascontiguousarray(anchor_h_train, dtype=np.float64).reshape(self.n_rows, -1)
        )
        self.anchor_h_val = (
            None
            if anchor_h_val is None
            else np.ascontiguousarray(anchor_h_val, dtype=np.float64)
        )
        self.tau_h2 = float(tau_h2)
        self.groups: List[CandidateGroup] = []
        self.kernels: List[KernelBlock] = []
        self.robust_d = np.zeros(self.n_rows, dtype=np.float64)

        self.signal_cov = np.zeros((self.n_rows, self.n_rows), dtype=np.float64)
        self.cov = np.zeros((self.n_rows, self.n_rows), dtype=np.float64)
        self._chol_lower: Optional[np.ndarray] = None
        self._log_det: float = 0.0
        self.inv_cov = np.zeros((self.n_rows, self.n_rows), dtype=np.float64)
        self.alpha = np.zeros(self.n_rows, dtype=np.float64)
        self.qr_basis = np.zeros((self.n_rows, 0), dtype=np.float64)
        self.refactorize()

    @property
    def chol_lower(self) -> np.ndarray:
        if self._chol_lower is None:
            self._chol_lower = sla.cholesky(self.cov, lower=True, check_finite=False)
        return self._chol_lower

    def _noise_diag(self) -> np.ndarray:
        return np.maximum(self.base_sigma_n2 + self.robust_d, self._scale_jitter)

    def _rebuild_qr_basis(self) -> None:
        cols: List[np.ndarray] = []
        if self.anchor_h_train is not None and self.tau_h2 > 0.0:
            cols.append(self.anchor_h_train)
        for grp in self.groups:
            if grp.weight > 0.0:
                cols.append(grp.phi_train)
        if not cols:
            self.qr_basis = np.zeros((self.n_rows, 0), dtype=np.float64)
            return
        u_mat = np.hstack(cols)
        inv_sqrt_d = 1.0 / np.sqrt(self._noise_diag())[:, None]
        weighted = inv_sqrt_d * u_mat
        if weighted.shape[1] == 1:
            norm = float(np.linalg.norm(weighted[:, 0]))
            if norm > 1e-10:
                self.qr_basis = np.ascontiguousarray(weighted / norm, dtype=np.float64)
            else:
                self.qr_basis = np.zeros((self.n_rows, 0), dtype=np.float64)
            return
        q_mat, r_mat, _ = sla.qr(weighted, mode="economic", pivoting=True)
        if r_mat.size == 0:
            self.qr_basis = np.zeros((self.n_rows, 0), dtype=np.float64)
            return
        diag_abs = np.abs(np.diag(r_mat))
        tol = max(1e-10, 1e-10 * float(diag_abs[0]))
        rank = int(np.sum(diag_abs > tol))
        self.qr_basis = np.ascontiguousarray(q_mat[:, :rank], dtype=np.float64)

    def refactorize(self) -> None:
        """Exact refactorization of `Sigma` + `inv_cov` + `alpha` + QR basis (`O(n^2)` rank<=1, `O(n^3)` general)."""
        d_diag = self._noise_diag()
        has_active_groups = any(grp.weight > 0.0 for grp in self.groups)
        has_active_kernels = any(ker.variance > 0.0 for ker in self.kernels)
        anchor_rank = 0 if (self.anchor_h_train is None or self.tau_h2 <= 0.0) else self.anchor_h_train.shape[1]

        if not has_active_groups and not has_active_kernels and anchor_rank <= 1:
            inv_d = 1.0 / d_diag
            if anchor_rank == 0:
                self.signal_cov = np.zeros((self.n_rows, self.n_rows), dtype=np.float64)
                self.cov = np.diag(d_diag)
                self._chol_lower = np.diag(np.sqrt(d_diag))
                self._log_det = float(np.sum(np.log(d_diag)))
                self.inv_cov = np.diag(inv_d)
            else:
                assert self.anchor_h_train is not None
                h_vec = self.anchor_h_train[:, 0]
                self.signal_cov = self.tau_h2 * np.outer(h_vec, h_vec)
                self.cov = self.signal_cov + np.diag(d_diag)
                self._chol_lower = None
                inv_d_h = inv_d * h_vec
                cap = 1.0 + self.tau_h2 * float(np.dot(h_vec, inv_d_h))
                self._log_det = float(np.sum(np.log(d_diag))) + math.log(cap)
                self.inv_cov = np.diag(inv_d) - (self.tau_h2 / cap) * np.outer(inv_d_h, inv_d_h)
            self.alpha = self.inv_cov @ self.residual
            self._rebuild_qr_basis()
            return

        sig = np.zeros((self.n_rows, self.n_rows), dtype=np.float64)
        if self.anchor_h_train is not None and self.tau_h2 > 0.0:
            sig += self.tau_h2 * (self.anchor_h_train @ self.anchor_h_train.T)
        for grp in self.groups:
            if grp.weight > 0.0:
                sig += float(grp.weight) * (grp.phi_train @ grp.phi_train.T)
        for ker in self.kernels:
            if ker.variance > 0.0:
                sig += float(ker.variance) * ker.k_train
        self.signal_cov = sig
        self.cov = sig + np.diag(d_diag)
        chol = sla.cholesky(self.cov, lower=True, check_finite=False)
        self._chol_lower = chol
        self._log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
        eye = np.eye(self.n_rows, dtype=np.float64)
        self.inv_cov = sla.cho_solve((chol, True), eye, check_finite=False)
        self.alpha = sla.cho_solve((chol, True), self.residual, check_finite=False)
        self._rebuild_qr_basis()

    def update_robust_noise_irls(self) -> np.ndarray:
        """Inflate diagonal row noise `d_robust` on extreme LOO outlier rows (`cl_h458` defense)."""
        if not self.cfg.robust_irls:
            self.robust_d = np.zeros(self.n_rows, dtype=np.float64)
            return self.robust_d
        for _ in range(max(1, int(self.cfg.irls_passes))):
            e_loo = self.loo_residuals()
            med = float(np.median(e_loo))
            mad = 1.4826 * float(np.median(np.abs(e_loo - med)))
            scale = mad if mad > 1e-9 else max(float(np.std(e_loo)), 1e-9)
            cutoff = float(self.cfg.huber_k) * scale
            ratio = np.abs(e_loo - med) / cutoff
            infl = np.where(ratio > 1.0, self.base_sigma_n2 * (ratio * ratio - 1.0), 0.0)
            self.robust_d = np.minimum(infl, 1e6 * self.base_sigma_n2)
            self.refactorize()
        return self.robust_d.copy()

    def marginal_log_evidence(self) -> float:
        """Exact `-0.5 * log|Sigma| - 0.5 * r^T Sigma^{-1} r`."""
        quad = float(np.dot(self.residual, self.alpha))
        return -0.5 * self._log_det - 0.5 * quad

    def state_digest(self) -> str:
        """Deterministic 64-hex SHA-256 over full covariance, active bases/kernels, `robust_d`, and `alpha` (`F06`)."""
        h = hashlib.sha256()
        h.update(f"N:{self.n_rows};S2:{self.base_sigma_n2:.16e};TH2:{self.tau_h2:.16e};".encode("utf-8"))
        h.update(np.ascontiguousarray(self.robust_d, dtype=np.float64).tobytes())
        if self.anchor_h_train is not None:
            ah = np.ascontiguousarray(self.anchor_h_train, dtype=np.float64)
            h.update(f"AH:{ah.shape}:".encode("utf-8"))
            h.update(hashlib.sha256(ah.tobytes()).digest())
        for grp in self.groups:
            p_tr = np.ascontiguousarray(grp.phi_train, dtype=np.float64)
            h.update(f"G:{grp.name}:{grp.family_tag}:{grp.weight:.16e}:{p_tr.shape}:".encode("utf-8"))
            h.update(hashlib.sha256(p_tr.tobytes()).digest())
        for ker in self.kernels:
            k_tr = np.ascontiguousarray(ker.k_train, dtype=np.float64)
            h.update(f"K:{ker.name}:{ker.family_tag}:{ker.variance:.16e}:{k_tr.shape}:".encode("utf-8"))
            h.update(hashlib.sha256(k_tr.tobytes()).digest())
        h.update(hashlib.sha256(np.ascontiguousarray(self.cov, dtype=np.float64).tobytes()).digest())
        h.update(np.ascontiguousarray(self.alpha, dtype=np.float64).tobytes())
        return h.hexdigest()

    def select_loo_noise_scale(
        self, rho_grid: Optional[Sequence[float]] = None
    ) -> float:
        """Select scale-equivariant noise fraction `sigma_n2 = rho * Var(r)` via closed-form LOO (`ARENA D1`)."""
        res_var = float(np.var(self.residual))
        if res_var <= 0.0 or (not self.groups and not self.kernels):
            return self.base_sigma_n2
        grid = (
            rho_grid
            if rho_grid is not None
            else (self.cfg.loo_sigma_n2_grid or (1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 5e-2, 0.2, 0.5, 1.0))
        )
        # Scale unit-prior group/kernel variances proportionally to Var(r) if not yet scaled
        orig_s2 = self.base_sigma_n2
        best_s2 = orig_s2
        best_loo_mse = float(np.mean(self.loo_residuals() ** 2))
        for rho in grid:
            cand_s2 = max(float(rho) * res_var, self._scale_jitter)
            self.base_sigma_n2 = cand_s2
            self.refactorize()
            loo_mse = float(np.mean(self.loo_residuals() ** 2))
            if loo_mse < best_loo_mse:
                best_loo_mse = loo_mse
                best_s2 = cand_s2
        self.base_sigma_n2 = best_s2
        self.refactorize()
        return self.base_sigma_n2

    def loo_residuals(self) -> np.ndarray:
        """Exact closed-form Leave-One-Out residual `e_i = alpha_i / C_ii` (primal hat-matrix when kernel-free)."""
        has_active_kernels = any(ker.variance > 0.0 for ker in self.kernels)
        if not has_active_kernels:
            u_tr_parts: List[np.ndarray] = []
            inv_w_parts: List[np.ndarray] = []
            if self.anchor_h_train is not None and self.tau_h2 > 0.0:
                u_tr_parts.append(self.anchor_h_train)
                inv_w_parts.append(np.full(self.anchor_h_train.shape[1], 1.0 / float(self.tau_h2), dtype=np.float64))
            for grp in self.groups:
                if grp.weight > 0.0:
                    u_tr_parts.append(grp.phi_train)
                    inv_w_parts.append(np.full(grp.phi_train.shape[1], 1.0 / float(grp.weight), dtype=np.float64))
            if not u_tr_parts:
                return self.residual.copy()
            u_tr = np.hstack(u_tr_parts)
            inv_w = np.concatenate(inv_w_parts)
            inv_d = 1.0 / self._noise_diag()
            ut_dinv = u_tr.T * inv_d[None, :]
            gram_p = ut_dinv @ u_tr + np.diag(inv_w)
            chol_p = sla.cho_factor(gram_p, lower=True, check_finite=False)
            beta_p = sla.cho_solve(chol_p, ut_dinv @ self.residual, check_finite=False)
            a_inv_ut = sla.cho_solve(chol_p, u_tr.T, check_finite=False)
            hat_diag = inv_d * np.sum(u_tr.T * a_inv_ut, axis=0)
            return (self.residual - u_tr @ beta_p) / np.maximum(1.0 - hat_diag, 1e-12)

        c_floor = 1e-30 / max(self.base_sigma_n2, 1e-30)
        c_diag = np.maximum(np.diag(self.inv_cov), c_floor)
        return self.alpha / c_diag

    def lfo_residuals(self, folds: Sequence[np.ndarray]) -> np.ndarray:
        """Exact closed-form K-fold Leave-Fold-Out residuals `e_B = (C_BB)^{-1} alpha_B`."""
        out = np.zeros(self.n_rows, dtype=np.float64)
        for fold_idx in folds:
            idx = np.asarray(fold_idx, dtype=np.int64)
            if idx.size == 0:
                continue
            c_bb = self.inv_cov[np.ix_(idx, idx)]
            chol_bb = sla.cho_factor(c_bb, lower=True, check_finite=False)
            out[idx] = sla.cho_solve(chol_bb, self.alpha[idx], check_finite=False)
        return out

    def span_innovation(self, phi: np.ndarray) -> float:
        """Unregularized QR span-novelty ratio `||(I - Q Q^T) D^{-1/2} phi||_2^2 / ||D^{-1/2} phi||_2^2`."""
        mat = np.ascontiguousarray(phi, dtype=np.float64).reshape(self.n_rows, -1)
        inv_sqrt_d = 1.0 / np.sqrt(self._noise_diag())[:, None]
        w_phi = inv_sqrt_d * mat
        denom = float(np.sum(w_phi * w_phi))
        if denom <= 1e-18:
            return 0.0
        if self.qr_basis.shape[1] == 0:
            return 1.0
        proj = self.qr_basis @ (self.qr_basis.T @ w_phi)
        orth = w_phi - proj
        return max(0.0, min(1.0, float(np.sum(orth * orth)) / denom))

    def exact_block_delta_log_evidence(self, phi: np.ndarray, weight: float) -> float:
        """Exact marginal log-evidence change `0.5 * q^T S^{-1} q - 0.5 * log|I + w Phi^T C Phi|`."""
        if weight <= 0.0:
            return 0.0
        mat = np.ascontiguousarray(phi, dtype=np.float64).reshape(self.n_rows, -1)
        p_dim = mat.shape[1]
        u_mat = self.inv_cov @ mat
        m_mat = mat.T @ u_mat
        q_vec = mat.T @ self.alpha
        i_plus_wm = np.eye(p_dim, dtype=np.float64) + float(weight) * m_mat
        chol_iwm = sla.cholesky(i_plus_wm, lower=True, check_finite=False)
        log_det = 2.0 * float(np.sum(np.log(np.diag(chol_iwm))))
        sol = sla.cho_solve((chol_iwm, True), q_vec, check_finite=False)
        quad = float(weight) * float(np.dot(q_vec, sol))
        return 0.5 * quad - 0.5 * log_det

    def optimal_rank1_weight_and_gain(
        self, col: np.ndarray, use_robust_alpha: bool = False
    ) -> Tuple[float, float, float]:
        """Return `(z2, w_star, max_delta_log_evidence)` for a 1D column under Tipping's exact step."""
        vec = np.ascontiguousarray(col, dtype=np.float64).reshape(self.n_rows)
        c_vec = self.inv_cov @ vec
        m_val = float(np.dot(vec, c_vec))
        if m_val <= 1e-15:
            return 0.0, 0.0, 0.0
        alpha_use = _winsorize_vector(self.alpha, self.cfg.winsor_alpha_k) if use_robust_alpha else self.alpha
        g_val = float(np.dot(vec, alpha_use))
        z2 = (g_val * g_val) / m_val
        if z2 <= 1.0:
            return z2, 0.0, 0.0
        w_star = (z2 - 1.0) / m_val
        gain = 0.5 * (z2 - 1.0 - math.log(z2))
        return z2, w_star, gain

    def screen_dictionary(
        self,
        cand_matrix: np.ndarray,
        cand_names: Sequence[str],
        cand_val_matrix: Optional[np.ndarray] = None,
        c0_override: Optional[float] = None,
    ) -> AdmissionDecision:
        """Vectorized DGEMM screen across ALL ranked candidates (`Finding #11`) + QR + spike guards."""
        pool = np.ascontiguousarray(cand_matrix, dtype=np.float64).reshape(self.n_rows, -1)
        n_cands = pool.shape[1]
        c0 = float(self.cfg.foster_george_c0 if c0_override is None else c0_override)
        tau_z2 = foster_george_threshold(n_cands, c0=c0)
        if n_cands == 0:
            return AdmissionDecision("NONE", -1, "", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, tau_z2)

        c_pool = self.inv_cov @ pool
        m_vec = np.maximum(np.einsum("ij,ij->j", pool, c_pool), 1e-15)
        chi2_scale = max(float(np.dot(self.residual, self.alpha)) / float(self.n_rows), 1e-15)
        g_raw = pool.T @ self.alpha
        z2_raw_all = (g_raw * g_raw) / (m_vec * chi2_scale)

        alpha_rob = _winsorize_vector(self.alpha, self.cfg.winsor_alpha_k)
        chi2_rob = max(float(np.dot(alpha_rob, self.cov @ alpha_rob)) / float(self.n_rows), 1e-15)
        g_rob = pool.T @ alpha_rob
        z2_rob_all = (g_rob * g_rob) / (m_vec * chi2_rob)

        score_rank = np.argsort(-np.minimum(z2_raw_all, z2_rob_all) if self.cfg.robust_irls else -z2_raw_all)
        vetoed: List[int] = []
        best_decision: Optional[AdmissionDecision] = None

        for idx_i in score_rank:
            idx = int(idx_i)
            col = pool[:, idx]
            n_eff = compute_effective_support(col)
            contrib_abs = np.abs((col - float(np.mean(col))) * self.alpha)
            max_share = float(np.max(contrib_abs)) / max(float(np.sum(contrib_abs)), 1e-15)
            z2_raw = float(z2_raw_all[idx])
            z2_rob = float(z2_rob_all[idx])
            eff_z2 = min(z2_raw, z2_rob) if self.cfg.robust_irls else z2_raw

            if self.cfg.robust_irls and (
                n_eff < self.cfg.min_effective_support
                or max_share > self.cfg.max_single_row_score_share
            ):
                vetoed.append(idx)
                continue

            innov = self.span_innovation(col)
            _, w_opt, rob_gain = self.optimal_rank1_weight_and_gain(
                col, use_robust_alpha=self.cfg.robust_irls
            )
            gauss_ev_delta = self.exact_block_delta_log_evidence(col, w_opt) if w_opt > 0.0 else 0.0
            if innov < self.cfg.span_novelty_eps:
                if eff_z2 > 1.0:
                    return AdmissionDecision(
                        action="IN_SPAN_REWEIGHT",
                        candidate_index=idx,
                        candidate_name=str(cand_names[idx]),
                        raw_score_z2=z2_raw,
                        robust_score_z2=z2_rob,
                        delta_log_evidence=gauss_ev_delta,
                        optimal_weight=w_opt,
                        span_innovation=innov,
                        effective_support=n_eff,
                        max_row_share=max_share,
                        threshold_z2=tau_z2,
                        vetoed_indices=tuple(vetoed),
                        frozen_gaussian_evidence_delta=gauss_ev_delta,
                        robust_acquisition_gain=rob_gain,
                    )
                if best_decision is None:
                    best_decision = AdmissionDecision(
                        action="IN_SPAN_REWEIGHT",
                        candidate_index=idx,
                        candidate_name=str(cand_names[idx]),
                        raw_score_z2=z2_raw,
                        robust_score_z2=z2_rob,
                        delta_log_evidence=gauss_ev_delta,
                        optimal_weight=w_opt,
                        span_innovation=innov,
                        effective_support=n_eff,
                        max_row_share=max_share,
                        threshold_z2=tau_z2,
                        vetoed_indices=tuple(vetoed),
                        frozen_gaussian_evidence_delta=gauss_ev_delta,
                        robust_acquisition_gain=rob_gain,
                    )
                continue

            if eff_z2 >= tau_z2 and w_opt > 0.0:
                return AdmissionDecision(
                    action="ADD",
                    candidate_index=idx,
                    candidate_name=str(cand_names[idx]),
                    raw_score_z2=z2_raw,
                    robust_score_z2=z2_rob,
                    delta_log_evidence=gauss_ev_delta,
                    optimal_weight=w_opt,
                    span_innovation=innov,
                    effective_support=n_eff,
                    max_row_share=max_share,
                    threshold_z2=tau_z2,
                    vetoed_indices=tuple(vetoed),
                    frozen_gaussian_evidence_delta=gauss_ev_delta,
                    robust_acquisition_gain=rob_gain,
                )

            return AdmissionDecision(
                action="REJECT",
                candidate_index=idx,
                candidate_name=str(cand_names[idx]),
                raw_score_z2=z2_raw,
                robust_score_z2=z2_rob,
                delta_log_evidence=gauss_ev_delta,
                optimal_weight=w_opt,
                span_innovation=innov,
                effective_support=n_eff,
                max_row_share=max_share,
                threshold_z2=tau_z2,
                vetoed_indices=tuple(vetoed),
                frozen_gaussian_evidence_delta=gauss_ev_delta,
                robust_acquisition_gain=rob_gain,
            )

        if best_decision is not None:
            return dataclasses.replace(best_decision, vetoed_indices=tuple(vetoed))
        idx0 = int(score_rank[0])
        col0 = pool[:, idx0]
        return AdmissionDecision(
            action="REJECT_SPIKE",
            candidate_index=idx0,
            candidate_name=str(cand_names[idx0]),
            raw_score_z2=float(z2_raw_all[idx0]),
            robust_score_z2=float(z2_rob_all[idx0]),
            delta_log_evidence=0.0,
            optimal_weight=0.0,
            span_innovation=self.span_innovation(col0),
            effective_support=compute_effective_support(col0),
            max_row_share=1.0,
            threshold_z2=tau_z2,
            vetoed_indices=tuple(vetoed),
        )

    def add_group(
        self,
        name: str,
        phi_train: np.ndarray,
        phi_val: Optional[np.ndarray] = None,
        weight: Optional[float] = None,
        family_tag: str = "feature",
    ) -> None:
        w_val = float(self.cfg.default_prior_w if weight is None else weight)
        p_tr = np.ascontiguousarray(phi_train, dtype=np.float64).reshape(self.n_rows, -1)
        p_va = (
            None
            if phi_val is None
            else np.ascontiguousarray(phi_val, dtype=np.float64).reshape(-1, p_tr.shape[1])
        )
        self.groups.append(CandidateGroup(str(name), p_tr, p_va, max(w_val, 1e-12), str(family_tag)))
        self.refactorize()

    def add_kernel(
        self,
        name: str,
        k_train: np.ndarray,
        k_val_train: Optional[np.ndarray] = None,
        variance: float = 0.25,
        family_tag: str = "kernel",
    ) -> None:
        kt = np.ascontiguousarray(k_train, dtype=np.float64).reshape(self.n_rows, self.n_rows)
        kv = None if k_val_train is None else np.ascontiguousarray(k_val_train, dtype=np.float64)
        self.kernels.append(KernelBlock(str(name), kt, kv, max(float(variance), 1e-12), str(family_tag)))
        self.refactorize()

    def reestimate_group_weights_empirical_bayes(self, n_passes: int = 3) -> None:
        """Coordinate Empirical Bayes / MacKay-Tipping weight update across active groups."""
        for _ in range(max(1, int(n_passes))):
            changed = False
            for grp in self.groups:
                if grp.phi_train.shape[1] != 1:
                    continue
                old_w = grp.weight
                grp.weight = 0.0
                self.refactorize()
                _, w_star, _ = self.optimal_rank1_weight_and_gain(
                    grp.phi_train[:, 0], use_robust_alpha=self.cfg.robust_irls
                )
                grp.weight = max(w_star, 0.0)
                if abs(grp.weight - old_w) > 1e-6:
                    changed = True
            self.groups = [g for g in self.groups if g.weight > 1e-10]
            if self.cfg.auto_scale_sigma_n2 and self.groups:
                res_var = float(np.var(self.residual))
                if res_var > 0.0:
                    # Posterior fitted train signal: Sigma_signal @ alpha = residual - D_noise * alpha
                    noise_part = self._noise_diag() * self.alpha
                    dof = max(float(self.n_rows - len(self.groups) - 1), 1.0)
                    est_s2 = float(np.dot(noise_part, noise_part)) / dof
                    lo_s2 = max(0.002 * res_var, self.cfg.jitter)
                    hi_s2 = max(self.cfg.sigma_n2_var_frac * res_var, lo_s2)
                    new_s2 = min(hi_s2, max(lo_s2, est_s2))
                    if abs(new_s2 - self.base_sigma_n2) > 1e-9 * hi_s2:
                        self.base_sigma_n2 = new_s2
                        changed = True
            self.refactorize()
            if not changed:
                break

    def raw_residual_prediction_val(self) -> np.ndarray:
        """Posterior mean residual `Sigma_{val, train} alpha` on validation/query rows.

        When no dense kernel blocks are active, evaluates the algebraically identical primal
        Woodbury system `(U^T D^{-1} U + W^{-1}) beta = U^T D^{-1} r`, avoiding `O(w / sigma_n^2)`
        conditioning loss at tiny `rho` (`ARENA D1` exact affine equivariance to `< 1e-14`).
        """
        n_val = 0
        if self.inc_val is not None:
            n_val = int(self.inc_val.shape[0])
        elif self.groups and self.groups[0].phi_val is not None:
            n_val = int(self.groups[0].phi_val.shape[0])
        elif self.kernels and self.kernels[0].k_val_train is not None:
            n_val = int(self.kernels[0].k_val_train.shape[0])

        if n_val == 0:
            return np.zeros(0, dtype=np.float64)

        has_active_kernels = any(ker.variance > 0.0 for ker in self.kernels)
        if not has_active_kernels:
            u_tr_parts: List[np.ndarray] = []
            u_va_parts: List[np.ndarray] = []
            inv_w_parts: List[np.ndarray] = []
            if self.anchor_h_train is not None and self.anchor_h_val is not None and self.tau_h2 > 0.0:
                p_h = self.anchor_h_train.shape[1]
                u_tr_parts.append(self.anchor_h_train)
                u_va_parts.append(self.anchor_h_val.reshape(n_val, p_h))
                inv_w_parts.append(np.full(p_h, 1.0 / float(self.tau_h2), dtype=np.float64))
            for grp in self.groups:
                if grp.weight > 0.0 and grp.phi_val is not None:
                    p_g = grp.phi_train.shape[1]
                    u_tr_parts.append(grp.phi_train)
                    u_va_parts.append(grp.phi_val.reshape(n_val, p_g))
                    inv_w_parts.append(np.full(p_g, 1.0 / float(grp.weight), dtype=np.float64))
            if not u_tr_parts:
                return np.zeros(n_val, dtype=np.float64)
            u_tr = np.hstack(u_tr_parts)
            u_va = np.hstack(u_va_parts)
            inv_w = np.concatenate(inv_w_parts)
            inv_d = 1.0 / self._noise_diag()
            ut_dinv = u_tr.T * inv_d[None, :]
            gram_p = ut_dinv @ u_tr + np.diag(inv_w)
            rhs_p = ut_dinv @ self.residual
            chol_p = sla.cho_factor(gram_p, lower=True, check_finite=False)
            beta_p = sla.cho_solve(chol_p, rhs_p, check_finite=False)
            return u_va @ beta_p

        pred_r = np.zeros(n_val, dtype=np.float64)
        if self.anchor_h_train is not None and self.anchor_h_val is not None and self.tau_h2 > 0.0:
            beta_h = self.tau_h2 * (self.anchor_h_train.T @ self.alpha)
            pred_r += self.anchor_h_val @ beta_h
        for grp in self.groups:
            if grp.weight > 0.0 and grp.phi_val is not None:
                beta_g = float(grp.weight) * (grp.phi_train.T @ self.alpha)
                pred_r += grp.phi_val @ beta_g
        for ker in self.kernels:
            if ker.variance > 0.0 and ker.k_val_train is not None:
                pred_r += float(ker.variance) * (ker.k_val_train @ self.alpha)
        return pred_r

    def select_damping_eta_from_oof(
        self, oof_residual_pred: np.ndarray, folds: Sequence[np.ndarray]
    ) -> Tuple[float, Dict[str, float]]:
        """Select `eta in {0, 0.25, 0.5, 0.75, 1.0}` strictly from nested OOF scores (`N2`).

        Supports both standalone fold-floor insurance (`fold_floor_rel_tol`) and router-slot
        `min_improved_folds` (e.g. `4` of `5` folds) + `min_rel_rmse_gain` (e.g. `0.004`).
        """
        r_hat_oof = np.asarray(oof_residual_pred, dtype=np.float64).reshape(self.n_rows)
        inc_fold_mse = np.array(
            [float(np.mean(self.residual[f] ** 2)) for f in folds if len(f) > 0],
            dtype=np.float64,
        )
        inc_mse = float(np.mean(self.residual ** 2))
        inc_rmse = math.sqrt(max(inc_mse, 1e-18))
        best_eta = 0.0
        best_mse = inc_mse
        eta_scores: Dict[str, float] = {"0.0": inc_mse}

        for eta in self.cfg.damping_grid:
            eta_f = float(eta)
            if eta_f == 0.0:
                continue
            rem = self.residual - eta_f * r_hat_oof
            cand_mse = float(np.mean(rem ** 2))
            eta_scores[f"{eta_f}"] = cand_mse
            cand_rmse = math.sqrt(max(cand_mse, 1e-18))
            rel_rmse_gain = (inc_rmse - cand_rmse) / inc_rmse
            if rel_rmse_gain < float(self.cfg.min_rel_rmse_gain):
                continue

            fold_mses = np.array(
                [float(np.mean(rem[f] ** 2)) for f in folds if len(f) > 0],
                dtype=np.float64,
            )
            n_improved = int(np.sum(fold_mses < inc_fold_mse - 1e-12))
            if n_improved < int(self.cfg.min_improved_folds):
                continue
            rel_deg = (fold_mses - inc_fold_mse) / np.maximum(inc_fold_mse, 1e-12)
            worst_deg = float(np.max(rel_deg)) if rel_deg.size > 0 else 0.0
            median_improved = float(np.median(fold_mses)) <= float(np.median(inc_fold_mse))
            if self.cfg.enforce_hard_fold_floor:
                if worst_deg > self.cfg.fold_floor_rel_tol:
                    continue
            elif worst_deg > self.cfg.fold_floor_rel_tol and not median_improved:
                continue
            if cand_mse < best_mse - 1e-12:
                best_mse = cand_mse
                best_eta = eta_f
        return best_eta, eta_scores

    def select_damping_eta(self, folds: Sequence[np.ndarray]) -> Tuple[float, Dict[str, float]]:
        """Fast fixed-hyperparameter closed-form LFO `eta` selector (use nested OOF for G1!)."""
        e_lfo = self.lfo_residuals(folds)
        return self.select_damping_eta_from_oof(self.residual - e_lfo, folds)

    def predict_val(
        self,
        eta: float = 1.0,
        apply_range_clamp: bool = True,
        clamp_mode: str = "search_iqr",
    ) -> np.ndarray:
        """Predict on `val` rows; `eta == 0.0` returns `inc_val` bit-identically (`max_abs_diff == 0`)."""
        base = (
            np.full_like(self.raw_residual_prediction_val(), self._affine_offset)
            if self.inc_val is None
            else self.inc_val.copy()
        )
        if float(eta) == 0.0:
            return base
        out = base + float(eta) * self.raw_residual_prediction_val()
        if (
            not apply_range_clamp
            or clamp_mode == "none"
            or self.cfg.range_clamp_iqr_mult >= 1e6
        ):
            return out
        if clamp_mode == "mad3":
            return apply_mad3_clamp(out, self.y_train)
        return apply_search_range_clamp(out, self.y_train, iqr_mult=self.cfg.range_clamp_iqr_mult)
