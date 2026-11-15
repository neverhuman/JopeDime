"""Joint Residual-Covariance Equilibrium Core (`eq_core.py`).

Unifies the six `tips/core` formulations under one exact `float64` state:
  Sigma = tau_H^2 H H^T + sum_g w_g Phi_g Phi_g^T + sum_c sigma_c^2 K_c + diag(sigma_n^2 + d_robust)

Key Phase-1 properties (incorporating `cl_fwriter`, `cl_mtc`, and `cl_h458` 2026-11-15 rulings):
- Vectorized score screening z_m^2 = (phi_m^T alpha)^2 / (phi_m^T C phi_m) + exact rank-p Schur
  marginal log-evidence and fixed-lambda ridge objective gains.
- Unregularized QR span-novelty guard (`Compiler §18.7` defense): distinguishes true span
  expansion (`ADD`) from collinear variance re-allocation (`IN_SPAN_REWEIGHT`).
- Closed-form O(1) LOO (e_i = alpha_i / C_ii) and K-fold block LFO (e_B = C_BB^{-1} alpha_B).
- Heavy-tail / 1-row spike defenses (`cl_h458` `4286` 26-sd defense):
  1. Huber/Student-t IRLS heteroscedastic row-noise inflation `d_robust`;
  2. Effective support `n_eff` + winsorized/leverage-capped score screening so a single outlier
     row cannot admit a spurious spike feature;
  3. SEARCH target-range safety clamp + per-fold floor insurance with exact `eta=0` fallback.
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import linalg as sla


@dataclasses.dataclass(frozen=True)
class EquilibriumConfig:
    """Hyperparameters and safety guards for `EquilibriumState`."""

    sigma_n2: float = 0.05
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
    damping_grid: Tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)
    jitter: float = 1e-10


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
    replaced_group_index: Optional[int] = None


def foster_george_threshold(n_candidates: int, c0: float = 2.5) -> float:
    """Multiplicity-calibrated score threshold `tau(N) = 2 ln(max(N, 1)) + c0`."""
    return 2.0 * math.log(max(int(n_candidates), 1)) + float(c0)


def calibrate_placebo_c0(
    state: "EquilibriumState",
    x_pool: np.ndarray,
    n_trials: int = 200,
    target_false_admissions_per_fit: float = 0.05,
    seed: int = 460001,
) -> float:
    """Calibrate `c0` in `tau(N) = 2 ln N + c0` via hybrid permutation + Gaussian-cov placebos."""
    rng = np.random.default_rng(seed)
    pool = np.ascontiguousarray(x_pool, dtype=np.float64)
    n_rows, n_cands = pool.shape
    if n_cands == 0:
        return float(state.cfg.foster_george_c0)
    base_2ln = 2.0 * math.log(max(n_cands, 1))
    n_b = max(60, int(n_trials))
    max_perm = np.zeros(n_b, dtype=np.float64)
    max_gauss = np.zeros(n_b, dtype=np.float64)
    c_diag_col = np.einsum("ij,ji->i", pool.T @ state.inv_cov, pool)
    denom = np.maximum(c_diag_col, 1e-15)

    def _max_null_z2(r_vec: np.ndarray, alpha_vec: np.ndarray) -> float:
        lt_alpha = state.chol_lower.T @ alpha_vec
        chi2_raw = max(float(np.dot(lt_alpha, lt_alpha)) / float(n_rows), 1e-15)
        g_raw = pool.T @ alpha_vec
        z2_raw = (g_raw * g_raw) / (denom * chi2_raw)
        if not state.cfg.robust_irls:
            return float(np.max(z2_raw))
        alpha_rob = _winsorize_vector(alpha_vec, state.cfg.winsor_alpha_k)
        lt_rob = state.chol_lower.T @ alpha_rob
        chi2_rob = max(float(np.dot(lt_rob, lt_rob)) / float(n_rows), 1e-15)
        g_rob = pool.T @ alpha_rob
        z2_rob = (g_rob * g_rob) / (denom * chi2_rob)
        return float(np.max(np.minimum(z2_raw, z2_rob)))

    for t in range(n_b):
        perm = rng.permutation(n_rows)
        r_perm = state.residual[perm]
        max_perm[t] = _max_null_z2(r_perm, state.inv_cov @ r_perm)

        xi = rng.normal(size=n_rows)
        r_g = state.chol_lower @ xi
        alpha_g = sla.solve_triangular(state.chol_lower.T, xi, lower=False, check_finite=False)
        max_gauss[t] = _max_null_z2(r_g, alpha_g)

    alpha_f = max(1e-4, min(0.5, float(target_false_admissions_per_fit)))
    rank_idx = min(n_b - 1, int(math.ceil((1.0 - alpha_f) * (n_b + 1))))
    q_perm = float(np.sort(max_perm)[rank_idx])
    q_gauss = float(np.sort(max_gauss)[rank_idx])
    q_val = max(q_perm, q_gauss) + 0.5
    return max(float(state.cfg.foster_george_c0), q_val - base_2ln)


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


def exact_ridge_duplicate_gain(
    x_active: Optional[np.ndarray],
    x_new: np.ndarray,
    y_vec: np.ndarray,
    lam: float,
) -> float:
    """Exact unhalved ridge objective reduction `2*(J(X) - J([X, x_new]))` (`Compiler §18.7`).

    Uses `Sigma_u = lam * I + X X^T` so `2 J*(X) = lam * y^T Sigma_u^{-1} y`.
    By Woodbury, adding `x_new` reduces `2 J*` by `lam * (x_new^T Sigma_u^{-1} y)^2 / (1 + x_new^T Sigma_u^{-1} x_new)`.
    """
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
        self.inc_train = (
            np.zeros(self.n_rows, dtype=np.float64)
            if incumbent_train is None
            else np.ascontiguousarray(incumbent_train, dtype=np.float64).reshape(-1)
        )
        self.inc_val = (
            None
            if incumbent_val is None
            else np.ascontiguousarray(incumbent_val, dtype=np.float64).reshape(-1)
        )
        self.residual = self.y_train - self.inc_train
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
        self.chol_lower = np.zeros((self.n_rows, self.n_rows), dtype=np.float64)
        self.inv_cov = np.zeros((self.n_rows, self.n_rows), dtype=np.float64)
        self.alpha = np.zeros(self.n_rows, dtype=np.float64)
        self.qr_basis = np.zeros((self.n_rows, 0), dtype=np.float64)
        self.refactorize()

    def _noise_diag(self) -> np.ndarray:
        return np.maximum(self.cfg.sigma_n2 + self.robust_d, self.cfg.jitter)

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
        q_mat, r_mat, _ = sla.qr(weighted, mode="economic", pivoting=True)
        if r_mat.size == 0:
            self.qr_basis = np.zeros((self.n_rows, 0), dtype=np.float64)
            return
        diag_abs = np.abs(np.diag(r_mat))
        tol = max(1e-10, 1e-10 * float(diag_abs[0]))
        rank = int(np.sum(diag_abs > tol))
        self.qr_basis = np.ascontiguousarray(q_mat[:, :rank], dtype=np.float64)

    def refactorize(self) -> None:
        """Exact O(n^3) Cholesky refactorization of `Sigma` + `inv_cov` + `alpha` + QR basis."""
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
        noise_d = self._noise_diag()
        self.cov = sig + np.diag(noise_d)
        self.chol_lower = sla.cholesky(self.cov, lower=True, check_finite=False)
        eye = np.eye(self.n_rows, dtype=np.float64)
        self.inv_cov = sla.cho_solve((self.chol_lower, True), eye, check_finite=False)
        self.alpha = self.inv_cov @ self.residual
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
            infl = np.where(ratio > 1.0, self.cfg.sigma_n2 * (ratio * ratio - 1.0), 0.0)
            self.robust_d = np.minimum(infl, 1e6 * self.cfg.sigma_n2)
            self.refactorize()
        return self.robust_d.copy()

    def marginal_log_evidence(self) -> float:
        """Exact `-0.5 * log|Sigma| - 0.5 * r^T Sigma^{-1} r`."""
        log_det = 2.0 * float(np.sum(np.log(np.diag(self.chol_lower))))
        quad = float(np.dot(self.residual, self.alpha))
        return -0.5 * log_det - 0.5 * quad

    def state_digest(self) -> str:
        """Deterministic 64-hex SHA-256 over active names, weights, and `alpha`."""
        h = hashlib.sha256()
        for grp in self.groups:
            h.update(f"G:{grp.name}:{grp.weight:.16e};".encode("utf-8"))
        for ker in self.kernels:
            h.update(f"K:{ker.name}:{ker.variance:.16e};".encode("utf-8"))
        h.update(np.ascontiguousarray(self.alpha, dtype=np.float64).tobytes())
        return h.hexdigest()

    def loo_residuals(self) -> np.ndarray:
        """Exact closed-form Leave-One-Out residual `e_i = alpha_i / C_ii`."""
        c_diag = np.maximum(np.diag(self.inv_cov), 1e-15)
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
        # S = w^{-1} I + M = w^{-1} (I + w M), so q^T S^{-1} q = w * q^T (I + w M)^{-1} q
        sol = sla.cho_solve((chol_iwm, True), q_vec, check_finite=False)
        quad = float(weight) * float(np.dot(q_vec, sol))
        return 0.5 * quad - 0.5 * log_det

    def optimal_rank1_weight_and_gain(self, col: np.ndarray, use_robust_alpha: bool = False) -> Tuple[float, float, float]:
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
        """Vectorized DGEMM screen + QR span guard + single-row spike veto + Foster-George gate."""
        pool = np.ascontiguousarray(cand_matrix, dtype=np.float64).reshape(self.n_rows, -1)
        n_cands = pool.shape[1]
        c0 = float(self.cfg.foster_george_c0 if c0_override is None else c0_override)
        tau_z2 = foster_george_threshold(n_cands, c0=c0)
        if n_cands == 0:
            return AdmissionDecision("NONE", -1, "", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, tau_z2)

        c_pool = self.inv_cov @ pool
        m_vec = np.maximum(np.einsum("ij,ij->j", pool, c_pool), 1e-15)
        lt_alpha = self.chol_lower.T @ self.alpha
        chi2_scale = max(float(np.dot(lt_alpha, lt_alpha)) / float(self.n_rows), 1e-15)
        g_raw = pool.T @ self.alpha
        z2_raw_all = (g_raw * g_raw) / (m_vec * chi2_scale)

        alpha_rob = _winsorize_vector(self.alpha, self.cfg.winsor_alpha_k)
        lt_rob = self.chol_lower.T @ alpha_rob
        chi2_rob = max(float(np.dot(lt_rob, lt_rob)) / float(self.n_rows), 1e-15)
        g_rob = pool.T @ alpha_rob
        z2_rob_all = (g_rob * g_rob) / (m_vec * chi2_rob)

        # Evaluate top-K candidates in detail for span novelty and single-row spike guards
        score_rank = np.argsort(-np.minimum(z2_raw_all, z2_rob_all) if self.cfg.robust_irls else -z2_raw_all)
        top_k = score_rank[: min(24, n_cands)]

        best_decision: Optional[AdmissionDecision] = None
        for idx_i in top_k:
            idx = int(idx_i)
            col = pool[:, idx]
            n_eff = compute_effective_support(col)
            contrib_abs = np.abs((col - float(np.mean(col))) * self.alpha)
            max_share = float(np.max(contrib_abs)) / max(float(np.sum(contrib_abs)), 1e-15)
            z2_raw = float(z2_raw_all[idx])
            z2_rob = float(z2_rob_all[idx])
            eff_z2 = min(z2_raw, z2_rob) if self.cfg.robust_irls else z2_raw

            # Single-row spike / outlier-hijack veto (`cl_h458` 26-sd guard)
            if self.cfg.robust_irls and (
                n_eff < self.cfg.min_effective_support
                or max_share > self.cfg.max_single_row_score_share
            ):
                continue

            innov = self.span_innovation(col)
            _, w_opt, delta_ev = self.optimal_rank1_weight_and_gain(
                col, use_robust_alpha=self.cfg.robust_irls
            )
            if innov < self.cfg.span_novelty_eps:
                # Collinear with existing active span: route to zero-penalty reweight, never ADD
                cand_dec = AdmissionDecision(
                    action="IN_SPAN_REWEIGHT",
                    candidate_index=idx,
                    candidate_name=str(cand_names[idx]),
                    raw_score_z2=z2_raw,
                    robust_score_z2=z2_rob,
                    delta_log_evidence=delta_ev,
                    optimal_weight=w_opt,
                    span_innovation=innov,
                    effective_support=n_eff,
                    max_row_share=max_share,
                    threshold_z2=tau_z2,
                )
                if best_decision is None and eff_z2 > 1.0:
                    best_decision = cand_dec
                continue

            if eff_z2 >= tau_z2 and w_opt > 0.0:
                return AdmissionDecision(
                    action="ADD",
                    candidate_index=idx,
                    candidate_name=str(cand_names[idx]),
                    raw_score_z2=z2_raw,
                    robust_score_z2=z2_rob,
                    delta_log_evidence=delta_ev,
                    optimal_weight=w_opt,
                    span_innovation=innov,
                    effective_support=n_eff,
                    max_row_share=max_share,
                    threshold_z2=tau_z2,
                )

            if best_decision is None:
                best_decision = AdmissionDecision(
                    action="REJECT",
                    candidate_index=idx,
                    candidate_name=str(cand_names[idx]),
                    raw_score_z2=z2_raw,
                    robust_score_z2=z2_rob,
                    delta_log_evidence=delta_ev,
                    optimal_weight=w_opt,
                    span_innovation=innov,
                    effective_support=n_eff,
                    max_row_share=max_share,
                    threshold_z2=tau_z2,
                )

        if best_decision is not None:
            return best_decision
        idx0 = int(top_k[0])
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
        p_va = None if phi_val is None else np.ascontiguousarray(phi_val, dtype=np.float64).reshape(-1, p_tr.shape[1])
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
        """Coordinate Empirical Bayes / MacKay-Tipping weight update across active groups & kernels."""
        for _ in range(max(1, int(n_passes))):
            changed = False
            for idx, grp in enumerate(self.groups):
                if grp.phi_train.shape[1] != 1:
                    continue
                # Remove group idx virtually via rank-1 downdate or temporary zeroing
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
            self.refactorize()
            if not changed:
                break

    def raw_residual_prediction_val(self) -> np.ndarray:
        """Posterior mean residual `Sigma_{val, train} alpha` on validation/query rows."""
        n_val = 0
        if self.inc_val is not None:
            n_val = int(self.inc_val.shape[0])
        elif self.groups and self.groups[0].phi_val is not None:
            n_val = int(self.groups[0].phi_val.shape[0])
        elif self.kernels and self.kernels[0].k_val_train is not None:
            n_val = int(self.kernels[0].k_val_train.shape[0])

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

    def select_damping_eta(self, folds: Sequence[np.ndarray]) -> Tuple[float, Dict[str, float]]:
        """Select `eta in {0, 0.25, 0.5, 0.75, 1.0}` via K-fold LFO with per-fold floor insurance."""
        e_lfo = self.lfo_residuals(folds)
        r_hat_lfo = self.residual - e_lfo  # out-of-fold residual prediction

        inc_fold_mse = np.array(
            [float(np.mean(self.residual[f] ** 2)) for f in folds if len(f) > 0],
            dtype=np.float64,
        )
        best_eta = 0.0
        best_mse = float(np.mean(self.residual ** 2))
        eta_scores: Dict[str, float] = {"0.0": best_mse}

        for eta in self.cfg.damping_grid:
            eta_f = float(eta)
            if eta_f == 0.0:
                continue
            rem = self.residual - eta_f * r_hat_lfo
            cand_mse = float(np.mean(rem ** 2))
            eta_scores[f"{eta_f}"] = cand_mse
            fold_mses = np.array(
                [float(np.mean(rem[f] ** 2)) for f in folds if len(f) > 0],
                dtype=np.float64,
            )
            rel_deg = (fold_mses - inc_fold_mse) / np.maximum(inc_fold_mse, 1e-12)
            worst_deg = float(np.max(rel_deg)) if rel_deg.size > 0 else 0.0
            median_improved = float(np.median(fold_mses)) <= float(np.median(inc_fold_mse))
            if worst_deg > self.cfg.fold_floor_rel_tol and not median_improved:
                continue
            if cand_mse < best_mse - 1e-12:
                best_mse = cand_mse
                best_eta = eta_f

        return best_eta, eta_scores

    def predict_val(self, eta: float = 1.0, apply_range_clamp: bool = True) -> np.ndarray:
        """Predict on `val` rows; `eta == 0.0` returns `inc_val` bit-identically (`max_abs_diff == 0`)."""
        base = (
            np.zeros_like(self.raw_residual_prediction_val())
            if self.inc_val is None
            else self.inc_val.copy()
        )
        if float(eta) == 0.0:
            return base
        out = base + float(eta) * self.raw_residual_prediction_val()
        if not apply_range_clamp:
            return out
        q25, q75 = np.percentile(self.y_train, [25.0, 75.0])
        iqr = float(q75 - q25)
        pad = self.cfg.range_clamp_iqr_mult * (iqr if iqr > 1e-9 else max(float(np.std(self.y_train)), 1e-6))
        lo = float(np.min(self.y_train)) - pad
        hi = float(np.max(self.y_train)) + pad
        return np.clip(out, lo, hi)
