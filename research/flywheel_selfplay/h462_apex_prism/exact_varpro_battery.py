"""Strictly Inductive Algebraic `EXACT-VARPRO` Battery with Anchor Cancellation (`exact_varpro_battery.py`).

Implements `Stage B` (`ARENA §6`, `Learner Compiler §6.4`, `Solver-Trained §4`, `cl_h458` `06:05Z`):
  1. Searches `w in {"identity", "log"}` (**`Yeo-Johnson` explicitly banned**), feature views
     `z in {"x", "x_plus_logx"}`, and top-screened degree-`(p, q) <= (3, 2)` rational/polynomial laws.
  2. Solves each candidate via SVD initialization of `[M_P, -diag(t) M_Q]`, 2-pass `1/|Q|`
     Sanathanan-Koerner reweighting, and 6 damped Gauss-Newton Variable-Projection steps in original
     target units.
  3. **Strictly Inductive Pole Guard (`T1` `max_abs_diff == 0.0`):** Never inspects `x_query` during
     fitting or selection; checks `min |Q| >= 0.08 * median(|Q|)` on inner-fold validation rows AND
     `200` `X_search` radial tail-extrapolation bootstrap probes.
  4. **Anchor Cancellation (`delta_a in [-1, 0]`):** Allows exact laws to cancel a corrupted smooth or
     `yj` incumbent (`y_hat = (1 + delta_a) * anchor + law(x)`, `delta_a -> -1`) rather than stacking
     residuals on top of a tail explosion.
  5. **Strict `+0.50 u` on `5/5` Folds Gate:** Admits an exact law ONLY if its unclipped utility beats
     uncapped smooth Matérn-5/2 GP by `>= +0.50 u` on **every single fold (`5/5`)**.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import linalg as sla

from .assay_regime import _matern52_gram


@dataclasses.dataclass(frozen=True)
class ExactLawSpec:
    """Selected algebraic law family specification (hyperparameters only; refit per fold)."""

    name: str
    warp: str
    view: str
    cols: Tuple[int, ...]
    num_exps: Tuple[Tuple[int, ...], ...]
    den_exps: Tuple[Tuple[int, ...], ...]
    use_anchor_cancel: bool


@dataclasses.dataclass(frozen=True)
class FittedExactLaw:
    """Inductively fitted exact law ready for `predict(x_query, anchor_query)`."""

    spec: ExactLawSpec
    z_med: np.ndarray
    z_scale: np.ndarray
    x_pos_shift: np.ndarray
    y_shift: float
    p_coef: np.ndarray
    q_coef: np.ndarray
    delta_anchor: float
    q_med_train: float

    def predict(
        self,
        x_query: np.ndarray,
        anchor_query: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """Predict row-by-row on `x_query` (`0` cross-query interaction -> `T1 == 0.0`)."""
        zq = _transform_view(x_query, self.spec.view, self.x_pos_shift, self.z_med, self.z_scale)
        zc = zq[:, list(self.spec.cols)]
        m_p = _eval_monomials(zc, self.spec.num_exps)
        m_q = _eval_monomials(zc, self.spec.den_exps)
        num = m_p @ self.p_coef
        den = np.ones(zc.shape[0], dtype=np.float64) + (m_q @ self.q_coef if m_q.shape[1] > 0 else 0.0)
        min_q_ratio = float(np.min(np.abs(den))) / max(self.q_med_train, 1e-12)
        safe_den = np.where(np.abs(den) >= 1e-9, den, np.where(den >= 0.0, 1e-9, -1e-9))
        law_raw = num / safe_den
        if self.spec.warp == "log":
            law_y = np.exp(np.clip(law_raw, -30.0, 30.0)) - self.y_shift
        else:
            law_y = law_raw
        if anchor_query is not None:
            anc = np.asarray(anchor_query, dtype=np.float64).reshape(-1)
            return (1.0 + self.delta_anchor) * anc + law_y, min_q_ratio
        return law_y, min_q_ratio


@dataclasses.dataclass(frozen=True)
class ExactBatteryResult:
    """Result of `run_exact_varpro_battery` on `SEARCH`."""

    accepted: bool
    selected_law_name: Optional[str]
    delta_anchor: float
    oof_pred: np.ndarray
    gp_oof_pred: np.ndarray
    fold_u_exact: Tuple[float, ...]
    fold_u_gp: Tuple[float, ...]
    min_fold_margin_u: float
    min_bootstrap_pole_ratio: float
    fitted_full: Optional[FittedExactLaw]

    def predict_query(
        self,
        x_query: np.ndarray,
        fallback_query: np.ndarray,
        anchor_query: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if not self.accepted or self.fitted_full is None:
            return np.asarray(fallback_query, dtype=np.float64).copy()
        pred_q, _ = self.fitted_full.predict(x_query, anchor_query=anchor_query)
        return pred_q


def _fit_view_stats(x_tr: np.ndarray, view: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.ascontiguousarray(x_tr, dtype=np.float64)
    pos_shift = np.maximum(0.0, -np.min(x, axis=0) + 1.0)
    if view == "x_plus_logx":
        raw = np.hstack([x, np.log(np.maximum(x + pos_shift[None, :], 1e-9))])
    else:
        raw = x
    med = np.median(raw, axis=0)
    q25, q75 = np.percentile(raw, [25.0, 75.0], axis=0)
    scale = np.where((q75 - q25) > 1e-9, q75 - q25, np.maximum(np.std(raw, axis=0), 1e-9))
    return pos_shift, med, scale


def _transform_view(
    x_in: np.ndarray,
    view: str,
    pos_shift: np.ndarray,
    med: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    x = np.ascontiguousarray(x_in, dtype=np.float64)
    if view == "x_plus_logx":
        raw = np.hstack([x, np.log(np.maximum(x + pos_shift[None, :], 1e-9))])
    else:
        raw = x
    return (raw - med[None, :]) / scale[None, :]


def _generate_exponents(dim: int, max_deg: int, include_const: bool) -> Tuple[Tuple[int, ...], ...]:
    out: List[Tuple[int, ...]] = []

    def _rec(idx: int, rem: int, cur: List[int]) -> None:
        if idx == dim:
            deg = sum(cur)
            if deg == 0 and not include_const:
                return
            if deg <= max_deg:
                out.append(tuple(cur))
            return
        for e in range(rem + 1):
            _rec(idx + 1, rem - e, cur + [e])

    _rec(0, max_deg, [])
    out.sort(key=lambda tup: (sum(tup), tup))
    return tuple(out)


def _eval_monomials(zc: np.ndarray, exps: Sequence[Tuple[int, ...]]) -> np.ndarray:
    n_rows = zc.shape[0]
    if not exps:
        return np.zeros((n_rows, 0), dtype=np.float64)
    cols = []
    for exp_t in exps:
        col = np.ones(n_rows, dtype=np.float64)
        for j, pwr in enumerate(exp_t):
            if pwr == 1:
                col = col * zc[:, j]
            elif pwr == 2:
                col = col * (zc[:, j] * zc[:, j])
            elif pwr > 2:
                col = col * (zc[:, j] ** pwr)
        cols.append(col)
    return np.column_stack(cols)


def _inductive_bootstrap_probes(z_tr: np.ndarray, n_probes: int = 200, seed: int = 462101) -> np.ndarray:
    """Generate strictly `SEARCH`-derived radial tail-extrapolation probes (`1.35x` envelope)."""
    rng = np.random.default_rng(int(seed))
    n_rows, n_cols = z_tr.shape
    med = np.median(z_tr, axis=0)
    idx_a = rng.integers(0, n_rows, size=int(n_probes))
    idx_b = rng.integers(0, n_rows, size=int(n_probes))
    stretch = rng.uniform(1.05, 1.35, size=(int(n_probes), 1))
    mix = rng.uniform(-0.20, 1.20, size=(int(n_probes), n_cols))
    base = mix * z_tr[idx_a] + (1.0 - mix) * z_tr[idx_b]
    return med[None, :] + stretch * (base - med[None, :])


def _safe_pos_solve(gram: np.ndarray, rhs: np.ndarray, rel_eps: float = 1e-10) -> Optional[np.ndarray]:
    """Scale-invariant regularized symmetric solve with `lstsq` fallback for collinear/extreme-scale tables."""
    if not (np.all(np.isfinite(gram)) and np.all(np.isfinite(rhs))):
        return None
    d = gram.shape[0]
    diag_scale = float(np.mean(np.abs(np.diag(gram)))) if d > 0 else 1.0
    reg = max(1e-10, rel_eps * max(diag_scale, 1.0))
    g_reg = 0.5 * (gram + gram.T) + reg * np.eye(d, dtype=np.float64)
    try:
        sol = sla.solve(g_reg, rhs, assume_a="pos", check_finite=False)
        if np.all(np.isfinite(sol)):
            return sol
    except (sla.LinAlgError, np.linalg.LinAlgError, ValueError):
        pass
    try:
        sol_ls = sla.lstsq(g_reg, rhs, check_finite=False)[0]
        if np.all(np.isfinite(sol_ls)):
            return sol_ls
    except (sla.LinAlgError, np.linalg.LinAlgError, ValueError):
        return None
    return None


def _fit_single_spec_from_init_delta(
    zc: np.ndarray,
    m_p: np.ndarray,
    m_q: np.ndarray,
    y_tr: np.ndarray,
    spec: ExactLawSpec,
    anchor_tr: Optional[np.ndarray],
    init_delta_a: float,
    n_gn_steps: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, float, float]]:
    """Run SVD + 2-pass SK + Gauss-Newton from a given initial `delta_a in [-1, 0]`."""
    n_p = m_p.shape[1]
    n_q = m_q.shape[1]
    delta_a = float(init_delta_a)

    if anchor_tr is not None:
        anc = np.asarray(anchor_tr, dtype=np.float64).reshape(-1)
        target_y = y_tr - (1.0 + delta_a) * anc
    else:
        anc = None
        target_y = y_tr.copy()

    if not np.all(np.isfinite(target_y)):
        return None

    if spec.warp == "log":
        min_y = float(np.min(target_y))
        if min_y <= 1e-6:
            return None
        t_fit = np.log(target_y)
    else:
        t_fit = target_y

    if n_q == 0:
        if anc is not None and spec.use_anchor_cancel and spec.warp == "identity":
            z_aug = np.column_stack([anc, m_p])
            c_aug = _safe_pos_solve(z_aug.T @ z_aug, z_aug.T @ y_tr, rel_eps=1e-10)
            if c_aug is None:
                return None
            delta_a = min(0.0, max(-1.0, float(c_aug[0]) - 1.0))
            t_eff = y_tr - (1.0 + delta_a) * anc
            try:
                p_coef = sla.lstsq(m_p, t_eff, check_finite=False)[0]
            except (sla.LinAlgError, np.linalg.LinAlgError, ValueError):
                return None
        else:
            try:
                p_coef = sla.lstsq(m_p, t_fit, check_finite=False)[0]
            except (sla.LinAlgError, np.linalg.LinAlgError, ValueError):
                return None
        if not np.all(np.isfinite(p_coef)):
            return None
        q_coef = np.zeros(0, dtype=np.float64)
    else:
        # Normalize target scale during SK initialization so `t_fit * m_q` stays O(1)
        t_scale = max(float(np.std(t_fit)), float(np.max(np.abs(t_fit))) * 1e-3, 1e-12)
        t_norm = t_fit / t_scale
        a_lin = np.hstack([m_p, -(t_norm[:, None] * m_q)])
        if not np.all(np.isfinite(a_lin)):
            return None
        col_nrm = np.maximum(np.linalg.norm(a_lin, axis=0), 1e-12)
        try:
            _, _, vt = sla.svd(a_lin / col_nrm[None, :], full_matrices=False, check_finite=False)
        except (sla.LinAlgError, np.linalg.LinAlgError, ValueError):
            return None
        v_last = vt[-1, :] / col_nrm
        p_coef = v_last[:n_p] * t_scale
        q_coef = v_last[n_p:]
        w_sk = np.ones(zc.shape[0], dtype=np.float64)
        for _ in range(2):
            a_w = np.hstack([m_p, -(t_norm[:, None] * m_q)]) * w_sk[:, None]
            b_w = t_norm * w_sk
            sol = _safe_pos_solve(a_w.T @ a_w, a_w.T @ b_w, rel_eps=1e-9)
            if sol is None:
                break
            p_coef = sol[:n_p] * t_scale
            q_coef = sol[n_p:]
            den_cur = 1.0 + m_q @ q_coef
            if float(np.min(np.abs(den_cur))) < 1e-4:
                q_coef = 0.25 * q_coef
                den_cur = 1.0 + m_q @ q_coef
            w_sk = 1.0 / np.maximum(np.abs(den_cur), 1e-2)

        damp = 1e-4
        for _ in range(max(1, int(n_gn_steps))):
            den = 1.0 + m_q @ q_coef
            if not np.all(np.isfinite(den)) or float(np.min(np.abs(den))) < 1e-3:
                break
            inv_q = 1.0 / den
            z_vp = m_p * inv_q[:, None]
            if anc is not None and spec.use_anchor_cancel and spec.warp == "identity":
                z_aug = np.column_stack([anc, z_vp])
                c_aug = _safe_pos_solve(z_aug.T @ z_aug, z_aug.T @ y_tr, rel_eps=1e-10)
                if c_aug is None:
                    break
                delta_a = min(0.0, max(-1.0, float(c_aug[0]) - 1.0))
                t_eff = y_tr - (1.0 + delta_a) * anc
            else:
                t_eff = t_fit

            g_p = z_vp.T @ z_vp
            p_cand = _safe_pos_solve(g_p, z_vp.T @ t_eff, rel_eps=1e-10)
            if p_cand is None:
                break
            p_coef = p_cand
            f_val = z_vp @ p_coef
            res = t_eff - f_val
            cur_sse = float(np.dot(res, res))

            dz_p = -((f_val * inv_q)[:, None] * m_q)
            proj_rhs = _safe_pos_solve(g_p, z_vp.T @ dz_p, rel_eps=1e-10)
            if proj_rhs is None:
                break
            proj = z_vp @ proj_rhs
            j_vp = dz_p - proj
            jt_j = j_vp.T @ j_vp + damp * max(float(np.mean(np.abs(np.diag(j_vp.T @ j_vp)))), 1.0) * np.eye(n_q, dtype=np.float64)
            step_q = _safe_pos_solve(jt_j, -(j_vp.T @ res), rel_eps=1e-12)
            if step_q is None:
                break

            accepted_step = False
            for ls in (1.0, 0.5, 0.25, 0.1):
                cand_q = q_coef + ls * step_q
                cand_den = 1.0 + m_q @ cand_q
                if not np.all(np.isfinite(cand_den)) or float(np.min(np.abs(cand_den))) < 0.05 * float(np.median(np.abs(cand_den))):
                    continue
                cand_z = m_p / cand_den[:, None]
                cand_p = _safe_pos_solve(cand_z.T @ cand_z, cand_z.T @ t_eff, rel_eps=1e-10)
                if cand_p is None:
                    continue
                cand_sse = float(np.sum((t_eff - cand_z @ cand_p) ** 2))
                if math.isfinite(cand_sse) and cand_sse <= cur_sse:
                    q_coef = cand_q
                    p_coef = cand_p
                    damp = max(1e-7, 0.5 * damp)
                    accepted_step = True
                    break
            if not accepted_step:
                damp = min(1e2, 4.0 * damp)

    den_final = np.ones(zc.shape[0], dtype=np.float64) + (m_q @ q_coef if n_q > 0 else 0.0)
    q_med = max(float(np.median(np.abs(den_final))), 1e-9)
    if float(np.min(np.abs(den_final))) < 0.08 * q_med:
        return None
    law_raw = (m_p @ p_coef) / den_final
    law_y = np.exp(np.clip(law_raw, -30.0, 30.0)) if spec.warp == "log" else law_raw
    pred_tr = ((1.0 + delta_a) * anc + law_y) if anc is not None else law_y
    sse_orig = float(np.sum((y_tr - pred_tr) ** 2))
    return p_coef, q_coef, delta_a, sse_orig


def _fit_single_spec(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    spec: ExactLawSpec,
    anchor_tr: Optional[np.ndarray] = None,
    n_gn_steps: int = 6,
) -> Optional[FittedExactLaw]:
    """Fit `ExactLawSpec` via SVD + 2-pass `1/|Q|` + 6-step Gauss-Newton Variable Projection."""
    if spec.warp not in ("identity", "log"):
        raise ValueError(f"Forbidden warp {spec.warp!r}; Yeo-Johnson is banned")

    pos_shift, z_med, z_scale = _fit_view_stats(x_tr, spec.view)
    z_all = _transform_view(x_tr, spec.view, pos_shift, z_med, z_scale)
    if max(spec.cols) >= z_all.shape[1]:
        return None
    zc = z_all[:, list(spec.cols)]
    m_p = _eval_monomials(zc, spec.num_exps)
    m_q = _eval_monomials(zc, spec.den_exps)
    n_q = m_q.shape[1]

    init_deltas: List[float] = [0.0]
    if anchor_tr is not None and spec.use_anchor_cancel:
        anc = np.asarray(anchor_tr, dtype=np.float64).reshape(-1)
        design_init = np.column_stack([anc, m_p])
        coef_init, _, _, _ = sla.lstsq(design_init, y_tr, check_finite=False)
        partial_d = min(0.0, max(-1.0, float(coef_init[0]) - 1.0))
        # Always evaluate both full anchor cancellation (`delta_a = -1.0`, which cleanly strips
        # non-smooth `_yj` tail blowups before SVD initialization) and partial cancellation.
        init_deltas = [-1.0, partial_d]

    best_sol: Optional[Tuple[np.ndarray, np.ndarray, float, float]] = None
    for d0 in init_deltas:
        cand = _fit_single_spec_from_init_delta(zc, m_p, m_q, y_tr, spec, anchor_tr, d0, n_gn_steps)
        if cand is not None and (best_sol is None or cand[3] < best_sol[3]):
            best_sol = cand

    if best_sol is None:
        return None
    p_coef, q_coef, delta_a, _ = best_sol

    den_final = np.ones(zc.shape[0], dtype=np.float64) + (m_q @ q_coef if n_q > 0 else 0.0)
    q_med = max(float(np.median(np.abs(den_final))), 1e-9)

    # Strictly inductive pole check on 200 X_search bootstrap tail-extrapolation probes
    probes = _inductive_bootstrap_probes(zc, n_probes=200)
    if n_q > 0:
        den_probes = 1.0 + _eval_monomials(probes, spec.den_exps) @ q_coef
        if float(np.min(np.abs(den_probes))) < 0.08 * q_med:
            return None

    return FittedExactLaw(
        spec=spec,
        z_med=z_med,
        z_scale=z_scale,
        x_pos_shift=pos_shift,
        y_shift=0.0,
        p_coef=p_coef,
        q_coef=q_coef,
        delta_anchor=delta_a,
        q_med_train=q_med,
    )


def _predict_uncapped_gp_oof(
    x_search: np.ndarray,
    y_search: np.ndarray,
    lab_folds: np.ndarray,
) -> np.ndarray:
    """5-fold OOF uncapped Matérn-5/2 smooth GP baseline (`float64`, `rho_min = 1e-8`)."""
    x = np.ascontiguousarray(x_search, dtype=np.float64)
    y = np.ascontiguousarray(y_search, dtype=np.float64).reshape(-1)
    lab = np.asarray(lab_folds, dtype=np.int64).reshape(-1)
    oof = np.zeros_like(y)

    for k in range(5):
        va = lab == k
        tr = ~va
        x_tr, y_tr, x_va = x[tr], y[tr], x[va]
        y_mu = float(np.mean(y_tr))
        yc = y_tr - y_mu
        med = np.nanmedian(x_tr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        x_tr_imp = np.where(np.isfinite(x_tr), x_tr, med)
        x_va_imp = np.where(np.isfinite(x_va), x_va, med)
        q25, q75 = np.percentile(x_tr_imp, [25.0, 75.0], axis=0)
        sc_raw = np.where((q75 - q25) > 1e-9, q75 - q25, np.std(x_tr_imp, axis=0))
        sc = np.where(np.isfinite(sc_raw) & (sc_raw > 1e-9), sc_raw, 1.0)
        z_tr = np.clip((x_tr_imp - med) / sc, -25.0, 25.0)
        z_va = np.clip((x_va_imp - med) / sc, -25.0, 25.0)

        best_loo = float("inf")
        best_pred = np.full(x_va.shape[0], y_mu, dtype=np.float64)
        for ell in (0.6, 1.4, 3.2):
            k_tt = _matern52_gram(z_tr, z_tr, ell)
            ev, vecs = sla.eigh(0.5 * (k_tt + k_tt.T), check_finite=False)
            ev = np.maximum(ev, 0.0)
            vt_y = vecs.T @ yc
            v_sq = vecs * vecs
            for rho in (1e-8, 1e-6, 1e-4, 1e-2, 1e-1):
                inv_d = 1.0 / (ev + rho)
                alpha = vecs @ (vt_y * inv_d)
                e_loo = alpha / np.maximum(v_sq @ inv_d, 1e-15)
                rmse = float(math.sqrt(np.mean(e_loo * e_loo)))
                if rmse < best_loo:
                    best_loo = rmse
                    k_vt = _matern52_gram(z_va, z_tr, ell)
                    best_pred = y_mu + k_vt @ alpha
        oof[va] = best_pred
    return oof


def _corr_abs(u: np.ndarray, v: np.ndarray) -> float:
    uc = u - float(np.mean(u))
    vc = v - float(np.mean(v))
    denom = max(float(np.linalg.norm(uc)) * float(np.linalg.norm(vc)), 1e-12)
    return abs(float(np.dot(uc, vc))) / denom


def _build_candidate_specs(
    x_search: np.ndarray,
    y_search: np.ndarray,
    allow_anchor_cancel: bool,
) -> List[ExactLawSpec]:
    """Screen top feature subsets on `SEARCH` (linear, quadratic, and rational-interaction aware)."""
    x = np.ascontiguousarray(x_search, dtype=np.float64)
    y = np.ascontiguousarray(y_search, dtype=np.float64).reshape(-1)
    n_cols = x.shape[1]

    med = np.median(x, axis=0)
    q25, q75 = np.percentile(x, [25.0, 75.0], axis=0)
    sc = np.where((q75 - q25) > 1e-9, q75 - q25, np.maximum(np.std(x, axis=0), 1e-9))
    z = (x - med[None, :]) / sc[None, :]

    lin_corrs = [_corr_abs(z[:, j], y) for j in range(n_cols)]
    j_lead = int(np.argmax(lin_corrs))
    z_lead = z[:, j_lead]
    # Linear residual of y wrt lead column exposes multiplicative & rational-denominator terms
    b_lead = float(np.dot(z_lead - np.mean(z_lead), y - np.mean(y))) / max(
        float(np.sum((z_lead - np.mean(z_lead)) ** 2)), 1e-12
    )
    y_res = (y - np.mean(y)) - b_lead * (z_lead - np.mean(z_lead))
    y_abs = np.abs(y - float(np.median(y)))

    scores = []
    for j in range(n_cols):
        zj = z[:, j]
        zj2 = zj * zj
        s_j = max(
            lin_corrs[j],
            _corr_abs(zj2, y),
            _corr_abs(zj2, y_abs),
            _corr_abs(z_lead * zj, y_res),
            _corr_abs(z_lead * zj2, y_res),
        )
        scores.append(s_j)

    rank_cols = [int(i) for i in np.argsort(-np.asarray(scores))]
    top3 = tuple(rank_cols[: min(3, n_cols)])
    pair_candidates: List[Tuple[int, int]] = []
    for a_idx in range(len(top3)):
        for b_idx in range(a_idx + 1, len(top3)):
            pair_candidates.append((top3[a_idx], top3[b_idx]))

    specs: List[ExactLawSpec] = []
    cancel_modes = (True, False) if allow_anchor_cancel else (False,)
    warps = ["identity"]
    if float(np.min(y)) > 1e-3 and not allow_anchor_cancel:
        warps.append("log")

    for cancel in cancel_modes:
        tag_c = "_ancCancel" if cancel else ""
        for warp in warps:
            if warp == "log" and cancel:
                continue
            for pair in pair_candidates[:2]:
                specs.append(
                    ExactLawSpec(
                        name=f"rat2_deg32_{warp}{tag_c}_{pair}",
                        warp=warp,
                        view="x",
                        cols=pair,
                        num_exps=_generate_exponents(2, 3, include_const=True),
                        den_exps=_generate_exponents(2, 2, include_const=False),
                        use_anchor_cancel=cancel,
                    )
                )
            if len(top3) >= 1:
                specs.append(
                    ExactLawSpec(
                        name=f"rat3_deg21_{warp}{tag_c}_{top3}",
                        warp=warp,
                        view="x",
                        cols=top3,
                        num_exps=_generate_exponents(len(top3), 2, include_const=True),
                        den_exps=_generate_exponents(len(top3), 1, include_const=False),
                        use_anchor_cancel=cancel,
                    )
                )
                specs.append(
                    ExactLawSpec(
                        name=f"poly3_deg3_{warp}{tag_c}_{top3}",
                        warp=warp,
                        view="x",
                        cols=top3,
                        num_exps=_generate_exponents(len(top3), 3, include_const=True),
                        den_exps=(),
                        use_anchor_cancel=cancel,
                    )
                )
            if float(np.min(x)) > 0.0 and pair_candidates:
                top2 = pair_candidates[0]
                log_cols = tuple(list(top2) + [c + n_cols for c in top2])
                specs.append(
                    ExactLawSpec(
                        name=f"logpoly_deg2_{warp}{tag_c}_{top2}",
                        warp=warp,
                        view="x_plus_logx",
                        cols=log_cols,
                        num_exps=_generate_exponents(len(log_cols), 2, include_const=True),
                        den_exps=(),
                        use_anchor_cancel=cancel,
                    )
                )
    return specs


def run_exact_varpro_battery(
    x_search: np.ndarray,
    y_search: np.ndarray,
    lab_folds: np.ndarray,
    anchor_oof: Optional[np.ndarray] = None,
    anchor_full_train: Optional[np.ndarray] = None,
    min_fold_margin_u: float = 0.50,
) -> ExactBatteryResult:
    """Fit & certify `EXACT-VARPRO` strictly on `(x_search, y_search, lab_folds)` (`T1 == 0.0`)."""
    x = np.ascontiguousarray(x_search, dtype=np.float64)
    y = np.ascontiguousarray(y_search, dtype=np.float64).reshape(-1)
    lab = np.asarray(lab_folds, dtype=np.int64).reshape(-1)
    y_std = max(float(np.std(y)), 1e-30)
    rmse_floor = max(1e-15 * y_std, 1e-300)

    gp_oof = _predict_uncapped_gp_oof(x, y, lab)
    gp_fold_u = tuple(
        float(math.log(y_std / max(float(math.sqrt(np.mean((y[lab == k] - gp_oof[lab == k]) ** 2))), rmse_floor)))
        for k in range(5)
    )
    gp_pooled_rmse = float(math.sqrt(np.mean((y - gp_oof) ** 2)))
    gp_pooled_u = float(math.log(y_std / max(gp_pooled_rmse, rmse_floor)))

    specs = _build_candidate_specs(x, y, allow_anchor_cancel=(anchor_oof is not None))
    best_pooled_u = -1e9
    best_res: Optional[Tuple[ExactLawSpec, np.ndarray, Tuple[float, ...], float]] = None

    for spec in specs:
        oof = np.zeros_like(y)
        min_pole_ratio = 1e9
        valid = True
        for k in range(5):
            va = lab == k
            tr = ~va
            anc_tr = None if anchor_oof is None else anchor_oof[tr]
            anc_va = None if anchor_oof is None else anchor_oof[va]
            fitted_k = _fit_single_spec(x[tr], y[tr], spec, anchor_tr=anc_tr)
            if fitted_k is None:
                valid = False
                break
            pred_va, q_ratio = fitted_k.predict(x[va], anchor_query=anc_va)
            if q_ratio < 0.08 or not np.all(np.isfinite(pred_va)):
                valid = False
                break
            min_pole_ratio = min(min_pole_ratio, q_ratio)
            oof[va] = pred_va
        if not valid:
            continue

        fold_u = tuple(
            float(math.log(y_std / max(float(math.sqrt(np.mean((y[lab == k] - oof[lab == k]) ** 2))), rmse_floor)))
            for k in range(5)
        )
        margins = [fold_u[k] - gp_fold_u[k] for k in range(5)]
        min_margin = min(margins)
        pooled_rmse = float(math.sqrt(np.mean((y - oof) ** 2)))
        pooled_u = float(math.log(y_std / max(pooled_rmse, rmse_floor)))
        pooled_margin = pooled_u - gp_pooled_u
        if min(min_margin, pooled_margin) >= float(min_fold_margin_u) and pooled_u > best_pooled_u:
            best_pooled_u = pooled_u
            best_res = (spec, oof, fold_u, min_pole_ratio)

    if best_res is None:
        fallback = gp_oof if anchor_oof is None else np.asarray(anchor_oof, dtype=np.float64).copy()
        return ExactBatteryResult(
            accepted=False,
            selected_law_name=None,
            delta_anchor=0.0,
            oof_pred=fallback,
            gp_oof_pred=gp_oof,
            fold_u_exact=gp_fold_u,
            fold_u_gp=gp_fold_u,
            min_fold_margin_u=0.0,
            min_bootstrap_pole_ratio=0.0,
            fitted_full=None,
        )

    win_spec, win_oof, win_fold_u, win_pole = best_res
    anc_full = anchor_oof if anchor_full_train is None else anchor_full_train
    fitted_full = _fit_single_spec(x, y, win_spec, anchor_tr=anc_full)
    if fitted_full is None:
        fallback = gp_oof if anchor_oof is None else np.asarray(anchor_oof, dtype=np.float64).copy()
        return ExactBatteryResult(
            accepted=False,
            selected_law_name=None,
            delta_anchor=0.0,
            oof_pred=fallback,
            gp_oof_pred=gp_oof,
            fold_u_exact=gp_fold_u,
            fold_u_gp=gp_fold_u,
            min_fold_margin_u=0.0,
            min_bootstrap_pole_ratio=0.0,
            fitted_full=None,
        )

    margins_win = [win_fold_u[k] - gp_fold_u[k] for k in range(5)]
    return ExactBatteryResult(
        accepted=True,
        selected_law_name=win_spec.name,
        delta_anchor=fitted_full.delta_anchor,
        oof_pred=win_oof,
        gp_oof_pred=gp_oof,
        fold_u_exact=win_fold_u,
        fold_u_gp=gp_fold_u,
        min_fold_margin_u=float(min(margins_win)),
        min_bootstrap_pole_ratio=float(win_pole),
        fitted_full=fitted_full,
    )
