"""Router-Slot Non-Smooth Dictionary & 4-Arm Frozen Ratchet Harness (`router_ratchet.py`).

Implements the exact router-callable interface and 4-arm shared harness specified in `cl_mtc`'s
`PREREG_G1_DRAFT_v1.md` (§2, §3, §8, §8b) and `cl_fwriter`'s reviews (`04:34Z` `#1–#11`,
`04:36Z` `G0_8` + `N1–N6`, `04:54Z` `D1–D4`, `05:02Z`):
  1. `fit_router_fold_dictionary(x_train, r_train, x_query, ...)`:
     `(fold training X, fold residual, query X) -> RouterFoldFit` (admitted components, query
     residual prediction, nested-placebo `c0`, candidate count, and state digest), rebuilding the
     entire dictionary (`scan_best_step_features`, `scan_best_equality_features`,
     `build_gated_smooth_features`, `fit_rational_law_vp`, `B1/B2` kernels) on every null draw.
  2. `fit_router_fold_static_control(x_train, r_train, x_query, ...)`:
     Equal-compute smooth random-Fourier-feature (`RFF`) + linear ridge bootstrap ensemble on
     `(x_train, r_train)` for `R3 C_STATIC` (`Finding #3`, `N3`, `PREREG_G1_DRAFT_v1` §2), budgeted
     via `time.process_time()` to match `R1`'s worker CPU-seconds per context within `±10%`.
  3. Frozen configuration grids (`D3`):
     - `ROUTER_R0_CONFIG_GRID`: Exact `27` `(kind, ell, lam, eta)` configs of
       `guard_matern_core.fit_l2_residual_gate` (`L1322–1336`).
     - `ROUTER_R1_CONFIG_GRID`: Frozen `30` configs (`27` `R0` Matérn configs + `3`
       `("dict_eq", 0.0, 0.0, eta)` equilibrium dictionary configs for `eta in (0.35, 0.65, 1.0)`).
     - `ROUTER_R2_L3_CONFIG_GRID`: Frozen `30` configs (`27` `R0` configs + `3`
       `("l3_ridge", 0.0, 0.0, eta)` simultaneous inner-CV dense ridge configs on the same bank).
     - `ROUTER_R3_STATIC_CONFIG_GRID`: Frozen `30` configs (`27` `R0` configs + `3`
       `("c_static", 0.0, 0.0, eta)` equal-compute smooth RFF ridge ensemble configs).
  4. `run_router_ratchet_harness(...)`:
     Executes `R0`, `R1`, `R2`, and `R3` in **one harness** (`D2`): identical `4-of-5` (`0.9995`) +
     `0.004` log-RMSE ratchet, zero fallback, nested `0.5 * r_bag + 0.5 * r_full` (`D4`),
     `±2.5 MAD` AUDIT clip alongside unclamped outputs (`N4`), unified `"fired"` + subset
     `"dictionary_fired"` (`D1`), `changed_vs_r0`, and `§8b` SEARCH-OOF `u >= 2` precision proxy.
"""
from __future__ import annotations

import dataclasses
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import linalg as sla

from .eq_core import (
    EquilibriumConfig,
    EquilibriumState,
    calibrate_nested_discovery_placebo_c0,
)
from .jacobi_operators import PrecomputedXContext, build_production_router_dictionary

ROUTER_ELL_GRID: Tuple[float, ...] = (1.2, 2.5, 50.0)
ROUTER_LAM_GRID: Tuple[float, ...] = (2.0, 8.0, 25.0)
ROUTER_ETA_GRID: Tuple[float, ...] = (0.35, 0.65, 1.0)

ROUTER_R0_CONFIG_GRID: Tuple[Tuple[str, float, float, float], ...] = tuple(
    ("matern32", ell, lam, eta)
    for ell in ROUTER_ELL_GRID
    for lam in ROUTER_LAM_GRID
    for eta in ROUTER_ETA_GRID
)
ROUTER_R1_CONFIG_GRID: Tuple[Tuple[str, float, float, float], ...] = ROUTER_R0_CONFIG_GRID + tuple(
    ("dict_eq", 0.0, 0.0, eta) for eta in ROUTER_ETA_GRID
)
ROUTER_R2_L3_CONFIG_GRID: Tuple[Tuple[str, float, float, float], ...] = ROUTER_R0_CONFIG_GRID + tuple(
    ("l3_ridge", 0.0, 0.0, eta) for eta in ROUTER_ETA_GRID
)
ROUTER_R3_STATIC_CONFIG_GRID: Tuple[Tuple[str, float, float, float], ...] = ROUTER_R0_CONFIG_GRID + tuple(
    ("c_static", 0.0, 0.0, eta) for eta in ROUTER_ETA_GRID
)

assert len(ROUTER_R0_CONFIG_GRID) == 27
assert len(ROUTER_R1_CONFIG_GRID) == 30
assert len(ROUTER_R2_L3_CONFIG_GRID) == 30
assert len(ROUTER_R3_STATIC_CONFIG_GRID) == 30


@dataclasses.dataclass(frozen=True)
class RouterFoldFit:
    """Output of `fit_router_fold_dictionary` (`PREREG_G1_DRAFT_v1` §8)."""

    admitted_components: Tuple[str, ...]
    query_residual_pred: np.ndarray
    c0_nested: float
    n_bank_candidates: int
    state_digest: str
    bank_train: np.ndarray
    bank_query: np.ndarray


def _matern32_gram(za: np.ndarray, zb: np.ndarray, ell: float) -> np.ndarray:
    """Bit-for-bit replica of `guard_matern_core._matern32_gram` (`L1248–1257`, `G0_8 D3`)."""
    d_cols = max(int(za.shape[1]), 1)
    sc = float(ell) * math.sqrt(float(d_cols))
    a_scaled = za / sc
    b_scaled = zb / sc
    an2 = np.sum(a_scaled * a_scaled, axis=1, keepdims=True)
    bn2 = np.sum(b_scaled * b_scaled, axis=1, keepdims=True).T
    dist = np.sqrt(np.maximum(an2 + bn2 - 2.0 * (a_scaled @ b_scaled.T), 0.0))
    s_mat = math.sqrt(3.0) * dist
    return (1.0 + s_mat) * np.exp(-s_mat) + 0.20 * (a_scaled @ b_scaled.T)


def target_stratified_kfold_labels(y: np.ndarray, n_splits: int = 5, seed: int = 460) -> np.ndarray:
    """Target-quantile-stratified K-fold labels (`y`-sorted blocks of `n_splits`).

    Places exactly one row from every sorted `n_splits`-quantile block into each fold, ensuring
    heavy-tailed rows (`tail > 12`) are tested in all `5/5` folds rather than `1/5`.
    """
    y_vec = np.asarray(y, dtype=np.float64).reshape(-1)
    n_rows = y_vec.shape[0]
    k_val = max(2, int(n_splits))
    order = np.argsort(y_vec, kind="mergesort")
    rng = np.random.default_rng(int(seed))
    labels = np.zeros(n_rows, dtype=np.int64)
    for start in range(0, n_rows, k_val):
        block = order[start : min(start + k_val, n_rows)]
        perm = rng.permutation(k_val)[: block.size]
        labels[block] = perm
    return labels


def _passes_fold_ratchet(
    base_f: np.ndarray,
    new_f: np.ndarray,
    min_gain: float = 0.004,
    min_fold_t: float = 2.776,
) -> Tuple[bool, float]:
    """Require `>= 4/5` fold wins (`0.9995`), `mean_gain >= min_gain`, and `t_4 >= min_fold_t` (`D3`)."""
    folds_won = int(np.sum(new_f < base_f * 0.9995))
    fold_gains = np.log(base_f / np.maximum(new_f, 1e-15))
    mean_gain = float(np.mean(fold_gains))
    if folds_won < 4 or mean_gain < float(min_gain):
        return False, mean_gain
    se_gain = float(np.std(fold_gains, ddof=1)) / math.sqrt(max(int(fold_gains.size), 1))
    t_stat = mean_gain / max(se_gain, 1e-12)
    return bool(t_stat >= float(min_fold_t)), mean_gain


def _safe_eigh(gram: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ev, vecs = sla.eigh(0.5 * (gram + gram.T), check_finite=False)
    return np.maximum(ev, 0.0), vecs


def fit_router_fold_dictionary(
    x_train: np.ndarray,
    r_train: np.ndarray,
    x_query: np.ndarray,
    eq_cfg: Optional[EquilibriumConfig] = None,
    placebo_trials: int = 39,
    target_false_admissions_per_fit: float = 0.025,
    seed: int = 460701,
) -> RouterFoldFit:
    """Router-callable single-scope entry point (`PREREG_G1_DRAFT_v1` §8, `N1`, `D4`).

    Fits strictly on `(x_train, r_train)` (`incumbent = 0` since `r_train` is already the router's
    clipped L1 OOF residual), runs nested `G0_8` placebo rebuilding the whole dictionary on each
    permuted residual, equilibrates admitted components, and predicts on `x_query`.
    """
    x_tr = np.ascontiguousarray(x_train, dtype=np.float64)
    r_tr = np.ascontiguousarray(r_train, dtype=np.float64).reshape(-1)
    x_qu = np.ascontiguousarray(x_query, dtype=np.float64)
    n_tr = int(r_tr.shape[0])
    n_qu = int(x_qu.shape[0])
    cfg = eq_cfg or EquilibriumConfig(robust_irls=True)

    res_var = max(float(np.var(r_tr)), 1e-9)
    state = EquilibriumState(
        y_train=r_tr,
        incumbent_train=np.zeros(n_tr, dtype=np.float64),
        incumbent_val=np.zeros(n_qu, dtype=np.float64),
        anchor_h_train=np.ones((n_tr, 1), dtype=np.float64),
        anchor_h_val=np.ones((n_qu, 1), dtype=np.float64),
        tau_h2=max(0.25 * res_var, 1e-6),
        cfg=cfg,
    )
    state.update_robust_noise_irls()

    pre_full = PrecomputedXContext.from_x(x_tr, x_qu)
    pre_tr = pre_full.without_val()
    bank_tr, bank_qu_opt, cand_names = build_production_router_dictionary(
        x_tr, state.alpha, state.inv_cov, x_val=x_qu, precomputed=pre_full
    )
    bank_qu = np.zeros((n_qu, 0), dtype=np.float64) if bank_qu_opt is None else bank_qu_opt
    if bank_tr.shape[1] == 0:
        return RouterFoldFit(
            (), np.zeros(n_qu, dtype=np.float64), cfg.foster_george_c0, 0, state.state_digest(), bank_tr, bank_qu
        )

    def _run_admission_stage(
        b_tr: np.ndarray,
        b_qu: np.ndarray,
        c_names: Sequence[str],
        c0_stage: float,
        admitted_out: List[str],
    ) -> None:
        mu_s = np.mean(b_tr, axis=0)
        sd_s = np.maximum(np.std(b_tr, axis=0), 1e-9)
        z_tr_s = (b_tr - mu_s) / sd_s
        z_qu_s = (b_qu - mu_s) / sd_s
        available = np.ones(z_tr_s.shape[1], dtype=bool)
        while len(state.groups) < cfg.max_active_groups:
            avail_idx = np.flatnonzero(available)
            if avail_idx.size == 0:
                break
            dec = state.screen_dictionary(
                z_tr_s[:, avail_idx],
                [c_names[int(i)] for i in avail_idx],
                cand_val_matrix=z_qu_s[:, avail_idx],
                c0_override=c0_stage,
            )
            for v_sub in dec.vetoed_indices:
                available[int(avail_idx[int(v_sub)])] = False
            orig_idx = int(avail_idx[dec.candidate_index]) if dec.candidate_index >= 0 else -1
            if dec.action == "IN_SPAN_REWEIGHT" and orig_idx >= 0:
                available[orig_idx] = False
                state.reestimate_group_weights_empirical_bayes(n_passes=1)
                continue
            if dec.action != "ADD" or orig_idx < 0:
                break
            available[orig_idx] = False
            state.add_group(
                name=str(c_names[orig_idx]),
                phi_train=z_tr_s[:, orig_idx],
                phi_val=z_qu_s[:, orig_idx],
                weight=dec.optimal_weight,
                family_tag="router_dict",
            )
            admitted_out.append(str(c_names[orig_idx]))
            state.reestimate_group_weights_empirical_bayes(n_passes=2)
            state.update_robust_noise_irls()

    c0_cal = calibrate_nested_discovery_placebo_c0(
        state,
        discovery_fn=lambda null_st: build_production_router_dictionary(
            x_tr, null_st.alpha, null_st.inv_cov, x_val=None, precomputed=pre_tr
        )[0],
        n_trials=placebo_trials,
        target_false_admissions_per_fit=target_false_admissions_per_fit,
        seed=seed,
    )

    admitted: List[str] = []
    _run_admission_stage(bank_tr, bank_qu, cand_names, c0_cal, admitted)

    # Stage-1 orthogonalized refresh: runs ONLY if Stage-0 admitted >= 1 signal component (so H_0
    # pure-noise false-fire rate is 100% invariant!), discovering secondary regimes masked by Stage-0.
    if state.groups and len(state.groups) < cfg.max_active_groups:
        bank_tr2, bank_qu2_opt, cand_names2 = build_production_router_dictionary(
            x_tr, state.alpha, state.inv_cov, x_val=x_qu, precomputed=pre_full
        )
        bank_qu2 = np.zeros((n_qu, 0), dtype=np.float64) if bank_qu2_opt is None else bank_qu2_opt
        if bank_tr2.shape[1] > 0:
            c0_cal2 = calibrate_nested_discovery_placebo_c0(
                state,
                discovery_fn=lambda null_st: build_production_router_dictionary(
                    x_tr, null_st.alpha, null_st.inv_cov, x_val=None, precomputed=pre_tr
                )[0],
                n_trials=placebo_trials,
                target_false_admissions_per_fit=target_false_admissions_per_fit,
                seed=seed + 7919,
            )
            _run_admission_stage(bank_tr2, bank_qu2, cand_names2, c0_cal2, admitted)
            bank_tr = np.hstack([bank_tr, bank_tr2])
            bank_qu = np.hstack([bank_qu, bank_qu2])

    pred_qu = np.zeros(n_qu, dtype=np.float64) if not state.groups else state.raw_residual_prediction_val()
    return RouterFoldFit(
        admitted_components=tuple(admitted),
        query_residual_pred=pred_qu,
        c0_nested=float(c0_cal),
        n_bank_candidates=int(bank_tr.shape[1]),
        state_digest=state.state_digest(),
        bank_train=bank_tr,
        bank_query=bank_qu,
    )


def _fit_inner_cv_ridge_on_bank(
    r_train: np.ndarray,
    bank_train: np.ndarray,
    bank_query: np.ndarray,
    lam_grid: Sequence[float] = (2.0, 10.0, 50.0, 200.0),
) -> np.ndarray:
    """Simultaneous centered L2 ridge with inner 4-fold CV `lambda` for `R2` (`D2`)."""
    n_rows, n_cols = bank_train.shape
    if n_cols == 0 or n_rows < 8:
        return np.zeros(bank_query.shape[0], dtype=np.float64)
    r_mu = float(np.mean(r_train))
    r_c = r_train - r_mu
    b_mu = np.mean(bank_train, axis=0)
    b_c = bank_train - b_mu

    inner_folds = [np.arange(k, n_rows, 4) for k in range(4)]
    all_i = np.arange(n_rows)
    best_lam = float(lam_grid[0])
    best_mse = float("inf")
    for lam in lam_grid:
        oof = np.zeros(n_rows, dtype=np.float64)
        for f_i in inner_folds:
            mask = np.ones(n_rows, dtype=bool)
            mask[f_i] = False
            tr_i = all_i[mask]
            gram = b_c[tr_i].T @ b_c[tr_i] + float(lam) * np.eye(n_cols, dtype=np.float64)
            w = sla.solve(gram, b_c[tr_i].T @ r_c[tr_i], assume_a="pos", check_finite=False)
            oof[f_i] = b_c[f_i] @ w
        mse = float(np.mean((r_c - oof) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_lam = float(lam)

    gram_f = b_c.T @ b_c + best_lam * np.eye(n_cols, dtype=np.float64)
    w_f = sla.solve(gram_f, b_c.T @ r_c, assume_a="pos", check_finite=False)
    return r_mu + (bank_query - b_mu) @ w_f


def fit_router_fold_static_control(
    x_train: np.ndarray,
    r_train: np.ndarray,
    x_query: np.ndarray,
    target_cpu_s: Optional[float] = None,
    fixed_n_members: Optional[int] = None,
    n_rff: int = 96,
    seed: int = 460801,
) -> Tuple[np.ndarray, int, float]:
    """`R3 C_STATIC` equal-compute smooth RFF + linear ridge ensemble (`PREREG_G1_DRAFT_v1` §2).

    Spends genuine numerical ridge-ensemble work (random smooth Fourier features across lengthscales
    + inner-CV ridge solves, zero adaptive dictionary search or covariance equilibrium) matched to
    `target_cpu_s` (`time.process_time()`). Returns `(query_pred, n_members_run, cpu_s_spent)`.
    """
    t0 = time.process_time()
    x_tr = np.ascontiguousarray(x_train, dtype=np.float64)
    r_tr = np.ascontiguousarray(r_train, dtype=np.float64).reshape(-1)
    x_qu = np.ascontiguousarray(x_query, dtype=np.float64)
    n_tr, n_cols = x_tr.shape
    n_qu = x_qu.shape[0]
    if n_tr < 8 or n_cols == 0:
        return np.zeros(n_qu, dtype=np.float64), 0, time.process_time() - t0

    x_mu = np.mean(x_tr, axis=0)
    x_sd = np.maximum(np.std(x_tr, axis=0), 1e-9)
    z_tr = (x_tr - x_mu) / x_sd
    z_qu = (x_qu - x_mu) / x_sd

    pred_acc = np.zeros(n_qu, dtype=np.float64)
    n_done = 0
    scales = (0.7, 1.5, 3.5)
    lam_grid = (4.0, 16.0, 64.0)

    while True:
        rng = np.random.default_rng(int(seed) + 1009 * n_done)
        ell = scales[n_done % len(scales)]
        omega = rng.normal(scale=1.0 / ell, size=(n_cols, int(n_rff)))
        phase = rng.uniform(0.0, 2.0 * math.pi, size=(1, int(n_rff)))
        scale_f = math.sqrt(2.0 / float(n_rff))
        feat_tr = np.hstack([z_tr, scale_f * np.cos(z_tr @ omega + phase)])
        feat_qu = np.hstack([z_qu, scale_f * np.cos(z_qu @ omega + phase)])
        pred_acc += _fit_inner_cv_ridge_on_bank(r_tr, feat_tr, feat_qu, lam_grid=lam_grid)
        n_done += 1

        elapsed = time.process_time() - t0
        if fixed_n_members is not None:
            if n_done >= max(1, int(fixed_n_members)):
                break
        elif target_cpu_s is None:
            if n_done >= 4:
                break
        else:
            if elapsed >= float(target_cpu_s):
                break

    cpu_spent = time.process_time() - t0
    return pred_acc / float(max(n_done, 1)), n_done, cpu_spent


def run_router_ratchet_harness(
    zs_search: np.ndarray,
    za_audit: np.ndarray,
    y_search: np.ndarray,
    lab: np.ndarray,
    p1_oof: np.ndarray,
    p1_aud: np.ndarray,
    x_raw_search: Optional[np.ndarray] = None,
    x_raw_audit: Optional[np.ndarray] = None,
    baseline_search_rmse: Optional[float] = None,
    placebo_trials: int = 39,
    seed: int = 460750,
    include_r2_l3: bool = True,
    include_r3_static: bool = False,
) -> Dict[str, Any]:
    """Run `R0`, `R1`, `R2`, and `R3` in one shared harness (`PREREG_G1_DRAFT_v1` §2–§3, `D1–D4`)."""
    zs = np.ascontiguousarray(zs_search, dtype=np.float64)
    za = np.ascontiguousarray(za_audit, dtype=np.float64)
    xs_dict = zs if x_raw_search is None else np.ascontiguousarray(x_raw_search, dtype=np.float64)
    xa_dict = za if x_raw_audit is None else np.ascontiguousarray(x_raw_audit, dtype=np.float64)
    y = np.ascontiguousarray(y_search, dtype=np.float64).reshape(-1)
    lab_arr = np.asarray(lab, dtype=np.int64).reshape(-1)
    p1_o = np.ascontiguousarray(p1_oof, dtype=np.float64).reshape(-1)
    p1_a = np.ascontiguousarray(p1_aud, dtype=np.float64).reshape(-1)

    r_raw = y - p1_o
    s_base = float(np.sqrt(np.mean(r_raw * r_raw)))
    sy = max(float(np.std(y)), 1e-9)
    search_oof_u_p1 = (
        None
        if baseline_search_rmse is None or baseline_search_rmse <= 0.0
        else float(math.log(float(baseline_search_rmse) / max(s_base, 1e-15)))
    )
    is_precision_proxy = None if search_oof_u_p1 is None else bool(search_oof_u_p1 >= 2.0)

    def _zero_arm(grid_len: int, cpu_s: float = 0.0, **extra: Any) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "best_cfg": None,
            "best_gain": 0.0,
            "dictionary_fired": False,
            "fired": False,
            "grid_size": grid_len,
            "pred_aud": p1_a.copy(),
            "pred_aud_unclamped": p1_a.copy(),
            "worker_cpu_s": round(cpu_s, 6),
        }
        out.update(extra)
        return out

    if s_base < 1e-5 * sy or len(y) > 3000:
        return {
            "R0": _zero_arm(len(ROUTER_R0_CONFIG_GRID)),
            "R1": _zero_arm(len(ROUTER_R1_CONFIG_GRID), fold_admissions=0),
            "R2": _zero_arm(len(ROUTER_R2_L3_CONFIG_GRID)),
            "R3": _zero_arm(len(ROUTER_R3_STATIC_CONFIG_GRID), static_members_total=0),
            "changed_r1_vs_r0": False,
            "is_precision_search_proxy": is_precision_proxy,
            "search_oof_rmse_p1": s_base,
            "search_oof_u_p1": search_oof_u_p1,
        }

    base_f = np.array([float(np.sqrt(np.mean(r_raw[lab_arr == k] ** 2))) for k in range(5)])
    if np.any(base_f < 1e-12):
        return {
            "R0": _zero_arm(len(ROUTER_R0_CONFIG_GRID)),
            "R1": _zero_arm(len(ROUTER_R1_CONFIG_GRID), fold_admissions=0),
            "R2": _zero_arm(len(ROUTER_R2_L3_CONFIG_GRID)),
            "R3": _zero_arm(len(ROUTER_R3_STATIC_CONFIG_GRID), static_members_total=0),
            "changed_r1_vs_r0": False,
            "is_precision_search_proxy": is_precision_proxy,
            "search_oof_rmse_p1": s_base,
            "search_oof_u_p1": search_oof_u_p1,
        }

    mad_r = max(1.4826 * float(np.median(np.abs(r_raw - np.median(r_raw)))), 1e-6 * sy)
    r_train = np.clip(r_raw, -3.0 * mad_r, 3.0 * mad_r)

    # 1. Shared 27-config Matern-3/2 OOF sweep (`R0` base)
    t_r0_start = time.process_time()
    r0_best_gain = 0.0
    r0_best_cfg: Optional[Tuple[str, float, float, float]] = None
    for ell in ROUTER_ELL_GRID:
        k_ss = _matern32_gram(zs, zs, ell)
        fold_eigh = []
        for k in range(5):
            va = lab_arr == k
            tr = ~va
            ev, v_mat = _safe_eigh(k_ss[np.ix_(tr, tr)])
            vt_r = v_mat.T @ r_train[tr]
            kv_va = k_ss[np.ix_(va, tr)] @ v_mat
            fold_eigh.append((va, ev, vt_r, kv_va))
        for lam in ROUTER_LAM_GRID:
            oof_r = np.zeros_like(r_raw)
            for va, ev, vt_r, kv_va in fold_eigh:
                oof_r[va] = kv_va @ (vt_r / (ev + lam))
            for eta in ROUTER_ETA_GRID:
                cand_oof = p1_o + eta * oof_r
                new_f = np.array(
                    [float(np.sqrt(np.mean((y[lab_arr == k] - cand_oof[lab_arr == k]) ** 2))) for k in range(5)]
                )
                passed_r, gain = _passes_fold_ratchet(base_f, new_f)
                if passed_r and gain > r0_best_gain:
                    r0_best_gain = gain
                    r0_best_cfg = ("matern32", ell, lam, eta)

    def _predict_matern_cfg(cfg_t: Tuple[str, float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
        _, ell, lam, eta = cfg_t
        k_ss = _matern32_gram(zs, zs, ell)
        k_as = _matern32_gram(za, zs, ell)
        r_bag = np.zeros(len(p1_a), dtype=np.float64)
        for k in range(5):
            tr = lab_arr != k
            ev_k, v_k = _safe_eigh(k_ss[np.ix_(tr, tr)])
            alpha_k = v_k @ ((v_k.T @ r_train[tr]) / (ev_k + lam))
            r_bag += (k_as[:, tr] @ alpha_k) / 5.0
        ev_f, v_f = _safe_eigh(k_ss)
        alpha_f = v_f @ ((v_f.T @ r_train) / (ev_f + lam))
        r_full = k_as @ alpha_f
        r_raw_blend = 0.5 * r_bag + 0.5 * r_full
        r_clip = np.clip(r_raw_blend, -2.5 * mad_r, 2.5 * mad_r)
        return p1_a + eta * r_clip, p1_a + eta * r_raw_blend

    if r0_best_cfg is None or r0_best_gain < 0.004:
        r0_cpu_s = time.process_time() - t_r0_start
        r0_out = _zero_arm(len(ROUTER_R0_CONFIG_GRID), cpu_s=r0_cpu_s)
    else:
        r0_pred_c, r0_pred_u = _predict_matern_cfg(r0_best_cfg)
        r0_cpu_s = time.process_time() - t_r0_start
        r0_out = {
            "best_cfg": r0_best_cfg,
            "best_gain": r0_best_gain,
            "dictionary_fired": False,
            "fired": True,
            "grid_size": len(ROUTER_R0_CONFIG_GRID),
            "pred_aud": r0_pred_c,
            "pred_aud_unclamped": r0_pred_u,
            "worker_cpu_s": round(r0_cpu_s, 6),
        }

    # 2. Nested fold-internal dictionary discovery (`N1`) for R1 & R2
    t_r1_extra_start = time.process_time()
    oof_dict = np.zeros_like(r_raw)
    fold_fits_aud: List[RouterFoldFit] = []
    fold_banks: List[Tuple[np.ndarray, np.ndarray, int]] = []
    fold_admissions = 0

    for k in range(5):
        va = lab_arr == k
        tr = ~va
        x_qu_both = np.vstack([xs_dict[va], xa_dict])
        n_va = int(np.sum(va))
        ffit = fit_router_fold_dictionary(
            xs_dict[tr],
            r_train[tr],
            x_qu_both,
            placebo_trials=placebo_trials,
            seed=seed + 10 * k,
        )
        oof_dict[va] = ffit.query_residual_pred[:n_va]
        fold_fits_aud.append(
            dataclasses.replace(
                ffit,
                query_residual_pred=ffit.query_residual_pred[n_va:],
                bank_query=ffit.bank_query[n_va:],
            )
        )
        fold_banks.append((ffit.bank_train, ffit.bank_query, n_va))
        if ffit.admitted_components:
            fold_admissions += 1

    r1_best_gain = r0_best_gain
    r1_best_cfg = r0_best_cfg
    for eta in ROUTER_ETA_GRID:
        cand_oof = p1_o + eta * oof_dict
        new_f = np.array(
            [float(np.sqrt(np.mean((y[lab_arr == k] - cand_oof[lab_arr == k]) ** 2))) for k in range(5)]
        )
        passed_r, gain = _passes_fold_ratchet(base_f, new_f)
        if passed_r and gain > r1_best_gain:
            r1_best_gain = gain
            r1_best_cfg = ("dict_eq", 0.0, 0.0, eta)

    full_fit_cached: Optional[RouterFoldFit] = None
    if r1_best_cfg is None or r1_best_gain < 0.004:
        r1_cpu_s = r0_cpu_s + (time.process_time() - t_r1_extra_start)
        r1_out = _zero_arm(len(ROUTER_R1_CONFIG_GRID), cpu_s=r1_cpu_s, fold_admissions=fold_admissions)
    elif r1_best_cfg[0] == "matern32":
        if r1_best_cfg == r0_best_cfg and r0_out["fired"]:
            pred_c, pred_u = r0_out["pred_aud"].copy(), r0_out["pred_aud_unclamped"].copy()
        else:
            pred_c, pred_u = _predict_matern_cfg(r1_best_cfg)
        r1_cpu_s = r0_cpu_s + (time.process_time() - t_r1_extra_start)
        r1_out = {
            "best_cfg": r1_best_cfg,
            "best_gain": r1_best_gain,
            "dictionary_fired": False,
            "fired": True,
            "fold_admissions": fold_admissions,
            "grid_size": len(ROUTER_R1_CONFIG_GRID),
            "pred_aud": pred_c,
            "pred_aud_unclamped": pred_u,
            "worker_cpu_s": round(r1_cpu_s, 6),
        }
    else:
        eta = r1_best_cfg[3]
        r_bag = np.mean(np.column_stack([ff.query_residual_pred for ff in fold_fits_aud]), axis=1)
        full_fit_cached = fit_router_fold_dictionary(
            xs_dict, r_train, xa_dict, placebo_trials=placebo_trials, seed=seed + 99
        )
        r_full = full_fit_cached.query_residual_pred
        r_raw_blend = 0.5 * r_bag + 0.5 * r_full
        r_clip = np.clip(r_raw_blend, -2.5 * mad_r, 2.5 * mad_r)
        r1_cpu_s = r0_cpu_s + (time.process_time() - t_r1_extra_start)
        r1_out = {
            "admitted_full": list(full_fit_cached.admitted_components),
            "best_cfg": r1_best_cfg,
            "best_gain": r1_best_gain,
            "dictionary_fired": True,
            "fired": True,
            "fold_admissions": fold_admissions,
            "grid_size": len(ROUTER_R1_CONFIG_GRID),
            "pred_aud": p1_a + eta * r_clip,
            "pred_aud_unclamped": p1_a + eta * r_raw_blend,
            "worker_cpu_s": round(r1_cpu_s, 6),
        }

    # 3. R2 (`ROUTER_L3`: dense ridge on the same fold-discovered banks in the identical harness)
    if not include_r2_l3:
        r2_out = _zero_arm(len(ROUTER_R2_L3_CONFIG_GRID))
    else:
        t_r2_start = time.process_time()
        oof_l3 = np.zeros_like(r_raw)
        fold_l3_aud: List[np.ndarray] = []
        for k in range(5):
            va = lab_arr == k
            tr = ~va
            b_tr, b_qu_both, n_va = fold_banks[k]
            l3_both = _fit_inner_cv_ridge_on_bank(r_train[tr], b_tr, b_qu_both)
            oof_l3[va] = l3_both[:n_va]
            fold_l3_aud.append(l3_both[n_va:])

        r2_best_gain = r0_best_gain
        r2_best_cfg = r0_best_cfg
        for eta in ROUTER_ETA_GRID:
            cand_oof = p1_o + eta * oof_l3
            new_f = np.array(
                [float(np.sqrt(np.mean((y[lab_arr == k] - cand_oof[lab_arr == k]) ** 2))) for k in range(5)]
            )
            passed_r, gain = _passes_fold_ratchet(base_f, new_f)
            if passed_r and gain > r2_best_gain:
                r2_best_gain = gain
                r2_best_cfg = ("l3_ridge", 0.0, 0.0, eta)

        if r2_best_cfg is None or r2_best_gain < 0.004:
            r2_out = _zero_arm(len(ROUTER_R2_L3_CONFIG_GRID), cpu_s=r0_cpu_s + (time.process_time() - t_r2_start))
        elif r2_best_cfg[0] == "matern32":
            pred_c2, pred_u2 = (
                (r0_out["pred_aud"].copy(), r0_out["pred_aud_unclamped"].copy())
                if r2_best_cfg == r0_best_cfg and r0_out["fired"]
                else _predict_matern_cfg(r2_best_cfg)
            )
            r2_out = {
                "best_cfg": r2_best_cfg,
                "best_gain": r2_best_gain,
                "dictionary_fired": False,
                "fired": True,
                "grid_size": len(ROUTER_R2_L3_CONFIG_GRID),
                "pred_aud": pred_c2,
                "pred_aud_unclamped": pred_u2,
                "worker_cpu_s": round(r0_cpu_s + (time.process_time() - t_r2_start), 6),
            }
        else:
            eta2 = r2_best_cfg[3]
            r_bag2 = np.mean(np.column_stack(fold_l3_aud), axis=1)
            full_b = (
                full_fit_cached
                if full_fit_cached is not None
                else fit_router_fold_dictionary(xs_dict, r_train, xa_dict, placebo_trials=placebo_trials, seed=seed + 99)
            )
            r_full2 = _fit_inner_cv_ridge_on_bank(r_train, full_b.bank_train, full_b.bank_query)
            r_raw2 = 0.5 * r_bag2 + 0.5 * r_full2
            r_clip2 = np.clip(r_raw2, -2.5 * mad_r, 2.5 * mad_r)
            r2_out = {
                "best_cfg": r2_best_cfg,
                "best_gain": r2_best_gain,
                "dictionary_fired": True,
                "fired": True,
                "grid_size": len(ROUTER_R2_L3_CONFIG_GRID),
                "pred_aud": p1_a + eta2 * r_clip2,
                "pred_aud_unclamped": p1_a + eta2 * r_raw2,
                "worker_cpu_s": round(r0_cpu_s + (time.process_time() - t_r2_start), 6),
            }

    # 4. R3 (`C_STATIC`: one-sided >= 0.9 * R1 CPU-s per `cl_fwriter` 05:04Z Q5, interleaved per context)
    if not include_r3_static:
        r3_out = _zero_arm(len(ROUTER_R3_STATIC_CONFIG_GRID), static_members_total=0)
    else:
        t_r3_extra_start = time.process_time()
        extra_budget = max(0.01, 1.00 * (r1_cpu_s - r0_cpu_s))
        oof_st = np.zeros_like(r_raw)
        fold_st_aud: List[np.ndarray] = []
        total_members = 0
        # Reserve 5/6 of extra_budget across the 5 folds and 1/6 for full-SEARCH fit so total CPU matches R1!
        for k in range(5):
            va = lab_arr == k
            tr = ~va
            x_qu_both = np.vstack([xs_dict[va], xa_dict])
            n_va = int(np.sum(va))
            spent_so_far = time.process_time() - t_r3_extra_start
            target_cum_k = (float(k + 1) / 6.0) * extra_budget
            fold_target = max(0.002, target_cum_k - spent_so_far)
            st_pred, n_mem, _ = fit_router_fold_static_control(
                xs_dict[tr], r_train[tr], x_qu_both, target_cpu_s=fold_target, seed=seed + 500 + 13 * k
            )
            oof_st[va] = st_pred[:n_va]
            fold_st_aud.append(st_pred[n_va:])
            total_members += n_mem

        # Always run the 6th (full-SEARCH) static ensemble fit using the remaining 1/6 budget so
        # R3's per-context worker CPU-seconds matches R1 within ±10% regardless of ratchet outcome!
        rem_full = max(0.002, extra_budget - (time.process_time() - t_r3_extra_start))
        r_full3, n_mem_full, _ = fit_router_fold_static_control(
            xs_dict, r_train, xa_dict, target_cpu_s=rem_full, seed=seed + 599
        )
        total_members += n_mem_full

        r3_best_gain = r0_best_gain
        r3_best_cfg = r0_best_cfg
        for eta in ROUTER_ETA_GRID:
            cand_oof = p1_o + eta * oof_st
            new_f = np.array(
                [float(np.sqrt(np.mean((y[lab_arr == k] - cand_oof[lab_arr == k]) ** 2))) for k in range(5)]
            )
            passed_r, gain = _passes_fold_ratchet(base_f, new_f)
            if passed_r and gain > r3_best_gain:
                r3_best_gain = gain
                r3_best_cfg = ("c_static", 0.0, 0.0, eta)

        r3_cpu_s = r0_cpu_s + (time.process_time() - t_r3_extra_start)
        if r3_best_cfg is None or r3_best_gain < 0.004:
            r3_out = _zero_arm(
                len(ROUTER_R3_STATIC_CONFIG_GRID), cpu_s=r3_cpu_s, static_members_total=total_members
            )
        elif r3_best_cfg[0] == "matern32":
            pred_c3, pred_u3 = (
                (r0_out["pred_aud"].copy(), r0_out["pred_aud_unclamped"].copy())
                if r3_best_cfg == r0_best_cfg and r0_out["fired"]
                else _predict_matern_cfg(r3_best_cfg)
            )
            r3_out = {
                "best_cfg": r3_best_cfg,
                "best_gain": r3_best_gain,
                "dictionary_fired": False,
                "fired": True,
                "grid_size": len(ROUTER_R3_STATIC_CONFIG_GRID),
                "pred_aud": pred_c3,
                "pred_aud_unclamped": pred_u3,
                "static_members_total": total_members,
                "worker_cpu_s": round(r3_cpu_s, 6),
            }
        else:
            eta3 = r3_best_cfg[3]
            r_bag3 = np.mean(np.column_stack(fold_st_aud), axis=1)
            r_raw3 = 0.5 * r_bag3 + 0.5 * r_full3
            r_clip3 = np.clip(r_raw3, -2.5 * mad_r, 2.5 * mad_r)
            r3_out = {
                "best_cfg": r3_best_cfg,
                "best_gain": r3_best_gain,
                "dictionary_fired": False,
                "fired": True,
                "grid_size": len(ROUTER_R3_STATIC_CONFIG_GRID),
                "pred_aud": p1_a + eta3 * r_clip3,
                "pred_aud_unclamped": p1_a + eta3 * r_raw3,
                "static_members_total": total_members,
                "worker_cpu_s": round(r3_cpu_s, 6),
            }

    changed_r1_vs_r0 = not np.allclose(r1_out["pred_aud"], r0_out["pred_aud"], atol=1e-12, rtol=0.0)
    return {
        "R0": r0_out,
        "R1": r1_out,
        "R2": r2_out,
        "R3": r3_out,
        "changed_r1_vs_r0": bool(changed_r1_vs_r0),
        "is_precision_search_proxy": is_precision_proxy,
        "search_oof_rmse_p1": s_base,
        "search_oof_u_p1": search_oof_u_p1,
    }
