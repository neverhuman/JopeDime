"""Gate G1 Fixed-Bank & Nested Router-Dictionary Benchmark (`replay_g1_s9_pool.py`).

Incorporates all `cl_fwriter` (`04:34Z`, `04:36Z`, `04:37Z`) & `cl_mtc` (`04:32Z`, `04:37Z`) gates:
  - `Finding #1` & `N6`: Real `.npz` bank loader derives `"board"` strictly from verified on-disk
    inputs (`manifest.json` / `.npz` `board_id` + per-file SHA-256s + `context_ids`), NEVER from a
    CLI flag. Unconditionally refuses `p201e` (`P201e` has no SEARCH/AUDIT bank per `cl_mtc`
    `04:37Z`) and refuses `p201b_s9_80` / `p201b_descriptive` without `--s9-selection-receipt`.
  - `Finding #2` & `N1`: Fixed-bank mode requires certified out-of-fold (`cand_is_oof`) columns;
    nested router-dictionary mode (`solve_nested_router_dictionary`) reruns the ENTIRE production
    discovery (`build_production_router_dictionary`) + `calibrate_nested_discovery_placebo_c0` +
    EB weights + IRLS inside EVERY inner fold.
  - `Finding #3` & `N3`: Non-synthetic runs require an explicit equal-compute `C_static` predictor
    with positive `static_ctrl_compute_s`, evaluated alongside `L0` (incumbent / switch-off V0)
    and `L3` (simultaneous dense ridge with inner-CV `lambda`).
  - `Finding #4` & `N4`: Identical output clamp (`mad3` or `search_iqr`) across every arm plus
    unclamped predictions reported alongside.
  - `Finding #5` & `N2`: Damping `eta`, `0.004` relative-RMSE floor, and `4-of-5` fold ratchet are
    selected strictly from nested out-of-fold predictions.
  - `Finding #7` & `N5`: `FitScopeV2` hashes `bank_*`, `incumbent_*`, and `static_ctrl_val`; loader
    never reads `y_val` / `ya` / `y_audit` from `.npz` files (`max_abs_diff == 0.0` under `+1000`
    on-disk AUDIT label poisoning).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import linalg as sla

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from research.flywheel_selfplay.h460_apex_eq.eq_core import (
        EquilibriumConfig,
        EquilibriumState,
        apply_mad3_clamp,
        apply_search_range_clamp,
        calibrate_nested_discovery_placebo_c0,
        calibrate_placebo_c0,
    )
    from research.flywheel_selfplay.h460_apex_eq.fit_closure import FitScopeV2, learn_whole_procedure
    from research.flywheel_selfplay.h460_apex_eq.jacobi_operators import (
        PrecomputedXContext,
        build_production_router_dictionary,
    )
else:
    from .eq_core import (
        EquilibriumConfig,
        EquilibriumState,
        apply_mad3_clamp,
        apply_search_range_clamp,
        calibrate_nested_discovery_placebo_c0,
        calibrate_placebo_c0,
    )
    from .fit_closure import FitScopeV2, learn_whole_procedure
    from .jacobi_operators import PrecomputedXContext, build_production_router_dictionary

FORBIDDEN_PATH_TOKENS = (
    "confirm",
    "holdout",
    "sealed",
    "dl-val",
    "dl_val",
    "fmgap_b",
    "fmgap_c",
    "probe_ya",
)


def assert_clean_train_path(path: Path) -> None:
    """Refuse any sealed, confirm, or holdout evaluation artifact path."""
    low = str(path).lower()
    for tok in FORBIDDEN_PATH_TOKENS:
        if tok in low:
            raise ValueError(f"Quarantine refusal: forbidden token '{tok}' in path {path}")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _apply_arm_clamp(
    preds: np.ndarray,
    y_train: np.ndarray,
    clamp_mode: str,
    iqr_mult: float = 1.5,
) -> np.ndarray:
    if clamp_mode == "none":
        return np.asarray(preds, dtype=np.float64).copy()
    if clamp_mode == "mad3":
        return apply_mad3_clamp(preds, y_train)
    return apply_search_range_clamp(preds, y_train, iqr_mult=iqr_mult)


def make_deterministic_folds(n_rows: int, n_folds: int = 5, seed: int = 460101) -> List[np.ndarray]:
    """Deterministic row partition for inner K-fold nested discovery and damping selection."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(int(n_rows))
    return [np.sort(chunk) for chunk in np.array_split(perm, max(2, int(n_folds)))]


def _fit_sequential_greedy_l2(
    y_train: np.ndarray,
    inc_train: np.ndarray,
    inc_val: np.ndarray,
    bank_train: np.ndarray,
    bank_val: np.ndarray,
    max_steps: int = 8,
    shrinkage: float = 0.7,
    ridge_pen: float = 5.0,
) -> np.ndarray:
    """`L2`: Serial greedy residual boosting (diagnostic only; does not satisfy `N3` alone)."""
    raw_res = y_train - inc_train
    bias = float(np.mean(raw_res))
    res = raw_res - bias
    pred_val = inc_val + bias
    if bank_train.shape[1] == 0:
        return pred_val
    b_mu = np.mean(bank_train, axis=0)
    b_tr = bank_train - b_mu
    b_va = bank_val - b_mu
    used: set[int] = set()
    n_cands = b_tr.shape[1]

    for _ in range(min(int(max_steps), n_cands)):
        best_idx = -1
        best_corr = 0.0
        for j in range(n_cands):
            if j in used:
                continue
            col = b_tr[:, j]
            denom = float(np.dot(col, col)) + float(ridge_pen)
            score = (float(np.dot(col, res)) ** 2) / max(denom, 1e-12)
            if score > best_corr:
                best_corr = score
                best_idx = j
        if best_idx < 0 or best_corr <= 1e-12:
            break
        used.add(best_idx)
        col_b = b_tr[:, best_idx]
        coef = float(shrinkage) * float(np.dot(col_b, res)) / (float(np.dot(col_b, col_b)) + float(ridge_pen))
        res = res - coef * col_b
        pred_val = pred_val + coef * b_va[:, best_idx]
    return pred_val


def _fit_simultaneous_ridge_l3(
    y_train: np.ndarray,
    inc_train: np.ndarray,
    inc_val: np.ndarray,
    bank_train: np.ndarray,
    bank_val: np.ndarray,
    folds: Sequence[np.ndarray],
    lam_grid: Sequence[float] = (0.5, 2.0, 10.0, 50.0, 200.0),
) -> np.ndarray:
    """`L3`: Simultaneous centered L2 Ridge over all bank columns with inner CV `lambda` (`N3`)."""
    raw_res = y_train - inc_train
    n_rows, n_cands = bank_train.shape
    if n_cands == 0:
        return inc_val + float(np.mean(raw_res))

    best_lam = float(lam_grid[0])
    best_mse = float("inf")
    all_idx = np.arange(n_rows)
    for lam in lam_grid:
        oof = np.zeros(n_rows, dtype=np.float64)
        for f_idx in folds:
            tr_mask = np.ones(n_rows, dtype=bool)
            tr_mask[f_idx] = False
            tr_i = all_idx[tr_mask]
            r_mu = float(np.mean(raw_res[tr_i]))
            b_mu = np.mean(bank_train[tr_i], axis=0)
            b_tr = bank_train[tr_i] - b_mu
            gram = b_tr.T @ b_tr + float(lam) * np.eye(n_cands, dtype=np.float64)
            w_hat = sla.solve(gram, b_tr.T @ (raw_res[tr_i] - r_mu), assume_a="pos", check_finite=False)
            oof[f_idx] = r_mu + (bank_train[f_idx] - b_mu) @ w_hat
        mse = float(np.mean((raw_res - oof) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_lam = float(lam)

    r_mu_full = float(np.mean(raw_res))
    b_mu_full = np.mean(bank_train, axis=0)
    b_tr_full = bank_train - b_mu_full
    gram_full = b_tr_full.T @ b_tr_full + best_lam * np.eye(n_cands, dtype=np.float64)
    w_full = sla.solve(gram_full, b_tr_full.T @ (raw_res - r_mu_full), assume_a="pos", check_finite=False)
    return inc_val + r_mu_full + ((bank_val - b_mu_full) @ w_full)


def _admit_and_equilibrate(
    state: EquilibriumState,
    bank_train: np.ndarray,
    bank_val: np.ndarray,
    cand_names: Sequence[str],
    c0_cal: float,
) -> Tuple[List[str], int]:
    """Run greedy admission + QR novelty + EB weight re-estimation + IRLS on standardized bank."""
    if bank_train.shape[1] == 0:
        return [], 0
    mu = np.mean(bank_train, axis=0)
    sd = np.maximum(np.std(bank_train, axis=0), 1e-9)
    z_tr = (bank_train - mu) / sd
    z_va = (bank_val - mu) / sd

    admitted: List[str] = []
    reweighted_in_span = 0
    available = np.ones(z_tr.shape[1], dtype=bool)

    for _ in range(state.cfg.max_active_groups):
        avail_idx = np.flatnonzero(available)
        if avail_idx.size == 0:
            break
        sub_names = [cand_names[int(i)] for i in avail_idx]
        dec = state.screen_dictionary(
            z_tr[:, avail_idx],
            sub_names,
            cand_val_matrix=z_va[:, avail_idx],
            c0_override=c0_cal,
        )
        for v_sub in dec.vetoed_indices:
            available[int(avail_idx[int(v_sub)])] = False
        orig_idx = int(avail_idx[dec.candidate_index]) if dec.candidate_index >= 0 else -1
        if dec.action == "IN_SPAN_REWEIGHT" and orig_idx >= 0:
            available[orig_idx] = False
            reweighted_in_span += 1
            state.reestimate_group_weights_empirical_bayes(n_passes=1)
            continue
        if dec.action != "ADD" or orig_idx < 0:
            break
        available[orig_idx] = False
        state.add_group(
            name=str(cand_names[orig_idx]),
            phi_train=z_tr[:, orig_idx],
            phi_val=z_va[:, orig_idx],
            weight=dec.optimal_weight,
            family_tag="bank_col",
        )
        admitted.append(str(cand_names[orig_idx]))
        state.reestimate_group_weights_empirical_bayes(n_passes=2)
        state.update_robust_noise_irls()

    return admitted, reweighted_in_span


def _fit_apex_eq_single_pass(
    y_train: np.ndarray,
    inc_train: np.ndarray,
    inc_val: np.ndarray,
    bank_train: np.ndarray,
    bank_val: np.ndarray,
    cand_names: Sequence[str],
    eq_cfg: EquilibriumConfig,
    calibrate_placebo: bool = True,
    placebo_trials: int = 80,
    seed: int = 460201,
) -> Tuple[EquilibriumState, List[str], int, float]:
    """Single-scope `APEX-EQ` pass over a genuinely fixed, label-free OOF bank."""
    n_tr = int(y_train.shape[0])
    n_va = int(inc_val.shape[0])
    res_var = max(float(np.var(y_train - inc_train)), 1e-6)
    state = EquilibriumState(
        y_train=y_train,
        incumbent_train=inc_train,
        incumbent_val=inc_val,
        anchor_h_train=np.ones((n_tr, 1), dtype=np.float64),
        anchor_h_val=np.ones((n_va, 1), dtype=np.float64),
        tau_h2=max(1.0, 4.0 * res_var),
        cfg=eq_cfg,
    )
    state.update_robust_noise_irls()

    mu = np.mean(bank_train, axis=0) if bank_train.shape[1] > 0 else np.zeros(0)
    sd = np.maximum(np.std(bank_train, axis=0), 1e-9) if bank_train.shape[1] > 0 else np.ones(0)
    z_tr = (bank_train - mu) / sd if bank_train.shape[1] > 0 else bank_train

    c0_cal = (
        calibrate_placebo_c0(state, z_tr, n_trials=placebo_trials, seed=seed)
        if calibrate_placebo and z_tr.shape[1] > 0
        else eq_cfg.foster_george_c0
    )
    admitted, reweighted_in_span = _admit_and_equilibrate(
        state, bank_train, bank_val, cand_names, c0_cal=c0_cal
    )
    return state, admitted, reweighted_in_span, float(c0_cal)


def _fit_nested_discovery_single_scope(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    inc_train: np.ndarray,
    inc_val: np.ndarray,
    eq_cfg: EquilibriumConfig,
    placebo_trials: int = 49,
    target_false_admissions_per_fit: float = 0.025,
    seed: int = 460401,
) -> Tuple[EquilibriumState, np.ndarray, np.ndarray, List[str], List[str], float]:
    """Build production dictionary on `(x_train, y_train)` + nested `G0_8` placebo + equilibrate."""
    n_tr = int(y_train.shape[0])
    n_va = int(inc_val.shape[0])
    res_var = max(float(np.var(y_train - inc_train)), 1e-6)
    state = EquilibriumState(
        y_train=y_train,
        incumbent_train=inc_train,
        incumbent_val=inc_val,
        anchor_h_train=np.ones((n_tr, 1), dtype=np.float64),
        anchor_h_val=np.ones((n_va, 1), dtype=np.float64),
        tau_h2=max(1.0, 4.0 * res_var),
        cfg=eq_cfg,
    )
    state.update_robust_noise_irls()

    pre_full = PrecomputedXContext.from_x(x_train, x_val)
    pre_tr_only = pre_full.without_val()
    bank_tr, bank_va_opt, cand_names = build_production_router_dictionary(
        x_train, state.alpha, state.inv_cov, x_val=x_val, precomputed=pre_full
    )
    bank_va = np.zeros((n_va, 0), dtype=np.float64) if bank_va_opt is None else bank_va_opt

    c0_cal = calibrate_nested_discovery_placebo_c0(
        state,
        discovery_fn=lambda null_st: build_production_router_dictionary(
            x_train, null_st.alpha, null_st.inv_cov, x_val=None, precomputed=pre_tr_only
        )[0],
        n_trials=placebo_trials,
        target_false_admissions_per_fit=target_false_admissions_per_fit,
        seed=seed,
    )
    admitted, _ = _admit_and_equilibrate(state, bank_tr, bank_va, cand_names, c0_cal=c0_cal)
    if state.groups and len(state.groups) < eq_cfg.max_active_groups:
        bank_tr2, bank_va2_opt, cand_names2 = build_production_router_dictionary(
            x_train, state.alpha, state.inv_cov, x_val=x_val, precomputed=pre_full
        )
        bank_va2 = np.zeros((n_va, 0), dtype=np.float64) if bank_va2_opt is None else bank_va2_opt
        if bank_tr2.shape[1] > 0:
            c0_cal2 = calibrate_nested_discovery_placebo_c0(
                state,
                discovery_fn=lambda null_st: build_production_router_dictionary(
                    x_train, null_st.alpha, null_st.inv_cov, x_val=None, precomputed=pre_tr_only
                )[0],
                n_trials=placebo_trials,
                target_false_admissions_per_fit=target_false_admissions_per_fit,
                seed=seed + 7919,
            )
            adm2, _ = _admit_and_equilibrate(state, bank_tr2, bank_va2, cand_names2, c0_cal=c0_cal2)
            admitted.extend(adm2)
            bank_tr = np.hstack([bank_tr, bank_tr2])
            bank_va = np.hstack([bank_va, bank_va2])
            cand_names = list(cand_names) + list(cand_names2)
    return state, bank_tr, bank_va, cand_names, admitted, float(c0_cal)


def solve_nested_router_dictionary(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    incumbent_train: np.ndarray,
    incumbent_val: np.ndarray,
    static_ctrl_val: Optional[np.ndarray] = None,
    static_ctrl_compute_s: Optional[float] = None,
    require_c_static: bool = False,
    clamp_mode: str = "mad3",
    cfg: Optional[EquilibriumConfig] = None,
    placebo_trials: int = 49,
    fold_placebo_trials: int = 39,
    seed: int = 460501,
) -> Dict[str, Any]:
    """End-to-end nested router-dictionary & `JAC_V1` solver satisfying `G0_8` and `N1-N5`.

    - `N1`: Reruns `build_production_router_dictionary` + `calibrate_nested_discovery_placebo_c0`
      + admission + EB weights + IRLS inside EVERY inner fold on fold-training rows only.
    - `N2`: Selects `eta`, `min_rel_rmse_gain` (`0.004`), and `min_improved_folds` (`4` of `5`)
      strictly on those nested OOF predictions (`eta == 0` -> bit-identical `incumbent_val`).
    - `N3` & `N4`: Evaluates `L0`, `L1_c_static`, `L3_dense_ridge`, and `L4_nested_router` with
      both identical output clamping and unclamped outputs.
    """
    if require_c_static:
        if static_ctrl_val is None or static_ctrl_compute_s is None or float(static_ctrl_compute_s) <= 0.0:
            raise ValueError(
                "G1 refusal (Finding #3 / N3): non-synthetic G1 requires an explicit equal-compute "
                "C_static column with positive static_ctrl_compute_s."
            )

    inc_tr_ro = np.array(incumbent_train, dtype=np.float64, copy=True).reshape(-1)
    inc_va_ro = np.array(incumbent_val, dtype=np.float64, copy=True).reshape(-1)
    c_stat_ro = (
        inc_va_ro.copy()
        if static_ctrl_val is None
        else np.array(static_ctrl_val, dtype=np.float64, copy=True).reshape(-1)
    )
    for arr in (inc_tr_ro, inc_va_ro, c_stat_ro):
        arr.setflags(write=False)

    extra_map = {
        "incumbent_train": inc_tr_ro,
        "incumbent_val": inc_va_ro,
        "static_ctrl_val": c_stat_ro,
    }
    router_cfg = cfg or EquilibriumConfig(min_improved_folds=4, min_rel_rmse_gain=0.004)

    def _learner(
        x_tr: np.ndarray,
        y_tr: np.ndarray,
        x_va: np.ndarray,
        scope: FitScopeV2,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        folds = make_deterministic_folds(y_tr.shape[0], n_folds=5, seed=seed)
        n_rows = y_tr.shape[0]
        all_idx = np.arange(n_rows)
        oof_r_pred = np.zeros(n_rows, dtype=np.float64)

        for f_k, f_idx in enumerate(folds):
            if len(f_idx) == 0:
                continue
            tr_mask = np.ones(n_rows, dtype=bool)
            tr_mask[f_idx] = False
            tr_i = all_idx[tr_mask]
            fold_st, _, _, _, _, _ = _fit_nested_discovery_single_scope(
                x_train=x_tr[tr_i],
                y_train=y_tr[tr_i],
                x_val=x_tr[f_idx],
                inc_train=inc_tr_ro[tr_i],
                inc_val=inc_tr_ro[f_idx],
                eq_cfg=router_cfg,
                placebo_trials=fold_placebo_trials,
                target_false_admissions_per_fit=0.025,
                seed=seed + 100 + f_k,
            )
            oof_r_pred[f_idx] = fold_st.raw_residual_prediction_val()

        full_st, bank_tr, bank_va, _, admitted, c0_cal = _fit_nested_discovery_single_scope(
            x_train=x_tr,
            y_train=y_tr,
            x_val=x_va,
            inc_train=inc_tr_ro,
            inc_val=inc_va_ro,
            eq_cfg=router_cfg,
            placebo_trials=placebo_trials,
            target_false_admissions_per_fit=0.025,
            seed=seed,
        )
        eta_star, eta_scores = full_st.select_damping_eta_from_oof(oof_r_pred, folds)
        l4_u = full_st.predict_val(eta=eta_star, apply_range_clamp=False)
        l4_c = full_st.predict_val(eta=eta_star, apply_range_clamp=True, clamp_mode=clamp_mode)
        l3_u = _fit_simultaneous_ridge_l3(y_tr, inc_tr_ro, inc_va_ro, bank_tr, bank_va, folds)

        unclamped = {
            "L0_incumbent": inc_va_ro.copy(),
            "L1_c_static": c_stat_ro.copy(),
            "L3_dense_ridge": l3_u,
            "L4_apex_eq": l4_u,
        }
        clamped = {
            k: (
                l4_c
                if k == "L4_apex_eq"
                else _apply_arm_clamp(v, y_tr, clamp_mode, router_cfg.range_clamp_iqr_mult)
            )
            for k, v in unclamped.items()
        }
        l4_meta = {
            "admitted_columns": admitted,
            "c0_nested_calibrated": c0_cal,
            "eta_grid_nested_oof_mse": eta_scores,
            "eta_is_nested_discovery": True,
            "eta_selected": float(eta_star),
            "n_active_groups": len(full_st.groups),
            "state_digest": full_st.state_digest(),
        }
        return l4_c, {
            "c_static_compute_s": None if static_ctrl_compute_s is None else float(static_ctrl_compute_s),
            "clamp_mode": clamp_mode,
            "fit_scope_sha256": scope.scope_sha256,
            "l4_meta": l4_meta,
            "predictions": clamped,
            "predictions_clamped": clamped,
            "predictions_unclamped": unclamped,
            "state_digest": l4_meta["state_digest"],
        }

    pred_l4, scope, bundle = learn_whole_procedure(
        x_train, y_train, x_val, _learner, fold_id="g1_nested_router", extra_arrays=extra_map
    )
    bundle["primary_prediction"] = pred_l4
    bundle["scope"] = dataclasses.asdict(scope)
    return bundle


def _fit_apex_eq_l4(
    y_train: np.ndarray,
    inc_train: np.ndarray,
    inc_val: np.ndarray,
    bank_train: np.ndarray,
    bank_val: np.ndarray,
    cand_names: Sequence[str],
    folds: Sequence[np.ndarray],
    cfg: Optional[EquilibriumConfig] = None,
    calibrate_placebo: bool = True,
    nested_eta: bool = True,
    clamp_mode: str = "search_iqr",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """`L4`: Fixed-bank `APEX-EQ` with genuine 5-fold nested `eta` selection (`Finding #5`)."""
    eq_cfg = cfg or EquilibriumConfig()
    full_state, admitted, reweighted_in_span, c0_cal = _fit_apex_eq_single_pass(
        y_train=y_train,
        inc_train=inc_train,
        inc_val=inc_val,
        bank_train=bank_train,
        bank_val=bank_val,
        cand_names=cand_names,
        eq_cfg=eq_cfg,
        calibrate_placebo=calibrate_placebo,
        placebo_trials=80,
        seed=460201,
    )

    if nested_eta and len(folds) >= 2 and y_train.shape[0] >= 20:
        n_rows = y_train.shape[0]
        all_idx = np.arange(n_rows)
        oof_r_pred = np.zeros(n_rows, dtype=np.float64)
        for f_k, f_idx in enumerate(folds):
            if len(f_idx) == 0:
                continue
            tr_mask = np.ones(n_rows, dtype=bool)
            tr_mask[f_idx] = False
            tr_i = all_idx[tr_mask]
            fold_state, _, _, _ = _fit_apex_eq_single_pass(
                y_train=y_train[tr_i],
                inc_train=inc_train[tr_i],
                inc_val=inc_train[f_idx],
                bank_train=bank_train[tr_i],
                bank_val=bank_train[f_idx],
                cand_names=cand_names,
                eq_cfg=eq_cfg,
                calibrate_placebo=calibrate_placebo,
                placebo_trials=60,
                seed=460210 + f_k,
            )
            oof_r_pred[f_idx] = fold_state.raw_residual_prediction_val()
        eta_star, eta_grid_mse = full_state.select_damping_eta_from_oof(oof_r_pred, folds)
    else:
        eta_star, eta_grid_mse = full_state.select_damping_eta(folds)

    pred_unclamped = full_state.predict_val(eta=eta_star, apply_range_clamp=False)
    pred_clamped = full_state.predict_val(eta=eta_star, apply_range_clamp=True, clamp_mode=clamp_mode)
    meta = {
        "admitted_columns": admitted,
        "base_sigma_n2": float(full_state.base_sigma_n2),
        "c0_calibrated": float(c0_cal),
        "eta_grid_nested_oof_mse": eta_grid_mse,
        "eta_is_nested": bool(nested_eta),
        "eta_selected": float(eta_star),
        "marginal_log_evidence": full_state.marginal_log_evidence(),
        "n_active_groups": len(full_state.groups),
        "reweighted_in_span": reweighted_in_span,
        "state_digest": full_state.state_digest(),
    }
    return pred_unclamped, pred_clamped, meta


def solve_fixed_bank_l0_to_l4(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    incumbent_train: np.ndarray,
    incumbent_val: np.ndarray,
    bank_train: np.ndarray,
    bank_val: np.ndarray,
    cand_names: Sequence[str],
    static_ctrl_val: Optional[np.ndarray] = None,
    static_ctrl_compute_s: Optional[float] = None,
    cand_is_oof: Optional[Sequence[bool]] = None,
    require_oof_bank: bool = False,
    require_c_static: bool = False,
    clamp_mode: str = "search_iqr",
    cfg: Optional[EquilibriumConfig] = None,
) -> Dict[str, Any]:
    """Run `L0..L4` strictly inside `learn_whole_procedure` with full `FitScopeV2` hashing."""
    if require_oof_bank:
        if cand_is_oof is None or len(cand_is_oof) != bank_train.shape[1] or not all(bool(v) for v in cand_is_oof):
            raise ValueError(
                "G1 refusal (Finding #2): every bank_train column must be certified cross-fitted out-of-fold."
            )
    if require_c_static:
        if static_ctrl_val is None or static_ctrl_compute_s is None or float(static_ctrl_compute_s) <= 0.0:
            raise ValueError(
                "G1 refusal (Finding #3): non-synthetic G1 requires an explicit equal-compute C_static "
                "column with positive static_ctrl_compute_s."
            )

    inc_tr_ro = np.array(incumbent_train, dtype=np.float64, copy=True).reshape(-1)
    inc_va_ro = np.array(incumbent_val, dtype=np.float64, copy=True).reshape(-1)
    b_tr_ro = np.array(bank_train, dtype=np.float64, copy=True)
    b_va_ro = np.array(bank_val, dtype=np.float64, copy=True)
    c_stat_ro = (
        inc_va_ro.copy()
        if static_ctrl_val is None
        else np.array(static_ctrl_val, dtype=np.float64, copy=True).reshape(-1)
    )
    for arr in (inc_tr_ro, inc_va_ro, b_tr_ro, b_va_ro, c_stat_ro):
        arr.setflags(write=False)

    extra_map = {
        "bank_train": b_tr_ro,
        "bank_val": b_va_ro,
        "incumbent_train": inc_tr_ro,
        "incumbent_val": inc_va_ro,
        "static_ctrl_val": c_stat_ro,
    }

    def _learner(
        _x_tr: np.ndarray,
        y_tr: np.ndarray,
        _x_va: np.ndarray,
        scope: FitScopeV2,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        eq_cfg = cfg or EquilibriumConfig()
        folds = make_deterministic_folds(y_tr.shape[0], n_folds=5)
        l0_u = inc_va_ro.copy()
        l1_u = c_stat_ro.copy()
        l2_u = _fit_sequential_greedy_l2(y_tr, inc_tr_ro, inc_va_ro, b_tr_ro, b_va_ro)
        l3_u = _fit_simultaneous_ridge_l3(y_tr, inc_tr_ro, inc_va_ro, b_tr_ro, b_va_ro, folds)
        l4_u, l4_c, l4_meta = _fit_apex_eq_l4(
            y_tr,
            inc_tr_ro,
            inc_va_ro,
            b_tr_ro,
            b_va_ro,
            cand_names,
            folds,
            cfg=eq_cfg,
            clamp_mode=clamp_mode,
        )

        unclamped = {
            "L0_incumbent": l0_u,
            "L1_c_static": l1_u,
            "L2_serial_boost": l2_u,
            "L3_dense_ridge": l3_u,
            "L4_apex_eq": l4_u,
        }
        clamped = {
            k: (
                l4_c
                if k == "L4_apex_eq"
                else _apply_arm_clamp(v, y_tr, clamp_mode, eq_cfg.range_clamp_iqr_mult)
            )
            for k, v in unclamped.items()
        }
        return l4_c, {
            "c_static_compute_s": None if static_ctrl_compute_s is None else float(static_ctrl_compute_s),
            "clamp_mode": clamp_mode,
            "fit_scope_sha256": scope.scope_sha256,
            "l4_meta": l4_meta,
            "predictions": clamped,
            "predictions_clamped": clamped,
            "predictions_unclamped": unclamped,
            "state_digest": l4_meta["state_digest"],
        }

    pred_l4, scope, bundle = learn_whole_procedure(
        x_train, y_train, x_val, _learner, fold_id="g1_fixed_bank", extra_arrays=extra_map
    )
    bundle["primary_prediction"] = pred_l4
    bundle["scope"] = dataclasses.asdict(scope)
    return bundle


def run_bank_directory(
    bank_dir: Path,
    mode: str = "fixed_bank",
    s9_selection_receipt: Optional[Path] = None,
    expected_board: Optional[str] = None,
) -> Dict[str, Any]:
    """Load real `.npz` contexts; derive `board` strictly from verified input files (`N5`, `N6`)."""
    assert_clean_train_path(bank_dir)
    if not bank_dir.is_dir():
        raise SystemExit(f"REFUSED: --bank-dir does not exist or is not a directory: {bank_dir}")
    npz_files = sorted(p for p in bank_dir.glob("*.npz") if p.is_file())
    if not npz_files:
        raise SystemExit(f"REFUSED: --bank-dir contains 0 .npz context files: {bank_dir}")

    manifest_path = bank_dir / "manifest.json"
    manifest_sha: Optional[str] = None
    manifest_board: Optional[str] = None
    if manifest_path.is_file():
        assert_clean_train_path(manifest_path)
        manifest_sha = _sha256_file(manifest_path)
        mdoc = json.loads(manifest_path.read_text())
        if "board_id" in mdoc:
            manifest_board = str(mdoc["board_id"])

    input_manifest: List[Dict[str, Any]] = []
    ctx_outputs: Dict[str, Any] = {}
    observed_boards: set[str] = set()

    for npz_p in npz_files:
        assert_clean_train_path(npz_p)
        file_sha = _sha256_file(npz_p)
        with np.load(npz_p, allow_pickle=False) as doc:
            cid = str(doc["cid"].item()) if "cid" in doc.files else npz_p.stem
            file_board = str(doc["board_id"].item()) if "board_id" in doc.files else manifest_board
            if not file_board:
                raise SystemExit(
                    f"REFUSED (N6): {npz_p} lacks 'board_id' and no manifest.json board_id is present; "
                    "board labels must come from loader inputs, never from a CLI flag."
                )
            observed_boards.add(file_board)

            # Explicitly ignore any held-out label keys (`y_val`, `ya`, `y_audit`) on disk (`N5`)!
            x_tr = np.asarray(doc["x_train"], dtype=np.float64)
            y_tr = np.asarray(doc["y_train"], dtype=np.float64).reshape(-1)
            x_va = np.asarray(doc["x_val"], dtype=np.float64)
            inc_tr = np.asarray(doc["incumbent_train"], dtype=np.float64).reshape(-1)
            inc_va = np.asarray(doc["incumbent_val"], dtype=np.float64).reshape(-1)
            c_stat = (
                np.asarray(doc["static_ctrl_val"], dtype=np.float64).reshape(-1)
                if "static_ctrl_val" in doc.files
                else None
            )
            c_sec = (
                float(doc["static_ctrl_compute_s"].item())
                if "static_ctrl_compute_s" in doc.files
                else None
            )
            if mode == "nested_router":
                res = solve_nested_router_dictionary(
                    x_train=x_tr,
                    y_train=y_tr,
                    x_val=x_va,
                    incumbent_train=inc_tr,
                    incumbent_val=inc_va,
                    static_ctrl_val=c_stat,
                    static_ctrl_compute_s=c_sec,
                    require_c_static=True,
                )
            else:
                b_tr = np.asarray(doc["bank_train"], dtype=np.float64)
                b_va = np.asarray(doc["bank_val"], dtype=np.float64)
                names = [str(s) for s in doc["cand_names"].tolist()]
                is_oof = (
                    [bool(b) for b in doc["cand_is_oof"].tolist()]
                    if "cand_is_oof" in doc.files
                    else None
                )
                res = solve_fixed_bank_l0_to_l4(
                    x_train=x_tr,
                    y_train=y_tr,
                    x_val=x_va,
                    incumbent_train=inc_tr,
                    incumbent_val=inc_va,
                    bank_train=b_tr,
                    bank_val=b_va,
                    cand_names=names,
                    static_ctrl_val=c_stat,
                    static_ctrl_compute_s=c_sec,
                    cand_is_oof=is_oof,
                    require_oof_bank=True,
                    require_c_static=True,
                )

        input_manifest.append({"board_id": file_board, "cid": cid, "file": npz_p.name, "sha256": file_sha})
        ctx_outputs[cid] = {
            "fit_scope_sha256": res["fit_scope_sha256"],
            "l4_meta": res["l4_meta"],
            "predictions_clamped": {k: v.tolist() for k, v in res["predictions_clamped"].items()},
            "predictions_unclamped": {k: v.tolist() for k, v in res["predictions_unclamped"].items()},
            "state_digest": res["state_digest"],
        }

    if len(observed_boards) != 1:
        raise SystemExit(f"REFUSED (N6): mixed board_ids inside {bank_dir}: {sorted(observed_boards)}")
    loader_board = next(iter(observed_boards))
    if loader_board.lower() == "p201e":
        raise SystemExit("REFUSED: P201e has no SEARCH/AUDIT bank in Track S tooling (cl_mtc 04:37Z).")
    if expected_board is not None and expected_board != loader_board:
        raise SystemExit(
            f"REFUSED (N6): CLI --board '{expected_board}' disagrees with loader board '{loader_board}'."
        )
    if loader_board in ("p201b_descriptive", "p201b_s9_80", "s9_p201b_80"):
        if s9_selection_receipt is None or not s9_selection_receipt.is_file():
            raise SystemExit(
                f"REFUSED: loader board '{loader_board}' touches S9 P201B-80 and requires an existing "
                "--s9-selection-receipt (cl_mtc 04:07Z ruling)."
            )
        assert_clean_train_path(s9_selection_receipt)

    return {
        "board": loader_board,
        "board_manifest_sha256": manifest_sha,
        "board_source": "loader_inputs",
        "context_ids": [m["cid"] for m in input_manifest],
        "contexts": ctx_outputs,
        "input_manifest": input_manifest,
        "mode": mode,
        "n_contexts": len(input_manifest),
        "schema": "jope.h460.g1_bank_replay/v2",
    }


def _run_synthetic_smoke() -> Dict[str, Any]:
    rng = np.random.default_rng(460301)
    x_tr = rng.normal(size=(180, 6))
    x_va = rng.normal(size=(80, 6))
    true_fn_tr = 1.4 * (x_tr[:, 0] > 0.2).astype(float) + 0.8 * np.sin(x_tr[:, 1]) - 0.6 * x_tr[:, 2]
    true_fn_va = 1.4 * (x_va[:, 0] > 0.2).astype(float) + 0.8 * np.sin(x_va[:, 1]) - 0.6 * x_va[:, 2]
    y_tr = true_fn_tr + 0.15 * rng.normal(size=180)
    inc_tr = 0.5 * np.sin(x_tr[:, 1])
    inc_va = 0.5 * np.sin(x_va[:, 1])

    t0 = time.perf_counter()
    gram = x_tr.T @ x_tr + 5.0 * np.eye(6, dtype=np.float64)
    w_lin = sla.solve(gram, x_tr.T @ (y_tr - np.mean(y_tr)), assume_a="pos")
    c_stat_va = float(np.mean(y_tr)) + x_va @ w_lin
    c_stat_s = max(time.perf_counter() - t0, 1e-6)

    noise_tr = rng.normal(size=(180, 20))
    noise_va = rng.normal(size=(80, 20))
    bank_tr = np.column_stack([(x_tr[:, 0] > 0.2).astype(float), np.sin(x_tr[:, 1]), x_tr[:, 2], noise_tr])
    bank_va = np.column_stack([(x_va[:, 0] > 0.2).astype(float), np.sin(x_va[:, 1]), x_va[:, 2], noise_va])
    names = ["step_c0", "sin_c1", "lin_c2"] + [f"noise_{i}" for i in range(20)]
    is_oof = [True] * len(names)

    res = solve_fixed_bank_l0_to_l4(
        x_tr,
        y_tr,
        x_va,
        inc_tr,
        inc_va,
        bank_tr,
        bank_va,
        names,
        static_ctrl_val=c_stat_va,
        static_ctrl_compute_s=c_stat_s,
        cand_is_oof=is_oof,
        require_oof_bank=True,
        require_c_static=True,
    )
    mse_clamped = {
        k: round(float(np.mean((true_fn_va - v) ** 2)), 6) for k, v in res["predictions_clamped"].items()
    }
    mse_unclamped = {
        k: round(float(np.mean((true_fn_va - v) ** 2)), 6) for k, v in res["predictions_unclamped"].items()
    }
    return {
        "board": "synthetic",
        "board_source": "synthetic_generator",
        "l4_meta": res["l4_meta"],
        "mse_clamped": mse_clamped,
        "mse_unclamped": mse_unclamped,
        "schema": "jope.h460.g1_synthetic_smoke/v2",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--board",
        default=None,
        help="Optional cross-check against loader board_id (or 'synthetic' when --bank-dir is omitted).",
    )
    ap.add_argument("--bank-dir", type=Path, default=None, help="Directory of `.npz` contexts")
    ap.add_argument("--mode", choices=("fixed_bank", "nested_router"), default="fixed_bank")
    ap.add_argument("--s9-selection-receipt", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.board is not None and args.board.lower() == "p201e":
        raise SystemExit(
            "REFUSED (Finding #1 / cl_mtc 04:37Z): P201e has no SEARCH/AUDIT bank in Track S tooling."
        )

    if args.bank_dir is not None:
        summary = run_bank_directory(
            args.bank_dir,
            mode=args.mode,
            s9_selection_receipt=args.s9_selection_receipt,
            expected_board=args.board,
        )
    else:
        if args.board not in (None, "synthetic"):
            raise SystemExit(
                f"REFUSED (Finding #1 / N6): --board '{args.board}' requires --bank-dir of real .npz inputs."
            )
        summary = _run_synthetic_smoke()

    if args.out_json is not None:
        assert_clean_train_path(args.out_json)
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
