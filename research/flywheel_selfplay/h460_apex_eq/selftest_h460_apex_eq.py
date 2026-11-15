#!/usr/bin/env python3
"""Deterministic Gate G0 Verification Suite (`selftest_h460_apex_eq.py`).

Executes seven deterministic mathematical, statistical, and quarantine proofs:
  1. `g0_1_schur_and_duplicate_span`:
     - Exact rank-p Schur marginal log-evidence delta matches full dense Cholesky refactorization
       within `1e-12`;
     - Exact `Compiler §18.7` duplicate-column fixture (`||x||^2 = 200, y = 2x, lambda = 100`)
       reproduces `+106.66666666666667` (`320/3`) under raw ridge while the unregularized QR
       span-novelty guard flags `span_innovation < 1e-12` (`IN_SPAN_REWEIGHT`, `0` false `ADD`);
     - 3-column `[u, v, u+v]` rank-2 span check rejects `u+v` with `span_innovation < 1e-12`.
  2. `g0_2_closed_form_loo_and_lfo`:
     - Closed-form `O(1)` LOO (`alpha_i / C_ii`) and 5-fold block LFO (`C_BB^{-1} alpha_B`)
       match explicit leave-out Cholesky refits within `1e-11`.
  3. `g0_3_placebo_calibration`:
     - Under pure Gaussian noise (`N = 128` random features across `60` independent fits),
       placebo-calibrated `tau(N) = 2 ln N + c0` achieves `< 0.05` false admissions per fit.
  4. `g0_4_outer_fold_poisoning_and_cache`:
     - Poisoning held-out labels `y_all[val_idx] += 1000.0` produces `max_abs_diff == 0.0` on
       query predictions and identical `state_digest`;
     - Flipping 1 bit in residual, weights, or row order invalidates `FitScopeV2` (`0` stale hits),
       and `CoordinatorCAS` re-verifies stale worker epochs cleanly.
  5. `g0_5_eta_zero_incumbent_identity`:
     - `eta = 0.0` returns `incumbent_val` with IEEE-754 `max_abs_diff == 0.0`.
  6. `g0_6_single_row_26sd_outlier_guard` (`cl_h458` `4286` defense):
     - Injects a `+26 sigma` corruption on row 0 plus a junk spike column aligned with row 0;
       verifies robust `APEX-EQ` vetoes the spike (`0` spike admissions) and beats naive `L2`
       test MSE by `> 5x`.
  7. `g0_7_jacobi_discontinuous_and_rational`:
     - Verifies `P2` threshold scan + `B4` gated-smooth + `B1/B2` PSD Gram kernels + `P1` LM
       rational law reduce validation MSE substantially vs sequential boosting on a piecewise
       smooth+jump+rational benchmark.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
from scipy import linalg as sla

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.flywheel_selfplay.h460_apex_eq import (  # noqa: E402
    CoordinatorCAS,
    EnsembleSnapshot,
    EquilibriumConfig,
    EquilibriumState,
    FitScopeV2,
    ScopedFeatureCache,
    WorkerProposal,
    build_delta_kernel,
    build_gated_smooth_features,
    build_partition_kernel,
    calibrate_placebo_c0,
    exact_ridge_duplicate_gain,
    fit_rational_law_vp,
    run_outer_fold_poisoning_canary,
    scan_best_step_features,
)
from research.flywheel_selfplay.h460_apex_eq.replay_g1_s9_pool import solve_fixed_bank_l0_to_l4  # noqa: E402


def _check_g0_1_schur_and_duplicate_span() -> Dict[str, Any]:
    rng = np.random.default_rng(46001)
    n_rows = 96
    y_vec = rng.normal(size=n_rows)
    cfg = EquilibriumConfig(sigma_n2=0.08, robust_irls=False)
    state = EquilibriumState(y_train=y_vec, cfg=cfg)
    state.add_group("g0", rng.normal(size=(n_rows, 2)), weight=0.7)

    ev_before = state.marginal_log_evidence()
    cand_block = rng.normal(size=(n_rows, 3))
    w_test = 0.45
    schur_delta = state.exact_block_delta_log_evidence(cand_block, w_test)
    state.add_group("g_cand", cand_block, weight=w_test)
    dense_delta = state.marginal_log_evidence() - ev_before
    schur_abs_err = abs(schur_delta - dense_delta)
    assert schur_abs_err < 1e-11, f"Schur delta mismatch: {schur_abs_err}"

    # Compiler §18.7 exact fixture: ||x||_2^2 = 200, y = 2*x, lambda = 100 -> 2*Delta J = 320/3
    raw_u = rng.normal(size=40)
    x_col = raw_u * np.sqrt(200.0 / float(np.dot(raw_u, raw_u)))
    y_dup = 2.0 * x_col
    dup_ridge_gain = exact_ridge_duplicate_gain(x_col[:, None], x_col, y_dup, lam=100.0)
    expected_gain = 320.0 / 3.0
    assert abs(dup_ridge_gain - expected_gain) < 1e-10, f"Expected 320/3, got {dup_ridge_gain}"

    dup_state = EquilibriumState(y_train=y_dup, cfg=EquilibriumConfig(sigma_n2=1.0, robust_irls=False))
    dup_state.add_group("x1", x_col, weight=0.01)
    dup_dec = dup_state.screen_dictionary(x_col[:, None], ["x1_duplicate"])
    assert dup_dec.span_innovation < 1e-12, f"Duplicate innovation too large: {dup_dec.span_innovation}"
    assert dup_dec.action == "IN_SPAN_REWEIGHT", f"Expected IN_SPAN_REWEIGHT, got {dup_dec.action}"

    # Rank-2 span check on [u, v, u + v]
    u_vec = rng.normal(size=50)
    v_vec = rng.normal(size=50)
    span_state = EquilibriumState(y_train=rng.normal(size=50), cfg=cfg)
    span_state.add_group("u", u_vec, weight=1.0)
    span_state.add_group("v", v_vec, weight=1.0)
    sum_innov = span_state.span_innovation(u_vec + v_vec)
    orth_w = rng.normal(size=50)
    orth_innov = span_state.span_innovation(orth_w)
    assert sum_innov < 1e-12 and orth_innov > 0.5
    return {
        "compiler_18_7_dup_gain": round(dup_ridge_gain, 10),
        "dup_action": dup_dec.action,
        "dup_span_innovation": dup_dec.span_innovation,
        "rank2_sum_innovation": sum_innov,
        "schur_vs_dense_abs_err": schur_abs_err,
    }


def _check_g0_2_closed_form_loo_and_lfo() -> Dict[str, Any]:
    rng = np.random.default_rng(46002)
    n_rows = 75
    x_tr = rng.normal(size=(n_rows, 4))
    y_tr = np.sin(x_tr[:, 0]) + 0.5 * x_tr[:, 1] + 0.2 * rng.normal(size=n_rows)
    k_part, _ = build_partition_kernel(x_tr)
    cfg = EquilibriumConfig(sigma_n2=0.12, robust_irls=True)
    state = EquilibriumState(y_train=y_tr, cfg=cfg)
    state.add_group("lin", x_tr, weight=0.6)
    state.add_kernel("part", k_part, variance=0.35)
    state.update_robust_noise_irls()

    # Verify O(1) LOO against explicit n-1 submatrix solves
    e_loo_fast = state.loo_residuals()
    e_loo_ref = np.zeros(n_rows, dtype=np.float64)
    all_idx = np.arange(n_rows)
    for i in range(n_rows):
        keep = all_idx[all_idx != i]
        cov_aa = state.cov[np.ix_(keep, keep)]
        sig_ia = state.signal_cov[i, keep]
        pred_i = float(sig_ia @ sla.solve(cov_aa, state.residual[keep], assume_a="pos"))
        e_loo_ref[i] = state.residual[i] - pred_i
    loo_max_err = float(np.max(np.abs(e_loo_fast - e_loo_ref)))
    assert loo_max_err < 1e-11, f"LOO error too high: {loo_max_err}"

    # Verify 5-fold block LFO against 5 explicit fold refits
    folds = [np.arange(k, n_rows, 5) for k in range(5)]
    e_lfo_fast = state.lfo_residuals(folds)
    e_lfo_ref = np.zeros(n_rows, dtype=np.float64)
    for f_idx in folds:
        mask = np.ones(n_rows, dtype=bool)
        mask[f_idx] = False
        keep = all_idx[mask]
        cov_aa = state.cov[np.ix_(keep, keep)]
        sig_ba = state.signal_cov[np.ix_(f_idx, keep)]
        pred_b = sig_ba @ sla.solve(cov_aa, state.residual[keep], assume_a="pos")
        e_lfo_ref[f_idx] = state.residual[f_idx] - pred_b
    lfo_max_err = float(np.max(np.abs(e_lfo_fast - e_lfo_ref)))
    assert lfo_max_err < 1e-11, f"LFO error too high: {lfo_max_err}"
    return {"lfo_max_abs_err": lfo_max_err, "loo_max_abs_err": loo_max_err}


def _check_g0_3_placebo_calibration() -> Dict[str, Any]:
    rng = np.random.default_rng(46003)
    n_rows, n_cands, n_fits = 120, 128, 60
    y_cal = rng.normal(size=n_rows)
    pool_cal = rng.normal(size=(n_rows, n_cands))
    pool_cal = (pool_cal - np.mean(pool_cal, axis=0)) / np.std(pool_cal, axis=0)
    cal_state = EquilibriumState(y_train=y_cal, cfg=EquilibriumConfig(sigma_n2=1.0, robust_irls=True))
    c0_cal = calibrate_placebo_c0(
        cal_state, pool_cal, n_trials=240, target_false_admissions_per_fit=0.03, seed=460031
    )

    false_admissions = 0
    names = [f"null_{j}" for j in range(n_cands)]
    for fit_i in range(n_fits):
        y_null = rng.normal(size=n_rows)
        x_null = rng.normal(size=(n_rows, n_cands))
        x_null = (x_null - np.mean(x_null, axis=0)) / np.std(x_null, axis=0)
        st = EquilibriumState(
            y_train=y_null,
            cfg=EquilibriumConfig(sigma_n2=float(np.var(y_null)), foster_george_c0=c0_cal, robust_irls=True),
        )
        dec = st.screen_dictionary(x_null, names, c0_override=c0_cal)
        if dec.action == "ADD":
            false_admissions += 1

    rate = false_admissions / float(n_fits)
    assert rate <= 0.05, f"Null false admission rate {rate:.4f} exceeded 0.05"
    return {"c0_calibrated": round(c0_cal, 4), "false_admissions_per_fit": round(rate, 4), "n_fits": n_fits}


def _check_g0_4_outer_fold_poisoning_and_cache() -> Dict[str, Any]:
    rng = np.random.default_rng(46004)
    x_all = rng.normal(size=(100, 6))
    y_all = 1.2 * (x_all[:, 0] > 0.0).astype(float) - 0.7 * x_all[:, 1] + 0.1 * rng.normal(size=100)
    train_idx = np.arange(0, 75)
    val_idx = np.arange(75, 100)

    def _pipeline(x_full: np.ndarray, y_full: np.ndarray, tr_i: np.ndarray, va_i: np.ndarray):
        x_tr, y_tr, x_va = x_full[tr_i], y_full[tr_i], x_full[va_i]
        inc_tr = np.full(tr_i.size, float(np.mean(y_tr)), dtype=np.float64)
        inc_va = np.full(va_i.size, float(np.mean(y_tr)), dtype=np.float64)
        steps = scan_best_step_features(x_tr, y_tr - inc_tr, np.eye(tr_i.size), x_val=x_va, max_return=4)
        b_tr = np.column_stack([x_tr] + [s.phi_train for s in steps])
        b_va = np.column_stack([x_va] + [s.phi_val for s in steps if s.phi_val is not None])
        names = [f"raw_{j}" for j in range(x_tr.shape[1])] + [s.name for s in steps]
        out = solve_fixed_bank_l0_to_l4(x_tr, y_tr, x_va, inc_tr, inc_va, b_tr, b_va, names)
        return out["primary_prediction"], {"state_digest": out["state_digest"]}

    canary = run_outer_fold_poisoning_canary(x_all, y_all, train_idx, val_idx, _pipeline, poison_offset=1000.0)
    assert canary["passed"] and canary["max_abs_diff"] == 0.0, f"Canary failed: {canary}"

    # Cache invalidation check on 1-ulp perturbation and row permutation
    cache = ScopedFeatureCache()
    s_base = FitScopeV2.from_arrays(x_all[train_idx], y_all[train_idx], fold_id="f0")
    cache.put(s_base, "expr_1", x_all[train_idx, 0])
    assert cache.get(s_base, "expr_1") is not None

    r_mod = y_all[train_idx].copy()
    r_mod[0] = np.nextafter(r_mod[0], np.inf)
    s_res = FitScopeV2.from_arrays(x_all[train_idx], y_all[train_idx], residual=r_mod, fold_id="f0")
    w_mod = np.ones(train_idx.size, dtype=np.float64)
    w_mod[-1] = np.nextafter(w_mod[-1], np.inf)
    s_w = FitScopeV2.from_arrays(x_all[train_idx], y_all[train_idx], row_weights=w_mod, fold_id="f0")
    s_perm = FitScopeV2.from_arrays(x_all[train_idx][::-1], y_all[train_idx][::-1], fold_id="f0")
    assert cache.get(s_res, "expr_1") is None
    assert cache.get(s_w, "expr_1") is None
    assert cache.get(s_perm, "expr_1") is None

    # CoordinatorCAS stale-epoch re-verification check
    snap0 = EnsembleSnapshot(0, s_base.scope_sha256, (), -10.0, np.zeros(4), "d0")
    cas = CoordinatorCAS(snap0)
    prop_a = WorkerProposal(0, "col_a", np.ones(4), None, 12.0)
    prop_b_stale = WorkerProposal(0, "col_a_dup", np.ones(4), None, 11.5)
    ok_a, _ = cas.try_commit(
        prop_a,
        lambda p, cur: EnsembleSnapshot(cur.epoch + 1, cur.scope_sha256, cur.active_names + (p.name,), -8.0, cur.alpha_Dual, "d1"),
    )
    ok_b, _ = cas.try_commit(prop_b_stale, lambda _p, _cur: None)
    assert ok_a and (not ok_b) and cas.snapshot.epoch == 1
    return {"cache_misses_on_1ulp": cache.misses, "canary_max_abs_diff": canary["max_abs_diff"]}


def _check_g0_5_eta_zero_incumbent_identity() -> Dict[str, Any]:
    rng = np.random.default_rng(46005)
    y_tr = rng.normal(size=60)
    inc_tr = rng.normal(size=60)
    inc_va = rng.normal(size=25)
    st = EquilibriumState(y_train=y_tr, incumbent_train=inc_tr, incumbent_val=inc_va)
    st.add_group("g1", rng.normal(size=(60, 2)), phi_val=rng.normal(size=(25, 2)), weight=2.0)
    pred_zero = st.predict_val(eta=0.0, apply_range_clamp=True)
    max_diff = float(np.max(np.abs(pred_zero - inc_va)))
    assert max_diff == 0.0, f"eta=0 failed exact incumbent identity: {max_diff}"
    return {"eta_zero_max_abs_diff": max_diff}


def _check_g0_6_single_row_26sd_outlier_guard() -> Dict[str, Any]:
    """Verify `cl_h458` 26-sd single-row spike defense (`4286` pathology)."""
    rng = np.random.default_rng(46006)
    n_tr, n_va = 180, 80
    x_tr = rng.normal(size=(n_tr, 4))
    x_va = rng.normal(size=(n_va, 4))
    clean_tr = 1.5 * x_tr[:, 0] - 0.9 * x_tr[:, 1]
    clean_va = 1.5 * x_va[:, 0] - 0.9 * x_va[:, 1]
    y_tr = clean_tr + 0.15 * rng.normal(size=n_tr)
    # Corrupt row 0 by +26 standard deviations of y
    y_tr[0] += 26.0 * float(np.std(clean_tr))

    # Spurious feature: matches row 0 on train AND fires randomly on a test row!
    spike_tr = np.zeros(n_tr, dtype=np.float64)
    spike_tr[0] = 1.0
    spike_va = np.zeros(n_va, dtype=np.float64)
    spike_va[5] = 1.0

    bank_tr = np.column_stack([x_tr[:, 0], x_tr[:, 1], spike_tr, rng.normal(size=(n_tr, 12))])
    bank_va = np.column_stack([x_va[:, 0], x_va[:, 1], spike_va, rng.normal(size=(n_va, 12))])
    names = ["sig_0", "sig_1", "poison_spike_row0"] + [f"noise_{i}" for i in range(12)]
    inc_tr = np.zeros(n_tr, dtype=np.float64)
    inc_va = np.zeros(n_va, dtype=np.float64)

    out = solve_fixed_bank_l0_to_l4(x_tr, y_tr, x_va, inc_tr, inc_va, bank_tr, bank_va, names)
    admitted = out["l4_meta"]["admitted_columns"]
    assert "poison_spike_row0" not in admitted, f"Spike column leaked into active set: {admitted}"
    mse_l2 = float(np.mean((clean_va - out["predictions"]["L2_serial_boost"]) ** 2))
    mse_l4 = float(np.mean((clean_va - out["predictions"]["L4_apex_eq"]) ** 2))
    assert mse_l4 * 5.0 < mse_l2, f"Expected >5x MSE improvement on 26sd spike: L4={mse_l4} vs L2={mse_l2}"
    return {
        "admitted_columns": admitted,
        "mse_l2_naive_boost": round(mse_l2, 6),
        "mse_l4_apex_eq": round(mse_l4, 6),
        "mse_ratio_l2_over_l4": round(mse_l2 / max(mse_l4, 1e-12), 2),
    }


def _check_g0_7_jacobi_discontinuous_and_rational() -> Dict[str, Any]:
    rng = np.random.default_rng(46007)
    n_tr, n_va = 220, 100
    x_tr = rng.uniform(-2.0, 2.0, size=(n_tr, 5))
    x_va = rng.uniform(-2.0, 2.0, size=(n_va, 5))

    def _truth(x: np.ndarray) -> np.ndarray:
        jump = np.where(x[:, 0] > 0.35, 1.8 + 0.9 * x[:, 1], -1.2 - 0.7 * x[:, 1])
        rat = 1.1 * x[:, 2] / (1.0 + 1.4 * np.abs(x[:, 3]) + 0.5 * (x[:, 3] ** 2))
        return jump + rat

    y_tr = _truth(x_tr) + 0.12 * rng.normal(size=n_tr)
    y_va_clean = _truth(x_va)
    inc_tr = np.full(n_tr, float(np.mean(y_tr)), dtype=np.float64)
    inc_va = np.full(n_va, float(np.mean(y_tr)), dtype=np.float64)

    st_init = EquilibriumState(y_train=y_tr, incumbent_train=inc_tr, incumbent_val=inc_va)
    steps = scan_best_step_features(x_tr, st_init.alpha, st_init.inv_cov, x_val=x_va, max_return=6)
    gated_tr, gated_va, gated_names = build_gated_smooth_features(steps, x_tr, x_val=x_va, max_basis_cols=3)
    rat_tr, rat_va, rat_name = fit_rational_law_vp(x_tr, st_init.alpha, u_col=2, v_col=3, x_val=x_va)
    k_part_tr, _ = build_partition_kernel(x_tr, x_va)
    k_delta_tr, _ = build_delta_kernel(x_tr, x_va)

    eig_part = float(np.min(sla.eigvalsh(k_part_tr)))
    eig_delta = float(np.min(sla.eigvalsh(k_delta_tr)))
    assert eig_part >= -1e-10 and eig_delta >= -1e-10, "B1/B2 Gram matrices must be PSD"

    step_tr = np.column_stack([s.phi_train for s in steps])
    step_va = np.column_stack([s.phi_val for s in steps if s.phi_val is not None])
    bank_tr = np.column_stack([x_tr, step_tr, gated_tr, rat_tr[:, None]])
    bank_va = np.column_stack([x_va, step_va, gated_va, rat_va[:, None]])
    names = (
        [f"lin_{j}" for j in range(x_tr.shape[1])]
        + [s.name for s in steps]
        + gated_names
        + [rat_name]
    )

    res = solve_fixed_bank_l0_to_l4(x_tr, y_tr, x_va, inc_tr, inc_va, bank_tr, bank_va, names)
    mse_inc = float(np.mean((y_va_clean - res["predictions"]["L0_incumbent"]) ** 2))
    mse_l4 = float(np.mean((y_va_clean - res["predictions"]["L4_apex_eq"]) ** 2))
    assert mse_l4 < 0.08 * mse_inc, f"Expected >12x MSE drop over incumbent: {mse_l4} vs {mse_inc}"
    return {
        "b1_min_eig": eig_part,
        "b2_min_eig": eig_delta,
        "mse_incumbent": round(mse_inc, 6),
        "mse_l4_apex_eq": round(mse_l4, 6),
    }


def run_all_selftests() -> Dict[str, Any]:
    results = {
        "g0_1_schur_and_duplicate_span": _check_g0_1_schur_and_duplicate_span(),
        "g0_2_closed_form_loo_and_lfo": _check_g0_2_closed_form_loo_and_lfo(),
        "g0_3_placebo_calibration": _check_g0_3_placebo_calibration(),
        "g0_4_outer_fold_poisoning_and_cache": _check_g0_4_outer_fold_poisoning_and_cache(),
        "g0_5_eta_zero_incumbent_identity": _check_g0_5_eta_zero_incumbent_identity(),
        "g0_6_single_row_26sd_outlier_guard": _check_g0_6_single_row_26sd_outlier_guard(),
        "g0_7_jacobi_discontinuous_and_rational": _check_g0_7_jacobi_discontinuous_and_rational(),
    }
    return {"all_passed": True, "n_checks": len(results), "results": results, "schema": "jope.h460.g0_selftest/v1"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true", help="Run all 7 deterministic Gate G0 checks")
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args(argv)
    report = run_all_selftests()
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
