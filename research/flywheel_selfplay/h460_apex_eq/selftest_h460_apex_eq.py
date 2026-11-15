#!/usr/bin/env python3
"""Deterministic Gate G0 Verification Suite v2 (`selftest_h460_apex_eq.py`).

Executes eight deterministic mathematical, statistical, and quarantine proofs (`G0_1..G0_8`):
  1. `g0_1_schur_duplicate_scale_and_rank_scan`:
     - Exact rank-p Schur marginal log-evidence delta matches dense Cholesky within `1e-11`;
     - Exact `Compiler §18.7` duplicate-column fixture (`||x||^2 = 200, y = 2x, lambda = 100`)
       reproduces `320/3 = 106.6666666667` under raw ridge while QR novelty flags `IN_SPAN_REWEIGHT`;
     - Rank-2 `[u, v, u+v]` check (`span_innovation < 1e-12`);
     - Scale covariance (`Finding #8`): `1000x` target scaling preserves admissions & predictions;
     - Full-rank scan (`Finding #11`): vetoes 25 leading single-row spikes and admits column #26.
  2. `g0_2_closed_form_loo_and_lfo`:
     - Closed-form `O(1)` LOO (`alpha_i / C_ii`) and 5-fold block LFO (`C_BB^{-1} alpha_B`)
       match explicit leave-out Cholesky refits within `1e-11`.
  3. `g0_3_placebo_calibration` (`Finding #9`):
     - Fixed label-free OOF pool (`N = 128` random features) across `F = 400` independent null fits
       achieves `<= 0.05` false admissions per fit with 95% Clopper-Pearson bound reported.
  4. `g0_4_outer_fold_poisoning_and_cache` (`Findings #1, #2, #3, #7` + `N5` + `N6`):
     - Real on-disk `.npz` CLI AUDIT-poison canary (`y_val += 1000` on disk via `replay_g1_s9_pool.main`)
       produces `max_abs_diff == 0.0` across all `L0..L4` clamped & unclamped predictions;
     - 1-ulp mutation of `bank_train` or `incumbent_train` invalidates `FitScopeV2`;
     - Verifies `--board p201e` refusal, missing `--bank-dir` refusal, mismatched `--board` vs
       loader `board_id` refusal (`N6`), non-OOF bank refusal (`Finding #2`), and missing `C_static`
       refusal (`Finding #3`).
  5. `g0_5_eta_zero_incumbent_identity`:
     - `eta = 0.0` returns `incumbent_val` with IEEE-754 `max_abs_diff == 0.0` (`search_iqr` & `mad3`).
  6. `g0_6_single_row_26sd_outlier_guard` (`cl_h458` `4286` defense + `Findings #4, #10` + `N4`):
     - Injects `+26 sigma` corruption on row 0 + spurious row-0 spike firing on validation row 5;
       verifies `0` spike admissions and `L4` beats BOTH `L2` AND `L3` dense ridge by `> 3x` on
       both clamped and unclamped predictions.
  7. `g0_7_jacobi_discontinuous_and_rational`:
     - Verifies `P2`, `B1`, `B2`, `B4`, `P1` PSD & regime-jump recovery and verifies
       `solve_nested_router_dictionary` (`N1 + N2 + N4`) beats `L0` and `L3`.
  8. `g0_8_nested_discovery_placebo` (`cl_fwriter` `04:36Z` `G0_8` spec):
     - Across `F = 400` independent pure-noise fits (`n = 180, p = 6, x ~ N(0,1), y ~ N(0,1)`),
       builds the full production bank (`build_production_router_dictionary`) on each fit's own
       residual, calibrates `c0` via `calibrate_nested_discovery_placebo_c0` (rebuilding the bank
       on every permuted residual), asserts `nested_rate <= 0.05` (with one-sided 95% Clopper-Pearson
       bound), AND runs the `38ff3faf` fixed-pool power check on the SAME `400` fits proving
       `fixed_pool_rate > 0.10` (`>> 0.05`).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from scipy import linalg as sla
from scipy import stats as sstats

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.flywheel_selfplay.h460_apex_eq import (  # noqa: E402
    CoordinatorCAS,
    EnsembleSnapshot,
    EquilibriumConfig,
    EquilibriumState,
    FitScopeV2,
    PrecomputedXContext,
    ROUTER_R0_CONFIG_GRID,
    ROUTER_R1_CONFIG_GRID,
    ROUTER_R2_L3_CONFIG_GRID,
    ROUTER_R3_STATIC_CONFIG_GRID,
    ScopedFeatureCache,
    WorkerProposal,
    build_delta_kernel,
    build_gated_smooth_features,
    build_partition_kernel,
    build_production_router_dictionary,
    calibrate_nested_discovery_placebo_c0,
    calibrate_placebo_c0,
    exact_ridge_duplicate_gain,
    fit_rational_law_varpro_full,
    fit_rational_law_vp,
    fit_router_fold_dictionary,
    run_outer_fold_poisoning_canary,
    run_router_ratchet_harness,
    scan_best_equality_features,
    scan_best_step_features,
    solve_fixed_bank_l0_to_l4,
    solve_nested_router_dictionary,
    target_stratified_kfold_labels,
)
from research.flywheel_selfplay.h460_apex_eq import replay_g1_s9_pool  # noqa: E402


def _clopper_pearson_upper_95(k_fail: int, n_total: int) -> float:
    if k_fail >= n_total:
        return 1.0
    return float(sstats.beta.ppf(0.95, k_fail + 1, n_total - k_fail))


def _package_code_sha256() -> str:
    h = hashlib.sha256()
    for py_file in sorted(PKG_DIR.glob("*.py")):
        h.update(f"{py_file.name}:".encode("utf-8"))
        h.update(py_file.read_bytes())
    return h.hexdigest()


def _scratch_canary_dir() -> Path:
    """Create non-/tmp scratch directory inside `.agent` or user home, cleaned up in `finally`."""
    agent_root = REPO_ROOT / ".agent" / "h460_selftest_scratch"
    base = agent_root if (REPO_ROOT / ".agent").exists() else (Path.home() / ".cache" / "jope_h460_scratch")
    d = base / f"canary_{os.getpid()}_{int(time.time() * 1e6)}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _check_g0_1_schur_duplicate_scale_and_rank_scan() -> Dict[str, Any]:
    rng = np.random.default_rng(46001)
    n_rows = 96
    y_vec = rng.normal(size=n_rows)
    cfg = EquilibriumConfig(sigma_n2=0.08, auto_scale_sigma_n2=False, robust_irls=False)
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

    dup_state = EquilibriumState(
        y_train=y_dup,
        cfg=EquilibriumConfig(sigma_n2=1.0, auto_scale_sigma_n2=False, robust_irls=False),
    )
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
    orth_innov = span_state.span_innovation(rng.normal(size=50))
    assert sum_innov < 1e-12 and orth_innov > 0.5

    # Finding #8: scale covariance under 1000x target rescaling
    x_sc = rng.normal(size=(80, 3))
    x_sc_va = rng.normal(size=(30, 3))
    y_sc = 1.5 * x_sc[:, 0] + 0.2 * rng.normal(size=80)
    inc_sc = np.zeros(80)
    inc_sc_va = np.zeros(30)
    r1 = solve_fixed_bank_l0_to_l4(x_sc, y_sc, x_sc_va, inc_sc, inc_sc_va, x_sc, x_sc_va, ["c0", "c1", "c2"])
    r1000 = solve_fixed_bank_l0_to_l4(
        x_sc, 1000.0 * y_sc, x_sc_va, inc_sc, inc_sc_va, x_sc, x_sc_va, ["c0", "c1", "c2"]
    )
    assert r1["l4_meta"]["admitted_columns"] == r1000["l4_meta"]["admitted_columns"]
    scale_rel_err = float(
        np.max(np.abs(1000.0 * r1["primary_prediction"] - r1000["primary_prediction"]))
        / max(float(np.std(r1000["primary_prediction"])), 1e-9)
    )
    assert scale_rel_err < 1e-10, f"Scale covariance drift: {scale_rel_err}"

    # Finding #11: full-rank scan past 25 3-row spikes (n_eff = 3 < 4, unclipped by winsor) to admit #26
    clean_sig = rng.normal(size=80)
    clean_sig = (clean_sig - np.mean(clean_sig)) / np.std(clean_sig)
    y_sp = 0.25 * clean_sig + 0.30 * rng.normal(size=80)
    # Give rows 0..2 +1.45 residuals (~3.6 MAD, below 4.0 Huber / 4.5 winsor, z2 ~ 45 > clean_sig z2 ~ 25)
    y_sp[:3] = 1.45
    spikes = []
    for s_i in range(25):
        sp = np.zeros(80, dtype=np.float64)
        sp[0] = 1.0
        sp[1] = 1.0 + 0.005 * s_i
        sp[2] = 1.0 - 0.004 * s_i
        sp = (sp - np.mean(sp)) / np.std(sp)
        spikes.append(sp)
    pool_26 = np.column_stack(spikes + [clean_sig])
    names_26 = [f"spike3_{i}" for i in range(25)] + ["clean_signal_26"]
    st_26 = EquilibriumState(y_train=y_sp, cfg=EquilibriumConfig(robust_irls=True))
    st_26.update_robust_noise_irls()
    dec_26 = st_26.screen_dictionary(pool_26, names_26)
    assert dec_26.action == "ADD" and dec_26.candidate_name == "clean_signal_26", f"Got {dec_26}"
    assert len(dec_26.vetoed_indices) == 25, f"Expected 25 vetoed, got {len(dec_26.vetoed_indices)}"

    # Verify exact O(n^2) rank-1 Sherman-Morrison matches dense Cholesky to < 1e-13
    st_sm = EquilibriumState(
        y_train=y_sp,
        anchor_h_train=np.ones((80, 1), dtype=np.float64),
        tau_h2=2.5,
        cfg=EquilibriumConfig(robust_irls=True),
    )
    st_sm.update_robust_noise_irls()
    chol_ref = sla.cholesky(st_sm.cov, lower=True, check_finite=False)
    inv_ref = sla.cho_solve((chol_ref, True), np.eye(80, dtype=np.float64), check_finite=False)
    ev_ref = -float(np.sum(np.log(np.diag(chol_ref)))) - 0.5 * float(np.dot(st_sm.residual, inv_ref @ st_sm.residual))
    sm_inv_err = float(np.max(np.abs(st_sm.inv_cov - inv_ref)))
    sm_ev_err = abs(st_sm.marginal_log_evidence() - ev_ref)
    assert sm_inv_err < 1e-10 and sm_ev_err < 1e-10, f"Sherman-Morrison drift: {sm_inv_err}, {sm_ev_err}"

    return {
        "compiler_18_7_dup_gain": round(dup_ridge_gain, 10),
        "dup_action": dup_dec.action,
        "rank2_sum_innovation": sum_innov,
        "scale_1000x_rel_err": scale_rel_err,
        "schur_vs_dense_abs_err": schur_abs_err,
        "sherman_morrison_inv_err": sm_inv_err,
        "spikes_vetoed_before_clean": len(dec_26.vetoed_indices),
    }


def _check_g0_2_closed_form_loo_and_lfo() -> Dict[str, Any]:
    rng = np.random.default_rng(46002)
    n_rows = 75
    x_tr = rng.normal(size=(n_rows, 4))
    y_tr = np.sin(x_tr[:, 0]) + 0.5 * x_tr[:, 1] + 0.2 * rng.normal(size=n_rows)
    k_part, _ = build_partition_kernel(x_tr)
    cfg = EquilibriumConfig(sigma_n2=0.12, auto_scale_sigma_n2=False, robust_irls=True)
    state = EquilibriumState(y_train=y_tr, cfg=cfg)
    state.add_group("lin", x_tr, weight=0.6)
    state.add_kernel("part", k_part, variance=0.35)
    state.update_robust_noise_irls()

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
    """Fixed label-free OOF pool null test across `F = 400` fits (`Finding #9`)."""
    rng = np.random.default_rng(46003)
    n_rows, n_cands, n_fits = 120, 128, 400
    false_admissions = 0
    names = [f"null_{j}" for j in range(n_cands)]

    for fit_i in range(n_fits):
        y_null = rng.normal(size=n_rows)
        x_null = rng.normal(size=(n_rows, n_cands))
        x_null = (x_null - np.mean(x_null, axis=0)) / np.std(x_null, axis=0)
        st = EquilibriumState(
            y_train=y_null,
            incumbent_train=np.full(n_rows, float(np.mean(y_null))),
            anchor_h_train=np.ones((n_rows, 1), dtype=np.float64),
            tau_h2=4.0,
            cfg=EquilibriumConfig(robust_irls=True),
        )
        st.update_robust_noise_irls()
        c0_cal = calibrate_placebo_c0(
            st, x_null, n_trials=79, target_false_admissions_per_fit=0.025, seed=4600300 + fit_i
        )
        dec = st.screen_dictionary(x_null, names, c0_override=c0_cal)
        if dec.action == "ADD":
            false_admissions += 1

    rate = false_admissions / float(n_fits)
    cp95 = _clopper_pearson_upper_95(false_admissions, n_fits)
    assert rate <= 0.05, f"Fixed-pool null false admission rate {rate:.4f} exceeded 0.05"
    return {
        "clopper_pearson_upper_95": round(cp95, 4),
        "false_admissions": false_admissions,
        "false_admissions_per_fit": round(rate, 4),
        "n_fits": n_fits,
    }


def _check_g0_4_outer_fold_poisoning_and_cache() -> Dict[str, Any]:
    rng = np.random.default_rng(46004)
    x_all = rng.normal(size=(100, 5))
    y_all = 1.2 * (x_all[:, 0] > 0.0).astype(float) - 0.7 * x_all[:, 1] + 0.1 * rng.normal(size=100)
    train_idx = np.arange(0, 75)
    val_idx = np.arange(75, 100)

    def _nested_pipeline(x_full: np.ndarray, y_full: np.ndarray, tr_i: np.ndarray, va_i: np.ndarray):
        x_tr, y_tr, x_va = x_full[tr_i], y_full[tr_i], x_full[va_i]
        inc_tr = np.full(tr_i.size, float(np.mean(y_tr)), dtype=np.float64)
        inc_va = np.full(va_i.size, float(np.mean(y_tr)), dtype=np.float64)
        out = solve_nested_router_dictionary(
            x_tr, y_tr, x_va, inc_tr, inc_va, placebo_trials=39, fold_placebo_trials=39
        )
        return out["primary_prediction"], {
            "fit_scope_sha256": out["fit_scope_sha256"],
            "state_digest": out["state_digest"],
        }

    canary = run_outer_fold_poisoning_canary(x_all, y_all, train_idx, val_idx, _nested_pipeline, poison_offset=1000.0)
    assert canary["passed"] and canary["max_abs_diff"] == 0.0, f"In-memory canary failed: {canary}"

    # N5 & N6: Real on-disk .npz CLI entry-point AUDIT-poison canary + refusals
    scratch = _scratch_canary_dir()
    try:
        bank_dir = scratch / "bank_clean_train"
        bank_dir.mkdir(parents=True, exist_ok=True)
        x_tr, y_tr, x_va, y_va = x_all[train_idx], y_all[train_idx], x_all[val_idx], y_all[val_idx]
        inc_tr = np.full(train_idx.size, float(np.mean(y_tr)), dtype=np.float64)
        inc_va = np.full(val_idx.size, float(np.mean(y_tr)), dtype=np.float64)
        b_tr = x_tr.copy()
        b_va = x_va.copy()
        c_names = np.array([f"oof_{j}" for j in range(x_tr.shape[1])])
        c_oof = np.array([True] * x_tr.shape[1])
        c_stat = inc_va + 0.05 * x_va[:, 0]

        npz_path = bank_dir / "ctx_001.npz"
        base_payload = {
            "cid": np.array("ctx_001"),
            "board_id": np.array("p201b_non_s9_121"),
            "x_train": x_tr,
            "y_train": y_tr,
            "x_val": x_va,
            "y_val": y_va,
            "ya": y_va,
            "y_audit": y_va,
            "incumbent_train": inc_tr,
            "incumbent_val": inc_va,
            "bank_train": b_tr,
            "bank_val": b_va,
            "cand_names": c_names,
            "cand_is_oof": c_oof,
            "static_ctrl_val": c_stat,
            "static_ctrl_compute_s": np.array(0.012),
        }
        np.savez(npz_path, **base_payload)
        out_clean = scratch / "out_clean.json"
        rc1 = replay_g1_s9_pool.main(["--bank-dir", str(bank_dir), "--out-json", str(out_clean)])
        assert rc1 == 0
        doc_clean = json.loads(out_clean.read_text())
        assert doc_clean["board"] == "p201b_non_s9_121" and doc_clean["board_source"] == "loader_inputs"

        # Poison all held-out label keys on disk by +1000.0
        poison_payload = dict(base_payload)
        poison_payload["y_val"] = y_va + 1000.0
        poison_payload["ya"] = y_va + 1000.0
        poison_payload["y_audit"] = y_va + 1000.0
        np.savez(npz_path, **poison_payload)
        out_poison = scratch / "out_poison.json"
        rc2 = replay_g1_s9_pool.main(["--bank-dir", str(bank_dir), "--out-json", str(out_poison)])
        assert rc2 == 0
        doc_poison = json.loads(out_poison.read_text())

        c_ctx = doc_clean["contexts"]["ctx_001"]
        p_ctx = doc_poison["contexts"]["ctx_001"]
        assert c_ctx["fit_scope_sha256"] == p_ctx["fit_scope_sha256"]
        assert c_ctx["state_digest"] == p_ctx["state_digest"]
        cli_max_diff = 0.0
        for dict_key in ("predictions_clamped", "predictions_unclamped"):
            for arm_k, vals_c in c_ctx[dict_key].items():
                vals_p = p_ctx[dict_key][arm_k]
                cli_max_diff = max(cli_max_diff, float(np.max(np.abs(np.array(vals_c) - np.array(vals_p)))))
        assert cli_max_diff == 0.0, f"CLI on-disk canary diff: {cli_max_diff}"

        # Finding #7: 1-ulp mutation in bank_train or incumbent_train MUST change fit_scope_sha256
        b_mut = b_tr.copy()
        b_mut[0, 0] = np.nextafter(b_mut[0, 0], np.inf)
        mut_payload = dict(base_payload)
        mut_payload["bank_train"] = b_mut
        np.savez(npz_path, **mut_payload)
        out_mut = scratch / "out_mut.json"
        replay_g1_s9_pool.main(["--bank-dir", str(bank_dir), "--out-json", str(out_mut)])
        doc_mut = json.loads(out_mut.read_text())
        assert doc_mut["contexts"]["ctx_001"]["fit_scope_sha256"] != c_ctx["fit_scope_sha256"]

        # Verify Finding #1, #2, #3 & N6 refusals
        refused_p201e = False
        try:
            replay_g1_s9_pool.main(["--board", "p201e"])
        except SystemExit:
            refused_p201e = True
        assert refused_p201e

        refused_board_mismatch = False
        try:
            replay_g1_s9_pool.main(["--bank-dir", str(bank_dir), "--board", "fitsel_106"])
        except SystemExit:
            refused_board_mismatch = True
        assert refused_board_mismatch

        refused_non_oof = False
        try:
            solve_fixed_bank_l0_to_l4(
                x_tr, y_tr, x_va, inc_tr, inc_va, b_tr, b_va, ["c0"] * b_tr.shape[1],
                static_ctrl_val=c_stat, static_ctrl_compute_s=0.01,
                cand_is_oof=[True, False] + [True] * (b_tr.shape[1] - 2),
                require_oof_bank=True, require_c_static=True,
            )
        except ValueError:
            refused_non_oof = True
        assert refused_non_oof

        refused_no_c_static = False
        try:
            solve_fixed_bank_l0_to_l4(
                x_tr, y_tr, x_va, inc_tr, inc_va, b_tr, b_va, ["c0"] * b_tr.shape[1],
                static_ctrl_val=None, static_ctrl_compute_s=None,
                cand_is_oof=[True] * b_tr.shape[1],
                require_oof_bank=True, require_c_static=True,
            )
        except ValueError:
            refused_no_c_static = True
        assert refused_no_c_static
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    # ScopedFeatureCache + CoordinatorCAS checks
    cache = ScopedFeatureCache()
    s_base = FitScopeV2.from_arrays(x_all[train_idx], y_all[train_idx], fold_id="f0")
    cache.put(s_base, "expr_1", x_all[train_idx, 0])
    assert cache.get(s_base, "expr_1") is not None
    r_mod = y_all[train_idx].copy()
    r_mod[0] = np.nextafter(r_mod[0], np.inf)
    assert cache.get(FitScopeV2.from_arrays(x_all[train_idx], y_all[train_idx], residual=r_mod, fold_id="f0"), "expr_1") is None

    snap0 = EnsembleSnapshot(0, s_base.scope_sha256, (), -10.0, np.zeros(4), "d0")
    cas = CoordinatorCAS(snap0)
    ok_a, _ = cas.try_commit(
        WorkerProposal(0, "col_a", np.ones(4), None, 12.0),
        lambda p, cur: EnsembleSnapshot(cur.epoch + 1, cur.scope_sha256, cur.active_names + (p.name,), -8.0, cur.alpha_Dual, "d1"),
    )
    ok_b, _ = cas.try_commit(WorkerProposal(0, "col_a_dup", np.ones(4), None, 11.5), lambda _p, _cur: None)
    assert ok_a and (not ok_b) and cas.snapshot.epoch == 1
    return {
        "cli_disk_canary_max_abs_diff": cli_max_diff,
        "in_memory_canary_max_abs_diff": canary["max_abs_diff"],
        "refusals_verified": 4,
    }


def _check_g0_5_eta_zero_incumbent_identity() -> Dict[str, Any]:
    rng = np.random.default_rng(46005)
    y_tr = rng.normal(size=60)
    inc_tr = rng.normal(size=60)
    inc_va = rng.normal(size=25)
    st = EquilibriumState(y_train=y_tr, incumbent_train=inc_tr, incumbent_val=inc_va)
    st.add_group("g1", rng.normal(size=(60, 2)), phi_val=rng.normal(size=(25, 2)), weight=2.0)
    d_iqr = float(np.max(np.abs(st.predict_val(eta=0.0, apply_range_clamp=True, clamp_mode="search_iqr") - inc_va)))
    d_mad = float(np.max(np.abs(st.predict_val(eta=0.0, apply_range_clamp=True, clamp_mode="mad3") - inc_va)))
    assert d_iqr == 0.0 and d_mad == 0.0
    return {"eta_zero_iqr_max_diff": d_iqr, "eta_zero_mad3_max_diff": d_mad}


def _check_g0_6_single_row_26sd_outlier_guard() -> Dict[str, Any]:
    """Verify `cl_h458` 26-sd spike defense vs BOTH `L2` AND `L3` dense ridge (`Finding #10`)."""
    rng = np.random.default_rng(46006)
    n_tr, n_va = 180, 80
    x_tr = rng.normal(size=(n_tr, 4))
    x_va = rng.normal(size=(n_va, 4))
    clean_tr = 1.5 * x_tr[:, 0] - 0.9 * x_tr[:, 1]
    clean_va = 1.5 * x_va[:, 0] - 0.9 * x_va[:, 1]
    y_tr = clean_tr + 0.15 * rng.normal(size=n_tr)
    y_tr[0] += 26.0 * float(np.std(clean_tr))

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

    for mode_key in ("predictions_clamped", "predictions_unclamped"):
        preds = out[mode_key]
        m_l2 = float(np.mean((clean_va - preds["L2_serial_boost"]) ** 2))
        m_l3 = float(np.mean((clean_va - preds["L3_dense_ridge"]) ** 2))
        m_l4 = float(np.mean((clean_va - preds["L4_apex_eq"]) ** 2))
        assert m_l4 * 3.0 < m_l2, f"{mode_key}: L4={m_l4} did not beat L2={m_l2} by >3x"
        assert m_l4 * 3.0 < m_l3, f"{mode_key}: L4={m_l4} did not beat L3={m_l3} by >3x"

    mse_l2_c = float(np.mean((clean_va - out["predictions_clamped"]["L2_serial_boost"]) ** 2))
    mse_l3_c = float(np.mean((clean_va - out["predictions_clamped"]["L3_dense_ridge"]) ** 2))
    mse_l4_c = float(np.mean((clean_va - out["predictions_clamped"]["L4_apex_eq"]) ** 2))
    return {
        "admitted_columns": admitted,
        "mse_clamped_l2": round(mse_l2_c, 6),
        "mse_clamped_l3_dense_ridge": round(mse_l3_c, 6),
        "mse_clamped_l4_apex_eq": round(mse_l4_c, 6),
        "ratio_l3_over_l4": round(mse_l3_c / max(mse_l4_c, 1e-12), 2),
    }


def _check_g0_7_jacobi_discontinuous_and_rational() -> Dict[str, Any]:
    rng = np.random.default_rng(46007)
    n_tr, n_va = 220, 100
    x_tr = rng.uniform(-2.0, 2.0, size=(n_tr, 5))
    x_va = rng.uniform(-2.0, 2.0, size=(n_va, 5))
    # Inject discrete repeated-value regime on column 4 to exercise `scan_best_equality_features`
    x_tr[:36, 4] = 0.0
    x_va[:18, 4] = 0.0

    def _truth(x: np.ndarray) -> np.ndarray:
        jump = np.where(x[:, 0] > 0.35, 1.8 + 0.9 * x[:, 1], -1.2 - 0.7 * x[:, 1])
        eq_jump = np.where(np.round(x[:, 4], 8) == 0.0, 1.25, 0.0)
        rat = 1.1 * x[:, 2] / (1.0 + 1.4 * np.abs(x[:, 3]) + 0.5 * (x[:, 3] ** 2))
        return jump + eq_jump + rat

    y_tr = _truth(x_tr) + 0.12 * rng.normal(size=n_tr)
    y_va_clean = _truth(x_va)
    inc_tr = np.full(n_tr, float(np.mean(y_tr)), dtype=np.float64)
    inc_va = np.full(n_va, float(np.mean(y_tr)), dtype=np.float64)

    k_part_tr, _ = build_partition_kernel(x_tr, x_va)
    k_delta_tr, _ = build_delta_kernel(x_tr, x_va)
    eig_part = float(np.min(sla.eigvalsh(k_part_tr)))
    eig_delta = float(np.min(sla.eigvalsh(k_delta_tr)))
    assert eig_part >= -1e-10 and eig_delta >= -1e-10, "B1/B2 Gram matrices must be PSD"

    eq_tr, _, eq_names = scan_best_equality_features(
        x_tr, y_tr - inc_tr, np.eye(n_tr, dtype=np.float64), x_val=x_va
    )
    assert eq_tr.shape[1] >= 1 and any("eq(c4==0)" in nm for nm in eq_names), f"Expected eq(c4==0), got {eq_names}"

    # 1. End-to-end N1+N2+N4 standalone nested solver (`S1 JAC_V1` vs `L0` and `L3`)
    res = solve_nested_router_dictionary(
        x_tr, y_tr, x_va, inc_tr, inc_va, clamp_mode="search_iqr", placebo_trials=49, fold_placebo_trials=39
    )
    mse_inc = float(np.mean((y_va_clean - res["predictions_clamped"]["L0_incumbent"]) ** 2))
    mse_l3 = float(np.mean((y_va_clean - res["predictions_clamped"]["L3_dense_ridge"]) ** 2))
    mse_l4 = float(np.mean((y_va_clean - res["predictions_clamped"]["L4_apex_eq"]) ** 2))
    assert mse_l4 < 0.12 * mse_inc and mse_l4 < mse_l3, f"L4={mse_l4} vs inc={mse_inc}, L3={mse_l3}"

    # 2. 4-arm shared router harness (`R0`, `R1`, `R2`, `R3 C_STATIC`, PREREG_G1_DRAFT_v1 §2-§3, D1-D4)
    assert len(ROUTER_R0_CONFIG_GRID) == 27
    assert len(ROUTER_R1_CONFIG_GRID) == 30
    assert len(ROUTER_R2_L3_CONFIG_GRID) == 30
    assert len(ROUTER_R3_STATIC_CONFIG_GRID) == 30
    lab = np.arange(n_tr, dtype=np.int64) % 5
    x_mu = np.mean(x_tr, axis=0)
    x_sd = np.maximum(np.std(x_tr, axis=0), 1e-9)
    zs = (x_tr - x_mu) / x_sd
    za = (x_va - x_mu) / x_sd
    router_res = run_router_ratchet_harness(
        zs,
        za,
        y_tr,
        lab,
        inc_tr,
        inc_va,
        x_raw_search=x_tr,
        x_raw_audit=x_va,
        baseline_search_rmse=float(np.std(y_tr)) * 10.0,
        placebo_trials=39,
        include_r2_l3=True,
        include_r3_static=True,
    )
    assert router_res["R1"]["fired"] and router_res["R1"]["dictionary_fired"]
    assert router_res["changed_r1_vs_r0"] is True
    assert router_res["is_precision_search_proxy"] is True
    mse_r0 = float(np.mean((y_va_clean - router_res["R0"]["pred_aud"]) ** 2))
    mse_r1 = float(np.mean((y_va_clean - router_res["R1"]["pred_aud"]) ** 2))
    mse_r2 = float(np.mean((y_va_clean - router_res["R2"]["pred_aud"]) ** 2))
    mse_r3 = float(np.mean((y_va_clean - router_res["R3"]["pred_aud"]) ** 2))
    assert mse_r1 < mse_r0 and mse_r1 < mse_r2 and mse_r1 < mse_r3, (
        f"Router R1 ({mse_r1}) must beat R0 ({mse_r0}), R2 ({mse_r2}), and R3 ({mse_r3})"
    )
    r1_cpu = float(router_res["R1"]["worker_cpu_s"])
    r3_cpu = float(router_res["R3"]["worker_cpu_s"])
    cpu_ratio = r3_cpu / max(r1_cpu, 1e-6)
    assert cpu_ratio >= 0.90, f"R3 one-sided compute floor failed (Q5): R1={r1_cpu:.4f}s, R3={r3_cpu:.4f}s"

    return {
        "b1_min_eig": eig_part,
        "b2_min_eig": eig_delta,
        "equality_feature_detected": eq_names[0],
        "mse_incumbent": round(mse_inc, 6),
        "mse_l3_dense_ridge": round(mse_l3, 6),
        "mse_l4_nested_router": round(mse_l4, 6),
        "router_harness": {
            "mse_r0_asis": round(mse_r0, 6),
            "mse_r1_dict": round(mse_r1, 6),
            "mse_r2_l3": round(mse_r2, 6),
            "mse_r3_c_static": round(mse_r3, 6),
            "r1_cpu_s": round(r1_cpu, 4),
            "r3_cpu_s": round(r3_cpu, 4),
            "r3_over_r1_cpu_ratio": round(cpu_ratio, 4),
        },
    }


def _g0_8_single_fit_worker(fit_idx: int) -> Tuple[bool, bool, bool, bool, bool, float, float]:
    """Run one pure-noise `G0_8` fit (`n=180, p=6`) in BOTH standalone AND router-ratchet (`D3`) forms."""
    seed = 4608000 + int(fit_idx)
    rng = np.random.default_rng(seed)
    n_rows, n_cols = 180, 6
    x_tr = rng.normal(size=(n_rows, n_cols))
    x_va = rng.normal(size=(36, n_cols))
    y_tr = rng.normal(size=n_rows)
    inc_tr = np.full(n_rows, float(np.mean(y_tr)), dtype=np.float64)
    inc_va = np.full(36, float(np.mean(y_tr)), dtype=np.float64)

    eq_cfg = EquilibriumConfig(robust_irls=True)
    res_var = max(float(np.var(y_tr - inc_tr)), 1e-6)
    st = EquilibriumState(
        y_train=y_tr,
        incumbent_train=inc_tr,
        anchor_h_train=np.ones((n_rows, 1), dtype=np.float64),
        tau_h2=max(1.0, 4.0 * res_var),
        cfg=eq_cfg,
    )
    st.update_robust_noise_irls()

    pre_x = PrecomputedXContext.from_x(x_tr, None)
    bank_tr, _, cand_names = build_production_router_dictionary(
        x_tr, st.alpha, st.inv_cov, x_val=None, precomputed=pre_x
    )
    mu = np.mean(bank_tr, axis=0)
    sd = np.maximum(np.std(bank_tr, axis=0), 1e-9)
    z_tr = (bank_tr - mu) / sd

    # 1. Production nested placebo (G0_8.4): rebuilds the WHOLE bank on every permuted residual
    c0_nested = calibrate_nested_discovery_placebo_c0(
        st,
        discovery_fn=lambda null_st: build_production_router_dictionary(
            x_tr, null_st.alpha, null_st.inv_cov, x_val=None, precomputed=pre_x
        )[0],
        n_trials=49,
        target_false_admissions_per_fit=0.025,
        seed=4609000 + int(fit_idx),
    )
    dec_nested = st.screen_dictionary(z_tr, cand_names, c0_override=c0_nested)

    # 2. Power check (G0_8.6): exact 38ff3faf fixed-pool placebo on the SAME built bank
    c0_fixed_38ff3faf = calibrate_placebo_c0(
        st,
        z_tr,
        n_trials=200,
        target_false_admissions_per_fit=0.05,
        seed=4610000 + int(fit_idx),
        legacy_38ff3faf_fudge=0.5,
    )
    dec_fixed = st.screen_dictionary(z_tr, cand_names, c0_override=c0_fixed_38ff3faf)

    # 3. Router-form G0_8 (`cl_fwriter` D3 & PREREG_G1_DRAFT_v1 §4.1): R0 (27) vs R1 (30) through ratchet
    lab = np.arange(n_rows, dtype=np.int64) % 5
    x_mu = np.mean(x_tr, axis=0)
    x_sd = np.maximum(np.std(x_tr, axis=0), 1e-9)
    r_out = run_router_ratchet_harness(
        (x_tr - x_mu) / x_sd,
        (x_va - x_mu) / x_sd,
        y_tr,
        lab,
        inc_tr,
        inc_va,
        x_raw_search=x_tr,
        x_raw_audit=x_va,
        placebo_trials=39,
        seed=4620000 + 100 * int(fit_idx),
        include_r2_l3=False,
        include_r3_static=False,
    )
    return (
        dec_nested.action == "ADD",
        dec_fixed.action == "ADD",
        bool(r_out["R0"]["fired"]),
        bool(r_out["R1"]["fired"]),
        bool(r_out["R1"]["dictionary_fired"]),
        float(c0_nested),
        float(c0_fixed_38ff3faf),
    )


def _check_g0_8_nested_discovery_placebo() -> Dict[str, Any]:
    """`cl_fwriter` `G0_8` + `D3` router-form `G0_8`: `F = 400` pure-noise fits."""
    t0 = time.perf_counter()
    n_fits = 400
    n_workers = max(1, min(4, os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        outcomes = list(pool.map(_g0_8_single_fit_worker, range(n_fits), chunksize=10))

    nested_false = sum(1 for item in outcomes if item[0])
    fixed_false = sum(1 for item in outcomes if item[1])
    r0_fires = sum(1 for item in outcomes if item[2])
    r1_fires = sum(1 for item in outcomes if item[3])
    r1_dict_fires = sum(1 for item in outcomes if item[4])
    c0_nested_vals = np.array([item[5] for item in outcomes], dtype=np.float64)
    c0_fixed_vals = np.array([item[6] for item in outcomes], dtype=np.float64)

    nested_rate = nested_false / float(n_fits)
    fixed_rate = fixed_false / float(n_fits)
    r0_fire_rate = r0_fires / float(n_fits)
    r1_fire_rate = r1_fires / float(n_fits)
    r1_dict_fire_rate = r1_dict_fires / float(n_fits)

    cp95_nested = _clopper_pearson_upper_95(nested_false, n_fits)
    cp95_fixed = _clopper_pearson_upper_95(fixed_false, n_fits)
    cp95_r0_fire = _clopper_pearson_upper_95(r0_fires, n_fits)
    cp95_r1_fire = _clopper_pearson_upper_95(r1_fires, n_fits)
    cp95_r1_dict_fire = _clopper_pearson_upper_95(r1_dict_fires, n_fits)
    wall_s = time.perf_counter() - t0

    assert nested_rate <= 0.05, (
        f"G0_8 FAIL: nested discovery placebo false admission rate {nested_rate:.4f} > 0.05"
    )
    assert fixed_rate >= 0.10, (
        f"G0_8 POWER CHECK FAIL: fixed-pool 38ff3faf rate {fixed_rate:.4f} was not clearly above 0.05"
    )
    assert r0_fire_rate <= 0.05 and r1_dict_fire_rate <= 0.05 and r1_fire_rate <= 0.05, (
        f"G0_8 D3 ROUTER FAIL: R0={r0_fire_rate:.4f}, R1={r1_fire_rate:.4f} (dict={r1_dict_fire_rate:.4f}) > 0.05"
    )
    return {
        "c0_fixed_38ff3faf_median": round(float(np.median(c0_fixed_vals)), 4),
        "c0_nested_median": round(float(np.median(c0_nested_vals)), 4),
        "c0_nested_q90": round(float(np.quantile(c0_nested_vals, 0.90)), 4),
        "clopper_pearson_upper_95_fixed": round(cp95_fixed, 4),
        "clopper_pearson_upper_95_nested": round(cp95_nested, 4),
        "code_sha256": _package_code_sha256(),
        "f_fits": n_fits,
        "fixed_pool_38ff3faf_false_admissions": fixed_false,
        "fixed_pool_38ff3faf_rate": round(fixed_rate, 4),
        "n_workers": n_workers,
        "nested_false_admissions": nested_false,
        "nested_rate": round(nested_rate, 4),
        "router_form_d3": {
            "clopper_pearson_upper_95_r0_fire": round(cp95_r0_fire, 4),
            "clopper_pearson_upper_95_r1_dict_fire": round(cp95_r1_dict_fire, 4),
            "clopper_pearson_upper_95_r1_fire": round(cp95_r1_fire, 4),
            "r0_grid_size": len(ROUTER_R0_CONFIG_GRID),
            "r0_null_fire_rate": round(r0_fire_rate, 4),
            "r0_null_fires": r0_fires,
            "r1_grid_size": len(ROUTER_R1_CONFIG_GRID),
            "r1_null_dictionary_fire_rate": round(r1_dict_fire_rate, 4),
            "r1_null_dictionary_fires": r1_dict_fires,
            "r1_null_fire_rate": round(r1_fire_rate, 4),
            "r1_null_fires": r1_fires,
        },
        "seed_range": [4608000, 4608000 + n_fits - 1],
        "settings": {
            "fold_placebo_trials_router": 39,
            "include_b1_b2_kernels": True,
            "include_equality_features": True,
            "incumbent": "train_mean",
            "max_gated_basis_cols": 4,
            "max_rational_pairs": 6,
            "max_step_return": 12,
            "n_rows": 180,
            "p_features": 6,
            "placebo_trials": 49,
            "target_false_admissions_per_fit": 0.025,
            "top_per_col": 3,
        },
        "wall_time_s": round(wall_s, 3),
    }


def _check_g0_9_v2_adversarial_witnesses() -> Dict[str, Any]:
    """Verify all `v2` spec hardening fixes (`D1`, `D3/D4`, `F03/B.1`, `F05/B.3`, `F06/B.2`, `F08`, `F10`)."""
    rng = np.random.default_rng(46009)

    # 1. ARENA D1: Full affine target equivariance y -> 37.5 * y - 142.0 with LOO noise-fraction selection
    x_tr = rng.normal(size=(90, 4))
    x_va = rng.normal(size=(35, 4))
    y_base = 1.8 * x_tr[:, 0] - 0.7 * (x_tr[:, 1] ** 2) + 0.05 * rng.normal(size=90)
    a_aff, b_aff = 37.5, -142.0
    y_aff = a_aff * y_base + b_aff
    phi_tr = np.column_stack([x_tr[:, 0], x_tr[:, 1] ** 2])
    phi_va = np.column_stack([x_va[:, 0], x_va[:, 1] ** 2])

    def _fit_affine_state(y_in: np.ndarray) -> np.ndarray:
        cfg_d = EquilibriumConfig.for_regime("DETERMINISTIC")
        st = EquilibriumState(y_train=y_in, cfg=cfg_d)
        var_y = max(float(np.var(st.residual)), 1e-12)
        st.add_group("basis", phi_tr, phi_val=phi_va, weight=10.0 * var_y)
        st.select_loo_noise_scale()
        return st.predict_val(eta=1.0, apply_range_clamp=True)

    p_base = _fit_affine_state(y_base)
    p_aff = _fit_affine_state(y_aff)
    aff_rel_err = float(
        np.max(np.abs((a_aff * p_base + b_aff) - p_aff)) / max(float(np.std(p_aff)), 1e-9)
    )
    assert aff_rel_err < 1e-10, f"D1 affine equivariance drift: {aff_rel_err}"

    # 2. Learner Compiler Witness B.1 (F03): hard per-fold floor blocks 44% single-fold blowup
    folds_5 = [np.array([k], dtype=np.int64) for k in range(5)]
    res_5 = np.ones(5, dtype=np.float64)
    # Candidate leaves residuals [0.8, 0.8, 0.8, 1.0, -1.2] -> fold MSEs [0.64, 0.64, 0.64, 1.0, 1.44]
    oof_b1 = np.array([0.2, 0.2, 0.2, 0.0, 2.2], dtype=np.float64)
    st_legacy = EquilibriumState(
        y_train=res_5,
        cfg=EquilibriumConfig(enforce_hard_fold_floor=False, damping_grid=(0.0, 1.0)),
    )
    eta_legacy, _ = st_legacy.select_damping_eta_from_oof(oof_b1, folds_5)
    st_hard = EquilibriumState(
        y_train=res_5,
        cfg=EquilibriumConfig(enforce_hard_fold_floor=True, damping_grid=(0.0, 1.0)),
    )
    eta_hard, _ = st_hard.select_damping_eta_from_oof(oof_b1, folds_5)
    assert eta_legacy == 1.0 and eta_hard == 0.0, f"Witness B.1 failed: legacy={eta_legacy}, hard={eta_hard}"

    # 3. Learner Compiler Witness B.2 (F06): zero-residual covariance collision check
    z_zero = np.zeros(24, dtype=np.float64)
    st_cov1 = EquilibriumState(y_train=z_zero, cfg=EquilibriumConfig(auto_scale_sigma_n2=False))
    st_cov2 = EquilibriumState(y_train=z_zero, cfg=EquilibriumConfig(auto_scale_sigma_n2=False))
    st_cov1.add_group("same_name", rng.normal(size=(24, 1)), weight=1.0)
    st_cov2.add_group("same_name", rng.normal(size=(24, 1)), weight=1.0)
    assert np.allclose(st_cov1.alpha, st_cov2.alpha) and st_cov1.state_digest() != st_cov2.state_digest()

    # 4. Learner Compiler Witness B.3 (F05): frozen_gaussian_evidence_delta == exact dense Cholesky delta
    y_out = 1.6 * x_tr[:60, 0] + 0.2 * rng.normal(size=60)
    y_out[0] += 12.0
    st_ev = EquilibriumState(y_train=y_out, cfg=EquilibriumConfig(robust_irls=True))
    st_ev.update_robust_noise_irls()
    ev_0 = st_ev.marginal_log_evidence()
    col_ev = (x_tr[:60, 0] - np.mean(x_tr[:60, 0])) / np.std(x_tr[:60, 0])
    dec_ev = st_ev.screen_dictionary(col_ev[:, None], ["c0_sig"], c0_override=0.5)
    assert dec_ev.action == "ADD"
    st_ev.add_group("c0_sig", col_ev, weight=dec_ev.optimal_weight)
    ev_1 = st_ev.marginal_log_evidence()
    ev_diff = abs(dec_ev.frozen_gaussian_evidence_delta - (ev_1 - ev_0))
    assert ev_diff < 1e-11 and dec_ev.frozen_gaussian_evidence_delta != dec_ev.robust_acquisition_gain

    # 5. Learner Compiler F08: read-only snapshot/proposal buffers + scope/role CAS verification
    snap_ro = EnsembleSnapshot(0, "scope_ok", (), -5.0, np.ones(8), "dig_0")
    prop_ro = WorkerProposal(0, "p1", np.ones(8), np.ones(4), 9.0, proposal_scope_sha256="scope_wrong")
    mutated = False
    try:
        snap_ro.alpha_Dual[0] = 99.0
        mutated = True
    except ValueError:
        pass
    try:
        prop_ro.train_col[0] = 99.0
        mutated = True
    except ValueError:
        pass
    assert not mutated, "Snapshot/proposal buffers must be strictly read-only"
    cas_ro = CoordinatorCAS(snap_ro)
    ok_wrong, _ = cas_ro.try_commit(
        prop_ro,
        lambda p, cur: EnsembleSnapshot(cur.epoch + 1, cur.scope_sha256, (p.name,), -4.0, cur.alpha_Dual, "dig_1"),
    )
    assert not ok_wrong, "CoordinatorCAS must reject mismatched proposal_scope_sha256"

    # 6. Solver-Trained F10: Full-bank Golub-Pereyra VarPro gradient parity + anchor cancellation
    x_vp = rng.uniform(-1.5, 1.5, size=(140, 3))
    x_vp_va = rng.uniform(-1.5, 1.5, size=(50, 3))

    def _rat_truth(xm: np.ndarray) -> np.ndarray:
        zu = (xm[:, 0] - float(np.mean(x_vp[:, 0]))) / float(np.std(x_vp[:, 0]))
        zv = (xm[:, 1] - float(np.mean(x_vp[:, 1]))) / float(np.std(x_vp[:, 1]))
        return 2.5 * zu / (1.0 + 0.9 * np.abs(zv) + 0.6 * (zv ** 2))

    y_vp = _rat_truth(x_vp)
    y_vp_va = _rat_truth(x_vp_va)
    bad_anchor_tr = -3.0 * (x_vp[:, 1] ** 3)  # corrupted incumbent like YJ tail explosion
    bad_anchor_va = -3.0 * (x_vp_va[:, 1] ** 3)
    vp_res = fit_rational_law_varpro_full(
        x_vp,
        y_vp - bad_anchor_tr,
        u_col=0,
        v_col=1,
        x_val=x_vp_va,
        anchor_train=bad_anchor_tr,
        anchor_val=bad_anchor_va,
        allow_anchor_cancel=True,
        n_steps=12,
    )
    eval_fn = vp_res["eval_fn"]
    eps_fd = 1e-6
    g_fd = np.array(
        [
            (eval_fn(-0.5 + eps_fd, -1.5) - eval_fn(-0.5 - eps_fd, -1.5)) / (2.0 * eps_fd),
            (eval_fn(-0.5, -1.5 + eps_fd) - eval_fn(-0.5, -1.5 - eps_fd)) / (2.0 * eps_fd),
        ],
        dtype=np.float64,
    )
    vp_grad_err = float(np.max(np.abs(vp_res["initial_grad"] - g_fd)))
    assert vp_grad_err < 1e-6, f"VarPro reduced gradient mismatch: {vp_grad_err}"
    repaired_pred_va = bad_anchor_va + vp_res["pred_val"]
    vp_val_rmse = float(np.sqrt(np.mean((y_vp_va - repaired_pred_va) ** 2)))
    assert vp_res["delta_anchor"] < -0.95 and vp_val_rmse < 1e-3, (
        f"Anchor cancellation failed: delta_a={vp_res['delta_anchor']}, rmse={vp_val_rmse}"
    )

    # 7. Target-stratified K-fold check: top-5 tail rows distributed 1-per-fold (5/5 folds)
    y_heavy = np.concatenate([rng.normal(size=95), np.array([25.0, 30.0, 40.0, 55.0, 80.0])])
    strat_lab = target_stratified_kfold_labels(y_heavy, n_splits=5, seed=460)
    assert len(set(strat_lab[-5:].tolist())) == 5

    return {
        "affine_d1_rel_err": aff_rel_err,
        "anchor_cancel_delta_a": round(float(vp_res["delta_anchor"]), 6),
        "anchor_cancel_val_rmse": vp_val_rmse,
        "b1_hard_floor_eta": eta_hard,
        "b3_gaussian_ev_abs_err": ev_diff,
        "stratified_top5_distinct_folds": len(set(strat_lab[-5:].tolist())),
        "varpro_grad_fd_max_err": vp_grad_err,
    }


def run_all_selftests() -> Dict[str, Any]:
    results = {
        "g0_1_schur_duplicate_scale_and_rank_scan": _check_g0_1_schur_duplicate_scale_and_rank_scan(),
        "g0_2_closed_form_loo_and_lfo": _check_g0_2_closed_form_loo_and_lfo(),
        "g0_3_placebo_calibration": _check_g0_3_placebo_calibration(),
        "g0_4_outer_fold_poisoning_and_cache": _check_g0_4_outer_fold_poisoning_and_cache(),
        "g0_5_eta_zero_incumbent_identity": _check_g0_5_eta_zero_incumbent_identity(),
        "g0_6_single_row_26sd_outlier_guard": _check_g0_6_single_row_26sd_outlier_guard(),
        "g0_7_jacobi_discontinuous_and_rational": _check_g0_7_jacobi_discontinuous_and_rational(),
        "g0_8_nested_discovery_placebo": _check_g0_8_nested_discovery_placebo(),
        "g0_9_v2_adversarial_witnesses": _check_g0_9_v2_adversarial_witnesses(),
    }
    return {
        "all_passed": True,
        "code_sha256": _package_code_sha256(),
        "n_checks": len(results),
        "results": results,
        "schema": "jope.h460.g0_selftest/v2",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true", help="Run all 8 deterministic Gate G0 checks")
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
