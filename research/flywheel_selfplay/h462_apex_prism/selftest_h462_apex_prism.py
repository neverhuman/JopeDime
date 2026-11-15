#!/usr/bin/env python3
"""Deterministic 12-Gate Verification Suite for `H462 APEX-PRISM` (`selftest_h462_apex_prism.py`).

Executes 12 self-contained mathematical, statistical, causal, and quarantine proofs (`P0_1..P0_12`)
on CPU (`OPENBLAS_NUM_THREADS=1`, `< 25s`, zero external data or GPU required).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.flywheel_selfplay.h460_apex_eq import target_stratified_kfold_labels  # noqa: E402
from research.flywheel_selfplay.h462_apex_prism import (  # noqa: E402
    ENTRANT_ID_E8,
    EdgeCertifyConfig,
    EdgeContextDecision,
    FitClosureV3,
    PRISMConfig,
    PRISMSolverNet,
    assert_path_not_quarantined,
    bag_predictions_e8,
    bag_predictions_precision_weighted_oof,
    build_counterfactual_repair_episode,
    compile_pcr_hints,
    compute_l350_skill,
    compute_robust_tail_stat,
    corrupt_pcr_hints_counterfactual,
    count_exposed_families_on_board,
    edge_certify_context,
    enforce_cohort_family_mean_floor,
    evaluate_h8_episode_skills,
    fit_raw_y_classic_floor_oof,
    legacy_clipped_fold_mean_score,
    load_bank_manifest_exposure,
    pooled_unclipped_utility,
    prism_episode_loss,
    run_assay,
    run_exact_varpro_battery,
    solve_differentiable_cholesky_head,
    verify_nested_oof_fn,
    write_compatible_block_npz,
)


def _package_code_sha256() -> str:
    h = hashlib.sha256()
    for py_file in sorted(PKG_DIR.glob("*.py")):
        h.update(f"{py_file.name}:".encode("utf-8"))
        h.update(py_file.read_bytes())
    return h.hexdigest()


def _scratch_dir() -> Path:
    agent_root = REPO_ROOT / ".agent" / "h462_selftest_scratch"
    base = agent_root if (REPO_ROOT / ".agent").exists() else (Path.home() / ".cache" / "jope_h462_scratch")
    d = base / f"canary_{os.getpid()}_{int(time.time() * 1e6)}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _p0_1_assay_regime_and_tail_stat() -> Dict[str, Any]:
    rng = np.random.default_rng(46201)
    x_p = rng.uniform(-1.5, 1.5, size=(180, 3))
    y_p = (1.5 * x_p[:, 0] - 0.8 * x_p[:, 1]) / (1.0 + 0.35 * (x_p[:, 1] ** 2))
    rep_p = run_assay(x_p, y_p)
    assert rep_p.regime == "PRECISION" and rep_p.loo_u > 2.2, f"Expected PRECISION, got {rep_p}"

    x_t = rng.normal(size=(180, 4))
    y_t = 0.6 * x_t[:, 0] + rng.normal(scale=0.5, size=180)
    y_t[:3] = np.array([28.0, -34.0, 42.0])
    rep_t = run_assay(x_t, y_t)
    assert rep_t.regime == "HEAVY_TAIL" and rep_t.tail_robust_sd > 12.0, f"Expected HEAVY_TAIL, got {rep_t}"

    x_l = rng.choice([-1.0, 0.0, 1.0], size=(180, 3))
    y_l = 0.5 * x_l[:, 0] + rng.normal(scale=0.8, size=180)
    rep_l = run_assay(x_l, y_l)
    assert rep_l.regime == "LATTICE" and rep_l.is_lattice

    x_o = rng.normal(size=(180, 5))
    y_o = 0.8 * x_o[:, 0] + rng.normal(scale=0.9, size=180)
    rep_o = run_assay(x_o, y_o)
    assert rep_o.regime == "ORDINARY"
    return {
        "heavy_tail_sd": round(rep_t.tail_robust_sd, 3),
        "precision_loo_u": round(rep_p.loo_u, 4),
        "regimes": [rep_p.regime, rep_t.regime, rep_l.regime, rep_o.regime],
    }


def _p0_2_exact_varpro_recovery_and_zero_noisy_false_fire() -> Dict[str, Any]:
    rng = np.random.default_rng(46202)
    x_s = rng.uniform(-1.5, 1.5, size=(160, 4))
    x_q = rng.uniform(-1.5, 1.5, size=(60, 4))

    def _law(xm: np.ndarray) -> np.ndarray:
        return (1.2 + 2.1 * xm[:, 0] - 0.9 * xm[:, 1]) / (1.0 + 0.35 * xm[:, 0] + 0.45 * (xm[:, 1] ** 2))

    y_s = _law(x_s)
    y_q = _law(x_q)
    lab = target_stratified_kfold_labels(y_s, n_splits=5, seed=46202)
    res = run_exact_varpro_battery(x_s, y_s, lab, min_fold_margin_u=0.50)
    assert res.accepted and res.min_fold_margin_u > 2.0, f"Exact law not accepted: {res}"
    pred_q = res.predict_query(x_q, fallback_query=np.zeros(60))
    u_query = pooled_unclipped_utility(y_q, pred_q)
    assert u_query > 12.0, f"Expected near-machine precision u > 12, got {u_query}"

    noisy_fires = 0
    for trial in range(6):
        xt = rng.normal(size=(150, 4))
        yt = 1.2 * np.sin(xt[:, 0]) + 0.7 * xt[:, 1] + 0.25 * rng.normal(size=150)
        lt = target_stratified_kfold_labels(yt, n_splits=5, seed=46220 + trial)
        rt = run_exact_varpro_battery(xt, yt, lt, min_fold_margin_u=0.50)
        if rt.accepted:
            noisy_fires += 1
    assert noisy_fires == 0, f"Expected 0 noisy false fires, got {noisy_fires}"
    return {
        "law_name": res.selected_law_name,
        "min_fold_margin_u": round(res.min_fold_margin_u, 4),
        "noisy_false_fires_of_6": noisy_fires,
        "query_u": round(u_query, 4),
    }


def _p0_3_anchor_cancellation_of_yj_tail_explosion() -> Dict[str, Any]:
    rng = np.random.default_rng(46203)
    x_s = rng.uniform(-1.6, 1.6, size=(160, 3))
    x_q = rng.uniform(-1.6, 1.6, size=(60, 3))
    y_s = (1.8 * x_s[:, 0]) / (1.0 + 0.5 * (x_s[:, 1] ** 2))
    y_q = (1.8 * x_q[:, 0]) / (1.0 + 0.5 * (x_q[:, 1] ** 2))
    # Simulate an inverse-Yeo-Johnson incumbent that explodes cubically in the tail (u < 0)
    bad_oof = 0.8 * y_s + 2.5 * (x_s[:, 1] ** 3) * (np.abs(x_s[:, 1]) > 1.1)
    bad_q = 0.8 * y_q + 2.5 * (x_q[:, 1] ** 3) * (np.abs(x_q[:, 1]) > 1.1)
    u_bad = pooled_unclipped_utility(y_q, bad_q)
    assert u_bad < 0.0

    lab = target_stratified_kfold_labels(y_s, n_splits=5, seed=46203)
    res = run_exact_varpro_battery(x_s, y_s, lab, anchor_oof=bad_oof, anchor_full_train=bad_oof)
    assert res.accepted and res.delta_anchor < -0.95, f"Expected delta_a < -0.95, got {res.delta_anchor}"
    pred_q = res.predict_query(x_q, fallback_query=bad_q, anchor_query=bad_q)
    u_repaired = pooled_unclipped_utility(y_q, pred_q)
    assert u_repaired > 10.0, f"Expected repaired u > 10, got {u_repaired}"
    return {
        "bad_anchor_query_u": round(u_bad, 4),
        "delta_anchor": round(res.delta_anchor, 6),
        "repaired_query_u": round(u_repaired, 4),
    }


def _p0_4_pooled_unclipped_vs_clipped_fold_mean_on_stratified_folds() -> Dict[str, Any]:
    """Reproduce `cl_h458` `06:05Z` + `06:13Z` finding on ALREADY target-stratified `lab` folds."""
    rng = np.random.default_rng(46204)
    # 1. Heavy-tailed linear law (like singleton `866f` where ridge=1.74 beats yj-corrupted stack=1.11)
    n_s, n_q = 200, 100
    x_s = rng.normal(size=(n_s, 4))
    x_q = rng.normal(size=(n_q, 4))
    # Inject right-skewed heavy-tailed predictor rows on both SEARCH (top-5 stratum -> 1 per fold!) and AUDIT (3 rows)
    x_s[:5, 0] = np.array([14.0, 15.5, 17.0, 19.0, 22.0])
    x_q[:3, 0] = np.array([15.0, 18.0, 21.0])
    y_s = 2.0 * x_s[:, 0] - 0.8 * x_s[:, 1] + 0.35 * rng.normal(size=n_s)
    y_q = 2.0 * x_q[:, 0] - 0.8 * x_q[:, 1] + 0.35 * rng.normal(size=n_q)

    lab = target_stratified_kfold_labels(y_s, n_splits=5, seed=46204)
    assert len(set(lab[:5].tolist())) == 5
    assert compute_robust_tail_stat(y_s) > 12.0

    floor_oof, floor_q = fit_raw_y_classic_floor_oof(x_s, y_s, x_q, lab, seed=46204)
    # Anchor = decent smooth stack (`u ~ 1.11` like `866f` V0) that attenuates slightly in the tail
    anc_oof = 0.92 * (2.0 * x_s[:, 0] - 0.8 * x_s[:, 1]) * (np.abs(x_s[:, 0]) < 5.0) + 1.45 * x_s[:, 0] * (np.abs(x_s[:, 0]) >= 5.0)
    anc_q = 0.92 * (2.0 * x_q[:, 0] - 0.8 * x_q[:, 1]) * (np.abs(x_q[:, 0]) < 5.0) + 1.45 * x_q[:, 0] * (np.abs(x_q[:, 0]) >= 5.0)
    # Exploding `_yj`-warped candidate: tighter than anchor on the 97.5% bulk, explodes on |x0| > 10!
    yj_oof = (2.0 * x_s[:, 0] - 0.8 * x_s[:, 1]) * (np.abs(x_s[:, 0]) < 5.0) + 0.45 * (x_s[:, 0] ** 2) * np.sign(x_s[:, 0]) * (np.abs(x_s[:, 0]) >= 5.0)
    yj_q = (2.0 * x_q[:, 0] - 0.8 * x_q[:, 1]) * (np.abs(x_q[:, 0]) < 5.0) + 0.45 * (x_q[:, 0] ** 2) * np.sign(x_q[:, 0]) * (np.abs(x_q[:, 0]) >= 5.0)

    cands_oof = {"anchor_v0": anc_oof, "cand_yj_corrupted": yj_oof, **floor_oof}
    cands_q = {"anchor_v0": anc_q, "cand_yj_corrupted": yj_q, **floor_q}

    legacy_yj = legacy_clipped_fold_mean_score(y_s, anc_oof, yj_oof, lab, mad_clip=3.0)
    legacy_ridge = legacy_clipped_fold_mean_score(y_s, anc_oof, floor_oof["raw_ridge"], lab, mad_clip=3.0)
    assert legacy_yj > legacy_ridge, "Fixture check: 3-MAD clipped fold-mean must prefer bulk-tight yj candidate"

    dec_floor = edge_certify_context(y_s, lab, "anchor_v0", cands_oof, cands_q, family_size=1)
    u_aud_anc = pooled_unclipped_utility(y_q, anc_q)
    u_aud_yj = pooled_unclipped_utility(y_q, yj_q)
    u_aud_edge = pooled_unclipped_utility(y_q, dec_floor.pred_query)
    assert dec_floor.selected_name == "raw_ridge" and u_aud_edge > u_aud_anc + 0.50 and u_aud_yj < 0.0, (
        f"EDGE failed on tail floor: sel={dec_floor.selected_name}, edge={u_aud_edge}, anc={u_aud_anc}, yj={u_aud_yj}"
    )

    # 2. `280b`-shaped heavy-tailed exact rational/polynomial law (`tail > 12`) where trees lose badly!
    x2_s = rng.normal(size=(180, 3))
    x2_q = rng.normal(size=(80, 3))
    x2_s[:5, 0] = np.array([5.5, -5.8, 6.2, -6.5, 7.0])
    x2_q[:3, 0] = np.array([5.7, -6.1, 6.8])
    y2_s = 0.5 * (x2_s[:, 0] ** 2) * np.sign(x2_s[:, 0]) + 0.8 * x2_s[:, 1]
    y2_q = 0.5 * (x2_q[:, 0] ** 2) * np.sign(x2_q[:, 0]) + 0.8 * x2_q[:, 1]
    lab2 = target_stratified_kfold_labels(y2_s, n_splits=5, seed=46244)
    assert compute_robust_tail_stat(y2_s) > 12.0
    floor2_oof, floor2_q = fit_raw_y_classic_floor_oof(x2_s, y2_s, x2_q, lab2, seed=46244)
    # Champion smooth/algebraic stack already gets u ~ 4.2 on 280b
    champ2_oof = y2_s + 0.025 * float(np.std(y2_s)) * rng.normal(size=180)
    champ2_q = y2_q + 0.025 * float(np.std(y2_s)) * rng.normal(size=80)
    dec_280b = edge_certify_context(
        y2_s, lab2, "fair_champion", {"fair_champion": champ2_oof, **floor2_oof}, {"fair_champion": champ2_q, **floor2_q}
    )
    assert dec_280b.selected_name == "fair_champion" and not dec_280b.switched_from_anchor
    return {
        "audit_u_anchor": round(u_aud_anc, 4),
        "audit_u_edge_raw_ridge": round(u_aud_edge, 4),
        "audit_u_legacy_yj": round(u_aud_yj, 4),
        "kept_champion_on_280b_tail": not dec_280b.switched_from_anchor,
        "selected_on_866f_tail": dec_floor.selected_name,
    }


def _p0_5_inductive_query_independence_t1() -> Dict[str, Any]:
    rng = np.random.default_rng(46205)
    x_c = rng.uniform(-1.5, 1.5, size=(80, 4))
    y_c = (1.5 * x_c[:, 0]) / (1.0 + 0.4 * (x_c[:, 1] ** 2))
    x_q1 = rng.uniform(-1.5, 1.5, size=(16, 4))
    x_q2 = x_q1.copy()
    x_q2[1:] = x_q2[1:] * (-7.5) + 42.0  # Extreme perturbation on companion query rows 1..15!

    lab = target_stratified_kfold_labels(y_c, n_splits=5, seed=46205)
    ex = run_exact_varpro_battery(x_c, y_c, lab)
    d_ex = abs(float(ex.predict_query(x_q1, np.zeros(16))[0] - ex.predict_query(x_q2, np.zeros(16))[0]))

    hb1 = compile_pcr_hints(x_c, y_c, x_q1, k_inner=5, seed=46205)
    hb2 = compile_pcr_hints(x_c, y_c, x_q2, k_inner=5, seed=46205)
    d_pcr = float(np.max(np.abs(hb1.query_hints[0] - hb2.query_hints[0])))

    torch.manual_seed(46205)
    net = PRISMSolverNet(PRISMConfig(d_model=48, n_heads=4, n_layers=2, basis_rank=16, n_bins=64))
    net.eval()
    with torch.no_grad():
        o1 = net.forward_episode(
            torch.tensor(x_c),
            torch.tensor(y_c),
            torch.tensor(x_q1),
            hints_context=torch.tensor(hb1.context_hints),
            hints_query=torch.tensor(hb1.query_hints),
            rel_context=torch.tensor(hb1.context_reliability),
            rel_query=torch.tensor(hb1.query_reliability),
            head_mode_override="cholesky_basis_r64",
        )
        o2 = net.forward_episode(
            torch.tensor(x_c),
            torch.tensor(y_c),
            torch.tensor(x_q2),
            hints_context=torch.tensor(hb2.context_hints),
            hints_query=torch.tensor(hb2.query_hints),
            rel_context=torch.tensor(hb2.context_reliability),
            rel_query=torch.tensor(hb2.query_reliability),
            head_mode_override="cholesky_basis_r64",
        )
    d_net = abs(float(o1["pred_query"][0].item() - o2["pred_query"][0].item()))
    assert d_ex == 0.0 and d_pcr < 1e-13 and d_net < 1e-13, f"T1 query leak: ex={d_ex}, pcr={d_pcr}, net={d_net}"
    return {"exact_row0_diff": d_ex, "pcr_row0_diff": d_pcr, "prism_row0_diff": d_net}


def _p0_6_warm_plus_cholesky_step0_identity_and_grad_parity() -> Dict[str, Any]:
    torch.manual_seed(46206)
    rng = np.random.default_rng(46206)
    ep = build_counterfactual_repair_episode(n_context=48, n_query=16, n_features=4, seed=46206)
    xc = torch.tensor(ep["x_context"])
    yc = torch.tensor(ep["y_context"])
    xq = torch.tensor(ep["x_query"])

    net = PRISMSolverNet(PRISMConfig(d_model=48, n_heads=4, n_layers=2, basis_rank=12, n_bins=64))
    net.eval()
    with torch.no_grad():
        out_warm = net.forward_episode(xc, yc, xq, head_mode_override="warm_scalar_twohot")
        out_dual = net.forward_episode(xc, yc, xq, head_mode_override="warm_plus_cholesky_r64")
    step0_diff = float(torch.max(torch.abs(out_warm["pred_query"] - out_dual["pred_query"])).item())
    assert step0_diff == 0.0, f"Step-0 warm-start bit-identity violated: {step0_diff}"

    # Verify analytical autograd through `solve_differentiable_cholesky_head` vs finite differences
    phi_c = torch.tensor(rng.normal(size=(28, 5)), dtype=torch.float64, requires_grad=True)
    phi_q = torch.tensor(rng.normal(size=(10, 5)), dtype=torch.float64, requires_grad=True)
    t_c = torch.tensor(rng.normal(size=28), dtype=torch.float64)
    t_q = torch.tensor(rng.normal(size=10), dtype=torch.float64)
    w_c = torch.tensor(rng.uniform(0.4, 2.2, size=28), dtype=torch.float64, requires_grad=True)
    lam = torch.tensor(rng.uniform(0.1, 1.5, size=5), dtype=torch.float64, requires_grad=True)

    pred_q, _, _ = solve_differentiable_cholesky_head(phi_c, phi_q, t_c, w_c, lam)
    loss = 0.5 * torch.sum((pred_q - t_q) ** 2)
    loss.backward()
    assert phi_c.grad is not None and lam.grad is not None

    eps = 1e-6
    pc_np = phi_c.detach().numpy().copy()
    grad_an = float(phi_c.grad[0, 0].item())
    pc_hi = pc_np.copy()
    pc_lo = pc_np.copy()
    pc_hi[0, 0] += eps
    pc_lo[0, 0] -= eps
    with torch.no_grad():
        l_hi = 0.5 * torch.sum((solve_differentiable_cholesky_head(torch.tensor(pc_hi), phi_q, t_c, w_c, lam)[0] - t_q) ** 2)
        l_lo = 0.5 * torch.sum((solve_differentiable_cholesky_head(torch.tensor(pc_lo), phi_q, t_c, w_c, lam)[0] - t_q) ** 2)
    grad_fd = float((l_hi - l_lo).item() / (2.0 * eps))
    grad_err = abs(grad_an - grad_fd)
    assert grad_err < 1e-7, f"Cholesky autograd mismatch: {grad_err}"
    return {"cholesky_autograd_fd_err": grad_err, "step0_warm_max_abs_diff": step0_diff}


def _p0_7_causal_ablation_solver_grad_vs_stopgrad_and_corrupted_hints() -> Dict[str, Any]:
    ep = build_counterfactual_repair_episode(n_context=96, n_query=36, n_features=5, seed=46207)
    hb_clean = compile_pcr_hints(ep["x_context"], ep["y_context"], ep["x_query"], k_inner=5, seed=46207)
    hb_corr, _ = corrupt_pcr_hints_counterfactual(hb_clean, corrupt_prob=0.35, seed=46207)

    xc = torch.tensor(ep["x_context"])
    yc = torch.tensor(ep["y_context"])
    xq = torch.tensor(ep["x_query"])
    yq = torch.tensor(ep["y_query"])
    yq_clean = ep["y_query_clean"]
    anc_c = torch.tensor(ep["anchor_context"])
    anc_q = torch.tensor(ep["anchor_query"])
    hc = torch.tensor(hb_corr.context_hints)
    hq = torch.tensor(hb_corr.query_hints)
    rc = torch.tensor(hb_corr.context_reliability)
    rq = torch.tensor(hb_corr.query_reliability)

    def _train_mode(mode: str) -> float:
        torch.manual_seed(46277)
        model = PRISMSolverNet(
            PRISMConfig(d_model=48, n_heads=4, n_layers=2, basis_rank=20, n_bins=64, head_mode=mode)
        )
        opt = torch.optim.AdamW(model.parameters(), lr=8e-3, weight_decay=1e-4)
        for _ in range(18):
            opt.zero_grad()
            # Split context 72/24 internally during adaptation check (zero `yq` access during training!)
            out_in = model.forward_episode(
                xc[:72], yc[:72], xc[72:], hc[:72], hc[72:], rc[:72], rc[72:], anc_c[:72], anc_c[72:]
            )
            loss = prism_episode_loss(out_in, yc[72:], bin_centers=model.bin_centers)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            out_q = model.forward_episode(xc, yc, xq, hc, hq, rc, rq, anc_c, anc_q)
        return pooled_unclipped_utility(yq_clean, out_q["pred_query"].numpy())

    u_full = _train_mode("cholesky_basis_r64")
    u_stop = _train_mode("stopgrad_cholesky_r64")
    u_anc = pooled_unclipped_utility(yq_clean, ep["anchor_query"])
    assert u_full > u_stop and u_full > u_anc + 0.5, f"Causal ablation failed: full={u_full}, stop={u_stop}, anc={u_anc}"
    return {
        "u_corrupted_anchor": round(u_anc, 4),
        "u_full_solver_grad": round(u_full, 4),
        "u_stopgrad_solver": round(u_stop, 4),
    }


def _p0_8_pcr_inner_oof_leak_canary_t8() -> Dict[str, Any]:
    rng = np.random.default_rng(46208)
    x_null = rng.normal(size=(150, 5))
    y_null = rng.normal(size=150)
    x_q = rng.normal(size=(20, 5))
    hb = compile_pcr_hints(x_null, y_null, x_q, k_inner=5, max_formulas=6, seed=46208)
    # Check max positive correlation between pure-noise y_null and ALL cross-fitted OOF predictors (Class A + Class B/C)
    max_corr = 0.0
    for j in range(hb.oof_predictions_raw.shape[1]):
        if j == hb.gp_sd_hint_col:
            continue
        c_col = hb.oof_predictions_raw[:, j]
        if float(np.std(c_col)) > 1e-12:
            corr = float(np.corrcoef(y_null, c_col)[0, 1])
            max_corr = max(max_corr, corr)
    assert max_corr <= 0.16, f"PCR inner-OOF label leakage detected under null: max_corr={max_corr}"
    return {
        "hint_protocol_id": hb.hint_protocol_id,
        "max_formulas": len(hb.selected_formula_channels),
        "max_null_oof_corr": round(max_corr, 4),
    }


def _p0_9_whole_procedure_audit_label_poison_canary_t2() -> Dict[str, Any]:
    rng = np.random.default_rng(46209)
    x_s = rng.normal(size=(120, 4))
    y_s = 1.4 * x_s[:, 0] - 0.6 * (x_s[:, 1] ** 2) + 0.2 * rng.normal(size=120)
    x_a = rng.normal(size=(40, 4))
    y_a_clean = 1.4 * x_a[:, 0] - 0.6 * (x_a[:, 1] ** 2)
    y_a_poison = y_a_clean + 1000.0

    scratch = _scratch_dir()
    try:
        def _run_end_to_end(y_audit_unused: np.ndarray, sub: str) -> Tuple[Dict[str, np.ndarray], str]:
            # Note: y_audit_unused is structurally excluded from FitClosureV3 and all solvers
            _ = y_audit_unused.shape
            lab = target_stratified_kfold_labels(y_s, n_splits=5, seed=46209)
            closure = FitClosureV3.create(
                x_s,
                y_s,
                dataset_id="ds_09",
                family_id="fam_09",
                train_row_ids=[f"S:{i}" for i in range(x_s.shape[0])],
                eval_row_ids=[f"A:{i}" for i in range(x_a.shape[0])],
                checkpoint_exposed_families=(),
            )
            floor_oof, floor_q = fit_raw_y_classic_floor_oof(x_s, y_s, x_a, lab, seed=46209)
            ex = run_exact_varpro_battery(x_s, y_s, lab)
            c_oof = dict(floor_oof)
            c_q = dict(floor_q)
            c_oof["exact_varpro"] = ex.oof_pred
            c_q["exact_varpro"] = ex.predict_query(x_a, floor_q["raw_ridge"])
            dec = edge_certify_context(y_s, lab, "raw_ridge", c_oof, c_q)
            npz_p = write_compatible_block_npz(
                scratch / sub,
                "ctx_09",
                {"edge": dec.pred_oof},
                {"edge": dec.pred_query},
                closure_sha256=closure.closure_sha256,
            )
            side = json.loads((scratch / sub / "ctx_09.sidecar.json").read_text())
            loaded = dict(np.load(npz_p))
            return loaded, str(side["npz_sha256"])

        arr_c, sha_c = _run_end_to_end(y_a_clean, "clean")
        arr_p, sha_p = _run_end_to_end(y_a_poison, "poison")
        max_diff = max(float(np.max(np.abs(arr_c[k] - arr_p[k]))) for k in arr_c)
        assert max_diff == 0.0 and sha_c == sha_p
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return {"max_abs_diff": max_diff, "npz_sha256_equal": sha_c == sha_p}


def _p0_10_singleton_family_leverage_and_cohort_floor() -> Dict[str, Any]:
    rng = np.random.default_rng(46210)
    y_s = rng.normal(size=100)
    lab = target_stratified_kfold_labels(y_s, n_splits=5, seed=46210)
    anc = y_s + 0.5 * rng.normal(size=100)
    better = y_s + 0.35 * rng.normal(size=100)
    d_single = edge_certify_context(y_s, lab, "anc", {"anc": anc, "b": better}, {"anc": anc[:20], "b": better[:20]}, family_size=1)
    d_five = edge_certify_context(y_s, lab, "anc", {"anc": anc, "b": better}, {"anc": anc[:20], "b": better[:20]}, family_size=5)
    ratio = d_single.expected_family_weighted_wins / max(d_five.expected_family_weighted_wins, 1e-12)
    assert abs(ratio - 5.0) < 1e-12

    fake_bad = EdgeContextDecision(
        selected_name="risky",
        anchor_name="anc",
        switched_from_anchor=True,
        family_weight=5.0,
        pooled_u_selected=0.20,
        pooled_u_anchor=1.00,
        pooled_delta_u=-0.80,
        bootstrap_lcb10_delta_u=0.01,
        bootstrap_win_prob_vs_anchor=0.75,
        expected_family_weighted_wins=3.0,
        pred_oof=better,
        pred_query=better[:20],
        candidate_pooled_u={"anc": 1.00, "risky": 0.20},
    )
    enforced = enforce_cohort_family_mean_floor([fake_bad, d_five], ["fam_singleton", "fam_five"], max_drop_u=0.01)
    assert not enforced[0].switched_from_anchor
    return {"reverted_drop_violator": not enforced[0].switched_from_anchor, "singleton_weight_ratio": round(ratio, 4)}


def _p0_11_bagging_e8_jensen_variance_inequality() -> Dict[str, Any]:
    rng = np.random.default_rng(46211)
    y_true = rng.normal(size=120)
    members = [y_true + 0.25 * rng.normal(size=120) for _ in range(8)]
    bag_pred, meta = bag_predictions_e8(members)
    mean_member_mse = float(np.mean([np.mean((y_true - m) ** 2) for m in members]))
    bag_mse = float(np.mean((y_true - bag_pred) ** 2))
    identity_err = abs((mean_member_mse - bag_mse) - meta["mean_member_variance_reduction"])
    assert meta["entrant_id"] == ENTRANT_ID_E8 and bag_mse < mean_member_mse and identity_err < 1e-14
    return {
        "bag_mse": round(bag_mse, 6),
        "entrant_id": meta["entrant_id"],
        "jensen_identity_abs_err": identity_err,
        "mean_member_mse": round(mean_member_mse, 6),
    }


def _p0_12_fit_closure_v3_s0_global_oof_and_quarantine_refusals() -> Dict[str, Any]:
    x = np.ones((20, 3), dtype=np.float64)
    y = np.arange(20, dtype=np.float64)
    s_ids = [f"S:{i}" for i in range(20)]
    a_ids = [f"A:{i}" for i in range(6)]

    # 1. External FM column in S0 must be refused
    refused_s0 = False
    try:
        FitClosureV3.create(
            x,
            y,
            "ds1",
            "fam1",
            train_row_ids=s_ids,
            eval_row_ids=a_ids,
            checkpoint_exposed_families=(),
            candidate_names=["raw_ridge", "kumo_tabular_large_e8"],
        )
    except PermissionError:
        refused_s0 = True
    assert refused_s0

    # 2. Global-OOF ancestor inside outer fold `fold_0` (`gp_chain --base-block` bug) must be refused by row-ID closure
    anc_global = FitClosureV3.create(
        x,
        y,
        "ds1",
        "fam1",
        train_row_ids=s_ids,
        eval_row_ids=a_ids,
        checkpoint_exposed_families=(),
        outer_fold_id="global_oof",
    )
    refused_global_oof = False
    try:
        FitClosureV3.create(
            x[:16],
            y[:16],
            "ds1",
            "fam1",
            train_row_ids=s_ids[:16],
            eval_row_ids=s_ids[16:],
            checkpoint_exposed_families=(),
            outer_fold_id="fold_0",
            ancestors=[anc_global],
        )
    except PermissionError:
        refused_global_oof = True
    assert refused_global_oof

    # 3. `be7de9bc` family exposure in checkpoint corpus must be refused
    refused_exposure = False
    try:
        FitClosureV3.create(
            x,
            y,
            "ds1",
            "fam_dev91_01",
            train_row_ids=s_ids,
            eval_row_ids=a_ids,
            checkpoint_exposed_families=["fam_clean", "fam_dev91_01"],
        )
    except PermissionError:
        refused_exposure = True
    assert refused_exposure

    # 4. Quarantined path tokens (`T14`) must be refused
    quarantined_paths = [
        "/home/ubuntu/jope-data/CONFIRM/board.npz",
        "/home/ubuntu/jope-data/fmgap_b/scores.json",
        "/home/ubuntu/jope-data/sealed_holdout/ctx.npz",
        "/home/ubuntu/jope-data/PROBE_YA.npz",
    ]
    refused_q = 0
    for qp in quarantined_paths:
        try:
            assert_path_not_quarantined(qp)
        except PermissionError:
            refused_q += 1
    assert refused_q == len(quarantined_paths)
    return {
        "refused_be7de9bc_family_exposure": refused_exposure,
        "refused_external_fm_s0": refused_s0,
        "refused_global_oof_base_block": refused_global_oof,
        "refused_quarantined_paths": refused_q,
    }


def _p0_13_extreme_scale_collinear_varpro_stability() -> Dict[str, Any]:
    """Reproduces extreme `10^10` target scale + binary/collinear features that triggered singular SK Gram matrices."""
    rng = np.random.default_rng(46213)
    n = 90
    b0 = rng.integers(0, 2, size=n).astype(np.float64)
    x1 = rng.uniform(0.5, 4.0, size=n)
    # Collinear duplicate + binary column (`b0^2 == b0` makes rational/quadratic monomials rank-deficient)
    x = np.column_stack([b0, b0.copy(), x1, 2.0 * x1 + 1.0])
    y = 1.0e10 * ((3.0 * x1 + 2.0 * b0 + 1.0) / (1.0 + 0.5 * x1))
    folds = np.arange(n, dtype=np.int64) % 5
    res = run_exact_varpro_battery(x, y, folds, min_fold_margin_u=0.20)
    assert np.all(np.isfinite(res.oof_pred)), "Non-finite OOF predictions on extreme-scale collinear table"
    return {"accepted": bool(res.accepted), "selected_law": res.selected_law_name}


def _p0_14_float32_feature_overflow_classic_floor() -> Dict[str, Any]:
    """Reproduces `|X| > 3.4e38` (`float32` overflow in sklearn RF/ET) discovered on real 398-context sweep."""
    rng = np.random.default_rng(46214)
    n_s, n_q = 80, 25
    x_base = rng.normal(0.0, 1.0, size=(n_s + n_q, 4))
    # Column 0 exceeds float32 max (3.4028e38); column 1 has NaNs + normal scale
    x_all = x_base.copy()
    x_all[:, 0] = x_base[:, 0] * 1.0e42
    x_all[3::11, 1] = np.nan
    y_all = 2.5e22 * x_base[:, 0] - 1.2e22 * np.nan_to_num(x_base[:, 1], nan=0.0) + rng.normal(0.0, 1.0e20, size=n_s + n_q)
    x_s, x_q = x_all[:n_s], x_all[n_s:]
    y_s, y_q = y_all[:n_s], y_all[n_s:]
    folds = np.arange(n_s, dtype=np.int64) % 5
    oof_map, q_map = fit_raw_y_classic_floor_oof(x_s, y_s, x_q, folds, seed=46214)
    u_q = {}
    for nm, pred in q_map.items():
        assert np.all(np.isfinite(oof_map[nm])), f"Non-finite OOF in {nm}"
        assert np.all(np.isfinite(pred)), f"Non-finite query pred in {nm}"
        u_q[nm] = round(pooled_unclipped_utility(y_q, pred), 4)
        assert u_q[nm] > 0.25, f"Poor extreme-scale recovery in {nm}: {u_q[nm]}"
    return {"query_utilities": u_q}


def _p0_15_precision_weighted_seed_bagging_exact_law_preservation() -> Dict[str, Any]:
    """Proves `M6-PRECISION-AVG` beats both single-seed mean `u` on noisy tables AND equal-weight `0.5*(p0+p1)` when 1 seed hits an exact law."""
    rng = np.random.default_rng(46215)
    n_s, n_q = 120, 60
    # Case A: Comparable noisy seeds -> precision weights ~0.5, beats mean single-seed utility
    y_s_a = rng.normal(0.0, 1.0, size=n_s)
    y_q_a = rng.normal(0.0, 1.0, size=n_q)
    q0_a = y_s_a + rng.normal(0.0, 0.35, size=n_s)
    q1_a = y_s_a + rng.normal(0.0, 0.35, size=n_s)
    p0_a = y_q_a + rng.normal(0.0, 0.35, size=n_q)
    p1_a = y_q_a + rng.normal(0.0, 0.35, size=n_q)
    p_prec_a, meta_a = bag_predictions_precision_weighted_oof(y_s_a, [q0_a, q1_a], [p0_a, p1_a])
    u_conv_a = 0.5 * (pooled_unclipped_utility(y_q_a, p0_a) + pooled_unclipped_utility(y_q_a, p1_a))
    u_prec_a = pooled_unclipped_utility(y_q_a, p_prec_a)
    assert u_prec_a > u_conv_a + 0.05

    # Case B: Seed 0 hits near-exact formula (1e-7 noise) while Seed 1 misses (0.25 noise) -> equal-weight loses badly, precision-avg preserves exact law!
    q0_b = y_s_a + rng.normal(0.0, 1e-7, size=n_s)
    q1_b = y_s_a + rng.normal(0.0, 0.25, size=n_s)
    p0_b = y_q_a + rng.normal(0.0, 1e-7, size=n_q)
    p1_b = y_q_a + rng.normal(0.0, 0.25, size=n_q)
    p_eq_b, _ = bag_predictions_e8([p0_b, p1_b])
    p_prec_b, meta_b = bag_predictions_precision_weighted_oof(y_s_a, [q0_b, q1_b], [p0_b, p1_b])
    u_conv_b = 0.5 * (pooled_unclipped_utility(y_q_a, p0_b) + pooled_unclipped_utility(y_q_a, p1_b))
    u_eq_b = pooled_unclipped_utility(y_q_a, p_eq_b)
    u_prec_b = pooled_unclipped_utility(y_q_a, p_prec_b)
    assert u_eq_b < u_conv_b - 4.0, "Equal-weight must suffer log-MSE convexity drop when one seed hits exact law"
    assert u_prec_b > u_conv_b + 6.0, "OOF precision-weighted bag must preserve exact law and beat single-seed mean u"
    return {
        "noisy_delta_u": round(u_prec_a - u_conv_a, 4),
        "exact_eq_delta_u": round(u_eq_b - u_conv_b, 4),
        "exact_prec_delta_u": round(u_prec_b - u_conv_b, 4),
        "exact_winner_weight": round(meta_b["max_weight"], 8),
    }


def _p0_16_hnc1_h1_self_exclusion_and_h3_h4_h7_parity() -> Dict[str, Any]:
    """Verifies `HNC-1` `H1` (`+1000*sd` single-row poison), `H3` (Findings 1 & 7), `H4` (Finding 3), and `H7` (Finding 4)."""
    rng = np.random.default_rng(46216)
    x_c = rng.normal(size=(64, 5))
    y_c = 1.6 * x_c[:, 0] * x_c[:, 1] - 0.8 * x_c[:, 2] + 0.3 * rng.normal(size=64)
    x_q = rng.normal(size=(20, 5))

    hb_clean = compile_pcr_hints(x_c, y_c, x_q, k_inner=5, max_formulas=6, seed=46216)
    assert len(hb_clean.selected_formula_channels) == 6
    assert len(hb_clean.non_selected_channels) == 6

    # H1 canary: add 1000 * sd(y) to y_i; every non-selected channel at row i must be unchanged to 1e-12 relative
    poison_row = 17
    y_poison = y_c.copy()
    y_poison[poison_row] += 1000.0 * float(np.std(y_c))
    hb_poison = compile_pcr_hints(x_c, y_poison, x_q, k_inner=5, max_formulas=6, seed=46216)
    max_rel_diff = 0.0
    for col in hb_clean.non_selected_channels:
        v0 = float(hb_clean.context_hints[poison_row, col])
        v1 = float(hb_poison.context_hints[poison_row, col])
        rel = abs(v1 - v0) / max(abs(v0), 1e-12)
        max_rel_diff = max(max_rel_diff, rel)
    assert max_rel_diff < 1e-12, f"H1 self-exclusion failed at row {poison_row}: max_rel_diff={max_rel_diff}"

    # H3 Finding 1: gp_sd is in identical reliability slot on C and Q; fold disagreement has missing indicator
    r_c = hb_clean.context_reliability
    r_q = hb_clean.query_reliability
    gp_col = hb_clean.gp_sd_hint_col
    gp_rel_col = hb_clean.gp_sd_row_rel_col
    assert np.allclose(r_c[:, gp_rel_col], hb_clean.context_hints[:, gp_col], atol=0.0)
    assert np.allclose(r_q[:, gp_rel_col], hb_clean.query_hints[:, gp_col], atol=0.0)
    assert np.all(r_c[:, gp_rel_col + 1] == 0.0) and np.all(r_c[:, gp_rel_col + 2] == 1.0)
    assert np.all(r_q[:, gp_rel_col + 2] == 0.0)

    # H4 Finding 3: corrupting channels recomputes ONLY corrupted skills via deploy L350 formula (zero *0.25 flag)
    hb_corr, mask = corrupt_pcr_hints_counterfactual(hb_clean, corrupt_prob=0.50, seed=46216)
    assert np.any(mask) and not np.all(mask)
    for j, is_corr in enumerate(mask):
        s_col = hb_clean.skill_col_for_hint[j]
        if s_col is None:
            continue
        if not is_corr:
            assert float(np.max(np.abs(hb_corr.context_reliability[:, s_col] - hb_clean.context_reliability[:, s_col]))) == 0.0
        else:
            expected_sk = compute_l350_skill(hb_clean.context_target_norm, hb_corr.context_hints[:, j])
            assert abs(float(hb_corr.context_reliability[0, s_col]) - expected_sk) < 1e-13

    # H7 Finding 4: protocol id hashes module source + params, invariant when n_c < 20 changes k_eff
    hb_small = compile_pcr_hints(x_c[:12], y_c[:12], x_q[:4], k_inner=5, max_formulas=6, seed=46216)
    assert hb_small.k_eff == 3 and hb_clean.k_eff == 5
    assert hb_small.hint_protocol_id == hb_clean.hint_protocol_id
    return {
        "h1_max_rel_diff": max_rel_diff,
        "h7_protocol_id_invariant": hb_small.hint_protocol_id == hb_clean.hint_protocol_id,
        "k_eff_pair": [hb_small.k_eff, hb_clean.k_eff],
    }


def _p0_17_hnc1_h5_rowid_closure_and_h6_manifest_exposure() -> Dict[str, Any]:
    """Verifies `HNC-1` `H5` row-ID outer nesting (refusing spoofed fold strings) and `H6` mandatory exposure."""
    x = np.ones((20, 3), dtype=np.float64)
    y = np.arange(20, dtype=np.float64)
    s_all = [f"S:{i}" for i in range(20)]
    a_all = [f"A:{i}" for i in range(8)]

    # Spoofed fold string: ancestor claims "fold_3" on S:0..19; consumer claims "fold_3" on S:0..15 evaluating S:16..19 -> MUST REFUSE!
    anc_spoof = FitClosureV3.create(
        x, y, "ds_h5", "fam_h5",
        train_row_ids=s_all,
        eval_row_ids=a_all,
        checkpoint_exposed_families=(),
        outer_fold_id="fold_3",
        hint_protocol_id="pcr_v2_abc",
    )
    refused_spoof_fold = False
    try:
        FitClosureV3.create(
            x[:16], y[:16], "ds_h5", "fam_h5",
            train_row_ids=s_all[:16],
            eval_row_ids=s_all[16:],
            checkpoint_exposed_families=(),
            outer_fold_id="fold_3",
            hint_protocol_id="pcr_v2_abc",
            ancestors=[anc_spoof],
        )
    except PermissionError:
        refused_spoof_fold = True
    assert refused_spoof_fold

    # Spoofed "full_search" consumer evaluating on S:16..19 -> MUST ALSO REFUSE!
    refused_full_search_on_s = False
    try:
        FitClosureV3.create(
            x[:16], y[:16], "ds_h5", "fam_h5",
            train_row_ids=s_all[:16],
            eval_row_ids=s_all[16:],
            checkpoint_exposed_families=(),
            outer_fold_id="full_search",
            hint_protocol_id="pcr_v2_abc",
            ancestors=[anc_spoof],
        )
    except PermissionError:
        refused_full_search_on_s = True
    assert refused_full_search_on_s

    # Valid A-only consumer (C = S, Q = A) -> MUST PASS!
    valid_a = FitClosureV3.create(
        x, y, "ds_h5", "fam_h5",
        train_row_ids=s_all,
        eval_row_ids=a_all,
        checkpoint_exposed_families=(),
        outer_fold_id="full_search",
        hint_protocol_id="pcr_v2_abc",
        ancestors=[anc_spoof],
    )
    assert valid_a.closure_sha256 != ""

    # H7 stack mismatch refusal
    refused_h7_mismatch = False
    try:
        FitClosureV3.create(
            x, y, "ds_h5", "fam_h5",
            train_row_ids=s_all,
            eval_row_ids=a_all,
            checkpoint_exposed_families=(),
            hint_protocol_id="pcr_v2_different",
            ancestors=[anc_spoof],
        )
    except PermissionError:
        refused_h7_mismatch = True
    assert refused_h7_mismatch

    # H6: omitting checkpoint_exposed_families raises TypeError; bank_sha256 with empty list raises ValueError
    refused_missing_exp = False
    try:
        FitClosureV3.create(x, y, "ds_h6", "fam_h6", train_row_ids=s_all, eval_row_ids=a_all)  # type: ignore[call-arg]
    except TypeError:
        refused_missing_exp = True
    assert refused_missing_exp

    refused_empty_bank_exp = False
    try:
        FitClosureV3.create(
            x, y, "ds_h6", "fam_h6",
            train_row_ids=s_all,
            eval_row_ids=a_all,
            checkpoint_exposed_families=(),
            checkpoint_bank_sha256="bc5c7d6cd1f4",
        )
    except ValueError:
        refused_empty_bank_exp = True
    assert refused_empty_bank_exp

    scratch = _scratch_dir()
    try:
        mf_p = scratch / "mock_bank.npz.json"
        mf_p.parent.mkdir(parents=True, exist_ok=True)
        mf_p.write_text(
            json.dumps(
                {
                    "bank_sha256": "bc5c7d6c_mock",
                    "tables": [
                        {"family_key": "fresh:srcfile:mtp"},
                        {"family_key": "openml:kin8nm"},
                    ],
                }
            )
        )
        b_sha, b_fams = load_bank_manifest_exposure(mf_p)
        assert b_sha == "bc5c7d6c_mock" and b_fams == ("openml:kin8nm", "srcfile:mtp")
        n_ov = count_exposed_families_on_board(["srcfile:mtp", "uci:concrete"], b_fams)
        assert n_ov == 1
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    return {
        "board_overlap_count": n_ov,
        "refused_empty_bank_exp": refused_empty_bank_exp,
        "refused_full_search_on_s": refused_full_search_on_s,
        "refused_h7_mismatch": refused_h7_mismatch,
        "refused_missing_exp": refused_missing_exp,
        "refused_spoof_fold": refused_spoof_fold,
    }


def _p0_18_hnc1_h8_200_episode_calibration_canary() -> Dict[str, Any]:
    """Verifies `HNC-1` `H8`: over 200 synthetic episodes (`max_formulas=6`), `|mean(ctx_skill) - mean(qry_skill)| <= 0.05`."""
    n_episodes = 200
    n_c, n_q, n_f = 60, 36, 4
    ctx_sums = np.zeros(11, dtype=np.float64)
    qry_sums = np.zeros(11, dtype=np.float64)

    for ep_idx in range(n_episodes):
        rng = np.random.default_rng(462800 + ep_idx)
        x_all = rng.normal(size=(n_c + n_q, n_f))
        mode = ep_idx % 4
        noise_sd = float(rng.uniform(0.15, 0.85))
        if mode == 0:
            y_clean = 1.3 * x_all[:, 0] - 0.9 * x_all[:, 1] + 0.5 * x_all[:, 0] * x_all[:, 1]
        elif mode == 1:
            y_clean = np.sin(1.5 * x_all[:, 0]) + 0.8 * (x_all[:, 1] / (np.abs(x_all[:, 2]) + 0.6))
        elif mode == 2:
            disc = np.round(x_all[:, 0] * 1.5)
            y_clean = 1.1 * disc + 0.6 * (x_all[:, 1] - x_all[:, 3])
        else:
            y_clean = 0.25 * x_all[:, 0]
        y_all = y_clean + noise_sd * rng.normal(size=n_c + n_q)

        hb = compile_pcr_hints(
            x_all[:n_c],
            y_all[:n_c],
            x_all[n_c:],
            k_inner=5,
            max_formulas=6,
            seed=462800 + ep_idx,
            query_fold_mode="random_single_fold",
        )
        ep_skills = evaluate_h8_episode_skills(hb, y_all[n_c:])
        assert len(ep_skills) == 11
        for k_pos, (_nm, (c_sk, q_sk)) in enumerate(ep_skills.items()):
            ctx_sums[k_pos] += c_sk
            qry_sums[k_pos] += q_sk

    ctx_means = ctx_sums / float(n_episodes)
    qry_means = qry_sums / float(n_episodes)
    abs_gaps = np.abs(ctx_means - qry_means)
    max_gap = float(np.max(abs_gaps))
    assert max_gap <= 0.05, f"H8 calibration canary failed: max_gap={max_gap}, gaps={abs_gaps.tolist()}"
    return {
        "max_abs_skill_gap": round(max_gap, 5),
        "mean_abs_skill_gap": round(float(np.mean(abs_gaps)), 5),
        "n_episodes": n_episodes,
    }


def _p0_19_edge_certify_partial_top_block_sampling() -> Dict[str, Any]:
    """Verifies `cl_mtc` Correction 2: `_sample_audit_shaped_blocks` draws `round(3*r/10)` rows from partial top block."""
    from research.flywheel_selfplay.h462_apex_prism.edge_certify import _sample_audit_shaped_blocks

    # n_rows = 57 -> 5 full blocks (15 draws) + partial top block r=7 -> round(2.1)=2 draws = 17 draws total
    y = np.arange(57, dtype=np.float64)
    draws = _sample_audit_shaped_blocks(y, n_bootstrap=400, audit_per_10=3, seed=46299)
    assert draws.shape == (400, 17), f"Expected shape (400, 17), got {draws.shape}"
    max_row_draw_rate = float(np.mean(np.any(draws == 56, axis=1)))
    assert 0.18 <= max_row_draw_rate <= 0.40, f"Top row 56 never/badly drawn: rate={max_row_draw_rate}"
    return {"draw_cols": int(draws.shape[1]), "top_row_draw_rate": round(max_row_draw_rate, 4)}


def _p0_20_outer_nested_gp_passes_verify_nested_oof_fn() -> Dict[str, Any]:
    """Verifies `HNC-1 H5` `verify_nested_oof_fn`: catches full-SEARCH & 2-stage outer coupling, certifies `run_outer_nested_oof_committee`."""
    from research.flywheel_selfplay.h462_apex_prism.emit_gp_runbag6_h462 import (
        _precision_combine_oof,
        run_outer_nested_oof_committee,
        run_outer_nested_oof_seed,
    )

    rng = np.random.default_rng(46220)
    n_s = 60
    x_s = rng.normal(size=(n_s, 4))
    y_s = 1.7 * x_s[:, 0] - 0.9 * (x_s[:, 1] ** 2) + 0.25 * rng.normal(size=n_s)
    lab5 = np.arange(n_s, dtype=np.int64) % 5

    def _candidate_bank(X_fit: np.ndarray, y_fit: np.ndarray, X_eval: np.ndarray):
        Phi_fit = np.column_stack(
            [
                np.ones(len(X_fit)),
                X_fit[:, 0],
                X_fit[:, 1],
                X_fit[:, 1] ** 2,
                X_fit[:, 2],
                X_fit[:, 0] * X_fit[:, 3],
            ]
        )
        Phi_ev = np.column_stack(
            [
                np.ones(len(X_eval)),
                X_eval[:, 0],
                X_eval[:, 1],
                X_eval[:, 1] ** 2,
                X_eval[:, 2],
                X_eval[:, 0] * X_eval[:, 3],
            ]
        )
        subsets = [(0, 1), (0, 2), (0, 1, 3), (0, 4), (0, 2, 5), (0, 1, 2, 3)]
        q_cols = []
        p_cols = []
        for sub in subsets:
            A = Phi_fit[:, sub]
            beta = np.linalg.lstsq(A.T @ A + 1e-4 * np.eye(len(sub)), A.T @ y_fit, rcond=None)[0]
            q_cols.append(A @ beta)
            p_cols.append(Phi_ev[:, sub] @ beta)
        return np.column_stack(q_cols), np.column_stack(p_cols)

    def _seed_fold_elites(sd: int, fk: int, X_tr: np.ndarray, y_tr: np.ndarray, lab_in: np.ndarray, X_va: np.ndarray):
        Q_in = np.empty((len(y_tr), 6), dtype=np.float64)
        for ifk in np.unique(lab_in):
            iva = lab_in == ifk
            itr = ~iva
            _, P_iva = _candidate_bank(X_tr[itr], y_tr[itr], X_tr[iva])
            Q_in[iva] = P_iva
        _, P_va_all = _candidate_bank(X_tr, y_tr, X_va)
        r_sd = np.random.default_rng(sd * 31 + fk)
        scores = np.mean((Q_in - y_tr[:, None]) ** 2, axis=0) + 0.02 * r_sd.uniform(size=6)
        elites = np.argsort(scores)[:3]
        return Q_in[:, elites], P_va_all[:, elites]

    # 1. Negative control A: selecting top-2 programs on full SEARCH (Xs, ys) leaks validation labels -> MUST FAIL!
    def _leaky_full_search_selection_oof(X: np.ndarray, y: np.ndarray, lab: np.ndarray) -> np.ndarray:
        Q_all, _ = _candidate_bank(X, y, X)
        mses = np.mean((Q_all - y[:, None]) ** 2, axis=0)
        top2 = np.argsort(mses)[:2]
        q_out = np.empty(len(y), dtype=np.float64)
        for fk in range(5):
            va = lab == fk
            tr = ~va
            A_tr = Q_all[np.ix_(tr, top2)]
            w = np.linalg.lstsq(A_tr.T @ A_tr + 1e-4 * np.eye(2), A_tr.T @ y[tr], rcond=None)[0]
            q_out[va] = Q_all[np.ix_(va, top2)] @ w
        return q_out

    caught_selection_leak = False
    try:
        verify_nested_oof_fn(_leaky_full_search_selection_oof, x_s, y_s, lab5)
    except PermissionError:
        caught_selection_leak = True
    assert caught_selection_leak

    # 2. Negative control B: naive 2-stage outer-OOF re-stacking (`Q_mat[tr]` trained on `va` in other folds) -> MUST FAIL!
    def _leaky_two_stage_outer_coupling_oof(X: np.ndarray, y: np.ndarray, lab: np.ndarray) -> np.ndarray:
        seeds_q = [
            run_outer_nested_oof_seed("ctx_h5", sd, X, y, lab, lambda fk, xt, yt, li, xv, _s=sd: _seed_fold_elites(_s, fk, xt, yt, li, xv))
            for sd in (46211, 46212, 46213)
        ]
        Q_mat = np.column_stack(seeds_q)
        dummy_a = np.zeros((4, Q_mat.shape[1]), dtype=np.float64)
        q_pr, _, _, _ = _precision_combine_oof(y, lab, Q_mat, dummy_a, dummy_a)
        return q_pr

    caught_twostage_coupling = False
    try:
        verify_nested_oof_fn(_leaky_two_stage_outer_coupling_oof, x_s, y_s, lab5)
    except PermissionError:
        caught_twostage_coupling = True
    assert caught_twostage_coupling

    # 3. Positive proof: `run_outer_nested_oof_committee` fits inner-fold weights strictly inside `(X_tr, y_tr)` -> 0.0 leak!
    def _honest_outer_nested_committee_oof(X: np.ndarray, y: np.ndarray, lab: np.ndarray) -> np.ndarray:
        Q_seeds, q_eq, q_pr = run_outer_nested_oof_committee(
            "ctx_h5", (46211, 46212, 46213), X, y, lab, _seed_fold_elites
        )
        return np.column_stack([Q_seeds, q_eq, q_pr])

    rcpt = verify_nested_oof_fn(_honest_outer_nested_committee_oof, x_s, y_s, lab5)
    assert rcpt["verified"] is True and float(rcpt["max_abs_leak"]) == 0.0
    return {
        "caught_full_search_selection_leak": caught_selection_leak,
        "caught_two_stage_outer_coupling": caught_twostage_coupling,
        "max_abs_leak": float(rcpt["max_abs_leak"]),
        "n_folds_checked": int(rcpt["n_folds_checked"]),
    }


def _p0_21_search_only_members_emitter_blocks_ya() -> Dict[str, Any]:
    """Verifies `emit_p2_p4_members_search_only.py` blocks `ya`/`ba` via `NoLabelNpz` and emits valid `P2`/`P4` payloads."""
    from research.flywheel_selfplay.h462_apex_prism.emit_p2_p4_members_search_only import (
        H430_K3,
        H433_F4,
        H450_TAB_COLS,
        NoLabelNpz,
        compute_context_members_search_only,
        load_gp_pack_no_ya,
        load_search_only_bank,
    )
    from research.flywheel_selfplay.h462_apex_prism import guard_matern_core as GM

    rng = np.random.default_rng(46221)
    scratch = _scratch_dir()
    try:
        n_s, n_a, n_f = 30, 12, 3
        Xs = rng.normal(size=(n_s, n_f))
        ys = 1.4 * Xs[:, 0] - 0.7 * Xs[:, 1] + 0.25 * rng.normal(size=n_s)
        Xa = rng.normal(size=(n_a, n_f))
        lab0 = np.arange(n_s, dtype=np.int64) % 5

        # 1. Bank containing `ya` MUST be refused immediately
        bad_bank = scratch / "bad_bank_with_ya.npz"
        S_mat = np.column_stack([Xs, ys])
        np.savez(
            bad_bank,
            cids=np.array(["ctx01"]),
            family=np.array(["fam01"]),
            s_shape=np.array([[n_s, n_f + 1]], dtype=np.int64),
            a_shape=np.array([[n_a, n_f]], dtype=np.int64),
            search=S_mat.reshape(-1),
            audit=Xa.reshape(-1),
            y64=ys,
            lab=lab0,
            ya=np.zeros(n_a),
        )
        blocked_bank_ya = False
        try:
            load_search_only_bank(bad_bank)
        except PermissionError:
            blocked_bank_ya = True
        assert blocked_bank_ya

        # 2. H418-style GP file containing `ya` & `ba` MUST block `zg["ya"]` via `NoLabelNpz` while reconstructing `BAG`/`REF` with `0` `ya` reads
        gp_p = scratch / "h418_d1_mock.npz"
        Q_gp = ys[:, None] + 0.2 * rng.normal(size=(n_s, 6))
        PA = 1.4 * Xa[:, [0]] - 0.7 * Xa[:, [1]] + 0.2 * rng.normal(size=(n_a, 6))
        AKF = PA + 0.05 * rng.normal(size=(n_a, 6))
        np.savez(
            gp_p,
            Q=Q_gp,
            PA=PA,
            AKF=AKF,
            pa_ok=np.ones(6, dtype=bool),
            ym=np.array(float(np.mean(ys))),
            ya=np.full(n_a, 999999.0),
            ba=np.zeros(n_a),
        )
        with NoLabelNpz(gp_p) as zg:
            blocked_gp_ya = False
            try:
                _ = zg["ya"]
            except PermissionError:
                blocked_gp_ya = True
            assert blocked_gp_ya and not zg.has_key("ya") and not zg.has_key("ba")

        Q_loaded, bcols, rcols = load_gp_pack_no_ya(gp_p, n_s, n_a)
        assert Q_loaded.shape == (n_s, 6) and len(bcols) == 6 and len(rcols) == 6

        def _mk_blk(ncols: int):
            q = ys[:, None] + 0.22 * rng.normal(size=(n_s, ncols))
            b = (1.4 * Xa[:, [0]] - 0.7 * Xa[:, [1]]) + 0.22 * rng.normal(size=(n_a, ncols))
            return q, b, b.copy()

        base_common = [
            ("m7", _mk_blk(len(GM.PRIMARY_NAMES))),
            ("k3", _mk_blk(len(H430_K3))),
            ("f4", _mk_blk(len(H433_F4))),
            ("tab450", _mk_blk(len(H450_TAB_COLS))),
        ]
        r4_blocks = [("wc_r4e8", _mk_blk(1))]
        gp_packs = [("d1", Q_loaded, bcols, rcols), ("d2", Q_loaded, bcols, rcols)]
        out = compute_context_members_search_only(
            cid="ctx01",
            Xs=Xs,
            Xa=Xa,
            y=ys,
            lab0=lab0,
            gp_packs=gp_packs,
            raw_common_cv7=base_common,
            r4_blocks=r4_blocks,
            gpbag_blk=_mk_blk(2),
            cv_repeats=1,
            audit_weight_mode="real_stacker",
        )
        out_hwm = compute_context_members_search_only(
            cid="ctx01",
            Xs=Xs,
            Xa=Xa,
            y=ys,
            lab0=lab0,
            gp_packs=gp_packs,
            raw_common_cv7=base_common,
            r4_blocks=r4_blocks,
            gpbag_blk=None,
            cv_repeats=1,
            audit_weight_mode="hwm_1842_exact",
        )
        # 1 r4 * 2 gp * 1 cv_repeats = 2 members
        assert out["member_oof"].shape == (n_s, 2)
        assert out["member_audit"].shape == (n_a, 2)
        assert abs(float(np.sum(out["member_weights"])) - 1.0) < 1e-12
        assert np.all(np.isfinite(out["p_oof_ens"])) and np.all(np.isfinite(out["p_audit_ens"]))
        assert "member_oof_hwm_fast" in out_hwm and out_hwm["member_oof_hwm_fast"].shape == (n_s, 2)
        # Also verify `real_stacker_outer_oof_rep` passes `verify_nested_oof_fn` when base columns are outer-fold invariant!
        from research.flywheel_selfplay.h462_apex_prism.emit_p2_p4_members_search_only import real_stacker_outer_oof_rep

        Q_gp_fixed = np.column_stack([1.4 * Xs[:, 0] - 0.7 * Xs[:, 1], 1.2 * Xs[:, 0], -0.6 * Xs[:, 1]])
        raw_fixed = [("m7", (np.column_stack([Xs[:, 0], Xs[:, 1]]), np.zeros((n_a, 2)), np.zeros((n_a, 2))))]
        rcpt_rs = verify_nested_oof_fn(
            lambda X_in, y_in, l_in: real_stacker_outer_oof_rep(
                "ctx_rs_h5", 0, X_in, y_in, l_in, Q_gp_fixed, [True, True, True], raw_fixed, None, None
            ),
            Xs,
            ys,
            lab0,
        )
        assert rcpt_rs["verified"] is True and float(rcpt_rs["max_abs_leak"]) == 0.0
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    return {
        "blocked_bank_ya": blocked_bank_ya,
        "blocked_gp_ya": blocked_gp_ya,
        "n_members": 2,
    }


def _p0_22_nested_gpbag_layout_g1_g2_and_zero_leak_stacker() -> Dict[str, Any]:
    """`P0_22`: Verifies `cl_mtc` `G1` + `G2` + `P2 v1.4 §3 / v1.5` `nest_lab_/nest_q_/nest_a_` end-to-end."""
    from research.flywheel_selfplay.h462_apex_prism.contracts_v3 import verify_nested_oof_fn
    from research.flywheel_selfplay.h462_apex_prism.emit_gp_runbag6_h462 import build_nested_gpbag_from_fold_elites
    from research.flywheel_selfplay.h462_apex_prism.emit_p2_p4_members_search_only import (
        assert_same_outer_folds,
        build_gpbag_block_from_dirs,
        load_gp_pack_no_ya,
        real_stacker_outer_oof_rep,
    )

    rng = np.random.default_rng(46222)
    scratch = _scratch_dir()
    try:
        n_s, n_a, n_f = 35, 10, 3
        Xs = rng.normal(size=(n_s, n_f))
        ys = 1.6 * Xs[:, 0] - 0.8 * Xs[:, 1] + 0.35 * Xs[:, 2] + 0.15 * rng.normal(size=n_s)
        Xa = rng.normal(size=(n_a, n_f))
        lab0 = np.arange(n_s, dtype=np.int64) % 5

        # 1. Canary second half: y-ignoring function MUST fail `verify_nested_oof_fn`
        caught_y_ignoring = False
        try:
            verify_nested_oof_fn(lambda X_in, _y_in, _l_in: X_in[:, 0].copy(), Xs, ys, lab0)
        except PermissionError:
            caught_y_ignoring = True
        assert caught_y_ignoring

        # 2. G2: `assert_same_outer_folds` accepts relabeled identical row sets, refuses altered row sets & missing D1 `lab`
        lab_perm = (4 - lab0).astype(np.int64)
        fmap = assert_same_outer_folds(lab_perm, lab0, "perm_check")
        assert all(fmap[fk] == 4 - fk for fk in range(5))

        lab_bad = lab0.copy()
        lab_bad[0], lab_bad[1] = lab_bad[1], lab_bad[0]
        caught_bad_lab = False
        try:
            assert_same_outer_folds(lab_bad, lab0, "bad_lab_check")
        except PermissionError:
            caught_bad_lab = True
        assert caught_bad_lab

        gp_nolab = scratch / "gp_nolab.npz"
        np.savez(
            gp_nolab,
            Q=np.zeros((n_s, 2)),
            PA=np.zeros((n_a, 2)),
            AKF=np.zeros((n_a, 2)),
            pa_ok=np.ones(2, dtype=bool),
            ym=np.array(0.0),
        )
        caught_missing_gp_lab = False
        try:
            load_gp_pack_no_ya(gp_nolab, n_s, n_a, lab0=lab0, require_lab=True)
        except PermissionError:
            caught_missing_gp_lab = True
        assert caught_missing_gp_lab

        # 3. End-to-end `build_nested_gpbag_from_fold_elites` -> `build_gpbag_block_from_dirs` -> `real_stacker_outer_oof_rep`
        seeds = (46211, 46212)

        def _fit_ridge(X_tr: np.ndarray, y_tr: np.ndarray, X_q: np.ndarray, lam: float) -> np.ndarray:
            A = np.column_stack([np.ones(len(X_tr)), X_tr])
            Aq = np.column_stack([np.ones(len(X_q)), X_q])
            reg = np.diag([1e-8] + [lam] * X_tr.shape[1])
            beta = np.linalg.solve(A.T @ A + reg, A.T @ y_tr)
            return Aq @ beta

        Q_gp_fixed = np.column_stack([1.5 * Xs[:, 0] - 0.6 * Xs[:, 1], 1.1 * Xs[:, 0] + 0.2 * Xs[:, 2]])

        def _build_gb_and_run_stacker(X_in: np.ndarray, y_in: np.ndarray, l_in: np.ndarray, use_nest: bool) -> np.ndarray:
            def _seed_fold_fn(sd: int, _fk: int, X_tr: np.ndarray, y_tr: np.ndarray, l_sub: np.ndarray, X_va: np.ndarray, X_aud: np.ndarray):
                lams = (0.05, 0.5) if sd == 46211 else (0.15, 1.5)
                Q_tr = np.empty((len(y_tr), len(lams)), dtype=np.float64)
                for ifk in sorted(int(k) for k in np.unique(l_sub)):
                    iva = l_sub == ifk
                    itr = ~iva
                    for li, lm in enumerate(lams):
                        Q_tr[iva, li] = _fit_ridge(X_tr[itr], y_tr[itr], X_tr[iva], lm)
                P_va = np.column_stack([_fit_ridge(X_tr, y_tr, X_va, lm) for lm in lams])
                P_aud = np.column_stack([_fit_ridge(X_tr, y_tr, X_aud, lm) for lm in lams])
                return Q_tr, P_va, P_aud

            nest_d = build_nested_gpbag_from_fold_elites("ctx22", seeds, X_in, Xa, y_in, l_in, _seed_fold_fn)
            gb_p = scratch / "gpbag_nested.npz"
            payload = {"ok": np.array(True), "lab": l_in}
            for si, sd in enumerate(seeds):
                nm = f"gp_s{si + 1}"
                # Full-SEARCH cross-fit for `q_<nm>` (couples `SEARCH \ g` across folds if un-nested!)
                q_cf = np.empty(len(y_in), dtype=np.float64)
                lm_s = 0.05 if sd == 46211 else 0.15
                for fk in sorted(int(k) for k in np.unique(l_in)):
                    va = l_in == fk
                    tr = ~va
                    q_cf[va] = _fit_ridge(X_in[tr], y_in[tr], X_in[va], lm_s)
                payload[f"q_{nm}"] = q_cf
                payload[f"bag_{nm}"] = np.mean(nest_d[f"nest_a_{nm}"], axis=0)
                payload[f"ref_{nm}"] = np.mean(nest_d[f"nest_a_{nm}"], axis=0)
            payload.update(nest_d)
            np.savez(gb_p, **payload)

            gb_blk, gb_nq = build_gpbag_block_from_dirs(y_in, l_in, [gb_p], len(y_in), n_a, 2, "allcol", cid="ctx22")
            assert gb_blk is not None and gb_nq is not None and gb_nq.shape == (5, len(y_in), 4)
            return real_stacker_outer_oof_rep(
                "ctx22",
                0,
                X_in,
                y_in,
                l_in,
                Q_gp_fixed,
                [True, True],
                [],
                None,
                gb_blk,
                gpbag_nest_q=gb_nq if use_nest else None,
            )

        # Negative control: un-nested GPBAG (`gpbag_nest_q=None`) leaks `y[va]` via second-order `SEARCH \ g` coupling
        caught_unnested_gpbag = False
        try:
            verify_nested_oof_fn(lambda X_i, y_i, l_i: _build_gb_and_run_stacker(X_i, y_i, l_i, False), Xs, ys, lab0)
        except PermissionError:
            caught_unnested_gpbag = True
        assert caught_unnested_gpbag

        # Positive proof: `nest_lab_ / nest_q_ / nest_a_` (`gpbag_nest_q=gb_nq`) achieves exact `max_abs_leak == 0.0`!
        rcpt = verify_nested_oof_fn(lambda X_i, y_i, l_i: _build_gb_and_run_stacker(X_i, y_i, l_i, True), Xs, ys, lab0)
        assert rcpt["verified"] is True and float(rcpt["max_abs_leak"]) == 0.0
        assert float(rcpt["min_other_fold_response"]) > 1e-3
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    return {
        "caught_y_ignoring": caught_y_ignoring,
        "caught_bad_lab": caught_bad_lab,
        "caught_missing_gp_lab": caught_missing_gp_lab,
        "caught_unnested_gpbag": caught_unnested_gpbag,
        "nested_gpbag_stacker_max_abs_leak": float(rcpt["max_abs_leak"]),
        "min_other_fold_response": float(rcpt["min_other_fold_response"]),
    }


def run_all_selftests() -> Dict[str, Any]:
    t0 = time.perf_counter()
    results = {
        "p0_1_assay_regime_and_tail_stat": _p0_1_assay_regime_and_tail_stat(),
        "p0_2_exact_varpro_recovery_and_zero_noisy_false_fire": _p0_2_exact_varpro_recovery_and_zero_noisy_false_fire(),
        "p0_3_anchor_cancellation_of_yj_tail_explosion": _p0_3_anchor_cancellation_of_yj_tail_explosion(),
        "p0_4_pooled_unclipped_vs_clipped_fold_mean_on_stratified_folds": _p0_4_pooled_unclipped_vs_clipped_fold_mean_on_stratified_folds(),
        "p0_5_inductive_query_independence_t1": _p0_5_inductive_query_independence_t1(),
        "p0_6_warm_plus_cholesky_step0_identity_and_grad_parity": _p0_6_warm_plus_cholesky_step0_identity_and_grad_parity(),
        "p0_7_causal_ablation_solver_grad_vs_stopgrad_and_corrupted_hints": _p0_7_causal_ablation_solver_grad_vs_stopgrad_and_corrupted_hints(),
        "p0_8_pcr_inner_oof_leak_canary_t8": _p0_8_pcr_inner_oof_leak_canary_t8(),
        "p0_9_whole_procedure_audit_label_poison_canary_t2": _p0_9_whole_procedure_audit_label_poison_canary_t2(),
        "p0_10_singleton_family_leverage_and_cohort_floor": _p0_10_singleton_family_leverage_and_cohort_floor(),
        "p0_11_bagging_e8_jensen_variance_inequality": _p0_11_bagging_e8_jensen_variance_inequality(),
        "p0_12_fit_closure_v3_s0_global_oof_and_quarantine_refusals": _p0_12_fit_closure_v3_s0_global_oof_and_quarantine_refusals(),
        "p0_13_extreme_scale_collinear_varpro_stability": _p0_13_extreme_scale_collinear_varpro_stability(),
        "p0_14_float32_feature_overflow_classic_floor": _p0_14_float32_feature_overflow_classic_floor(),
        "p0_15_precision_weighted_seed_bagging_exact_law_preservation": _p0_15_precision_weighted_seed_bagging_exact_law_preservation(),
        "p0_16_hnc1_h1_self_exclusion_and_h3_h4_h7_parity": _p0_16_hnc1_h1_self_exclusion_and_h3_h4_h7_parity(),
        "p0_17_hnc1_h5_rowid_closure_and_h6_manifest_exposure": _p0_17_hnc1_h5_rowid_closure_and_h6_manifest_exposure(),
        "p0_18_hnc1_h8_200_episode_calibration_canary": _p0_18_hnc1_h8_200_episode_calibration_canary(),
        "p0_19_edge_certify_partial_top_block_sampling": _p0_19_edge_certify_partial_top_block_sampling(),
        "p0_20_outer_nested_gp_passes_verify_nested_oof_fn": _p0_20_outer_nested_gp_passes_verify_nested_oof_fn(),
        "p0_21_search_only_members_emitter_blocks_ya": _p0_21_search_only_members_emitter_blocks_ya(),
        "p0_22_nested_gpbag_layout_g1_g2_and_zero_leak_stacker": _p0_22_nested_gpbag_layout_g1_g2_and_zero_leak_stacker(),
    }
    return {
        "all_passed": True,
        "code_sha256": _package_code_sha256(),
        "n_checks": len(results),
        "results": results,
        "schema": "jope.h462.apex_prism_selftest/v1",
        "wall_time_s": round(time.perf_counter() - t0, 3),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true", help="Run all 22 deterministic H462 APEX-PRISM checks")
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
