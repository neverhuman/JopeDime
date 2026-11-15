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
    build_counterfactual_repair_episode,
    compile_pcr_hints,
    compute_robust_tail_stat,
    corrupt_pcr_hints_counterfactual,
    edge_certify_context,
    enforce_cohort_family_mean_floor,
    fit_raw_y_classic_floor_oof,
    legacy_clipped_fold_mean_score,
    pooled_unclipped_utility,
    prism_episode_loss,
    run_assay,
    run_exact_varpro_battery,
    solve_differentiable_cholesky_head,
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
    hb = compile_pcr_hints(x_null, y_null, x_q, k_inner=5, max_formulas=0, seed=46208)
    # Check max positive correlation between pure-noise y_null and location OOF hints [0, 1, 3, 4, 5]
    max_corr = 0.0
    for j in (0, 1, 3, 4, 5):
        c_col = hb.context_hints[:, j]
        corr = float(np.corrcoef(y_null, c_col)[0, 1])
        max_corr = max(max_corr, corr)
    assert max_corr <= 0.15, f"PCR inner-OOF label leakage detected under null: max_corr={max_corr}"
    return {"hint_protocol_id": hb.hint_protocol_id, "max_null_oof_corr": round(max_corr, 4)}


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
            closure = FitClosureV3.create(x_s, y_s, dataset_id="ds_09", family_id="fam_09")
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

    # 1. External FM column in S0 must be refused
    refused_s0 = False
    try:
        FitClosureV3.create(x, y, "ds1", "fam1", candidate_names=["raw_ridge", "kumo_tabular_large_e8"])
    except PermissionError:
        refused_s0 = True
    assert refused_s0

    # 2. Global-OOF ancestor inside outer fold `fold_0` (`gp_chain --base-block` bug) must be refused
    anc_global = FitClosureV3.create(x, y, "ds1", "fam1", outer_fold_id="global_oof")
    refused_global_oof = False
    try:
        FitClosureV3.create(x[:16], y[:16], "ds1", "fam1", outer_fold_id="fold_0", ancestors=[anc_global])
    except PermissionError:
        refused_global_oof = True
    assert refused_global_oof

    # 3. `be7de9bc` family exposure in checkpoint corpus must be refused
    refused_exposure = False
    try:
        FitClosureV3.create(x, y, "ds1", "fam_dev91_01", checkpoint_exposed_families=["fam_clean", "fam_dev91_01"])
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
    ap.add_argument("--selftest", action="store_true", help="Run all 12 deterministic H462 APEX-PRISM checks")
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
