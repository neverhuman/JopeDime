"""Unit tests for North Star Phase C/D modules (NS3, NS4, NS5, NS7)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from jope.north_star import (
    DatasetEvalRecord,
    DopeFreezeViolationError,
    assert_optimizer_excludes_dope,
    batched_roc_auc_gpu,
    compile_and_eval_rust_fitness,
    compute_family_grouped_bradley_terry_elo,
    emit_rust_only_flywheel_source,
    evaluate_unified_regression_and_binary_batched_gpu,
    get_trainable_param_counts_by_submodule,
    lock_dope_weights_and_eval_mode,
    resolve_authoritative_family_and_task,
    snapshot_dope_weights,
    verify_dope_frozen_unchanged,
)


def test_ns3_dope_freeze_guard_and_counterexamples() -> None:
    torch.manual_seed(42)

    class ToyFusedNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dope_model = nn.Module()
            self.dope_model.sub = nn.Linear(4, 4, bias=False)
            self.dope_model.bn = nn.BatchNorm1d(4)
            self.head = nn.Linear(4, 2, bias=False)
            self.root_scale = nn.Parameter(torch.ones(3))

    model = ToyFusedNet()
    lock_dope_weights_and_eval_mode(model)
    snap = snapshot_dope_weights(model)

    counts = get_trainable_param_counts_by_submodule(model)
    assert counts["dope_model"]["trainable_params"] == 0
    assert counts["dope_model"]["training_mode"] is False
    assert counts["head"]["trainable_params"] == 8
    assert counts["__root_direct__"]["trainable_params"] == 3
    assert counts["__global_total__"]["trainable_params"] == 11

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-2,
    )
    assert_optimizer_excludes_dope(opt, model)

    # Parent train(True) and nested train(True) must keep DOPE in eval mode
    model.train(True)
    model.dope_model.sub.train(True)
    assert not model.dope_model.training
    assert not model.dope_model.sub.training
    verify_dope_frozen_unchanged(model, snap)

    # Direct nested attribute mutation must be caught by both snapshot check and counts
    model.dope_model.sub.training = True
    with pytest.raises(DopeFreezeViolationError):
        get_trainable_param_counts_by_submodule(model)
    with pytest.raises(DopeFreezeViolationError):
        verify_dope_frozen_unchanged(model, snap)
    model.dope_model.sub.training = False

    # Buffer drift must be caught
    with torch.no_grad():
        model.dope_model.bn.running_mean.add_(1.0)
    with pytest.raises(DopeFreezeViolationError):
        verify_dope_frozen_unchanged(model, snap)


def test_ns4_binary_clf_metrics_and_valid_mask_nan_counterexample() -> None:
    device = torch.device("cpu")

    # Equal-score tie handling -> 0.5
    p_eq = torch.tensor([[0.5, 0.5, 0.5, 0.5]], device=device)
    y_eq = torch.tensor([[0.0, 1.0, 0.0, 1.0]], device=device)
    m_eq = torch.ones((1, 4), dtype=torch.bool, device=device)
    auc_eq = batched_roc_auc_gpu(p_eq, y_eq, m_eq)
    assert abs(float(auc_eq[0].item()) - 0.5) < 1e-6

    # Partial tie handling -> 0.875
    p_pt = torch.tensor([[0.1, 0.5, 0.5, 0.9]], device=device)
    y_pt = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=device)
    auc_pt = batched_roc_auc_gpu(p_pt, y_pt, m_eq)
    assert abs(float(auc_pt[0].item()) - 0.875) < 1e-6

    # Masked NaN/Inf padding invariance
    raw_tr_fin = torch.tensor([[-2.0, 2.0, -1.5, 1.8, 0.0, 0.0]], device=device)
    raw_tr_nan = torch.tensor([[-2.0, 2.0, -1.5, 1.8, float("nan"), float("inf")]], device=device)
    y_tr_fin = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0, 0.0]], device=device)
    y_tr_nan = torch.tensor([[0.0, 1.0, 0.0, 1.0, float("nan"), float("-inf")]], device=device)
    m_tr = torch.tensor([[True, True, True, True, False, False]], device=device)

    raw_te_fin = torch.tensor([[-2.0, 2.0, -1.5, 1.8, 0.0, 0.0]], device=device)
    raw_te_nan = torch.tensor([[-2.0, 2.0, -1.5, 1.8, float("nan"), float("-inf")]], device=device)
    y_te_fin = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0, 0.0]], device=device)
    y_te_nan = torch.tensor([[0.0, 1.0, 0.0, 1.0, float("nan"), float("inf")]], device=device)
    m_te = torch.tensor([[True, True, True, True, False, False]], device=device)

    ev_fin = evaluate_unified_regression_and_binary_batched_gpu(
        raw_tr_fin, y_tr_fin, m_tr, raw_te_fin, y_te_fin, m_te
    )
    ev_nan = evaluate_unified_regression_and_binary_batched_gpu(
        raw_tr_nan, y_tr_nan, m_tr, raw_te_nan, y_te_nan, m_te
    )
    assert bool(ev_fin["valid_eval_mask"][0].item())
    assert bool(ev_nan["valid_eval_mask"][0].item())
    assert abs(float(ev_fin["roc_auc"][0].item()) - float(ev_nan["roc_auc"][0].item())) < 1e-6
    assert abs(float(ev_fin["unified_r2_score"][0].item()) - float(ev_nan["unified_r2_score"][0].item())) < 1e-6

    # Codex 13:41:58Z counterexample: 2 valid holdout rows with BOTH predictions NaN
    # Must NEVER return 1.0; must mark valid_eval_mask=False and return NaN (or raise on reject_nonfinite_eval=True)
    raw_tr_ce = torch.tensor([[1.0, 2.0, 3.0, 0.0]], device=device)
    y_tr_ce = torch.tensor([[1.0, 2.0, 3.0, 0.0]], device=device)
    m_tr_ce = torch.tensor([[True, True, True, False]], device=device)
    raw_te_ce = torch.tensor([[float("nan"), float("nan"), 0.0, 0.0]], device=device)
    y_te_ce = torch.tensor([[1.5, 2.5, 0.0, 0.0]], device=device)
    m_te_ce = torch.tensor([[True, True, False, False]], device=device)

    ev_ce = evaluate_unified_regression_and_binary_batched_gpu(
        raw_tr_ce, y_tr_ce, m_tr_ce, raw_te_ce, y_te_ce, m_te_ce
    )
    assert not bool(ev_ce["valid_eval_mask"][0].item())
    assert math.isnan(float(ev_ce["unified_r2_score"][0].item()))
    assert math.isnan(float(ev_ce["regression_r2"][0].item()))

    with pytest.raises(ValueError, match="Invalid evaluation batch"):
        evaluate_unified_regression_and_binary_batched_gpu(
            raw_tr_ce,
            y_tr_ce,
            m_tr_ce,
            raw_te_ce,
            y_te_ce,
            m_te_ce,
            reject_nonfinite_eval=True,
        )


def test_ns5_rust_fitness_target() -> None:
    rng = np.random.default_rng(123)
    X = rng.normal(size=(64, 4))
    w_true = np.array([1.5, -0.8, 0.5, 0.3])
    y_reg = X @ w_true
    mu = np.zeros(4)
    sd = np.ones(4)

    rs_src = emit_rust_only_flywheel_source(
        dataset_id="ns5_test",
        selected_dsl="(pipe X (standardize) (ridge_gcv))",
        weights=w_true,
        intercept=0.0,
        mu_tr=mu,
        sd_tr=sd,
        is_binary=False,
    )
    assert "pub unsafe extern \"C\" fn predict_batch_f64" in rs_src
    res = compile_and_eval_rust_fitness(rs_src, X, y_reg, is_binary=False)
    assert abs(float(res["fitness_r2"]) - 1.0) < 1e-6


def test_ns7_elo_tracker_and_null_family_rejection() -> None:
    # 1. Duplicate-family invariance
    base_records = [
        DatasetEvalRecord("d1", "fam_1", "regression", {"M_A": 0.90, "M_B": 0.70}),
        DatasetEvalRecord("d2", "fam_2", "binary_classification", {"M_A": 0.60, "M_B": 0.80}),
    ]
    dup_records = base_records + [
        DatasetEvalRecord(f"d1_clone_{k}", "fam_1", "regression", {"M_A": 0.90, "M_B": 0.70})
        for k in range(5)
    ]
    res_base = compute_family_grouped_bradley_terry_elo(
        base_records, anchor_contender="M_B"
    )["ratings"]
    res_dup = compute_family_grouped_bradley_terry_elo(
        dup_records, anchor_contender="M_B"
    )["ratings"]
    assert abs(float(res_base["M_A"]) - 1500.0) < 1e-4
    assert abs(float(res_dup["M_A"]) - 1500.0) < 1e-4

    # 2. Incomplete panel rejected by default, balanced under opt-in
    inc_records = [
        DatasetEvalRecord("d1", "fam_win_2", "regression", {"anchor": 0.70, "challenger": 0.80}),
        DatasetEvalRecord("d2", "fam_win_2", "regression", {"anchor": 0.72}),
        DatasetEvalRecord("d3", "fam_lose_1", "regression", {"anchor": 0.85, "challenger": 0.65}),
    ]
    with pytest.raises(ValueError, match="Incomplete panel"):
        compute_family_grouped_bradley_terry_elo(inc_records, anchor_contender="anchor")

    res_inc = compute_family_grouped_bradley_terry_elo(
        inc_records,
        anchor_contender="anchor",
        allow_incomplete_panel=True,
    )["ratings"]
    assert abs(float(res_inc["challenger"]) - 1500.0) < 1e-4

    # 3. Null / "None" / "unknown" family_key rejection (Codex 13:41:58Z counterexample)
    for bad_fam in [None, "", "None", "null", "unknown", "nan", "N/A"]:
        with pytest.raises(ValueError, match="unresolved/null family_id"):
            compute_family_grouped_bradley_terry_elo(
                [
                    DatasetEvalRecord("d1", bad_fam, "regression", {"M_A": 0.9, "M_B": 0.7}),  # type: ignore[arg-type]
                ],
                anchor_contender="M_B",
            )

    with pytest.raises(ValueError, match="unresolved/null family_key"):
        resolve_authoritative_family_and_task(
            "dbc501fc7f6a40b5",
            {
                "dbc501fc7f6a40b5": {
                    "family_id": None,
                    "task_type": "regression",
                    "origin": "real",
                    "has_meta_test": False,
                }
            },
        )
