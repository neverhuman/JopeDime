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
    assert_research_optimizer_policy,
    batched_roc_auc_gpu,
    compile_and_eval_rust_fitness,
    compute_family_grouped_bradley_terry_elo,
    configure_research_dope,
    emit_rust_only_flywheel_source,
    evaluate_unified_regression_and_binary_batched_gpu,
    get_trainable_param_counts_by_submodule,
    lock_dope_weights_and_eval_mode,
    reject_research_dope_unfreeze,
    resolve_authoritative_family_and_task,
    snapshot_dope_weights,
    verify_dope_frozen_unchanged,
)
from jope.north_star.dope_freeze_guard import DOPE_NORMALIZER_BUFFERS


class _ToyFusedNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dope_model = nn.Module()
        self.dope_model.sub = nn.Linear(4, 4, bias=False)
        self.dope_model.bn = nn.BatchNorm1d(4)
        self.head = nn.Linear(4, 2, bias=False)
        self.root_scale = nn.Parameter(torch.ones(3))
        for name in DOPE_NORMALIZER_BUFFERS:
            self.register_buffer(name, torch.zeros(1))


def test_ns3_dope_freeze_guard_and_counterexamples() -> None:
    torch.manual_seed(42)

    model = _ToyFusedNet()
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


def test_ns3_missing_normalizer_buffer_is_rejected() -> None:
    model = _ToyFusedNet()
    lock_dope_weights_and_eval_mode(model)
    delattr(model, DOPE_NORMALIZER_BUFFERS[0])

    with pytest.raises(DopeFreezeViolationError, match="required normalizer buffer.*missing"):
        snapshot_dope_weights(model)


def test_ns3_extra_trainable_dope_parameter_is_rejected() -> None:
    model = _ToyFusedNet()
    lock_dope_weights_and_eval_mode(model)
    model.dope_model.extra = nn.Parameter(torch.ones(1))

    with pytest.raises(DopeFreezeViolationError, match="dope_model.extra.*requires_grad=True"):
        snapshot_dope_weights(model)


def test_ns3_reshaped_normalizer_buffer_is_detected() -> None:
    model = _ToyFusedNet()
    lock_dope_weights_and_eval_mode(model)
    before = snapshot_dope_weights(model)
    model.step2_lm_mean_1x256 = torch.zeros(1, 1)

    with pytest.raises(DopeFreezeViolationError, match="hash mismatch"):
        verify_dope_frozen_unchanged(model, before)


def test_ns3_mutated_normalizer_buffer_is_detected() -> None:
    model = _ToyFusedNet()
    lock_dope_weights_and_eval_mode(model)
    before = snapshot_dope_weights(model)
    with torch.no_grad():
        model.step2_lm_mean_1x256.add_(1.0)

    with pytest.raises(DopeFreezeViolationError, match="hash mismatch"):
        verify_dope_frozen_unchanged(model, before)


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
    assert (
        abs(float(ev_fin["unified_r2_score"][0].item()) - float(ev_nan["unified_r2_score"][0].item())) < 1e-6
    )

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
    assert 'pub unsafe extern "C" fn predict_batch_f64' in rs_src
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
    res_base = compute_family_grouped_bradley_terry_elo(base_records, anchor_contender="M_B")["ratings"]
    res_dup = compute_family_grouped_bradley_terry_elo(dup_records, anchor_contender="M_B")["ratings"]
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


@pytest.mark.parametrize("flag", ["1", "true", "", "2"])
def test_ns3_serving_and_flywheel_refuse_research_flag(monkeypatch, flag) -> None:
    monkeypatch.setenv("JOPE_RESEARCH_DOPE_UNFREEZE", flag)
    monkeypatch.setenv("JOPE_RESEARCH_RUN_ID", "H206-test")
    model = _ToyFusedNet()
    with pytest.raises(DopeFreezeViolationError, match="refuses"):
        lock_dope_weights_and_eval_mode(model)
    with pytest.raises(DopeFreezeViolationError, match="refuses"):
        reject_research_dope_unfreeze("fused closed-loop flywheel")


def test_ns3_research_default_stays_byte_identical(monkeypatch) -> None:
    monkeypatch.delenv("JOPE_RESEARCH_DOPE_UNFREEZE", raising=False)
    model = _ToyFusedNet()
    metadata = {}
    configure_research_dope(model, launch_metadata=metadata)
    before = snapshot_dope_weights(model)
    opt = torch.optim.SGD(model.head.parameters(), lr=0.1)
    assert_research_optimizer_policy(opt, model, research_run_id="H206-default")
    model.train()
    opt.zero_grad()
    model.head(model.dope_model.sub(torch.ones(2, 4))).sum().backward()
    opt.step()
    verify_dope_frozen_unchanged(model, before)
    assert metadata["dope_policy"]["research_dope_unfreeze"] is False


@pytest.mark.parametrize("use_env", [False, True])
def test_ns3_explicit_research_harness_updates_dope_only(monkeypatch, use_env) -> None:
    monkeypatch.delenv("JOPE_RESEARCH_DOPE_UNFREEZE", raising=False)
    model = _ToyFusedNet()
    model.gamenet_anchor = nn.Linear(4, 4)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    metadata = {}
    if use_env:
        monkeypatch.setenv("JOPE_RESEARCH_DOPE_UNFREEZE", "1")
        monkeypatch.setenv("JOPE_RESEARCH_RUN_ID", "H206-unit")
        configure_research_dope(model, launch_metadata=metadata)
    else:
        configure_research_dope(model, launch_metadata=metadata, unfreeze=True, research_run_id="H206-unit")
    model.train()
    model.dope_model.sub.train()
    assert all(not m.training for m in model.dope_model.modules())
    assert all(p.requires_grad for p in model.dope_model.parameters())
    opt = torch.optim.SGD(model.dope_model.parameters(), lr=0.1)
    assert assert_research_optimizer_policy(opt, model, research_run_id="H206-unit")
    with pytest.raises(DopeFreezeViolationError):
        assert_optimizer_excludes_dope(opt, model)
    with pytest.raises(DopeFreezeViolationError, match="run id"):
        assert_research_optimizer_policy(opt, model, research_run_id="another-run")
    opt.zero_grad()
    model.dope_model.bn(model.dope_model.sub(torch.ones(2, 4))).sum().backward()
    opt.step()
    assert not torch.equal(original["dope_model.sub.weight"], model.dope_model.sub.weight)
    for k, v in model.state_dict().items():
        if k.startswith("gamenet_anchor.") or k not in dict(model.named_parameters()):
            assert torch.equal(v, original[k]), k
    assert metadata["dope_policy"] == {
        "research_dope_unfreeze": True,
        "research_run_id": "H206-unit",
        "entrypoint": "research",
        "dope_eval_mode": True,
        "serving_eligible": False,
    }
    monkeypatch.delenv("JOPE_RESEARCH_DOPE_UNFREEZE", raising=False)
    lock_dope_weights_and_eval_mode(model)
    model.train()
    assert all(not p.requires_grad and p.grad is None for p in model.dope_model.parameters())
    with pytest.raises(DopeFreezeViolationError):
        assert_research_optimizer_policy(opt, model, research_run_id="H206-unit")


@pytest.mark.parametrize("run_id", [None, "", "bad/run", "bad run", "x" * 129])
def test_ns3_research_unfreeze_requires_run_identity(monkeypatch, run_id) -> None:
    monkeypatch.delenv("JOPE_RESEARCH_RUN_ID", raising=False)
    with pytest.raises(DopeFreezeViolationError, match="run id"):
        configure_research_dope(_ToyFusedNet(), launch_metadata={}, unfreeze=True, research_run_id=run_id)


def test_ns3_research_flag_cannot_hide_environment_or_unfreeze_anchor(monkeypatch) -> None:
    monkeypatch.setenv("JOPE_RESEARCH_DOPE_UNFREEZE", "1")
    with pytest.raises(DopeFreezeViolationError, match="conflicts"):
        configure_research_dope(_ToyFusedNet(), launch_metadata={}, unfreeze=False)
    model = _ToyFusedNet()
    model.gamenet_anchor = model.dope_model
    with pytest.raises(DopeFreezeViolationError, match="share parameters"):
        configure_research_dope(model, launch_metadata={}, research_run_id="H206-unit")
