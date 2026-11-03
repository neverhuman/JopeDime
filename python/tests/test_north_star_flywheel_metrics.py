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
from test_north_star_flywheel_guards import _ToyFusedNet


def test_ns14_rust_fitness_tie_averaged_roc_auc_and_validation() -> None:
    # 1. Constant zero-weight contender must return tie-averaged roc_auc == 0.5 and brier_r2 == 0.0
    # regardless of whether holdout labels are ordered [0,0,1,1] or [1,1,0,0].
    X_tie = np.array([[-1.0, 0.5], [0.2, -0.3], [0.8, 1.1], [-0.4, 0.9]], dtype=np.float64)
    rs_zero = emit_rust_only_flywheel_source(
        dataset_id="ns14_tie_test",
        selected_dsl="(pipe X (standardize) (ridge_gcv))",
        weights=np.zeros(2, dtype=np.float64),
        intercept=0.0,
        mu_tr=np.zeros(2, dtype=np.float64),
        sd_tr=np.ones(2, dtype=np.float64),
        is_binary=True,
        platt_a=1.0,
        platt_b=0.0,
        pred_mu=0.0,
        pred_sd=1.0,
    )
    res_0011 = compile_and_eval_rust_fitness(
        rs_zero, X_tie, np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64), is_binary=True
    )
    res_1100 = compile_and_eval_rust_fitness(
        rs_zero, X_tie, np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float64), is_binary=True
    )
    assert abs(float(res_0011["roc_auc"]) - 0.5) < 1e-6
    assert abs(float(res_1100["roc_auc"]) - 0.5) < 1e-6
    assert abs(float(res_0011["brier_r2"]) - 0.0) < 1e-6
    assert abs(float(res_1100["brier_r2"]) - 0.0) < 1e-6

    # 2. Partial tie permutation invariance: X = [[-1], [0], [0], [1]] with y = [0, 0, 1, 1] -> AUC = 0.875
    rs_unit = emit_rust_only_flywheel_source(
        dataset_id="ns14_partial_tie",
        selected_dsl="(pipe X (standardize) (ridge_gcv))",
        weights=np.array([1.0], dtype=np.float64),
        intercept=0.0,
        mu_tr=np.array([0.0], dtype=np.float64),
        sd_tr=np.array([1.0], dtype=np.float64),
        is_binary=True,
        platt_a=1.0,
        platt_b=0.0,
        pred_mu=0.0,
        pred_sd=1.0,
    )
    X_pt_fwd = np.array([[-1.0], [0.0], [0.0], [1.0]], dtype=np.float64)
    y_pt_fwd = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    X_pt_rev = np.array([[1.0], [0.0], [0.0], [-1.0]], dtype=np.float64)
    y_pt_rev = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float64)
    auc_fwd = float(compile_and_eval_rust_fitness(rs_unit, X_pt_fwd, y_pt_fwd, is_binary=True)["roc_auc"])
    auc_rev = float(compile_and_eval_rust_fitness(rs_unit, X_pt_rev, y_pt_rev, is_binary=True)["roc_auc"])
    assert abs(auc_fwd - 0.875) < 1e-6
    assert abs(auc_rev - 0.875) < 1e-6

    # 3. Emission parameter validation rejects non-finite scalars/arrays and shape mismatches
    with pytest.raises(ValueError, match="weights must contain only finite"):
        emit_rust_only_flywheel_source(
            dataset_id="bad_w",
            selected_dsl="(pipe X)",
            weights=np.array([1.0, math.nan]),
            intercept=0.0,
            mu_tr=np.zeros(2),
            sd_tr=np.ones(2),
        )
    with pytest.raises(ValueError, match="mu_tr and sd_tr must have identical shape"):
        emit_rust_only_flywheel_source(
            dataset_id="bad_shape",
            selected_dsl="(pipe X)",
            weights=np.ones(2),
            intercept=0.0,
            mu_tr=np.zeros(2),
            sd_tr=np.ones(3),
        )
    with pytest.raises(ValueError, match="weights length.*cannot exceed feature count"):
        emit_rust_only_flywheel_source(
            dataset_id="bad_w_len",
            selected_dsl="(pipe X)",
            weights=np.ones(3),
            intercept=0.0,
            mu_tr=np.zeros(2),
            sd_tr=np.ones(2),
        )
    with pytest.raises(ValueError, match="pred_sd must be non-zero"):
        emit_rust_only_flywheel_source(
            dataset_id="bad_sd",
            selected_dsl="(pipe X)",
            weights=np.ones(2),
            intercept=0.0,
            mu_tr=np.zeros(2),
            sd_tr=np.ones(2),
            pred_sd=0.0,
        )

    # 4. Holdout target validation rejects non-finite labels, constant regression targets, and invalid binary targets
    with pytest.raises(ValueError, match="y_val must contain only finite"):
        compile_and_eval_rust_fitness(rs_unit, X_pt_fwd, np.array([0.0, math.nan, 1.0, 1.0]), is_binary=False)
    with pytest.raises(ValueError, match="Degenerate constant regression holdout target"):
        compile_and_eval_rust_fitness(rs_unit, X_pt_fwd, np.ones(4, dtype=np.float64), is_binary=False)
    with pytest.raises(ValueError, match="Degenerate single-class binary holdout target"):
        compile_and_eval_rust_fitness(rs_unit, X_pt_fwd, np.zeros(4, dtype=np.float64), is_binary=True)
    with pytest.raises(ValueError, match="Binary holdout target must contain exactly two distinct classes"):
        compile_and_eval_rust_fitness(
            rs_unit, X_pt_fwd, np.array([0.0, 0.5, 1.0, 1.0], dtype=np.float64), is_binary=True
        )


@pytest.mark.parametrize("label", [0.0, 1.0])
@pytest.mark.parametrize("coverage", ["all", "one", "masked_opposite"])
def test_ns4_single_class_auc_is_unavailable(label: float, coverage: str) -> None:
    scores = torch.tensor([[0.1, 0.7, 0.2, 0.9]])
    labels = torch.full_like(scores, label)
    mask = torch.ones_like(scores, dtype=torch.bool)
    if coverage == "one":
        mask[:, 1:] = False
    elif coverage == "masked_opposite":
        labels[:, 2:] = 1.0 - label
        mask[:, 2:] = False
        scores[:, 2:] = torch.tensor([math.nan, math.inf])
    result = batched_roc_auc_gpu(scores, labels, mask)
    assert result.shape == (1,)
    assert result.dtype == scores.dtype
    assert result.device == scores.device
    assert torch.isnan(result).all()


def test_ns4_auc_batch_distinguishes_missing_classes_from_measured_ties() -> None:
    scores = torch.tensor(
        [
            [0.1, 0.7, 0.2, 0.9],
            [0.1, 0.7, 0.2, 0.9],
            [0.5, 0.5, 0.5, 0.5],
            [0.1, 0.5, 0.5, 0.9],
            [math.nan, 0.7, 0.2, 0.9],
            [0.1, 0.7, 0.2, 0.9],
        ]
    )
    labels = torch.tensor(
        [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 1, 1]]
    ).float()
    mask = torch.ones_like(labels, dtype=torch.bool)
    mask[-1] = False
    result = batched_roc_auc_gpu(scores, labels, mask)
    expected = torch.tensor([math.nan, math.nan, 0.5, 0.875, math.nan, math.nan])
    torch.testing.assert_close(result, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("label", [0.0, 1.0])
def test_ns4_unified_single_class_auc_preserves_other_metrics(label: float) -> None:
    train_y = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    result = evaluate_unified_regression_and_binary_batched_gpu(
        torch.tensor([[-2.0, 2.0, -1.0, 1.0]]),
        train_y,
        torch.ones_like(train_y, dtype=torch.bool),
        torch.tensor([[-1.0, 1.0, 2.0, 3.0]]),
        torch.full_like(train_y, label),
        torch.ones_like(train_y, dtype=torch.bool),
        reject_nonfinite_eval=True,
    )
    assert result["is_binary"].tolist() == [True]
    assert result["valid_eval_mask"].tolist() == [True]
    assert torch.isnan(result["roc_auc"]).all()
    for name in [
        "unified_r2_score",
        "regression_r2",
        "brier_r2",
        "log_loss",
        "balanced_accuracy",
        "platt_a",
        "platt_b",
    ]:
        assert torch.isfinite(result[name]).all(), name


def test_ns4_valid_auc_matches_pairwise_definition_with_masks_and_ties() -> None:
    rng = torch.Generator().manual_seed(1504)
    scores = torch.randint(0, 5, (8, 16), generator=rng).float() / 4.0
    labels = torch.randint(0, 2, (8, 16), generator=rng).float()
    labels[:, 0], labels[:, 1] = 0.0, 1.0
    mask = torch.rand(8, 16, generator=rng) > 0.3
    mask[:, :2] = True
    scores = torch.where(mask, scores, torch.full_like(scores, math.nan))
    expected = []
    for score_row, label_row, mask_row in zip(scores, labels, mask):
        positive = score_row[mask_row & (label_row == 1)]
        negative = score_row[mask_row & (label_row == 0)]
        pairs = (positive[:, None] > negative[None, :]).float()
        pairs += 0.5 * (positive[:, None] == negative[None, :]).float()
        expected.append(pairs.mean())
    result = batched_roc_auc_gpu(scores, labels, mask)
    torch.testing.assert_close(result, torch.stack(expected), rtol=0, atol=1e-7)
    permutation = torch.randperm(scores.shape[1], generator=rng)
    reordered = batched_roc_auc_gpu(scores[:, permutation], labels[:, permutation], mask[:, permutation])
    torch.testing.assert_close(reordered, result, rtol=0, atol=0)


@pytest.fixture
def ns16_gate_inputs():
    from jope.north_star import check_phase_d_elo_gate

    ratings = {"ROUND_0": 1490.0, "ROUND_1": 1500.0, "ROUND_2": 1510.0, "H": 1500.0}
    return check_phase_d_elo_gate, {
        "round_names": ["ROUND_0", "ROUND_1", "ROUND_2"],
        "overall_ratings": dict(ratings),
        "regression_ratings": dict(ratings),
        "binary_ratings": dict(ratings),
        "h_baseline_name": "H",
    }


@pytest.mark.parametrize("axis", ["overall", "regression", "binary"])
@pytest.mark.parametrize("contender", ["ROUND_0", "ROUND_1", "ROUND_2", "H"])
@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf, True, None])
def test_ns16_gate_rejects_invalid_requested_ratings(ns16_gate_inputs, axis, contender, invalid):
    gate, inputs = ns16_gate_inputs
    inputs[f"{axis}_ratings"][contender] = invalid
    with pytest.raises(ValueError):
        gate(**inputs)


@pytest.mark.parametrize("numeric_strings", [False, True])
def test_ns16_gate_preserves_finite_policy(ns16_gate_inputs, numeric_strings):
    gate, inputs = ns16_gate_inputs
    # Overall is compared at the final round; only regression and binary must rise.
    inputs["overall_ratings"]["ROUND_0"] = 1600.0
    for axis in ["overall", "regression", "binary"]:
        inputs[f"{axis}_ratings"]["UNREQUESTED"] = math.nan
        if numeric_strings:
            inputs[f"{axis}_ratings"] = {key: str(value) for key, value in inputs[f"{axis}_ratings"].items()}
    result = gate(**inputs)
    assert result["gate_pass"] is True
    assert result["num_rounds"] == 3
    assert result["overall_elo_trajectory"] == {"ROUND_0": 1600.0, "ROUND_1": 1500.0, "ROUND_2": 1510.0}
    assert all(
        result[key] is True
        for key in [
            "regression_trending_up",
            "binary_clf_trending_up",
            "beats_h_baseline_regression",
            "beats_h_baseline_binary_clf",
            "beats_h_baseline_overall",
        ]
    )


@pytest.mark.parametrize(
    "reason", ["one_round", "two_rounds", "regression_plateau", "binary_plateau", "overall_below_baseline"]
)
def test_ns16_gate_preserves_finite_rejections(ns16_gate_inputs, reason):
    gate, inputs = ns16_gate_inputs
    if reason == "one_round":
        inputs["round_names"] = ["ROUND_2"]
    elif reason == "two_rounds":
        inputs["round_names"] = ["ROUND_1", "ROUND_2"]
    elif reason.endswith("_plateau"):
        axis = reason.removesuffix("_plateau")
        inputs[f"{axis}_ratings"]["ROUND_2"] = 1500.0
    else:
        inputs["overall_ratings"]["H"] = 1520.0
    assert gate(**inputs)["gate_pass"] is False


def test_ns19_dope_freeze_storage_aliases_normalizers_and_anchor_guards(monkeypatch) -> None:
    monkeypatch.delenv("JOPE_RESEARCH_DOPE_UNFREEZE", raising=False)
    model = _ToyFusedNet()
    model.gamenet_anchor = nn.Linear(4, 4, bias=False)
    # Simulate a pre-existing requires_grad=True buffer before locking
    model.step2_lm_mean_1x256.requires_grad_(True)
    lock_dope_weights_and_eval_mode(model)
    assert not model.step2_lm_mean_1x256.requires_grad

    snap = snapshot_dope_weights(model)
    counts = get_trainable_param_counts_by_submodule(model)
    assert counts["gamenet_anchor"]["trainable_params"] == 0
    assert counts["gamenet_anchor"]["training_mode"] is False

    # 1. Storage-aliased view / wrapped Parameter of a DOPE parameter must be rejected by assert_optimizer_excludes_dope
    dope_param = next(model.dope_model.parameters())
    dope_view_alias = nn.Parameter(dope_param.data.view(-1))
    assert id(dope_view_alias) != id(dope_param)
    opt_dope_alias = torch.optim.SGD([dope_view_alias], lr=1e-2)
    with pytest.raises(DopeFreezeViolationError, match="storage alias"):
        assert_optimizer_excludes_dope(opt_dope_alias, model)

    # 2. Storage-aliased view of gamenet_anchor or DOPE normalizer buffer must also be rejected
    anchor_param = next(model.gamenet_anchor.parameters())
    anchor_view_alias = nn.Parameter(anchor_param.data.view(-1))
    opt_anchor_alias = torch.optim.SGD([anchor_view_alias], lr=1e-2)
    with pytest.raises(DopeFreezeViolationError, match="storage alias"):
        assert_optimizer_excludes_dope(opt_anchor_alias, model)

    buf_alias = nn.Parameter(model.step2_lm_mean_1x256.data.view(-1))
    opt_buf_alias = torch.optim.SGD([buf_alias], lr=1e-2)
    with pytest.raises(DopeFreezeViolationError, match="storage alias"):
        assert_optimizer_excludes_dope(opt_buf_alias, model)

    # 3. Normalizer buffer requires_grad=True or non-finite NaN/Inf must be rejected by snapshot_dope_weights
    model.step2_lm_mean_1x256.requires_grad_(True)
    with pytest.raises(DopeFreezeViolationError, match="normalizer.step2_lm_mean_1x256.*requires_grad=True"):
        snapshot_dope_weights(model)
    model.step2_lm_mean_1x256.requires_grad_(False)

    orig_buf = model.step2_lm_mean_1x256.clone()
    with torch.no_grad():
        model.step2_lm_mean_1x256.fill_(float("nan"))
    with pytest.raises(DopeFreezeViolationError, match="non-finite"):
        snapshot_dope_weights(model)
    with torch.no_grad():
        model.step2_lm_mean_1x256.copy_(orig_buf)

    # 4. Frozen anchor requires_grad=True or training=True must be rejected by snapshot_dope_weights and counts
    anchor_param.requires_grad_(True)
    with pytest.raises(DopeFreezeViolationError, match="gamenet_anchor.*requires_grad=True"):
        snapshot_dope_weights(model)
    with pytest.raises(DopeFreezeViolationError, match="gamenet_anchor has non-zero trainable_params"):
        get_trainable_param_counts_by_submodule(model)
    anchor_param.requires_grad_(False)

    model.gamenet_anchor.training = True
    with pytest.raises(DopeFreezeViolationError, match="gamenet_anchor.*training"):
        snapshot_dope_weights(model)
    with pytest.raises(DopeFreezeViolationError, match="gamenet_anchor.*training"):
        get_trainable_param_counts_by_submodule(model)
    model.gamenet_anchor.training = False

    verify_dope_frozen_unchanged(model, snap)

    # 5. Research policy also rejects storage-aliased anchor or normalizer buffer in optimizer
    configure_research_dope(model, launch_metadata={}, unfreeze=True, research_run_id="H206-ns19")
    with pytest.raises(DopeFreezeViolationError, match="frozen anchor"):
        assert_research_optimizer_policy(opt_anchor_alias, model, research_run_id="H206-ns19")
    with pytest.raises(DopeFreezeViolationError, match="normalizer buffer"):
        assert_research_optimizer_policy(opt_buf_alias, model, research_run_id="H206-ns19")


def test_ns21_unified_eval_holdout_degeneracy_ood_binary_and_shape_guards() -> None:
    from jope.north_star import (
        detect_binary_target_batched_gpu,
        fit_batched_platt_calibration_gpu,
    )

    # 1. Constant regression holdout target (sst <= 1e-12) must never award R^2 = 1.0,
    # while preserving valid regression/binary sibling tables in the same batch.
    raw_tr = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],  # b=0: valid regression
            [1.0, 2.0, 3.0, 4.0],  # b=1: regression with constant holdout y_te
            [-2.0, 2.0, -1.5, 1.5],  # b=2: binary {0, 1} with out-of-domain holdout label 2.0
        ],
        dtype=torch.float32,
    )
    y_tr = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    m_tr = torch.ones((3, 4), dtype=torch.bool)

    raw_te = torch.tensor(
        [
            [1.5, 2.5, 3.5, 4.5],
            [5.0, 5.0, 5.0, 5.0],  # exact match to constant y_te
            [-1.0, 1.0, -1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    y_te = torch.tensor(
        [
            [1.5, 2.5, 3.5, 4.5],
            [5.0, 5.0, 5.0, 5.0],  # constant holdout regression target
            [0.0, 1.0, 0.0, 2.0],  # 2.0 is out-of-domain for {0.0, 1.0}
        ],
        dtype=torch.float32,
    )
    m_te = torch.ones((3, 4), dtype=torch.bool)

    res = evaluate_unified_regression_and_binary_batched_gpu(raw_tr, y_tr, m_tr, raw_te, y_te, m_te)
    assert res["is_binary"].tolist() == [False, False, True]
    assert res["valid_eval_mask"].tolist() == [True, False, False]
    assert abs(float(res["unified_r2_score"][0]) - 1.0) < 1e-6
    for idx in (1, 2):
        for metric_key in (
            "unified_r2_score",
            "regression_r2",
            "brier_r2",
            "roc_auc",
            "log_loss",
            "balanced_accuracy",
        ):
            assert math.isnan(float(res[metric_key][idx])), (metric_key, idx, res[metric_key])

    with pytest.raises(ValueError, match="Invalid evaluation batch"):
        evaluate_unified_regression_and_binary_batched_gpu(
            raw_tr[1:2], y_tr[1:2], m_tr[1:2], raw_te[1:2], y_te[1:2], m_te[1:2], reject_nonfinite_eval=True
        )
    with pytest.raises(ValueError, match="Invalid evaluation batch"):
        evaluate_unified_regression_and_binary_batched_gpu(
            raw_tr[2:3], y_tr[2:3], m_tr[2:3], raw_te[2:3], y_te[2:3], m_te[2:3], reject_nonfinite_eval=True
        )

    # Masked-out out-of-domain binary padding must be ignored when mask_te is False on that column
    m_te_masked_ood = m_te[2:3].clone()
    m_te_masked_ood[0, 3] = False
    res_masked_ood = evaluate_unified_regression_and_binary_batched_gpu(
        raw_tr[2:3], y_tr[2:3], m_tr[2:3], raw_te[2:3], y_te[2:3], m_te_masked_ood, reject_nonfinite_eval=True
    )
    assert res_masked_ood["valid_eval_mask"].tolist() == [True]
    assert bool(torch.isfinite(res_masked_ood["unified_r2_score"]).all())

    # 2. batched_roc_auc_gpu rejects non-binary required labels (outside {0, 1}) as NaN per table
    auc_nonbin = batched_roc_auc_gpu(
        torch.tensor([[0.1, 0.4, 0.6, 0.9], [0.1, 0.4, 0.6, 0.9]]),
        torch.tensor([[0.0, 0.5, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]),
        torch.ones((2, 4), dtype=torch.bool),
    )
    assert math.isnan(float(auc_nonbin[0]))
    assert abs(float(auc_nonbin[1]) - 1.0) < 1e-6

    # 3. Shape, rank, hyperparameter, and mask domain validation
    with pytest.raises(ValueError, match="2D torch.Tensor"):
        detect_binary_target_batched_gpu(torch.zeros(4), torch.ones(4, dtype=torch.bool))
    with pytest.raises(ValueError, match="does not match expected shape"):
        detect_binary_target_batched_gpu(torch.zeros(2, 4), torch.ones(1, 4, dtype=torch.bool))
    with pytest.raises(ValueError, match="tol must be a finite non-negative"):
        detect_binary_target_batched_gpu(torch.zeros(2, 4), torch.ones(2, 4, dtype=torch.bool), tol=-1e-5)
    with pytest.raises(ValueError, match="non-finite values"):
        detect_binary_target_batched_gpu(torch.zeros(1, 4), torch.tensor([[1.0, 0.0, float("nan"), 1.0]]))
    with pytest.raises(ValueError, match="contain only \\{0, 1\\}"):
        detect_binary_target_batched_gpu(torch.zeros(1, 4), torch.tensor([[1, 0, 2, 1]]))

    with pytest.raises(ValueError, match="l2_reg must be a finite positive"):
        fit_batched_platt_calibration_gpu(
            torch.zeros(1, 4), torch.zeros(1, 4), torch.ones(1, 4, dtype=torch.bool), l2_reg=0.0
        )
    with pytest.raises(ValueError, match="n_iters must be a non-negative int"):
        fit_batched_platt_calibration_gpu(
            torch.zeros(1, 4), torch.zeros(1, 4), torch.ones(1, 4, dtype=torch.bool), n_iters=-1
        )
    with pytest.raises(ValueError, match="Batch size mismatch"):
        evaluate_unified_regression_and_binary_batched_gpu(
            raw_tr[:2], y_tr[:2], m_tr[:2], raw_te[:1], y_te[:1], m_te[:1]
        )
