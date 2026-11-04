#!/usr/bin/env python3
"""
NORTH STAR Phase D (NS4): Unified Regression + Binary Classification Head & Batched GPU Metrics
===============================================================================================
Implements a single-model ("one model preferred") extension for `FusedJopeNetwork` so it handles
both continuous Regression and Binary Classification without modifying frozen DOPE weights:

1. `detect_binary_target_batched_gpu(y_tr, valid_mask)`:
   - Vectorized GPU detection of binary targets (exact 2 distinct finite values `v0 < v1` per table),
     mapping `y -> {0.0, 1.0}` and computing positive prevalence `p1`.
2. Why Frozen DOPE-MLC-v3 Works Unchanged on Binary Targets:
   - Standardized binary `y_z = (y - p1) / sqrt(p1 * (1 - p1))` turns Stage 1A's Pearson correlation
     probes into exact Point-Biserial correlations (two-sample Welch/Fisher discriminant statistics),
     while target skewness `(1 - 2*p1)/sqrt(p1*(1-p1))` and kurtosis `1/(p1*(1-p1)) - 6` in `ms_376`
     uniquely encode class imbalance `p1`.
3. `fit_batched_platt_or_logistic_head_gpu(...)`:
   - Fits a batched 5-step Newton-Raphson (IRLS) L2-penalized logistic / Platt calibration layer on
     GPU (`p_hat = sigmoid(scale * raw_pred + bias)`) for binary classification tables while keeping
     identity link for regression tables.
4. `compute_binary_and_regression_metrics_batched_gpu(...)`:
   - Regression: `hold_r2`, `rmse`, `mae`.
   - Binary Classification:
     * `brier_r2 = 1.0 - Brier / (p1 * (1 - p1))` (Efron/Brier skill score on the exact same
       `(-inf, 1.0]` scale as regression `R^2`, unifying Stage 2 `rev_f = clamp(-score, -1, 2)`
       and Stage 3 GCV selection inside a single network!).
     * `roc_auc` (exact vectorized Mann-Whitney U rank statistic on GPU, tie-aware).
     * `log_loss` (clipped binary cross-entropy) and `null_log_loss_r2` (McFadden pseudo-R^2).
     * `accuracy` and `balanced_accuracy`.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

if __package__:
    from .binary_clf_head import (
        BatchedTaskDetection as BatchedTaskDetection,
    )
    from .binary_clf_head import (
        _sanitize_masked as _sanitize_masked,
    )
    from .binary_clf_head import (
        _validate_2d_mask as _validate_2d_mask,
    )
    from .binary_clf_head import (
        _validate_2d_tensor as _validate_2d_tensor,
    )
    from .binary_clf_head import (
        detect_binary_target_batched_gpu as detect_binary_target_batched_gpu,
    )
    from .binary_clf_head import (
        fit_batched_platt_calibration_gpu as fit_batched_platt_calibration_gpu,
    )
    from .binary_clf_metrics import (
        batched_roc_auc_gpu,
        evaluate_unified_regression_and_binary_batched_gpu,
    )
else:
    from binary_clf_head import (
        BatchedTaskDetection as BatchedTaskDetection,
    )
    from binary_clf_head import (
        _sanitize_masked as _sanitize_masked,
    )
    from binary_clf_head import (
        _validate_2d_mask as _validate_2d_mask,
    )
    from binary_clf_head import (
        _validate_2d_tensor as _validate_2d_tensor,
    )
    from binary_clf_head import (
        detect_binary_target_batched_gpu as detect_binary_target_batched_gpu,
    )
    from binary_clf_head import (
        fit_batched_platt_calibration_gpu as fit_batched_platt_calibration_gpu,
    )
    from binary_clf_metrics import (
        batched_roc_auc_gpu,
        evaluate_unified_regression_and_binary_batched_gpu,
    )


def run_self_test() -> dict:
    """
    Verifies unified Regression + Binary Classification evaluation on mixed batches
    (2 regression tables + 2 binary classification tables, including {-1, +1} and {0, 1} labels).
    """
    torch.manual_seed(7)
    B, N_tr, N_te = 4, 256, 128
    mask_tr = torch.ones((B, N_tr), dtype=torch.bool)
    mask_te = torch.ones((B, N_te), dtype=torch.bool)

    x_tr = torch.randn(B, N_tr)
    x_te = torch.randn(B, N_te)

    y_tr = torch.zeros(B, N_tr)
    y_te = torch.zeros(B, N_te)
    raw_tr = torch.zeros(B, N_tr)
    raw_te = torch.zeros(B, N_te)

    # b=0, 1: Continuous regression
    for b in (0, 1):
        y_tr[b] = 2.5 * x_tr[b] + 0.3 * torch.randn(N_tr)
        y_te[b] = 2.5 * x_te[b] + 0.3 * torch.randn(N_te)
        raw_tr[b] = 2.45 * x_tr[b]
        raw_te[b] = 2.45 * x_te[b]

    # b=2: Binary classification in {0.0, 1.0}
    prob_tr_2 = torch.sigmoid(2.2 * x_tr[2] - 0.4)
    prob_te_2 = torch.sigmoid(2.2 * x_te[2] - 0.4)
    y_tr[2] = torch.bernoulli(prob_tr_2)
    y_te[2] = torch.bernoulli(prob_te_2)
    raw_tr[2] = 0.8 * x_tr[2]
    raw_te[2] = 0.8 * x_te[2]

    # b=3: Binary classification in {-1.0, +1.0}
    prob_tr_3 = torch.sigmoid(-1.8 * x_tr[3] + 0.2)
    prob_te_3 = torch.sigmoid(-1.8 * x_te[3] + 0.2)
    y_tr[3] = torch.where(torch.bernoulli(prob_tr_3) > 0.5, 1.0, -1.0)
    y_te[3] = torch.where(torch.bernoulli(prob_te_3) > 0.5, 1.0, -1.0)
    raw_tr[3] = -0.9 * x_tr[3]
    raw_te[3] = -0.9 * x_te[3]

    res = evaluate_unified_regression_and_binary_batched_gpu(raw_tr, y_tr, mask_tr, raw_te, y_te, mask_te)
    assert res["is_binary"].tolist() == [False, False, True, True]
    assert float(res["regression_r2"][0]) > 0.90
    assert float(res["roc_auc"][2]) > 0.80
    assert float(res["roc_auc"][3]) > 0.75
    assert float(res["brier_r2"][2]) > 0.25

    # Codex Counterexample #1: Four equal 0.5 scores with [0,0,1,1] and [1,1,0,0] must both return 0.5
    tied_scores = torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]], dtype=torch.float32)
    tied_labels = torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    tied_mask = torch.ones((2, 4), dtype=torch.bool)
    tied_aucs = batched_roc_auc_gpu(tied_scores, tied_labels, tied_mask)
    assert torch.allclose(tied_aucs, torch.tensor([0.5, 0.5]), atol=1e-6), f"Tied AUC failed: {tied_aucs}"

    # Codex Counterexample #2: Partial tie + permutation invariance
    # Scores [0.1, 0.5, 0.5, 0.9] with labels [0, 0, 1, 1] -> U = 0.5 (tie at 0.5) + 3 = 3.5 / 4 = 0.875
    p_scores = torch.tensor([[0.1, 0.5, 0.5, 0.9], [0.9, 0.5, 0.5, 0.1]], dtype=torch.float32)
    p_labels = torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    p_aucs = batched_roc_auc_gpu(p_scores, p_labels, tied_mask)
    assert torch.allclose(p_aucs, torch.tensor([0.875, 0.875]), atol=1e-6), (
        f"Partial-tie AUC failed: {p_aucs}"
    )

    # Codex Counterexample #3: Masked-out NaN / Inf in padded positions must match finite padding exactly
    raw_tr_pad_fin = torch.cat([raw_tr, torch.zeros(B, 16)], dim=1)
    y_tr_pad_fin = torch.cat([y_tr, torch.zeros(B, 16)], dim=1)
    m_tr_pad = torch.cat([mask_tr, torch.zeros((B, 16), dtype=torch.bool)], dim=1)
    raw_te_pad_fin = torch.cat([raw_te, torch.zeros(B, 16)], dim=1)
    y_te_pad_fin = torch.cat([y_te, torch.zeros(B, 16)], dim=1)
    m_te_pad = torch.cat([mask_te, torch.zeros((B, 16), dtype=torch.bool)], dim=1)

    raw_tr_pad_nan = raw_tr_pad_fin.clone()
    y_tr_pad_nan = y_tr_pad_fin.clone()
    raw_te_pad_nan = raw_te_pad_fin.clone()
    y_te_pad_nan = y_te_pad_fin.clone()
    raw_tr_pad_nan[:, N_tr:] = float("nan")
    y_tr_pad_nan[:, N_tr:] = float("inf")
    raw_te_pad_nan[:, N_te:] = float("nan")
    y_te_pad_nan[:, N_te:] = float("-inf")

    res_fin = evaluate_unified_regression_and_binary_batched_gpu(
        raw_tr_pad_fin, y_tr_pad_fin, m_tr_pad, raw_te_pad_fin, y_te_pad_fin, m_te_pad
    )
    res_nan = evaluate_unified_regression_and_binary_batched_gpu(
        raw_tr_pad_nan, y_tr_pad_nan, m_tr_pad, raw_te_pad_nan, y_te_pad_nan, m_te_pad
    )
    for k in ("unified_r2_score", "regression_r2", "brier_r2", "roc_auc", "log_loss", "balanced_accuracy"):
        assert torch.isfinite(res_nan[k]).all(), f"NaN leaked into {k}: {res_nan[k]}"
        assert torch.allclose(res_fin[k], res_nan[k], atol=1e-6), (
            f"Padding mismatch on {k}: {res_fin[k]} vs {res_nan[k]}"
        )

    # Codex Counterexample #4 (13:41:58Z): 2 holdout rows explicitly marked valid and BOTH predictions NaN
    # Must NEVER silently drop required evaluation rows or award unified_r2=1.0 / regression_r2=1.0!
    c4_raw_tr = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
    c4_y_tr = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
    c4_m_tr = torch.ones((1, 4), dtype=torch.bool)
    c4_raw_te = torch.tensor([[float("nan"), float("nan")]], dtype=torch.float32)
    c4_y_te = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    c4_m_te = torch.ones((1, 2), dtype=torch.bool)

    res_c4 = evaluate_unified_regression_and_binary_batched_gpu(
        c4_raw_tr, c4_y_tr, c4_m_tr, c4_raw_te, c4_y_te, c4_m_te
    )
    assert res_c4["valid_eval_mask"].tolist() == [False]
    for k in ("unified_r2_score", "regression_r2", "brier_r2", "roc_auc", "log_loss", "balanced_accuracy"):
        assert bool(torch.isnan(res_c4[k]).all()), (
            f"Expected NaN on valid-mask NaN predictions for {k}, got {res_c4[k]}"
        )

    try:
        evaluate_unified_regression_and_binary_batched_gpu(
            c4_raw_tr, c4_y_tr, c4_m_tr, c4_raw_te, c4_y_te, c4_m_te, reject_nonfinite_eval=True
        )
        strict_rejected = False
    except ValueError:
        strict_rejected = True
    assert strict_rejected is True

    # Zero valid holdout coverage must also return NaN (never 1.0)
    res_zero_cov = evaluate_unified_regression_and_binary_batched_gpu(
        c4_raw_tr, c4_y_tr, c4_m_tr, torch.zeros((1, 2)), c4_y_te, torch.zeros((1, 2), dtype=torch.bool)
    )
    assert res_zero_cov["valid_eval_mask"].tolist() == [False]
    assert bool(torch.isnan(res_zero_cov["unified_r2_score"]).all())
    assert bool(torch.isnan(res_zero_cov["regression_r2"]).all())

    out = {
        "task": "NS4_BINARY_CLASSIFICATION_HEAD_AND_METRICS",
        "status": "PASS",
        "primary_ranking_metric_note": (
            "Per NORTH_STAR / phase_push_1352, binary classification contenders are ranked by head-to-head "
            "Method Elo on calibrated Brier R^2 (unified_r2_score); ROC-AUC is a secondary diagnostic."
        ),
        "is_binary": res["is_binary"].tolist(),
        "unified_r2_score": [round(float(x), 5) for x in res["unified_r2_score"]],
        "regression_r2": [round(float(x), 5) for x in res["regression_r2"]],
        "brier_r2": [round(float(x), 5) for x in res["brier_r2"]],
        "roc_auc": [round(float(x), 5) for x in res["roc_auc"]],
        "log_loss": [round(float(x), 5) for x in res["log_loss"]],
        "balanced_accuracy": [round(float(x), 5) for x in res["balanced_accuracy"]],
        "codex_counterexample_checks": {
            "four_equal_05_scores_0011_and_1100_auc": [round(float(x), 6) for x in tied_aucs],
            "partial_tie_permutation_invariant_auc": [round(float(x), 6) for x in p_aucs],
            "masked_nan_inf_padding_matches_finite_padding": True,
            "valid_holdout_both_nan_predictions_returns_nan_never_1": True,
            "valid_holdout_both_nan_strict_rejected": strict_rejected,
            "zero_holdout_coverage_returns_nan_never_1": True,
        },
    }
    (Path(__file__).resolve().parent / "ns4_binary_clf_receipt.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run_self_test()
