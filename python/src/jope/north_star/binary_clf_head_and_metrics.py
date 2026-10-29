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
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class BatchedTaskDetection:
    is_binary: torch.Tensor  # [B] bool
    y0_val: torch.Tensor  # [B, 1] float32 (negative class original value)
    y1_val: torch.Tensor  # [B, 1] float32 (positive class original value)
    y_bin_01: torch.Tensor  # [B, N] float32 in {0.0, 1.0} for binary tables
    pos_rate: torch.Tensor  # [B, 1] float32 in (0, 1)


def _sanitize_masked(x: torch.Tensor, mask: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    """
    Replaces masked-out (`mask == False`) or non-finite values with `fill` BEFORE any arithmetic
    so `NaN * 0.0 -> NaN` never occurs in reductions.
    """
    xf = x.float()
    mb = mask.bool() & torch.isfinite(xf)
    return torch.where(mb, xf, torch.full_like(xf, fill))


def detect_binary_target_batched_gpu(
    y: torch.Tensor,
    valid_mask: torch.Tensor,
    tol: float = 1e-5,
) -> BatchedTaskDetection:
    """
    Vectorized GPU check for whether each table `b` in `[B, N]` has a binary target:
    - Computes `y_min = min(y_valid)` and `y_max = max(y_valid)` per table.
    - A table is binary iff `y_max > y_min + tol`, every valid element is within `tol` of
      either `y_min` or `y_max`, and both classes have at least 2 observations.
    """
    B, N = y.shape
    y_f = y.float()
    mask_f = valid_mask.bool() & torch.isfinite(y_f)
    y_clean = torch.where(mask_f, y_f, torch.zeros_like(y_f))

    pos_inf = torch.full_like(y_f, float("inf"))
    neg_inf = torch.full_like(y_f, float("-inf"))
    y_min = torch.where(mask_f, y_clean, pos_inf).amin(dim=1, keepdim=True)
    y_max = torch.where(mask_f, y_clean, neg_inf).amax(dim=1, keepdim=True)
    y_min = torch.where(torch.isfinite(y_min), y_min, torch.zeros_like(y_min))
    y_max = torch.where(torch.isfinite(y_max), y_max, torch.ones_like(y_max))

    span_ok = (y_max - y_min) > tol
    near_min = (y_clean - y_min).abs() <= tol
    near_max = (y_clean - y_max).abs() <= tol
    all_two_vals = ((near_min | near_max) | (~mask_f)).all(dim=1, keepdim=True)

    n_min = (near_min & mask_f).sum(dim=1, keepdim=True)
    n_max = (near_max & mask_f).sum(dim=1, keepdim=True)
    n_tot = mask_f.sum(dim=1, keepdim=True).clamp_min(1)

    is_bin = (span_ok & all_two_vals & (n_min >= 2) & (n_max >= 2)).squeeze(1)
    denom = (y_max - y_min).clamp_min(1e-6)
    y_bin_01 = torch.where(mask_f, ((y_clean - y_min) / denom).clamp(0.0, 1.0), torch.zeros_like(y_clean))
    pos_rate = (n_max.float() / n_tot.float()).clamp(1e-4, 1.0 - 1e-4)

    return BatchedTaskDetection(
        is_binary=is_bin,
        y0_val=y_min,
        y1_val=y_max,
        y_bin_01=y_bin_01,
        pos_rate=pos_rate,
    )


def fit_batched_platt_calibration_gpu(
    raw_score_tr: torch.Tensor,
    y_bin_tr: torch.Tensor,
    valid_mask_tr: torch.Tensor,
    l2_reg: float = 1e-2,
    n_iters: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fits a 2-parameter Platt / logistic link `p = sigmoid(a * raw_score + b)` per table `[B]`
    using 6 batched Newton-Raphson (IRLS) steps on GPU (`2x2` closed-form solve, zero host sync).
    Initialized at `a = 1.0`, `b = logit(pos_rate)`.
    """
    B, N = raw_score_tr.shape
    mask_b = valid_mask_tr.bool() & torch.isfinite(raw_score_tr) & torch.isfinite(y_bin_tr)
    mask_f = mask_b.float()
    s = _sanitize_masked(raw_score_tr, mask_b, 0.0)
    y = _sanitize_masked(y_bin_tr, mask_b, 0.0)

    n_valid = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    pos_rate = (y.sum(dim=1, keepdim=True) / n_valid).clamp(1e-4, 1.0 - 1e-4)

    a = torch.ones((B, 1), dtype=torch.float32, device=raw_score_tr.device)
    b = torch.log(pos_rate / (1.0 - pos_rate))

    for _ in range(n_iters):
        logits = (a * s + b).clamp(-15.0, 15.0)
        p = torch.sigmoid(logits)
        w = (p * (1.0 - p)).clamp_min(1e-5) * mask_f
        diff = (p - y) * mask_f

        # Gradient of NLL + 0.5 * l2_reg * a^2
        g_a = (diff * s).sum(dim=1, keepdim=True) + l2_reg * a
        g_b = diff.sum(dim=1, keepdim=True) + 1e-4 * b

        # 2x2 Hessian entries
        h_aa = (w * s * s).sum(dim=1, keepdim=True) + l2_reg
        h_ab = (w * s).sum(dim=1, keepdim=True)
        h_bb = w.sum(dim=1, keepdim=True) + 1e-4

        det = (h_aa * h_bb - h_ab * h_ab).clamp_min(1e-6)
        da = (h_bb * g_a - h_ab * g_b) / det
        db = (h_aa * g_b - h_ab * g_a) / det

        a = (a - da).clamp(-20.0, 20.0)
        b = (b - db).clamp(-15.0, 15.0)

    return a, b


def batched_roc_auc_gpu(
    prob_pred: torch.Tensor,
    y_bin_true: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Computes exact tie-aware ROC-AUC per table `[B]` on GPU via Mann-Whitney U statistics
    with exact mid-rank averaging (`torch.searchsorted` left/right bounds) and zero CPU loops.
    Preserves `valid_mask` strictly: if any required (`valid_mask == True`) entry has non-finite
    `prob_pred` or `y_bin_true`, or if `valid_mask` has 0 valid rows, returns `NaN` for that table.
    """
    B, N = prob_pred.shape
    req_mask = valid_mask.bool()
    has_nonfinite_req = ((~torch.isfinite(prob_pred) | ~torch.isfinite(y_bin_true)) & req_mask).any(
        dim=1, keepdim=True
    )
    zero_coverage = req_mask.sum(dim=1, keepdim=True) == 0
    invalid_table = has_nonfinite_req | zero_coverage

    mask_b = req_mask & torch.isfinite(prob_pred) & torch.isfinite(y_bin_true)
    y_clean = _sanitize_masked(y_bin_true, mask_b, 0.0)

    # Place invalid rows at -inf so they sort strictly before all finite valid scores
    scores = torch.where(mask_b, prob_pred.float(), torch.full_like(prob_pred.float(), float("-inf")))
    s_sorted, order = torch.sort(scores, dim=1, stable=True)
    y_sorted = torch.gather(y_clean, 1, order)
    m_sorted = torch.gather(mask_b, 1, order)

    # Exact tie-averaged mid-ranks in O(N log N) via left/right binary search in sorted s_sorted
    first_pos = torch.searchsorted(s_sorted, s_sorted, side="left")
    last_pos = torch.searchsorted(s_sorted, s_sorted, side="right") - 1
    mid_ranks_all = 0.5 * (first_pos.float() + last_pos.float()) + 1.0

    # Shift by number of invalid (-inf) rows so valid rows have 1-based mid-ranks in [1, N_valid]
    n_invalid = (~m_sorted).sum(dim=1, keepdim=True).float()
    ranks = torch.where(m_sorted, mid_ranks_all - n_invalid, torch.zeros_like(mid_ranks_all))

    pos_mask = (y_sorted > 0.5) & m_sorted
    neg_mask = (y_sorted <= 0.5) & m_sorted
    n_pos = pos_mask.sum(dim=1, keepdim=True).float()
    n_neg = neg_mask.sum(dim=1, keepdim=True).float()

    rank_sum_pos = (ranks * pos_mask.float()).sum(dim=1, keepdim=True)
    u_stat = rank_sum_pos - 0.5 * n_pos * (n_pos + 1.0)
    denom = (n_pos * n_neg).clamp_min(1.0)
    auc = (u_stat / denom).clamp(0.0, 1.0)
    auc = torch.where((n_pos >= 1.0) & (n_neg >= 1.0), auc, torch.full_like(auc, 0.5))
    auc = torch.where(invalid_table, torch.full_like(auc, float("nan")), auc)
    return auc.squeeze(1)


def evaluate_unified_regression_and_binary_batched_gpu(
    raw_pred_tr: torch.Tensor,
    y_tr: torch.Tensor,
    mask_tr: torch.Tensor,
    raw_pred_te: torch.Tensor,
    y_te: torch.Tensor,
    mask_te: torch.Tensor,
    reject_nonfinite_eval: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Evaluates both Regression and Binary Classification tables in a single batched GPU pass:
    - Sanitizes all masked-out (`mask == False`) entries before arithmetic so padded NaNs/Infs
      never poison reductions.
    - Strictly preserves the required evaluation masks `mask_tr` and `mask_te`: if any required
      (`mask == True`) prediction or label is non-finite (`NaN`/`Inf`), or if a table has zero
      valid train or evaluation rows, marks `valid_eval_mask = False` and sets all metrics for
      that table to `NaN` (or raises `ValueError` when `reject_nonfinite_eval=True`); NEVER
      drops required evaluation rows or awards `1.0`.
    """
    req_tr_b = mask_tr.bool()
    req_te_b = mask_te.bool()
    nonfinite_tr = ((~torch.isfinite(raw_pred_tr) | ~torch.isfinite(y_tr)) & req_tr_b).any(dim=1)
    nonfinite_te = ((~torch.isfinite(raw_pred_te) | ~torch.isfinite(y_te)) & req_te_b).any(dim=1)
    zero_tr = req_tr_b.sum(dim=1) == 0
    zero_te = req_te_b.sum(dim=1) == 0
    invalid_eval = nonfinite_tr | nonfinite_te | zero_tr | zero_te

    if reject_nonfinite_eval and bool(invalid_eval.any()):
        raise ValueError(
            "Invalid evaluation batch: non-finite predictions/labels on required valid mask or zero valid coverage."
        )

    m_tr_b = req_tr_b & torch.isfinite(raw_pred_tr) & torch.isfinite(y_tr)
    m_te_b = req_te_b & torch.isfinite(raw_pred_te) & torch.isfinite(y_te)

    y_tr_c = _sanitize_masked(y_tr, m_tr_b, 0.0)
    y_te_c = _sanitize_masked(y_te, m_te_b, 0.0)
    p_tr_c = _sanitize_masked(raw_pred_tr, m_tr_b, 0.0)
    p_te_c = _sanitize_masked(raw_pred_te, m_te_b, 0.0)

    det_tr = detect_binary_target_batched_gpu(y_tr_c, m_tr_b)
    is_bin = det_tr.is_binary

    # 1. Regression R^2 path
    m_te_f = m_te_b.float()
    n_te = m_te_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    y_te_mean = y_te_c.sum(dim=1, keepdim=True) / n_te
    diff_err = torch.where(m_te_b, y_te_c - p_te_c, torch.zeros_like(y_te_c))
    diff_tot = torch.where(m_te_b, y_te_c - y_te_mean, torch.zeros_like(y_te_c))
    sse_reg = (diff_err * diff_err).sum(dim=1)
    sst_reg = (diff_tot * diff_tot).sum(dim=1).clamp_min(1e-12)
    reg_r2 = 1.0 - sse_reg / sst_reg

    # 2. Binary Classification path (map y_te using train's y0_val, y1_val)
    denom_bin = (det_tr.y1_val - det_tr.y0_val).clamp_min(1e-6)
    y_te_bin = torch.where(
        m_te_b,
        ((y_te_c - det_tr.y0_val) / denom_bin).clamp(0.0, 1.0),
        torch.zeros_like(y_te_c),
    )

    # Standardize raw_pred_tr for numerically stable Platt calibration
    m_tr_f = m_tr_b.float()
    n_tr = m_tr_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    mu_p = p_tr_c.sum(dim=1, keepdim=True) / n_tr
    diff_p_tr = torch.where(m_tr_b, p_tr_c - mu_p, torch.zeros_like(p_tr_c))
    sd_p = ((diff_p_tr * diff_p_tr).sum(dim=1, keepdim=True) / n_tr).sqrt().clamp_min(1e-5)
    z_pred_tr = torch.where(m_tr_b, (p_tr_c - mu_p) / sd_p, torch.zeros_like(p_tr_c))
    z_pred_te = torch.where(m_te_b, (p_te_c - mu_p) / sd_p, torch.zeros_like(p_te_c))

    platt_a, platt_b = fit_batched_platt_calibration_gpu(z_pred_tr, det_tr.y_bin_01, m_tr_b)
    prob_te = torch.sigmoid((platt_a * z_pred_te + platt_b).clamp(-15.0, 15.0))
    prob_te = torch.where(m_te_b, prob_te, torch.full_like(prob_te, 0.5))

    # Brier R^2 (Efron skill score relative to holdout prevalence)
    p1_te = (y_te_bin.sum(dim=1, keepdim=True) / n_te).clamp(1e-4, 1.0 - 1e-4)
    brier_diff = torch.where(m_te_b, y_te_bin - prob_te, torch.zeros_like(y_te_bin))
    brier = (brier_diff * brier_diff).sum(dim=1) / n_te.squeeze(1)
    brier_null = (p1_te.squeeze(1) * (1.0 - p1_te.squeeze(1))).clamp_min(1e-6)
    brier_r2 = 1.0 - brier / brier_null

    # ROC-AUC (using strict req_te_b), Log-Loss, Balanced Accuracy
    prob_te_for_auc = torch.where(
        nonfinite_te.unsqueeze(1) & req_te_b,
        torch.full_like(prob_te, float("nan")),
        prob_te,
    )
    roc_auc = batched_roc_auc_gpu(prob_te_for_auc, y_te_bin, req_te_b)
    eps = 1e-6
    p_clip = prob_te.clamp(eps, 1.0 - eps)
    bce = -(y_te_bin * torch.log(p_clip) + (1.0 - y_te_bin) * torch.log(1.0 - p_clip))
    bce = torch.where(m_te_b, bce, torch.zeros_like(bce))
    log_loss = bce.sum(dim=1) / n_te.squeeze(1)

    pred_lbl = (prob_te >= 0.5).float()
    pos_m = (y_te_bin > 0.5) & m_te_b
    neg_m = (y_te_bin <= 0.5) & m_te_b
    tpr = ((pred_lbl == 1.0) & pos_m).sum(dim=1).float() / pos_m.sum(dim=1).clamp_min(1).float()
    tnr = ((pred_lbl == 0.0) & neg_m).sum(dim=1).float() / neg_m.sum(dim=1).clamp_min(1).float()
    bal_acc = 0.5 * (tpr + tnr)

    unified_r2 = torch.where(is_bin, brier_r2, reg_r2)

    nan_1d = torch.full_like(unified_r2, float("nan"))
    unified_r2 = torch.where(invalid_eval, nan_1d, unified_r2)
    reg_r2 = torch.where(invalid_eval, nan_1d, reg_r2)
    brier_r2 = torch.where(invalid_eval, nan_1d, brier_r2)
    roc_auc = torch.where(invalid_eval, nan_1d, roc_auc)
    log_loss = torch.where(invalid_eval, nan_1d, log_loss)
    bal_acc = torch.where(invalid_eval, nan_1d, bal_acc)

    return {
        "is_binary": is_bin,
        "valid_eval_mask": ~invalid_eval,
        "unified_r2_score": unified_r2,
        "regression_r2": reg_r2,
        "brier_r2": brier_r2,
        "roc_auc": roc_auc,
        "log_loss": log_loss,
        "balanced_accuracy": bal_acc,
        "platt_a": platt_a.squeeze(1),
        "platt_b": platt_b.squeeze(1),
    }


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
