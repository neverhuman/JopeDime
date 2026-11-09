from __future__ import annotations

import torch

if __package__:
    from .binary_clf_head import (
        _sanitize_masked,
        _validate_2d_mask,
        _validate_2d_tensor,
        detect_binary_target_batched_gpu,
        fit_batched_platt_calibration_gpu,
    )
else:
    from binary_clf_head import (
        _sanitize_masked,
        _validate_2d_mask,
        _validate_2d_tensor,
        detect_binary_target_batched_gpu,
        fit_batched_platt_calibration_gpu,
    )


def batched_roc_auc_gpu(
    prob_pred: torch.Tensor,
    y_bin_true: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Computes exact tie-aware ROC-AUC per table `[B]` on GPU via Mann-Whitney U statistics
    with exact mid-rank averaging (`torch.searchsorted` left/right bounds) and zero CPU loops.
    Preserves `valid_mask` strictly: if any required (`valid_mask == True`) entry has non-finite
    `prob_pred` or `y_bin_true`, or non-binary `y_bin_true` outside `{0.0, 1.0}`, or if
    `valid_mask` has 0 valid rows, returns `NaN` for that table.
    AUC is also unavailable (`NaN`) when either target class has no observed valid row.
    """
    B, N = _validate_2d_tensor(prob_pred, "prob_pred")
    _validate_2d_tensor(y_bin_true, "y_bin_true", expected_shape=(B, N))
    req_mask = _validate_2d_mask(valid_mask, (B, N), "valid_mask")
    has_nonfinite_req = ((~torch.isfinite(prob_pred) | ~torch.isfinite(y_bin_true)) & req_mask).any(
        dim=1, keepdim=True
    )
    y_f = y_bin_true.float()
    is_zero_or_one = (y_f.abs() <= 1e-5) | ((y_f - 1.0).abs() <= 1e-5)
    has_nonbinary_req = (req_mask & torch.isfinite(y_bin_true) & ~is_zero_or_one).any(dim=1, keepdim=True)
    zero_coverage = req_mask.sum(dim=1, keepdim=True) == 0
    invalid_table = has_nonfinite_req | has_nonbinary_req | zero_coverage

    mask_b = req_mask & torch.isfinite(prob_pred) & torch.isfinite(y_bin_true) & is_zero_or_one
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
    auc = torch.where((n_pos >= 1.0) & (n_neg >= 1.0), auc, torch.full_like(auc, float("nan")))
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
    - Validates 2D tensor shapes and boolean/{0,1} masks across train and evaluation inputs.
    - Sanitizes all masked-out (`mask == False`) entries before arithmetic so padded NaNs/Infs
      never poison reductions.
    - Strictly preserves the required evaluation masks `mask_tr` and `mask_te`: if any required
      (`mask == True`) prediction or label is non-finite (`NaN`/`Inf`), if a table has zero
      valid train or evaluation rows, if a regression table has a zero-variance (constant)
      holdout target (`sst <= 1e-12`), or if a binary table has out-of-domain holdout labels
      outside `{y0_val, y1_val}`, marks `valid_eval_mask = False` and sets all evaluation metrics
      for that table to `NaN` (or raises `ValueError` when `reject_nonfinite_eval=True`); NEVER
      drops required evaluation rows or awards `1.0`.
    - ROC-AUC additionally requires both classes in the evaluation rows. A single-class
      evaluation has unavailable (`NaN`) AUC while input validity and other metrics are preserved.
    """
    B_tr, N_tr = _validate_2d_tensor(raw_pred_tr, "raw_pred_tr")
    _validate_2d_tensor(y_tr, "y_tr", expected_shape=(B_tr, N_tr))
    req_tr_b = _validate_2d_mask(mask_tr, (B_tr, N_tr), "mask_tr")

    B_te, N_te = _validate_2d_tensor(raw_pred_te, "raw_pred_te")
    if B_te != B_tr:
        raise ValueError(f"Batch size mismatch between train ({B_tr}) and evaluation ({B_te}) tensors")
    _validate_2d_tensor(y_te, "y_te", expected_shape=(B_te, N_te))
    req_te_b = _validate_2d_mask(mask_te, (B_te, N_te), "mask_te")

    nonfinite_tr = ((~torch.isfinite(raw_pred_tr) | ~torch.isfinite(y_tr)) & req_tr_b).any(dim=1)
    nonfinite_te = ((~torch.isfinite(raw_pred_te) | ~torch.isfinite(y_te)) & req_te_b).any(dim=1)
    zero_tr = req_tr_b.sum(dim=1) == 0
    zero_te = req_te_b.sum(dim=1) == 0

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
    sst_raw = (diff_tot * diff_tot).sum(dim=1)
    sst_reg = sst_raw.clamp_min(1e-12)
    reg_r2 = 1.0 - sse_reg / sst_reg

    # Constant holdout target on a regression table has undefined R^2 (must never award 1.0)
    degen_reg_te = (~is_bin) & (sst_raw <= 1e-12)

    # 2. Binary Classification path (map y_te using train's y0_val, y1_val)
    near_y0_te = (y_te_c - det_tr.y0_val).abs() <= 1e-5
    near_y1_te = (y_te_c - det_tr.y1_val).abs() <= 1e-5
    ood_bin_te = (is_bin.unsqueeze(1) & m_te_b & ~near_y0_te & ~near_y1_te).any(dim=1)

    invalid_eval = nonfinite_tr | nonfinite_te | zero_tr | zero_te | degen_reg_te | ood_bin_te

    if reject_nonfinite_eval and bool(invalid_eval.any()):
        raise ValueError(
            "Invalid evaluation batch: non-finite predictions/labels on required valid mask, "
            "zero valid coverage, degenerate constant regression holdout target, or out-of-domain binary holdout labels."
        )

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
