from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def _validate_2d_tensor(
    x: torch.Tensor, name: str, expected_shape: tuple[int, int] | None = None
) -> tuple[int, int]:
    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D torch.Tensor, got {type(x).__name__} with ndim={getattr(x, 'ndim', None)}"
        )
    B, N = int(x.shape[0]), int(x.shape[1])
    if B < 1 or N < 1:
        raise ValueError(f"{name} must have non-empty 2D shape [B >= 1, N >= 1], got {tuple(x.shape)}")
    if expected_shape is not None and (B, N) != expected_shape:
        raise ValueError(f"{name} shape {(B, N)} does not match expected shape {expected_shape}")
    return B, N


def _validate_2d_mask(
    mask: torch.Tensor, expected_shape: tuple[int, int], name: str = "valid_mask"
) -> torch.Tensor:
    _validate_2d_tensor(mask, name, expected_shape=expected_shape)
    if mask.dtype == torch.bool:
        return mask
    if mask.is_complex():
        raise ValueError(f"{name} cannot be a complex tensor")
    if mask.is_floating_point() and not bool(torch.isfinite(mask).all()):
        raise ValueError(f"{name} contains non-finite values (NaN/Inf)")
    valid_vals = (mask == 0) | (mask == 1)
    if not bool(valid_vals.all()):
        raise ValueError(f"{name} must be bool or contain only {{0, 1}} values")
    return mask.bool()


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
    if (
        isinstance(tol, bool)
        or not isinstance(tol, (int, float))
        or not math.isfinite(float(tol))
        or float(tol) < 0.0
    ):
        raise ValueError(f"tol must be a finite non-negative float, got {tol!r}")
    B, N = _validate_2d_tensor(y, "y")
    req_mask = _validate_2d_mask(valid_mask, (B, N), "valid_mask")
    y_f = y.float()
    mask_f = req_mask & torch.isfinite(y_f)
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
    if (
        isinstance(l2_reg, bool)
        or not isinstance(l2_reg, (int, float))
        or not math.isfinite(float(l2_reg))
        or float(l2_reg) <= 0.0
    ):
        raise ValueError(f"l2_reg must be a finite positive float, got {l2_reg!r}")
    if isinstance(n_iters, bool) or not isinstance(n_iters, int) or n_iters < 0:
        raise ValueError(f"n_iters must be a non-negative int, got {n_iters!r}")
    B, N = _validate_2d_tensor(raw_score_tr, "raw_score_tr")
    _validate_2d_tensor(y_bin_tr, "y_bin_tr", expected_shape=(B, N))
    req_mask = _validate_2d_mask(valid_mask_tr, (B, N), "valid_mask_tr")
    mask_b = req_mask & torch.isfinite(raw_score_tr) & torch.isfinite(y_bin_tr)
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


