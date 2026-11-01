"""Invalid share through the unchanged H188 kernel; hard veto over unchanged native costs."""

import numpy as np
import torch
from parents import h188

LABELS = ("INF", "Q90", "Q75", "Q50")
QUANTILES = (0.90, 0.75, 0.50)


def revealed_invalid(fitness, gate):
    """Original float64 utility exactly -2, including a passing fitness of +2."""
    return (
        np.where(np.asarray(gate, bool), -np.asarray(fitness, np.float64), -2.0) == -2.0
    )


def invalid_share(ad, static, scale, rev, raw, gate, invalid):
    if invalid.shape != rev.shape:
        raise ValueError("revealed indicator shape")
    args = (
        ad,
        static["sem24"],
        static["cand_q_norm"],
        static["eff_prior"],
        scale,
        rev,
        raw,
        gate,
    )
    numerator = h188.kernel_term(
        *args, residual_transform=lambda residual: invalid.to(residual.dtype)
    )
    denominator = h188.kernel_term(*args, residual_transform=torch.ones_like)
    if not torch.isfinite(denominator).all() or not (denominator > 0).all():
        raise RuntimeError("nonpositive kernel normalization")
    probability = (numerator / denominator).clamp(0.0, 1.0)
    if not torch.isfinite(probability).all():
        raise RuntimeError("nonfinite invalid share")
    return probability


def theta_grid(fit_probabilities):
    values = np.concatenate([np.asarray(p, np.float64) for p in fit_probabilities])
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("finite probabilities in [0,1] required")
    positive = values[values > 0]
    return [None] + (
        np.quantile(positive, QUANTILES, method="linear").tolist()
        if len(positive)
        else [None] * 3
    )


def veto_pick(cost, probability, theta):
    cost, probability = np.asarray(cost), np.asarray(probability)
    if (
        cost.ndim != 1
        or cost.size == 0
        or cost.shape != probability.shape
        or not np.isfinite(cost).all()
        or not np.isfinite(probability).all()
    ):
        raise ValueError("matching finite vectors required")
    if theta is not None and (not np.isfinite(theta) or not 0 <= theta <= 1):
        raise ValueError("theta must be null(infinity) or [0,1]")
    veto = probability > (float("inf") if theta is None else theta)
    fallback = bool(veto.all())
    pick = (
        int(np.argmin(cost))
        if fallback
        else int(np.argmin(np.where(veto, np.inf, cost)))
    )
    return pick, dict(
        vetoed=int(veto.sum()), candidates=cost.size, fallback=int(fallback), contexts=1
    )


def median_theta(values):
    return sorted(values, key=lambda x: float("inf") if x is None else x)[
        len(values) // 2
    ]
