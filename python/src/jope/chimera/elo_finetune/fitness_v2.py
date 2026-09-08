"""Fitness v2: bounded-margin, Elo-impact-weighted, group-CVaR fitness on
never-repeating episodes vs a live zoo. See docs/fitness_v2.md.

Per episode e, opponent o:
    m = (rmse_o − rmse_θ)/s_e ; s_e = median opponent rmse
    m̃ = M·tanh(m/M)                      (bounded margin, M=2)
    p = σ(m̃/τ) ; w = p(1−p)              (Elo-impact weight)
    ℓ = w·(m̃ − m*)²                      (IPO-form target margin, m*=0.5)
    L_e = Σℓ/Σw
    L = (1−λ)·mean_bucket(L̄_b) + λ·worst_bucket   (group CVaR, α=0.25 of 4)
    F = −L − λ_sp·||θ−θ₀||²/n_params

Improving F on fresh episodes cannot be done by memorization (episodes never
repeat) — only by genuine in-context skill against the zoo.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from .episodes import Episode
from .eval_protocol import ProtocolRunner

M_BOUND = 2.0
M_STAR = 0.5
TAU = 1.0
LAM_DRO = 0.25
DORO_TRIM = 0.03
LAM_SP = 3e-3


@dataclass
class FitnessReport:
    fitness: float
    loss_task: float
    mean_margin: float
    mean_winrate: float
    sp_term: float
    n_episodes: int
    per_bucket_loss: dict[str, float]
    per_episode_margin: list[float]  # CRN-pairable across checkpoints


def model_rmse(runner: ProtocolRunner, ep: Episode) -> float:
    """Model rmse on the episode (identity view, native NaN handling)."""
    pred = runner._mean_single(ep.ctx_x, ep.ctx_y, ep.qry_x)
    err = pred.astype(np.float64) - ep.qry_y.astype(np.float64)
    return float(np.sqrt(np.mean(err * err)))


def episode_scores(rmse_model: float, ep: Episode) -> tuple[float, float, float]:
    """Returns (episode loss L_e, mean bounded margin, winrate vs zoo)."""
    opp = np.array([v for v in ep.zoo_rmse.values() if np.isfinite(v)])
    s_e = float(np.median(opp))
    if not math.isfinite(rmse_model) or s_e <= 1e-12:
        return float("nan"), float("nan"), float("nan")
    m = (opp - rmse_model) / s_e
    m_t = M_BOUND * np.tanh(m / M_BOUND)
    p = 1.0 / (1.0 + np.exp(-m_t / TAU))
    w = p * (1.0 - p)
    if w.sum() <= 1e-9:
        loss = 0.0  # fully saturated pairs carry no Elo impact either way
    else:
        loss = float((w * (m_t - M_STAR) ** 2).sum() / w.sum())
    return loss, float(m_t.mean()), float((m > 0).mean())


def evaluate(runner: ProtocolRunner, episodes: list[Episode],
             theta_sp_sq: float = 0.0) -> FitnessReport:
    per_bucket: dict[str, list[float]] = defaultdict(list)
    margins, winrates, ep_losses = [], [], []
    for ep in episodes:
        rmse = model_rmse(runner, ep)
        loss, margin, wr = episode_scores(rmse, ep)
        if not math.isfinite(loss):
            continue
        per_bucket[ep.bucket].append(loss)
        ep_losses.append(loss)
        margins.append(margin)
        winrates.append(wr)

    # DORO trim: drop the top-ε episode losses (broken/unlearnable episodes)
    if ep_losses:
        cut = np.quantile(ep_losses, 1.0 - DORO_TRIM)
        per_bucket = {
            b: [v for v in vals if v <= cut] for b, vals in per_bucket.items()
        }
    bucket_means = {b: float(np.mean(v)) for b, v in per_bucket.items() if v}
    if bucket_means:
        means = np.array(list(bucket_means.values()))
        loss_task = float((1 - LAM_DRO) * means.mean() + LAM_DRO * means.max())
    else:
        loss_task = float("nan")

    sp = LAM_SP * theta_sp_sq
    return FitnessReport(
        fitness=-loss_task - sp,
        loss_task=loss_task,
        mean_margin=float(np.mean(margins)) if margins else float("nan"),
        mean_winrate=float(np.mean(winrates)) if winrates else float("nan"),
        sp_term=sp,
        n_episodes=len(ep_losses),
        per_bucket_loss=bucket_means,
        per_episode_margin=margins,
    )


def sp_distance_sq(model, base_model) -> float:
    """||θ−θ₀||² / n_params (per-parameter mean, so λ_sp is scale-free)."""
    total, count = 0.0, 0
    with torch.no_grad():
        for p, p0 in zip(model.parameters(), base_model.parameters()):
            total += float(((p - p0) ** 2).sum())
            count += p.numel()
    return total / max(count, 1)
