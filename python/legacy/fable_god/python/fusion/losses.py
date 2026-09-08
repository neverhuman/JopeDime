"""Fusion losses — proper-scoring-first + calibration + decayed distillation +
group-DRO robustness + gate-entropy anneal + tiny weight decay.

Rationale in docs/best_training_strategy.md: the gate is the new Goodhart surface,
so we add a CVaR tail over datasets and regularize the gate (entropy early, only
generalizable features). Regression uses Huber (robust point) + calibration; the
DRO tail is applied by the trainer over per-dataset losses.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def huber(pred, y, delta=0.05):
    return F.huber_loss(pred, y, delta=delta, reduction="mean")


def gate_entropy(w, eps=1e-9):
    return -(w.clamp_min(eps) * w.clamp_min(eps).log()).sum(-1).mean()


def regression_step_loss(out, y, best_expert_mean, lam):
    """Per-dataset scalar loss (the trainer aggregates with a CVaR tail).
    lam: dict(cal, distill, ent, sp) with current (annealed) coefficients."""
    final = out["final"]
    l_task = huber(final, y)
    # calibration: each expert's calibrated mean should itself be decent
    l_cal = torch.stack([huber(out["cal_means"][:, e], y)
                         for e in range(out["cal_means"].shape[1])]).mean()
    # distillation: don't fall below the best single expert (decayed anchor)
    l_dist = F.mse_loss(final, best_expert_mean.detach())
    l_ent = -gate_entropy(out["w"])            # MAXIMIZE entropy early -> negative sign
    total = l_task + lam["cal"] * l_cal + lam["distill"] * l_dist + lam["ent"] * l_ent
    return total, dict(task=float(l_task), cal=float(l_cal), dist=float(l_dist),
                       ent=float(gate_entropy(out["w"])))


def cvar_tail(per_dataset_losses: torch.Tensor, alpha=0.25):
    """Mean of the worst-alpha fraction of datasets (group-DRO)."""
    k = max(1, int(alpha * per_dataset_losses.numel()))
    worst = torch.topk(per_dataset_losses, k).values
    return worst.mean()
