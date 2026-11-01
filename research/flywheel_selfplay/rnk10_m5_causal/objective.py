"""Preregistered RNK-10 landmark-causality auxiliary objective."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

F12_SCALE = 0.4208
AUXILIARY_WEIGHT = 0.1
CAUSAL_MARGIN = 0.1 * F12_SCALE


def choose_training_pair(fitness: torch.Tensor, gate: torch.Tensor) -> tuple[int, int]:
    """Choose best/worst gated unrevealed candidates; stable source-order ties."""
    if fitness.ndim != 1 or gate.shape != fitness.shape or fitness.numel() <= 4:
        raise ValueError("expected aligned slate with at least one unrevealed candidate")
    eligible = torch.nonzero(gate[4:].bool(), as_tuple=False).flatten() + 4
    if eligible.numel() < 2:
        raise ValueError("at least two gated unrevealed candidates are required")
    values = fitness[eligible]
    if not torch.isfinite(values).all():
        raise ValueError("training fitness must be finite")
    order = torch.argsort(values, stable=True)
    return int(eligible[order[0]]), int(eligible[order[-1]])


def landmark_causality_hinge(
    baseline_costs: torch.Tensor,
    ablated_costs: torch.Tensor,
    best_indices: torch.Tensor,
    worst_indices: torch.Tensor,
    margin: float = CAUSAL_MARGIN,
) -> torch.Tensor:
    """Train the baseline good-vs-bad margin to exceed its ablated counterpart."""
    if baseline_costs.shape != ablated_costs.shape or baseline_costs.ndim != 2:
        raise ValueError("baseline and ablated cost batches must have matching [B, N] shape")
    if best_indices.shape != worst_indices.shape or best_indices.ndim != 1:
        raise ValueError("best/worst indices must be aligned vectors")
    if best_indices.numel() != baseline_costs.shape[0]:
        raise ValueError("one best/worst pair is required per slate")
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("causal margin must be positive and finite")
    if not torch.isfinite(baseline_costs).all() or not torch.isfinite(ablated_costs).all():
        raise ValueError("costs must be finite")

    batch = torch.arange(baseline_costs.shape[0], device=baseline_costs.device)
    base_margin = baseline_costs[batch, worst_indices] - baseline_costs[batch, best_indices]
    ablated_margin = ablated_costs[batch, worst_indices] - ablated_costs[batch, best_indices]
    return F.relu(margin - (base_margin - ablated_margin)).mean()


def training_loss(
    baseline_costs: torch.Tensor,
    ablated_costs: torch.Tensor,
    targets: torch.Tensor,
    best_indices: torch.Tensor,
    worst_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep the source regression/KL objective and add the single preregistered term."""
    if baseline_costs.shape != targets.shape:
        raise ValueError("regression targets must match baseline costs")
    regression = F.smooth_l1_loss(baseline_costs, targets) + 0.1 * F.kl_div(
        F.log_softmax(-baseline_costs / 0.2, dim=-1),
        F.softmax(-targets / 0.2, dim=-1),
        reduction="batchmean",
    )
    causal = landmark_causality_hinge(
        baseline_costs, ablated_costs, best_indices, worst_indices
    )
    return regression + AUXILIARY_WEIGHT * causal, regression, causal
