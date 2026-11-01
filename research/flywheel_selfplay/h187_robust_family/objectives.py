"""H180 per-context live loss, frozen-anchor normalization and Huber influence bound."""

import torch
import torch.nn.functional as F

SCALE_FLOOR = 1e-3
HUBER_DELTA = 2.0


def per_context_loss(pred, fitness, gate):
    target = torch.where(gate, fitness.clamp(-4.0, 2.0), torch.full_like(fitness, 2.0))
    return F.smooth_l1_loss(pred, target, reduction="none").mean(1) + 0.1 * F.kl_div(
        F.log_softmax(-pred / 0.2, dim=-1),
        F.softmax(-target / 0.2, dim=-1),
        reduction="none",
    ).sum(1)


def huber(z):
    a = z.abs()
    return torch.where(
        a <= HUBER_DELTA, 0.5 * z * z, HUBER_DELTA * (a - 0.5 * HUBER_DELTA)
    )


def normalized_loss(pred, anchor_pred, fitness, gate):
    scale = (
        per_context_loss(anchor_pred.detach(), fitness, gate)
        .detach()
        .clamp_min(SCALE_FLOOR)
    )
    z = per_context_loss(pred, fitness, gate) / scale
    return huber(z).mean(), dict(
        anchor_loss_scale=scale,
        standardized_loss=z,
        influence=z.clamp(-HUBER_DELTA, HUBER_DELTA),
    )


def paired_initial_states(initial, arms):
    return {
        arm: {name: value.detach().clone() for name, value in initial.items()}
        for arm in arms
    }
