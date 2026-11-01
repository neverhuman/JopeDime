"""Data-free H197 per-context gradient clipping primitives."""

import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def per_context_losses(prediction, fitness, gate):
    """H146 live loss before its context mean; labels/targets are unchanged."""
    if prediction.shape != fitness.shape or prediction.shape != gate.shape:
        raise ValueError("prediction/fitness/gate shape mismatch")
    target = torch.where(gate, fitness.clamp(-4.0, 2.0), torch.full_like(fitness, 2.0))
    return F.smooth_l1_loss(prediction, target, reduction="none").mean(
        1
    ) + 0.1 * F.kl_div(
        F.log_softmax(-prediction / 0.2, dim=-1),
        F.softmax(-target / 0.2, dim=-1),
        reduction="none",
    ).sum(1)


def gradient_norm(gradients):
    present = [g.detach().reshape(-1).double() for g in gradients if g is not None]
    if not present:
        return 0.0
    norm = torch.linalg.vector_norm(torch.cat(present))
    value = float(norm.cpu())
    if not np.isfinite(value):
        raise ValueError("non-finite per-context gradient norm")
    return value


def context_gradients(losses, parameters):
    """Compute separate gradients for each context from the common forward graph."""
    if losses.ndim != 1 or losses.numel() < 1:
        raise ValueError("one scalar loss per context required")
    rows, norms = [], []
    for i in range(losses.numel()):
        grads = torch.autograd.grad(
            losses[i],
            parameters,
            retain_graph=i + 1 < losses.numel(),
            allow_unused=True,
        )
        rows.append(grads)
        norms.append(gradient_norm(grads))
    return rows, np.asarray(norms, dtype=np.float64)


def clip_factors(norms, threshold):
    norms = np.asarray(norms, dtype=np.float64)
    if (
        norms.ndim != 1
        or not np.isfinite(norms).all()
        or not np.isfinite(threshold)
        or threshold <= 0
    ):
        raise ValueError("finite positive threshold and finite norm vector required")
    return np.minimum(1.0, threshold / np.maximum(norms, np.finfo(np.float64).tiny))


def aggregate_context_gradients(rows, factors):
    factors = np.asarray(factors, dtype=np.float64)
    if not rows or factors.shape != (len(rows),) or not np.isfinite(factors).all():
        raise ValueError("one finite clip factor per context required")
    if np.any((factors < 0) | (factors > 1)):
        raise ValueError("clip factors must be in [0,1]")
    count = len(rows)
    aggregate = []
    for column in zip(*rows, strict=True):
        present = [
            (g, float(factors[i])) for i, g in enumerate(column) if g is not None
        ]
        if not present:
            aggregate.append(None)
            continue
        value = torch.zeros_like(present[0][0])
        for grad, factor in present:
            value.add_(grad, alpha=factor / count)
        aggregate.append(value)
    return tuple(aggregate)


def set_gradients(parameters, gradients):
    if len(parameters) != len(gradients):
        raise ValueError("parameter/gradient inventory mismatch")
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.grad = None if gradient is None else gradient.detach().clone()


def shuffle_factors(factors, seed):
    factors = np.asarray(factors, dtype=np.float64)
    if factors.ndim != 1 or len(factors) < 2:
        raise ValueError(
            "within-batch factor permutation requires at least two contexts"
        )
    permutation = np.random.default_rng(seed).permutation(len(factors))
    return factors[permutation], permutation


def dry_thresholds(step0_norms, dry_pcgc_aggregate_norms):
    step0_norms = np.asarray(step0_norms, dtype=np.float64)
    dry_pcgc_aggregate_norms = np.asarray(dry_pcgc_aggregate_norms, dtype=np.float64)
    if (
        step0_norms.ndim != 1
        or dry_pcgc_aggregate_norms.ndim != 1
        or not len(step0_norms)
        or not len(dry_pcgc_aggregate_norms)
        or not np.isfinite(step0_norms).all()
        or not np.isfinite(dry_pcgc_aggregate_norms).all()
    ):
        raise ValueError("H197 dry FIT norms must be finite non-empty vectors")
    c = float(np.median(step0_norms))
    global_c = float(np.median(dry_pcgc_aggregate_norms))
    if c <= 0 or global_c <= 0:
        raise ValueError("zero clipping threshold is not admissible")
    return c, global_c


def validate_b_lock(path, expected_panels):
    """Require a fixed-step, on-disk B lock with every referenced panel intact."""
    path = Path(path)
    if not path.is_file():
        raise ValueError("A scoring requires an on-disk B lock")
    lock = json.loads(path.read_text())
    panels = lock.get("b_panel_files", {})
    if (
        lock.get("wave") != "H197"
        or lock.get("candidate_step") != 40
        or lock.get("policy") != "fixed candidate step 40; no B-based step selection"
        or lock.get("A_scoring_authorized_after_lock") is not True
        or set(panels) != set(expected_panels)
    ):
        raise ValueError("B lock does not match the frozen H197 policy/panels")
    for name, digest in panels.items():
        target = path.parent / name
        if not target.is_file():
            raise ValueError("B lock panel is absent: " + name)
        h = hashlib.sha256(target.read_bytes()).hexdigest()
        if h != digest:
            raise ValueError("B lock panel hash mismatch: " + name)
    return hashlib.sha256(path.read_bytes()).hexdigest()
