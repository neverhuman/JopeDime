"""FIT-only invalid-margin hinge; the native live objective and inference stay intact."""

import numpy as np
import torch

LAMBDA = 0.1


def fit_margin(costs, gates, fit_ids):
    if set(costs) != set(fit_ids) or set(gates) != set(fit_ids):
        raise ValueError("margin closure must equal FIT only")
    gaps = {}
    for cid in fit_ids:
        v = np.asarray(costs[cid], np.float32)[np.asarray(gates[cid], bool)]
        if not np.isfinite(v).all():
            raise ValueError("nonfinite FIT prediction")
        if len(v) >= 2:
            pair = np.partition(v, 1)[:2]
            gaps[cid] = float(abs(float(pair[1]) - float(pair[0])))
    if not gaps:
        raise ValueError("no FIT contexts with two valid candidates")
    return float(np.median(list(gaps.values()))), gaps


def shuffled_groups(gate, rng):
    gate = np.asarray(gate, bool)
    pseudo = np.zeros_like(gate)
    comparison = np.zeros_like(gate)
    feasible = []
    for i, valid in enumerate(gate):
        count = int((~valid).sum())
        ids = np.flatnonzero(valid)
        ok = 0 < count < len(ids)
        feasible.append(ok)
        if ok:
            chosen = rng.choice(ids, count, replace=False)
            pseudo[i, chosen] = True
            comparison[i] = valid & ~pseudo[i]
    return pseudo, comparison, np.asarray(feasible)


def hinge(pred, valid, invalid, margin):
    if pred.shape != valid.shape or pred.shape != invalid.shape:
        raise ValueError("hinge group shape mismatch")
    if torch.any(valid & invalid) or not np.isfinite(margin) or margin < 0:
        raise ValueError("disjoint masks and fixed nonnegative margin required")
    active = valid.any(dim=1) & invalid.any(dim=1)
    lo_valid = pred.masked_fill(~valid, float("inf")).min(dim=1).values
    lo_invalid = pred.masked_fill(~invalid, float("inf")).min(dim=1).values
    # Mask before subtracting avoids inf-inf in degenerate contexts.
    gap = torch.where(active, lo_invalid, torch.zeros_like(lo_invalid)) - torch.where(
        active, lo_valid, torch.zeros_like(lo_valid)
    )
    per_context = torch.where(
        active, torch.relu(pred.new_tensor(margin) - gap), torch.zeros_like(gap)
    )
    return per_context.mean(), active


def loss(pred, fitness, gate, live_loss, arm, margin, rng):
    base = live_loss(pred, fitness, gate)
    if arm == "CTRL":
        return base, dict(hinge=0.0, active=0, infeasible=0, pseudo_indices=[])
    if arm == "IMH":
        invalid, valid = ~gate, gate
        info = dict(infeasible=0, pseudo_indices=[])
    elif arm == "IMH_SHUF":
        inv, val, feasible = shuffled_groups(gate.detach().numpy(), rng)
        invalid, valid = torch.from_numpy(inv), torch.from_numpy(val)
        info = dict(
            infeasible=int(((~gate).any(dim=1).numpy() & ~feasible).sum()),
            pseudo_indices=[np.flatnonzero(row).tolist() for row in inv],
        )
    else:
        raise ValueError("unknown H194 arm")
    h, active = hinge(pred, valid, invalid, margin)
    return base + LAMBDA * h, dict(
        **info, hinge=float(h.detach()), active=int(active.sum())
    )
