"""FIT-only within-family mixup. Dataset labels and source tensors remain immutable."""

import numpy as np
import torch


def plan(cids, groups, family_by_cid, seed):
    rng = np.random.default_rng(seed)
    partners, weights = [], []
    for cid in cids:
        family = family_by_cid[cid]
        options = [other for other in groups[family] if other != cid]
        if options and rng.random() < 0.5:
            partners.append(options[int(rng.integers(len(options)))])
            weights.append(float(rng.beta(0.4, 0.4)))
        else:
            partners.append(cid)
            weights.append(1.0)
    return partners, np.asarray(weights, dtype=np.float64)


def blend(a, b, weights):
    if a.shape != b.shape or a.shape[0] != len(weights):
        raise ValueError("paired tensor shapes required")
    if not a.is_floating_point():
        a, b = a.float(), b.float()
    w = torch.as_tensor(weights, dtype=a.dtype, device=a.device)
    w = w.reshape((len(weights),) + (1,) * (a.ndim - 1))
    return w * a + (1 - w) * b


def cost_target(fitness, gate):
    return torch.where(gate, fitness.clamp(-4.0, 2.0), torch.full_like(fitness, 2.0))


def batch(left, right, left_anchor, right_anchor, weights):
    a, at, am, af, ag = left
    b, bt, bm, bf, bg = right
    if not torch.equal(a["revealed_indices"], b["revealed_indices"]):
        raise ValueError("paired reveal slots must match")
    # All continuous adapter inputs are mixed, including frozen anchor outputs.
    kwargs = {
        k: a[k].clone() if k == "revealed_indices" else blend(a[k], b[k], weights)
        for k in a
    }
    target = blend(cost_target(af, ag), cost_target(bf, bg), weights)
    return (
        kwargs,
        blend(at, bt, weights),
        blend(am, bm, weights),
        target,
        torch.ones_like(target, dtype=torch.bool),
        tuple(
            blend(x, y, weights) for x, y in zip(left_anchor, right_anchor, strict=True)
        ),
    )
