"""Current-top5 inversion logistic loss, exact utilities and adapter gradient mask."""

import numpy as np
import torch
import torch.nn.functional as F


def adapter_mask(net):
    names = []
    for name, param in net.named_parameters():
        train = name.startswith("adapter_10tok.") and not name.startswith(
            "adapter_10tok.anchor."
        )
        param.requires_grad_(train)
        if not train:
            param.grad = None
        else:
            names.append(name)
    return tuple(names)


def assert_grad_mask(net, names):
    names = set(names)
    for name, param in net.named_parameters():
        if param.requires_grad != (name in names):
            raise RuntimeError("parameter gradient mask changed: " + name)
        if name not in names and param.grad is not None:
            raise RuntimeError("non-adapter received a gradient: " + name)


def shuffled_utilities(utility, seed):
    utility = np.asarray(utility, dtype=np.float64)
    return utility[np.random.default_rng(seed).permutation(len(utility))].copy()


def hard_negative(prediction, utility):
    """Equal-weight inverted ordered pairs within stable current top5; no margin tuning."""
    if (
        prediction.ndim != 1
        or len(prediction) < 5
        or not torch.isfinite(prediction).all()
    ):
        raise ValueError(
            "finite full-pool cost vector with at least five candidates required"
        )
    utility = torch.as_tensor(utility, dtype=torch.float64, device=prediction.device)
    if utility.shape != prediction.shape or not torch.isfinite(utility).all():
        raise ValueError("finite true utilities must match candidate ordering")
    top = torch.argsort(prediction.detach(), stable=True)[:5]
    pairs = [
        (int(top[j]), int(top[i]))
        for i in range(5)
        for j in range(i + 1, 5)
        if utility[top[j]] > utility[top[i]]
    ]
    if pairs:
        better, worse = zip(*pairs, strict=True)
        loss = F.softplus(prediction[list(better)] - prediction[list(worse)]).mean()
    else:
        loss = prediction.sum() * 0
    return loss, dict(
        top5=top.tolist(),
        pairs=[list(p) for p in pairs],
        pair_count=len(pairs),
        top5_costs=prediction.detach()[top].tolist(),
    )
