"""Opt-in CPU loss primitives, not a trainer or evidence-admission validator."""

from __future__ import annotations

import math
from pathlib import Path
from runpy import run_path

# H141 retains the byte-pinned canonical target builder used by both W3 waves.
targets = run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "h141_w3_retention"
        / "w3_losses.py"
    )
)["targets"]


def _tensor(value, shape=None):
    import torch

    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError("CPU float32 or float64 tensors required")
    if (
        value.ndim != 2
        or (shape is not None and value.shape != shape)
        or not torch.isfinite(value).all()
    ):
        raise ValueError("finite matching batch/slate tensor shapes required")
    return value


def compose_cost(prior_cost, residual_cost):
    """Keep the externally supplied W1 prior fixed; zero residual preserves choices."""
    _tensor(prior_cost)
    _tensor(residual_cost, prior_cost.shape)
    if prior_cost.dtype != residual_cost.dtype:
        raise ValueError("prior and residual dtypes must match")
    total = prior_cost.detach() + residual_cost
    return _tensor(total, prior_cost.shape)


def w3_loss(
    arm, prior_cost=None, residual_cost=None, rows=None, *, enabled=False, heads=None
):
    """Return None when disabled; enabled results never certify row provenance or a run."""
    if enabled is False:
        return None
    if enabled is not True or arm not in {"a", "b", "c", "d"}:
        raise ValueError("explicit enable and registered arm required")
    import torch
    from torch.nn import functional as F

    pred = compose_cost(prior_cost, residual_cost)
    prepared = targets(rows, top_resolved=arm != "b")
    if len(prepared) != pred.shape[0] or len(prepared[0]["rows"]) != pred.shape[1]:
        raise ValueError("row/tensor shape mismatch")
    if arm == "d":
        if not isinstance(heads, dict) or set(heads) != {"quality", "residual", "gate"}:
            raise ValueError(
                "d requires quality, passing-residual and gate-logit heads"
            )
        for head in heads.values():
            _tensor(head, pred.shape)
            if head.dtype != pred.dtype:
                raise ValueError("head dtype mismatch")
        terms = []
        counts = {}
        for name in ("quality", "residual", "gate"):
            mask = torch.tensor(
                [[r[name + "_mask"] for r in s["rows"]] for s in prepared]
            )
            target = torch.tensor(
                [[r[name] for r in s["rows"]] for s in prepared], dtype=pred.dtype
            )
            counts[name] = int(mask.sum())
            # Equal slate weight among slates with this measured component.
            per_slate = []
            for i in range(len(prepared)):
                if mask[i].any():
                    h, t = heads[name][i, mask[i]], target[i, mask[i]]
                    per_slate.append(
                        F.binary_cross_entropy_with_logits(h, t)
                        if name == "gate"
                        else F.smooth_l1_loss(h, t)
                    )
            if per_slate:
                terms.append(torch.stack(per_slate).mean())
        zero = sum(h.reshape(-1)[:0].sum() for h in heads.values())
        loss = sum(terms, zero)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite loss")
        return {
            "loss": loss,
            "component_counts": counts,
            "status": "LOSS_ONLY" if terms else "NO_SUPERVISION",
        }

    list_terms, gate_terms, pair_terms = [], [], []
    count = 0
    for i, slate in enumerate(prepared):
        ix = [j for j, r in enumerate(slate["rows"]) if r["rank_mask"]]
        if not ix:
            continue
        count += len(ix)
        p = pred[i, ix]
        rs = [slate["rows"][j] for j in ix]
        t = torch.tensor([r["cost"] for r in rs], dtype=pred.dtype)
        if arm == "a":
            list_terms.append(
                F.smooth_l1_loss(p, t)
                + 0.1
                * F.kl_div(
                    F.log_softmax(-p / 0.2, dim=0),
                    F.softmax(-t / 0.2, dim=0),
                    reduction="sum",
                )
            )
            continue
        regret = torch.tensor([r["regret"] for r in rs], dtype=pred.dtype)
        list_terms.append(
            0.7
            * F.kl_div(
                F.log_softmax(-p / 0.15, dim=0),
                F.softmax(-regret / 0.10, dim=0),
                reduction="sum",
            )
            + 0.3
            * F.kl_div(
                F.log_softmax(-p / 0.40, dim=0),
                F.softmax(-regret / 0.35, dim=0),
                reduction="sum",
            )
        )
        pi = [j for j, r in enumerate(rs) if r["gate"]]
        fi = [j for j, r in enumerate(rs) if not r["gate"]]
        if pi:
            pp, tt = p[pi], t[pi]
            order = torch.argsort(tt, stable=True)
            if fi:
                gate_terms.append(
                    F.relu(0.50 + pp[order[0]] - p[fi]).mean()
                    + 0.5 * F.relu(0.25 + pp[order[:8]].mean() - p[fi]).mean()
                )
            if len(pi) >= 8:
                top, bottom = order[: min(4, len(pi) // 2)], order[len(pi) // 2 :]
                strict = tt[top, None] < tt[bottom][None, :] - 1e-6
                if strict.any():
                    pair_terms.append(
                        F.relu(0.15 + pp[top, None] - pp[bottom][None, :])[
                            strict
                        ].mean()
                    )
    zero = pred.reshape(-1)[:0].sum()

    def average(terms):
        return torch.stack(terms).mean() if terms else zero

    loss = average(list_terms) + 0.50 * average(gate_terms) + 0.25 * average(pair_terms)
    if not torch.isfinite(loss):
        raise ValueError("nonfinite loss")
    return {
        "loss": loss,
        "component_counts": {"rank": count},
        "status": "LOSS_ONLY" if count else "NO_SUPERVISION",
    }
