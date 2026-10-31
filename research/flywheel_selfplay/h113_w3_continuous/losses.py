"""Opt-in CPU loss primitives, not a trainer or evidence-admission validator."""

from __future__ import annotations

import math

STATES = {"completed_pass", "completed_gate_fail", "timeout", "error", "unknown"}
FIELDS = {"status", "fitness", "logratio", "gate", "revealed"}


def targets(rows, *, top_resolved=True):
    """Build detached target values from caller-admitted rows; never infer completion."""
    if not rows or any(not r or len(r) != len(rows[0]) for r in rows):
        raise ValueError("nonempty rectangular slates required")
    output = []
    for slate in rows:
        records = []
        for row in slate:
            if set(row) != FIELDS or row["status"] not in STATES:
                raise ValueError(
                    "closed row fields and explicit execution state required"
                )
            if type(row["revealed"]) is not bool or row["gate"] not in (
                None,
                True,
                False,
            ):
                raise ValueError("explicit boolean masks required")
            if row["gate"] is not None and type(row["gate"]) is not bool:
                raise ValueError("gate must be boolean or null")
            for name in ("fitness", "logratio"):
                v = row[name]
                if v is not None and (
                    type(v) not in (int, float) or not math.isfinite(v)
                ):
                    raise ValueError(
                        "measured components must be finite numbers or null"
                    )
            completed = row["status"] in {"completed_pass", "completed_gate_fail"}
            if not completed and any(
                row[k] is not None for k in ("fitness", "logratio", "gate")
            ):
                raise ValueError(
                    "uncompleted rows must have null supervision components"
                )
            passed = row["status"] == "completed_pass"
            if completed and row["gate"] is not None and row["gate"] != passed:
                raise ValueError("recorded gate contradicts completion state")
            active = completed and not row["revealed"]
            gate_known = active and row["gate"] is not None
            quality_known = active and row["logratio"] is not None
            residual_known = (
                gate_known and passed and quality_known and row["fitness"] is not None
            )
            rank_known = gate_known and (not passed or row["fitness"] is not None)
            records.append(
                {
                    "rank_mask": rank_known,
                    "gate_mask": gate_known,
                    "quality_mask": quality_known,
                    "residual_mask": residual_known,
                    "gate": float(passed) if gate_known else 0.0,
                    "cost": (float(row["fitness"]) if passed else 2.0)
                    if rank_known
                    else 0.0,
                    "quality": float(row["logratio"]) if quality_known else 0.0,
                    "residual": float(row["fitness"] - 0.9 * row["logratio"])
                    if residual_known
                    else 0.0,
                    "regret": 0.0,
                }
            )
        passing = sorted(r["cost"] for r in records if r["rank_mask"] and r["gate"])
        if passing:
            center = passing[0]
            if top_resolved:
                scale = max(passing[min(16, len(passing)) - 1] - center, 0.05)
            else:
                mean = sum(passing) / len(passing)
                scale = max(
                    math.sqrt(sum((v - mean) ** 2 for v in passing) / len(passing)),
                    0.05,
                )
        else:
            center, scale = 0.0, 0.05
        if not math.isfinite(scale) or any(
            not math.isfinite(r[k])
            for r in records
            for k in ("cost", "quality", "residual")
        ):
            raise ValueError("derived target or scale overflow")
        for r in records:
            if r["rank_mask"]:
                r["regret"] = (
                    min(6.0, max(0.0, (r["cost"] - center) / scale))
                    if r["gate"]
                    else 8.0
                )
        output.append({"rows": records, "scale": scale, "passing_count": len(passing)})
    return output


def _tensor(value, shape=None):
    import torch

    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or not value.is_floating_point()
    ):
        raise ValueError("CPU floating tensors required")
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
