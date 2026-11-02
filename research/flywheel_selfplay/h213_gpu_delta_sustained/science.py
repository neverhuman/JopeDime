"""H213 fixed GPU head, role-safe selection and paired-family inference."""

import hashlib
import math
import numpy as np
import torch
from torch import nn

SEEDS = (213101, 213102, 213103)
THRESHOLDS = (0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, float("inf"))
ARMS = ("R882", "DELTA_RAW", "DELTA_SAFE", "SHUF_SAFE")


def stream(*parts):
    return int.from_bytes(
        hashlib.sha256(":".join(map(str, ("H213", *parts))).encode()).digest()[:8],
        "little",
    )


def partition(rows, seed, outer):
    families = sorted({r["family_id"] for r in rows})
    if len(families) != 205 or seed not in SEEDS or outer not in range(5):
        raise ValueError("unregistered family partition")
    groups = np.array_split(np.random.default_rng(seed + 40).permutation(families), 5)
    a, b = set(groups[outer]), set(groups[(outer + 1) % 5])
    return tuple(
        [
            r
            for r in rows
            if (
                r["family_id"] in a
                if role == "A"
                else r["family_id"] in b
                if role == "B"
                else r["family_id"] not in a | b
            )
        ]
        for role in ("FIT", "B", "A")
    )


class DeltaHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(744, 64), nn.GELU(), nn.Linear(64, 1, bias=False)
        )
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, x, reference):
        raw = self.net(x).squeeze(-1)
        return raw - raw[reference]


def features(program, tokens, mask, native):
    p, t, m, c = map(np.asarray, (program, tokens, mask, native))
    if (
        p.shape != (len(c), 412)
        or t.shape != (10, 32)
        or m.shape != (10,)
        or not all(np.isfinite(v).all() for v in (p, t, c))
    ):
        raise ValueError("finite pinned feature shapes required")
    ref = int(c.argmin())
    context = np.r_[(t * m[:, None]).ravel(), m.astype(np.float32)]
    return np.concatenate(
        (
            np.arcsinh(p),
            np.broadcast_to(context, (len(c), 330)),
            np.clip(c, -10, 10)[:, None],
            np.clip(c - c[ref], -10, 10)[:, None],
        ),
        axis=1,
    ).astype(np.float32)


def balanced_loss(pred, utility, reference):
    target = (utility - utility[reference]).clamp(-4, 4)
    loss = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
    positive = target > 0
    if positive.any() and (~positive).any():
        return 0.5 * loss[positive].mean() + 0.5 * loss[~positive].mean()
    return loss.mean()


def pick(delta, reference, threshold=0.0):
    values = np.asarray(delta)
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or threshold < 0
        or np.isnan(threshold)
    ):
        raise ValueError("finite improvement scores and valid threshold required")
    candidate = int(values.argmax())
    return candidate if values[candidate] > threshold else reference


def family_mean(values, families):
    values = np.asarray(values, np.float64)
    return np.stack(
        [
            values[..., np.asarray(families) == f].mean(-1)
            for f in sorted(set(families))
        ],
        axis=-1,
    )


def select_threshold(gains, families):
    """B-only precomputed candidate/reference deltas by threshold; infinity fallback."""
    if set(gains) != set(THRESHOLDS):
        raise ValueError("complete frozen threshold grid required")
    reports = {}
    for threshold, values in gains.items():
        if not np.isfinite(values).all():
            raise ValueError("finite selection outcomes required")
        fm = family_mean(values, families)
        reports[threshold] = (
            math.fsum(fm.tolist()) / len(fm),
            float((fm < -0.01).mean()),
        )
    eligible = [
        t for t in THRESHOLDS if reports[t][0] > 0.001 and reports[t][1] <= 0.05
    ]
    chosen = (
        max(eligible, key=lambda t: (reports[t][0], t)) if eligible else float("inf")
    )
    return chosen, {
        str(t): {"mean_gain": g, "harm": h} for t, (g, h) in reports.items()
    }


def contrast(panel, base, families):
    delta = family_mean(np.asarray(panel) - base, families)
    draw = np.random.default_rng(213150).integers(
        0, delta.shape[1], (10000, delta.shape[1])
    )
    boot = np.median(delta[:, draw].mean(-1), axis=0)
    means = delta.mean(-1)
    return {
        "per_seed_gain": means.tolist(),
        "median_gain": float(np.median(means)),
        "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "p": float((1 + (boot <= 0).sum()) / 10001),
        "harm": float((delta.mean(0) < -0.01).mean()),
        "positive_seeds": int((means > 0).sum()),
    }


def verdict(primary):
    return (
        "CANDIDATE"
        if primary["ci95"][0] > 0 and primary["p"] < 0.05 and primary["harm"] <= 0.1
        else "NO_ADVANCE"
    )


def require_native_reference(observed, expected):
    actual, reference = (
        np.asarray(observed, np.float64),
        np.asarray(expected, np.float64),
    )
    if (
        actual.shape != (91,)
        or reference.shape != (91,)
        or not np.array_equal(actual, reference)
        or float(actual.mean()) != 1.1377941898164525
    ):
        raise RuntimeError("native91 exact utility mismatch")
