"""H227 outcome-free pool construction and paired family analysis."""

from __future__ import annotations
import hashlib
import numpy as np

SEEDS = (227101, 227102, 227103)
FOLD_SEEDS = (227141, 227142, 227143)
ARMS = ("NATIVE16", "NATIVE64", "FPS16of64")
CONTRASTS = ("NATIVE64", "FPS16of64")
BOOT_SEED = 227150
BOOT_DRAWS = 10_000
MIN_GAIN = 0.002


def stream_seed(seed, cid):
    if (
        seed not in SEEDS
        or len(cid) != 64
        or any(c not in "0123456789abcdef" for c in cid)
    ):
        raise ValueError("registered seed and external SHA256 context ID required")
    return (
        int.from_bytes(
            hashlib.sha256(f"H227:{seed}:{cid}".encode()).digest()[:8], "little"
        )
        & 0x7FFFFFFF
    )


def unique(parent, proposals, width):
    rows = [parent, *proposals]
    ids = [hashlib.sha256(p["dsl"].encode()).hexdigest() for p in rows]
    if (
        len(rows) != width
        or len(set(ids)) != width
        or ids != [p["program_sha256"] for p in rows]
    ):
        raise ValueError("exact unique width and DSL identity required; no refill")
    return rows


def fps(rows, vectors, width=16):
    x = np.asarray([vectors[r["program_sha256"]] for r in rows], np.float64)
    if x.shape != (64, 16) or not np.isfinite(x).all():
        raise ValueError("64 finite native proj16 vectors required")
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.where(norm > 0, norm, 1)
    keep = [0]
    distance = np.sum((x - x[0]) ** 2, axis=1)
    while len(keep) < width:
        distance[keep] = -np.inf
        i = int(np.argmax(distance))
        keep.append(i)
        distance = np.minimum(distance, np.sum((x - x[i]) ** 2, axis=1))
    return [rows[i] for i in keep]


def build_pools(parent, proposer, arguments, featurizer):
    small = unique(parent, proposer(**arguments, n_proposals=15), 16)
    large = unique(parent, proposer(**arguments, n_proposals=63), 64)
    vectors = {p["program_sha256"]: featurizer(p["dsl"])[1] for p in large}
    return dict(NATIVE16=small, NATIVE64=large, FPS16of64=fps(large, vectors))


def score(cost, results):
    c = np.asarray(cost, np.float64)
    f = np.asarray([r["fitness"] for r in results], np.float64)
    g = np.asarray([r["cv_gate_passed"] for r in results], bool)
    if (
        c.ndim != 1
        or not c.size
        or c.shape != f.shape
        or not np.isfinite(c).all()
        or not np.isfinite(f).all()
    ):
        raise ValueError("finite complete candidate cost/fitness panel required")
    i = int(np.argmin(c))
    utilities = np.where(g, -f, -2.0)
    return dict(
        utility=float(utilities[i]),
        selected_index=i,
        oracle_utility=float(utilities.max()),
    )


def holm(p):
    ordered = sorted(p, key=lambda k: (p[k], k))
    out, running = {}, 0.0
    for i, key in enumerate(ordered):
        if not np.isfinite(p[key]) or not 0 <= p[key] <= 1:
            raise ValueError("finite probabilities required")
        running = max(running, min(1.0, (len(p) - i) * p[key]))
        out[key] = running
    return out


def gain_stats(values, families):
    values = np.asarray(values, np.float64)
    labels = np.asarray(families)
    order = sorted(set(families))
    if (
        values.shape != (3, 91)
        or len(order) != 19
        or labels.shape != (91,)
        or not np.isfinite(values).all()
    ):
        raise ValueError("complete three-seed DL-Val91/19-family panel required")
    means = np.stack([values[:, labels == f].mean(axis=1) for f in order], axis=1)
    ix = np.random.default_rng(BOOT_SEED).integers(0, 19, (BOOT_DRAWS, 19))
    draws = np.median(means[:, ix].mean(axis=2), axis=0)
    return dict(
        seed_context_mean_gain=values.mean(axis=1).tolist(),
        plateau_median_gain=float(np.median(values.mean(axis=1))),
        seed_equal_family_mean_gain=means.mean(axis=1).tolist(),
        median_gain=float(np.median(means.mean(axis=1))),
        ci95=np.quantile(draws, [0.025, 0.975]).tolist(),
        p_one_sided=float((1 + np.sum(draws <= 0)) / (BOOT_DRAWS + 1)),
        family_harm_fraction_by_seed=(means < 0).mean(axis=1).tolist(),
    )


def decide(stats):
    if set(stats) != set(CONTRASTS):
        return dict(verdict="INCOMPLETE", accepted_arms=[], holm_p=None)
    for s in stats.values():
        if (
            not np.isfinite(
                [
                    s["plateau_median_gain"],
                    s["median_gain"],
                    *s["ci95"],
                    s["p_one_sided"],
                ]
            ).all()
            or s["ci95"][0] > s["ci95"][1]
        ):
            raise ValueError("finite ordered statistics required")
    adjusted = holm({a: stats[a]["p_one_sided"] for a in CONTRASTS})
    accepted = [
        a
        for a in CONTRASTS
        if stats[a]["plateau_median_gain"] > MIN_GAIN
        and stats[a]["ci95"][0] > 0
        and adjusted[a] < 0.05
    ]
    return dict(
        verdict="ADVANCE" if "NATIVE64" in accepted else "NO_ADVANCE",
        accepted_arms=accepted,
        secondary_advance="FPS16of64" in accepted,
        holm_p=adjusted,
        holm_m=2,
    )
