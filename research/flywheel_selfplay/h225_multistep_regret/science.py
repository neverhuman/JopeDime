"""Prospective H225 partitions, proposal streams, selection and native readouts."""

import hashlib
import itertools
import math
import random

import numpy as np

SEEDS = (225101, 225102, 225103)
FOLD_SEEDS = dict(zip(SEEDS, (225141, 225142, 225143), strict=True))
ARMS = ("MSCR", "MSTOP1", "MSCR_SHUF")
TEMPERATURE = 0.2
DRAWS = 10_000
BASELINE = 1.1377941898164525
CONFIGS = tuple(
    dict(name=f"m{steps}-lr{lr:g}", steps=steps, lr_scale=lr, replica=0)
    for steps in (8, 32)
    for lr in (0.5, 1.0, 2.0)
) + tuple(
    dict(name=f"m32-lr1-rep{replica}", steps=32, lr_scale=1.0, replica=replica)
    for replica in (1, 2)
)


def stream_seed(*parts):
    return (
        int.from_bytes(
            hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little"
        )
        & 0x7FFFFFFF
    )


def folds(families, seed):
    families = sorted(set(families))
    if len(families) != 205 or seed not in SEEDS:
        raise ValueError("registered seed and 205 TRAIN families required")
    return [
        sorted(x.tolist())
        for x in np.array_split(
            np.random.default_rng(FOLD_SEEDS[seed]).permutation(families), 5
        )
    ]


def cell(k):
    if k not in range(5):
        raise ValueError("five outer folds required")
    a, b = k, (k + 1) % 5
    return a, b, tuple(j for j in range(5) if j not in (a, b))


def exclusions():
    outer = sorted({tuple(sorted(cell(k)[:2])) for k in range(5)})
    inner = sorted(
        {tuple(sorted((*cell(k)[:2], j))) for k in range(5) for j in cell(k)[2]}
    )
    return outer, inner


def partition_jaccard(a, b):
    return float(
        np.mean(
            [max(len(set(x) & set(y)) / len(set(x) | set(y)) for y in b) for x in a]
        )
    )


def batch_plan(rows, orders, seed, steps):
    """Uniform family, then uniform context; all configurations share a prefix."""
    if steps not in (2, 8, 32) or not rows:
        raise ValueError("registered nonempty fit required")
    if any(r["dl_split"] != "DL-Train" for r in rows):
        raise ValueError("TRAIN-only optimizer closure")
    by_family = {}
    for row in rows:
        by_family.setdefault(row["family_id"], []).append(row["context_sha256"])
    families = sorted(by_family)
    for ids in by_family.values():
        ids.sort()
    result = []
    for step in range(steps):
        stream = stream_seed("H225-batch", seed, step)
        rng = np.random.default_rng(stream)
        py_rng = random.Random(stream)
        cids = [py_rng.choice(by_family[py_rng.choice(families)]) for _ in range(16)]
        indices = []
        for cid in cids:
            order = orders[cid]
            top = min(48, len(order))
            rest = order[top:] if len(order) > top else order
            indices.append(
                np.concatenate(
                    [
                        order[:top],
                        rng.choice(rest, 192 - top, replace=len(rest) < 192 - top),
                    ]
                )
            )
        result.append((stream, cids, indices))
    return result


def gaps(fitness, gate):
    utility = np.where(np.asarray(gate, bool), -np.asarray(fitness, float), -2.0)
    if utility.ndim != 1 or not len(utility) or not np.isfinite(utility).all():
        raise ValueError("finite complete utility labels required")
    return utility.max() - utility


def regret(cost, labels):
    cost, labels = np.asarray(cost, float), np.asarray(labels, float)
    if cost.shape != labels.shape or cost.ndim != 1 or not np.isfinite(cost).all():
        raise ValueError("matching complete costs and gap labels required")
    if not np.isfinite(labels).all() or np.any(labels < 0):
        raise ValueError("finite nonnegative gap labels required")
    weights = np.exp(-(cost - cost.min()) / TEMPERATURE)
    return float(np.dot(weights / weights.sum(), labels))


def shuffled_gaps(rows, labels, seed):
    """Permute candidate-gap labels only inside each FIT family, not model costs."""
    groups = {}
    for r in rows:
        groups.setdefault(r["family_id"], []).append(r["context_sha256"])
    result = {}
    for family, ids in sorted(groups.items()):
        ids.sort()
        values = np.concatenate([labels[c] for c in ids])
        values = np.random.default_rng(stream_seed(seed, family)).permutation(values)
        offset = 0
        for cid in ids:
            length = len(labels[cid])
            result[cid] = values[offset : offset + length]
            offset += length
    return result


def family_matrix(values, families):
    values, labels = np.asarray(values, float), np.asarray(families)
    if values.shape[-1] != len(labels) or not np.isfinite(values).all():
        raise ValueError("complete finite aligned family panel required")
    return np.stack(
        [values[..., labels == f].mean(-1) for f in sorted(set(labels))], -1
    )


def require_score_role(families, fit_families, role, locked, forbidden=()):
    if role not in ("FIT", "A", "DL-Val"):
        raise ValueError("explicit score role required")
    if set(families) & set(fit_families):
        raise ValueError("fitted families cannot supply OOF/A/Val scores")
    if role == "FIT" and set(families) & set(forbidden):
        raise ValueError("A/B family in FIT selection")
    if role != "FIT" and not locked:
        raise ValueError("candidate A/Val scoring requires immutable selection lock")


def select(arm, panels, reference, families, seed):
    """Best mean score; identical strict paired-family LCB policy for all arms."""
    names = [c["name"] for c in CONFIGS]
    if arm not in ARMS or set(panels) != set(names):
        raise ValueError("exact registered arm and complete eight proposals required")
    improvements = {
        name: family_matrix(
            np.asarray(value) - reference
            if arm == "MSTOP1"
            else reference - np.asarray(value),
            families,
        )
        for name, value in panels.items()
    }
    if any(value.ndim != 1 for value in improvements.values()):
        raise ValueError("one OOF vector per proposal required")
    n = len(next(iter(improvements.values())))
    draws = np.random.default_rng(seed).integers(0, n, (DRAWS, n))
    inventory = {
        name: dict(
            mean_improvement=float(values.mean()),
            lcb95=float(np.quantile(values[draws].mean(-1), 0.025)),
        )
        for name, values in improvements.items()
    }
    best = max(names, key=lambda name: inventory[name]["mean_improvement"])
    chosen = best if inventory[best]["lcb95"] > 0 else None
    return dict(chosen=chosen, best=best, scores=inventory, fallback=chosen is None)


def contrast(panel, reference, families, seed, *, plateau=False):
    delta = np.asarray(panel, float) - np.asarray(reference, float)
    expected = (3, 91 if plateau else 1219)
    if delta.shape != expected or not np.isfinite(delta).all():
        raise ValueError("all three seeds and complete registered contexts required")
    family = family_matrix(delta, families)
    n = family.shape[-1]
    draws = np.random.default_rng(seed).integers(0, n, (DRAWS, n))
    if plateau:
        labels = np.asarray(families)
        counts = np.array([(labels == f).sum() for f in sorted(set(labels))])
        sums = family * counts
        boot = sums[:, draws].sum(-1) / counts[draws].sum(-1)
        per_seed = np.array([math.fsum(row.tolist()) / len(row) for row in delta])
    else:
        boot = family[:, draws].mean(-1)
        per_seed = family.mean(-1)
    ci = np.quantile(np.median(boot, axis=0), [0.025, 0.975])
    harm = (family < -0.01).mean(-1)
    point = float(np.median(per_seed))
    return dict(
        median_gain=point,
        ci95=ci.tolist(),
        per_seed_gain=per_seed.tolist(),
        per_seed_positive=(per_seed > 0).tolist(),
        per_seed_harm=harm.tolist(),
        median_harm=float(np.median(harm)),
        strict_pass=bool(
            point > (0.002 if plateau else 0)
            and ci[0] > 0
            and (per_seed > 0).all()
            and (plateau or (harm <= 0.10).all())
        ),
    )


def elo(panel, reference, families):
    delta = family_matrix(np.asarray(panel) - np.asarray(reference), families)
    wins = np.where(delta > 0, 1.0, np.where(delta < 0, 0.0, 0.5)).sum(-1)
    values = 1500 + 400 * np.log10((wins + 0.01) / (delta.shape[-1] - wins + 0.01))
    return dict(per_seed=values.tolist(), median=float(np.median(values)), anchor=1500)


def fit_inventory():
    outer, inner = exclusions()
    updates = sum(c["steps"] for c in CONFIGS)
    return dict(
        outer_proposals=3 * len(outer) * 8,
        inner_cv_fits=3 * len(inner) * 8,
        outer_fit_refits=3 * len(outer) * 8,
        all_train_refits=0,
        total_fits=3 * (len(inner) + len(outer)) * 8,
        optimizer_updates=3 * (len(inner) + len(outer)) * updates,
    )


def fold_comparisons(current, historical):
    return {
        wave: {
            f"{s}-{t}": partition_jaccard(a, b)
            for (s, a), (t, b) in itertools.product(current.items(), old.items())
        }
        for wave, old in historical.items()
    }
