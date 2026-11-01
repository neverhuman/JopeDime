"""Prospective H144 statistics: equal-family gains, median over three seeds."""

import hashlib
import random
import numpy as np

SEEDS = (144101, 144102, 144103)
FOLD_SEED = 144140
ARMS = {"CTRL": 1e-4, "WD_1E2": 1e-2, "WD_1E1": 1e-1}
DROPPED = ("DROP_00", "DROP_30")
DRAWS = 10000
BOOT_SEED = 144141
HOLM_M = 4  # Preserve dispatched multiplicity; dropped arms have p=1.


def folds(rows):
    if not rows or len({r["context_sha256"] for r in rows}) != len(rows):
        raise ValueError("unique contexts required")
    if any(
        r["dl_split"] != "DL-Train"
        or r.get("has_meta_test") is True
        or r.get("source_split") not in ("train", "inner")
        for r in rows
    ):
        raise ValueError("DL-Train inner rows only")
    fams = sorted({r["family_id"] for r in rows})
    return [
        sorted(x.tolist())
        for x in np.array_split(np.random.default_rng(FOLD_SEED).permutation(fams), 5)
    ]


def batch_plan(fit_ids, orders, seed):
    """H140 paired stream: 20 rounds, consume 8 draws but retain first 2 steps."""
    if seed not in SEEDS or not fit_ids:
        raise ValueError("registered seed and fit contexts required")
    ids = tuple(sorted(fit_ids))
    sampler = seed * 1_000_003 + 10_007
    plan = []
    for rnd in range(1, 21):
        rng = np.random.RandomState((sampler + rnd * 104729) % 2**32)
        rot = rnd % len(ids)
        rotated = ids[rot:] + ids[:rot]
        for step in range(8):
            stream = int.from_bytes(
                hashlib.sha256(
                    f"flywheel-batch/v1:{sampler}:{rnd}:{step}".encode()
                ).digest()[:8],
                "little",
            )
            cids = random.Random(stream).choices(rotated, k=16)
            indexes = []
            for cid in cids:
                order = orders[cid]
                k = min(48, 192, len(order))
                rest = order[k:] if len(order) > k else order
                indexes.append(
                    np.concatenate(
                        [
                            order[:k],
                            rng.choice(rest, 192 - k, replace=len(rest) < 192 - k),
                        ]
                    )
                )
            if step < 2:
                plan.append((stream, cids, indexes))
    return plan


def family_matrix(rows, values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (3, len(rows)) or not np.isfinite(values).all():
        raise ValueError("all three complete finite seed panels required")
    families = sorted({r["family_id"] for r in rows})
    if len({r["context_sha256"] for r in rows}) != len(rows):
        raise ValueError("duplicate context")
    return families, np.stack(
        [
            values[:, [i for i, r in enumerate(rows) if r["family_id"] == f]].mean(
                axis=1
            )
            for f in families
        ],
        axis=1,
    )


def contrast(rows, values, reference):
    fams, delta = family_matrix(rows, np.asarray(values) - np.asarray(reference))
    gains = delta.mean(axis=1)
    ix = np.random.default_rng(BOOT_SEED).integers(0, len(fams), (DRAWS, len(fams)))
    boot = np.median(delta[:, ix].mean(axis=2), axis=0)
    # The family contribution vector comes from the seed defining the median point estimate.
    mid = sorted(range(3), key=lambda i: (gains[i], SEEDS[i]))[1]
    positive = np.maximum(delta[mid], 0)
    share = (
        float(np.sort(positive)[-3:].sum() / positive.sum()) if positive.sum() else 1.0
    )
    score = np.where(delta > 0, 1.0, np.where(delta < 0, 0.0, 0.5))

    def elo(x):
        wins = x.sum(axis=-1)
        n = x.shape[-1]
        return 1500 + 400 * np.log10((wins + 0.01) / (n - wins + 0.01))

    elo_draws = np.median(elo(score[:, ix]), axis=0)
    return dict(
        families=len(fams),
        per_seed_gain=gains.tolist(),
        median_gain=float(np.median(gains)),
        ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
        p_one_sided=float((1 + (boot <= 0).sum()) / (DRAWS + 1)),
        seeds_beating=int((gains > 0).sum()),
        median_seed=SEEDS[mid],
        top3_positive_share=share,
        elo=float(np.median(elo(score))),
        elo_ci95=np.quantile(elo_draws, [0.025, 0.975]).tolist(),
    )


def decide(contrasts):
    if set(contrasts) != set(ARMS) - {"CTRL"}:
        raise ValueError("complete registered contrasts required")
    p = {a: contrasts[a]["p_one_sided"] for a in contrasts}
    p.update({a: 1.0 for a in DROPPED})
    result = {}
    adjusted = 0.0
    for rank, a in enumerate(sorted(p, key=lambda a: (p[a], a))):
        adjusted = max(adjusted, min(1.0, p[a] * (HOLM_M - rank)))
        if a in DROPPED:
            continue
        v = contrasts[a]
        gates = dict(
            margin=v["median_gain"] > 0.001,
            ci=v["ci95"][0] > 0,
            two_seeds=v["seeds_beating"] >= 2,
            concentration=v["top3_positive_share"] <= 0.8,
            holm=adjusted < 0.05,
        )
        result[a] = dict(
            holm_adjusted_p=adjusted, gates=gates, advance=all(gates.values())
        )
    return result
