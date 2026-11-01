"""H158 paired family inference; KD-only Holm family, fixed before execution."""

import hashlib
import random
import numpy as np

SEEDS = (158101, 158102, 158103)
FOLD_SEED = 158140
BOOT_SEED = 158141
DRAWS = 10000
ARMS = {
    "CTRL": (0.0, False),
    "KD03": (0.3, True),
    "KD07": (0.7, True),
    "KD07_NOVETO": (0.7, False),
}
KD_ARMS = ("KD03", "KD07", "KD07_NOVETO")


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
    """H140/H146 paired stream: 20 rounds, consume 8 draws but retain first 2 steps."""
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
        holm_lcb_candidates=[
            float(np.quantile(boot, 0.05 / (2 * k))) for k in (3, 2, 1)
        ],
        p_one_sided=float((1 + (boot <= 0).sum()) / (DRAWS + 1)),
        seeds_beating=int((gains > 0).sum()),
        median_seed=SEEDS[mid],
        top3_positive_share=share,
        elo=float(np.median(elo(score))),
        elo_ci95=np.quantile(elo_draws, [0.025, 0.975]).tolist(),
    )


def decide(contrasts):
    """Fixed KD family; step-down 95% two-sided percentile lower bounds.

    Rank by one-sided p (name breaks ties), use .025/(3-rank) quantiles,
    and require the whole step-down prefix plus Holm p<.05. CTRL is diagnostic.
    """
    if set(contrasts) != set(KD_ARMS):
        raise ValueError("all three KD hypotheses required")
    result = {}
    adjusted = 0.0
    prefix_pass = True
    for rank, arm in enumerate(
        sorted(KD_ARMS, key=lambda a: (contrasts[a]["p_one_sided"], a))
    ):
        value = contrasts[arm]
        if not np.isfinite(
            [value["median_gain"], value["p_one_sided"], *value["holm_lcb_candidates"]]
        ).all():
            raise ValueError("nonfinite contrast")
        adjusted = max(adjusted, min(1.0, value["p_one_sided"] * (3 - rank)))
        lower = value["holm_lcb_candidates"][rank]
        prefix_pass = prefix_pass and lower > 0 and adjusted < 0.05
        gates = dict(margin=value["median_gain"] > 0.001, holm_lcb95=prefix_pass)
        result[arm] = dict(
            holm_rank=rank + 1,
            holm_adjusted_p=adjusted,
            holm_lcb95=lower,
            holm_prefix_pass=prefix_pass,
            gates=gates,
            advance=all(gates.values()),
        )
    return result
