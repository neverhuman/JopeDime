"""H164 H148 family cross-fit/selection and canonical BT inference."""

import hashlib
import random
import numpy as np

SEEDS = (164101, 164102, 164103)
FOLD_SEED = 164140
BOOT_SEED = 164141
STEPS = (0, 10, 20, 30, 40)
DRAWS = 10000
ARMS = {
    "CTRL": (0.0, False),
    "SAM01": (0.01, False),
    "SAM05": (0.05, False),
    "ASAM05": (0.5, True),
}
COLUMNS = ("r882", *ARMS)


def folds(rows):
    ids = [r["context_sha256"] for r in rows]
    families = sorted({r["family_id"] for r in rows})
    if (
        len(ids) != len(set(ids))
        or len(families) < 5
        or any(
            r.get("dl_split") != "DL-Train" or r.get("has_meta_test") is True
            for r in rows
        )
    ):
        raise ValueError("unique contexts and at least five families required")
    return [
        sorted(g.tolist())
        for g in np.array_split(
            np.random.default_rng(FOLD_SEED).permutation(families), 5
        )
    ]


def partition(rows, outer):
    groups = folds(rows)
    a, b = set(groups[outer]), set(groups[(outer + 1) % 5])
    fit = [r for r in rows if r["family_id"] not in a | b]
    select = [r for r in rows if r["family_id"] in b]
    assess = [r for r in rows if r["family_id"] in a]
    assert not a & b and not {r["family_id"] for r in fit} & (a | b)
    return fit, select, assess


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


def family_matrix(values, families):
    order = sorted(set(families))
    matrix = np.asarray(
        [
            np.asarray(values)[np.asarray(families) == family].mean(axis=0)
            for family in order
        ]
    )
    if not np.isfinite(matrix).all():
        raise ValueError("nonfinite complete panel")
    return order, matrix


def interval(values):
    return {
        "se": float(np.std(values, ddof=1)),
        "ci95": np.quantile(values, [0.025, 0.975], method="linear").tolist(),
    }


def bt_ratings(wins, anchor, prior=0.01):
    """Canonical 400-iteration MM, with optional leading bootstrap dimension."""
    wins = np.asarray(wins, dtype=np.float64)
    n = wins.shape[-1]
    games = wins + np.swapaxes(wins, -1, -2)
    offdiag = 1.0 - np.eye(n)
    totals = wins.sum(axis=-1) + prior * (n - 1)
    strengths = np.ones(wins.shape[:-1])
    for _ in range(400):
        denom = (
            (games + 2.0 * prior * offdiag)
            / np.maximum(strengths[..., :, None] + strengths[..., None, :], 1e-12)
        ).sum(axis=-1)
        strengths = np.clip(totals / denom, 1e-6, 1e6)
        strengths /= np.maximum(strengths[..., anchor, None], 1e-12)
    return 1500.0 + 400.0 * np.log10(np.maximum(strengths, 1e-9))


def family_wins(values, families, tie_margin=1e-4):
    diff = values[:, :, None] - values[:, None, :]
    outcomes = np.where(diff > tie_margin, 1.0, np.where(diff < -tie_margin, 0.0, 0.5))
    outcomes *= 1.0 - np.eye(values.shape[1])
    return family_matrix(outcomes, families)[1]


def choose(rows, panel, seed):
    """No assessment argument. Earliest best step; reject if margin/paired LB fail."""
    values = np.asarray(panel, dtype=float)
    if values.shape != (len(rows), len(STEPS)) or not np.isfinite(values).all():
        raise ValueError("complete five-snapshot selector required")
    _, fm = family_matrix(values, [r["family_id"] for r in rows])
    means = fm.mean(axis=0)
    best = int(np.flatnonzero(means >= means.max() - 1e-12)[0])
    nonzero = 1 + int(np.flatnonzero(means[1:] >= means[1:].max() - 1e-12)[0])
    delta = fm[:, best] - fm[:, 0]
    ix = np.random.default_rng(seed).integers(0, len(fm), (DRAWS, len(fm)))
    ci = np.quantile(delta[ix].mean(axis=1), [0.025, 0.975]).tolist()
    gain = float(delta.mean())
    accepted = gain > 0.001 + 1e-12 and ci[0] > 0
    return dict(
        best_step=STEPS[best],
        best_nonzero_step=STEPS[nonzero],
        best_nonzero_gain=float(means[nonzero] - means[0]),
        selected_step=STEPS[best] if accepted else 0,
        gain=gain,
        ci95=ci,
        accepted=bool(accepted),
        family_means=means.tolist(),
    )


def summarize(rows, panels):
    values = np.asarray(panels, dtype=np.float64)
    if values.shape != (3, len(rows), len(COLUMNS)) or not np.isfinite(values).all():
        raise ValueError("complete finite three-seed/five-policy panel required")
    if len({r["context_sha256"] for r in rows}) != len(rows):
        raise ValueError("duplicate context")
    families = [r["family_id"] for r in rows]
    fm = np.stack([family_matrix(v, families)[1] for v in values])
    n = fm.shape[1]
    ix = np.random.default_rng(BOOT_SEED).integers(0, n, (DRAWS, n))
    means = fm.mean(axis=1)
    boot = fm[:, ix].mean(axis=2)
    contrasts = {}
    for col, arm in enumerate(ARMS, 1):
        contrasts[arm] = {}
        for ref, rc in [("r882", 0), ("CTRL", 1)]:
            per = means[:, col] - means[:, rc]
            draws = np.median(boot[:, :, col] - boot[:, :, rc], axis=0)
            contrasts[arm]["vs_" + ref] = dict(
                per_seed_gain=per.tolist(),
                median_gain=float(np.median(per)),
                ci95=np.quantile(draws, [0.025, 0.975]).tolist(),
                p_one_sided=float((1 + (draws <= 0).sum()) / (DRAWS + 1)),
                positive_seeds=int((per > 0).sum()),
            )
    fw = np.stack([family_wins(v, families) for v in values])
    elo = bt_ratings(fw.sum(axis=1), 0)
    ei = np.random.default_rng(BOOT_SEED + 1).integers(0, n, (1000, n))
    eb = np.median(bt_ratings(fw[:, ei].sum(axis=2), 0), axis=0)
    return dict(
        contexts=len(rows),
        families=n,
        columns=list(COLUMNS),
        contrasts=contrasts,
        family_utility_by_seed=means.tolist(),
        context_utility_by_seed=values.mean(axis=1).tolist(),
        elo_by_seed=elo.tolist(),
        median_elo=np.median(elo, axis=0).tolist(),
        elo_ci95=np.quantile(eb, [0.025, 0.975], axis=0).T.tolist(),
        identical_seed_panels=bool(
            np.array_equal(values[0], values[1])
            and np.array_equal(values[1], values[2])
        ),
    )


def decide(summary):
    cs = summary["contrasts"]
    if set(cs) != set(ARMS):
        raise ValueError("all four registered arm contrasts required")
    result = {}
    adjusted = 0.0
    for rank, arm in enumerate(
        sorted(ARMS, key=lambda a: (cs[a]["vs_r882"]["p_one_sided"], a))
    ):
        v = cs[arm]["vs_r882"]
        if not np.isfinite([v["median_gain"], *v["ci95"], v["p_one_sided"]]).all():
            raise ValueError("nonfinite contrast")
        adjusted = max(adjusted, min(1.0, v["p_one_sided"] * (4 - rank)))
        gates = dict(
            margin=v["median_gain"] > 0.001 + 1e-12,
            ci=v["ci95"][0] > 0,
            holm=adjusted < 0.05,
        )
        result[arm] = dict(
            holm_rank=rank + 1,
            holm_adjusted_p=adjusted,
            gates=gates,
            advance=all(gates.values()),
        )
    return result
