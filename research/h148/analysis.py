"""H148 prospective family partitions, strict selection, paired inference and anchored BT Elo."""

import hashlib
import random
import numpy as np

SEEDS = (148101, 148102, 148103)
FOLD_SEED = 148140
BOOT_SEED = 148141
STEPS = (0, 10, 20, 30, 40)
DRAWS = 10000


def folds(rows):
    ids = [r["context_sha256"] for r in rows]
    families = sorted({r["family_id"] for r in rows})
    if len(ids) != len(set(ids)) or len(families) < 5:
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
    if values.shape != (len(rows), len(STEPS)):
        raise ValueError("complete five-snapshot selector required")
    _, fm = family_matrix(values, [r["family_id"] for r in rows])
    means = fm.mean(axis=0)
    best = int(np.argmax(means))
    delta = fm[:, best] - fm[:, 0]
    ix = np.random.default_rng(seed).integers(0, len(fm), (DRAWS, len(fm)))
    ci = np.quantile(delta[ix].mean(axis=1), [0.025, 0.975]).tolist()
    gain = float(delta.mean())
    accepted = gain > 0.001 + 1e-12 and ci[0] > 0
    return dict(
        best_step=STEPS[best],
        selected_step=STEPS[best] if accepted else 0,
        gain=gain,
        ci95=ci,
        accepted=bool(accepted),
        family_means=means.tolist(),
    )


def summarize(rows, panels):
    """Three seed panels, columns r882/XFIT/VALSEL, equal-family shared bootstrap."""
    values = np.asarray(panels, dtype=float)
    if values.shape != (3, len(rows), 3) or not np.isfinite(values).all():
        raise ValueError("complete finite three-seed/three-rule panel required")
    families = [r["family_id"] for r in rows]
    fm = np.stack([family_matrix(v, families)[1] for v in values])
    n = fm.shape[1]
    ix = np.random.default_rng(BOOT_SEED).integers(0, n, (DRAWS, n))
    means = fm.mean(axis=1)
    boot = fm[:, ix].mean(axis=2)
    pairs = {}
    for name, left, right in [
        ("XFIT_minus_VALSEL", 1, 2),
        ("XFIT_minus_r882", 1, 0),
        ("VALSEL_minus_r882", 2, 0),
    ]:
        per_seed = means[:, left] - means[:, right]
        draws = np.median(boot[:, :, left] - boot[:, :, right], axis=0)
        pairs[name] = dict(
            per_seed_gain=per_seed.tolist(),
            median_gain=float(np.median(per_seed)),
            ci95=np.quantile(draws, [0.025, 0.975]).tolist(),
            positive_seeds=int((per_seed > 0).sum()),
        )
    # Canonical H145 400-step MM, family mean context-pairwise wins, tie margin 1e-4.
    fw = np.stack([family_wins(v, families) for v in values])
    elo = bt_ratings(fw.sum(axis=1), 0)
    ei = np.random.default_rng(BOOT_SEED + 1).integers(0, n, (1000, n))
    elo_boot = np.median(bt_ratings(fw[:, ei].sum(axis=2), 0), axis=0)
    return dict(
        contexts=len(rows),
        families=n,
        contrasts=pairs,
        family_utility_by_seed=means.tolist(),
        elo_by_seed=elo.tolist(),
        median_elo=np.median(elo, axis=0).tolist(),
        elo_ci95=np.quantile(elo_boot, [0.025, 0.975], axis=0).T.tolist(),
        identical_seed_panels=bool(
            np.array_equal(values[0], values[1])
            and np.array_equal(values[1], values[2])
        ),
    )


def verdict(primary, selections):
    if len(selections) != 15:
        raise ValueError("15 complete cells required")
    if all(s[r]["selected_step"] == 0 for s in selections for r in ("XFIT", "VALSEL")):
        return "RULE_NEUTRAL"
    p = primary["contrasts"]["XFIT_minus_VALSEL"]
    if p["median_gain"] > 0.001 and p["ci95"][0] > 0 and p["positive_seeds"] >= 2:
        return "RULE_HELPS"
    if p["median_gain"] < -0.001 and p["ci95"][1] < 0 and p["positive_seeds"] <= 1:
        return "RULE_HURTS"
    return "RULE_NEUTRAL"
