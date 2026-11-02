"""Prospective H222 folds, paired H140 training streams and family readouts."""

import hashlib
import random
import numpy as np

SEEDS = (222101, 222102, 222103)
ARMS = ("CTRL", "SN", "JAC")
LAMBDAS = (0.001, 0.01)
STEPS = 40


def folds(families, seed):
    ordered = sorted(set(families))
    if len(ordered) != 205 or seed not in (222141, 222142, 222143):
        raise ValueError("registered 205-family fold seed required")
    return [
        sorted(x.tolist())
        for x in np.array_split(np.random.default_rng(seed).permutation(ordered), 5)
    ]


def batch_plan(fit_ids, orders, seed):
    """H144/H140 sampler body; only the caller's seed whitelist is generalized."""
    if not fit_ids:
        raise ValueError("nonempty FIT required")
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


def family_matrix(panel, families):
    values = np.asarray(panel, dtype=np.float64)
    labels = np.asarray(families)
    return np.stack(
        [values[..., labels == f].mean(-1) for f in sorted(set(families))], -1
    )


def choose_lambda(panels, families):
    if set(panels) != set(LAMBDAS):
        raise ValueError("both complete TRAIN-only lambda panels required")
    values = {lam: family_matrix(panels[lam], families).mean() for lam in LAMBDAS}
    if not np.isfinite(list(values.values())).all():
        raise ValueError("incomplete lambda CV")
    return min(LAMBDAS, key=lambda lam: (-values[lam], lam)), values


def contrast(panel, reference, families):
    delta = family_matrix(np.asarray(panel) - np.asarray(reference), families)
    if delta.shape != (3, 19) or not np.isfinite(delta).all():
        raise ValueError("complete three-seed, 19-family panel required")
    ix = np.random.default_rng(222150).integers(0, 19, (10000, 19))
    seed_means = delta.mean(-1)
    seed_draws = delta[:, ix].mean(-1)
    median_draws = np.median(seed_draws, axis=0)
    seed_ci = np.quantile(seed_draws, [0.025, 0.975], axis=1).T
    return dict(
        per_seed_gain=seed_means.tolist(),
        per_seed_positive=(seed_means > 0).tolist(),
        median_gain=float(np.median(seed_means)),
        ci95=np.quantile(median_draws, [0.025, 0.975]).tolist(),
        per_seed_ci95=seed_ci.tolist(),
        per_seed_harm_rate=(delta < 0).mean(-1).tolist(),
        median_harm_rate=float(np.median((delta < 0).mean(-1))),
        strict_pass=bool(
            (seed_means > 0).all()
            and (seed_ci[:, 0] > 0).all()
            and np.quantile(median_draws, 0.025) > 0
        ),
    )


def readout(panels, families):
    if set(panels) != {"R882", *ARMS} or any(
        np.asarray(x).shape != (3, 91) or not np.isfinite(x).all()
        for x in panels.values()
    ):
        raise ValueError("all registered outcomes required")
    contrasts = {
        arm + "_vs_" + ref: contrast(panels[arm], panels[ref], families)
        for arm in ARMS
        for ref in ("R882",)
        if arm != ref
    }
    for arm in ("SN", "JAC"):
        contrasts[arm + "_vs_CTRL"] = contrast(panels[arm], panels["CTRL"], families)
    elo = {}
    for arm in ARMS:
        d = family_matrix(
            np.asarray(panels[arm]) - np.asarray(panels["R882"]), families
        )
        wins = np.where(d > 0, 1.0, np.where(d < 0, 0.0, 0.5)).sum(-1)
        values = 1500 + 400 * np.log10((wins + 0.01) / (19 - wins + 0.01))
        elo[arm] = dict(
            per_seed=values.tolist(),
            median=float(np.median(values)),
            anchor="r882=1500; paired family Bradley-Terry, prior0.01; exact ties half wins",
        )
    admitted = [
        a
        for a in ("SN", "JAC")
        if contrasts[a + "_vs_CTRL"]["strict_pass"]
        and contrasts[a + "_vs_R882"]["strict_pass"]
    ]
    return dict(
        contrasts=contrasts,
        elo_vs_r882=elo,
        admitted=admitted,
        verdict="CANDIDATE" if admitted else "NO_ADVANCE",
        family_count=19,
        bootstrap_draws=10000,
        bootstrap_seed=222150,
    )
