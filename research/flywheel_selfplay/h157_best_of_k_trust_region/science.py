"""H157 family partition, strict selector and paired bootstrap summaries."""

import numpy as np

SEEDS = (157101, 157102, 157103)
FOLD_SEED = 157140
INNER_DRAWS = 2000
PRIMARY_DRAWS = 10000
MARGIN = 0.001


def folds(families):
    ordered = sorted(set(families))
    if len(ordered) != 205:
        raise ValueError("expected 205 eligible TRAIN families")
    return [
        sorted(g.tolist())
        for g in np.array_split(
            np.random.default_rng(FOLD_SEED).permutation(ordered), 5
        )
    ]


def bootstrap_median(gains, seed, draws=INNER_DRAWS):
    values = np.asarray(gains, dtype=np.float64)
    if values.ndim != 1 or values.size != 41 or not np.isfinite(values).all():
        raise ValueError("expected 41 finite family gains")
    ix = np.random.default_rng(seed).integers(0, len(values), (draws, len(values)))
    samples = np.median(values[ix], axis=1)
    return dict(
        median=float(np.median(values)),
        ci95=np.quantile(samples, [0.025, 0.975], method="linear").tolist(),
        bootstrap_seed=int(seed),
        bootstrap_draws=int(draws),
    )


def admits(score):
    tie = abs(score["median"] - MARGIN) <= 1e-12
    return bool(not tie and score["median"] > MARGIN and score["ci95"][0] > 0.0)


def choose_winner(candidates, seed):
    """Select the observed median winner; failed winner never falls back."""
    if not candidates:
        raise ValueError("candidate set is empty")
    scored = {
        name: bootstrap_median(gains, seed + index + 1)
        for index, (name, gains) in enumerate(sorted(candidates.items()))
    }
    ranked = sorted(scored, key=lambda name: (-scored[name]["median"], name))
    winner = ranked[0]
    tied_for_best = len(ranked) > 1 and scored[ranked[1]]["median"] == scored[winner]["median"]
    score = scored[winner]
    admitted = not tied_for_best and admits(score)
    return dict(
        winner=winner,
        admitted=bool(admitted),
        selected=winner if admitted else "R882",
        tied_for_best=bool(tied_for_best),
        candidates=scored,
    )


def primary_summary(base, treatments, seed=157142, draws=PRIMARY_DRAWS):
    """Per treatment, median seed mean gain with shared-family bootstrap and Holm."""
    base = np.asarray(base, dtype=np.float64)
    if base.shape != (3, 19) or not np.isfinite(base).all():
        raise ValueError("baseline must be 3 seeds by 19 families")
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, 19, (draws, 19))
    out = {}
    for name, values in treatments.items():
        values = np.asarray(values, dtype=np.float64)
        if values.shape != base.shape or not np.isfinite(values).all():
            raise ValueError("treatment must be a complete 3x19 family panel")
        seed_gains = (values - base).mean(axis=1)
        boot_by_seed = (values - base)[:, ix].mean(axis=2)
        boot = np.median(boot_by_seed, axis=0)
        out[name] = dict(
            per_seed_gain=seed_gains.tolist(),
            median_gain=float(np.median(seed_gains)),
            ci95=np.quantile(boot, [0.025, 0.975], method="linear").tolist(),
            bootstrap_lower_tail=float((np.count_nonzero(boot <= 0.0) + 1) / (draws + 1)),
            _bootstrap=boot,
        )
    ordered = sorted(out, key=lambda name: (out[name]["bootstrap_lower_tail"], name))
    for rank, name in enumerate(ordered):
        alpha = 0.05 / (2 - rank)
        boot = out[name].pop("_bootstrap")
        out[name]["holm_rank"] = rank + 1
        out[name]["holm_adjusted_p"] = min(
            1.0,
            max((2 - prior_rank) * out[prior]["bootstrap_lower_tail"]
                for prior_rank, prior in enumerate(ordered[: rank + 1])),
        )
        out[name]["holm_lcb95"] = float(np.quantile(boot, alpha, method="linear"))
    return dict(seed=int(seed), draws=int(draws), treatments=out)


def family_wins(values, families, tie_margin=1e-4):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(families) != len(values):
        raise ValueError("context x arm values and family IDs required")
    diff = values[:, :, None] - values[:, None, :]
    outcomes = np.where(diff > tie_margin, 1.0, np.where(diff < -tie_margin, 0.0, 0.5))
    outcomes *= 1.0 - np.eye(values.shape[1])
    family_ids = sorted(set(families))
    return np.asarray([outcomes[np.asarray(families) == f].mean(axis=0) for f in family_ids])


def bt_ratings(wins, anchor=0, prior=0.01):
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
