"""H243 fixed cross-fit protocol for zero-update module-wise soups of r882 and r782 (merged H242/H241 inference
machinery; fresh per-seed family folds 243141-243143); model-free inference only."""

import hashlib
import numpy as np

SEEDS = (243101, 243102, 243103)
FOLD_SEEDS = {
    243101: 243141,
    243102: 243142,
    243103: 243143,
}  # hw_t0435: fresh partitions
BOOT_SEED = 243150
ARMS = ("CONTROL", "MODSWAP_SELECT", "GREEDY_SOUP", "PLACEBO")
DECISION_ARMS = ("MODSWAP_SELECT", "GREEDY_SOUP")  # Holm m = 2; PLACEBO is report-only
GREEDY_MAX_INGREDIENTS = 5
CLIFF_CONTEXT = 27  # H152 gate-cliff DL-Val context (fam-9f67f144...): reported separately, never excluded
ALL_ARMS = ("r882",) + ARMS
DRAWS, EPS = 10000, 1e-12
ADVANCE_MARGIN = 0.002  # hw_t0435: median gain vs r882 > +0.002
PLATEAU_BEST = 1.1389572  # r3175
R882_MEAN = 1.1377941898164525


def stream_seed(*parts):
    return (
        int.from_bytes(
            hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little"
        )
        & 0x7FFFFFFF
    )


def folds(rows, seed):
    fams = sorted({r["family_id"] for r in rows})
    if len(fams) != 205:
        raise ValueError("205 TRAIN families required")
    order = np.random.default_rng(FOLD_SEEDS[seed]).permutation(205)
    return [sorted(fams[i] for i in group) for group in np.array_split(order, 5)]


def partition(rows, outer, seed):
    """FIT = the four non-outer family folds (selection only); ASSESS = the outer fold (descriptive OOF)."""
    held = set(folds(rows, seed)[outer])
    return (
        [r for r in rows if r["family_id"] not in held],
        [r for r in rows if r["family_id"] in held],
    )


def family_matrix(values, families):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] != len(families) or not np.isfinite(values).all():
        raise ValueError("invalid context panel")
    fams = sorted(set(families))
    return fams, np.array(
        [values[np.asarray(families) == f].mean(axis=0) for f in fams]
    )


def choose(rows, panel, seed, labels):
    """Strict selector on FIT rows: column 0 is r882. The best column by equal-family mean is accepted only if its
    gain over column 0 exceeds .001 AND its family-bootstrap LCB95 > 0; otherwise r882. Numerical ties pick the
    earliest column (r882 first, then group order with beta .5 before 1.0)."""
    mat = np.asarray(panel, dtype=np.float64)
    if (
        mat.shape != (len(rows), len(labels))
        or len(labels) < 2
        or not np.isfinite(mat).all()
    ):
        raise ValueError("finite FIT panel with one column per label required")
    _, fm = family_matrix(mat, [r["family_id"] for r in rows])
    means = fm.mean(axis=0)
    best = next(i for i, v in enumerate(means) if v >= means.max() - EPS)
    diff = fm[:, best] - fm[:, 0]
    ix = np.random.default_rng(seed).integers(0, len(fm), (DRAWS, len(fm)))
    ci = np.quantile(diff[ix].mean(axis=1), [0.025, 0.975]).tolist()
    gain = float(diff.mean())
    accepted = best > 0 and gain > 0.001 + EPS and ci[0] > 0
    return dict(
        best=labels[best],
        selected=labels[best] if accepted else labels[0],
        gain=gain,
        ci95=ci,
        accepted=bool(accepted),
        family_means=means.tolist(),
    )


def bt_ratings(wins, anchor=0, prior=0.01):
    """H148 canonical 400-iteration MM, r882 anchor, pseudo-win prior .01."""
    wins = np.asarray(wins, dtype=np.float64)
    n = wins.shape[-1]
    games = wins + np.swapaxes(wins, -1, -2)
    totals = wins.sum(axis=-1) + prior * (n - 1)
    strengths = np.ones(wins.shape[:-1])
    for _ in range(400):
        denom = (
            (games + 2 * prior * (1 - np.eye(n)))
            / np.maximum(strengths[..., :, None] + strengths[..., None, :], 1e-12)
        ).sum(axis=-1)
        strengths = np.clip(totals / denom, 1e-6, 1e6)
        strengths /= np.maximum(strengths[..., anchor, None], 1e-12)
    return 1500 + 400 * np.log10(np.maximum(strengths, 1e-9))


def family_wins(values, families):
    diff = values[:, :, None] - values[:, None, :]
    outcomes = np.where(diff > 1e-4, 1.0, np.where(diff < -1e-4, 0.0, 0.5)) * (
        1 - np.eye(values.shape[1])
    )
    return family_matrix(outcomes, families)[1]


def holm(pvals):
    result, previous = {}, 0.0
    for rank, key in enumerate(sorted(pvals, key=pvals.get)):
        previous = max(previous, min(1.0, pvals[key] * (len(pvals) - rank)))
        result[key] = previous
    return result


def harm(fam_ids, diff):
    """Families whose equal-family utility drops by more than 1e-4 against r882."""
    return [f for f, d in zip(fam_ids, diff) if d < -1e-4]


def summarize(rows, panel, seed=BOOT_SEED):
    values = np.asarray(panel, dtype=np.float64)
    if values.shape != (3, len(rows), len(ALL_ARMS)) or not np.isfinite(values).all():
        raise ValueError("complete 3-seed 5-policy panel required")
    families = [r["family_id"] for r in rows]
    fam_ids = family_matrix(values[0], families)[0]
    fm = np.stack([family_matrix(v, families)[1] for v in values])
    n = fm.shape[1]
    ix = np.random.default_rng(seed).integers(0, n, (DRAWS, n))
    means = fm.mean(axis=1)
    pairs = (
        [(a, "r882") for a in ARMS]
        + [(a, "CONTROL") for a in ARMS[1:]]
        + [("PLACEBO", "MODSWAP_SELECT")]
    )
    contrasts = {}
    for arm, ref in pairs:
        diff = fm[:, :, ALL_ARMS.index(arm)] - fm[:, :, ALL_ARMS.index(ref)]
        per_seed = diff.mean(axis=1)
        boot = np.median(diff[:, ix].mean(axis=2), axis=0)
        contrasts[arm + "_minus_" + ref] = dict(
            per_seed_gain=per_seed.tolist(),
            median_gain=float(np.median(per_seed)),
            ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
            positive_seeds=int((per_seed > 0).sum()),
            p_one_sided=float((1 + (boot <= 0).sum()) / (DRAWS + 1)),
            harm_families_by_seed=[harm(fam_ids, d) for d in diff],
        )
    adjusted = holm(
        {arm: contrasts[arm + "_minus_r882"]["p_one_sided"] for arm in DECISION_ARMS}
    )
    for arm in DECISION_ARMS:
        c = contrasts[arm + "_minus_r882"]
        c["holm_p"] = adjusted[arm]
        c["advance"] = bool(
            c["median_gain"] > ADVANCE_MARGIN + EPS
            and c["ci95"][0] > 0
            and adjusted[arm] < 0.05
            and c["positive_seeds"] >= 2
        )
    fw = np.stack([family_wins(v, families) for v in values])
    elo = bt_ratings(fw.sum(axis=1))
    ei = np.random.default_rng(seed + 1).integers(0, n, (1000, n))
    elo_boot = np.median(bt_ratings(fw[:, ei].sum(axis=2)), axis=0)
    context_means = values.mean(axis=1)
    return dict(
        contexts=len(rows),
        families=n,
        contrasts=contrasts,
        family_utility_by_seed=means.tolist(),
        context_utility_by_seed=context_means.tolist(),
        vs_plateau_best_by_seed=(context_means - PLATEAU_BEST).tolist(),
        elo_by_seed=elo.tolist(),
        median_elo=np.median(elo, axis=0).tolist(),
        elo_ci95=np.quantile(elo_boot, [0.025, 0.975], axis=0).T.tolist(),
        utility_bootstrap_draws=DRAWS,
        elo_bootstrap_draws=1000,
    )


def fixed_readout(rows, panel, names, seed=BOOT_SEED):
    """Descriptive (never admitted): each fixed policy on all rows, r882 in column 0."""
    values = np.asarray(panel, dtype=np.float64)
    if values.shape != (len(rows), 1 + len(names)) or not np.isfinite(values).all():
        raise ValueError("r882 plus one column per fixed policy required")
    families = [r["family_id"] for r in rows]
    fam_ids, fm = family_matrix(values, families)
    n = len(fm)
    ix = np.random.default_rng(seed).integers(0, n, (DRAWS, n))
    elo = bt_ratings(family_wins(values, families).sum(axis=0))
    out = {}
    for col, name in enumerate(names, 1):
        diff = fm[:, col] - fm[:, 0]
        boot = diff[ix].mean(axis=1)
        out[name] = dict(
            gain=float(diff.mean()),
            ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
            p_one_sided=float((1 + (boot <= 0).sum()) / (DRAWS + 1)),
            families_wtl=[
                int((diff > 1e-4).sum()),
                int((np.abs(diff) <= 1e-4).sum()),
                int((diff < -1e-4).sum()),
            ],
            harm_families=harm(fam_ids, diff),
            context_mean=float(values[:, col].mean()),
            vs_plateau_best=float(values[:, col].mean() - PLATEAU_BEST),
            elo=float(elo[col]),
        )
    return out


def family_mean(rows, values):
    return float(family_matrix(values, [r["family_id"] for r in rows])[1].mean())


def greedy(rows, base, singles, betas, score, seed):
    """Wortsman et al. 2022 greedy soup over module groups, on FIT rows only.

    `singles[(g, beta)]` holds each single-group policy's FIT values and `base` r882's. Each group's ingredient is
    its best beta by FIT family mean (ties: the smaller beta). Groups whose ingredient beats r882 by more than .001
    are ranked by that gain (ties: group order) and at most GREEDY_MAX_INGREDIENTS are tried in order; an ingredient
    is kept only if the soup's FIT family mean strictly improves by more than .001. `score(recipe)` returns the FIT
    values of a recipe {group: beta}. The final soup then passes the same strict check as `choose` (gain > .001 AND
    family LCB95 > 0 over r882), else r882."""
    r0 = family_mean(rows, base)
    groups = list(dict.fromkeys(g for g, _ in singles))
    ingredients = []
    for order, g in enumerate(groups):
        best = max(betas, key=lambda b: (family_mean(rows, singles[(g, b)]), -b))
        gain = family_mean(rows, singles[(g, best)]) - r0
        if gain > 0.001 + EPS:
            ingredients.append((gain, order, g, best))
    ingredients.sort(key=lambda x: (-x[0], x[1]))
    ingredients = ingredients[:GREEDY_MAX_INGREDIENTS]
    recipe, current, trace = {}, r0, []
    for gain, _, g, beta in ingredients:
        trial = dict(recipe, **{g: beta})
        values = score(trial)
        value = family_mean(rows, values)
        kept = value > current + 0.001 + EPS
        trace.append(
            dict(
                group=g, beta=beta, single_gain=gain, soup_value=value, kept=bool(kept)
            )
        )
        if kept:
            recipe, current = trial, value
    if not recipe:
        return dict(
            recipe={},
            selected={},
            accepted=False,
            trace=trace,
            gain=0.0,
            ci95=[0.0, 0.0],
        )
    final = np.stack([base, score(recipe)], axis=1)
    _, fm = family_matrix(final, [r["family_id"] for r in rows])
    diff = fm[:, 1] - fm[:, 0]
    ix = np.random.default_rng(seed).integers(0, len(fm), (DRAWS, len(fm)))
    ci = np.quantile(diff[ix].mean(axis=1), [0.025, 0.975]).tolist()
    accepted = float(diff.mean()) > 0.001 + EPS and ci[0] > 0
    return dict(
        recipe=recipe,
        selected=recipe if accepted else {},
        accepted=bool(accepted),
        trace=trace,
        gain=float(diff.mean()),
        ci95=ci,
    )


def recipe_label(recipe):
    """Canonical policy label of a soup recipe; {} is r882."""
    if not recipe:
        return "r882"
    if len(recipe) == 1:
        ((g, b),) = recipe.items()
        return f"{g}@{b:g}"
    return "SOUP:" + "+".join(f"{g}@{b:g}" for g, b in sorted(recipe.items()))
