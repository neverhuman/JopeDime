"""Fixed H138 statistics; no model loading or optimizer operations."""

import itertools
import numpy as np

ARMS = ("SWA4_UNIFORM", "SWA2_LATE", "EMA_0.5")
ROUNDS = (782, 785, 859, 882)
FOLD_SEEDS = (138181, 138182, 138183)
WEIGHTS = {
    "R882": (0.0, 0.0, 0.0, 1.0),
    "SWA4_UNIFORM": (0.25, 0.25, 0.25, 0.25),
    "SWA2_LATE": (0.0, 0.0, 0.5, 0.5),
    "EMA_0.5": (1 / 6, 1 / 6, 1 / 6, 0.5),
}


def folds(rows):
    if len({r["context_sha256"] for r in rows}) != len(rows):
        raise ValueError("duplicate context identity")
    families = sorted({r["family_id"] for r in rows})
    if len(families) != 205 or len(rows) != 1219:
        raise ValueError("full205-family/1219-context DL-Train cohort required")
    if any(
        r["dl_split"] != "DL-Train"
        or r["task"] != "regression"
        or r["source_split"] not in ("train", "inner")
        for r in rows
    ):
        raise ValueError("DL-Train regression only")
    result = {}
    for seed in FOLD_SEEDS:
        shuffled = np.random.default_rng(seed).permutation(families).tolist()
        result[str(seed)] = {f: i % 5 for i, f in enumerate(shuffled)}
    return result


def holm(pvalues):
    result, alive, previous = {}, True, 0.0
    for rank, arm in enumerate(sorted(pvalues, key=lambda a: (pvalues[a], a))):
        p = pvalues[arm]
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("finite probability required")
        threshold = 0.05 / (len(pvalues) - rank)
        alive = bool(alive and p <= threshold)
        previous = max(previous, min(1.0, p * (len(pvalues) - rank)))
        result[arm] = dict(p=p, adjusted_p=previous, threshold=threshold, passed=alive)
    return result


def elo(gain):
    points = np.where(gain > 0, 1.0, np.where(gain < 0, 0.0, 0.5)).sum(axis=-1)
    return 1500 + 400 * np.log10((points + 0.01) / (gain.shape[-1] - points + 0.01))


def correlation(a, b):
    return None if min(np.std(a), np.std(b)) == 0 else float(np.corrcoef(a, b)[0, 1])


def primary(rows, values):
    if set(values) != {"R882", *ARMS}:
        raise ValueError("all fixed arms required")
    partitions = folds(rows)
    families = sorted(partitions[str(FOLD_SEEDS[0])])
    members = [
        np.array([i for i, r in enumerate(rows) if r["family_id"] == f])
        for f in families
    ]
    v = {a: np.asarray(x, dtype=float) for a, x in values.items()}
    if any(x.shape != (1219,) or not np.isfinite(x).all() for x in v.values()):
        raise ValueError("finite complete native utilities required")
    family = {a: np.array([x[m].mean() for m in members]) for a, x in v.items()}
    draws = np.random.default_rng(138100).integers(0, 205, size=(10000, 205))
    signs = np.random.default_rng(138101).choice([-1.0, 1.0], size=(10000, 205))
    result, pvalues = {}, {}
    for arm in ARMS:
        g = family[arm] - family["R882"]
        gain = float(g.mean())
        boot = g[draws].mean(axis=1)
        pvalues[arm] = float((1 + np.count_nonzero((signs @ g) / 205 >= gain)) / 10001)
        order = sorted(range(205), key=lambda i: (-abs(g[i]), families[i]))
        total = float(g.sum())
        absolute = float(np.abs(g).sum())
        # Every partition evaluates all205 families once; no refit or duplicated sample size.
        by_seed = [gain for _ in FOLD_SEEDS]
        fold_means = {
            str(seed): [
                float(
                    np.mean(
                        [
                            g[i]
                            for i, f in enumerate(families)
                            if partitions[str(seed)][f] == k
                        ]
                    )
                )
                for k in range(5)
            ]
            for seed in FOLD_SEEDS
        }
        ci = np.quantile(boot, [0.025, 0.975]).tolist()
        result[arm] = dict(
            median_gain=float(np.median(by_seed)),
            gain_by_fold_seed=by_seed,
            fold_mean_gains=fold_means,
            family_bootstrap_ci95=ci,
            family_median_gain=float(np.median(g)),
            family_balanced_utility=float(family[arm].mean()),
            context_mean_utility=float(v[arm].mean()),
            family_harm_count=int((g < 0).sum()),
            family_harm_fraction=float((g < 0).mean()),
            failure_count=int((v[arm] == -2).sum()),
            median_elo=float(elo(g)),
            elo_ci95=np.quantile(elo(g[draws]), [0.025, 0.975]).tolist(),
            family_wins=int((g > 0).sum()),
            family_ties=int((g == 0).sum()),
            family_losses=int((g < 0).sum()),
            top10_families=[families[i] for i in order[:10]],
            top10_signed_share=None
            if total == 0
            else float(g[order[:10]].sum() / total),
            top10_absolute_share=None
            if absolute == 0
            else float(np.abs(g[order[:10]]).sum() / absolute),
            fold_seed_correlations=[
                dict(seeds=[a, b], pearson=correlation(g, g), identical=True)
                for a, b in itertools.combinations(FOLD_SEEDS, 2)
            ],
            identical_to_reference=bool(np.array_equal(v[arm], v["R882"])),
        )
    adjusted = holm(pvalues)
    for arm, x in result.items():
        x["holm"] = adjusted[arm]
        x["gates"] = dict(
            median_gt_001=x["median_gain"] > 0.001,
            ci_lower_gt_zero=x["family_bootstrap_ci95"][0] > 0,
            holm=adjusted[arm]["passed"],
            all_fold_seeds_positive=all(g > 0 for g in x["gain_by_fold_seed"]),
            harm_le_10pct=x["family_harm_fraction"] <= 0.10,
            not_tied=not x["identical_to_reference"],
        )
        x["passed"] = all(x["gates"].values())
    eligible = [a for a in ARMS if result[a]["passed"]]
    selected = (
        min(eligible, key=lambda a: (-result[a]["median_gain"], ARMS.index(a)))
        if eligible
        else None
    )
    return dict(
        status="COMPLETE",
        reference_family_utility=float(family["R882"].mean()),
        reference_context_utility=float(v["R882"].mean()),
        reference_failure_count=int((v["R882"] == -2).sum()),
        reference_elo=1500,
        contexts=1219,
        families=205,
        fold_seeds=list(FOLD_SEEDS),
        bootstrap_draws=10000,
        bootstrap_seed=138100,
        permutation_draws=10000,
        permutation_seed=138101,
        arms=result,
        selected_arm=selected,
        arm_correlations=[
            dict(
                arms=[a, b],
                pearson=correlation(
                    family[a] - family["R882"], family[b] - family["R882"]
                ),
            )
            for a, b in itertools.combinations(ARMS, 2)
        ],
        verdict="PRIMARY_PASS_PENDING_SECONDARY" if selected else "NO_ADVANCE",
        interpretation="Fixed retained DL-Train-exposed checkpoints. Five-fold grouped evaluation for each seed, zero refits. All seeds cover the same205 families and their identical vectors are not independent replications; bootstrap/sign-flip unit is one family.",
        harm_definition="Strictly negative family mean gain, no tolerance; exact ties are not gains.",
    )


def secondary(rows, reference, candidate, selected):
    if selected not in ARMS or len(rows) != 91:
        raise ValueError("one preselected arm and exact91-context cohort required")
    ref = np.asarray(reference, dtype=float)
    v = np.asarray(candidate, dtype=float)
    if (
        ref.shape != (91,)
        or v.shape != (91,)
        or not np.isfinite(v).all()
        or not np.isfinite(ref).all()
    ):
        raise ValueError("complete finite secondary panel required")
    if abs(ref.mean() - 1.1377942) > 1e-6:
        raise ValueError("r882 regression identity changed")
    families = sorted({r["family_id"] for r in rows})
    members = [
        np.array([i for i, r in enumerate(rows) if r["family_id"] == f])
        for f in families
    ]
    assert len(families) == 19
    gain = v - ref
    counts = np.array([len(m) for m in members])
    sums = np.array([gain[m].sum() for m in members])
    fg = sums / counts
    draws = np.random.default_rng(138102).integers(0, 19, size=(10000, 19))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    ci = np.quantile(boot, [0.025, 0.975]).tolist()
    return dict(
        arm=selected,
        contexts=91,
        families=19,
        reference_mean=float(ref.mean()),
        mean_utility=float(v.mean()),
        median_gain=float(gain.mean()),
        gain_interpretation="One fixed candidate; median across identical fold-seed reports equals this context-weighted mean gain, not three independent validation reads.",
        family_bootstrap_ci95=ci,
        bootstrap_seed=138102,
        bootstrap_draws=10000,
        median_elo=float(elo(fg)),
        elo_ci95=np.quantile(elo(fg[draws]), [0.025, 0.975]).tolist(),
        family_harm_count=int((fg < 0).sum()),
        failure_count=int((v == -2).sum()),
        verdict="BREAKTHROUGH" if ci[0] > 0 else "NO_BREAKTHROUGH",
        serving_promotion=False,
    )
