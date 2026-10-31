"""H135 fixed native-utility estimand; no model or data access."""

import itertools
import numpy as np

SEEDS = (135, 136, 137)
ARMS = ("BLEND_025", "PURE_GBM", "BLEND_050")


def native(cost, fitness, gate):
    cost, fitness, gate = np.asarray(cost), np.asarray(fitness), np.asarray(gate)
    if (
        cost.ndim != 1
        or cost.size == 0
        or cost.shape != fitness.shape
        or cost.shape != gate.shape
        or not np.isfinite(cost).all()
    ):
        raise ValueError("invalid complete pool")
    i = int(np.argmin(cost))
    value = float(-fitness[i] if gate[i] else -2.0)
    if not np.isfinite(value):
        raise ValueError("nonfinite selected utility")
    return value, i


def sanity(values, reference):
    values, expected = (
        np.asarray(values),
        np.asarray(reference["utility"]["per_context"]),
    )
    if (
        values.shape != (91,)
        or expected.shape != (91,)
        or not np.isfinite(values).all()
    ):
        raise ValueError("91 finite reference contexts required")
    error = float(np.max(np.abs(values - expected)))
    mean_error = abs(float(values.mean()) - reference["best_utility"])
    if (
        error > 1e-6
        or mean_error > 1e-6
        or abs(float(values.mean()) - 1.1377942) > 1e-6
    ):
        raise RuntimeError("r882 sanity gate FAIL; STOP before fitting or blend")
    return dict(
        status="PASS",
        max_context_error=error,
        mean_error=mean_error,
        mean=float(values.mean()),
    )


def summarize(rows, reference, arms):
    ids = [r["context_sha256"] for r in rows]
    families = sorted({r["family_id"] for r in rows})
    members = [
        np.array([i for i, r in enumerate(rows) if r["family_id"] == f])
        for f in families
    ]
    n = len(rows)
    reference = np.asarray(reference, dtype=float)
    if n != 91 or reference.shape != (n,) or len(set(ids)) != n:
        raise ValueError("exact unique 91-context cohort required")
    draws = np.random.default_rng(135100).integers(
        0, len(families), size=(10000, len(families))
    )
    counts = np.array([len(x) for x in members])
    result = {}
    for arm in ARMS:
        values = np.asarray(arms[arm], dtype=float)
        if values.shape != (3, n) or not np.isfinite(values).all():
            raise ValueError("three complete finite seeds required")
        gain = values - reference
        per_seed = gain.mean(axis=1)
        sums = np.stack([gain[:, m].sum(axis=1) for m in members], axis=1)
        family_gain = sums / counts
        boot = np.median(sums[:, draws].sum(axis=2) / counts[draws].sum(axis=1), axis=0)
        median = float(np.median(per_seed))
        lcb = float(np.quantile(boot, 0.05))
        context_gain = gain.mean(axis=0)
        order = sorted(range(n), key=lambda i: (-abs(context_gain[i]), ids[i]))
        trimmed = float(context_gain[order[10:]].mean())
        signed = float(context_gain.sum())
        absolute = float(np.abs(context_gain).sum())
        harm = family_gain.mean(axis=0) < -0.01
        pairs = []
        for a, b in itertools.combinations(range(3), 2):
            identical = bool(np.array_equal(values[a], values[b]))
            corr = (
                None
                if min(gain[a].std(), gain[b].std()) == 0
                else float(np.corrcoef(gain[a], gain[b])[0, 1])
            )
            pairs.append(
                dict(
                    seeds=[SEEDS[a], SEEDS[b]],
                    pearson=corr,
                    identical_utilities=identical,
                )
            )
        independent = not any(p["identical_utilities"] for p in pairs)
        # Equal-family pairwise Elo, exact ties are half points and zero utility gain.
        points = np.stack(
            [
                np.where(gain[:, m] > 0, 1.0, np.where(gain[:, m] < 0, 0.0, 0.5)).mean(
                    axis=1
                )
                for m in members
            ],
            axis=1,
        )
        wins = points.sum(axis=1) + 0.01
        losses = len(families) + 0.02 - wins
        elo = 1500 + 400 * np.log10(wins / losses)
        gates = dict(
            median_gt_001=median > 0.001,
            lcb_gt_zero=lcb > 0,
            all_seeds_positive=bool((per_seed > 0).all()),
            harm_le_10pct=bool(harm.mean() <= 0.10),
            trim_gt_zero=trimmed > 0,
            seed_independence=independent,
        )
        result[arm] = dict(
            mean_utility_by_seed=values.mean(axis=1).tolist(),
            gain_by_seed=per_seed.tolist(),
            median_gain=median,
            lcb95=lcb,
            ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
            harm_count=int(harm.sum()),
            harm_fraction=float(harm.mean()),
            families=len(families),
            trimmed_mean=trimmed,
            top10_context_ids=[ids[i] for i in order[:10]],
            top10_signed_share=None
            if signed == 0
            else float(context_gain[order[:10]].sum() / signed),
            top10_absolute_share=None
            if absolute == 0
            else float(np.abs(context_gain[order[:10]]).sum() / absolute),
            failure_counts=(values == -2).sum(axis=1).tolist(),
            seed_pairs=pairs,
            elo_by_seed=elo.tolist(),
            median_elo=float(np.median(elo)),
            gates=gates,
            admission_assessed=arm == "BLEND_025",
            candidate=arm == "BLEND_025" and all(gates.values()),
        )
    return dict(
        protocol="fixed-dev-cost-argmin-negative-fitness/v1",
        reference_mean=float(reference.mean()),
        reference_failure_count=int((reference == -2).sum()),
        reference_elo=1500,
        seeds=list(SEEDS),
        contexts=n,
        families=len(families),
        bootstrap_draws=10000,
        bootstrap_seed=135100,
        arms=result,
        verdict="CANDIDATE_REVIEW_ONLY"
        if result["BLEND_025"]["candidate"]
        else "NO_ADVANCE",
        promotion=False,
    )
