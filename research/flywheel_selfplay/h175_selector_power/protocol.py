"""H175 finite-cohort counterfactual audit; no optimizer or fitted policy."""

import hashlib
import numpy as np

SEEDS = (175101, 175102, 175103)
FOLD_SEED, BOOT_SEED = 175140, 175141
FRACTIONS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
SIMULATIONS, SELECTOR_DRAWS, REPORT_DRAWS = 200, 1000, 10000
EPS, MARGIN = 1e-12, 0.001


def stream(*parts):
    return (
        int.from_bytes(
            hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little"
        )
        & 0x7FFFFFFF
    )


def folds(rows):
    families = sorted({r["family_id"] for r in rows})
    if len(families) != 205:
        raise ValueError("205 TRAIN families required")
    order = np.random.default_rng(FOLD_SEED).permutation(len(families))
    return [sorted(families[i] for i in g) for g in np.array_split(order, 5)]


def family_mean(values, families):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1] != len(families) or not np.isfinite(values).all():
        raise ValueError("finite context panel required")
    fams = sorted(set(families))
    return fams, np.stack(
        [values[..., np.asarray(families) == f].mean(axis=-1) for f in fams], axis=-1
    )


def slate_indices(cid, n, seed, kind):
    if kind == "full":
        return np.arange(n)
    if kind != "24" or n < 24:
        raise ValueError("24 candidates required")
    # Preserve the four storage-order revealed candidates of native evaluation.
    rest = np.random.default_rng(stream(seed, cid, "slate24")).choice(
        np.arange(4, n), 20, replace=False
    )
    return np.r_[np.arange(4), rest]


def inject(rows, payloads, scope, seed, fraction, variant, direction, slate):
    """Pointwise oracle simulation reads only supplied role-local payloads.

    IDs/families, never outcomes, determine activation. The same hypothetical policy
    is reproducible when a family is visited independently in its A and B role.
    """
    if {r["context_sha256"] for r in rows} != set(scope):
        raise ValueError("injection scope mismatch")
    if variant not in ("context", "family") or direction not in (
        "oracle",
        "random_nonoracle",
    ):
        raise ValueError("unknown injection")
    base, oracle, delta, wins, changes, failures, avoidable = [], [], [], [], [], [], []
    for row in rows:
        cid, family = row["context_sha256"], row["family_id"]
        predicted, utility = payloads[cid]
        ix = slate_indices(cid, len(utility), seed, slate)
        p, u = np.asarray(predicted)[ix], np.asarray(utility)[ix]
        if not np.isfinite(p).all() or not np.isfinite(u).all():
            raise ValueError("nonfinite candidate values")
        baseline = float(u[np.argmin(p)])
        best = float(u.max())
        key = cid if variant == "context" else family
        active = (
            np.random.default_rng(stream(seed, variant, key, "activation")).random(
                SIMULATIONS
            )
            < fraction
        )
        if direction == "oracle":
            replacement = np.full(SIMULATIONS, best)
        else:
            nonoracle = np.flatnonzero(u < best)
            replacement = (
                np.full(SIMULATIONS, baseline)
                if not len(nonoracle)
                else u[
                    nonoracle[
                        np.random.default_rng(
                            stream(seed, cid, slate, "nonoracle")
                        ).integers(0, len(nonoracle), SIMULATIONS)
                    ]
                ]
            )
        diff = np.where(active, replacement - baseline, 0.0)
        base.append(baseline)
        oracle.append(best)
        delta.append(diff)
        wins.append(np.where(diff > 1e-4, 1.0, np.where(diff < -1e-4, 0.0, 0.5)))
        changes.append(active)
        failures.append(baseline == -2.0)
        avoidable.append(baseline == -2.0 and best > -2.0)
    return dict(
        base=np.array(base),
        oracle=np.array(oracle),
        delta=np.array(delta).T,
        wins=np.array(wins).T,
        activated=np.array(changes).T,
        failures=np.array(failures),
        avoidable=np.array(avoidable),
    )


def bootstrap_counts(n, seed, draws=SELECTOR_DRAWS):
    ix = np.random.default_rng(seed).integers(0, n, (draws, n))
    counts = np.zeros((draws, n), dtype=np.float64)
    for j in range(n):
        counts[:, j] = (ix == j).sum(axis=1)
    return counts / n


def select(family_deltas, indices):
    """One counterfactual candidate vs baseline; same strict paired-family rule.

    Common bootstrap draws across simulations change Monte Carlo coupling, not
    the selector. One candidate has no best-of-four snapshot multiplicity.
    """
    fm = np.asarray(family_deltas, dtype=np.float64)
    if fm.ndim != 2 or fm.shape[1] != indices.shape[1] or not np.isfinite(fm).all():
        raise ValueError("invalid B family panel")
    gain = fm.mean(axis=1)
    boot = np.empty((len(fm), len(indices)))
    for start in range(0, len(fm), 16):
        boot[start : start + 16] = np.ascontiguousarray(
            fm[start : start + 16][:, indices]
        ).mean(axis=2)
    ci = np.quantile(boot, [0.025, 0.975], axis=1).T
    new = gain > MARGIN + EPS
    strict = new & (ci[:, 0] > 0)
    return gain, ci, np.stack([strict, new])


def report_weights(n, seed=BOOT_SEED):
    return bootstrap_counts(n, seed, REPORT_DRAWS)


def summarize(family_values, weights):
    """Median of three seed family means, with shared family resampling."""
    values = np.asarray(family_values, dtype=np.float64)
    if values.shape != (3, weights.shape[1]):
        raise ValueError("three seed panels required")
    per_seed = values.mean(axis=1)
    boot = np.median(values @ weights.T, axis=0)
    return dict(
        per_seed=per_seed.tolist(),
        median=float(np.median(per_seed)),
        ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
    )


def elo(family_winprob, weights):
    """Exact two-policy BT fit with H148 family count and .01 pseudo-win prior."""
    n = family_winprob.shape[1]

    def rating(prob):
        return 1500 + 400 * np.log10((n * prob + 0.01) / (n * (1 - prob) + 0.01))

    points = rating(family_winprob.mean(axis=1))
    boot = np.median(rating(family_winprob @ weights.T), axis=0)
    return dict(
        r882=1500.0,
        per_seed=points.tolist(),
        median=float(np.median(points)),
        ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
        interpretation="expected family-weighted wins of randomized counterfactual policy, not Elo of mean utility",
    )


def headroom(rows, panels):
    fams = [r["family_id"] for r in rows]
    weights = report_weights(len(set(fams)))
    bases = np.stack([x["base"] for x in panels])
    best = np.stack([x["oracle"] for x in panels])
    gap = best - bases
    fm = np.stack([family_mean(v, fams)[1] for v in gap])
    wins = np.stack([family_mean(np.where(v > 1e-4, 1.0, 0.5), fams)[1] for v in gap])
    return dict(
        contexts=len(rows),
        families=len(set(fams)),
        equal_family_headroom=summarize(fm, weights),
        oracle_elo=elo(wins, weights),
        context_gap_quantiles_by_seed=[
            np.quantile(x, [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]).tolist() for x in gap
        ],
        family_gap_quantiles_by_seed=[
            np.quantile(x, [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]).tolist() for x in fm
        ],
        oracle_optimal_contexts_by_seed=[int((x == 0).sum()) for x in gap],
        failure_contexts_by_seed=[int(x["failures"].sum()) for x in panels],
        avoidable_failures_by_seed=[int(x["avoidable"].sum()) for x in panels],
        baseline_family_utility_by_seed=[
            float(family_mean(x, fams)[1].mean()) for x in bases
        ],
    )


def mde(curve, headroom_mean):
    """Smallest tested f with median power>=.8, no interpolation/extrapolation."""
    for row in curve:
        if row["power"]["median"] >= 0.8:
            g = row["injected_gain"]["median"]
            return dict(
                status="GRID_DETECTED",
                fraction=row["fraction"],
                gain=g,
                headroom_percent=100 * g / headroom_mean if headroom_mean > 0 else None,
            )
    last = curve[-1]
    return dict(
        status="RIGHT_CENSORED",
        fraction=None,
        gain=None,
        gain_lower_tested_bound=last["injected_gain"]["median"],
        headroom_percent_lower_tested_bound=100
        * last["injected_gain"]["median"]
        / headroom_mean
        if headroom_mean > 0
        else None,
    )
