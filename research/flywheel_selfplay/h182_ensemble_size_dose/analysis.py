"""H182 grouped-fold ensemble-size inference and data-free design controls."""

import hashlib
import math

import numpy as np

SEEDS = (182101, 182102, 182103)
MEMBER_COUNT = 20
MEMBER_COUNTS = (5, 10, 20)
FOLD_SEED, BOOT_SEED = 182140, 182141
ARMS = ("R882", "ENS5", "ENS10", "ENS20")
FRESH_ARMS = ARMS[1:]
DRAWS = 10_000
TIE_MARGIN = 1e-4


def stream_seed(*parts):
    return (
        int.from_bytes(
            hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little"
        )
        & 0x7FFFFFFF
    )


def family_folds(family_ids, seed=FOLD_SEED):
    families = sorted(set(family_ids))
    if len(families) != 205:
        raise ValueError("205 TRAIN families required")
    order = np.random.default_rng(seed).permutation(len(families))
    return [sorted(families[i] for i in group) for group in np.array_split(order, 5)]


def partition(family_ids, outer, folds=None):
    groups = family_folds(family_ids) if folds is None else folds
    if len(groups) != 5 or not 0 <= outer < 5:
        raise ValueError("five family folds required")
    a, b = set(groups[outer]), set(groups[(outer + 1) % 5])
    fit = set().union(
        *(set(groups[i]) for i in range(5) if i not in (outer, (outer + 1) % 5))
    )
    if fit & (a | b) or a & b or (len(fit), len(b), len(a)) != (123, 41, 41):
        raise ValueError("family fold overlap or count mismatch")
    return fit, b, a


def context_utility(fitness, gate):
    fitness, gate = np.asarray(fitness, dtype=np.float64), np.asarray(gate, dtype=bool)
    if fitness.shape != gate.shape or not np.isfinite(fitness).all():
        raise ValueError("invalid outcome arrays")
    return np.where(gate, -fitness, -2.0)


def batch_plan(cids, orders, seed, updates=40):
    cids = sorted(cids)
    if len(cids) < 8 or updates < 1:
        raise ValueError(
            "H156 sampler requires eight FIT contexts and positive updates"
        )
    plan = []
    for step in range(1, updates + 1):
        stream = stream_seed(seed, step)
        rng = np.random.default_rng(stream)
        chosen = [cids[i] for i in rng.choice(len(cids), 8, replace=False)]
        indexes = []
        for cid in chosen:
            order = np.asarray(orders[cid], dtype=np.int64)
            indexes.append(
                np.concatenate((order[:4], rng.choice(order[4:], 20, replace=False)))
            )
        plan.append((stream, chosen, indexes))
    return plan


def validate_fit_plan(plan, fit_context_ids, a_context_ids, b_context_ids):
    """Reject any planned training context outside FIT or inside held-out A/B."""
    fit, a, b = set(fit_context_ids), set(a_context_ids), set(b_context_ids)
    if fit & (a | b) or a & b:
        raise ValueError("FIT/A/B context overlap")
    for _, chosen, indexes in plan:
        if (
            len(chosen) != 8
            or len(indexes) != 8
            or not set(chosen) <= fit
            or set(chosen) & (a | b)
        ):
            raise ValueError("training batch contains a non-FIT context")
    return True


def fresh_member_seeds(outer_seed, count=MEMBER_COUNT):
    if outer_seed not in SEEDS:
        raise ValueError("unknown outer seed")
    if count != MEMBER_COUNT:
        raise ValueError("H182 requires 20 members per outer seed")
    offset = 100 * (SEEDS.index(outer_seed) + 1)
    return tuple(182100 + offset + i for i in range(1, MEMBER_COUNT + 1))


def validate_member_init_hashes(hashes, expected=MEMBER_COUNT):
    """Require distinct constructor-produced adapter hashes before training."""
    values = tuple(hashes)
    if (
        expected != MEMBER_COUNT
        or len(values) != expected
        or any(not isinstance(value, str) or not value for value in values)
        or len(set(values)) != expected
    ):
        raise ValueError("20 distinct fresh-member initialization hashes required")
    return values


def member_indices(k):
    """Return the strictly nested prefix of independently seeded members for K."""
    if k not in MEMBER_COUNTS:
        raise ValueError("ensemble K must be 5, 10, or 20")
    return tuple(range(k))


def score_ensemble(member_scores, k):
    """Average the first K member scores per candidate before every argmin."""
    scores = np.asarray(member_scores, dtype=np.float64)
    indexes = member_indices(k)
    if (
        scores.ndim != 3
        or scores.shape[0] < k
        or not np.isfinite(scores[list(indexes)]).all()
    ):
        raise ValueError("expected finite 20-member x context x candidate score tensor")
    return scores[list(indexes)].mean(axis=0)


def family_fresh_subsets(fit_families):
    """All nested ensemble arms train on the exact same 123 FIT families."""
    fams = frozenset(fit_families)
    if len(fams) != 123:
        raise ValueError("fresh members require exactly the 123 FIT families")
    return {arm: fams for arm in FRESH_ARMS}


def family_values(rows, values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(rows),) or not np.isfinite(values).all():
        raise ValueError("one finite utility per context required")
    families = sorted({r["family_id"] for r in rows})
    by_family = np.array(
        [values[[r["family_id"] == f for r in rows]].mean() for f in families]
    )
    return families, by_family


def family_win_matrix(values_by_context, family_ids, tie_margin=TIE_MARGIN):
    values = np.asarray(values_by_context, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] != len(family_ids)
        or not np.isfinite(values).all()
    ):
        raise ValueError("context x arm utility matrix required")
    diff = values[:, :, None] - values[:, None, :]
    wins = np.where(diff > tie_margin, 1.0, np.where(diff < -tie_margin, 0.0, 0.5))
    wins *= 1.0 - np.eye(values.shape[1])
    families = sorted(set(family_ids))
    return np.array(
        [wins[np.asarray(family_ids) == fam].mean(axis=0) for fam in families]
    )


def bt_ratings(wins, anchor=0, prior=0.01):
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


def holm_adjust(p_values):
    """Holm step-down adjusted p-values, returned in original order."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or np.any((p < 0) | (p > 1)):
        raise ValueError("invalid p-values")
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    n = len(p)
    for rank, index in enumerate(order):
        running = max(running, (n - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _ci(values):
    return np.quantile(np.asarray(values), [0.025, 0.975], axis=0).tolist()


def paired_family_bootstrap(delta, draw):
    """Median over seeds for each paired-family bootstrap draw."""
    return np.median(delta[:, draw].mean(axis=2), axis=0)


def ensemble_summary(
    family_ids,
    family_panel,
    seed_ids=SEEDS,
    boot_seed=BOOT_SEED,
    context_panel=None,
    context_families=None,
):
    """Paired family bootstrap, one-sided centered bootstrap tests, Holm m=3, and Elo."""
    panel = np.asarray(family_panel, dtype=np.float64)
    if (
        panel.shape != (len(seed_ids), len(family_ids), len(ARMS))
        or not np.isfinite(panel).all()
        or len(set(family_ids)) != len(family_ids)
    ):
        raise ValueError("expected complete seed x family x arm panel")
    if context_panel is not None:
        cp = np.asarray(context_panel, dtype=np.float64)
        if (
            context_families is None
            or cp.ndim != 3
            or cp.shape[0] != len(seed_ids)
            or cp.shape[2] != len(ARMS)
            or cp.shape[1] != len(context_families)
            or not np.isfinite(cp).all()
        ):
            raise ValueError("invalid context Elo panel")
        if set(context_families) != set(family_ids):
            raise ValueError("context/family closure mismatch")
    n = len(family_ids)
    rng = np.random.default_rng(boot_seed)
    draw = rng.integers(0, n, (DRAWS, n))
    means = panel.mean(axis=1)
    out = {
        "family_count": n,
        "seed_ids": list(seed_ids),
        "arms": {},
        "family_bootstrap_draws": DRAWS,
    }
    for j, arm in enumerate(ARMS):
        bs = np.take(panel[:, :, j], draw, axis=1).mean(axis=2)
        med = np.median(bs, axis=0)
        out["arms"][arm] = {
            "per_seed_equal_family_utility": means[:, j].tolist(),
            "median_over_seeds": float(np.median(means[:, j])),
            "family_bootstrap_ci95": _ci(med),
        }
    gains, tests = {}, {}
    for arm in ARMS[1:]:
        j = ARMS.index(arm)
        delta = panel[:, :, j] - panel[:, :, 0]
        point = np.median(delta.mean(axis=1))
        boot = paired_family_bootstrap(delta, draw)
        gains[arm] = {
            "per_seed": delta.mean(axis=1).tolist(),
            "median_over_seeds": float(point),
            "family_bootstrap_ci95": _ci(boot),
        }
        if arm in FRESH_ARMS:
            # Center paired family bootstrap under the point null; upper tail is prespecified.
            tests[arm] = float(
                (1 + int(np.count_nonzero(boot - point >= point))) / (DRAWS + 1)
            )
    adjusted = holm_adjust([tests[a] for a in FRESH_ARMS])
    for i, arm in enumerate(FRESH_ARMS):
        gains[arm]["one_sided_centered_bootstrap_p"] = tests[arm]
        gains[arm]["holm_m3_adjusted_p"] = float(adjusted[i])
    out["gain_vs_r882"] = gains
    out["elo_vs_r882"] = {}
    if context_panel is not None:
        wins = np.stack(
            [family_win_matrix(cp[s], context_families) for s in range(len(seed_ids))]
        )
    else:
        wins = np.stack(
            [family_win_matrix(panel[s], family_ids) for s in range(len(seed_ids))]
        )
    ratings = bt_ratings(wins.sum(axis=1))
    elo_boot = []
    for start in range(0, DRAWS, 100):
        sample = np.take(wins, draw[start : start + 100], axis=1).sum(axis=2)
        elo_boot.append(bt_ratings(sample))
    median_elo = np.median(np.concatenate(elo_boot, axis=1), axis=0)
    for j, arm in enumerate(ARMS[1:], 1):
        out["elo_vs_r882"][arm] = {
            "per_seed": ratings[:, j].tolist(),
            "median_over_seeds": float(np.median(ratings[:, j])),
            "family_bootstrap_ci95": _ci(median_elo[:, j]),
        }
    return out


def verdict(summary, h175_mde80_205fam=0.04234124834299788):
    primary = summary["gain_vs_r882"]["ENS20"]
    passed = (
        primary["median_over_seeds"] > 0.001
        and primary["family_bootstrap_ci95"][0] > 0
        and all(g > 0 for g in primary["per_seed"])
        and primary["holm_m3_adjusted_p"] < 0.05
    )
    summary["verdict"] = "ADVANCE_CANDIDATE" if passed else "NO_ADVANCE"
    summary["underpowered_null_per_H175_MDE80"] = bool(
        not passed and abs(primary["median_over_seeds"]) < h175_mde80_205fam
    )
    summary["H175_MDE80_reference"] = {
        "pooled_205_family_planning_heuristic": h175_mde80_205fam,
        "measured_for_this_rule": False,
        "basis": "H175 family-clustered full-slate MDE80 .094678 divided by sqrt(5); planning heuristic only",
    }
    summary["verdict_rule"] = (
        "ADVANCE_CANDIDATE iff ENS20 median gain > .001, paired-family bootstrap LCB95 > 0, all three seed gains > 0, and Holm m=3 adjusted one-sided centered-bootstrap p < .05 across ENS5/ENS10/ENS20; ties reject"
    )
    return summary


def execution_plan(
    seconds_per_update,
    max_updates=40,
    cap_seconds=14400,
    reserve_seconds=900,
    members=MEMBER_COUNT,
):
    if not math.isfinite(seconds_per_update) or seconds_per_update <= 0:
        raise ValueError("positive finite update timing required")
    if members != MEMBER_COUNT:
        raise ValueError("H182 execution estimate must cover all 20 members")
    full_projection = (
        5 * members * len(SEEDS) * max_updates * seconds_per_update + reserve_seconds
    )
    if full_projection <= cap_seconds:
        return {
            "outer_folds": 5,
            "updates_per_member": max_updates,
            "projected_seconds": full_projection,
            "within_cap": True,
        }
    reduced_projection = (
        3 * members * len(SEEDS) * max_updates * seconds_per_update + reserve_seconds
    )
    if reduced_projection <= cap_seconds:
        return {
            "outer_folds": 3,
            "updates_per_member": max_updates,
            "projected_seconds": reduced_projection,
            "within_cap": True,
        }
    return {
        "outer_folds": 3,
        "updates_per_member": max_updates,
        "projected_seconds": reduced_projection,
        "within_cap": False,
    }
