"""H176 family-grouped learning-curve summaries; no data access or model code."""

import hashlib
import math

import numpy as np

SEEDS = (176101, 176102, 176103)
FOLD_SEED, BOOT_SEED = 176140, 176141
FRACTIONS = (0.25, 0.50, 1.00)
FRACTION_FAMILIES = (31, 62, 123)
ARMS = ("R882", "FRESH_25", "FRESH_50", "FRESH_100", "R882_CONT_100")
DRAWS = 10_000
TIE_MARGIN = 1e-4


def stream_seed(*parts):
    return int.from_bytes(
        hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little"
    ) & 0x7FFFFFFF


def family_folds(family_ids, seed=FOLD_SEED):
    families = sorted(set(family_ids))
    if len(families) != 205:
        raise ValueError("205 TRAIN families required")
    order = np.random.default_rng(seed).permutation(len(families))
    return [
        sorted(families[i] for i in group)
        for group in np.array_split(order, 5)
    ]


def partition(family_ids, outer, folds=None):
    groups = family_folds(family_ids) if folds is None else folds
    if len(groups) != 5 or not 0 <= outer < 5:
        raise ValueError("five family folds required")
    a, b = set(groups[outer]), set(groups[(outer + 1) % 5])
    fit = set().union(*(set(groups[i]) for i in range(5) if i not in (outer, (outer + 1) % 5)))
    if fit & (a | b) or a & b:
        raise ValueError("family fold overlap")
    return fit, b, a


def nested_family_subsets(fit_families, seed):
    families = sorted(set(fit_families))
    if len(families) != 123:
        raise ValueError("123 FIT families required")
    ordered = np.random.default_rng(seed).permutation(families).tolist()
    result = {size: frozenset(ordered[:size]) for size in FRACTION_FAMILIES}
    if not (result[31] < result[62] < result[123] == frozenset(families)):
        raise AssertionError("nested family subsets")
    return result


def contexts_for_families(rows, families):
    selected = [r["context_sha256"] for r in rows if r["family_id"] in families]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("empty or duplicated context subset")
    if any(r["family_id"] not in families for r in rows if r["context_sha256"] in selected):
        raise AssertionError("context outside selected family subset")
    return selected


def context_utility(fitness, gate):
    fitness = np.asarray(fitness, dtype=np.float64)
    gate = np.asarray(gate, dtype=bool)
    if fitness.shape != gate.shape or not np.isfinite(fitness).all():
        raise ValueError("invalid outcome arrays")
    return np.where(gate, -fitness, -2.0)


def batch_plan(cids, orders, seed, updates=40):
    """H156 sampler: eight FIT contexts, four lowest-fitness plus twenty random candidates."""
    cids = sorted(cids)
    if len(cids) < 8 or updates < 1:
        raise ValueError("H156 needs at least eight FIT contexts and a positive budget")
    plan = []
    for step in range(1, updates + 1):
        stream = stream_seed(seed, step)
        rng = np.random.default_rng(stream)
        chosen = [cids[i] for i in rng.choice(len(cids), 8, replace=False)]
        indexes = []
        for cid in chosen:
            order = np.asarray(orders[cid], dtype=np.int64)
            if len(order) < 24:
                raise ValueError("H156 needs at least 24 candidates per context")
            indexes.append(np.concatenate([order[:4], rng.choice(order[4:], 20, replace=False)]))
        plan.append((stream, chosen, indexes))
    return plan


def family_values(rows, values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(rows),) or not np.isfinite(values).all():
        raise ValueError("one finite utility per context required")
    families = sorted({r["family_id"] for r in rows})
    by_family = np.array(
        [values[[r["family_id"] == f for r in rows]].mean() for f in families],
        dtype=np.float64,
    )
    return families, by_family


def zero_update_gap(train_rows, train_r882, train_oracle, val_rows, val_r882, val_oracle):
    train_families, train_f = family_values(train_rows, train_r882)
    val_families, val_f = family_values(val_rows, val_r882)
    _, train_o = family_values(train_rows, train_oracle)
    _, val_o = family_values(val_rows, val_oracle)
    train_mean, val_mean = float(train_f.mean()), float(val_f.mean())
    train_oracle_mean, val_oracle_mean = float(train_o.mean()), float(val_o.mean())
    headroom = train_oracle_mean - train_mean
    gap = train_mean - val_mean
    return dict(
        train=dict(
            contexts=len(train_rows), families=len(train_families),
            equal_family_utility=train_mean,
            family_q05=float(np.quantile(train_f, 0.05)),
            family_q25=float(np.quantile(train_f, 0.25)),
            family_median=float(np.median(train_f)),
            family_q75=float(np.quantile(train_f, 0.75)),
            family_q95=float(np.quantile(train_f, 0.95)),
            minus2_contexts=int(np.count_nonzero(np.asarray(train_r882) == -2.0)),
            oracle_equal_family_utility=train_oracle_mean,
        ),
        dl_val=dict(
            contexts=len(val_rows), families=len(val_families),
            equal_family_utility=val_mean,
            family_q05=float(np.quantile(val_f, 0.05)),
            family_q25=float(np.quantile(val_f, 0.25)),
            family_median=float(np.median(val_f)),
            family_q75=float(np.quantile(val_f, 0.75)),
            family_q95=float(np.quantile(val_f, 0.95)),
            minus2_contexts=int(np.count_nonzero(np.asarray(val_r882) == -2.0)),
            oracle_equal_family_utility=val_oracle_mean,
        ),
        generalization_gap_train_minus_val=gap,
        oracle_headroom_train=headroom,
        oracle_headroom_normalized_gap=(gap / headroom if headroom > 0 else None),
        normalization_definition="(r882 equal-family TRAIN utility - r882 equal-family DL-Val utility) / (TRAIN oracle utility - r882 TRAIN utility)",
        family_values={"train": dict(zip(train_families, train_f.tolist())), "dl_val": dict(zip(val_families, val_f.tolist()))},
    )


def family_win_matrix(values_by_context, family_ids, tie_margin=TIE_MARGIN):
    v = np.asarray(values_by_context, dtype=np.float64)
    if v.ndim != 2 or v.shape[0] != len(family_ids) or not np.isfinite(v).all():
        raise ValueError("context x policy finite matrix required")
    diff = v[:, :, None] - v[:, None, :]
    wins = np.where(diff > tie_margin, 1.0, np.where(diff < -tie_margin, 0.0, 0.5))
    wins *= 1.0 - np.eye(v.shape[1])
    families = sorted(set(family_ids))
    return np.array(
        [wins[np.asarray(family_ids) == family].mean(axis=0) for family in families]
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


def _percentile_ci(x):
    return np.quantile(np.asarray(x), [0.025, 0.975], axis=0).tolist()


def learning_curve_summary(family_ids, family_panels, seed_ids=SEEDS, boot_seed=BOOT_SEED, context_panels=None, context_families=None):
    """Summarize family means; Elo uses context panels when provided for canonical H148 aggregation."""
    panels = np.asarray(family_panels, dtype=np.float64)
    if panels.shape != (len(seed_ids), len(family_ids), len(ARMS)) or not np.isfinite(panels).all():
        raise ValueError("incomplete seed/family/arm panel")
    if context_panels is not None:
        context_panels = np.asarray(context_panels, dtype=np.float64)
        if context_families is None or context_panels.ndim != 3 or context_panels.shape[0] != len(seed_ids) or context_panels.shape[2] != len(ARMS) or context_panels.shape[1] != len(context_families) or not np.isfinite(context_panels).all():
            raise ValueError("incomplete context-level Elo panel")
        if set(context_families) != set(family_ids):
            raise ValueError("context-level Elo family closure differs")
    if len(set(family_ids)) != len(family_ids):
        raise ValueError("duplicate families")
    n = len(family_ids)
    rng = np.random.default_rng(boot_seed)
    draw = rng.integers(0, n, (DRAWS, n))
    means = panels.mean(axis=1)
    out = {"family_count": n, "seed_ids": list(seed_ids), "arms": {}}
    for j, arm in enumerate(ARMS):
        boot_seed_means = np.take(panels[:, :, j], draw, axis=1).mean(axis=2)
        boot_medians = np.median(boot_seed_means, axis=0)
        out["arms"][arm] = dict(
            per_seed_equal_family_utility=means[:, j].tolist(),
            median_over_seeds=float(np.median(means[:, j])),
            family_bootstrap_ci95=_percentile_ci(boot_medians),
        )
    differences = {}
    for label, left, right in [("25_to_50", 1, 2), ("50_to_100", 2, 3), ("25_to_100", 1, 3)]:
        delta = panels[:, :, right] - panels[:, :, left]
        boot = np.median(delta[:, draw].mean(axis=2), axis=1)
        differences[label] = dict(
            per_seed_equal_family_gain=delta.mean(axis=1).tolist(),
            median_over_seeds=float(np.median(delta.mean(axis=1))),
            family_bootstrap_ci95=_percentile_ci(boot),
        )
    x = np.array([-2.0, -1.0, 0.0])
    slopes = np.stack([np.polyfit(x, means[s, 1:4], 1)[0] for s in range(len(seed_ids))])
    boot_slopes = np.empty(DRAWS)
    for start in range(0, DRAWS, 100):
        sampled_curves = np.take(panels[:, :, 1:4], draw[start : start + 100], axis=1).mean(axis=2)
        for offset in range(sampled_curves.shape[1]):
            means_b = sampled_curves[:, offset, :]
            boot_slopes[start + offset] = np.median([np.polyfit(x, means_b[s], 1)[0] for s in range(len(seed_ids))])
    out["paired_differences"] = differences
    out["slope_per_log2_doubling"] = dict(
        per_seed=slopes.tolist(), median_over_seeds=float(np.median(slopes)),
        family_bootstrap_ci95=_percentile_ci(boot_slopes),
    )
    if context_panels is None:
        win_panels = np.stack([family_win_matrix(panels[s], family_ids) for s in range(len(seed_ids))])
    else:
        win_panels = np.stack([family_win_matrix(context_panels[s], context_families) for s in range(len(seed_ids))])
    elo = bt_ratings(win_panels.sum(axis=1))
    elo_boot_parts = []
    for start in range(0, DRAWS, 100):
        sample = np.take(win_panels, draw[start : start + 100], axis=1).sum(axis=2)
        elo_boot_parts.append(bt_ratings(sample))
    elo_boot = np.concatenate(elo_boot_parts, axis=1)  # seed x draw x policy
    median_boot_elo = np.median(elo_boot, axis=0)  # draw x policy
    out["elo_vs_r882"] = {
        arm: dict(
            per_seed=elo[:, j].tolist(), median_over_seeds=float(np.median(elo[:, j])),
            family_bootstrap_ci95=_percentile_ci(median_boot_elo[:, j]),
        )
        for j, arm in enumerate(ARMS[1:], 1)
    }
    ci_25_50 = differences["25_to_50"]["family_bootstrap_ci95"]
    ci_50_100 = differences["50_to_100"]["family_bootstrap_ci95"]
    slope_ci = out["slope_per_log2_doubling"]["family_bootstrap_ci95"]
    if ci_50_100[0] > 0:
        verdict = "DATA_LIMITED"
    elif ci_50_100[0] <= 0 <= ci_50_100[1] and (ci_25_50[0] <= 0 <= ci_25_50[1] or slope_ci[1] < 0.001):
        verdict = "DATA_SATURATED"
    else:
        verdict = "INCONCLUSIVE"
    out["verdict"] = verdict
    out["bootstrap_draws"] = DRAWS
    out["bootstrap_seed"] = boot_seed
    out["bootstrap_method"] = "10,000 paired family-resamples shared across seeds and all curve arms; calculate equal-family policy means per seed, then median over seeds per draw; linear percentile CI"
    return out


def pilot_budget(seconds_per_update, folds=5, arms=4, seeds=3, max_updates=40, cap_seconds=10800, reserve_seconds=900):
    if not math.isfinite(seconds_per_update) or seconds_per_update <= 0:
        raise ValueError("positive finite update timing required")
    usable = cap_seconds - reserve_seconds
    cell_count = folds * arms * seeds
    updates = min(max_updates, int(usable // (cell_count * seconds_per_update)))
    return max(0, updates)


def execution_plan(seconds_per_update, max_updates=40, cap_seconds=10800, reserve_seconds=900):
    """Use five folds if the max budget fits, otherwise three; update count is timing-only."""
    if not math.isfinite(seconds_per_update) or seconds_per_update <= 0:
        raise ValueError("positive finite update timing required")
    arms, seeds = 4, len(SEEDS)
    for folds in (5, 3):
        projected = folds * arms * seeds * max_updates * seconds_per_update + reserve_seconds
        if projected <= cap_seconds:
            return {"outer_folds": folds, "updates_per_cell": max_updates, "projected_seconds": projected}
    folds = 3
    updates = pilot_budget(seconds_per_update, folds=folds, max_updates=max_updates, cap_seconds=cap_seconds, reserve_seconds=reserve_seconds)
    if updates < 4:
        return {"outer_folds": folds, "updates_per_cell": 0, "projected_seconds": None}
    projected = folds * arms * seeds * updates * seconds_per_update + reserve_seconds
    return {"outer_folds": folds, "updates_per_cell": updates, "projected_seconds": projected}
