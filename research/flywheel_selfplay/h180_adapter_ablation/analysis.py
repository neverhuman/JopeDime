"""H180 adapter-contribution ablation + fresh-init update-budget dose: family-grouped summaries; no data or model code.

Fold, partition, sampler, utility, family win-matrix and canonical Bradley-Terry functions are copied verbatim from
merged H176 (research/flywheel_selfplay/h176_data_scaling/analysis.py, including the #1512 axis=0 paired bootstrap);
only constants and the arm summary/verdict are H180's.
"""

import hashlib
import math

import numpy as np

SEEDS = (180101, 180102, 180103)
FOLD_SEED, BOOT_SEED = 180140, 180141
ARMS = ("R882", "ZERO_ADAPTER", "RAND0", "FRESH40", "FRESH160")
CANDIDATES = ARMS[1:]  # Holm family m=4 for ADVANCE_CANDIDATE
DOSE = (40, 160)
DRAWS = 10_000
TIE_MARGIN = 1e-4
MARGIN = 0.001
TIE = 1e-12
INERT_ABS = 0.005
PILOT_UPDATES = 4
WALL_RULE_SECONDS = 10_800
WALL_HARD_SECONDS = 14_400
EVAL_RESERVE_SECONDS = 1_800


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
    if fit & (a | b) or a & b:
        raise ValueError("family fold overlap")
    return fit, b, a


def contexts_for_families(rows, families):
    selected = [r["context_sha256"] for r in rows if r["family_id"] in families]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("empty or duplicated context subset")
    if any(
        r["family_id"] not in families for r in rows if r["context_sha256"] in selected
    ):
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
            indexes.append(
                np.concatenate([order[:4], rng.choice(order[4:], 20, replace=False)])
            )
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


def zero_update_gap(
    train_rows, train_r882, train_oracle, val_rows, val_r882, val_oracle
):
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
            contexts=len(train_rows),
            families=len(train_families),
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
            contexts=len(val_rows),
            families=len(val_families),
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
        family_values={
            "train": dict(zip(train_families, train_f.tolist())),
            "dl_val": dict(zip(val_families, val_f.tolist())),
        },
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


def summarize(
    family_ids,
    family_panels,
    context_panels,
    context_families,
    minus2,
    agreement,
    seed_ids=SEEDS,
    boot_seed=BOOT_SEED,
):
    """family_panels / context_panels: [seed, family|context, arm] with arm order ARMS (column 0 = R882).
    minus2 / agreement: {arm: [per-seed value]}. Paired family bootstrap shared across arms and seeds; for each draw,
    each seed's equal-family mean contrast, then the median over seeds (axis=0, as corrected in H176 #1512)."""
    panels = np.asarray(family_panels, dtype=np.float64)
    if (
        panels.shape != (len(seed_ids), len(family_ids), len(ARMS))
        or not np.isfinite(panels).all()
    ):
        raise ValueError("incomplete seed/family/arm panel")
    if len(set(family_ids)) != len(family_ids):
        raise ValueError("duplicate families")
    context_panels = np.asarray(context_panels, dtype=np.float64)
    if (
        context_panels.ndim != 3
        or context_panels.shape[0] != len(seed_ids)
        or context_panels.shape[2] != len(ARMS)
        or context_panels.shape[1] != len(context_families)
        or not np.isfinite(context_panels).all()
    ):
        raise ValueError("incomplete context-level panel")
    if set(context_families) != set(family_ids):
        raise ValueError("context-level family closure differs")
    n = len(family_ids)
    draw = np.random.default_rng(boot_seed).integers(0, n, (DRAWS, n))
    means = panels.mean(axis=1)
    out = {"family_count": n, "seed_ids": list(seed_ids), "arms": {}, "contrasts": {}}
    for j, arm in enumerate(ARMS):
        boot = np.median(np.take(panels[:, :, j], draw, axis=1).mean(axis=2), axis=0)
        out["arms"][arm] = dict(
            per_seed_equal_family_utility=means[:, j].tolist(),
            median_over_seeds=float(np.median(means[:, j])),
            family_bootstrap_ci95=_percentile_ci(boot),
            minus2_contexts_per_seed=list(minus2[arm]),
            choice_agreement_with_r882_per_seed=list(agreement[arm]),
            identical_across_seeds=bool(
                all(
                    np.array_equal(context_panels[0, :, j], context_panels[s, :, j])
                    for s in range(len(seed_ids))
                )
            ),
        )
    pairs = [(f"{a}_minus_R882", a, "R882") for a in CANDIDATES] + [
        ("FRESH160_minus_FRESH40", "FRESH160", "FRESH40")
    ]
    for label, left, right in pairs:
        delta = panels[:, :, ARMS.index(left)] - panels[:, :, ARMS.index(right)]
        per_seed = delta.mean(axis=1)
        boot = np.median(delta[:, draw].mean(axis=2), axis=0)
        out["contrasts"][label] = dict(
            per_seed_equal_family_gain=per_seed.tolist(),
            median_over_seeds=float(np.median(per_seed)),
            family_bootstrap_ci95=_percentile_ci(boot),
            p_one_sided=float((1 + np.count_nonzero(boot <= 0)) / (DRAWS + 1)),
            positive_seeds=int(np.count_nonzero(per_seed > 0)),
        )
    win_panels = np.stack(
        [
            family_win_matrix(context_panels[s], context_families)
            for s in range(len(seed_ids))
        ]
    )
    elo = bt_ratings(win_panels.sum(axis=1))
    parts = [
        bt_ratings(np.take(win_panels, draw[start : start + 100], axis=1).sum(axis=2))
        for start in range(0, DRAWS, 100)
    ]
    median_boot_elo = np.median(np.concatenate(parts, axis=1), axis=0)
    out["elo_vs_r882"] = {
        arm: dict(
            per_seed=elo[:, j].tolist(),
            median_over_seeds=float(np.median(elo[:, j])),
            family_bootstrap_ci95=_percentile_ci(median_boot_elo[:, j]),
        )
        for j, arm in enumerate(ARMS)
    }
    out.update(decide(out["contrasts"]))
    out.update(
        bootstrap_draws=DRAWS,
        bootstrap_seed=boot_seed,
        bootstrap_method="10,000 paired family resamples shared across seeds and arms; each seed's equal-family mean contrast per draw, then median over seeds (axis=0); linear percentile CI",
    )
    return out


def decide(contrasts):
    """Preregistered verdicts. Ties (|x - margin| <= 1e-12) reject."""
    z = contrasts["ZERO_ADAPTER_minus_R882"]
    lo, hi = z["family_bootstrap_ci95"]
    if lo <= 0 <= hi and abs(z["median_over_seeds"]) < INERT_ABS:
        adapter = "ADAPTER_INERT"
    elif hi < 0:
        adapter = "ADAPTER_MATTERS"
    elif lo > 0:
        adapter = "ADAPTER_HARMFUL"
    else:
        adapter = "ADAPTER_INCONCLUSIVE"
    d = contrasts["FRESH160_minus_FRESH40"]["family_bootstrap_ci95"]
    dose = (
        "BUDGET_HELPS" if d[0] > 0 else ("BUDGET_HURTS" if d[1] < 0 else "BUDGET_FLAT")
    )
    p = {a: contrasts[f"{a}_minus_R882"]["p_one_sided"] for a in CANDIDATES}
    candidates, adjusted = {}, 0.0
    for rank, arm in enumerate(sorted(CANDIDATES, key=lambda a: (p[a], a))):
        adjusted = max(adjusted, min(1.0, p[arm] * (len(CANDIDATES) - rank)))
        c = contrasts[f"{arm}_minus_R882"]
        gates = dict(
            margin=c["median_over_seeds"] > MARGIN + TIE,
            lcb95_gt_0=c["family_bootstrap_ci95"][0] > 0,
            three_of_three_seeds=c["positive_seeds"] == 3,
            holm=adjusted < 0.05,
        )
        candidates[arm] = dict(
            holm_adjusted_p=adjusted, gates=gates, advance_candidate=all(gates.values())
        )
    next_step = (
        "stop optimizer/regularizer adapter waves; pivot to adapter inputs/representation"
        if adapter == "ADAPTER_INERT"
        else "continue (adapter contribution not shown inert)"
    )
    return dict(
        adapter_verdict=adapter,
        dose_verdict=dose,
        advance_candidates=candidates,
        any_advance_candidate=any(c["advance_candidate"] for c in candidates.values()),
        next_step=next_step,
    )


def execution_plan(seconds_per_update, seeds=len(SEEDS), updates=DOSE[1]):
    """Timing-only fold rule: five folds if the projected wall time fits 3 h, otherwise three (disclosed); if even
    three folds exceed the 4 h hard limit, the attempt is INCOMPLETE after the pilot."""
    if not math.isfinite(seconds_per_update) or seconds_per_update <= 0:
        raise ValueError("positive finite update timing required")
    for folds, cap in ((5, WALL_RULE_SECONDS), (3, WALL_HARD_SECONDS)):
        projected = folds * seeds * updates * seconds_per_update + EVAL_RESERVE_SECONDS
        if projected <= cap:
            return {
                "outer_folds": folds,
                "updates_per_cell": updates,
                "projected_seconds": projected,
            }
    return {"outer_folds": 0, "updates_per_cell": 0, "projected_seconds": None}


def choice_agreement(choices, reference):
    """Fraction of contexts where an arm's argmin candidate equals R882's."""
    keys = sorted(reference)
    if set(choices) != set(keys) or not keys:
        raise ValueError("choice closure")
    return float(np.mean([choices[k] == reference[k] for k in keys]))
