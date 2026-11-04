"""Paired family bootstrap and frozen chronological acceptance rules; no ML imports."""

import bisect
import numpy as np

SEEDS = (289101, 289102, 289103)
DRAWS = 10000
TIE = 1e-4
MARGIN = 0.002
RULES = ("R0", "R1", "R2", "R3")


def family_layout(families):
    ids = sorted(set(families))
    f = np.asarray(families)
    groups = [np.flatnonzero(f == x) for x in ids]
    return ids, groups, np.array([len(x) for x in groups])


def make_draws(n, seeds=SEEDS, draws=DRAWS):
    ix = []
    counts = []
    for seed in seeds:
        x = np.random.default_rng(seed).integers(0, n, (draws, n))
        c = np.zeros((draws, n), float)
        np.add.at(c, (np.arange(draws)[:, None], x), 1)
        ix.append(x)
        counts.append(c)
    return ix, counts


def distribution(samples):
    rows = []
    for seed, x in zip(SEEDS, samples):
        rows.append(
            dict(
                seed=seed,
                median=float(np.median(x)),
                ci95=np.quantile(x, [0.025, 0.975]).tolist(),
            )
        )
    return dict(
        by_seed=rows,
        median=float(np.median([r["median"] for r in rows])),
        ci95=np.median([r["ci95"] for r in rows], axis=0).tolist(),
        interpretation="median of bootstrap-seed medians/endpoints; Monte Carlo stability, not independent datasets",
    )


def context_bootstrap(delta, groups, sizes, counts):
    sums = np.array([np.sum(delta[g]) for g in groups])
    return [(c @ sums) / (c @ sizes) for c in counts]


def elo_samples(delta, groups, counts):
    wins = np.array(
        [
            np.mean(np.where(delta[g] > TIE, 1.0, np.where(delta[g] < -TIE, 0.0, 0.5)))
            for g in groups
        ]
    )
    losses = 1 - wins
    point = float(1500 + 400 * np.log10((wins.sum() + 0.01) / (losses.sum() + 0.01)))
    return point, [
        1500 + 400 * np.log10((c @ wins + 0.01) / (c @ losses + 0.01)) for c in counts
    ]


def comparison(candidate, baseline, groups, sizes, counts):
    d = np.asarray(candidate) - np.asarray(baseline)
    boot = context_bootstrap(d, groups, sizes, counts)
    f = np.array([d[g].mean() for g in groups])
    elo, eboot = elo_samples(d, groups, counts)
    return dict(
        context_mean_gain=float(d.mean()),
        equal_family_mean_gain=float(f.mean()),
        paired_family_median_gain=float(np.median(f)),
        bootstrap=distribution(boot),
        elo_base1500=dict(point=elo, bootstrap=distribution(eboot)),
        family_wtl=dict(
            wins=int((f > TIE).sum()),
            ties=int((abs(f) <= TIE).sum()),
            losses=int((f < -TIE).sum()),
        ),
        harmed_families_loss_gt_0_01=int((f < -0.01).sum()),
    )


def family_median_gate(gain, indices, margin=MARGIN, raw_median=None):
    point = float(np.median(gain))
    if point <= margin or (raw_median is not None and raw_median <= TIE):
        return False, dict(
            point=point, ci95=None, bootstrap_draws=0, reason="strict_margin_or_raw_tie"
        )
    boot = np.median(np.asarray(gain)[indices], axis=1)
    ci = np.quantile(boot, [0.025, 0.975])
    accept = bool(ci[0] > 0)
    return accept, dict(
        point=point,
        ci95=ci.tolist(),
        bootstrap_draws=len(indices),
        reason="accept" if accept else "LCB_not_positive",
    )


def midrank(history, value):
    lo = bisect.bisect_left(history, value - TIE)
    hi = bisect.bisect_right(history, value + TIE)
    return (lo + 0.5 * (hi - lo)) / len(history)


def replay(values, rounds, families, draws, emit=None):
    _, groups, sizes = family_layout(families)
    means = np.array([[v[g].mean() for g in groups] for v in values])
    clipped = np.array(
        [[np.maximum(v[g], -1.0).mean() for g in groups] for v in values]
    )
    context_means = values.mean(axis=1)
    runs = []
    for si, (seed, indices) in enumerate(zip(SEEDS, draws)):
        incumbents = {rule: 0 for rule in RULES}
        accepted = {rule: [] for rule in RULES}
        rank_history = [[float(v)] for v in means[0]]
        for i in range(1, len(rounds)):
            for rule in RULES:
                ref = incumbents[rule]
                if rule == "R0":
                    point = float(context_means[i] - context_means[ref])
                    yes = point > TIE
                    detail = dict(
                        point=point,
                        ci95=None,
                        bootstrap_draws=0,
                        reason="accept" if yes else "strict_mean_or_tie",
                    )
                else:
                    raw_median = None
                    if rule == "R1":
                        delta = means[i] - means[ref]
                    elif rule == "R2":
                        delta = clipped[i] - clipped[ref]
                    else:
                        delta = np.array(
                            [
                                midrank(h, a) - midrank(h, b)
                                for h, a, b in zip(rank_history, means[i], means[ref])
                            ]
                        )
                        raw_median = float(np.median(means[i] - means[ref]))
                    yes, detail = family_median_gate(
                        delta, indices, raw_median=raw_median
                    )
                    if raw_median is not None:
                        detail["raw_paired_family_median_gain"] = raw_median
                event = dict(
                    seed=seed,
                    rule=rule,
                    round=int(rounds[i]),
                    previous_round=int(rounds[ref]),
                    accepted=bool(yes),
                    **detail,
                )
                if emit is not None:
                    emit(event)
                if yes:
                    accepted[rule].append(event)
                    incumbents[rule] = i
            for h, v in zip(rank_history, means[i]):
                bisect.insort(h, float(v))
        runs.append(
            dict(
                seed=seed,
                rules={
                    r: dict(
                        accept_count=len(accepted[r]),
                        accepted=accepted[r],
                        terminal_round=int(rounds[incumbents[r]]),
                    )
                    for r in RULES
                },
            )
        )
    return runs


def winner_analysis(values, rounds, groups, sizes, draws, counts, latest):
    ix = np.flatnonzero((rounds >= latest - 199) & (rounds <= latest))
    base = values[0]
    window = values[ix]
    assert len(window) > 0
    plateau = window.mean(axis=0)
    lead = base - plateau
    lead_boot = context_bootstrap(lead, groups, sizes, counts)
    per_round = []
    for i in ix:
        d = base - values[i]
        bs = context_bootstrap(d, groups, sizes, counts)
        c = comparison(values[i], base, groups, sizes, counts)
        per_round.append(
            dict(
                round=int(rounds[i]),
                baseline_minus_round_mean=float(d.mean()),
                baseline_lead=distribution(bs),
                bootstrap_probability_baseline_gt_round=[
                    float(np.mean(x > 0)) for x in bs
                ],
                bootstrap_strict_win_probability=[float(np.mean(x > TIE)) for x in bs],
                candidate_vs_baseline=c,
            )
        )
    delta = window - base
    centered = delta - delta.mean(axis=1)[:, None]
    group_sums = np.stack([centered[:, g].sum(axis=1) for g in groups], axis=0)
    ns = [n for n in [1, 5, 10, 25, 50, 100, 200] if n <= len(window)]
    if len(window) not in ns:
        ns.append(len(window))
    best = {n: [] for n in ns}
    for c in counts:
        boot = (c @ group_sums) / (c @ sizes)[:, None]
        for n in ns:
            best[n].append(np.max(boot[:, :n], axis=1))
    null = {str(n): distribution(b) for n, b in best.items()}
    lead_summary = distribution(lead_boot)
    within = []
    for i in range(len(SEEDS)):
        ci = lead_summary["by_seed"][i]["ci95"]
        limit = null[str(len(window))]["by_seed"][i]["ci95"][1]
        within.append(bool(ci[0] <= 0 <= ci[1] and lead.mean() <= limit))
    summary = dict(
        window_first_requested=int(latest - 199),
        window_last_requested=int(latest),
        window_records=len(ix),
        plateau_definition="per-context arithmetic mean over recorded window",
        baseline_minus_plateau_point=float(lead.mean()),
        baseline_minus_plateau_bootstrap=lead_summary,
        best_of_N_centered_null=null,
        within_noise_by_seed=within,
        within_noise_all_three=all(within),
    )
    return summary, per_round, ix


def attribution(values, invalid, rounds, cids, families, window):
    base = values[0]
    bi = invalid[0]
    rows = []
    deltas = base - values[window]
    masks = invalid[window] | bi
    for i, d, mask in zip(window, deltas, masks):
        clipped = np.maximum(base, -1.0) - np.maximum(values[i], -1.0)
        rows.append(
            dict(
                round=int(rounds[i]),
                lead=float(d.mean()),
                invalid_tail_contribution=float(np.where(mask, d, 0.0).mean()),
                both_valid_contribution=float(np.where(~mask, d, 0.0).mean()),
                excess_penalty_removed_by_clip=float((d - clipped).mean()),
                base_invalid=int(bi.sum()),
                candidate_invalid=int(invalid[i].sum()),
            )
        )
    per_context = []
    for j, cid in enumerate(cids):
        per_context.append(
            dict(
                cid=str(cid),
                family=str(families[j]),
                mean_loss=float(deltas[:, j].mean()),
                mean_contribution_to_91_mean=float(deltas[:, j].mean() / len(base)),
                invalid_frequency=float(invalid[window, j].mean()),
                base_invalid=bool(bi[j]),
                invalid_tail_contribution=float(
                    np.where(masks[:, j], deltas[:, j], 0.0).mean() / len(base)
                ),
            )
        )
    lead = float(deltas.mean())
    tail = float(np.where(masks, deltas, 0.0).mean())
    broad = float(np.where(~masks, deltas, 0.0).mean())
    _, groups, _ = family_layout(families)
    summary = dict(
        mean_lead=lead,
        invalid_tail_contribution=tail,
        both_valid_contribution=broad,
        invalid_tail_fraction=None if abs(lead) < 1e-12 else tail / lead,
        excess_penalty_removed_by_clip=float(
            np.mean([r["excess_penalty_removed_by_clip"] for r in rows])
        ),
        base_invalid=int(bi.sum()),
        mean_candidate_invalid=float(invalid[window].sum(axis=1).mean()),
        top10_contexts=sorted(
            per_context,
            key=lambda r: abs(r["mean_contribution_to_91_mean"]),
            reverse=True,
        )[:10],
        family_contributions=[
            dict(
                family=str(families[g[0]]),
                contexts=len(g),
                contribution=float(deltas[:, g].sum(axis=1).mean() / len(base)),
            )
            for g in groups
        ],
    )
    return summary, rows, per_context


def decide(runs, comparisons, winner, records):
    if records < 1:
        return dict(
            verdict="INSUFFICIENT_RECORDS", qualifying_rounds=[], shadow_only=True
        )
    accepted = {
        e["round"]
        for r in runs
        for rule in ["R1", "R2", "R3"]
        for e in r["rules"][rule]["accepted"]
    }
    witness = []
    for n in sorted(accepted):
        b = comparisons[str(n)]["bootstrap"]
        if b["median"] >= 0 and all(x["ci95"][0] > -0.002 for x in b["by_seed"]):
            witness.append(n)
    return dict(
        verdict="RULE_CHANGE_RECOMMENDED"
        if witness and winner["within_noise_all_three"]
        else "KEEP_RULE",
        qualifying_rounds=witness,
        winner_lead_within_noise_all_three=winner["within_noise_all_three"],
        shadow_only=True,
        fresh_test_required_before_adoption=True,
    )
