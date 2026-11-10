"""SELECT-only rankers and family-level inference for H307; no training imports."""

import hashlib
import numpy as np

TIE = 1e-4
RULES = (
    "S0_ARGMAX_MEAN",
    "S1_ARGMAX_FAMILY_MEDIAN",
    "S2_ARGMAX_LCB95",
    "S3_EB_SHRUNK",
    "S4_TOPK_SMOOTH",
)
SEEDS = (30401, 30402, 30403)


def split_indices(families, seed):
    parity = np.array(
        [
            hashlib.sha256(
                b"H304-split-v1\0" + str(seed).encode() + b"\0" + str(f).encode()
            ).digest()[-1]
            % 2
            for f in families
        ]
    )
    select, evaluation = np.flatnonzero(parity == 0), np.flatnonzero(parity == 1)
    assert len(select) and len(evaluation)
    return select, evaluation


def balanced_splits(n_families=19, n_splits=200):
    """Exact H304 generator: RNG permutation first9, sorted, duplicates skipped."""
    rng = np.random.default_rng(304150)
    seen, result = set(), []
    while len(result) < n_splits:
        s = tuple(sorted(rng.permutation(n_families)[: n_families // 2].tolist()))
        if s in seen:
            continue
        seen.add(s)
        result.append(
            (np.array(s), np.array([i for i in range(n_families) if i not in s]))
        )
    return result


def family_matrix(utilities, context_families):
    names = np.array(sorted(set(context_families)))
    return names, np.column_stack(
        [utilities[:, context_families == name].mean(1) for name in names]
    )


def earliest_max(scores, rounds, tolerance=TIE):
    scores, rounds = np.asarray(scores), np.asarray(rounds)
    assert scores.shape == rounds.shape and np.isfinite(scores).all()
    tied = np.flatnonzero(scores.max() - scores <= tolerance)
    return int(tied[np.argmin(rounds[tied])])


def bootstrap_weights(n, draws, seed):
    indices = np.random.default_rng(seed).integers(0, n, size=(draws, n))
    counts = np.zeros((draws, n))
    np.add.at(counts, (np.arange(draws)[:, None], indices), 1)
    return counts / n


def shrunk_scores(means, variances):
    mu = float(np.mean(means))
    tau2 = max(float(np.var(means, ddof=1) - np.mean(variances)), 0.0)
    factors = np.zeros_like(means) if tau2 == 0 else tau2 / (tau2 + variances)
    return mu + factors * (means - mu), dict(
        grand_mean=mu,
        tau2=tau2,
        variance_min=float(variances.min()),
        variance_max=float(variances.max()),
        factor_min=float(factors.min()),
        factor_median=float(np.median(factors)),
        factor_max=float(factors.max()),
    )


def smooth_scores(means, rounds):
    return np.array([means[np.abs(rounds - r) <= 5].mean() for r in rounds])


def select_candidates(select_family_utilities, rounds, seed):
    """This interface receives ONLY SELECT values, never a full/EVAL matrix."""
    x = np.asarray(select_family_utilities)
    assert x.ndim == 2 and len(x) == len(rounds) and np.isfinite(x).all()
    means = x.mean(1)
    weights = bootstrap_weights(x.shape[1], 1000, seed)
    boot = x @ weights.T
    shrunk, eb = shrunk_scores(means, boot.var(1, ddof=1))
    scores = np.stack(
        [
            means,
            np.median(x, axis=1),
            np.quantile(boot, 0.05, axis=1),
            shrunk,
            smooth_scores(means, rounds),
        ]
    )
    picks = np.array([earliest_max(score, rounds) for score in scores], dtype=np.int64)
    return picks, scores, eb


def interval(values):
    return dict(
        ci95=np.quantile(values, [0.025, 0.975]).tolist(),
        lcb95=float(np.quantile(values, 0.05)),
    )


def elo_mm_counts(wins, losses):
    """Two-player MM update, normalized to r3175 strength1. Ties excluded."""
    wins, losses = np.asarray(wins, float), np.asarray(losses, float)
    # With both starting strengths1 the MM denominator is identical for the two
    # players. Normalization cancels it; the fixed-point strength ratio follows.
    return 1500 + 400 * np.log10((wins + 0.01) / (losses + 0.01))


def elo_family(delta, weights=None):
    if weights is None:
        return float(elo_mm_counts(np.sum(delta > TIE), np.sum(delta < -TIE)))
    # Weights sum1; convert back to match counts at the original family sample size.
    return elo_mm_counts(
        weights @ (delta > TIE) * len(delta), weights @ (delta < -TIE) * len(delta)
    )


def eval_split(family_utilities, picks, select, evaluation):
    weights = bootstrap_weights(len(evaluation), 10000, 304150)
    baseline = family_utilities[0, evaluation]
    s0 = family_utilities[picks[0], evaluation]
    output = {}
    for name, pick in zip(RULES, picks):
        chosen = family_utilities[pick, evaluation]
        delta, versus_s0 = chosen - baseline, chosen - s0
        select_gain = float(
            (family_utilities[pick, select] - family_utilities[0, select]).mean()
        )
        select_vs_s0 = float(
            (family_utilities[pick, select] - family_utilities[picks[0], select]).mean()
        )
        output[name] = dict(
            eval_level=float(chosen.mean()),
            eval_level_interval=interval(weights @ chosen),
            baseline_eval_level=float(baseline.mean()),
            gain_vs_r3175=float(delta.mean()),
            gain_vs_r3175_interval=interval(weights @ delta),
            gain_vs_s0=float(versus_s0.mean()),
            gain_vs_s0_interval=interval(weights @ versus_s0),
            select_gain_vs_r3175=select_gain,
            shrinkage_vs_r3175=select_gain - float(delta.mean()),
            select_gain_vs_s0=select_vs_s0,
            shrinkage_vs_s0=select_vs_s0 - float(versus_s0.mean()),
            family_gains_vs_r3175=delta.tolist(),
            family_gains_vs_s0=versus_s0.tolist(),
            family_wins_ties_losses=[
                int(np.sum(delta > TIE)),
                int(np.sum(np.abs(delta) <= TIE)),
                int(np.sum(delta < -TIE)),
            ],
            elo_r3175_1500=elo_family(delta),
            elo_interval=interval(elo_family(delta, weights)),
        )
    return output


def common_family_weights(n, eval_sets, draws=10000):
    """One shared multinomial19 draw, restricted to each overlapping EVAL half."""
    rng = np.random.default_rng(304150)
    accepted = []
    rejected = 0
    while len(accepted) < draws:
        counts = np.bincount(rng.integers(0, n, size=n), minlength=n)
        if any(counts[e].sum() == 0 for e in eval_sets):
            rejected += 1
            continue
        accepted.append(counts)
    counts = np.array(accepted)
    weights = [counts[:, e] / counts[:, e].sum(1, keepdims=True) for e in eval_sets]
    return weights, rejected


def summarize_primary(family_utilities, all_picks, splits, per_seed):
    weights, rejected = common_family_weights(
        family_utilities.shape[1], [e for s, e in splits]
    )
    output = {}
    for ri, name in enumerate(RULES):
        values = {key: [] for key in ("level", "gain", "vs_s0", "elo", "baseline")}
        for si, ((select, evaluation), picks, w) in enumerate(
            zip(splits, all_picks, weights)
        ):
            selected = family_utilities[picks[ri], evaluation]
            baseline = family_utilities[0, evaluation]
            s0 = family_utilities[picks[0], evaluation]
            values["level"].append(w @ selected)
            values["baseline"].append(w @ baseline)
            values["gain"].append(w @ (selected - baseline))
            values["vs_s0"].append(w @ (selected - s0))
            values["elo"].append(elo_family(selected - baseline, w))
        entries = [per_seed[str(seed)]["rules"][name] for seed in SEEDS]
        output[name] = dict(
            median_eval_level=float(np.median([v["eval_level"] for v in entries])),
            median_eval_level_interval=interval(np.median(values["level"], axis=0)),
            median_baseline_eval_level=float(
                np.median([v["baseline_eval_level"] for v in entries])
            ),
            median_baseline_eval_level_interval=interval(
                np.median(values["baseline"], axis=0)
            ),
            median_gain_vs_r3175=float(
                np.median([v["gain_vs_r3175"] for v in entries])
            ),
            median_gain_vs_r3175_interval=interval(np.median(values["gain"], axis=0)),
            median_gain_vs_s0=float(np.median([v["gain_vs_s0"] for v in entries])),
            median_gain_vs_s0_interval=interval(np.median(values["vs_s0"], axis=0)),
            median_shrinkage_vs_r3175=float(
                np.median([v["shrinkage_vs_r3175"] for v in entries])
            ),
            median_shrinkage_vs_s0=float(
                np.median([v["shrinkage_vs_s0"] for v in entries])
            ),
            median_elo_r3175_1500=float(
                np.median([v["elo_r3175_1500"] for v in entries])
            ),
            median_elo_interval=interval(np.median(values["elo"], axis=0)),
            positive_lcb95_seeds=sum(
                v["gain_vs_r3175_interval"]["lcb95"] > 0 for v in entries
            ),
            beats_s0_seeds=sum(v["gain_vs_s0"] > TIE for v in entries),
        )
    return output, dict(
        draws=10000,
        seed=304150,
        rejected_empty_draws=rejected,
        scheme="Shared19-family multinomial draws restricted to EVAL sets; fixed picks",
    )


def decision(best_rule, primary):
    s = primary[best_rule]
    conditions = [
        s["median_gain_vs_r3175"] > 0.001,
        s["positive_lcb95_seeds"] == 3,
        s["beats_s0_seeds"] >= 2,
    ]
    return dict(
        best_non_s0=best_rule,
        chosen_on_seed=30401,
        median_gain_gt_001=conditions[0],
        positive_lcb95_3of3=conditions[1],
        beats_s0_2of3=conditions[2],
        verdict="RESELECT_ADVANCES" if all(conditions) else "RESELECT_NOT_ADVANCED",
    )
