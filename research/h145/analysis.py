"""H145 fixed-family construction, paired uncertainty, selector cross-fit and Elo."""

from __future__ import annotations

import hashlib
import numpy as np


def family_folds(rows):
    ids = [row["context_sha256"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate context")
    families = sorted({row["family_id"] for row in rows})
    if len(families) < 5:
        raise ValueError("at least five resolved families required")
    groups = np.array_split(np.random.default_rng(145191).permutation(families), 5)
    result = []
    for ordinal, group in enumerate(groups):
        assessment = set(group.tolist())
        fit = set(families) - assessment
        assert not assessment & fit
        result.append(
            {
                "fold": ordinal,
                "fit_families": sorted(fit),
                "assessment_families": sorted(assessment),
                "fit_contexts": [
                    r["context_sha256"] for r in rows if r["family_id"] in fit
                ],
                "assessment_contexts": [
                    r["context_sha256"] for r in rows if r["family_id"] in assessment
                ],
            }
        )
    assert sorted(
        cid for fold in result for cid in fold["assessment_contexts"]
    ) == sorted(ids)
    return result


def family_matrix(values, families):
    order = sorted(set(families))
    matrix = np.asarray(
        [
            np.asarray(values)[np.asarray(families) == family].mean(axis=0)
            for family in order
        ]
    )
    if not np.isfinite(matrix).all():
        raise ValueError("nonfinite complete panel")
    return order, matrix


def interval(values):
    return {
        "se": float(np.std(values, ddof=1)),
        "ci95": np.quantile(values, [0.025, 0.975], method="linear").tolist(),
    }


def bt_ratings(wins, anchor, prior=0.01):
    """Canonical 400-iteration MM, with optional leading bootstrap dimension."""
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


def family_wins(values, families, tie_margin=1e-4):
    diff = values[:, :, None] - values[:, None, :]
    outcomes = np.where(diff > tie_margin, 1.0, np.where(diff < -tie_margin, 0.0, 0.5))
    outcomes *= 1.0 - np.eye(values.shape[1])
    return family_matrix(outcomes, families)[1]


def analyze(rows, values_by_arm, aliases, rounds):
    arms = sorted(set(aliases.values()), key=lambda arm: rounds[arm])
    reference = arms.index("R882")
    families = [row["family_id"] for row in rows]
    values = np.column_stack([values_by_arm[arm] for arm in arms]).astype(np.float64)
    if values.shape != (len(rows), len(arms)) or not np.isfinite(values).all():
        raise ValueError("invalid complete paired adapter panel")
    order, fm = family_matrix(values, families)
    rng = np.random.default_rng(145000)
    sample = rng.integers(0, len(order), size=(10_000, len(order)))
    boot = fm[sample].mean(axis=1)
    gain = boot - boot[:, reference, None]
    means = fm.mean(axis=0)

    # The small TRAIN control is chosen from identities, before looking at utility.
    small_indices = sorted(
        range(len(rows)),
        key=lambda i: hashlib.sha256(
            ("145091:" + rows[i]["context_sha256"]).encode()
        ).hexdigest(),
    )[:91]
    small_families = [families[i] for i in small_indices]
    small_order, small = family_matrix(values[small_indices], small_families)
    small_sample = np.random.default_rng(145000).integers(
        0, len(small_order), size=(10_000, len(small_order))
    )
    small_boot = small[small_sample].mean(axis=1)
    small_gain = small_boot - small_boot[:, reference, None]

    fold_result, selected_gain = [], np.empty(len(order))
    family_index = {family: i for i, family in enumerate(order)}
    for fold in family_folds(rows):
        fit = [family_index[f] for f in fold["fit_families"]]
        held = [family_index[f] for f in fold["assessment_families"]]
        chosen = int(
            np.argmax(fm[fit].mean(axis=0))
        )  # arms already in earliest-round order
        selected_gain[held] = fm[held, chosen] - fm[held, reference]
        fold_result.append(
            {
                "fold": fold["fold"],
                "selected": arms[chosen],
                "fit_mean": float(fm[fit, chosen].mean()),
                "assessment_gain": float(selected_gain[held].mean()),
            }
        )
    selected_boot = selected_gain[sample].mean(axis=1)

    best = boot == boot.max(axis=1, keepdims=True)
    first_probability = (best / best.sum(axis=1, keepdims=True)).mean(axis=0)
    # Average rank for exact ties, with rank 1 best.
    ranks = 1.0 + (boot[:, None, :] > boot[:, :, None]).sum(axis=2)
    ranks += 0.5 * ((boot[:, None, :] == boot[:, :, None]).sum(axis=2) - 1)

    wins = family_wins(values, families)
    point_elo = bt_ratings(wins.sum(axis=0), reference)
    elo_sample = np.random.default_rng(145001).integers(
        0, len(order), size=(1000, len(order))
    )
    elo_boot = bt_ratings(wins[elo_sample].sum(axis=1), reference)
    results = {}
    ratios = []
    for i, arm in enumerate(arms):
        uncertainty, small_uncertainty = (
            interval(gain[:, i]),
            interval(small_gain[:, i]),
        )
        ratio = (
            uncertainty["se"] / small_uncertainty["se"]
            if small_uncertainty["se"] > 0
            else None
        )
        if i != reference and ratio is not None:
            ratios.append(ratio)
        results[arm] = {
            "equal_family_mean_utility": float(means[i]),
            "utility_uncertainty": interval(boot[:, i]),
            "mean_gain_vs_R882": float(means[i] - means[reference]),
            "gain_uncertainty": uncertainty,
            "small_control_gain_uncertainty": small_uncertainty,
            "full_small_gain_se_ratio": ratio,
            "first_probability": float(first_probability[i]),
            "mean_bootstrap_rank": float(ranks[:, i].mean()),
            "rank_ci95": np.quantile(ranks[:, i], [0.025, 0.975]).tolist(),
            "pairwise_positive_probability": {
                other: float((boot[:, i] > boot[:, j]).mean())
                for j, other in enumerate(arms)
            },
            "elo": float(point_elo[i]),
            "elo_ci95": np.quantile(elo_boot[:, i], [0.025, 0.975]).tolist(),
            "invalid_picks": int((values[:, i] == -2).sum()),
        }
    winner = int(np.argmax(means))
    gates = {
        "complete_panel": len(rows) == 1219 and len(order) == 205 and len(aliases) == 5,
        "median_se_ratio_lte_0_75": len(ratios) >= 2
        and float(np.median(ratios)) <= 0.75,
        "top_probability_gte_0_80": first_probability[winner] >= 0.80,
        "selected_in_four_folds": sum(
            row["selected"] == arms[winner] for row in fold_result
        )
        >= 4,
        "crossfit_gain_gt_0_001": selected_gain.mean() > 0.001,
        "crossfit_lower_bound_gt_0": interval(selected_boot)["ci95"][0] > 0,
    }
    for arm, representative in aliases.items():
        if arm != representative:
            results[arm] = dict(results[representative], alias_of=representative)
    result = {
        "contexts": len(rows),
        "families": len(order),
        "unique_functions": arms,
        "small_train_control": {
            "contexts": len(small_indices),
            "families": len(small_order),
        },
        "arms": results,
        "ranking": sorted(
            results, key=lambda a: (-results[a]["equal_family_mean_utility"], rounds[a])
        ),
        "selector_crossfit": {
            "folds": fold_result,
            "mean_gain": float(selected_gain.mean()),
            **interval(selected_boot),
        },
        "median_full_small_se_ratio": float(np.median(ratios)) if ratios else None,
        "decision_gates": {key: bool(value) for key, value in gates.items()},
        "proposal": "LARGER_PANEL_SHADOW_TRIAL"
        if all(gates.values())
        else "KEEP_AS_RESEARCH_NO_GATE_SWITCH",
        "live_switch": False,
        "bootstrap_replicates": 10_000,
        "elo_bootstrap_replicates": 1000,
        "historical_training_exposure": True,
    }
    arrays = {
        "context_values": values,
        "family_values": fm,
        "bootstrap_mean": boot,
        "elo_bootstrap": elo_boot,
        "selected_family_gain": selected_gain,
    }
    return result, arrays
