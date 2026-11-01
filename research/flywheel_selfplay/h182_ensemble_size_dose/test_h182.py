"""H182 data-free ensemble-size, inference, leakage, and bootstrap controls."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import analysis as stat


def test_five_fresh_family_folds_are_disjoint_and_assess_each_once():
    families = {f"f{i:03}" for i in range(205)}
    folds = stat.family_folds(families, seed=182140)
    assert [len(group) for group in folds] == [41] * 5
    seen = []
    for outer in range(5):
        fit, b, a = stat.partition(families, outer, folds)
        assert (len(fit), len(b), len(a)) == (123, 41, 41)
        assert not fit & (a | b) and not a & b
        seen.extend(a)
    assert len(seen) == len(set(seen)) == 205


def test_twenty_member_seeds_are_distinct_and_ensemble_prefixes_nested():
    seeds = [stat.fresh_member_seeds(seed) for seed in stat.SEEDS]
    assert seeds == [
        tuple(range(182201, 182221)),
        tuple(range(182301, 182321)),
        tuple(range(182401, 182421)),
    ]
    flattened = [member for group in seeds for member in group]
    assert len(flattened) == len(set(flattened)) == 60
    assert stat.member_indices(5) == tuple(range(5))
    assert stat.member_indices(10) == tuple(range(10))
    assert stat.member_indices(20) == tuple(range(20))
    assert (
        set(stat.member_indices(5))
        < set(stat.member_indices(10))
        < set(stat.member_indices(20))
    )
    with pytest.raises(ValueError):
        stat.member_indices(15)


def test_twenty_distinct_init_hashes_and_manifest_bounds():
    hashes = tuple(
        hashlib.sha256(f"synthetic-init-{seed}".encode()).hexdigest()
        for seed in stat.fresh_member_seeds(182101)
    )
    assert stat.validate_member_init_hashes(hashes) == hashes
    with pytest.raises(ValueError):
        stat.validate_member_init_hashes([hashes[0]] * 20)
    manifest = json.loads((Path(__file__).parent / "manifest.json").read_text())
    assert manifest["max_trajectories"] == 300
    assert manifest["max_total_updates"] == 12000
    assert manifest["max_updates_per_member"] == 40
    assert manifest["member_seeds"]["182101"] == list(range(182201, 182221))
    assert "fractions" not in manifest


def test_candidate_scores_are_averaged_before_argmin_for_all_nested_sizes():
    scores = np.arange(40, dtype=float).reshape(20, 1, 2)
    scores[:, 0, 0] = np.arange(20, dtype=float)
    scores[:, 0, 1] = np.arange(20, 0, -1, dtype=float)
    ens5 = stat.score_ensemble(scores, 5)
    ens10 = stat.score_ensemble(scores, 10)
    ens20 = stat.score_ensemble(scores, 20)
    assert ens5[0].tolist() == pytest.approx([2.0, 18.0])
    assert ens10[0].tolist() == pytest.approx([4.5, 15.5])
    assert ens20[0].tolist() == pytest.approx([9.5, 10.5])
    assert (
        int(np.argmin(ens5[0]))
        == int(np.argmin(ens10[0]))
        == int(np.argmin(ens20[0]))
        == 0
    )


def test_training_plan_contains_fit_contexts_only_and_rejects_a_or_b():
    fit = {f"fit-{i}" for i in range(123)}
    a = {f"a-{i}" for i in range(41)}
    b = {f"b-{i}" for i in range(41)}
    ids = sorted(fit)
    orders = {cid: np.arange(48) for cid in ids}
    plan = stat.batch_plan(ids, orders, 182140)
    assert len(plan) == 40
    assert stat.validate_fit_plan(plan, fit, a, b)
    _, chosen, indexes = plan[0]
    contaminated = [(plan[0][0], chosen[:-1] + [sorted(a)[0]], indexes)] + plan[1:]
    with pytest.raises(ValueError, match="non-FIT"):
        stat.validate_fit_plan(contaminated, fit, a, b)


def test_paired_bootstrap_uses_axis_zero_and_keeps_10000_non_degenerate_draws():
    families = [f"f{i:03}" for i in range(40)]
    delta = np.stack([np.tile([-amp, amp], 20) for amp in (0.01, 0.02, 0.03)])
    assert [float(np.var(seed_delta)) for seed_delta in delta] == pytest.approx(
        [0.0001, 0.0004, 0.0009]
    )
    draws = np.random.default_rng(182141).integers(
        0, len(families), size=(stat.DRAWS, len(families))
    )
    bootstrap = stat.paired_family_bootstrap(delta, draws)
    assert bootstrap.shape == (10000,)
    assert np.ptp(bootstrap) > 0
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    assert lower < 0 < upper


def test_holm_and_verdict_keep_ens20_as_strict_primary():
    assert stat.holm_adjust([0.01, 0.02, 0.04]).tolist() == pytest.approx(
        [0.03, 0.04, 0.04]
    )
    good = {
        "gain_vs_r882": {
            arm: {
                "median_over_seeds": 0.002,
                "family_bootstrap_ci95": [0.001, 0.003],
                "per_seed": [0.001, 0.002, 0.003],
                "holm_m3_adjusted_p": 0.03,
            }
            for arm in stat.FRESH_ARMS
        }
    }
    assert stat.verdict(good.copy())["verdict"] == "ADVANCE_CANDIDATE"
    for key, value in (
        ("median_over_seeds", 0.001),
        ("family_bootstrap_ci95", [0.0, 0.01]),
        ("per_seed", [0.1, 0.0, 0.1]),
        ("holm_m3_adjusted_p", 0.05),
    ):
        bad = {
            "gain_vs_r882": {
                arm: dict(good["gain_vs_r882"][arm]) for arm in stat.FRESH_ARMS
            }
        }
        bad["gain_vs_r882"]["ENS20"][key] = value
        assert stat.verdict(bad)["verdict"] == "NO_ADVANCE"


def test_timing_rule_uses_three_folds_only_when_full_projection_exceeds_four_hours():
    assert stat.execution_plan(0.5)["outer_folds"] == 5
    reduced = stat.execution_plan(1.2)
    assert reduced == {
        "outer_folds": 3,
        "updates_per_member": 40,
        "projected_seconds": 9540.0,
        "within_cap": True,
    }
    too_slow = stat.execution_plan(2.0)
    assert too_slow["outer_folds"] == 3 and too_slow["updates_per_member"] == 40
    assert too_slow["within_cap"] is False
    with pytest.raises(ValueError):
        stat.execution_plan(0)


def test_context_utilities_use_minus_two_floor():
    utility = stat.context_utility([[1.0, 2.0, 3.0]], [[True, False, True]])
    assert utility.tolist() == [[-1.0, -2.0, -3.0]]
