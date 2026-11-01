"""H179 statistical and leakage controls; fixtures are synthetic and data-free."""

import numpy as np
import pytest

import analysis as stat


def test_fold_partitions_are_family_disjoint_and_assess_each_once():
    families = {f"f{i:03}" for i in range(205)}
    folds = stat.family_folds(families)
    assert [len(group) for group in folds] == [41] * 5
    seen = []
    for outer in range(5):
        fit, b, a = stat.partition(families, outer, folds)
        assert (len(fit), len(b), len(a)) == (123, 41, 41)
        assert not fit & (a | b) and not a & b
        seen.extend(a)
    assert len(seen) == len(set(seen)) == 205


def test_member_seeds_are_distinct_per_outer_seed_and_ens3_is_nested():
    per_seed = [stat.fresh_member_seeds(seed) for seed in stat.SEEDS]
    assert per_seed == [tuple(range(179201, 179206)), tuple(range(179301, 179306)), tuple(range(179401, 179406))]
    assert len(set(sum((list(x) for x in per_seed), []))) == 15
    fit = {f"fit-{i}" for i in range(123)}
    ensembles = stat.family_fresh_subsets(fit)
    assert ensembles["FRESH1"] == ensembles["ENS3"] == ensembles["ENS5"] == fit


def test_score_averaging_precedes_argmin_and_ens3_is_member_subset():
    # Choice-level voting would select candidate 0 here, while mean scores select 1.
    scores = np.array([[[0.0, 3.0]], [[2.0, 1.0]], [[2.0, 1.0]], [[9.0, 0.0]], [[8.0, 0.0]]])
    ens3 = stat.score_ensemble(scores, 3)
    ens5 = stat.score_ensemble(scores, 5)
    assert ens3.shape == (1, 2)
    assert ens3[0].tolist() == pytest.approx([4 / 3, 5 / 3])
    assert ens5[0].tolist() == pytest.approx([4.2, 1.0])
    assert int(np.argmin(ens3[0])) == 0
    assert int(np.argmin(ens5[0])) == 1
    with pytest.raises(ValueError):
        stat.score_ensemble(scores, 2)


def test_fit_only_batch_plan_never_contains_a_or_b_contexts():
    fit = {f"fit-{i}" for i in range(123)}
    a = {f"a-{i}" for i in range(41)}
    b = {f"b-{i}" for i in range(41)}
    ids = sorted(fit)
    orders = {cid: np.arange(48) for cid in ids}
    plan = stat.batch_plan(ids, orders, 1234)
    assert len(plan) == 40
    for _, chosen, indexes in plan:
        assert set(chosen) <= fit and not set(chosen) & (a | b)
        assert all(len(ix) == 24 and np.array_equal(ix[:4], np.arange(4)) for ix in indexes)


def test_holm_and_verdict_require_all_strict_primary_conditions():
    assert stat.holm_adjust([0.01, 0.02, 0.04]).tolist() == pytest.approx([0.03, 0.04, 0.04])
    base = {
        "gain_vs_r882": {
            "ENS5": {"median_over_seeds": 0.002, "family_bootstrap_ci95": [0.001, 0.003], "per_seed": [0.001, 0.002, 0.003], "holm_m3_adjusted_p": 0.03}
        }
    }
    assert stat.verdict(base.copy())["verdict"] == "ADVANCE_CANDIDATE"
    for key, value in (("median_over_seeds", 0.001), ("family_bootstrap_ci95", [0.0, 0.01]), ("per_seed", [0.1, 0.0, 0.1]), ("holm_m3_adjusted_p", 0.05)):
        bad = {"gain_vs_r882": {"ENS5": {**base["gain_vs_r882"]["ENS5"], key: value}}}
        assert stat.verdict(bad)["verdict"] == "NO_ADVANCE"


def test_pilot_budget_accounts_for_six_trajectories_and_three_hour_cap():
    assert stat.execution_plan(0.1)["outer_folds"] == 5
    assert stat.execution_plan(10.0)["outer_folds"] == 3
    assert stat.execution_plan(40.0)["updates_per_member"] == 4
    assert stat.execution_plan(50.0)["updates_per_member"] == 0
    with pytest.raises(ValueError):
        stat.execution_plan(0)


def test_context_utilities_use_gate_and_minus_two_floor():
    utility = stat.context_utility([[1.0, 2.0, 3.0]], [[True, False, True]])
    assert utility.tolist() == [[-1.0, -2.0, -3.0]]
