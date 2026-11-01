"""H176 statistical/design controls; all fixtures are synthetic and data-free."""

import numpy as np
import pytest

import analysis as stat


def rows(n_families=205):
    return [
        {"context_sha256": f"c{i:04}", "family_id": f"f{i // 3:03}"}
        for i in range(n_families * 3)
    ]


def test_five_fold_family_crossfit_is_disjoint_and_assesses_each_family_once():
    source = rows()
    families = {r["family_id"] for r in source}
    groups = stat.family_folds(families)
    assert [len(g) for g in groups] == [41] * 5
    assessed = []
    for outer in range(5):
        fit, select, assess = stat.partition(families, outer, groups)
        assert (len(fit), len(select), len(assess)) == (123, 41, 41)
        assert not fit & select and not fit & assess and not select & assess
        assessed += list(assess)
    assert len(assessed) == len(set(assessed)) == 205
    with pytest.raises(ValueError):
        stat.family_folds(families - {"f000"})


def test_nested_family_subsets_are_deterministic_seeded_and_fit_only():
    source = rows()
    fit, select, assess = stat.partition({r["family_id"] for r in source}, 0)
    subsets = stat.nested_family_subsets(fit, stat.stream_seed(176101, 0, "subset"))
    assert [len(subsets[n]) for n in (31, 62, 123)] == [31, 62, 123]
    assert subsets[31] < subsets[62] < subsets[123]
    assert subsets == stat.nested_family_subsets(fit, stat.stream_seed(176101, 0, "subset"))
    assert subsets != stat.nested_family_subsets(fit, stat.stream_seed(176102, 0, "subset"))
    for subset in subsets.values():
        assert subset <= fit and not subset & select and not subset & assess
        cids = stat.contexts_for_families(source, subset)
        assert len(cids) == 3 * len(subset)


def test_oracle_utility_and_zero_update_gap_are_equal_family_and_gate_aware():
    train = [
        {"context_sha256": "a", "family_id": "f1"},
        {"context_sha256": "b", "family_id": "f1"},
        {"context_sha256": "c", "family_id": "f2"},
    ]
    val = [{"context_sha256": "v", "family_id": "vf"}]
    actual = stat.context_utility([[1, 4], [2, 3], [5, 6]], [[1, 1], [1, 0], [0, 0]])
    assert actual.tolist() == [[-1.0, -4.0], [-2.0, -2.0], [-2.0, -2.0]]
    oracle = np.array([-0.5, -0.5, -1.0])
    gap = stat.zero_update_gap(train, [-1.0, -2.0, -2.0], oracle, val, [-1.5], [-0.5])
    assert gap["train"]["equal_family_utility"] == pytest.approx(-1.75)
    assert gap["train"]["minus2_contexts"] == 2
    assert gap["generalization_gap_train_minus_val"] == pytest.approx(-0.25)
    assert gap["oracle_headroom_train"] == pytest.approx(1.0)
    assert gap["oracle_headroom_normalized_gap"] == pytest.approx(-0.25)


def test_curve_paired_differences_slope_verdict_and_elo_use_family_resamples():
    families = [f"f{i:03}" for i in range(41)]
    panel = np.zeros((3, 41, 5), dtype=float)
    panel[:, :, 1] = 0.0
    panel[:, :, 2] = 0.002
    panel[:, :, 3] = 0.006
    panel[:, :, 4] = 0.004
    result = stat.learning_curve_summary(families, panel)
    assert result["verdict"] == "DATA_LIMITED"
    assert result["paired_differences"]["50_to_100"]["median_over_seeds"] == pytest.approx(0.004)
    assert result["slope_per_log2_doubling"]["median_over_seeds"] == pytest.approx(0.003)
    assert result["elo_vs_r882"]["FRESH_100"]["median_over_seeds"] > 1500
    assert result["bootstrap_draws"] == 10_000
    panel[:, :, 2:] = 0.0
    result = stat.learning_curve_summary(families, panel)
    assert result["verdict"] == "DATA_SATURATED"
    with pytest.raises(ValueError):
        stat.learning_curve_summary(families, panel[:2])


def test_pilot_budget_is_fixed_by_timing_only_and_respects_time_cap():
    assert stat.pilot_budget(0.5, folds=5) == 40
    assert stat.pilot_budget(3.0, folds=5) == 40
    assert stat.pilot_budget(100.0, folds=3) == 2
    assert stat.execution_plan(4.0) == {"outer_folds": 5, "updates_per_cell": 40, "projected_seconds": 10500.0}
    assert stat.execution_plan(5.0)["outer_folds"] == 3
    assert stat.execution_plan(10.0)["updates_per_cell"] == 27
    assert stat.execution_plan(100.0)["updates_per_cell"] == 0
    with pytest.raises(ValueError):
        stat.pilot_budget(0.0)


def test_h156_batch_plan_contains_only_fit_ids_and_fixed_candidate_recipe():
    cids = [f"c{i:03}" for i in range(80)]
    orders = {cid: np.arange(48) for cid in cids}
    plan = stat.batch_plan(cids, orders, stat.stream_seed(176101, 0, "25%"))
    assert len(plan) == 40
    reversed_plan = stat.batch_plan(list(reversed(cids)), orders, stat.stream_seed(176101, 0, "25%"))
    for (stream, chosen, indexes), (stream2, chosen2, indexes2) in zip(plan, reversed_plan):
        assert stream == stream2 and chosen == chosen2 and len(chosen) == 8
        assert set(chosen) <= set(cids)
        for ix, ix2 in zip(indexes, indexes2):
            assert np.array_equal(ix, ix2) and len(ix) == 24
            assert np.array_equal(ix[:4], np.arange(4))
            assert len(set(ix[4:])) == 20 and min(ix[4:]) >= 4


def test_h148_elo_uses_context_pair_outcomes_before_equal_family_weighting():
    family_ids = ["large", "large", "small"]
    contexts = np.zeros((3, 5))
    contexts[:, 1:] = np.array([0.0002, -0.00005, -0.0002])[:, None]
    family_panels = np.array([[0.0, 0.000075, 0.000075, 0.000075, 0.000075], [0.0, -0.0002, -0.0002, -0.0002, -0.0002]])
    family_panels = np.repeat(family_panels[None, :, :], 3, axis=0)
    context_panels = np.repeat(contexts[None, :, :], 3, axis=0)
    result = stat.learning_curve_summary(["large", "small"], family_panels, context_panels=context_panels, context_families=family_ids)
    assert result["elo_vs_r882"]["FRESH_25"]["median_over_seeds"] != pytest.approx(
        stat.learning_curve_summary(["large", "small"], family_panels)["elo_vs_r882"]["FRESH_25"]["median_over_seeds"]
    )
