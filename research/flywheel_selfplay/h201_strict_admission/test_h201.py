"""Data-free H201 host, family inference and counterfactual replay controls."""

import numpy as np
import pytest

import analysis as a
import monitor as m
import runner


@pytest.mark.parametrize("host", ["xbabe0", "xbabe3", "unknown"])
def test_excluded_host_rejected_before_input_read(monkeypatch, host):
    monkeypatch.setattr(m.socket, "gethostname", lambda: host)
    monkeypatch.setattr(runner.Path, "read_bytes", lambda _: pytest.fail("input read"))
    monkeypatch.setattr(runner.Path, "read_text", lambda _: pytest.fail("input read"))
    with pytest.raises(RuntimeError, match="excludes this host"):
        runner.main()


def test_allowed_hosts_and_host_specific_disk_guard():
    sample = dict(
        cpu=5,
        available=16 * 2**30,
        rss=1e9,
        output=100,
        elapsed=5,
        cores=2,
        free=50e9,
        used_fraction=0.937,
    )
    assert m.admit_host("xbabe1") == "xbabe1"
    assert m.admit_host("xbabe2") == "xbabe2"
    assert m.violations(sample, "xbabe1") == []
    assert m.violations(sample, "xbabe2") == ["xbabe2_disk93.6"]
    sample.update(cores=3, rss=4e9, output=200e6, elapsed=5401)
    assert {"cores2", "RSS4GB", "output200MB", "wall90min"} <= set(
        m.violations(sample, "xbabe1")
    )


def test_native_parent_imports_keep_their_own_statistics_namespace():
    base, endpoint = runner.native_modules()
    assert base.stat.__file__.endswith("h189_swa_ema/analysis.py")
    assert (
        endpoint.native(
            np.array([0.0, 1.0]), np.array([0.5, 0.2]), np.array([True, True])
        )
        == -0.5
    )


def test_family_mean_does_not_weight_large_families_more():
    names, values = a.family_panel([[1, 1, 1, 9]], ["A", "A", "A", "B"])
    assert names == ["A", "B"]
    assert values.mean() == 5
    with pytest.raises(ValueError):
        a.family_panel([[np.nan, 1]], ["A", "B"])


def test_bootstrap_is_paired_normalized_reproducible_and_seeded():
    x = a.weights(19, 201101, draws=100)
    np.testing.assert_allclose(x.sum(axis=1), 1)
    np.testing.assert_array_equal(x, a.weights(19, 201101, draws=100))
    assert not np.array_equal(x, a.weights(19, 201102, draws=100))
    assert np.all(x @ np.zeros(19) == 0)


def test_sma_rejects_ties_and_harm_accepts_uniform_improvement():
    boot = a.weights(19, 201150)
    zero = np.zeros(19)
    assert a.strict_accept(zero, zero, boot) == (False, 0.0)
    assert not a.strict_accept(zero - 0.1, zero, boot)[0]
    assert a.strict_accept(zero + 0.1, zero, boot)[0]


def test_sma_is_counterfactual_and_observed_screen_is_separate():
    # An actual tie can update the observed incumbent even when SMA rejected it.
    # The next proposal is judged against the retained SMA state, not that update.
    boot = a.weights(3, 201150)
    noisy = np.array([0.4, -0.1, -0.1])
    assert not a.strict_accept(noisy, np.zeros(3), boot)[0]
    summary, decisions = a.sma_replay(
        [1, 2],
        [noisy, [0.01, 0.01, 0.01]],
        np.zeros(3),
        ["accept_tie", "reject_regression"],
    )
    assert decisions[1]["SMA_accept"]
    assert not decisions[1]["observed_incumbent_screen_accept"]
    assert summary["SMA_final_admitted_round"] == 2
    assert not summary["H2"]


def test_joint_null_preserves_duplicate_candidate_dependence():
    one = np.array([[0.1, -0.1, 0.0]])
    r1 = a.null_max(one, np.zeros(3))
    r2 = a.null_max(np.repeat(one, 8, axis=0), np.zeros(3))
    assert r1["expected_max"] == r2["expected_max"]
    assert r1["ci95"] == r2["ci95"]


def test_split_selection_uses_whole_disjoint_families_and_baseline_ties():
    result = a.split_selection(
        np.zeros((2, 19)), np.zeros(19), [3100, 3175], list(map(str, range(19)))
    )
    assert not result["all_three_B_positive"]
    for row in result["per_seed"]:
        assert len(row["A_families"]) == 9 and len(row["B_families"]) == 10
        assert not set(row["A_families"]) & set(row["B_families"])
        assert row["selected_round"] == 882 and row["B_gain"] == 0


def test_bt_reference_and_symmetric_ties():
    np.testing.assert_allclose(a.bt_ratings(np.ones((3, 3)) - np.eye(3)), 1500)


def test_verdict_conjunction_and_missing_coverage():
    assert a.verdict(0, 0.2, False, 1490) == "WINNERS_CURSE"
    assert a.verdict(0.01, 0.01, True, 1501) == "CANDIDATE_REAL"
    assert a.verdict(0.01, 0.01, False, 1501) == "INCONCLUSIVE"
    assert a.verdict(0.01, 0.01, True, 1500) == "INCONCLUSIVE"
    assert a.verdict(-0.1, 0.2, False, 1400, complete=False) == "INCONCLUSIVE"
