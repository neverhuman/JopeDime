"""Synthetic H200 protocol tests; these never open cohort or model data."""

from pathlib import Path

import numpy as np
import pytest

import analysis as stat
import runner
import monitor


def test_registered_seeds_and_fit_plan_are_fresh_and_exact():
    assert stat.SEEDS == (200101, 200102, 200103)
    assert tuple(stat.FOLD_SEEDS.values()) == (200141, 200142, 200143)
    outers, inners, calibrators = stat.fit_plan()
    assert len(outers) == 5 and len(inners) == 10 and len(calibrators) == 5
    for k in range(5):
        a, b, fit = stat.cell(k)
        assert tuple(sorted((a, b))) in outers
        assert all(set(stat.inner_model_for(k, j)) == {a, b, j} for j in fit)
    assert len(set(outers + inners + calibrators)) == 20


def test_lambda_choice_requires_strict_margin_and_breaks_exact_ties_low():
    zero = {lam: 0.0 for lam in stat.LAMBDA_GRID}
    assert stat.choose_lambda(zero) == 0.0
    gains = dict(zero)
    gains[0.1] = 0.001
    assert stat.choose_lambda(gains) == 0.0
    gains[0.1] = 0.002
    gains[0.2] = 0.002
    assert stat.choose_lambda(gains) == 0.1
    gains[0.2] = 0.00201
    assert stat.choose_lambda(gains) == 0.2


def test_rank_normalization_averages_ties_and_orients_low_cost_first():
    rank = stat.average_rank01(np.array([3.0, 1.0, 1.0, 2.0]))
    assert rank[1] == rank[2] == 1 / 6
    assert rank[0] == 1 and rank[3] == 2 / 3
    mixed = stat.rank_blend(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0]))
    assert np.array_equal(mixed, np.array([0.25, 0.5, 0.75]))


def test_permutation_preserves_score_multiset_and_is_seeded():
    scores = np.array([1.0, 1.0, 3.0, 4.0])
    a = stat.permute_scores(scores, stat.stream_seed(1, "inner", 0, "ctx"))
    b = stat.permute_scores(scores, stat.stream_seed(1, "inner", 0, "ctx"))
    assert np.array_equal(a, b)
    assert np.array_equal(np.sort(a), np.sort(scores))


def test_host_guard_rejects_xbabe0_2_3():
    for host in ("xbabe0", "xbabe2", "xbabe3"):
        with pytest.raises(RuntimeError, match="xbabe1"):
            runner.require_host(host)
        with pytest.raises(RuntimeError, match="xbabe1"):
            monitor.require_host(host)
    runner.require_host("xbabe1")


def test_wrong_host_run_rejects_before_any_file_read(monkeypatch):
    monkeypatch.setattr(runner.socket, "gethostname", lambda: "xbabe2")

    def forbidden(*args, **kwargs):
        raise AssertionError("file read happened before host refusal")

    monkeypatch.setattr(Path, "open", forbidden)
    with pytest.raises(RuntimeError, match="xbabe1"):
        runner.run(Path("/not/read"), Path("/not/read"))


def test_wrong_host_monitor_rejects_before_any_path_access(monkeypatch):
    import sys

    monkeypatch.setattr(monitor.socket, "gethostname", lambda: "xbabe3")
    monkeypatch.setattr(sys, "argv", ["monitor.py", "--output", "/no/write",
                                      "--root", "/no/write", "--launch", "/no/read"])

    def forbidden(*args, **kwargs):
        raise AssertionError("path access happened before host refusal")

    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    with pytest.raises(RuntimeError, match="xbabe1"):
        monitor.main()


def test_native_contrast_ties_are_not_gains():
    values = np.tile(np.arange(60, dtype=np.float64), (3, 1))
    families = [f"f{i:03}" for i in range(60)]
    c = stat.contrast(values, values.copy(), families)
    assert c["median_gain"] == 0.0
    assert c["positive_seeds"] == 0
    assert stat.verdict(c, 1.0)["verdict"] == "NO_ADVANCE"


def test_monitor_enforces_resource_caps():
    sample = dict(load=1.0, cpu=10, available=20 * 2**30, free=50_000_000_000,
                  disk_used_fraction=0.5, rss=1, tree_cores=1.0)
    assert monitor.violations(sample, "xbabe1") == []
    sample["rss"] = 6_000_000_000
    assert "aggregate RSS>=6GB" in monitor.violations(sample, "xbabe1")
