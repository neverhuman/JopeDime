"""H195 design/statistics controls; all fixtures are synthetic and data-free (no checkpoint, corpus or model)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import analysis as stat

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("h195_test_monitor", HERE / "monitor.py")
monitor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(monitor)


def families(n=205):
    return [f"f{f:03}" for f in range(n)]


def test_cells_partition_folds_cyclically():
    for k in range(5):
        a, b, fit = stat.cell(k)
        assert (a, b) == (k, (k + 1) % 5)
        assert sorted((a, b, *fit)) == list(range(5)) and len(fit) == 3


def test_fit_plan_is_minimal_and_covers_every_cell():
    outers, inners = stat.fit_plan()
    assert (
        len(outers) == 5 and len(inners) == 10
    )  # all adjacent pairs, all C(5,3) triples
    for k in range(5):
        a, b, fit = stat.cell(k)
        assert tuple(sorted((a, b))) in outers
        used = [stat.inner_model_for(k, j) for j in fit]
        assert len(set(used)) == 3 and all(m in inners for m in used)
        for j, m in zip(fit, used):
            assert {a, b, j} == set(
                m
            )  # inner model never saw A, B or its own calibration fold
    with pytest.raises(ValueError):
        stat.inner_model_for(0, 1)
    budget = sum(len(stat.FIT_SEEDS[t]) * 15 for t in stat.FIT_SEEDS)
    assert budget == 60 and stat.FIT_SEEDS["BLEND"] == stat.SEEDS


def test_seed_folds_reproduce_h124_algorithm_and_differ_by_seed():
    fams = families()
    for s in stat.SEEDS:
        groups = stat.seed_folds(fams + fams[:7], s)  # duplicates collapse to families
        expected = [
            sorted(x.tolist())
            for x in np.array_split(
                np.random.default_rng(stat.FOLD_SEEDS[s]).permutation(sorted(fams)), 5
            )
        ]
        assert groups == expected and all(len(g) == 41 for g in groups)
    assert len({json.dumps(stat.seed_folds(fams, s)) for s in stat.SEEDS}) == 3
    with pytest.raises(ValueError):
        stat.seed_folds(fams[:-1], stat.SEEDS[0])


def test_registered_fold_manifest_is_consistent():
    m = json.loads((HERE / "fold_manifest.json").read_text())
    assert m["family_count"] == 205 and m["context_count"] == 1219
    assert m["fold_seeds"] == {str(s): stat.FOLD_SEEDS[s] for s in stat.SEEDS}
    fams = sorted(
        m["families_by_seed"][str(stat.SEEDS[0])][0]
        + sum(m["families_by_seed"][str(stat.SEEDS[0])][1:], [])
    )
    for s in stat.SEEDS:
        assert m["families_by_seed"][str(s)] == stat.seed_folds(fams, s)
        assert sum(m["fold_context_counts_by_seed"][str(s)]) == 1219
    assert m["pairwise_fold_jaccard"] == stat.fold_jaccard(
        {s: m["families_by_seed"][str(s)] for s in stat.SEEDS}
    )
    for v in m["pairwise_fold_jaccard"].values():
        assert v["best_match"] < 0.5  # independent partitions


def test_shuffle_within_family_keeps_family_multisets():
    rng = np.random.default_rng(0)
    y = rng.normal(size=600).astype(np.float32)
    fam = np.repeat([f"f{i}" for i in range(6)], 100)
    z = stat.shuffle_within_family(y, fam, 7)
    assert not np.array_equal(y, z)
    for f in set(fam.tolist()):
        assert np.array_equal(np.sort(y[fam == f]), np.sort(z[fam == f]))
    assert np.array_equal(z, stat.shuffle_within_family(y, fam, 7))
    assert stat.stream_seed(1, 2) == stat.stream_seed(1, 2) != stat.stream_seed(2, 1)


def test_blend_is_h129_cost_interpolation():
    ref, gbm = np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])
    assert np.allclose(stat.blend(ref, gbm), ref + 0.25 * (gbm - ref))
    for bad in [dict(lam=0.5), dict()]:
        if bad:
            with pytest.raises(ValueError):
                stat.blend(ref, gbm, **bad)
    with pytest.raises(ValueError):
        stat.blend(ref, np.array([np.nan, 1.0, 1.0]))
    with pytest.raises(ValueError):
        stat.blend(ref, gbm[:2])


def panel(shift, seed=0, fams=60, per=3, noise=0.05, seeds=3):
    rng = np.random.default_rng(seed)
    f = [f"f{i:02}" for i in range(fams) for _ in range(per)]
    base = np.tile(rng.normal(size=fams * per), (seeds, 1))
    arm = base + shift + rng.normal(0, noise, size=base.shape)
    return arm, base, f


def test_contrast_detects_gain_and_rejects_ties():
    arm, base, f = panel(0.05)
    c = stat.contrast(arm, base, f)
    assert (
        c["ci95"][0] > 0
        and c["p_one_sided_vs_margin"] < 0.05
        and c["positive_seeds"] == 3
    )
    assert stat.verdict(c)["verdict"] == "CANDIDATE: BLEND_025"
    tie = stat.contrast(*panel(0.0)[:2], f)
    assert stat.verdict(tie)["verdict"] == "NO_ADVANCE"
    small = stat.contrast(
        *panel(0.0005, noise=0.0001)[:2], f
    )  # above 0 but not above the +0.001 margin
    assert small["ci95"][0] > 0 and not stat.verdict(small)["gates"]["p_lt_005"]


def test_harm_gate_blocks_a_gain_with_harmed_families():
    arm, base, f = panel(0.05)
    arm[:, :21] -= 0.5  # 7 of 60 families (11.7%) clearly harmed
    c = stat.contrast(arm, base, f)
    assert c["harm_families"] == 7 and c["harm_fraction"] > 0.10
    assert stat.verdict(c)["verdict"] == "NO_ADVANCE"


def test_single_seed_descriptive_contrast_works():
    arm, base, f = panel(0.05, seeds=1)
    c = stat.contrast(arm, base, f)
    assert len(c["per_seed_gain"]) == 1 and c["median_gain"] > 0


def test_bt_elo_orientation():
    w = np.array([[0.0, 5.0], [5.0, 0.0]])
    assert np.allclose(stat.bt_ratings(w), 1500)
    arm, base, f = panel(0.05)
    e = stat.elo_vs_reference(arm, base, f)
    assert e["median"] > 1500 and e["ci95"][0] > 1500
    e2 = stat.elo_vs_reference(base, arm, f)
    assert e2["median"] < 1500


def ok_sample(**kw):
    s = dict(
        load=1.0,
        cpu=10.0,
        available=64 * 2**30,
        free=100_000_000_000,
        disk_used_fraction=0.93,
        rss=1_000_000_000,
        threads=9,
        tree_cores=3.9,
    )
    s.update(kw)
    return s


def test_monitor_guards():
    assert (
        monitor.violations(ok_sample(), "xbabe2") == []
    )  # OpenMP helper threads alone are not a violation
    for kw in [
        dict(tree_cores=4.2),
        dict(available=15 * 2**30),
        dict(free=39_000_000_000),
        dict(disk_used_fraction=0.938),
        dict(rss=6_000_000_000),
        dict(cpu=86.0),
        dict(load=monitor.LOAD_LIMIT + 1),
    ]:
        assert len(monitor.violations(ok_sample(**kw), "xbabe2")) == 1
    assert (
        monitor.violations(ok_sample(disk_used_fraction=0.95), "xbabe1") == []
    )  # xbabe1 rule: free >= 40 GB
    assert monitor.DEADLINE == 4 * 3600 and monitor.CORE_LIMIT == 4.0
