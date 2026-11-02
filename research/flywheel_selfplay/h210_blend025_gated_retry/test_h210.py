"""H210 design/statistics controls; all fixtures are synthetic and data-free (no checkpoint, corpus or model)."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

import analysis as stat

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("h210_test_monitor", HERE / "monitor.py")
monitor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(monitor)


def _shared_h203_test_helpers():
    path = HERE.parent / "h203_blend025_gated" / "test_h203.py"
    spec = importlib.util.spec_from_file_location("h203_shared_test_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def families(n=205):
    return [f"f{f:03}" for f in range(n)]


def test_cells_partition_folds_cyclically():
    for k in range(5):
        a, b, fit = stat.cell(k)
        assert (a, b) == (k, (k + 1) % 5)
        assert sorted((a, b, *fit)) == list(range(5)) and len(fit) == 3


def test_fit_plan_is_minimal_and_covers_every_cell():
    _shared_h203_test_helpers()._assert_fit_plan_is_minimal_and_covers_every_cell(stat)


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


def panel(shift, seed=0, fams=60, per=3, noise=0.05, seeds=3):
    rng = np.random.default_rng(seed)
    f = [f"f{i:02}" for i in range(fams) for _ in range(per)]
    base = np.tile(rng.normal(size=fams * per), (seeds, 1))
    arm = base + shift + rng.normal(0, noise, size=base.shape)
    return arm, base, f


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
        load5=1.0,
        load_over_s=0.0,
        cpu_over_s=0.0,
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
        monitor.violations(ok_sample()) == []
    )  # OpenMP helper threads alone are not a violation
    for kw in [
        dict(tree_cores=4.2),
        dict(available=15 * 2**30),
        dict(free=39_000_000_000),
        dict(disk_used_fraction=0.938),
        dict(rss=6_000_000_000),
        dict(cpu_over_s=300.0),
        dict(load_over_s=300.0),
        dict(load5=monitor.LOAD_LIMIT + 1),
    ]:
        assert len(monitor.violations(ok_sample(**kw))) == 1
    assert monitor.violations(ok_sample(disk_used_fraction=0.9379)) == []
    assert monitor.DEADLINE == 4 * 3600 and monitor.CORE_LIMIT == 4.0


def test_monitor_ignores_short_load_spikes_and_trips_on_sustained_overload():
    # H203 lost both attempts to one-minute spikes (load1 120.9 and 154.6 > 108.8); H210 needs 300 s or load5.
    spike = dict(load=monitor.LOAD_LIMIT + 50, cpu=99.0)
    since = {}
    s = monitor.track_overload(ok_sample(**spike), since, 1000.0)
    assert s["load_over_s"] == 0.0 and monitor.violations(s) == []
    s = monitor.track_overload(ok_sample(**spike), since, 1299.0)
    assert s["load_over_s"] == 299.0 and monitor.violations(s) == []
    s = monitor.track_overload(ok_sample(**spike), since, 1300.0)
    assert monitor.violations(s) == [
        "load1>85% of host online cores for 300 s",
        "host CPU>85% for 300 s",
    ]
    # a dip below the limit restarts the clock
    monitor.track_overload(ok_sample(), since, 1301.0)
    s = monitor.track_overload(ok_sample(**spike), since, 1302.0)
    assert s["load_over_s"] == 0.0 and s["cpu_over_s"] == 0.0


def test_monitor_admission_needs_a_quiet_host_and_disk_margin():
    assert monitor.admission_refusals(ok_sample()) == []
    half = monitor.ADMIT_LOAD
    assert monitor.admission_refusals(ok_sample(load=half + 1)) == [
        "admission: load1>50% of cores"
    ]
    assert monitor.admission_refusals(ok_sample(load5=half + 1)) == [
        "admission: load5>50% of cores"
    ]
    assert monitor.admission_refusals(ok_sample(cpu=51.0)) == [
        "admission: host CPU>50%"
    ]
    assert monitor.admission_refusals(ok_sample(disk_used_fraction=0.935)) == [
        "admission: disk used>=93.5%"
    ]
    assert monitor.ADMIT_LOAD == 0.5 * monitor.HOST_CORES and monitor.SUSTAIN == 300.0


@pytest.mark.parametrize("host", ["xbabe2", "xbabe3", "xbabe0"])
def test_monitor_rejects_excluded_hosts_before_any_side_effect(
    host, tmp_path, monkeypatch
):
    assert monitor.ALLOWED_HOSTS == ("xbabe1",)
    claims, out = tmp_path / "claims", tmp_path / "logs" / "run-x"
    monkeypatch.setattr(monitor, "CLAIMS", claims)
    monkeypatch.setattr(monitor.socket, "gethostname", lambda: host)
    monkeypatch.setattr(monitor.os, "getpriority", lambda *a: 19)
    launched = []
    monkeypatch.setattr(monitor.subprocess, "Popen", lambda *a, **k: launched.append(a))
    argv = [
        "monitor.py",
        "--output",
        str(out),
        "--root",
        str(tmp_path / "run-x"),
        "--launch",
        str(tmp_path / "l.json"),
    ]
    monkeypatch.setattr(monitor.sys, "argv", argv)
    with pytest.raises(RuntimeError, match="xbabe1"):
        monitor.main()
    assert not out.parent.exists() and not claims.exists() and not launched
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("host", ["xbabe2", "xbabe3", "xbabe0"])
def test_runner_rejects_excluded_hosts_before_any_read_or_root(host, monkeypatch):
    import runner

    assert set(runner.HOST_RULES) == {"xbabe1"}
    monkeypatch.setattr(runner.socket, "gethostname", lambda: host)
    monkeypatch.setattr(runner.os, "getpriority", lambda *a: 19)
    root = runner.DATA / "run-h199-host-admission-test"
    missing = runner.DATA / "no-such-launch.json"
    with pytest.raises(RuntimeError, match="xbabe1"):
        runner.run(
            root, missing
        )  # a missing launch file would raise differently if reached
    with pytest.raises(RuntimeError, match="xbabe1"):
        runner.guards(root)
    assert not root.exists()


def test_runner_admits_xbabe1_at_nice_19_only(monkeypatch):
    import runner

    monkeypatch.setattr(runner.socket, "gethostname", lambda: "xbabe1.local")
    monkeypatch.setattr(runner.os, "getpriority", lambda *a: 19)
    assert runner.admit_host() == "xbabe1"
    monkeypatch.setattr(runner.os, "getpriority", lambda *a: 10)
    with pytest.raises(RuntimeError):
        runner.admit_host()


def test_runner_guards_pass_on_xbabe1_with_healthy_resources(tmp_path, monkeypatch):
    _shared_h203_test_helpers()._assert_runner_guards_pass_on_xbabe1_with_healthy_resources(
        tmp_path, monkeypatch
    )


def test_blend_is_h129_cost_interpolation_with_fixed_lambda():
    ref, gbm = np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])
    assert np.allclose(stat.blend(ref, gbm), ref + 0.25 * (gbm - ref))
    assert stat.LAMBDAS == {"BLEND_025": 0.25}
    for bad in (0.15, 0.35, 0.5):
        with pytest.raises(ValueError):
            stat.blend(ref, gbm, bad)


def test_zgap_is_outcome_free_and_zero_when_picks_agree():
    r = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert stat.zgap(r, r + 0.1) == 0.0  # same first argmin
    b = r.copy()
    b[3] = -1.0  # the blend picks slot 3, which r882 ranks 4th
    assert stat.zgap(r, b) == pytest.approx((3.0 - 0.0) / np.std(r))
    assert (
        stat.zgap(np.ones(8), np.arange(8.0)[::-1]) == math.inf
    )  # constant r882: gate stays closed


def test_disagreement_matches_h129_definition():
    r = np.arange(10.0)
    assert stat.disagreement(r, r) == pytest.approx(0.0)
    assert stat.disagreement(-r, r) == pytest.approx(2.0)
    assert stat.disagreement(np.ones(10), r) == math.inf


def test_tau_grid_spans_closed_quantiles_and_open():
    g = stat.tau_grid([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, math.inf])
    assert g[0] == -math.inf and g[-1] == math.inf and len(g) == 7
    assert g[1:-1] == pytest.approx(
        np.quantile([1.0, 2.0, 3.0, 4.0], [0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
    )
    assert stat.tau_grid([0.0, 0.0]) == [-math.inf, math.inf]


def gate_fixture():
    """40 families x 3 contexts. Low-zgap switches help (+0.2); high-zgap switches hurt (-0.3)."""
    fams, s, r, b = [], [], [], []
    for f in range(40):
        for i in range(3):
            fams.append(f"f{f:02}")
            r.append(1.0)
            if f < 20:
                s.append(0.1 + 0.01 * f)
                b.append(1.2)
            else:
                s.append(2.0 + 0.01 * f)
                b.append(0.7)
    return s, r, b, fams


def test_choose_tau_admits_helpful_low_risk_switches_only():
    s, r, b, fams = gate_fixture()
    c = stat.choose_tau(s, r, b, fams)
    assert c["fit_harm"] <= stat.FIT_HARM_TARGET and c["fit_gain"] > 0
    assert all(
        x <= c["tau"] for x, f in zip(s, fams) if int(f[1:]) < 20
    )  # every helpful switch admitted
    assert all(
        x > c["tau"] for x, f in zip(s, fams) if int(f[1:]) >= 20
    )  # every harmful switch refused
    always = next(x for x in c["grid"] if x["tau"] == math.inf)
    assert always["harm"] == 0.5  # the ungated blend would harm half the families


def test_choose_tau_falls_back_to_r882_when_nothing_is_safe_and_ties_go_conservative():
    s, r, b, fams = gate_fixture()
    worse = [x - 0.5 for x in b]  # every switch now harms
    c = stat.choose_tau(s, r, worse, fams)
    assert c["tau"] == -math.inf and c["fit_gain"] == 0.0 and c["fit_harm"] == 0.0
    same = stat.choose_tau(s, r, r, fams)  # no effect anywhere: all taus tie at gain 0
    assert same["tau"] == -math.inf


def test_arm_costs_gate_falls_back_to_r882_and_is_outcome_free():
    _shared_h203_test_helpers()._assert_arm_costs_gate_falls_back_to_r882_and_is_outcome_free(
        stat
    )


def test_jsonable_encodes_infinite_thresholds_only():
    import runner

    assert runner.jsonable({"t": [-math.inf, 1.5, math.inf]}) == {
        "t": ["-inf", 1.5, "inf"]
    }
    with pytest.raises(ValueError):
        json.dumps(runner.jsonable({"x": math.nan}), allow_nan=False)


def test_verdict_needs_lcb_p_and_harm():
    arm, base, f = panel(0.05)
    c = stat.contrast(arm, base, f)
    assert c["ci95"][0] > 0 and c["p_one_sided_vs_margin"] < 0.05
    assert stat.verdict(c)["verdict"] == "CANDIDATE: GATED_025"
    assert set(stat.verdict(c)["gates"]) == {"lcb95_gt_0", "p_lt_005", "harm_le_10pct"}
    tie = stat.contrast(
        base.copy(), base, f
    )  # identical picks: exact ties are zero gain
    assert tie["median_gain"] == 0.0 and stat.verdict(tie)["verdict"] == "NO_ADVANCE"
    small = stat.contrast(
        *panel(0.0005, noise=0.0001)[:2], f
    )  # positive, but not past the +0.001 margin
    assert small["ci95"][0] > 0 and stat.verdict(small)["verdict"] == "NO_ADVANCE"


def test_runner_guard_uses_load5_and_the_938_disk_rule(tmp_path, monkeypatch):
    import collections

    import runner

    monkeypatch.setattr(runner.socket, "gethostname", lambda: "xbabe1")
    monkeypatch.setattr(runner.os, "getpriority", lambda *a: 19)
    monkeypatch.setattr(runner, "MEM_MIN", 0)
    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(
        runner.shutil, "disk_usage", lambda p: usage(10**13, 9300 * 10**9, 700 * 10**9)
    )
    monkeypatch.setattr(runner.os, "getloadavg", lambda: (10_000.0, 1.0, 1.0))
    assert (
        runner.guards(tmp_path / "absent")["load5"] == 1.0
    )  # a load1 spike alone passes
    monkeypatch.setattr(runner.os, "getloadavg", lambda: (1.0, 10_000.0, 1.0))
    with pytest.raises(RuntimeError, match="load5"):
        runner.guards(tmp_path / "absent")
    monkeypatch.setattr(runner.os, "getloadavg", lambda: (1.0, 1.0, 1.0))
    monkeypatch.setattr(
        runner.shutil, "disk_usage", lambda p: usage(10**13, 9381 * 10**9, 619 * 10**9)
    )
    with pytest.raises(RuntimeError, match="disk guard"):
        runner.guards(tmp_path / "absent")
