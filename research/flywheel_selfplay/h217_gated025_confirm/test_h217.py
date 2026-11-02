"""H217 design/statistics/guard controls; all fixtures are synthetic and data-free (no checkpoint, corpus or model)."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

import analysis as stat

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("h217_test_monitor", HERE / "monitor.py")
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
    assert (
        budget == 45
        and stat.FIT_SEEDS["BLEND"] == stat.SEEDS
        and stat.FIT_SEEDS["SHUF"] == ()
    )


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
        tree_cores=4.4,
    )
    s.update(kw)
    return s


def test_monitor_limits_use_the_effective_cpu_affinity():
    # H210 E1: 102 CPUs allowed of 128 online; every limit is sized from the allowed set.
    assert monitor.HOST_CORES == len(monitor.os.sched_getaffinity(0))
    assert monitor.LOAD_LIMIT == 0.85 * monitor.HOST_CORES
    assert monitor.ADMIT_LOAD == 0.50 * monitor.HOST_CORES
    import runner

    assert runner.CORES == monitor.HOST_CORES


def test_monitor_guards():
    assert (
        monitor.violations(ok_sample()) == []
    )  # OpenMP helper threads up to 4.5 cores are not a violation
    for kw in [
        dict(tree_cores=4.6),
        dict(available=19 * 2**30),
        dict(free=39_000_000_000),
        dict(disk_used_fraction=0.938),
        dict(rss=6_000_000_000),
        dict(cpu_over_s=300.0),
        dict(load_over_s=300.0),
        dict(load5=monitor.LOAD_LIMIT + 1),
    ]:
        assert len(monitor.violations(ok_sample(**kw))) == 1
    assert monitor.violations(ok_sample(disk_used_fraction=0.9379)) == []
    assert monitor.DEADLINE == 4 * 3600 and monitor.CORE_LIMIT == 4.5
    assert monitor.WINDOW == 30.0 and monitor.MEM_MIN == 20 * 2**30


def test_monitor_ignores_short_load_spikes_and_trips_on_sustained_overload():
    # H203 lost both attempts to one-minute spikes on xbabe2; H217 needs 300 s or load5.
    spike = dict(load=monitor.LOAD_LIMIT + 50, cpu=99.0)
    since = {}
    s = monitor.track_overload(ok_sample(**spike), since, 1000.0)
    assert s["load_over_s"] == 0.0 and monitor.violations(s) == []
    s = monitor.track_overload(ok_sample(**spike), since, 1299.0)
    assert s["load_over_s"] == 299.0 and monitor.violations(s) == []
    s = monitor.track_overload(ok_sample(**spike), since, 1300.0)
    assert monitor.violations(s) == [
        "load1>85% of effective cores for 300 s",
        "host CPU>85% for 300 s",
    ]
    # a dip below the limit restarts the clock
    monitor.track_overload(ok_sample(), since, 1301.0)
    s = monitor.track_overload(ok_sample(**spike), since, 1302.0)
    assert s["load_over_s"] == 0.0 and s["cpu_over_s"] == 0.0


def test_monitor_admission_needs_a_quiet_host_and_margins():
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
    assert monitor.admission_refusals(ok_sample(available=23 * 2**30)) == [
        "admission: MemAvailable<24GiB"
    ]
    assert monitor.admission_refusals(ok_sample(disk_used_fraction=0.937)) == [
        "admission: disk used>=93.7%"
    ]
    assert monitor.SUSTAIN == 300.0


def test_attempt_deadline_is_absolute_and_inherited_by_the_resume(tmp_path):
    # H210 E1: the monitor clock restarted per invocation; H217 records one deadline in the attempt claim.
    assert monitor.attempt_deadline(tmp_path, False, 1000.0) == 1000.0 + 4 * 3600
    with pytest.raises(RuntimeError, match="attempt claim"):
        monitor.attempt_deadline(tmp_path, True, 2000.0)
    (tmp_path / "H217_attempt.claim").write_text(
        json.dumps({"deadline_epoch": 15400.0})
    )
    assert monitor.attempt_deadline(tmp_path, True, 9000.0) == 15400.0
    with pytest.raises(RuntimeError, match="passed"):
        monitor.attempt_deadline(tmp_path, True, 15400.0)


def monitor_argv(tmp_path):
    return [
        "monitor.py",
        "--output",
        str(tmp_path / "logs" / "run-x"),
        "--root",
        str(tmp_path / "run-x"),
        "--launch",
        str(tmp_path / "l.json"),
    ]


@pytest.mark.parametrize("host", ["xbabe1", "xbabe3", "xbabe0"])
def test_monitor_rejects_excluded_hosts_before_any_side_effect(
    host, tmp_path, monkeypatch
):
    assert monitor.ALLOWED_HOSTS == ("xbabe2",)
    claims = tmp_path / "claims"
    monkeypatch.setattr(monitor, "CLAIMS", claims)
    monkeypatch.setattr(monitor.socket, "gethostname", lambda: host)
    monkeypatch.setattr(monitor.os, "getpriority", lambda *a: 19)
    launched = []
    monkeypatch.setattr(monitor.subprocess, "Popen", lambda *a, **k: launched.append(a))
    monkeypatch.setattr(monitor.subprocess, "run", lambda *a, **k: launched.append(a))
    monkeypatch.setattr(monitor.sys, "argv", monitor_argv(tmp_path))
    with pytest.raises(RuntimeError, match="xbabe2"):
        monitor.main()
    assert not claims.exists() and not launched
    assert list(tmp_path.iterdir()) == []


def test_monitor_checks_the_source_binding_before_any_claim(tmp_path, monkeypatch):
    # H210 D1: the claim was written before bind_source failed. H217 binds first.
    import types

    claims = tmp_path / "claims"
    monkeypatch.setattr(monitor, "CLAIMS", claims)
    monkeypatch.setattr(monitor.socket, "gethostname", lambda: "xbabe2")
    monkeypatch.setattr(monitor.os, "getpriority", lambda *a: 19)
    calls, launched = [], []
    monkeypatch.setattr(
        monitor.subprocess,
        "run",
        lambda cmd, **k: (
            calls.append(cmd)
            or types.SimpleNamespace(
                returncode=1, stdout="", stderr="fatal: bad object"
            )
        ),
    )
    monkeypatch.setattr(monitor.subprocess, "Popen", lambda *a, **k: launched.append(a))
    monkeypatch.setattr(monitor.sys, "argv", monitor_argv(tmp_path))
    with pytest.raises(RuntimeError, match="source binding"):
        monitor.main()
    assert len(calls) == 1 and "--check-launch" in calls[0]
    assert not claims.exists() and not launched


@pytest.mark.parametrize("host", ["xbabe1", "xbabe3", "xbabe0"])
def test_runner_rejects_excluded_hosts_before_any_read_or_root(host, monkeypatch):
    import runner

    assert set(runner.HOST_RULES) == {"xbabe2"}
    monkeypatch.setattr(runner.socket, "gethostname", lambda: host)
    monkeypatch.setattr(runner.os, "getpriority", lambda *a: 19)
    root = runner.DATA / "run-h217-host-admission-test"
    missing = runner.DATA / "no-such-launch.json"
    with pytest.raises(RuntimeError, match="xbabe2"):
        runner.run(
            root, missing
        )  # a missing launch file would raise differently if reached
    with pytest.raises(RuntimeError, match="xbabe2"):
        runner.guards(root)
    assert not root.exists()


def test_runner_admits_xbabe2_at_nice_19_only_and_only_under_the_monitor(monkeypatch):
    import runner

    monkeypatch.setattr(runner.socket, "gethostname", lambda: "xbabe2.local")
    monkeypatch.setattr(runner.os, "getpriority", lambda *a: 19)
    assert runner.admit_host() == "xbabe2"
    monkeypatch.delenv(runner.DEADLINE_ENV, raising=False)
    root = runner.DATA / "run-h217-deadline-test"
    with pytest.raises(RuntimeError, match="deadline missing"):
        runner.run(root, runner.DATA / "no-such-launch.json")
    assert not root.exists()
    monkeypatch.setattr(runner.os, "getpriority", lambda *a: 10)
    with pytest.raises(RuntimeError):
        runner.admit_host()


def guarded_runner(
    monkeypatch, load=(1.0, 1.0, 1.0), disk=(10**13, 9300 * 10**9, 700 * 10**9)
):
    import collections

    import runner

    monkeypatch.setattr(runner.socket, "gethostname", lambda: "xbabe2")
    monkeypatch.setattr(runner.os, "getpriority", lambda *a: 19)
    monkeypatch.setattr(runner, "MEM_MIN", 0)
    monkeypatch.setattr(runner.os, "getloadavg", lambda: load)
    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda p: usage(*disk))
    return runner


def test_runner_guard_uses_effective_load5_the_938_disk_rule_and_the_deadline(
    tmp_path, monkeypatch
):
    runner = guarded_runner(monkeypatch, load=(10_000.0, 1.0, 1.0))
    monkeypatch.delenv(runner.DEADLINE_ENV, raising=False)
    g = runner.guards(tmp_path / "absent")
    assert (
        g["load5"] == 1.0 and g["effective_cores"] == runner.CORES
    )  # a load1 spike alone passes
    runner = guarded_runner(monkeypatch, load=(1.0, 0.85 * runner.CORES + 0.1, 1.0))
    with pytest.raises(RuntimeError, match="load5"):
        runner.guards(tmp_path / "absent")
    runner = guarded_runner(monkeypatch, disk=(10**13, 9381 * 10**9, 619 * 10**9))
    with pytest.raises(RuntimeError, match="disk guard"):
        runner.guards(tmp_path / "absent")
    runner = guarded_runner(monkeypatch)
    monkeypatch.setenv(runner.DEADLINE_ENV, "1.0")
    with pytest.raises(RuntimeError, match="deadline"):
        runner.guards(tmp_path / "absent")
    monkeypatch.setenv(runner.DEADLINE_ENV, repr(runner.time.time() + 3600))
    assert runner.guards(tmp_path / "absent")["deadline_epoch"] > runner.time.time()


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
    import runner

    base = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    gbm = np.array(
        [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, -30.0]
    )  # pulls the blend pick to slot 7
    closed = {arm: -math.inf for arm in stat.GATES}
    costs, opened, signal = runner.arm_costs(stat, base, gbm, closed)
    assert not any(opened.values()) and signal["zgap"] > 0
    assert all(np.array_equal(costs[arm], base) for arm in stat.GATES)
    assert int(np.argmin(costs["BLEND_025"])) == 7
    costs, opened, _ = runner.arm_costs(
        stat, base, gbm, {arm: math.inf for arm in stat.GATES}
    )
    assert all(opened.values()) and all(
        np.array_equal(costs[a], costs["BLEND_025"]) for a in stat.GATES
    )
    assert set(costs) == set(stat.ARMS) and stat.PRIMARY == "GATED_025"


def test_jsonable_encodes_infinite_thresholds_only():
    import runner

    assert runner.jsonable({"t": [-math.inf, 1.5, math.inf]}) == {
        "t": ["-inf", 1.5, "inf"]
    }
    with pytest.raises(ValueError):
        json.dumps(runner.jsonable({"x": math.nan}), allow_nan=False)


def val_fixture(shift, noise=0.0, seed=1):
    """91 contexts over 19 families, as DL-Val flywheel91."""
    rng = np.random.default_rng(seed)
    fams = [f"v{i % 19:02}" for i in range(91)]
    ref = rng.normal(1.1, 0.3, size=91)
    return ref + shift + rng.normal(0, noise, size=91), ref, fams


def test_plateau_win_needs_gain_above_0002_and_a_positive_lcb():
    arm, ref, fams = val_fixture(0.01, noise=0.001)
    p = stat.plateau(arm, ref, fams)
    assert p["win"] and p["gain"] > stat.PLATEAU_MARGIN and p["ci95"][0] > 0
    assert p["contexts"] == 91 and p["families"] == 19
    small = stat.plateau(*val_fixture(0.0015, noise=0.0001))
    assert small["ci95"][0] > 0 and not small["win"]  # positive, but not past +0.002
    tie = stat.plateau(ref.copy(), ref, fams)
    assert tie["gain"] == 0.0 and not tie["win"]  # ties are not a win
    with pytest.raises(ValueError):
        stat.plateau(arm[:90], ref, fams)


def test_plateau_bootstrap_resamples_families_not_contexts():
    arm, ref, fams = val_fixture(0.0)
    arm = arm.copy()
    arm[[i for i, f in enumerate(fams) if f == "v00"]] += (
        1.0  # one family carries all the gain
    )
    p = stat.plateau(arm, ref, fams)
    # a large mean gain carried by one family: dropping it is a common draw, so LCB95 = 0 and it is not a win
    assert p["gain"] > stat.PLATEAU_MARGIN and p["ci95"][0] == 0.0 and not p["win"]


def test_verdict_needs_lcb_all_three_seeds_and_harm():
    arm, base, f = panel(0.05)
    c = stat.contrast(arm, base, f)
    win = stat.plateau(*val_fixture(0.01, noise=0.001))
    lose = stat.plateau(*val_fixture(0.0))
    v = stat.verdict(c, win)
    assert v["verdict"] == "CONFIRMED: GATED_025 + PLATEAU_WIN"
    assert set(v["gates"]) == {"lcb95_gt_0", "all_3_seeds_positive", "harm_le_10pct"}
    assert stat.verdict(c, lose)["verdict"] == "CONFIRMED: GATED_025 + NO_PLATEAU_WIN"
    one_bad = dict(c, per_seed_gain=[0.05, 0.05, -0.001], positive_seeds=2)
    assert stat.verdict(one_bad, win)["primary"] == "NOT_CONFIRMED"
    tie = stat.contrast(
        base.copy(), base, f
    )  # identical picks: exact ties are zero gain
    assert (
        tie["median_gain"] == 0.0
        and stat.verdict(tie, lose)["primary"] == "NOT_CONFIRMED"
    )
    harmed = arm.copy()
    harmed[:, :21] -= 0.5  # 7 of 60 families (11.7%) clearly harmed
    h = stat.contrast(harmed, base, f)
    assert (
        h["harm_fraction"] > 0.10 and stat.verdict(h, win)["primary"] == "NOT_CONFIRMED"
    )


def test_frozen_gate_recipe_matches_h210():
    # No retuning: the gate statistic, grid, FIT harm target, lambda and decision margins are H210's.
    assert stat.GATES == {"GATED_025": "zgap", "GATED_025_DIS": "disagreement"}
    assert stat.FIT_HARM_TARGET == 0.05 and stat.LAMBDA == 0.25
    assert stat.HARM_DELTA == -0.01 and stat.HARM_CAP == 0.10 and stat.DRAWS == 10_000
    assert stat.ARMS == ("R882", "GATED_025", "BLEND_025", "GATED_025_DIS")
    h210 = (HERE.parent / "h210_blend025_gated_retry" / "analysis.py").read_text()
    for name in (
        "def zgap",
        "def disagreement",
        "def tau_grid",
        "def gated",
        "def choose_tau",
        "def blend",
    ):

        def block(text, name=name):
            body = text.split(name, 1)[1].split("\ndef ", 1)[0]
            return body.replace("H217", "H210")  # wave names in error messages only

        assert block((HERE / "analysis.py").read_text()) == block(h210), name


def test_fold_manifest_is_fresh_against_h203_and_h210():
    m = json.loads((HERE / "fold_manifest.json").read_text())
    assert set(m["cross_wave_fold_jaccard"]) == {"H203", "H210"}
    for wave, pairs in m["cross_wave_fold_jaccard"].items():
        assert len(pairs) == 9
        assert all(v["best_match"] < 0.5 for v in pairs.values()), wave
    fams = sorted(sum(m["families_by_seed"][str(stat.SEEDS[0])], []))
    earlier = {
        w: json.loads((HERE.parent / d / "fold_manifest.json").read_text())[
            "families_by_seed"
        ]
        for w, d in (
            ("H203", "h203_blend025_gated"),
            ("H210", "h210_blend025_gated_retry"),
        )
    }
    rows = [{"family_id": f} for f in fams]
    assert (
        stat.fold_manifest(rows, earlier)["cross_wave_fold_jaccard"]
        == m["cross_wave_fold_jaccard"]
    )
    reused = {"H210": {"1": stat.seed_folds(fams, stat.SEEDS[0])}}
    with pytest.raises(ValueError, match="distinct"):
        stat.fold_manifest(rows, reused)
