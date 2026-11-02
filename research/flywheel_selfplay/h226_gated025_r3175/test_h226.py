"""H226 design/statistics/guard controls; all fixtures are synthetic and data-free (no checkpoint, corpus or model)."""

import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analysis as stat  # noqa: E402

_spec = importlib.util.spec_from_file_location("h226_test_monitor", HERE / "monitor.py")
monitor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(monitor)

_rspec = importlib.util.spec_from_file_location("h226_test_runner", HERE / "runner.py")
runner = importlib.util.module_from_spec(_rspec)
_rspec.loader.exec_module(runner)


def families(n=205):
    return [f"f{f:03}" for f in range(n)]


def test_cells_partition_folds_cyclically():
    for k in range(5):
        a, b, fit = stat.cell(k)
        assert (a, b) == (k, (k + 1) % 5)
        assert sorted((a, b, *fit)) == list(range(5)) and len(fit) == 3


def test_fit_plan_is_minimal_and_covers_every_cell():
    outers, inners = stat.fit_plan()
    assert len(outers) == 5 and len(inners) == 10
    for k in range(5):
        a, b, fit = stat.cell(k)
        assert tuple(sorted((a, b))) in outers
        used = [stat.inner_model_for(k, j) for j in fit]
        assert len(set(used)) == 3 and all(m in inners for m in used)
        for j, m in zip(fit, used):
            assert {a, b, j} == set(m)
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
        groups = stat.seed_folds(fams + fams[:7], s)
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


def test_registered_fold_manifest_is_consistent_and_fresh():
    m = json.loads((HERE / "fold_manifest.json").read_text())
    assert m["wave"] == "H226"
    assert m["family_count"] == 205 and m["context_count"] == 1219
    assert m["fold_seeds"] == {str(s): stat.FOLD_SEEDS[s] for s in stat.SEEDS}
    fams = sorted(sum(m["families_by_seed"][str(stat.SEEDS[0])], []))
    for s in stat.SEEDS:
        assert m["families_by_seed"][str(s)] == stat.seed_folds(fams, s)
        assert sum(m["fold_context_counts_by_seed"][str(s)]) == 1219
    assert m["pairwise_fold_jaccard"] == stat.fold_jaccard(
        {s: m["families_by_seed"][str(s)] for s in stat.SEEDS}
    )
    for v in m["pairwise_fold_jaccard"].values():
        assert v["best_match"] <= stat.JACCARD_CAP
    expected_waves = {"H203", "H210", "H217", "H218", "H222", "H223", "H224", "H225"}
    assert set(m["cross_wave_fold_jaccard"]) == expected_waves
    for wave, pairs in m["cross_wave_fold_jaccard"].items():
        assert len(pairs) == 9
        assert all(v["best_match"] <= stat.JACCARD_CAP for v in pairs.values()), wave
    rows = [{"family_id": f} for f in fams]
    assert (
        stat.fold_manifest(rows)["cross_wave_fold_jaccard"]
        == m["cross_wave_fold_jaccard"]
    )
    reused = {"H217": {"1": stat.seed_folds(fams, stat.SEEDS[0])}}
    with pytest.raises(ValueError, match="distinct"):
        stat.fold_manifest(rows, reused)


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
        free=465_000_000_000,
        disk_used_fraction=0.93,
        rss=1_000_000_000,
        threads=5,
        tree_cores=3.9,
        worker_affinity_cores=4,
    )
    s.update(kw)
    return s


def test_monitor_limits_and_worker_cpu_selection():
    assert monitor.HOST_CORES == len(monitor.os.sched_getaffinity(0))
    assert monitor.WORKER_CORES == 4 and monitor.THREADS == 4
    assert monitor.CORE_LIMIT == 4.5 and monitor.RSS_MAX == 4 * 2**30
    assert monitor.MIN_FREE == 300_000_000_000 and monitor.ADMIT_FREE == 320_000_000_000
    assert monitor.MEM_MIN == 20 * 2**30 and monitor.ADMIT_MEM == 24 * 2**30
    assert monitor.LOAD5_MAX == 60.0 and monitor.LOAD5_RESUME == 55.0
    assert monitor.RUN_END == 1793644200.0
    cpus = monitor.select_worker_cpus([0, 1, 2, 3, 4, 5], 4)
    assert len(cpus) == 4 and set(cpus) <= {0, 1, 2, 3, 4, 5}


def test_monitor_hard_violations_and_load_pause_hysteresis():
    assert monitor.violations(ok_sample()) == []
    for kw in [
        dict(tree_cores=4.6),
        dict(worker_affinity_cores=5),
        dict(available=19 * 2**30),
        dict(free=299_000_000_000),
        dict(rss=4 * 2**30),
    ]:
        assert len(monitor.violations(ok_sample(**kw))) == 1
    # Host load pauses rather than aborting:
    high_load = ok_sample(load5=61.0)
    assert monitor.violations(high_load) == []
    assert monitor.overloaded(high_load, False) is True
    assert monitor.overloaded(ok_sample(load5=56.0), True) is True  # stays paused > 55
    assert monitor.overloaded(ok_sample(load5=55.0), True) is False  # resumes at <= 55


def test_monitor_admission_needs_a_quiet_host_and_margins():
    assert monitor.admission_refusals(ok_sample()) == []
    assert monitor.admission_refusals(ok_sample(load=61.0)) == ["admission: load1>60"]
    assert monitor.admission_refusals(ok_sample(load5=56.0)) == ["admission: load5>55"]
    assert monitor.admission_refusals(ok_sample(cpu=51.0)) == [
        "admission: host CPU>50%"
    ]
    assert monitor.admission_refusals(ok_sample(available=23 * 2**30)) == [
        "admission: MemAvailable<24GiB"
    ]
    assert monitor.admission_refusals(ok_sample(free=319_000_000_000)) == [
        "admission: disk free<320GB"
    ]


def test_attempt_deadline_is_capped_at_1830z_and_inherited_by_resume(tmp_path):
    assert monitor.attempt_deadline(tmp_path, False, 1000.0) == 1000.0 + 4 * 3600
    assert (
        monitor.attempt_deadline(tmp_path, False, monitor.RUN_END - 1800.0)
        == monitor.RUN_END
    )
    with pytest.raises(RuntimeError, match="attempt claim"):
        monitor.attempt_deadline(tmp_path, True, 2000.0)
    (tmp_path / "H226_attempt.claim").write_text(
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
    assert set(runner.HOST_RULES) == {"xbabe2"}
    monkeypatch.setattr(runner.socket, "gethostname", lambda: host)
    monkeypatch.setattr(runner.os, "getpriority", lambda *a: 19)
    root = runner.DATA / "run-h226-host-admission-test"
    missing = runner.DATA / "no-such-launch.json"
    with pytest.raises(RuntimeError, match="xbabe2"):
        runner.run(root, missing)
    with pytest.raises(RuntimeError, match="xbabe2"):
        runner.guards(root)
    assert not root.exists()


def test_runner_admits_xbabe2_at_nice_19_only_and_only_under_the_monitor(monkeypatch):
    monkeypatch.setattr(runner.socket, "gethostname", lambda: "xbabe2.local")
    monkeypatch.setattr(runner.os, "getpriority", lambda *a: 19)
    assert runner.admit_host() == "xbabe2"
    monkeypatch.delenv(runner.DEADLINE_ENV, raising=False)
    root = runner.DATA / "run-h226-deadline-test"
    with pytest.raises(RuntimeError, match="deadline missing"):
        runner.run(root, runner.DATA / "no-such-launch.json")
    assert not root.exists()
    monkeypatch.setattr(runner.os, "getpriority", lambda *a: 10)
    with pytest.raises(RuntimeError):
        runner.admit_host()


def test_runner_prereg_sha_matches_prereg_md():
    assert runner.sha(HERE / "PREREG.md") == runner.PREREG_SHA


def test_runner_wait_for_load_sleeps_until_load5_resumes(monkeypatch):
    loads = iter([(70.0, 62.0, 50.0), (65.0, 58.0, 50.0), (50.0, 54.0, 50.0)])
    monkeypatch.setattr(runner.os, "getloadavg", lambda: next(loads))
    slept = []
    monkeypatch.setattr(runner.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setenv(runner.DEADLINE_ENV, repr(runner.time.time() + 3600))
    runner.wait_for_load()
    assert slept == [60.0]


def test_blend_is_h129_cost_interpolation_with_fixed_lambda():
    ref, gbm = np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])
    assert np.allclose(stat.blend(ref, gbm), ref + 0.25 * (gbm - ref))
    assert stat.LAMBDAS == {"BLEND_025": 0.25}
    for bad in (0.15, 0.35, 0.5):
        with pytest.raises(ValueError):
            stat.blend(ref, gbm, bad)


def test_zgap_is_outcome_free_and_zero_when_picks_agree():
    r = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert stat.zgap(r, r + 0.1) == 0.0
    b = r.copy()
    b[3] = -1.0
    assert stat.zgap(r, b) == pytest.approx((3.0 - 0.0) / np.std(r))
    assert stat.zgap(np.ones(8), np.arange(8.0)[::-1]) == math.inf


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
    fams, s, r, b = [], [], [], []
    for f in range(40):
        for _ in range(3):
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
    assert all(x <= c["tau"] for x, f in zip(s, fams) if int(f[1:]) < 20)
    assert all(x > c["tau"] for x, f in zip(s, fams) if int(f[1:]) >= 20)


def test_choose_tau_falls_back_to_base_when_nothing_is_safe():
    s, r, b, fams = gate_fixture()
    worse = [x - 0.5 for x in b]
    c = stat.choose_tau(s, r, worse, fams)
    assert c["tau"] == -math.inf and c["fit_gain"] == 0.0 and c["fit_harm"] == 0.0
    same = stat.choose_tau(s, r, r, fams)
    assert same["tau"] == -math.inf


def test_arm_costs_gate_falls_back_to_respective_base_and_is_outcome_free():
    base_882 = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    base_3175 = np.array([1.0, 0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    bases = {"R882": base_882, "R3175": base_3175}
    gbm = np.array([9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, -30.0])
    closed = {arm: -math.inf for arm in stat.GATES}
    costs, opened, signal = runner.arm_costs(stat, bases, gbm, closed)
    assert not any(opened.values())
    assert signal["R3175"]["zgap"] > 0 and signal["R882"]["zgap"] > 0
    assert np.array_equal(costs["G3175"], base_3175)
    assert np.array_equal(costs["G3175_DIS"], base_3175)
    assert np.array_equal(costs["G882"], base_882)
    assert int(np.argmin(costs["B3175"])) == 7 and int(np.argmin(costs["B882"])) == 7
    costs, opened, _ = runner.arm_costs(
        stat, bases, gbm, {arm: math.inf for arm in stat.GATES}
    )
    assert all(opened.values())
    assert np.array_equal(costs["G3175"], costs["B3175"])
    assert np.array_equal(costs["G3175_DIS"], costs["B3175"])
    assert np.array_equal(costs["G882"], costs["B882"])
    assert set(costs) == set(stat.ARMS) and stat.PRIMARY == "G3175"


def val_fixture(shift, noise=0.0, seed=1):
    rng = np.random.default_rng(seed)
    fams = [f"v{i % 19:02}" for i in range(91)]
    ref = rng.normal(1.1, 0.3, size=91)
    return ref + shift + rng.normal(0, noise, size=91), ref, fams


def test_verdict_requires_primary_and_both_r882_and_r3175_plateau_gates():
    arm, base, f = panel(0.05)
    c = stat.contrast(arm, base, f)
    g3175_val, r882_val, vf = val_fixture(0.01, noise=0.001)
    r3175_val = r882_val + 0.004  # G3175 still beats R3175 by ~+0.006 across families
    p_r882 = stat.plateau(g3175_val, r882_val, vf)
    p_r3175 = stat.plateau(g3175_val, r3175_val, vf)
    v = stat.verdict(c, p_r882, p_r3175)
    assert v["verdict"] == "CONFIRMED: G3175 + PLATEAU_WIN"
    assert set(v["plateau_gates"]) == {
        "vs_r882_gain_gt_0002",
        "vs_r882_lcb95_gt_0",
        "vs_r3175_lcb95_gt_0",
    }
    # If G3175 equals R3175 on DL-Val (winner's curse of R3175 alone), vs_r3175_lcb95_gt_0 is False -> NO_PLATEAU_WIN!
    p_same_as_3175 = stat.plateau(g3175_val, g3175_val.copy(), vf)
    v_curse = stat.verdict(c, p_r882, p_same_as_3175)
    assert v_curse["verdict"] == "CONFIRMED: G3175 + NO_PLATEAU_WIN"
    assert v_curse["plateau_gates"]["vs_r3175_lcb95_gt_0"] is False


def test_top1_flips_diagnostic():
    ref = np.zeros((15, 91), dtype=np.int64)
    arm = np.zeros((15, 91), dtype=np.int64)
    arm[:, :3] = 1  # 3 contexts flip in all 15 cells
    arm[:8, 3] = 2  # 1 context flips in 8/15 cells (majority)
    arm[0, 4] = 3  # 1 context flips in 1/15 cells
    flips = stat.top1_flips(arm, ref)
    assert flips["contexts_all_cells"] == 3
    assert flips["contexts_majority_cells"] == 4
    assert flips["contexts_any_cell"] == 5
    assert flips["total_contexts"] == 91
    assert len(flips["per_cell_counts"]) == 15


def test_frozen_gate_recipe_matches_h210_and_h217():
    assert stat.FIT_HARM_TARGET == 0.05 and stat.LAMBDA == 0.25
    assert stat.HARM_DELTA == -0.01 and stat.HARM_CAP == 0.10 and stat.DRAWS == 10_000
    h210 = (HERE.parent / "h210_blend025_gated_retry" / "analysis.py").read_text()
    h217 = (HERE.parent / "h217_gated025_confirm" / "analysis.py").read_text()
    h226 = (HERE / "analysis.py").read_text()
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
            return body.replace("H226", "H210").replace("H217", "H210")

        assert block(h226) == block(h210) == block(h217), name


def test_closeout_readme_reconciles_metrics_resources_and_custody_replay():
    readme = (HERE / "README.md").read_text()
    metrics_path = HERE / "metrics.json"
    custody_path = HERE / "custody.json"
    assert (
        runner.sha(metrics_path)
        == "5d74d13ebf1e11984bd99058848a26d6297f75287de90bf8d5fd5d043ee3d8f3"
    )
    assert (
        runner.sha(custody_path)
        == "cd5e5822a0d7770166f12cf8871c5911f2a24f1c80c7102953acf1681b9aa3d2"
    )
    m = json.loads(metrics_path.read_text())
    val = m["descriptive_base_comparison_from_saved_smoke"][
        "dl_val_91_contexts_19_families"
    ]
    assert val["r3175_vs_r882_native_equal_family_contrast"]["ci95"] == [
        -0.00645137774296966,
        0.002692315338785731,
    ]
    assert val["r3175_vs_r882_native_equal_family_contrast"]["harm_families"] == 1
    assert val["r3175_vs_r882_plateau"]["ci95"] == [
        -0.0066668236369076,
        0.00290733477716234,
    ]
    for token in (
        "[-0.006451, +0.002692]",
        "5.26%` (`1/19`)",
        "[0.000000, +0.000245]",
        "1502.41` `[1497.52, 1507.75]",
        "1077.789049994899 s",
        "300.79945918009616 s",
        "977,838,080 B",
        "1.1441300444495006",
        "59,802,333,184 B",
        "445,921,017,856 B",
        "911.93 s",
        "6YSADvHFQJgfL43IueLP0yr7kyRF5RXP",
        "069fdd50340c8a0234cafeae9ee0e96701a5b19482965625bfdcbb445260f85b",
    ):
        assert token in readme, token
