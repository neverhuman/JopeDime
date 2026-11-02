"""Data-free scientific boundary and CPU supervisor controls for H224."""

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import types
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(
        "h224_test_" + name, HERE / (name + ".py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


stat, gate, runner, monitor = (
    load(n) for n in ["analysis", "gate", "runner", "monitor"]
)


def test_reference_function_bodies_match_h217():
    m = json.loads((HERE / "manifest.json").read_text())
    old = subprocess.check_output(
        [
            "git",
            "-C",
            "/home/ubuntu/JopeDime",
            "show",
            m["parent_commit"]
            + ":research/flywheel_selfplay/h217_gated025_confirm/analysis.py",
        ]
    )

    def functions(raw):
        return {
            n.name: ast.dump(n, include_attributes=False)
            for n in ast.parse(raw).body
            if isinstance(n, ast.FunctionDef)
        }

    a, b = functions(old), functions((HERE / "analysis.py").read_text())
    for name in [
        "blend",
        "zgap",
        "tau_grid",
        "gated",
        "choose_tau",
        "family_matrix",
        "elo_vs_reference",
    ]:
        assert a[name] == b[name], name


def test_features_use_costs_only_and_known_coordinates():
    r = np.array([0.0, 2.0, 3.0])
    p = np.array([9.0, -9.0, 1.0])
    b = stat.blend(r, p)
    x, switch = gate.features(r, p, b, stat.zgap)
    assert switch and np.allclose(x, [stat.zgap(r, b), 2.0, 0.5, 18.0, 3.0])
    assert set(__import__("inspect").signature(gate.features).parameters) == {
        "reference",
        "prior",
        "blended",
        "zgap",
    }


def test_constant_cost_gap_is_finite_and_stable_ties():
    x, switch = gate.features(
        [0.0, 0.0, 0.0], [5.0, -5.0, 0.0], [1.0, -1.0, 0.0], stat.zgap
    )
    assert switch and x[0] == 1e6 and np.isfinite(x).all()
    x, switch = gate.features([0.0, 1.0], [0.0, 1.0], [0.0, 1.0], stat.zgap)
    assert not switch and x[2] == 0


@pytest.mark.parametrize("bad", [[np.nan, 0.0], [np.inf, 0.0]])
def test_nonfinite_features_refused(bad):
    with pytest.raises(ValueError):
        gate.features(bad, [0.0, 1.0], [0.0, 1.0], stat.zgap)


def fixture():
    rng = np.random.default_rng(42)
    families = np.repeat(np.arange(18), 3)
    groups = families % 3
    x = rng.normal(size=(54, 5))
    delta = np.where(x[:, 0] > 0, 0.1, -0.1)
    return x, delta, families, groups, np.ones(54, bool)


def test_gate_cv_only_fits_two_family_groups(monkeypatch):
    x, delta, fam, group, switch = fixture()
    original = gate.fit
    seen = []

    def spy(xx, yy, ff, c):
        seen.append((set(ff), c))
        return original(xx, yy, ff, c)

    monkeypatch.setattr(gate, "fit", spy)
    choice = gate.train(x, delta, fam, group, switch, 224101)
    assert len(seen) == 13 and choice["threshold_uses_gate_oof"]
    for i, (f, _) in enumerate(seen[:12]):
        assert all(k % 3 != i % 3 for k in f)
    assert seen[-1][0] == set(fam) and choice["C"] in gate.CS
    assert np.isfinite(gate.predict(choice["model"], x)).all()


def test_cv_refuses_family_alias():
    x, d, f, g, s = fixture()
    g[0] = (g[0] + 1) % 3
    with pytest.raises(ValueError, match="crosses"):
        gate.train(x, d, f, g, s, 1)


def test_placebo_preserves_each_family_label_multiset():
    y = np.arange(30) % 2
    f = np.repeat(np.arange(6), 5)
    shuf = gate.shuffled_labels(y, f, 22)
    assert not np.array_equal(y, shuf)
    for family in set(f):
        assert sorted(y[f == family]) == sorted(shuf[f == family])
    assert np.array_equal(y, np.arange(30) % 2)


@pytest.mark.parametrize("switch", [True, False])
def test_zero_gain_falls_back_closed(switch):
    x, d, f, g, s = fixture()
    choice = gate.train(x, np.zeros(len(d)), f, g, np.full(len(d), switch), 1)
    assert choice["threshold"] == "closed" and choice["C"] == 0.01
    assert not gate.opens(choice, x, s).any()


def test_threshold_is_strict():
    choice = dict(model=dict(constant=0.5), threshold=0.5)
    assert not gate.opens(choice, np.zeros((2, 5)), [True, True]).any()


def test_calibration_and_gate_labels_are_only_inner_fit_predictions():
    rows = [dict(context_sha256=str(k), family_id="f" + str(k)) for k in range(15)]
    mapping = {str(k): k % 5 for k in range(15)}
    base = {str(k): np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]) for k in range(15)}
    pools = {
        str(k): dict(
            fitness=np.full(6, 99999.0 if k % 5 in [0, 1] else float(k)),
            gate=np.ones(6, bool),
        )
        for k in range(15)
    }
    calls = []

    def preds(seed, excluded):
        assert len(excluded) == 3 and 0 in excluded and 1 in excluded
        calls.append(excluded)
        return {
            str(k): np.arange(6, dtype=float) for k in range(15) if k % 5 in excluded
        }

    def calibration(pred, fitness, gate0):
        assert 99999.0 not in fitness
        return 0.0, 1.0

    h139 = types.SimpleNamespace(
        native=lambda cost, fitness, gates: float(-fitness[np.argmin(cost)])
    )
    cell = runner.fit_cell(
        stat,
        gate,
        types.SimpleNamespace(calibration=calibration),
        h139,
        224101,
        0,
        rows,
        mapping,
        base,
        pools,
        preds,
    )
    assert (
        set(calls) == {(0, 1, 2), (0, 1, 3), (0, 1, 4)}
        and cell["fit_contexts"] == 9
        and cell["fit_family_count"] == 9
    )


def test_folds_fresh_and_overlap_bound():
    m = json.loads((HERE / "fold_manifest.json").read_text())
    groups = m["families_by_seed"]
    assert len(groups) == 3
    for folds in groups.values():
        assert sorted(map(len, folds)) == [41] * 5 and len(set(sum(folds, []))) == 205
    assert m["best_match_max"] <= 0.20 and set(m["cross_wave_fold_jaccard"]) == {
        "H203",
        "H210",
        "H217",
        "H218",
        "H223",
    }


def test_45_unique_fits_60_logical_uses():
    outer, inner = stat.fit_plan()
    assert len(outer) == 5 and len(inner) == 10
    uses = []
    for k in range(5):
        a, b, fit = stat.cell(k)
        uses.append(tuple(sorted((a, b))))
        for j in fit:
            uses.append(stat.inner_model_for(k, j))
    assert len(uses) * 3 == 60 and len(set(uses)) * 3 == 45


def test_primary_and_plateau_strict_boundaries():
    p = dict(ci95=[0.001, 0.02], positive_seeds=3, harm_fraction=0.1)
    assert stat.verdict(p, dict(win=True))["verdict"] == "CANDIDATE + PLATEAU_WIN"
    for bad in [
        dict(p, ci95=[0.0, 0.02]),
        dict(p, positive_seeds=2),
        dict(p, harm_fraction=0.100001),
    ]:
        assert stat.verdict(bad, dict(win=False))["primary"] == "NO_ADVANCE"
    r = np.zeros(91)
    f = np.array(["f" + str(i % 19) for i in range(91)])
    assert not stat.plateau(np.full(91, 0.002), r, f)["win"]


@pytest.mark.parametrize(
    "field,value", [("load", 61), ("free", 299_999_999_999), ("ram", 20 * 2**30 - 1)]
)
def test_resource_stops(monkeypatch, field, value):
    values = dict(load=0, free=400_000_000_000, ram=30 * 2**30)
    values[field] = value
    monkeypatch.setattr(monitor.os, "getloadavg", lambda: (0, values["load"], 0))
    monkeypatch.setattr(
        monitor.psutil,
        "disk_usage",
        lambda p: types.SimpleNamespace(free=values["free"]),
    )
    monkeypatch.setattr(
        monitor.psutil,
        "virtual_memory",
        lambda: types.SimpleNamespace(available=values["ram"]),
    )
    with pytest.raises(RuntimeError, match="resource stop"):
        monitor.resources()


def test_stage_lock_and_cpu_source_guards():
    source = (HERE / "runner.py").read_text()
    assert (
        source.index("store.put(lock)") < source.index("stage='score'")
        if "stage='score'" in source
        else source.index("store.put(lock)") < source.index('stage = "score"')
    )
    assert "threshold_uses_gate_oof" in (HERE / "gate.py").read_text()
    monitor_source = (HERE / "monitor.py").read_text()
    assert (
        "qualification.lock" not in monitor_source
        and "sched_setaffinity" in monitor_source
        and "SIGKILL" in monitor_source
    )


def test_plateau_accepts_only_strictly_above_representable_margin():
    families = np.array(["f" + str(i % 19) for i in range(91)])
    for margin, expected in [
        (np.nextafter(0.002, 0), False),
        (0.002, False),
        (np.nextafter(0.002, 1), True),
    ]:
        result = stat.plateau(np.full(91, margin), np.zeros(91), families)
        assert result["win"] == expected


def test_tree_rss_limit_is_strict(monkeypatch):
    monkeypatch.setattr(monitor.os, "getloadavg", lambda: (0, 0, 0))
    monkeypatch.setattr(monitor.os, "sched_getaffinity", lambda pid: {1, 2, 3, 4})
    monkeypatch.setattr(
        monitor.psutil,
        "disk_usage",
        lambda path: types.SimpleNamespace(free=400_000_000_000),
    )
    monkeypatch.setattr(
        monitor.psutil,
        "virtual_memory",
        lambda: types.SimpleNamespace(available=30 * 2**30),
    )
    parent = types.SimpleNamespace(
        pid=11,
        children=lambda recursive: [],
        is_running=lambda: True,
        memory_info=lambda: types.SimpleNamespace(rss=4 * 2**30),
        threads=lambda: [],
    )
    with pytest.raises(RuntimeError, match="resource stop"):
        monitor.resources(parent)
    parent.memory_info = lambda: types.SimpleNamespace(rss=4 * 2**30 - 1)
    assert monitor.resources(parent)["tree_rss"] == 4 * 2**30 - 1


def test_thread_affinity_escape_is_refused(monkeypatch):
    monkeypatch.setattr(monitor.os, "getloadavg", lambda: (0, 0, 0))
    monkeypatch.setattr(
        monitor.os,
        "sched_getaffinity",
        lambda pid: {1, 2, 3, 4} if pid == 11 else {1, 2, 3, 4, 5},
    )
    monkeypatch.setattr(
        monitor.psutil,
        "disk_usage",
        lambda path: types.SimpleNamespace(free=400_000_000_000),
    )
    monkeypatch.setattr(
        monitor.psutil,
        "virtual_memory",
        lambda: types.SimpleNamespace(available=30 * 2**30),
    )
    parent = types.SimpleNamespace(
        pid=11,
        children=lambda recursive: [],
        is_running=lambda: True,
        memory_info=lambda: types.SimpleNamespace(rss=1000),
        threads=lambda: [types.SimpleNamespace(id=12)],
    )
    with pytest.raises(RuntimeError, match="escaped"):
        monitor.resources(parent)


@pytest.mark.parametrize("valid", [True, False])
def test_s3_exact_version_and_bytes(tmp_path, valid):
    import io

    store = runner.Store.__new__(runner.Store)
    store.root = tmp_path
    store.prefix = "H224/"
    store.receipts = {}

    def put(**kw):
        assert kw["IfNoneMatch"] == "*"
        return dict(VersionId="v1")

    def get(**kw):
        assert kw["VersionId"] == "v1"
        return dict(VersionId="v1", Body=io.BytesIO(b"evidence" if valid else b"bad"))

    store.client = types.SimpleNamespace(put_object=put, get_object=get)
    path = tmp_path / "artifact"
    path.write_bytes(b"evidence")
    if valid:
        assert store.put(path)["version_id"] == "v1"
    else:
        with pytest.raises(RuntimeError, match="recovery mismatch"):
            store.put(path)
        assert not store.receipts


def test_exited_short_lived_child_does_not_abort_resource_sample(monkeypatch):
    monkeypatch.setattr(monitor.os, "getloadavg", lambda: (0, 0, 0))
    monkeypatch.setattr(monitor.os, "sched_getaffinity", lambda pid: {1, 2, 3, 4})
    monkeypatch.setattr(
        monitor.psutil,
        "disk_usage",
        lambda path: types.SimpleNamespace(free=400_000_000_000),
    )
    monkeypatch.setattr(
        monitor.psutil,
        "virtual_memory",
        lambda: types.SimpleNamespace(available=30 * 2**30),
    )

    def exited():
        raise monitor.psutil.NoSuchProcess(12)

    child = types.SimpleNamespace(
        pid=12, is_running=lambda: True, memory_info=exited, threads=exited
    )
    parent = types.SimpleNamespace(
        pid=11,
        children=lambda recursive: [child],
        is_running=lambda: True,
        memory_info=lambda: types.SimpleNamespace(rss=1000),
        threads=lambda: [],
    )
    assert monitor.resources(parent)["tree_rss"] == 1000
