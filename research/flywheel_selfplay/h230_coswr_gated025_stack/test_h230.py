"""H230 data-free schedule, FIT boundary, stack, estimator and resource tests."""

import ast
import importlib.util
import io
import json
from pathlib import Path
import types

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(
        "h230_test_" + name, HERE / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stat, gate, runner, monitor = [
    load(x) for x in ["analysis", "gate", "runner", "monitor"]
]


def test_frozen_h217_gate_bodies():
    for name in ["blend", "zgap", "choose_tau", "gated"]:
        assert getattr(stat, name) is getattr(stat.parent, name)


def test_coswr_exact_frozen_schedule():
    recipe = gate.recipe()
    expected = [
        1e-4 * (u + 1) / 4
        if u < 4
        else 0.5e-4 * (1 + np.cos(np.pi * ((u - 4) % (40 / 3)) / (40 / 3)))
        for u in range(40)
    ]
    np.testing.assert_allclose(
        [recipe.coswr_learning_rate(u) for u in range(40)], expected, rtol=0, atol=1e-18
    )
    assert (
        recipe.coswr_learning_rate(0) == 2.5e-5
        and recipe.coswr_learning_rate(3) == 1e-4
    )
    with pytest.raises(ValueError):
        recipe.coswr_learning_rate(40)


def test_control_stream_is_fit_only_and_restores_rng():
    r = gate.recipe()
    ids = ["a", "b", "c"]
    pools = {c: dict(pids=np.arange(256), fitness=np.arange(256)) for c in ids}
    orders = {c: np.arange(256) for c in ids}
    np.random.seed(123)
    expected = np.random.random(5)
    np.random.seed(123)
    plan = r.build_control_or_coswr_batch_plan(ids, pools, orders, 230101)
    assert np.array_equal(np.random.random(5), expected)
    assert len(plan) == 40 and all(len(s["microbatches"]) == 1 for s in plan)
    for step in plan:
        mb = step["microbatches"][0]
        assert len(mb["context_ids"]) == 16 and set(mb["context_ids"]) <= set(ids)
        assert all(
            len(ix) == 192 and np.array_equal(ix[:48], np.arange(48))
            for ix in mb["candidate_indices"]
        )
    other = r.build_control_or_coswr_batch_plan(ids, pools, orders, 230102)
    assert (
        plan[0]["microbatches"][0]["candidate_pool_sha256"]
        != other[0]["microbatches"][0]["candidate_pool_sha256"]
    )


def test_frozen_targets_and_loss_gradient():
    import torch

    r = gate.recipe()
    f = torch.tensor([[-7.0, 3.0, 9.0]])
    g = torch.tensor([[True, True, False]])
    target = r.h59.preregistered_targets(f, g)
    assert torch.equal(target, torch.tensor([[-4.0, 2.0, 2.0]]))
    pred = torch.tensor([[0.1, -0.2, 0.3]], requires_grad=True)
    loss = r.per_context_losses(pred, target).mean()
    loss.backward()
    assert (
        torch.isfinite(loss)
        and torch.isfinite(pred.grad).all()
        and pred.grad.abs().sum() > 0
    )


def test_lazy_predictor_matches_dense_features_and_reveals():
    import torch

    r = gate.recipe()
    pool = dict(
        program=np.arange(8 * 412, dtype=np.float32).reshape(8, 412) / 100,
        proj16=np.ones((8, 16), np.float32),
        fitness=np.arange(8, dtype=float),
        gate=np.ones(8, bool),
    )

    class Adapter:
        def predict_costs(self, **kw):
            assert torch.equal(
                kw["revealed_raw_f"], torch.tensor([[6.0, 2.0, 7.0, 1.0]])
            )
            return dict(total_cost=kw["prog_412"].sum(-1) + kw["proj_16"].sum(-1))

    net = types.SimpleNamespace(adapter_10tok=Adapter())
    tokens = dict(a=torch.zeros(10, 32))
    masks = dict(a=torch.ones(10, dtype=torch.bool))
    indices = [np.array([6, 2, 7, 1, 0])]
    dense = r.h48.make_predictor(net, dict(a=pool), tokens, masks, torch.device("cpu"))(
        ["a"], indices
    )
    reader = types.SimpleNamespace(rows=lambda *a: pool["proj16"])
    h = types.SimpleNamespace(program=lambda pools, c: pool["program"])
    lazy = dict(a=gate.LazyPool(dict(a=pool), "a", h, (reader, dict(a=(0, 8)), None)))
    actual = r.h48.make_predictor(net, lazy, tokens, masks, torch.device("cpu"))(
        ["a"], indices
    )
    assert torch.equal(dense, actual)


def test_fit_cell_never_reads_a_or_b_labels():
    rows = [dict(context_sha256=str(i), family_id=str(i)) for i in range(5)]
    mapping = {str(i): i for i in range(5)}
    base = {str(i): np.arange(6, dtype=float) for i in range(5)}

    class BoundPool(dict):
        def __getitem__(self, k):
            if dict.__getitem__(self, "id") in ["0", "1"] and k in ["fitness", "gate"]:
                raise AssertionError("A/B label opened")
            return dict.__getitem__(self, k)

    pools = {
        str(i): BoundPool(id=str(i), fitness=np.arange(6), gate=np.ones(6, bool))
        for i in range(5)
    }
    calls = []

    def train_cell(seed, outer, groups, m):
        calls.append(tuple(groups))
        return base

    controller = types.SimpleNamespace(
        train_cell=train_cell, receipts={(230101, 0): dict(updates=40)}
    )
    h124 = types.SimpleNamespace(calibration=lambda *a: (0.0, 1.0))
    h139 = types.SimpleNamespace(
        native=lambda cost, fit, valid: float(-fit[np.argmin(cost)])
    )

    def preds(seed, excluded):
        return {str(i): -np.arange(6, dtype=float) for i in range(5)}

    result = runner.fit_cell(
        stat, controller, h124, h139, 230101, 0, rows, mapping, base, pools, preds
    )
    assert (
        calls == [(2, 3, 4)]
        and result["FIT"] == [2, 3, 4]
        and set(result["choices"]) == {"G025", "STACK"}
    )


def test_stack_changes_reference_and_closed_fallback():
    base = np.array([0.0, 2.0, 4.0])
    coswr = np.array([2.0, 0.0, 4.0])
    prior = np.array([4.0, 4.0, -9.0])
    controller = types.SimpleNamespace(predictions=lambda *a: dict(c=coswr))
    cell = dict(
        seed=230101,
        outer=0,
        choices={a: dict(tau=float("-inf")) for a in ["G025", "STACK"]},
    )
    got = runner.costs_for(stat, controller, base, prior, cell, "c")
    assert np.array_equal(got["G025"], base) and np.array_equal(got["STACK"], coswr)
    for v in cell["choices"].values():
        v["tau"] = float("inf")
    got = runner.costs_for(stat, controller, base, prior, cell, "c")
    assert np.array_equal(got["G025"], stat.blend(base, prior)) and np.array_equal(
        got["STACK"], stat.blend(coswr, prior)
    )


def test_all_scoring_calls_supply_context_id():
    tree = ast.parse((HERE / "runner.py").read_text())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "costs_for"
    ]
    assert len(calls) == 2 and all(len(n.args) == 6 for n in calls)


@pytest.mark.parametrize(
    "change",
    [
        dict(median_gain=0.002),
        dict(median_gain=0.002 + 5e-13),
        dict(ci95=[0, 0.01]),
        dict(harm_fraction=0.10001),
        dict(per_seed_gain=[0.01, 0, 0]),
    ],
)
def test_primary_rejects_ties_or_failed_guards(change):
    result = dict(
        median_gain=0.004,
        ci95=[0.001, 0.008],
        harm_fraction=0.1,
        per_seed_gain=[0.004, 0.005, -0.001],
    )
    result.update(change)
    assert stat.verdict(result, dict(win=True))["verdict"] == "NO_ADVANCE"


def test_both_endpoints_required():
    primary = dict(
        median_gain=0.004,
        ci95=[0.001, 0.008],
        harm_fraction=0.1,
        per_seed_gain=[0.004, 0.005, -0.001],
    )
    assert stat.verdict(primary, dict(win=True))["verdict"] == "ADVANCE"
    assert stat.verdict(primary, dict(win=False))["verdict"] == "NO_ADVANCE"


def test_plateau_exact_and_near_ties_reject():
    fam = [str(i % 19) for i in range(91)]
    for gain in [0.002, 0.002 + 5e-13]:
        assert not stat.plateau(np.full(91, gain), np.zeros(91), fam)["win"]
    assert stat.plateau(np.full(91, 0.00201), np.zeros(91), fam)["win"]


def test_fit_inventory_and_fresh_families():
    pairs, triples = stat.fit_plan()
    assert len(pairs) == 5 and len(triples) == 10
    families = [str(i) for i in range(205)]
    folds = [stat.seed_folds(families, s) for s in stat.SEEDS]
    assert len({json.dumps(x) for x in folds}) == 3 and all(
        sorted(sum(x, [])) == sorted(families) for x in folds
    )


def test_load_pause_hysteresis_keeps_deadline():
    p = monitor.LoadPause()
    assert p.update(60, 0) is None
    assert p.update(60.1, 1) == "pause"
    assert p.update(50, 60) is None
    assert p.update(55.1, 61) is None
    assert p.update(55, 121) == "resume"
    assert monitor.DEADLINE == 1793676600


@pytest.mark.parametrize("kind", ["rss", "ram", "disk", "output"])
def test_hard_resource_stops(monkeypatch, tmp_path, kind):
    monkeypatch.setattr(
        monitor.psutil,
        "disk_usage",
        lambda _: types.SimpleNamespace(
            free=299_999_999_999 if kind == "disk" else 400_000_000_000
        ),
    )
    monkeypatch.setattr(
        monitor.psutil,
        "virtual_memory",
        lambda: types.SimpleNamespace(
            available=20 * 2**30 - 1 if kind == "ram" else 40 * 2**30
        ),
    )
    monkeypatch.setattr(
        monitor.psutil, "swap_memory", lambda: types.SimpleNamespace(free=0)
    )
    monkeypatch.setattr(monitor.os, "getloadavg", lambda: (90, 90, 90))
    process = None
    if kind == "rss":
        process = types.SimpleNamespace(
            children=lambda recursive: [],
            memory_info=lambda: types.SimpleNamespace(rss=4 * 2**30),
        )
    if kind == "output":
        with (tmp_path / "sparse").open("wb") as f:
            f.truncate(2_000_000_000)
    with pytest.raises(RuntimeError):
        monitor.resources(process, tmp_path)


def test_s3_exact_version_and_hash(tmp_path, monkeypatch):
    f = tmp_path / "a.json"
    f.write_bytes(b"payload")
    store = runner.Store.__new__(runner.Store)
    store.root = tmp_path
    store.prefix = "test/"
    store.receipts = {}

    class Client:
        def put_object(self, **kw):
            assert kw["IfNoneMatch"] == "*"
            return dict(VersionId="v1")

        def get_object(self, **kw):
            assert kw["VersionId"] == "v1"
            return dict(VersionId="wrong", Body=io.BytesIO(b"payload"))

    store.client = Client()
    with pytest.raises(RuntimeError, match="recovery"):
        store.put(f)
