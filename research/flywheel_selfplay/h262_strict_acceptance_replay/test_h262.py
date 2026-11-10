"""Data-free fixtures only; no study pools, trained checkpoints or outcome reads."""

import ast
import copy
import json
import math
from pathlib import Path
import numpy as np
import pytest
import torch
import analysis as a
import engine
import monitor
import runner
from treatment import ControlAdamW

HERE = Path(__file__).parent


def test_current_ties_worsening_nonfinite_and_context_weighting():
    assert a.current_accept([1, 1], [1, 1])["accepted"]
    assert a.current_accept([1, 1.01], [1, 1])["accepted"]
    assert not a.current_accept([1, 0.99], [1, 1])["accepted"]
    # Context mean weights unequal-sized families exactly like the live rule.
    assert a.current_accept([2, 2, 0], [1, 1, 1])["accepted"]
    with pytest.raises(ValueError):
        a.current_accept([float("nan")], [0])
    with pytest.raises(ValueError):
        a.current_accept([float("inf")], [0])


def test_strict_margin_ties_nan_and_family_weighting():
    f = ["a", "a", "b"]
    assert not a.strict_accept([0, 0, 0], [0, 0, 0], f, 1)["accepted"]
    assert not a.strict_accept([0.002] * 3, [0] * 3, f, 1)["accepted"]
    assert a.strict_accept([0.003] * 3, [0] * 3, f, 1)["accepted"]
    assert not a.strict_accept([float("nan"), 0, 0], [0] * 3, f, 1)["accepted"]
    # Positive context mean cannot compensate for a harmed equally weighted family.
    assert not a.strict_accept([2, 2, -3], [0] * 3, f, 1)["accepted"]


def test_partition_and_paired_streams():
    rows = [dict(context_sha256=str(i), family_id=f"f{i // 2}") for i in range(410)]
    fit, accept = a.partition(rows)
    assert len({r["family_id"] for r in fit}) == 143
    assert len({r["family_id"] for r in accept}) == 62
    assert not {r["family_id"] for r in fit} & {r["family_id"] for r in accept}
    ids = {r["context_sha256"] for r in fit}
    orders = {c: np.arange(40) for c in ids}
    p1 = a.batch_plan(ids, orders, a.stream_seed(262101, 1))
    assert len(p1) == 40 and a.plan_digest(p1) == a.plan_digest(
        a.batch_plan(ids, orders, a.stream_seed(262101, 1))
    )
    assert a.plan_digest(p1) != a.plan_digest(
        a.batch_plan(ids, orders, a.stream_seed(262101, 2))
    )
    assert all(
        set(cids) <= ids
        and all(
            len(ix) == 24 and np.array_equal(ix[:4], np.arange(4)) for ix in indexes
        )
        for _, cids, indexes in p1
    )


def test_optimizer_exact_adamw_and_restore_state():
    torch.set_num_threads(1)
    p = torch.nn.Parameter(torch.tensor([0.2, -0.3]))
    q = torch.nn.Parameter(p.detach().clone())
    opt = ControlAdamW([p])
    ref = torch.optim.AdamW([q], lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)
    for step in range(40):
        g = torch.tensor([step / 40.0, -0.2])
        p.grad = g.clone()
        q.grad = g.clone()
        opt.step()
        ref.step()
    assert torch.equal(p, q)
    old = p.detach().clone()
    state = copy.deepcopy(opt.state_dict())
    p.grad = torch.ones_like(p)
    opt.step()
    with torch.no_grad():
        p.copy_(old)
    opt.load_state_dict(state)
    assert torch.equal(p, old) and int(opt.state[p]["step"]) == 40
    assert torch.equal(opt.state[p]["exp_avg"], ref.state[q]["exp_avg"])


def test_top4_h258_gated_cost_stable_ties_and_ungated():
    c = np.array([0, 0.001, 0.002, 0.003, 2.0], dtype=np.float32)
    fit = np.array([-0.2, -0.9, -0.9, -0.8, -10])
    gate = np.ones(5, dtype=bool)
    result = a.top4_pick(c, fit, gate)
    assert result["choice"] == 1 and result["reveals"] == 4
    gate[1] = False
    assert a.top4_pick(c, fit, gate)["choice"] == 2
    assert (
        a.top4_pick(np.array([0.0, 10.0, 11.0, 12.0, 13.0]), fit, gate)["reveals"] == 0
    )
    assert a.top4_pick(np.ones(5), fit, gate)["reveals"] == 0


def test_holm_primary_secondary_and_tie_panels():
    rows = [dict(family_id=f"f{i}") for i in range(19)]
    z = np.zeros((3, 19, 3))
    tie = a.summarize(rows, z)
    assert all(
        not c["advance"] for k, c in tie["contrasts"].items() if k.startswith("STRICT")
    )
    z[:, :, 1] = -0.01
    z[:, :, 2] = 0.01
    s = a.summarize(rows, z)
    assert s["contrasts"]["STRICT_LCB_minus_r882"]["advance"]
    assert s["contrasts"]["STRICT_LCB_minus_CURRENT"]["advance"]
    z[:, :, 2] = 0
    s = a.summarize(rows, z)
    assert not s["contrasts"]["STRICT_LCB_minus_r882"]["advance"]
    assert s["contrasts"]["STRICT_LCB_minus_CURRENT"]["advance"]
    assert s["median_elo"][0] == 1500


def sample(t=0, available=35, load=50):
    return dict(
        utc=t,
        available=available * 2**30,
        load5=load,
        rss=1000000,
        free=400000000000,
        threads=2,
        affinity=[48, 50, 51, 52],
    )


def test_admission_and_temporal_pause_guards():
    assert not monitor.admission_refusals(sample())
    assert monitor.admission_refusals(sample(available=29))
    g = monitor.Guard()
    assert g.update(sample(1, available=19))[1]
    assert g.update(sample(2, available=25))[1]
    assert g.update(sample(31, available=25))[1]
    assert not g.update(sample(32, available=25))[1]
    assert g.update(sample(40, load=109))[1]
    assert g.update(sample(41, load=101))[1]
    assert not g.update(sample(42, load=100))[1]
    assert monitor.violations(sample(monitor.COMPUTE_STOP))
    assert monitor.COMPUTE_STOP == 1793729700


def test_final_lock_rejects_missing_incomplete_and_mutation(tmp_path):
    with pytest.raises(FileNotFoundError):
        runner.assert_final_lock(tmp_path)
    f = tmp_path / "final.npz"
    f.write_bytes(b"fixture")
    lock = dict(
        completed_rounds=59, optimizer_updates=2400, files={f.name: engine.sha(f)}
    )
    engine.save(tmp_path / "final-lock.json", lock)
    with pytest.raises(RuntimeError):
        runner.assert_final_lock(tmp_path)
    lock["completed_rounds"] = 60
    engine.save(tmp_path / "final-lock.json", lock)
    assert runner.assert_final_lock(tmp_path)
    f.write_bytes(b"changed")
    with pytest.raises(RuntimeError):
        runner.assert_final_lock(tmp_path)


def test_no_intermediate_dlval_in_round_loop():
    tree = ast.parse((HERE / "runner.py").read_text())
    loops = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.For)
        and isinstance(n.target, ast.Name)
        and n.target.id == "round_idx"
    ]
    assert len(loops) == 1
    for node in ast.walk(loops[0]):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "evaluate"
        ):
            assert isinstance(node.args[0], ast.Name) and node.args[0].id == "accept"
    assert "assert_final_lock(out)" in (HERE / "runner.py").read_text()


def test_round_read_boundary_audit(tmp_path):
    pins = json.loads((HERE / "manifest.json").read_text())
    rt = engine.runtime(pins)
    allowed = "/home/ubuntu/jope-data/fixture/allowed.json"
    audit = rt.Audit({allowed}, tmp_path)
    audit.check("open", (allowed, "r", 0))
    with pytest.raises(RuntimeError):
        audit.check("open", (allowed, "w", 0))
    with pytest.raises(RuntimeError):
        audit.check("open", ("/home/ubuntu/jope-data/unlisted.json", "r", 0))
    with pytest.raises(RuntimeError):
        audit.check("open", ("/home/ubuntu/jope-data/holdout/a.json", "r", 0))


def test_monitor_affinity_matches_engine_and_runner():
    assert "[48, 50, 51, 52]" in (HERE / "engine.py").read_text()
    assert "[48, 50, 51, 52]" in (HERE / "monitor.py").read_text()
    # Formatting-independent AST comparison of the explicit runner affinity contract.
    assert any(
        isinstance(n, ast.List)
        and all(isinstance(v, ast.Constant) for v in n.elts)
        and [v.value for v in n.elts] == [48, 50, 51, 52]
        for n in ast.walk(ast.parse((HERE / "runner.py").read_text()))
    )


def test_finite_statistics_shape_refusal():
    rows = [dict(family_id="f")]
    with pytest.raises(ValueError):
        a.summarize(rows, np.zeros((2, 1, 3)))
    values = np.zeros((3, 1, 3))
    values[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        a.summarize(rows, values)
    assert math.isfinite(a.compute_r882_top_zgap(np.ones(4)))


def test_current_comparator_against_verbatim_live_decide():
    from types import SimpleNamespace

    reference = json.loads((HERE / "current_rule_source.json").read_text())
    namespace = {"copy": copy, "math": math}
    exec(reference["verbatim_methods"]["decide"], namespace)
    for new, old in [(1.0, 1.0), (1.1, 1.0), (0.9, 1.0), (-2.0, -2.0)]:
        restored = []
        state = SimpleNamespace(
            identity="fixed",
            strict_admission=False,
            min_gain=None,
            best_utility=old,
            best_round=0,
            best_checkpoint="r882",
            _mean=lambda x: float(x["mean"]),
            restore=lambda *args: restored.append(True),
        )
        model = SimpleNamespace(state_dict=lambda: {"adapter": 1})
        optimizer = SimpleNamespace(state_dict=lambda: {"step": 40})
        exact = namespace["decide"](
            state, model, optimizer, {"mean": new}, "fixed", "candidate", 1
        )
        ours = a.current_accept([new], [old])
        assert ours["accepted"] == exact["accepted"]
        assert bool(restored) == (not ours["accepted"])
