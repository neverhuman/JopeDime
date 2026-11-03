"""Data-free tests; fixtures are not study observations."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn.functional as F
import analysis as stat
import monitor as mon
import treatment as tr

HERE = Path(__file__).resolve().parent
GIB = 2**30


def ctrl(pred, fitness, gate):
    target = torch.where(gate, fitness.clamp(-4, 2), torch.full_like(fitness, 2))
    return (
        F.smooth_l1_loss(pred, target, reduction="none").mean(1)
        + 0.1
        * F.kl_div(
            F.log_softmax(-pred / 0.2, dim=-1),
            F.softmax(-target / 0.2, dim=-1),
            reduction="none",
        ).sum(1)
    ).mean()


def rows(n=205):
    return [dict(context_sha256=f"c{i:03}", family_id=f"f{i:03}") for i in range(n)]


def test_partition_and_training_batch_identity():
    r = rows()
    groups = [stat.folds(r, s) for s in stat.SEEDS]
    assert groups[0] == groups[1] == groups[2]
    for outer in range(5):
        fit, select, assess = stat.partition(r, outer, stat.SEEDS[0])
        sets = [{x["family_id"] for x in part} for part in (fit, select, assess)]
        assert list(map(len, sets)) == [123, 41, 41]
        assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
        ids = [x["context_sha256"] for x in fit]
        orders = {cid: np.arange(30) for cid in ids}
        a = stat.batch_plan(ids, orders, 257101)
        b = stat.batch_plan(ids, orders, 257101)
        assert len(a) == 40
        for (stream, cids, ix), (stream2, cids2, ix2) in zip(a, b, strict=True):
            assert stream == stream2 and cids == cids2 and set(cids) <= set(ids)
            for q, z in zip(ix, ix2, strict=True):
                assert (
                    np.array_equal(q, z)
                    and np.array_equal(q[:4], np.arange(4))
                    and len(set(q)) == 24
                )


def test_selector_ties_noise_and_strict_margin():
    r = rows(40)
    flat = np.ones((40, 5))
    d = stat.choose(r, flat, 1)
    assert d["selected_step"] == d["best_step"] == 0
    noisy = flat.copy()
    noisy[:, 4] += np.where(np.arange(40) % 2 == 0, 0.5, -0.49)
    d = stat.choose(r, noisy, 1)
    assert d["best_step"] == 40 and d["selected_step"] == 0
    flat[:, 1:] += 0.001
    assert stat.choose(r, flat, 1)["selected_step"] == 0
    flat[:, 1:] += 0.01
    assert stat.choose(r, flat, 1)["selected_step"] == 10


def test_holm_two_and_advance_rule():
    assert stat.holm({"a": 0.03, "b": 0.04}) == {"a": 0.06, "b": 0.06}
    r = rows(19)
    panel = np.ones((3, 19, 4))
    out = stat.summarize(r, panel)
    for arm in stat.ARMS[1:]:
        assert not out["contrasts"][arm + "_minus_r882"]["advance"]
    panel[:, :, 2] += 0.01
    out = stat.summarize(r, panel)
    assert out["contrasts"]["ADAN_minus_r882"]["advance"]
    assert not out["contrasts"]["ADAN_HALF_minus_r882"]["advance"]
    assert "ADAN_HALF_minus_CONTROL" in out["contrasts"]
    assert stat.CLIFF_CONTEXT == 27


def sample(t, ram, load=50):
    return dict(
        utc=t,
        load5=load,
        available=ram * GIB,
        free=10**15,
        rss=GIB,
        threads=2,
        affinity=[52, 53, 54, 55],
    )


def test_temporal_a1_memory_and_load_guards():
    t = 1793700000.0
    g = mon.Guard()
    assert g.update(sample(t, 19)) == ([], True)
    assert g.update(sample(t + 1, 26))[1]
    assert g.update(sample(t + 30, 26))[1]
    assert g.update(sample(t + 31, 26)) == ([], False)
    g = mon.Guard()
    assert not g.update(sample(t, 11))[0]
    assert not g.update(sample(t + 29, 11))[0]
    assert "MemAvailable<12GiB sustained30s" in g.update(sample(t + 30, 11))[0]
    assert "MemAvailable<6GiB" in mon.Guard().update(sample(t, 5))[0]
    g = mon.Guard()
    assert g.update(sample(t, 40, 109))[1]
    assert g.update(sample(t + 1, 40, 101))[1]
    assert not g.update(sample(t + 2, 40, 100))[1]
    assert (
        mon.COMPUTE_STOP == 1793721600.0
        and "absolute compute stop 16:00Z"
        in mon.violations(sample(mon.COMPUTE_STOP, 40))
    )


def test_archived_plateau_elo_ties_and_direction():
    r = rows(3)
    values = np.ones((3, 3, 4))
    reference = np.ones(3)
    j = stat.archived_plateau_elo(r, values, reference)
    assert all(
        v["median_rating"] == 1500 and v["ci95_lift"] == [0.0, 0.0]
        for v in j["arms"].values()
    )
    values[:, :, 2] += 1.0
    j = stat.archived_plateau_elo(r, values, reference)
    assert j["arms"]["ADAN"]["median_rating"] > 1500


def test_readonly_protocol_allowlist_rejects_writes_and_unlisted_data():
    import subprocess
    import types

    manifest = json.loads((HERE / "manifest.json").read_text())
    module = types.ModuleType("h257_audit_test")
    raw = subprocess.check_output(
        [
            "git",
            "show",
            manifest["parent_commit"] + ":research/flywheel_selfplay/runtime_audit.py",
        ],
        cwd="/home/ubuntu/JopeDime",
    )
    exec(compile(raw, "runtime_audit.py", "exec"), module.__dict__)
    name = "/home/ubuntu/jope-data/h257-fixture/source/schedules.json"
    audit = module.Audit({name}, Path("/home/ubuntu/jope-data/h257-fixture/output"))
    audit.check("open", (name, "r", 0))
    with pytest.raises(RuntimeError):
        audit.check("open", (name, "w", 0))
    with pytest.raises(RuntimeError):
        audit.check(
            "open", ("/home/ubuntu/jope-data/h257-fixture/unlisted.json", "r", 0)
        )


def test_control_loss_and_updates_exact():
    gen = torch.Generator().manual_seed(8)
    a = torch.nn.Parameter(torch.randn(3, 7, generator=gen))
    b = torch.nn.Parameter(a.detach().clone())
    idle = torch.nn.Parameter(torch.ones(1))
    ours = tr.make_optimizer("CONTROL", [a, idle])
    ref = torch.optim.AdamW([b], lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)
    fitness = torch.randn(3, 7, generator=gen)
    gate = fitness > -0.2
    for arm in stat.ARMS:
        x = tr.arm_loss(arm, ctrl, a, fitness, gate)
        y = ctrl(a, fitness, gate)
        assert torch.equal(x, y)
        assert torch.equal(
            torch.autograd.grad(x, a, retain_graph=True)[0],
            torch.autograd.grad(y, a)[0],
        )
    for _ in range(40):
        g = torch.randn(3, 7, generator=gen)
        a.grad = g.clone()
        b.grad = g.clone()
        ours.step()
        ref.step()
    assert torch.equal(a, b) and idle.item() == 1


def test_adan_scalar_recurrence_and_half_lr():
    # Separate scalar recurrence checks initialization, bias correction and proximal decay.
    for arm, lr in [("ADAN", 1e-4), ("ADAN_HALF", 5e-5)]:
        p = torch.nn.Parameter(torch.tensor([0.8], dtype=torch.float64))
        opt = tr.make_optimizer(arm, [p])
        value = 0.8
        m = d = v = 0.0
        prev = None
        for i, g in enumerate([0.7, -0.3, 0.0, 1e-12, 0.9], 1):
            delta = 0.0 if prev is None else g - prev
            m = 0.98 * m + 0.02 * g
            d = 0.92 * d + 0.08 * delta
            v = 0.99 * v + 0.01 * (g + 0.92 * delta) ** 2
            value = (
                value
                - lr
                * (m / (1 - 0.98**i) + 0.92 * d / (1 - 0.92**i))
                / ((v / (1 - 0.99**i)) ** 0.5 + 1e-8)
            ) / (1 + lr * 0.02)
            p.grad = torch.tensor([g], dtype=torch.float64)
            opt.step()
            prev = g
            assert p.item() == pytest.approx(value, abs=2e-15)


def test_adan_none_zero_and_resume():
    import copy

    a = torch.nn.Parameter(torch.tensor([0.8, -0.2], dtype=torch.float64))
    idle = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    opt = tr.make_optimizer("ADAN", [a, idle])
    a.grad = torch.zeros_like(a)
    opt.step()
    assert torch.equal(
        a, torch.tensor([0.8, -0.2], dtype=torch.float64) / (1 + 1e-4 * 0.02)
    )
    assert idle.item() == 1 and idle not in opt.state
    b = torch.nn.Parameter(a.detach().clone())
    idle2 = torch.nn.Parameter(idle.detach().clone())
    other = tr.make_optimizer("ADAN", [b, idle2])
    other.load_state_dict(copy.deepcopy(opt.state_dict()))
    for i in range(10):
        a.grad = torch.tensor([i / 10, -0.2], dtype=torch.float64)
        b.grad = a.grad.clone()
        opt.step()
        other.step()
    assert torch.equal(a, b) and idle2.item() == 1
    a.grad = torch.tensor([float("nan"), 0.0], dtype=torch.float64)
    with pytest.raises(RuntimeError):
        opt.step()


def test_frozen_settings():
    m = json.loads((HERE / "manifest.json").read_text())
    s = json.loads((HERE / "schedules.json").read_text())["arms"]
    assert stat.SEEDS == (257101, 257102, 257103) and stat.FOLD_SEED == 257140
    assert m["total_updates"] == 1800 and m["smoke_updates"] == 3
    assert m["resource_limits"]["affinity"] == [52, 53, 54, 55]
    for arm in stat.ARMS:
        for i, row in enumerate(s[arm], 1):
            assert tuple(
                row[k] for k in ["lr", "beta1", "beta2", "weight_decay", "beta3", "eps"]
            ) == tr.settings(arm, i)
    assert tr.settings("ADAN", 40) == (1e-4, 0.98, 0.92, 0.02, 0.99, 1e-8)
    assert tr.settings("ADAN_HALF", 1)[0] == 5e-5


def test_three_positive_seeds_required():
    panel = np.ones((3, 19, 4))
    panel[:2, :, 2] += 0.02
    panel[2, :, 2] -= 0.0001
    c = stat.summarize(rows(19), panel)["contrasts"]["ADAN_minus_r882"]
    assert (
        c["median_gain"] > 0.002
        and c["ci95"][0] > 0
        and c["holm_p"] < 0.05
        and c["positive_seeds"] == 2
        and not c["advance"]
    )
    panel[2, :, 2] += 0.02
    assert stat.summarize(rows(19), panel)["contrasts"]["ADAN_minus_r882"]["advance"]


def test_launch_requires_published_prereg():
    j = dict(
        published_source_verified=True,
        publication_branch="v1/next",
        source_commit="h",
        reviewed_source="h",
        review_status="PASS",
        manifest_sha256="m",
        prereg_sha256="p",
        fold_manifest_sha256="f",
    )
    tr.validate_launch(j, "h", "m", "p", "f")
    for key, value in [
        ("publication_branch", "main"),
        ("published_source_verified", False),
        ("review_status", "FAIL"),
        ("reviewed_source", "bad"),
        ("fold_manifest_sha256", "bad"),
    ]:
        with pytest.raises(RuntimeError):
            tr.validate_launch(dict(j, **{key: value}), "h", "m", "p", "f")
    with pytest.raises(RuntimeError):
        tr.validate_launch(j, "h", "m", "p", "f", smoke=True)


def test_runtime_affinity_matches_guard_and_manifest():
    import ast

    expected = json.loads((HERE / "manifest.json").read_text())["resource_limits"][
        "affinity"
    ]
    for name in ["runner.py", "monitor.py"]:
        tree = ast.parse((HERE / name).read_text())
        compares = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Compare) and "sched_getaffinity" in ast.unparse(n)
        ]
        assert any(ast.literal_eval(n.comparators[0]) == expected for n in compares)
