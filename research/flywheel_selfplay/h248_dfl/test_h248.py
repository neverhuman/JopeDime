"""Data-free gradient, weighting, exact-control, selector, launch and A1 guard checks. Toy data are not study data."""

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


def test_expected_regret_gradient_direction_and_temperature():
    scores = torch.tensor([[0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    fitness = torch.tensor([[0.0, 2.0]], dtype=torch.float64)
    gate = torch.ones((1, 2), dtype=torch.bool)
    weights = torch.ones(1, dtype=torch.float64)
    ceilings = torch.tensor([3.0])
    loss = tr.expected_regret(scores, fitness, gate, 2.0, weights, ceilings)
    grad = torch.autograd.grad(loss, scores)[0]
    assert loss.item() == 1 and torch.allclose(
        grad, torch.tensor([[0.25, -0.25]], dtype=torch.float64)
    )
    stepped = scores.detach() - 0.1 * grad
    assert tr.expected_regret(stepped, fitness, gate, 2.0, weights, ceilings) < loss
    assert torch.allclose(
        tr.expected_regret(scores + 17, fitness, gate, 2.0, weights, ceilings), loss
    )
    for bad in [0.0, -1.0, float("nan")]:
        with pytest.raises(ValueError):
            tr.expected_regret(scores, fitness, gate, bad, weights, ceilings)


def test_failure_cost_sign_and_full_context_ceiling():
    fitness = torch.tensor([[-1.0, 0.5, 9000.0, 2.0]])
    gate = torch.tensor([[True, True, False, True]])
    cap = tr.failure_ceiling(fitness.numpy()[0], gate.numpy()[0])
    assert cap == 3.0
    regret = tr.regret_targets(fitness, gate, torch.tensor([cap]))
    assert torch.equal(regret, torch.tensor([[0.0, 1.5, 4.0, 3.0]]))
    # A valid cost +2 is not an invalid-gate failure, and valid labels are not CTRL-clipped.
    assert (
        tr.failure_ceiling(np.array([-8.0, 4.0, 2.0]), np.array([True, True, False]))
        == 5
    )
    with pytest.raises(ValueError):
        tr.failure_ceiling(np.array([2.0]), np.array([False]))


def test_equal_family_weighting_under_uniform_context_sampling():
    r = [
        dict(context_sha256="a", family_id="A"),
        dict(context_sha256="b", family_id="A"),
        dict(context_sha256="c", family_id="B"),
    ]
    w = tr.family_weights(r, ["a", "b", "c"])
    assert torch.equal(w, torch.tensor([0.75, 0.75, 1.5]))
    losses = torch.tensor([2.0, 4.0, 10.0])
    assert (w * losses).mean() == 6.5
    # Fixed inverse-frequency weights, not batch-local renormalization.
    assert tr.family_weights(r, ["a"]).item() == 0.75


def test_loss_composition_and_exact_control_gradients():
    gen = torch.Generator().manual_seed(5)
    x = torch.randn(3, 7, generator=gen, requires_grad=True)
    f = torch.randn(3, 7, generator=gen)
    g = torch.ones_like(f, dtype=torch.bool)
    g[0, -1] = False
    w = torch.tensor([0.5, 1.0, 1.5])
    ceil = torch.tensor([3.0, 3.0, 3.0])
    ref = ctrl(x, f, g)
    actual = tr.arm_loss("CONTROL", ctrl, x, f, g, 0.7, w, ceil)
    assert torch.equal(actual, ref)
    assert torch.equal(
        torch.autograd.grad(actual, x, retain_graph=True)[0],
        torch.autograd.grad(ref, x, retain_graph=True)[0],
    )
    d = tr.expected_regret(x, f, g, 0.7, w, ceil)
    assert torch.equal(
        tr.arm_loss("DFL_MIX", ctrl, x, f, g, 0.7, w, ceil), ref + 0.5 * d
    )
    target = torch.where(g, f.clamp(-4, 2), torch.full_like(f, 2))
    kl = (
        0.1
        * F.kl_div(
            F.log_softmax(-x / 0.2, dim=-1),
            F.softmax(-target / 0.2, dim=-1),
            reduction="none",
        )
        .sum(1)
        .mean()
    )
    assert torch.equal(tr.arm_loss("DFL", ctrl, x, f, g, 0.7, w, ceil), d + kl)
    with pytest.raises(ValueError):
        tr.arm_loss("OTHER", ctrl, x, f, g, 0.7, w, ceil)


def test_control_update_matches_native_adamw():
    torch.manual_seed(3)
    a = torch.nn.Parameter(torch.randn(5, 3))
    b = torch.nn.Parameter(a.detach().clone())
    idle = torch.nn.Parameter(torch.ones(1))
    ours = tr.make_optimizer("CONTROL", [a, idle])
    ref = torch.optim.AdamW([b], lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)
    for _ in range(5):
        grad = torch.randn(5, 3)
        a.grad = grad.clone()
        b.grad = grad.clone()
        ours.filter_gradients()
        ours.step()
        ref.step()
    assert torch.equal(a, b) and idle.item() == 1
    for arm in stat.ARMS:
        assert type(tr.make_optimizer(arm, [a])) is tr.ControlAdamW


def test_frozen_schedule_and_temperature_binding():
    assert (
        stat.ARMS == ("CONTROL", "DFL_MIX", "DFL")
        and stat.SEEDS == (248101, 248102, 248103)
        and stat.FOLD_SEED == 248140
    )
    schedule = json.loads((HERE / "schedules.json").read_text())["arms"]
    for arm in stat.ARMS:
        for t, row in enumerate(schedule[arm], 1):
            assert (
                (row["lr"], row["beta1"], row["beta2"], row["weight_decay"])
                == tr.settings(arm, t)
                == (1e-4, 0.9, 0.999, 1e-4)
            )
    temperatures = json.loads((HERE / "temperatures.json").read_text())
    assert (
        temperatures["DLVal_scored"]
        == temperatures["optimizer_updates"]
        == temperatures["utility_outcomes_scored"]
        == 0
    )
    assert set(temperatures["by_outer"]) == set("01234")
    assert all(
        x["temperature"] > 0 and x["fit_families"] == 123
        for x in temperatures["by_outer"].values()
    )


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
        a = stat.batch_plan(ids, orders, 248101)
        b = stat.batch_plan(ids, orders, 248101)
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
    assert out["contrasts"]["DFL_MIX_minus_r882"]["advance"]
    assert not out["contrasts"]["DFL_minus_r882"]["advance"]
    assert "DFL_minus_CONTROL" in out["contrasts"]
    assert stat.CLIFF_CONTEXT == 27


def test_launch_requires_merged_binding():
    j = dict(
        manifest_sha256="m",
        merged_source_verified=True,
        source_commit="h",
        review_id=5,
        prereg_pr=7,
        prereg_sha256="p",
        fold_manifest_sha256="f",
    )
    tr.validate_launch(j, "h", "m", "p", "f")
    for field, value in [
        ("source_commit", "bad"),
        ("review_id", 0),
        ("fold_manifest_sha256", "bad"),
    ]:
        with pytest.raises(RuntimeError):
            tr.validate_launch(dict(j, **{field: value}), "h", "m", "p", "f")


def sample(t, ram, load=50):
    return dict(
        utc=t,
        load5=load,
        available=ram * GIB,
        free=10**15,
        rss=GIB,
        threads=2,
        affinity=[48, 49, 50, 51],
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
        mon.COMPUTE_STOP == 1793718000.0
        and "absolute compute stop 15:00Z"
        in mon.violations(sample(mon.COMPUTE_STOP, 40))
    )


def test_manifest_and_schema():
    import jsonschema

    m = json.loads((HERE / "manifest.json").read_text())
    assert (
        m["wave"] == "H248" and m["total_updates"] == 1800 and m["smoke_updates"] == 3
    )
    assert m["inputs"][m["checkpoint"]].startswith("0084479a") and m[
        "dope_composite_sha256"
    ].startswith("c51a56d0")
    jsonschema.validate(
        json.loads((HERE / "metrics.json").read_text()),
        json.loads((HERE / "metrics.schema.json").read_text()),
    )
