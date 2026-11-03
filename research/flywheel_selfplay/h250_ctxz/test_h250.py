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


def test_standardization_affine_invariance_and_floor():
    x = torch.tensor(
        [[1.0, 2.0, 4.0], [0.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True
    )
    z = tr.standardize(x, 0.01)
    assert torch.allclose(z, tr.standardize(3 * x + 7, 0.01), atol=1e-12)
    assert torch.allclose(z.mean(1), torch.zeros(2, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(
        z[0].std(correction=0), torch.tensor(1.0, dtype=torch.float64)
    )
    g = torch.autograd.grad((z * torch.tensor([1.0, 2.0, 3.0])).sum(), x)[0]
    assert torch.isfinite(g).all()
    small = torch.tensor([[0.0, 0.001]])
    assert torch.allclose(tr.standardize(small, 0.1), torch.tensor([[-0.005, 0.005]]))
    for bad in [0.0, -1.0, float("nan")]:
        with pytest.raises(ValueError):
            tr.standardize(x, bad)


def test_failure_sign_full_context_and_detached_labels():
    f = torch.tensor([[-1.0, 0.5, 9000.0, 2.0]], requires_grad=True)
    gate = torch.tensor([[True, True, False, True]])
    cap = tr.failure_ceiling(f.detach().numpy()[0], gate.numpy()[0])
    assert cap == 3.0
    targets = tr.clipped_targets(f, gate, torch.tensor([cap]))
    assert (
        torch.equal(targets, torch.tensor([[-1.0, 0.5, 3.0, 2.0]]))
        and not targets.requires_grad
    )
    assert (
        tr.failure_ceiling(np.array([-8.0, 4.0, 2.0]), np.array([True, True, False]))
        == 5.0
    )
    with pytest.raises(ValueError):
        tr.failure_ceiling(np.array([2.0]), np.array([False]))


def test_z_loss_formula_gradients_and_exact_control():
    gen = torch.Generator().manual_seed(5)
    x = torch.randn(3, 7, generator=gen, requires_grad=True)
    f = torch.randn(3, 7, generator=gen)
    gate = torch.ones_like(f, dtype=torch.bool)
    gate[0, -1] = False
    ceil = torch.tensor([3.0, 3.0, 3.0])
    ref = ctrl(x, f, gate)
    actual = tr.arm_loss("CONTROL", ctrl, x, f, gate, 0.01, ceil)
    assert torch.equal(actual, ref)
    assert torch.equal(
        torch.autograd.grad(actual, x, retain_graph=True)[0],
        torch.autograd.grad(ref, x, retain_graph=True)[0],
    )
    target = torch.where(gate, f, ceil[:, None])
    zp = (x - x.mean(1, keepdim=True)) / x.std(1, correction=0, keepdim=True)
    zt = (target - target.mean(1, keepdim=True)) / target.std(
        1, correction=0, keepdim=True
    )
    expected = (
        F.smooth_l1_loss(zp, zt, reduction="none").mean(1)
        + 0.1
        * F.kl_div(
            F.log_softmax(-zp / 0.2, dim=-1),
            F.softmax(-zt / 0.2, dim=-1),
            reduction="none",
        ).sum(1)
    ).mean()
    for arm in ["CTXZ_FB", "CTXZ"]:
        actual = tr.arm_loss(arm, ctrl, x, f, gate, 0.01, ceil)
        assert torch.equal(actual, expected)
        assert torch.allclose(
            actual,
            tr.arm_loss(arm, ctrl, 3 * x + 7, 2 * f - 1, gate, 0.01, 2 * ceil - 1),
            atol=2e-6,
        )
        assert torch.isfinite(
            torch.autograd.grad(actual, x, retain_graph=True)[0]
        ).all()
    with pytest.raises(ValueError):
        tr.arm_loss("BAD", ctrl, x, f, gate, 0.01, ceil)


def test_balanced_plan_uniform_family_contract_and_exact_control_order():
    fit = [
        dict(context_sha256=f"c{i:03}-{j}", family_id=f"f{i:03}")
        for i in range(123)
        for j in range(1 + i % 4)
    ]
    orders = {r["context_sha256"]: np.arange(30) for r in fit}
    family = {r["context_sha256"]: r["family_id"] for r in fit}
    original = stat.batch_plan(list(orders), orders, 250101)
    for arm in ["CONTROL", "CTXZ"]:
        assert stat.plan_digest(original) == stat.plan_digest(
            stat.arm_batch_plan(fit, orders, 250101, 0, arm)
        )
    plan = stat.arm_batch_plan(fit, orders, 250101, 0, "CTXZ_FB")
    assert len(plan) == 40 and stat.plan_digest(plan) == stat.plan_digest(
        stat.arm_batch_plan(fit, orders, 250101, 0, "CTXZ_FB")
    )
    assert stat.plan_digest(plan) != stat.plan_digest(
        stat.arm_batch_plan(fit, orders, 250101, 1, "CTXZ_FB")
    )
    for step, (stream, cids, indexes) in enumerate(plan, 1):
        assert len(cids) == 8 and len({family[c] for c in cids}) == 8
        rng = np.random.default_rng(stat.stream_seed(250101, 0, "CTXZ_FB", step))
        fams = sorted(set(family.values()))
        chosen = [fams[i] for i in rng.choice(123, 8, replace=False)]
        assert [family[c] for c in cids] == chosen
        for fam, cid in zip(chosen, cids, strict=True):
            choices = sorted(c for c in family if family[c] == fam)
            assert choices[rng.integers(len(choices))] == cid
        for ix in indexes:
            assert np.array_equal(ix[:4], np.arange(4)) and len(set(ix)) == 24


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


def test_frozen_schedule_and_normalization_binding():
    assert (
        stat.ARMS == ("CONTROL", "CTXZ_FB", "CTXZ")
        and stat.SEEDS == (250101, 250102, 250103)
        and stat.FOLD_SEED == 250140
    )
    schedule = json.loads((HERE / "schedules.json").read_text())["arms"]
    for arm in stat.ARMS:
        for t, row in enumerate(schedule[arm], 1):
            assert (
                (row["lr"], row["beta1"], row["beta2"], row["weight_decay"])
                == tr.settings(arm, t)
                == (1e-4, 0.9, 0.999, 1e-4)
            )
    normalization = json.loads((HERE / "normalization.json").read_text())
    assert (
        normalization["DLVal_scored"]
        == normalization["optimizer_updates"]
        == normalization["utility_outcomes_scored"]
        == 0
    )
    assert set(normalization["by_outer"]) == set("01234")
    assert all(
        x["epsilon"] > 0 and x["fit_families"] == 123
        for x in normalization["by_outer"].values()
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
        a = stat.batch_plan(ids, orders, 250101)
        b = stat.batch_plan(ids, orders, 250101)
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
    assert out["contrasts"]["CTXZ_FB_minus_r882"]["advance"]
    assert not out["contrasts"]["CTXZ_minus_r882"]["advance"]
    assert "CTXZ_minus_CONTROL" in out["contrasts"]
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
        m["wave"] == "H250" and m["total_updates"] == 1800 and m["smoke_updates"] == 3
    )
    assert m["inputs"][m["checkpoint"]].startswith("0084479a") and m[
        "dope_composite_sha256"
    ].startswith("c51a56d0")
    jsonschema.validate(
        json.loads((HERE / "metrics.json").read_text()),
        json.loads((HERE / "metrics.schema.json").read_text()),
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
    assert j["arms"]["CTXZ_FB"]["median_rating"] > 1500


def test_readonly_protocol_allowlist_rejects_writes_and_unlisted_data():
    import subprocess
    import types

    manifest = json.loads((HERE / "manifest.json").read_text())
    module = types.ModuleType("h250_audit_test")
    raw = subprocess.check_output(
        [
            "git",
            "show",
            manifest["parent_commit"] + ":research/flywheel_selfplay/runtime_audit.py",
        ],
        cwd="/home/ubuntu/JopeDime",
    )
    exec(compile(raw, "runtime_audit.py", "exec"), module.__dict__)
    name = "/home/ubuntu/jope-data/h250-fixture/source/schedules.json"
    audit = module.Audit({name}, Path("/home/ubuntu/jope-data/h250-fixture/output"))
    audit.check("open", (name, "r", 0))
    with pytest.raises(RuntimeError):
        audit.check("open", (name, "w", 0))
    with pytest.raises(RuntimeError):
        audit.check(
            "open", ("/home/ubuntu/jope-data/h250-fixture/unlisted.json", "r", 0)
        )
