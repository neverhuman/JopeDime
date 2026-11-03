"""Data-free H251 controls: first-order Reptile (inner plain SGD, outer interpolation, k = 5 / k = 1), the one-family
batch plan, exact CTRL AdamW, frozen settings, folds, selector, decision rule with the mechanism contrast and the A1
guard. Toy tensors and toy rows are never study data."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import analysis as stat
import monitor as mon
from treatment import (
    BETAS,
    CONTROL_LR,
    CONTROL_WD,
    INNER_LR,
    INNER_STEPS,
    META_EPS,
    ControlAdamW,
    Reptile,
    adapter_drift,
    arm_loss,
    make_optimizer,
    position,
    settings,
    validate_launch,
)

HERE = Path(__file__).resolve().parent
GIB = 2**30


def rows(n=205):
    return [dict(context_sha256=f"c{i:03}", family_id=f"f{i:03}") for i in range(n)]


def family_rows():
    """Toy FIT rows: 12 families with 1..12 contexts each."""
    out = []
    for f in range(12):
        out += [
            dict(context_sha256=f"f{f:02}c{j:02}", family_id=f"f{f:02}")
            for j in range(f + 1)
        ]
    return out


def test_frozen_hyperparameters_and_arms():
    assert stat.ARMS == ("CONTROL", "REPTILE", "REPTILE_K1")
    assert (
        stat.SEEDS == (251101, 251102, 251103)
        and stat.FOLD_SEED == 251140
        and stat.BOOT_SEED == 251150
    )
    assert (CONTROL_LR, CONTROL_WD, BETAS) == (1e-4, 1e-4, (0.9, 0.999))
    assert (
        INNER_LR == pytest.approx(1e-3)
        and INNER_LR == 10 * CONTROL_LR
        and META_EPS == 0.5
    )
    assert INNER_STEPS == stat.INNER_STEPS == {"REPTILE": 5, "REPTILE_K1": 1}
    for arm in stat.ARMS[1:]:
        assert (
            sum(settings(arm, t)["meta_update"] for t in range(1, 41))
            == 40 // INNER_STEPS[arm]
        )
    assert (
        settings("CONTROL", 3)["optimizer"] == "AdamW"
        and settings("REPTILE", 3)["optimizer"] == "SGD"
    )
    with pytest.raises(ValueError):
        settings("REPTILE", 41)


def test_schedule_table_matches_settings():
    table = json.loads((HERE / "schedules.json").read_text())["arms"]
    assert set(table) == set(stat.ARMS)
    for arm in stat.ARMS:
        assert table[arm] == [dict(step=t, **settings(arm, t)) for t in range(1, 41)]


def test_positions_and_snapshot_boundaries():
    assert [position("REPTILE", t) for t in (1, 5, 6, 40)] == [
        (1, 1),
        (1, 5),
        (2, 1),
        (8, 5),
    ]
    assert [position("REPTILE_K1", t) for t in (1, 2, 40)] == [(1, 1), (2, 1), (40, 1)]
    for arm in stat.ARMS[1:]:
        assert all(
            settings(arm, s)["meta_update"] for s in stat.STEPS[1:]
        )  # snapshots at outer boundaries


def test_reptile_k1_is_scaled_sgd_and_k5_interpolates():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(6, 3))
    theta = p.detach().clone()
    opt = make_optimizer("REPTILE_K1", [p])
    g = torch.randn(6, 3)
    p.grad = g.clone()
    assert opt.step() is True
    assert torch.allclose(p.detach(), theta - META_EPS * INNER_LR * g, atol=1e-7)
    q = torch.nn.Parameter(torch.randn(4))
    start = q.detach().clone()
    opt = make_optimizer("REPTILE", [q])
    phi = start.clone()
    for i in range(5):
        grad = torch.randn(4, generator=torch.Generator().manual_seed(i))
        q.grad = grad.clone()
        assert opt.step() is (i == 4)
        phi = phi - INNER_LR * grad
        if i < 4:
            assert torch.allclose(q.detach(), phi, atol=1e-7) and opt.inner == i + 1
    assert torch.allclose(q.detach(), start + META_EPS * (phi - start), atol=1e-7)
    assert opt.inner == 0 and opt.outer_updates == 1 and opt.theta is None


def test_reptile_inner_sgd_is_plain_and_skips_none():
    a, b = torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.ones(2))
    opt = Reptile([a, b], 2)
    group = opt.param_groups[0]
    assert (
        group["lr"],
        group["momentum"],
        group["weight_decay"],
        group["nesterov"],
    ) == (INNER_LR, 0.0, 0.0, False)
    for _ in range(2):
        a.grad, b.grad = torch.ones(3), None
        opt.step()
    assert torch.equal(b.detach(), torch.ones(2))
    assert torch.allclose(
        a.detach(), torch.ones(3) - META_EPS * 2 * INNER_LR, atol=1e-7
    )
    opt.zero_grad()
    assert a.grad is None
    with pytest.raises(ValueError):
        Reptile([a], 0)
    with pytest.raises(ValueError):
        make_optimizer("MUON", [a])


def test_control_is_exact_torch_adamw():
    torch.manual_seed(3)
    a = torch.nn.Parameter(torch.randn(5, 3))
    b = torch.nn.Parameter(a.detach().clone())
    ours, ref = (
        make_optimizer("CONTROL", [a]),
        torch.optim.AdamW([b], lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4),
    )
    assert type(ours) is ControlAdamW
    for _ in range(5):
        g = torch.randn(5, 3)
        a.grad, b.grad = g.clone(), g.clone()
        ours.filter_gradients()
        assert ours.step() is True
        ref.step()
    assert torch.equal(a.detach(), b.detach())


def test_family_plan_one_family_per_outer_step():
    fit = family_rows()
    orders = {r["context_sha256"]: np.arange(30)[::-1] for r in fit}
    family = {r["context_sha256"]: r["family_id"] for r in fit}
    plans = {
        arm: stat.family_plan(fit, orders, 251101, 2, INNER_STEPS[arm])
        for arm in stat.ARMS[1:]
    }
    for arm, plan in plans.items():
        k = INNER_STEPS[arm]
        assert len(plan) == 40
        for t, (stream, cids, indexes, fam, o, i) in enumerate(plan, 1):
            assert (o, i) == position(arm, t)
            assert len(cids) == 8 and {family[c] for c in cids} == {fam}
            assert all(
                len(ix) == 24 and list(ix[:4]) == [29, 28, 27, 26] for ix in indexes
            )
            if len([r for r in fit if r["family_id"] == fam]) >= 8:
                assert len(set(cids)) == 8
        assert len({p[3] for p in plan[:k]}) == 1
    reptile, k1 = plans["REPTILE"], plans["REPTILE_K1"]
    assert [reptile[5 * j][3] for j in range(8)] == [
        k1[j][3] for j in range(8)
    ]  # shared family stream
    assert reptile[0][:2] == k1[0][:2]  # same first minibatch
    again = stat.family_plan(fit, orders, 251101, 2, 5)
    assert [(p[0], p[1], p[3]) for p in again] == [(p[0], p[1], p[3]) for p in reptile]
    other = stat.family_plan(fit, orders, 251101, 3, 5)
    assert [p[0] for p in other] != [p[0] for p in reptile]


def test_every_arm_uses_ctrl_loss_and_drift():
    calls = []

    def ctrl(pred, fitness, gate):
        calls.append(1)
        return pred.sum()

    for arm in stat.ARMS:
        assert float(arm_loss(arm, ctrl, torch.ones(2), None, None)) == 2.0
    assert len(calls) == 3
    with pytest.raises(ValueError):
        arm_loss("SAM", ctrl, torch.ones(2), None, None)
    assert adapter_drift({"x": torch.zeros(2)}, {"x": torch.tensor([3.0, 4.0])}) == 5.0


def test_shared_fold_seed_partition_and_manifest():
    r = rows()
    groups = [stat.folds(r, s) for s in stat.SEEDS]
    assert groups[0] == groups[1] == groups[2]
    for outer in range(5):
        fit, select, assess = stat.partition(r, outer, stat.SEEDS[0])
        fams = [{x["family_id"] for x in part} for part in (fit, select, assess)]
        assert (
            not (fams[0] & fams[1])
            and not (fams[0] & fams[2])
            and not (fams[1] & fams[2])
        )
    fm = json.loads((HERE / "fold_manifest.json").read_text())
    assert fm["fold_seed"] == 251140
    assert all([len(g) for g in v] == [41] * 5 for v in fm["families_by_seed"].values())


def test_selector_keeps_r882_on_ties_and_noise():
    r = rows(40)
    flat = np.ones((40, 5))
    d = stat.choose(r, flat, 1)
    assert d["selected_step"] == 0 and d["best_step"] == 0
    noisy = flat.copy()
    noisy[:, 4] += np.where(np.arange(40) % 2 == 0, 0.5, -0.49)
    d = stat.choose(r, noisy, 1)
    assert d["best_step"] == 40 and d["selected_step"] == 0


def test_decision_rule_four_gates_and_contrasts():
    r = rows(19)
    panel = np.ones((3, 19, 4))
    out = stat.summarize(r, panel)
    assert set(out["contrasts"]) == {
        "CONTROL_minus_r882",
        "REPTILE_minus_r882",
        "REPTILE_minus_CONTROL",
        "REPTILE_minus_REPTILE_K1",
        "REPTILE_K1_minus_r882",
        "REPTILE_K1_minus_CONTROL",
    }
    for arm in stat.ARMS[1:]:
        c = out["contrasts"][arm + "_minus_r882"]
        assert not c["advance"] and c["holm_p"] == 1.0
        assert c["harm_families_by_seed"] == [[], [], []]
    assert "holm_p" not in out["contrasts"]["CONTROL_minus_r882"]
    assert out["vs_plateau_best_by_seed"][0][0] == pytest.approx(1 - stat.PLATEAU_BEST)
    lift = panel.copy()
    lift[:, :, 2] += 0.01
    out = stat.summarize(r, lift)
    assert out["contrasts"]["REPTILE_minus_r882"]["advance"]
    assert out["contrasts"]["REPTILE_minus_REPTILE_K1"]["median_gain"] == pytest.approx(
        0.01
    )
    assert not out["contrasts"]["REPTILE_K1_minus_r882"]["advance"]
    small = panel.copy()
    small[:, :, 3] += 0.0015
    assert not stat.summarize(r, small)["contrasts"]["REPTILE_K1_minus_r882"]["advance"]
    one_seed = panel.copy()
    one_seed[0, :, 2] += 0.05
    assert not stat.summarize(r, one_seed)["contrasts"]["REPTILE_minus_r882"]["advance"]
    harm = panel.copy()
    harm[:, 0, 2] -= 1
    assert (
        stat.summarize(r, harm)["contrasts"]["REPTILE_minus_r882"][
            "harm_families_by_seed"
        ]
        == [["f000"]] * 3
    )
    assert stat.CLIFF_CONTEXT == 27


def test_validate_launch_binding():
    launch = dict(
        manifest_sha256="m",
        merged_source_verified=True,
        source_commit="h",
        review_id=5,
        prereg_pr=7,
        prereg_sha256="p",
        fold_manifest_sha256="f",
    )
    validate_launch(launch, "h", "m", "p", "f")
    for key, bad in [
        ("source_commit", "z"),
        ("review_id", 0),
        ("fold_manifest_sha256", "q"),
    ]:
        with pytest.raises(RuntimeError):
            validate_launch(dict(launch, **{key: bad}), "h", "m", "p", "f")


def sample(t, available, load5=50.0):
    return dict(
        utc=t,
        load5=load5,
        available=available,
        free=10**15,
        rss=GIB,
        threads=2,
        affinity=[44, 45, 46, 47],
    )


def test_a1_memory_guard_and_deadline():
    g = mon.Guard()
    t0 = 1793710000.0
    assert g.update(sample(t0, 19 * GIB)) == ([], True)
    assert g.update(sample(t0 + 32, 26 * GIB))[1] is True
    assert g.update(sample(t0 + 64, 26 * GIB)) == ([], False)
    h = mon.Guard()
    h.update(sample(t0, 11 * GIB))
    assert "MemAvailable<12GiB sustained30s" in h.update(sample(t0 + 31, 11 * GIB))[0]
    assert "MemAvailable<6GiB" in mon.Guard().update(sample(t0, 5 * GIB))[0]
    assert mon.COMPUTE_STOP == 1793718000.0  # 2026-11-03T15:00:00Z
    assert "absolute compute stop 15:00Z" in mon.violations(
        sample(mon.COMPUTE_STOP, 40 * GIB)
    )
    assert mon.Guard().update(sample(t0, 40 * GIB, load5=109.0))[1] is True


def test_manifest_pins():
    m = json.loads((HERE / "manifest.json").read_text())
    assert m["wave"] == "H251" and m["total_gradient_evaluations"] == 1800
    assert m["inputs"][m["checkpoint"]].startswith("0084479a")
    assert m["dope_composite_sha256"].startswith("c51a56d0")
    assert m["reptile"] == dict(
        inner_optimizer="SGD",
        inner_lr=1e-3,
        meta_eps=0.5,
        inner_steps={"REPTILE": 5, "REPTILE_K1": 1},
    )
    assert (
        "treatment.py" in m["source_sha256"] and "schedules.json" in m["source_sha256"]
    )
