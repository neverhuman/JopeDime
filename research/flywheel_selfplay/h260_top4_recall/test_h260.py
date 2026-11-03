"""Data-free H260 controls: exact H258 TOP4_PICK and zgap, oracle labels, the hinge@4 and exact Plackett-Luce top-4
objectives (brute-force checked), exact CTRL AdamW, frozen settings, folds, the TOP4 selector, the decision rule with
the single-pick guard and the A1 guard. Toy tensors and toy rows are never study data."""

from itertools import permutations
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import analysis as stat
import monitor as mon
from treatment import (
    AUX_WEIGHT,
    BETAS,
    CONTROL_LR,
    CONTROL_WD,
    MARGIN_FRACTION,
    PL_TAU,
    ControlAdamW,
    adapter_drift,
    arm_loss,
    hinge_at4,
    make_optimizer,
    oracle_best,
    pl_topk_prob,
    settings,
    validate_launch,
)

HERE = Path(__file__).resolve().parent
GIB = 2**30


def rows(n=205):
    return [dict(context_sha256=f"c{i:03}", family_id=f"f{i:03}") for i in range(n)]


def test_frozen_constants_and_arms():
    assert stat.ARMS == ("CONTROL", "T4_HINGE", "T4_PL")
    assert (
        stat.SEEDS == (260101, 260102, 260103)
        and stat.FOLD_SEED == 260140
        and stat.BOOT_SEED == 260150
    )
    assert (CONTROL_LR, CONTROL_WD, BETAS) == (1e-4, 1e-4, (0.9, 0.999))
    assert (AUX_WEIGHT, MARGIN_FRACTION, PL_TAU) == (0.5, 0.05, 1.0)
    assert stat.GATE_ZGAP == 0.25 and stat.TOP4_R882_DLVAL == 1.2403601682250796
    assert stat.SEEDS_POSITIVE_REQUIRED == 3 and stat.SINGLE_PICK_GUARD == 0.002
    assert (
        settings("T4_HINGE", 1)["aux_loss"] == "hinge@4"
        and settings("CONTROL", 1)["aux_weight"] is None
    )
    with pytest.raises(ValueError):
        settings("T4_PL", 41)


def test_schedule_table_matches_settings():
    table = json.loads((HERE / "schedules.json").read_text())["arms"]
    assert set(table) == set(stat.ARMS)
    for arm in stat.ARMS:
        assert table[arm] == [dict(step=t, **settings(arm, t)) for t in range(1, 41)]


def test_top_zgap_matches_h252_definition():
    cost = np.array([0.5, 0.1, 0.3, 0.9], dtype=np.float32)
    sd = float(np.std(cost))
    assert stat.top_zgap(cost) == pytest.approx((0.3 - 0.1) / sd, rel=1e-6)
    assert stat.top_zgap(np.ones(5, dtype=np.float32)) == 999.0


def test_top4_pick_gated_ungated_ties_and_recall():
    fitness = np.array([-1.0, -3.0, -2.0, -0.5, -5.0, -4.0])
    gate = np.array([True, True, False, True, True, True])
    # Gated: tiny gap between the two lowest costs. Top-4 by cost = [0, 1, 2, 3]; gated costs -1, -3, 2.0, -0.5.
    cost = np.array([0.10, 0.11, 0.5, 0.6, 0.9, 1.0], dtype=np.float32)
    u, gated, rec = stat.top4_pick(cost, fitness, gate)
    assert (
        gated and u == 3.0 and rec is False
    )  # oracle best (index 4, -5) is outside the top-4
    # Ungated: large gap -> single pick of argmin cost, 0 reveals.
    cost = np.array([0.0, 5.0, 5.1, 5.2, 5.3, 5.4], dtype=np.float32)
    u, gated, rec = stat.top4_pick(cost, fitness, gate)
    assert not gated and u == 1.0
    # A non-gated (invalid) pick scores -2.0; recall counts gated-cost ties.
    u, _, _ = stat.top4_pick(
        np.array([0.0, 9, 9, 9, 9, 9], dtype=np.float32),
        fitness,
        np.array([False] + [True] * 5),
    )
    assert u == -2.0
    tie_f = np.array([-1.0, -1.0, 0.0, 0.0, 0.0])
    u, gated, rec = stat.top4_pick(
        np.array([0.2, 0.1, 0.15, 0.3, 0.9], dtype=np.float32), tie_f, np.ones(5, bool)
    )
    assert gated and u == 1.0 and rec
    # Tie in revealed gated cost -> model (cost) order: index 1 (cost 0.1) before index 0.
    order_pick = np.argsort(np.array([0.2, 0.1, 0.15, 0.3, 0.9]), kind="stable")[:4]
    assert int(order_pick[0]) == 1


def test_oracle_best_uses_gated_cost_first_index():
    fitness = torch.tensor([[-1.0, -9.0, -1.0], [0.5, 0.5, 0.7]])
    gate = torch.tensor([[True, False, True], [True, True, True]])
    assert oracle_best(fitness, gate).tolist() == [0, 0]


def brute_force_topk(w, i, k):
    total = 0.0
    n = len(w)
    for perm in permutations(range(n)):
        p, rem = 1.0, 1.0
        for j in perm:
            p *= w[j] / rem
            rem -= w[j]
        if i in perm[:k]:
            total += p
    return total


def test_plackett_luce_top4_is_exact():
    torch.manual_seed(0)
    pred = torch.randn(3, 6)
    best = torch.tensor([0, 3, 5])
    got = pl_topk_prob(pred, best, tau=1.0, k=4)
    w = torch.softmax(-pred.double(), dim=1).numpy()
    for r in range(3):
        assert got[r].item() == pytest.approx(
            brute_force_topk(w[r], int(best[r]), 4), rel=1e-10
        )
    full = pl_topk_prob(pred, best, tau=1.0, k=6)
    assert torch.allclose(full, torch.ones(3, dtype=torch.float64))
    pred24 = torch.randn(2, 24, requires_grad=True)
    loss = -torch.log(pl_topk_prob(pred24, torch.tensor([3, 7]))).mean()
    loss.backward()
    assert torch.isfinite(pred24.grad).all() and pred24.grad.abs().sum() > 0


def test_hinge_at4_boundary_and_gradient():
    pred = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]], requires_grad=True)
    best = torch.tensor([5])
    loss = hinge_at4(pred, best, 0.1)
    assert loss.item() == pytest.approx(5.0 - 3.0 + 0.1)
    loss.backward()
    assert pred.grad[0, 5].item() == 1.0 and pred.grad[0, 3].item() == -1.0
    inside = hinge_at4(
        torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]]), torch.tensor([0]), 0.1
    )
    assert inside.item() == 0.0  # best well inside the top-4 by more than the margin


def test_arm_loss_composition():
    def ctrl(pred, fitness, gate):
        return pred.sum() * 0

    pred = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])
    fitness = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, -1.0]])
    gate = torch.ones(1, 6, dtype=torch.bool)
    total, aux = arm_loss("CONTROL", ctrl, pred, fitness, gate)
    assert aux is None and total.item() == 0.0
    total, aux = arm_loss("T4_HINGE", ctrl, pred, fitness, gate, 0.1)
    assert total.item() == pytest.approx(AUX_WEIGHT * 2.1)
    total, aux = arm_loss("T4_PL", ctrl, pred, fitness, gate)
    want = -math.log(pl_topk_prob(pred, torch.tensor([5])).item())
    assert total.item() == pytest.approx(AUX_WEIGHT * want, rel=1e-6)
    with pytest.raises(ValueError):
        arm_loss("T4_HINGE", ctrl, pred, fitness, gate, None)
    with pytest.raises(ValueError):
        arm_loss("SAM", ctrl, pred, fitness, gate)


def test_control_is_exact_torch_adamw_for_every_arm():
    for arm in stat.ARMS:
        torch.manual_seed(3)
        a = torch.nn.Parameter(torch.randn(5, 3))
        b = torch.nn.Parameter(a.detach().clone())
        ours, ref = (
            make_optimizer(arm, [a]),
            torch.optim.AdamW([b], lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4),
        )
        assert type(ours) is ControlAdamW
        for _ in range(3):
            g = torch.randn(5, 3)
            a.grad, b.grad = g.clone(), g.clone()
            ours.filter_gradients()
            ours.step()
            ref.step()
        assert torch.equal(a.detach(), b.detach())
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
    assert fm["fold_seed"] == 260140
    assert all([len(g) for g in v] == [41] * 5 for v in fm["families_by_seed"].values())


def test_selector_keeps_r882_on_ties_and_noise():
    r = rows(40)
    flat = np.ones((40, 5))
    assert stat.choose(r, flat, 1)["selected_step"] == 0
    noisy = flat.copy()
    noisy[:, 4] += np.where(np.arange(40) % 2 == 0, 0.5, -0.49)
    d = stat.choose(r, noisy, 1)
    assert d["best_step"] == 40 and d["selected_step"] == 0


def test_decision_rule_with_single_pick_guard():
    r = rows(19)
    flat = np.ones((3, 19, 4))
    lift = flat.copy()
    lift[:, :, 2] += 0.01
    single_ok = flat.copy()
    out = stat.decide(stat.summarize(r, lift), stat.summarize(r, single_ok))
    assert out["contrasts"]["T4_HINGE_minus_r882"]["advance"]
    assert not out["contrasts"]["T4_PL_minus_r882"]["advance"]
    single_bad = flat.copy()
    single_bad[:, :, 2] -= 0.003  # single-pick regresses by more than 0.002
    out = stat.decide(stat.summarize(r, lift), stat.summarize(r, single_bad))
    c = out["contrasts"]["T4_HINGE_minus_r882"]
    assert not c["advance"] and not c["single_pick_guard_pass"]
    two = flat.copy()
    two[:2, :, 2] += 0.01
    assert not stat.decide(stat.summarize(r, two), stat.summarize(r, flat))[
        "contrasts"
    ]["T4_HINGE_minus_r882"]["advance"]
    assert set(stat.summarize(r, flat)["contrasts"]) == {
        "CONTROL_minus_r882",
        "T4_HINGE_minus_r882",
        "T4_HINGE_minus_CONTROL",
        "T4_PL_minus_r882",
        "T4_PL_minus_CONTROL",
    }


def test_validate_launch_binding_v1_next():
    launch = dict(
        manifest_sha256="m",
        merged_source_verified=True,
        source_commit="h",
        publication_ref="v1/next",
        published_ancestor_verified=True,
        review_receipt_sha256="a" * 64,
        prereg_sha256="p",
        fold_manifest_sha256="f",
    )
    validate_launch(launch, "h", "m", "p", "f")
    for key, bad in [
        ("source_commit", "z"),
        ("publication_ref", "main"),
        ("review_receipt_sha256", "x"),
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
    t0 = 1793716000.0
    assert g.update(sample(t0, 19 * GIB)) == ([], True)
    assert g.update(sample(t0 + 32, 26 * GIB))[1] is True
    assert g.update(sample(t0 + 64, 26 * GIB)) == ([], False)
    assert "MemAvailable<6GiB" in mon.Guard().update(sample(t0, 5 * GIB))[0]
    assert mon.Guard().update(sample(t0, 40 * GIB, load5=109.0))[1] is True
    assert mon.COMPUTE_STOP == 1793726100.0  # 2026-11-03T17:15:00Z
    assert "absolute compute stop 17:15Z" in mon.violations(
        sample(mon.COMPUTE_STOP, 40 * GIB)
    )


def test_manifest_pins():
    m = json.loads((HERE / "manifest.json").read_text())
    assert m["wave"] == "H260" and m["total_updates"] == 1800
    assert m["inputs"][m["checkpoint"]].startswith("0084479a")
    assert m["dope_composite_sha256"].startswith("c51a56d0")
    assert m["recall4"]["aux_weight"] == 0.5 and m["recall4"]["margin_fraction"] == 0.05
    assert (
        "treatment.py" in m["source_sha256"] and "schedules.json" in m["source_sha256"]
    )
