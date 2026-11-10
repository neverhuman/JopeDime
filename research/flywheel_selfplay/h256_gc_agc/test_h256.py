"""Data-free H256 controls: Gradient Centralization, unit-wise Adaptive Gradient Clipping, the GC -> AGC order, exact
CTRL AdamW, frozen settings, folds, selector, the decision rule (Holm m=2, 3/3 seeds) with the mechanism contrast and
the A1 guard. Toy tensors and toy rows are never study data."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import analysis as stat
import monitor as mon
from treatment import (
    AGC_EPS,
    AGC_LAMBDA,
    BETAS,
    CONTROL_LR,
    CONTROL_WD,
    FilteredAdamW,
    adapter_drift,
    adaptive_clip,
    arm_loss,
    centralize,
    make_optimizer,
    settings,
    validate_launch,
)

HERE = Path(__file__).resolve().parent
GIB = 2**30


def rows(n=205):
    return [dict(context_sha256=f"c{i:03}", family_id=f"f{i:03}") for i in range(n)]


def reference_agc(w, g, lamb=AGC_LAMBDA, eps=AGC_EPS):
    """Independent NFNet AGC reference (unitwise_norm: per output row for >= 2-D, whole tensor for <= 1-D)."""
    if g.dim() >= 2:
        wn = w.reshape(w.shape[0], -1).norm(dim=1).reshape([-1] + [1] * (g.dim() - 1))
        gn = g.reshape(g.shape[0], -1).norm(dim=1).reshape([-1] + [1] * (g.dim() - 1))
    else:
        wn, gn = w.norm(), g.norm()
    bound = lamb * torch.clamp(wn, min=eps)
    return torch.where(gn > bound, g * (bound / torch.clamp(gn, min=1e-6)), g)


def test_frozen_constants_and_arms():
    assert stat.ARMS == ("CONTROL", "GC_AGC", "AGC_ONLY")
    assert (
        stat.SEEDS == (256101, 256102, 256103)
        and stat.FOLD_SEED == 256140
        and stat.BOOT_SEED == 256150
    )
    assert (CONTROL_LR, CONTROL_WD, BETAS) == (1e-4, 1e-4, (0.9, 0.999))
    assert (AGC_LAMBDA, AGC_EPS) == (0.01, 1e-3)
    assert stat.SEEDS_POSITIVE_REQUIRED == 3 and stat.ADVANCE_MARGIN == 0.002
    assert (
        settings("GC_AGC", 1)["gradient_centralization"]
        and settings("GC_AGC", 1)["agc_lambda"] == 0.01
    )
    assert (
        not settings("AGC_ONLY", 1)["gradient_centralization"]
        and settings("AGC_ONLY", 1)["agc_lambda"] == 0.01
    )
    assert settings("CONTROL", 1)["agc_lambda"] is None
    with pytest.raises(ValueError):
        settings("GC_AGC", 41)


def test_schedule_table_matches_settings():
    table = json.loads((HERE / "schedules.json").read_text())["arms"]
    assert set(table) == set(stat.ARMS)
    for arm in stat.ARMS:
        assert table[arm] == [dict(step=t, **settings(arm, t)) for t in range(1, 41)]


def test_gradient_centralization_rows_and_3d():
    g = torch.randn(5, 7, generator=torch.Generator().manual_seed(0))
    want = g - g.mean(dim=1, keepdim=True)
    assert centralize(g) is True
    assert torch.allclose(g, want, atol=1e-7) and torch.allclose(
        g.mean(dim=1), torch.zeros(5), atol=1e-6
    )
    t = torch.randn(3, 2, 4, generator=torch.Generator().manual_seed(1))
    want = t - t.reshape(3, -1).mean(dim=1).reshape(3, 1, 1)
    centralize(t)
    assert torch.allclose(t, want, atol=1e-7)
    v = torch.randn(6)
    before = v.clone()
    assert centralize(v) is False and torch.equal(v, before)


def test_agc_matches_reference_for_every_shape():
    gen = torch.Generator().manual_seed(2)
    for shape in [(4, 6), (1, 9), (3, 2, 5), (8,), ()]:
        w = torch.randn(shape, generator=gen) * 0.05
        g = torch.randn(shape, generator=gen)
        want = reference_agc(w, g.clone())
        clipped, units = adaptive_clip(torch.nn.Parameter(w), g)
        assert torch.allclose(g, want, atol=1e-7)
        assert units == (shape[0] if len(shape) >= 2 else 1) and 0 <= clipped <= units
    w = torch.ones(2, 2)
    small = torch.full((2, 2), 1e-4)  # ||g_row|| = 1.4e-4 < 0.01 * sqrt(2): untouched
    assert adaptive_clip(w, small) == (0, 2) and torch.equal(
        small, torch.full((2, 2), 1e-4)
    )
    zero_w = torch.zeros(1, 3)
    g = torch.ones(1, 3)
    adaptive_clip(zero_w, g)  # eps floor: ||g|| clipped to lambda * 1e-3
    assert torch.linalg.vector_norm(g).item() == pytest.approx(
        AGC_LAMBDA * AGC_EPS, rel=1e-5
    )


def test_filter_order_gc_then_agc_and_none_skipped():
    w = torch.nn.Parameter(
        torch.randn(3, 4, generator=torch.Generator().manual_seed(3))
    )
    idle = torch.nn.Parameter(torch.randn(2, 2))
    g = torch.randn(3, 4, generator=torch.Generator().manual_seed(4))
    opt = make_optimizer("GC_AGC", [w, idle])
    w.grad, idle.grad = g.clone(), None
    stats = opt.filter_gradients()
    expected = reference_agc(w.detach(), g - g.mean(dim=1, keepdim=True))
    assert torch.allclose(w.grad, expected, atol=1e-7) and idle.grad is None
    assert stats["gc_tensors"] == 1 and stats["agc_units"] == 3
    only = make_optimizer("AGC_ONLY", [w])
    w.grad = g.clone()
    only.filter_gradients()
    assert torch.allclose(w.grad, reference_agc(w.detach(), g), atol=1e-7)
    ctrl = make_optimizer("CONTROL", [w])
    w.grad = g.clone()
    assert ctrl.filter_gradients() == dict(
        gc_tensors=0, agc_clipped_units=0, agc_units=0
    )
    assert torch.equal(w.grad, g)
    with pytest.raises(ValueError):
        make_optimizer("MUON", [w])


def test_control_is_exact_torch_adamw():
    torch.manual_seed(3)
    a = torch.nn.Parameter(torch.randn(5, 3))
    b = torch.nn.Parameter(a.detach().clone())
    ours, ref = (
        make_optimizer("CONTROL", [a]),
        torch.optim.AdamW([b], lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4),
    )
    assert type(ours) is FilteredAdamW and not ours.gc and not ours.agc
    for _ in range(5):
        g = torch.randn(5, 3)
        a.grad, b.grad = g.clone(), g.clone()
        ours.filter_gradients()
        ours.step()
        ref.step()
    assert torch.equal(a.detach(), b.detach())


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
    assert fm["fold_seed"] == 256140
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


def test_decision_rule_gates_and_contrasts():
    r = rows(19)
    panel = np.ones((3, 19, 4))
    out = stat.summarize(r, panel)
    assert set(out["contrasts"]) == {
        "CONTROL_minus_r882",
        "GC_AGC_minus_r882",
        "GC_AGC_minus_CONTROL",
        "GC_AGC_minus_AGC_ONLY",
        "AGC_ONLY_minus_r882",
        "AGC_ONLY_minus_CONTROL",
    }
    for arm in stat.ARMS[1:]:
        c = out["contrasts"][arm + "_minus_r882"]
        assert not c["advance"] and c["holm_p"] == 1.0
    lift = panel.copy()
    lift[:, :, 2] += 0.01
    out = stat.summarize(r, lift)
    assert out["contrasts"]["GC_AGC_minus_r882"]["advance"]
    assert out["contrasts"]["GC_AGC_minus_AGC_ONLY"]["median_gain"] == pytest.approx(
        0.01
    )
    assert not out["contrasts"]["AGC_ONLY_minus_r882"]["advance"]
    two_of_three = panel.copy()
    two_of_three[:2, :, 2] += 0.01  # 2/3 seeds positive is NOT enough under hw_t1335
    assert not stat.summarize(r, two_of_three)["contrasts"]["GC_AGC_minus_r882"][
        "advance"
    ]
    small = panel.copy()
    small[:, :, 3] += 0.0015
    assert not stat.summarize(r, small)["contrasts"]["AGC_ONLY_minus_r882"]["advance"]
    harm = panel.copy()
    harm[:, 0, 2] -= 1
    assert (
        stat.summarize(r, harm)["contrasts"]["GC_AGC_minus_r882"][
            "harm_families_by_seed"
        ]
        == [["f000"]] * 3
    )
    assert stat.CLIFF_CONTEXT == 27


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
        ("published_ancestor_verified", False),
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
    t0 = 1793712000.0
    assert g.update(sample(t0, 19 * GIB)) == ([], True)
    assert g.update(sample(t0 + 32, 26 * GIB))[1] is True
    assert g.update(sample(t0 + 64, 26 * GIB)) == ([], False)
    h = mon.Guard()
    h.update(sample(t0, 11 * GIB))
    assert "MemAvailable<12GiB sustained30s" in h.update(sample(t0 + 31, 11 * GIB))[0]
    assert "MemAvailable<6GiB" in mon.Guard().update(sample(t0, 5 * GIB))[0]
    assert mon.Guard().update(sample(t0, 40 * GIB, load5=109.0))[1] is True
    assert mon.COMPUTE_STOP == 1793721600.0  # 2026-11-03T16:00:00Z
    assert "absolute compute stop 16:00Z" in mon.violations(
        sample(mon.COMPUTE_STOP, 40 * GIB)
    )


def test_manifest_pins():
    m = json.loads((HERE / "manifest.json").read_text())
    assert m["wave"] == "H256" and m["total_updates"] == 1800
    assert m["inputs"][m["checkpoint"]].startswith("0084479a")
    assert m["dope_composite_sha256"].startswith("c51a56d0")
    assert m["gc_agc"]["agc_lambda"] == 0.01 and m["gc_agc"]["agc_eps"] == 0.001
    assert (
        "treatment.py" in m["source_sha256"] and "schedules.json" in m["source_sha256"]
    )
