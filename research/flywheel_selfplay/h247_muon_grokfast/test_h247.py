"""Data-free H247 controls: Muon (Newton-Schulz-5, momentum, scaling, parameter split), Grokfast-EMA filter, exact
CTRL AdamW, frozen settings, folds, selector, decision rule and A1 guard. Toy tensors are never study data."""

import json
import math
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
    GROK_ALPHA,
    GROK_LAMBDA,
    MUON_LR,
    MUON_MOMENTUM,
    MUON_WD,
    Composite,
    ControlAdamW,
    GrokfastAdamW,
    Muon,
    adapter_drift,
    arm_loss,
    make_optimizer,
    newton_schulz5,
    settings,
    validate_launch,
)

HERE = Path(__file__).resolve().parent
GIB = 2**30


def rows(n=205):
    return [dict(context_sha256=f"c{i:03}", family_id=f"f{i:03}") for i in range(n)]


def test_frozen_hyperparameters_and_arms():
    assert stat.ARMS == ("CONTROL", "MUON", "GROKFAST")
    assert stat.SEEDS == (247101, 247102, 247103) and stat.FOLD_SEED == 247140
    assert (CONTROL_LR, CONTROL_WD, BETAS) == (1e-4, 1e-4, (0.9, 0.999))
    assert (MUON_LR, MUON_MOMENTUM, MUON_WD) == (2e-4, 0.95, 1e-4)
    assert (GROK_ALPHA, GROK_LAMBDA) == (0.98, 2.0)
    for arm in stat.ARMS:
        assert all(
            settings(arm, t)[:4] == (1e-4, 0.9, 0.999, 1e-4) for t in range(1, 41)
        )
    assert settings("MUON", 1)[4] == 2e-4 and settings("CONTROL", 1)[4] is None
    assert settings("GROKFAST", 7)[5:] == (0.98, 2.0)
    with pytest.raises(ValueError):
        settings("MUON", 41)


def test_schedule_table_matches_settings():
    table = json.loads((HERE / "schedules.json").read_text())["arms"]
    for arm in stat.ARMS:
        assert [r["step"] for r in table[arm]] == list(range(1, 41))
        for r in table[arm]:
            assert (
                r["lr"],
                r["beta1"],
                r["beta2"],
                r["weight_decay"],
                r["muon_lr"],
                r["grok_alpha"],
                r["grok_lambda"],
            ) == settings(arm, r["step"])


def test_newton_schulz_approximates_orthogonal_polar_factor():
    g = torch.randn(24, 50, generator=torch.Generator().manual_seed(1))
    o = newton_schulz5(g)
    assert o.shape == g.shape
    sv = torch.linalg.svdvals(o)
    assert (
        float(sv.min()) > 0.5 and float(sv.max()) < 1.3
    )  # quintic NS: singular values pushed near 1
    u, _, vh = torch.linalg.svd(g, full_matrices=False)
    assert float((o * (u @ vh)).sum() / (o.norm() * (u @ vh).norm())) > 0.95
    tall = newton_schulz5(g.T)
    assert torch.allclose(tall, o.T, atol=1e-5)
    with pytest.raises(ValueError):
        newton_schulz5(torch.randn(5))


def test_muon_step_formula_momentum_and_scaling():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(48, 16))
    start = p.detach().clone()
    opt = Muon([p])
    g1, g2 = torch.randn(48, 16), torch.randn(48, 16)
    p.grad = g1.clone()
    opt.step()
    buf = g1.clone()
    want = start * (1 - MUON_LR * MUON_WD) - MUON_LR * math.sqrt(
        48 / 16
    ) * newton_schulz5(g1 + MUON_MOMENTUM * buf)
    assert torch.allclose(p.detach(), want, atol=1e-6)
    mid = p.detach().clone()
    p.grad = g2.clone()
    opt.step()
    buf = MUON_MOMENTUM * buf + g2
    want = mid * (1 - MUON_LR * MUON_WD) - MUON_LR * math.sqrt(3.0) * newton_schulz5(
        g2 + MUON_MOMENTUM * buf
    )
    assert torch.allclose(p.detach(), want, atol=1e-6)
    wide = torch.nn.Parameter(torch.zeros(16, 48))
    wide.grad = torch.randn(16, 48)
    Muon([wide]).step()  # scale sqrt(max(1, 16/48)) = 1
    assert torch.allclose(
        wide.detach(),
        -MUON_LR * newton_schulz5(wide.grad + MUON_MOMENTUM * wide.grad),
        atol=1e-6,
    )
    with pytest.raises(ValueError):
        Muon([torch.nn.Parameter(torch.zeros(3))])


def test_muon_arm_splits_matrices_and_others_and_skips_none():
    mat = torch.nn.Parameter(torch.randn(8, 4))
    vec = torch.nn.Parameter(torch.randn(8))
    idle = torch.nn.Parameter(torch.randn(4, 4))
    opt = make_optimizer("MUON", [mat, vec, idle])
    assert isinstance(opt, Composite)
    assert type(opt.adamw) is ControlAdamW and [
        p for g in opt.adamw.param_groups for p in g["params"]
    ] == [vec]
    assert [p for g in opt.muon.param_groups for p in g["params"]] == [mat, idle]
    assert (
        opt.param_groups[0]["lr"] == CONTROL_LR
        and opt.param_groups[0]["betas"] == BETAS
    )
    before = idle.detach().clone()
    mat.grad, vec.grad = torch.randn(8, 4), torch.randn(8)
    opt.filter_gradients()
    opt.step()
    assert torch.equal(idle.detach(), before)
    opt.zero_grad(set_to_none=True)
    assert mat.grad is None and vec.grad is None


def test_control_is_exact_torch_adamw():
    torch.manual_seed(3)
    a = torch.nn.Parameter(torch.randn(5, 3))
    b = torch.nn.Parameter(a.detach().clone())
    ours, ref = (
        make_optimizer("CONTROL", [a]),
        torch.optim.AdamW([b], lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4),
    )
    for _ in range(5):
        g = torch.randn(5, 3)
        a.grad, b.grad = g.clone(), g.clone()
        ours.filter_gradients()
        ours.step()
        ref.step()
    assert torch.equal(a.detach(), b.detach())


def test_grokfast_ema_filter_matches_reference():
    p = torch.nn.Parameter(torch.zeros(4))
    q = torch.nn.Parameter(torch.zeros(4))
    opt = make_optimizer("GROKFAST", [p, q])
    assert isinstance(opt, GrokfastAdamW)
    grads = [
        torch.randn(4, generator=torch.Generator().manual_seed(i)) for i in range(4)
    ]
    ema = None
    for g in grads:
        p.grad, q.grad = g.clone(), None
        opt.filter_gradients()
        ema = g.clone() if ema is None else ema
        ema = GROK_ALPHA * ema + (1 - GROK_ALPHA) * g
        assert torch.allclose(p.grad, g + GROK_LAMBDA * ema, atol=1e-7)
        assert q.grad is None
    first = make_optimizer("GROKFAST", [torch.nn.Parameter(torch.zeros(2))])
    par = first.param_groups[0]["params"][0]
    par.grad = torch.tensor([1.0, -2.0])
    first.filter_gradients()
    assert torch.allclose(
        par.grad, torch.tensor([3.0, -6.0])
    )  # g + 2 g on the first step


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
    a = {"x": torch.zeros(2)}
    assert adapter_drift(a, {"x": torch.tensor([3.0, 4.0])}) == 5.0


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
    assert fm["fold_seed"] == 247140
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


def test_decision_rule_four_gates_and_reports():
    r = rows(19)
    panel = np.ones((3, 19, 4))
    out = stat.summarize(r, panel)
    for arm in stat.ARMS[1:]:
        c = out["contrasts"][arm + "_minus_r882"]
        assert not c["advance"] and c["holm_p"] == 1.0
        assert c["harm_families_by_seed"] == [[], [], []]
    assert out["vs_plateau_best_by_seed"][0][0] == pytest.approx(1 - stat.PLATEAU_BEST)
    lift = panel.copy()
    lift[:, :, 2] += 0.01
    out = stat.summarize(r, lift)
    assert out["contrasts"]["MUON_minus_r882"]["advance"]
    assert not out["contrasts"]["GROKFAST_minus_r882"]["advance"]
    small = panel.copy()
    small[:, :, 3] += 0.0015
    assert not stat.summarize(r, small)["contrasts"]["GROKFAST_minus_r882"]["advance"]
    harm = panel.copy()
    harm[:, 0, 2] -= 1
    assert (
        stat.summarize(r, harm)["contrasts"]["MUON_minus_r882"]["harm_families_by_seed"]
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
    t0 = 1793680000.0
    assert g.update(sample(t0, 19 * GIB)) == ([], True)
    assert g.update(sample(t0 + 32, 26 * GIB))[1] is True
    assert g.update(sample(t0 + 64, 26 * GIB)) == ([], False)
    h = mon.Guard()
    h.update(sample(t0, 11 * GIB))
    assert "MemAvailable<12GiB sustained30s" in h.update(sample(t0 + 31, 11 * GIB))[0]
    assert "MemAvailable<6GiB" in mon.Guard().update(sample(t0, 5 * GIB))[0]
    assert mon.COMPUTE_STOP == 1793698200.0  # 2026-11-03T09:30:00Z
    assert "absolute compute stop 09:30Z" in mon.violations(
        sample(mon.COMPUTE_STOP, 40 * GIB)
    )


def test_manifest_pins():
    m = json.loads((HERE / "manifest.json").read_text())
    assert m["wave"] == "H247" and m["total_updates"] == 1800
    assert m["inputs"][m["checkpoint"]].startswith("0084479a")
    assert m["dope_composite_sha256"].startswith("c51a56d0")
    assert (
        "treatment.py" in m["source_sha256"] and "schedules.json" in m["source_sha256"]
    )
