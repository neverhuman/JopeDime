"""Data-free H249 controls: zero-init low-rank residual side branch (SIDE4 rank 4, SIDE16 rank 16) on frozen r882
2-D adapter matrices, update-0 bit-identity, gradient flow through (A, B) with frozen r882 weights untouched, exact
CTRL AdamW, frozen schedules, fold seed 249140 partition, strict selector, Holm m=2 decision rule, and H239 A1 guard.
Toy tensors are never study data."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import analysis as stat
import monitor as mon
from treatment import (
    ALPHAS,
    BETAS,
    CONTROL_LR,
    CONTROL_WD,
    RANKS,
    SIDE_LR,
    SIDE_WD,
    TRAINABLE_PARAMS,
    ControlAdamW,
    SideBranch,
    SideBranchAdamW,
    adapter_drift,
    arm_loss,
    make_optimizer,
    settings,
    setup_arm,
    validate_launch,
)

HERE = Path(__file__).resolve().parent
GIB = 2**30


def rows(n=205):
    return [dict(context_sha256=f"c{i:03}", family_id=f"f{i:03}") for i in range(n)]


def test_frozen_hyperparameters_and_arms():
    assert stat.ARMS == ("CONTROL", "SIDE4", "SIDE16")
    assert stat.SEEDS == (249101, 249102, 249103) and stat.FOLD_SEED == 249140
    assert (CONTROL_LR, CONTROL_WD, BETAS) == (1e-4, 1e-4, (0.9, 0.999))
    assert (SIDE_LR, SIDE_WD) == (1e-4, 1e-2)
    assert RANKS == {"CONTROL": None, "SIDE4": 4, "SIDE16": 16}
    assert ALPHAS == {"CONTROL": None, "SIDE4": 4.0, "SIDE16": 16.0}
    assert TRAINABLE_PARAMS == {"CONTROL": 29211, "SIDE4": 6232, "SIDE16": 24928}
    assert all(
        settings("CONTROL", t) == (1e-4, 0.9, 0.999, 1e-4, None, None, 29211)
        for t in range(1, 41)
    )
    assert all(
        settings("SIDE4", t) == (1e-4, 0.9, 0.999, 1e-2, 4, 4.0, 6232)
        for t in range(1, 41)
    )
    assert all(
        settings("SIDE16", t) == (1e-4, 0.9, 0.999, 1e-2, 16, 16.0, 24928)
        for t in range(1, 41)
    )
    with pytest.raises(ValueError):
        settings("SIDE4", 41)


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
                r["rank"],
                r["alpha"],
                r["trainable_params"],
            ) == settings(arm, r["step"])


def test_side_branch_step0_bit_identity_and_kaiming_b_zero():
    initial = {
        "l1.weight": torch.randn(16, 32, generator=torch.Generator().manual_seed(1)),
        "l1.bias": torch.randn(16, generator=torch.Generator().manual_seed(2)),
        "l2.weight": torch.randn(8, 16, generator=torch.Generator().manual_seed(3)),
    }
    b4 = SideBranch(initial, rank=4, seed=42)
    b16 = SideBranch(initial, rank=16, seed=42)
    assert len(b4.parameters()) == 4 and sum(
        p.numel() for p in b4.parameters()
    ) == 4 * (16 + 32) + 4 * (8 + 16)
    assert len(b16.parameters()) == 4 and sum(
        p.numel() for p in b16.parameters()
    ) == 16 * (16 + 32) + 16 * (8 + 16)
    for n in ("l1.weight", "l2.weight"):
        a, b = b4.pairs[n]
        assert torch.count_nonzero(a) > 0
        assert torch.count_nonzero(b) == 0
        assert torch.equal(b4.delta(n), torch.zeros_like(initial[n]))
    merged = b4.merged_state()
    for n in initial:
        assert torch.equal(merged[n], initial[n])
    assert adapter_drift(initial, merged) == 0.0
    assert b4.branch_norm() > 0.0
    with pytest.raises(ValueError):
        SideBranch(initial, rank=8)
    with pytest.raises(ValueError):
        SideBranch(initial, rank=4, alpha=8.0)


def test_side_branch_applied_training_leaves_base_frozen_and_matches_merged():
    class ToyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(6, 4)
            self.l2 = torch.nn.Linear(4, 2)

        def forward(self, x):
            return self.l2(torch.tanh(self.l1(x)))

    torch.manual_seed(7)
    net = ToyNet()
    names = tuple(n for n, _ in net.named_parameters())
    initial = {n: net.state_dict()[n].detach().clone() for n in names}
    branch, params, _, opt = setup_arm("SIDE4", net, names, initial, seed=99)
    assert isinstance(opt, SideBranchAdamW)
    assert all(not p.requires_grad for p in net.parameters())
    x = torch.randn(5, 6)
    with torch.no_grad():
        ref0 = net(x)
        with branch.applied(net):
            out0 = net(x)
    assert torch.equal(ref0, out0)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        with branch.applied(net):
            pred = net(x)
        loss = (pred**2).mean()
        loss.backward()
        opt.step()
    for n in names:
        assert torch.equal(net.state_dict()[n], initial[n])
    assert adapter_drift(initial, branch.merged_state()) > 0.0
    with torch.no_grad():
        with branch.applied(net):
            out_applied = net(x)
        merged = branch.merged_state()
        for n, p in net.named_parameters():
            p.copy_(merged[n])
        out_merged = net(x)
    assert torch.equal(out_applied, out_merged)


def test_control_is_exact_torch_adamw():
    torch.manual_seed(3)
    a = torch.nn.Parameter(torch.randn(5, 3))
    b = torch.nn.Parameter(a.detach().clone())
    ours, ref = (
        make_optimizer("CONTROL", [a]),
        torch.optim.AdamW([b], lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4),
    )
    assert isinstance(ours, ControlAdamW)
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
    assert fm["fold_seed"] == 249140
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
        assert c["harm_family_counts_by_seed"] == [0, 0, 0]
    assert out["vs_plateau_best_by_seed"][0][0] == pytest.approx(1 - stat.PLATEAU_BEST)
    lift = panel.copy()
    lift[:, :, 2] += 0.01
    out = stat.summarize(r, lift)
    assert out["contrasts"]["SIDE4_minus_r882"]["advance"]
    assert not out["contrasts"]["SIDE16_minus_r882"]["advance"]
    small = panel.copy()
    small[:, :, 3] += 0.0015
    assert not stat.summarize(r, small)["contrasts"]["SIDE16_minus_r882"]["advance"]
    harm = panel.copy()
    harm[:, 0, 2] -= 1
    assert (
        stat.summarize(r, harm)["contrasts"]["SIDE4_minus_r882"][
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
        affinity=[52, 53, 54, 55],
    )


def test_a1_memory_guard_and_deadline():
    g = mon.Guard()
    t0 = 1793700000.0
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


def test_manifest_pins():
    m = json.loads((HERE / "manifest.json").read_text())
    assert m["wave"] == "H249" and m["total_updates"] == 1800
    assert m["inputs"][m["checkpoint"]].startswith("0084479a")
    assert m["dope_composite_sha256"].startswith("c51a56d0")
    assert (
        "treatment.py" in m["source_sha256"] and "schedules.json" in m["source_sha256"]
    )
