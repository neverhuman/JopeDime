"""Data-free H235 controls; toy gradients are never study-training datasets."""

from types import SimpleNamespace
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import analysis as stat
import monitor as mon
from treatment import (
    ADD_WEIGHT,
    BETA,
    HEAD_ARMS,
    LOGVAR_CLAMP,
    arm_loss,
    beta_nll,
    ctrl_listwise,
    ctrl_regression,
    log_variance,
    make_head,
    make_optimizer,
    settings,
    gradient_norm,
    adapter_drift,
    validate_launch,
)

HERE = Path(__file__).resolve().parent


def rows():
    return [dict(context_sha256=f"c{i:03}", family_id=f"f{i:03}") for i in range(205)]


def test_every_arm_uses_the_identical_h170_ctrl_adamw_recipe():
    assert stat.ARMS == ("CONTROL", "BNLL_MIX", "BNLL_ADD") and HEAD_ARMS == (
        "BNLL_MIX",
        "BNLL_ADD",
    )
    for arm in stat.ARMS:
        assert all(settings(arm, t) == (1e-4, 0.9, 0.999, 1e-4) for t in range(1, 41))
        opt = make_optimizer(arm, [torch.nn.Parameter(torch.ones(1))])
        assert isinstance(opt, torch.optim.AdamW)
        assert (
            opt.param_groups[0]["lr"] == 1e-4
            and opt.param_groups[0]["weight_decay"] == 1e-4
        )
    for step in [0, 41]:
        with pytest.raises(ValueError):
            settings("BNLL_MIX", step)
    table = json.loads((HERE / "schedules.json").read_text())["arms"]
    assert table == {
        a: [
            dict(step=t, lr=1e-4, beta1=0.9, beta2=0.999, weight_decay=1e-4)
            for t in range(1, 41)
        ]
        for a in stat.ARMS
    }


def ctrl_loss_reference(pred, fitness, gate):
    import torch.nn.functional as F

    target = torch.where(gate, fitness.clamp(-4.0, 2.0), torch.full_like(fitness, 2.0))
    return (
        F.smooth_l1_loss(pred, target, reduction="none").mean(1)
        + 0.1
        * F.kl_div(
            F.log_softmax(-pred / 0.2, dim=-1),
            F.softmax(-target / 0.2, dim=-1),
            reduction="none",
        ).sum(1)
    ).mean()


def toy(seed=5, b=8, n=24):
    rng = np.random.default_rng(seed)
    pred = torch.tensor(
        rng.normal(size=(b, n)), dtype=torch.float64, requires_grad=True
    )
    fitness = torch.tensor(rng.normal(size=(b, n)), dtype=torch.float64)
    gate = torch.tensor(rng.random((b, n)) > 0.1)
    return pred, fitness, gate


def test_ctrl_terms_split_exactly_into_regression_and_listwise():
    pred, fitness, gate = toy()
    split = (
        ctrl_regression(pred, fitness, gate) + ctrl_listwise(pred, fitness, gate)
    ).mean()
    assert torch.allclose(
        split, ctrl_loss_reference(pred, fitness, gate), rtol=0, atol=1e-12
    )


def test_beta_nll_matches_seitzer_with_stopgrad_weights():
    pred, fitness, gate = toy()
    logvar = torch.tensor(
        np.random.default_rng(9).uniform(-2, 2, size=(8, 24)), requires_grad=True
    )
    got = beta_nll(pred, logvar, fitness, gate)
    target = torch.where(gate, fitness.clamp(-4.0, 2.0), torch.full_like(fitness, 2.0))
    var = logvar.detach().exp()
    ref = (
        var**0.5 * 0.5 * (logvar.detach() + (target - pred.detach()) ** 2 / var)
    ).mean(1)
    assert torch.allclose(got.detach(), ref, rtol=0, atol=1e-12)
    got.sum().backward()
    # d/dlogvar with sigma^(2 beta) held constant: w * 0.5 * (1 - r^2 / sigma^2) / n
    r2 = (target - pred.detach()) ** 2
    expect = var**0.5 * 0.5 * (1 - r2 / var) / 24
    assert torch.allclose(logvar.grad, expect, rtol=0, atol=1e-12)
    # sigma^2 = 1 (log-variance 0): beta-NLL is half the squared error
    zero = beta_nll(
        pred.detach(), torch.zeros(8, 24, dtype=torch.float64), fitness, gate
    )
    assert torch.allclose(zero, 0.5 * r2.mean(1), rtol=0, atol=1e-12)
    assert BETA == 0.5


def test_variance_head_starts_at_zero_and_is_clamped():
    head = make_head(7)
    hidden = torch.randn(3, 5, 7)
    assert torch.equal(log_variance(head, hidden), torch.zeros(3, 5))
    with torch.no_grad():
        head.bias.fill_(50.0)
    assert torch.equal(log_variance(head, hidden), torch.full((3, 5), 6.0))
    with torch.no_grad():
        head.bias.fill_(-50.0)
    assert torch.equal(log_variance(head, hidden), torch.full((3, 5), -6.0))
    assert (
        LOGVAR_CLAMP == 6.0 and sum(p.numel() for p in make_head(7).parameters()) == 8
    )


def test_arm_loss_composition():
    pred, fitness, gate = toy()
    logvar = torch.zeros(8, 24, dtype=torch.float64)
    ctrl = ctrl_loss_reference(pred, fitness, gate)
    bn = beta_nll(pred, logvar, fitness, gate)
    assert torch.equal(
        arm_loss("CONTROL", ctrl_loss_reference, pred, fitness, gate), ctrl
    )
    mix = arm_loss("BNLL_MIX", ctrl_loss_reference, pred, fitness, gate, logvar)
    assert torch.allclose(
        mix, (bn + ctrl_listwise(pred, fitness, gate)).mean(), rtol=0, atol=1e-12
    )
    add = arm_loss("BNLL_ADD", ctrl_loss_reference, pred, fitness, gate, logvar)
    assert torch.allclose(add, ctrl + 0.5 * bn.mean(), rtol=0, atol=1e-12)
    assert ADD_WEIGHT == 0.5
    with pytest.raises(ValueError):
        arm_loss("BNLL_MIX", ctrl_loss_reference, pred, fitness, gate)
    with pytest.raises(ValueError):
        arm_loss("LAMBDA", ctrl_loss_reference, pred, fitness, gate, logvar)


def test_inherited_ctrl_regression_term_is_smooth_l1_plus_listwise_kl():
    """Pins the reading recorded in the PREREG: the H146 CTRL loss's regression term is smooth-L1 (Huber, squared
    below |error| 1), and its other term is the ListNet-style KL; BNLL_MIX replaces only the regression term."""
    src = (
        HERE.parents[1] / "flywheel_selfplay/h146_lookahead_exec/runtime.py"
    ).read_text()
    body = src.split("def loss(", 1)[1].split("\ndef ", 1)[0]
    assert (
        "F.smooth_l1_loss(pred, target" in body
        and "F.kl_div" in body
        and "mse" not in body.lower()
    )


def test_none_gradients_are_zero_in_norm_and_drift_includes_all_coordinates():
    a = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    b = torch.nn.Parameter(torch.ones(6))
    a.grad = torch.tensor([3.0, 4.0])
    assert gradient_norm([a, b]) == 5.0 and gradient_norm([b]) == 0.0
    assert b.grad is None
    initial = {"a": torch.zeros(2), "b": torch.ones(6)}
    final = {"a": torch.tensor([3.0, 4.0]), "b": torch.ones(6)}
    assert adapter_drift(initial, final) == 5.0
    with pytest.raises(ValueError):
        adapter_drift(initial, {"a": final["a"]})


def test_family_folds_fit_exclusion_and_exact_assessment_cover():
    source = rows()
    cover = []
    for outer in range(5):
        fit, select, assess = stat.partition(source, outer, stat.SEEDS[0])
        groups = [{r["family_id"] for r in rs} for rs in [fit, select, assess]]
        assert [len(g) for g in groups] == [123, 41, 41]
        assert not (
            groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]
        )
        cover += list(groups[2])
    assert len(cover) == len(set(cover)) == 205
    fit, _, _ = stat.partition(source, 0, stat.SEEDS[0])
    assert (
        len({json.dumps(stat.folds(source, s)) for s in stat.SEEDS}) == 3
    )  # one fresh partition per seed
    ids = [r["context_sha256"] for r in fit]
    orders = {c: np.arange(48) for c in ids}
    first = stat.batch_plan(ids, orders, 235101)
    other = stat.batch_plan(list(reversed(ids)), orders, 235101)
    assert len(first) == 40
    for (s, c, ix), (s2, c2, ix2) in zip(first, other):
        assert s == s2 and c == c2 and len(c) == 8 and set(c) <= set(ids)
        for a, b in zip(ix, ix2):
            assert np.array_equal(a, b) and len(set(a)) == 24


def test_strict_selector_reachability_ties_and_no_second_best_fallback():
    source = rows()[:41]
    panel = np.zeros((41, 5))
    assert stat.choose(source, panel, 1)["selected_step"] == 0
    panel[:, 1] = 0.005
    panel[:, 2] = 0.005 + 5e-13
    assert stat.choose(source, panel, 1)["selected_step"] == 10
    panel[:, 1:] = 0.001
    assert stat.choose(source, panel, 1)["selected_step"] == 0
    panel[:] = 0
    panel[:, 1] = np.r_[-np.ones(20), np.ones(21)]
    panel[:, 2] = 0.005
    pick = stat.choose(source, panel, 1)
    assert (
        pick["best_step"] == 10 and pick["ci95"][0] < 0 and pick["selected_step"] == 0
    )


def test_median_family_bootstrap_holm_and_canonical_elo():
    source = [{"family_id": "big"}] * 9 + [{"family_id": "small"}]
    panel = np.zeros((3, 10, 4))
    panel[:, :9, 1:] = 1.0
    panel[:, 9, 1:] = -1.0
    s = stat.summarize(source, panel)
    assert s["contrasts"]["BNLL_MIX_minus_r882"]["median_gain"] == 0.0
    assert np.allclose(s["median_elo"], 1500.0)
    panel[:] = 0
    panel[:, :, 2] = 0.005
    s = stat.summarize(source, panel)
    assert s["contrasts"]["BNLL_MIX_minus_r882"]["advance"]
    assert s["contrasts"]["BNLL_MIX_minus_r882"]["ci95"][0] > 0
    assert (
        "BNLL_MIX_minus_CONTROL" in s["contrasts"]
        and "CONTROL_minus_r882" in s["contrasts"]
    )
    # hw_t2120 margin: a +0.002 gain is a tie with the margin, not an advance
    panel[:, :, 2] = 0.002
    assert not stat.summarize(source, panel)["contrasts"]["BNLL_MIX_minus_r882"][
        "advance"
    ]
    panel[:, :, 2] = 0.0021
    assert stat.summarize(source, panel)["contrasts"]["BNLL_MIX_minus_r882"]["advance"]
    assert stat.ADVANCE_MARGIN == 0.002
    assert s["median_elo"][2] > 1500.0 and s["median_elo"][0] == 1500.0
    assert stat.holm({"a": 0.01, "b": 0.04, "c": 0.02}) == {
        "a": 0.03,
        "c": 0.04,
        "b": 0.04,
    }


def test_exact_utility_duplicates_not_equal_means_and_signed_zero():
    assert stat.duplicate_groups({"a": [1, -1], "b": [-1, 1]}) == []
    assert stat.duplicate_groups({"a": [-0.0, 1.0], "b": [0.0, 1.0]}) == [["a", "b"]]
    assert stat.duplicate_groups({"a": "hash1", "b": "hash1", "c": "hash2"}) == [
        ["a", "b"]
    ]
    with pytest.raises(ValueError):
        stat.duplicate_groups({"bad": [np.nan]})


def test_pseudoreplication_discloses_identical_predictions_despite_distinct_adapters():
    panel = np.zeros((3, 2, 4))
    cells = []
    for outer in range(5):
        for seed in stat.SEEDS:
            cells.append(
                dict(
                    outer_fold=outer,
                    seed=seed,
                    selector_panels={
                        a: [dict(values=[0.0, 0.0]) for _ in stat.STEPS]
                        for a in stat.ARMS
                    },
                    adapter_sha256={
                        a: {"0": "r882", "40": f"{outer}:{seed}:{a}"} for a in stat.ARMS
                    },
                    decisions={a: {"selected_step": 0} for a in stat.ARMS},
                )
            )
    report = stat.pseudoreplication(panel, panel, panel, panel, cells)
    assert len(report["strict_OOF_identical"][0]) == 12  # 3 seeds x (r882 + 3 arms)
    assert (
        report["step40_adapter_identical"] == []
        and report["all_arms_selector_identical_to_CONTROL"]
    )
    cells[0]["selector_panels"]["BNLL_MIX"][1]["values"] = [1.0, 0.0]
    assert not stat.pseudoreplication(panel, panel, panel, panel, cells)[
        "all_arms_selector_identical_to_CONTROL"
    ]


def test_hard_resource_boundaries_and_load_is_a_pause():
    good = dict(
        load=500.0,
        load5=500.0,
        cpu=100.0,
        available=20 * 2**30,
        free=300_000_000_000,
        rss=4 * 2**30 - 1,
        threads=4,
        utc=mon.COMPUTE_STOP - 1,
    )
    assert mon.violations(good) == []  # host load alone never stops the attempt
    for key, val in dict(
        available=20 * 2**30 - 1,
        free=299_999_999_999,
        rss=4 * 2**30,
        threads=5,
        utc=mon.COMPUTE_STOP,
    ).items():
        assert mon.violations(dict(good, **{key: val}))
    assert mon.COMPUTE_STOP == 1793682000.0  # 2026-11-03T05:00:00Z
    assert not mon.paused(dict(load5=60.0), False) and mon.paused(
        dict(load5=60.1), False
    )
    assert mon.paused(dict(load5=58.0), True) and not mon.paused(dict(load5=55.0), True)
    assert mon.admission_refusals(dict(good, load5=55.0)) == []
    assert mon.admission_refusals(dict(good, load5=55.1)) == ["admission: load5>55"]


def test_resource_tree_counts_supervisor_and_descendants(monkeypatch):
    grand = SimpleNamespace(
        pid=3, memory_info=lambda: SimpleNamespace(rss=30), num_threads=lambda: 1
    )
    child = SimpleNamespace(
        pid=2,
        memory_info=lambda: SimpleNamespace(rss=20),
        num_threads=lambda: 2,
        children=lambda recursive: [grand],
    )
    supervisor = SimpleNamespace(
        pid=1, memory_info=lambda: SimpleNamespace(rss=10), num_threads=lambda: 2
    )
    monkeypatch.setattr(
        mon.psutil, "Process", lambda pid=None: supervisor if pid is None else child
    )
    sample = mon.snapshot(SimpleNamespace(pid=2))
    assert sample["threads"] == 5 and sample["rss"] == 60
    assert "aggregate threads>4" in mon.violations(sample)


def test_external_launch_requires_review_and_all_source_protocol_bindings():
    launch = dict(
        manifest_sha256="m",
        source_commit="h",
        prereg_sha256="p",
        fold_manifest_sha256="f",
        review_id=123,
        prereg_pr=456,
        merged_source_verified=True,
    )
    validate_launch(launch, "h", "m", "p", "f")
    for key, bad in dict(
        manifest_sha256="x",
        source_commit="x",
        prereg_sha256="x",
        fold_manifest_sha256="x",
        review_id=0,
        prereg_pr=0,
        merged_source_verified=False,
    ).items():
        with pytest.raises(RuntimeError):
            validate_launch(dict(launch, **{key: bad}), "h", "m", "p", "f")
    validate_launch(dict(manifest_sha256="m"), "h", "m", "p", "f", smoke=True)


def test_boolean_ids_cannot_stand_in_for_a_real_external_review():
    launch = dict(
        manifest_sha256="m",
        source_commit="h",
        prereg_sha256="p",
        fold_manifest_sha256="f",
        review_id=True,
        prereg_pr=456,
        merged_source_verified=True,
    )
    with pytest.raises(RuntimeError):
        validate_launch(launch, "h", "m", "p", "f")


def test_runtime_loader_releases_the_early_research_namespace():
    """Amendment A2: after loading the pinned H146 runtime, no `research` package binding may remain, so the pinned
    H50 fused module can import JopeDimeMicro's regular `research` package (research.levels)."""
    import sys

    import runner

    rt = runner.runtime()
    assert rt.Audit.__name__ == "Audit"
    assert not [n for n in sys.modules if n == "research" or n.startswith("research.")]
