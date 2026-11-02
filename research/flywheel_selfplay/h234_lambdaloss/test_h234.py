"""Data-free H234 controls; toy gradients are never study-training datasets."""

from types import SimpleNamespace
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import analysis as stat
import monitor as mon
from treatment import (
    LAMBDA_WEIGHT,
    arm_loss,
    lambda_loss,
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
    assert stat.ARMS == ("CONTROL", "LAMBDAMIX", "LAMBDA")
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
            settings("LAMBDA", step)
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


def test_lambda_loss_orders_ndcg_weights_and_ties():
    fitness = torch.tensor(
        [[-3.0, -2.0, -1.0, 0.0]]
    )  # target cost = fitness; best candidate is index 0
    gate = torch.ones_like(fitness, dtype=torch.bool)
    perfect = fitness.clone()  # predicted cost ordered like the target
    reversed_ = -fitness
    assert lambda_loss(perfect, fitness, gate) < lambda_loss(reversed_, fitness, gate)
    # constant targets: no strictly ordered pair -> zero loss
    flat = torch.zeros(1, 4)
    assert (
        lambda_loss(torch.randn(1, 4), flat, torch.ones(1, 4, dtype=torch.bool)) == 0.0
    )
    # a failed gate is a cost-2 target, i.e. the worst candidate
    gated = torch.tensor([[True, True, True, False]])
    assert lambda_loss(
        perfect, torch.tensor([[-3.0, -2.0, -1.0, -9.0]]), gated
    ) == pytest.approx(float(lambda_loss(perfect, fitness, gate)), abs=0.2)
    # gradient lowers the cost (raises the score) of the best candidate
    pred = torch.zeros(1, 4, requires_grad=True)
    lambda_loss(pred, fitness, gate).backward()
    assert (
        pred.grad[0, 0] > 0 and pred.grad[0, 3] < 0
    )  # descending pred[0] lowers its cost
    assert torch.isfinite(pred.grad).all()


def test_lambda_loss_matches_an_explicit_pairwise_reference():
    rng = np.random.default_rng(3)
    pred = torch.tensor(rng.normal(size=(2, 6)), dtype=torch.float64)
    fitness = torch.tensor(rng.normal(size=(2, 6)), dtype=torch.float64)
    gate = torch.tensor(rng.random((2, 6)) > 0.2)
    got = float(lambda_loss(pred, fitness, gate))
    losses = []
    for b in range(2):
        t = [float(fitness[b, i]) if gate[b, i] else 2.0 for i in range(6)]
        t = [min(max(x, -4.0), 2.0) if gate[b, i] else x for i, x in enumerate(t)]
        u = [-x for x in t]
        s = [-float(pred[b, i]) for i in range(6)]
        lo, hi = min(u), max(u)
        gain = [2 ** ((x - lo) / (hi - lo)) - 1 for x in u]
        order = sorted(range(6), key=lambda i: -s[i])
        rank = {i: r + 1 for r, i in enumerate(order)}
        idcg = sum(g / np.log2(r + 2) for r, g in enumerate(sorted(gain, reverse=True)))
        num = den = 0.0
        for i in range(6):
            for j in range(6):
                if u[i] > u[j]:
                    w = (
                        abs(gain[i] - gain[j])
                        * abs(1 / np.log2(1 + rank[i]) - 1 / np.log2(1 + rank[j]))
                        / idcg
                    )
                    num += w * np.log1p(np.exp(-(s[i] - s[j])))
                    den += w
        losses.append(num / den if den > 0 else 0.0)
    assert got == pytest.approx(float(np.mean(losses)), abs=1e-9)


def test_arm_loss_composition():
    rng = np.random.default_rng(5)
    pred = torch.tensor(rng.normal(size=(8, 24)), dtype=torch.float32)
    fitness = torch.tensor(rng.normal(size=(8, 24)), dtype=torch.float32)
    gate = torch.tensor(rng.random((8, 24)) > 0.1)
    ctrl = ctrl_loss_reference(pred, fitness, gate)
    lam = lambda_loss(pred, fitness, gate)
    assert torch.equal(
        arm_loss("CONTROL", ctrl_loss_reference, pred, fitness, gate), ctrl
    )
    assert torch.allclose(
        arm_loss("LAMBDAMIX", ctrl_loss_reference, pred, fitness, gate),
        ctrl + 0.5 * lam,
    )
    assert torch.equal(
        arm_loss("LAMBDA", ctrl_loss_reference, pred, fitness, gate), lam
    )
    assert LAMBDA_WEIGHT == 0.5
    with pytest.raises(ValueError):
        arm_loss("LION", ctrl_loss_reference, pred, fitness, gate)


def test_inherited_ctrl_loss_already_contains_a_listnet_term():
    """Why H234 tests LambdaLoss and not ListNet (hw_t2220 rule): the merged H146 CTRL loss already includes a
    listwise softmax cross-entropy (KL of softmax(-target/0.2) vs softmax(-pred/0.2), weight 0.1)."""
    src = (
        HERE.parents[1] / "flywheel_selfplay/h146_lookahead_exec/runtime.py"
    ).read_text()
    body = src.split("def loss(", 1)[1].split("\ndef ", 1)[0]
    assert (
        "F.kl_div" in body
        and "F.log_softmax(-pred / 0.2" in body
        and "F.softmax(-target / 0.2" in body
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
    first = stat.batch_plan(ids, orders, 234101)
    other = stat.batch_plan(list(reversed(ids)), orders, 234101)
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
    assert s["contrasts"]["LAMBDAMIX_minus_r882"]["median_gain"] == 0.0
    assert np.allclose(s["median_elo"], 1500.0)
    panel[:] = 0
    panel[:, :, 2] = 0.005
    s = stat.summarize(source, panel)
    assert s["contrasts"]["LAMBDAMIX_minus_r882"]["advance"]
    assert s["contrasts"]["LAMBDAMIX_minus_r882"]["ci95"][0] > 0
    assert (
        "LAMBDAMIX_minus_CONTROL" in s["contrasts"]
        and "CONTROL_minus_r882" in s["contrasts"]
    )
    # hw_t2120 margin: a +0.002 gain is a tie with the margin, not an advance
    panel[:, :, 2] = 0.002
    assert not stat.summarize(source, panel)["contrasts"]["LAMBDAMIX_minus_r882"][
        "advance"
    ]
    panel[:, :, 2] = 0.0021
    assert stat.summarize(source, panel)["contrasts"]["LAMBDAMIX_minus_r882"]["advance"]
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
    cells[0]["selector_panels"]["LAMBDAMIX"][1]["values"] = [1.0, 0.0]
    assert not stat.pseudoreplication(panel, panel, panel, panel, cells)[
        "all_arms_selector_identical_to_CONTROL"
    ]


def test_hard_resource_boundaries_and_load_is_a_pause():
    good = dict(
        load=500.0,
        load5=500.0,
        cpu=100.0,
        available=20 * 2**30,
        free=150_000_000_000,
        rss=4 * 2**30 - 1,
        threads=4,
        utc=mon.COMPUTE_STOP - 1,
    )
    assert mon.violations(good) == []  # host load alone never stops the attempt
    for key, val in dict(
        available=20 * 2**30 - 1,
        free=149_999_999_999,
        rss=4 * 2**30,
        threads=5,
        utc=mon.COMPUTE_STOP,
    ).items():
        assert mon.violations(dict(good, **{key: val}))
    assert mon.COMPUTE_STOP == 1793680200.0  # 2026-11-03T04:30:00Z
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
