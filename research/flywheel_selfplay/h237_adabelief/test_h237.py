"""Data-free H237 controls; toy gradients are never study-training datasets."""

from types import SimpleNamespace
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import analysis as stat
import monitor as mon
from treatment import (
    ABEL_EPS,
    AdaBelief,
    arm_loss,
    make_optimizer,
    settings,
    gradient_norm,
    adapter_drift,
    validate_launch,
)

HERE = Path(__file__).resolve().parent


def rows():
    return [dict(context_sha256=f"c{i:03}", family_id=f"f{i:03}") for i in range(205)]


def test_arm_settings_and_optimizer_classes():
    assert stat.ARMS == ("CONTROL", "ABEL", "RADAM")
    for arm in stat.ARMS:
        assert all(settings(arm, t) == (1e-4, 0.9, 0.999, 1e-4) for t in range(1, 41))
    for step in [0, 41]:
        with pytest.raises(ValueError):
            settings("ABEL", step)
    p = [torch.nn.Parameter(torch.ones(1))]
    assert type(make_optimizer("CONTROL", p)) is torch.optim.AdamW
    assert type(make_optimizer("ABEL", p)) is AdaBelief
    radam = make_optimizer("RADAM", p)
    assert (
        type(radam) is torch.optim.RAdam
        and radam.param_groups[0]["decoupled_weight_decay"] is True
    )
    for opt in (make_optimizer(a, p) for a in stat.ARMS):
        g = opt.param_groups[0]
        assert (
            g["lr"] == 1e-4 and g["betas"] == (0.9, 0.999) and g["weight_decay"] == 1e-4
        )
    assert make_optimizer("ABEL", p).param_groups[0]["eps"] == ABEL_EPS == 1e-16
    table = json.loads((HERE / "schedules.json").read_text())["arms"]
    assert table == {
        a: [
            dict(step=t, lr=1e-4, beta1=0.9, beta2=0.999, weight_decay=1e-4)
            for t in range(1, 41)
        ]
        for a in stat.ARMS
    }


def test_adabelief_matches_the_reference_update_and_skips_none_gradients():
    lr, b1, b2, eps, wd = 0.01, 0.9, 0.999, 1e-16, 0.3
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64))
    inactive = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    opt = AdaBelief([p, inactive], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)
    ref = p.detach().clone()
    m = torch.zeros(3, dtype=torch.float64)
    s = torch.zeros(3, dtype=torch.float64)
    rng = np.random.default_rng(0)
    for t in range(1, 31):
        g = torch.tensor(rng.normal(size=3))
        p.grad = g.clone()
        opt.step()
        ref = ref * (1 - lr * wd)
        m = b1 * m + (1 - b1) * g
        s = b2 * s + (1 - b2) * (g - m) ** 2 + eps
        ref = ref - lr * (m / (1 - b1**t)) / (torch.sqrt(s / (1 - b2**t)) + eps)
        assert torch.allclose(p.detach(), ref, rtol=0, atol=1e-12)
    assert inactive.grad is None and inactive not in opt.state
    assert torch.equal(inactive.detach(), torch.ones(2, dtype=torch.float64))
    with pytest.raises(ValueError):
        AdaBelief([p], lr=0.0)


def test_belief_takes_larger_steps_than_adam_on_consistent_gradients():
    """The defining property: with a constant gradient the belief variance (g - m)^2 shrinks, so AdaBelief steps
    are larger than Adam's at the same lr."""
    pa = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    pb = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    adam = torch.optim.Adam([pa], lr=1e-3, betas=(0.9, 0.999))
    abel = AdaBelief([pb], lr=1e-3, betas=(0.9, 0.999), eps=1e-16, weight_decay=0.0)
    for _ in range(20):
        pa.grad = torch.ones(1, dtype=torch.float64)
        pb.grad = torch.ones(1, dtype=torch.float64)
        adam.step()
        abel.step()
    assert abs(float(pb)) > abs(float(pa))


def test_every_arm_changes_parameters_with_its_own_optimizer_and_ctrl_loss():
    final = []
    for arm in stat.ARMS:
        p = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
        opt = make_optimizer(arm, [p])
        rng = np.random.default_rng(1)
        for _ in range(40):
            p.grad = torch.tensor(rng.normal(size=2))
            torch.nn.utils.clip_grad_norm_([p], 1.0)
            opt.step()
        final.append(p.detach().numpy().tobytes())
    assert len(set(final)) == 3
    sentinel = object()
    assert arm_loss("ABEL", lambda *a: sentinel, None, None, None) is sentinel
    with pytest.raises(ValueError):
        arm_loss("LION", lambda *a: sentinel, None, None, None)


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
    first = stat.batch_plan(ids, orders, 237101)
    other = stat.batch_plan(list(reversed(ids)), orders, 237101)
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
    assert s["contrasts"]["ABEL_minus_r882"]["median_gain"] == 0.0
    assert np.allclose(s["median_elo"], 1500.0)
    panel[:] = 0
    panel[:, :, 2] = 0.005
    s = stat.summarize(source, panel)
    assert s["contrasts"]["ABEL_minus_r882"]["advance"]
    assert s["contrasts"]["ABEL_minus_r882"]["ci95"][0] > 0
    assert (
        "ABEL_minus_CONTROL" in s["contrasts"]
        and "CONTROL_minus_r882" in s["contrasts"]
    )
    # hw_t2120 margin: a +0.002 gain is a tie with the margin, not an advance
    panel[:, :, 2] = 0.002
    assert not stat.summarize(source, panel)["contrasts"]["ABEL_minus_r882"]["advance"]
    panel[:, :, 2] = 0.0021
    assert stat.summarize(source, panel)["contrasts"]["ABEL_minus_r882"]["advance"]
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
    cells[0]["selector_panels"]["ABEL"][1]["values"] = [1.0, 0.0]
    assert not stat.pseudoreplication(panel, panel, panel, panel, cells)[
        "all_arms_selector_identical_to_CONTROL"
    ]


def test_hard_resource_boundaries_and_load_and_memory_are_pauses():
    good = dict(
        load=500.0,
        load5=500.0,
        cpu=100.0,
        available=12 * 2**30,
        free=150_000_000_000,
        rss=4 * 2**30 - 1,
        threads=4,
        utc=mon.COMPUTE_STOP - 1,
    )
    assert (
        mon.violations(good) == []
    )  # load and memory above the 12 GiB floor never stop the attempt
    for key, val in dict(
        available=12 * 2**30 - 1,
        free=149_999_999_999,
        rss=4 * 2**30,
        threads=5,
        utc=mon.COMPUTE_STOP,
    ).items():
        assert mon.violations(dict(good, **{key: val}))
    assert mon.COMPUTE_STOP == 1793687400.0  # 2026-11-03T06:30:00Z
    ok = dict(load5=10.0, available=30 * 2**30)
    assert not mon.paused(ok, False)
    assert mon.paused(dict(ok, load5=60.1), False) and not mon.paused(
        dict(ok, load5=60.0), False
    )
    assert mon.paused(dict(ok, available=20 * 2**30 - 1), False)
    assert not mon.paused(dict(ok, available=20 * 2**30), False)
    # once paused: resume only when load5 <= 55 AND MemAvailable >= 22 GiB
    assert mon.paused(dict(ok, available=21 * 2**30), True)
    assert mon.paused(dict(ok, load5=58.0), True)
    assert not mon.paused(dict(ok, load5=55.0, available=22 * 2**30), True)
    adm = dict(good, load5=55.0, available=22 * 2**30)
    assert mon.admission_refusals(adm) == []
    assert mon.admission_refusals(dict(adm, load5=55.1)) == ["admission: load5>55"]
    assert mon.admission_refusals(dict(adm, available=22 * 2**30 - 1)) == [
        "admission: MemAvailable<22GiB"
    ]


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
