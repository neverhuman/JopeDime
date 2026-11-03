"""Data-free optimizer, selection and boundary tests; toy tensors are not study data."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import analysis as stat
import monitor as mon
import runner
from treatment import (
    ARMS,
    ScheduleFreeAdamW,
    adapter_drift,
    arm_loss,
    gradient_norm,
    make_optimizer,
    optimizer_mode,
    settings,
    validate_launch,
)

HERE = Path(__file__).resolve().parent


def rows():
    return [dict(context_sha256=f"c{i:03}", family_id=f"f{i:03}") for i in range(205)]


@pytest.mark.parametrize("arm", ["SF", "SF_LR3"])
def test_sf_matches_independent_explicit_xyz_recurrence_through_warmup(arm):
    # Three explicit reference iterates, independently expressed equations.
    x = torch.tensor([1.25, -3.0, 0.0], dtype=torch.float64)
    z, v = x.clone(), torch.zeros_like(x)
    p = torch.nn.Parameter(x.clone())
    opt = make_optimizer(arm, [p])
    total = 0.0
    base = 1e-4 * (3 if arm == "SF_LR3" else 1)
    for step in range(1, 41):
        opt.train()
        y = 0.9 * x + 0.1 * z
        assert torch.allclose(p, y, rtol=0, atol=2e-14)
        g = torch.tensor([np.sin(step), np.cos(step), step / 40], dtype=p.dtype)
        v = 0.999 * v + 0.001 * g.square()
        direction = g / ((v / (1 - 0.999**step)).sqrt() + 1e-8) + 1e-4 * y
        lr = base * min(1.0, step / 4)
        weight = lr**2
        total += weight
        z = z - lr * direction
        x = (1 - weight / total) * x + (weight / total) * z
        p.grad = g.clone()
        opt.step()
        opt.eval()
        assert torch.allclose(p, x, rtol=0, atol=2e-14)
        assert opt.param_groups[0]["scheduled_lr"] == settings(arm, step)[0]
        assert set(opt.state[p]) == {"z", "exp_avg_sq"}


def test_update_zero_modes_are_bit_exact_and_repeated_modes_are_noops():
    p = torch.nn.Parameter(torch.tensor([0.0, -0.0, 1.1], dtype=torch.float64))
    original = p.detach().view(torch.int64).clone()
    opt = make_optimizer("SF", [p])
    for training in [False, False, True, True, False]:
        optimizer_mode(opt, training)
        assert torch.equal(p.detach().view(torch.int64), original)
    opt.train()
    p.grad = torch.ones_like(p)
    opt.step()
    opt.eval()
    averaged = p.detach().clone()
    opt.eval()
    assert torch.equal(p, averaged)
    opt.train()
    training = p.detach().clone()
    opt.train()
    assert torch.equal(p, training)
    with pytest.raises(RuntimeError):
        opt.eval().step()


def test_none_gradient_receives_no_decay_or_optimizer_state():
    active = torch.nn.Parameter(torch.tensor([2.0]))
    inactive = torch.nn.Parameter(torch.tensor([3.0]))
    opt = make_optimizer("SF", [active, inactive])
    for _ in range(40):
        opt.train()
        active.grad = torch.ones_like(active)
        inactive.grad = None
        opt.step()
        opt.eval()
    assert torch.equal(inactive, torch.tensor([3.0]))
    assert not opt.state[inactive]
    assert gradient_norm([inactive]) == 0
    assert adapter_drift({"a": torch.zeros(2)}, {"a": torch.tensor([3.0, 4.0])}) == 5


def test_control_matches_exact_torch_adamw_and_all_arms_use_same_loss():
    p = torch.nn.Parameter(torch.tensor([2.0, -1.0], dtype=torch.float64))
    q = torch.nn.Parameter(p.detach().clone())
    opt = make_optimizer("CONTROL", [p])
    ref = torch.optim.AdamW([q], lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)
    for step in range(1, 41):
        p.grad = torch.tensor([step, -step / 2], dtype=p.dtype)
        q.grad = p.grad.clone()
        optimizer_mode(opt, True)
        opt.step()
        ref.step()
        optimizer_mode(opt, False)
        assert torch.equal(p, q)
    sentinel = object()
    for arm in ARMS:
        assert arm_loss(arm, lambda *args: sentinel, p, p, p) is sentinel
    table = json.loads((HERE / "schedules.json").read_text())["arms"]
    for arm in ARMS:
        for step, row in enumerate(table[arm], 1):
            assert tuple(
                row[k] for k in ["lr", "beta1", "beta2", "weight_decay"]
            ) == settings(arm, step)
    with pytest.raises(ValueError):
        settings("SF", 0)
    assert isinstance(make_optimizer("SF", [p]), ScheduleFreeAdamW)


def test_runner_swaps_before_snapshot_and_preserves_native_smoke_full_order():
    source = (HERE / "runner.py").read_text()
    assert source.index(
        "baseline, costs = engine.evaluate(val, native=True"
    ) < source.index("for arm in stat.ARMS:")
    assert source.index('out / "smoke.json"') < source.index("engine.load(train)")
    segment = source[source.index("if step in stat.STEPS:") :]
    assert segment.index("optimizer_mode(") < segment.index("snapshot = {")
    assert "head = None" not in source and "make_head" not in source


def test_immutable_runtime_loader_releases_namespace_without_model_construction():
    import sys

    pins = json.loads((HERE / "manifest.json").read_text())
    rt = runner.runtime(pins)
    assert rt.Audit.__name__ == "Audit"
    assert rt.DOPE.startswith("c51a56d0")
    assert not [n for n in sys.modules if n == "research" or n.startswith("research.")]
    assert rt.HERE == Path("/home/ubuntu/JopeDime")


def test_external_launch_is_head_bound_even_in_smoke_mode():
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
        review_id=True,
        prereg_pr=0,
        merged_source_verified=False,
    ).items():
        with pytest.raises(RuntimeError):
            validate_launch(dict(launch, **{key: bad}), "h", "m", "p", "f", smoke=True)


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
    first = stat.batch_plan(ids, orders, 236101)
    other = stat.batch_plan(list(reversed(ids)), orders, 236101)
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
    assert s["contrasts"]["SF_minus_r882"]["median_gain"] == 0.0
    assert np.allclose(s["median_elo"], 1500.0)
    panel[:] = 0
    panel[:, :, 2] = 0.005
    s = stat.summarize(source, panel)
    assert s["contrasts"]["SF_minus_r882"]["advance"]
    assert s["contrasts"]["SF_minus_r882"]["ci95"][0] > 0
    assert (
        "SF_minus_CONTROL" in s["contrasts"] and "CONTROL_minus_r882" in s["contrasts"]
    )
    # hw_t2120 margin: a +0.002 gain is a tie with the margin, not an advance
    panel[:, :, 2] = 0.002
    assert not stat.summarize(source, panel)["contrasts"]["SF_minus_r882"]["advance"]
    panel[:, :, 2] = 0.0021
    assert stat.summarize(source, panel)["contrasts"]["SF_minus_r882"]["advance"]
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
    cells[0]["selector_panels"]["SF"][1]["values"] = [1.0, 0.0]
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
    assert mon.COMPUTE_STOP == 1793685600.0  # 2026-11-03T06:00:00Z
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
