"""Data-free H170 controls; toy gradients are never study-training datasets."""

from types import SimpleNamespace
import math
import numpy as np
import pytest
import torch
import analysis as stat
import monitor as mon
from treatment import (
    settings,
    cycle,
    MEANMATCH_LR,
    apply_settings,
    gradient_norm,
    adapter_drift,
    validate_launch,
)


def rows():
    return [dict(context_sha256=f"c{i:03}", family_id=f"f{i:03}") for i in range(205)]


def test_cycle_exact_endpoints_inverse_momentum_and_discrete_dose():
    for arm, peak in [("OC3", 3e-4), ("OC15", 1.5e-4)]:
        values = np.array([settings(arm, t) for t in range(1, 41)])
        assert values[0, 0] == 1e-5 and values[11, 0] == peak and values[-1, 0] == 1e-6
        assert (
            values[0, 1] == 0.95
            and values[11, 1] == pytest.approx(0.85)
            and values[-1, 1] == 0.95
        )
        assert np.all(np.diff(values[:12, 0]) > 0) and np.all(
            np.diff(values[11:, 0]) < 0
        )
        assert np.all(np.diff(values[:12, 1]) < 0) and np.all(
            np.diff(values[11:, 1]) > 0
        )
    assert MEANMATCH_LR == pytest.approx(0.0001481125, rel=0, abs=1e-19)
    assert math.fsum(settings("OC3", t)[0] for t in range(1, 41)) == pytest.approx(
        40 * MEANMATCH_LR, abs=1e-18
    )
    assert all(settings("CTRL", t) == (1e-4, 0.9) for t in range(1, 41))
    assert all(
        settings("CTRL_MEANMATCH", t) == (MEANMATCH_LR, 0.9) for t in range(1, 41)
    )
    for step in [0, 41]:
        with pytest.raises(ValueError):
            settings("OC3", step)
    with pytest.raises(ValueError):
        cycle(1, 2e-4)


def test_actual_adamw_receives_pre_step_settings_and_changes_parameters():
    final = []
    for arm in stat.ARMS:
        p = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
        inactive = torch.nn.Parameter(torch.ones(3, dtype=torch.float64))
        optimizer = torch.optim.AdamW(
            [p, inactive], lr=7.0, betas=(0.1, 0.2), weight_decay=0.0
        )
        for t in range(1, 41):
            lr, beta = apply_settings(optimizer, arm, t)
            p.grad = torch.tensor([3.0, 4.0], dtype=torch.float64)
            torch.nn.utils.clip_grad_norm_([p, inactive], 1.0)
            assert gradient_norm([p, inactive]) == pytest.approx(1.0, abs=3e-7)
            before = p.detach().clone()
            optimizer.step()
            assert optimizer.param_groups[0]["lr"] == settings(arm, t)[0]
            assert optimizer.param_groups[0]["betas"] == (beta, 0.999)
            if t == 1:
                assert torch.allclose(
                    before - p, torch.full_like(p, lr), atol=1e-10, rtol=0
                )
            assert inactive.grad is None and inactive not in optimizer.state
        assert torch.equal(inactive, torch.ones_like(inactive))
        final.append(p.detach().numpy().tobytes())
    assert len(set(final)) == 4


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
        fit, select, assess = stat.partition(source, outer)
        groups = [{r["family_id"] for r in rs} for rs in [fit, select, assess]]
        assert [len(g) for g in groups] == [123, 41, 41]
        assert not (
            groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]
        )
        cover += list(groups[2])
    assert len(cover) == len(set(cover)) == 205
    fit, _, _ = stat.partition(source, 0)
    ids = [r["context_sha256"] for r in fit]
    orders = {c: np.arange(48) for c in ids}
    first = stat.batch_plan(ids, orders, 170101)
    other = stat.batch_plan(list(reversed(ids)), orders, 170101)
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
    panel = np.zeros((3, 10, 5))
    panel[:, :9, 1:] = 1.0
    panel[:, 9, 1:] = -1.0
    s = stat.summarize(source, panel)
    assert s["contrasts"]["OC3_minus_r882"]["median_gain"] == 0.0
    assert np.allclose(s["median_elo"], 1500.0)
    panel[:] = 0
    panel[:, :, 2] = 0.005
    s = stat.summarize(source, panel)
    assert s["contrasts"]["OC3_minus_r882"]["advance"]
    assert s["contrasts"]["OC3_minus_r882"]["ci95"][0] > 0
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
    panel = np.zeros((3, 2, 5))
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
    assert len(report["strict_OOF_identical"][0]) == 15
    assert (
        report["step40_adapter_identical"] == []
        and report["all_arms_selector_identical_to_CTRL"]
    )
    cells[0]["selector_panels"]["OC3"][1]["values"] = [1.0, 0.0]
    assert not stat.pseudoreplication(panel, panel, panel, panel, cells)[
        "all_arms_selector_identical_to_CTRL"
    ]


def test_every_terminal_resource_boundary():
    good = dict(
        load=mon.LOAD_LIMIT,
        cpu=85.0,
        available=16 * 2**30,
        free=40_000_000_000,
        disk_used_fraction=0.94,
        rss=5_999_999_999,
        threads=4,
    )
    assert mon.violations(good) == []
    for key, val in dict(
        load=mon.LOAD_LIMIT + 0.001,
        cpu=85.01,
        available=16 * 2**30 - 1,
        free=39_999_999_999,
        disk_used_fraction=0.95,
        rss=6_000_000_000,
        threads=5,
    ).items():
        assert mon.violations(dict(good, **{key: val}))


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
