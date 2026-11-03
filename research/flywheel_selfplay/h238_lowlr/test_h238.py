"""Data-free H238 optimizer doses, horizon labels, paired sampler and boundary controls."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch
import analysis as stat
import monitor as mon
import runner
from treatment import (
    ARMS,
    LEARNING_RATES,
    UPDATES,
    arm_loss,
    make_optimizer,
    settings,
    validate_launch,
)

HERE = Path(__file__).resolve().parent


def rows():
    return [dict(context_sha256=f"c{i:03}", family_id=f"f{i:03}") for i in range(205)]


@pytest.mark.parametrize("arm", ARMS)
def test_exact_adamw_recipe_with_only_lr_and_horizon_changed(arm):
    p = torch.nn.Parameter(torch.tensor([2.0, -1.0], dtype=torch.float64))
    q = torch.nn.Parameter(p.detach().clone())
    inactive = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.float64))
    opt = make_optimizer(arm, [p, inactive])
    ref = torch.optim.AdamW(
        [q], lr=LEARNING_RATES[arm], betas=(0.9, 0.999), weight_decay=1e-4
    )
    assert type(opt) is torch.optim.AdamW and opt.param_groups[0]["eps"] == 1e-8
    for step in range(1, UPDATES[arm] + 1):
        p.grad = torch.tensor([step, -step / 2], dtype=p.dtype)
        q.grad = p.grad.clone()
        inactive.grad = None
        torch.nn.utils.clip_grad_norm_([p, inactive], 1.0)
        torch.nn.utils.clip_grad_norm_([q], 1.0)
        opt.step()
        ref.step()
        assert torch.equal(p, q)
        assert settings(arm, step) == (LEARNING_RATES[arm], 0.9, 0.999, 1e-4)
    assert (
        torch.equal(inactive, torch.tensor([3.0], dtype=p.dtype))
        and inactive not in opt.state
    )
    with pytest.raises(ValueError):
        settings(arm, UPDATES[arm] + 1)
    with pytest.raises(ValueError):
        settings(arm, 0)


def test_all_arms_use_identical_ctrl_loss_and_constant_schedules():
    sentinel = object()
    table = json.loads((HERE / "schedules.json").read_text())
    assert stat.ARM_UPDATES == UPDATES == dict(CONTROL=40, LR3E5=40, LR1E5_120=120)
    assert 15 * sum(UPDATES.values()) == 3000
    for arm in ARMS:
        assert arm_loss(arm, lambda *args: sentinel, None, None, None) is sentinel
        assert table["candidate_steps"][arm] == list(stat.ARM_STEPS[arm])
        assert len(table["arms"][arm]) == UPDATES[arm]
        for step, row in enumerate(table["arms"][arm], 1):
            assert row == dict(
                step=step,
                lr=LEARNING_RATES[arm],
                beta1=0.9,
                beta2=0.999,
                weight_decay=1e-4,
            )


def test_120_batch_stream_preserves_exact_paired_first40_and_extends_sampler():
    ids = [r["context_sha256"] for r in rows()[:123]]
    orders = {c: np.arange(48) for c in ids}
    short = stat.batch_plan(ids, orders, 238101)
    long = stat.batch_plan(list(reversed(ids)), orders, 238101, updates=120)
    assert len(short) == 40 and len(long) == 120
    for i, (stream, cids, indexes) in enumerate(long):
        assert stream == stat.stream_seed(238101, i + 1)
        assert len(cids) == 8 and set(cids) <= set(ids)
        for ix in indexes:
            assert len(set(ix)) == 24 and np.array_equal(ix[:4], np.arange(4))
        if i < 40:
            a, b, c = short[i]
            assert (
                a == stream
                and b == cids
                and all(np.array_equal(x, y) for x, y in zip(c, indexes))
            )


def test_arm_specific_long_selector_returns_actual_step30_not10():
    source = rows()[:41]
    panel = np.zeros((41, 5))
    panel[:, 1:] = 0.005
    assert stat.choose(source, panel, 1)["selected_step"] == 10
    assert (
        stat.choose(source, panel, 1, steps=stat.ARM_STEPS["LR1E5_120"])[
            "selected_step"
        ]
        == 30
    )
    panel[:] = 0
    assert (
        stat.choose(source, panel, 1, steps=stat.ARM_STEPS["LR1E5_120"])[
            "selected_step"
        ]
        == 0
    )
    for invalid in [(0, 30, 30, 90, 120), (10, 20, 30, 40, 50), (0, 30, 20, 90, 120)]:
        with pytest.raises(ValueError):
            stat.choose(source, panel, 1, steps=invalid)


def test_runner_uses_per_arm_budget_candidates_and_preserves_native_smoke_order():
    s = (HERE / "runner.py").read_text()
    assert (
        "batch_cache[: stat.ARM_UPDATES[arm]]" in s
        or "batch_cache[:stat.ARM_UPDATES[arm]]" in s
    )
    assert "if step in stat.ARM_STEPS[arm]" in s
    assert "steps=stat.ARM_STEPS[arm]" in s
    assert s.index("baseline, costs = engine.evaluate(val, native=True") < s.index(
        "for arm in stat.ARMS:"
    )
    assert s.index('out / "smoke.json"') < s.index("engine.load(train)")
    assert "optimizer_mode" not in s and "1.0, error_if_nonfinite=True" in s


def test_ram_pause_resume_hysteresis_and_hard_floor_are_distinct():
    base = dict(
        load5=100.0,
        available=22 * 2**30,
        rss=4 * 2**30 - 1,
        threads=4,
        free=300_000_000_000,
        utc=mon.COMPUTE_STOP - 1,
    )
    assert mon.admission_refusals(base) == []
    assert not mon.paused(dict(base, available=20 * 2**30), False)
    assert mon.paused(dict(base, available=20 * 2**30 - 1), False)
    assert mon.paused(dict(base, available=22 * 2**30 - 1), True)
    assert not mon.paused(base, True)
    assert mon.violations(dict(base, available=12 * 2**30)) == []
    assert mon.violations(dict(base, available=12 * 2**30 - 1)) == [
        "MemAvailable<12GiB"
    ]
    assert mon.paused(dict(base, load5=108.1), False) and not mon.paused(
        dict(base, load5=108), False
    )
    assert mon.paused(dict(base, load5=100.1), True)
    assert mon.admission_refusals(dict(base, available=22 * 2**30 - 1))
    for key, value in dict(
        rss=4 * 2**30, threads=5, free=299_999_999_999, utc=mon.COMPUTE_STOP
    ).items():
        assert mon.violations(dict(base, **{key: value}))
    assert mon.COMPUTE_STOP == 1793689200.0


def test_immutable_runtime_releases_namespace_and_launch_is_exact_head_bound():
    import sys

    rt = runner.runtime(json.loads((HERE / "manifest.json").read_text()))
    assert rt.Audit.__name__ == "Audit" and rt.DOPE.startswith("c51a56d0")
    assert not [n for n in sys.modules if n == "research" or n.startswith("research.")]
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
    for key, value in dict(
        source_commit="x",
        manifest_sha256="x",
        prereg_sha256="x",
        fold_manifest_sha256="x",
        review_id=True,
        prereg_pr=0,
        merged_source_verified=False,
    ).items():
        with pytest.raises(RuntimeError):
            validate_launch(
                dict(launch, **{key: value}), "h", "m", "p", "f", smoke=True
            )


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
    first = stat.batch_plan(ids, orders, 238101)
    other = stat.batch_plan(list(reversed(ids)), orders, 238101)
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
    assert s["contrasts"]["LR3E5_minus_r882"]["median_gain"] == 0.0
    assert np.allclose(s["median_elo"], 1500.0)
    panel[:] = 0
    panel[:, :, 2] = 0.005
    s = stat.summarize(source, panel)
    assert s["contrasts"]["LR3E5_minus_r882"]["advance"]
    assert s["contrasts"]["LR3E5_minus_r882"]["ci95"][0] > 0
    assert (
        "LR3E5_minus_CONTROL" in s["contrasts"]
        and "CONTROL_minus_r882" in s["contrasts"]
    )
    # hw_t2120 margin: a +0.002 gain is a tie with the margin, not an advance
    panel[:, :, 2] = 0.002
    assert not stat.summarize(source, panel)["contrasts"]["LR3E5_minus_r882"]["advance"]
    panel[:, :, 2] = 0.0021
    assert stat.summarize(source, panel)["contrasts"]["LR3E5_minus_r882"]["advance"]
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
                        a: {
                            "0": "r882",
                            str(stat.ARM_UPDATES[a]): f"{outer}:{seed}:{a}",
                        }
                        for a in stat.ARMS
                    },
                    decisions={a: {"selected_step": 0} for a in stat.ARMS},
                )
            )
    report = stat.pseudoreplication(panel, panel, panel, panel, cells)
    assert len(report["strict_OOF_identical"][0]) == 12  # 3 seeds x (r882 + 3 arms)
    assert (
        report["terminal_adapter_identical"] == []
        and report["all_arms_candidate_index_panels_identical_to_CONTROL"]
    )
    cells[0]["selector_panels"]["LR3E5"][1]["values"] = [1.0, 0.0]
    assert not stat.pseudoreplication(panel, panel, panel, panel, cells)[
        "all_arms_candidate_index_panels_identical_to_CONTROL"
    ]
