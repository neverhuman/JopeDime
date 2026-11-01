"""H180 design/statistics controls; all fixtures are synthetic and data-free (no checkpoint, corpus or model)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

import analysis as stat
import treatment

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("h180_test_monitor", HERE / "monitor.py")
monitor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(monitor)


def rows(n_families=205):
    return [
        {"context_sha256": f"c{i:04}", "family_id": f"f{i // 3:03}"}
        for i in range(n_families * 3)
    ]


def test_registered_constants():
    assert (
        stat.SEEDS == (180101, 180102, 180103)
        and stat.FOLD_SEED == 180140
        and stat.BOOT_SEED == 180141
    )
    assert stat.ARMS == (
        "R882",
        "ZERO_ADAPTER",
        "RAND0",
        "FRESH40",
        "FRESH160",
    ) and stat.DOSE == (40, 160)
    assert stat.CANDIDATES == ("ZERO_ADAPTER", "RAND0", "FRESH40", "FRESH160")


def test_five_fold_family_crossfit_is_disjoint_and_assesses_each_family_once():
    families = {r["family_id"] for r in rows()}
    groups = stat.family_folds(families)
    assert [len(g) for g in groups] == [41] * 5 and groups == stat.family_folds(
        families, seed=180140
    )
    assessed = []
    for outer in range(5):
        fit, select, assess = stat.partition(families, outer, groups)
        assert (len(fit), len(select), len(assess)) == (123, 41, 41)
        assert not fit & select and not fit & assess and not select & assess
        assessed += list(assess)
    assert len(assessed) == len(set(assessed)) == 205


def test_no_a_leak_batch_plan_draws_only_fit_contexts():
    source = rows()
    fit, select, assess = stat.partition({r["family_id"] for r in source}, 2)
    fit_ids = [r["context_sha256"] for r in source if r["family_id"] in fit]
    held = {r["context_sha256"] for r in source if r["family_id"] in select | assess}
    orders = {c: np.arange(48) for c in fit_ids}
    plan = stat.batch_plan(
        fit_ids,
        orders,
        stat.stream_seed(180101, 2, "fit-123", "h156-batches-v1"),
        updates=160,
    )
    drawn = set().union(*(set(c) for _, c, _ in plan))
    assert len(plan) == 160 and drawn <= set(fit_ids) and not drawn & held


def test_fresh40_is_a_prefix_of_the_fresh160_trajectory():
    cids = [f"c{i:03}" for i in range(80)]
    orders = {c: np.arange(48) for c in cids}
    seed = stat.stream_seed(180101, 0, "fit-123", "h156-batches-v1")
    p40, p160 = (
        stat.batch_plan(cids, orders, seed, updates=40),
        stat.batch_plan(cids, orders, seed, updates=160),
    )
    for (s1, c1, i1), (s2, c2, i2) in zip(p40, p160[:40], strict=True):
        assert (
            s1 == s2 and c1 == c2 and all(np.array_equal(a, b) for a, b in zip(i1, i2))
        )

    def trained(
        split,
    ):  # one AdamW trajectory; continuing the same optimizer equals the unsplit run
        torch.manual_seed(0)
        w = torch.nn.Parameter(torch.randn(5))
        opt = torch.optim.AdamW([w], lr=1e-2, weight_decay=1e-4)
        snaps = {}
        for step in range(1, 161):
            opt.zero_grad()
            ((w - step / 160.0) ** 2).sum().backward()
            opt.step()
            if step in split:
                snaps[step] = w.detach().clone()
        return snaps

    assert torch.equal(trained({40, 160})[160], trained({160})[160])


def test_zero_adapter_is_the_anchor_only_scorer_of_the_live_decomposition():
    anchor = torch.tensor([[0.3, -1.2, 2.0]])
    delta = torch.tensor([[0.5, 0.25, -0.75]])
    live = {"total_cost": anchor + delta, "adapter_total_delta": delta}
    assert treatment.assert_anchor_decomposition(live, anchor)
    zero = treatment.zero_adapter_scores(anchor)
    assert torch.equal(
        zero, live["total_cost"] - live["adapter_total_delta"]
    ) and torch.equal(zero, anchor)
    assert int(torch.argmin(zero)) == 1 and int(torch.argmin(live["total_cost"])) == 1
    with pytest.raises(RuntimeError):
        treatment.assert_anchor_decomposition(
            {"total_cost": anchor + delta + 1e-6, "adapter_total_delta": delta}, anchor
        )
    with pytest.raises(RuntimeError):
        treatment.assert_anchor_decomposition({"total_cost": anchor}, anchor)


def panels(n=205, gains=None, noise=0.0, seed=0, seed_offsets=None):
    families = [f"f{i:03}" for i in range(n)]
    rng = np.random.default_rng(seed)
    p = np.zeros((3, n, 5))
    for j, arm in enumerate(stat.ARMS):
        if arm == "R882":
            continue
        g = (gains or {}).get(arm, 0.0)
        e = rng.normal(0.0, noise, size=(3, n)) if noise else np.zeros((3, n))
        e -= e.mean(axis=1, keepdims=True)
        off = np.asarray((seed_offsets or {}).get(arm, [0.0, 0.0, 0.0]))[:, None]
        p[:, :, j] = g + e + off
    return families, p


def run_summary(families, p):
    minus2 = {a: [0, 0, 0] for a in stat.ARMS}
    agree = {a: [1.0, 1.0, 1.0] for a in stat.ARMS}
    return stat.summarize(families, p, p, families, minus2, agree)


def test_identical_arms_give_elo_1500_and_adapter_inert():
    families, p = panels(41)
    s = run_summary(families, p)
    assert all(
        v["median_over_seeds"] == pytest.approx(1500.0)
        for v in s["elo_vs_r882"].values()
    )
    assert (
        s["adapter_verdict"] == "ADAPTER_INERT"
        and s["dose_verdict"] == "BUDGET_FLAT"
        and not s["any_advance_candidate"]
    )
    assert s["arms"]["ZERO_ADAPTER"]["identical_across_seeds"]


def test_paired_bootstrap_reduces_over_seeds_axis0_not_draws():
    # Per-seed ZERO-R882 means are all slightly negative, but family variance is large: the axis=0 reduction keeps
    # the family uncertainty (CI spans 0, |median| < 0.005 -> INERT), whereas reducing over draws would collapse the
    # interval to three negative seed medians.
    families, p = panels(
        205,
        noise=0.035,
        seed=180999,
        seed_offsets={"ZERO_ADAPTER": [-0.0018, -0.0021, -0.0038]},
    )
    s = run_summary(families, p)
    lo, hi = s["contrasts"]["ZERO_ADAPTER_minus_R882"]["family_bootstrap_ci95"]
    assert lo < 0.0 < hi and hi - lo > 0.005
    assert s["adapter_verdict"] == "ADAPTER_INERT"


def test_adapter_and_dose_verdict_boundaries_and_advance_gates():
    families, p = panels(
        41, gains={"ZERO_ADAPTER": -0.05, "FRESH40": 0.0, "FRESH160": 0.02}
    )
    s = run_summary(families, p)
    assert (
        s["adapter_verdict"] == "ADAPTER_MATTERS"
        and s["dose_verdict"] == "BUDGET_HELPS"
    )
    assert s["advance_candidates"]["FRESH160"]["advance_candidate"]
    assert not s["advance_candidates"]["ZERO_ADAPTER"]["advance_candidate"]
    families, p = panels(41, gains={"ZERO_ADAPTER": 0.02, "FRESH160": -0.02})
    s = run_summary(families, p)
    assert (
        s["adapter_verdict"] == "ADAPTER_HARMFUL"
        and s["dose_verdict"] == "BUDGET_HURTS"
    )
    families, p = panels(
        41, gains={"RAND0": 0.001}
    )  # exactly the margin is a tie: rejected
    assert not run_summary(families, p)["advance_candidates"]["RAND0"]["gates"][
        "margin"
    ]
    families, p = panels(
        205, noise=0.15, seed=3, seed_offsets={"ZERO_ADAPTER": [0.006, 0.007, 0.008]}
    )
    assert (
        run_summary(families, p)["adapter_verdict"] == "ADAPTER_INCONCLUSIVE"
    )  # CI spans 0 but |gain| >= 0.005


def test_execution_plan_is_timing_only():
    assert stat.execution_plan(1.0)["outer_folds"] == 5
    assert (
        stat.execution_plan(4.0)["outer_folds"] == 3
    )  # 5 folds would exceed 3 h; 3 folds fit 4 h
    assert stat.execution_plan(100.0)["outer_folds"] == 0
    with pytest.raises(ValueError):
        stat.execution_plan(0.0)


def test_choice_agreement_and_launch_review_or_waiver():
    assert stat.choice_agreement({"a": 1, "b": 2}, {"a": 1, "b": 3}) == 0.5
    with pytest.raises(ValueError):
        stat.choice_agreement({"a": 1}, {"a": 1, "b": 2})
    base = dict(
        manifest_sha256="m",
        merged_source_verified=True,
        source_commit="h",
        prereg_pr=1,
        prereg_sha256="p",
        fold_manifest_sha256="f",
    )
    treatment.validate_launch(dict(base, review_id=5), "h", "m", "p", "f")
    treatment.validate_launch(
        dict(base, review_waiver="rd_argon gave no review within the 15 min window"),
        "h",
        "m",
        "p",
        "f",
    )
    with pytest.raises(RuntimeError):
        treatment.validate_launch(dict(base, review_waiver="none"), "h", "m", "p", "f")
    with pytest.raises(RuntimeError):
        treatment.validate_launch(base, "h", "m", "p", "f")


def test_monitor_disk_and_host_guards():
    ok = dict(
        load=20.0,
        cpu=30.0,
        available=70 * 2**30,
        free=500e9,
        disk_used_fraction=0.93,
        rss=1e9,
        threads=3,
    )
    assert monitor.violations(ok) == []
    assert monitor.violations(dict(ok, disk_used_fraction=0.938)) == [
        "disk user-visible used>=93.8%"
    ]
    assert monitor.violations(dict(ok, available=15 * 2**30)) == ["MemAvailable<16GiB"]
    assert monitor.violations(dict(ok, cpu=86.0)) == ["host CPU>85%"]
