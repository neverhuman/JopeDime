"""Data-free controls for H160 (synthetic vectors, rows and panels only; no checkpoint, corpus or model)."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis as stat  # noqa: E402

MSPEC = importlib.util.spec_from_file_location("h160_test_monitor", HERE / "monitor.py")
mon = importlib.util.module_from_spec(MSPEC)
MSPEC.loader.exec_module(mon)


def rows(n_fam=205, per=6):
    return [
        dict(context_sha256=f"c{f}-{i}", family_id=f"f{f:03d}")
        for f in range(n_fam)
        for i in range(per)
    ]


def chain(n=8, dim=50, seed=0):
    rng = np.random.default_rng(seed)
    steps = rng.standard_normal((n - 1, dim)) * np.linspace(1, 0.1, n - 1)[:, None]
    return np.vstack([np.zeros(dim), np.cumsum(steps, axis=0)])


def test_registered_constants():
    assert (
        stat.SEEDS == (160101, 160102, 160103)
        and stat.FOLD_SEED == 160140
        and stat.BOOT_SEED == 160141
    )
    assert stat.ARMS == (
        "CTRL",
        "PROJ_K2",
        "PROJ_K4",
        "CTRL_DRIFT",
    ) and stat.PRIMARY == ("PROJ_K2", "PROJ_K4")
    assert (
        stat.K == {"PROJ_K2": 2, "PROJ_K4": 4}
        and stat.CHAIN[-1] == 882
        and len(stat.CHAIN) == 8
    )
    assert (
        stat.BASE_LR == 1e-4
        and stat.WEIGHT_DECAY == 1e-4
        and stat.STEPS == (0, 10, 20, 30, 40)
    )


def test_basis_is_orthonormal_top_directions_of_deltas():
    v = chain()
    u, info = stat.basis(v, 4)
    assert u.shape == (50, 4) and np.allclose(u.T @ u, np.eye(4), atol=1e-12)
    assert (
        info["rank"] == 7
        and 0 < info["explained_energy"] <= 1
        and len(info["delta_l2"]) == 7
    )
    deltas = np.diff(v, axis=0)
    _, s, vt = np.linalg.svd(deltas, full_matrices=False)
    assert np.allclose(
        np.abs(u.T @ vt[:4].T), np.eye(4), atol=1e-10
    )  # same span, sign-free


def test_basis_refuses_k_above_delta_rank():
    v = chain(
        n=4
    )  # 3 deltas: rank <= 3, so k = 4 is impossible (why H160 needs the 8-point chain)
    with pytest.raises(ValueError, match="exceeds delta rank"):
        stat.basis(v, 4)
    dup = np.vstack(
        [chain(n=5), chain(n=5)[-1]]
    )  # a duplicated endpoint adds a zero delta, not a direction
    assert stat.basis(dup, 4)[1]["rank"] == 4


def test_projection_idempotent_and_adam_like_step_leaks_until_displacement_projected():
    u, _ = stat.basis(chain(), 2)
    rng = np.random.default_rng(1)
    g = rng.standard_normal(50)
    gp = stat.project(u, g)
    assert np.allclose(stat.project(u, gp), gp) and stat.out_of_span(u, gp) < 1e-12
    adam_step = -1e-4 * np.sign(
        gp
    )  # elementwise preconditioning of an in-span gradient leaves the span
    assert stat.out_of_span(u, adam_step) > 0.1
    displaced = stat.project(u, adam_step)
    assert (
        stat.out_of_span(u, displaced) < 1e-12
        and stat.out_of_span(u, np.zeros(50)) == 0.0
    )


def test_drift_lr_and_inclusive_match_band():
    assert stat.drift_lr([0.01, 0.02, 0.03], [0.4, 0.42, 0.44]) == pytest.approx(
        1e-4 * 0.02 / 0.42
    )
    with pytest.raises(ValueError):
        stat.drift_lr([0.0], [0.4])
    assert (
        stat.drift_match([0.02], [0.017])["valid"]
        and stat.drift_match([0.02], [0.023])["valid"]
    )
    assert (
        not stat.drift_match([0.02], [0.0169])["valid"]
        and not stat.drift_match([0.02], [0.0231])["valid"]
    )


def test_partition_disjoint_complete_and_deterministic():
    r = rows()
    g = stat.folds(r)
    assert [len(x) for x in g] == [41] * 5 and g == stat.folds(r)
    for outer in range(5):
        fit, sel, ass = stat.partition(r, outer)
        ff, fs, fa = ({x["family_id"] for x in p} for p in (fit, sel, ass))
        assert (
            not ff & fs
            and not ff & fa
            and not fs & fa
            and len(ff) == 123
            and len(fs) == len(fa) == 41
        )


def test_batch_plan_forty_fit_only_steps_and_registered_seed():
    r = rows()
    fit, _, _ = stat.partition(r, 0)
    ids = {x["context_sha256"] for x in fit}
    orders = {c: np.arange(300) for c in ids}
    plan = stat.batch_plan(ids, orders, 160101)
    assert len(plan) == 40 and all(set(c) <= ids and len(c) == 16 for _, c, _ in plan)
    assert all(len(ix) == 192 for _, _, ixs in plan for ix in ixs)
    with pytest.raises(ValueError):
        stat.batch_plan(ids, orders, 148101)


def selector(rows_, gains):
    fam = sorted({x["family_id"] for x in rows_})
    idx = {f: i for i, f in enumerate(fam)}
    base = np.array([1.0 + 0.01 * idx[x["family_id"]] for x in rows_])
    return np.stack([base + g for g in gains], axis=1)


def test_choose_strict_margin_ties_and_lower_bound():
    r = rows(41, 3)
    tie = stat.choose(r, selector(r, [0, 0.001, 0, 0, 0]), 1)
    assert (
        tie["best_step"] == 10 and tie["selected_step"] == 0
    )  # exactly +0.001 is a tie: reject
    ok = stat.choose(r, selector(r, [0, 0.002, 0.002, 0, 0]), 1)
    assert ok["selected_step"] == 10 and ok["accepted"]  # earliest of equal best steps
    noisy = selector(r, [0, 0, 0, 0, 0])
    noisy[:, 4] += np.where(np.arange(len(r)) % 2, 0.5, -0.49)  # positive mean, LB <= 0
    assert stat.choose(r, noisy, 1)["selected_step"] == 0
    seeds = {
        stat.selection_seed(o, s, a)
        for o in range(5)
        for s in range(3)
        for a in stat.ARMS
    }
    assert len(seeds) == 60


def panels(r, gains_by_arm):
    base = np.array([1.0 + 0.01 * (int(x["family_id"][1:]) % 7) for x in r])
    cols = [base] + [base + gains_by_arm.get(a, 0.0) for a in stat.ARMS]
    return np.stack([np.stack(cols, axis=1)] * 3)


def test_identical_arms_have_elo_1500_and_rule_neutral():
    r = rows(41, 3)
    s = stat.summarize(r, panels(r, {}))
    assert all(v == pytest.approx(1500.0) for v in s["median_elo"].values())
    assert (
        s["contrasts"]["PROJ_K4_minus_r882"]["median_gain"] == 0
        and s["identical_seed_panels"]
    )
    zero = {a: 0 for a in stat.ARMS}
    d = stat.decide(s, zero, stat.drift_match([0.02], [0.02]))
    assert d["verdict"] == "RULE_NEUTRAL" and d["mechanism"] == "NO_SELECTION_BOTH"


def test_decision_gates_holm_and_mechanism_labels():
    r = rows(41, 3)
    s = stat.summarize(r, panels(r, {"PROJ_K4": 0.01, "CTRL_DRIFT": 0.0}))
    picks = {"CTRL": 0, "PROJ_K2": 0, "PROJ_K4": 5, "CTRL_DRIFT": 3}
    d = stat.decide(s, picks, stat.drift_match([0.02], [0.021]))
    assert d["verdict"] == "ADVANCE" and d["primary_arms"]["PROJ_K4"]["advance"]
    assert (
        not d["primary_arms"]["PROJ_K2"]["advance"]
        and d["mechanism"] == "SUBSPACE_SPECIFIC"
    )
    assert (
        stat.decide(s, picks, stat.drift_match([0.02], [0.05]))["mechanism"]
        == "DRIFT_MATCH_INVALID"
    )
    s2 = stat.summarize(r, panels(r, {"PROJ_K4": 0.0005}))
    assert (
        stat.decide(s2, picks, stat.drift_match([0.02], [0.02]))["verdict"]
        == "NO_ADVANCE"
    )  # below margin


def test_supervisor_hard_stops():
    ok = dict(
        rss=1e9,
        threads=3,
        free=600e9,
        disk_used=0.92,
        scratch=1e8,
        load=40.0,
        cores=102,
        available=70 * 1024**3,
    )
    assert mon.classify(ok) == ("run", None)
    for key, bad, why in [
        ("rss", 6e9, "RSS"),
        ("threads", 5, "TOTAL_THREADS"),
        ("free", 39e9, "DISK"),
        ("disk_used", 0.95, "DISK_PERCENT"),
        ("scratch", 1e9, "SCRATCH"),
        ("load", 86.8, "LOAD"),
        ("available", 15 * 1024**3, "MEMORY"),
    ]:
        assert mon.classify(dict(ok, **{key: bad})) == ("abort", why)


def test_amendment_a1_project_grads_zero_fills_missing_gradients():
    import torch

    rspec = importlib.util.spec_from_file_location("h160_test_run", HERE / "run.py")
    run = importlib.util.module_from_spec(rspec)
    rspec.loader.exec_module(run)
    u_np, _ = stat.basis(chain(dim=7), 2)
    u = torch.as_tensor(u_np)
    a = torch.nn.Parameter(torch.zeros(4))
    b = torch.nn.Parameter(
        torch.zeros(3)
    )  # receives no gradient this step (grad is None)
    a.grad = torch.arange(1.0, 5.0)
    missing = run.project_grads(torch, [a, b], [4, 3], u)
    assert missing == [1] and b.grad is not None
    full = np.r_[np.arange(1.0, 5.0), np.zeros(3)]
    got = np.r_[a.grad.numpy(), b.grad.numpy()]
    assert (
        np.allclose(got, stat.project(u_np, full), atol=1e-6)
        and stat.out_of_span(u_np, got) < 1e-6
    )
    assert run.project_grads(torch, [a, b], [4, 3], u) == []  # both now have gradients
