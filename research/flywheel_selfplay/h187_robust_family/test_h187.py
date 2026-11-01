"""Data-free H187 exclusion, pairing, influence, bootstrap and stage-order controls."""

import json
import numpy as np
import pytest
import torch
import design as d
import objectives as o
from monitor import violations, LOAD_LIMIT


def rows():
    return [dict(context_sha256=str(i), family_id="F" + str(i)) for i in range(205)]


def test_fresh_folds_fit_only_and_complete_cyclic_coverage():
    allrows = rows()
    seenB = []
    seenA = []
    for outer in range(5):
        fit, b, a = d.partition(allrows, outer)
        assert [len(x) for x in (fit, b, a)] == [123, 41, 41]
        seenB.extend(r["context_sha256"] for r in b)
        seenA.extend(r["context_sha256"] for r in a)
        with pytest.raises(ValueError):
            d.require_fit([a[0]["context_sha256"]], fit)
        with pytest.raises(ValueError):
            d.require_fit([b[0]["context_sha256"]], fit)
        assert not ({r["family_id"] for r in fit} & {r["family_id"] for r in a + b})
    assert (
        sorted(seenB) == sorted(seenA) == sorted(r["context_sha256"] for r in allrows)
    )


def test_identical_initial_weights_across_arms_are_independent_clones():
    initial = {"one": torch.arange(5.0), "two": torch.zeros(3)}
    paired = o.paired_initial_states(initial, d.ARMS)
    for arm in d.ARMS:
        for n, value in initial.items():
            assert (
                torch.equal(value, paired[arm][n])
                and value.data_ptr() != paired[arm][n].data_ptr()
            )
    paired[d.ARMS[0]]["one"][0] = 99
    assert paired[d.ARMS[1]]["one"][0] == initial["one"][0] == 0


def test_huber_influence_clips_at_two_without_zero_gradient_tail():
    z = torch.tensor([-100.0, -2.0, -1.0, 0.0, 1.0, 2.0, 100.0], requires_grad=True)
    o.huber(z).sum().backward()
    torch.testing.assert_close(
        z.grad, torch.tensor([-2.0, -2.0, -1.0, 0.0, 1.0, 2.0, 2.0])
    )
    assert z.grad.abs().max() == 2


def test_rnorm_detaches_anchor_scale_and_bounds_standardized_influence():
    pred = torch.tensor([[100.0, 0.0], [2.0, -1.0]], requires_grad=True)
    anchor = torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)
    fitness = torch.zeros_like(pred)
    gate = torch.ones_like(pred, dtype=torch.bool)
    loss, info = o.normalized_loss(pred, anchor, fitness, gate)
    loss.backward()
    assert anchor.grad is None
    assert info["anchor_loss_scale"].min() >= o.SCALE_FLOOR
    assert info["influence"].abs().max() <= 2
    assert torch.isfinite(pred.grad).all()


def test_per_context_loss_matches_h180_live_mean_on_fixture():
    runtime = d.sibling(
        "h187_test_runtime", "research/flywheel_selfplay/h146_lookahead_exec/runtime.py"
    )
    g = torch.Generator().manual_seed(187)
    p = torch.randn(8, 24, generator=g)
    f = torch.randn(8, 24, generator=g) * 3
    gate = torch.rand(8, 24, generator=g) > 0.2
    torch.testing.assert_close(
        o.per_context_loss(p, f, gate).mean(), runtime.loss(p, f, gate), rtol=0, atol=0
    )


def test_family_weights_sum_one_and_each_family_gets_equal_mass():
    r = [
        dict(context_sha256=f"{f}:{j}", family_id=f)
        for f, n in [("a", 1), ("b", 5), ("c", 17)]
        for j in range(n)
    ]
    w = d.family_probabilities(r)
    assert w.sum() == pytest.approx(1)
    for f in ["a", "b", "c"]:
        assert sum(
            p for row, p in zip(r, w, strict=True) if row["family_id"] == f
        ) == pytest.approx(1 / 3)
    orders = {row["context_sha256"]: np.arange(30) for row in r}
    plan = d.balanced_plan(r, orders, 5, 160)
    assert len(plan) == 160 and all(
        len(c) == 8 and all(len(ix) == 24 for ix in indexes) for _, c, indexes in plan
    )
    assert plan[0][0] == d.h177.stream_seed(5, 1)


def test_pilot_never_drops_folds_seeds_or_arms():
    p = d.execution_plan(0.06)
    assert p["updates_per_cell"] == 160 and p["total_optimizer_updates"] == 7200
    p = d.execution_plan(1.5)
    assert p["updates_per_cell"] == 80 and p["total_optimizer_updates"] == 3600
    assert p["folds"] == 5 and p["seeds"] == 3
    with pytest.raises(RuntimeError):
        d.execution_plan(3.0)


def test_B_lock_before_A_and_immutable_step(tmp_path):
    guard = d.StageOrderGuard(expected_steps={a: 160 for a in d.h177.ARMS})
    with pytest.raises(RuntimeError):
        guard.require_b_locked()
    with pytest.raises(RuntimeError):
        guard.set_candidate_step("ACC4", 80)
    lock = tmp_path / "pooled_b_selection_lock.json"
    lock.write_text(
        json.dumps(
            dict(
                wave="H177",
                pooled_b_families=205,
                decisions={a: {} for a in d.h177.ARMS},
            )
        )
    )
    guard.mark_b_locked(lock)
    assert guard.require_b_locked()["pooled_b_families"] == 205
    lock.write_text("{}")
    with pytest.raises(RuntimeError):
        guard.require_b_locked()


def test_imported_rule_m3_branch_and_tie_rejection():
    assert d.H177_DECISION_RULE is d.h177.H177_DECISION_RULE
    assert d.StageOrderGuard is d.h177.StageOrderGuard
    base = np.zeros((3, 205))
    candidate = {
        a: np.full((3, 205), 0.02 if i else 0.001 + d.EPS) for i, a in enumerate(d.ARMS)
    }
    decisions, native = d.select(rows(), base, candidate)
    for arm in d.ARMS:
        assert decisions[arm]["holm_p"] == native[d.SLOTS[arm]]["holm_p_treatments3"]
        assert (
            decisions[arm]["admitted"] == native[d.SLOTS[arm]]["admitted_treatments3"]
        )
        assert decisions[arm]["holm_multiplicity"] == 3
    assert not decisions[d.ARMS[0]]["admitted"] and decisions[d.ARMS[1]]["admitted"]


def test_axis0_shared_seed_reduction_non_degenerate_interval(monkeypatch):
    monkeypatch.setattr(d, "DRAWS", 200)
    rng = np.random.default_rng(11)
    base = rng.normal(size=(3, 205))
    gain = np.stack([rng.normal(0, z, 205) for z in [0.01, 1, 20]])
    panel = np.stack([base, base + gain, base + gain / 2, base - gain / 2], axis=2)
    summary = d.summarize(rows(), panel)
    ix = np.random.default_rng(d.BOOT_SEED).integers(0, 205, (200, 205))
    order = np.argsort([r["family_id"] for r in rows()])
    expected = np.median(gain[:, order][:, ix].mean(axis=2), axis=0)
    c = summary["contrasts"]["FRESH160_minus_R882"]
    np.testing.assert_allclose(
        c["ci95"], np.quantile(expected, [0.025, 0.975]), atol=1e-13
    )
    assert expected.shape == (200,) and np.ptp(expected) > 0.05
    assert summary["Elo"]["R882"]["median"] == 1500


def test_advance_requires_B_plus_positive_three_seed_A_and_LCB():
    b = {a: dict(admitted=True) for a in d.ARMS}
    A = dict(
        contrasts={
            a + "_minus_R882": dict(median=0.02, positive_seeds=3, ci95=[0.001, 0.03])
            for a in d.ARMS
        }
    )
    assert d.verdict(b, A)["verdict"] == "ADVANCE"
    for a in d.ARMS:
        A["contrasts"][a + "_minus_R882"]["ci95"][0] = 0
    assert d.verdict(b, A)["verdict"] == "NO_ADVANCE"
    for a in d.ARMS:
        A["contrasts"][a + "_minus_R882"].update(ci95=[0.001, 0.03], positive_seeds=2)
    assert d.verdict(b, A)["verdict"] == "NO_ADVANCE"


def test_resource_limits_match_owner_boundary():
    x = dict(
        load=0,
        cpu=10,
        available=30 * 2**30,
        free=50e9,
        disk_used_fraction=0.9,
        rss=1e9,
        threads=3,
    )
    assert not violations(x)
    for k, v in [
        ("cpu", 85.1),
        ("available", 15 * 2**30),
        ("free", 39e9),
        ("disk_used_fraction", 0.938),
        ("rss", 6e9),
        ("threads", 5),
        ("load", LOAD_LIMIT + 1),
    ]:
        assert violations(dict(x, **{k: v}))


def test_committed_calibration_receipt_matches_imported_source():
    import hashlib

    receipt = json.loads(
        (
            d.ROOT
            / "research/flywheel_selfplay/h178_pooled_calibration/h178_receipt.json"
        ).read_text()
    )
    sha = hashlib.sha256(
        (
            d.ROOT / "research/flywheel_selfplay/h177_pooled_selector/analysis.py"
        ).read_bytes()
    ).hexdigest()
    result = d.H177_DECISION_RULE.resolve_verdict(
        True, receipt, expected_analysis_sha256=sha
    )
    assert result["verdict"] == "ADVANCE" and result["lift_qualified"]


def test_relaunch_sustained_load_and_xbabe1_resource_floors():
    from monitor import next_load_streak

    x = dict(
        load=LOAD_LIMIT + 1,
        cpu=36.5,
        available=30 * 2**30,
        free=100e9,
        disk_used_fraction=0.95,
        rss=1e9,
        threads=3,
        output_bytes=0,
    )
    streak = 0
    for i in range(1, 4):
        streak = next_load_streak(streak, x)
        assert bool(
            violations(dict(x, disk_used_fraction=0.9), streak, host="xbabe2")
        ) == (i == 3)
    assert next_load_streak(streak, dict(x, load=0)) == 0
    healthy = dict(x, load=0)
    assert not violations(healthy, 0, host="xbabe1")
    assert violations(dict(healthy, free=79e9), 0, host="xbabe1")
    assert violations(dict(healthy, output_bytes=1e9), 0, host="xbabe1")
    assert violations(dict(healthy, cpu=85.1), 0, host="xbabe1")
