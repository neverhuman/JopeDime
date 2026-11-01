"""Synthetic controls only: no study inputs, checkpoint or optimizer updates."""

# ruff: noqa: E402
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis as a
from distill import (
    require_fit,
    student_objective,
    teacher_distribution,
    validate_initial_hashes,
)
from monitor import LOAD_LIMIT, violations
from runner import fit_cache, persist_scores


def test_fresh_folds_and_disjoint_fit():
    families = [f"f{i:03}" for i in range(205)]
    folds = a.family_folds(families)
    assert len(folds) == 5 and all(len(f) == 41 for f in folds)
    assert len(set(sum(folds, []))) == 205
    assert folds != a.BASE.family_folds(families, seed=182140)
    for outer in range(5):
        fit, b, assess = a.partition(families, outer, folds)
        assert (len(fit), len(b), len(assess)) == (123, 41, 41)
        assert not fit & b and not fit & assess and not b & assess
        require_fit(list(fit), fit, b | assess)
        with pytest.raises(ValueError):
            require_fit(list(assess), fit, b | assess)


def test_all_initial_seeds_distinct():
    teachers = [t for s in a.SEEDS for t in a.teacher_seeds(s)]
    assert teachers == list(range(183201, 183211)) + list(range(183301, 183311)) + list(
        range(183401, 183411)
    )
    assert len(set(teachers + list(a.STUDENT_SEEDS))) == 33
    validate_initial_hashes([str(t) for t in teachers[:10]], "student")
    with pytest.raises(ValueError):
        validate_initial_hashes(["same"] * 10, "student")
    with pytest.raises(ValueError):
        validate_initial_hashes([str(t) for t in teachers[:10]], str(teachers[0]))


def test_common_student_batch_prefix_and_scope():
    ids = [f"fit{i}" for i in range(12)]
    orders = {c: np.arange(32) for c in ids}
    p40 = a.batch_plan(ids, orders, 1234, updates=40)
    p80 = a.batch_plan(ids, orders, 1234, updates=80)
    for x, y in zip(p40, p80[:40], strict=True):
        assert x[:2] == y[:2]
        np.testing.assert_array_equal(x[2], y[2])
    a.validate_fit_plan(p80, ids, ["A"], ["B"])
    poisoned = copy.deepcopy(p80)
    poisoned[0][1][0] = "A"
    with pytest.raises(ValueError):
        a.validate_fit_plan(poisoned, ids, ["A"], ["B"])


def test_target_is_softmax_of_mean_costs_and_detached():
    x = np.tile(np.array([[[0.0, 1.0, 2.0]]], np.float32), (10, 1, 1))
    x[0] = [[3.0, -1.0, 0.0]]
    mean, prob = teacher_distribution(x)
    np.testing.assert_array_equal(mean, x.mean(axis=0, dtype=np.float32))
    expected = torch.softmax(-torch.from_numpy(mean) / 0.2, dim=-1)
    torch.testing.assert_close(prob, expected, rtol=0, atol=0)
    assert not prob.requires_grad
    assert not torch.allclose(
        prob, torch.softmax(-torch.from_numpy(x) / 0.2, dim=-1).mean(dim=0)
    )
    with pytest.raises(ValueError):
        teacher_distribution(x[:9])


def test_loss_weights_temperature_and_no_t_squared():
    prediction = torch.tensor([[0.1, -0.2, 0.4]], requires_grad=True)
    target = torch.tensor([[0.2, 0.7, 0.1]])

    def live_fn(p, f, g):
        return p.square().mean()

    loss, live, kl = student_objective(prediction, None, None, live_fn, target)
    expected_kl = F.kl_div(
        F.log_softmax(-prediction / 0.2, dim=-1), target, reduction="batchmean"
    )
    torch.testing.assert_close(kl, expected_kl)
    torch.testing.assert_close(
        loss, 0.5 * prediction.square().mean() + 0.5 * expected_kl
    )
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert target.grad is None
    with pytest.raises(ValueError):
        student_objective(
            prediction, None, None, live_fn, target.clone().requires_grad_()
        )
    assert student_objective(prediction, None, None, live_fn)[2] == 0


def test_poisoned_outer_context_never_reaches_fit_cache():
    class Engine:
        def __init__(self):
            self.seen = []

        def batch(self, ids, indexes):
            if "A" in ids or "B" in ids:
                raise AssertionError("poisoned outer outcome accessed")
            self.seen += ids
            return ({}, None, None, None, None)

        def anchored(self, kwargs):
            return None

    engine = Engine()
    fit_cache(engine, [(1, ["fit"], [np.arange(24)])], ["fit"], ["A", "B"])
    assert engine.seen == ["fit"]
    with pytest.raises(ValueError):
        fit_cache(engine, [(1, ["A"], [])], ["fit"], ["A", "B"])
    assert engine.seen == ["fit"]


def test_bootstrap_preserves_all_draws_axis_zero():
    delta = np.array(
        [[0.0, 1.0, 10.0, 3.0], [1.0, 0.0, -2.0, 7.0], [9.0, -1.0, 2.0, 4.0]]
    )
    draw = np.random.default_rng(183141).integers(0, 4, (10000, 4))
    actual = a.paired_family_bootstrap(delta, draw)
    expected = np.array([np.median(delta[:, row].mean(axis=1)) for row in draw])
    assert actual.shape == (10000,)
    np.testing.assert_array_equal(actual, expected)
    assert len(np.unique(actual)) > 10


def verdict(gain=0.002, lcb=0.0001, seeds=(0.002, 0.003, 0.004), p=0.01):
    return dict(
        median_over_seeds=gain,
        ci95=[lcb, 0.01],
        per_seed=list(seeds),
        holm_m2_adjusted_p=p,
    )


@pytest.mark.parametrize(
    "bad",
    [dict(gain=0.001), dict(lcb=0.0), dict(seeds=(0.0, 0.003, 0.004)), dict(p=0.05)],
)
def test_primary_gate_rejects_boundary_or_secondary_only(bad):
    s = dict(gain_vs_r882={"DIST10": verdict(**bad), "DIST10_80": verdict()})
    assert a.decide(s)["verdict"] == "NO_ADVANCE"
    assert s["secondary_only"] and not s["serving_promotion"]


def test_primary_gate_accepts_only_dist40():
    s = dict(gain_vs_r882={"DIST10": verdict(), "DIST10_80": verdict(gain=-0.1)})
    assert a.decide(s)["verdict"] == "ADVANCE_CANDIDATE"
    assert not s["serving_promotion"]
    np.testing.assert_allclose(a.holm_adjust([0.01, 0.04]), [0.02, 0.04])


def test_ragged_scores_native_minus2_and_first_tie(tmp_path):
    rows = [
        dict(context_sha256="a", family_id="f"),
        dict(context_sha256="b", family_id="g"),
    ]
    engine = SimpleNamespace(
        pools={
            "a": dict(fitness=np.array([-1.0, -2.0]), gate=np.array([False, True])),
            "b": dict(
                fitness=np.array([-1.0, -2.0, -3.0]), gate=np.array([True, True, True])
            ),
        }
    )
    u, c = persist_scores(
        tmp_path,
        "test",
        rows,
        {"a": np.array([0.0, 0.0]), "b": np.array([2.0, 1.0, 0.0])},
        engine,
    )
    assert u == {"a": -2.0, "b": 3.0} and c == {"a": 0, "b": 2}
    with np.load(tmp_path / "test.npz", allow_pickle=False) as z:
        assert z["scores"].dtype == np.float32
        assert z["offsets"].tolist() == [0, 2, 5]
        assert z["candidate_counts"].tolist() == [2, 3]


def test_fit_only_timing_choice_keeps_three_seeds():
    assert a.execution_plan(100)["outer_folds"] == 5
    assert a.execution_plan(1000)["outer_folds"] == 3
    assert a.execution_plan(1000)["assessed_families"] == 123
    assert not a.execution_plan(1500)["within_cap"]
    assert a.execution_plan(1500)["seeds"] == 3


def test_strict_resource_guards():
    good = dict(
        load=0.0,
        cpu=0.0,
        available=20 * 2**30,
        free=50_000_000_000,
        disk_used_fraction=0.937,
        rss=1_000_000_000,
        threads=3,
    )
    assert violations(good) == []
    for key, value in dict(
        load=LOAD_LIMIT + 1,
        cpu=86,
        available=15 * 2**30,
        free=39_999_999_999,
        disk_used_fraction=0.938,
        rss=6_000_000_000,
        threads=5,
    ).items():
        assert violations(dict(good, **{key: value}))


def test_full_summary_equal_family_reduction_and_diagnostic_control():
    panel = np.zeros((3, 6, 6))
    families = ["f", "f", "f", "f", "g", "h"]
    panel[:, :4, a.ARMS.index("ENS10")] = 0.09
    panel[:, 4:, a.ARMS.index("RENS10")] = 0.03
    panel[:, :, a.ARMS.index("DIST10_80")] = 0.2
    summary = a.summarize(panel, families, draws=100)
    assert summary["arms"]["ENS10"]["median_over_seeds"] == pytest.approx(0.03)
    assert summary["contrasts"]["ENS10_minus_RENS10"][
        "median_over_seeds"
    ] == pytest.approx(0.01)
    assert set(a.PRIMARY_FAMILY) == {"DIST10", "DIST10_80"}
    assert "holm_m2_adjusted_p" not in summary["gain_vs_r882"]["RENS10"]
    assert summary["verdict"] == "NO_ADVANCE" and summary["secondary_only"]
    assert summary["family_bootstrap_draws"] == 100
    assert summary["elo_vs_r882"]["R882"]["median_over_seeds"] == 1500.0
