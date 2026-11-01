"""Meaningful CPU checks: orientation, calibration effect, isolation and inference."""

import numpy as np
import torch
from runner import pairwise, fit_temperature, split_families, summarize, decisions, ARMS


def test_rank_orientation_and_smoothing_are_effective():
    target = torch.tensor([[0.0, 1.0, 2.0]])
    good = torch.tensor([[0.0, 1.0, 2.0]])
    bad = -good
    assert pairwise(good, target, 0.0, 1) < pairwise(bad, target, 0.0, 1)
    pred = torch.tensor([[0.0, 4.0, 8.0]], requires_grad=True)
    g0 = torch.autograd.grad(pairwise(pred, target, 0.0, 1), pred, retain_graph=True)[0]
    gs = torch.autograd.grad(pairwise(pred, target, 0.1, 1), pred)[0]
    assert not torch.equal(g0, gs)
    assert torch.isfinite(gs).all()


def test_all_ties_produce_finite_zero_gradient():
    pred = torch.tensor([[0.2, -0.1]], requires_grad=True)
    value = pairwise(pred, torch.ones_like(pred), 0.1, 1)
    value.backward()
    assert value == 0 and torch.equal(pred.grad, torch.zeros_like(pred))


def test_calibration_reduces_overconfidence_but_cannot_change_argmin():
    logits = torch.tensor([12.0, 12.0, 12.0, 12.0])
    labels = torch.tensor([1.0, 1.0, 1.0, 0.0])
    temperature, losses = fit_temperature(logits, labels, 0.1)
    assert 1 < temperature <= 20.000001
    assert losses[-1] < losses[0]
    pred = np.array([-0.2, 3.0, 0.5])
    assert np.argmin(pred) == np.argmin(pred / temperature)


def test_family_split_is_deterministic_and_disjoint():
    rows = [{"family_id": f"f{i:03d}"} for i in range(205)]
    groups = split_families(rows)
    assert groups == split_families(list(reversed(rows)))
    assert len(set(sum(groups, []))) == 205
    assert [len(g) for g in groups] == [41] * 5
    fit = set(sum(groups[2:], []))
    assert not fit & set(groups[0] + groups[1])


def test_elo_weights_families_equally_and_ties_at_baseline():
    rows = [{"family_id": "big"}] * 9 + [{"family_id": "small"}]
    values = np.array([[1.0] * 9 + [-1.0]] * 3)
    summary = summarize(rows, values, np.zeros(10))
    assert summary["median_family_gain"] == 0
    assert summary["median_elo"] == 1500
    tied = summarize(rows, np.zeros((3, 10)), np.zeros(10))
    assert tied["median_elo"] == 1500 and tied["p_one_sided"] == 1


def test_negative_r882_contrast_blocks_promising_despite_control_lift():
    positive = dict(
        p_one_sided=0.001,
        median_family_gain=0.1,
        gain_ci95=[0.01, 0.2],
        seeds_beating=3,
    )
    negative = dict(positive, median_family_gain=-0.01, gain_ci95=[-0.02, -0.001])
    inner = {a: {"vs_CTRL": positive, "vs_r882": negative} for a in ARMS}
    assert not any(x["promising"] for x in decisions(inner).values())
