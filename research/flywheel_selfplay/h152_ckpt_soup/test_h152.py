"""Data-free controls for H152 soups and readout (synthetic tensors and gains only)."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("h152_test_runner", HERE / "runner.py")
r = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r
SPEC.loader.exec_module(r)


def states():
    base = {
        "adapter_10tok.w": torch.tensor([0.0, 4.0]),
        "adapter_10tok.anchor.b": torch.tensor([1.0]),
        "dope_model.x": torch.tensor([7.0]),
    }
    out = {}
    for rnd, v in ((782, 2.0), (785, 4.0), (859, 6.0), (882, 8.0)):
        s = {k: t.clone() for k, t in base.items()}
        s["adapter_10tok.w"] = torch.tensor([v, v])
        out[rnd] = s
    return out


def test_arms_are_fixed_and_weights_sum_to_one():
    assert set(r.ARMS) == {"R882", "MID", "SOUP4", "A025", "A075"}
    assert r.ADMISSIBLE == ("MID", "SOUP4") and set(r.EXPLORATORY) == {"A025", "A075"}
    assert all(abs(sum(w.values()) - 1) < 1e-12 for w in r.ARMS.values())
    assert r.ARMS["MID"] == {782: 0.5, 882: 0.5} and r.SEEDS == (152101, 152102, 152103)


def test_blend_averages_only_trainable_tensors_and_keeps_r882_otherwise():
    s = states()
    s[782]["dope_model.x"] = torch.tensor(
        [99.0]
    )  # a differing frozen tensor must not leak into the soup
    mid = r.blend(s, r.ARMS["MID"], {"adapter_10tok.w"})
    assert mid["adapter_10tok.w"].tolist() == [5.0, 5.0]
    assert mid["dope_model.x"].tolist() == [7.0] and mid[
        "adapter_10tok.anchor.b"
    ].tolist() == [1.0]
    soup = r.blend(s, r.ARMS["SOUP4"], {"adapter_10tok.w"})
    assert soup["adapter_10tok.w"].tolist() == [5.0, 5.0]
    assert r.frozen_identity(s, {"adapter_10tok.w"}) == ["dope_model.x"]
    with pytest.raises(ValueError):
        r.blend(s, {782: 0.7, 882: 0.7}, {"adapter_10tok.w"})


def test_gain_readout_is_deterministic_and_ties_score_half():
    g = np.zeros(19)
    out = r.gain_readout(g)
    assert (
        out["gain"] == 0
        and out["elo"] == pytest.approx(1500.0)
        and out["families_win_tie_loss"] == [0, 19, 0]
    )
    assert out == r.gain_readout(g)


def test_decision_requires_positive_gain_ci_and_holm():
    pos = r.gain_readout(np.full(19, 0.01))
    tie = r.gain_readout(np.zeros(19))
    d = r.decide(pos, tie)
    assert d["MID"]["advance"] and not d["SOUP4"]["advance"]
    d = r.decide(tie, tie)
    assert not d["MID"]["advance"] and not d["MID"]["gates"]["gain_gt_0"]
    neg = r.gain_readout(np.full(19, -0.01))
    assert not r.decide(neg, neg)["MID"]["advance"]


def test_concentrated_gain_is_not_admitted_without_ci_support():
    g = np.zeros(19)
    g[0] = 0.5  # one family carries everything
    out = r.gain_readout(g)
    assert out["median_ci95"][0] <= 0 and not r.decide(out, out)["MID"]["advance"]


def test_elo_ci_brackets_point():
    g = np.r_[np.full(10, 0.01), np.full(9, -0.01)]
    lo, hi = r.elo_ci(g)
    assert lo <= r.gain_readout(g)["elo"] <= hi
