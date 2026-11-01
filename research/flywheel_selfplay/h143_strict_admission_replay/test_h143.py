"""Synthetic CPU controls for the H143 strict-admission shadow replay (no receipts, checkpoints or model passes)."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("h143_test_runner", HERE / "runner.py")
r = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r
SPEC.loader.exec_module(r)

FAMILIES = [f"f{i % 19:02d}" for i in range(91)]
BASE = [0.5] * 91


def _with(changes):
    v = list(BASE)
    for k, x in changes.items():
        v[k] = x
    return v


def test_fixed_protocol_constants():
    assert r.SEEDS == (143101, 143102, 143103) and r.DRAWS == 2000 and r.MARGIN == 0.001
    assert r.ROUNDS == (775, 2530) and set(r.ARMS) == {"R_STRICT", "R_STRICT_NOINV"}


def test_live_rule_accepts_improvements_and_exact_ties():
    assert r.live_accept(1.1, 1.0) == (True, "accept_improvement")
    assert r.live_accept(1.0, 1.0) == (True, "accept_tie")
    assert r.live_accept(0.9, 1.0) == (False, "restore_best_model_and_optimizer")


def test_strict_rejects_ties_and_small_gains_and_accepts_broad_gains():
    rng = np.random.default_rng(0)
    assert not r.strict_accept(BASE, BASE, FAMILIES, rng)[0]  # exact tie
    assert not r.strict_accept([x + 0.0005 for x in BASE], BASE, FAMILIES, rng)[
        0
    ]  # below margin
    assert r.strict_accept([x + 0.01 for x in BASE], BASE, FAMILIES, rng)[0]


def test_single_cliff_flip_is_rejected_when_invalid_contexts_are_excluded():
    inc = _with({0: -2.0})  # the incumbent picked an invalid program in one context
    cand = list(BASE)  # the candidate fixes only that context: a pure cliff flip
    accepted_strict, _ = r.strict_accept(cand, inc, FAMILIES, np.random.default_rng(1))
    accepted_noinv, _ = r.strict_accept(
        cand, inc, FAMILIES, np.random.default_rng(1), drop_invalid=True
    )
    assert accepted_strict and not accepted_noinv


def test_family_gains_drop_empty_families_without_invalid_contexts():
    fam = ["a", "a", "b"]
    g = r.family_gains([1.0, -2.0, 1.0], [0.0, 0.0, -2.0], fam, drop_invalid=True)
    assert g.tolist() == [1.0]


def test_replay_tracks_incumbent_and_is_seed_reproducible():
    stream = [
        {"round": 1, "per_context": [x + 0.01 for x in BASE], "mean": 0.51},
        {
            "round": 2,
            "per_context": [x + 0.0105 for x in BASE],
            "mean": 0.5105,
        },  # +0.0005 over new incumbent
        {"round": 3, "per_context": [x + 0.03 for x in BASE], "mean": 0.53},
    ]
    start = (0, BASE, 0.5)
    live = r.replay(stream, start, FAMILIES, "S_LIVE")
    assert live["admissions"] == [1, 2, 3] and live["final_round"] == 3
    a = r.replay(stream, start, FAMILIES, "R_STRICT", 143101)
    b = r.replay(stream, start, FAMILIES, "R_STRICT", 143101)
    assert a["admissions"] == b["admissions"] == [1, 3]


def test_cliff_driven_flags_admissions_explained_by_invalid_flips():
    inc = _with({0: -2.0})
    stream = {
        5: {"per_context": list(BASE)},
        6: {"per_context": [x + 0.02 for x in BASE]},
    }
    assert r.cliff_driven(stream, [5, 6], inc, FAMILIES) == [5]


@pytest.mark.parametrize(
    "dl,expected",
    [
        (
            {
                "1": {"available": True, "ci95": [0.001, 0.01]},
                "2": {"available": True, "ci95": [0.002, 0.01]},
                "3": {"available": True, "ci95": [-0.01, 0.01]},
            },
            "RULE_HELPS",
        ),
        (
            {
                "1": {"available": True, "ci95": [-0.01, -0.001]},
                "2": {"available": True, "ci95": [-0.02, -0.001]},
                "3": {"available": True, "ci95": [-0.01, 0.01]},
            },
            "RULE_HURTS",
        ),
        (
            {s: {"available": True, "ci95": [-0.01, 0.01]} for s in "123"},
            "RULE_NEUTRAL",
        ),
        (
            {
                "1": {"available": True, "ci95": [0.01, 0.02]},
                "2": {"available": False, "ci95": None},
                "3": {"available": False, "ci95": None},
            },
            "INCOMPLETE_COVERAGE",
        ),
    ],
)
def test_verdict_labels(dl, expected):
    assert r.verdict(dl) == expected


def test_native_endpoint():
    assert r.native([2.0, 1.0], [0.3, 0.4], [True, True]) == pytest.approx(-0.4)
    assert r.native([2.0, 1.0], [0.3, 0.4], [True, False]) == -2.0
