"""Synthetic CPU controls for the H139 selection-rule audit (no checkpoints, corpus or model passes)."""

import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("h139_test_runner", HERE / "runner.py")
r = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r
SPEC.loader.exec_module(r)

SAMPLES = np.random.default_rng(0).integers(0, 50, size=(2000, 50))


def test_arms_are_fixed_and_degenerate_arms_are_excluded_from_inference():
    assert list(r.ARMS) == ["R882", "R782", "R859", "R1703", "R2481", "REJ"]
    assert set(r.DEGENERATE) | set(r.DISTINCT) | {r.ANCHOR} == set(r.ARMS)
    assert not set(r.DEGENERATE) & set(r.DISTINCT)
    assert r.FOLD_SEEDS == (139191, 139192, 139193)


def test_native_endpoint_is_first_argmin_with_invalid_minus_two():
    assert r.native(
        [3.0, 1.0, 1.0], [5.0, 0.2, 0.1], [True, True, True]
    ) == pytest.approx(-0.2)
    assert r.native([3.0, 1.0], [5.0, 0.2], [True, False]) == -2.0
    with pytest.raises(ValueError):
        r.native([np.nan, 1.0], [1.0, 1.0], [True, True])


def test_family_matrix_weights_families_equally():
    order, fm = r.family_matrix({"a": [1.0, 3.0, 10.0]}, ["f1", "f1", "f2"])
    assert order == ["f1", "f2"] and fm["a"].tolist() == [2.0, 10.0]


def test_gain_stats_ci_and_p_value():
    worse = r.gain_stats(np.full(50, -0.05) + 0.001 * np.arange(50) / 50, SAMPLES)
    assert (
        worse["ci95"][1] < 0
        and worse["p_two_sided"] < 0.01
        and worse["harm_fraction"] == 1.0
    )
    noise = r.gain_stats(np.random.default_rng(1).normal(0, 0.01, 50), SAMPLES)
    assert noise["ci95"][0] < 0 < noise["ci95"][1]
    zero = r.gain_stats(np.zeros(50), SAMPLES)
    assert zero["p_two_sided"] == 1.0 and zero["families_win_tie_loss"] == [0, 50, 0]


def test_holm_step_down():
    assert r.holm({"a": 0.01, "b": 0.02, "c": 0.04}) == {
        "a": True,
        "b": True,
        "c": True,
    }
    assert r.holm({"a": 0.03, "b": 0.001, "c": 0.5}) == {
        "a": False,
        "b": True,
        "c": False,
    }


def test_spearman_handles_ties_and_constant_vectors():
    assert r.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert r.spearman([1, 1, 2, 3], [4, 3, 2, 1]) < 0
    assert math.isnan(r.spearman([1, 1, 1, 1], [1, 2, 3, 4]))


def test_decision_rule():
    stats = {a: {"ci95": [-0.03, -0.01]} for a in r.DISTINCT}
    assert (
        r.decide(stats, (-0.5, 0.9))[0] == "PLATEAU_REAL"
    )  # every distinct arm significantly worse
    stats["R859"] = {"ci95": [-0.01, 0.002]}
    assert r.decide(stats, (-0.5, 0.9))[0] == "SELECTION_ARTEFACT"
    assert (
        r.decide(stats, (0.2, 0.9))[0] == "PLATEAU_REAL"
    )  # val ranking predicts cross-fit ranking
    assert (
        r.decide(stats, (math.nan, math.nan))[0] == "SELECTION_ARTEFACT"
    )  # undefined CI counts as including 0


def test_cross_fit_selection_optimism_uses_only_training_folds():
    families = [f"f{i:02d}" for i in range(50)]
    fm = {a: np.zeros(50) for a in r.ARMS}
    fm["R782"] = np.where(
        np.arange(50) < 40, 0.02, -0.5
    )  # looks best in-sample, terrible on a tail
    out = r.selection_optimism(fm, families, 139191, tuple(r.ARMS))
    assert len(out) == 5 and all(
        set(x) == {"fold", "chosen", "in_sample_gain", "held_out_gain"} for x in out
    )
    for x in out:
        if x["chosen"] == "R882":
            assert x["in_sample_gain"] == 0
    folds = r.folds(families, 139191)
    assert sorted(f for g in folds for f in g) == sorted(families) and len(folds) == 5


def test_regression_check_requires_exact_91_context_parity():
    ref = {"utility": {"per_context": [1.1377942] * 91}, "best_utility": 1.1377942}
    assert r.regression_check([1.1377942] * 91, ref)["contexts_matched"] == 91
    bad = [1.1377942] * 90 + [1.2]
    with pytest.raises(RuntimeError, match="regression FAIL"):
        r.regression_check(bad, ref)
