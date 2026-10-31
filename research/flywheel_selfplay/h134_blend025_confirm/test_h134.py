"""Synthetic CPU controls for the H134 single-primary BLEND_025 confirmation (no corpus, checkpoint or fits)."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("h134_test_runner", HERE / "runner.py")
r = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r
SPEC.loader.exec_module(r)

SAMPLES = np.random.default_rng(3).integers(0, 205, size=(400, 205))


def test_h129_runner_is_pinned_and_only_fold_seeds_are_rebound():
    h129 = r.load_h129()
    assert (h129.SEEDS, h129.FOLD_SEED, h129.SAMPLE_SEED) == (
        (141, 142, 143),
        134141,
        134142,
    )
    assert h129.LAMBDAS["BLEND_025"] == 0.25 and h129.H124_SHA.startswith("9ebce965")
    h124 = h129.load_h124()
    assert (h124.SEEDS, h124.FOLD_SEED, h124.SAMPLE_SEED) == (
        (141, 142, 143),
        134141,
        134142,
    )
    used = {64, 65, 66, 91, 92, 93, 51, 52, 53}
    assert not set(r.SEEDS) & used and r.FOLD_SEED not in (12400, 129029, 11900)


def _gain(level=0.01, noise=0.002, harmed=0, seed=0):
    rng = np.random.default_rng(seed)
    gain = level + noise * rng.standard_normal((3, 205))
    gain[:, :harmed] = -0.05
    return gain


def test_clean_positive_gain_is_admitted():
    out = r.readout(_gain(), SAMPLES)
    assert out["admitted"] and all(out["gates"].values())


@pytest.mark.parametrize(
    "gain,failed",
    [
        (_gain(level=0.0008, noise=0.0), "median_seed_gain_gt_0.001"),
        (_gain(level=0.012, harmed=21), "harm_le_10pct"),
    ],
)
def test_each_gate_can_fail(gain, failed):
    out = r.readout(gain, SAMPLES)
    assert not out["gates"][failed] and not out["admitted"]


def test_any_negative_seed_fails():
    gain = _gain()
    gain[2] = -0.002
    out = r.readout(gain, SAMPLES)
    assert not out["gates"]["all_seeds_gt_0"] and not out["admitted"]


def test_trimmed_mean_rejects_gain_carried_by_ten_families():
    gain = np.full((3, 205), -0.0005)
    gain[:, :10] = 0.5  # a ten-family tail produces the whole gain
    out = r.readout(gain, SAMPLES)
    assert out["median_seed_gain"] > 0.001
    assert (
        out["trimmed_mean_drop10_largest_abs"] < 0
        and not out["gates"]["trimmed_mean_gt_0"]
    )
    assert out["top10_gain_share"] > 1 and not out["admitted"]


def test_tail_checks_definitions():
    g = np.arange(20, dtype=float) - 5  # -5..14
    t = r.tail_checks(g)
    assert t["family_median_gain"] == pytest.approx(4.5)
    assert t["top10_gain_share"] == pytest.approx(sum(range(5, 15)) / g.sum())
    keep = np.argsort(-np.abs(g), kind="stable")[10:]
    assert t["trimmed_mean_drop10_largest_abs"] == pytest.approx(g[keep].mean())


def test_dry_run_is_data_free():
    out = r.cpu_dry_run()
    assert (
        out["status"] == "CPU_DRY_RUN_PASS"
        and out["data_reads"] == 0
        and out["estimator_fits"] == 0
    )
