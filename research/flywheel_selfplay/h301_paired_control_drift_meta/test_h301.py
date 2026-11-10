"""Unit and regression tests for H301 (h301_paired_control_drift_meta)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis_h301 as a301  # noqa: E402
import run_h301  # noqa: E402


def test_step1_regression_receipt_bit_for_bit() -> None:
    """Verify H297 and H298 exact-VersionId S3 regression receipt matches committed metrics bit-for-bit."""
    reg_path = HERE / "regression_receipt.json"
    assert reg_path.exists()
    reg = json.loads(reg_path.read_text())
    assert reg["status"] == "PASS"
    assert reg["h297_exact_s3_get"]["results_tar_version_id"] == "K7zNgkztjDJ32iTNlNc9clWVrzZDw2tF"
    assert reg["h297_exact_s3_get"]["bit_for_bit_match_all_arms"] is True
    assert reg["h297_exact_s3_get"]["recomputed_median_lift"] == -0.008461025318455606
    assert reg["h297_exact_s3_get"]["recomputed_median_lift_ci95"] == [
        -0.019228082690923506,
        0.0003517660130095988,
    ]
    assert reg["h298_exact_s3_get"]["results_tar_version_id"] == "lG1UYi3tUz3vNfkFllycDAWzTvOLIbOE"
    assert reg["h298_exact_s3_get"]["bit_for_bit_match_all_arms"] is True
    assert reg["h298_exact_s3_get"]["recomputed_median_lift_vs_r3175"] == -0.033528458894059865
    assert reg["h298_exact_s3_get"]["recomputed_median_ci95_vs_r3175"] == [
        -0.09813891237891575,
        0.0035028817498420945,
    ]


def test_evaluate_trajectory_and_mm_bradley_terry_elo() -> None:
    """Verify evaluate_trajectory_vs_r3175 computes exact lift, W/T/L, and MM Bradley-Terry Elo."""
    dev_fams = [f"fam_{i % 19}" for i in range(91)]
    counts = a301.make_family_bootstrap_counts(19, 200, 30100)
    base = np.ones(91, dtype=np.float64)
    progs = [f"p_{i}" for i in range(91)]

    # Identical candidate -> 0 lift, 0 wins, 91 ties, 0 losses, Elo = 1500.0
    res_eq = a301.evaluate_trajectory_vs_r3175(
        wave="unit",
        label="eq",
        seed=1,
        n_updates=2,
        lr=1e-4,
        cand_per_context=base,
        cand_programs=progs,
        r3175_per_context=base,
        r3175_programs=progs,
        dev_families=dev_fams,
        counts=counts,
    )
    assert res_eq["lift_vs_r3175"] == 0.0
    assert res_eq["wtl_vs_r3175"] == {"wins": 0, "ties": 91, "losses": 0}
    assert res_eq["changed_contexts"] == 0
    assert res_eq["elo_vs_r3175"]["point"] == pytest.approx(1500.0, abs=1e-12)

    # Strictly losing candidate -> negative lift, Elo < 1500.0
    cand_lose = base.copy()
    cand_lose[:8] -= 0.05
    cand_lose[8:10] += 0.01
    res_lose = a301.evaluate_trajectory_vs_r3175(
        wave="unit",
        label="lose",
        seed=2,
        n_updates=8,
        lr=1e-4,
        cand_per_context=cand_lose,
        cand_programs=None,
        r3175_per_context=base,
        r3175_programs=progs,
        dev_families=dev_fams,
        counts=counts,
    )
    assert res_lose["lift_vs_r3175"] < 0.0
    assert res_lose["wtl_vs_r3175"] == {"wins": 2, "ties": 81, "losses": 8}
    assert res_lose["elo_vs_r3175"]["point"] < 1500.0


def test_fit_drift_slope_bootstrap_known_linear_trend() -> None:
    """Verify OLS slope and bootstrap CI95 on a synthetic negative drift trajectory."""
    x = np.array([2.0, 8.0, 16.0, 64.0, 256.0], dtype=np.float64)
    true_slope = -0.0005
    true_intercept = -0.01
    y_pt = true_intercept + true_slope * x
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 0.0002, size=(len(x), 1000))
    y_boot = y_pt[:, None] + noise
    res = a301.fit_drift_slope_bootstrap(x, y_pt, y_boot)
    assert res["slope_per_unit"] == pytest.approx(true_slope, abs=1e-15)
    assert res["intercept"] == pytest.approx(true_intercept, abs=1e-15)
    assert res["slope_strictly_negative_95"] is True
    assert res["slope_ci95"][1] < 0.0


def test_analysis_only_guard_blocks_torch_and_checkpoints() -> None:
    """Verify the audit hook rejects torch imports and .pt / fresh_cohort file opens."""
    code_torch = (
        "import sys; sys.path.insert(0, %r); import run_h301; "
        "run_h301.install_analysis_only_guard(); import torch"
    ) % str(HERE)
    p1 = subprocess.run([sys.executable, "-c", code_torch], capture_output=True, text=True)
    assert p1.returncode != 0
    assert "Forbidden neural framework import" in p1.stderr

    code_pt = (
        "import sys; sys.path.insert(0, %r); import run_h301; "
        "run_h301.install_analysis_only_guard(); open('/tmp/round-3175.pt', 'rb')"
    ) % str(HERE)
    p2 = subprocess.run([sys.executable, "-c", code_pt], capture_output=True, text=True)
    assert p2.returncode != 0
    assert "Forbidden file open" in p2.stderr


def test_metrics_json_schema_verdict_and_size() -> None:
    """Verify metrics.json is < 50 KB, schema-complete, and reproduces SELECTION_NOISE_PLATEAU."""
    m_path = HERE / "metrics.json"
    assert m_path.exists()
    assert m_path.stat().st_size < 50 * 1024

    m = json.loads(m_path.read_text())
    assert m["wave_id"] == "H301"
    assert m["slug"] == "h301_paired_control_drift_meta"
    assert m["mode"] == "ANALYSIS_ONLY"
    assert m["neural_updates"] == 0
    assert m["neural_forwards"] == 0
    assert m["dope_untouched"] is True
    assert m["verdict"] == "SELECTION_NOISE_PLATEAU"

    # Check primary Criterion 1: median control lift across wave/horizon groups < 0 with CI95 excluding 0
    all_waves = m["median_over_waves_control_summary"]["all_waves_2_to_256_updates"]
    assert all_waves["median_lift_vs_r3175"] == pytest.approx(-0.033462432115435314, abs=1e-12)
    assert all_waves["ci95_family_bootstrap"][1] < 0.0
    assert all_waves["ci95_excludes_zero_strictly_negative"] is True

    # Check Drift Slope (a): both linear per update and per log2 doubling have 95% CIs strictly < 0
    slope_lin = m["drift_slope_vs_n_updates"]["wave_medians_linear_per_update_2_to_256"]
    assert slope_lin["slope_per_unit"] < 0.0
    assert slope_lin["slope_ci95"][1] < 0.0
    assert slope_lin["slope_strictly_negative_95"] is True

    slope_log = m["drift_slope_vs_n_updates"]["wave_medians_per_doubling_log2_updates_2_to_256"]
    assert slope_log["slope_per_unit"] < 0.0
    assert slope_log["slope_ci95"][1] < 0.0
    assert slope_log["slope_strictly_negative_95"] is True

    # Check Winner's Curse (b) & Criterion 2: r3175 - q95 < 2 * SE_fam under both marginal and paired SE
    wc = m["winners_curse_analysis"]["post_r3175_2update_distribution"]
    assert wc["n_rounds"] == 1499
    assert wc["r3175_minus_q95"] == pytest.approx(0.003518451949980239, abs=1e-12)
    assert wc["r3175_minus_q95_in_marginal_family_se"] < 2.0
    assert wc["r3175_minus_q95_in_paired_family_se"] < 2.0
    assert wc["exceeds_q95_by_lt_2_marginal_se"] is True
    assert wc["exceeds_q95_by_lt_2_paired_se"] is True
    assert wc["expected_max_gaussian_order_stat"] > m["r3175_baseline_mean"]

    # Check all 19 trajectories are present and every single one has lift_vs_r3175 < 0 and Elo < 1500
    assert len(m["trajectories"]) == 19
    for t in m["trajectories"]:
        assert t["lift_vs_r3175"] < 0.0
        assert t["elo_vs_r3175"]["point"] < 1500.0
        assert "auc" not in json.dumps(t).lower()
