"""Unit and integrity tests for H291 (h291_harm_veto_abstention)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import analysis_h291 as ana
import harm_veto as hv

CAPSULE_DIR = Path(__file__).resolve().parent
EXPORT_DIR = Path("/home/ubuntu/jope-data/h291/export")


def test_step1_h287_regression_bit_exact() -> None:
    metrics = json.loads((CAPSULE_DIR / "metrics.json").read_text(encoding="utf-8"))
    reg = metrics["step1_h287_regression"]
    assert reg["bit_exact"] is True
    assert reg["seed"] == 287101
    assert reg["n_families"] == 205
    assert reg["n_contexts"] == 1219
    assert reg["utility_vector_max_abs_diff_vs_h287_panel"] == 0.0
    assert reg["gain_abs_diff"] == 0.0
    assert reg["harmed_families_gt_0p01"] == 31
    assert reg["raw_neg2_invalid_picks"] == 13
    assert reg["changed_contexts"] == 472


def test_fit_lock_integrity_and_zero_leakage() -> None:
    fit_lock_path = CAPSULE_DIR / "fit_lock.json"
    assert fit_lock_path.exists()
    fit_lock = json.loads(fit_lock_path.read_text(encoding="utf-8"))
    assert fit_lock["part_b_pools_read_during_part_a"] == 0
    assert fit_lock["fresh_pools_read"] == 0
    assert fit_lock["feature_dim"] == 644

    check_copy = dict(fit_lock)
    expected_sha = check_copy.pop("fit_lock_content_sha256")
    actual_sha = hashlib.sha256(
        json.dumps(check_copy, indent=2, sort_keys=True).encode("utf-8")
    ).hexdigest()
    assert actual_sha == expected_sha == "cd04dec452bacd84ea68a1c08e53a214ba2452b09c982652b5aeaf7fff2498cf"

    for entry in fit_lock["exported_harm_veto_models"]:
        mpath = EXPORT_DIR / entry["file"]
        assert mpath.exists()
        assert hv.sha256_file(mpath) == entry["sha256"]


def test_metrics_json_schema_and_size_cap() -> None:
    mpath = CAPSULE_DIR / "metrics.json"
    assert mpath.exists()
    assert mpath.stat().st_size < 50_000

    m = json.loads(mpath.read_text(encoding="utf-8"))
    assert m["hypothesis_id"] == "H291"
    assert m["slug"] == "h291_harm_veto_abstention"
    assert m["verdict"] == "HARMVETO_NOT_ADVANCED"
    assert m["neural_forwards"] == 0
    assert m["neural_updates"] == 0
    assert m["part_b_pools_read_during_part_a"] == 0
    assert m["part_b_semi_fresh_h286_v2_even"]["reused"] is True
    assert m["part_b_semi_fresh_h286_v2_even"]["fresh_claim"] is False
    assert m["part_b_semi_fresh_h286_v2_even"]["r882_fit_bit_exact_with_h286"] is True


def test_sign_test_and_mm_bt_elo_properties() -> None:
    # Ties never count as wins
    assert ana.binom_two_sided_pvalue(10, 10) == 1.0
    assert ana.binom_one_sided_greater_pvalue(10, 2) < 0.05

    fam_mat = np.array(
        [
            [1.0, 1.2, 1.0 + 5e-5],
            [1.0, 1.3, 1.0 - 5e-5],
            [1.0, 1.1, 1.0],
            [1.0, 0.9, 1.0],
        ],
        dtype=np.float64,
    )
    elos = ana.mm_bradley_terry_elo(fam_mat, anchor_idx=0, anchor_elo=1500.0, tie_band=1e-4)
    assert abs(float(elos[0]) - 1500.0) < 1e-9
    # Arm 2 is tied within 1e-4 on all 4 families vs arm 0, and loses 3/4 vs arm 1
    assert float(elos[1]) > float(elos[0])
