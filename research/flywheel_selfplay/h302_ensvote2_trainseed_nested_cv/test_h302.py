"""Unit and regression tests for H302 (h302_ensvote2_trainseed_nested_cv)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis_h302 as a302  # noqa: E402
import run_h302  # noqa: E402


def test_step1_regression_receipt_and_outer_fold_split() -> None:
    """Verify Step 1 H287 64-context regression receipt and pre-declared outer fold sizes."""
    reg_path = HERE / "regression_receipt.json"
    assert reg_path.exists()
    reg = json.loads(reg_path.read_text())
    assert reg["status"] == "PASS"
    assert reg["h287_export_commit"] == "213b30ad52e1ce48927bba960ed609bd98ab4dfe"
    assert reg["selected_json_sha256"] == "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"
    assert reg["bit_exact_all_deterministic_and_per_seed_arms"] is True
    assert reg["ens_vote2_64ctx_changed_count"] == 19
    assert reg["ens_vote2_64ctx_picks_sha256"] == "8ef2837ae43e22721b7f7ffdf30371c29ba18e1ad3d9f0f788462d7c350d6d01"

    fsum = reg["h302_outer_fold_summary"]
    assert sum(int(fsum[str(k)]["n_test_families"]) for k in range(4)) == 205
    assert sum(int(fsum[str(k)]["n_test_contexts"]) for k in range(4)) == 1219


def test_outer_fold_of_family_deterministic() -> None:
    """Verify SHA256('H302-outer-v1' NUL family) last byte mod 4 is deterministic and in 0..3."""
    for i in range(50):
        f = f"fam_{i}"
        k1 = a302.outer_fold_of_family(f)
        k2 = a302.outer_fold_of_family(f)
        assert k1 == k2
        assert 0 <= k1 < 4


def test_summarize_and_criteria_synthetic() -> None:
    """Verify summarize_three_seeds and evaluate_h302_criteria on a synthetic winning panel."""
    n_fam = 40
    fams = [f"f_{i}" for i in range(n_fam)]
    u = np.ones((3, n_fam, 2), dtype=np.float64)
    # Give all 3 seeds positive gain on 30 families, tie on 8, small loss on 2
    u[:, :30, 1] += 0.05
    u[:, 38:, 1] -= 0.02
    inv = np.zeros((3, n_fam, 2), dtype=bool)
    inv[:, :3, 0] = True
    inv[:, :2, 1] = True
    picks = [[1 if i < 30 else 0 for i in range(n_fam)] for _ in range(3)]

    s = a302.summarize_three_seeds(u, inv, fams, picks, boot_seed=302150, n_draws=500)
    c = a302.evaluate_h302_criteria(s, inv, n_fam)
    assert c["condition_1_median_gain_gt_0_001"] is True
    assert c["condition_2_lcb95_gt_0_on_3of3_seeds"] is True
    assert c["condition_3_harmed_share_le_35pct_each_seed"] is True
    assert c["condition_4_invalid_noninferior_each_seed"] is True
    assert c["verdict"] == "INNERCV_CONFIRMED_TRAINING_SEEDS"
    assert s["median_elo_across_seeds"] > 1500.0


def test_audit_guard_blocks_fresh_and_checkpoints() -> None:
    """Verify install_h302_audit_guard blocks opening fresh_cohort, dl-test, and .pt files."""
    for bad_path in ("/tmp/fresh_cohort_v5.json", "/tmp/round-3175.pt", "/tmp/dl-test.json"):
        code = (
            "import sys; sys.path.insert(0, %r); import run_h302; "
            "run_h302.install_h302_audit_guard(); open(%r, 'rb')"
        ) % (str(HERE), bad_path)
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert p.returncode != 0
        assert "Forbidden path open in H302" in p.stderr


def test_metrics_json_schema_and_size() -> None:
    """Verify metrics.json exists, is < 50 KB, and contains complete H302 fields."""
    m_path = HERE / "metrics.json"
    assert m_path.exists()
    assert m_path.stat().st_size < 50 * 1024

    m = json.loads(m_path.read_text())
    assert m["wave_id"] == "H302"
    assert m["slug"] == "h302_ensvote2_trainseed_nested_cv"
    assert m["verdict"] in ("INNERCV_CONFIRMED_TRAINING_SEEDS", "NOT_CONFIRMED_TRAINING_SEEDS")
    assert m["optimizer_updates"] == 0
    assert m["gpu_used"] is False
    assert m["fresh_pools_read"] == 0
    assert m["outer_test_rows_read_before_fold_lock"] == 0
    assert m["dope_untouched"] is True
    assert m["root_training_seeds"] == [30001, 30002, 30003]
    assert m["shared_bootstrap_seed"] == 302150
    assert len(m["outer_fold_receipts"]) == 4
    assert "auc" not in m_path.read_text().lower()
