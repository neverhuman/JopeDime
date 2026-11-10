"""Unit tests for H295 (h295_label_free_validity_guard)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import analysis_h295 as stat
import monitor
import runner
import valguard


def test_h287_ens_vote2_regression_bit_exact() -> None:
    receipt = valguard.verify_h287_ens_vote2_regression(
        oof_cache_path=runner.OOF_CACHE_PATH,
        h287_panel_path=runner.H287_PANEL_PATH,
        h287_metrics_path=runner.H287_METRICS_PATH,
    )
    assert receipt["bit_exact"] is True
    assert max(receipt["utility_vector_max_abs_diff_by_seed"]) == 0.0
    assert receipt["median_gain_round6"] == 0.059932
    assert receipt["per_seed_gain_round6"] == [0.053903, 0.059932, 0.062204]
    assert receipt["harmed_001_by_seed"] == [24, 17, 22]
    assert receipt["invalid_by_seed"] == [14, 13, 14]
    assert receipt["changed_by_seed"] == [296, 268, 265]


def test_isolation_audit_hook_blocks_forbidden_splits() -> None:
    monitor.install_isolation_audit_hook()
    for bad_path in (
        "/tmp/fake_dl_val_split.npz",
        "/tmp/fake_dl_test_split.npz",
        "/tmp/fake_holdout_split.npz",
        "/tmp/fake_sealed_split.npz",
        "/tmp/fake_fresh_cohort_v5.npz",
        "/tmp/fake_h294_pools.npz",
    ):
        with pytest.raises(PermissionError):
            open(bad_path, "r")


def test_candidate_feature_builder_and_voted_fallback() -> None:
    rng = np.random.default_rng(295)
    X_dummy = rng.standard_normal((10, 8, 568)).astype(np.float32)
    X_aug = valguard.build_candidate_guard_features(X_dummy)
    assert X_aug.shape == (10, 8, 652)
    assert np.allclose(X_aug[:, :, :568], X_dummy)

    # Test voted fallback: 2 members vote cand 5, 1 member votes cand 3
    u_pbag = np.zeros((1, 3, 2), dtype=np.int32)
    u_pbag[0, :, 0] = [5, 5, 3]
    u_pbag[0, :, 1] = [5, 5, 0]
    u_votes = np.full((1, 3, 2), 7, dtype=np.int32)
    u_raw = np.zeros((1, 3, 2, 8), dtype=np.float64)
    p_inv = np.full((2, 8), 0.01, dtype=np.float64)
    # On ctx 0: cand 5 has p_inv=0.08 (> tau=0.04), next voted cand 3 has p_inv=0.01 (<= tau) -> falls back to 3
    p_inv[0, 5] = 0.08
    p_inv[0, 3] = 0.01
    # On ctx 1: cand 5 has p_inv=0.08 (> tau=0.04), no other voted cand -> falls back to slot 0
    p_inv[1, 5] = 0.08
    p_inv[1, 0] = 0.01

    picks = valguard.apply_valguard_on_indices(
        np.array([0, 1], dtype=np.int64), 0, u_pbag, u_votes, u_raw, p_inv, tau=0.04
    )
    assert picks.tolist() == [3, 0]


def test_ties_never_count_as_wins_and_mm_bt_elo() -> None:
    # 4 families, 2 arms: diffs within 1e-4 tie band must count as Ties, never Wins
    u_panel = np.zeros((3, 4, 5), dtype=np.float64)
    u_panel[:, :, 2] = 5e-5  # within tie band 1e-4
    picks_panel = np.zeros((3, 4, 5), dtype=np.int32)
    gate_k = np.ones((4, 8), dtype=bool)
    fams = ["f0", "f1", "f2", "f3"]
    ev = stat.evaluate_panel(
        u_panel=u_panel,
        picks_panel=picks_panel,
        gate_k=gate_k,
        fams=fams,
        arm_names=valguard.ARM_NAMES,
        seeds=valguard.H295_SEEDS,
        n_boot=200,
    )
    wtl = ev["arms"]["ENS_VOTE2+VALGUARD"]["wtl_by_seed"][0]
    assert wtl == {"W": 0, "T": 4, "L": 0}
    assert abs(ev["arms"]["SLOT0"]["elo_median"] - 1500.0) < 1e-9
