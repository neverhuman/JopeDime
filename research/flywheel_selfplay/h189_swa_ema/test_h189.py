"""Synthetic/data-free H189 design and selector tests."""

import json
import hashlib

import numpy as np
import pytest
import torch

import analysis as stat
import treatment


def _rows():
    return [
        {"family_id": f"F{i:03d}", "context_sha256": f"c{i:04d}"} for i in range(205)
    ]


def test_swa_is_uniform_mean_of_fixed_adapter_only_snapshots():
    snapshots = {
        step: {"adapter.weight": np.full((2,), step, np.float32)}
        for step in stat.SNAPSHOT_STEPS
    }
    result = treatment.uniform_swa(snapshots)
    assert result["adapter.weight"].dtype == np.float32
    assert result["adapter.weight"].tolist() == [60.0, 60.0]


def test_state_array_preserves_scalar_adapter_tensors_as_zero_dim_arrays():
    result = treatment.state_array(np.float32(1.25))
    assert isinstance(result, np.ndarray)
    assert result.shape == ()
    assert result.dtype == np.float32
    assert result.item() == pytest.approx(1.25)
    assert torch.from_numpy(result).item() == pytest.approx(1.25)


def test_swa_rejects_missing_or_nonfinite_snapshot():
    snapshots = {
        step: {"adapter.weight": np.ones((2,), np.float32)}
        for step in stat.SNAPSHOT_STEPS[:-1]
    }
    with pytest.raises(ValueError, match="exactly 40"):
        treatment.uniform_swa(snapshots)
    snapshots[80] = {"adapter.weight": np.array([np.nan, 1], np.float32)}
    with pytest.raises(ValueError, match="finite"):
        treatment.uniform_swa(snapshots)


def test_ema_uses_all_80_post_update_adapter_states():
    states = [
        {"adapter.weight": np.array([float(i)], np.float32)} for i in range(1, 81)
    ]
    result = treatment.ema_over_trajectory(states, decay=0.95)
    expected = 1.0
    for i in range(2, 81):
        expected = 0.95 * expected + 0.05 * i
    assert result["adapter.weight"][0] == pytest.approx(expected, abs=2e-5)


def test_fold_partitions_are_exact_and_disjoint():
    treatment.validate_partitions(range(123), range(123, 164), range(164, 205))
    with pytest.raises(ValueError, match="disjoint"):
        treatment.validate_partitions(range(123), range(122, 163), range(164, 205))


def test_pooled_b_selector_uses_two_arm_holm_and_strict_ties():
    rows = _rows()
    baseline = np.zeros(205)
    # Exact ties never clear the strict +0.001 threshold.
    tie = np.zeros((3, 205))
    decisions = stat.pooled_b_select(
        rows, baseline, {"SWA": tie, "EMA": tie}, seed=stat.BOOT_SEED
    )
    assert set(decisions) == {"SWA", "EMA"}
    assert all(
        d["holm_multiplicity"] == 2 and not d["admitted"] for d in decisions.values()
    )
    # A positive shift is detected, with three identical synthetic seeds.
    positive = np.full((3, 205), 0.02)
    decisions = stat.pooled_b_select(
        rows, baseline, {"SWA": positive, "EMA": tie}, seed=stat.BOOT_SEED
    )
    assert decisions["SWA"]["admitted"]
    assert not decisions["EMA"]["admitted"]


def test_selector_refuses_missing_arm_and_wrong_cohort():
    rows = _rows()
    with pytest.raises(ValueError, match="exactly"):
        stat.pooled_b_select(rows, np.zeros(205), {"SWA": np.zeros((3, 205))})
    with pytest.raises(ValueError, match="205"):
        stat.pooled_b_select(
            rows[:-1],
            np.zeros(204),
            {"SWA": np.zeros((3, 204)), "EMA": np.zeros((3, 204))},
        )


def test_a_stage_requires_fsynced_h189_b_lock(tmp_path):
    guard = stat.StageOrderGuard()
    with pytest.raises(RuntimeError, match="before"):
        guard.require_b_locked()
    lock = tmp_path / "pooled_b_selection_lock.json"
    stat.write_json_fsync(
        lock,
        {"wave": "H189", "pooled_b_families": 205, "decisions": {"SWA": {}, "EMA": {}}},
    )
    guard.mark_b_locked(lock)
    assert guard.require_b_locked()["wave"] == "H189"


def test_a_confirmation_requires_three_positive_seeds_and_positive_lcb():
    rows = _rows()
    baseline = np.zeros(205)
    positive = np.full((3, 205), 0.02)
    result = stat.pooled_a_confirm(rows, baseline, {"SWA": positive, "EMA": positive})
    assert result["SWA"]["advance_a_gate"]
    assert result["SWA"]["positive_seed_count"] == 3
    harmed = np.full((3, 205), -0.01)
    result = stat.pooled_a_confirm(rows, baseline, {"SWA": harmed, "EMA": positive})
    assert not result["SWA"]["advance_a_gate"]


def test_h178_receipt_is_retained_as_four_arm_provenance_not_m2_claim():
    receipt = json.loads(
        (
            stat.Path(__file__).parent.parent
            / "h178_pooled_calibration"
            / "h178_receipt.json"
        ).read_text()
    )
    assert receipt["status"] == "COMPLETE"
    assert receipt["replayed_four_arm_holm_m"] == 4
    assert stat.ARMS == ("SWA", "EMA")


def test_family_elo_identical_panels_are_r882_1500():
    rows = _rows()
    baseline = np.stack([np.linspace(-1.0, 1.0, 205)] * 3)
    ratings = stat.family_elo_summary(
        rows,
        baseline,
        {"SWA": baseline.copy(), "EMA": baseline.copy()},
        draws=20,
    )
    assert ratings["R882"]["elo"] == pytest.approx(1500.0)
    assert ratings["SWA"]["elo_delta_vs_r882"] == pytest.approx(0.0)
    assert ratings["EMA"]["elo_delta_ci95"] == pytest.approx([0.0, 0.0])


def test_launch_requires_owner_binding_and_exact_tree_self_review(tmp_path):
    tree = "a" * 40
    receipt = {
        "conclusion": "PASS",
        "reviewer_kind": "same-agent substantive review",
        "reviewed_tree_sha256": tree,
    }
    receipt_path = tmp_path / "review.json"
    receipt_bytes = (json.dumps(receipt, sort_keys=True) + "\n").encode()
    receipt_path.write_bytes(receipt_bytes)
    launch = {
        "manifest_sha256": "m",
        "merged_source_verified": True,
        "source_commit": "merge-commit",
        "prereg_pr": 1900,
        "prereg_sha256": "p",
        "fold_manifest_sha256": "f",
        "owner_exception": "/home/ubuntu/nudges/hwave_t0805_h189_go.md",
        "owner_ci_exception": True,
        "same_agent_review": {
            **receipt,
            "receipt_path": str(receipt_path),
            "review_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        },
    }
    treatment.validate_launch(launch, "merge-commit", "m", "p", "f", source_tree=tree)
    with pytest.raises(RuntimeError, match="exact-tree"):
        treatment.validate_launch(
            launch, "merge-commit", "m", "p", "f", source_tree="b" * 40
        )
