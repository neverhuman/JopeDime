from __future__ import annotations

import hashlib

import pytest

from phase3_gatekeeper import (
    GateError,
    compare_round,
    deterministic_verification_sample,
    misses_to_training_pairs,
    qualify_round,
    verify_sample_results,
)
from phase3_keyed_subset import build_keyed_subset_manifest
from phase3_round_runner import qualify_phase3_bundle


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_keyed_subset_uses_only_recorded_real_families_and_sealed_exclusions():
    rows = []
    for task, count in (("regression", 10), ("binary", 5)):
        for index in range(count):
            family = f"{task}-family-{index // 2}"
            rows.append({
                "id": f"ds-{task}-{index}",
                "family_key": family,
                "task": task,
                "origin": "real",
                "has_meta_test": False,
                "manifest_sha256": _sha(f"manifest-{task}-{index}"),
            })
    rows[0]["has_meta_test"] = True
    rows.append({"id": "unkeyed", "family_key": None, "task": "regression", "origin": "real", "has_meta_test": False})
    rows.append({"id": "synthetic", "family_key": "fake", "task": "regression", "origin": "synthetic", "has_meta_test": False})
    result = build_keyed_subset_manifest(
        rows, inventory_sha256=_sha("inventory"),
        sealed_excluded_family_ids={"binary-family-0"}, expected_counts=None,
    )
    assert result["sealed_excluded_applied"] is True
    assignments = result["assignments"]
    assert "ds-regression-1" not in {row["dataset_id"] for row in assignments}
    assert not any(row["family_id"] == "binary-family-0" for row in assignments)
    assert all(row["origin"] == "real" and row["family_id"] for row in assignments)
    folds_by_family = {}
    for row in assignments:
        folds_by_family.setdefault(row["family_id"], set()).add(row["outer_fold"])
    assert all(len(folds) == 1 for folds in folds_by_family.values())
    assert {row["outer_fold"] for row in assignments} == set(range(5))


def test_keyed_subset_does_not_claim_missing_sealed_roster_was_applied():
    result = build_keyed_subset_manifest(
        [{"id": "ds", "family_key": "family", "task": "regression", "origin": "real", "has_meta_test": False}],
        inventory_sha256=_sha("inventory"), expected_counts=None,
    )
    assert result["sealed_excluded_applied"] is False


def test_verification_membership_is_stable_and_exact():
    winners = [{"winner_id": f"w{i}"} for i in range(37)]
    sample_a = deterministic_verification_sample(winners, seed="s", fraction=0.1)
    sample_b = deterministic_verification_sample(winners, seed="s", fraction=0.1)
    assert [row["winner_id"] for row in sample_a] == [row["winner_id"] for row in sample_b]
    assert len(sample_a) == 4
    result = [{
        "fused_winner_id": row["winner_id"],
        "gp_best_winner_id": row["winner_id"],
        "search_completed": True,
        "partition": "development",
    } for row in sample_a]
    assert len(verify_sample_results(sample_a, result)) == 4
    with pytest.raises(GateError, match="membership"):
        verify_sample_results(sample_a, result[:-1])


def test_phase3_runner_rejects_unkeyed_scope():
    with pytest.raises(GateError, match="scope must be keyed_subset"):
        qualify_phase3_bundle({"scope": "DL-Train"})


def test_misses_become_pairs_and_comparison_requires_quality_without_speed_regression():
    rows = [{
        "dataset_id": "ds1", "round": 3, "verification_id": "v1",
        "fused_winner_id": "fused", "gp_best_winner_id": "gp",
        "fused_loss": 2.0, "gp_best_loss": 1.0, "search_completed": True,
    }]
    pairs = misses_to_training_pairs(rows)
    assert len(pairs) == 1
    assert pairs[0]["preferred_winner_id"] == "gp"
    assert len(pairs[0]["pair_id"]) == 64
    baseline = {"match_rate": 0.9, "holdout_r2_vs_gp": 0.2, "wins_over_chained": 4, "rust_tok_s": 100.0}
    candidate = {"match_rate": 0.91, "holdout_r2_vs_gp": 0.2, "wins_over_chained": 5, "rust_tok_s": 100.0}
    assert compare_round(candidate, baseline)["action"] == "keep"
    candidate["rust_tok_s"] = 99.0
    assert compare_round(candidate, baseline)["action"] == "rollback"


def test_grouped_s3_and_dope_receipts_are_required_for_qualification():
    family_rows = []
    for fold in range(5):
        family_rows.append({
            "dataset_id": f"ds{fold}", "family_id": f"fam{fold}",
            "outer_fold": fold, "origin": "real", "partition": "development",
        })
    manifest = {
        "provenance": "authoritative_s3_family_roster",
        "family_roster_sha256": _sha("families"),
        "fold_map_sha256": _sha("folds"),
        "sealed_partition_included": False,
        "assignments": family_rows,
    }
    winners = [{"winner_id": f"w{fold}", "dataset_id": f"ds{fold}", "origin": "real", "partition": "development"} for fold in range(5)]
    sample = deterministic_verification_sample(winners, seed="round", fraction=0.2)
    verification = [{
        "fused_winner_id": row["winner_id"], "gp_best_winner_id": row["winner_id"],
        "search_completed": True, "partition": "development",
    } for row in sample]
    fold_runs = []
    all_families = {row["family_id"] for row in family_rows}
    for fold in range(5):
        held = {f"fam{fold}"}
        fold_runs.append({
            "outer_fold": fold,
            "train_family_ids": sorted(all_families - held),
            "eval_family_ids": sorted(held),
            "train_dataset_ids": [f"ds{i}" for i in range(5) if i != fold],
            "eval_dataset_ids": [f"ds{fold}"],
        })
    training = {
        "consumed_pair_ids": [], "fused_head_retrained": True,
        "fold_map_sha256": manifest["fold_map_sha256"], "outer_fold_runs": fold_runs,
    }
    dope = {"before_sha256": _sha("dope"), "after_sha256": _sha("dope"), "dope_trainable_params": 0}
    metrics = {"round_seed": "round", "match_rate": .9, "holdout_r2_vs_gp": .1, "wins_over_chained": 3, "rust_tok_s": 120.0}
    previous = {"match_rate": .8, "holdout_r2_vs_gp": .0, "wins_over_chained": 2, "rust_tok_s": 110.0}
    result = qualify_round(
        winners=winners, verification_results=verification,
        s3_fold_manifest=manifest, training_receipt=training,
        dope_freeze_receipt=dope, current_metrics=metrics, previous_metrics=previous,
    )
    assert result["qualified"] is True
    training["fold_map_sha256"] = _sha("wrong")
    with pytest.raises(GateError, match="different grouped fold map"):
        qualify_round(
            winners=winners, verification_results=verification,
            s3_fold_manifest=manifest, training_receipt=training,
            dope_freeze_receipt=dope, current_metrics=metrics, previous_metrics=previous,
        )
