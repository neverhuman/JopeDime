"""Fail-closed Phase 3 gate primitives for fused GP self-play.

Stage into research/flywheel_selfplay and call from the round boundary. This
module intentionally does not fetch data, invoke GP search, or access sealed
partitions; the owner runner supplies those authorized results.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence


class GateError(ValueError):
    """Raised when a round lacks proof required for qualification."""


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


def deterministic_verification_sample(
    winners: Sequence[Mapping[str, Any]], *, seed: str, fraction: float = 0.10
) -> list[Mapping[str, Any]]:
    """Select exactly ceil(fraction*n) distinct winners, stable across machines."""
    if not 0.0 < fraction <= 1.0:
        raise GateError("verification fraction must be in (0, 1]")
    keyed: list[tuple[bytes, Mapping[str, Any]]] = []
    seen: set[str] = set()
    for row in winners:
        winner_id = row.get("winner_id")
        if not isinstance(winner_id, str) or not winner_id or winner_id in seen:
            raise GateError("every winner needs a unique nonempty winner_id")
        seen.add(winner_id)
        digest = hashlib.sha256(f"phase3-v1\0{seed}\0{winner_id}".encode()).digest()
        keyed.append((digest, row))
    if not keyed:
        raise GateError("cannot verify an empty winner set")
    count = max(1, math.ceil(len(keyed) * fraction))
    return [row for _, row in sorted(keyed, key=lambda pair: pair[0])[:count]]


def verify_sample_results(
    sample: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]], *, fraction: float = 0.10
) -> list[dict[str, Any]]:
    """Bind fresh GP search results to the exact selected winners."""
    expected = {row["winner_id"] for row in sample}
    by_id: dict[str, Mapping[str, Any]] = {}
    for result in results:
        winner_id = result.get("fused_winner_id")
        if not isinstance(winner_id, str) or winner_id in by_id:
            raise GateError("verification results need unique fused_winner_id values")
        by_id[winner_id] = result
    if set(by_id) != expected:
        raise GateError("verification membership differs from deterministic sample")
    verified: list[dict[str, Any]] = []
    for winner_id in sorted(expected):
        result = by_id[winner_id]
        if result.get("search_completed") is not True:
            raise GateError(f"GP search did not complete for {winner_id}")
        if not isinstance(result.get("gp_best_winner_id"), str):
            raise GateError(f"GP best winner missing for {winner_id}")
        if result.get("partition") in {"sealed", "meta_test"}:
            raise GateError("sealed partition label in verification result")
        verified.append(dict(result))
    minimum = max(1, math.ceil(len(expected) * fraction))
    if len(verified) < minimum:
        raise GateError("GP verification sample is below the configured minimum")
    return verified


def misses_to_training_pairs(results: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Turn each verified GP win over the fused choice into a labeled pair."""
    pairs: list[dict[str, Any]] = []
    for row in results:
        fused_id, gp_id = row.get("fused_winner_id"), row.get("gp_best_winner_id")
        if not isinstance(fused_id, str) or not isinstance(gp_id, str):
            raise GateError("pair generation requires both winner identities")
        if row.get("search_completed") is not True:
            raise GateError("cannot train from an unverified candidate")
        if fused_id == gp_id:
            continue
        fused_loss, gp_loss = row.get("fused_loss"), row.get("gp_best_loss")
        if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in (fused_loss, gp_loss)):
            raise GateError("miss pair needs finite losses from the same verification evaluation")
        if gp_loss >= fused_loss:
            raise GateError("reported GP miss does not beat fused candidate on the shared objective")
        dataset_id = row.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise GateError("pair needs real-S3 dataset identity")
        pair = {
            "dataset_id": dataset_id,
            "preferred_winner_id": gp_id,
            "rejected_winner_id": fused_id,
            "preferred_loss": float(gp_loss),
            "rejected_loss": float(fused_loss),
            "source_round": row.get("round"),
            "source_verification_id": row.get("verification_id"),
        }
        pair["pair_id"] = hashlib.sha256(repr(sorted(pair.items())).encode()).hexdigest()
        pairs.append(pair)
    return pairs


def validate_grouped_s3_manifest(
    dataset_ids: Iterable[str], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Require complete authoritative family/fold coverage for every S3 row."""
    if manifest.get("provenance") != "authoritative_s3_family_roster":
        raise GateError("family roster provenance is not authoritative")
    if not _sha256(manifest.get("family_roster_sha256")) or not _sha256(manifest.get("fold_map_sha256")):
        raise GateError("family roster and fold map must be content-addressed")
    if manifest.get("sealed_partition_included") is not False:
        raise GateError("sealed partition must be explicitly excluded")
    rows = manifest.get("assignments")
    if not isinstance(rows, list):
        raise GateError("fold manifest assignments missing")
    assignments: dict[str, tuple[str, int]] = {}
    family_folds: dict[str, int] = {}
    for row in rows:
        dataset_id = row.get("dataset_id")
        family = row.get("family_id")
        fold = row.get("outer_fold")
        if row.get("origin") != "real" or row.get("partition") in {"sealed", "meta_test"}:
            raise GateError("non-real or sealed row in grouped S3 manifest")
        if not isinstance(dataset_id, str) or not dataset_id or not isinstance(family, str) or not family:
            raise GateError("dataset and authoritative family identifiers are required")
        if not isinstance(fold, int) or fold not in range(5):
            raise GateError("outer_fold must be an integer in [0,4]")
        if dataset_id in assignments:
            raise GateError(f"duplicate dataset assignment: {dataset_id}")
        if family in family_folds and family_folds[family] != fold:
            raise GateError(f"family leaks across folds: {family}")
        assignments[dataset_id] = (family, fold)
        family_folds[family] = fold
    requested = set(dataset_ids)
    if not requested or not requested.issubset(assignments):
        raise GateError("grouped S3 fold map does not cover all round datasets")
    return {
        "dataset_count": len(requested),
        "family_count": len({assignments[dataset_id][0] for dataset_id in requested}),
        "family_roster_sha256": manifest["family_roster_sha256"],
        "fold_map_sha256": manifest["fold_map_sha256"],
        "assignments": {dataset_id: {"family_id": family, "outer_fold": fold}
                        for dataset_id, (family, fold) in assignments.items()},
    }


def compare_round(current: Mapping[str, Any], previous: Mapping[str, Any]) -> dict[str, Any]:
    """Require reported outcomes and enforce keep/rollback rule."""
    fields = ("match_rate", "holdout_r2_vs_gp", "wins_over_chained", "rust_tok_s")
    for label, metrics in (("current", current), ("previous", previous)):
        for name in fields:
            value = metrics.get(name)
            if not isinstance(value, (float, int)) or not math.isfinite(value):
                raise GateError(f"{label} comparison lacks finite {name}")
    no_speed_regression = current["rust_tok_s"] >= previous["rust_tok_s"]
    quality_improved = (
        current["match_rate"] > previous["match_rate"]
        or current["holdout_r2_vs_gp"] > previous["holdout_r2_vs_gp"]
    )
    return {
        "match_rate": current["match_rate"],
        "holdout_r2_vs_gp": current["holdout_r2_vs_gp"],
        "wins_over_chained": current["wins_over_chained"],
        "rust_tok_s": current["rust_tok_s"],
        "no_tok_s_regression": no_speed_regression,
        "quality_improved": quality_improved,
        "keep_round": bool(no_speed_regression and quality_improved),
        "action": "keep" if no_speed_regression and quality_improved else "rollback",
    }


def qualify_round(
    *,
    winners: Sequence[Mapping[str, Any]],
    verification_results: Sequence[Mapping[str, Any]],
    s3_fold_manifest: Mapping[str, Any],
    training_receipt: Mapping[str, Any],
    dope_freeze_receipt: Mapping[str, Any],
    current_metrics: Mapping[str, Any],
    previous_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a round receipt; any missing gate raises and cannot be counted."""
    sample = deterministic_verification_sample(winners, seed=str(current_metrics.get("round_seed", "")))
    verified = verify_sample_results(sample, verification_results)
    pairs = misses_to_training_pairs(verified)
    pair_ids = {pair["pair_id"] for pair in pairs}
    consumed_pair_ids = set(training_receipt.get("consumed_pair_ids", []))
    if consumed_pair_ids != pair_ids:
        raise GateError("fused-head training receipt does not consume exactly the verified miss pairs")
    if training_receipt.get("fused_head_retrained") is not True:
        raise GateError("fused head was not retrained after miss-pair ingestion")
    if dope_freeze_receipt.get("before_sha256") != dope_freeze_receipt.get("after_sha256"):
        raise GateError("DOPE checksum changed during the round")
    if not _sha256(dope_freeze_receipt.get("before_sha256")):
        raise GateError("DOPE freeze receipt lacks a valid checksum")
    if dope_freeze_receipt.get("dope_trainable_params") != 0:
        raise GateError("DOPE parameters were trainable")
    dataset_ids: list[str] = []
    for row in winners:
        dataset_id = row.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise GateError("each fused winner needs a real-S3 dataset_id")
        if row.get("origin") != "real" or row.get("partition") in {"sealed", "meta_test"}:
            raise GateError("winner input is not an authorized real-S3 development row")
        dataset_ids.append(dataset_id)
    if len(dataset_ids) != len(set(dataset_ids)):
        raise GateError("each Phase 3 round must have one selected winner per dataset")
    grouped = validate_grouped_s3_manifest(
        dataset_ids, s3_fold_manifest
    )
    if training_receipt.get("fold_map_sha256") != grouped["fold_map_sha256"]:
        raise GateError("training used a different grouped fold map")
    fold_runs = training_receipt.get("outer_fold_runs")
    if not isinstance(fold_runs, list) or len(fold_runs) != 5:
        raise GateError("training receipt must include all five outer folds")
    assignments = grouped["assignments"]
    all_families = {item["family_id"] for item in assignments.values()}
    seen_folds: set[int] = set()
    for fold_run in fold_runs:
        fold = fold_run.get("outer_fold")
        if not isinstance(fold, int) or fold not in range(5) or fold in seen_folds:
            raise GateError("outer fold runs must contain each fold 0 through 4 once")
        seen_folds.add(fold)
        expected_eval = {fam for fam, outer in
                         ((item["family_id"], item["outer_fold"]) for item in assignments.values())
                         if outer == fold}
        train_families = set(fold_run.get("train_family_ids", []))
        eval_families = set(fold_run.get("eval_family_ids", []))
        if eval_families != expected_eval or train_families != all_families - expected_eval:
            raise GateError(f"outer fold {fold} family partition differs from frozen fold map")
        train_ids = set(fold_run.get("train_dataset_ids", []))
        eval_ids = set(fold_run.get("eval_dataset_ids", []))
        if not train_ids.isdisjoint(eval_ids):
            raise GateError(f"outer fold {fold} has dataset leakage")
        for dataset_id in train_ids:
            if dataset_id not in assignments or assignments[dataset_id]["outer_fold"] == fold:
                raise GateError(f"outer fold {fold} trains on an evaluation-family dataset")
        for dataset_id in eval_ids:
            if dataset_id not in assignments or assignments[dataset_id]["outer_fold"] != fold:
                raise GateError(f"outer fold {fold} evaluates outside its assigned family fold")
    if seen_folds != set(range(5)):
        raise GateError("one or more outer grouped folds are missing")
    comparison = compare_round(current_metrics, previous_metrics)
    if not comparison["keep_round"]:
        raise GateError("round failed comparison admission and must roll back")
    return {
        "qualified": True,
        "verification": {"sampled": len(sample), "verified": len(verified), "misses": len(pairs)},
        "miss_training_pairs": pairs,
        "training_receipt": dict(training_receipt),
        "dope_freeze": dict(dope_freeze_receipt),
        "grouped_s3": {key: value for key, value in grouped.items() if key != "assignments"},
        "comparison": comparison,
    }
