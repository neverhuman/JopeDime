#!/usr/bin/env python3
"""Development-only audit of a fixed production-compatible raw model floor.

The candidate set is declared in source and contains only model heads already
supported by the production ``feat_math`` evaluator.  CV5 screens a candidate
against both frozen round-2 seed predictors.  Only the single frozen candidate
that beats both seeds by at least one percent advances to paired CV7, where it
must pass the same fold, bootstrap, and worst-fold guards as the Hyperion gate.

No test path or Hyperion sidecar is accepted.  The resulting evidence decides
whether these simple candidates belong in every future search finale; it is not
a benchmark or confirmation result.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import hyperion_train_audit as common


SCHEMA = "RawFloorDevelopmentAudit.v1"
MINIMUM_SCREEN_IMPROVEMENT = 0.01
CANDIDATES = {
    "ridge": "(pipe X (ridge lambda_frac=0.0))",
    "ridge_gcv": "(pipe X (ridge_gcv))",
    "elasticnet": "(pipe X (elasticnet))",
    "histgbm2": "(pipe X (histgbm2))",
    "stack": "(pipe X (stack))",
    "portfolio_core": "(pipe X (portfolio preset=core))",
}


def pooled_rmse(evidence: Sequence[common.FoldEvidence], arm: str) -> float:
    actual = [value for fold in evidence for value in fold.actual]
    predicted = [value for fold in evidence for value in fold.arms[arm]]
    return common.rmse(actual, predicted)


def evaluate_candidate(
    evaluator: Path,
    genome: str,
    raw: np.ndarray,
    targets: np.ndarray,
    groups: Sequence[str],
    folds: Sequence[common.Fold],
    timeout_seconds: int,
) -> list[common.FoldEvidence]:
    evidence: list[common.FoldEvidence] = []
    with tempfile.TemporaryDirectory(prefix="jope-raw-floor-audit-") as temporary:
        root = Path(temporary)
        for fold in folds:
            staged = root / f"fold-{fold.index}.csv"
            common.stage_csv(staged, raw, targets, fold.hold_rows)
            predictions = common.evaluate_predictions(
                evaluator, genome, staged, fold, timeout_seconds
            )
            hold = list(fold.hold_rows)
            evidence.append(
                common.FoldEvidence(
                    index=fold.index,
                    row_groups=[groups[index] for index in hold],
                    actual=[float(targets[index]) for index in hold],
                    raw=predictions,
                    arms={"candidate": predictions},
                )
            )
    return evidence


def evaluate_pair(
    evaluator: Path,
    raw_genome: str,
    candidate_genome: str,
    raw: np.ndarray,
    targets: np.ndarray,
    groups: Sequence[str],
    folds: Sequence[common.Fold],
    timeout_seconds: int,
) -> list[common.FoldEvidence]:
    evidence: list[common.FoldEvidence] = []
    with tempfile.TemporaryDirectory(prefix="jope-raw-floor-pair-") as temporary:
        root = Path(temporary)
        for fold in folds:
            staged = root / f"fold-{fold.index}.csv"
            common.stage_csv(staged, raw, targets, fold.hold_rows)
            raw_predictions = common.evaluate_predictions(
                evaluator, raw_genome, staged, fold, timeout_seconds
            )
            candidate_predictions = common.evaluate_predictions(
                evaluator, candidate_genome, staged, fold, timeout_seconds
            )
            hold = list(fold.hold_rows)
            evidence.append(
                common.FoldEvidence(
                    index=fold.index,
                    row_groups=[groups[index] for index in hold],
                    actual=[float(targets[index]) for index in hold],
                    raw=raw_predictions,
                    arms={"candidate": candidate_predictions},
                )
            )
    return evidence


def frozen_winner(round2_root: Path, dataset_id: str, seed: int) -> tuple[str, str]:
    path = common.find_round2_result(round2_root, dataset_id, seed)
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("dataset_hash") != dataset_id
        or result.get("master_seed") != seed
        or result.get("test_csv_read") is not False
    ):
        raise ValueError(f"round-2 result identity or train-only attestation failed: {path}")
    genome = result.get("winner_genome")
    if not isinstance(genome, str) or not genome:
        raise ValueError(f"round-2 result has no winner: {path}")
    return genome, common.sha256_file(path)


def historical_raw_cv5(
    hyperion_report: dict[str, Any], dataset_id: str, seed: int
) -> float | None:
    for result in hyperion_report.get("results", []):
        if result.get("dataset_id") == dataset_id and result.get("seed") == seed:
            candidates = result.get("cv5", [])
            if candidates:
                return float(candidates[0]["pooled_raw_rmse"])
    return None


def screen_choice(candidate_rmse: dict[str, float], raw_by_seed: dict[int, float]) -> dict[str, Any] | None:
    if len(raw_by_seed) != 2:
        return None
    ranked: list[tuple[float, str]] = []
    for name, value in candidate_rmse.items():
        minimum_log_improvement = min(math.log(raw / value) for raw in raw_by_seed.values())
        if minimum_log_improvement >= math.log1p(MINIMUM_SCREEN_IMPROVEMENT):
            ranked.append((minimum_log_improvement, name))
    if not ranked:
        return None
    minimum_log_improvement, name = min(ranked, key=lambda item: (-item[0], item[1]))
    return {
        "candidate": name,
        "candidate_rmse": candidate_rmse[name],
        "minimum_seed_log_rmse_improvement": minimum_log_improvement,
    }


def audit_dataset(
    dataset_id: str,
    quant_root: Path,
    round2_root: Path,
    evaluator: Path,
    hyperion_report: dict[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    dataset_root = quant_root / dataset_id
    train_path = common.checked_data_path(dataset_root, "train.csv")
    raw, targets, groups = common.read_training_csv(train_path)
    thresholds = common.Thresholds(minimum_relative_improvement=MINIMUM_SCREEN_IMPROVEMENT)
    cv5_folds = common.grouped_folds(
        groups, 5, common.CV5_SEED, thresholds.minimum_atomic_groups
    )
    candidate_rmse: dict[str, float] = {}
    candidate_failures: dict[str, str] = {}
    for name, genome in CANDIDATES.items():
        try:
            evidence = evaluate_candidate(
                evaluator, genome, raw, targets, groups, cv5_folds, timeout_seconds
            )
            candidate_rmse[name] = pooled_rmse(evidence, "candidate")
        except Exception as error:
            candidate_failures[name] = str(error)

    raw_by_seed: dict[int, float] = {}
    result_hashes: dict[str, str] = {}
    winners: dict[int, str] = {}
    for seed in (101, 202):
        genome, result_sha256 = frozen_winner(round2_root, dataset_id, seed)
        winners[seed] = genome
        result_hashes[str(seed)] = result_sha256
        historical = historical_raw_cv5(hyperion_report, dataset_id, seed)
        if historical is None:
            evidence = evaluate_candidate(
                evaluator, genome, raw, targets, groups, cv5_folds, timeout_seconds
            )
            historical = pooled_rmse(evidence, "candidate")
        raw_by_seed[seed] = historical

    choice = screen_choice(candidate_rmse, raw_by_seed)
    confirmations: dict[str, Any] = {}
    selected = "raw"
    if choice is not None:
        cv7_folds = common.grouped_folds(
            groups, 7, common.CV7_SEED, thresholds.minimum_atomic_groups
        )
        for seed in (101, 202):
            try:
                paired = evaluate_pair(
                    evaluator,
                    winners[seed],
                    CANDIDATES[choice["candidate"]],
                    raw,
                    targets,
                    groups,
                    cv7_folds,
                    timeout_seconds,
                )
                confirmations[str(seed)] = common.evaluate_arm(
                    "cv7", paired, "candidate", thresholds
                )
            except Exception as error:
                confirmations[str(seed)] = {"passed": False, "failure": str(error)}
        if all(confirmations[str(seed)].get("passed") is True for seed in (101, 202)):
            selected = choice["candidate"]
    deltas = [
        math.log(
            confirmations[str(seed)]["pooled_raw_rmse"]
            / confirmations[str(seed)]["pooled_arm_rmse"]
        )
        for seed in (101, 202)
        if confirmations.get(str(seed), {}).get("passed") is True
    ]
    return {
        "dataset_id": dataset_id,
        "rows": int(raw.shape[0]),
        "features": int(raw.shape[1]),
        "train_sha256": common.sha256_file(train_path),
        "round2_result_sha256s": result_hashes,
        "cv5_raw_rmse": {str(seed): value for seed, value in raw_by_seed.items()},
        "cv5_candidate_rmse": candidate_rmse,
        "cv5_candidate_failures": candidate_failures,
        "cv5_choice": choice,
        "cv7_confirmation": confirmations,
        "selected": selected,
        "mean_confirmed_log_rmse_improvement": statistics.mean(deltas) if len(deltas) == 2 else 0.0,
    }


def summarize(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    deltas = [result["mean_confirmed_log_rmse_improvement"] for result in results]
    selected = [result for result in results if result["selected"] != "raw"]
    screen = [result for result in results if result["cv5_choice"] is not None]
    by_candidate: dict[str, int] = {}
    for result in selected:
        by_candidate[result["selected"]] = by_candidate.get(result["selected"], 0) + 1
    return {
        "datasets_complete": len(results),
        "datasets_screened_into_cv7": len(screen),
        "datasets_confirmed_both_seeds": len(selected),
        "confirmed_candidates": by_candidate,
        "datasets_with_positive_gated_delta": sum(delta > 0.0 for delta in deltas),
        "median_dataset_log_rmse_improvement": statistics.median(deltas) if deltas else None,
        "mean_dataset_log_rmse_improvement": statistics.mean(deltas) if deltas else None,
        "bootstrap_95_lower_bound_mean_dataset_delta": common.dataset_bootstrap_lower_bound(deltas),
        "maximum_confirmed_dataset_log_rmse_improvement": max(deltas, default=None),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--round2-root", type=Path, required=True)
    parser.add_argument("--quant-root", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--hyperion-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    dataset_ids = protocol.get("cohorts", {}).get("development")
    if not isinstance(dataset_ids, list) or len(dataset_ids) != 22:
        raise ValueError("raw-floor audit requires the exact development cohort")
    hyperion_report = json.loads(args.hyperion_audit.read_text(encoding="utf-8"))
    if hyperion_report.get("cohort") != "development" or hyperion_report.get("test_artifacts_read") is not False:
        raise ValueError("Hyperion audit is not a train-only development report")
    evaluator = args.evaluator.resolve()
    if not evaluator.is_file() or not os.access(evaluator, os.X_OK):
        raise ValueError("evaluator must be executable")
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {
            executor.submit(
                audit_dataset,
                dataset_id,
                args.quant_root,
                args.round2_root,
                evaluator,
                hyperion_report,
                args.timeout_seconds,
            ): dataset_id
            for dataset_id in dataset_ids
        }
        for future in as_completed(pending):
            dataset_id = pending[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"{dataset_id} screen={result['cv5_choice']} selected={result['selected']} "
                    f"delta={result['mean_confirmed_log_rmse_improvement']:.6f}",
                    flush=True,
                )
            except Exception as error:
                failures.append({"dataset_id": dataset_id, "reason": str(error)})
                print(f"{dataset_id} failed={error}", flush=True)
    results.sort(key=lambda result: result["dataset_id"])
    failures.sort(key=lambda result: result["dataset_id"])
    report = {
        "schema": SCHEMA,
        "cohort": "development",
        "test_artifacts_read": False,
        "promotion_eligible": False,
        "promotion_blocker": "development audit; confirmation cohort remains unopened",
        "minimum_cv5_screen_improvement": MINIMUM_SCREEN_IMPROVEMENT,
        "candidates": CANDIDATES,
        "protocol_sha256": common.sha256_file(args.protocol),
        "evaluator_sha256": common.sha256_file(evaluator),
        "hyperion_audit_sha256": common.sha256_file(args.hyperion_audit),
        "results": results,
        "failures": failures,
        "summary": summarize(results),
    }
    report["report_sha256"] = common.canonical_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0 if not failures and len(results) == len(dataset_ids) else 1


if __name__ == "__main__":
    raise SystemExit(main())
