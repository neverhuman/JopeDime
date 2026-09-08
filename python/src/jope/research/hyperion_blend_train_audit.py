#!/usr/bin/env python3
"""Protected-frontier audit of cell-based auxiliary prediction blends.

Unlike direct feature augmentation, this policy never changes the frozen raw
GP fit.  A fixed production-compatible head fits an independent raw+cell view,
then a small predeclared weight blends its OOF predictions with the raw GP.
CV5 chooses at most one blend per seed; CV7 confirms it unchanged.

The historical full-train cell matrix makes this a development probe only.
Any positive result still requires fresh fold-local encoder receipts before it
can become promotion evidence.  Test artifacts are neither accepted nor read.
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


SCHEMA = "HyperionCellBlendDevelopmentAudit.v1"
HEADS = {
    "ridge_gcv": "(pipe X (ridge_gcv))",
    "elasticnet": "(pipe X (elasticnet))",
    "histgbm2": "(pipe X (histgbm2))",
    "stack": "(pipe X (stack))",
    "portfolio_core": "(pipe X (portfolio preset=core))",
}
WEIGHTS = (0.10, 0.25, 0.50)


def candidate_name(head: str, weight: float) -> str:
    return f"{head}@{weight:.2f}"


def blend(raw: Sequence[float], auxiliary: Sequence[float], weight: float) -> list[float]:
    if len(raw) != len(auxiliary):
        raise ValueError("blend inputs have different lengths")
    return [
        (1.0 - weight) * left + weight * right
        for left, right in zip(raw, auxiliary, strict=True)
    ]


def frozen_winner(round2_root: Path, dataset_id: str, seed: int) -> tuple[str, str]:
    result_path = common.find_round2_result(round2_root, dataset_id, seed)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("dataset_hash") != dataset_id
        or result.get("master_seed") != seed
        or result.get("test_csv_read") is not False
    ):
        raise ValueError(f"invalid round-2 authority: {result_path}")
    genome = result.get("winner_genome")
    if not isinstance(genome, str) or not genome:
        raise ValueError(f"round-2 authority has no frozen winner: {result_path}")
    return genome, common.sha256_file(result_path)


def auxiliary_predictions(
    evaluator: Path,
    heads: dict[str, str],
    raw: np.ndarray,
    targets: np.ndarray,
    groups: Sequence[str],
    cell: np.ndarray,
    folds: Sequence[common.Fold],
    timeout_seconds: int,
) -> tuple[dict[str, list[list[float]]], dict[str, str]]:
    predictions = {head: [] for head in heads}
    failures: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="jope-cell-blend-aux-") as temporary:
        root = Path(temporary)
        for fold in folds:
            transformed = common.transform_cell(cell, fold.fit_rows)
            staged = root / f"fold-{fold.index}.csv"
            common.stage_csv(
                staged,
                np.concatenate((raw, transformed), axis=1),
                targets,
                fold.hold_rows,
            )
            for head, genome in heads.items():
                if head in failures:
                    continue
                try:
                    predictions[head].append(
                        common.evaluate_predictions(
                            evaluator, genome, staged, fold, timeout_seconds
                        )
                    )
                except Exception as error:
                    failures[head] = str(error)
    return {
        head: values for head, values in predictions.items() if head not in failures and len(values) == len(folds)
    }, failures


def raw_predictions(
    evaluator: Path,
    genome: str,
    raw: np.ndarray,
    targets: np.ndarray,
    folds: Sequence[common.Fold],
    timeout_seconds: int,
) -> list[list[float]]:
    predictions: list[list[float]] = []
    with tempfile.TemporaryDirectory(prefix="jope-cell-blend-raw-") as temporary:
        root = Path(temporary)
        for fold in folds:
            staged = root / f"fold-{fold.index}.csv"
            common.stage_csv(staged, raw, targets, fold.hold_rows)
            predictions.append(
                common.evaluate_predictions(evaluator, genome, staged, fold, timeout_seconds)
            )
    return predictions


def paired_evidence(
    folds: Sequence[common.Fold],
    targets: np.ndarray,
    groups: Sequence[str],
    raw: Sequence[Sequence[float]],
    auxiliary: dict[str, Sequence[Sequence[float]]],
) -> list[common.FoldEvidence]:
    evidence: list[common.FoldEvidence] = []
    for fold in folds:
        arms: dict[str, list[float]] = {}
        for head, head_folds in auxiliary.items():
            for weight in WEIGHTS:
                arms[candidate_name(head, weight)] = blend(
                    raw[fold.index], head_folds[fold.index], weight
                )
        hold = list(fold.hold_rows)
        evidence.append(
            common.FoldEvidence(
                index=fold.index,
                row_groups=[groups[index] for index in hold],
                actual=[float(targets[index]) for index in hold],
                raw=list(raw[fold.index]),
                arms=arms,
            )
        )
    return evidence


def parse_candidate(name: str) -> tuple[str, float]:
    head, raw_weight = name.rsplit("@", 1)
    weight = float(raw_weight)
    if head not in HEADS or weight not in WEIGHTS:
        raise ValueError("candidate is outside the frozen blend grid")
    return head, weight


def audit_dataset(
    dataset_id: str,
    quant_root: Path,
    round2_root: Path,
    evaluator: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    dataset_root = quant_root / dataset_id
    train_path = common.checked_data_path(dataset_root, "train.csv")
    cell_path = common.checked_data_path(dataset_root, "train3.vs")
    raw, targets, groups = common.read_training_csv(train_path)
    cell = common.read_matrix(cell_path, raw.shape[0], raw.shape[1])
    thresholds = common.Thresholds()
    winners: dict[int, str] = {}
    result_hashes: dict[str, str] = {}
    for seed in (101, 202):
        winners[seed], result_hashes[str(seed)] = frozen_winner(
            round2_root, dataset_id, seed
        )

    cv5_folds = common.grouped_folds(
        groups, 5, common.CV5_SEED, thresholds.minimum_atomic_groups
    )
    cv5_auxiliary, cv5_failures = auxiliary_predictions(
        evaluator, HEADS, raw, targets, groups, cell, cv5_folds, timeout_seconds
    )
    seeds: dict[str, Any] = {}
    for seed in (101, 202):
        raw_cv5 = raw_predictions(
            evaluator, winners[seed], raw, targets, cv5_folds, timeout_seconds
        )
        cv5_evidence = paired_evidence(
            cv5_folds, targets, groups, raw_cv5, cv5_auxiliary
        )
        cv5_results = [
            common.evaluate_arm("cv5", cv5_evidence, arm, thresholds)
            for arm in sorted(cv5_evidence[0].arms)
        ]
        passed = [result for result in cv5_results if result["passed"]]
        advanced = (
            min(passed, key=lambda item: (-item["relative_improvement"], item["arm"]))["arm"]
            if passed
            else None
        )
        seeds[str(seed)] = {
            "cv5": cv5_results,
            "cv5_advanced": advanced,
            "cv7": None,
            "selected": "raw",
            "selected_log_rmse_improvement": 0.0,
        }

    advanced_heads = sorted(
        {parse_candidate(value["cv5_advanced"])[0] for value in seeds.values() if value["cv5_advanced"]}
    )
    cv7_failures: dict[str, str] = {}
    if advanced_heads:
        cv7_folds = common.grouped_folds(
            groups, 7, common.CV7_SEED, thresholds.minimum_atomic_groups
        )
        cv7_auxiliary, cv7_failures = auxiliary_predictions(
            evaluator,
            {head: HEADS[head] for head in advanced_heads},
            raw,
            targets,
            groups,
            cell,
            cv7_folds,
            timeout_seconds,
        )
        for seed in (101, 202):
            advanced = seeds[str(seed)]["cv5_advanced"]
            if advanced is None:
                continue
            head, weight = parse_candidate(advanced)
            if head not in cv7_auxiliary:
                seeds[str(seed)]["cv7"] = {
                    "passed": False,
                    "failure": cv7_failures.get(head, "missing auxiliary CV7 predictions"),
                }
                continue
            raw_cv7 = raw_predictions(
                evaluator, winners[seed], raw, targets, cv7_folds, timeout_seconds
            )
            auxiliary = {
                head: cv7_auxiliary[head]
            }
            cv7_evidence = paired_evidence(
                cv7_folds, targets, groups, raw_cv7, auxiliary
            )
            # paired_evidence emits every frozen weight; confirm only the exact
            # CV5 winner and never reopen arm selection on CV7.
            confirmation = common.evaluate_arm(
                "cv7", cv7_evidence, candidate_name(head, weight), thresholds
            )
            seeds[str(seed)]["cv7"] = confirmation
            if confirmation["passed"]:
                seeds[str(seed)]["selected"] = advanced
                seeds[str(seed)]["selected_log_rmse_improvement"] = math.log(
                    confirmation["pooled_raw_rmse"] / confirmation["pooled_arm_rmse"]
                )

    return {
        "dataset_id": dataset_id,
        "rows": int(raw.shape[0]),
        "features": int(raw.shape[1]),
        "train_sha256": common.sha256_file(train_path),
        "historical_cell_sha256": common.sha256_file(cell_path),
        "round2_result_sha256s": result_hashes,
        "cv5_auxiliary_failures": cv5_failures,
        "cv7_auxiliary_failures": cv7_failures,
        "seeds": seeds,
    }


def summarize(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    seed_rows = [seed for result in results for seed in result["seeds"].values()]
    activations = [seed for seed in seed_rows if seed["selected"] != "raw"]
    dataset_deltas = [
        statistics.mean(
            result["seeds"][str(seed)]["selected_log_rmse_improvement"]
            for seed in (101, 202)
        )
        for result in results
    ]
    return {
        "datasets_complete": len(results),
        "seed_runs": len(seed_rows),
        "cv5_advances": sum(seed["cv5_advanced"] is not None for seed in seed_rows),
        "cv7_activations": len(activations),
        "activation_rate": len(activations) / len(seed_rows) if seed_rows else 0.0,
        "selected_views": sorted(seed["selected"] for seed in activations),
        "datasets_with_positive_gated_delta": sum(delta > 0.0 for delta in dataset_deltas),
        "median_dataset_log_rmse_improvement": statistics.median(dataset_deltas)
        if dataset_deltas
        else None,
        "maximum_seed_log_rmse_improvement": max(
            (seed["selected_log_rmse_improvement"] for seed in seed_rows), default=None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--round2-root", type=Path, required=True)
    parser.add_argument("--quant-root", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    dataset_ids = protocol.get("best5_screen", {}).get("datasets")
    if not isinstance(dataset_ids, list) or len(dataset_ids) != 2:
        raise ValueError("cell-blend audit requires the exact protected two-dataset screen")
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
                args.timeout_seconds,
            ): dataset_id
            for dataset_id in dataset_ids
        }
        for future in as_completed(pending):
            dataset_id = pending[future]
            try:
                result = future.result()
                results.append(result)
                selected = {seed: value["selected"] for seed, value in result["seeds"].items()}
                print(f"{dataset_id} selected={selected}", flush=True)
            except Exception as error:
                failures.append({"dataset_id": dataset_id, "reason": str(error)})
                print(f"{dataset_id} failed={error}", flush=True)
    results.sort(key=lambda result: result["dataset_id"])
    failures.sort(key=lambda result: result["dataset_id"])
    report = {
        "schema": SCHEMA,
        "cohort": "development-protected-frontier",
        "promotion_eligible": False,
        "promotion_blocker": "historical full-train cell matrix is not a fold-local encoder receipt",
        "test_artifacts_read": False,
        "heads": HEADS,
        "blend_weights": list(WEIGHTS),
        "thresholds": common.Thresholds().__dict__,
        "protocol_sha256": common.sha256_file(args.protocol),
        "evaluator_sha256": common.sha256_file(evaluator),
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
