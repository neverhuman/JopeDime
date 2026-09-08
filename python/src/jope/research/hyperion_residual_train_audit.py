#!/usr/bin/env python3
"""Protected train-only audit of a cross-fitted Hyperion residual corrector.

The frozen round-2 GP remains the predictor.  Within every outer fold, inner
atomic-group folds generate out-of-fold GP residuals.  A small ridge model maps
fold-train ECDF-transformed historical Hyperion cell signals to those residuals
and adds a fixed shrunken correction to the outer-fold GP prediction.

This is development evidence only: historical ``train3.vs`` was encoded over
the full training table and cannot authorize promotion.  A positive protected
screen would justify rebuilding fold-local encoder receipts; a negative screen
closes the residual-correction branch without any test access.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import expanded_gp_protected_screen as protected
import hyperion_train_audit as common
import raw_floor_train_audit as raw_floor


SCHEMA = "HyperionResidualProtectedAudit.v1"
INNER_FOLDS = 5
CORRECTIONS = {
    "cell_residual_a001_w025": (0.01, 0.25),
    "cell_residual_a001_w050": (0.01, 0.50),
    "cell_residual_a001_w100": (0.01, 1.00),
    "cell_residual_a010_w025": (0.10, 0.25),
    "cell_residual_a010_w050": (0.10, 0.50),
    "cell_residual_a010_w100": (0.10, 1.00),
}


def inner_folds(groups: Sequence[str], outer: common.Fold) -> list[common.Fold]:
    outer_rows = list(outer.fit_rows)
    local_groups = [groups[row] for row in outer_rows]
    local = common.grouped_folds(
        local_groups,
        INNER_FOLDS,
        common.seed64(("hyperion-residual-inner", str(outer.index))),
        common.Thresholds().minimum_atomic_groups,
    )
    return [
        common.Fold(
            index=fold.index,
            fit_rows=tuple(outer_rows[index] for index in fold.fit_rows),
            hold_rows=tuple(outer_rows[index] for index in fold.hold_rows),
            hold_groups=fold.hold_groups,
        )
        for fold in local
    ]


def ridge_residual_predictions(
    fit_features: np.ndarray,
    residuals: np.ndarray,
    hold_features: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if fit_features.shape[0] != residuals.shape[0] or fit_features.shape[1] != hold_features.shape[1]:
        raise ValueError("residual ridge shape mismatch")
    means = np.mean(fit_features, axis=0)
    scales = np.maximum(np.std(fit_features, axis=0), 1e-12)
    fit = (fit_features - means) / scales
    hold = (hold_features - means) / scales
    residual_mean = float(np.mean(residuals))
    centered = residuals - residual_mean
    gram = fit.T @ fit
    penalty = alpha * fit.shape[0]
    coefficients = np.linalg.solve(
        gram + penalty * np.eye(gram.shape[0], dtype=np.float64),
        fit.T @ centered,
    )
    predictions = residual_mean + hold @ coefficients
    if not np.isfinite(predictions).all():
        raise ValueError("residual ridge produced non-finite predictions")
    return predictions


def execute_stage(
    evaluator: Path,
    genome: str,
    raw: np.ndarray,
    targets: np.ndarray,
    groups: Sequence[str],
    cell: np.ndarray,
    folds: Sequence[common.Fold],
    corrections: dict[str, tuple[float, float]],
    timeout_seconds: int,
) -> list[common.FoldEvidence]:
    all_rows = set(range(raw.shape[0]))
    evidence: list[common.FoldEvidence] = []
    with tempfile.TemporaryDirectory(prefix="jope-hyperion-residual-") as temporary:
        root = Path(temporary)
        for outer in folds:
            outer_stage = root / f"outer-{outer.index}.csv"
            common.stage_csv(outer_stage, raw, targets, outer.hold_rows)
            raw_hold = common.evaluate_predictions(
                evaluator, genome, outer_stage, outer, timeout_seconds
            )

            outer_fit = list(outer.fit_rows)
            fit_position = {row: index for index, row in enumerate(outer_fit)}
            residuals = np.empty(len(outer_fit), dtype=np.float64)
            residual_features = np.empty((len(outer_fit), cell.shape[1]), dtype=np.float64)
            covered: set[int] = set()
            for inner in inner_folds(groups, outer):
                # The child sees real targets only for its fitting rows.  Outer
                # hold rows and this inner hold are both replaced with dummies.
                hidden = tuple(sorted(all_rows - set(inner.fit_rows)))
                inner_stage = root / f"outer-{outer.index}-inner-{inner.index}.csv"
                common.stage_csv(inner_stage, raw, targets, hidden)
                inner_predictions = common.evaluate_predictions(
                    evaluator, genome, inner_stage, inner, timeout_seconds
                )
                transformed = common.transform_cell(cell, inner.fit_rows)
                for offset, row in enumerate(inner.hold_rows):
                    position = fit_position[row]
                    residuals[position] = targets[row] - inner_predictions[offset]
                    residual_features[position] = transformed[row]
                    covered.add(row)
            if covered != set(outer_fit):
                raise ValueError("inner folds did not cross-fit every outer fitting row")

            outer_transformed = common.transform_cell(cell, outer.fit_rows)
            hold_features = outer_transformed[np.asarray(outer.hold_rows, dtype=np.int64)]
            arms: dict[str, list[float]] = {}
            by_alpha = {
                alpha: ridge_residual_predictions(
                    residual_features, residuals, hold_features, alpha
                )
                for alpha, _ in set(corrections.values())
            }
            raw_vector = np.asarray(raw_hold, dtype=np.float64)
            for arm, (alpha, weight) in corrections.items():
                arms[arm] = [
                    float(value) for value in raw_vector + weight * by_alpha[alpha]
                ]
            hold = list(outer.hold_rows)
            evidence.append(
                common.FoldEvidence(
                    index=outer.index,
                    row_groups=[groups[row] for row in hold],
                    actual=[float(targets[row]) for row in hold],
                    raw=raw_hold,
                    arms=arms,
                )
            )
    return evidence


def audit_one(
    dataset_id: str,
    seed: int,
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
    genome, round2_sha256 = raw_floor.frozen_winner(round2_root, dataset_id, seed)
    thresholds = common.Thresholds()
    cv5_folds = common.grouped_folds(
        groups, 5, common.CV5_SEED, thresholds.minimum_atomic_groups
    )
    cv5_evidence = execute_stage(
        evaluator,
        genome,
        raw,
        targets,
        groups,
        cell,
        cv5_folds,
        CORRECTIONS,
        timeout_seconds,
    )
    cv5 = {
        arm: common.evaluate_arm("cv5", cv5_evidence, arm, thresholds)
        for arm in CORRECTIONS
    }
    passed = [evidence for evidence in cv5.values() if evidence["passed"]]
    advanced = (
        min(passed, key=lambda item: (-item["relative_improvement"], item["arm"]))["arm"]
        if passed
        else None
    )
    cv7: dict[str, Any] | None = None
    if advanced is not None:
        cv7_folds = common.grouped_folds(
            groups, 7, common.CV7_SEED, thresholds.minimum_atomic_groups
        )
        cv7_evidence = execute_stage(
            evaluator,
            genome,
            raw,
            targets,
            groups,
            cell,
            cv7_folds,
            {advanced: CORRECTIONS[advanced]},
            timeout_seconds,
        )
        cv7 = common.evaluate_arm("cv7", cv7_evidence, advanced, thresholds)
    selected = advanced if cv7 and cv7["passed"] else "raw"
    return {
        "dataset_id": dataset_id,
        "seed": seed,
        "rows": int(raw.shape[0]),
        "raw_features": int(raw.shape[1]),
        "source_sha256": {
            "train": common.sha256_file(train_path),
            "historical_cell": common.sha256_file(cell_path),
            "round2_result": round2_sha256,
        },
        "cv5": cv5,
        "cv5_advanced_arm": advanced,
        "cv7": cv7,
        "selected": selected,
        "selected_log_rmse_improvement": (
            math.log(cv7["pooled_raw_rmse"] / cv7["pooled_arm_rmse"])
            if selected != "raw"
            else 0.0
        ),
    }


def summarize(results: Sequence[dict[str, Any]], expected: int) -> dict[str, Any]:
    best = [
        max(result["cv5"].values(), key=lambda evidence: evidence["relative_improvement"])
        for result in results
    ]
    selected = [result for result in results if result["selected"] != "raw"]
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_dataset.setdefault(result["dataset_id"], []).append(result)
    return {
        "complete_seed_dataset_audits": len(results),
        "expected_seed_dataset_audits": expected,
        "cv5_seed_advances": sum(result["cv5_advanced_arm"] is not None for result in results),
        "cv7_seed_passes": len(selected),
        "protected_datasets_both_seeds_confirmed": sum(
            len(rows) == len(protected.SEEDS) and all(row["selected"] != "raw" for row in rows)
            for rows in by_dataset.values()
        ),
        "maximum_observed_cv5_relative_improvement": max(
            (evidence["relative_improvement"] for evidence in best), default=None
        ),
        "median_best_observed_cv5_relative_improvement": (
            statistics.median(evidence["relative_improvement"] for evidence in best)
            if best
            else None
        ),
        "selected_arms": {
            arm: sum(result["selected"] == arm for result in results)
            for arm in ("raw", *CORRECTIONS)
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--round2-root", required=True, type=Path)
    parser.add_argument("--quant-root", required=True, type=Path)
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.timeout_seconds < 1:
        raise ValueError("workers and timeout must be positive")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    dataset_ids = protected.protected_ids(protocol)
    evaluator = args.evaluator.resolve()
    if not evaluator.is_file() or not os.access(evaluator, os.X_OK):
        raise ValueError("evaluator must be executable")
    jobs = [(dataset_id, seed) for dataset_id in dataset_ids for seed in protected.SEEDS]
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {
            executor.submit(
                audit_one,
                dataset_id,
                seed,
                args.quant_root,
                args.round2_root,
                evaluator,
                args.timeout_seconds,
            ): (dataset_id, seed)
            for dataset_id, seed in jobs
        }
        for future in as_completed(pending):
            dataset_id, seed = pending[future]
            try:
                result = future.result()
                results.append(result)
                best = max(
                    result["cv5"].values(), key=lambda evidence: evidence["relative_improvement"]
                )
                print(
                    f"{dataset_id} seed={seed} best={best['arm']} "
                    f"cv5={best['relative_improvement']:.6f} selected={result['selected']}",
                    flush=True,
                )
            except Exception as error:
                failures.append({"dataset_id": dataset_id, "seed": seed, "reason": str(error)})
                print(f"{dataset_id} seed={seed} failed={error}", flush=True)
    results.sort(key=lambda result: (result["dataset_id"], result["seed"]))
    failures.sort(key=lambda failure: (failure["dataset_id"], failure["seed"]))
    summary = summarize(results, len(jobs))
    passed = (
        not failures
        and len(results) == len(jobs)
        and summary["protected_datasets_both_seeds_confirmed"] == len(dataset_ids)
    )
    report = {
        "schema": SCHEMA,
        "cohort": "development-protected",
        "protected_dataset_ids": list(dataset_ids),
        "seeds": list(protected.SEEDS),
        "test_artifacts_read": False,
        "allowed_data_basenames": ["train.csv", "train3.vs"],
        "promotion_eligible": False,
        "promotion_blocker": "historical full-train cell sidecar is not a fold-local encoder receipt",
        "protected_screen_passed": passed,
        "inner_folds": INNER_FOLDS,
        "corrections": {
            arm: {"ridge_alpha_per_row": alpha, "shrinkage_weight": weight}
            for arm, (alpha, weight) in CORRECTIONS.items()
        },
        "thresholds": asdict(common.Thresholds()),
        "protocol_sha256": common.sha256_file(args.protocol),
        "evaluator_sha256": common.sha256_file(evaluator),
        "results": results,
        "failures": failures,
        "summary": summary,
    }
    report["report_sha256"] = common.canonical_sha256(report)
    protected.atomic_json(args.output, report)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if not failures and len(results) == len(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
