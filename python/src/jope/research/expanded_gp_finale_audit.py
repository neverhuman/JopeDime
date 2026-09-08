#!/usr/bin/env python3
"""Train-only protected audit of predeclared expanded-GP finalizers.

The expanded search emits five finalizer choices before any external gate is
run: repeated-CV 0.5-SE, CV7 argmax, bootstrap-stable, uniform top-5, and a
greedy ensemble (with replacement).  This audit evaluates that fixed arm set
against each seed's frozen round-2 predictor using atomic-group CV5.  At most
one arm per seed advances to CV7.

Only the two protected development datasets and ``train.csv`` are accepted.
Search results and receipts are immutable inputs; no search is rerun here.
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

import expanded_gp_protected_screen as expanded
import hyperion_train_audit as common
import raw_floor_train_audit as raw_floor


SCHEMA = "ExpandedGpFinaleProtectedAudit.v1"
ARM_KEYS = {
    "rcv_1se": "t4_rcv_1se",
    "cv7_argmax": "control_cv7",
    "bootstrap_stable": "t3_bootstrap",
    "uniform5": "t2_uniform5",
    "greedy8": "t2_greedy",
}


def extract_arms(result: dict[str, Any]) -> dict[str, list[tuple[str, str]]]:
    top = result.get("top")
    finales = result.get("finales")
    if not isinstance(top, list) or not isinstance(finales, dict):
        raise ValueError("expanded result lacks finale evidence")
    genomes: dict[str, str] = {}
    for member in top:
        semantic = member.get("semantic")
        canonical = member.get("canonical")
        if not isinstance(semantic, str) or not isinstance(canonical, str):
            raise ValueError("expanded result has malformed top member")
        if semantic in genomes and genomes[semantic] != canonical:
            raise ValueError("semantic identity maps to multiple genomes")
        genomes[semantic] = canonical
    arms: dict[str, list[tuple[str, str]]] = {}
    for arm, key in ARM_KEYS.items():
        references = finales.get(key)
        if isinstance(references, str):
            references = [references]
        if not isinstance(references, list) or not references:
            raise ValueError(f"expanded result lacks {key}")
        if arm == "uniform5" and len(references) != 5:
            raise ValueError("uniform5 must contain exactly five members")
        if arm == "greedy8" and len(references) > 8:
            raise ValueError("greedy8 exceeds its fixed member budget")
        members: list[tuple[str, str]] = []
        for semantic in references:
            if not isinstance(semantic, str) or semantic not in genomes:
                raise ValueError(f"{key} references a missing top member")
            members.append((semantic, genomes[semantic]))
        arms[arm] = members
    return arms


def average_predictions(predictions: Sequence[Sequence[float]]) -> list[float]:
    if not predictions or any(len(values) != len(predictions[0]) for values in predictions):
        raise ValueError("cannot average empty or ragged predictions")
    return [
        sum(values[row] for values in predictions) / len(predictions)
        for row in range(len(predictions[0]))
    ]


def execute_stage(
    evaluator: Path,
    control_genome: str,
    arms: dict[str, list[tuple[str, str]]],
    raw: np.ndarray,
    targets: np.ndarray,
    groups: Sequence[str],
    folds: Sequence[common.Fold],
    timeout_seconds: int,
) -> list[common.FoldEvidence]:
    unique = {
        semantic: canonical
        for members in arms.values()
        for semantic, canonical in members
    }
    evidence: list[common.FoldEvidence] = []
    with tempfile.TemporaryDirectory(prefix="jope-expanded-finale-") as temporary:
        root = Path(temporary)
        for fold in folds:
            staged = root / f"fold-{fold.index}.csv"
            common.stage_csv(staged, raw, targets, fold.hold_rows)
            control = common.evaluate_predictions(
                evaluator, control_genome, staged, fold, timeout_seconds
            )
            by_semantic = {
                semantic: common.evaluate_predictions(
                    evaluator, canonical, staged, fold, timeout_seconds
                )
                for semantic, canonical in sorted(unique.items())
            }
            arm_predictions = {
                arm: average_predictions([by_semantic[semantic] for semantic, _ in members])
                for arm, members in arms.items()
            }
            hold = list(fold.hold_rows)
            evidence.append(
                common.FoldEvidence(
                    index=fold.index,
                    row_groups=[groups[index] for index in hold],
                    actual=[float(targets[index]) for index in hold],
                    raw=control,
                    arms=arm_predictions,
                )
            )
    return evidence


def audit_one(
    dataset_id: str,
    seed: int,
    quant_root: Path,
    round2_root: Path,
    search_root: Path,
    evaluator: Path,
    timeout_seconds: int,
    search_evaluations: int,
) -> dict[str, Any]:
    train_path = common.checked_data_path(quant_root / dataset_id, "train.csv")
    raw, targets, groups = common.read_training_csv(train_path)
    result_path = search_root / "jobs" / dataset_id / f"seed-{seed}" / "result.json"
    receipt_path = search_root / "jobs" / dataset_id / f"seed-{seed}" / "receipt.v1.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expanded.validate_search_result(
        result,
        dataset_id,
        seed,
        strategy=f"c{search_evaluations}",
        search_evaluations=search_evaluations,
    )
    if receipt.get("result_sha256") != common.sha256_file(result_path):
        raise ValueError("expanded search receipt does not bind result")
    if receipt.get("test_artifacts_read") is not False:
        raise ValueError("expanded search receipt lacks train-only attestation")
    arms = extract_arms(result)
    control_genome, control_result_sha256 = raw_floor.frozen_winner(
        round2_root, dataset_id, seed
    )
    thresholds = common.Thresholds()
    cv5_folds = common.grouped_folds(
        groups, 5, common.CV5_SEED, thresholds.minimum_atomic_groups
    )
    cv5_evidence = execute_stage(
        evaluator,
        control_genome,
        arms,
        raw,
        targets,
        groups,
        cv5_folds,
        timeout_seconds,
    )
    cv5 = {
        arm: common.evaluate_arm("cv5", cv5_evidence, arm, thresholds)
        for arm in ARM_KEYS
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
            control_genome,
            {advanced: arms[advanced]},
            raw,
            targets,
            groups,
            cv7_folds,
            timeout_seconds,
        )
        cv7 = common.evaluate_arm("cv7", cv7_evidence, advanced, thresholds)
    selected = advanced if cv7 and cv7["passed"] else "round2_raw"
    selected_delta = (
        math.log(cv7["pooled_raw_rmse"] / cv7["pooled_arm_rmse"])
        if selected != "round2_raw"
        else 0.0
    )
    return {
        "dataset_id": dataset_id,
        "seed": seed,
        "train_sha256": common.sha256_file(train_path),
        "expanded_result_sha256": common.sha256_file(result_path),
        "expanded_receipt_sha256": common.sha256_file(receipt_path),
        "control_round2_result_sha256": control_result_sha256,
        "arms": {
            arm: {
                "members": [semantic for semantic, _ in members],
                "member_count_with_replacement": len(members),
                "unique_member_count": len({semantic for semantic, _ in members}),
            }
            for arm, members in arms.items()
        },
        "cv5": cv5,
        "cv5_advanced_arm": advanced,
        "cv7": cv7,
        "selected": selected,
        "selected_log_rmse_improvement": selected_delta,
    }


def summarize(results: Sequence[dict[str, Any]], expected: int) -> dict[str, Any]:
    cv5_passes = [
        (result["dataset_id"], result["seed"], arm)
        for result in results
        for arm, evidence in result["cv5"].items()
        if evidence["passed"]
    ]
    selected = [result for result in results if result["selected"] != "round2_raw"]
    best_observed = [
        max(result["cv5"].values(), key=lambda value: value["relative_improvement"])
        for result in results
    ]
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_dataset.setdefault(result["dataset_id"], []).append(result)
    deltas = [result["selected_log_rmse_improvement"] for result in results]
    return {
        "complete_seed_dataset_audits": len(results),
        "expected_seed_dataset_audits": expected,
        "cv5_arm_passes": len(cv5_passes),
        "cv5_pass_details": [list(value) for value in cv5_passes],
        "cv7_seed_passes": len(selected),
        "protected_datasets_both_seeds_confirmed": sum(
            len(rows) == len(expanded.SEEDS)
            and all(row["selected"] != "round2_raw" for row in rows)
            for rows in by_dataset.values()
        ),
        "maximum_observed_cv5_relative_improvement": max(
            (value["relative_improvement"] for value in best_observed), default=None
        ),
        "median_best_observed_cv5_relative_improvement": (
            statistics.median(value["relative_improvement"] for value in best_observed)
            if best_observed
            else None
        ),
        "mean_selected_log_rmse_improvement": statistics.mean(deltas) if deltas else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--round2-root", required=True, type=Path)
    parser.add_argument("--quant-root", required=True, type=Path)
    parser.add_argument("--search-root", required=True, type=Path)
    parser.add_argument("--search-report", required=True, type=Path)
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
    dataset_ids = expanded.protected_ids(protocol)
    search_report = json.loads(args.search_report.read_text(encoding="utf-8"))
    if (
        search_report.get("schema") != expanded.SCHEMA
        or search_report.get("test_artifacts_read") is not False
        or search_report.get("protected_dataset_ids") != list(dataset_ids)
    ):
        raise ValueError("expanded search report is not the protected train-only evidence")
    search_evaluations = search_report.get("evaluation_accounting_per_search", {}).get(
        "search_evaluations"
    )
    if not isinstance(search_evaluations, int) or search_evaluations < 1000:
        raise ValueError("expanded search report lacks evaluation accounting")
    evaluator = args.evaluator.resolve()
    if not evaluator.is_file() or not os.access(evaluator, os.X_OK):
        raise ValueError("evaluator must be executable")
    jobs = [(dataset_id, seed) for dataset_id in dataset_ids for seed in expanded.SEEDS]
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
                args.search_root,
                evaluator,
                args.timeout_seconds,
                search_evaluations,
            ): (dataset_id, seed)
            for dataset_id, seed in jobs
        }
        for future in as_completed(pending):
            dataset_id, seed = pending[future]
            try:
                result = future.result()
                results.append(result)
                best = max(
                    result["cv5"].values(), key=lambda value: value["relative_improvement"]
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
        "source_campaign_id": search_report.get("campaign_id"),
        "cohort": "development-protected",
        "protected_dataset_ids": list(dataset_ids),
        "seeds": list(expanded.SEEDS),
        "test_artifacts_read": False,
        "allowed_data_basenames": ["train.csv"],
        "fixed_arm_keys": ARM_KEYS,
        "thresholds": asdict(common.Thresholds()),
        "protected_screen_passed": passed,
        "promotion_eligible": False,
        "promotion_blocker": (
            "must expand the frozen winning finalizer to the remaining frontier development datasets"
            if passed
            else "no predeclared expanded-GP finalizer passed the protected development screen"
        ),
        "protocol_sha256": common.sha256_file(args.protocol),
        "evaluator_sha256": common.sha256_file(evaluator),
        "expanded_search_report_sha256": common.sha256_file(args.search_report),
        "results": results,
        "failures": failures,
        "summary": summary,
    }
    report["report_sha256"] = common.canonical_sha256(report)
    expanded.atomic_json(args.output, report)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if not failures and len(results) == len(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
