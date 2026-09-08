#!/usr/bin/env python3
"""Matched train-only CV-search screen for precomputed Hyperion cell features.

For each protected development dataset and seed, this campaign runs the same
strict two-fold pooled-OOF search twice: once on raw features and once on
``raw + train3.vs``.  Both searches start from the identical sealed r2000
top-ten archive and consume exactly 2,000 search fits.  CV5 compares the two
frozen winners on common grouped folds; CV7 confirms only an arm that passes
CV5.  A fixed single-worker execution policy gives every candidate a 300-second
deadline and aborts immediately on incomplete fitness.  No confirmation dataset
and no test-side basename is accepted.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import expanded_gp_protected_screen as expanded
import hyperion_train_audit as common


SCHEMA = "HyperionMatchedCvSearchProtectedScreen.v2"
RECEIPT_SCHEMA = "HyperionMatchedCvSearchReceipt.v2"
CAMPAIGN_ID = "hyperion-matched-cv2-r2000-complete-protected-v2"
STRATEGY_PREFIX = "hyperion-cv2-r2000-complete"
VIEWS = ("raw", "hyperion_cell")
SEEDS = (101, 202)
SEARCH_EVALUATIONS = 2_000
SEARCH_FOLDS = 2
SEARCH_GENERATIONS = 25
EXPECTED_WORKERS = 1
EXPECTED_RAYON_THREADS = 24
EXPECTED_EVALUATOR_TIMEOUT_SECONDS = 300
EXPECTED_SEARCH_TIMEOUT_SECONDS = 21_600
EXPECTED_GATE_TIMEOUT_SECONDS = 300
ALLOWED_DATA_BASENAMES = frozenset({"train.csv", "train3.vs"})
TREATMENT = {
    "schema": "JopeStrategyScreenConfig.v2",
    "vocab": "expanded",
    "baseline_seeds": False,
    "pooled_oof": True,
    "winner_rule": "rcv_1se",
    "drop_failed_folds": False,
    "se_band": 0.5,
    "perturb_fitness": False,
    "search_folds": SEARCH_FOLDS,
    "search_evaluations": SEARCH_EVALUATIONS,
    "cell_preprocessing": "fold_train_empirical_percentile_per_column.v1",
    "completion_policy": "abort_on_first_incomplete_candidate.v1",
    "workers": EXPECTED_WORKERS,
    "rayon_threads": EXPECTED_RAYON_THREADS,
    "candidate_timeout_seconds": EXPECTED_EVALUATOR_TIMEOUT_SECONDS,
    "search_timeout_seconds": EXPECTED_SEARCH_TIMEOUT_SECONDS,
    "gate_timeout_seconds": EXPECTED_GATE_TIMEOUT_SECONDS,
}


def strategy_name(view: str) -> str:
    if view not in VIEWS:
        raise ValueError(f"unsupported feature view {view}")
    return f"{STRATEGY_PREFIX}-{view}"


def checked_train_path(dataset_root: Path, basename: str) -> Path:
    if basename not in ALLOWED_DATA_BASENAMES:
        raise ValueError(f"forbidden matched-search input: {basename}")
    path = dataset_root / basename
    if path.name != basename or not path.is_file():
        raise FileNotFoundError(path)
    return path


def parent_result_path(parent_root: Path, dataset_id: str, seed: int) -> Path:
    return parent_root / "jobs" / dataset_id / f"seed-{seed}" / "result.json"


def strict_v2_config(
    base: dict[str, Any],
    parent: dict[str, Any],
    dataset_id: str,
    seed: int,
    view: str,
) -> dict[str, Any]:
    if view not in VIEWS:
        raise ValueError(f"unsupported feature view {view}")
    if base.get("schema") != "JopeStrategyScreenConfig.v1":
        raise ValueError("matched search requires the frozen round-2 V1 base")
    if base.get("dataset_hash") != dataset_id or base.get("master_seed") != seed:
        raise ValueError("round-2 base identity mismatch")
    if base.get("budget") != expanded.EXPECTED_BUDGET:
        raise ValueError("round-2 base budget drift")
    expanded.validate_search_result(
        parent,
        dataset_id,
        seed,
        strategy="c2000",
        search_evaluations=2_000,
    )
    top = parent.get("top")
    if not isinstance(top, list) or len(top) != 10:
        raise ValueError("matched search requires the sealed r2000 top-ten archive")
    archive = [member.get("canonical") for member in top]
    if any(not isinstance(genome, str) or not genome for genome in archive):
        raise ValueError("sealed r2000 archive contains a malformed genome")

    config = copy.deepcopy(base)
    config.update(
        {
            "schema": TREATMENT["schema"],
            "strategy": strategy_name(view),
            "dl_genes": archive,
            "vocab": TREATMENT["vocab"],
            "baseline_seeds": TREATMENT["baseline_seeds"],
            "pooled_oof": TREATMENT["pooled_oof"],
            "winner_rule": TREATMENT["winner_rule"],
            "drop_failed_folds": TREATMENT["drop_failed_folds"],
            "se_band": TREATMENT["se_band"],
            "perturb_fitness": TREATMENT["perturb_fitness"],
        }
    )
    config["budget"].update(
        {
            "generations": SEARCH_GENERATIONS,
            "splits_per_gen": 0,
            "cv_fitness": SEARCH_FOLDS,
        }
    )
    return config


def validate_result(
    result: dict[str, Any], dataset_id: str, seed: int, view: str
) -> None:
    strict_contract = result.get("strict_fold_contract")
    if (
        result.get("schema") != "JopeStrategyScreenResult.v2"
        or result.get("strategy") != strategy_name(view)
        or result.get("dataset_hash") != dataset_id
        or result.get("master_seed") != seed
        or result.get("test_csv_read") is not False
        or result.get("search_evaluations") != SEARCH_EVALUATIONS
        or result.get("final_cv_evaluations") != 70
        or not isinstance(result.get("winner_genome"), str)
        or not result["winner_genome"]
        or not isinstance(strict_contract, dict)
        or strict_contract.get("complete_predictions_required") is not True
        or strict_contract.get("fail_on_incomplete_candidate") is not True
        or strict_contract.get("candidate_timeout_seconds")
        != EXPECTED_EVALUATOR_TIMEOUT_SECONDS
    ):
        raise ValueError(
            "matched CV-search result is incomplete, foreign, or not train-only"
        )
    generations = result.get("generations")
    if not isinstance(generations, list) or len(generations) != SEARCH_GENERATIONS:
        raise ValueError("matched CV-search generation coverage is incomplete")
    if any(generation.get("invalid_members") != 0 for generation in generations):
        raise ValueError("matched CV-search contains incomplete candidate fitness")


def recover_completed_result(
    root: Path,
    dataset_id: str,
    seed: int,
    view: str,
) -> tuple[dict[str, Any], Path] | None:
    valid: list[tuple[dict[str, Any], Path]] = []
    for candidate in sorted(root.glob(".result.json.tmp.*")):
        try:
            result = json.loads(candidate.read_text(encoding="utf-8"))
            validate_result(result, dataset_id, seed, view)
            valid.append((result, candidate))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    if len(valid) > 1:
        raise ValueError(
            "multiple complete temporary results require manual quarantine"
        )
    if not valid:
        return None
    return valid[0]


def stage_view(
    dataset_root: Path, view: str, destination: Path
) -> tuple[Path, dict[str, Any]]:
    train_path = checked_train_path(dataset_root, "train.csv")
    raw, targets, _ = common.read_training_csv(train_path)
    metadata: dict[str, Any] = {
        "view": view,
        "rows": int(raw.shape[0]),
        "raw_features": int(raw.shape[1]),
        "train_sha256": common.sha256_file(train_path),
        "test_artifacts_read": False,
    }
    if view == "raw":
        features = raw
        metadata["cell_sha256"] = None
    elif view == "hyperion_cell":
        cell_path = checked_train_path(dataset_root, "train3.vs")
        cell = common.read_matrix(cell_path, raw.shape[0], raw.shape[1])
        # Fit this deterministic preprocessing on the complete training table
        # for search.  The external CV5/CV7 gate refits it fold-locally.
        cell = common.transform_cell(cell, tuple(range(raw.shape[0])))
        features = np.concatenate((raw, cell), axis=1)
        metadata["cell_sha256"] = common.sha256_file(cell_path)
    else:
        raise ValueError(f"unsupported feature view {view}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    common.stage_csv(destination, features, targets, ())
    metadata.update(
        {
            "output_features": int(features.shape[1]),
            "staged_sha256": common.sha256_file(destination),
        }
    )
    return destination, metadata


def run_search(
    dataset_id: str,
    seed: int,
    view: str,
    quant_root: Path,
    round2_root: Path,
    parent_root: Path,
    evaluator: Path,
    output_root: Path,
    source_sha256s: dict[str, str],
    recovery_launcher_sha256: str | None,
    rayon_threads: int,
    evaluator_timeout_seconds: int,
    search_timeout_seconds: int,
) -> dict[str, Any]:
    base_path = expanded.base_config_path(round2_root, dataset_id, seed)
    parent_path = parent_result_path(parent_root, dataset_id, seed)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    config = strict_v2_config(base, parent, dataset_id, seed, view)
    root = output_root / "jobs" / dataset_id / f"seed-{seed}" / view
    config_path = root / "config.v2.json"
    staged_path = root / "train-view.csv"
    result_path = root / "result.json"
    receipt_path = root / "receipt.v1.json"
    expanded.atomic_json(config_path, config)
    _, staged = stage_view(quant_root / dataset_id, view, staged_path)
    expected = {
        "schema": RECEIPT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "dataset_id": dataset_id,
        "seed": seed,
        "view": view,
        "config_sha256": common.sha256_file(config_path),
        "base_config_sha256": common.sha256_file(base_path),
        "parent_result_sha256": common.sha256_file(parent_path),
        "staged": staged,
        "evaluator_sha256": common.sha256_file(evaluator),
        "source_sha256s": source_sha256s,
        "treatment": TREATMENT,
        "test_artifacts_read": False,
    }
    prior = expanded.matching_receipt(receipt_path, result_path, expected)
    if prior is not None:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        validate_result(result, dataset_id, seed, view)
        return {"result": result, "receipt": prior, "reused": True}

    recovered = recover_completed_result(root, dataset_id, seed, view)
    if recovered is not None:
        if recovery_launcher_sha256 is None:
            raise ValueError(
                "complete temporary result exists but no recovery launcher authority was supplied"
            )
        if len(recovery_launcher_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in recovery_launcher_sha256
        ):
            raise ValueError("recovery launcher authority must be a lowercase SHA-256")
        result, temporary_path = recovered
        execution_sources = dict(source_sha256s)
        launcher_key = next(
            (
                path
                for path in execution_sources
                if path.endswith("scripts/hyperion_cv_search_protected.py")
            ),
            None,
        )
        if launcher_key is None:
            raise ValueError("recovery requires the launcher in --source authority")
        execution_sources[launcher_key] = recovery_launcher_sha256
        os.replace(temporary_path, result_path)
        receipt = {
            **expected,
            "argv": [
                "run",
                "--config",
                str(config_path),
                "--train",
                str(staged_path),
                "--output",
                "<job-private-temporary-result>",
            ],
            "environment": {
                "RAYON_NUM_THREADS": str(rayon_threads),
                "JSS_EVAL_TIMEOUT_SECS": str(evaluator_timeout_seconds),
            },
            "elapsed_seconds": result.get("runtime_seconds"),
            "result_sha256": common.sha256_file(result_path),
            "recovery": {
                "kind": "completed-temporary-after-launcher-validation-failure",
                "temporary_basename": temporary_path.name,
                "execution_source_sha256s": execution_sources,
                "result_recomputed": False,
            },
        }
        receipt["receipt_sha256"] = common.canonical_sha256(receipt)
        expanded.atomic_json(receipt_path, receipt)
        return {"result": result, "receipt": receipt, "reused": True}

    temporary = result_path.with_name(f".{result_path.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    environment = os.environ.copy()
    environment["RAYON_NUM_THREADS"] = str(rayon_threads)
    environment["JSS_EVAL_TIMEOUT_SECS"] = str(evaluator_timeout_seconds)
    started = time.monotonic()
    completed = subprocess.run(
        [
            str(evaluator),
            "run",
            "--config",
            str(config_path),
            "--train",
            str(staged_path),
            "--output",
            str(temporary),
        ],
        text=True,
        capture_output=True,
        timeout=search_timeout_seconds,
        check=False,
        env=environment,
    )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        temporary.unlink(missing_ok=True)
        reason = completed.stderr.strip()[-1_000:] or f"exit={completed.returncode}"
        raise RuntimeError(f"matched {view} search failed: {reason}")
    result = json.loads(temporary.read_text(encoding="utf-8"))
    validate_result(result, dataset_id, seed, view)
    os.replace(temporary, result_path)
    receipt = {
        **expected,
        "argv": [
            "run",
            "--config",
            str(config_path),
            "--train",
            str(staged_path),
            "--output",
            "<job-private-temporary-result>",
        ],
        "environment": {
            "RAYON_NUM_THREADS": str(rayon_threads),
            "JSS_EVAL_TIMEOUT_SECS": str(evaluator_timeout_seconds),
        },
        "elapsed_seconds": elapsed,
        "result_sha256": common.sha256_file(result_path),
        "recovery": None,
    }
    receipt["receipt_sha256"] = common.canonical_sha256(receipt)
    expanded.atomic_json(receipt_path, receipt)
    return {"result": result, "receipt": receipt, "reused": False}


def paired_view_evidence(
    evaluator: Path,
    raw_genome: str,
    cell_genome: str,
    raw: np.ndarray,
    targets: np.ndarray,
    groups: Sequence[str],
    cell: np.ndarray,
    folds: Sequence[common.Fold],
    timeout_seconds: int,
) -> list[common.FoldEvidence]:
    evidence: list[common.FoldEvidence] = []
    with tempfile.TemporaryDirectory(
        prefix="jope-hyperion-cv-search-gate-"
    ) as temporary:
        root = Path(temporary)
        for fold in folds:
            transformed = common.transform_cell(cell, fold.fit_rows)
            raw_path = root / f"fold-{fold.index}-raw.csv"
            cell_path = root / f"fold-{fold.index}-cell.csv"
            common.stage_csv(raw_path, raw, targets, fold.hold_rows)
            common.stage_csv(
                cell_path,
                np.concatenate((raw, transformed), axis=1),
                targets,
                fold.hold_rows,
            )
            raw_predictions = common.evaluate_predictions(
                evaluator, raw_genome, raw_path, fold, timeout_seconds
            )
            cell_predictions = common.evaluate_predictions(
                evaluator, cell_genome, cell_path, fold, timeout_seconds
            )
            hold = list(fold.hold_rows)
            evidence.append(
                common.FoldEvidence(
                    index=fold.index,
                    row_groups=[groups[row] for row in hold],
                    actual=[float(targets[row]) for row in hold],
                    raw=raw_predictions,
                    arms={"hyperion_cell": cell_predictions},
                )
            )
    return evidence


def paired_gate(
    dataset_id: str,
    quant_root: Path,
    evaluator: Path,
    raw_genome: str,
    cell_genome: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    root = quant_root / dataset_id
    train_path = checked_train_path(root, "train.csv")
    cell_path = checked_train_path(root, "train3.vs")
    raw, targets, groups = common.read_training_csv(train_path)
    cell = common.read_matrix(cell_path, raw.shape[0], raw.shape[1])
    thresholds = common.Thresholds()
    cv5_folds = common.grouped_folds(
        groups, 5, common.CV5_SEED, thresholds.minimum_atomic_groups
    )
    cv5 = common.evaluate_arm(
        "cv5",
        paired_view_evidence(
            evaluator,
            raw_genome,
            cell_genome,
            raw,
            targets,
            groups,
            cell,
            cv5_folds,
            timeout_seconds,
        ),
        "hyperion_cell",
        thresholds,
    )
    cv7 = None
    if cv5["passed"]:
        cv7_folds = common.grouped_folds(
            groups, 7, common.CV7_SEED, thresholds.minimum_atomic_groups
        )
        cv7 = common.evaluate_arm(
            "cv7",
            paired_view_evidence(
                evaluator,
                raw_genome,
                cell_genome,
                raw,
                targets,
                groups,
                cell,
                cv7_folds,
                timeout_seconds,
            ),
            "hyperion_cell",
            thresholds,
        )
    selected = bool(cv7 and cv7["passed"])
    return {
        "raw_genome_sha256": common.canonical_sha256(raw_genome),
        "cell_genome_sha256": common.canonical_sha256(cell_genome),
        "train_sha256": common.sha256_file(train_path),
        "cell_sha256": common.sha256_file(cell_path),
        "cv5": cv5,
        "cv7": cv7,
        "selected_view": "hyperion_cell" if selected else "raw",
        "selected_log_rmse_improvement": (
            math.log(cv7["pooled_raw_rmse"] / cv7["pooled_arm_rmse"])
            if selected
            else 0.0
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--round2-root", required=True, type=Path)
    parser.add_argument("--parent-search-root", required=True, type=Path)
    parser.add_argument("--quant-root", required=True, type=Path)
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--source", required=True, action="append", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=EXPECTED_WORKERS)
    parser.add_argument("--rayon-threads", type=int, default=EXPECTED_RAYON_THREADS)
    parser.add_argument(
        "--evaluator-timeout-seconds",
        type=int,
        default=EXPECTED_EVALUATOR_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--search-timeout-seconds",
        type=int,
        default=EXPECTED_SEARCH_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--gate-timeout-seconds", type=int, default=EXPECTED_GATE_TIMEOUT_SECONDS
    )
    parser.add_argument("--recovery-launcher-sha256")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        min(
            args.workers,
            args.rayon_threads,
            args.evaluator_timeout_seconds,
            args.search_timeout_seconds,
            args.gate_timeout_seconds,
        )
        < 1
    ):
        raise ValueError("worker, thread, and timeout values must be positive")
    actual_execution_policy = {
        "workers": args.workers,
        "rayon_threads": args.rayon_threads,
        "candidate_timeout_seconds": args.evaluator_timeout_seconds,
        "search_timeout_seconds": args.search_timeout_seconds,
        "gate_timeout_seconds": args.gate_timeout_seconds,
    }
    expected_execution_policy = {
        "workers": EXPECTED_WORKERS,
        "rayon_threads": EXPECTED_RAYON_THREADS,
        "candidate_timeout_seconds": EXPECTED_EVALUATOR_TIMEOUT_SECONDS,
        "search_timeout_seconds": EXPECTED_SEARCH_TIMEOUT_SECONDS,
        "gate_timeout_seconds": EXPECTED_GATE_TIMEOUT_SECONDS,
    }
    if actual_execution_policy != expected_execution_policy:
        raise ValueError(
            "campaign execution policy drift: "
            f"expected {expected_execution_policy}, got {actual_execution_policy}"
        )
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    dataset_ids = expanded.protected_ids(protocol)
    evaluator = args.evaluator.resolve()
    if not evaluator.is_file() or not os.access(evaluator, os.X_OK):
        raise ValueError("evaluator must be executable")
    source_sha256s = {str(path): common.sha256_file(path) for path in args.source}
    jobs = [(dataset_id, seed) for dataset_id in dataset_ids for seed in SEEDS]
    outcomes: dict[tuple[str, int], dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []

    def run_pair(dataset_id: str, seed: int) -> dict[str, Any]:
        views = {
            view: run_search(
                dataset_id,
                seed,
                view,
                args.quant_root,
                args.round2_root,
                args.parent_search_root,
                evaluator,
                args.output_root,
                source_sha256s,
                args.recovery_launcher_sha256,
                args.rayon_threads,
                args.evaluator_timeout_seconds,
                args.search_timeout_seconds,
            )
            for view in VIEWS
        }
        gate = paired_gate(
            dataset_id,
            args.quant_root,
            evaluator,
            views["raw"]["result"]["winner_genome"],
            views["hyperion_cell"]["result"]["winner_genome"],
            args.gate_timeout_seconds,
        )
        return {
            "dataset_id": dataset_id,
            "seed": seed,
            "views": {
                view: {
                    "result_sha256": views[view]["receipt"]["result_sha256"],
                    "receipt_sha256": views[view]["receipt"]["receipt_sha256"],
                    "winner_semantic": views[view]["result"]["winner_semantic"],
                    "reused": views[view]["reused"],
                }
                for view in VIEWS
            },
            "gate": gate,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {
            executor.submit(run_pair, dataset_id, seed): (dataset_id, seed)
            for dataset_id, seed in jobs
        }
        for future in as_completed(pending):
            dataset_id, seed = pending[future]
            try:
                outcomes[(dataset_id, seed)] = future.result()
                print(
                    f"matched {dataset_id} seed={seed} "
                    f"selected={outcomes[(dataset_id, seed)]['gate']['selected_view']}",
                    flush=True,
                )
            except Exception as error:
                failures.append(
                    {"dataset_id": dataset_id, "seed": seed, "reason": str(error)}
                )
                print(f"matched {dataset_id} seed={seed} failed={error}", flush=True)
    results = [outcomes[key] for key in sorted(outcomes)]
    both_seed_wins = sum(
        all(
            outcomes.get((dataset_id, seed), {}).get("gate", {}).get("selected_view")
            == "hyperion_cell"
            for seed in SEEDS
        )
        for dataset_id in dataset_ids
    )
    summary = {
        "complete_seed_dataset_pairs": len(results),
        "expected_seed_dataset_pairs": len(jobs),
        "cv5_passes": sum(result["gate"]["cv5"]["passed"] for result in results),
        "cv7_passes": sum(
            result["gate"]["selected_view"] == "hyperion_cell" for result in results
        ),
        "protected_datasets_both_seeds_confirmed": both_seed_wins,
    }
    passed = (
        not failures
        and len(results) == len(jobs)
        and both_seed_wins == len(dataset_ids)
    )
    report = {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "cohort": "development-protected",
        "protected_dataset_ids": list(dataset_ids),
        "seeds": list(SEEDS),
        "allowed_data_basenames": sorted(ALLOWED_DATA_BASENAMES),
        "test_artifacts_read": False,
        "treatment": TREATMENT,
        "execution_policy": expected_execution_policy,
        "thresholds": asdict(common.Thresholds()),
        "protocol_sha256": common.sha256_file(args.protocol),
        "evaluator_sha256": common.sha256_file(evaluator),
        "source_sha256s": source_sha256s,
        "parent_search_root": str(args.parent_search_root),
        "protected_screen_passed": passed,
        "promotion_eligible": False,
        "promotion_blocker": (
            "must expand unchanged to the remaining frontier development datasets"
            if passed
            else "matched Hyperion CV search did not pass the protected development screen"
        ),
        "results": results,
        "failures": sorted(failures, key=lambda row: (row["dataset_id"], row["seed"])),
        "summary": summary,
    }
    report["report_sha256"] = common.canonical_sha256(report)
    expanded.atomic_json(args.report, report)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if not failures and len(results) == len(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
