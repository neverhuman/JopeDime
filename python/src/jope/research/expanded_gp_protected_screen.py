#!/usr/bin/env python3
"""Train-only protected screen for the expanded GP search vocabulary.

This phase changes the search algorithm, not the data boundary.  It replays the
frozen c1000 round-2 budget on the two protocol-protected development datasets
with the expanded transform vocabulary, pooled out-of-fold scoring, and the
repeated-CV half-standard-error winner rule.  Each resulting genome is then
compared with the matching frozen round-2 winner using atomic-group CV5 and,
only after a pass, CV7.

The launcher accepts ``train.csv`` only.  It has no test-path argument and all
search/config/result identities are bound into resumable receipts.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import hyperion_train_audit as common
import raw_floor_train_audit as raw_floor


SCHEMA = "ExpandedGpProtectedScreen.v1"
RECEIPT_SCHEMA = "ExpandedGpSearchReceipt.v1"
SEEDS = (101, 202)
TREATMENT = {
    "vocab": "expanded",
    "baseline_seeds": False,
    "pooled_oof": True,
    "winner_rule": "rcv_1se",
    "drop_failed_folds": True,
    "se_band": 0.5,
}
EXPECTED_BUDGET = {
    "population": 40,
    "generations": 25,
    "splits_per_gen": 1,
    "elites": 3,
    "cv_fitness": 0,
}
SELECTION_ACCOUNTING = {
    "search_evaluations": 1000,
    "final_cv_refits": 70,
    "repeated_cv_refits": 5 * 10 * 7,
    "bootstrap_refits": 20 * 10,
    "ensemble_oof_refits": 10 * 7,
    "total_train_only_fits": 1000 + 70 + 5 * 10 * 7 + 20 * 10 + 10 * 7,
}


def selection_accounting(search_evaluations: int) -> dict[str, int]:
    accounting = dict(SELECTION_ACCOUNTING)
    accounting["search_evaluations"] = search_evaluations
    accounting["total_train_only_fits"] = (
        search_evaluations
        + accounting["final_cv_refits"]
        + accounting["repeated_cv_refits"]
        + accounting["bootstrap_refits"]
        + accounting["ensemble_oof_refits"]
    )
    return accounting


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def protected_ids(protocol: dict[str, Any]) -> tuple[str, str]:
    development = protocol.get("cohorts", {}).get("development")
    protected = protocol.get("best5_screen", {}).get("datasets")
    if not isinstance(development, list) or len(development) != 22:
        raise ValueError("protocol must contain the exact 22-dataset development cohort")
    if not isinstance(protected, list) or len(protected) != 2:
        raise ValueError("protocol must declare exactly two protected screen datasets")
    if len(set(protected)) != 2 or not set(protected).issubset(development):
        raise ValueError("protected datasets must be distinct development members")
    confirmation = set(protocol.get("cohorts", {}).get("confirmation", []))
    if confirmation.intersection(protected):
        raise ValueError("protected screen overlaps confirmation cohort")
    return protected[0], protected[1]


def round2_job_name(dataset_id: str, seed: int) -> str:
    return f"js2-c1000-{dataset_id[:12]}-s{seed}"


def base_config_path(round2_root: Path, dataset_id: str, seed: int) -> Path:
    return round2_root / "screen-jobs2" / round2_job_name(dataset_id, seed) / "config.json"


def treatment_config(base: dict[str, Any], dataset_id: str, seed: int) -> dict[str, Any]:
    if base.get("schema") != "JopeStrategyScreenConfig.v1":
        raise ValueError("unexpected round-2 config schema")
    if base.get("strategy") != "c1000":
        raise ValueError("protected treatment requires c1000 strategy")
    if base.get("dataset_hash") != dataset_id or base.get("master_seed") != seed:
        raise ValueError("round-2 config identity mismatch")
    if base.get("budget") != EXPECTED_BUDGET:
        raise ValueError("round-2 search budget drift")
    if base.get("final_cv_folds") != 7 or base.get("final_top_k") != 10:
        raise ValueError("round-2 finale budget drift")
    config = copy.deepcopy(base)
    config.update(TREATMENT)
    return config


def continuation_config(
    base: dict[str, Any],
    parent_result: dict[str, Any],
    dataset_id: str,
    seed: int,
    search_evaluations: int,
) -> dict[str, Any]:
    if search_evaluations <= 1000 or search_evaluations % EXPECTED_BUDGET["population"]:
        raise ValueError("continuation rung must be a population-aligned value above 1000")
    validate_search_result(parent_result, dataset_id, seed)
    top = parent_result.get("top")
    if not isinstance(top, list) or len(top) != 10:
        raise ValueError("continuation requires the sealed parent top-10 archive")
    archive: list[str] = []
    for member in top:
        canonical = member.get("canonical")
        if not isinstance(canonical, str) or not canonical:
            raise ValueError("parent archive contains a malformed genome")
        archive.append(canonical)
    config = treatment_config(base, dataset_id, seed)
    config["strategy"] = f"c{search_evaluations}"
    config["budget"]["generations"] = search_evaluations // config["budget"]["population"]
    config["dl_genes"] = archive
    return config


def validate_search_result(
    result: dict[str, Any],
    dataset_id: str,
    seed: int,
    strategy: str = "c1000",
    search_evaluations: int = 1000,
) -> None:
    if result.get("schema") != "JopeStrategyScreenResult.v1":
        raise ValueError("unexpected search result schema")
    if result.get("strategy") != strategy:
        raise ValueError("search result strategy drift")
    if result.get("dataset_hash") != dataset_id or result.get("master_seed") != seed:
        raise ValueError("search result identity mismatch")
    if result.get("test_csv_read") is not False:
        raise ValueError("search result lacks train-only attestation")
    if (
        result.get("search_evaluations") != search_evaluations
        or result.get("final_cv_evaluations") != 70
    ):
        raise ValueError("search result budget drift")
    if not isinstance(result.get("winner_genome"), str) or not result["winner_genome"]:
        raise ValueError("search result has no winner genome")
    if not isinstance(result.get("winner_semantic"), str) or not result["winner_semantic"]:
        raise ValueError("search result has no winner semantic")


def matching_receipt(
    receipt_path: Path,
    result_path: Path,
    expected: dict[str, Any],
) -> dict[str, Any] | None:
    if not receipt_path.is_file() or not result_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if any(receipt.get(key) != value for key, value in expected.items()):
        return None
    if receipt.get("result_sha256") != common.sha256_file(result_path):
        return None
    claimed_hash = receipt.get("receipt_sha256")
    unhashed = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if claimed_hash != common.canonical_sha256(unhashed):
        return None
    return receipt


def run_search(
    dataset_id: str,
    seed: int,
    quant_root: Path,
    round2_root: Path,
    evaluator: Path,
    output_root: Path,
    source_hashes: dict[str, str],
    rayon_threads: int,
    evaluator_timeout_seconds: int,
    search_timeout_seconds: int,
    search_evaluations: int,
    parent_search_root: Path | None,
) -> dict[str, Any]:
    train_path = common.checked_data_path(quant_root / dataset_id, "train.csv")
    base_path = base_config_path(round2_root, dataset_id, seed)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    parent_result_path: Path | None = None
    parent_receipt_path: Path | None = None
    parent_result: dict[str, Any] | None = None
    if search_evaluations == 1000:
        if parent_search_root is not None:
            raise ValueError("the first rung must not declare a parent search")
        config = treatment_config(base, dataset_id, seed)
    else:
        if parent_search_root is None:
            raise ValueError("continuation rung requires a parent search root")
        parent_result_path = (
            parent_search_root / "jobs" / dataset_id / f"seed-{seed}" / "result.json"
        )
        parent_receipt_path = (
            parent_search_root / "jobs" / dataset_id / f"seed-{seed}" / "receipt.v1.json"
        )
        parent_result = json.loads(parent_result_path.read_text(encoding="utf-8"))
        parent_receipt = json.loads(parent_receipt_path.read_text(encoding="utf-8"))
        if parent_receipt.get("result_sha256") != common.sha256_file(parent_result_path):
            raise ValueError("parent receipt does not bind its search result")
        config = continuation_config(
            base, parent_result, dataset_id, seed, search_evaluations
        )
    job_root = output_root / "jobs" / dataset_id / f"seed-{seed}"
    config_path = job_root / "config.json"
    result_path = job_root / "result.json"
    receipt_path = job_root / "receipt.v1.json"
    atomic_json(config_path, config)
    expected = {
        "schema": RECEIPT_SCHEMA,
        "dataset_id": dataset_id,
        "seed": seed,
        "config_sha256": common.sha256_file(config_path),
        "base_config_sha256": common.sha256_file(base_path),
        "train_sha256": common.sha256_file(train_path),
        "evaluator_sha256": common.sha256_file(evaluator),
        "source_sha256s": source_hashes,
        "treatment": TREATMENT,
        "test_artifacts_read": False,
    }
    if parent_result_path is not None and parent_receipt_path is not None:
        expected.update(
            {
                "campaign_id": f"expanded-gp-r{search_evaluations}-protected-v1",
                "parent_result_sha256": common.sha256_file(parent_result_path),
                "parent_receipt_sha256": common.sha256_file(parent_receipt_path),
            }
        )
    prior = matching_receipt(receipt_path, result_path, expected)
    if prior is not None:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        validate_search_result(
            result,
            dataset_id,
            seed,
            strategy=config["strategy"],
            search_evaluations=search_evaluations,
        )
        return {"result": result, "receipt": prior, "reused": True}

    temporary_result = result_path.with_name(f".{result_path.name}.tmp.{os.getpid()}")
    temporary_result.unlink(missing_ok=True)
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
            str(train_path),
            "--output",
            str(temporary_result),
        ],
        text=True,
        capture_output=True,
        timeout=search_timeout_seconds,
        check=False,
        env=environment,
    )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        temporary_result.unlink(missing_ok=True)
        reason = completed.stderr.strip()[-1000:] or f"exit={completed.returncode}"
        raise RuntimeError(f"expanded search failed: {reason}")
    result = json.loads(temporary_result.read_text(encoding="utf-8"))
    validate_search_result(
        result,
        dataset_id,
        seed,
        strategy=config["strategy"],
        search_evaluations=search_evaluations,
    )
    os.replace(temporary_result, result_path)
    receipt = {
        **expected,
        "argv": [
            "run",
            "--config",
            str(config_path),
            "--train",
            str(train_path),
            "--output",
            "<job-private-temporary-result>",
        ],
        "environment": {
            "RAYON_NUM_THREADS": str(rayon_threads),
            "JSS_EVAL_TIMEOUT_SECS": str(evaluator_timeout_seconds),
        },
        "elapsed_seconds": elapsed,
        "result_sha256": common.sha256_file(result_path),
        "search_runtime_seconds": result.get("runtime_seconds"),
    }
    receipt["receipt_sha256"] = common.canonical_sha256(receipt)
    atomic_json(receipt_path, receipt)
    return {"result": result, "receipt": receipt, "reused": False}


def paired_gate(
    dataset_id: str,
    seed: int,
    quant_root: Path,
    round2_root: Path,
    evaluator: Path,
    treatment_genome: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    train_path = common.checked_data_path(quant_root / dataset_id, "train.csv")
    raw, targets, groups = common.read_training_csv(train_path)
    control_genome, control_sha256 = raw_floor.frozen_winner(round2_root, dataset_id, seed)
    thresholds = common.Thresholds()
    cv5_folds = common.grouped_folds(
        groups, 5, common.CV5_SEED, thresholds.minimum_atomic_groups
    )
    cv5_evidence = raw_floor.evaluate_pair(
        evaluator,
        control_genome,
        treatment_genome,
        raw,
        targets,
        groups,
        cv5_folds,
        timeout_seconds,
    )
    cv5 = common.evaluate_arm("cv5", cv5_evidence, "candidate", thresholds)
    cv7: dict[str, Any] | None = None
    if cv5["passed"]:
        cv7_folds = common.grouped_folds(
            groups, 7, common.CV7_SEED, thresholds.minimum_atomic_groups
        )
        cv7_evidence = raw_floor.evaluate_pair(
            evaluator,
            control_genome,
            treatment_genome,
            raw,
            targets,
            groups,
            cv7_folds,
            timeout_seconds,
        )
        cv7 = common.evaluate_arm("cv7", cv7_evidence, "candidate", thresholds)
    selected = bool(cv7 and cv7["passed"])
    evidence = cv7 if selected else cv5
    return {
        "control_round2_result_sha256": control_sha256,
        "control_genome_sha256": common.canonical_sha256(control_genome),
        "treatment_genome_sha256": common.canonical_sha256(treatment_genome),
        "cv5": cv5,
        "cv7": cv7,
        "selected": "expanded_gp" if selected else "round2_raw",
        "selected_log_rmse_improvement": (
            float(math.log(evidence["pooled_raw_rmse"] / evidence["pooled_arm_rmse"]))
            if selected
            else 0.0
        ),
    }


def summarize(results: Sequence[dict[str, Any]], expected_jobs: int) -> dict[str, Any]:
    cv5_passes = sum(result["gate"]["cv5"]["passed"] for result in results)
    cv7_passes = sum(bool(result["gate"]["cv7"] and result["gate"]["cv7"]["passed"]) for result in results)
    observed = [result["gate"]["cv5"]["relative_improvement"] for result in results]
    selected = [result["gate"]["selected_log_rmse_improvement"] for result in results]
    datasets: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        datasets.setdefault(result["dataset_id"], []).append(result)
    return {
        "complete_searches": len(results),
        "expected_searches": expected_jobs,
        "reused_searches": sum(result["search_reused"] for result in results),
        "cv5_seed_passes": cv5_passes,
        "cv7_seed_passes": cv7_passes,
        "protected_datasets_both_seeds_confirmed": sum(
            len(rows) == len(SEEDS) and all(row["gate"]["selected"] == "expanded_gp" for row in rows)
            for rows in datasets.values()
        ),
        "maximum_cv5_relative_improvement": max(observed, default=None),
        "median_cv5_relative_improvement": statistics.median(observed) if observed else None,
        "mean_selected_log_rmse_improvement": statistics.mean(selected) if selected else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--round2-root", required=True, type=Path)
    parser.add_argument("--quant-root", required=True, type=Path)
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--source", required=True, action="append", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rayon-threads", type=int, default=24)
    parser.add_argument("--evaluator-timeout-seconds", type=int, default=30)
    parser.add_argument("--search-timeout-seconds", type=int, default=14_400)
    parser.add_argument("--gate-timeout-seconds", type=int, default=120)
    parser.add_argument("--rung-evaluations", type=int, default=1000)
    parser.add_argument("--parent-search-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(
        args.workers,
        args.rayon_threads,
        args.evaluator_timeout_seconds,
        args.search_timeout_seconds,
        args.gate_timeout_seconds,
    ) < 1:
        raise ValueError("worker, thread, and timeout values must be positive")
    if args.rung_evaluations < 1000 or args.rung_evaluations % EXPECTED_BUDGET["population"]:
        raise ValueError("rung evaluations must be population-aligned and at least 1000")
    if (args.rung_evaluations == 1000) != (args.parent_search_root is None):
        raise ValueError("only continuation rungs require --parent-search-root")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    dataset_ids = protected_ids(protocol)
    evaluator = args.evaluator.resolve()
    if not evaluator.is_file() or not os.access(evaluator, os.X_OK):
        raise ValueError("evaluator must be executable")
    source_hashes = {str(path): common.sha256_file(path) for path in args.source}
    jobs = [(dataset_id, seed) for dataset_id in dataset_ids for seed in SEEDS]
    searched: list[tuple[str, int, dict[str, Any]]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {
            executor.submit(
                run_search,
                dataset_id,
                seed,
                args.quant_root,
                args.round2_root,
                evaluator,
                args.output_root,
                source_hashes,
                args.rayon_threads,
                args.evaluator_timeout_seconds,
                args.search_timeout_seconds,
                args.rung_evaluations,
                args.parent_search_root,
            ): (dataset_id, seed)
            for dataset_id, seed in jobs
        }
        for future in as_completed(pending):
            dataset_id, seed = pending[future]
            try:
                outcome = future.result()
                searched.append((dataset_id, seed, outcome))
                print(
                    f"search {dataset_id} seed={seed} reused={outcome['reused']} "
                    f"runtime={outcome['result'].get('runtime_seconds')}",
                    flush=True,
                )
            except Exception as error:
                failures.append({"dataset_id": dataset_id, "seed": seed, "phase": "search", "reason": str(error)})
                print(f"search {dataset_id} seed={seed} failed={error}", flush=True)

    results: list[dict[str, Any]] = []
    for dataset_id, seed, outcome in sorted(searched):
        try:
            gate = paired_gate(
                dataset_id,
                seed,
                args.quant_root,
                args.round2_root,
                evaluator,
                outcome["result"]["winner_genome"],
                args.gate_timeout_seconds,
            )
            result = {
                "dataset_id": dataset_id,
                "seed": seed,
                "search_result_sha256": outcome["receipt"]["result_sha256"],
                "search_receipt_sha256": outcome["receipt"]["receipt_sha256"],
                "search_runtime_seconds": outcome["result"].get("runtime_seconds"),
                "search_reused": outcome["reused"],
                "winner_semantic": outcome["result"]["winner_semantic"],
                "gate": gate,
            }
            results.append(result)
            print(
                f"gate {dataset_id} seed={seed} cv5={gate['cv5']['relative_improvement']:.6f} "
                f"selected={gate['selected']}",
                flush=True,
            )
        except Exception as error:
            failures.append({"dataset_id": dataset_id, "seed": seed, "phase": "gate", "reason": str(error)})
            print(f"gate {dataset_id} seed={seed} failed={error}", flush=True)
    results.sort(key=lambda result: (result["dataset_id"], result["seed"]))
    failures.sort(key=lambda failure: (failure["dataset_id"], failure["seed"], failure["phase"]))
    summary = summarize(results, len(jobs))
    passed_screen = (
        not failures
        and summary["complete_searches"] == len(jobs)
        and summary["protected_datasets_both_seeds_confirmed"] == len(dataset_ids)
    )
    report = {
        "schema": SCHEMA,
        "campaign_id": f"expanded-gp-r{args.rung_evaluations}-protected-v1",
        "cohort": "development-protected",
        "protected_dataset_ids": list(dataset_ids),
        "seeds": list(SEEDS),
        "test_artifacts_read": False,
        "allowed_data_basenames": ["train.csv"],
        "promotion_eligible": False,
        "protected_screen_passed": passed_screen,
        "promotion_blocker": (
            "must expand unchanged to the remaining five frontier development datasets"
            if passed_screen
            else "expanded GP treatment did not pass the protected development screen"
        ),
        "treatment": TREATMENT,
        "budget": {
            **EXPECTED_BUDGET,
            "generations": args.rung_evaluations // EXPECTED_BUDGET["population"],
        },
        "evaluation_accounting_per_search": selection_accounting(args.rung_evaluations),
        "parent_search_root": str(args.parent_search_root) if args.parent_search_root else None,
        "protocol_sha256": common.sha256_file(args.protocol),
        "evaluator_sha256": common.sha256_file(evaluator),
        "source_sha256s": source_hashes,
        "thresholds": asdict(common.Thresholds()),
        "results": results,
        "failures": failures,
        "summary": summary,
    }
    report["report_sha256"] = common.canonical_sha256(report)
    atomic_json(args.report, report)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if not failures and len(results) == len(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
