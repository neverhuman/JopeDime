#!/usr/bin/env python3
"""One-way train-only CV7 benchmark for the frozen Hyperion cell policy.

The policy is frozen before any confirmation-table metric is computed.  Two
development-selected, production-compatible GP genomes form a fixed 50/50
ensemble.  On two predeclared grouped CV7 partitions, the script compares the
same ensemble on raw features and raw plus fold-fit ECDF Hyperion cell features.

This is a train-only estimate, not a held-out test claim.  The only admitted
dataset files are ``train.csv`` and an encoder-authorized ``train3.vs``.  Tiny
or duplicate-heavy datasets that cannot support honest CV7 retain raw without
running the evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import backfill_hyperion_stale_confirmation_cells as backfill
import expanded_gp_protected_screen as expanded
import hyperion_train_audit as common


SCHEMA = "HyperionFinalTrainCv7Benchmark.v1"
POLICY_SCHEMA = "HyperionFinalTrainCv7PolicyFreeze.v1"
SEED_RECEIPT_SCHEMA = "HyperionFinalTrainCv7SeedReceipt.v1"
CAMPAIGN_ID = "hyperion-final-fixed-ensemble-cv7-confirmation-train-v1"
CONFIRMATION_SEEDS = (303, 404)
MINIMUM_FIT_ROWS = 32
MINIMUM_HOLD_ROWS = 2
MINIMUM_HOLD_GROUPS = 2
EXPERTS = (
    {
        "name": "development-global-gpr-rbf-linear",
        "semantic": "0669959f71f2d237450d43b9952e81f23b5d75674df60a9cf986829bd30738e8",
        "genome": "(pipe X (gpr kernel=rbf+linear m=300 steps=40))",
        "weight": 0.5,
    },
    {
        "name": "development-robust-core-portfolio",
        "semantic": "36f97d4d30053fb2815945c912175404c078778cea29883699d914121570dc38",
        "genome": "(pipe X (standardize) (portfolio blend=nnls folds=5 preset=core repeats=1))",
        "weight": 0.5,
    },
)
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--coverage", required=True, type=Path)
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--development-prior", required=True, type=Path)
    parser.add_argument("--quant-root", required=True, type=Path)
    parser.add_argument("--authority-root", required=True, type=Path)
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--source", required=True, action="append", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    return parser.parse_args()


def confirmation_ids(protocol: dict[str, Any]) -> list[str]:
    values = protocol.get("cohorts", {}).get("confirmation")
    if (
        protocol.get("schema") != "JopeDlgpBudgetFirstProtocol.v1"
        or not isinstance(values, list)
        or len(values) != 20
        or len(set(values)) != 20
        or any(not isinstance(value, str) or not value for value in values)
    ):
        raise ValueError("confirmation cohort is incomplete or foreign")
    return values


def validate_expert_authority(prior: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(prior, dict) or len(prior) != 22:
        raise ValueError("development leave-one-dataset prior is incomplete")
    evidence: dict[str, dict[str, Any]] = {}
    top_counts: dict[str, int] = {}
    for entries in prior.values():
        if not isinstance(entries, list) or not entries:
            raise ValueError("development prior contains an empty ranking")
        semantic = entries[0].get("semantic")
        top_counts[semantic] = top_counts.get(semantic, 0) + 1
        for entry in entries:
            row = evidence.setdefault(
                entry.get("semantic"),
                {"genomes": set(), "mean_pct": [], "worst_pct": [], "appearances": 0},
            )
            row["genomes"].add(entry.get("genome"))
            row["mean_pct"].append(float(entry.get("mean_pct")))
            row["worst_pct"].append(float(entry.get("worst_pct")))
            row["appearances"] += 1
    output: dict[str, Any] = {}
    for expert in EXPERTS:
        row = evidence.get(expert["semantic"])
        if row is None or row["genomes"] != {expert["genome"]}:
            raise ValueError(
                f"development authority is missing expert {expert['name']}"
            )
        output[expert["name"]] = {
            "semantic": expert["semantic"],
            "top_rank_count": top_counts.get(expert["semantic"], 0),
            "appearances": row["appearances"],
            "median_mean_percentile": statistics.median(row["mean_pct"]),
            "median_worst_percentile": statistics.median(row["worst_pct"]),
        }
    if output[EXPERTS[0]["name"]]["top_rank_count"] != 21:
        raise ValueError("primary expert no longer has its frozen 21/22 rank authority")
    return output


def authority_maps(
    coverage: dict[str, Any], authority: dict[str, Any], expected_ids: Sequence[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if (
        coverage.get("schema") != "HyperionPrecomputedCellCoverage.v1"
        or coverage.get("confirmation_cell_available_dataset_count") != 20
        or coverage.get("test_artifacts_read") is not False
        or authority.get("schema") != "HyperionConfirmationCellAuthority.v2"
        or authority.get("encoder_authorized_dataset_count") != 20
        or authority.get("raw_fallback_count") != 0
        or authority.get("test_artifacts_read") is not False
    ):
        raise ValueError("Hyperion cell coverage or encoder authority is incomplete")
    coverage_rows = {
        row["dataset_id"]: row
        for row in coverage.get("datasets", [])
        if row.get("cohort") == "confirmation"
    }
    authority_rows = {row["dataset_id"]: row for row in authority.get("datasets", [])}
    if set(coverage_rows) != set(expected_ids) or set(authority_rows) != set(
        expected_ids
    ):
        raise ValueError(
            "cell authority cohort differs from the frozen confirmation cohort"
        )
    return coverage_rows, authority_rows


def materialize_train_only_overlay(
    dataset_ids: Sequence[str],
    coverage_rows: dict[str, dict[str, Any]],
    authority_rows: dict[str, dict[str, Any]],
    quant_root: Path,
    authority_root: Path,
    overlay_root: Path,
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        coverage = coverage_rows[dataset_id]
        authority = authority_rows[dataset_id]
        train_source = quant_root / dataset_id / "train.csv"
        cell_root = (
            authority_root if authority["source"] == "campaign-backfill" else quant_root
        )
        cell_source = cell_root / authority["relative_path"]
        if (
            not train_source.is_file()
            or not cell_source.is_file()
            or common.sha256_file(train_source) != coverage["train_sha256"]
            or common.sha256_file(cell_source) != authority["cell_sha256"]
        ):
            raise ValueError(f"{dataset_id}: train or cell bytes differ from authority")
        destination = overlay_root / dataset_id
        backfill.immutable_write(destination / "train.csv", train_source.read_bytes())
        backfill.immutable_write(destination / "train3.vs", cell_source.read_bytes())
        receipts.append(
            {
                "dataset_id": dataset_id,
                "rows": coverage["rows"],
                "raw_features": coverage["raw_features"],
                "train_sha256": coverage["train_sha256"],
                "cell_sha256": authority["cell_sha256"],
                "cell_source": authority["source"],
                "allowed_basenames": ["train.csv", "train3.vs"],
                "test_artifacts_read": False,
            }
        )
    return receipts


def fold_plan(groups: Sequence[str], seed: int) -> list[common.Fold]:
    plan = common.grouped_folds(groups, 7, seed, minimum_groups=14)
    for fold in plan:
        if (
            len(fold.fit_rows) < MINIMUM_FIT_ROWS
            or len(fold.hold_rows) < MINIMUM_HOLD_ROWS
            or len(fold.hold_groups) < MINIMUM_HOLD_GROUPS
        ):
            raise ValueError(
                f"fold {fold.index} is too small: fit={len(fold.fit_rows)} "
                f"hold={len(fold.hold_rows)} groups={len(fold.hold_groups)}"
            )
    return plan


def weighted_average(values: Sequence[Sequence[float]]) -> list[float]:
    if len(values) != len(EXPERTS) or len({len(row) for row in values}) != 1:
        raise ValueError("expert prediction coverage is incomplete")
    return [
        sum(
            float(EXPERTS[index]["weight"]) * values[index][row]
            for index in range(len(EXPERTS))
        )
        for row in range(len(values[0]))
    ]


def pooled_improvement(
    folds: Sequence[common.FoldEvidence], expert_name: str
) -> dict[str, Any]:
    actual: list[float] = []
    raw: list[float] = []
    cell: list[float] = []
    fold_wins = 0
    arm = f"{expert_name}:cell"
    raw_key = f"{expert_name}:raw"
    for fold in folds:
        raw_fold = fold.arms[raw_key]
        cell_fold = fold.arms[arm]
        fold_wins += int(
            common.rmse(fold.actual, cell_fold) < common.rmse(fold.actual, raw_fold)
        )
        actual.extend(fold.actual)
        raw.extend(raw_fold)
        cell.extend(cell_fold)
    raw_rmse = common.rmse(actual, raw)
    cell_rmse = common.rmse(actual, cell)
    return {
        "raw_rmse": raw_rmse,
        "cell_rmse": cell_rmse,
        "relative_improvement": 0.0
        if raw_rmse <= 1e-300
        else 1.0 - cell_rmse / raw_rmse,
        "fold_wins": fold_wins,
    }


def evaluate_seed(
    dataset_id: str,
    seed: int,
    overlay_root: Path,
    evaluator: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    root = overlay_root / dataset_id
    train_path = root / "train.csv"
    cell_path = root / "train3.vs"
    raw, targets, groups = common.read_training_csv(train_path)
    cell = common.read_matrix(cell_path, raw.shape[0], raw.shape[1])
    try:
        folds = fold_plan(groups, common.seed64((CAMPAIGN_ID, "cv7", str(seed))))
    except ValueError as error:
        return {
            "schema": SEED_RECEIPT_SCHEMA,
            "dataset_id": dataset_id,
            "seed": seed,
            "status": "structural_raw_fallback",
            "reason": str(error),
            "selected_view": "raw",
            "test_artifacts_read": False,
        }
    evidence: list[common.FoldEvidence] = []
    started = time.monotonic()
    with tempfile.TemporaryDirectory(
        prefix=f"hyperion-final-cv7-{dataset_id}-{seed}-"
    ) as temporary:
        temporary_root = Path(temporary)
        for fold in folds:
            transformed = common.transform_cell(cell, fold.fit_rows)
            feature_views = {
                "raw": raw,
                "cell": np.concatenate((raw, transformed), axis=1),
            }
            predictions: dict[str, list[list[float]]] = {"raw": [], "cell": []}
            expert_arms: dict[str, list[float]] = {}
            for view, features in feature_views.items():
                staged = temporary_root / f"fold-{fold.index}-{view}.csv"
                common.stage_csv(staged, features, targets, fold.hold_rows)
                for expert in EXPERTS:
                    values = common.evaluate_predictions(
                        evaluator, expert["genome"], staged, fold, timeout_seconds
                    )
                    predictions[view].append(values)
                    expert_arms[f"{expert['name']}:{view}"] = values
            hold = list(fold.hold_rows)
            evidence.append(
                common.FoldEvidence(
                    index=fold.index,
                    row_groups=[groups[row] for row in hold],
                    actual=[float(targets[row]) for row in hold],
                    raw=weighted_average(predictions["raw"]),
                    arms={
                        "hyperion_cell": weighted_average(predictions["cell"]),
                        **expert_arms,
                    },
                )
            )
    gate = common.evaluate_arm("cv7", evidence, "hyperion_cell", common.Thresholds())
    experts = {
        expert["name"]: pooled_improvement(evidence, expert["name"])
        for expert in EXPERTS
    }
    consensus = all(row["relative_improvement"] >= 0.0 for row in experts.values())
    selected = bool(gate["passed"] and consensus)
    gate["expert_consensus_nonnegative"] = consensus
    gate["policy_passed"] = selected
    return {
        "schema": SEED_RECEIPT_SCHEMA,
        "dataset_id": dataset_id,
        "seed": seed,
        "status": "complete",
        "fold_plan_seed": common.seed64((CAMPAIGN_ID, "cv7", str(seed))),
        "fold_count": 7,
        "rows": int(raw.shape[0]),
        "raw_features": int(raw.shape[1]),
        "cell_features": int(cell.shape[1]),
        "train_sha256": common.sha256_file(train_path),
        "cell_sha256": common.sha256_file(cell_path),
        "gate": gate,
        "experts": experts,
        "selected_view": "hyperion_cell" if selected else "raw",
        "elapsed_seconds": time.monotonic() - started,
        "test_artifacts_read": False,
    }


def freeze_policy(
    args: argparse.Namespace,
    dataset_ids: Sequence[str],
    expert_authority: dict[str, Any],
) -> dict[str, Any]:
    policy: dict[str, Any] = {
        "schema": POLICY_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "confirmation_dataset_ids": list(dataset_ids),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "experts": list(EXPERTS),
        "expert_authority": expert_authority,
        "ensemble": {
            "kind": "fixed_weighted_mean",
            "weights_selected_on_confirmation": False,
        },
        "cell_preprocessing": "fold_train_empirical_percentile_per_column.v1",
        "folds": 7,
        "minimum_fit_rows": MINIMUM_FIT_ROWS,
        "minimum_hold_rows": MINIMUM_HOLD_ROWS,
        "minimum_hold_groups": MINIMUM_HOLD_GROUPS,
        "gate_thresholds": asdict(common.Thresholds()),
        "extra_activation_rule": "every constituent expert has nonnegative pooled cell lift",
        "raw_fallback": "any structural, evaluator, shape, finiteness, authority, or gate failure",
        "interpretation": "paired train-only CV estimate; never a held-out test win claim",
        "input_sha256s": {
            "protocol": common.sha256_file(args.protocol),
            "coverage": common.sha256_file(args.coverage),
            "authority": common.sha256_file(args.authority),
            "development_prior": common.sha256_file(args.development_prior),
            "evaluator": common.sha256_file(args.evaluator),
        },
        "source_sha256s": {str(path): common.sha256_file(path) for path in args.source},
        "execution": {
            "workers": args.workers,
            "timeout_seconds": args.timeout_seconds,
            "environment": THREAD_ENVIRONMENT,
        },
        "test_artifacts_read": False,
    }
    policy["policy_sha256"] = common.canonical_sha256(policy)
    if args.policy.exists():
        existing = json.loads(args.policy.read_bytes())
        if existing != policy:
            raise ValueError(
                "existing final CV7 policy differs from the requested freeze"
            )
    else:
        expanded.atomic_json(args.policy, policy)
    return policy


def main() -> int:
    args = parse_args()
    if args.workers != 4 or args.timeout_seconds != 300:
        raise ValueError(
            "final benchmark execution policy is frozen at workers=4, timeout=300"
        )
    for name, value in THREAD_ENVIRONMENT.items():
        os.environ[name] = value
    protocol = json.loads(args.protocol.read_bytes())
    coverage = json.loads(args.coverage.read_bytes())
    authority = json.loads(args.authority.read_bytes())
    prior = json.loads(args.development_prior.read_bytes())
    dataset_ids = confirmation_ids(protocol)
    expert_authority = validate_expert_authority(prior)
    coverage_rows, authority_rows = authority_maps(coverage, authority, dataset_ids)
    evaluator = args.evaluator.resolve()
    if not evaluator.is_file() or not os.access(evaluator, os.X_OK):
        raise ValueError("evaluator is not executable")
    policy = freeze_policy(args, dataset_ids, expert_authority)
    overlay = materialize_train_only_overlay(
        dataset_ids,
        coverage_rows,
        authority_rows,
        args.quant_root,
        args.authority_root,
        args.output_root / "train-only-overlay",
    )

    results: dict[tuple[str, int], dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    jobs = [
        (dataset_id, seed) for dataset_id in dataset_ids for seed in CONFIRMATION_SEEDS
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {
            executor.submit(
                evaluate_seed,
                dataset_id,
                seed,
                args.output_root / "train-only-overlay",
                evaluator,
                args.timeout_seconds,
            ): (dataset_id, seed)
            for dataset_id, seed in jobs
        }
        for future in as_completed(pending):
            dataset_id, seed = pending[future]
            try:
                result = future.result()
            except Exception as error:
                result = {
                    "schema": SEED_RECEIPT_SCHEMA,
                    "dataset_id": dataset_id,
                    "seed": seed,
                    "status": "evaluation_raw_fallback",
                    "reason": str(error),
                    "selected_view": "raw",
                    "test_artifacts_read": False,
                }
                failures.append(
                    {"dataset_id": dataset_id, "seed": seed, "reason": str(error)}
                )
            result["receipt_sha256"] = common.canonical_sha256(result)
            results[(dataset_id, seed)] = result
            expanded.atomic_json(
                args.output_root
                / "seed-receipts"
                / dataset_id
                / f"seed-{seed}.v1.json",
                result,
            )
            print(
                f"cv7 {dataset_id} seed={seed} status={result['status']} "
                f"selected={result['selected_view']}",
                flush=True,
            )

    ordered = [results[key] for key in sorted(results)]
    robust_wins = [
        dataset_id
        for dataset_id in dataset_ids
        if all(
            results[(dataset_id, seed)]["selected_view"] == "hyperion_cell"
            for seed in CONFIRMATION_SEEDS
        )
    ]
    complete = [row for row in ordered if row["status"] == "complete"]
    structural = [row for row in ordered if row["status"] == "structural_raw_fallback"]
    selected = [row for row in ordered if row["selected_view"] == "hyperion_cell"]
    positive = [row for row in complete if row["gate"]["relative_improvement"] > 0.0]
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "policy_sha256": policy["policy_sha256"],
        "policy_frozen_before_metrics": True,
        "cohort": "budget-first-confirmation-train-only",
        "dataset_count": len(dataset_ids),
        "seed_count": len(CONFIRMATION_SEEDS),
        "expected_seed_receipts": len(jobs),
        "complete_seed_receipts": len(ordered),
        "evaluable_seed_receipts": len(complete),
        "structural_raw_fallback_seed_receipts": len(structural),
        "evaluation_failure_seed_receipts": len(failures),
        "selected_seed_receipts": len(selected),
        "positive_delta_seed_receipts": len(positive),
        "robust_cv7_win_dataset_ids": robust_wins,
        "robust_cv7_wins": len(robust_wins),
        "robust_cv7_win_rate_all_20": len(robust_wins) / len(dataset_ids),
        "seed_activation_rate_all_40": len(selected) / len(jobs),
        "overlay": overlay,
        "results": ordered,
        "failures": failures,
        "target_statistics_scope": "train-only paired out-of-fold metrics",
        "held_out_test_win_rate": None,
        "held_out_test_claim_available": False,
        "test_artifacts_read": False,
    }
    report["report_sha256"] = common.canonical_sha256(report)
    expanded.atomic_json(args.report, report)
    print(
        json.dumps(
            {
                "robust_cv7_wins": len(robust_wins),
                "robust_cv7_win_rate_all_20": report["robust_cv7_win_rate_all_20"],
                "seed_activation_rate_all_40": report["seed_activation_rate_all_40"],
                "evaluation_failures": len(failures),
                "test_artifacts_read": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if len(ordered) == len(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
