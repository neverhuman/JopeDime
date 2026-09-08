#!/usr/bin/env python3
"""Train-only audit of historical Hyperion feature blocks.

This is an exploratory development-cohort probe, not promotion evidence.  The
historical ``train3.vs`` and ``hyperion3train.vs`` files were encoded over the
complete training table, so they cannot substitute for the fold-local cache
receipts required by ``jope_hyperion_gate``.  The audit nevertheless answers a
useful question without reading any test artifact: does the frozen round-2 GP
predictor contain repeatable OOF lift that justifies the fold-local backfill?

Only these inputs are admitted:

* the frozen budget-first protocol;
* round-2 ``result.json`` files;
* ``train.csv``, ``train3.vs``, and ``hyperion3train.vs``;
* a pinned ``jope_strategy_screen`` evaluator.

Validation targets are retained only by this controller for scoring.  Every
evaluator subprocess receives a staged CSV in which held-fold targets are 0.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


SCHEMA = "HyperionHistoricalTrainAudit.v1"
CV5_SEED = 0x4A4F_5045_4859_5035
CV7_SEED = 0x4A4F_5045_4859_5037
BOOTSTRAP_REPLICATES = 4096
ALLOWED_DATA_FILES = frozenset({"train.csv", "train3.vs", "hyperion3train.vs"})
VIEWS = ("hyperion_cell", "hyperion_row128", "hyperion_both")


@dataclass(frozen=True)
class Thresholds:
    minimum_relative_improvement: float = 0.005
    cv5_minimum_fold_wins: int = 4
    cv5_minimum_best_arm_wins: int = 4
    cv5_bootstrap_one_sided_confidence: float = 0.90
    cv7_minimum_fold_wins: int = 5
    cv7_bootstrap_one_sided_confidence: float = 0.95
    maximum_fold_regression: float = 0.02
    minimum_atomic_groups: int = 8


@dataclass(frozen=True)
class Fold:
    index: int
    fit_rows: tuple[int, ...]
    hold_rows: tuple[int, ...]
    hold_groups: tuple[str, ...]


@dataclass
class FoldEvidence:
    index: int
    row_groups: list[str]
    actual: list[float]
    raw: list[float]
    arms: dict[str, list[float]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def checked_data_path(dataset_root: Path, name: str) -> Path:
    if name not in ALLOWED_DATA_FILES:
        raise ValueError(f"forbidden Hyperion audit input: {name}")
    path = dataset_root / name
    if path.name != name or not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_training_csv(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if path.name != "train.csv":
        raise ValueError("audit accepts train.csv only")
    features: list[list[float]] = []
    targets: list[float] = []
    groups: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row_number, fields in enumerate(csv.reader(handle), start=1):
            if len(fields) < 2:
                raise ValueError(f"{path}:{row_number}: expected features plus target")
            stripped = [field.strip() for field in fields]
            try:
                values = [float(field) for field in stripped]
            except ValueError as error:
                raise ValueError(f"{path}:{row_number}: non-numeric field") from error
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path}:{row_number}: non-finite value")
            features.append(values[:-1])
            targets.append(values[-1])
            group_bytes = "\x1f".join(stripped[:-1]).encode("utf-8")
            groups.append(hashlib.sha256(group_bytes).hexdigest())
    if not features or any(len(row) != len(features[0]) for row in features):
        raise ValueError(f"{path}: empty or ragged training matrix")
    return np.asarray(features, dtype=np.float64), np.asarray(targets), groups


def read_matrix(path: Path, rows: int, columns: int) -> np.ndarray:
    if path.name not in ALLOWED_DATA_FILES - {"train.csv"}:
        raise ValueError(f"forbidden sidecar input: {path.name}")
    matrix = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    if matrix.shape != (rows, columns):
        raise ValueError(f"{path}: shape {matrix.shape}, expected {(rows, columns)}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{path}: non-finite sidecar value")
    return matrix


def seed64(parts: Iterable[str]) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int.from_bytes(bytes.fromhex(digest[:16]), "big")


def grouped_folds(groups: Sequence[str], count: int, seed: int, minimum_groups: int = 8) -> list[Fold]:
    grouped: dict[str, list[int]] = {}
    for row, group in enumerate(groups):
        if not group:
            raise ValueError("atomic group must not be empty")
        grouped.setdefault(group, []).append(row)
    if len(grouped) < minimum_groups:
        raise ValueError("insufficient atomic groups")
    ordered = sorted(grouped.items(), key=lambda item: seed64((str(seed), item[0])))
    assignments: list[list[tuple[str, list[int]]]] = [[] for _ in range(count)]
    sizes = [0] * count
    for group, rows in ordered:
        fold_index = min(range(count), key=lambda index: (sizes[index], index))
        sizes[fold_index] += len(rows)
        assignments[fold_index].append((group, rows))
    all_rows = set(range(len(groups)))
    folds: list[Fold] = []
    for index, assigned in enumerate(assignments):
        hold = tuple(sorted(row for _, rows in assigned for row in rows))
        folds.append(
            Fold(
                index=index,
                fit_rows=tuple(sorted(all_rows - set(hold))),
                hold_rows=hold,
                hold_groups=tuple(sorted(group for group, _ in assigned)),
            )
        )
    return folds


def transform_cell(cell: np.ndarray, fit_rows: Sequence[int]) -> np.ndarray:
    fit = np.asarray(fit_rows, dtype=np.int64)
    if fit.size == 0:
        raise ValueError("fold has no fitting rows")
    sorted_cell = np.sort(cell[fit], axis=0)
    transformed_cell = np.empty_like(cell)
    denominator = 2.0 * fit.size
    for column in range(cell.shape[1]):
        values = cell[:, column]
        lower = np.searchsorted(sorted_cell[:, column], values, side="left")
        upper = np.searchsorted(sorted_cell[:, column], values, side="right")
        transformed_cell[:, column] = (lower + upper) / denominator
    return transformed_cell


def transform_blocks(cell: np.ndarray, row: np.ndarray, fit_rows: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    fit = np.asarray(fit_rows, dtype=np.int64)
    transformed_cell = transform_cell(cell, fit_rows)
    means = np.mean(row[fit], axis=0)
    scales = np.maximum(np.sqrt(np.mean(np.square(row[fit] - means), axis=0)), 1e-12)
    return transformed_cell, (row - means) / scales


def feature_view(raw: np.ndarray, cell: np.ndarray, row: np.ndarray, view: str) -> np.ndarray:
    if view == "raw":
        return raw
    if view == "hyperion_cell":
        return np.concatenate((raw, cell), axis=1)
    if view == "hyperion_row128":
        return np.concatenate((raw, row), axis=1)
    if view == "hyperion_both":
        return np.concatenate((raw, cell, row), axis=1)
    raise ValueError(f"unknown feature view {view}")


def stage_csv(path: Path, features: np.ndarray, targets: np.ndarray, hold_rows: Sequence[int]) -> None:
    staged_targets = targets.copy()
    staged_targets[np.asarray(hold_rows, dtype=np.int64)] = 0.0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for values, target in zip(features, staged_targets, strict=True):
            writer.writerow([*(format(float(value), ".17g") for value in values), format(float(target), ".17g")])


def evaluate_predictions(
    evaluator: Path,
    genome: str,
    staged_csv: Path,
    fold: Fold,
    timeout_seconds: int,
) -> list[float]:
    payload = json.dumps(
        {
            "genome": genome,
            "train_ids": list(fold.fit_rows),
            "valid_ids": list(fold.hold_rows),
            "want": "pred",
        },
        separators=(",", ":"),
    )
    completed = subprocess.run(
        [str(evaluator), "eval-one", "--train", str(staged_csv)],
        input=payload,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        reason = completed.stderr.strip()[-500:] or f"exit={completed.returncode}"
        raise RuntimeError(f"evaluator failed: {reason}")
    values = json.loads(completed.stdout)
    if len(values) != len(fold.hold_rows) or not all(math.isfinite(value) for value in values):
        raise RuntimeError("evaluator returned incomplete or non-finite predictions")
    return [float(value) for value in values]


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(actual, predicted, strict=True)) / len(actual))


def bootstrap_lower_bound(
    folds: Sequence[FoldEvidence], arm: str, confidence: float, seed: int
) -> float:
    grouped: dict[str, tuple[list[float], list[float]]] = {}
    for fold in folds:
        for index, group in enumerate(fold.row_groups):
            raw_sq, arm_sq = grouped.setdefault(group, ([], []))
            raw_sq.append((fold.actual[index] - fold.raw[index]) ** 2)
            arm_sq.append((fold.actual[index] - fold.arms[arm][index]) ** 2)
    groups = list(grouped.values())
    if len(groups) < Thresholds().minimum_atomic_groups:
        raise ValueError("insufficient groups for bootstrap")
    state = max(seed, 1)
    samples: list[float] = []
    mask = (1 << 64) - 1
    for _ in range(BOOTSTRAP_REPLICATES):
        raw_sum = arm_sum = 0.0
        rows = 0
        for _ in groups:
            state ^= (state << 13) & mask
            state ^= state >> 7
            state ^= (state << 17) & mask
            state &= mask
            raw_group, arm_group = groups[state % len(groups)]
            raw_sum += sum(raw_group)
            arm_sum += sum(arm_group)
            rows += len(raw_group)
        raw_value = math.sqrt(raw_sum / rows)
        arm_value = math.sqrt(arm_sum / rows)
        samples.append(math.log((raw_value + 1e-300) / (arm_value + 1e-300)))
    samples.sort()
    index = math.floor((1.0 - confidence) * (len(samples) - 1))
    return samples[index]


def evaluate_arm(stage: str, folds: Sequence[FoldEvidence], arm: str, thresholds: Thresholds) -> dict[str, Any]:
    actual: list[float] = []
    raw_predictions: list[float] = []
    arm_predictions: list[float] = []
    fold_wins = 0
    best_arm_wins = 0
    worst_regression = -math.inf
    for fold in folds:
        raw_fold = rmse(fold.actual, fold.raw)
        arm_fold = rmse(fold.actual, fold.arms[arm])
        fold_wins += int(arm_fold < raw_fold)
        relative_regression = math.inf if raw_fold <= 1e-300 < arm_fold else (
            0.0 if raw_fold <= 1e-300 else arm_fold / raw_fold - 1.0
        )
        worst_regression = max(worst_regression, relative_regression)
        best = min(fold.arms, key=lambda name: (rmse(fold.actual, fold.arms[name]), name))
        best_arm_wins += int(best == arm)
        actual.extend(fold.actual)
        raw_predictions.extend(fold.raw)
        arm_predictions.extend(fold.arms[arm])
    raw_value = rmse(actual, raw_predictions)
    arm_value = rmse(actual, arm_predictions)
    improvement = 0.0 if raw_value <= 1e-300 else 1.0 - arm_value / raw_value
    confidence = (
        thresholds.cv5_bootstrap_one_sided_confidence
        if stage == "cv5"
        else thresholds.cv7_bootstrap_one_sided_confidence
    )
    lower = bootstrap_lower_bound(folds, arm, confidence, seed64((stage, arm)))
    wins_pass = (
        fold_wins >= thresholds.cv5_minimum_fold_wins
        and best_arm_wins >= thresholds.cv5_minimum_best_arm_wins
        if stage == "cv5"
        else fold_wins >= thresholds.cv7_minimum_fold_wins
    )
    passed = (
        wins_pass
        and improvement >= thresholds.minimum_relative_improvement
        and lower > 0.0
        and worst_regression <= thresholds.maximum_fold_regression
    )
    return {
        "stage": stage,
        "arm": arm,
        "folds": len(folds),
        "fold_wins": fold_wins,
        "fold_best_arm_wins": best_arm_wins,
        "pooled_raw_rmse": raw_value,
        "pooled_arm_rmse": arm_value,
        "relative_improvement": improvement,
        "bootstrap_lower_bound_log_improvement": lower,
        "worst_fold_regression": worst_regression,
        "passed": passed,
    }


def execute_stage(
    evaluator: Path,
    genome: str,
    raw: np.ndarray,
    targets: np.ndarray,
    groups: Sequence[str],
    cell: np.ndarray,
    row: np.ndarray,
    folds: Sequence[Fold],
    arms: Sequence[str],
    timeout_seconds: int,
) -> list[FoldEvidence]:
    evidence: list[FoldEvidence] = []
    with tempfile.TemporaryDirectory(prefix="jope-hyp-train-audit-") as temporary:
        root = Path(temporary)
        for fold in folds:
            transformed_cell, transformed_row = transform_blocks(cell, row, fold.fit_rows)
            predictions: dict[str, list[float]] = {}
            for view in ("raw", *arms):
                staged = root / f"fold-{fold.index}-{view}.csv"
                stage_csv(staged, feature_view(raw, transformed_cell, transformed_row, view), targets, fold.hold_rows)
                predictions[view] = evaluate_predictions(evaluator, genome, staged, fold, timeout_seconds)
            hold = list(fold.hold_rows)
            evidence.append(
                FoldEvidence(
                    index=fold.index,
                    row_groups=[groups[index] for index in hold],
                    actual=[float(targets[index]) for index in hold],
                    raw=predictions.pop("raw"),
                    arms=predictions,
                )
            )
    return evidence


def find_round2_result(round2_root: Path, dataset_id: str, seed: int) -> Path:
    expected = round2_root / "screen-results2" / f"js2-c1000-{dataset_id[:12]}-s{seed}" / "result.json"
    if not expected.is_file():
        raise FileNotFoundError(expected)
    return expected


def audit_one(
    dataset_id: str,
    seed: int,
    quant_root: Path,
    round2_root: Path,
    evaluator: Path,
    thresholds: Thresholds,
    timeout_seconds: int,
) -> dict[str, Any]:
    dataset_root = quant_root / dataset_id
    train_path = checked_data_path(dataset_root, "train.csv")
    cell_path = checked_data_path(dataset_root, "train3.vs")
    row_path = checked_data_path(dataset_root, "hyperion3train.vs")
    raw, targets, groups = read_training_csv(train_path)
    cell = read_matrix(cell_path, raw.shape[0], raw.shape[1])
    row = read_matrix(row_path, raw.shape[0], 128)
    result_path = find_round2_result(round2_root, dataset_id, seed)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("dataset_hash") != dataset_id or result.get("master_seed") != seed:
        raise ValueError(f"round-2 result identity mismatch: {result_path}")
    if result.get("test_csv_read") is not False:
        raise ValueError(f"round-2 result lacks train-only attestation: {result_path}")
    genome = result.get("winner_genome")
    if not isinstance(genome, str) or not genome:
        raise ValueError(f"round-2 result has no frozen winner: {result_path}")

    cv5_folds = grouped_folds(groups, 5, CV5_SEED, thresholds.minimum_atomic_groups)
    cv5_evidence = execute_stage(
        evaluator, genome, raw, targets, groups, cell, row, cv5_folds, VIEWS, timeout_seconds
    )
    cv5 = [evaluate_arm("cv5", cv5_evidence, arm, thresholds) for arm in VIEWS]
    passed = [result for result in cv5 if result["passed"]]
    advanced = min(passed, key=lambda item: (-item["relative_improvement"], item["arm"]))["arm"] if passed else None
    confirmation = None
    selected = "raw"
    if advanced is not None:
        cv7_folds = grouped_folds(groups, 7, CV7_SEED, thresholds.minimum_atomic_groups)
        cv7_evidence = execute_stage(
            evaluator, genome, raw, targets, groups, cell, row, cv7_folds, (advanced,), timeout_seconds
        )
        confirmation = evaluate_arm("cv7", cv7_evidence, advanced, thresholds)
        if confirmation["passed"]:
            selected = advanced
    selected_result = confirmation if selected != "raw" else None
    return {
        "dataset_id": dataset_id,
        "seed": seed,
        "rows": int(raw.shape[0]),
        "raw_features": int(raw.shape[1]),
        "source_sha256": {
            "train": sha256_file(train_path),
            "historical_cell": sha256_file(cell_path),
            "historical_row128": sha256_file(row_path),
            "round2_result": sha256_file(result_path),
        },
        "winner_semantic": result.get("winner_semantic"),
        "cv5": cv5,
        "cv5_advanced_arm": advanced,
        "cv7_confirmation": confirmation,
        "selected_view": selected,
        "selected_log_rmse_improvement": (
            math.log(selected_result["pooled_raw_rmse"] / selected_result["pooled_arm_rmse"])
            if selected_result is not None
            else 0.0
        ),
    }


def dataset_bootstrap_lower_bound(deltas: Sequence[float], confidence: float = 0.95) -> float:
    if not deltas:
        return math.nan
    state = 0x4859_5045_5249_4F4E
    mask = (1 << 64) - 1
    samples: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        values: list[float] = []
        for _ in deltas:
            state ^= (state << 13) & mask
            state ^= state >> 7
            state ^= (state << 17) & mask
            state &= mask
            values.append(deltas[state % len(deltas)])
        samples.append(statistics.mean(values))
    samples.sort()
    return samples[math.floor((1.0 - confidence) * (len(samples) - 1))]


def summarize(results: Sequence[dict[str, Any]], dataset_ids: Sequence[str], seeds: Sequence[int]) -> dict[str, Any]:
    by_dataset: dict[str, dict[int, dict[str, Any]]] = {dataset_id: {} for dataset_id in dataset_ids}
    for result in results:
        by_dataset[result["dataset_id"]][result["seed"]] = result
    dataset_rows: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        selected = by_dataset[dataset_id]
        complete = all(seed in selected for seed in seeds)
        deltas = [selected[seed]["selected_log_rmse_improvement"] for seed in seeds] if complete else []
        dataset_rows.append(
            {
                "dataset_id": dataset_id,
                "complete": complete,
                "selected_views": {str(seed): selected[seed]["selected_view"] for seed in seeds if seed in selected},
                "mean_log_rmse_improvement": statistics.mean(deltas) if deltas else None,
                "both_seeds_activated": complete and all(selected[seed]["selected_view"] != "raw" for seed in seeds),
            }
        )
    deltas = [row["mean_log_rmse_improvement"] for row in dataset_rows if row["mean_log_rmse_improvement"] is not None]
    activations: dict[str, int] = {"raw": 0, **{view: 0 for view in VIEWS}}
    for result in results:
        activations[result["selected_view"]] += 1
    return {
        "complete_seed_dataset_runs": len(results),
        "expected_seed_dataset_runs": len(dataset_ids) * len(seeds),
        "activation_counts": activations,
        "activation_rate": 1.0 - activations["raw"] / len(results) if results else 0.0,
        "datasets_with_positive_gated_delta": sum(delta > 0.0 for delta in deltas),
        "datasets_with_both_seeds_activated": sum(row["both_seeds_activated"] for row in dataset_rows),
        "median_dataset_log_rmse_improvement": statistics.median(deltas) if deltas else None,
        "mean_dataset_log_rmse_improvement": statistics.mean(deltas) if deltas else None,
        "bootstrap_95_lower_bound_mean_dataset_delta": dataset_bootstrap_lower_bound(deltas),
        "maximum_seed_log_rmse_improvement": max(
            (result["selected_log_rmse_improvement"] for result in results), default=None
        ),
        "datasets": dataset_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--round2-root", type=Path, required=True)
    parser.add_argument("--quant-root", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    dataset_ids = protocol.get("cohorts", {}).get("development")
    if not isinstance(dataset_ids, list) or len(dataset_ids) != 22:
        raise ValueError("audit requires the exact 22-dataset development cohort")
    seeds = (101, 202)
    if args.workers < 1 or args.timeout_seconds < 1:
        raise ValueError("workers and timeout must be positive")
    evaluator = args.evaluator.resolve()
    if not evaluator.is_file() or not os.access(evaluator, os.X_OK):
        raise ValueError("evaluator must be an executable regular file")
    thresholds = Thresholds()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    jobs = [(dataset_id, seed) for dataset_id in dataset_ids for seed in seeds]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {
            executor.submit(
                audit_one,
                dataset_id,
                seed,
                args.quant_root,
                args.round2_root,
                evaluator,
                thresholds,
                args.timeout_seconds,
            ): (dataset_id, seed)
            for dataset_id, seed in jobs
        }
        for future in as_completed(pending):
            dataset_id, seed = pending[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"{dataset_id} seed={seed} selected={result['selected_view']} "
                    f"delta={result['selected_log_rmse_improvement']:.6f}",
                    flush=True,
                )
            except Exception as error:  # fail closed but retain complete diagnostic coverage
                failures.append({"dataset_id": dataset_id, "seed": seed, "reason": str(error)})
                print(f"{dataset_id} seed={seed} failed={error}", flush=True)
    results.sort(key=lambda result: (result["dataset_id"], result["seed"]))
    failures.sort(key=lambda result: (result["dataset_id"], result["seed"]))
    report = {
        "schema": SCHEMA,
        "promotion_eligible": False,
        "promotion_blocker": "historical full-train sidecars are not fold-local encoder receipts",
        "cohort": "development",
        "test_artifacts_read": False,
        "allowed_data_basenames": sorted(ALLOWED_DATA_FILES),
        "protocol_sha256": sha256_file(args.protocol),
        "evaluator_sha256": sha256_file(evaluator),
        "thresholds": asdict(thresholds),
        "seeds": list(seeds),
        "results": results,
        "failures": failures,
        "summary": summarize(results, dataset_ids, seeds),
    }
    report["report_sha256"] = canonical_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0 if not failures and len(results) == len(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
