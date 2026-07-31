#!/usr/bin/env python3
"""Test-adaptive fitness, ranking labels, and gene evidence v2."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Mapping, Protocol

from test_exact_common import SELFPLAY_SELECTION_BIAS, sha256_file, sha256_text


class AdaptiveEstimator(Protocol):
    def fit(self, features: list[list[float]], targets: list[float]) -> object: ...

    def predict(self, features: list[list[float]]) -> list[float]: ...


def read_last_target_csv(path: Path) -> tuple[list[list[float]], list[float]]:
    features, targets = [], []
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            values = [float(value) for value in row]
            if len(values) < 2 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path}: invalid numeric row")
            features.append(values[:-1])
            targets.append(values[-1])
    if not features:
        raise ValueError(f"{path}: dataset is empty")
    return features, targets


def r2_score(targets: list[float], predictions: list[float]) -> float:
    mean = sum(targets) / len(targets)
    denominator = sum((target - mean) ** 2 for target in targets)
    if denominator == 0:
        return float("nan")
    return 1.0 - sum(
        (target - prediction) ** 2
        for target, prediction in zip(targets, predictions, strict=True)
    ) / denominator


def auc_score(targets: list[float], predictions: list[float]) -> float:
    positives = sum(target == 1.0 for target in targets)
    negatives = sum(target == 0.0 for target in targets)
    if positives == 0 or negatives == 0:
        return float("nan")
    ordered = sorted(
        zip(predictions, targets, strict=True), key=lambda row: row[0]
    )
    rank_sum = 0.0
    offset = 0
    while offset < len(ordered):
        end = offset + 1
        while end < len(ordered) and ordered[end][0] == ordered[offset][0]:
            end += 1
        average_rank = (offset + 1 + end) / 2.0
        rank_sum += average_rank * sum(
            target == 1.0 for _, target in ordered[offset:end]
        )
        offset = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def log_loss(targets: list[float], predictions: list[float]) -> float:
    epsilon = 1e-15
    return -sum(
        target * math.log(min(1.0 - epsilon, max(epsilon, prediction)))
        + (1.0 - target)
        * math.log(min(1.0 - epsilon, max(epsilon, 1.0 - prediction)))
        for target, prediction in zip(targets, predictions, strict=True)
    ) / len(targets)


def evaluate_adaptive(
    task: str,
    train_csv: Path,
    test_csv: Path,
    estimator_factory: Callable[[], AdaptiveEstimator],
) -> dict:
    """Fit once on train and use only held-out test fitness; never CV."""

    train_x, train_y = read_last_target_csv(train_csv)
    test_x, test_y = read_last_target_csv(test_csv)
    if len(train_x[0]) != len(test_x[0]):
        raise ValueError("train/test feature widths differ")
    estimator = estimator_factory()
    estimator.fit(train_x, train_y)
    predictions = [float(value) for value in estimator.predict(test_x)]
    if len(predictions) != len(test_y) or not all(
        math.isfinite(value) for value in predictions
    ):
        raise ValueError("adaptive predictions are invalid")
    if task == "regression":
        score = r2_score(test_y, predictions)
        secondary_metric = "rmse"
        secondary = math.sqrt(
            sum(
                (target - prediction) ** 2
                for target, prediction in zip(test_y, predictions, strict=True)
            )
            / len(test_y)
        )
    elif task == "binary":
        if set(train_y + test_y) - {0.0, 1.0}:
            raise ValueError("binary adaptive targets must be zero/one")
        score = auc_score(test_y, predictions)
        secondary_metric = "log_loss"
        secondary = log_loss(test_y, predictions)
    else:
        raise ValueError(f"unsupported adaptive task {task}")
    return {
        "task": task,
        "metric": "r2" if task == "regression" else "auc",
        "score": score,
        "secondary_metric": secondary_metric,
        "secondary_score": secondary,
        "fit_split": "train.csv",
        "fitness_split": "test.csv",
        "cross_validation_fitness": False,
        "test_adaptive": True,
        "benchmark_valid": False,
        "selection_bias": SELFPLAY_SELECTION_BIAS,
        "valid": math.isfinite(score) and math.isfinite(secondary),
    }


def attach_ranking_labels(
    rows: Iterable[Mapping[str, object]],
    incumbents: Mapping[tuple[str, str], float] | None = None,
) -> list[dict]:
    by_dataset: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        by_dataset[(str(row["task"]), str(row["dataset_hash"]))].append(row)
    result = []
    for identity in sorted(by_dataset):
        values = by_dataset[identity]
        valid = [
            row
            for row in values
            if row.get("valid") is True
            and isinstance(row.get("score"), (int, float))
            and math.isfinite(float(row["score"]))
        ]
        valid.sort(
            key=lambda row: (
                -float(row["score"]),
                float(row.get("secondary_score") or 0.0)
                if identity[0] == "binary"
                else 0.0,
                str(row.get("semantic_sha256") or ""),
            )
        )
        count = len(valid)
        for rank, row in enumerate(valid, 1):
            incumbent = None if incumbents is None else incumbents.get(identity)
            row.update(
                within_dataset_rank=rank,
                within_dataset_percentile=(
                    1.0 if count == 1 else (count - rank) / (count - 1)
                ),
                regret_to_incumbent=(
                    None
                    if incumbent is None
                    else float(incumbent) - float(row["score"])
                ),
                incumbent_available=incumbent is not None,
            )
            result.append(row)
        valid_ids = {id(row) for row in valid}
        for row in values:
            if id(row) not in valid_ids:
                row.update(
                    within_dataset_rank=None,
                    within_dataset_percentile=None,
                    regret_to_incumbent=None,
                    incumbent_available=(
                        incumbents is not None and identity in incumbents
                    ),
                )
                result.append(row)
    return result


def discover_prior_evidence(roots: Iterable[Path]) -> list[dict]:
    """Index report-backed evidence without asserting cross-protocol comparability."""

    rows = []
    for root in roots:
        if not root.exists():
            continue
        paths = root.rglob("*.json") if root.is_dir() else [root]
        for path in paths:
            try:
                value = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(value, Mapping):
                continue
            score = (
                value.get("score")
                or value.get("test_auc")
                or value.get("r2")
                or value.get("final_r2")
            )
            genome = value.get("canonical_genome") or value.get("genome")
            dataset = value.get("dataset_hash") or value.get("dataset")
            if (
                not dataset
                or not genome
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
            ):
                continue
            rows.append(
                {
                    "source_report": str(path),
                    "source_report_sha256": sha256_file(path),
                    "dataset_hash": str(dataset),
                    "canonical_genome": str(genome),
                    "semantic_sha256": sha256_text(str(genome)),
                    "score": float(score),
                    "protocol_comparable": False,
                    "requires_current_protocol_replay": True,
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            row["dataset_hash"],
            row["semantic_sha256"],
            row["source_report"],
        ),
    )


def gene_evidence_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    classification: str = "adaptive",
) -> list[dict]:
    if classification not in {"adaptive", "sealed"}:
        raise ValueError("gene evidence classification must be adaptive or sealed")
    result = []
    for raw in rows:
        row = dict(raw)
        required = (
            "task",
            "dataset_hash",
            "semantic_sha256",
            "protocol_sha256",
            "score",
            "canonical_genome",
            "receipt_sha256",
        )
        if any(row.get(field) is None for field in required):
            raise ValueError("gene evidence row is not decision-complete")
        result.append(
            {
                "schema": "JopeGeneEvidence.v2",
                "task": row["task"],
                "dataset_hash": row["dataset_hash"],
                "semantic_sha256": row["semantic_sha256"],
                "protocol_sha256": row["protocol_sha256"],
                "score": row["score"],
                "metric": row.get("metric"),
                "secondary_metric": row.get("secondary_metric"),
                "secondary_score": row.get("secondary_score"),
                "within_dataset_rank": row.get("within_dataset_rank"),
                "within_dataset_percentile": row.get(
                    "within_dataset_percentile"
                ),
                "incumbent_delta": (
                    None
                    if row.get("regret_to_incumbent") is None
                    else -float(row["regret_to_incumbent"])
                ),
                "canonical_genome": row["canonical_genome"],
                "checkpoint_epoch": row.get("checkpoint_epoch"),
                "model_checkpoint_sha256": row.get(
                    "model_checkpoint_sha256"
                ),
                "proposal_slot": row.get("proposal_slot"),
                "candidate_source": row.get("candidate_source"),
                "receipt_sha256": row["receipt_sha256"],
                "classification": classification,
                "test_adaptive": classification == "adaptive",
                "benchmark_valid": classification == "sealed",
                "validity": row.get("validity", "valid"),
                "qualification_state": row.get("qualification_state"),
                "compiled_parity_state": row.get(
                    "compiled_parity_state", "not_required"
                ),
            }
        )
    return result


def write_gene_evidence(
    rows: Iterable[Mapping[str, object]],
    output: Path,
    *,
    classification: str = "adaptive",
) -> dict:
    values = gene_evidence_rows(rows, classification=classification)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        for row in values:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return {
        "schema": "JopeGeneEvidenceExport.v2",
        "output": str(output),
        "output_sha256": sha256_file(output),
        "rows": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = discover_prior_evidence(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "output": str(args.output),
                "output_sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
