"""Paired offline and prospective promotion gates."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .contracts import content_sha256


def normalized_regret(scores: Sequence[float], selected_index: int) -> float:
    values = np.asarray(scores, dtype=np.float64)
    if (
        values.ndim != 1
        or len(values) < 2
        or not np.all(np.isfinite(values))
        or not 0 <= selected_index < len(values)
    ):
        raise ValueError("normalized regret requires a finite candidate slate")
    span = float(values.max() - values.min())
    return 0.0 if span <= 1e-12 else float(
        (values.max() - values[selected_index]) / span
    )


def paired_bootstrap_interval(
    improvements: Sequence[float],
    *,
    seed: int,
    samples: int = 20_000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(improvements, dtype=np.float64)
    if (
        values.ndim != 1
        or not len(values)
        or not np.all(np.isfinite(values))
        or samples < 1
        or not 0.0 < confidence < 1.0
    ):
        raise ValueError("paired bootstrap inputs are invalid")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 2_000):
        count = min(2_000, samples - start)
        indexes = rng.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[indexes].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(means, tail)),
        float(np.quantile(means, 1.0 - tail)),
    )


def evaluate_offline_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    production: bool = True,
) -> dict[str, Any]:
    if production and len(rows) != 119:
        raise ValueError("offline gate requires exactly 119 cold-real datasets")
    if not rows:
        raise ValueError("offline gate rows are empty")
    datasets = [str(row["dataset_hash"]) for row in rows]
    if len(datasets) != len(set(datasets)):
        raise ValueError("offline gate contains duplicate datasets")
    baseline_regret = np.asarray(
        [float(row["real_only_regret"]) for row in rows], dtype=np.float64
    )
    mixed_regret = np.asarray(
        [float(row["mixed_regret"]) for row in rows], dtype=np.float64
    )
    baseline_pairwise = np.asarray(
        [float(row["real_only_pairwise_accuracy"]) for row in rows],
        dtype=np.float64,
    )
    mixed_pairwise = np.asarray(
        [float(row["mixed_pairwise_accuracy"]) for row in rows],
        dtype=np.float64,
    )
    baseline_validity = np.asarray(
        [float(row["real_only_validity"]) for row in rows], dtype=np.float64
    )
    mixed_validity = np.asarray(
        [float(row["mixed_validity"]) for row in rows], dtype=np.float64
    )
    arrays = (
        baseline_regret,
        mixed_regret,
        baseline_pairwise,
        mixed_pairwise,
        baseline_validity,
        mixed_validity,
    )
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("offline gate metrics must be finite")
    improvement = baseline_regret - mixed_regret
    interval = paired_bootstrap_interval(improvement, seed=seed)
    baseline_mean = float(baseline_regret.mean())
    relative_reduction = (
        float(improvement.mean() / baseline_mean)
        if baseline_mean > 1e-12
        else -math.inf
    )
    pairwise_delta = float(mixed_pairwise.mean() - baseline_pairwise.mean())
    validity_delta = float(mixed_validity.mean() - baseline_validity.mean())
    conditions = {
        "regret_reduction_at_least_10_percent": relative_reduction >= 0.10,
        "paired_bootstrap_lower_above_zero": interval[0] > 0.0,
        "pairwise_regression_within_one_point": pairwise_delta >= -0.01,
        "validity_regression_within_one_point": validity_delta >= -0.01,
    }
    result = {
        "schema": "BroadOfflineGate.v1",
        "datasets": len(rows),
        "seed": seed,
        "mean_real_only_regret": baseline_mean,
        "mean_mixed_regret": float(mixed_regret.mean()),
        "relative_regret_reduction": relative_reduction,
        "mean_paired_regret_improvement": float(improvement.mean()),
        "paired_bootstrap_95": list(interval),
        "pairwise_accuracy_delta": pairwise_delta,
        "validity_delta": validity_delta,
        "conditions": conditions,
        "passed": all(conditions.values()),
    }
    result["gate_sha256"] = content_sha256(result)
    return result


def evaluate_prospective_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    production: bool = True,
) -> dict[str, Any]:
    if production and len(rows) != 64:
        raise ValueError("prospective gate requires exactly 64 sealed real datasets")
    if not rows:
        raise ValueError("prospective gate rows are empty")
    datasets = [str(row["dataset_hash"]) for row in rows]
    if len(datasets) != len(set(datasets)):
        raise ValueError("prospective gate contains duplicate datasets")
    improvement = np.asarray(
        [
            float(row["mixed_best_robust_r2"])
            - float(row["real_only_best_robust_r2"])
            for row in rows
        ],
        dtype=np.float64,
    )
    baseline_validity = np.asarray(
        [float(row["real_only_validity"]) for row in rows], dtype=np.float64
    )
    mixed_validity = np.asarray(
        [float(row["mixed_validity"]) for row in rows], dtype=np.float64
    )
    baseline_wall = np.asarray(
        [float(row["real_only_wall_seconds"]) for row in rows], dtype=np.float64
    )
    mixed_wall = np.asarray(
        [float(row["mixed_wall_seconds"]) for row in rows], dtype=np.float64
    )
    if not all(
        np.all(np.isfinite(array))
        for array in (
            improvement,
            baseline_validity,
            mixed_validity,
            baseline_wall,
            mixed_wall,
        )
    ) or np.any(baseline_wall <= 0) or np.any(mixed_wall < 0):
        raise ValueError("prospective gate metrics must be finite with positive baseline time")
    interval = paired_bootstrap_interval(improvement, seed=seed)
    validity_delta = float(mixed_validity.mean() - baseline_validity.mean())
    wall_ratio = float(statistics.median(mixed_wall) / statistics.median(baseline_wall))
    conditions = {
        "paired_bootstrap_lower_above_zero": interval[0] > 0.0,
        "median_improvement_nonnegative": float(np.median(improvement)) >= 0.0,
        "validity_regression_within_one_point": validity_delta >= -0.01,
        "median_wall_time_within_1_25x": wall_ratio <= 1.25,
    }
    result = {
        "schema": "BroadProspectiveGate.v1",
        "datasets": len(rows),
        "seed": seed,
        "mean_best_robust_r2_improvement": float(improvement.mean()),
        "median_best_robust_r2_improvement": float(np.median(improvement)),
        "paired_bootstrap_95": list(interval),
        "validity_delta": validity_delta,
        "median_wall_time_ratio": wall_ratio,
        "conditions": conditions,
        "passed": all(conditions.values()),
    }
    result["gate_sha256"] = content_sha256(result)
    return result


def recovery_decision(
    *,
    pilot_gate_passed: bool,
    labels: int,
    recovery_attempted: bool,
    recovery_gate_passed: bool | None,
) -> str:
    if pilot_gate_passed:
        return "prospective_gate"
    if labels < 100_000:
        return "continue_fixed_probe_to_100000"
    if not recovery_attempted:
        return "synthetic_pretrain_real_finetune"
    if recovery_gate_passed:
        return "prospective_gate"
    return "cap_synthetic_at_10_percent_and_acquire_real"
