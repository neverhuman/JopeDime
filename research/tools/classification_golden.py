#!/usr/bin/env python3
"""One-off scikit-learn oracle for OBJ-03; EXPECTED is the committed golden output."""

from __future__ import annotations

import json

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

LABELS = [0, 0, 1, 1, 0, 1]
POSITIVE_PROBABILITIES = [0.1, 0.5, 0.5, 0.9, 0.9, 0.1]
COST_MATRIX = [[1.0, 4.0], [2.0, 0.0]]  # rows actual, columns predicted
EXPECTED = {
    "roc_auc": 0.5,
    "pr_auc": 0.5,
    "f1_at_0_5": 4.0 / 7.0,
    "f1_at_0_9": 0.4,
    "balanced_accuracy_at_0_5": 0.5,
    "cost_matrix_loss_at_0_5": 11.0 / 6.0,
}
# Output from this oracle on the tied-score fixture:
# {
#   "balanced_accuracy_at_0_5": 0.5,
#   "cost_matrix_loss_at_0_5": 1.8333333333333333,
#   "f1_at_0_5": 0.5714285714285714,
#   "f1_at_0_9": 0.4,
#   "pr_auc": 0.5,
#   "roc_auc": 0.5
# }


def calculate() -> dict[str, float]:
    scores = POSITIVE_PROBABILITIES
    predicted_at_0_5 = [int(score >= 0.5) for score in scores]
    matrix = confusion_matrix(LABELS, predicted_at_0_5, labels=[0, 1])
    cost = sum(
        matrix[actual][predicted] * COST_MATRIX[actual][predicted]
        for actual in range(2)
        for predicted in range(2)
    ) / len(LABELS)
    return {
        "roc_auc": float(roc_auc_score(LABELS, scores)),
        "pr_auc": float(average_precision_score(LABELS, scores)),
        "f1_at_0_5": float(f1_score(LABELS, predicted_at_0_5, zero_division=0)),
        "f1_at_0_9": float(
            f1_score(
                LABELS,
                [int(score >= 0.9) for score in scores],
                zero_division=0,
            )
        ),
        "balanced_accuracy_at_0_5": float(
            balanced_accuracy_score(LABELS, predicted_at_0_5)
        ),
        "cost_matrix_loss_at_0_5": float(cost),
    }


def main() -> None:
    actual = calculate()
    for metric, expected in EXPECTED.items():
        if abs(actual[metric] - expected) > 1e-12:
            raise SystemExit(
                f"{metric} changed: expected {expected:.17g}, got {actual[metric]:.17g}"
            )
    print(json.dumps(actual, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
