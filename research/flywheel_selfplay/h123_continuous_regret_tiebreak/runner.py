"""Pure CPU helpers for H123's preregistered candidate tie-break."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np


def family_folds(rows: Sequence[Mapping], *, seed: int = 12300) -> dict:
    """Build deterministic folds after enforcing the real DL-Train boundary."""
    families = set()
    for row in rows:
        if (
            row.get("dl_split") != "DL-Train"
            or row.get("task") != "regression"
            or row.get("has_meta_test") is True
            or row.get("origin") not in (None, "real")
            or not isinstance(row.get("family_id"), str)
            or not row["family_id"]
        ):
            raise ValueError("fold roster must contain real DL-Train regression rows")
        families.add(row["family_id"])
    if len(families) < 5:
        raise ValueError("at least five families are required")
    ordered = sorted(families)
    permutation = np.random.default_rng(seed).permutation(len(ordered))
    folds = [[] for _ in range(5)]
    for index, family_index in enumerate(permutation):
        folds[index % 5].append(ordered[int(family_index)])
    folds = [sorted(fold) for fold in folds]
    return {
        "seed": seed,
        "folds": folds,
        "train_families": sorted(set().union(*folds[:4])),
        "score_families": folds[4],
    }


def mean_regret_by_family(
    true_utility: Sequence[Sequence[float]],
    predicted_utility: Sequence[Sequence[float]],
    family_ids: Sequence[str],
) -> float:
    """Return equally weighted family mean regret over candidate predictions."""
    if not true_utility or len(true_utility) != len(predicted_utility):
        raise ValueError("truth and prediction must have matching non-empty rows")
    if len(family_ids) != len(true_utility):
        raise ValueError("one family ID is required per context")
    by_family: dict[str, list[float]] = defaultdict(list)
    for truth, prediction, family in zip(true_utility, predicted_utility, family_ids):
        if not family or len(truth) < 2 or len(truth) != len(prediction):
            raise ValueError("invalid family or candidate row")
        values = [*truth, *prediction]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("utilities must be finite")
        oracle = max(truth)
        by_family[family].append(
            sum(oracle - predicted for predicted in prediction) / len(prediction)
        )
    return sum(sum(rows) / len(rows) for rows in by_family.values()) / len(by_family)


def select_tied_candidate(
    candidates: Sequence[Mapping], *, minimum_artifacts: int = 3
) -> Mapping:
    """Pick minimum training regret among the verified maximum-utility tie set."""
    if minimum_artifacts < 3:
        raise ValueError("H123 requires at least three verified artifacts")
    if not candidates:
        raise ValueError("candidate inventory is empty")
    utilities = [candidate.get("utility") for candidate in candidates]
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in utilities):
        raise ValueError("candidate utilities must be finite numbers")
    maximum = max(utilities)
    tied = [candidate for candidate in candidates if candidate["utility"] == maximum]
    usable = [
        candidate
        for candidate in tied
        if isinstance(candidate.get("round"), int)
        and candidate["round"] > 0
        and isinstance(candidate.get("artifact_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", candidate["artifact_sha256"])
        and isinstance(candidate.get("train_mean_regret"), (int, float))
        and math.isfinite(candidate["train_mean_regret"])
    ]
    if len(usable) < minimum_artifacts:
        raise ValueError("fewer than three verified top-tied artifacts")
    return min(
        usable,
        key=lambda item: (
            item["train_mean_regret"], item["round"], item["artifact_sha256"]
        ),
    )
