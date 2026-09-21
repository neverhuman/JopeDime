"""Per-cut train-mean baseline B. Diagnostics only; does not rewrite ranking labels."""

from __future__ import annotations

import math


def baseline_rmse(train_targets, validation_targets):
    if not train_targets or not validation_targets:
        raise ValueError("baseline requires train and validation targets")
    if any(not math.isfinite(v) for v in list(train_targets) + list(validation_targets)):
        raise ValueError("nonfinite target")
    mean = sum(train_targets) / len(train_targets)
    sse = sum((y - mean) ** 2 for y in validation_targets)
    return math.sqrt(sse / len(validation_targets))


def beats_baseline(score, baseline):
    if type(score) not in (int, float) or type(baseline) not in (int, float):
        return None
    if not math.isfinite(score) or not math.isfinite(baseline) or baseline < 0 or score < 0:
        return None
    return score < baseline


def disagreement_vs_unit(score, baseline):
    """True when 'beat 1.0' and 'beat this cut B' disagree."""
    if beats_baseline(score, baseline) is None:
        return None
    return (score < 1.0) != (score < baseline)


def annotate_observation(metrics_rmse, baseline, units="cut_normalized_target"):
    if units not in ("cut_normalized_target", "original_target"):
        raise ValueError("unknown RMSE units")
    valid = beats_baseline(metrics_rmse, baseline) is not None
    ratio = metrics_rmse / baseline if valid and baseline > 0 else None
    if ratio is not None and not math.isfinite(ratio):
        ratio = None
    squared = ratio * ratio if ratio is not None else None
    return {
        "baseline_rmse": baseline,
        "baseline_rule": "train_mean_on_validation",
        "metrics_units": units,
        "search_best_units": "original_target",
        "baseline_units": units,
        "baseline_available": valid,
        "relative_rmse": ratio,
        "rmse_skill": 1 - ratio if ratio is not None else None,
        "squared_error_skill": 1 - squared if squared is not None and math.isfinite(squared) else None,
        "ratio_status": "available"
        if ratio is not None
        else "zero_baseline"
        if valid and baseline == 0
        else "unavailable",
        "beats_unit_one": beats_baseline(metrics_rmse, 1.0),
        "beats_cut_baseline": beats_baseline(metrics_rmse, baseline),
        "unit_vs_baseline_disagrees": disagreement_vs_unit(metrics_rmse, baseline),
    }
