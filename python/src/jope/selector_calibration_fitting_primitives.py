"""Calibration fitting, scoring, feature extraction, and genome helpers."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from .data.active import parse_pipeline_actions

SCHEMA_VERSION = "active_selector_calibration.v1"

MODEL_REALIZATION_SCHEMA = "jope-prime-model-realization-v2"

PORTFOLIO_META_POLICY = "portfolio_meta_v1"

PORTFOLIO_BASELINE_POLICY = "portfolio_baseline"

BIN_POLICIES = ("raw_reward", "head_reward_bins", "robust_pattern_bins")

POLICIES = (*BIN_POLICIES, PORTFOLIO_META_POLICY)

_GLOBAL_HEAD = "__global__"

_DEFAULT_ENGINE = "featmath"

_TARGET_DELTA_FLOOR = -1.0

_CALIBRATED_SCORE_WEIGHT = 0.1

_META_RIDGE_L2 = 1e-3

_PATTERN_NAMES = (
    "target_bin_encode",
    "repeated_scalers",
    "repeated_explosive_transform",
    "excessive_appended_expansion",
    "unstable_gpr_periodic_with_expansion",
    "unstable_gpr_large_expansion",
)

_EXPANSIVE_TRANSFORMS = {
    "bin_onehot",
    "cheb",
    "hinge",
    "oblique_hinge",
    "pair_prod",
    "poly_interact",
    "quantile_bins",
    "rand_proj",
    "rff_cos",
    "rff_sin",
}

_EXPLOSIVE_TRANSFORMS = _EXPANSIVE_TRANSFORMS | {"sigmoid"}

_SCALER_TRANSFORMS = {"standardize", "robust_scale"}

_PORTFOLIO_PREFERRED_HEADS = ("histgbm2", "gpr", "ridge")

_PORTFOLIO_DEFERRED_HEADS = {"ksmooth", "histgbm"}

_META_NUMERIC_FEATURES = (
    "reward",
    "reward_percentile_in_engine",
    "reward_std",
    "p_bad",
    "transform_count",
    "repeated_scaler",
    "target_bin",
    "appended_expansion_count",
    "gpr_periodic",
    "calibrated_expected_delta",
    "calibrated_win_rate",
    "calibrated_tail_risk",
    "bin_score",
    "pattern_penalty",
)

def pattern_flags(actions: dict[str, Any] | None) -> list[str]:
    actions = actions or {}
    transforms = [stage for stage in actions.get("transforms") or [] if isinstance(stage, dict)]
    head = actions.get("head") if isinstance(actions.get("head"), dict) else {}
    head_op = str(head.get("op") or "")
    counts = Counter(str(stage.get("op") or "") for stage in transforms)
    expansion_count = sum(count for op, count in counts.items() if op in _EXPANSIVE_TRANSFORMS)
    appended_expansion_count = sum(
        1
        for stage in transforms
        if str(stage.get("op") or "") in _EXPANSIVE_TRANSFORMS
        and bool((stage.get("params") or {}).get("append"))
    )
    flags: list[str] = []
    if counts.get("target_bin_encode", 0) > 0:
        flags.append("target_bin_encode")
    if sum(counts.get(op, 0) for op in _SCALER_TRANSFORMS) > 1:
        flags.append("repeated_scalers")
    if any(op in _EXPLOSIVE_TRANSFORMS and count > 1 for op, count in counts.items()):
        flags.append("repeated_explosive_transform")
    if expansion_count >= 5 or appended_expansion_count >= 3:
        flags.append("excessive_appended_expansion")
    if head_op == "gpr":
        kernel = str((head.get("params") or {}).get("kernel") or "rbf")
        if "periodic" in kernel and expansion_count >= 2:
            flags.append("unstable_gpr_periodic_with_expansion")
        if len(transforms) > 4 and expansion_count >= 3:
            flags.append("unstable_gpr_large_expansion")
    return flags


def _bin_index(reward: float, thresholds: list[float], bin_count: int) -> int:
    idx = int(np.searchsorted(np.asarray(thresholds, dtype=np.float64), float(reward), side="right"))
    return max(0, min(idx, max(1, bin_count) - 1))


def _stats(deltas: list[float]) -> dict[str, Any]:
    values = [float(value) for value in deltas if math.isfinite(float(value))]
    if not values:
        return {
            "count": 0,
            "expected_delta": 0.0,
            "median_delta": 0.0,
            "win_rate": 0.0,
            "loss_rate": 0.0,
            "tail_risk": 0.0,
            "score": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    tail_risk = max(0.0, -float(np.quantile(arr, 0.1)))
    expected_delta = float(np.mean(arr))
    win_rate = float(np.mean(arr > 0.0))
    score = expected_delta + 0.05 * win_rate - 0.10 * tail_risk
    return {
        "count": len(values),
        "expected_delta": expected_delta,
        "median_delta": float(np.median(arr)),
        "win_rate": win_rate,
        "loss_rate": float(np.mean(arr <= 0.0)),
        "tail_risk": float(tail_risk),
        "score": float(score),
    }

def _engine_head_key(engine: str, head: str) -> str:
    return f"{engine or _DEFAULT_ENGINE}::{head or 'unknown'}"


def _parse_actions(genome: str) -> dict[str, Any] | None:
    try:
        return parse_pipeline_actions(genome)
    except Exception:
        return None


def _parse_actions_for_engine(genome: str, engine: str) -> dict[str, Any]:
    if engine != "prime":
        actions = _parse_actions(genome)
        if actions is None:
            raise ValueError(f"cannot parse {engine!r} realization candidate")
        return actions
    from .grammars import grammar_for
    from .pipeline_text import parse_pipeline_actions as parse_registry_pipeline

    try:
        return parse_registry_pipeline(genome, grammar_for("prime"))
    except (TypeError, ValueError) as error:
        raise ValueError("cannot parse Prime realization candidate") from error


def _head_from_genome(genome: str) -> str | None:
    return _head_from_actions(_parse_actions(genome))


def _head_from_actions(actions: dict[str, Any] | None) -> str | None:
    if not actions:
        return None
    head = actions.get("head") if isinstance(actions.get("head"), dict) else {}
    op = head.get("op")
    return str(op) if op else None


def _correlation_lists(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return None
