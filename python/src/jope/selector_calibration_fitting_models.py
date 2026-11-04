"""Calibration and portfolio model fitting/scoring helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from .selector_calibration_fitting_primitives import (
    _CALIBRATED_SCORE_WEIGHT,
    _DEFAULT_ENGINE,
    _EXPANSIVE_TRANSFORMS,
    _GLOBAL_HEAD,
    _META_NUMERIC_FEATURES,
    _META_RIDGE_L2,
    _PATTERN_NAMES,
    _SCALER_TRANSFORMS,
    _TARGET_DELTA_FLOOR,
    PORTFOLIO_META_POLICY,
    SCHEMA_VERSION,
    _bin_index,
    _correlation_lists,
    _engine_head_key,
    _finite_float,
    _head_from_actions,
    _head_from_genome,
    _parse_actions,
    _stats,
    pattern_flags,
)


def _fit_calibration_model(
    rows: list[dict[str, Any]],
    *,
    bin_count: int = 5,
    min_head_rows: int = 12,
    min_pattern_rows: int = 8,
) -> dict[str, Any]:
    global_bins = _fit_reward_bins(rows, bin_count=bin_count, head=_GLOBAL_HEAD)
    by_head: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_engine: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_engine_head: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        head = str(row.get("head_type") or "unknown")
        engine = str(row.get("engine") or _DEFAULT_ENGINE)
        by_head[head].append(row)
        by_engine[engine].append(row)
        by_engine_head[_engine_head_key(engine, head)].append(row)
    head_bins: dict[str, Any] = {}
    for head, head_rows in sorted(by_head.items()):
        if len(head_rows) >= min_head_rows:
            head_bins[head] = _fit_reward_bins(head_rows, bin_count=bin_count, head=head)
        else:
            head_bins[head] = {
                "head": head,
                "fallback": True,
                "source": _GLOBAL_HEAD,
                "count": len(head_rows),
            }
    engine_head_bins: dict[str, Any] = {}
    for key, key_rows in sorted(by_engine_head.items()):
        engine, head = key.split("::", 1)
        if len(key_rows) >= min_head_rows:
            spec = _fit_reward_bins(key_rows, bin_count=bin_count, head=head)
            spec["engine"] = engine
            engine_head_bins[key] = spec
        else:
            engine_head_bins[key] = {
                "engine": engine,
                "head": head,
                "fallback": True,
                "source": head
                if head in head_bins and not bool(head_bins[head].get("fallback"))
                else _GLOBAL_HEAD,
                "count": len(key_rows),
            }
    return {
        "global_bins": global_bins,
        "head_bins": head_bins,
        "engine_head_bins": engine_head_bins,
        "engine_stats": {
            engine: _stats([row["delta"] for row in engine_rows])
            for engine, engine_rows in sorted(by_engine.items())
        },
        "pattern_penalties": _fit_pattern_penalties(rows, min_pattern_rows=min_pattern_rows),
        "global_stats": _stats([row["delta"] for row in rows]),
        "target_delta_floor": _TARGET_DELTA_FLOOR,
        "calibrated_score_weight": _CALIBRATED_SCORE_WEIGHT,
    }


def _fit_portfolio_meta_model(rows: list[dict[str, Any]], calibration_fit: dict[str, Any]) -> dict[str, Any]:
    datasets = {str(row.get("dataset_hash")) for row in rows if row.get("dataset_hash")}
    if len(rows) < 8 or len(datasets) < 2:
        return {
            "enabled": False,
            "reason": "insufficient_rows",
            "row_count": len(rows),
            "dataset_count": len(datasets),
        }
    calibration = {
        "schema_version": SCHEMA_VERSION,
        "selected_policy": "robust_pattern_bins",
        "fit": calibration_fit,
    }
    categories = _meta_categories(rows)
    matrix, targets, feature_names = _portfolio_meta_matrix(rows, calibration, categories)
    if matrix.shape[0] < 2 or matrix.shape[1] == 0:
        return {
            "enabled": False,
            "reason": "empty_feature_matrix",
            "row_count": len(rows),
            "dataset_count": len(datasets),
        }
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale < 1e-9, 1.0, scale)
    x = (matrix - mean) / scale
    design = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * _META_RIDGE_L2
    penalty[0, 0] = 0.0
    try:
        coef = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(design.T @ design + penalty) @ design.T @ targets
    predictions = design @ coef
    return {
        "enabled": True,
        "policy": PORTFOLIO_META_POLICY,
        "row_count": int(matrix.shape[0]),
        "dataset_count": len(datasets),
        "target": "delta_clipped",
        "ridge_l2": _META_RIDGE_L2,
        "feature_names": feature_names,
        "feature_mean": [float(value) for value in mean.tolist()],
        "feature_scale": [float(value) for value in scale.tolist()],
        "intercept": float(coef[0]),
        "coef": [float(value) for value in coef[1:].tolist()],
        "train_correlation": _correlation_lists(predictions.tolist(), targets.tolist()),
        "categories": categories,
    }


def _portfolio_meta_matrix(
    rows: list[dict[str, Any]],
    calibration: dict[str, Any],
    categories: dict[str, list[str]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_names = _portfolio_meta_feature_names(categories)
    matrix: list[list[float]] = []
    targets: list[float] = []
    for row in rows:
        reward = _finite_float(row.get("reward"))
        delta = _finite_float(row.get("delta"))
        if reward is None or delta is None:
            continue
        bin_stats, bin_components = _lookup_bin_stats(row, calibration, reward)
        pattern_penalty = float(sum(_candidate_pattern_penalties(row, calibration).values()))
        values = _portfolio_meta_feature_values(row, bin_stats, bin_components, pattern_penalty, categories)
        matrix.append([float(values.get(name, 0.0)) for name in feature_names])
        targets.append(float(delta))
    if matrix:
        return (
            np.asarray(matrix, dtype=np.float64),
            np.asarray(targets, dtype=np.float64),
            feature_names,
        )
    return (
        np.zeros((0, len(feature_names)), dtype=np.float64),
        np.zeros((0,), dtype=np.float64),
        feature_names,
    )


def _fit_reward_bins(rows: list[dict[str, Any]], *, bin_count: int, head: str) -> dict[str, Any]:
    rewards = [float(row["reward"]) for row in rows if _finite_float(row.get("reward")) is not None]
    if rewards:
        quantiles = np.quantile(
            np.asarray(rewards, dtype=np.float64), np.linspace(0.0, 1.0, max(1, bin_count) + 1)
        ).tolist()
    else:
        quantiles = [0.0 for _ in range(max(1, bin_count) + 1)]
    thresholds = quantiles[1:-1]
    all_stats = _stats([row["delta"] for row in rows])
    grouped: list[list[float]] = [[] for _ in range(max(1, bin_count))]
    for row in rows:
        reward = _finite_float(row.get("reward"))
        if reward is None:
            continue
        idx = _bin_index(reward, thresholds, len(grouped))
        grouped[idx].append(float(row["delta"]))
    bins = []
    for idx, deltas in enumerate(grouped):
        stats = _stats(deltas) if deltas else {**all_stats, "count": 0}
        bins.append(
            {
                "index": idx,
                "lo": float(quantiles[idx]),
                "hi": float(quantiles[idx + 1]),
                **stats,
            }
        )
    return {
        "head": head,
        "fallback": False,
        "count": len(rows),
        "reward_quantiles": [float(value) for value in quantiles],
        "bins": bins,
    }


def _fit_pattern_penalties(rows: list[dict[str, Any]], *, min_pattern_rows: int) -> dict[str, Any]:
    global_stats = _stats([row["delta"] for row in rows])
    by_pattern: dict[str, list[float]] = {name: [] for name in _PATTERN_NAMES}
    for row in rows:
        for pattern in row.get("patterns") or []:
            if pattern in by_pattern:
                by_pattern[pattern].append(float(row["delta"]))
    out: dict[str, Any] = {}
    for pattern in _PATTERN_NAMES:
        deltas = by_pattern[pattern]
        stats = _stats(deltas)
        loss_heavy = len(deltas) >= min_pattern_rows and (
            float(stats["win_rate"]) < float(global_stats["win_rate"])
            or float(stats["expected_delta"]) < float(global_stats["expected_delta"])
        )
        win_gap = max(0.0, float(global_stats["win_rate"]) - float(stats["win_rate"])) if loss_heavy else 0.0
        delta_gap = (
            max(0.0, float(global_stats["expected_delta"]) - float(stats["expected_delta"]))
            if loss_heavy
            else 0.0
        )
        tail_gap = (
            max(0.0, float(stats["tail_risk"]) - float(global_stats["tail_risk"])) if loss_heavy else 0.0
        )
        penalty = min(2.0, 0.08 * win_gap + 0.35 * delta_gap + 0.05 * tail_gap)
        out[pattern] = {
            **stats,
            "loss_heavy": bool(loss_heavy),
            "penalty": float(penalty),
        }
    return out


def _lookup_bin_stats(
    row: dict[str, Any], calibration: dict[str, Any], reward: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    fit = calibration.get("fit") if isinstance(calibration.get("fit"), dict) else {}
    global_bins = (
        fit.get("global_bins")
        if isinstance(fit.get("global_bins"), dict)
        else _fit_reward_bins([], bin_count=5, head=_GLOBAL_HEAD)
    )
    head = str(row.get("head_type") or _head_from_genome(str(row.get("canonical_genome") or "")) or "unknown")
    engine = str(row.get("engine") or _DEFAULT_ENGINE)
    head_bins = fit.get("head_bins") if isinstance(fit.get("head_bins"), dict) else {}
    engine_head_bins = fit.get("engine_head_bins") if isinstance(fit.get("engine_head_bins"), dict) else {}
    engine_key = _engine_head_key(engine, head)
    spec = engine_head_bins.get(engine_key) if isinstance(engine_head_bins.get(engine_key), dict) else None
    used_engine_fallback = False
    used_global_fallback = False
    if not spec or bool(spec.get("fallback")):
        spec = head_bins.get(head) if isinstance(head_bins.get(head), dict) else None
        used_engine_fallback = True
    if not spec or bool(spec.get("fallback")):
        used_engine_fallback = True
        used_global_fallback = True
        spec = None
    if used_global_fallback:
        spec = global_bins
    quantiles = spec.get("reward_quantiles") if isinstance(spec.get("reward_quantiles"), list) else []
    thresholds = [float(value) for value in quantiles[1:-1]]
    bins = spec.get("bins") if isinstance(spec.get("bins"), list) else []
    idx = _bin_index(reward, thresholds, max(1, len(bins)))
    stats = bins[idx] if idx < len(bins) and isinstance(bins[idx], dict) else _stats([])
    return stats, {
        "head_type": head,
        "engine": engine,
        "bin_head": str(spec.get("head") or _GLOBAL_HEAD),
        "bin_engine": str(spec.get("engine") or _GLOBAL_HEAD),
        "reward_bin": int(idx),
        "used_engine_fallback": bool(used_engine_fallback),
        "used_global_fallback": bool(used_global_fallback),
    }


def _candidate_pattern_penalties(row: dict[str, Any], calibration: dict[str, Any]) -> dict[str, float]:
    fit = calibration.get("fit") if isinstance(calibration.get("fit"), dict) else {}
    penalties = fit.get("pattern_penalties") if isinstance(fit.get("pattern_penalties"), dict) else {}
    actions = _parse_actions(str(row.get("canonical_genome") or row.get("genome") or ""))
    out: dict[str, float] = {}
    for pattern in pattern_flags(actions):
        spec = penalties.get(pattern) if isinstance(penalties.get(pattern), dict) else {}
        penalty = _finite_float(spec.get("penalty"))
        if penalty is not None and penalty > 0.0:
            out[pattern] = float(penalty)
    return out


def _lookup_engine_stats(row: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
    fit = calibration.get("fit") if isinstance(calibration.get("fit"), dict) else {}
    engine = str(row.get("engine") or _DEFAULT_ENGINE)
    engine_stats = fit.get("engine_stats") if isinstance(fit.get("engine_stats"), dict) else {}
    stats = engine_stats.get(engine) if isinstance(engine_stats.get(engine), dict) else None
    if stats is not None:
        return stats
    global_stats = fit.get("global_stats") if isinstance(fit.get("global_stats"), dict) else {}
    return global_stats


def _score_portfolio_meta(
    row: dict[str, Any],
    calibration: dict[str, Any],
    bin_stats: dict[str, Any],
    bin_components: dict[str, Any],
    pattern_penalty: float,
) -> float | None:
    fit = calibration.get("fit") if isinstance(calibration.get("fit"), dict) else {}
    model = fit.get(PORTFOLIO_META_POLICY) if isinstance(fit.get(PORTFOLIO_META_POLICY), dict) else {}
    if not model.get("enabled"):
        return None
    names = model.get("feature_names") if isinstance(model.get("feature_names"), list) else []
    mean = np.asarray(model.get("feature_mean") or [], dtype=np.float64)
    scale = np.asarray(model.get("feature_scale") or [], dtype=np.float64)
    coef = np.asarray(model.get("coef") or [], dtype=np.float64)
    if (
        not names
        or mean.shape[0] != len(names)
        or scale.shape[0] != len(names)
        or coef.shape[0] != len(names)
    ):
        return None
    categories = (
        model.get("categories") if isinstance(model.get("categories"), dict) else _meta_categories([])
    )
    values = _portfolio_meta_feature_values(row, bin_stats, bin_components, pattern_penalty, categories)
    x = np.asarray([float(values.get(str(name), 0.0)) for name in names], dtype=np.float64)
    scale = np.where(np.abs(scale) < 1e-9, 1.0, scale)
    intercept = _finite_float(model.get("intercept")) or 0.0
    return float(intercept + ((x - mean) / scale) @ coef)


def _portfolio_meta_feature_names(categories: dict[str, list[str]]) -> list[str]:
    names = list(_META_NUMERIC_FEATURES)
    names.extend(f"engine={engine}" for engine in categories.get("engines", []))
    names.extend(f"head={head}" for head in categories.get("heads", []))
    return names


def _portfolio_meta_feature_values(
    row: dict[str, Any],
    bin_stats: dict[str, Any],
    bin_components: dict[str, Any],
    pattern_penalty: float,
    categories: dict[str, list[str]],
) -> dict[str, float]:
    actions = _parse_actions(str(row.get("canonical_genome") or row.get("genome") or ""))
    action_features = _action_feature_values(actions)
    engine = str(row.get("engine") or _DEFAULT_ENGINE)
    head = str(row.get("head_type") or _head_from_actions(actions) or "unknown")
    values = {
        "reward": _finite_float(row.get("reward")) or 0.0,
        "reward_percentile_in_engine": _finite_float(row.get("reward_percentile_in_engine")) or 0.0,
        "reward_std": _finite_float(row.get("reward_std")) or 0.0,
        "p_bad": _finite_float(row.get("p_bad")) or 0.0,
        "transform_count": float(action_features["transform_count"]),
        "repeated_scaler": float(action_features["repeated_scaler"]),
        "target_bin": float(action_features["target_bin"]),
        "appended_expansion_count": float(action_features["appended_expansion_count"]),
        "gpr_periodic": float(action_features["gpr_periodic"]),
        "calibrated_expected_delta": _finite_float(bin_stats.get("expected_delta")) or 0.0,
        "calibrated_win_rate": _finite_float(bin_stats.get("win_rate")) or 0.0,
        "calibrated_tail_risk": _finite_float(bin_stats.get("tail_risk")) or 0.0,
        "bin_score": _finite_float(bin_stats.get("score")) or 0.0,
        "pattern_penalty": float(pattern_penalty),
    }
    for category in categories.get("engines", []):
        values[f"engine={category}"] = 1.0 if engine == category else 0.0
    for category in categories.get("heads", []):
        values[f"head={category}"] = 1.0 if head == category else 0.0
    values.update(
        {key: float(value) for key, value in bin_components.items() if isinstance(value, (int, float, bool))}
    )
    return values


def _meta_categories(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    engines = sorted({str(row.get("engine") or _DEFAULT_ENGINE) for row in rows})
    heads = sorted({str(row.get("head_type") or "unknown") for row in rows})
    return {"engines": engines, "heads": heads}


def _action_feature_values(actions: dict[str, Any] | None) -> dict[str, Any]:
    actions = actions or {}
    transforms = [stage for stage in actions.get("transforms") or [] if isinstance(stage, dict)]
    head = actions.get("head") if isinstance(actions.get("head"), dict) else {}
    head_op = str(head.get("op") or "")
    counts = Counter(str(stage.get("op") or "") for stage in transforms)
    appended_expansion_count = sum(
        1
        for stage in transforms
        if str(stage.get("op") or "") in _EXPANSIVE_TRANSFORMS
        and bool((stage.get("params") or {}).get("append"))
    )
    gpr_periodic = 0
    if head_op == "gpr":
        kernel = str((head.get("params") or {}).get("kernel") or "rbf")
        gpr_periodic = int("periodic" in kernel)
    return {
        "transform_count": int(
            actions.get("transform_count")
            if isinstance(actions.get("transform_count"), int)
            else len(transforms)
        ),
        "repeated_scaler": int(sum(counts.get(op, 0) for op in _SCALER_TRANSFORMS) > 1),
        "target_bin": int(counts.get("target_bin_encode", 0) > 0),
        "appended_expansion_count": int(appended_expansion_count),
        "gpr_periodic": int(gpr_periodic),
    }
