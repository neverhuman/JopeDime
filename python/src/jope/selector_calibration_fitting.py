"""Calibration fitting, scoring, feature extraction, and genome helpers."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from .selector_calibration_fitting_models import (
    _action_feature_values as _action_feature_values,
)
from .selector_calibration_fitting_models import (
    _candidate_pattern_penalties as _candidate_pattern_penalties,
)
from .selector_calibration_fitting_models import (
    _fit_calibration_model as _fit_calibration_model,
)
from .selector_calibration_fitting_models import (
    _fit_pattern_penalties as _fit_pattern_penalties,
)
from .selector_calibration_fitting_models import (
    _fit_portfolio_meta_model as _fit_portfolio_meta_model,
)
from .selector_calibration_fitting_models import (
    _fit_reward_bins as _fit_reward_bins,
)
from .selector_calibration_fitting_models import (
    _lookup_bin_stats as _lookup_bin_stats,
)
from .selector_calibration_fitting_models import (
    _lookup_engine_stats as _lookup_engine_stats,
)
from .selector_calibration_fitting_models import (
    _meta_categories as _meta_categories,
)
from .selector_calibration_fitting_models import (
    _portfolio_meta_feature_names as _portfolio_meta_feature_names,
)
from .selector_calibration_fitting_models import (
    _portfolio_meta_feature_values as _portfolio_meta_feature_values,
)
from .selector_calibration_fitting_models import (
    _portfolio_meta_matrix as _portfolio_meta_matrix,
)
from .selector_calibration_fitting_models import (
    _score_portfolio_meta as _score_portfolio_meta,
)
from .selector_calibration_fitting_primitives import (
    _CALIBRATED_SCORE_WEIGHT as _CALIBRATED_SCORE_WEIGHT,
)
from .selector_calibration_fitting_primitives import (
    _DEFAULT_ENGINE as _DEFAULT_ENGINE,
)
from .selector_calibration_fitting_primitives import (
    _EXPANSIVE_TRANSFORMS as _EXPANSIVE_TRANSFORMS,
)
from .selector_calibration_fitting_primitives import (
    _EXPLOSIVE_TRANSFORMS as _EXPLOSIVE_TRANSFORMS,
)
from .selector_calibration_fitting_primitives import (
    _GLOBAL_HEAD as _GLOBAL_HEAD,
)
from .selector_calibration_fitting_primitives import (
    _META_NUMERIC_FEATURES as _META_NUMERIC_FEATURES,
)
from .selector_calibration_fitting_primitives import (
    _META_RIDGE_L2 as _META_RIDGE_L2,
)
from .selector_calibration_fitting_primitives import (
    _PATTERN_NAMES as _PATTERN_NAMES,
)
from .selector_calibration_fitting_primitives import (
    _PORTFOLIO_DEFERRED_HEADS as _PORTFOLIO_DEFERRED_HEADS,
)
from .selector_calibration_fitting_primitives import (
    _PORTFOLIO_PREFERRED_HEADS as _PORTFOLIO_PREFERRED_HEADS,
)
from .selector_calibration_fitting_primitives import (
    _SCALER_TRANSFORMS as _SCALER_TRANSFORMS,
)
from .selector_calibration_fitting_primitives import (
    _TARGET_DELTA_FLOOR as _TARGET_DELTA_FLOOR,
)
from .selector_calibration_fitting_primitives import (
    BIN_POLICIES as BIN_POLICIES,
)
from .selector_calibration_fitting_primitives import (
    MODEL_REALIZATION_SCHEMA as MODEL_REALIZATION_SCHEMA,
)
from .selector_calibration_fitting_primitives import (
    POLICIES as POLICIES,
)
from .selector_calibration_fitting_primitives import (
    PORTFOLIO_BASELINE_POLICY as PORTFOLIO_BASELINE_POLICY,
)
from .selector_calibration_fitting_primitives import (
    PORTFOLIO_META_POLICY as PORTFOLIO_META_POLICY,
)
from .selector_calibration_fitting_primitives import (
    SCHEMA_VERSION as SCHEMA_VERSION,
)
from .selector_calibration_fitting_primitives import (
    _bin_index as _bin_index,
)
from .selector_calibration_fitting_primitives import (
    _correlation_lists as _correlation_lists,
)
from .selector_calibration_fitting_primitives import (
    _engine_head_key as _engine_head_key,
)
from .selector_calibration_fitting_primitives import (
    _finite_float as _finite_float,
)
from .selector_calibration_fitting_primitives import (
    _head_from_actions as _head_from_actions,
)
from .selector_calibration_fitting_primitives import (
    _head_from_genome as _head_from_genome,
)
from .selector_calibration_fitting_primitives import (
    _metric_value as _metric_value,
)
from .selector_calibration_fitting_primitives import (
    _parse_actions as _parse_actions,
)
from .selector_calibration_fitting_primitives import (
    _parse_actions_for_engine as _parse_actions_for_engine,
)
from .selector_calibration_fitting_primitives import (
    _policy_metric_key as _policy_metric_key,
)
from .selector_calibration_fitting_primitives import (
    _portfolio_diversity_key as _portfolio_diversity_key,
)
from .selector_calibration_fitting_primitives import (
    _rank_rows as _rank_rows,
)
from .selector_calibration_fitting_primitives import (
    _score_value as _score_value,
)
from .selector_calibration_fitting_primitives import (
    _stats as _stats,
)
from .selector_calibration_fitting_primitives import (
    pattern_flags as pattern_flags,
)


def score_candidate_with_calibration(
    row: dict[str, Any],
    calibration: dict[str, Any] | None = None,
    *,
    policy: str | None = None,
) -> dict[str, Any]:
    if calibration is None:
        raise ValueError("selector calibration is required")
    if row.get("schema_version") == SCHEMA_VERSION and calibration.get("schema_version") != SCHEMA_VERSION:
        row, calibration = calibration, row
    policy = policy or str(calibration.get("selected_policy") or "raw_reward")
    if policy not in POLICIES:
        policy = "raw_reward"
    reward = _finite_float(row.get("reward"))
    if reward is None:
        reward = 0.0
    bin_stats, bin_components = _lookup_bin_stats(row, calibration, reward)
    engine_stats = _lookup_engine_stats(row, calibration)
    pattern_penalties = (
        _candidate_pattern_penalties(row, calibration)
        if policy in {"robust_pattern_bins", PORTFOLIO_META_POLICY}
        else {}
    )
    pattern_penalty = float(sum(pattern_penalties.values()))
    calibrated_score = float(bin_stats["score"]) - pattern_penalty
    meta_score = None
    if policy == "raw_reward":
        selection_score = reward
    elif policy == PORTFOLIO_META_POLICY:
        meta_score = _score_portfolio_meta(row, calibration, bin_stats, bin_components, pattern_penalty)
        selection_score = meta_score if meta_score is not None else reward
    else:
        selection_score = reward + _CALIBRATED_SCORE_WEIGHT * calibrated_score
    components = {
        "raw_reward": reward,
        "bin_score": float(bin_stats["score"]),
        "calibrated_score": calibrated_score,
        "calibrated_score_weight": _CALIBRATED_SCORE_WEIGHT,
        "pattern_penalty": pattern_penalty,
        "pattern_penalties": pattern_penalties,
        "engine_prior_score": float(engine_stats.get("score", 0.0)),
        "engine_prior_win_rate": float(engine_stats.get("win_rate", 0.0)),
        "portfolio_meta_score": meta_score,
        "portfolio_meta_enabled": meta_score is not None,
        **bin_components,
    }
    return {
        "selection_score": float(selection_score),
        "selection_policy": policy,
        "calibrated_expected_delta": float(bin_stats["expected_delta"]),
        "calibrated_win_rate": float(bin_stats["win_rate"]),
        "calibrated_tail_risk": float(bin_stats["tail_risk"]),
        "selection_score_components": components,
    }


def _selection_metrics(
    rows: list[dict[str, Any]], calibration: dict[str, Any], policy: str
) -> dict[str, Any]:
    selected = _select_top_by_dataset(rows, calibration, policy)
    return {
        **_delta_metrics([row["delta"] for row in selected]),
        "selection_objective": "top1_by_dataset",
    }


def _portfolio_delta_metrics(groups: list[list[dict[str, Any]]]) -> dict[str, Any]:
    best_deltas: list[float] = []
    selected_count = 0
    for group in groups:
        selected_count += len(group)
        deltas = [float(row["delta"]) for row in group if _finite_float(row.get("delta")) is not None]
        if deltas:
            best_deltas.append(max(deltas))
    metrics = _delta_metrics(best_deltas)
    metrics["proposal_selected_count"] = int(selected_count)
    return metrics


def _select_top_by_dataset(
    rows: list[dict[str, Any]], calibration: dict[str, Any], policy: str
) -> list[dict[str, Any]]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset_hash"])].append(row)
    selected: list[dict[str, Any]] = []
    for dataset_hash in sorted(by_dataset):
        ranked = sorted(
            by_dataset[dataset_hash],
            key=lambda row: (
                float(score_candidate_with_calibration(row, calibration, policy=policy)["selection_score"]),
                float(row.get("reward") or 0.0),
                -(row.get("proposal_index") if isinstance(row.get("proposal_index"), int) else 10**9),
                str(row.get("canonical_genome") or ""),
            ),
            reverse=True,
        )
        if ranked:
            selected.append(ranked[0])
    return selected


def _select_portfolio_by_dataset(
    rows: list[dict[str, Any]],
    calibration: dict[str, Any],
    policy: str,
) -> list[list[dict[str, Any]]]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset_hash"])].append(row)
    groups: list[list[dict[str, Any]]] = []
    for dataset_hash in sorted(by_dataset):
        scored: list[dict[str, Any]] = []
        for row in by_dataset[dataset_hash]:
            enriched = dict(row)
            enriched.update(score_candidate_with_calibration(row, calibration, policy=policy))
            scored.append(enriched)
        groups.append(
            _select_portfolio_rows(
                scored, calibrated_score_key="selection_score" if policy != "raw_reward" else None
            )
        )
    return groups


def _select_portfolio_rows(
    rows: list[dict[str, Any]], *, calibrated_score_key: str | None
) -> list[dict[str, Any]]:
    safe = [row for row in rows if not row.get("candidate_risk_reason")]
    if not safe:
        return []
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    selected_diversity: set[str] = set()
    selected_canonicals: set[str] = set()

    def add(row: dict[str, Any]) -> None:
        selected.append(row)
        selected_ids.add(id(row))
        selected_diversity.add(_portfolio_diversity_key(row))
        selected_canonicals.add(str(row.get("canonical_genome") or ""))

    add(_rank_rows(safe, "reward")[0])
    if calibrated_score_key is not None:
        for row in _rank_rows(safe, calibrated_score_key):
            if id(row) in selected_ids:
                continue
            if _portfolio_diversity_key(row) in selected_diversity:
                continue
            if str(row.get("canonical_genome") or "") in selected_canonicals:
                continue
            add(row)
            break
    for head in _PORTFOLIO_PREFERRED_HEADS:
        bucket = [
            row for row in safe if id(row) not in selected_ids and str(row.get("head_type") or "") == head
        ]
        if not bucket:
            continue
        diverse = [row for row in bucket if _portfolio_diversity_key(row) not in selected_diversity]
        add(_rank_rows(diverse or bucket, calibrated_score_key or "reward")[0])
        break
    for pool in (
        [
            row
            for row in safe
            if id(row) not in selected_ids and _portfolio_diversity_key(row) not in selected_diversity
        ],
        [row for row in safe if id(row) not in selected_ids],
    ):
        for row in _rank_rows(pool, calibrated_score_key or "reward"):
            if id(row) in selected_ids:
                continue
            if len(selected) >= 3:
                return selected
            add(row)
    return selected[:3]


def _choose_policy(cv: dict[str, Any]) -> str:
    policies = cv.get("policies") if isinstance(cv.get("policies"), dict) else {}
    raw_key = _policy_metric_key(policies.get("raw_reward") or {})
    best_policy = "raw_reward"
    best_key = raw_key
    for policy in ("head_reward_bins", "robust_pattern_bins"):
        key = _policy_metric_key(policies.get(policy) or {})
        if key > best_key:
            best_policy = policy
            best_key = key
    portfolio_key = _policy_metric_key(policies.get(PORTFOLIO_BASELINE_POLICY) or {})
    meta_key = _policy_metric_key(policies.get(PORTFOLIO_META_POLICY) or {})
    if meta_key > max(raw_key, portfolio_key):
        return PORTFOLIO_META_POLICY
    return best_policy if best_key > raw_key else "raw_reward"


def _delta_metrics(deltas: list[float]) -> dict[str, Any]:
    values = [float(value) for value in deltas if math.isfinite(float(value))]
    if not values:
        return {
            "selected_count": 0,
            "win_rate": 0.0,
            "mean_delta": None,
            "median_delta": None,
            "p10_delta": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "selected_count": len(values),
        "win_rate": float(np.mean(arr > 0.0)),
        "mean_delta": float(np.mean(arr)),
        "median_delta": float(np.median(arr)),
        "p10_delta": float(np.quantile(arr, 0.1)),
    }
