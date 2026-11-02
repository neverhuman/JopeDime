"""Calibration fitting, scoring, feature extraction, and genome helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .selector_calibration_fitting_primitives import (
    BIN_POLICIES,
    MODEL_REALIZATION_SCHEMA,
    POLICIES,
    PORTFOLIO_BASELINE_POLICY,
    PORTFOLIO_META_POLICY,
    SCHEMA_VERSION,
    _CALIBRATED_SCORE_WEIGHT,
    _DEFAULT_ENGINE,
    _EXPANSIVE_TRANSFORMS,
    _EXPLOSIVE_TRANSFORMS,
    _GLOBAL_HEAD,
    _META_NUMERIC_FEATURES,
    _META_RIDGE_L2,
    _PATTERN_NAMES,
    _PORTFOLIO_DEFERRED_HEADS,
    _PORTFOLIO_PREFERRED_HEADS,
    _SCALER_TRANSFORMS,
    _TARGET_DELTA_FLOOR,
    _bin_index,
    _correlation_lists,
    _engine_head_key,
    _finite_float,
    _head_from_actions,
    _head_from_genome,
    _metric_value,
    _parse_actions,
    _parse_actions_for_engine,
    _policy_metric_key,
    _portfolio_diversity_key,
    _rank_rows,
    _score_value,
    _stats,
    pattern_flags,
)
from .selector_calibration_fitting_models import (
    _action_feature_values,
    _candidate_pattern_penalties,
    _fit_calibration_model,
    _fit_pattern_penalties,
    _fit_portfolio_meta_model,
    _fit_reward_bins,
    _lookup_bin_stats,
    _lookup_engine_stats,
    _meta_categories,
    _portfolio_meta_feature_names,
    _portfolio_meta_feature_values,
    _portfolio_meta_matrix,
    _score_portfolio_meta,
)

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


def _rank_rows(rows: list[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -_score_value(row, score_key),
            -_score_value(row, "reward"),
            int(row.get("proposal_index") if isinstance(row.get("proposal_index"), int) else 10**9),
            str(row.get("canonical_genome") or ""),
        ),
    )


def _score_value(row: dict[str, Any], key: str) -> float:
    value = _finite_float(row.get(key))
    if value is not None:
        return value
    reward = _finite_float(row.get("reward"))
    return reward if reward is not None else float("-inf")


def _portfolio_diversity_key(row: dict[str, Any]) -> str:
    existing = row.get("candidate_diversity_key")
    if isinstance(existing, str) and existing:
        return existing
    actions = _parse_actions(str(row.get("canonical_genome") or row.get("genome") or ""))
    transforms = [stage for stage in (actions or {}).get("transforms") or [] if isinstance(stage, dict)]
    first = str(transforms[0].get("op") if transforms else "none")
    return f"{row.get('head_type') or _head_from_actions(actions) or 'unknown'}:{first}"


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


def _policy_metric_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (
        _metric_value(metrics.get("win_rate")),
        _metric_value(metrics.get("median_delta")),
        _metric_value(metrics.get("mean_delta")),
    )


def _metric_value(value: Any) -> float:
    numeric = _finite_float(value)
    return float("-inf") if numeric is None else numeric


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
