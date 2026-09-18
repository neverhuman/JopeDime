from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .data.active import parse_pipeline_actions
from .io import read_jsonl
from .prime_config import load_config

SCHEMA_VERSION = "active_selector_calibration.v1"
MODEL_REALIZATION_SCHEMA = "jope-prime-model-realization-v2"
PORTFOLIO_META_POLICY = "portfolio_meta_v1"
PORTFOLIO_BASELINE_POLICY = "portfolio_baseline"
BIN_POLICIES = ("raw_reward", "head_reward_bins", "robust_pattern_bins")
POLICIES = (*BIN_POLICIES, PORTFOLIO_META_POLICY)
DEFAULT_TRAIN_FEEDBACK_RUN = (
    load_config().runs / "active_v2_train_feedback_current_code_20260705_232346"
)
DEFAULT_EVALUATION_JSONL = DEFAULT_TRAIN_FEEDBACK_RUN / "evaluation.jsonl"
DEFAULT_PROPOSALS_JSONL = DEFAULT_TRAIN_FEEDBACK_RUN / "proposals.jsonl"
DEFAULT_OUTPUT_JSON = DEFAULT_TRAIN_FEEDBACK_RUN / "selector_calibration.json"

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


def load_selector_calibration(path_or_data: str | Path | dict[str, Any] | None) -> dict[str, Any] | None:
    if path_or_data is None:
        return None
    if isinstance(path_or_data, dict):
        data = path_or_data
    else:
        data = json.loads(Path(path_or_data).read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported selector calibration schema: {data.get('schema_version')!r}")
    if data.get("selected_policy") not in POLICIES:
        raise ValueError(f"unsupported selector policy: {data.get('selected_policy')!r}")
    return data


def build_selector_calibration(
    *,
    evaluation_jsonl: str | Path = DEFAULT_EVALUATION_JSONL,
    proposals_jsonl: str | Path | None = DEFAULT_PROPOSALS_JSONL,
    extra_training_jsonl: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
    split: str = "train",
    folds: int = 3,
    bin_count: int = 5,
    min_head_rows: int = 12,
    min_pattern_rows: int = 8,
) -> dict[str, Any]:
    rows = load_selector_training_rows(
        evaluation_jsonl,
        proposals_jsonl=proposals_jsonl,
        extra_training_jsonl=extra_training_jsonl,
        split=split,
    )
    cv = cross_validate_selector_policies(
        rows,
        folds=folds,
        bin_count=bin_count,
        min_head_rows=min_head_rows,
        min_pattern_rows=min_pattern_rows,
    )
    selected_policy = _choose_policy(cv)
    fit = _fit_calibration_model(
        rows,
        bin_count=bin_count,
        min_head_rows=min_head_rows,
        min_pattern_rows=min_pattern_rows,
    )
    fit[PORTFOLIO_META_POLICY] = _fit_portfolio_meta_model(rows, fit)
    return {
        "schema_version": SCHEMA_VERSION,
        "selected_policy": selected_policy,
        "candidate_policies": list(POLICIES),
        "source": {
            "evaluation_jsonl": str(evaluation_jsonl),
            "proposals_jsonl": str(proposals_jsonl) if proposals_jsonl else None,
            "extra_training_jsonl": _stringify_paths(extra_training_jsonl),
            "split": split,
            "row_count": len(rows),
            "dataset_count": len({row["dataset_hash"] for row in rows}),
            "target_delta_floor": _TARGET_DELTA_FLOOR,
            "calibrated_score_weight": _CALIBRATED_SCORE_WEIGHT,
        },
        "fit": {
            **fit,
            "bin_count": int(bin_count),
            "min_head_rows": int(min_head_rows),
            "min_pattern_rows": int(min_pattern_rows),
        },
        "cv": cv,
    }


def write_selector_calibration(calibration: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def benchmark_selector(
    *,
    evaluation_jsonl: str | Path = DEFAULT_EVALUATION_JSONL,
    proposals_jsonl: str | Path | None = DEFAULT_PROPOSALS_JSONL,
    calibration: str | Path | dict[str, Any] | None = None,
    extra_training_jsonl: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
    split: str = "train",
    folds: int = 3,
) -> dict[str, Any]:
    rows = load_selector_training_rows(
        evaluation_jsonl,
        proposals_jsonl=proposals_jsonl,
        extra_training_jsonl=extra_training_jsonl,
        split=split,
    )
    raw_calibration = {
        "schema_version": SCHEMA_VERSION,
        "selected_policy": "raw_reward",
        "fit": _fit_calibration_model(rows),
    }
    metrics = {"raw_reward": _selection_metrics(rows, raw_calibration, "raw_reward")}
    data = load_selector_calibration(calibration) if calibration is not None else None
    if data is not None:
        policy = str(data.get("selected_policy") or "raw_reward")
        metrics[policy] = _selection_metrics(rows, data, policy)
    else:
        metrics["cross_validation"] = cross_validate_selector_policies(rows, folds=folds)
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_jsonl": str(evaluation_jsonl),
        "proposals_jsonl": str(proposals_jsonl) if proposals_jsonl else None,
        "extra_training_jsonl": _stringify_paths(extra_training_jsonl),
        "split": split,
        "row_count": len(rows),
        "dataset_count": len({row["dataset_hash"] for row in rows}),
        "metrics": metrics,
    }


def load_selector_training_rows(
    evaluation_jsonl: str | Path,
    *,
    proposals_jsonl: str | Path | None = None,
    extra_training_jsonl: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
    split: str = "train",
) -> list[dict[str, Any]]:
    proposal_lookup = _proposal_lookup(proposals_jsonl) if proposals_jsonl else {}
    rows: list[dict[str, Any]] = []
    for raw in read_jsonl(evaluation_jsonl):
        if split and raw.get("split") != split:
            continue
        if raw.get("valid") is False:
            continue
        dataset_hash = str(raw.get("dataset_hash") or "")
        if not dataset_hash:
            continue
        proposal_index = raw.get("proposal_index")
        canonical = str(raw.get("canonical_genome") or raw.get("genome") or "")
        engine = str(raw.get("engine") or _DEFAULT_ENGINE)
        proposal = _lookup_proposal(proposal_lookup, dataset_hash, engine, canonical, proposal_index)
        merged = {**proposal, **raw}
        engine = str(merged.get("engine") or engine or _DEFAULT_ENGINE)
        canonical = str(merged.get("canonical_genome") or merged.get("genome") or "")
        reward = _finite_float(merged.get("reward"))
        raw_delta = _finite_float(merged.get("oof_delta_vs_incumbent"))
        if raw_delta is None:
            cv = merged.get("cv") if isinstance(merged.get("cv"), dict) else {}
            oof_r2 = _finite_float(cv.get("oof_r2"))
            incumbent = _finite_float(merged.get("incumbent_oof_r2"))
            raw_delta = oof_r2 - incumbent if oof_r2 is not None and incumbent is not None else None
        if reward is None or raw_delta is None or not canonical:
            continue
        delta = max(float(raw_delta), _TARGET_DELTA_FLOOR)
        actions = _parse_actions(canonical)
        head = str(merged.get("head_type") or _head_from_actions(actions) or "unknown")
        patterns = pattern_flags(actions)
        action_features = _action_feature_values(actions)
        rows.append(
            {
                "dataset_hash": dataset_hash,
                "engine": engine,
                "proposal_index": int(proposal_index) if isinstance(proposal_index, int) else None,
                "canonical_genome": canonical,
                "reward": reward,
                "reward_std": _finite_float(merged.get("reward_std") if merged.get("reward_std") is not None else merged.get("uncertainty")),
                "reward_percentile_in_engine": _finite_float(merged.get("reward_percentile_in_engine")),
                "p_bad": _finite_float(merged.get("p_bad")),
                "delta": delta,
                "raw_delta": float(raw_delta),
                "win": delta > 0.0,
                "head_type": head,
                "patterns": patterns,
                **action_features,
                "split": raw.get("split"),
            }
        )
    rows.extend(_load_extra_training_rows(extra_training_jsonl, split=split))
    rows.sort(
        key=lambda row: (
            row["dataset_hash"],
            row["engine"],
            row["proposal_index"] if row["proposal_index"] is not None else 10**9,
            row["canonical_genome"],
        )
    )
    return rows


def load_realization_selector_rows(
    realization_jsonl: str | Path | list[str | Path] | tuple[str | Path, ...],
    *,
    candidate_pool_jsonl: str | Path | list[str | Path] | tuple[str | Path, ...],
    incumbent_realized_scores: Mapping[str, float],
    split: str = "rest",
) -> list[dict[str, Any]]:
    """Join verified realization labels to the proposal features that produced them.

    This is deliberately opt-in and separate from the legacy canonical-genome join.
    ``semantic_plan_hash`` is the candidate identity; ``artifact_hash`` remains
    language-specific provenance.  An explicit realized incumbent is required because
    substituting a CV incumbent would mix incompatible label semantics.
    """
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in _iter_paths(candidate_pool_jsonl):
        for raw in read_jsonl(path):
            semantic_plan_hash = str(raw.get("semantic_plan_hash") or "")
            if not semantic_plan_hash:
                continue
            dataset_hash = str(raw.get("dataset_hash") or "")
            if not dataset_hash or not _is_sha256(semantic_plan_hash):
                raise ValueError(
                    "realization candidate identities require dataset_hash and a "
                    "lowercase SHA-256 semantic_plan_hash"
                )
            candidates[(dataset_hash, semantic_plan_hash)].append(dict(raw))

    grouped_labels: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in _iter_paths(realization_jsonl):
        for raw in read_jsonl(path):
            if raw.get("schema") != MODEL_REALIZATION_SCHEMA:
                raise ValueError(
                    f"realization adapter requires {MODEL_REALIZATION_SCHEMA!r} rows"
                )
            if split and str(raw.get("split") or "") != split:
                continue
            _validate_selector_realization_label(raw)
            key = (str(raw["dataset_hash"]), str(raw["semantic_plan_hash"]))
            grouped_labels[key].append(dict(raw))
    if not grouped_labels:
        raise ValueError(f"no verified realization labels found for split {split!r}")

    rows: list[dict[str, Any]] = []
    for key in sorted(grouped_labels):
        labels = grouped_labels[key]
        anchor = labels[0]
        _validate_realization_label_group(labels)
        matches = candidates.get(key, [])
        if len(matches) != 1:
            raise ValueError(
                "realization identity must map to exactly one candidate row: "
                f"dataset_hash={key[0]!r} semantic_plan_hash={key[1]!r} "
                f"matches={len(matches)}"
            )
        candidate = matches[0]
        canonical = str(candidate.get("canonical_genome") or candidate.get("genome") or "")
        label_canonical = str(anchor.get("canonical_genome") or "")
        if not canonical or canonical != label_canonical:
            raise ValueError("realization candidate canonical_genome mismatch")
        engine = str(candidate.get("engine") or "")
        if engine != "prime" or str(anchor.get("engine") or "") != "prime":
            raise ValueError("realization selector evidence must use the Prime engine")
        for field in ("target_hash", "fold_plan_hash"):
            candidate_value = candidate.get(field)
            if candidate_value is not None and candidate_value != anchor.get(field):
                raise ValueError(f"realization candidate {field} mismatch")

        incumbent = _finite_float(incumbent_realized_scores.get(key[0]))
        if incumbent is None:
            raise ValueError(
                f"missing finite realized incumbent for dataset_hash={key[0]!r}"
            )
        reward = _finite_float(
            candidate.get("reward_mean")
            if candidate.get("reward_mean") is not None
            else candidate.get("reward")
        )
        if reward is None:
            raise ValueError("realization candidate has no finite proposal reward")
        realized_score = float(anchor["realized_score"])
        raw_delta = realized_score - incumbent
        actions = _parse_actions_for_engine(canonical, engine)
        head = str(
            candidate.get("head_type") or _head_from_actions(actions) or "unknown"
        )
        artifact_hashes = {
            str(label["target"]): str(label["artifact_hash"])
            for label in sorted(labels, key=lambda item: str(item["target"]))
        }
        rows.append(
            {
                "dataset_hash": key[0],
                "engine": engine,
                "proposal_index": candidate.get("proposal_index"),
                "candidate_pool_id": candidate.get("candidate_pool_id"),
                "canonical_genome": canonical,
                "semantic_plan_hash": key[1],
                "artifact_hashes": artifact_hashes,
                "target_hash": anchor["target_hash"],
                "fold_plan_hash": anchor["fold_plan_hash"],
                "reward": reward,
                "reward_std": _finite_float(candidate.get("reward_std")),
                "reward_percentile_in_engine": _finite_float(
                    candidate.get("reward_percentile_in_engine")
                ),
                "p_bad": _finite_float(candidate.get("p_bad")),
                "realized_score": realized_score,
                "incumbent_realized_score": incumbent,
                "delta": max(raw_delta, _TARGET_DELTA_FLOOR),
                "raw_delta": raw_delta,
                "win": raw_delta > 0.0,
                "head_type": head,
                "patterns": pattern_flags(actions),
                **_action_feature_values(actions),
                "split": split,
                "source": "prime_model_realization",
                "execution_runtime_ms": _finite_float(
                    anchor.get("execution_runtime_ms")
                ),
                "peak_memory_bytes": anchor.get("peak_memory_bytes"),
            }
        )
    return rows


def dataset_group_folds(rows: list[dict[str, Any]], folds: int = 3) -> dict[str, int]:
    datasets = sorted({str(row.get("dataset_hash")) for row in rows if row.get("dataset_hash") is not None})
    if not datasets:
        return {}
    fold_count = max(1, min(int(folds), len(datasets)))
    return {dataset_hash: idx % fold_count for idx, dataset_hash in enumerate(datasets)}


def cross_validate_selector_policies(
    rows: list[dict[str, Any]],
    *,
    folds: int = 3,
    bin_count: int = 5,
    min_head_rows: int = 12,
    min_pattern_rows: int = 8,
) -> dict[str, Any]:
    fold_map = dataset_group_folds(rows, folds=folds)
    fold_count = max(fold_map.values(), default=-1) + 1
    metrics: dict[str, dict[str, Any]] = {}
    for policy in BIN_POLICIES:
        selected_rows: list[dict[str, Any]] = []
        fold_details = []
        for fold_idx in range(fold_count):
            train = [row for row in rows if fold_map.get(row["dataset_hash"]) != fold_idx]
            heldout = [row for row in rows if fold_map.get(row["dataset_hash"]) == fold_idx]
            calibration = {
                "schema_version": SCHEMA_VERSION,
                "selected_policy": policy,
                "fit": _fit_calibration_model(
                    train,
                    bin_count=bin_count,
                    min_head_rows=min_head_rows,
                    min_pattern_rows=min_pattern_rows,
                ),
            }
            selected = _select_top_by_dataset(heldout, calibration, policy)
            selected_rows.extend(selected)
            fold_details.append({"fold": fold_idx, **_delta_metrics([row["delta"] for row in selected])})
        metrics[policy] = {
            **_delta_metrics([row["delta"] for row in selected_rows]),
            "folds": fold_details,
            "selection_objective": "top1_by_dataset",
        }
    for policy in (PORTFOLIO_BASELINE_POLICY, PORTFOLIO_META_POLICY):
        selected_groups: list[list[dict[str, Any]]] = []
        fold_details = []
        for fold_idx in range(fold_count):
            train = [row for row in rows if fold_map.get(row["dataset_hash"]) != fold_idx]
            heldout = [row for row in rows if fold_map.get(row["dataset_hash"]) == fold_idx]
            fit = _fit_calibration_model(
                train,
                bin_count=bin_count,
                min_head_rows=min_head_rows,
                min_pattern_rows=min_pattern_rows,
            )
            if policy == PORTFOLIO_META_POLICY:
                fit[PORTFOLIO_META_POLICY] = _fit_portfolio_meta_model(train, fit)
                calibration_policy = PORTFOLIO_META_POLICY
            else:
                calibration_policy = "raw_reward"
            calibration = {
                "schema_version": SCHEMA_VERSION,
                "selected_policy": calibration_policy,
                "fit": fit,
            }
            groups = _select_portfolio_by_dataset(heldout, calibration, calibration_policy)
            selected_groups.extend(groups)
            fold_details.append({"fold": fold_idx, **_portfolio_delta_metrics(groups)})
        metrics[policy] = {
            **_portfolio_delta_metrics(selected_groups),
            "folds": fold_details,
            "selection_objective": "best_of_portfolio_by_dataset",
        }
    return {
        "folds": int(fold_count),
        "dataset_fold_map": fold_map,
        "policies": metrics,
        "selected_policy": _choose_policy({"policies": metrics}),
    }


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
    pattern_penalties = _candidate_pattern_penalties(row, calibration) if policy in {"robust_pattern_bins", PORTFOLIO_META_POLICY} else {}
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
        if str(stage.get("op") or "") in _EXPANSIVE_TRANSFORMS and bool((stage.get("params") or {}).get("append"))
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
                "source": head if head in head_bins and not bool(head_bins[head].get("fallback")) else _GLOBAL_HEAD,
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
    return np.zeros((0, len(feature_names)), dtype=np.float64), np.zeros((0,), dtype=np.float64), feature_names


def _fit_reward_bins(rows: list[dict[str, Any]], *, bin_count: int, head: str) -> dict[str, Any]:
    rewards = [float(row["reward"]) for row in rows if _finite_float(row.get("reward")) is not None]
    if rewards:
        quantiles = np.quantile(np.asarray(rewards, dtype=np.float64), np.linspace(0.0, 1.0, max(1, bin_count) + 1)).tolist()
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
        loss_heavy = (
            len(deltas) >= min_pattern_rows
            and (
                float(stats["win_rate"]) < float(global_stats["win_rate"])
                or float(stats["expected_delta"]) < float(global_stats["expected_delta"])
            )
        )
        win_gap = max(0.0, float(global_stats["win_rate"]) - float(stats["win_rate"])) if loss_heavy else 0.0
        delta_gap = max(0.0, float(global_stats["expected_delta"]) - float(stats["expected_delta"])) if loss_heavy else 0.0
        tail_gap = max(0.0, float(stats["tail_risk"]) - float(global_stats["tail_risk"])) if loss_heavy else 0.0
        penalty = min(2.0, 0.08 * win_gap + 0.35 * delta_gap + 0.05 * tail_gap)
        out[pattern] = {
            **stats,
            "loss_heavy": bool(loss_heavy),
            "penalty": float(penalty),
        }
    return out


def _selection_metrics(rows: list[dict[str, Any]], calibration: dict[str, Any], policy: str) -> dict[str, Any]:
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


def _select_top_by_dataset(rows: list[dict[str, Any]], calibration: dict[str, Any], policy: str) -> list[dict[str, Any]]:
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
        groups.append(_select_portfolio_rows(scored, calibrated_score_key="selection_score" if policy != "raw_reward" else None))
    return groups


def _select_portfolio_rows(rows: list[dict[str, Any]], *, calibrated_score_key: str | None) -> list[dict[str, Any]]:
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
        bucket = [row for row in safe if id(row) not in selected_ids and str(row.get("head_type") or "") == head]
        if not bucket:
            continue
        diverse = [row for row in bucket if _portfolio_diversity_key(row) not in selected_diversity]
        add(_rank_rows(diverse or bucket, calibrated_score_key or "reward")[0])
        break
    for pool in (
        [row for row in safe if id(row) not in selected_ids and _portfolio_diversity_key(row) not in selected_diversity],
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


def _lookup_bin_stats(row: dict[str, Any], calibration: dict[str, Any], reward: float) -> tuple[dict[str, Any], dict[str, Any]]:
    fit = calibration.get("fit") if isinstance(calibration.get("fit"), dict) else {}
    global_bins = fit.get("global_bins") if isinstance(fit.get("global_bins"), dict) else _fit_reward_bins([], bin_count=5, head=_GLOBAL_HEAD)
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
    if not names or mean.shape[0] != len(names) or scale.shape[0] != len(names) or coef.shape[0] != len(names):
        return None
    categories = model.get("categories") if isinstance(model.get("categories"), dict) else _meta_categories([])
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
    values.update({key: float(value) for key, value in bin_components.items() if isinstance(value, (int, float, bool))})
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
        if str(stage.get("op") or "") in _EXPANSIVE_TRANSFORMS and bool((stage.get("params") or {}).get("append"))
    )
    gpr_periodic = 0
    if head_op == "gpr":
        kernel = str((head.get("params") or {}).get("kernel") or "rbf")
        gpr_periodic = int("periodic" in kernel)
    return {
        "transform_count": int(actions.get("transform_count") if isinstance(actions.get("transform_count"), int) else len(transforms)),
        "repeated_scaler": int(sum(counts.get(op, 0) for op in _SCALER_TRANSFORMS) > 1),
        "target_bin": int(counts.get("target_bin_encode", 0) > 0),
        "appended_expansion_count": int(appended_expansion_count),
        "gpr_periodic": int(gpr_periodic),
    }


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


def _proposal_lookup(path: str | Path) -> dict[tuple[str, str, str, int | None], dict[str, Any]]:
    lookup: dict[tuple[str, str, str, int | None], dict[str, Any]] = {}
    try:
        iterator = read_jsonl(path)
        for row in iterator:
            dataset_hash = str(row.get("dataset_hash") or "")
            engine = str(row.get("engine") or _DEFAULT_ENGINE)
            canonical = str(row.get("canonical_genome") or row.get("genome") or "")
            proposal_index = row.get("proposal_index")
            index = int(proposal_index) if isinstance(proposal_index, int) else None
            if dataset_hash and canonical:
                lookup.setdefault((dataset_hash, engine, canonical, index), row)
                lookup.setdefault((dataset_hash, engine, canonical, None), row)
            if dataset_hash and index is not None:
                lookup.setdefault((dataset_hash, engine, "", index), row)
    except OSError:
        return {}
    return lookup


def _lookup_proposal(
    lookup: dict[tuple[str, str, str, int | None], dict[str, Any]],
    dataset_hash: str,
    engine: str,
    canonical: str,
    proposal_index: Any,
) -> dict[str, Any]:
    index = int(proposal_index) if isinstance(proposal_index, int) else None
    return (
        lookup.get((dataset_hash, engine, canonical, index))
        or lookup.get((dataset_hash, engine, canonical, None))
        or lookup.get((dataset_hash, engine, "", index))
        or lookup.get((dataset_hash, _DEFAULT_ENGINE, canonical, index))
        or lookup.get((dataset_hash, _DEFAULT_ENGINE, canonical, None))
        or lookup.get((dataset_hash, _DEFAULT_ENGINE, "", index))
        or {}
    )


def _load_extra_training_rows(
    paths: str | Path | list[str | Path] | tuple[str | Path, ...] | None,
    *,
    split: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in _iter_paths(paths):
        try:
            iterator = read_jsonl(path)
            for raw in iterator:
                row = _selector_row_from_training_json(raw, split=split)
                if row is not None:
                    out.append(row)
        except OSError:
            continue
    return out


def _selector_row_from_training_json(raw: dict[str, Any], *, split: str) -> dict[str, Any] | None:
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    row_split = str(metadata.get("split") or raw.get("split") or "train")
    if split and row_split != split:
        return None
    dataset_hash = str(raw.get("dataset_id") or metadata.get("dataset_hash") or raw.get("dataset_hash") or "")
    canonical = str(raw.get("canonical_genome") or raw.get("genome") or "")
    if not dataset_hash or not canonical:
        return None
    reward = _finite_float(
        metadata.get("surrogate_mu")
        if metadata.get("surrogate_mu") is not None
        else metadata.get("reward")
        if metadata.get("reward") is not None
        else raw.get("reward")
    )
    if reward is None:
        return None
    score = _finite_float(metadata.get("score_robust") if metadata.get("score_robust") is not None else raw.get("score"))
    raw_score = _finite_float(metadata.get("oof_r2_raw") if metadata.get("oof_r2_raw") is not None else raw.get("score"))
    delta = _finite_float(metadata.get("oof_delta_to_incumbent"))
    if delta is None:
        incumbent = _finite_float(metadata.get("best_known_robust") if metadata.get("best_known_robust") is not None else metadata.get("best_known_oof_r2"))
        if score is not None and incumbent is not None:
            delta = score - incumbent
    if delta is None:
        return None
    actions = metadata.get("actions") if isinstance(metadata.get("actions"), dict) else _parse_actions(canonical)
    head = str(raw.get("head_type") or _head_from_actions(actions) or "unknown")
    action_features = _action_feature_values(actions)
    p_bad = _finite_float(metadata.get("p_bad"))
    if p_bad is None and metadata.get("eval_error"):
        p_bad = 1.0
    return {
        "dataset_hash": dataset_hash,
        "engine": str(raw.get("engine") or metadata.get("engine") or _DEFAULT_ENGINE),
        "proposal_index": None,
        "canonical_genome": canonical,
        "reward": reward,
        "reward_std": _finite_float(metadata.get("surrogate_sigma") if metadata.get("surrogate_sigma") is not None else raw.get("reward_std")),
        "reward_percentile_in_engine": _finite_float(
            metadata.get("rank_percentile_engine")
            if metadata.get("rank_percentile_engine") is not None
            else raw.get("reward_percentile_in_engine")
        ),
        "p_bad": p_bad,
        "delta": max(float(delta), _TARGET_DELTA_FLOOR),
        "raw_delta": float(delta if raw_score is None else raw_score),
        "win": float(delta) > 0.0,
        "head_type": head,
        "patterns": pattern_flags(actions),
        **action_features,
        "split": row_split,
        "source": str(raw.get("source") or metadata.get("provenance") or "extra_training"),
    }


def _validate_selector_realization_label(raw: dict[str, Any]) -> None:
    if raw.get("promotion_eligible") is not True:
        raise ValueError("realization selector evidence must be promotion eligible")
    if raw.get("failure") not in (None, ""):
        raise ValueError("failed realization evidence cannot train the selector")
    if str(raw.get("target") or "") not in {"portable_python", "portable_rust"}:
        raise ValueError("realization selector evidence must come from a portable target")
    if not str(raw.get("dataset_hash") or ""):
        raise ValueError("realization selector evidence requires dataset_hash")
    if not str(raw.get("canonical_genome") or ""):
        raise ValueError("realization selector evidence requires canonical_genome")
    for field in (
        "semantic_plan_hash",
        "artifact_hash",
        "target_hash",
        "fold_plan_hash",
        "holdout_predictions_sha256",
        "oof_predictions_sha256",
    ):
        if not _is_sha256(str(raw.get(field) or "")):
            raise ValueError(f"realization selector evidence has invalid {field}")
    if _finite_float(raw.get("realized_score")) is None:
        raise ValueError("realization selector evidence requires finite realized_score")


def _validate_realization_label_group(labels: list[dict[str, Any]]) -> None:
    anchor = labels[0]
    shared_fields = (
        "dataset_id",
        "dataset_hash",
        "semantic_plan_hash",
        "canonical_genome",
        "target_hash",
        "fold_plan_hash",
        "realized_score",
        "holdout_predictions_sha256",
        "oof_predictions_sha256",
        "oof_r2",
        "plan_version",
        "registry_version",
        "registry_hash",
        "input_columns",
        "seeds",
        "operator_versions",
    )
    for label in labels[1:]:
        for field in shared_fields:
            if label.get(field) != anchor.get(field):
                raise ValueError(
                    "portable realization targets disagree for "
                    f"semantic_plan_hash={anchor['semantic_plan_hash']!r}: {field}"
                )
    artifacts_by_target: dict[str, set[str]] = defaultdict(set)
    for label in labels:
        artifacts_by_target[str(label["target"])].add(str(label["artifact_hash"]))
    if any(len(artifacts) != 1 for artifacts in artifacts_by_target.values()):
        raise ValueError("one portable target maps to multiple realization artifacts")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _iter_paths(paths: str | Path | list[str | Path] | tuple[str | Path, ...] | None) -> list[Path]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(path) for path in paths]


def _stringify_paths(paths: str | Path | list[str | Path] | tuple[str | Path, ...] | None) -> list[str] | str | None:
    if paths is None:
        return None
    if isinstance(paths, (str, Path)):
        return str(paths)
    return [str(path) for path in paths]


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
