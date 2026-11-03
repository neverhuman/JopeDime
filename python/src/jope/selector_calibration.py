from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .io import read_jsonl
from .prime_config import load_config

DEFAULT_TRAIN_FEEDBACK_RUN = load_config().runs / "active_v2_train_feedback_current_code_20260705_232346"
DEFAULT_EVALUATION_JSONL = DEFAULT_TRAIN_FEEDBACK_RUN / "evaluation.jsonl"
DEFAULT_PROPOSALS_JSONL = DEFAULT_TRAIN_FEEDBACK_RUN / "proposals.jsonl"
DEFAULT_OUTPUT_JSON = DEFAULT_TRAIN_FEEDBACK_RUN / "selector_calibration.json"


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
                "reward_std": _finite_float(
                    merged.get("reward_std")
                    if merged.get("reward_std") is not None
                    else merged.get("uncertainty")
                ),
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
                raise ValueError(f"realization adapter requires {MODEL_REALIZATION_SCHEMA!r} rows")
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
            raise ValueError(f"missing finite realized incumbent for dataset_hash={key[0]!r}")
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
        head = str(candidate.get("head_type") or _head_from_actions(actions) or "unknown")
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
                "reward_percentile_in_engine": _finite_float(candidate.get("reward_percentile_in_engine")),
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
                "execution_runtime_ms": _finite_float(anchor.get("execution_runtime_ms")),
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
    score = _finite_float(
        metadata.get("score_robust") if metadata.get("score_robust") is not None else raw.get("score")
    )
    raw_score = _finite_float(
        metadata.get("oof_r2_raw") if metadata.get("oof_r2_raw") is not None else raw.get("score")
    )
    delta = _finite_float(metadata.get("oof_delta_to_incumbent"))
    if delta is None:
        incumbent = _finite_float(
            metadata.get("best_known_robust")
            if metadata.get("best_known_robust") is not None
            else metadata.get("best_known_oof_r2")
        )
        if score is not None and incumbent is not None:
            delta = score - incumbent
    if delta is None:
        return None
    actions = (
        metadata.get("actions") if isinstance(metadata.get("actions"), dict) else _parse_actions(canonical)
    )
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
        "reward_std": _finite_float(
            metadata.get("surrogate_sigma")
            if metadata.get("surrogate_sigma") is not None
            else raw.get("reward_std")
        ),
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


def _stringify_paths(
    paths: str | Path | list[str | Path] | tuple[str | Path, ...] | None,
) -> list[str] | str | None:
    if paths is None:
        return None
    if isinstance(paths, (str, Path)):
        return str(paths)
    return [str(path) for path in paths]

from .selector_calibration_fitting import (
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
    _action_feature_values,
    _bin_index,
    _candidate_pattern_penalties,
    _choose_policy,
    _correlation_lists,
    _delta_metrics,
    _engine_head_key,
    _finite_float,
    _fit_calibration_model,
    _fit_pattern_penalties,
    _fit_portfolio_meta_model,
    _fit_reward_bins,
    _head_from_actions,
    _head_from_genome,
    _lookup_bin_stats,
    _lookup_engine_stats,
    _meta_categories,
    _metric_value,
    _parse_actions,
    _parse_actions_for_engine,
    _policy_metric_key,
    _portfolio_delta_metrics,
    _portfolio_diversity_key,
    _portfolio_meta_feature_names,
    _portfolio_meta_feature_values,
    _portfolio_meta_matrix,
    _rank_rows,
    _score_portfolio_meta,
    _score_value,
    _select_portfolio_by_dataset,
    _select_portfolio_rows,
    _select_top_by_dataset,
    _selection_metrics,
    _stats,
    pattern_flags,
    score_candidate_with_calibration,
)
