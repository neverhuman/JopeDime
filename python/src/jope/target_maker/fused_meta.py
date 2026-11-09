from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .classical import (
    PairIndex,
    RetrievalQuery,
    _model_artifact_name,
    _score_model,
    retrieval_metrics_from_scores,
)
from .data import (
    TASK_BINARY,
)
from .fused import (
    FusedTrainingConfig,
    _BatchEncoding,
    _cosine_scores,
    _jsonable_config,
    _model_pair_scores_for_batch,
    _safe_unit,
    _score_report_with_retrieval,
    _TargetEncoding,
)


def _meta_feature_names(classical_names: Sequence[str]) -> list[str]:
    names = [f"classical_score:{name}" for name in classical_names]
    names.extend(
        [
            "neural_best_cosine",
            "neural_final_cosine",
            "fused_vector_cosine",
            "classical_component_cosine",
            "classical_component_l2",
            "abs_log_length_delta",
            "abs_log_effective_delta",
            "abs_replacement_delta",
            "same_task",
            "both_binary",
            "both_regression",
        ]
    )
    return names


def _meta_features(
    batch: _BatchEncoding,
    pairs: Sequence[PairIndex],
    classical_score_by_name: Mapping[str, np.ndarray],
    classical_names: Sequence[str],
    fused_scores: np.ndarray,
) -> np.ndarray:
    left_idx = np.asarray([pair.left for pair in pairs], dtype=np.int64)
    right_idx = np.asarray([pair.right for pair in pairs], dtype=np.int64)
    rows = np.empty((len(pairs), len(_meta_feature_names(classical_names))), dtype=np.float32)
    cursor = 0
    for name in classical_names:
        rows[:, cursor] = np.asarray(classical_score_by_name[name], dtype=np.float32)
        cursor += 1

    class_left = batch.classical_components[left_idx]
    class_right = batch.classical_components[right_idx]
    left_norm = np.linalg.norm(class_left, axis=1)
    right_norm = np.linalg.norm(class_right, axis=1)
    denom = np.maximum(left_norm * right_norm, 1.0e-12)
    class_cos = np.sum(class_left * class_right, axis=1) / denom
    class_cos = np.where((left_norm > 1.0e-12) & (right_norm > 1.0e-12), class_cos, 0.0)
    class_l2 = np.linalg.norm(class_left - class_right, axis=1)
    left_task = batch.tasks[left_idx]
    right_task = batch.tasks[right_idx]
    log_lengths = np.log1p(batch.original_lengths)
    log_effective = np.log1p(batch.effective_counts)
    tail = np.column_stack(
        [
            np.sum(batch.neural_best[left_idx] * batch.neural_best[right_idx], axis=1),
            np.sum(batch.neural_final[left_idx] * batch.neural_final[right_idx], axis=1),
            np.asarray(fused_scores, dtype=np.float32),
            class_cos,
            class_l2,
            np.abs(log_lengths[left_idx] - log_lengths[right_idx]),
            np.abs(log_effective[left_idx] - log_effective[right_idx]),
            np.abs(batch.replacement_fractions[left_idx] - batch.replacement_fractions[right_idx]),
            (left_task == right_task).astype(np.float32),
            ((left_task == TASK_BINARY) & (right_task == TASK_BINARY)).astype(np.float32),
            ((left_task != TASK_BINARY) & (right_task != TASK_BINARY)).astype(np.float32),
        ]
    ).astype(np.float32, copy=False)
    rows[:, cursor : cursor + tail.shape[1]] = tail
    return np.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)


def _meta_features_for_two(
    left: _TargetEncoding,
    right: _TargetEncoding,
    classical_score_by_name: Mapping[str, float],
    classical_names: Sequence[str],
    fused_score: float,
) -> np.ndarray:
    row = np.empty((1, len(_meta_feature_names(classical_names))), dtype=np.float32)
    cursor = 0
    for name in classical_names:
        row[0, cursor] = float(classical_score_by_name[name])
        cursor += 1
    class_cos = float(np.dot(_safe_unit(left.classical_component), _safe_unit(right.classical_component)))
    class_l2 = float(np.linalg.norm(left.classical_component - right.classical_component))
    values = [
        float(np.dot(left.neural_best, right.neural_best)),
        float(np.dot(left.neural_final, right.neural_final)),
        float(fused_score),
        class_cos,
        class_l2,
        abs(math.log1p(float(left.original_length)) - math.log1p(float(right.original_length))),
        abs(math.log1p(float(left.effective_count)) - math.log1p(float(right.effective_count))),
        abs(float(left.replacement_fraction) - float(right.replacement_fraction)),
        float(left.task == right.task),
        float(left.task == TASK_BINARY and right.task == TASK_BINARY),
        float(left.task != TASK_BINARY and right.task != TASK_BINARY),
    ]
    row[0, cursor : cursor + len(values)] = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)


def _fit_meta_model(kind: str, seed: int) -> object:
    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2000,
                random_state=seed,
            ),
        )
    if kind == "hist_gbdt":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0e-3,
            random_state=seed,
        )
    raise ValueError(f"Unknown fused meta model kind: {kind!r}")


def _fit_meta_candidates(
    bundle_dir: Path,
    cfg: FusedTrainingConfig,
    batches_by_weight: Mapping[str, Mapping[str, _BatchEncoding]],
    pair_indices: Mapping[str, Sequence[PairIndex]],
    retrieval_queries: Mapping[str, Sequence[RetrievalQuery]],
    classical_bundles: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    import joblib

    scorer_dir = bundle_dir / "scorers"
    scorer_dir.mkdir(parents=True, exist_ok=True)
    classical_names = sorted(classical_bundles)
    y_train = np.asarray([pair.label for pair in pair_indices["train"]], dtype=np.int64)
    y_val = np.asarray([pair.label for pair in pair_indices["val"]], dtype=np.int64)
    y_test = np.asarray([pair.label for pair in pair_indices["test"]], dtype=np.int64)

    reports: dict[str, object] = {}
    artifacts: dict[str, object] = {}
    base_batches = next(iter(batches_by_weight.values()))
    classical_pair_scores = {
        split: {
            name: _model_pair_scores_for_batch(bundle, base_batches[split], pair_indices[split])
            for name, bundle in classical_bundles.items()
        }
        for split in ("train", "val", "test")
    }
    retrieval_flat = {split: _flatten_retrieval_pairs(retrieval_queries[split]) for split in ("val", "test")}
    retrieval_classical_scores = {
        split: {
            name: (
                _model_pair_scores_for_batch(bundle, base_batches[split], retrieval_flat[split][0])
                if retrieval_flat[split][0]
                else np.zeros(0, dtype=np.float64)
            )
            for name, bundle in classical_bundles.items()
        }
        for split in ("val", "test")
    }
    for weight_key, split_batches in batches_by_weight.items():
        train_batch = split_batches["train"]
        val_batch = split_batches["val"]
        test_batch = split_batches["test"]
        train_fused_scores = _cosine_scores(train_batch.vectors, pair_indices["train"])
        val_fused_scores = _cosine_scores(val_batch.vectors, pair_indices["val"])
        test_fused_scores = _cosine_scores(test_batch.vectors, pair_indices["test"])
        x_train = _meta_features(
            train_batch,
            pair_indices["train"],
            classical_pair_scores["train"],
            classical_names,
            train_fused_scores,
        )
        x_val = _meta_features(
            val_batch, pair_indices["val"], classical_pair_scores["val"], classical_names, val_fused_scores
        )
        x_test = _meta_features(
            test_batch,
            pair_indices["test"],
            classical_pair_scores["test"],
            classical_names,
            test_fused_scores,
        )

        for kind in ("logistic", "hist_gbdt"):
            model = _fit_meta_model(kind, cfg.seed)
            model.fit(x_train, y_train)
            val_scores = _score_model(model, x_val)
            test_scores = _score_model(model, x_test)
            val_retrieval = _meta_retrieval_from_flat(
                model,
                val_batch,
                retrieval_queries["val"],
                retrieval_flat["val"][0],
                retrieval_flat["val"][1],
                retrieval_classical_scores["val"],
                classical_names,
            )
            test_retrieval = _meta_retrieval_from_flat(
                model,
                test_batch,
                retrieval_queries["test"],
                retrieval_flat["test"][0],
                retrieval_flat["test"][1],
                retrieval_classical_scores["test"],
                classical_names,
            )
            key = f"{kind}__{weight_key}"
            artifact_path = scorer_dir / f"{_model_artifact_name(key)}.joblib"
            artifact = {
                "model": model,
                "kind": kind,
                "weight_key": weight_key,
                "feature_names": _meta_feature_names(classical_names),
                "classical_model_names": classical_names,
                "config": _jsonable_config(cfg),
            }
            joblib.dump(artifact, artifact_path)
            artifacts[key] = {**artifact, "artifact_path": str(artifact_path)}
            reports[key] = {
                **_score_report_with_retrieval(
                    y_val, val_scores, y_test, test_scores, val_retrieval, test_retrieval
                ),
                "kind": kind,
                "weight_key": weight_key,
                "artifact_path": str(artifact_path),
            }
    return reports, artifacts


def _meta_retrieval(
    model: object,
    batch: _BatchEncoding,
    queries: Sequence[RetrievalQuery],
    classical_bundles: Mapping[str, Mapping[str, object]],
    classical_names: Sequence[str],
) -> dict[str, float]:
    if not queries:
        return {}
    all_pairs, sizes = _flatten_retrieval_pairs(queries)
    classical_scores = {
        name: _model_pair_scores_for_batch(bundle, batch, all_pairs)
        for name, bundle in classical_bundles.items()
    }
    return _meta_retrieval_from_flat(
        model, batch, queries, all_pairs, sizes, classical_scores, classical_names
    )


def _flatten_retrieval_pairs(queries: Sequence[RetrievalQuery]) -> tuple[list[PairIndex], list[int]]:
    all_pairs: list[PairIndex] = []
    sizes: list[int] = []
    for query in queries:
        pairs = [PairIndex(int(query.query), int(candidate), 0) for candidate in query.candidates]
        sizes.append(len(pairs))
        all_pairs.extend(pairs)
    return all_pairs, sizes


def _meta_retrieval_from_flat(
    model: object,
    batch: _BatchEncoding,
    queries: Sequence[RetrievalQuery],
    all_pairs: Sequence[PairIndex],
    sizes: Sequence[int],
    classical_scores: Mapping[str, np.ndarray],
    classical_names: Sequence[str],
) -> dict[str, float]:
    if not queries:
        return {}
    fused_scores = _cosine_scores(batch.vectors, all_pairs)
    x = _meta_features(batch, all_pairs, classical_scores, classical_names, fused_scores)
    all_scores = _score_model(model, x)
    scores_by_query: list[np.ndarray] = []
    cursor = 0
    for size in sizes:
        scores_by_query.append(all_scores[cursor : cursor + size])
        cursor += size
    return retrieval_metrics_from_scores(scores_by_query, len(queries[0].candidates))


def _weight_grid() -> list[tuple[str, tuple[float, float, float]]]:
    candidates = [
        ("balanced", (1.0, 1.0, 1.0)),
        ("best_heavy", (1.5, 1.0, 0.75)),
        ("final_heavy", (1.0, 1.5, 0.75)),
        ("classical_heavy", (0.75, 0.75, 1.5)),
        ("neural_heavy", (1.5, 1.5, 0.5)),
        ("best_classical", (1.0, 0.0, 1.0)),
        ("final_classical", (0.0, 1.0, 1.0)),
        ("neural_only", (1.0, 1.0, 0.0)),
    ]
    return candidates
