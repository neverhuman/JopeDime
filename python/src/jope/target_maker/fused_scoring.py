from __future__ import annotations

import json
import math
import os
import shutil
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .artifact_pins import artifact_sha256, load_joblib_pinned
from .classical import (
    PairIndex,
    RetrievalQuery,
    WeightedScoreEnsemble,
    WindowFeature,
    _best_f1_threshold,
    _classification_metrics,
    _ensemble_members,
    _ensemble_requested,
    _limit_records,
    _model_artifact_name,
    _model_specs,
    _score_model,
    _split_records,
    _validation_weights,
    build_pair_indices,
    build_retrieval_queries,
    pair_feature_names,
    pair_features,
    retrieval_metrics_from_scores,
    select_window_features,
    summarize_target_window,
)
from .data import (
    TASK_BINARY,
    DatasetRecord,
    bits_at_indices,
    compute_profile,
    deterministic_shuffle,
    load_manifest,
    read_target_input,
    sample_indices,
    stable_seed,
)
from .encode import encode_values, load_encoder, normalize_task
from .model import build_model_from_checkpoint

from .fused import (
    FusedTrainingConfig,
    FusedWindow,
    _BatchEncoding,
    _jsonable_config,
    _score_report_with_retrieval,
    _window_features,
)

def _cosine_scores(vectors: np.ndarray, pairs: Sequence[PairIndex]) -> np.ndarray:
    return np.asarray(
        [float(np.dot(vectors[pair.left], vectors[pair.right])) for pair in pairs], dtype=np.float64
    )

def _pair_features_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.nan_to_num(np.asarray(left, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    right = np.nan_to_num(np.asarray(right, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    diff = left - right
    absdiff = np.abs(diff)
    denominator = np.where(np.abs(right) > 1.0e-6, right, np.where(right >= 0.0, 1.0e-6, -1.0e-6))
    ratio = np.clip(left / denominator, -1.0e4, 1.0e4)
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denom = np.maximum(left_norm * right_norm, 1.0e-12)
    cosine = np.sum(left * right, axis=1) / denom
    cosine = np.where((left_norm > 1.0e-12) & (right_norm > 1.0e-12), cosine, 0.0).astype(np.float32)
    global_features = np.column_stack(
        [
            absdiff.mean(axis=1),
            absdiff.max(axis=1),
            np.sqrt(np.mean(diff * diff, axis=1)),
            cosine,
            1.0 - cosine,
            np.mean(left * right, axis=1),
            diff.mean(axis=1),
            absdiff.std(axis=1),
        ]
    ).astype(np.float32, copy=False)
    out = np.concatenate(
        [
            diff,
            absdiff,
            diff * diff,
            np.minimum(left, right),
            np.maximum(left, right),
            ratio,
            np.log1p(absdiff),
            left * right,
            0.5 * (left + right),
            global_features,
        ],
        axis=1,
    )
    return np.nan_to_num(out.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)

def _pair_matrix_from_feature_matrix(
    features: np.ndarray, pairs: Sequence[PairIndex], *, chunk_size: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    if not pairs:
        raise ValueError("At least one pair is required")
    left_idx = np.asarray([pair.left for pair in pairs], dtype=np.int64)
    right_idx = np.asarray([pair.right for pair in pairs], dtype=np.int64)
    y = np.asarray([pair.label for pair in pairs], dtype=np.int64)
    pair_dim = pair_features(features[0], features[0]).shape[0]
    x = np.empty((len(pairs), pair_dim), dtype=np.float32)
    chunk = len(pairs) if chunk_size is None else max(1, int(chunk_size))
    for start in range(0, len(pairs), chunk):
        end = min(len(pairs), start + chunk)
        x[start:end] = _pair_features_matrix(features[left_idx[start:end]], features[right_idx[start:end]])
    return x, y

def _score_pairs_from_feature_matrix(
    model: object,
    features: np.ndarray,
    pairs: Sequence[PairIndex],
    *,
    chunk_size: int = 8192,
) -> np.ndarray:
    if not pairs:
        return np.zeros(0, dtype=np.float64)
    left_idx = np.asarray([pair.left for pair in pairs], dtype=np.int64)
    right_idx = np.asarray([pair.right for pair in pairs], dtype=np.int64)
    scores = np.empty(len(pairs), dtype=np.float64)
    chunk = max(1, int(chunk_size))
    for start in range(0, len(pairs), chunk):
        end = min(len(pairs), start + chunk)
        x = _pair_features_matrix(features[left_idx[start:end]], features[right_idx[start:end]])
        scores[start:end] = _score_model(model, x)
    return scores

def _retrieval_from_vectors(vectors: np.ndarray, queries: Sequence[RetrievalQuery]) -> dict[str, float]:
    if not queries:
        return {}
    scores_by_query: list[np.ndarray] = []
    for query in queries:
        candidates = np.asarray(query.candidates, dtype=np.int64)
        scores_by_query.append(vectors[candidates] @ vectors[int(query.query)])
    return retrieval_metrics_from_scores(scores_by_query, len(queries[0].candidates))

def _model_pair_scores_for_batch(
    model_bundle: Mapping[str, object],
    batch: _BatchEncoding,
    pairs: Sequence[PairIndex],
) -> np.ndarray:
    indices = tuple(
        int(index)
        for index in model_bundle.get("selected_feature_indices", range(batch.window_features.shape[1]))
    )
    features = batch.window_features[:, np.asarray(indices, dtype=np.int64)]
    return _score_pairs_from_feature_matrix(model_bundle["model"], features, pairs, chunk_size=8192)

def _model_retrieval_for_batch(
    model_bundle: Mapping[str, object],
    batch: _BatchEncoding,
    queries: Sequence[RetrievalQuery],
) -> dict[str, float]:
    if not queries:
        return {}
    indices = tuple(
        int(index)
        for index in model_bundle.get("selected_feature_indices", range(batch.window_features.shape[1]))
    )
    features = batch.window_features[:, np.asarray(indices, dtype=np.int64)]
    all_pairs: list[PairIndex] = []
    sizes: list[int] = []
    for query in queries:
        pairs = [PairIndex(int(query.query), int(candidate), 0) for candidate in query.candidates]
        sizes.append(len(pairs))
        all_pairs.extend(pairs)
    all_scores = _score_pairs_from_feature_matrix(model_bundle["model"], features, all_pairs, chunk_size=8192)
    scores_by_query: list[np.ndarray] = []
    cursor = 0
    for size in sizes:
        scores_by_query.append(all_scores[cursor : cursor + size])
        cursor += size
    return retrieval_metrics_from_scores(scores_by_query, len(queries[0].candidates))

def _classical_retrieval_from_features(
    model: object,
    features: np.ndarray,
    queries: Sequence[RetrievalQuery],
) -> dict[str, float]:
    if not queries:
        return {}
    all_pairs: list[PairIndex] = []
    sizes: list[int] = []
    for query in queries:
        pairs = [PairIndex(int(query.query), int(candidate), 0) for candidate in query.candidates]
        sizes.append(len(pairs))
        all_pairs.extend(pairs)
    all_scores = _score_pairs_from_feature_matrix(model, features, all_pairs, chunk_size=8192)
    scores_by_query: list[np.ndarray] = []
    cursor = 0
    for size in sizes:
        scores_by_query.append(all_scores[cursor : cursor + size])
        cursor += size
    return retrieval_metrics_from_scores(scores_by_query, len(queries[0].candidates))

def _fit_classical_models(
    bundle_dir: Path,
    cfg: FusedTrainingConfig,
    sampled: Mapping[str, list[FusedWindow]],
    feature_names: Sequence[str],
    pair_indices: Mapping[str, Sequence[PairIndex]],
    retrieval_queries: Mapping[str, Sequence[RetrievalQuery]],
) -> tuple[dict[str, Mapping[str, object]], dict[str, object]]:
    import joblib

    model_dir = bundle_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    requested = tuple(dict.fromkeys(name.strip() for name in cfg.models if name.strip()))
    base_requested = [name for name in requested if name != "ensemble"]
    if not base_requested:
        base_requested = ["sgd_logistic"]
    specs = _model_specs(base_requested, cfg.seed, cfg.n_jobs)
    windows_by_split = {split: _window_features(windows) for split, windows in sampled.items()}
    full_indices = tuple(range(len(feature_names)))
    selected = {
        split: select_window_features(windows, full_indices) for split, windows in windows_by_split.items()
    }
    selected_features = {
        split: np.stack([window.features for window in windows]).astype(np.float32, copy=False)
        for split, windows in selected.items()
    }
    pair_names = pair_feature_names(feature_names)
    x_train, y_train = _pair_matrix_from_feature_matrix(
        selected_features["train"], pair_indices["train"], chunk_size=8192
    )
    x_val, y_val = _pair_matrix_from_feature_matrix(
        selected_features["val"], pair_indices["val"], chunk_size=8192
    )
    x_test, y_test = _pair_matrix_from_feature_matrix(
        selected_features["test"], pair_indices["test"], chunk_size=8192
    )

    bundles: dict[str, Mapping[str, object]] = {}
    reports: dict[str, object] = {}
    fitted: dict[str, object] = {}
    for model_name, model in specs.items():
        started = time.time()
        model.fit(x_train, y_train)
        val_scores = _score_model(model, x_val)
        test_scores = _score_model(model, x_test)
        val_retrieval = _classical_retrieval_from_features(
            model, selected_features["val"], retrieval_queries["val"]
        )
        test_retrieval = _classical_retrieval_from_features(
            model, selected_features["test"], retrieval_queries["test"]
        )
        artifact_name = _model_artifact_name(model_name)
        model_path = model_dir / f"{artifact_name}.joblib"
        bundle = {
            "model": model,
            "model_name": model_name,
            "feature_names": list(feature_names),
            "pair_feature_names": pair_names,
            "selected_feature_indices": list(full_indices),
            "config": _jsonable_config(cfg),
        }
        joblib.dump(bundle, model_path)
        bundles[model_name] = {**bundle, "model_path": str(model_path)}
        reports[model_name] = {
            **_score_report_with_retrieval(
                y_val, val_scores, y_test, test_scores, val_retrieval, test_retrieval
            ),
            "model_path": str(model_path),
            "fit_seconds": time.time() - started,
        }
        fitted[model_name] = model

    if _ensemble_requested(requested):
        member_names = _ensemble_members(reports, fitted)
        if member_names:
            weights = _validation_weights(member_names, reports, y_val)
            ensemble = WeightedScoreEnsemble([(name, fitted[name]) for name in member_names], weights)
            val_scores = _score_model(ensemble, x_val)
            test_scores = _score_model(ensemble, x_test)
            val_retrieval = _classical_retrieval_from_features(
                ensemble, selected_features["val"], retrieval_queries["val"]
            )
            test_retrieval = _classical_retrieval_from_features(
                ensemble, selected_features["test"], retrieval_queries["test"]
            )
            model_path = model_dir / "ensemble.joblib"
            bundle = {
                "model": ensemble,
                "model_name": "ensemble",
                "feature_names": list(feature_names),
                "pair_feature_names": pair_names,
                "selected_feature_indices": list(full_indices),
                "member_names": member_names,
                "weights": weights.tolist(),
                "config": _jsonable_config(cfg),
            }
            joblib.dump(bundle, model_path)
            bundles["ensemble"] = {**bundle, "model_path": str(model_path)}
            reports["ensemble"] = {
                **_score_report_with_retrieval(
                    y_val, val_scores, y_test, test_scores, val_retrieval, test_retrieval
                ),
                "model_path": str(model_path),
                "member_names": member_names,
                "weights": weights.tolist(),
                "fit_seconds": 0.0,
            }
    return bundles, reports

def _fit_classical_models_all(
    bundle_dir: Path,
    cfg: FusedTrainingConfig,
    windows: Sequence[FusedWindow],
    feature_names: Sequence[str],
    pair_indices: Sequence[PairIndex],
) -> dict[str, Mapping[str, object]]:
    import joblib

    model_dir = bundle_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    requested = tuple(dict.fromkeys(name.strip() for name in cfg.models if name.strip()))
    base_requested = [name for name in requested if name != "ensemble"]
    if not base_requested:
        base_requested = ["sgd_logistic"]
    specs = _model_specs(base_requested, cfg.seed, cfg.n_jobs)
    full_indices = tuple(range(len(feature_names)))
    selected = select_window_features(_window_features(windows), full_indices)
    selected_features = np.stack([window.features for window in selected]).astype(np.float32, copy=False)
    pair_names = pair_feature_names(feature_names)
    x_all, y_all = _pair_matrix_from_feature_matrix(selected_features, pair_indices, chunk_size=8192)
    bundles: dict[str, Mapping[str, object]] = {}
    fitted: dict[str, object] = {}
    for model_name, model in specs.items():
        model.fit(x_all, y_all)
        model_path = model_dir / f"{_model_artifact_name(model_name)}.joblib"
        bundle = {
            "model": model,
            "model_name": model_name,
            "feature_names": list(feature_names),
            "pair_feature_names": pair_names,
            "selected_feature_indices": list(full_indices),
            "config": _jsonable_config(cfg),
        }
        joblib.dump(bundle, model_path)
        bundles[model_name] = {**bundle, "model_path": str(model_path)}
        fitted[model_name] = model
    if _ensemble_requested(requested):
        member_names = [name for name in ("sgd_logistic", "hist_gbdt", "extra_trees") if name in fitted]
        if member_names:
            weights = np.ones(len(member_names), dtype=np.float64)
            ensemble = WeightedScoreEnsemble([(name, fitted[name]) for name in member_names], weights)
            model_path = model_dir / "ensemble.joblib"
            bundle = {
                "model": ensemble,
                "model_name": "ensemble",
                "feature_names": list(feature_names),
                "pair_feature_names": pair_names,
                "selected_feature_indices": list(full_indices),
                "member_names": member_names,
                "weights": weights.tolist(),
                "config": _jsonable_config(cfg),
            }
            joblib.dump(bundle, model_path)
            bundles["ensemble"] = {**bundle, "model_path": str(model_path)}
    return bundles
