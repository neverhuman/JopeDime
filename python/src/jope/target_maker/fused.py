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

BUNDLE_FORMAT = "target_maker_fused_bundle"
BUNDLE_VERSION = 1
DEFAULT_BEST_CHECKPOINT = Path("/home/ubuntu/target_runs/full_rich_instance_20260705_070543/best.pt")
DEFAULT_FINAL_CHECKPOINT = Path("/home/ubuntu/target_runs/full_rich_instance_20260705_070543/final.pt")


@dataclass
class FusedTrainingConfig:
    cache_dir: Path
    run_dir: Path
    best_checkpoint: Path = DEFAULT_BEST_CHECKPOINT
    final_checkpoint: Path = DEFAULT_FINAL_CHECKPOINT
    window_size: int = 96
    views_per_record: int = 4
    signature_views: int = 8
    limit_per_task: int | None = None
    train_pairs: int = 250_000
    val_pairs: int = 50_000
    test_pairs: int = 50_000
    retrieval_queries: int = 5000
    retrieval_candidates: int = 512
    seed: int = 17
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    positive_fraction: float = 0.50
    same_task_negative_fraction: float = 0.80
    include_cache_profile: bool = False
    models: tuple[str, ...] = ("sgd_logistic", "hist_gbdt", "extra_trees", "ensemble")
    vector_dim: int = 512
    classical_component_dim: int = 384
    n_jobs: int = -1
    device: str | None = None
    batch_size: int = 512


@dataclass(frozen=True)
class FusedWindow:
    dataset_id: str
    task: int
    original_length: int
    effective_count: int
    replacement_fraction: float
    values: np.ndarray
    features: np.ndarray


@dataclass
class _TargetEncoding:
    vector: np.ndarray
    neural_best: np.ndarray
    neural_final: np.ndarray
    classical_component: np.ndarray
    window_features: np.ndarray
    task: int
    original_length: int
    effective_count: int
    replacement_fraction: float


@dataclass
class _BatchEncoding:
    vectors: np.ndarray
    neural_best: np.ndarray
    neural_final: np.ndarray
    classical_components: np.ndarray
    window_features: np.ndarray
    tasks: np.ndarray
    original_lengths: np.ndarray
    effective_counts: np.ndarray
    replacement_fractions: np.ndarray


def _jsonable_config(cfg: FusedTrainingConfig) -> dict[str, object]:
    data = asdict(cfg)
    for key in ("cache_dir", "run_dir", "best_checkpoint", "final_checkpoint"):
        data[key] = str(data[key])
    data["models"] = list(cfg.models)
    return data


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def default_fusion_run_dir() -> Path:
    return Path("/home/ubuntu/target_runs") / f"fused_monster_20260705_{_utc_stamp()[9:]}"


def _safe_unit(x: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(arr))
    if norm <= 1.0e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / norm).astype(np.float32, copy=False)


def _component_pad(x: np.ndarray, dim: int) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(x, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == dim:
        return arr.astype(np.float32, copy=False)
    if arr.size > dim:
        return arr[:dim].astype(np.float32, copy=False)
    out = np.zeros(dim, dtype=np.float32)
    out[: arr.size] = arr
    return out


def _weighted_fused_vector(
    neural_best: np.ndarray,
    neural_final: np.ndarray,
    classical_component: np.ndarray,
    weights: Sequence[float],
    *,
    vector_dim: int,
    classical_component_dim: int,
) -> np.ndarray:
    best_w, final_w, classical_w = [float(x) for x in weights]
    parts = [
        best_w * _safe_unit(neural_best),
        final_w * _safe_unit(neural_final),
        classical_w * _safe_unit(_component_pad(classical_component, classical_component_dim)),
    ]
    fused = np.concatenate(parts).astype(np.float32, copy=False)
    fused = _component_pad(fused, vector_dim)
    return _safe_unit(fused)


def _combined_score(pair_metrics: Mapping[str, float], retrieval: Mapping[str, float]) -> float:
    return float(
        0.35 * float(pair_metrics.get("roc_auc", 0.5))
        + 0.35 * float(pair_metrics.get("average_precision", 0.0))
        + 0.15 * float(retrieval.get("retrieval@1", 0.0))
        + 0.15 * float(retrieval.get("retrieval@10", 0.0))
    )


def _score_report_with_retrieval(
    y_val: np.ndarray,
    val_scores: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
    val_retrieval: Mapping[str, float],
    test_retrieval: Mapping[str, float],
) -> dict[str, object]:
    threshold = _best_f1_threshold(y_val, val_scores)
    val_metrics = _classification_metrics(y_val, val_scores, threshold)
    test_metrics = _classification_metrics(y_test, test_scores, threshold)
    return {
        "val": val_metrics,
        "test": test_metrics,
        "val_retrieval": dict(val_retrieval),
        "retrieval": dict(test_retrieval),
        "validation_score": _combined_score(val_metrics, val_retrieval),
        "test_score": _combined_score(test_metrics, test_retrieval),
    }


def _window_features(windows: Sequence[FusedWindow]) -> list[WindowFeature]:
    return [WindowFeature(window.dataset_id, window.task, window.features) for window in windows]


def _values_from_npz(record: DatasetRecord, npz: Mapping[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    if record.task == TASK_BINARY:
        return bits_at_indices(npz["bits"], indices)
    return np.asarray(npz["values"], dtype=np.float32)[indices]


def _sample_windows(records: Sequence[DatasetRecord], cfg: FusedTrainingConfig) -> tuple[list[FusedWindow], list[str]]:
    windows: list[FusedWindow] = []
    feature_names: list[str] | None = None
    for record in records:
        with np.load(record.cache_path, allow_pickle=False) as npz:
            cache_profile = np.asarray(npz["profile"], dtype=np.float32) if cfg.include_cache_profile else None
            for view_index in range(cfg.views_per_record):
                rng = np.random.default_rng(stable_seed("fusion-window", cfg.seed, record.dataset_id, view_index, cfg.window_size))
                indices, effective_count, replacement_fraction = sample_indices(record.length, cfg.window_size, rng)
                values = np.asarray(_values_from_npz(record, npz, indices), dtype=np.float32)
                features, names = summarize_target_window(
                    values,
                    task=record.task,
                    original_length=record.length,
                    effective_count=effective_count,
                    replacement_fraction=replacement_fraction,
                    window_size=cfg.window_size,
                    cache_profile=cache_profile,
                )
                if feature_names is None:
                    feature_names = names
                elif feature_names != names:
                    raise RuntimeError("Feature names changed while sampling fused windows")
                windows.append(
                    FusedWindow(
                        dataset_id=record.dataset_id,
                        task=record.task,
                        original_length=record.length,
                        effective_count=int(effective_count),
                        replacement_fraction=float(replacement_fraction),
                        values=values,
                        features=features,
                    )
                )
    if feature_names is None:
        raise ValueError("No fused windows sampled")
    return windows, feature_names


def _signature_from_feature_stack(stack: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(stack, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([x.mean(axis=0), x.std(axis=0), x.min(axis=0), x.max(axis=0)]).astype(np.float32, copy=False)


def _record_signatures(
    records: Sequence[DatasetRecord],
    cfg: FusedTrainingConfig,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    signatures: dict[str, np.ndarray] = {}
    mean_features: dict[str, np.ndarray] = {}
    feature_names: list[str] | None = None
    for record in records:
        with np.load(record.cache_path, allow_pickle=False) as npz:
            cache_profile = np.asarray(npz["profile"], dtype=np.float32) if cfg.include_cache_profile else None
            rows: list[np.ndarray] = []
            for view_index in range(cfg.signature_views):
                rng = np.random.default_rng(stable_seed("fusion-signature", cfg.seed, record.dataset_id, view_index, cfg.window_size))
                indices, effective_count, replacement_fraction = sample_indices(record.length, cfg.window_size, rng)
                values = np.asarray(_values_from_npz(record, npz, indices), dtype=np.float32)
                features, names = summarize_target_window(
                    values,
                    task=record.task,
                    original_length=record.length,
                    effective_count=effective_count,
                    replacement_fraction=replacement_fraction,
                    window_size=cfg.window_size,
                    cache_profile=cache_profile,
                )
                if feature_names is None:
                    feature_names = names
                elif feature_names != names:
                    raise RuntimeError("Feature names changed while sampling fused signatures")
                rows.append(features)
            stack = np.stack(rows).astype(np.float32, copy=False)
            signatures[record.dataset_id] = _signature_from_feature_stack(stack)
            mean_features[record.dataset_id] = stack.mean(axis=0).astype(np.float32, copy=False)
    if feature_names is None:
        raise ValueError("No fused signatures sampled")
    return signatures, mean_features, feature_names


def _raw_signature_from_values(
    values: Iterable[float] | np.ndarray,
    task: str | int,
    *,
    window_size: int,
    signature_views: int,
    seed: int,
    include_cache_profile: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], int, int, float]:
    task_id = normalize_task(task)
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("Cannot encode an empty target")
    if task_id == TASK_BINARY:
        arr = (arr >= 0.5).astype(np.float32)
    else:
        arr = np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)
    original_length = int(arr.size)
    cache_profile = compute_profile(arr, task_id) if include_cache_profile else None
    rows: list[np.ndarray] = []
    names: list[str] | None = None
    effective_values: list[int] = []
    replacement_values: list[float] = []
    for view_index in range(max(1, int(signature_views))):
        rng = np.random.default_rng(stable_seed("fusion-signature-inference", seed, task_id, original_length, view_index, window_size))
        indices, effective_count, replacement_fraction = sample_indices(original_length, window_size, rng)
        features, feature_names = summarize_target_window(
            arr[indices],
            task=task_id,
            original_length=original_length,
            effective_count=effective_count,
            replacement_fraction=replacement_fraction,
            window_size=window_size,
            cache_profile=cache_profile,
        )
        if names is None:
            names = feature_names
        elif names != feature_names:
            raise RuntimeError("Feature names changed while encoding fused signature")
        rows.append(features)
        effective_values.append(int(effective_count))
        replacement_values.append(float(replacement_fraction))
    assert names is not None
    stack = np.stack(rows).astype(np.float32, copy=False)
    return (
        _signature_from_feature_stack(stack),
        stack.mean(axis=0).astype(np.float32, copy=False),
        names,
        original_length,
        int(round(float(np.mean(effective_values)))),
        float(np.mean(replacement_values)),
    )


def _fit_signature_vectorizer(signatures: Mapping[str, np.ndarray], component_dim: int, seed: int) -> dict[str, object]:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler

    keys = sorted(signatures)
    x = np.stack([signatures[key] for key in keys]).astype(np.float32, copy=False)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(x)
    requested = max(0, int(component_dim))
    reducer = None
    actual_dim = 0
    if requested > 0 and scaled.shape[1] > 0:
        n_components = min(requested, scaled.shape[1])
        reducer = TruncatedSVD(n_components=n_components, random_state=seed)
        reducer.fit(scaled)
        actual_dim = int(getattr(reducer, "components_", np.zeros((0, 0))).shape[0])
    return {
        "scaler": scaler,
        "reducer": reducer,
        "component_dim": requested,
        "actual_dim": actual_dim,
        "signature_dim": int(x.shape[1]),
        "fit_records": keys,
    }


def _transform_signatures(vectorizer: Mapping[str, object], signatures: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    keys = sorted(signatures)
    if not keys:
        return {}
    scaler = vectorizer.get("scaler")
    reducer = vectorizer.get("reducer")
    component_dim = int(vectorizer.get("component_dim", 0))
    x = np.stack([signatures[key] for key in keys]).astype(np.float32, copy=False)
    scaled = scaler.transform(x) if scaler is not None else x
    if reducer is None or component_dim <= 0:
        reduced = np.zeros((x.shape[0], 0), dtype=np.float32)
    else:
        reduced = np.asarray(reducer.transform(scaled), dtype=np.float32)
    return {key: _component_pad(reduced[row], component_dim) for row, key in enumerate(keys)}


@torch.no_grad()
def _encode_neural_windows(
    checkpoint_path: Path,
    windows: Sequence[FusedWindow],
    *,
    device: str | None,
    batch_size: int,
) -> np.ndarray:
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(Path(checkpoint_path), map_location=torch_device)
    model = build_model_from_checkpoint(checkpoint).to(torch_device)
    model.eval()
    embeddings: list[torch.Tensor] = []
    batch_size = max(1, int(batch_size))
    for start in range(0, len(windows), batch_size):
        chunk = windows[start : start + batch_size]
        values = torch.from_numpy(np.stack([window.values for window in chunk]).astype(np.float32, copy=False)).to(torch_device)
        task = torch.tensor([window.task for window in chunk], device=torch_device, dtype=torch.long)
        original_length = torch.tensor([window.original_length for window in chunk], device=torch_device, dtype=torch.float32)
        effective_count = torch.tensor([window.effective_count for window in chunk], device=torch_device, dtype=torch.float32)
        replacement_fraction = torch.tensor([window.replacement_fraction for window in chunk], device=torch_device, dtype=torch.float32)
        mask = torch.ones_like(values)
        out = model(
            values,
            task,
            original_length,
            effective_count,
            mask=mask,
            replacement_fraction=replacement_fraction,
            return_aux=False,
        )
        embeddings.append(F.normalize(out.float(), dim=1).cpu())
    return torch.cat(embeddings, dim=0).numpy().astype(np.float32, copy=False)


def _batch_from_components(
    windows: Sequence[FusedWindow],
    best_embeddings: np.ndarray,
    final_embeddings: np.ndarray,
    component_by_record: Mapping[str, np.ndarray],
    weights: Sequence[float],
    *,
    vector_dim: int,
    classical_component_dim: int,
) -> _BatchEncoding:
    vectors = np.empty((len(windows), vector_dim), dtype=np.float32)
    classical_components = np.empty((len(windows), classical_component_dim), dtype=np.float32)
    window_features = np.stack([window.features for window in windows]).astype(np.float32, copy=False)
    for index, window in enumerate(windows):
        classical_component = component_by_record[window.dataset_id]
        classical_components[index] = _component_pad(classical_component, classical_component_dim)
        vectors[index] = _weighted_fused_vector(
            best_embeddings[index],
            final_embeddings[index],
            classical_components[index],
            weights,
            vector_dim=vector_dim,
            classical_component_dim=classical_component_dim,
        )
    return _BatchEncoding(
        vectors=vectors,
        neural_best=best_embeddings.astype(np.float32, copy=False),
        neural_final=final_embeddings.astype(np.float32, copy=False),
        classical_components=classical_components,
        window_features=window_features,
        tasks=np.asarray([window.task for window in windows], dtype=np.int64),
        original_lengths=np.asarray([window.original_length for window in windows], dtype=np.float64),
        effective_counts=np.asarray([window.effective_count for window in windows], dtype=np.float64),
        replacement_fractions=np.asarray([window.replacement_fraction for window in windows], dtype=np.float64),
    )


def _cosine_scores(vectors: np.ndarray, pairs: Sequence[PairIndex]) -> np.ndarray:
    return np.asarray([float(np.dot(vectors[pair.left], vectors[pair.right])) for pair in pairs], dtype=np.float64)


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


def _pair_matrix_from_feature_matrix(features: np.ndarray, pairs: Sequence[PairIndex], *, chunk_size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
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
    indices = tuple(int(index) for index in model_bundle.get("selected_feature_indices", range(batch.window_features.shape[1])))
    features = batch.window_features[:, np.asarray(indices, dtype=np.int64)]
    return _score_pairs_from_feature_matrix(model_bundle["model"], features, pairs, chunk_size=8192)


def _model_retrieval_for_batch(
    model_bundle: Mapping[str, object],
    batch: _BatchEncoding,
    queries: Sequence[RetrievalQuery],
) -> dict[str, float]:
    if not queries:
        return {}
    indices = tuple(int(index) for index in model_bundle.get("selected_feature_indices", range(batch.window_features.shape[1])))
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
    selected = {split: select_window_features(windows, full_indices) for split, windows in windows_by_split.items()}
    selected_features = {
        split: np.stack([window.features for window in windows]).astype(np.float32, copy=False)
        for split, windows in selected.items()
    }
    pair_names = pair_feature_names(feature_names)
    x_train, y_train = _pair_matrix_from_feature_matrix(selected_features["train"], pair_indices["train"], chunk_size=8192)
    x_val, y_val = _pair_matrix_from_feature_matrix(selected_features["val"], pair_indices["val"], chunk_size=8192)
    x_test, y_test = _pair_matrix_from_feature_matrix(selected_features["test"], pair_indices["test"], chunk_size=8192)

    bundles: dict[str, Mapping[str, object]] = {}
    reports: dict[str, object] = {}
    fitted: dict[str, object] = {}
    for model_name, model in specs.items():
        started = time.time()
        model.fit(x_train, y_train)
        val_scores = _score_model(model, x_val)
        test_scores = _score_model(model, x_test)
        val_retrieval = _classical_retrieval_from_features(model, selected_features["val"], retrieval_queries["val"])
        test_retrieval = _classical_retrieval_from_features(model, selected_features["test"], retrieval_queries["test"])
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
            **_score_report_with_retrieval(y_val, val_scores, y_test, test_scores, val_retrieval, test_retrieval),
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
            val_retrieval = _classical_retrieval_from_features(ensemble, selected_features["val"], retrieval_queries["val"])
            test_retrieval = _classical_retrieval_from_features(ensemble, selected_features["test"], retrieval_queries["test"])
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
                **_score_report_with_retrieval(y_val, val_scores, y_test, test_scores, val_retrieval, test_retrieval),
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
    retrieval_flat = {
        split: _flatten_retrieval_pairs(retrieval_queries[split])
        for split in ("val", "test")
    }
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
        x_train = _meta_features(train_batch, pair_indices["train"], classical_pair_scores["train"], classical_names, train_fused_scores)
        x_val = _meta_features(val_batch, pair_indices["val"], classical_pair_scores["val"], classical_names, val_fused_scores)
        x_test = _meta_features(test_batch, pair_indices["test"], classical_pair_scores["test"], classical_names, test_fused_scores)

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
                **_score_report_with_retrieval(y_val, val_scores, y_test, test_scores, val_retrieval, test_retrieval),
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
    return _meta_retrieval_from_flat(model, batch, queries, all_pairs, sizes, classical_scores, classical_names)


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


def _write_bundle_manifest(
    bundle_dir: Path,
    cfg: FusedTrainingConfig,
    *,
    feature_names: Sequence[str],
    vectorizer_artifact: str,
    vectorizer: Mapping[str, object],
    checkpoint_best: str,
    checkpoint_final: str,
    classical_bundles: Mapping[str, Mapping[str, object]],
    vector_weights: Sequence[float],
    scorer: Mapping[str, object],
    report: Mapping[str, object],
) -> None:
    manifest = {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": _jsonable_config(cfg),
        "window_size": int(cfg.window_size),
        "signature_views": int(cfg.signature_views),
        "seed": int(cfg.seed),
        "include_cache_profile": bool(cfg.include_cache_profile),
        "vector_dim": int(cfg.vector_dim),
        "classical_component_dim": int(cfg.classical_component_dim),
        "feature_names": list(feature_names),
        "signature_feature_names": [f"{agg}:{name}" for agg in ("mean", "std", "min", "max") for name in feature_names],
        "vectorizer": {
            "artifact": vectorizer_artifact,
            "signature_dim": int(vectorizer.get("signature_dim", 0)),
            "component_dim": int(vectorizer.get("component_dim", 0)),
            "actual_dim": int(vectorizer.get("actual_dim", 0)),
        },
        "neural": {
            "best": checkpoint_best,
            "final": checkpoint_final,
        },
        "classical_models": {
            name: os.path.relpath(str(bundle["model_path"]), str(bundle_dir)) for name, bundle in classical_bundles.items()
        },
        "vector_weights": [float(x) for x in vector_weights],
        "scorer": dict(scorer),
        "report": dict(report),
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_checkpoints(bundle_dir: Path, best_checkpoint: Path, final_checkpoint: Path) -> tuple[str, str]:
    neural_dir = bundle_dir / "neural"
    neural_dir.mkdir(parents=True, exist_ok=True)
    best_dest = neural_dir / "best.pt"
    final_dest = neural_dir / "final.pt"
    shutil.copy2(best_checkpoint, best_dest)
    shutil.copy2(final_checkpoint, final_dest)
    return os.path.relpath(best_dest, bundle_dir), os.path.relpath(final_dest, bundle_dir)


def _relative_artifact(bundle_dir: Path, path: Path) -> str:
    return os.path.relpath(str(path), str(bundle_dir))


def _train_best_bundle(
    cfg: FusedTrainingConfig,
    records_by_split: Mapping[str, list[DatasetRecord]],
    bundle_dir: Path,
) -> dict[str, object]:
    import joblib

    bundle_dir.mkdir(parents=True, exist_ok=True)
    sampled: dict[str, list[FusedWindow]] = {}
    feature_names: list[str] | None = None
    for split, records in records_by_split.items():
        windows, names = _sample_windows(records, cfg)
        sampled[split] = windows
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise RuntimeError("Feature names changed across fused splits")
    assert feature_names is not None

    pair_indices = {
        "train": build_pair_indices(
            _window_features(sampled["train"]),
            cfg.train_pairs,
            seed=stable_seed("fusion-pairs", cfg.seed, "train"),
            positive_fraction=cfg.positive_fraction,
            same_task_negative_fraction=cfg.same_task_negative_fraction,
        ),
        "val": build_pair_indices(
            _window_features(sampled["val"]),
            cfg.val_pairs,
            seed=stable_seed("fusion-pairs", cfg.seed, "val"),
            positive_fraction=cfg.positive_fraction,
            same_task_negative_fraction=cfg.same_task_negative_fraction,
        ),
        "test": build_pair_indices(
            _window_features(sampled["test"]),
            cfg.test_pairs,
            seed=stable_seed("fusion-pairs", cfg.seed, "test"),
            positive_fraction=cfg.positive_fraction,
            same_task_negative_fraction=cfg.same_task_negative_fraction,
        ),
    }
    retrieval_queries = {
        "val": build_retrieval_queries(
            _window_features(sampled["val"]),
            queries=cfg.retrieval_queries,
            candidates=cfg.retrieval_candidates,
            seed=stable_seed("fusion-retrieval", cfg.seed, "val"),
        ),
        "test": build_retrieval_queries(
            _window_features(sampled["test"]),
            queries=cfg.retrieval_queries,
            candidates=cfg.retrieval_candidates,
            seed=stable_seed("fusion-retrieval", cfg.seed, "test"),
        ),
    }

    signature_records = [record for split in ("train", "val", "test") for record in records_by_split[split]]
    signatures, _, signature_feature_names = _record_signatures(signature_records, cfg)
    if signature_feature_names != feature_names:
        raise RuntimeError("Signature feature names do not match sampled window feature names")
    vectorizer = _fit_signature_vectorizer({record.dataset_id: signatures[record.dataset_id] for record in records_by_split["train"]}, cfg.classical_component_dim, cfg.seed)
    component_by_record = _transform_signatures(vectorizer, signatures)
    vectorizer_path = bundle_dir / "fused_vectorizer.joblib"
    joblib.dump({**vectorizer, "feature_names": feature_names}, vectorizer_path)

    classical_bundles, classical_reports = _fit_classical_models(bundle_dir, cfg, sampled, feature_names, pair_indices, retrieval_queries)
    best_embeddings = {
        split: _encode_neural_windows(cfg.best_checkpoint, windows, device=cfg.device, batch_size=cfg.batch_size)
        for split, windows in sampled.items()
    }
    final_embeddings = {
        split: _encode_neural_windows(cfg.final_checkpoint, windows, device=cfg.device, batch_size=cfg.batch_size)
        for split, windows in sampled.items()
    }

    batches_by_weight: dict[str, dict[str, _BatchEncoding]] = {}
    vector_reports: dict[str, object] = {}
    y_val = np.asarray([pair.label for pair in pair_indices["val"]], dtype=np.int64)
    y_test = np.asarray([pair.label for pair in pair_indices["test"]], dtype=np.int64)
    for weight_key, weights in _weight_grid():
        batches_by_weight[weight_key] = {
            split: _batch_from_components(
                sampled[split],
                best_embeddings[split],
                final_embeddings[split],
                component_by_record,
                weights,
                vector_dim=cfg.vector_dim,
                classical_component_dim=cfg.classical_component_dim,
            )
            for split in ("train", "val", "test")
        }
        val_scores = _cosine_scores(batches_by_weight[weight_key]["val"].vectors, pair_indices["val"])
        test_scores = _cosine_scores(batches_by_weight[weight_key]["test"].vectors, pair_indices["test"])
        vector_reports[weight_key] = _score_report_with_retrieval(
            y_val,
            val_scores,
            y_test,
            test_scores,
            _retrieval_from_vectors(batches_by_weight[weight_key]["val"].vectors, retrieval_queries["val"]),
            _retrieval_from_vectors(batches_by_weight[weight_key]["test"].vectors, retrieval_queries["test"]),
        )

    meta_reports, meta_artifacts = _fit_meta_candidates(
        bundle_dir,
        cfg,
        batches_by_weight,
        pair_indices,
        retrieval_queries,
        classical_bundles,
    )

    best_meta_key = max(meta_reports, key=lambda key: float(meta_reports[key].get("validation_score", -1.0)))
    best_vector_key = max(vector_reports, key=lambda key: float(vector_reports[key].get("validation_score", -1.0)))
    best_classical_name = max(classical_reports, key=lambda key: float(classical_reports[key].get("validation_score", -1.0)))
    hist_score = float(classical_reports.get("hist_gbdt", {}).get("validation_score", -1.0)) if isinstance(classical_reports, Mapping) else -1.0
    best_classical_score = float(classical_reports[best_classical_name].get("validation_score", -1.0))
    best_meta_score = float(meta_reports[best_meta_key].get("validation_score", -1.0))
    fusion_wins = best_meta_score > max(hist_score, best_classical_score)
    selected_key = best_meta_key if fusion_wins else best_classical_name
    selected_weight_key = str(meta_reports[best_meta_key].get("weight_key", best_vector_key))
    selected_weights = dict(_weight_grid())[selected_weight_key]

    scorer_artifact = None
    if fusion_wins:
        scorer_artifact = _relative_artifact(bundle_dir, Path(str(meta_artifacts[best_meta_key]["artifact_path"])))
        scorer = {
            "default": "fused_meta",
            "artifact": scorer_artifact,
            "meta_key": best_meta_key,
            "fallback_classical_model": best_classical_name,
            "fusion_wins": True,
        }
    else:
        scorer = {
            "default": "classical",
            "classical_model": best_classical_name,
            "candidate_meta_key": best_meta_key,
            "fusion_wins": False,
            "reason": "fused meta validation score did not beat the best validated classical scorer",
        }

    checkpoint_best, checkpoint_final = _copy_checkpoints(bundle_dir, cfg.best_checkpoint, cfg.final_checkpoint)
    report = {
        "split_records": {split: len(records) for split, records in records_by_split.items()},
        "split_windows": {split: len(windows) for split, windows in sampled.items()},
        "pair_counts": {split: len(pairs) for split, pairs in pair_indices.items()},
        "retrieval_queries": {split: len(queries) for split, queries in retrieval_queries.items()},
        "classical": classical_reports,
        "fused_vector": vector_reports,
        "fused_meta": meta_reports,
        "selection": {
            "best_meta_key": best_meta_key,
            "best_meta_validation_score": best_meta_score,
            "best_vector_key": best_vector_key,
            "best_classical_name": best_classical_name,
            "best_classical_validation_score": best_classical_score,
            "hist_gbdt_validation_score": hist_score,
            "selected_default": selected_key,
            "fusion_wins": fusion_wins,
            "vector_weight_key": selected_weight_key,
            "vector_weights": list(selected_weights),
        },
    }
    _write_bundle_manifest(
        bundle_dir,
        cfg,
        feature_names=feature_names,
        vectorizer_artifact=_relative_artifact(bundle_dir, vectorizer_path),
        vectorizer=vectorizer,
        checkpoint_best=checkpoint_best,
        checkpoint_final=checkpoint_final,
        classical_bundles=classical_bundles,
        vector_weights=selected_weights,
        scorer=scorer,
        report=report,
    )
    return report


def _train_final_bundle(
    cfg: FusedTrainingConfig,
    records: Sequence[DatasetRecord],
    bundle_dir: Path,
    selection: Mapping[str, object],
) -> dict[str, object]:
    import joblib

    bundle_dir.mkdir(parents=True, exist_ok=True)
    windows, feature_names = _sample_windows(records, cfg)
    signatures, _, signature_feature_names = _record_signatures(records, cfg)
    if signature_feature_names != feature_names:
        raise RuntimeError("Signature feature names do not match sampled window feature names")
    vectorizer = _fit_signature_vectorizer(signatures, cfg.classical_component_dim, cfg.seed)
    component_by_record = _transform_signatures(vectorizer, signatures)
    vectorizer_path = bundle_dir / "fused_vectorizer.joblib"
    joblib.dump({**vectorizer, "feature_names": feature_names}, vectorizer_path)
    total_pairs = int(cfg.train_pairs + cfg.val_pairs + cfg.test_pairs)
    pairs = build_pair_indices(
        _window_features(windows),
        total_pairs,
        seed=stable_seed("fusion-pairs", cfg.seed, "all"),
        positive_fraction=cfg.positive_fraction,
        same_task_negative_fraction=cfg.same_task_negative_fraction,
    )
    classical_bundles = _fit_classical_models_all(bundle_dir, cfg, windows, feature_names, pairs)
    best_embeddings = _encode_neural_windows(cfg.best_checkpoint, windows, device=cfg.device, batch_size=cfg.batch_size)
    final_embeddings = _encode_neural_windows(cfg.final_checkpoint, windows, device=cfg.device, batch_size=cfg.batch_size)
    weight_key = str(selection.get("vector_weight_key", "balanced"))
    weights = dict(_weight_grid()).get(weight_key, (1.0, 1.0, 1.0))
    batch = _batch_from_components(
        windows,
        best_embeddings,
        final_embeddings,
        component_by_record,
        weights,
        vector_dim=cfg.vector_dim,
        classical_component_dim=cfg.classical_component_dim,
    )
    y_all = np.asarray([pair.label for pair in pairs], dtype=np.int64)
    classical_names = sorted(classical_bundles)
    fused_scores = _cosine_scores(batch.vectors, pairs)
    classical_scores = {
        name: _model_pair_scores_for_batch(bundle, batch, pairs)
        for name, bundle in classical_bundles.items()
    }
    x_all = _meta_features(batch, pairs, classical_scores, classical_names, fused_scores)
    scorer_dir = bundle_dir / "scorers"
    scorer_dir.mkdir(parents=True, exist_ok=True)
    selected_meta_key = str(selection.get("best_meta_key", "logistic__balanced"))
    selected_kind = selected_meta_key.split("__", 1)[0] if "__" in selected_meta_key else "logistic"
    if selected_kind not in {"logistic", "hist_gbdt"}:
        selected_kind = "logistic"
    meta_model = _fit_meta_model(selected_kind, cfg.seed)
    meta_model.fit(x_all, y_all)
    meta_artifact_path = scorer_dir / "meta_scorer.joblib"
    meta_artifact = {
        "model": meta_model,
        "kind": selected_kind,
        "weight_key": weight_key,
        "feature_names": _meta_feature_names(classical_names),
        "classical_model_names": classical_names,
        "config": _jsonable_config(cfg),
    }
    joblib.dump(meta_artifact, meta_artifact_path)
    best_classical_name = str(selection.get("best_classical_name", "ensemble" if "ensemble" in classical_bundles else classical_names[0]))
    fusion_wins = bool(selection.get("fusion_wins", False))
    if fusion_wins:
        scorer = {
            "default": "fused_meta",
            "artifact": _relative_artifact(bundle_dir, meta_artifact_path),
            "meta_key": selected_meta_key,
            "fallback_classical_model": best_classical_name,
            "fusion_wins": True,
        }
    else:
        scorer = {
            "default": "classical",
            "classical_model": best_classical_name,
            "candidate_meta_artifact": _relative_artifact(bundle_dir, meta_artifact_path),
            "candidate_meta_key": selected_meta_key,
            "fusion_wins": False,
            "reason": "fused meta validation score did not beat the best validated classical scorer",
        }
    checkpoint_best, checkpoint_final = _copy_checkpoints(bundle_dir, cfg.best_checkpoint, cfg.final_checkpoint)
    report = {
        "record_count": len(records),
        "window_count": len(windows),
        "pair_count": len(pairs),
        "selection_from_best_bundle": dict(selection),
    }
    _write_bundle_manifest(
        bundle_dir,
        cfg,
        feature_names=feature_names,
        vectorizer_artifact=_relative_artifact(bundle_dir, vectorizer_path),
        vectorizer=vectorizer,
        checkpoint_best=checkpoint_best,
        checkpoint_final=checkpoint_final,
        classical_bundles=classical_bundles,
        vector_weights=weights,
        scorer=scorer,
        report=report,
    )
    return report


def train_fusion(cfg: FusedTrainingConfig) -> dict[str, object]:
    started = time.time()
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    if not Path(cfg.best_checkpoint).is_file():
        raise FileNotFoundError(f"Best neural checkpoint is missing: {cfg.best_checkpoint}")
    if not Path(cfg.final_checkpoint).is_file():
        raise FileNotFoundError(f"Final neural checkpoint is missing: {cfg.final_checkpoint}")
    records = load_manifest(cfg.cache_dir, split="all", seed=cfg.seed)
    records = _limit_records(records, cfg.limit_per_task, cfg.seed)
    if len(records) < 3:
        raise ValueError("Need at least three records for fused train/val/test splits")
    splits = _split_records(records, train_fraction=cfg.train_fraction, val_fraction=cfg.val_fraction, seed=cfg.seed)
    if not splits["train"] or not splits["val"] or not splits["test"]:
        raise ValueError(f"Non-empty train/val/test splits are required; got {[len(splits[key]) for key in ('train', 'val', 'test')]}")

    best_bundle = cfg.run_dir / "best.bundle"
    final_bundle = cfg.run_dir / "final.bundle"
    best_report = _train_best_bundle(cfg, splits, best_bundle)
    selection = best_report["selection"] if isinstance(best_report.get("selection"), Mapping) else {}
    final_report = _train_final_bundle(cfg, records, final_bundle, selection)
    link = cfg.run_dir / "inference.bundle"
    if link.is_symlink() or link.exists():
        if link.is_dir() and not link.is_symlink():
            raise FileExistsError(f"Cannot replace existing directory with inference bundle symlink: {link}")
        link.unlink()
    link.symlink_to(final_bundle.name)
    report = {
        "config": _jsonable_config(cfg),
        "record_count": len(records),
        "split_records": {split: len(value) for split, value in splits.items()},
        "best_bundle": str(best_bundle),
        "final_bundle": str(final_bundle),
        "inference_bundle": str(link),
        "best_report": best_report,
        "final_report": final_report,
        "elapsed_seconds": time.time() - started,
    }
    report_path = cfg.run_dir / "fused_training_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


class FusedTargetEncoder:
    def __init__(self, bundle_dir: Path, device: torch.device | str | None = None) -> None:
        self.bundle_dir = Path(bundle_dir).expanduser().resolve()
        manifest_path = self.bundle_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Fused bundle manifest is missing: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != BUNDLE_FORMAT:
            raise ValueError(f"Incompatible fused bundle format: {self.manifest.get('format')!r}")
        if int(self.manifest.get("version", -1)) != BUNDLE_VERSION:
            raise ValueError(f"Unsupported fused bundle version: {self.manifest.get('version')!r}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.window_size = int(self.manifest["window_size"])
        self.signature_views = int(self.manifest["signature_views"])
        self.seed = int(self.manifest["seed"])
        self.include_cache_profile = bool(self.manifest.get("include_cache_profile", False))
        self.vector_dim = int(self.manifest["vector_dim"])
        self.classical_component_dim = int(self.manifest["classical_component_dim"])
        self.feature_names = list(self.manifest["feature_names"])
        self.vector_weights = tuple(float(x) for x in self.manifest.get("vector_weights", [1.0, 1.0, 1.0]))

        vectorizer_path = self._artifact_path(self.manifest["vectorizer"]["artifact"])
        if not vectorizer_path.is_file():
            raise FileNotFoundError(f"Fused vectorizer artifact is missing: {vectorizer_path}")
        import joblib

        self.vectorizer = joblib.load(vectorizer_path)
        if list(self.vectorizer.get("feature_names", [])) != self.feature_names:
            raise ValueError("Fused vectorizer feature names do not match manifest feature names")

        neural = self.manifest.get("neural", {})
        best_path = self._artifact_path(neural.get("best"))
        final_path = self._artifact_path(neural.get("final"))
        if not best_path.is_file():
            raise FileNotFoundError(f"Fused neural best checkpoint is missing: {best_path}")
        if not final_path.is_file():
            raise FileNotFoundError(f"Fused neural final checkpoint is missing: {final_path}")
        self.best_model = load_encoder(best_path, device=self.device)
        self.final_model = load_encoder(final_path, device=self.device)

        self.classical_bundles: dict[str, Mapping[str, object]] = {}
        for name, rel_path in sorted(self.manifest.get("classical_models", {}).items()):
            path = self._artifact_path(rel_path)
            if not path.is_file():
                raise FileNotFoundError(f"Fused classical model artifact is missing for {name}: {path}")
            bundle = joblib.load(path)
            if not isinstance(bundle, Mapping) or bundle.get("model") is None:
                raise ValueError(f"Invalid fused classical model artifact for {name}: {path}")
            self.classical_bundles[str(name)] = bundle
        if not self.classical_bundles:
            raise ValueError("Fused bundle does not contain any classical model artifacts")

        self.scorer = dict(self.manifest.get("scorer", {}))
        self.meta_artifact: Mapping[str, object] | None = None
        meta_rel = self.scorer.get("artifact") or self.scorer.get("candidate_meta_artifact")
        if meta_rel:
            meta_path = self._artifact_path(str(meta_rel))
            if not meta_path.is_file():
                raise FileNotFoundError(f"Fused meta scorer artifact is missing: {meta_path}")
            self.meta_artifact = joblib.load(meta_path)

    def _artifact_path(self, rel_path: object) -> Path:
        if rel_path is None:
            raise ValueError("Manifest artifact path is missing")
        path = Path(str(rel_path))
        return path if path.is_absolute() else self.bundle_dir / path

    def encode_values(
        self,
        values: Iterable[float] | np.ndarray,
        task: str | int,
        *,
        seed: int | None = None,
    ) -> np.ndarray:
        return self._encode_target(values, task, seed=self.seed if seed is None else int(seed)).vector

    def score_values(
        self,
        left_values: Iterable[float] | np.ndarray,
        left_task: str | int,
        right_values: Iterable[float] | np.ndarray,
        right_task: str | int,
        *,
        seed: int | None = None,
    ) -> float:
        seed_value = self.seed if seed is None else int(seed)
        left = self._encode_target(left_values, left_task, seed=seed_value)
        right = self._encode_target(right_values, right_task, seed=seed_value)
        return self._score_encoded(left, right)

    def _encode_target(self, values: Iterable[float] | np.ndarray, task: str | int, *, seed: int) -> _TargetEncoding:
        raw_signature, mean_features, names, original_length, effective_count, replacement_fraction = _raw_signature_from_values(
            values,
            task,
            window_size=self.window_size,
            signature_views=self.signature_views,
            seed=seed,
            include_cache_profile=self.include_cache_profile,
        )
        if names != self.feature_names:
            missing = set(self.feature_names).symmetric_difference(names)
            sample = sorted(missing)[:5]
            raise ValueError(f"Fused bundle feature names are incompatible with input signature; sample mismatch: {sample}")
        component = _transform_single_signature(self.vectorizer, raw_signature, self.classical_component_dim)
        best = encode_values(
            self.best_model,
            values,
            task,
            window_size=self.window_size,
            stride=self.window_size,
            max_windows=self.signature_views,
            seed=seed,
            device=self.device,
        )
        final = encode_values(
            self.final_model,
            values,
            task,
            window_size=self.window_size,
            stride=self.window_size,
            max_windows=self.signature_views,
            seed=seed,
            device=self.device,
        )
        vector = _weighted_fused_vector(
            best,
            final,
            component,
            self.vector_weights,
            vector_dim=self.vector_dim,
            classical_component_dim=self.classical_component_dim,
        )
        return _TargetEncoding(
            vector=vector,
            neural_best=best,
            neural_final=final,
            classical_component=component,
            window_features=mean_features,
            task=normalize_task(task),
            original_length=original_length,
            effective_count=effective_count,
            replacement_fraction=replacement_fraction,
        )

    def _classical_scores_for_two(self, left: _TargetEncoding, right: _TargetEncoding) -> dict[str, float]:
        scores: dict[str, float] = {}
        for name, bundle in self.classical_bundles.items():
            indices = tuple(int(index) for index in bundle.get("selected_feature_indices", range(left.window_features.shape[0])))
            idx = np.asarray(indices, dtype=np.int64)
            x = pair_features(left.window_features[idx], right.window_features[idx])[None, :]
            scores[name] = float(_score_model(bundle["model"], x)[0])
        return scores

    def _score_encoded(self, left: _TargetEncoding, right: _TargetEncoding) -> float:
        classical_scores = self._classical_scores_for_two(left, right)
        default = self.scorer.get("default")
        if default == "classical":
            model_name = str(self.scorer.get("classical_model", "ensemble"))
            if model_name not in classical_scores:
                model_name = sorted(classical_scores)[0]
            return float(np.clip(classical_scores[model_name], 0.0, 1.0))
        if self.meta_artifact is None:
            fallback = str(self.scorer.get("fallback_classical_model", "ensemble"))
            if fallback not in classical_scores:
                fallback = sorted(classical_scores)[0]
            return float(np.clip(classical_scores[fallback], 0.0, 1.0))
        classical_names = [str(name) for name in self.meta_artifact.get("classical_model_names", sorted(classical_scores))]
        fused_score = float(np.dot(left.vector, right.vector))
        x = _meta_features_for_two(left, right, classical_scores, classical_names, fused_score)
        return float(np.clip(_score_model(self.meta_artifact["model"], x)[0], 0.0, 1.0))

    def encode_window_batch(self, windows: Sequence[object], *, batch_size: int = 512) -> _BatchEncoding:
        fused_windows = [
            FusedWindow(
                dataset_id=str(window.dataset_id),
                task=int(window.task),
                original_length=int(window.original_length),
                effective_count=int(window.effective_count),
                replacement_fraction=float(window.replacement_fraction),
                values=np.asarray(window.values, dtype=np.float32),
                features=np.asarray(window.features, dtype=np.float32),
            )
            for window in windows
        ]
        names = self.feature_names
        signatures: dict[str, list[np.ndarray]] = {}
        for window in fused_windows:
            if window.features.shape[0] != len(names):
                raise ValueError("Fused eval window feature dimension does not match bundle")
            signatures.setdefault(window.dataset_id, []).append(window.features)
        raw_signatures = {key: _signature_from_feature_stack(np.stack(rows)) for key, rows in signatures.items()}
        components = _transform_signatures(self.vectorizer, raw_signatures)
        best_path = self._artifact_path(self.manifest["neural"]["best"])
        final_path = self._artifact_path(self.manifest["neural"]["final"])
        best = _encode_neural_windows(best_path, fused_windows, device=str(self.device), batch_size=batch_size)
        final = _encode_neural_windows(final_path, fused_windows, device=str(self.device), batch_size=batch_size)
        return _batch_from_components(
            fused_windows,
            best,
            final,
            components,
            self.vector_weights,
            vector_dim=self.vector_dim,
            classical_component_dim=self.classical_component_dim,
        )

    def score_pairs_from_batch(self, batch: _BatchEncoding, pairs: Sequence[PairIndex]) -> np.ndarray:
        if self.scorer.get("default") == "classical":
            model_name = str(self.scorer.get("classical_model", "ensemble"))
            if model_name not in self.classical_bundles:
                model_name = sorted(self.classical_bundles)[0]
            return _model_pair_scores_for_batch(self.classical_bundles[model_name], batch, pairs)
        if self.meta_artifact is None:
            model_name = str(self.scorer.get("fallback_classical_model", "ensemble"))
            if model_name not in self.classical_bundles:
                model_name = sorted(self.classical_bundles)[0]
            return _model_pair_scores_for_batch(self.classical_bundles[model_name], batch, pairs)
        classical_names = [str(name) for name in self.meta_artifact.get("classical_model_names", sorted(self.classical_bundles))]
        fused_scores = _cosine_scores(batch.vectors, pairs)
        classical_scores = {
            name: _model_pair_scores_for_batch(self.classical_bundles[name], batch, pairs)
            for name in classical_names
            if name in self.classical_bundles
        }
        x = _meta_features(batch, pairs, classical_scores, list(classical_scores), fused_scores)
        return _score_model(self.meta_artifact["model"], x)


def _transform_single_signature(vectorizer: Mapping[str, object], signature: np.ndarray, component_dim: int) -> np.ndarray:
    scaler = vectorizer.get("scaler")
    reducer = vectorizer.get("reducer")
    x = np.asarray(signature, dtype=np.float32).reshape(1, -1)
    scaled = scaler.transform(x) if scaler is not None else x
    if reducer is None or component_dim <= 0:
        reduced = np.zeros(0, dtype=np.float32)
    else:
        reduced = np.asarray(reducer.transform(scaled), dtype=np.float32).reshape(-1)
    return _component_pad(reduced, component_dim)


def load_fused_encoder(bundle_dir: Path, device: torch.device | str | None = None) -> FusedTargetEncoder:
    return FusedTargetEncoder(bundle_dir, device=device)


def encode_values_fused(
    *args: object,
    bundle_dir: Path | str | None = None,
    device: torch.device | str | None = None,
    seed: int | None = None,
) -> np.ndarray:
    if len(args) == 3:
        encoder_or_bundle, values, task = args
        encoder = encoder_or_bundle if isinstance(encoder_or_bundle, FusedTargetEncoder) else load_fused_encoder(Path(str(encoder_or_bundle)), device=device)
    elif len(args) == 2 and bundle_dir is not None:
        values, task = args
        encoder = load_fused_encoder(Path(bundle_dir), device=device)
    else:
        raise TypeError("encode_values_fused expects (encoder_or_bundle, values, task) or (values, task, bundle_dir=...)")
    return encoder.encode_values(values, task, seed=seed)


def score_values_fused(
    *args: object,
    bundle_dir: Path | str | None = None,
    device: torch.device | str | None = None,
    seed: int | None = None,
) -> float:
    if len(args) == 5:
        encoder_or_bundle, left_values, left_task, right_values, right_task = args
        encoder = encoder_or_bundle if isinstance(encoder_or_bundle, FusedTargetEncoder) else load_fused_encoder(Path(str(encoder_or_bundle)), device=device)
    elif len(args) == 4 and bundle_dir is not None:
        left_values, left_task, right_values, right_task = args
        encoder = load_fused_encoder(Path(bundle_dir), device=device)
    else:
        raise TypeError(
            "score_values_fused expects (encoder_or_bundle, left_values, left_task, right_values, right_task) "
            "or (left_values, left_task, right_values, right_task, bundle_dir=...)"
        )
    return encoder.score_values(left_values, left_task, right_values, right_task, seed=seed)


def encode_fused_file(
    bundle_dir: Path,
    input_path: Path,
    task: str | int,
    *,
    output_path: Path | None = None,
    seed: int = 17,
    device: str | None = None,
) -> np.ndarray:
    target = deterministic_shuffle(read_target_input(input_path), str(Path(input_path).resolve()), seed)
    encoder = load_fused_encoder(bundle_dir, device=device)
    embedding = encoder.encode_values(target, task, seed=seed)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".json":
            output_path.write_text(json.dumps({"embedding": embedding.tolist()}) + "\n", encoding="utf-8")
        else:
            np.save(output_path, embedding)
    return embedding


def score_fused_files(
    bundle_dir: Path,
    left_input: Path,
    left_task: str | int,
    right_input: Path,
    right_task: str | int,
    *,
    seed: int = 17,
    device: str | None = None,
) -> float:
    left = deterministic_shuffle(read_target_input(left_input), str(Path(left_input).resolve()), seed)
    right = deterministic_shuffle(read_target_input(right_input), str(Path(right_input).resolve()), seed)
    encoder = load_fused_encoder(bundle_dir, device=device)
    return encoder.score_values(left, left_task, right, right_task, seed=seed)


def evaluate_fused_bundle_on_windows(
    bundle_dir: Path,
    sampled: Mapping[str, Sequence[object]],
    pair_indices: Mapping[str, Sequence[PairIndex]],
    retrieval_queries: Sequence[RetrievalQuery],
    *,
    device: str | None = None,
    batch_size: int = 512,
) -> dict[str, object]:
    encoder = load_fused_encoder(bundle_dir, device=device)
    val_batch = encoder.encode_window_batch(sampled["val"], batch_size=batch_size)
    test_batch = encoder.encode_window_batch(sampled["test"], batch_size=batch_size)
    y_val = np.asarray([pair.label for pair in pair_indices["val"]], dtype=np.int64)
    y_test = np.asarray([pair.label for pair in pair_indices["test"]], dtype=np.int64)
    val_scores = encoder.score_pairs_from_batch(val_batch, pair_indices["val"])
    test_scores = encoder.score_pairs_from_batch(test_batch, pair_indices["test"])
    test_retrieval = _fused_retrieval(encoder, test_batch, retrieval_queries)
    return {
        **_score_report_with_retrieval(y_val, val_scores, y_test, test_scores, {}, test_retrieval),
        "bundle_dir": str(bundle_dir),
        "default_scorer": encoder.scorer.get("default"),
    }


def _fused_retrieval(encoder: FusedTargetEncoder, batch: _BatchEncoding, queries: Sequence[RetrievalQuery]) -> dict[str, float]:
    if not queries:
        return {}
    if encoder.scorer.get("default") != "fused_meta":
        model_name = str(encoder.scorer.get("classical_model", "ensemble"))
        if model_name in encoder.classical_bundles:
            return _model_retrieval_for_batch(encoder.classical_bundles[model_name], batch, queries)
        return _retrieval_from_vectors(batch.vectors, queries)
    if encoder.meta_artifact is None:
        return _retrieval_from_vectors(batch.vectors, queries)
    classical_names = [str(name) for name in encoder.meta_artifact.get("classical_model_names", sorted(encoder.classical_bundles))]
    return _meta_retrieval(encoder.meta_artifact["model"], batch, queries, encoder.classical_bundles, classical_names)
