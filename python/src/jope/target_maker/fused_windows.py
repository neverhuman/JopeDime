from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .classical import (
    WindowFeature,
    summarize_target_window,
)
from .data import (
    TASK_BINARY,
    DatasetRecord,
    bits_at_indices,
    compute_profile,
    sample_indices,
    stable_seed,
)
from .encode import normalize_task
from .fused import (
    FusedTrainingConfig,
    FusedWindow,
    _BatchEncoding,
    _component_pad,
    _weighted_fused_vector,
)
from .model import build_model_from_checkpoint


def _window_features(windows: Sequence[FusedWindow]) -> list[WindowFeature]:
    return [WindowFeature(window.dataset_id, window.task, window.features) for window in windows]


def _values_from_npz(record: DatasetRecord, npz: Mapping[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    if record.task == TASK_BINARY:
        return bits_at_indices(npz["bits"], indices)
    return np.asarray(npz["values"], dtype=np.float32)[indices]


def _sample_windows(
    records: Sequence[DatasetRecord], cfg: FusedTrainingConfig
) -> tuple[list[FusedWindow], list[str]]:
    windows: list[FusedWindow] = []
    feature_names: list[str] | None = None
    for record in records:
        with np.load(record.cache_path, allow_pickle=False) as npz:
            cache_profile = (
                np.asarray(npz["profile"], dtype=np.float32) if cfg.include_cache_profile else None
            )
            for view_index in range(cfg.views_per_record):
                rng = np.random.default_rng(
                    stable_seed("fusion-window", cfg.seed, record.dataset_id, view_index, cfg.window_size)
                )
                indices, effective_count, replacement_fraction = sample_indices(
                    record.length, cfg.window_size, rng
                )
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
    return np.concatenate([x.mean(axis=0), x.std(axis=0), x.min(axis=0), x.max(axis=0)]).astype(
        np.float32, copy=False
    )


def _record_signatures(
    records: Sequence[DatasetRecord],
    cfg: FusedTrainingConfig,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    signatures: dict[str, np.ndarray] = {}
    mean_features: dict[str, np.ndarray] = {}
    feature_names: list[str] | None = None
    for record in records:
        with np.load(record.cache_path, allow_pickle=False) as npz:
            cache_profile = (
                np.asarray(npz["profile"], dtype=np.float32) if cfg.include_cache_profile else None
            )
            rows: list[np.ndarray] = []
            for view_index in range(cfg.signature_views):
                rng = np.random.default_rng(
                    stable_seed("fusion-signature", cfg.seed, record.dataset_id, view_index, cfg.window_size)
                )
                indices, effective_count, replacement_fraction = sample_indices(
                    record.length, cfg.window_size, rng
                )
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
    arr = np.asarray(
        list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float32
    ).reshape(-1)
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
        rng = np.random.default_rng(
            stable_seed("fusion-signature-inference", seed, task_id, original_length, view_index, window_size)
        )
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


def _fit_signature_vectorizer(
    signatures: Mapping[str, np.ndarray], component_dim: int, seed: int
) -> dict[str, object]:
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


def _transform_signatures(
    vectorizer: Mapping[str, object], signatures: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
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
        values = torch.from_numpy(
            np.stack([window.values for window in chunk]).astype(np.float32, copy=False)
        ).to(torch_device)
        task = torch.tensor([window.task for window in chunk], device=torch_device, dtype=torch.long)
        original_length = torch.tensor(
            [window.original_length for window in chunk], device=torch_device, dtype=torch.float32
        )
        effective_count = torch.tensor(
            [window.effective_count for window in chunk], device=torch_device, dtype=torch.float32
        )
        replacement_fraction = torch.tensor(
            [window.replacement_fraction for window in chunk], device=torch_device, dtype=torch.float32
        )
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
        replacement_fractions=np.asarray(
            [window.replacement_fraction for window in windows], dtype=np.float64
        ),
    )
