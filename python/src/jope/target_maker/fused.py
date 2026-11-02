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
        self.vectorizer = self._load_joblib_artifact(
            self.manifest["vectorizer"]["artifact"], context="Fused vectorizer"
        )
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
            bundle = self._load_joblib_artifact(rel_path, context=f"Fused classical model {name}")
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
            self.meta_artifact = self._load_joblib_artifact(meta_rel, context="Fused meta scorer")

    def _load_joblib_artifact(self, rel_path: object, *, context: str):
        pins = self.manifest.get("artifact_sha256", {})
        if not isinstance(pins, Mapping):
            raise ValueError("Fused manifest artifact SHA-256 pins must be a mapping")
        return load_joblib_pinned(self._artifact_path(rel_path), pins.get(str(rel_path)), context=context)

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

    def _encode_target(
        self, values: Iterable[float] | np.ndarray, task: str | int, *, seed: int
    ) -> _TargetEncoding:
        raw_signature, mean_features, names, original_length, effective_count, replacement_fraction = (
            _raw_signature_from_values(
                values,
                task,
                window_size=self.window_size,
                signature_views=self.signature_views,
                seed=seed,
                include_cache_profile=self.include_cache_profile,
            )
        )
        if names != self.feature_names:
            missing = set(self.feature_names).symmetric_difference(names)
            sample = sorted(missing)[:5]
            raise ValueError(
                f"Fused bundle feature names are incompatible with input signature; sample mismatch: {sample}"
            )
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
            indices = tuple(
                int(index)
                for index in bundle.get("selected_feature_indices", range(left.window_features.shape[0]))
            )
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
        classical_names = [
            str(name) for name in self.meta_artifact.get("classical_model_names", sorted(classical_scores))
        ]
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
        raw_signatures = {
            key: _signature_from_feature_stack(np.stack(rows)) for key, rows in signatures.items()
        }
        components = _transform_signatures(self.vectorizer, raw_signatures)
        best_path = self._artifact_path(self.manifest["neural"]["best"])
        final_path = self._artifact_path(self.manifest["neural"]["final"])
        best = _encode_neural_windows(
            best_path, fused_windows, device=str(self.device), batch_size=batch_size
        )
        final = _encode_neural_windows(
            final_path, fused_windows, device=str(self.device), batch_size=batch_size
        )
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
        classical_names = [
            str(name)
            for name in self.meta_artifact.get("classical_model_names", sorted(self.classical_bundles))
        ]
        fused_scores = _cosine_scores(batch.vectors, pairs)
        classical_scores = {
            name: _model_pair_scores_for_batch(self.classical_bundles[name], batch, pairs)
            for name in classical_names
            if name in self.classical_bundles
        }
        x = _meta_features(batch, pairs, classical_scores, list(classical_scores), fused_scores)
        return _score_model(self.meta_artifact["model"], x)


def _transform_single_signature(
    vectorizer: Mapping[str, object], signature: np.ndarray, component_dim: int
) -> np.ndarray:
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
        encoder = (
            encoder_or_bundle
            if isinstance(encoder_or_bundle, FusedTargetEncoder)
            else load_fused_encoder(Path(str(encoder_or_bundle)), device=device)
        )
    elif len(args) == 2 and bundle_dir is not None:
        values, task = args
        encoder = load_fused_encoder(Path(bundle_dir), device=device)
    else:
        raise TypeError(
            "encode_values_fused expects (encoder_or_bundle, values, task) or (values, task, bundle_dir=...)"
        )
    return encoder.encode_values(values, task, seed=seed)


def score_values_fused(
    *args: object,
    bundle_dir: Path | str | None = None,
    device: torch.device | str | None = None,
    seed: int | None = None,
) -> float:
    if len(args) == 5:
        encoder_or_bundle, left_values, left_task, right_values, right_task = args
        encoder = (
            encoder_or_bundle
            if isinstance(encoder_or_bundle, FusedTargetEncoder)
            else load_fused_encoder(Path(str(encoder_or_bundle)), device=device)
        )
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


def _fused_retrieval(
    encoder: FusedTargetEncoder, batch: _BatchEncoding, queries: Sequence[RetrievalQuery]
) -> dict[str, float]:
    if not queries:
        return {}
    if encoder.scorer.get("default") != "fused_meta":
        model_name = str(encoder.scorer.get("classical_model", "ensemble"))
        if model_name in encoder.classical_bundles:
            return _model_retrieval_for_batch(encoder.classical_bundles[model_name], batch, queries)
        return _retrieval_from_vectors(batch.vectors, queries)
    if encoder.meta_artifact is None:
        return _retrieval_from_vectors(batch.vectors, queries)
    classical_names = [
        str(name)
        for name in encoder.meta_artifact.get("classical_model_names", sorted(encoder.classical_bundles))
    ]
    return _meta_retrieval(
        encoder.meta_artifact["model"], batch, queries, encoder.classical_bundles, classical_names
    )


from .fused_windows import (
    _window_features,
    _values_from_npz,
    _sample_windows,
    _signature_from_feature_stack,
    _record_signatures,
    _raw_signature_from_values,
    _fit_signature_vectorizer,
    _transform_signatures,
    _encode_neural_windows,
    _batch_from_components,
)

from .fused_scoring import (
    _cosine_scores,
    _pair_features_matrix,
    _pair_matrix_from_feature_matrix,
    _score_pairs_from_feature_matrix,
    _retrieval_from_vectors,
    _model_pair_scores_for_batch,
    _model_retrieval_for_batch,
    _classical_retrieval_from_features,
    _fit_classical_models,
    _fit_classical_models_all,
)

from .fused_meta import (
    _meta_feature_names,
    _meta_features,
    _meta_features_for_two,
    _fit_meta_model,
    _fit_meta_candidates,
    _meta_retrieval,
    _flatten_retrieval_pairs,
    _meta_retrieval_from_flat,
    _weight_grid,
)

from .fused_training import (
    _write_bundle_manifest,
    _copy_checkpoints,
    _relative_artifact,
    _train_best_bundle,
    _train_final_bundle,
    train_fusion,
)
