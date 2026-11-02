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
    BUNDLE_FORMAT,
    BUNDLE_VERSION,
    FusedTrainingConfig,
    FusedWindow,
    _BatchEncoding,
    _batch_from_components,
    _cosine_scores,
    _encode_neural_windows,
    _fit_classical_models,
    _fit_classical_models_all,
    _fit_meta_candidates,
    _fit_meta_model,
    _fit_signature_vectorizer,
    _jsonable_config,
    _meta_feature_names,
    _meta_features,
    _model_pair_scores_for_batch,
    _record_signatures,
    _retrieval_from_vectors,
    _sample_windows,
    _score_report_with_retrieval,
    _transform_signatures,
    _weight_grid,
    _window_features,
)

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
        "signature_feature_names": [
            f"{agg}:{name}" for agg in ("mean", "std", "min", "max") for name in feature_names
        ],
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
            name: os.path.relpath(str(bundle["model_path"]), str(bundle_dir))
            for name, bundle in classical_bundles.items()
        },
        "vector_weights": [float(x) for x in vector_weights],
        "scorer": dict(scorer),
        "report": dict(report),
    }
    # Bind all joblib load paths, including an optional candidate meta scorer.
    artifact_paths = {vectorizer_artifact, *manifest["classical_models"].values()}
    artifact_paths.update(
        str(scorer[key]) for key in ("artifact", "candidate_meta_artifact") if scorer.get(key)
    )
    manifest["artifact_sha256"] = {
        rel_path: artifact_sha256(bundle_dir / rel_path) for rel_path in sorted(artifact_paths)
    }
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

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
    vectorizer = _fit_signature_vectorizer(
        {record.dataset_id: signatures[record.dataset_id] for record in records_by_split["train"]},
        cfg.classical_component_dim,
        cfg.seed,
    )
    component_by_record = _transform_signatures(vectorizer, signatures)
    vectorizer_path = bundle_dir / "fused_vectorizer.joblib"
    joblib.dump({**vectorizer, "feature_names": feature_names}, vectorizer_path)

    classical_bundles, classical_reports = _fit_classical_models(
        bundle_dir, cfg, sampled, feature_names, pair_indices, retrieval_queries
    )
    best_embeddings = {
        split: _encode_neural_windows(
            cfg.best_checkpoint, windows, device=cfg.device, batch_size=cfg.batch_size
        )
        for split, windows in sampled.items()
    }
    final_embeddings = {
        split: _encode_neural_windows(
            cfg.final_checkpoint, windows, device=cfg.device, batch_size=cfg.batch_size
        )
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
    best_vector_key = max(
        vector_reports, key=lambda key: float(vector_reports[key].get("validation_score", -1.0))
    )
    best_classical_name = max(
        classical_reports, key=lambda key: float(classical_reports[key].get("validation_score", -1.0))
    )
    hist_score = (
        float(classical_reports.get("hist_gbdt", {}).get("validation_score", -1.0))
        if isinstance(classical_reports, Mapping)
        else -1.0
    )
    best_classical_score = float(classical_reports[best_classical_name].get("validation_score", -1.0))
    best_meta_score = float(meta_reports[best_meta_key].get("validation_score", -1.0))
    fusion_wins = best_meta_score > max(hist_score, best_classical_score)
    selected_key = best_meta_key if fusion_wins else best_classical_name
    selected_weight_key = str(meta_reports[best_meta_key].get("weight_key", best_vector_key))
    selected_weights = dict(_weight_grid())[selected_weight_key]

    scorer_artifact = None
    if fusion_wins:
        scorer_artifact = _relative_artifact(
            bundle_dir, Path(str(meta_artifacts[best_meta_key]["artifact_path"]))
        )
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

    checkpoint_best, checkpoint_final = _copy_checkpoints(
        bundle_dir, cfg.best_checkpoint, cfg.final_checkpoint
    )
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
    best_embeddings = _encode_neural_windows(
        cfg.best_checkpoint, windows, device=cfg.device, batch_size=cfg.batch_size
    )
    final_embeddings = _encode_neural_windows(
        cfg.final_checkpoint, windows, device=cfg.device, batch_size=cfg.batch_size
    )
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
        name: _model_pair_scores_for_batch(bundle, batch, pairs) for name, bundle in classical_bundles.items()
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
    best_classical_name = str(
        selection.get(
            "best_classical_name", "ensemble" if "ensemble" in classical_bundles else classical_names[0]
        )
    )
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
    checkpoint_best, checkpoint_final = _copy_checkpoints(
        bundle_dir, cfg.best_checkpoint, cfg.final_checkpoint
    )
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
    splits = _split_records(
        records, train_fraction=cfg.train_fraction, val_fraction=cfg.val_fraction, seed=cfg.seed
    )
    if not splits["train"] or not splits["val"] or not splits["test"]:
        raise ValueError(
            f"Non-empty train/val/test splits are required; got {[len(splits[key]) for key in ('train', 'val', 'test')]}"
        )

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
