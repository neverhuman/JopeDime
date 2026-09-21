from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
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
    _top_features,
    _validation_weights,
    build_pair_indices,
    build_retrieval_queries,
    length_feature_indices,
    no_length_feature_indices,
    pair_dataset_from_indices,
    pair_feature_names,
    retrieval_metrics_for_model,
    retrieval_metrics_from_scores,
    select_window_features,
    summarize_target_window,
)
from .data import TASK_BINARY, DatasetRecord, bits_at_indices, load_manifest, sample_indices, stable_seed
from .model import build_model_from_checkpoint


@dataclass
class EvalRepresentationsConfig:
    cache_dir: Path
    run_dir: Path
    neural_checkpoints: tuple[Path, ...] = ()
    classical_model_dir: Path | None = None
    classical_models: tuple[str, ...] = ()
    fused_bundles: tuple[Path, ...] = ()
    window_size: int = 96
    views_per_record: int = 4
    limit_per_task: int | None = 5000
    train_pairs: int = 50_000
    val_pairs: int = 10_000
    test_pairs: int = 10_000
    retrieval_queries: int = 1000
    retrieval_candidates: int = 256
    seed: int = 17
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    positive_fraction: float = 0.50
    same_task_negative_fraction: float = 0.80
    include_cache_profile: bool = False
    n_jobs: int = -1
    device: str | None = None
    batch_size: int = 512


@dataclass(frozen=True)
class EvalWindow:
    dataset_id: str
    task: int
    original_length: int
    effective_count: int
    replacement_fraction: float
    values: np.ndarray
    features: np.ndarray


def _jsonable_config(cfg: EvalRepresentationsConfig) -> dict[str, object]:
    data = asdict(cfg)
    data["cache_dir"] = str(cfg.cache_dir)
    data["run_dir"] = str(cfg.run_dir)
    data["neural_checkpoints"] = [str(path) for path in cfg.neural_checkpoints]
    data["classical_model_dir"] = None if cfg.classical_model_dir is None else str(cfg.classical_model_dir)
    data["classical_models"] = list(cfg.classical_models)
    data["fused_bundles"] = [str(path) for path in cfg.fused_bundles]
    return data


def _window_features(windows: Sequence[EvalWindow]) -> list[WindowFeature]:
    return [WindowFeature(window.dataset_id, window.task, window.features) for window in windows]


def _sample_eval_windows(
    records: Sequence[DatasetRecord],
    cfg: EvalRepresentationsConfig,
) -> tuple[list[EvalWindow], list[str]]:
    windows: list[EvalWindow] = []
    feature_names: list[str] | None = None
    for record in records:
        with np.load(record.cache_path, allow_pickle=False) as npz:
            cache_profile = (
                np.asarray(npz["profile"], dtype=np.float32) if cfg.include_cache_profile else None
            )
            for view_index in range(cfg.views_per_record):
                rng = np.random.default_rng(
                    stable_seed("classical-window", cfg.seed, record.dataset_id, view_index, cfg.window_size)
                )
                indices, effective_count, replacement_fraction = sample_indices(
                    record.length, cfg.window_size, rng
                )
                if record.task == TASK_BINARY:
                    values = bits_at_indices(npz["bits"], indices)
                else:
                    values = np.asarray(npz["values"], dtype=np.float32)[indices]
                values = np.asarray(values, dtype=np.float32)
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
                    raise RuntimeError("Feature names changed while sampling evaluation windows")
                windows.append(
                    EvalWindow(
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
        raise ValueError("No evaluation windows sampled")
    return windows, feature_names


def _score_report(
    y_val: np.ndarray, val_scores: np.ndarray, y_test: np.ndarray, test_scores: np.ndarray
) -> dict[str, object]:
    threshold = _best_f1_threshold(y_val, val_scores)
    return {
        "val": _classification_metrics(y_val, val_scores, threshold),
        "test": _classification_metrics(y_test, test_scores, threshold),
    }


def _length_scores(windows: Sequence[EvalWindow], pairs: Sequence[PairIndex]) -> np.ndarray:
    meta = np.asarray(
        [
            [
                math.log1p(window.original_length),
                math.log1p(window.effective_count),
                window.replacement_fraction,
            ]
            for window in windows
        ],
        dtype=np.float64,
    )
    scores = []
    for pair in pairs:
        diff = meta[pair.left] - meta[pair.right]
        scores.append(-float(np.sqrt(np.mean(diff * diff))))
    return np.asarray(scores, dtype=np.float64)


def _length_retrieval(windows: Sequence[EvalWindow], queries: Sequence[RetrievalQuery]) -> dict[str, float]:
    if not queries:
        return {}
    meta = np.asarray(
        [
            [
                math.log1p(window.original_length),
                math.log1p(window.effective_count),
                window.replacement_fraction,
            ]
            for window in windows
        ],
        dtype=np.float64,
    )
    scores_by_query: list[np.ndarray] = []
    for query in queries:
        diff = meta[np.asarray(query.candidates, dtype=np.int64)] - meta[int(query.query)]
        scores_by_query.append(-np.sqrt(np.mean(diff * diff, axis=1)))
    return retrieval_metrics_from_scores(scores_by_query, len(queries[0].candidates))


def _classical_bundle_for(
    sampled_features: Mapping[str, list[WindowFeature]],
    feature_names: Sequence[str],
    pair_indices: Mapping[str, Sequence[PairIndex]],
    indices: tuple[int, ...],
):
    selected = {
        split: select_window_features(windows, indices) for split, windows in sampled_features.items()
    }
    selected_feature_names = [feature_names[index] for index in indices]
    selected_pair_names = pair_feature_names(selected_feature_names)
    x_val, y_val = pair_dataset_from_indices(selected["val"], pair_indices["val"])
    x_test, y_test = pair_dataset_from_indices(selected["test"], pair_indices["test"])
    return selected, selected_feature_names, selected_pair_names, x_val, y_val, x_test, y_test


def _evaluate_classical_model(
    model: object,
    sampled_features: Mapping[str, list[WindowFeature]],
    feature_names: Sequence[str],
    pair_indices: Mapping[str, Sequence[PairIndex]],
    retrieval_queries: Sequence[RetrievalQuery],
    indices: tuple[int, ...],
) -> dict[str, object]:
    selected, _, selected_pair_names, x_val, y_val, x_test, y_test = _classical_bundle_for(
        sampled_features,
        feature_names,
        pair_indices,
        indices,
    )
    val_scores = _score_model(model, x_val)
    test_scores = _score_model(model, x_test)
    return {
        **_score_report(y_val, val_scores, y_test, test_scores),
        "retrieval": retrieval_metrics_for_model(model, selected["test"], retrieval_queries),
        "top_features": _top_features(model, selected_pair_names),
    }


def _fit_eval_classical(
    cfg: EvalRepresentationsConfig,
    sampled_features: Mapping[str, list[WindowFeature]],
    feature_names: Sequence[str],
    pair_indices: Mapping[str, Sequence[PairIndex]],
    retrieval_queries: Sequence[RetrievalQuery],
) -> tuple[dict[str, object], dict[str, object]]:
    requested = tuple(dict.fromkeys(name.strip() for name in cfg.classical_models if name.strip()))
    if not requested:
        return {}, {}
    base_requested = [name for name in requested if name != "ensemble"]
    if "length_only" not in base_requested:
        base_requested.append("length_only")
    full_indices = tuple(range(len(feature_names)))
    length_indices_tuple = tuple(length_feature_indices(feature_names))
    no_length_indices_tuple = tuple(no_length_feature_indices(feature_names))
    model_dir = cfg.run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    def fit_one_set(
        specs: Mapping[str, object], indices: tuple[int, ...], prefix: str = ""
    ) -> tuple[dict[str, object], dict[str, object], np.ndarray]:
        selected, selected_feature_names, selected_pair_names, x_val, y_val, x_test, y_test = (
            _classical_bundle_for(
                sampled_features,
                feature_names,
                pair_indices,
                indices,
            )
        )
        x_train, y_train = pair_dataset_from_indices(selected["train"], pair_indices["train"])
        reports: dict[str, object] = {}
        fitted: dict[str, object] = {}
        import joblib

        for model_name, model in specs.items():
            model.fit(x_train, y_train)
            val_scores = _score_model(model, x_val)
            test_scores = _score_model(model, x_test)
            model_key = f"{prefix}{model_name}"
            model_path = model_dir / f"{_model_artifact_name(model_key)}.joblib"
            joblib.dump(
                {
                    "model": model,
                    "model_name": model_key,
                    "feature_names": selected_feature_names,
                    "pair_feature_names": selected_pair_names,
                    "selected_feature_indices": list(indices),
                    "config": _jsonable_config(cfg),
                },
                model_path,
            )
            reports[model_name] = {
                **_score_report(y_val, val_scores, y_test, test_scores),
                "retrieval": retrieval_metrics_for_model(model, selected["test"], retrieval_queries),
                "top_features": _top_features(model, selected_pair_names),
                "model_path": str(model_path),
            }
            fitted[model_name] = model
        return reports, fitted, y_val

    full_specs = _model_specs(base_requested, cfg.seed, cfg.n_jobs)
    full_reports: dict[str, object] = {}
    full_fitted: dict[str, object] = {}
    y_val_full = np.zeros(0, dtype=np.int64)
    for model_name, model in full_specs.items():
        indices = length_indices_tuple if model_name == "length_only" else full_indices
        reports, fitted, y_val_full = fit_one_set({model_name: model}, indices)
        full_reports.update(reports)
        full_fitted.update(fitted)
    if _ensemble_requested(requested):
        member_names = _ensemble_members(full_reports, full_fitted)
        if member_names:
            selected, selected_feature_names, selected_pair_names, x_val, y_val, x_test, y_test = (
                _classical_bundle_for(
                    sampled_features,
                    feature_names,
                    pair_indices,
                    full_indices,
                )
            )
            weights = _validation_weights(
                member_names, full_reports, y_val_full if y_val_full.size else y_val
            )
            ensemble = WeightedScoreEnsemble([(name, full_fitted[name]) for name in member_names], weights)
            model_path = model_dir / "ensemble.joblib"
            import joblib

            joblib.dump(
                {
                    "model": ensemble,
                    "model_name": "ensemble",
                    "feature_names": selected_feature_names,
                    "pair_feature_names": selected_pair_names,
                    "selected_feature_indices": list(full_indices),
                    "member_names": member_names,
                    "weights": weights.tolist(),
                    "config": _jsonable_config(cfg),
                },
                model_path,
            )
            full_reports["ensemble"] = {
                **_score_report(y_val, _score_model(ensemble, x_val), y_test, _score_model(ensemble, x_test)),
                "retrieval": retrieval_metrics_for_model(ensemble, selected["test"], retrieval_queries),
                "top_features": [],
                "member_names": member_names,
                "weights": weights.tolist(),
                "model_path": str(model_path),
            }

    no_length_names = [name for name in base_requested if name not in {"dummy", "length_only"}]
    if not no_length_names:
        no_length_names = ["sgd_logistic"]
    no_length_reports, no_length_fitted, y_val_no_length = fit_one_set(
        _model_specs(no_length_names, cfg.seed + 997, cfg.n_jobs),
        no_length_indices_tuple,
        prefix="no_length_",
    )
    if _ensemble_requested(requested):
        member_names = _ensemble_members(no_length_reports, no_length_fitted)
        if member_names:
            selected, selected_feature_names, selected_pair_names, x_val, y_val, x_test, y_test = (
                _classical_bundle_for(
                    sampled_features,
                    feature_names,
                    pair_indices,
                    no_length_indices_tuple,
                )
            )
            weights = _validation_weights(member_names, no_length_reports, y_val_no_length)
            ensemble = WeightedScoreEnsemble(
                [(name, no_length_fitted[name]) for name in member_names], weights
            )
            model_path = model_dir / "no_length_ensemble.joblib"
            import joblib

            joblib.dump(
                {
                    "model": ensemble,
                    "model_name": "no_length_ensemble",
                    "feature_names": selected_feature_names,
                    "pair_feature_names": selected_pair_names,
                    "selected_feature_indices": list(no_length_indices_tuple),
                    "member_names": member_names,
                    "weights": weights.tolist(),
                    "config": _jsonable_config(cfg),
                },
                model_path,
            )
            no_length_reports["ensemble"] = {
                **_score_report(y_val, _score_model(ensemble, x_val), y_test, _score_model(ensemble, x_test)),
                "retrieval": retrieval_metrics_for_model(ensemble, selected["test"], retrieval_queries),
                "top_features": [],
                "member_names": member_names,
                "weights": weights.tolist(),
                "model_path": str(model_path),
            }
    return full_reports, no_length_reports


def _load_eval_classical(
    model_dir: Path | None,
    sampled_features: Mapping[str, list[WindowFeature]],
    feature_names: Sequence[str],
    pair_indices: Mapping[str, Sequence[PairIndex]],
    retrieval_queries: Sequence[RetrievalQuery],
) -> dict[str, object]:
    if model_dir is None:
        return {}
    import joblib

    reports: dict[str, object] = {}
    for model_path in sorted(Path(model_dir).glob("*.joblib")):
        bundle = joblib.load(model_path)
        if isinstance(bundle, Mapping):
            model = bundle.get("model")
            model_name = str(bundle.get("model_name") or model_path.stem)
            indices = tuple(
                int(index) for index in bundle.get("selected_feature_indices", range(len(feature_names)))
            )
        else:
            model = bundle
            model_name = model_path.stem
            indices = tuple(range(len(feature_names)))
        if model is None:
            continue
        reports[model_name] = {
            **_evaluate_classical_model(
                model, sampled_features, feature_names, pair_indices, retrieval_queries, indices
            ),
            "model_path": str(model_path),
        }
    return reports


@torch.no_grad()
def _encode_windows(
    checkpoint_path: Path,
    windows: Sequence[EvalWindow],
    cfg: EvalRepresentationsConfig,
    *,
    ablate_length: bool,
) -> np.ndarray:
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    model = build_model_from_checkpoint(checkpoint).to(device)
    model.eval()
    embeddings: list[torch.Tensor] = []
    batch_size = max(1, int(cfg.batch_size))
    for start in range(0, len(windows), batch_size):
        chunk = windows[start : start + batch_size]
        values = torch.from_numpy(
            np.stack([window.values for window in chunk]).astype(np.float32, copy=False)
        ).to(device)
        task = torch.tensor([window.task for window in chunk], device=device, dtype=torch.long)
        if ablate_length:
            original_length = torch.full((len(chunk),), float(cfg.window_size), device=device)
            effective_count = torch.full((len(chunk),), float(cfg.window_size), device=device)
            replacement_fraction = torch.zeros(len(chunk), device=device)
        else:
            original_length = torch.tensor(
                [window.original_length for window in chunk], device=device, dtype=torch.float32
            )
            effective_count = torch.tensor(
                [window.effective_count for window in chunk], device=device, dtype=torch.float32
            )
            replacement_fraction = torch.tensor(
                [window.replacement_fraction for window in chunk], device=device, dtype=torch.float32
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


def _embedding_pair_scores(embeddings: np.ndarray, pairs: Sequence[PairIndex]) -> np.ndarray:
    scores = [float(np.dot(embeddings[pair.left], embeddings[pair.right])) for pair in pairs]
    return np.asarray(scores, dtype=np.float64)


def _embedding_retrieval(embeddings: np.ndarray, queries: Sequence[RetrievalQuery]) -> dict[str, float]:
    if not queries:
        return {}
    scores_by_query: list[np.ndarray] = []
    for query in queries:
        candidate_embeddings = embeddings[np.asarray(query.candidates, dtype=np.int64)]
        scores_by_query.append(candidate_embeddings @ embeddings[int(query.query)])
    return retrieval_metrics_from_scores(scores_by_query, len(queries[0].candidates))


def _checkpoint_key(path: Path) -> str:
    parent = path.parent.name
    return _model_artifact_name(f"{parent}_{path.stem}") if parent else _model_artifact_name(path.stem)


def _bundle_key(path: Path) -> str:
    parent = path.parent.name
    return _model_artifact_name(f"{parent}_{path.stem}") if parent else _model_artifact_name(path.stem)


def _evaluate_neural_checkpoint(
    checkpoint_path: Path,
    sampled: Mapping[str, list[EvalWindow]],
    pair_indices: Mapping[str, Sequence[PairIndex]],
    retrieval_queries: Sequence[RetrievalQuery],
    cfg: EvalRepresentationsConfig,
    *,
    ablate_length: bool,
) -> dict[str, object]:
    val_embeddings = _encode_windows(checkpoint_path, sampled["val"], cfg, ablate_length=ablate_length)
    test_embeddings = _encode_windows(checkpoint_path, sampled["test"], cfg, ablate_length=ablate_length)
    y_val = np.asarray([pair.label for pair in pair_indices["val"]], dtype=np.int64)
    y_test = np.asarray([pair.label for pair in pair_indices["test"]], dtype=np.int64)
    val_scores = _embedding_pair_scores(val_embeddings, pair_indices["val"])
    test_scores = _embedding_pair_scores(test_embeddings, pair_indices["test"])
    return {
        **_score_report(y_val, val_scores, y_test, test_scores),
        "retrieval": _embedding_retrieval(test_embeddings, retrieval_queries),
        "checkpoint": str(checkpoint_path),
    }


def eval_representations(cfg: EvalRepresentationsConfig) -> dict[str, object]:
    records = load_manifest(cfg.cache_dir, split="all", seed=cfg.seed)
    records = _limit_records(records, cfg.limit_per_task, cfg.seed)
    if len(records) < 3:
        raise ValueError("Need at least three records for train/val/test evaluation splits")
    splits = _split_records(
        records, train_fraction=cfg.train_fraction, val_fraction=cfg.val_fraction, seed=cfg.seed
    )
    if not splits["train"] or not splits["val"] or not splits["test"]:
        raise ValueError(
            f"Non-empty train/val/test splits are required; got {[len(splits[key]) for key in ('train', 'val', 'test')]}"
        )

    sampled: dict[str, list[EvalWindow]] = {}
    feature_names: list[str] | None = None
    for split_name, split_records in splits.items():
        windows, names = _sample_eval_windows(split_records, cfg)
        sampled[split_name] = windows
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise RuntimeError("Feature names changed across evaluation splits")
    assert feature_names is not None
    sampled_features = {split: _window_features(windows) for split, windows in sampled.items()}
    pair_indices = {
        "train": build_pair_indices(
            sampled_features["train"],
            cfg.train_pairs,
            seed=stable_seed("eval-pairs", cfg.seed, "train"),
            positive_fraction=cfg.positive_fraction,
            same_task_negative_fraction=cfg.same_task_negative_fraction,
        ),
        "val": build_pair_indices(
            sampled_features["val"],
            cfg.val_pairs,
            seed=stable_seed("eval-pairs", cfg.seed, "val"),
            positive_fraction=cfg.positive_fraction,
            same_task_negative_fraction=cfg.same_task_negative_fraction,
        ),
        "test": build_pair_indices(
            sampled_features["test"],
            cfg.test_pairs,
            seed=stable_seed("eval-pairs", cfg.seed, "test"),
            positive_fraction=cfg.positive_fraction,
            same_task_negative_fraction=cfg.same_task_negative_fraction,
        ),
    }
    retrieval_queries = build_retrieval_queries(
        sampled_features["test"],
        queries=cfg.retrieval_queries,
        candidates=cfg.retrieval_candidates,
        seed=stable_seed("eval-retrieval", cfg.seed, "shared"),
    )

    y_val = np.asarray([pair.label for pair in pair_indices["val"]], dtype=np.int64)
    y_test = np.asarray([pair.label for pair in pair_indices["test"]], dtype=np.int64)
    length_val_scores = _length_scores(sampled["val"], pair_indices["val"])
    length_test_scores = _length_scores(sampled["test"], pair_indices["test"])

    neural_reports: dict[str, object] = {}
    for checkpoint in cfg.neural_checkpoints:
        checkpoint = Path(checkpoint).expanduser()
        key = _checkpoint_key(checkpoint)
        neural_reports[key] = {
            "normal": _evaluate_neural_checkpoint(
                checkpoint, sampled, pair_indices, retrieval_queries, cfg, ablate_length=False
            ),
            "length_ablated": _evaluate_neural_checkpoint(
                checkpoint, sampled, pair_indices, retrieval_queries, cfg, ablate_length=True
            ),
        }

    trained_classical, no_length_classical = _fit_eval_classical(
        cfg,
        sampled_features,
        feature_names,
        pair_indices,
        retrieval_queries,
    )
    loaded_classical = _load_eval_classical(
        cfg.classical_model_dir,
        sampled_features,
        feature_names,
        pair_indices,
        retrieval_queries,
    )
    fused_reports: dict[str, object] = {}
    if cfg.fused_bundles:
        from .fused import evaluate_fused_bundle_on_windows

        for bundle in cfg.fused_bundles:
            bundle = Path(bundle).expanduser()
            fused_reports[_bundle_key(bundle)] = evaluate_fused_bundle_on_windows(
                bundle,
                sampled,
                pair_indices,
                retrieval_queries,
                device=cfg.device,
                batch_size=cfg.batch_size,
            )

    length_indices = length_feature_indices(feature_names)
    report: dict[str, object] = {
        "config": _jsonable_config(cfg),
        "record_count": len(records),
        "split_records": {key: len(value) for key, value in splits.items()},
        "split_windows": {key: len(value) for key, value in sampled.items()},
        "pair_counts": {key: len(value) for key, value in pair_indices.items()},
        "retrieval_queries": len(retrieval_queries),
        "window_feature_dim": len(feature_names),
        "pair_feature_dim": len(pair_feature_names(feature_names)),
        "length_feature_names": [feature_names[index] for index in length_indices],
        "no_length_window_feature_dim": len(no_length_feature_indices(feature_names)),
        "length_only": {
            **_score_report(y_val, length_val_scores, y_test, length_test_scores),
            "retrieval": _length_retrieval(sampled["test"], retrieval_queries),
        },
        "neural": neural_reports,
        "classical_trained": trained_classical,
        "classical_loaded": loaded_classical,
        "classical_no_length": no_length_classical,
        "fused": fused_reports,
    }

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    report_path = cfg.run_dir / "eval_representations_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
