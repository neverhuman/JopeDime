from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from jope.data.importers import import_quant_reports
from jope.data.labels import assign_elo, assign_ranks
from jope.data.target_encoding import TARGET_VS_EMBED_DIM, encode_target_embedding
from jope.data.vs import DATASET_SIGNATURE_DIM, VS_DIM, dataset_signature_from_vs, read_vs_matrix
from jope.grammar import HEADS, TRANSFORMS
from jope.io import write_jsonl
from jope.prime_config import load_config
from jope.schema import TrainingRow, normalize_embedding
from jope.validator import RustValidator

ACTIVE_TRAIN_SIDECAR = "hyperion3train128dim.vs"
ACTIVE_TEST_SIDECAR = "hyperion3test128dim.vs"
ACTIVE_TRAIN_TARGET_SIDECAR = "train_target.vs"
ACTIVE_TEST_TARGET_SIDECAR = "test_target.vs"
ACTIVE_MAX_TRANSFORMS = 6
TARGET_SUMMARY_DIM = TARGET_VS_EMBED_DIM
DEFAULT_QUANT_ROOT = load_config().regression_dataset_root


@dataclass(frozen=True)
class ActiveBuild:
    rows: list[TrainingRow]
    vocab: dict[str, Any]
    standardization: dict[str, Any]
    manifest: dict[str, Any]


def build_active_corpus(
    *,
    quant_root: str | Path = DEFAULT_QUANT_ROOT,
    split_json: str | Path | None = None,
    out_dir: str | Path | None = None,
    out_jsonl: str | Path | None = None,
    limit_datasets: int | None = None,
    validation_fraction: float = 0.1,
    seed: int = 0,
    validate: bool = True,
    target_encoder: str = "none",
    target_model: str | Path | None = None,
) -> ActiveBuild:
    root = Path(quant_root)
    sidecars = discover_train_sidecars(root)
    if limit_datasets is not None:
        keep = set(sorted(sidecars)[:limit_datasets])
        sidecars = {dataset_hash: path for dataset_hash, path in sidecars.items() if dataset_hash in keep}
    raw_vectors: dict[str, np.ndarray] = {}
    vs_row_counts: dict[str, int] = {}
    for dataset_hash, path in sidecars.items():
        matrix = read_vs_matrix(path)
        raw_vectors[dataset_hash] = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
        vs_row_counts[dataset_hash] = int(matrix.shape[0])
    target_embeddings = _build_target_embeddings(sidecars, target_encoder, target_model)
    split_by_dataset = _assign_splits(sorted(raw_vectors), split_json, validation_fraction, seed)
    mean, scale, train_count = _fit_standardizer(raw_vectors, split_by_dataset)

    rows = [row for row in import_quant_reports(root) if row.dataset_id in raw_vectors]
    if validate and rows:
        _canonicalize_rows(rows)
    rows = [row for row in rows if row.valid and _active_canonical(row)]

    for row in rows:
        vector = (raw_vectors[row.dataset_id] - mean) / scale
        target_embedding = target_embeddings.get(row.dataset_id)
        row.embedding = normalize_embedding(vector.tolist() + (target_embedding or []))
        row.target_embedding = normalize_embedding(target_embedding) if target_embedding is not None else None
        row.metadata.update(
            {
                "dataset_hash": row.dataset_id,
                "split": split_by_dataset.get(row.dataset_id, "train"),
                "vs_path": str(sidecars[row.dataset_id]),
                "x128_dim": VS_DIM,
                "x128_rows": vs_row_counts[row.dataset_id],
                "target_encoder": target_encoder,
                "target_embedding_dim": 0 if target_embedding is None else len(target_embedding),
                "actions": parse_pipeline_actions(row.canonical_genome or row.genome),
            }
        )
        row.metric = row.metric or "cv.oof_r2"

    _assign_relative_labels(rows)
    assign_ranks(rows)
    assign_elo(rows)
    vocab = build_active_vocab(row.canonical_genome or row.genome for row in rows)
    standardization = {
        "dim": VS_DIM,
        "fit_split": "train",
        "train_dataset_count": train_count,
        "mean": mean.astype(float).tolist(),
        "scale": scale.astype(float).tolist(),
    }
    manifest = {
        "quant_root": str(root),
        "split_json": str(split_json) if split_json else None,
        "datasets_with_sidecar": len(sidecars),
        "rows": len(rows),
        "splits": dict(sorted(_count_splits(split_by_dataset).items())),
        "active_max_transforms": ACTIVE_MAX_TRANSFORMS,
        "target_encoder": target_encoder,
    }
    build = ActiveBuild(rows=rows, vocab=vocab, standardization=standardization, manifest=manifest)
    if out_dir is not None:
        write_active_artifacts(out_dir, build)
    if out_jsonl is not None:
        write_jsonl(out_jsonl, rows)
    return build


def build_active_v2_corpus(
    *,
    quant_root: str | Path = DEFAULT_QUANT_ROOT,
    split_json: str | Path | None = None,
    out_dir: str | Path | None = None,
    out_jsonl: str | Path | None = None,
    limit_datasets: int | None = None,
    validation_fraction: float = 0.1,
    seed: int = 0,
    validate: bool = True,
    target_encoder: str = "train_target_vs",
    target_model: str | Path | None = None,
) -> ActiveBuild:
    root = Path(quant_root)
    sidecars = discover_train_sidecars(root)
    if limit_datasets is not None:
        keep = set(sorted(sidecars)[:limit_datasets])
        sidecars = {dataset_hash: path for dataset_hash, path in sidecars.items() if dataset_hash in keep}

    raw_vectors: dict[str, np.ndarray] = {}
    vs_row_counts: dict[str, int] = {}
    for dataset_hash, path in sidecars.items():
        matrix = read_vs_matrix(path)
        raw_vectors[dataset_hash] = dataset_signature_from_vs(path)
        vs_row_counts[dataset_hash] = int(matrix.shape[0])

    target_embeddings = _build_target_embeddings(sidecars, target_encoder, target_model)
    split_by_dataset = _assign_splits(sorted(raw_vectors), split_json, validation_fraction, seed)
    mean, scale, train_count = _fit_standardizer(raw_vectors, split_by_dataset)

    rows = [row for row in import_quant_reports(root) if row.dataset_id in raw_vectors]
    if validate and rows:
        _canonicalize_rows(rows)
    rows = [row for row in rows if row.valid and _active_canonical(row)]
    rows = _dedupe_active_rows(rows)

    for row in rows:
        actions = parse_pipeline_actions(row.canonical_genome or row.genome)
        canonical = format_pipeline_actions(actions)
        vector = (raw_vectors[row.dataset_id] - mean) / scale
        target_embedding = target_embeddings.get(row.dataset_id)
        target_summary = None if target_embedding is None else target_embedding[:TARGET_SUMMARY_DIM]
        sidecar = sidecars[row.dataset_id]
        train_csv = sidecar.parent / "train.csv"
        test_csv = sidecar.parent / "test.csv"
        train_target_vs = sidecar.parent / ACTIVE_TRAIN_TARGET_SIDECAR
        test_target_vs = sidecar.parent / ACTIVE_TEST_TARGET_SIDECAR
        row.canonical_genome = canonical
        row.embedding = normalize_embedding(vector.tolist())
        row.target_embedding = normalize_embedding(target_summary) if target_summary is not None else None
        row.metadata.update(
            {
                "dataset_hash": row.dataset_id,
                "split": split_by_dataset.get(row.dataset_id, "train"),
                "vs_path": str(sidecar),
                "train_csv": str(train_csv),
                "test_csv": str(test_csv) if test_csv.exists() else None,
                "train_target_vs": str(train_target_vs) if train_target_vs.exists() else None,
                "test_target_vs": str(test_target_vs) if test_target_vs.exists() else None,
                "x128_dim": VS_DIM,
                "x128_signature_dim": DATASET_SIGNATURE_DIM,
                "x128_signature_stats": ["mean", "std", "p10", "p90"],
                "x128_rows": vs_row_counts[row.dataset_id],
                "target_encoder": target_encoder,
                "target_embedding_dim": 0 if target_summary is None else len(target_summary),
                "target_sidecar_used": str(train_target_vs) if target_encoder == "train_target_vs" and train_target_vs.exists() else None,
                "actions": actions,
            }
        )
        row.metric = row.metric or "cv.oof_r2"

    _assign_relative_labels(rows)
    assign_ranks(rows)
    assign_elo(rows)
    vocab = build_active_vocab(row.canonical_genome or row.genome for row in rows)
    vocab["syntax"] = "active_jain_regressor_v2"
    standardization = {
        "dim": DATASET_SIGNATURE_DIM,
        "fit_split": "train",
        "train_dataset_count": train_count,
        "signature_stats": ["mean", "std", "p10", "p90"],
        "mean": mean.astype(float).tolist(),
        "scale": scale.astype(float).tolist(),
    }
    split_manifest = _build_split_manifest(split_by_dataset, sidecars, rows)
    label_summary = _build_label_summary(rows)
    manifest = {
        "version": "active_v2",
        "quant_root": str(root),
        "split_json": str(split_json) if split_json else None,
        "datasets_with_sidecar": len(sidecars),
        "rows": len(rows),
        "splits": dict(sorted(_count_splits(split_by_dataset).items())),
        "active_max_transforms": ACTIVE_MAX_TRANSFORMS,
        "dataset_signature_dim": DATASET_SIGNATURE_DIM,
        "target_summary_dim": TARGET_SUMMARY_DIM,
        "target_encoder": target_encoder,
        "target_source_root": str(root),
        "target_sidecar": ACTIVE_TRAIN_TARGET_SIDECAR if target_encoder == "train_target_vs" else None,
        "heldout_target_sidecar_locked": ACTIVE_TEST_TARGET_SIDECAR if target_encoder == "train_target_vs" else None,
        "split_manifest": split_manifest,
        "label_summary": label_summary,
    }
    build = ActiveBuild(rows=rows, vocab=vocab, standardization=standardization, manifest=manifest)
    if out_dir is not None:
        write_active_v2_artifacts(out_dir, build)
    if out_jsonl is not None:
        write_jsonl(out_jsonl, rows)
    return build


def discover_train_sidecars(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    candidates = {
        *root.glob(f"*/{ACTIVE_TRAIN_SIDECAR}"),
        *root.glob(f"regression/*/{ACTIVE_TRAIN_SIDECAR}"),
    }
    return {path.parent.name: path for path in sorted(candidates)}


def write_active_artifacts(out_dir: str | Path, build: ActiveBuild) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "active_rows.jsonl", build.rows)
    (out / "active_vocab.json").write_text(json.dumps(build.vocab, indent=2, sort_keys=True), encoding="utf-8")
    (out / "standardization.json").write_text(
        json.dumps(build.standardization, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out / "manifest.json").write_text(json.dumps(build.manifest, indent=2, sort_keys=True), encoding="utf-8")
    np.savez_compressed(
        out / "features.npz",
        dataset_hash=np.asarray([row.dataset_id for row in build.rows]),
        x128=np.asarray([row.embedding[:VS_DIM] for row in build.rows], dtype=np.float32),
        target=np.asarray([row.embedding[VS_DIM : VS_DIM + 64] for row in build.rows], dtype=np.float32),
        score=np.asarray([np.nan if row.score is None else row.score for row in build.rows], dtype=np.float32),
        rank=np.asarray([-1 if row.rank is None else row.rank for row in build.rows], dtype=np.int32),
        split=np.asarray([row.metadata.get("split", "train") for row in build.rows]),
    )


def write_active_v2_artifacts(out_dir: str | Path, build: ActiveBuild) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "active_rows.jsonl", build.rows)
    (out / "active_vocab.json").write_text(json.dumps(build.vocab, indent=2, sort_keys=True), encoding="utf-8")
    (out / "vocab.json").write_text(json.dumps(build.vocab, indent=2, sort_keys=True), encoding="utf-8")
    (out / "standardization.json").write_text(
        json.dumps(build.standardization, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out / "manifest.json").write_text(json.dumps(build.manifest, indent=2, sort_keys=True), encoding="utf-8")
    (out / "split_manifest.json").write_text(
        json.dumps(build.manifest.get("split_manifest", {}), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out / "label_summary.json").write_text(
        json.dumps(build.manifest.get("label_summary", {}), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    np.savez_compressed(
        out / "features.npz",
        dataset_hash=np.asarray([row.dataset_id for row in build.rows]),
        canonical_genome=np.asarray([row.canonical_genome or row.genome for row in build.rows]),
        signature=np.asarray([row.embedding[:DATASET_SIGNATURE_DIM] for row in build.rows], dtype=np.float32),
        target_summary=np.asarray([_target_summary(row) for row in build.rows], dtype=np.float32),
        score=np.asarray([np.nan if row.score is None else row.score for row in build.rows], dtype=np.float32),
        oof_delta_to_incumbent=np.asarray(
            [_float_or_nan(row.metadata.get("oof_delta_to_incumbent")) for row in build.rows],
            dtype=np.float32,
        ),
        rank_percentile=np.asarray(
            [_float_or_nan(row.metadata.get("rank_percentile")) for row in build.rows],
            dtype=np.float32,
        ),
        candidate_rank=np.asarray(
            [
                -1 if row.metadata.get("candidate_rank") is None else int(row.metadata["candidate_rank"])
                for row in build.rows
            ],
            dtype=np.int32,
        ),
        split=np.asarray([row.metadata.get("split", "train") for row in build.rows]),
        train_csv=np.asarray([row.metadata.get("train_csv", "") for row in build.rows]),
        test_csv=np.asarray([row.metadata.get("test_csv") or "" for row in build.rows]),
        train_target_vs=np.asarray([row.metadata.get("train_target_vs") or "" for row in build.rows]),
        test_target_vs=np.asarray([row.metadata.get("test_target_vs") or "" for row in build.rows]),
    )


def _build_target_embeddings(
    sidecars: dict[str, Path],
    target_encoder: str,
    target_model: str | Path | None,
) -> dict[str, list[float]]:
    if target_encoder == "none":
        return {}
    embeddings: dict[str, list[float]] = {}
    for dataset_hash, sidecar in sidecars.items():
        train_csv = sidecar.parent / "train.csv"
        train_target_vs = sidecar.parent / ACTIVE_TRAIN_TARGET_SIDECAR
        if target_encoder == "train_target_vs" and not train_target_vs.exists():
            continue
        embeddings[dataset_hash] = encode_target_embedding(
            train_csv,
            encoder=target_encoder,  # type: ignore[arg-type]
            model_path=target_model,
            train_target_vs=train_target_vs,
        )
    return embeddings


def build_active_vocab(genomes: Iterable[str]) -> dict[str, Any]:
    transforms: dict[str, dict[str, set[str]]] = {}
    heads: dict[str, dict[str, set[str]]] = {}
    transform_counts: set[int] = set()
    for genome in genomes:
        try:
            actions = parse_pipeline_actions(genome)
        except ValueError:
            continue
        transform_counts.add(int(actions["transform_count"]))
        for stage in actions["transforms"]:
            op = stage["op"]
            params = transforms.setdefault(op, {})
            for key, value in stage.get("params", {}).items():
                params.setdefault(key, set()).add(_param_token(value))
        head = actions["head"]
        params = heads.setdefault(head["op"], {})
        for key, value in head.get("params", {}).items():
            params.setdefault(key, set()).add(_param_token(value))

    return {
        "syntax": "active_jain_regressor_v1",
        "max_transforms": ACTIVE_MAX_TRANSFORMS,
        "transform_counts": sorted(transform_counts),
        "transforms": _freeze_vocab(transforms),
        "heads": _freeze_vocab(heads),
    }


def parse_pipeline_actions(genome: str) -> dict[str, Any]:
    stages = _split_pipeline_stages(genome)
    if not stages:
        raise ValueError("pipeline contains no stages")
    parsed = [_parse_stage(stage) for stage in stages]
    head = parsed[-1]
    transforms = parsed[:-1]
    if head["op"] not in HEADS:
        raise ValueError(f"{head['op']}: inactive head")
    inactive = [stage["op"] for stage in transforms if stage["op"] not in TRANSFORMS]
    if inactive:
        raise ValueError(f"inactive transforms: {inactive}")
    if len(transforms) > ACTIVE_MAX_TRANSFORMS:
        raise ValueError(f"too many transforms: {len(transforms)}")
    return {
        "transform_count": len(transforms),
        "transforms": transforms,
        "head": head,
    }


def format_pipeline_actions(actions: dict[str, Any]) -> str:
    stages = list(actions.get("transforms") or []) + [actions["head"]]
    return "(pipe X " + " ".join(_format_stage(stage) for stage in stages) + ")"


def parse_param_token(token: str) -> Any:
    return _parse_param(token)


def format_param_token(value: Any) -> str:
    return _format_param(value)


def _canonicalize_rows(rows: list[TrainingRow], *, batch_size: int = 512) -> None:
    validator = RustValidator()
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        outputs = validator.validate_many([row.genome for row in batch])
        for row, meta in zip(batch, outputs, strict=True):
            row.valid = bool(meta.get("valid"))
            row.canonical_genome = meta.get("canonical_genome")
            row.family_bitmap = dict(meta.get("family_bitmap") or {})
            row.complexity = int(meta.get("stage_count") or row.complexity or 0)
            row.metadata["validation"] = meta


def _active_canonical(row: TrainingRow) -> bool:
    try:
        parse_pipeline_actions(row.canonical_genome or row.genome)
    except ValueError:
        return False
    return True


def _assign_relative_labels(rows: list[TrainingRow]) -> None:
    best_by_dataset: dict[str, float] = {}
    for row in rows:
        if row.score is None:
            continue
        best_by_dataset[row.dataset_id] = max(best_by_dataset.get(row.dataset_id, float("-inf")), row.score)
    grouped: dict[str, list[TrainingRow]] = defaultdict(list)
    for row in rows:
        grouped[row.dataset_id].append(row)
    for dataset_id, group in grouped.items():
        scored = [row for row in group if row.score is not None]
        scored.sort(key=lambda row: row.score if row.score is not None else float("-inf"), reverse=True)
        n = max(1, len(scored) - 1)
        for rank, row in enumerate(scored, start=1):
            incumbent = best_by_dataset.get(dataset_id)
            row.metadata["candidate_rank"] = rank
            row.metadata["rank_percentile"] = 0.0 if len(scored) == 1 else (rank - 1) / n
            row.metadata["best_known_oof_r2"] = incumbent
            row.metadata["oof_delta_to_incumbent"] = None if incumbent is None else row.score - incumbent


def _dedupe_active_rows(rows: list[TrainingRow]) -> list[TrainingRow]:
    best: dict[tuple[str, str], TrainingRow] = {}
    for row in rows:
        try:
            actions = parse_pipeline_actions(row.canonical_genome or row.genome)
        except ValueError:
            continue
        canonical = format_pipeline_actions(actions)
        row.canonical_genome = canonical
        key = (row.dataset_id, canonical)
        previous = best.get(key)
        if previous is None or _score_sort_key(row) > _score_sort_key(previous):
            best[key] = row
    return sorted(best.values(), key=lambda row: (row.dataset_id, row.canonical_genome or row.genome))


def _score_sort_key(row: TrainingRow) -> tuple[int, float]:
    if row.score is None:
        return (0, float("-inf"))
    return (1, float(row.score))


def _build_split_manifest(
    split_by_dataset: dict[str, str],
    sidecars: dict[str, Path],
    rows: list[TrainingRow],
) -> dict[str, Any]:
    datasets: dict[str, dict[str, Any]] = {}
    for dataset_hash, sidecar in sorted(sidecars.items()):
        train_csv = sidecar.parent / "train.csv"
        test_csv = sidecar.parent / "test.csv"
        train_target_vs = sidecar.parent / ACTIVE_TRAIN_TARGET_SIDECAR
        test_target_vs = sidecar.parent / ACTIVE_TEST_TARGET_SIDECAR
        datasets[dataset_hash] = {
            "split": split_by_dataset.get(dataset_hash, "train"),
            "vs_path": str(sidecar),
            "train_csv": str(train_csv),
            "test_csv": str(test_csv) if test_csv.exists() else None,
            "train_target_vs": str(train_target_vs) if train_target_vs.exists() else None,
            "test_target_vs": str(test_target_vs) if test_target_vs.exists() else None,
            "candidate_rows": 0,
            "best_known_oof_r2": None,
        }
    for row in rows:
        entry = datasets.setdefault(
            row.dataset_id,
            {
                "split": row.metadata.get("split", "train"),
                "vs_path": row.metadata.get("vs_path"),
                "train_csv": row.metadata.get("train_csv"),
                "test_csv": row.metadata.get("test_csv"),
                "train_target_vs": row.metadata.get("train_target_vs"),
                "test_target_vs": row.metadata.get("test_target_vs"),
                "candidate_rows": 0,
                "best_known_oof_r2": None,
            },
        )
        entry["candidate_rows"] = int(entry.get("candidate_rows") or 0) + 1
        if row.score is not None:
            current = entry.get("best_known_oof_r2")
            entry["best_known_oof_r2"] = float(row.score) if current is None else max(float(current), float(row.score))
    return {
        "datasets": datasets,
        "counts": dict(sorted(_count_splits(split_by_dataset).items())),
    }


def _build_label_summary(rows: list[TrainingRow]) -> dict[str, Any]:
    heads: dict[str, int] = defaultdict(int)
    transform_counts: dict[str, int] = defaultdict(int)
    split_counts: dict[str, int] = defaultdict(int)
    deltas: list[float] = []
    rank_percentiles: list[float] = []
    for row in rows:
        split_counts[str(row.metadata.get("split", "train"))] += 1
        actions = row.metadata.get("actions")
        if not isinstance(actions, dict):
            try:
                actions = parse_pipeline_actions(row.canonical_genome or row.genome)
            except ValueError:
                actions = {}
        head = (actions.get("head") or {}).get("op") if isinstance(actions, dict) else None
        if head:
            heads[str(head)] += 1
        count = actions.get("transform_count") if isinstance(actions, dict) else None
        if isinstance(count, int):
            transform_counts[str(count)] += 1
        delta = _float_or_nan(row.metadata.get("oof_delta_to_incumbent"))
        if math.isfinite(delta):
            deltas.append(delta)
        percentile = _float_or_nan(row.metadata.get("rank_percentile"))
        if math.isfinite(percentile):
            rank_percentiles.append(percentile)
    return {
        "rows": len(rows),
        "datasets": len({row.dataset_id for row in rows}),
        "splits": dict(sorted(split_counts.items())),
        "heads": dict(sorted(heads.items())),
        "transform_counts": dict(sorted(transform_counts.items(), key=lambda item: int(item[0]))),
        "top_candidate_rows": sum(1 for row in rows if row.metadata.get("candidate_rank") == 1),
        "delta_mean": _mean_or_none(deltas),
        "delta_median": _median_or_none(deltas),
        "rank_percentile_mean": _mean_or_none(rank_percentiles),
    }


def _target_summary(row: TrainingRow) -> list[float]:
    values = row.target_embedding[:TARGET_SUMMARY_DIM] if row.target_embedding is not None else []
    out = [float(value) for value in values]
    if len(out) < TARGET_SUMMARY_DIM:
        out.extend([0.0] * (TARGET_SUMMARY_DIM - len(out)))
    return out[:TARGET_SUMMARY_DIM]


def _float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _mean_or_none(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _median_or_none(values: list[float]) -> float | None:
    return None if not values else float(np.median(values))


def _assign_splits(
    dataset_hashes: list[str],
    split_json: str | Path | None,
    validation_fraction: float,
    seed: int,
) -> dict[str, str]:
    explicit = _load_explicit_split(split_json)
    if explicit:
        mapping = {dataset_hash: "train" for dataset_hash in dataset_hashes}
        for key in explicit.get("real_train", set()):
            if key in mapping:
                mapping[key] = "train"
        for key in explicit.get("validation", set()):
            if key in mapping:
                mapping[key] = "validation"
        for key in explicit.get("real_test", set()):
            if key in mapping:
                mapping[key] = "test"
        if not explicit.get("validation"):
            _carve_validation(mapping, validation_fraction, seed)
        return mapping

    mapping = {dataset_hash: "train" for dataset_hash in dataset_hashes}
    _carve_validation(mapping, validation_fraction, seed)
    return mapping


def _load_explicit_split(split_json: str | Path | None) -> dict[str, set[str]]:
    if split_json is None:
        return {}
    path = Path(split_json)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, set[str]] = {}
    aliases = {
        "real_train": ("real_train", "train"),
        "validation": ("validation", "valid", "val"),
        "real_test": ("real_test", "test"),
    }
    for canonical, keys in aliases.items():
        values: set[str] = set()
        for key in keys:
            raw = data.get(key)
            if isinstance(raw, list):
                values.update(Path(str(item)).name for item in raw)
            elif isinstance(raw, dict):
                values.update(Path(str(item)).name for item in raw.values())
        if values:
            out[canonical] = values
    return out


def _carve_validation(mapping: dict[str, str], validation_fraction: float, seed: int) -> None:
    train_hashes = [key for key, split in mapping.items() if split == "train"]
    if len(train_hashes) < 2:
        return
    rng = random.Random(seed)
    shuffled = train_hashes[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * max(0.0, min(validation_fraction, 0.5)))))
    for key in shuffled[:n_val]:
        mapping[key] = "validation"


def _fit_standardizer(
    raw_vectors: dict[str, np.ndarray],
    split_by_dataset: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, int]:
    train_vectors = [
        raw_vectors[dataset_hash]
        for dataset_hash, split in split_by_dataset.items()
        if split == "train" and dataset_hash in raw_vectors
    ]
    if not train_vectors:
        train_vectors = list(raw_vectors.values())
    matrix = np.asarray(train_vectors, dtype=np.float64)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32), len(train_vectors)


def _count_splits(mapping: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for split in mapping.values():
        counts[split] += 1
    return counts


def _split_pipeline_stages(genome: str) -> list[str]:
    text = genome.strip()
    if not text.startswith("(pipe X") or not text.endswith(")"):
        raise ValueError("expected active pipeline genome")
    body = text[len("(pipe X") : -1].strip()
    stages: list[str] = []
    depth = 0
    start: int | None = None
    in_quote = False
    escaped = False
    for i, ch in enumerate(body):
        if in_quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_quote = False
            continue
        if ch == '"':
            in_quote = True
        elif ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start is not None:
                stages.append(body[start : i + 1])
                start = None
            if depth < 0:
                raise ValueError("unbalanced pipeline")
    if depth != 0 or in_quote:
        raise ValueError("unbalanced pipeline")
    return stages


def _parse_stage(stage: str) -> dict[str, Any]:
    if not stage.startswith("(") or not stage.endswith(")"):
        raise ValueError(f"bad stage: {stage}")
    tokens = _tokenize_stage(stage[1:-1].strip())
    if not tokens:
        raise ValueError("empty stage")
    op = tokens[0]
    params: dict[str, Any] = {}
    for token in tokens[1:]:
        if "=" not in token:
            raise ValueError(f"bad param token: {token}")
        key, raw = token.split("=", 1)
        params[key] = _parse_param(raw)
    return {"op": op, "params": params}


def _tokenize_stage(text: str) -> list[str]:
    tokens: list[str] = []
    buf: list[str] = []
    in_quote = False
    escaped = False
    for ch in text:
        if in_quote:
            buf.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_quote = False
            continue
        if ch == '"':
            in_quote = True
            buf.append(ch)
        elif ch.isspace():
            if buf:
                tokens.append("".join(buf))
                buf = []
        else:
            buf.append(ch)
    if buf:
        tokens.append("".join(buf))
    return tokens


def _parse_param(raw: str) -> Any:
    if raw == "true":
        return True
    if raw == "false":
        return False
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    if re.fullmatch(r"[-+]?\d+", raw):
        return int(raw)
    try:
        value = float(raw)
    except ValueError:
        return raw
    if not math.isfinite(value):
        raise ValueError(f"non-finite param: {raw}")
    return value


def _format_stage(stage: dict[str, Any]) -> str:
    parts = [stage["op"]]
    for key, value in sorted((stage.get("params") or {}).items()):
        parts.append(f"{key}={_format_param(value)}")
    return "(" + " ".join(parts) + ")"


def _format_param(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value - round(value)) < 1e-12:
            return f"{value:.1f}"
        return repr(value)
    text = str(value)
    if re.fullmatch(r"[A-Za-z0-9_+*.-]+", text):
        return text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _freeze_vocab(vocab: dict[str, dict[str, set[str]]]) -> dict[str, dict[str, list[str]]]:
    return {
        op: {key: sorted(values) for key, values in sorted(params.items())}
        for op, params in sorted(vocab.items())
    }


def _param_token(value: Any) -> str:
    return _format_param(value)
