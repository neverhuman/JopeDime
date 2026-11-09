from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jope.data.importers import import_quant_reports as import_quant_reports
from jope.data.labels import assign_elo as assign_elo
from jope.data.labels import assign_ranks as assign_ranks
from jope.data.target_encoding import TARGET_VS_EMBED_DIM, encode_target_embedding
from jope.data.vs import DATASET_SIGNATURE_DIM as DATASET_SIGNATURE_DIM
from jope.data.vs import VS_DIM as VS_DIM
from jope.data.vs import dataset_signature_from_vs as dataset_signature_from_vs
from jope.data.vs import read_vs_matrix as read_vs_matrix
from jope.grammar import HEADS as HEADS
from jope.grammar import TRANSFORMS as TRANSFORMS
from jope.io import write_jsonl as write_jsonl
from jope.prime_config import load_config
from jope.schema import TrainingRow
from jope.schema import normalize_embedding as normalize_embedding
from jope.validator import RustValidator

from .active_grammar import (
    ACTIVE_MAX_TRANSFORMS as ACTIVE_MAX_TRANSFORMS,
)
from .active_grammar import (
    _format_param as _format_param,
)
from .active_grammar import (
    _freeze_vocab,
    _param_token,
    format_pipeline_actions,
    parse_pipeline_actions,
)
from .active_grammar import (
    _parse_param as _parse_param,
)
from .active_grammar import (
    _parse_stage as _parse_stage,
)
from .active_grammar import (
    _split_pipeline_stages as _split_pipeline_stages,
)
from .active_grammar import (
    _tokenize_stage as _tokenize_stage,
)
from .active_grammar import (
    format_param_token as format_param_token,
)
from .active_grammar import (
    parse_param_token as parse_param_token,
)

ACTIVE_TRAIN_SIDECAR = "hyperion3train128dim.vs"
ACTIVE_TEST_SIDECAR = "hyperion3test128dim.vs"
ACTIVE_TRAIN_TARGET_SIDECAR = "train_target.vs"
ACTIVE_TEST_TARGET_SIDECAR = "test_target.vs"
TARGET_SUMMARY_DIM = TARGET_VS_EMBED_DIM
DEFAULT_QUANT_ROOT = load_config().regression_dataset_root


@dataclass(frozen=True)
class ActiveBuild:
    rows: list[TrainingRow]
    vocab: dict[str, Any]
    standardization: dict[str, Any]
    manifest: dict[str, Any]


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
            entry["best_known_oof_r2"] = (
                float(row.score) if current is None else max(float(current), float(row.score))
            )
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
    from .active_builders import build_active_corpus as build

    return build(
        quant_root=quant_root,
        split_json=split_json,
        out_dir=out_dir,
        out_jsonl=out_jsonl,
        limit_datasets=limit_datasets,
        validation_fraction=validation_fraction,
        seed=seed,
        validate=validate,
        target_encoder=target_encoder,
        target_model=target_model,
    )


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
    from .active_builders import build_active_v2_corpus as build

    return build(
        quant_root=quant_root,
        split_json=split_json,
        out_dir=out_dir,
        out_jsonl=out_jsonl,
        limit_datasets=limit_datasets,
        validation_fraction=validation_fraction,
        seed=seed,
        validate=validate,
        target_encoder=target_encoder,
        target_model=target_model,
    )
