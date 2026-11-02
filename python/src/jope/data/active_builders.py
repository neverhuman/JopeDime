"""Active corpus assembly entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from jope.data.importers import import_quant_reports
from jope.data.labels import assign_elo, assign_ranks
from jope.data.target_encoding import TARGET_VS_EMBED_DIM
from jope.data.vs import VS_DIM, dataset_signature_from_vs, read_vs_matrix
from jope.io import write_jsonl
from jope.schema import normalize_embedding
from .active import (
    ACTIVE_MAX_TRANSFORMS, ACTIVE_TEST_SIDECAR, ACTIVE_TEST_TARGET_SIDECAR,
    ACTIVE_TRAIN_SIDECAR, ACTIVE_TRAIN_TARGET_SIDECAR, DEFAULT_QUANT_ROOT,
    TARGET_SUMMARY_DIM, ActiveBuild, _active_canonical, _assign_relative_labels,
    _assign_splits, _build_label_summary, _build_split_manifest, _build_target_embeddings,
    _canonicalize_rows, _count_splits, _dedupe_active_rows, _fit_standardizer,
    _float_or_nan, _target_summary, build_active_vocab,
    format_pipeline_actions, parse_pipeline_actions,
)

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
                "target_sidecar_used": str(train_target_vs)
                if target_encoder == "train_target_vs" and train_target_vs.exists()
                else None,
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
        "heldout_target_sidecar_locked": ACTIVE_TEST_TARGET_SIDECAR
        if target_encoder == "train_target_vs"
        else None,
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
    (out / "active_vocab.json").write_text(
        json.dumps(build.vocab, indent=2, sort_keys=True), encoding="utf-8"
    )
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
        score=np.asarray(
            [np.nan if row.score is None else row.score for row in build.rows], dtype=np.float32
        ),
        rank=np.asarray([-1 if row.rank is None else row.rank for row in build.rows], dtype=np.int32),
        split=np.asarray([row.metadata.get("split", "train") for row in build.rows]),
    )


def write_active_v2_artifacts(out_dir: str | Path, build: ActiveBuild) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "active_rows.jsonl", build.rows)
    (out / "active_vocab.json").write_text(
        json.dumps(build.vocab, indent=2, sort_keys=True), encoding="utf-8"
    )
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
        score=np.asarray(
            [np.nan if row.score is None else row.score for row in build.rows], dtype=np.float32
        ),
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

