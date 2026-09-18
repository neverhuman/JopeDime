"""active_v3: dual-engine corpus build — historical featmath rows + generated seed rows.

Differences vs v2:
- Every row carries an ``engine``; canonicalization routes through the EngineRouter and
  the BRIDGE canonical is the stored identity (no Python re-formatting).
- Seed generation: per train dataset x engine, M grammar-sampled genomes (cost-capped,
  expr_fraction carrying map/derive symbolic stages) are CV-scored on train.csv ONLY
  (test.csv is sealed for evaluate-active) through the engine-aware ledger.
- Robust labels: score = max(oof_r2, -1); score_robust subtracts a fold-variance
  penalty; ranking pools candidates per dataset ACROSS engines (cross-engine pairs are
  the routing signal) with per-engine auxiliary ranks.
- Dedupe key is (dataset_id, engine, canonical).
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..engines import DEFAULT_ENGINE, ENGINES, EngineRouter
from ..eval_ledger import iter_eval_regression_with_ledger
from ..grammars import grammar_for
from ..grammars.sampler import sample_genome_from_spec
from ..grammars.vocab import build_active_vocab_v3
from ..io import write_jsonl
from ..pipeline_text import parse_pipeline_actions
from ..schema import TrainingRow, normalize_embedding
from .active import (
    ACTIVE_TEST_TARGET_SIDECAR,
    ACTIVE_TRAIN_TARGET_SIDECAR,
    DEFAULT_QUANT_ROOT,
    TARGET_SUMMARY_DIM,
    _assign_splits,
    _build_label_summary,
    _build_split_manifest,
    _build_target_embeddings,
    _count_splits,
    _fit_standardizer,
    _score_sort_key,
    discover_train_sidecars,
    write_active_v2_artifacts,
)
from .importers import import_quant_reports
from .labels import assign_elo, assign_ranks
from .vs import DATASET_SIGNATURE_DIM, VS_DIM, dataset_signature_from_vs, read_vs_matrix

# Fold-variance penalty weight for the robust score (single source of truth).
ROBUST_SCORE_CLIP = -1.0
ROBUST_VARIANCE_LAMBDA = 0.25


def robust_score(oof_r2: float | None, fold_r2: list[float] | None) -> float | None:
    """Clipped, fold-variance-penalized score used for regression ranking labels."""
    if oof_r2 is None or not math.isfinite(float(oof_r2)):
        return None
    clipped = max(float(oof_r2), ROBUST_SCORE_CLIP)
    std = 0.0
    if fold_r2:
        finite = [float(v) for v in fold_r2 if math.isfinite(float(v))]
        if len(finite) >= 2:
            std = float(np.std(finite))
    return clipped - ROBUST_VARIANCE_LAMBDA * min(std, 1.0)


def robust_score_binary(oof_auc: float | None, fold_auc: list[float] | None) -> float | None:
    """Fold-variance-penalized AUC label for binary ranking.

    No lower clip: AUC is already bounded [0,1] and AUC<0.5 is real (inverted) ordering
    signal the policy should learn to avoid, not floor away.
    """
    if oof_auc is None or not math.isfinite(float(oof_auc)):
        return None
    std = 0.0
    if fold_auc:
        finite = [float(v) for v in fold_auc if math.isfinite(float(v))]
        if len(finite) >= 2:
            std = float(np.std(finite))
    return float(oof_auc) - ROBUST_VARIANCE_LAMBDA * min(std, 1.0)


def robust_score_for(task: str, oof: float | None, folds: list[float] | None) -> float | None:
    """Task-dispatched robust label: regression (R²) vs binary (AUC)."""
    if task == "binary":
        return robust_score_binary(oof, folds)
    return robust_score(oof, folds)


@dataclass(frozen=True)
class ActiveBuildV3:
    rows: list[TrainingRow]
    vocab: dict[str, Any]
    standardization: dict[str, Any]
    manifest: dict[str, Any]


def build_active_v3_corpus(
    *,
    quant_root: str | Path = DEFAULT_QUANT_ROOT,
    split_json: str | Path | None = None,
    out_dir: str | Path | None = None,
    out_jsonl: str | Path | None = None,
    limit_datasets: int | None = None,
    validation_fraction: float = 0.1,
    seed: int = 0,
    engines: tuple[str, ...] = ENGINES,
    seed_genomes_per_engine: int = 24,
    expr_fraction: float = 0.2,
    max_seed_cost_class: str = "medium",
    skip_seed_eval: bool = False,
    ledger: str | Path | None = None,
    eval_workers: int | None = None,
    folds: int = 3,
    target_encoder: str = "train_target_vs",
    target_model: str | Path | None = None,
    router: EngineRouter | None = None,
) -> ActiveBuildV3:
    router = router or EngineRouter()
    root = Path(quant_root)
    sidecars = discover_train_sidecars(root)
    if limit_datasets is not None:
        keep = set(sorted(sidecars)[:limit_datasets])
        sidecars = {h: p for h, p in sidecars.items() if h in keep}

    raw_vectors: dict[str, np.ndarray] = {}
    vs_row_counts: dict[str, int] = {}
    for dataset_hash, path in sidecars.items():
        matrix = read_vs_matrix(path)
        raw_vectors[dataset_hash] = dataset_signature_from_vs(path)
        vs_row_counts[dataset_hash] = int(matrix.shape[0])

    target_embeddings = _build_target_embeddings(sidecars, target_encoder, target_model)
    split_by_dataset = _assign_splits(sorted(raw_vectors), split_json, validation_fraction, seed)
    mean, scale, train_count = _fit_standardizer(raw_vectors, split_by_dataset)

    # ---- historical featmath rows -------------------------------------------------
    rows = [row for row in import_quant_reports(root) if row.dataset_id in raw_vectors]
    for row in rows:
        row.engine = DEFAULT_ENGINE
        row.metadata.setdefault("provenance", "historical_import")
    _canonicalize_rows_v3(rows, router)
    rows = [row for row in rows if row.valid and row.canonical_genome]

    # ---- generated seed rows (both engines) ---------------------------------------
    seed_report: dict[str, Any] = {}
    if seed_genomes_per_engine > 0:
        seed_rows, seed_report = _generate_seed_rows(
            rows,
            sidecars,
            split_by_dataset,
            router,
            engines=engines,
            per_engine=seed_genomes_per_engine,
            expr_fraction=expr_fraction,
            max_cost_class=max_seed_cost_class,
            skip_eval=skip_seed_eval,
            ledger=ledger,
            eval_workers=eval_workers,
            folds=folds,
            seed=seed,
        )
        rows.extend(seed_rows)

    rows = _dedupe_active_rows_v3(rows)

    # ---- per-row features + metadata ----------------------------------------------
    for row in rows:
        spec = grammar_for(row.engine)
        canonical = row.canonical_genome or row.genome
        actions = parse_pipeline_actions(canonical, spec)
        vector = (raw_vectors[row.dataset_id] - mean) / scale
        target_embedding = target_embeddings.get(row.dataset_id)
        target_summary = None if target_embedding is None else target_embedding[:TARGET_SUMMARY_DIM]
        sidecar = sidecars[row.dataset_id]
        train_csv = sidecar.parent / "train.csv"
        test_csv = sidecar.parent / "test.csv"
        train_target_vs = sidecar.parent / ACTIVE_TRAIN_TARGET_SIDECAR
        test_target_vs = sidecar.parent / ACTIVE_TEST_TARGET_SIDECAR
        row.embedding = normalize_embedding(vector.tolist())
        row.target_embedding = normalize_embedding(target_summary) if target_summary is not None else None
        row.metadata.update(
            {
                "engine": row.engine,
                "dataset_hash": row.dataset_id,
                "split": split_by_dataset.get(row.dataset_id, "train"),
                "vs_path": str(sidecar),
                "train_csv": str(train_csv),
                "test_csv": str(test_csv) if test_csv.exists() else None,
                "train_target_vs": str(train_target_vs) if train_target_vs.exists() else None,
                "test_target_vs": str(test_target_vs) if test_target_vs.exists() else None,
                "x128_dim": VS_DIM,
                "x128_signature_dim": DATASET_SIGNATURE_DIM,
                "x128_rows": vs_row_counts[row.dataset_id],
                "target_encoder": target_encoder,
                "target_embedding_dim": 0 if target_summary is None else len(target_summary),
                "actions": actions,
            }
        )
        row.metric = row.metric or "cv.oof_r2"

    _assign_relative_labels_v3(rows)
    assign_ranks(rows)
    authority_rows = [row for row in rows if row.metadata.get("elo_authority", True)]
    assign_elo(authority_rows)
    assign_ranks(rows)

    specs = [grammar_for(engine) for engine in engines]
    vocab = build_active_vocab_v3(rows, specs)
    standardization = {
        "dim": DATASET_SIGNATURE_DIM,
        "fit_split": "train",
        "train_dataset_count": train_count,
        "signature_stats": ["mean", "std", "p10", "p90"],
        "mean": mean.astype(float).tolist(),
        "scale": scale.astype(float).tolist(),
    }
    split_manifest = _build_split_manifest(split_by_dataset, sidecars, rows)
    _add_engine_incumbents(split_manifest, rows)
    label_summary = _build_label_summary(rows)
    per_engine_rows = defaultdict(int)
    for row in rows:
        per_engine_rows[row.engine] += 1
    manifest = {
        "version": "active_v3",
        "quant_root": str(root),
        "split_json": str(split_json) if split_json else None,
        "datasets_with_sidecar": len(sidecars),
        "rows": len(rows),
        "rows_per_engine": dict(sorted(per_engine_rows.items())),
        "engines": list(engines),
        "grammar_versions": {spec.engine: spec.version for spec in specs},
        "evaluator_versions": {spec.engine: spec.evaluator_version for spec in specs},
        "splits": dict(sorted(_count_splits(split_by_dataset).items())),
        "robust_score": {"clip": ROBUST_SCORE_CLIP, "variance_lambda": ROBUST_VARIANCE_LAMBDA},
        "seed_generation": seed_report,
        "target_encoder": target_encoder,
        "target_sidecar": ACTIVE_TRAIN_TARGET_SIDECAR if target_encoder == "train_target_vs" else None,
        "split_manifest": split_manifest,
        "label_summary": label_summary,
    }
    build = ActiveBuildV3(rows=rows, vocab=vocab, standardization=standardization, manifest=manifest)
    if out_dir is not None:
        write_active_v3_artifacts(out_dir, build)
    if out_jsonl is not None:
        write_jsonl(out_jsonl, rows)
    return build


def _canonicalize_rows_v3(rows: list[TrainingRow], router: EngineRouter, *, batch_size: int = 512) -> None:
    """Canonicalize via the engine's bridge; the bridge canonical is the identity."""
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        outputs = router.validate_many([(row.engine, row.genome) for row in batch])
        for row, meta in zip(batch, outputs, strict=True):
            row.valid = bool(meta.get("valid"))
            row.canonical_genome = meta.get("canonical_genome")
            row.family_bitmap = dict(meta.get("family_bitmap") or {})
            row.complexity = int(meta.get("stage_count") or row.complexity or 0)
            row.metadata["validation"] = {
                k: meta.get(k)
                for k in ("valid", "engine", "head_type", "stage_count", "transform_count", "errors")
            }


def _generate_seed_rows(
    historical: list[TrainingRow],
    sidecars: dict[str, Path],
    split_by_dataset: dict[str, str],
    router: EngineRouter,
    *,
    engines: tuple[str, ...],
    per_engine: int,
    expr_fraction: float,
    max_cost_class: str,
    skip_eval: bool,
    ledger: str | Path | None,
    eval_workers: int | None,
    folds: int,
    seed: int,
) -> tuple[list[TrainingRow], dict[str, Any]]:
    known: set[tuple[str, str, str]] = {
        (row.dataset_id, row.engine, row.canonical_genome or row.genome) for row in historical
    }
    train_datasets = [h for h in sorted(sidecars) if split_by_dataset.get(h, "train") == "train"]
    rng = random.Random(seed)
    seed_rows: list[TrainingRow] = []
    report: dict[str, Any] = {"per_engine": {}, "train_datasets": len(train_datasets)}

    for engine in engines:
        spec = grammar_for(engine)
        candidates: list[TrainingRow] = []
        for dataset_hash in train_datasets:
            genomes: list[str] = []
            seen: set[str] = set()
            attempts = 0
            while len(genomes) < per_engine * 2 and attempts < per_engine * 12:
                attempts += 1
                genome = sample_genome_from_spec(
                    spec, rng, expr_fraction=expr_fraction, max_cost_class=max_cost_class
                )
                if genome not in seen:
                    seen.add(genome)
                    genomes.append(genome)
            outputs = router.validate_many([(engine, g) for g in genomes])
            kept = 0
            for meta in outputs:
                if kept >= per_engine:
                    break
                if not meta.get("valid"):
                    continue
                canonical = meta.get("canonical_genome")
                key = (dataset_hash, engine, canonical)
                if not canonical or key in known:
                    continue
                known.add(key)
                kept += 1
                candidates.append(
                    TrainingRow(
                        dataset_id=dataset_hash,
                        source="seed_sample",
                        embedding=[0.0],
                        genome=canonical,
                        canonical_genome=canonical,
                        valid=True,
                        engine=engine,
                        family_bitmap=dict(meta.get("family_bitmap") or {}),
                        complexity=int(meta.get("stage_count") or 0),
                        metadata={
                            "provenance": "seed",
                            "evidence_weight": 1.0,
                            "canonical_re_evaluation": True,
                            "elo_authority": True,
                        },
                    )
                )
        report["per_engine"][engine] = {"candidates": len(candidates)}
        if not skip_eval and candidates:
            eval_rows = [
                {
                    "engine": engine,
                    "dataset_hash": row.dataset_id,
                    "train_csv": str(sidecars[row.dataset_id].parent / "train.csv"),
                    "genome": row.canonical_genome,
                    "folds": folds,
                }
                for row in candidates
            ]
            scored = timeouts = failed = cache_hits = 0
            for idx, result in iter_eval_regression_with_ledger(
                eval_rows, ledger=ledger, router=router, workers=eval_workers
            ):
                row = candidates[idx]
                cv = result.get("cv") if isinstance(result.get("cv"), dict) else {}
                oof = cv.get("oof_r2")
                fold_r2 = cv.get("fold_r2") if isinstance(cv.get("fold_r2"), list) else None
                if result.get("cache_hit"):
                    cache_hits += 1
                if isinstance(oof, (int, float)) and math.isfinite(float(oof)):
                    row.score = max(float(oof), ROBUST_SCORE_CLIP)
                    row.metadata["oof_r2_raw"] = float(oof)
                    row.metadata["fold_r2_std"] = (
                        float(np.std([v for v in fold_r2 if math.isfinite(float(v))]))
                        if fold_r2 and len(fold_r2) >= 2
                        else 0.0
                    )
                    row.metadata["score_robust"] = robust_score(float(oof), fold_r2)
                    row.cost_ms = (result.get("timings") or {}).get("total_ms")
                    scored += 1
                else:
                    row.score = None
                    row.metadata["eval_error"] = (result.get("errors") or ["unknown"])[0]
                    if result.get("timeout"):
                        timeouts += 1
                    failed += 1
            report["per_engine"][engine].update(
                {"scored": scored, "failed": failed, "timeouts": timeouts, "cache_hits": cache_hits}
            )
        seed_rows.extend(candidates)
    return seed_rows, report


def _dedupe_active_rows_v3(rows: list[TrainingRow]) -> list[TrainingRow]:
    best: dict[tuple[str, str, str], TrainingRow] = {}
    for row in rows:
        canonical = row.canonical_genome or row.genome
        key = (row.dataset_id, row.engine, canonical)
        previous = best.get(key)
        if previous is None or _score_sort_key(row) > _score_sort_key(previous):
            best[key] = row
    return sorted(best.values(), key=lambda r: (r.dataset_id, r.engine, r.canonical_genome or r.genome))


def _row_robust(row: TrainingRow) -> float | None:
    value = row.metadata.get("score_robust")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if row.score is None:
        return None
    return max(float(row.score), ROBUST_SCORE_CLIP)


def _assign_relative_labels_v3(rows: list[TrainingRow]) -> None:
    """Pooled per-dataset ranking across engines (routing signal) + per-engine ranks."""
    grouped: dict[str, list[TrainingRow]] = defaultdict(list)
    for row in rows:
        grouped[row.dataset_id].append(row)
    for dataset_id, group in grouped.items():
        scored = [(row, _row_robust(row)) for row in group]
        scored = [(row, value) for row, value in scored if value is not None]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        incumbent = scored[0][1] if scored else None
        n = max(1, len(scored) - 1)
        for rank, (row, value) in enumerate(scored, start=1):
            row.metadata["candidate_rank"] = rank
            row.metadata["rank_percentile"] = 0.0 if len(scored) == 1 else (rank - 1) / n
            row.metadata["best_known_robust"] = incumbent
            row.metadata["best_known_oof_r2"] = incumbent
            row.metadata["oof_delta_to_incumbent"] = None if incumbent is None else value - incumbent
        # per-engine auxiliary ranks (for ablations)
        by_engine: dict[str, list[tuple[TrainingRow, float]]] = defaultdict(list)
        for row, value in scored:
            by_engine[row.engine].append((row, value))
        for engine, pairs in by_engine.items():
            pairs.sort(key=lambda pair: pair[1], reverse=True)
            n_e = max(1, len(pairs) - 1)
            for rank, (row, _) in enumerate(pairs, start=1):
                row.metadata["candidate_rank_engine"] = rank
                row.metadata["rank_percentile_engine"] = 0.0 if len(pairs) == 1 else (rank - 1) / n_e


def _add_engine_incumbents(split_manifest: dict[str, Any], rows: list[TrainingRow]) -> None:
    best: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        value = _row_robust(row)
        if value is None:
            continue
        engine_best = best[row.dataset_id]
        if value > engine_best.get(row.engine, float("-inf")):
            engine_best[row.engine] = value
    datasets = split_manifest.get("datasets")
    if isinstance(datasets, dict):
        for dataset_hash, entry in datasets.items():
            if isinstance(entry, dict) and dataset_hash in best:
                entry["best_known_oof_r2_by_engine"] = dict(sorted(best[dataset_hash].items()))


def write_active_v3_artifacts(out_dir: str | Path, build: ActiveBuildV3) -> None:
    from .active import ActiveBuild

    # Reuse the v2 writer for the common files, then extend features.npz with v3 columns.
    write_active_v2_artifacts(
        out_dir,
        ActiveBuild(
            rows=build.rows,
            vocab=build.vocab,
            standardization=build.standardization,
            manifest=build.manifest,
        ),
    )
    out = Path(out_dir)
    with np.load(out / "features.npz", allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["engine"] = np.asarray([row.engine for row in build.rows])
    arrays["provenance"] = np.asarray([str(row.metadata.get("provenance") or "") for row in build.rows])
    arrays["score_robust"] = np.asarray(
        [np.nan if _row_robust(row) is None else _row_robust(row) for row in build.rows],
        dtype=np.float32,
    )
    arrays["fold_r2_std"] = np.asarray(
        [float(row.metadata.get("fold_r2_std") or 0.0) for row in build.rows], dtype=np.float32
    )
    np.savez_compressed(out / "features.npz", **arrays)
