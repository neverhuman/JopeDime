"""Build a UNIFIED regression+binary corpus for the fused policy.

Reuses an existing (regression) v3 corpus verbatim — those rows are already CV-scored —
and appends freshly generated + eval-binary-scored rows for binary datasets that have
Hyperion .vs signatures. Every row is tagged ``metadata.task_type`` so the fused encoder
(TASK token) can condition on it, and so the corpus can be trivially split back into two
task-specific corpora if the fused model ever underperforms.

Binary embeddings are standardized with the SAME standardizer as the regression corpus, so
the fused model sees one consistent signature space. Binary labels use ``robust_score_binary``
(AUC − variance penalty, no lower clip). Each binary target embedding is bound to one of
five complementary views and is computed only from that view's training rows.
"""

from __future__ import annotations

import json
import math
import random
import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from ..engines import EngineRouter
from ..contracts import (
    TaskType,
    build_binary_target_summaries,
    build_split_view_manifest,
    freeze_source_family_split,
)
from ..eval_ledger import iter_eval_regression_with_ledger
from ..grammars import grammar_for
from ..grammars.sampler import sample_genome_from_spec
from ..io import read_jsonl, write_jsonl
from ..pipeline_text import parse_pipeline_actions
from ..schema import TrainingRow, normalize_embedding
from .active_v3 import assign_elo, assign_ranks, robust_score_binary
from .vs import DATASET_SIGNATURE_DIM, dataset_signature_from_vs, read_vs_matrix

ROW_SIDECAR = "hyperion3train128dim.vs"


def _standardize(sig: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> list[float]:
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    return normalize_embedding(((sig - mean) / scale).tolist())


def build_fused_corpus(
    *,
    base_corpus_dir: str | Path,
    binary_root: str | Path,
    out_dir: str | Path,
    max_binary_datasets: int = 1_244,
    binary_val_fraction: float = 0.20,
    seed_genomes_per_engine: int = 8,
    engines: tuple[str, ...] = ("featmath", "newmath"),
    max_cost_class: str = "medium",
    folds: int = 3,
    ledger: str | Path | None = None,
    eval_workers: int | None = None,
    seed: int = 0,
    router: EngineRouter | None = None,
) -> dict[str, Any]:
    router = router or EngineRouter()
    base = Path(base_corpus_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    standardization = json.loads(
        (base / "standardization.json").read_text(encoding="utf-8")
    )
    mean = np.asarray(standardization["mean"], dtype=np.float64)
    scale = np.asarray(standardization["scale"], dtype=np.float64)

    # ---- 1) reuse the regression rows verbatim, tag task_type ----------------------
    reg_rows = [
        TrainingRow.from_json(r) for r in read_jsonl(base / "active_rows.jsonl")
    ]
    for row in reg_rows:
        row.metadata.setdefault("task_type", "regression")

    # ---- 2) discover binary datasets that already have a .vs signature -------------
    binary_root = Path(binary_root)
    candidates = [
        d
        for d in sorted(binary_root.iterdir())
        if d.is_dir()
        and not d.name.startswith(".")
        and (d / ROW_SIDECAR).exists()
        and (d / "train.csv").exists()
    ][:max_binary_datasets]
    rng = random.Random(seed)

    # Admission precedes the frozen family split so reported train/validation counts
    # describe datasets that actually have a sound signature, train rows, and all five
    # strict train-only binary summaries.
    admitted: dict[str, dict[str, Any]] = {}
    for d in candidates:
        try:
            targets, row_ids = _training_targets(d / "train.csv")
            rows_n = int(read_vs_matrix(d / ROW_SIDECAR).shape[0])
            if rows_n != len(targets):
                continue
            sig = dataset_signature_from_vs(d / ROW_SIDECAR)
            standardized_signature = _standardize(sig, mean, scale)
            source_family = _source_family(d)
            train_sha256 = _sha256_file(d / "train.csv")
            test_sha256 = (
                _sha256_file(d / "test.csv") if (d / "test.csv").is_file() else None
            )
            # The outer split does not affect view membership. Validate all five class
            # summaries before admitting the dataset to the family-disjoint split.
            admission_manifest = build_split_view_manifest(
                dataset=d.name,
                source_family=source_family,
                task=TaskType.BINARY,
                outer_split="train",
                row_ids=row_ids,
                train_sha256=train_sha256,
                test_sha256=test_sha256,
            )
            build_binary_target_summaries(
                targets,
                row_ids=row_ids,
                split_view_manifest=admission_manifest,
            )
        except (OSError, ValueError, csv.Error, json.JSONDecodeError):
            continue
        admitted[d.name] = {
            "path": d,
            "signature": standardized_signature,
            "source_family": source_family,
            "targets": targets,
            "row_ids": row_ids,
            "train_sha256": train_sha256,
            "test_sha256": test_sha256,
        }

    source_families = {
        dataset_hash: str(record["source_family"])
        for dataset_hash, record in admitted.items()
    }
    validation_count = (
        248 if len(admitted) == 1_244 else round(len(admitted) * binary_val_fraction)
    )
    train_hashes, validation_hashes = freeze_source_family_split(
        source_families,
        train_count=len(admitted) - validation_count,
        holdout_count=validation_count,
        seed=seed,
    )
    train_hashes = set(train_hashes)
    val_hashes = set(validation_hashes)

    # ---- 3) generate + eval-binary-score seeds per binary dataset ------------------
    bin_rows: list[TrainingRow] = []
    eval_inputs: list[dict[str, Any]] = []
    eval_meta: list[TrainingRow] = []
    sig_cache: dict[str, list[float]] = {}
    target_summaries: dict[str, Any] = {}
    split_views: dict[str, Any] = {}
    for dataset_hash, record in admitted.items():
        d = record["path"]
        sig_cache[dataset_hash] = record["signature"]
        split = "validation" if dataset_hash in val_hashes else "train"
        split_view = build_split_view_manifest(
            dataset=dataset_hash,
            source_family=source_families[dataset_hash],
            task=TaskType.BINARY,
            outer_split="holdout" if split == "validation" else "train",
            row_ids=record["row_ids"],
            train_sha256=record["train_sha256"],
            test_sha256=record["test_sha256"],
        )
        split_views[dataset_hash] = split_view
        target_summaries[dataset_hash] = build_binary_target_summaries(
            record["targets"],
            row_ids=record["row_ids"],
            split_view_manifest=split_view,
        )
        for engine in engines:
            spec = grammar_for(engine)
            seen: set[str] = set()
            genomes: list[str] = []
            attempts = 0
            while (
                len(genomes) < seed_genomes_per_engine * 2
                and attempts < seed_genomes_per_engine * 12
            ):
                attempts += 1
                g = sample_genome_from_spec(
                    spec,
                    rng,
                    expr_fraction=0.15,
                    max_cost_class=max_cost_class,
                    task="binary",
                )
                if g not in seen:
                    seen.add(g)
                    genomes.append(g)
            metas = router.validate_many([(engine, g) for g in genomes])
            kept = 0
            for meta in metas:
                if kept >= seed_genomes_per_engine or not meta.get("valid"):
                    continue
                canonical = meta.get("canonical_genome")
                if not canonical:
                    continue
                kept += 1
                row = TrainingRow(
                    dataset_id=dataset_hash,
                    source="seed_binary",
                    embedding=[0.0],
                    genome=canonical,
                    canonical_genome=canonical,
                    valid=True,
                    engine=engine,
                    family_bitmap=dict(meta.get("family_bitmap") or {}),
                    complexity=int(meta.get("stage_count") or 0),
                    metadata={
                        "provenance": "seed",
                        "task_type": "binary",
                        "split": split,
                        "split_view_index": (kept - 1) % 5,
                    },
                )
                bin_rows.append(row)
                eval_inputs.append(
                    {
                        "engine": engine,
                        "task": "binary",
                        "dataset_hash": dataset_hash,
                        "train_csv": str(d / "train.csv"),
                        "genome": canonical,
                        "folds": folds,
                    }
                )
                eval_meta.append(row)

    # ---- 4) score binary seeds across the cluster (eval-binary) --------------------
    scored = failed = 0
    for idx, result in iter_eval_regression_with_ledger(
        eval_inputs, ledger=ledger, router=router, workers=eval_workers
    ):
        row = eval_meta[idx]
        cv = result.get("cv") if isinstance(result.get("cv"), dict) else {}
        oof = cv.get("oof_auc")
        fold_auc = cv.get("fold_auc") if isinstance(cv.get("fold_auc"), list) else None
        if isinstance(oof, (int, float)) and math.isfinite(float(oof)):
            row.score = float(oof)
            row.metric = "cv.oof_auc"
            row.metadata["oof_auc_raw"] = float(oof)
            row.metadata["score_robust"] = robust_score_binary(float(oof), fold_auc)
            row.cost_ms = (result.get("timings") or {}).get("total_ms")
            scored += 1
        else:
            row.score = None
            row.metadata["eval_error"] = (result.get("errors") or ["unknown"])[0]
            row.metadata["score_robust"] = 0.0  # AUC floor (chance) for failures
            failed += 1

    # ---- 5) finalize binary rows: embeddings, actions, per-dataset relative labels -
    for row in bin_rows:
        row.embedding = sig_cache.get(row.dataset_id) or [0.0] * DATASET_SIGNATURE_DIM
        view_index = int(row.metadata["split_view_index"])
        target_summary = target_summaries[row.dataset_id][view_index]
        row.target_embedding = list(target_summary.values)
        actions = parse_pipeline_actions(
            row.canonical_genome or row.genome, grammar_for(row.engine)
        )
        actions["task"] = "binary"
        d = binary_root / row.dataset_id
        row.metadata.update(
            {
                "engine": row.engine,
                "dataset_hash": row.dataset_id,
                "task_type": "binary",
                "split": "validation" if row.dataset_id in val_hashes else "train",
                "source_family": source_families[row.dataset_id],
                "split_view_manifest_sha256": split_views[row.dataset_id].sha256,
                "binary_target_summary_schema": "BinaryTargetSummary.v1",
                "binary_target_summary_view_index": view_index,
                "binary_target_summary_training_rows_hash": target_summary.training_rows_hash,
                "vs_path": str(d / ROW_SIDECAR),
                "train_csv": str(d / "train.csv"),
                "test_csv": str(d / "test.csv") if (d / "test.csv").exists() else None,
                "actions": actions,
            }
        )
    _assign_binary_relative_labels(bin_rows)
    assign_ranks(bin_rows)
    assign_elo(bin_rows)

    # ---- 6) merge + write ----------------------------------------------------------
    all_rows = reg_rows + [
        r for r in bin_rows if r.score is not None or r.metadata.get("eval_error")
    ]
    write_jsonl(out / "active_rows.jsonl", all_rows)
    (out / "standardization.json").write_text(
        json.dumps(standardization), encoding="utf-8"
    )
    summaries_dir = out / "binary_target_summaries"
    summaries_dir.mkdir(exist_ok=True)
    views_dir = out / "split_views"
    views_dir.mkdir(exist_ok=True)
    for dataset_hash, summaries in target_summaries.items():
        for target_summary in summaries:
            (
                summaries_dir / f"{dataset_hash}.view-{target_summary.view_index}.json"
            ).write_text(
                json.dumps(target_summary.to_json(), sort_keys=True), encoding="utf-8"
            )
        (views_dir / f"{dataset_hash}.json").write_text(
            json.dumps(split_views[dataset_hash].to_json(), sort_keys=True),
            encoding="utf-8",
        )

    # unified split manifest = regression manifest + binary entries
    manifest = json.loads((base / "split_manifest.json").read_text(encoding="utf-8"))
    for dataset_hash, record in admitted.items():
        d = record["path"]
        manifest.setdefault("datasets", {})[dataset_hash] = {
            "split": "validation" if dataset_hash in val_hashes else "train",
            "task": "binary",
            "source_family": source_families[dataset_hash],
            "split_view_manifest": str(views_dir / f"{dataset_hash}.json"),
            "split_view_manifest_sha256": split_views[dataset_hash].sha256,
            "binary_target_summaries": [
                str(summaries_dir / f"{dataset_hash}.view-{view_index}.json")
                for view_index in range(5)
            ],
            "vs_path": str(d / ROW_SIDECAR),
            "train_csv": str(d / "train.csv"),
            "test_csv": str(d / "test.csv") if (d / "test.csv").exists() else None,
            "best_reports_roc_auc": _best_report_auc(d),
        }
    (out / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    summary = {
        "out_dir": str(out),
        "regression_rows": len(reg_rows),
        "binary_rows": len([r for r in bin_rows if r.score is not None]),
        "binary_failed": failed,
        "binary_datasets": len(sig_cache),
        "binary_scored": scored,
        "total_rows": len(all_rows),
        "binary_train_datasets": len(train_hashes & set(admitted)),
        "binary_validation_datasets": len(val_hashes & set(admitted)),
        "binary_rejected_datasets": len(candidates) - len(admitted),
        "source_family_disjoint": not (
            {source_families[item] for item in train_hashes}
            & {source_families[item] for item in val_hashes}
        ),
    }
    (out / "fused_manifest.json").write_text(
        json.dumps(summary, sort_keys=True), encoding="utf-8"
    )
    return summary


def _assign_binary_relative_labels(rows: list[TrainingRow]) -> None:
    by_dataset: dict[str, list[TrainingRow]] = {}
    for row in rows:
        by_dataset.setdefault(row.dataset_id, []).append(row)
    for group in by_dataset.values():
        scored = [
            (r, r.metadata.get("score_robust"))
            for r in group
            if r.metadata.get("score_robust") is not None
        ]
        incumbent = max((s for _, s in scored), default=None)
        for row in group:
            value = row.metadata.get("score_robust")
            row.metadata["best_known_robust"] = incumbent
            row.metadata["oof_delta_to_incumbent"] = (
                None if (incumbent is None or value is None) else value - incumbent
            )


def _best_report_auc(d: Path) -> float | None:
    best = None
    reports = d / "reports"
    if reports.exists():
        for r in reports.glob("*.json"):
            try:
                data = json.loads(r.read_text(encoding="utf-8"))
                auc = (data.get("metrics") or {}).get("roc_auc")
                if isinstance(auc, (int, float)):
                    best = auc if best is None else max(best, auc)
            except (OSError, json.JSONDecodeError):
                continue
    return best


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _training_targets(path: Path) -> tuple[list[int], list[str]]:
    """Read strict binary targets and row identities from train.csv only."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        targets: list[int] = []
        row_ids: list[str] = []
        for index, row in enumerate(reader):
            if not row:
                continue
            raw = str(row[-1]).strip()
            if index == 0 and raw.lower() in {"target", "label", "y"}:
                continue
            if raw not in {"0", "1", "0.0", "1.0"}:
                raise ValueError(f"binary training target is not strict 0/1 in {path}")
            targets.append(int(float(raw)))
            row_ids.append(
                f"{index}:{hashlib.sha256(json.dumps(row, separators=(',', ':')).encode()).hexdigest()}"
            )
    return targets, row_ids


def _source_family(dataset_dir: Path) -> str:
    """Derive a stable lineage family; never use target/test statistics."""
    meta_path = dataset_dir / "meta.json"
    meta = (
        json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    )
    explicit = meta.get("source_family")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    real_source = meta.get("source_real_data")
    if real_source:
        material: Any = {"origin": "real", "source": real_source}
    else:
        recipe = meta.get("generator_recipe") or {}
        material = {
            "origin": meta.get("origin") or "unknown",
            "generator": meta.get("generator_name") or "unknown",
            # Counts define a generator lineage without using per-dataset seeds,
            # labels, acceptance scores, or any held-out/test measurement.
            "feature_family_counts": recipe.get("feature_family_counts") or {},
        }
    return f"family-{hashlib.sha256(json.dumps(material, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:16]}"
