"""Label normalization and corpus construction for Prime training data."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .prime_label_to_corpus_embeddings import (
    DUAL_EMBEDDING_DIM,
    TARGET_SUMMARY_DIM,
    PlaceholderEmbedder,
)
from .prime_label_to_corpus_identity import (
    DEFAULT_PROVENANCE,
    LEAGUE_LABEL_SCHEMA,
    GenomeEngineResolver,
    _finite,
    _league_label_evidence,
    _reward_pair_identity,
)
from .schema import TrainingRow
from .split_integrity import (
    DEVELOPMENT_SPLIT,
    TEMPORAL_AUDIT_SPLIT,
    assignment_for_dataset,
    is_training_authority_label,
    validate_split_manifest,
)


# --------------------------------------------------------------------------- #
# Row parsing (schema -> canonical fields)                                    #
# --------------------------------------------------------------------------- #
def _parse_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize any supported label schema into canonical adapter fields."""
    committed = row
    if isinstance(row.get("label"), dict):
        row = row["label"]
    schema = str(row.get("schema") or "")
    if schema == LEAGUE_LABEL_SCHEMA:
        if str(row.get("split") or "train").lower() != "train":
            return None
        genome = row.get("canonical_genome") or row.get("semantic_genome")
        dataset = row.get("dataset")
        if not genome or not dataset:
            return None
        evidence = _league_label_evidence(row)
        safe_provenance = evidence["provenance"]
        failure = row.get("failure")
        task_type = str(row.get("task") or "regression")
        return {
            "dataset_hash": str(dataset),
            "canonical_genome": str(genome),
            "engine": row.get("engine"),
            "reward": _finite(row.get("robust_score")),
            "oof_r2": _finite(safe_provenance.get("oof_r2")),
            "fold_r2": safe_provenance.get("fold_r2"),
            "folds": evidence["folds"],
            "fold_plan": evidence["fold_plan"],
            "evaluator_version": evidence["evaluator_version"],
            "evidence_tier": evidence["evidence_tier"],
            "runtime_seconds": evidence["runtime_seconds"],
            "gp_task_id": safe_provenance.get("gp_task_id"),
            "robust_formula": safe_provenance.get("robust_formula"),
            "reward_pair_identity": _reward_pair_identity(
                dataset=str(dataset),
                task=task_type,
                folds=evidence["folds"],
                fold_plan=evidence["fold_plan"],
                evaluator_version=evidence["evaluator_version"],
            ),
            "p_bad": 1.0 if failure else 0.0,
            "failed": bool(failure),
            "task_type": task_type,
            "semantic_plan_hash": row.get("semantic_genome"),
            "reward_std": None,
            "n_evidence": None,
            "promotion_eligible": row.get("failure") is None,
            "label_watermark": committed.get("watermark") or committed.get("_watermark"),
            "committed_at_ms": committed.get("committed_at_ms") or committed.get("_committed_at_ms"),
            "replay_hot_until_ms": committed.get("_replay_hot_until_ms"),
        }
    # aggregate-v1 and realization-v2 share dataset_hash + canonical_genome
    genome = row.get("canonical_genome")
    dataset = row.get("dataset_hash") or row.get("dataset_id")
    if not genome or not dataset:
        return None
    reward = _finite(row.get("reward"))
    if reward is None:
        reward = _finite(row.get("mean_realized"))
    if reward is None:
        reward = _finite(row.get("realized_score"))
    oof = _finite(row.get("oof_r2"))
    failed = bool(row.get("failure")) or (row.get("n_failed") or 0) > 0 and reward is None
    return {
        "dataset_hash": str(dataset),
        "canonical_genome": str(genome),
        "engine": row.get("engine"),
        "reward": reward,
        "oof_r2": oof if oof is not None else reward,
        "p_bad": _finite(row.get("p_bad")),
        "failed": bool(failed),
        "task_type": str(row.get("task") or "regression"),
        "semantic_plan_hash": row.get("semantic_plan_hash"),
        "reward_std": _finite(row.get("reward_std")),
        "n_evidence": row.get("n_evidence"),
        "fold_r2": None,
        "folds": None,
        "fold_plan": None,
        "evaluator_version": None,
        "evidence_tier": None,
        "runtime_seconds": None,
        "gp_task_id": None,
        "robust_formula": None,
        "reward_pair_identity": None,
        "promotion_eligible": bool(row.get("promotion_eligible", True)),
        "label_watermark": committed.get("watermark") or committed.get("_watermark"),
        "committed_at_ms": committed.get("committed_at_ms") or committed.get("_committed_at_ms"),
        "replay_hot_until_ms": committed.get("_replay_hot_until_ms"),
    }


def label_to_training_row(
    row: dict[str, Any],
    *,
    embedder: Any,
    resolver: GenomeEngineResolver,
    split: str = "train",
    provenance: str = DEFAULT_PROVENANCE,
) -> tuple[TrainingRow | None, str]:
    """Convert one label row into a ``TrainingRow``. Returns (row_or_None, reason)."""
    if not is_training_authority_label(row):
        return None, "report_only_evidence"
    parsed = _parse_row(row)
    if parsed is None:
        return None, "missing_identity"
    genome = parsed["canonical_genome"]
    engine = resolver.resolve(genome, parsed.get("engine"))
    if engine is None:
        return None, "unencodable_genome"

    reward = parsed["reward"]
    if reward is None and not parsed["failed"]:
        return None, "no_finite_reward"

    emb, tgt = embedder(parsed["dataset_hash"])
    synthetic = emb is None
    if synthetic:  # signature embedder missed this dataset -> deterministic placeholder
        emb, tgt = PlaceholderEmbedder(DUAL_EMBEDDING_DIM, TARGET_SUMMARY_DIM)(parsed["dataset_hash"])

    metadata: dict[str, Any] = {
        "provenance": provenance,
        "task_type": parsed["task_type"],
        "split": split,
        "dataset_hash": parsed["dataset_hash"],
        "semantic_plan_hash": parsed.get("semantic_plan_hash"),
        "p_bad": parsed.get("p_bad"),
        "reward_std": parsed.get("reward_std"),
        "n_evidence": parsed.get("n_evidence"),
        "promotion_eligible": parsed.get("promotion_eligible"),
        "realization_engine": parsed.get("engine"),
        "embedding_synthetic": synthetic,
        "label_watermark": parsed.get("label_watermark"),
        "committed_at_ms": parsed.get("committed_at_ms"),
        "replay_hot_until_ms": parsed.get("replay_hot_until_ms"),
        "fold_r2": parsed.get("fold_r2"),
        "folds": parsed.get("folds"),
        "fold_plan": parsed.get("fold_plan"),
        "evaluator_version": parsed.get("evaluator_version"),
        "evidence_tier": parsed.get("evidence_tier"),
        "gp_task_id": parsed.get("gp_task_id"),
        "robust_formula": parsed.get("robust_formula"),
        "reward_pair_identity": parsed.get("reward_pair_identity"),
    }
    if reward is not None:
        # feed all three keys train_v3._score_label reads, so the within-dataset
        # listwise/reward selector labels are recomputed from the realized reward.
        metadata["score_robust"] = reward
        metadata["oof_r2_raw"] = parsed["oof_r2"] if parsed["oof_r2"] is not None else reward
    if parsed["failed"]:
        # a realized failure is a hard negative for the p_bad / stability head.
        metadata["eval_error"] = True

    tr = TrainingRow(
        dataset_id=parsed["dataset_hash"],
        source=provenance,
        embedding=emb,
        target_embedding=tgt,
        genome=genome,
        canonical_genome=genome,
        engine=engine,
        valid=not parsed["failed"],
        score=reward,
        cost_ms=(parsed["runtime_seconds"] * 1_000.0 if parsed.get("runtime_seconds") is not None else None),
        metadata=metadata,
    )
    return tr, "ok"


def build_corpus(
    rows: Iterable[dict[str, Any]],
    *,
    embedder: Any | None = None,
    resolver: GenomeEngineResolver | None = None,
    val_fraction: float = 0.2,
    seed: int = 0,
    provenance: str = DEFAULT_PROVENANCE,
    validation_datasets: Iterable[str] | None = None,
    split_manifest: dict[str, Any] | None = None,
) -> tuple[list[TrainingRow], dict[str, Any]]:
    """Adapt an iterable of label rows into a policy_v3 ``TrainingRow`` corpus.

    Whole datasets are held out for the validation split (``train_v3`` requires one and
    the listwise loss groups by dataset). Returns ``(training_rows, stats)``.
    """
    rows = list(rows)
    resolver = resolver or GenomeEngineResolver()
    embedder = embedder or PlaceholderEmbedder()

    # Assign validation by DATASET so a dataset never straddles train/val.  A
    # continuous trainer passes the initially selected identities back on every
    # epoch: newly committed TRAIN labels can then enter replay without moving
    # the fixed internal validation boundary.
    def dataset_id(row: dict[str, Any]) -> str:
        label = row.get("label") if isinstance(row.get("label"), dict) else row
        return str(label.get("dataset_hash") or label.get("dataset_id") or label.get("dataset") or "")

    if split_manifest is not None:
        validate_split_manifest(split_manifest)
        if split_manifest["status"] != "complete":
            raise ValueError("cannot build a corpus from an incomplete split manifest")
    hashes = sorted({dataset_id(r) for r in rows if is_training_authority_label(r)} - {""})
    if split_manifest is not None:
        val_datasets = {
            dataset
            for dataset, record in split_manifest["assignments"].items()
            if record["split"] == DEVELOPMENT_SPLIT
        }
    elif validation_datasets is None:
        shuffled = hashes[:]
        random.Random(seed).shuffle(shuffled)
        n_val = max(1, round(val_fraction * len(shuffled))) if (val_fraction > 0 and len(shuffled) > 1) else 0
        val_datasets = set(shuffled[:n_val])
    else:
        val_datasets = {str(value) for value in validation_datasets}

    out: list[TrainingRow] = []
    reasons: dict[str, int] = {}
    engines: dict[str, int] = {}
    synthetic = 0
    audit_datasets: set[str] = set()
    for r in rows:
        dh = dataset_id(r)
        if not is_training_authority_label(r):
            reasons["report_only_evidence"] = reasons.get("report_only_evidence", 0) + 1
            continue
        assignment = assignment_for_dataset(split_manifest, dh) if split_manifest is not None else None
        if assignment is not None and assignment["split"] == TEMPORAL_AUDIT_SPLIT:
            audit_datasets.add(dh)
            reasons["report_only_temporal_audit"] = reasons.get("report_only_temporal_audit", 0) + 1
            continue
        split = (
            assignment["split"] if assignment is not None else "validation" if dh in val_datasets else "train"
        )
        tr, reason = label_to_training_row(
            r, embedder=embedder, resolver=resolver, split=split, provenance=provenance
        )
        reasons[reason] = reasons.get(reason, 0) + 1
        if tr is None:
            continue
        engines[tr.engine] = engines.get(tr.engine, 0) + 1
        if (tr.metadata or {}).get("embedding_synthetic"):
            synthetic += 1
        if assignment is not None:
            tr.metadata.update(
                {
                    "source_family": assignment["source_family"],
                    "split_assignment": assignment["split"],
                    "split_manifest_schema": split_manifest["schema"],
                    "split_manifest_sha256": split_manifest["manifest_sha256"],
                }
            )
        out.append(tr)

    train_datasets = {str(r.dataset_id) for r in out if (r.metadata or {}).get("split") == "train"}
    val_seen = {
        str(r.dataset_id) for r in out if (r.metadata or {}).get("split") in {"validation", DEVELOPMENT_SPLIT}
    }
    stats = {
        "input_rows": len(rows),
        "training_rows": len(out),
        "skipped": {k: v for k, v in reasons.items() if k != "ok"},
        "engines": engines,
        "datasets": len(train_datasets | val_seen),
        "train_datasets": len(train_datasets),
        "validation_datasets": len(val_seen),
        "validation_dataset_ids": sorted(val_datasets),
        "temporal_audit_datasets": len(audit_datasets),
        "temporal_audit_dataset_ids": sorted(audit_datasets),
        "split_manifest_sha256": (split_manifest["manifest_sha256"] if split_manifest is not None else None),
        "embedding_dim": len(out[0].embedding) if out else 0,
        "target_summary_dim": len(out[0].target_embedding or []) if out else 0,
        "synthetic_embedding_rows": synthetic,
    }
    return out, stats


def write_corpus(rows: list[TrainingRow], out_path: str | Path) -> Path:
    """Write ``TrainingRow`` corpus as JSONL (the format ``train_v3`` reads)."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_json()) + "\n")
    return out_path
