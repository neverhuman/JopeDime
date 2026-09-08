from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data.active import format_pipeline_actions, parse_pipeline_actions
from .data.labels import assign_elo, assign_ranks
from .io import read_jsonl, write_jsonl
from .schema import TrainingRow
from .validator import RustValidator


def ingest_feedback(report: str | Path, ledger: str | Path) -> dict:
    report_path = Path(report)
    ledger_path = Path(ledger)
    existing = [TrainingRow.from_json(row) for row in read_jsonl(ledger_path)] if ledger_path.exists() else []
    data = json.loads(report_path.read_text(encoding="utf-8"))
    rows = _rows_from_feedback(data, report_path)
    all_rows = existing + rows
    assign_ranks(all_rows)
    assign_elo(all_rows)
    write_jsonl(ledger_path, all_rows)
    return {"ingested": len(rows), "ledger": str(ledger_path), "total_rows": len(all_rows)}


def build_active_feedback_corpus(
    *,
    art_dir: str | Path,
    evaluation_jsonl: str | Path,
    out: str | Path,
    base_corpus: str | Path | None = None,
    allow_non_train: bool = False,
) -> dict:
    art = Path(art_dir)
    eval_path = Path(evaluation_jsonl)
    base_path = Path(base_corpus) if base_corpus is not None else art / "active_rows.jsonl"
    split_manifest = _read_json(art / "split_manifest.json")
    split_datasets = split_manifest.get("datasets") if isinstance(split_manifest.get("datasets"), dict) else {}
    base_rows = []
    for row in (TrainingRow.from_json(item) for item in read_jsonl(base_path)):
        _strip_holdout_target_metadata(row)
        split = str(row.metadata.get("split") or (split_datasets.get(row.dataset_id) or {}).get("split") or "train")
        if split != "train" and not allow_non_train:
            continue
        base_rows.append(row)
    templates = _template_rows_by_dataset(base_rows)
    evaluation_rows = list(read_jsonl(eval_path))
    feedback_rows = _rows_from_active_evaluations(
        evaluation_rows,
        templates=templates,
        split_datasets=split_datasets,
        evaluation_path=eval_path,
        allow_non_train=allow_non_train,
    )
    all_rows = base_rows + feedback_rows
    _assign_relative_metadata(all_rows)
    assign_ranks(all_rows)
    assign_elo(all_rows)
    written = write_jsonl(out, all_rows)
    return {
        "base_rows": len(base_rows),
        "feedback_rows": len(feedback_rows),
        "rows": written,
        "out": str(out),
        "evaluation_jsonl": str(eval_path),
        "allow_non_train": allow_non_train,
    }


def _rows_from_feedback(data: dict, path: Path) -> list[TrainingRow]:
    candidates = data.get("finalists") if isinstance(data.get("finalists"), list) else [data]
    rows = []
    for i, item in enumerate(candidates):
        genome = item.get("genome") or item.get("gene") or item.get("dsl") or item.get("dsl_expr")
        if not genome:
            continue
        cv = item.get("cv") if isinstance(item.get("cv"), dict) else {}
        score = cv.get("oof_r2") or cv.get("mean_r2") or item.get("score") or (data.get("metrics") or {}).get("r2")
        rows.append(
            TrainingRow(
                dataset_id=str(data.get("dataset_hash") or item.get("dataset_id") or path.stem),
                source="feedback_report",
                embedding=data.get("embedding"),
                genome=genome,
                valid=True,
                metric="r2" if score is not None else None,
                score=float(score) if score is not None else None,
                rank=item.get("rank") or i + 1,
                cost_ms=(data.get("timings") or {}).get("total_secs", 0.0) * 1000.0,
                feedback_origin="delayed_real",
                metadata={"path": str(path)},
            )
        )
    return rows


def _rows_from_active_evaluations(
    evaluation_rows: list[dict],
    *,
    templates: dict[str, TrainingRow],
    split_datasets: dict[str, dict],
    evaluation_path: Path,
    allow_non_train: bool,
) -> list[TrainingRow]:
    valid_items: list[tuple[dict, str, dict]] = []
    rejected: list[tuple[str, str]] = []
    for row in evaluation_rows:
        dataset_id = str(row.get("dataset_hash") or row.get("dataset_id") or "")
        if not dataset_id:
            continue
        entry = split_datasets.get(dataset_id) if isinstance(split_datasets.get(dataset_id), dict) else {}
        split = str(row.get("split") or entry.get("split") or "train")
        if split != "train" and not allow_non_train:
            rejected.append((dataset_id, split))
            continue
        genome = row.get("canonical_genome") or row.get("genome")
        cv = row.get("cv") if isinstance(row.get("cv"), dict) else {}
        score = cv.get("oof_r2")
        if not isinstance(genome, str) or not isinstance(score, (int, float)) or not row.get("valid", True):
            continue
        valid_items.append((row, genome, entry))
    if rejected:
        examples = ", ".join(f"{dataset}:{split}" for dataset, split in rejected[:5])
        raise ValueError(f"active feedback defaults to train split only; rejected {len(rejected)} non-train evaluations ({examples})")
    validator = RustValidator()
    validations = validator.validate_many([genome for _, genome, _ in valid_items]) if valid_items else []
    rows: list[TrainingRow] = []
    for (item, genome, entry), validation in zip(valid_items, validations, strict=True):
        if not validation.get("valid"):
            continue
        dataset_id = str(item.get("dataset_hash") or item.get("dataset_id"))
        template = templates.get(dataset_id)
        if template is None:
            continue
        canonical = str(validation.get("canonical_genome") or genome)
        actions = parse_pipeline_actions(canonical)
        canonical = format_pipeline_actions(actions)
        cv = item.get("cv") if isinstance(item.get("cv"), dict) else {}
        score = float(cv["oof_r2"])
        incumbent_value = item.get("incumbent_oof_r2")
        if incumbent_value is None:
            incumbent_value = entry.get("best_known_oof_r2")
        incumbent = _optional_float(incumbent_value)
        delta = score - incumbent if incumbent is not None else _optional_float(item.get("oof_delta_vs_incumbent"))
        timings = item.get("timings") if isinstance(item.get("timings"), dict) else {}
        metadata = {
            "source": "active_eval",
            "dataset_hash": dataset_id,
            "split": "train" if not allow_non_train else str(item.get("split") or entry.get("split") or "train"),
            "vs_path": entry.get("vs_path") or template.metadata.get("vs_path"),
            "train_csv": item.get("train_csv") or entry.get("train_csv") or template.metadata.get("train_csv"),
            "train_target_vs": item.get("train_target_vs") or entry.get("train_target_vs") or template.metadata.get("train_target_vs"),
            "actions": actions,
            "candidate_rank": None,
            "rank_percentile": None,
            "best_known_oof_r2": incumbent,
            "oof_delta_to_incumbent": delta,
            "active_eval": {
                "evaluation_jsonl": str(evaluation_path),
                "proposal_index": item.get("proposal_index"),
                "decoder": item.get("decoder"),
                "reward": item.get("reward"),
                "predicted_oof_delta": item.get("predicted_oof_delta"),
                "cache_key": item.get("cache_key"),
                "evaluator_version": item.get("evaluator_version"),
            },
        }
        rows.append(
            TrainingRow(
                dataset_id=dataset_id,
                source="active_eval",
                embedding=template.embedding,
                target_embedding=template.target_embedding,
                genome=canonical,
                canonical_genome=canonical,
                valid=True,
                metric="cv.oof_r2",
                score=score,
                cost_ms=_optional_float(timings.get("total_ms")),
                family_bitmap=dict(validation.get("family_bitmap") or {}),
                complexity=int(validation.get("stage_count") or 0),
                runtime_class=_runtime_class(_optional_float(timings.get("total_ms"))),
                feedback_origin="active_eval",
                metadata=metadata,
            )
        )
    return rows


def _template_rows_by_dataset(rows: list[TrainingRow]) -> dict[str, TrainingRow]:
    out: dict[str, TrainingRow] = {}
    for row in rows:
        out.setdefault(row.dataset_id, row)
    return out


def _assign_relative_metadata(rows: list[TrainingRow]) -> None:
    grouped: dict[str, list[TrainingRow]] = {}
    for row in rows:
        grouped.setdefault(row.dataset_id, []).append(row)
    for group in grouped.values():
        scored = [row for row in group if row.score is not None]
        scored.sort(key=lambda row: float(row.score), reverse=True)
        if not scored:
            continue
        best = float(scored[0].score)
        denom = max(1, len(scored) - 1)
        for rank, row in enumerate(scored, start=1):
            row.metadata["candidate_rank"] = rank
            row.metadata["rank_percentile"] = 0.0 if len(scored) == 1 else (rank - 1) / denom
            row.metadata.setdefault("best_known_oof_r2", best)
            if row.metadata.get("oof_delta_to_incumbent") is None:
                row.metadata["oof_delta_to_incumbent"] = float(row.score) - best


def _strip_holdout_target_metadata(row: TrainingRow) -> None:
    row.metadata.pop("test_target_vs", None)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _runtime_class(cost_ms: float | None) -> str | None:
    if cost_ms is None:
        return None
    if cost_ms < 50:
        return "cheap"
    if cost_ms < 5000:
        return "medium"
    return "expensive"
