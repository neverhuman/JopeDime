from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

from jope.grammar import estimate_complexity
from jope.schema import TrainingRow, normalize_embedding

GENOME_KEYS = ("genome", "gene", "dsl", "dsl_expr", "canonical_genome")
SCORE_KEYS = ("score", "oracle_score", "search_score", "mean_r2", "oof_r2", "r2")


def import_quant_reports(root: str | Path, limit: int | None = None) -> list[TrainingRow]:
    root = Path(root)
    paths = sorted(
        {
            *root.glob("regression/*/reports/jain_regressor_*.json"),
            *root.glob("*/reports/jain_regressor_*.json"),
        }
    )
    rows: list[TrainingRow] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for row in _rows_from_quant_report(data, path):
            identity = (row.dataset_id, row.canonical_genome or row.genome)
            if identity in seen:
                continue
            seen.add(identity)
            rows.append(row)
        if limit is not None and len(rows) >= limit:
            return rows[:limit]
    return rows


def import_jmath_jsonl(paths: Iterable[str | Path], limit: int | None = None) -> list[TrainingRow]:
    rows: list[TrainingRow] = []
    for pathlike in paths:
        path = Path(pathlike)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if limit is not None and len(rows) >= limit:
                    return rows
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                genome = _extract_first(data, GENOME_KEYS)
                if not genome:
                    continue
                score = _extract_score(data)
                rows.append(
                    TrainingRow(
                        dataset_id=str(data.get("dataset_id") or data.get("candidate_id") or path.stem),
                        source="jmath_jsonl",
                        embedding=normalize_embedding(data.get("embedding")),
                        genome=genome,
                        canonical_genome=data.get("canonical_genome"),
                        valid=bool(data.get("compile_ok", data.get("valid", True))),
                        metric=_metric_from_row(data),
                        score=score,
                        rank=_optional_int(data.get("rank")),
                        cost_ms=_optional_float(data.get("estimated_wall_ms")),
                        complexity=estimate_complexity(genome),
                        runtime_class=_runtime_class(_optional_float(data.get("estimated_wall_ms"))),
                        feedback_origin=str(data.get("origin") or data.get("origin_run") or "historical"),
                        engine="newmath",
                        metadata={"path": str(path), "line": line_no},
                    )
                )
    return rows


def _rows_from_quant_report(data: dict, path: Path) -> list[TrainingRow]:
    dataset_id = str(data.get("dataset_hash") or path.parent.parent.name)
    metric = "r2"
    cost_ms = _timing_ms(data)
    rows = []
    candidates = list(data.get("finalists") or [])
    if _extract_first(data, GENOME_KEYS):
        candidates.append(data)
    seen: set[str] = set()
    for candidate in candidates:
        genome = _extract_first(candidate, GENOME_KEYS)
        if not genome or genome in seen:
            continue
        seen.add(genome)
        cv = candidate.get("cv") if isinstance(candidate.get("cv"), dict) else {}
        # Historical evidence is accepted only when the candidate itself carries
        # train-CV evidence. Test/full-fit/report summary scores are never adapted.
        score = _optional_float(cv.get("oof_r2"))
        if score is None:
            score = _optional_float(cv.get("mean_r2"))
        if score is None or not math.isfinite(score):
            continue
        rows.append(
            TrainingRow(
                dataset_id=dataset_id,
                source="quant_report",
                embedding=normalize_embedding(data.get("embedding")),
                genome=genome,
                canonical_genome=candidate.get("canonical_genome"),
                valid=True,
                metric=metric,
                score=score,
                rank=_optional_int(candidate.get("rank")),
                cost_ms=cost_ms,
                complexity=estimate_complexity(genome),
                runtime_class=_runtime_class(cost_ms),
                feedback_origin="historical",
                metadata={
                    "path": str(path),
                    "algorithm": data.get("algorithm"),
                    "provenance": "historical_import",
                    "evidence_weight": 0.25,
                    "score_source": "train_cv",
                    "canonical_re_evaluation": False,
                    "elo_authority": False,
                },
            )
        )
    return rows


def _extract_first(data: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _extract_score(data: dict) -> float | None:
    for key in SCORE_KEYS:
        value = data.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    cv = data.get("cv")
    if isinstance(cv, dict):
        for key in ("oof_r2", "mean_r2", "r2"):
            value = cv.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    metrics = data.get("metrics")
    if isinstance(metrics, dict):
        for key in SCORE_KEYS:
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _metric_from_row(data: dict) -> str | None:
    if data.get("oracle_score") is not None:
        return "oracle_score"
    if data.get("score") is not None:
        return "score"
    return None


def _timing_ms(data: dict) -> float | None:
    timings = data.get("timings")
    if not isinstance(timings, dict):
        return None
    if isinstance(timings.get("total_secs"), (int, float)):
        return float(timings["total_secs"]) * 1000.0
    return None


def _runtime_class(cost_ms: float | None) -> str | None:
    if cost_ms is None:
        return None
    if cost_ms < 50:
        return "cheap"
    if cost_ms < 5000:
        return "medium"
    return "expensive"


def _optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
