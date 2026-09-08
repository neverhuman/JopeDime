from __future__ import annotations

import random
from pathlib import Path

from jope.data.importers import import_jmath_jsonl, import_quant_reports
from jope.data.labels import assign_elo, assign_ranks
from jope.grammar import estimate_complexity, iter_sampled
from jope.schema import TrainingRow
from jope.validator import RustValidator


def build_corpus(
    *,
    quant_root: str | Path | None = None,
    jmath_jsonl: list[str | Path] | None = None,
    generated: int = 0,
    invalid: int = 0,
    seed: int = 0,
    validate: bool = True,
) -> list[TrainingRow]:
    rows: list[TrainingRow] = []
    if quant_root is not None:
        rows.extend(import_quant_reports(quant_root))
    if jmath_jsonl:
        rows.extend(import_jmath_jsonl(jmath_jsonl))
    rows.extend(_generated_rows(generated, invalid, seed))
    if validate and rows:
        _apply_validation(rows)
    assign_ranks(rows)
    assign_elo(rows)
    return rows


def _generated_rows(generated: int, invalid: int, seed: int) -> list[TrainingRow]:
    rng = random.Random(seed)
    rows = []
    for i, (genome, valid) in enumerate(iter_sampled(generated, seed=seed, invalid=invalid)):
        rows.append(
            TrainingRow(
                dataset_id=f"generated-{seed}",
                source="grammar_sampler",
                embedding=[rng.uniform(-0.01, 0.01) for _ in range(512)],
                genome=genome,
                valid=valid,
                metric="validity",
                score=1.0 if valid else 0.0,
                complexity=estimate_complexity(genome),
                runtime_class="cheap",
                feedback_origin="synthetic",
                metadata={"sample_index": i},
            )
        )
    return rows


def _apply_validation(rows: list[TrainingRow]) -> None:
    validator = RustValidator()
    outputs = validator.validate_many([row.genome for row in rows])
    for row, meta in zip(rows, outputs, strict=True):
        row.valid = bool(meta.get("valid"))
        row.canonical_genome = meta.get("canonical_genome")
        row.family_bitmap = dict(meta.get("family_bitmap") or {})
        row.complexity = int(meta.get("stage_count") or row.complexity or 0)
        row.metadata["validation"] = meta

