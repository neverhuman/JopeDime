"""GP interop: seed jope genomes INTO the jmath GP engine, and harvest its winners back.

jope's newmath genomes are the GP engine's native wire form (both parse via
``newmath_core::dsl::parse_dsl``), so seeding is a pure JSONL file-drop — no jmath build.

**Write-isolation invariant**: everything this module writes lands under jope's own tree
(``artifacts/gp_export/<run>/`` for exports, a caller-chosen path for harvest). The jmath
state dir and winners dir are read ONLY. Injecting the exported files into a GP run is a
deliberate USER action (point ``gp-search --concepts`` at our file, or copy candidates into
the survivor pool) — this module never runs the GP engine or mutates its state.

``export``: top newmath genes above an oof threshold → ``concepts.jsonl`` (accepted
ConceptCards = verbatim gen-0 population) + ``candidates.jsonl`` (Candidates for breeding
stock, with a high acquisition_score so they survive elite truncation), deduped against the
GP's read-only novelty archive.

``harvest``: ``winner-*.json`` → jope TrainingRow JSONL with ``provenance="gp_report"``
(which the corpus builder already weights as a curated winner).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from .io import read_jsonl, write_jsonl
from .prime_config import load_config
from .schema import TrainingRow

DEFAULT_WINNERS_DIR = load_config().newmath_winners_dir
_OP_RE = re.compile(r"\(([a-z_0-9]+)")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ops_of(genome: str) -> list[str]:
    """Best-effort op list from a genome string (for ConceptCard.operators)."""
    ops = [m.group(1) for m in _OP_RE.finditer(genome or "")]
    return [op for op in ops if op != "pipe"]


def _novelty_keys(state_dir: Path | None) -> set[str]:
    """Read the GP's novelty archive (READ-ONLY) so we don't re-seed known genomes."""
    if state_dir is None:
        return set()
    archive = state_dir / "novelty_archive.jsonl"
    if not archive.exists():
        return set()
    keys: set[str] = set()
    for row in read_jsonl(archive):
        for field in ("novelty_hash", "canonical_expr", "dsl_expr"):
            value = row.get(field)
            if isinstance(value, str) and value:
                keys.add(value)
    return keys


def _concept_card(genome: str, oof: float, score_robust: float | None, task_family: str) -> dict[str, Any]:
    return {
        "concept_id": f"jope-{_sha(genome)[:12]}",
        "source_path": "jope:policy_v3",
        "source_sha256": _sha(genome),
        "task_family": task_family,
        "operators": _ops_of(genome),
        "canonical_expr": genome,
        "math_summary": "jope-proposed gene (policy_v3)",
        "novelty_tags": ["jope_seed"],
        "complexity": len(_ops_of(genome)),
        "applicability": [],
        "failure_modes": [],
        "judge_scores": {
            "jope_oof_r2": float(oof),
            **({"jope_score_robust": float(score_robust)} if score_robust is not None else {}),
        },
        "status": "accepted",  # accepted → gp-search injects verbatim as gen-0
    }


def _candidate(genome: str, oof: float, concept_id: str, acq_base: float, acq_scale: float) -> dict[str, Any]:
    # acquisition_score must clear typical GP acquisition so the seed survives elite
    # truncation; predicted_score carries the real jope oof as a prior.
    acq = acq_base + acq_scale * max(0.0, min(1.0, float(oof)))
    return {
        "candidate_id": f"jope-{_sha(genome)[:12]}",
        "dsl_expr": genome,
        "parent_concepts": [concept_id],
        "complexity": len(_ops_of(genome)),
        "mutation": "jope_seed",
        "origin": "jope",
        "generation": 0,
        "parents": [],
        "novelty_hash": "",  # normalize_candidate fills this on load
        "predicted_score": float(oof),
        "acquisition_score": float(acq),
    }


def _iter_source_rows(source_paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in source_paths:
        if path.exists():
            yield from read_jsonl(path)


def gp_export(
    *,
    sources: list[Path],
    out_dir: Path,
    top: int = 200,
    min_oof: float = 0.0,
    engine: str = "newmath",
    task_family: str = "regression",
    state_dir: Path | None = None,
    acq_base: float = 10.0,
    acq_scale: float = 5.0,
) -> dict[str, Any]:
    """Export top jope genes as GP ConceptCards + Candidates (write-isolated to out_dir).

    ``sources`` are jope JSONL files (loop_rows / evaluation / harvested rows) carrying a
    genome + an oof score. Only ``engine`` genes above ``min_oof`` are exported (GP is
    newmath-native); best-by-oof, deduped by canonical AND against the GP novelty archive.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    novelty = _novelty_keys(state_dir)
    best: dict[str, tuple[float, float | None]] = {}
    for row in _iter_source_rows(sources):
        row_engine = str(row.get("engine") or (row.get("metadata") or {}).get("engine") or "featmath")
        if row_engine != engine:
            continue
        genome = row.get("canonical_genome") or row.get("genome") or row.get("dsl_expr")
        if not isinstance(genome, str) or not genome:
            continue
        oof = _extract_oof(row)
        if oof is None or oof < min_oof:
            continue
        if genome in novelty:
            continue
        score_robust = _extract_robust(row)
        prev = best.get(genome)
        if prev is None or oof > prev[0]:
            best[genome] = (oof, score_robust)

    ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)[:top]
    concepts: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for genome, (oof, score_robust) in ranked:
        card = _concept_card(genome, oof, score_robust, task_family)
        concepts.append(card)
        candidates.append(_candidate(genome, oof, card["concept_id"], acq_base, acq_scale))

    concepts_path = out_dir / "concepts.jsonl"
    candidates_path = out_dir / "candidates.jsonl"
    write_jsonl(concepts_path, concepts)
    write_jsonl(candidates_path, candidates)
    return {
        "concepts": str(concepts_path),
        "candidates": str(candidates_path),
        "exported": len(concepts),
        "considered": len(best),
        "skipped_known": sum(1 for g in best if g in novelty),
        "engine": engine,
    }


def gp_harvest(*, winners_dir: Path = DEFAULT_WINNERS_DIR, out: Path) -> dict[str, Any]:
    """Read GP winners (READ-ONLY) → jope TrainingRow JSONL with provenance=gp_report."""
    winners_dir = Path(winners_dir)
    rows: list[TrainingRow] = []
    seen: set[str] = set()
    for path in sorted(winners_dir.glob("winner-*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        genome = data.get("dsl_expr") or (data.get("concept") or {}).get("canonical_expr")
        if not isinstance(genome, str) or not genome or genome in seen:
            continue
        seen.add(genome)
        receipt = data.get("benchmark_receipt") if isinstance(data.get("benchmark_receipt"), dict) else {}
        elo = _num(receipt.get("aggregate_elo"))
        delta_lcb = _num(receipt.get("aggregate_delta_lcb"))
        concept = data.get("concept") if isinstance(data.get("concept"), dict) else {}
        rows.append(
            TrainingRow(
                dataset_id="gp_winner",
                source="gp_report",
                embedding=[0.0],
                genome=genome,
                canonical_genome=genome,
                engine="newmath",
                score=delta_lcb,
                elo=elo,
                feedback_origin="gp_report",
                metadata={
                    "provenance": "gp_report",
                    "concept_id": concept.get("concept_id"),
                    "promotion_reason": data.get("promotion_reason"),
                    "aggregate_elo": elo,
                    "aggregate_delta_lcb": delta_lcb,
                    "winner_path": str(path),
                },
            )
        )
    out = Path(out)
    write_jsonl(out, rows)
    return {"out": str(out), "harvested": len(rows), "winners_dir": str(winners_dir)}


def _extract_oof(row: dict[str, Any]) -> float | None:
    cv = row.get("cv") if isinstance(row.get("cv"), dict) else {}
    for source in (
        row.get("oof_r2"),
        cv.get("oof_r2"),
        (row.get("metadata") or {}).get("oof_r2_raw"),
        row.get("score"),
    ):
        value = _num(source)
        if value is not None:
            return value
    return None


def _extract_robust(row: dict[str, Any]) -> float | None:
    meta = row.get("metadata") or {}
    return _num(meta.get("score_robust", row.get("score_robust")))


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        return out if out == out else None  # drop NaN
    except (TypeError, ValueError):
        return None
