#!/usr/bin/env python3
"""Build percentile-ready, per-dataset ledgers from immutable evolution receipts."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import io
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "jope-dime.all-regression-evolution-index.v1"
LEDGER_SCHEMA = "jope-dime.all-regression-evaluation.v1"
ARCHIVE_SCHEMA = "jope-dime.all-regression-analysis-archive.v1"

CSV_FIELDS = (
    "cohort",
    "dataset",
    "run_id",
    "generation",
    "slot",
    "logical_evaluation",
    "fresh_evaluation",
    "cached",
    "valid",
    "rmse",
    "mae",
    "r2",
    "rmse_rank_best_first",
    "empirical_performance_percentile",
    "human_best_rmse",
    "beats_human_best",
    "canonical_genome",
    "semantic_sha256",
    "complexity",
    "evolved_descendant",
    "evolutionary_winner_eligible",
    "ancestry",
    "failure",
    "runtime_seconds",
    "node",
    "scq_job_id",
    "attempt",
    "receipt_file",
    "receipt_sha256",
    "checkpoint_file",
    "checkpoint_sha256",
    "binary_sha256",
    "train_sha256",
    "test_sha256",
    "predictions_sha256",
)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as source:
            value = json.load(source)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def csv_bytes(records: Iterable[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for record in records:
        row = dict(record)
        for field in ("ancestry", "failure"):
            if row.get(field) is not None:
                row[field] = json.dumps(row[field], sort_keys=True, separators=(",", ":"))
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


def finite(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def checkpoint_slots(checkpoint: dict[str, Any]) -> Iterable[dict[str, Any]]:
    slots = checkpoint.get("slots")
    if isinstance(slots, list):
        yield from (slot for slot in slots if isinstance(slot, dict))
        return
    islands = checkpoint.get("islands")
    if isinstance(islands, list):
        for island in islands:
            if isinstance(island, dict) and isinstance(island.get("slots"), list):
                yield from (slot for slot in island["slots"] if isinstance(slot, dict))


def member_fields(slot: dict[str, Any]) -> tuple[str | None, str | None, Any, Any]:
    member = slot.get("member")
    if not isinstance(member, dict):
        candidate = slot.get("candidate")
        member = candidate if isinstance(candidate, dict) else {}
    genome = member.get("canonical_genome", member.get("syntax"))
    semantic = member.get("semantic_sha256", member.get("semantic_hash"))
    return genome, semantic, member.get("complexity"), member.get("ancestry")


def score_fields(slot: dict[str, Any]) -> tuple[bool, float | None]:
    metrics = slot.get("metrics")
    score = finite(metrics.get("rmse")) if isinstance(metrics, dict) else finite(slot.get("rmse"))
    valid = bool(slot.get("valid", score is not None)) and score is not None
    return valid, score


def evolutionary_eligible(record: dict[str, Any]) -> bool:
    ancestry = record.get("ancestry")
    if not isinstance(ancestry, dict):
        return False
    parents = ancestry.get("parents")
    semantic = record.get("semantic_sha256")
    return bool(
        int(record["generation"]) >= 1
        and not record["cached"]
        and record.get("evolved_descendant") is True
        and ancestry.get("operation") in {"mutation", "crossover"}
        and isinstance(parents, list)
        and parents
        and isinstance(semantic, str)
        and all(parent != semantic for parent in parents)
    )


def dataset_specs(repo: Path, golden_run: str, tabarena_run: str) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    golden_root = repo / "runs" / golden_run
    golden_manifest = golden_root / "aggressive-manifest.v1.json"
    golden = read_json(golden_manifest)
    if golden is not None:
        for item in golden.get("datasets", []):
            if not isinstance(item, dict):
                continue
            specs.append(
                {
                    "cohort": "golden42",
                    "run_id": golden_run,
                    "run_root": golden_root,
                    "dataset": item.get("dataset_hash"),
                    "leader_models": item.get("leader_models", []),
                    "leader_rmse": item.get("leader_rmse"),
                    "population": golden.get("population", 240),
                    "manifest_sha256": hashlib.sha256(golden_manifest.read_bytes()).hexdigest(),
                }
            )
    tabarena_root = repo / "runs" / tabarena_run
    tabarena_manifest = tabarena_root / "tabarena-island-manifest.v1.json"
    tabarena = read_json(tabarena_manifest)
    if tabarena is not None:
        for item in tabarena.get("datasets", []):
            if not isinstance(item, dict):
                continue
            specs.append(
                {
                    "cohort": "tabarena-lite-v0.1-regression",
                    "run_id": tabarena_run,
                    "run_root": tabarena_root,
                    "dataset": item.get("dataset"),
                    "leader_models": item.get("leader_models", []),
                    "leader_rmse": item.get("leader_rmse"),
                    "population": tabarena.get("population", 32),
                    "manifest_sha256": hashlib.sha256(tabarena_manifest.read_bytes()).hexdigest(),
                }
            )
    return [spec for spec in specs if isinstance(spec["dataset"], str)]


def build_dataset(
    spec: dict[str, Any], output_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cohort = str(spec["cohort"])
    dataset = str(spec["dataset"])
    population = int(spec["population"])
    leader_rmse = finite(spec["leader_rmse"])
    dataset_root = Path(spec["run_root"]) / "datasets" / dataset
    checkpoints = sorted((dataset_root / "checkpoints").glob("generation-*.v1.json"))
    records: list[dict[str, Any]] = []
    fresh_ordinal = 0
    best_rmse: float | None = None
    first_beat: dict[str, Any] | None = None
    first_evolutionary_beat: dict[str, Any] | None = None
    completed_generations = 0
    for path in checkpoints:
        checkpoint_bytes = path.read_bytes()
        checkpoint = read_json(path)
        if checkpoint is None:
            continue
        generation = int(checkpoint.get("generation", completed_generations))
        slots = sorted(checkpoint_slots(checkpoint), key=lambda slot: int(slot.get("slot", slot.get("global_slot", 0))))
        if not slots:
            continue
        completed_generations += 1
        for slot_value in slots:
            slot = int(slot_value.get("slot", slot_value.get("global_slot", 0)))
            valid, rmse = score_fields(slot_value)
            genome, semantic, complexity, ancestry = member_fields(slot_value)
            member = slot_value.get("member")
            if not isinstance(member, dict):
                member = slot_value.get("candidate")
            if not isinstance(member, dict):
                member = {}
            metrics = slot_value.get("metrics")
            if not isinstance(metrics, dict):
                metrics = {}
            receipt_file = slot_value.get("receipt_file")
            receipt = (
                read_json(dataset_root / receipt_file)
                if isinstance(receipt_file, str)
                else None
            ) or {}
            cached = slot_value.get("cache_source") is not None
            if not cached:
                fresh_ordinal += 1
            logical_ordinal = generation * population + slot + 1
            beats = bool(valid and leader_rmse is not None and rmse is not None and rmse < leader_rmse)
            record = {
                "schema": LEDGER_SCHEMA,
                "cohort": cohort,
                "dataset": dataset,
                "run_id": spec["run_id"],
                "generation": generation,
                "slot": slot,
                "logical_evaluation": logical_ordinal,
                "fresh_evaluation": None if cached else fresh_ordinal,
                "cached": cached,
                "valid": valid,
                "rmse": rmse,
                "mae": finite(metrics.get("mae")),
                "r2": finite(metrics.get("r2")),
                "human_best_rmse": leader_rmse,
                "beats_human_best": beats,
                "canonical_genome": genome,
                "semantic_sha256": semantic,
                "complexity": complexity,
                "ancestry": ancestry,
                "evolved_descendant": member.get("evolved_descendant"),
                "rust_source_len": member.get("rust_source_len"),
                "failure": slot_value.get("failure"),
                "cache_source": slot_value.get("cache_source"),
                "attempt": slot_value.get("attempt", receipt.get("attempt")),
                "runtime_seconds": finite(receipt.get("runtime_seconds")),
                "binary_sha256": receipt.get("binary_sha256"),
                "train_sha256": receipt.get("train_sha256"),
                "test_sha256": receipt.get("test_sha256"),
                "predictions_sha256": receipt.get("predictions_sha256"),
                "node": slot_value.get("node"),
                "scq_job_id": slot_value.get("scq_job_id"),
                "receipt_file": receipt_file,
                "receipt_sha256": slot_value.get("receipt_sha256"),
                "checkpoint_file": str(path.relative_to(Path(spec["run_root"]))),
                "checkpoint_sha256": sha256_bytes(checkpoint_bytes),
            }
            record["evolutionary_winner_eligible"] = evolutionary_eligible(record)
            records.append(record)
            if valid and rmse is not None and (best_rmse is None or rmse < best_rmse):
                best_rmse = rmse
            if beats and first_beat is None:
                first_beat = {
                    "generation": generation,
                    "logical_evaluation": logical_ordinal,
                    "fresh_evaluation": None if cached else fresh_ordinal,
                    "generation_barrier_evaluations": (generation + 1) * population,
                    "slot": slot,
                    "rmse": rmse,
                    "semantic_sha256": semantic,
                }
            if (
                beats
                and record["evolutionary_winner_eligible"]
                and first_evolutionary_beat is None
            ):
                first_evolutionary_beat = {
                    "generation": generation,
                    "logical_evaluation": logical_ordinal,
                    "fresh_evaluation": None if cached else fresh_ordinal,
                    "generation_barrier_evaluations": (generation + 1) * population,
                    "slot": slot,
                    "rmse": rmse,
                    "semantic_sha256": semantic,
                    "operation": ancestry.get("operation") if isinstance(ancestry, dict) else None,
                }

    valid_scores = sorted(
        float(record["rmse"])
        for record in records
        if record["valid"] and record["rmse"] is not None
    )
    for record in records:
        score = record["rmse"]
        if not record["valid"] or score is None or not valid_scores:
            record["rmse_rank_best_first"] = None
            record["empirical_performance_percentile"] = None
            continue
        left = bisect.bisect_left(valid_scores, float(score))
        record["rmse_rank_best_first"] = left + 1
        record["empirical_performance_percentile"] = (
            100.0 * (len(valid_scores) - left) / len(valid_scores)
        )

    destination = output_root / "datasets" / cohort / dataset
    ledger = b"".join(canonical_json(record) for record in records)
    atomic_write(destination / "evaluations.jsonl", ledger)
    atomic_write(destination / "evaluations.csv", csv_bytes(records))
    scores_sorted = sorted(
        (record for record in records if record["valid"] and record["rmse"] is not None),
        key=lambda record: (float(record["rmse"]), int(record["logical_evaluation"])),
    )
    atomic_write(
        destination / "scores.sorted.jsonl",
        b"".join(canonical_json(record) for record in scores_sorted),
    )
    atomic_write(destination / "scores.sorted.csv", csv_bytes(scores_sorted))
    by_generation: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_generation.setdefault(int(record["generation"]), []).append(record)
    for generation, generation_records in sorted(by_generation.items()):
        atomic_write(
            destination / "generations" / f"generation-{generation:03d}.evaluations.jsonl",
            b"".join(canonical_json(record) for record in generation_records),
        )
    final = read_json(dataset_root / "final-result.v1.json")
    confirmed = bool(final is not None and final.get("outcome") == "confirmed_strict_win")
    if confirmed:
        status = "confirmed_winner"
    elif first_evolutionary_beat is not None:
        status = "evolutionary_threshold_crossed"
    elif first_beat is not None:
        status = "generation_zero_threshold_crossed"
    else:
        status = "running" if records else "planned"
    summary = {
        "schema": "jope-dime.all-regression-dataset-index.v1",
        "cohort": cohort,
        "dataset": dataset,
        "run_id": spec["run_id"],
        "status": status,
        "human_best_models": spec["leader_models"],
        "human_best_rmse": leader_rmse,
        "population": population,
        "generations_completed": completed_generations,
        "logical_evaluations": len(records),
        "fresh_evaluations": fresh_ordinal,
        "valid_evaluations": sum(1 for record in records if record["valid"]),
        "best_rmse": best_rmse,
        "margin_vs_human_best": None if best_rmse is None or leader_rmse is None else leader_rmse - best_rmse,
        "first_human_best_beat": first_beat,
        "first_evolutionary_human_best_beat": first_evolutionary_beat,
        "final_result": final,
        "evaluation_ledger": str((destination / "evaluations.jsonl").relative_to(output_root)),
        "evaluation_csv": str((destination / "evaluations.csv").relative_to(output_root)),
        "scores_sorted_ledger": str(
            (destination / "scores.sorted.jsonl").relative_to(output_root)
        ),
        "evaluation_ledger_sha256": sha256_bytes(ledger),
        "source_manifest_sha256": spec["manifest_sha256"],
        "updated_at_unix_ms": time.time_ns() // 1_000_000,
    }
    atomic_write(destination / "summary.v1.json", canonical_json(summary))
    return summary, records


def update(repo: Path, output_root: Path, campaign_id: str, golden_run: str, tabarena_run: str) -> None:
    specs = dataset_specs(repo, golden_run, tabarena_run)
    built = [build_dataset(spec, output_root) for spec in specs]
    summaries = [summary for summary, _ in built]
    all_records = [record for _, records in built for record in records]
    summaries.sort(key=lambda item: (item["cohort"], item["dataset"]))
    index = {
        "schema": SCHEMA,
        "campaign_id": campaign_id,
        "golden42_run_id": golden_run,
        "tabarena_run_id": tabarena_run,
        "expected_datasets": 55,
        "indexed_datasets": len(summaries),
        "human_bests_beaten": sum(item["first_human_best_beat"] is not None for item in summaries),
        "evolutionary_human_bests_beaten": sum(
            item["first_evolutionary_human_best_beat"] is not None for item in summaries
        ),
        "confirmed_winners": sum(item["status"] == "confirmed_winner" for item in summaries),
        "generation_zero_only_crossings": sum(
            item["first_human_best_beat"] is not None
            and item["first_evolutionary_human_best_beat"] is None
            for item in summaries
        ),
        "total_generations_completed": sum(item["generations_completed"] for item in summaries),
        "total_logical_evaluations": sum(item["logical_evaluations"] for item in summaries),
        "total_fresh_evaluations": sum(item["fresh_evaluations"] for item in summaries),
        "datasets": summaries,
        "updated_at_unix_ms": time.time_ns() // 1_000_000,
    }
    atomic_write(output_root / "index.v1.json", canonical_json(index))
    atomic_write(output_root / "datasets.v1.jsonl", b"".join(canonical_json(item) for item in summaries))
    all_ledger = b"".join(canonical_json(record) for record in all_records)
    atomic_write(output_root / "all-evaluations.jsonl", all_ledger)
    atomic_write(output_root / "all-evaluations.csv", csv_bytes(all_records))
    archive = {
        "schema": ARCHIVE_SCHEMA,
        "campaign_id": campaign_id,
        "evaluation_records": len(all_records),
        "datasets": [
            {
                "cohort": summary["cohort"],
                "dataset": summary["dataset"],
                "records": summary["logical_evaluations"],
                "ledger": summary["evaluation_ledger"],
                "ledger_sha256": summary["evaluation_ledger_sha256"],
            }
            for summary in summaries
        ],
        "all_evaluations_jsonl": "all-evaluations.jsonl",
        "all_evaluations_jsonl_sha256": sha256_bytes(all_ledger),
        "all_evaluations_csv": "all-evaluations.csv",
        "all_evaluations_csv_sha256": sha256_bytes(csv_bytes(all_records)),
        "percentile_definition": (
            "For lower-is-better RMSE, 100 * count(valid RMSE >= this RMSE) / "
            "count(valid RMSE); tied scores receive the same percentile."
        ),
        "updated_at_unix_ms": time.time_ns() // 1_000_000,
    }
    atomic_write(output_root / "archive-manifest.v1.json", canonical_json(archive))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--golden-run", required=True)
    parser.add_argument("--tabarena-run", required=True)
    parser.add_argument("--watch-seconds", type=int, default=0)
    arguments = parser.parse_args()
    while True:
        update(
            arguments.repo.resolve(),
            arguments.output_root.resolve(),
            arguments.campaign_id,
            arguments.golden_run,
            arguments.tabarena_run,
        )
        if arguments.watch_seconds <= 0:
            return
        time.sleep(arguments.watch_seconds)


if __name__ == "__main__":
    main()
