#!/usr/bin/env python3
"""Combine immutable per-campaign evaluation ledgers without deduplication."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import io
import json
import os
import time
from collections import defaultdict
from pathlib import Path


CSV_PREFIX = [
    "cohort", "dataset", "source_campaign_id", "source_campaign_order",
    "source_ledger_line", "combined_process_evaluation", "generation", "slot",
    "logical_evaluation", "fresh_evaluation", "cached", "valid", "rmse", "mae",
    "r2", "combined_rmse_rank_best_first",
    "combined_empirical_performance_percentile", "human_best_rmse",
    "beats_human_best", "canonical_genome", "semantic_sha256", "complexity",
    "evolved_descendant", "ancestry", "failure", "runtime_seconds", "node",
    "scq_job_id", "attempt", "receipt_file", "receipt_sha256", "checkpoint_file",
    "checkpoint_sha256", "binary_sha256", "train_sha256", "test_sha256",
    "predictions_sha256",
]


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(temporary, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def canonical_json(record: dict[str, object]) -> bytes:
    return json.dumps(record, separators=(",", ":"), sort_keys=True).encode() + b"\n"


def csv_bytes(records: list[dict[str, object]]) -> bytes:
    extra = sorted({key for record in records for key in record} - set(CSV_PREFIX))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_PREFIX + extra, extrasaction="ignore")
    writer.writeheader()
    for record in records:
        row = dict(record)
        for key, value in row.items():
            if isinstance(value, (dict, list)):
                row[key] = json.dumps(value, separators=(",", ":"), sort_keys=True)
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


def evolutionary_eligible(record: dict[str, object]) -> bool:
    ancestry = record.get("ancestry")
    if not isinstance(ancestry, dict):
        return False
    parents = ancestry.get("parents")
    semantic = record.get("semantic_sha256")
    explicitly_evolved = record.get("evolved_descendant")
    return bool(
        int(record.get("generation", 0)) >= 1
        and not bool(record.get("cached"))
        and explicitly_evolved is not False
        and ancestry.get("operation") in {"mutation", "crossover"}
        and isinstance(parents, list)
        and parents
        and isinstance(semantic, str)
        and all(parent != semantic for parent in parents)
    )


def build(output_root: Path, inputs: list[Path]) -> None:
    records: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    campaigns: list[dict[str, object]] = []
    for source_order, root in enumerate(inputs):
        index_path = root / "index.v1.json"
        if not index_path.is_file():
            campaigns.append(
                {"root": str(root), "campaign_id": root.name, "indexed": False}
            )
            continue
        source_index = json.loads(index_path.read_text(encoding="utf-8"))
        campaign_id = str(source_index.get("campaign_id", root.name))
        source_records = 0
        for ledger in sorted((root / "datasets").glob("*/*/evaluations.jsonl")):
            for source_line, line in enumerate(
                ledger.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line:
                    continue
                record = json.loads(line)
                record["source_campaign_id"] = campaign_id
                record["source_campaign_root"] = str(root)
                record["source_campaign_order"] = source_order
                record["source_ledger_line"] = source_line
                record["evolutionary_winner_eligible"] = evolutionary_eligible(record)
                records[(str(record["cohort"]), str(record["dataset"]))].append(record)
                source_records += 1
        campaigns.append(
            {
                "root": str(root),
                "campaign_id": campaign_id,
                "indexed": True,
                "evaluation_records": source_records,
            }
        )

    dataset_summaries: list[dict[str, object]] = []
    all_evaluations: list[dict[str, object]] = []
    for (cohort, dataset), evaluations in sorted(records.items()):
        evaluations.sort(
            key=lambda record: (
                int(record["source_campaign_order"]),
                int(record["logical_evaluation"]),
            )
        )
        for ordinal, record in enumerate(evaluations, start=1):
            record["combined_process_evaluation"] = ordinal
        valid_scores = sorted(
            float(record["rmse"])
            for record in evaluations
            if bool(record.get("valid")) and isinstance(record.get("rmse"), (int, float))
        )
        for record in evaluations:
            score = record.get("rmse")
            if not bool(record.get("valid")) or not isinstance(score, (int, float)):
                record["combined_rmse_rank_best_first"] = None
                record["combined_empirical_performance_percentile"] = None
                continue
            left = bisect.bisect_left(valid_scores, float(score))
            record["combined_rmse_rank_best_first"] = left + 1
            record["combined_empirical_performance_percentile"] = (
                100.0 * (len(valid_scores) - left) / len(valid_scores)
            )
        ledger_path = output_root / "datasets" / cohort / dataset / "evaluations.jsonl"
        ledger = b"".join(canonical_json(record) for record in evaluations)
        atomic_write(ledger_path, ledger)
        atomic_write(ledger_path.with_suffix(".csv"), csv_bytes(evaluations))
        valid = [
            record
            for record in evaluations
            if bool(record.get("valid")) and isinstance(record.get("rmse"), (int, float))
        ]
        beats = [record for record in valid if bool(record.get("beats_human_best"))]
        evolutionary_beats = [
            record
            for record in beats
            if bool(record.get("evolutionary_winner_eligible"))
        ]
        scores_sorted = sorted(
            valid,
            key=lambda record: (
                float(record["rmse"]),
                int(record["combined_process_evaluation"]),
            ),
        )
        atomic_write(
            ledger_path.with_name("scores.sorted.jsonl"),
            b"".join(canonical_json(record) for record in scores_sorted),
        )
        atomic_write(
            ledger_path.with_name("scores.sorted.csv"), csv_bytes(scores_sorted)
        )
        campaign_counts: dict[str, int] = defaultdict(int)
        for record in evaluations:
            campaign_counts[str(record["source_campaign_id"])] += 1
        summary = {
            "schema": "jope-dime.combined-regression-dataset-index.v1",
            "cohort": cohort,
            "dataset": dataset,
            "evaluation_ledger": str(ledger_path.relative_to(output_root)),
            "evaluation_csv": str(ledger_path.with_suffix(".csv").relative_to(output_root)),
            "evaluation_ledger_sha256": hashlib.sha256(ledger).hexdigest(),
            "total_process_evaluations": len(evaluations),
            "valid_evaluations": len(valid),
            "best_rmse": min((float(record["rmse"]) for record in valid), default=None),
            "human_best_rmse": next(
                (record.get("human_best_rmse") for record in evaluations), None
            ),
            "first_human_best_beat": (
                {
                    key: beats[0].get(key)
                    for key in (
                        "combined_process_evaluation",
                        "source_campaign_id",
                        "logical_evaluation",
                        "fresh_evaluation",
                        "generation",
                        "slot",
                        "rmse",
                        "semantic_sha256",
                    )
                }
                if beats
                else None
            ),
            "first_evolutionary_human_best_beat": (
                {
                    key: evolutionary_beats[0].get(key)
                    for key in (
                        "combined_process_evaluation",
                        "source_campaign_id",
                        "logical_evaluation",
                        "fresh_evaluation",
                        "generation",
                        "slot",
                        "rmse",
                        "semantic_sha256",
                        "ancestry",
                    )
                }
                if evolutionary_beats
                else None
            ),
            "campaign_evaluation_counts": dict(sorted(campaign_counts.items())),
        }
        atomic_write(
            ledger_path.with_name("summary.v1.json"),
            (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(),
        )
        dataset_summaries.append(summary)
        all_evaluations.extend(evaluations)

    index = {
        "schema": "jope-dime.combined-regression-index.v1",
        "updated_at_unix_ms": time.time_ns() // 1_000_000,
        "campaigns": campaigns,
        "indexed_datasets": len(dataset_summaries),
        "evaluation_records": sum(
            int(summary["total_process_evaluations"]) for summary in dataset_summaries
        ),
        "datasets_with_any_threshold_crossing": sum(
            summary["first_human_best_beat"] is not None for summary in dataset_summaries
        ),
        "datasets_with_evolutionary_threshold_crossing": sum(
            summary["first_evolutionary_human_best_beat"] is not None
            for summary in dataset_summaries
        ),
        "datasets": dataset_summaries,
    }
    atomic_write(
        output_root / "index.v1.json",
        (json.dumps(index, indent=2, sort_keys=True) + "\n").encode(),
    )
    all_ledger = b"".join(canonical_json(record) for record in all_evaluations)
    all_csv = csv_bytes(all_evaluations)
    atomic_write(output_root / "all-evaluations.jsonl", all_ledger)
    atomic_write(output_root / "all-evaluations.csv", all_csv)
    archive = {
        "schema": "jope-dime.combined-regression-analysis-archive.v1",
        "updated_at_unix_ms": time.time_ns() // 1_000_000,
        "campaigns": campaigns,
        "datasets": [
            {
                "cohort": summary["cohort"],
                "dataset": summary["dataset"],
                "records": summary["total_process_evaluations"],
                "ledger": summary["evaluation_ledger"],
                "ledger_sha256": summary["evaluation_ledger_sha256"],
            }
            for summary in dataset_summaries
        ],
        "evaluation_records": len(all_evaluations),
        "all_evaluations_jsonl": "all-evaluations.jsonl",
        "all_evaluations_jsonl_sha256": hashlib.sha256(all_ledger).hexdigest(),
        "all_evaluations_csv": "all-evaluations.csv",
        "all_evaluations_csv_sha256": hashlib.sha256(all_csv).hexdigest(),
        "percentile_definition": (
            "Within each dataset across all source campaigns, for lower-is-better RMSE: "
            "100 * count(valid RMSE >= this RMSE) / count(valid RMSE)."
        ),
    }
    atomic_write(output_root / "archive-manifest.v1.json", canonical_json(archive))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--watch-seconds", type=float)
    args = parser.parse_args()
    while True:
        build(args.output_root, args.input)
        if args.watch_seconds is None:
            return 0
        time.sleep(args.watch_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
