#!/usr/bin/env python3
"""Freeze and qualify all-dataset winner candidates without publishing genomes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from pathlib import Path

from test_exact_common import (
    atomic_json,
    sha256_file,
    validate_catalog_counts,
)
from test_exact_adaptive import write_gene_evidence

ROOT = Path("/home/ubuntu/jope-flywheel/test_exact_gen0")
SOTARENA = Path("/home/ubuntu/sotarena")
TOLERANCE = 1e-12
QUALIFICATION_STATES = {
    "discovered",
    "data-hash-mismatch",
    "replay-pending",
    "runtime-parity-pending",
    "qualified",
    "superseded",
    "published",
}


def _selected_scores(sotarena: Path) -> dict[tuple[str, str], float]:
    result = {}
    paths = (
        (
            "regression",
            sotarena
            / "results/tabular-regression-golden42-test-adaptive-v1/scores.csv",
            "jope-dime",
        ),
        (
            "binary",
            sotarena / "results/tabular-binary-76-champions-v1/scores.csv",
            "aspfm-binary-76-champions-v1",
        ),
    )
    for task, path, method in paths:
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["method_id"] == method and row["status"] == "ok":
                    result[(task, row["dataset_id"])] = float(row["value"])
    return result


def _private_incumbents(path: Path | None) -> dict[tuple[str, str], dict]:
    if path is None or not path.is_file():
        return {}
    value = json.loads(path.read_text())
    return {
        (row["task"], row["dataset_hash"]): row
        for row in value.get("datasets", [])
    }


def strictly_improves(task: str, candidate: float, incumbent: float) -> bool:
    scale = max(1.0, abs(candidate), abs(incumbent))
    margin = TOLERANCE * scale
    return (
        candidate < incumbent - margin
        if task == "regression"
        else candidate > incumbent + margin
    )


def scan(
    *,
    corpus_path: Path,
    catalog_path: Path,
    output_root: Path,
    accepted_checkpoint_sha256: str,
    evaluator_sha256: str,
    prior_ledger: Path | None = None,
    sotarena: Path = SOTARENA,
) -> dict:
    catalog = json.loads(catalog_path.read_text())
    declared = catalog.get("task_counts") or {
        "regression": catalog.get("regression_count"),
        "binary": catalog.get("binary_count"),
    }
    counts = validate_catalog_counts(catalog["entries"], declared)
    selected = _selected_scores(sotarena)
    prior = _private_incumbents(prior_ledger)
    database = sqlite3.connect(f"file:{corpus_path}?mode=ro", uri=True)
    database.row_factory = sqlite3.Row
    watermark = int(
        database.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    )
    datasets = []
    discovered = selected_overlap_wins = 0
    for entry in sorted(
        catalog["entries"], key=lambda row: (row["task"], row["dataset_hash"])
    ):
        task, dataset = entry["task"], entry["dataset_hash"]
        observation = database.execute(
            """
            SELECT * FROM observations
            WHERE task=? AND dataset_hash=?
            ORDER BY score DESC,semantic_sha256
            LIMIT 1
            """,
            (task, dataset),
        ).fetchone()
        candidate = dict(observation) if observation is not None else None
        if candidate is not None:
            raw_primary = (
                candidate["secondary_score"]
                if task == "regression"
                else candidate["score"]
            )
            primary = None if raw_primary is None else float(raw_primary)
            if primary is not None and not math.isfinite(primary):
                raise ValueError(f"{task}:{dataset} candidate metric is non-finite")
        else:
            primary = None
        prior_record = prior.get((task, dataset))
        prior_primary = (
            None
            if prior_record is None
            else prior_record.get("retained_primary_metric")
        )
        published_primary = selected.get((task, dataset))
        incumbents = [
            float(value)
            for value in (prior_primary, published_primary)
            if value is not None
        ]
        actual_incumbent = (
            (min(incumbents) if task == "regression" else max(incumbents))
            if incumbents
            else None
        )
        win = (
            primary is not None
            and actual_incumbent is not None
            and strictly_improves(task, primary, actual_incumbent)
        )
        state = "discovered" if win else "superseded"
        if win:
            discovered += 1
            selected_overlap_wins += int(published_primary is not None)
        record = {
            "task": task,
            "dataset_hash": dataset,
            "train_sha256": entry["train_sha256"],
            "test_sha256": entry["test_sha256"],
            "source_train_csv": entry["train_csv"],
            "source_test_csv": entry["test_csv"],
            "source_family": entry["source_family"],
            "catalog_index": entry["catalog_index"],
            "candidate": candidate,
            "candidate_primary_metric": primary,
            "candidate_r2": (
                float(candidate["score"])
                if candidate is not None and task == "regression"
                else None
            ),
            "candidate_rmse": (
                float(candidate["secondary_score"])
                if candidate is not None
                and task == "regression"
                and candidate["secondary_score"] is not None
                else None
            ),
            "candidate_auc": (
                float(candidate["score"])
                if candidate is not None and task == "binary"
                else None
            ),
            "previous_private_primary_metric": prior_primary,
            "published_selected_primary_metric": published_primary,
            "actual_incumbent_primary_metric": actual_incumbent,
            "retained_primary_metric": (
                primary if win or actual_incumbent is None else actual_incumbent
            ),
            "qualification_state": state,
            "strictly_improves": win,
            "requires_prediction_replay": win,
            "requires_fitted_cpp_parity": win and published_primary is not None,
        }
        datasets.append(record)
    database.close()
    ledger = {
        "schema": "JopeAllDatasetWinnerLedger.v1",
        "corpus_watermark": watermark,
        "catalog_sha256": sha256_file(catalog_path),
        "accepted_checkpoint_sha256": accepted_checkpoint_sha256,
        "evaluator_sha256": evaluator_sha256,
        "dataset_counts": counts,
        "dataset_count": len(datasets),
        "discovered": discovered,
        "selected_cohort_discovered": selected_overlap_wins,
        "test_adaptive": True,
        "benchmark_valid": False,
        "selection_bias": "test-exact-reused",
        "datasets": datasets,
    }
    identity = hashlib.sha256(
        json.dumps(ledger, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    final = output_root / identity
    final.mkdir(parents=True, exist_ok=True)
    private_path = final / "all-dataset-winners.private.json"
    if not private_path.exists():
        atomic_json(private_path, ledger)
    public = {
        **{key: value for key, value in ledger.items() if key != "datasets"},
        "private_ledger_sha256": sha256_file(private_path),
        "datasets": [
            {
                "task": row["task"],
                "dataset_hash": row["dataset_hash"],
                "train_sha256": row["train_sha256"],
                "test_sha256": row["test_sha256"],
                "semantic_sha256": (
                    row["candidate"]["semantic_sha256"] if row["candidate"] else None
                ),
                "primary_metric": row["candidate_primary_metric"],
                "published_selected_primary_metric": row[
                    "published_selected_primary_metric"
                ],
                "actual_incumbent_primary_metric": row[
                    "actual_incumbent_primary_metric"
                ],
                "gain": (
                    None
                    if not row["strictly_improves"]
                    else (
                        row["actual_incumbent_primary_metric"]
                        - row["candidate_primary_metric"]
                        if row["task"] == "regression"
                        else row["candidate_primary_metric"]
                        - row["actual_incumbent_primary_metric"]
                    )
                ),
                "status": row["qualification_state"],
                "encrypted_record_ref": (
                    f"all-dataset-winners.private.json#datasets/{index}"
                ),
            }
            for index, row in enumerate(datasets)
        ],
    }
    atomic_json(final / "winner-scan.public.json", public)
    evidence_rows = []
    for row in datasets:
        candidate = row["candidate"]
        if candidate is None or not candidate.get("protocol_sha256"):
            continue
        evidence_rows.append(
            {
                **candidate,
                "metric": "r2" if row["task"] == "regression" else "auc",
                "within_dataset_rank": 1,
                "within_dataset_percentile": 1.0,
                "regret_to_incumbent": (
                    None
                    if row["task"] == "regression"
                    or row["actual_incumbent_primary_metric"] is None
                    else row["actual_incumbent_primary_metric"]
                    - float(candidate["score"])
                ),
                "checkpoint_epoch": (
                    f"checkpoint-{candidate['model_checkpoint_sha256']}"
                ),
                "compiled_parity_state": (
                    "pending"
                    if row["requires_fitted_cpp_parity"]
                    else "not_required"
                ),
                "qualification_state": row["qualification_state"],
                "validity": "valid",
            }
        )
    gene_export = write_gene_evidence(
        evidence_rows, final / "gene-evidence.v2.jsonl"
    )
    atomic_json(
        output_root / "current.json",
        {
            "ledger": str(private_path),
            "ledger_sha256": sha256_file(private_path),
            "public": str(final / "winner-scan.public.json"),
            "public_sha256": sha256_file(final / "winner-scan.public.json"),
            "gene_evidence": gene_export["output"],
            "gene_evidence_sha256": gene_export["output_sha256"],
            "gene_evidence_rows": gene_export["rows"],
        },
    )
    return public


def qualify(record: dict, evidence: dict) -> dict:
    if record["qualification_state"] not in {"discovered", "replay-pending", "runtime-parity-pending"}:
        raise ValueError("winner is not in a qualifiable state")
    if any(
        evidence.get(field) is not True
        for field in (
            "dataset_identity",
            "row_alignment",
            "finite_predictions",
            "two_replays_identical",
            "evaluator_identity",
            "receipt_semantic_agreement",
            "strictly_beats_published",
            "no_parser_fallback",
        )
    ):
        return {**record, "qualification_state": "replay-pending"}
    if record["requires_fitted_cpp_parity"] and evidence.get("fitted_cpp_parity") is not True:
        return {**record, "qualification_state": "runtime-parity-pending"}
    hashes = evidence.get("prediction_sha256s") or []
    if len(hashes) != 2 or hashes[0] != hashes[1]:
        return {**record, "qualification_state": "replay-pending"}
    return {
        **record,
        "qualification_state": "qualified",
        "qualification_evidence_sha256": hashlib.sha256(
            json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=ROOT / "corpus/strict-test-exact.sqlite")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "winner-scans")
    parser.add_argument("--accepted-checkpoint-sha256", required=True)
    parser.add_argument("--evaluator-sha256", required=True)
    parser.add_argument("--prior-ledger", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            scan(
                corpus_path=args.corpus,
                catalog_path=args.catalog,
                output_root=args.output_root,
                accepted_checkpoint_sha256=args.accepted_checkpoint_sha256,
                evaluator_sha256=args.evaluator_sha256,
                prior_ledger=args.prior_ledger,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
