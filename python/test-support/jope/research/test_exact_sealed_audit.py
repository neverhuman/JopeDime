#!/usr/bin/env python3
"""Sealed, one-shot audit registry isolated from the adaptive flywheel."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Mapping

from test_exact_common import atomic_json, sha256_file, sha256_text

ROOT = Path("/home/ubuntu/jope-flywheel/test_exact_gen0")
DEFAULT_REGISTRY = ROOT / "sealed-audit" / "registry.sqlite"


class SealedAuditRegistry:
    def __init__(self, path: Path = DEFAULT_REGISTRY):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS datasets(
              audit_id TEXT PRIMARY KEY,
              task TEXT NOT NULL,
              dataset_hash TEXT NOT NULL,
              source_family TEXT NOT NULL,
              train_sha256 TEXT NOT NULL,
              test_sha256 TEXT NOT NULL,
              incumbent_score REAL NOT NULL,
              checkpoint_sha256 TEXT NOT NULL,
              state TEXT NOT NULL,
              canonical_genome TEXT,
              semantic_sha256 TEXT,
              emitted_at_ms INTEGER,
              score REAL,
              receipt_sha256 TEXT,
              compiled_interpreted_parity INTEGER,
              prediction_parity INTEGER,
              scored_at_ms INTEGER,
              UNIQUE(task,dataset_hash),
              UNIQUE(task,dataset_hash,checkpoint_sha256)
            );
            """
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def register(
        self,
        entry: Mapping[str, object],
        *,
        checkpoint_sha256: str,
        forbidden_datasets: set[tuple[str, str]],
        forbidden_families: set[str],
    ) -> str:
        identity = (str(entry["task"]), str(entry["dataset_hash"]))
        family = str(entry["source_family"])
        if identity in forbidden_datasets:
            raise ValueError("sealed dataset appeared in adaptive evidence")
        if family in forbidden_families:
            raise ValueError("sealed dataset source family appeared in training/selection")
        audit_id = sha256_text(
            json.dumps(
                {
                    "checkpoint_sha256": checkpoint_sha256,
                    "dataset_hash": identity[1],
                    "source_family": family,
                    "task": identity[0],
                    "test_sha256": entry["test_sha256"],
                    "train_sha256": entry["train_sha256"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        self.db.execute(
            """
            INSERT INTO datasets(
              audit_id,task,dataset_hash,source_family,train_sha256,
              test_sha256,incumbent_score,checkpoint_sha256,state
            ) VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (
                audit_id,
                *identity,
                family,
                entry["train_sha256"],
                entry["test_sha256"],
                float(entry["incumbent_score"]),
                checkpoint_sha256,
                "registered",
            ),
        )
        self.db.commit()
        return audit_id

    def emit(self, audit_id: str, canonical_genome: str) -> dict:
        row = self.db.execute(
            "SELECT * FROM datasets WHERE audit_id=?", (audit_id,)
        ).fetchone()
        if row is None or row["state"] != "registered":
            raise RuntimeError("sealed audit permits exactly one pre-score emission")
        semantic = sha256_text(canonical_genome)
        emitted_at = int(time.time() * 1000)
        self.db.execute(
            """
            UPDATE datasets SET state='emitted',canonical_genome=?,
              semantic_sha256=?,emitted_at_ms=? WHERE audit_id=?
            """,
            (canonical_genome, semantic, emitted_at, audit_id),
        )
        self.db.commit()
        return {
            "audit_id": audit_id,
            "semantic_sha256": semantic,
            "emitted_at_ms": emitted_at,
            "deterministic_greedy": True,
            "emissions": 1,
        }

    def score(
        self,
        audit_id: str,
        *,
        score: float,
        receipt_sha256: str,
        compiled_interpreted_parity: bool,
        prediction_parity: bool,
    ) -> dict:
        row = self.db.execute(
            "SELECT * FROM datasets WHERE audit_id=?", (audit_id,)
        ).fetchone()
        if row is None or row["state"] != "emitted":
            raise RuntimeError("sealed audit must be emitted exactly once before scoring")
        strict_win = float(score) > float(row["incumbent_score"])
        super_human = (
            strict_win and compiled_interpreted_parity and prediction_parity
        )
        self.db.execute(
            """
            UPDATE datasets SET state='scored',score=?,receipt_sha256=?,
              compiled_interpreted_parity=?,prediction_parity=?,scored_at_ms=?
            WHERE audit_id=?
            """,
            (
                float(score),
                receipt_sha256,
                int(compiled_interpreted_parity),
                int(prediction_parity),
                int(time.time() * 1000),
                audit_id,
            ),
        )
        self.db.commit()
        return {
            "audit_id": audit_id,
            "strictly_beats_frozen_incumbent": strict_win,
            "compiled_interpreted_parity": compiled_interpreted_parity,
            "prediction_parity": prediction_parity,
            "super_human": super_human,
            "training_ingestion_eligible": False,
            "sampling_decision_eligible": False,
        }

    def status(self, checkpoint_sha256: str | None = None) -> dict:
        where = "" if checkpoint_sha256 is None else " WHERE checkpoint_sha256=?"
        args = () if checkpoint_sha256 is None else (checkpoint_sha256,)
        rows = self.db.execute(
            f"SELECT * FROM datasets{where} ORDER BY task,dataset_hash", args
        ).fetchall()
        return {
            "schema": "JopeSealedAuditStatus.v1",
            "checkpoint_sha256": checkpoint_sha256,
            "eligible_unseen_datasets": len(rows),
            "zero_shot_evidence": "available" if rows else "unavailable",
            "states": {
                state: sum(row["state"] == state for row in rows)
                for state in ("registered", "emitted", "scored")
            },
            "strict_super_human": sum(
                row["state"] == "scored"
                and float(row["score"]) > float(row["incumbent_score"])
                and bool(row["compiled_interpreted_parity"])
                and bool(row["prediction_parity"])
                for row in rows
            ),
            "isolated_from_training": True,
        }


def adaptive_identities(root: Path = ROOT) -> tuple[set[tuple[str, str]], set[str]]:
    datasets: set[tuple[str, str]] = set()
    families: set[str] = set()
    pointer_path = root / "catalog" / "current.json"
    if pointer_path.is_file():
        pointer = json.loads(pointer_path.read_text())
        catalog = json.loads(Path(pointer["catalog"]).read_text())
        for row in catalog["entries"]:
            datasets.add((row["task"], row["dataset_hash"]))
            families.add(row["source_family"])
    corpus_path = root / "corpus" / "strict-test-exact.sqlite"
    if corpus_path.is_file():
        corpus = sqlite3.connect(f"file:{corpus_path}?mode=ro", uri=True)
        datasets.update(
            (row[0], row[1])
            for row in corpus.execute(
                "SELECT DISTINCT task,dataset_hash FROM observations"
            )
        )
        corpus.close()
    return datasets, families


def milestone_status(checkpoint_sha256: str, root: Path = ROOT) -> dict:
    registry = SealedAuditRegistry(root / "sealed-audit" / "registry.sqlite")
    try:
        status = registry.status(checkpoint_sha256)
    finally:
        registry.close()
    path = root / "sealed-audit" / f"{checkpoint_sha256}.status.json"
    atomic_json(path, status)
    return {
        **status,
        "status": str(path),
        "status_sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status")
    status.add_argument("--checkpoint-sha256")
    args = parser.parse_args()
    registry = SealedAuditRegistry(args.registry)
    try:
        print(json.dumps(registry.status(args.checkpoint_sha256), sort_keys=True))
    finally:
        registry.close()


if __name__ == "__main__":
    main()
