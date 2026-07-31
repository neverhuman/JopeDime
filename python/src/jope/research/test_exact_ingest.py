#!/usr/bin/env python3
"""Fetch, validate, ingest, and atomically index strict test-only receipts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

from test_exact_common import (
    atomic_json,
    is_strict_receipt,
    manifest_authority_kind,
    receipt_authority_kind,
    sha256_file,
    sha256_text,
)
from test_exact_pair_registry import PairRegistry, protocol_for_manifest

SCQ = Path("/home/ubuntu/smart_cluster/target/release/scq")
ROOT = Path("/home/ubuntu/jope-flywheel/test_exact_gen0")
BACKLOG = ROOT / "ops" / "backlog.sqlite"
CORPUS = ROOT / "corpus" / "strict-test-exact.sqlite"
RECEIPTS = ROOT / "receipts"
SCQ_RUNS = Path("/home/ubuntu/smart_cluster/var/runs")
HOF_POINTER = ROOT / "hof" / "current.json"
GOLDEN42 = Path("/home/ubuntu/golden42/golden42_foundry/data/panel.json")
CATALOG_POINTER = ROOT / "catalog" / "current.json"


def command(args: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, capture_output=True, text=True, timeout=timeout, check=False
    )


def scheduler_jobs() -> dict[str, dict]:
    result = command([str(SCQ), "--json", "ls"])
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    rows = json.loads(result.stdout).get("Jobs", {}).get("jobs", [])
    return {row["id"]: row for row in rows if row.get("id")}


def corpus_db() -> sqlite3.Connection:
    CORPUS.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(CORPUS)
    con.row_factory = sqlite3.Row
    con.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS observations(
          task TEXT NOT NULL,dataset_hash TEXT NOT NULL,
          semantic_sha256 TEXT NOT NULL,canonical_genome TEXT NOT NULL,
          candidate_source TEXT NOT NULL,
          metric TEXT NOT NULL,score REAL NOT NULL,
          secondary_metric TEXT,secondary_score REAL,
          runtime_seconds REAL NOT NULL,
          train_sha256 TEXT NOT NULL,test_sha256 TEXT NOT NULL,
          receipt_sha256 TEXT NOT NULL,
          receipt_path TEXT NOT NULL,manifest_sha256 TEXT NOT NULL,
          frozen_seed_sha256 TEXT NOT NULL,
          model_checkpoint_sha256 TEXT NOT NULL,
          hall_of_fame_sha256 TEXT NOT NULL,
          created_at_ms INTEGER NOT NULL,
          authority_kind TEXT NOT NULL DEFAULT 'legacy-superseded',
          selection_bias TEXT,
          strict_wave TEXT,
          predictions_path TEXT,
          predictions_sha256 TEXT,
          prediction_count INTEGER,
          balanced_accuracy REAL,
          protocol_sha256 TEXT,
          PRIMARY KEY(task,dataset_hash,semantic_sha256)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS failures(
          task TEXT,dataset_hash TEXT,semantic_sha256 TEXT,
          failure TEXT,failure_message TEXT,runtime_seconds REAL,
          receipt_sha256 TEXT,created_at_ms INTEGER,
          PRIMARY KEY(task,dataset_hash,semantic_sha256,receipt_sha256)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS ingested_jobs(
          job_id TEXT PRIMARY KEY,manifest_sha256 TEXT NOT NULL,
          receipt_sha256 TEXT NOT NULL,rows INTEGER NOT NULL,
          valid INTEGER NOT NULL,ingested_at_ms INTEGER NOT NULL
        ) WITHOUT ROWID;
        """
    )
    columns = {
        row[1] for row in con.execute("PRAGMA table_info(observations)").fetchall()
    }
    for name, declaration in (
        ("authority_kind", "TEXT NOT NULL DEFAULT 'legacy-superseded'"),
        ("selection_bias", "TEXT"),
        ("strict_wave", "TEXT"),
        ("predictions_path", "TEXT"),
        ("predictions_sha256", "TEXT"),
        ("prediction_count", "INTEGER"),
        ("balanced_accuracy", "REAL"),
        ("protocol_sha256", "TEXT"),
    ):
        if name not in columns:
            con.execute(f"ALTER TABLE observations ADD COLUMN {name} {declaration}")
    con.commit()
    return con


def receipt_dir(fetch_root: Path) -> Path:
    direct = fetch_root / "receipt"
    if direct.is_dir():
        return direct
    matches = [path for path in fetch_root.rglob("SHARD_OK") if path.parent.is_dir()]
    if len(matches) != 1:
        raise RuntimeError(f"could not resolve receipt output under {fetch_root}")
    return matches[0].parent


def local_receipt_dir(job_id: str, runs: Path = SCQ_RUNS) -> Path | None:
    attempts = runs / job_id / "attempts"
    if not attempts.is_dir():
        return None
    for attempt in sorted(attempts.iterdir(), reverse=True):
        receipt = attempt / "outputs" / "receipt"
        if (receipt / "SHARD_OK").is_file():
            return receipt
    return None


def validate_and_ingest(
    job_id: str,
    output: Path,
    con: sqlite3.Connection,
    *,
    include_pair_results: bool = False,
) -> dict:
    portable = output / ".portable-manifest.json"
    if not portable.is_file() or not (output / "SHARD_OK").is_file():
        raise RuntimeError(f"{job_id}: portable manifest or SHARD_OK missing")
    manifest = json.loads(portable.read_text())
    candidate_by_semantic = {
        row["semantic_sha256"]: row for row in manifest["candidates"]
    }
    expected = set(candidate_by_semantic)
    authority = manifest["label_authority"]
    protocol_sha256 = protocol_for_manifest(manifest)
    authority_kind = manifest_authority_kind(authority, task=manifest["task"])
    if authority_kind is None:
        raise RuntimeError(f"{job_id}: invalid manifest label authority")
    path = output / f"{manifest['dataset_hash']}.receipts.jsonl"
    receipt_hash = sha256_file(path)
    rows, valid, failed = [], [], []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            if row.get("semantic_sha256") not in expected:
                raise RuntimeError(f"{job_id}:{line_number}: unknown candidate")
            if any(
                row.get(field) != authority.get(field)
                for field in (
                    "frozen_seed_sha256",
                    "model_checkpoint_sha256",
                    "hall_of_fame_sha256",
                )
            ):
                raise RuntimeError(f"{job_id}:{line_number}: provenance pin mismatch")
            if row.get("valid"):
                if not is_strict_receipt(row, allow_legacy=True):
                    raise RuntimeError(f"{job_id}:{line_number}: receipt is not strict")
                if receipt_authority_kind(row) != authority_kind:
                    raise RuntimeError(
                        f"{job_id}:{line_number}: receipt authority generation drift"
                    )
                if authority_kind == "corrected":
                    relative = Path(str(row.get("predictions_path") or ""))
                    prediction_path = (output / relative).resolve()
                    prediction_count = int(row.get("prediction_count") or 0)
                    if (
                        relative.is_absolute()
                        or not prediction_path.is_relative_to(output.resolve())
                        or not prediction_path.is_file()
                    ):
                        raise RuntimeError(
                            f"{job_id}:{line_number}: missing or unsafe prediction evidence"
                        )
                    if (
                        prediction_count <= 0
                        or prediction_path.stat().st_size != prediction_count * 8
                    ):
                        raise RuntimeError(
                            f"{job_id}:{line_number}: prediction row alignment drift"
                        )
                    if sha256_file(prediction_path) != row.get("predictions_sha256"):
                        raise RuntimeError(
                            f"{job_id}:{line_number}: prediction evidence hash drift"
                        )
                    row["_resolved_predictions_path"] = str(prediction_path)
                valid.append(row)
            else:
                if any(
                    row.get(field) != authority.get(field)
                    for field in (
                        "test_adaptive",
                        "benchmark_valid",
                        "selection_bias",
                        "generation",
                    )
                ):
                    raise RuntimeError(f"{job_id}:{line_number}: failure receipt authority drift")
                failed.append(row)
    if len(rows) != len(expected):
        raise RuntimeError(f"{job_id}: expected {len(expected)} rows, found {len(rows)}")
    for row in valid:
        existing = con.execute(
            """
            SELECT score,receipt_sha256,protocol_sha256 FROM observations
            WHERE task=? AND dataset_hash=? AND semantic_sha256=?
            """,
            (row["task"], row["dataset_hash"], row["semantic_sha256"]),
        ).fetchone()
        values = (
            row["task"],
            row["dataset_hash"],
            row["semantic_sha256"],
            row["canonical_genome"],
            str(candidate_by_semantic[row["semantic_sha256"]].get("source") or "unknown"),
            row["metric"],
            float(row["score"]),
            row.get("secondary_metric"),
            row.get("secondary_score"),
            float(row["runtime_seconds"]),
            row["train_sha256"],
            row["test_sha256"],
            receipt_hash,
            str(path),
            row["manifest_sha256"],
            row["frozen_seed_sha256"],
            row["model_checkpoint_sha256"],
            row["hall_of_fame_sha256"],
            int(row["created_at_ms"]),
            authority_kind,
            row.get("selection_bias"),
            row["generation"] if authority_kind == "corrected" else None,
            row.get("_resolved_predictions_path"),
            row.get("predictions_sha256"),
            row.get("prediction_count"),
            row.get("balanced_accuracy"),
            protocol_sha256,
        )
        if existing is None:
            con.execute(
                """
                INSERT INTO observations(
                  task,dataset_hash,semantic_sha256,canonical_genome,
                  candidate_source,metric,score,secondary_metric,secondary_score,
                  runtime_seconds,train_sha256,test_sha256,receipt_sha256,
                  receipt_path,manifest_sha256,frozen_seed_sha256,
                  model_checkpoint_sha256,hall_of_fame_sha256,created_at_ms,
                  authority_kind,selection_bias,strict_wave,predictions_path,
                  predictions_sha256,prediction_count,balanced_accuracy
                  ,protocol_sha256
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                values,
            )
        elif (
            existing["protocol_sha256"] != protocol_sha256
            or (float(row["score"]), receipt_hash)
            > (float(existing["score"]), existing["receipt_sha256"])
        ):
            con.execute(
                """
                REPLACE INTO observations(
                  task,dataset_hash,semantic_sha256,canonical_genome,
                  candidate_source,metric,score,secondary_metric,secondary_score,
                  runtime_seconds,train_sha256,test_sha256,receipt_sha256,
                  receipt_path,manifest_sha256,frozen_seed_sha256,
                  model_checkpoint_sha256,hall_of_fame_sha256,created_at_ms,
                  authority_kind,selection_bias,strict_wave,predictions_path,
                  predictions_sha256,prediction_count,balanced_accuracy
                  ,protocol_sha256
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                values,
            )
    con.executemany(
        "INSERT OR IGNORE INTO failures VALUES(?,?,?,?,?,?,?,?)",
        [
            (
                row["task"],
                row["dataset_hash"],
                row["semantic_sha256"],
                row.get("failure"),
                row.get("failure_message"),
                float(row["runtime_seconds"]),
                receipt_hash,
                int(row["created_at_ms"]),
            )
            for row in failed
        ],
    )
    con.execute(
        "INSERT INTO ingested_jobs VALUES(?,?,?,?,?,?)",
        (
            job_id,
            rows[0]["manifest_sha256"],
            receipt_hash,
            len(rows),
            len(valid),
            int(time.time() * 1000),
        ),
    )
    result = {
        "rows": len(rows),
        "valid": len(valid),
        "failed": len(failed),
    }
    if include_pair_results:
        result["pair_results"] = [
            {
                "task": row["task"],
                "dataset_hash": row["dataset_hash"],
                "semantic_sha256": row["semantic_sha256"],
                "protocol_sha256": protocol_sha256,
                "receipt_sha256": receipt_hash,
                "failure": None if row.get("valid") else row.get("failure"),
            }
            for row in rows
        ]
    return result


def reconcile(limit: int) -> dict:
    if not BACKLOG.is_file():
        return {"completed": 0, "ingested": 0, "failed": 0}
    jobs = scheduler_jobs()
    backlog = sqlite3.connect(BACKLOG, timeout=60.0)
    backlog.row_factory = sqlite3.Row
    backlog.execute("PRAGMA busy_timeout=60000")
    corpus = corpus_db()
    registry = PairRegistry()
    already = {row[0] for row in corpus.execute("SELECT job_id FROM ingested_jobs")}
    candidates = []
    for row in backlog.execute(
        "SELECT job_id,manifest_sha256 FROM claims "
        "WHERE state='submitted' "
        "OR state LIKE 'ingest_failed:%no such job%' "
        "OR state LIKE 'ingest_failed:%database is locked%' "
        "ORDER BY job_id"
    ):
        job_id = row["job_id"]
        if job_id in already:
            continue
        if (
            (RECEIPTS / job_id).is_dir()
            or local_receipt_dir(job_id) is not None
            or jobs.get(job_id, {}).get("state") == "completed"
        ):
            candidates.append(row)
        if len(candidates) == limit:
            break
    ingested = failures = rows_total = valid_total = 0
    for claim in candidates:
        job_id = claim["job_id"]
        final = RECEIPTS / job_id
        try:
            if not final.is_dir():
                final.parent.mkdir(parents=True, exist_ok=True)
                local = local_receipt_dir(job_id)
                if local is not None:
                    temporary = RECEIPTS / f".local-{job_id}-{os.getpid()}"
                    shutil.copytree(local, temporary)
                    os.replace(temporary, final)
                else:
                    temporary = RECEIPTS / f".fetch-{job_id}-{os.getpid()}"
                    temporary.mkdir(parents=True, exist_ok=False)
                    result = command(
                        [str(SCQ), "fetch", job_id, "--output", str(temporary)]
                    )
                    if result.returncode:
                        raise RuntimeError((result.stderr or result.stdout).strip())
                    source = receipt_dir(temporary)
                    final.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(source, final)
                    shutil.rmtree(temporary)
            summary = validate_and_ingest(
                job_id, final, corpus, include_pair_results=True
            )
            for pair in summary["pair_results"]:
                registry.record_result(**pair)
            corpus.commit()
            backlog.execute(
                "UPDATE claims SET state='ingested',updated_at_ms=? WHERE job_id=?",
                (int(time.time() * 1000), job_id),
            )
            backlog.commit()
            ingested += 1
            rows_total += summary["rows"]
            valid_total += summary["valid"]
        except Exception as error:
            corpus.rollback()
            failures += 1
            backlog.execute(
                "UPDATE claims SET state=?,updated_at_ms=? WHERE job_id=?",
                (f"ingest_failed:{str(error)[:500]}", int(time.time() * 1000), job_id),
            )
            backlog.commit()
    corpus.close()
    registry.close()
    backlog.close()
    indexes = rebuild_indexes()
    result = {
        "completed_seen": len(candidates),
        "ingested": ingested,
        "failures": failures,
        "rows": rows_total,
        "valid": valid_total,
        **indexes,
    }
    atomic_json(ROOT / "ops" / "last-ingest.json", result)
    return result


def rebuild_indexes() -> dict:
    con = corpus_db()
    hof_pointer = json.loads(HOF_POINTER.read_text())
    known = sqlite3.connect(f"file:{hof_pointer['known']}?mode=ro", uri=True)
    known.row_factory = sqlite3.Row
    golden_ids = set(json.loads(GOLDEN42.read_text())["dataset_ids"])
    champions, wins = [], []
    datasets = con.execute(
        "SELECT DISTINCT task,dataset_hash FROM observations ORDER BY task,dataset_hash"
    ).fetchall()
    for dataset in datasets:
        task, dataset_hash = dataset
        new = con.execute(
            """
            SELECT * FROM observations WHERE task=? AND dataset_hash=?
            ORDER BY score DESC,semantic_sha256 LIMIT 1
            """,
            (task, dataset_hash),
        ).fetchone()
        old = known.execute(
            """
            SELECT * FROM known WHERE task=? AND dataset_hash=?
            ORDER BY score DESC,semantic_sha256 LIMIT 1
            """,
            (task, dataset_hash),
        ).fetchone()
        improves = old is None or float(new["score"]) > float(old["score"])
        champions.append(
            {
                "task": task,
                "dataset_hash": dataset_hash,
                "recorded_incumbent": dict(old) if old else None,
                "best_new_strict": dict(new),
                "strictly_improves": improves,
                "promotion_status": (
                    "pending_fitted_cpp_parity" if improves else "incumbent_retained"
                ),
            }
        )
        if improves:
            win = {
                "task": task,
                "dataset_hash": dataset_hash,
                "old_score": float(old["score"]) if old else None,
                "new_score": float(new["score"]),
                "delta": None if old is None else float(new["score"]) - float(old["score"]),
                "old_genome": old["canonical_genome"] if old else None,
                "old_semantic_sha256": old["semantic_sha256"] if old else None,
                "old_receipt_sha256": (
                    old["receipt_ref"] if old else None
                ),
                "new_genome": new["canonical_genome"],
                "semantic_sha256": new["semantic_sha256"],
                "receipt_sha256": new["receipt_sha256"],
                "golden42_content_hash_match": (
                    task == "regression"
                    and dataset_hash in golden_ids
                    and (Path("/home/ubuntu/remote_super/quant/regression") / dataset_hash / "train.csv").is_file()
                    and sha256_file(
                        Path("/home/ubuntu/remote_super/quant/regression")
                        / dataset_hash
                        / "train.csv"
                    )
                    == new["train_sha256"]
                    and sha256_file(
                        Path("/home/ubuntu/remote_super/quant/regression")
                        / dataset_hash
                        / "test.csv"
                    )
                    == new["test_sha256"]
                ),
                "requires_fitted_cpp_confirmation": True,
                "model_checkpoint_sha256": new["model_checkpoint_sha256"],
                "checkpoint_origin": new["candidate_source"],
                "evidence_lane": "adaptive",
                "sealed": False,
            }
            wins.append(win)
    exact_rows = con.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    failures = con.execute("SELECT COUNT(*) FROM failures").fetchone()[0]
    con.close()
    known.close()
    atomic_json(ROOT / "indexes" / "champions.json", champions)
    atomic_json(ROOT / "indexes" / "strict-wins.json", wins)
    atomic_json(
        ROOT / "indexes" / "hall-of-fame.json",
        {
            "schema": "JopeTestOnlyStrictHallOfFame.v1",
            "observations": exact_rows,
            "failures": failures,
            "champions": champions,
            "strict_wins": wins,
        },
    )
    return {"exact_rows_total": exact_rows, "evaluator_failures_total": failures, "strict_wins": len(wins)}


def import_hof(tasks: list[str]) -> dict:
    pointer = json.loads(HOF_POINTER.read_text())
    known_path = Path(pointer["known"])
    if sha256_file(known_path) != pointer["known_sha256"]:
        raise RuntimeError("immutable known-exact snapshot hash drift")
    catalog_pointer = json.loads(CATALOG_POINTER.read_text())
    catalog = json.loads(Path(catalog_pointer["catalog"]).read_text())
    entries = {
        (entry["task"], entry["dataset_hash"]): entry
        for entry in catalog["entries"]
    }
    known = sqlite3.connect(f"file:{known_path}?mode=ro", uri=True)
    known.row_factory = sqlite3.Row
    corpus = corpus_db()
    imported = skipped = 0
    placeholders = ",".join("?" for _ in tasks)
    rows = known.execute(
        f"""
        SELECT * FROM known
        WHERE task IN ({placeholders})
        ORDER BY task,dataset_hash,semantic_sha256
        """,
        tasks,
    )
    for row in rows:
        entry = entries.get((row["task"], row["dataset_hash"]))
        if entry is None:
            skipped += 1
            continue
        receipt_ref = str(row["receipt_ref"] or "")
        receipt_hash = (
            receipt_ref
            if len(receipt_ref) == 64
            and all(character in "0123456789abcdef" for character in receipt_ref)
            else sha256_text(
                f"known-exact:{row['observation_id']}:{receipt_ref}:{row['source'] or ''}"
            )
        )
        before = corpus.total_changes
        corpus.execute(
            """
            INSERT OR IGNORE INTO observations(
              task,dataset_hash,semantic_sha256,canonical_genome,
              candidate_source,metric,score,secondary_metric,secondary_score,
              runtime_seconds,train_sha256,test_sha256,receipt_sha256,
              receipt_path,manifest_sha256,frozen_seed_sha256,
              model_checkpoint_sha256,hall_of_fame_sha256,created_at_ms
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                row["task"],
                row["dataset_hash"],
                row["semantic_sha256"],
                row["canonical_genome"],
                "historical_test_exact",
                "r2" if row["task"] == "regression" else "auc",
                float(row["score"]),
                None,
                None,
                0.0,
                entry["train_sha256"],
                entry["test_sha256"],
                receipt_hash,
                receipt_ref
                or f"{known_path}#observation-{row['observation_id']}",
                pointer["known_sha256"],
                sha256_text(
                    f"known-exact-seed:{row['task']}:{row['dataset_hash']}:"
                    f"{row['semantic_sha256']}"
                ),
                sha256_text("historical-test-exact-unattributed-checkpoint"),
                pointer["hall_of_fame_sha256"],
                0,
            ),
        )
        imported += corpus.total_changes - before
    corpus.commit()
    total = corpus.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    task_counts = {
        task: count
        for task, count in corpus.execute(
            "SELECT task,COUNT(*) FROM observations GROUP BY task"
        )
    }
    corpus.close()
    known.close()
    result = {
        "schema": "JopeTestOnlyHofImport.v1",
        "source": str(known_path),
        "source_sha256": pointer["known_sha256"],
        "strict_filter": {
            "regression": {"score_tier": "test_exact", "metric": "r2"},
            "binary": {"score_tier": "test_exact", "metric": "auc"},
        },
        "tasks": tasks,
        "imported": imported,
        "skipped_not_in_catalog": skipped,
        "task_counts": task_counts,
        "total": total,
    }
    atomic_json(ROOT / "corpus" / "hof-import.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    reconcile_command = commands.add_parser("reconcile")
    reconcile_command.add_argument("--limit", type=int, default=100)
    commands.add_parser("rebuild-indexes")
    import_command = commands.add_parser("import-hof")
    import_command.add_argument(
        "--task",
        action="append",
        choices=("regression", "binary"),
        dest="tasks",
        required=True,
    )
    args = parser.parse_args()
    if args.command == "reconcile":
        print(json.dumps(reconcile(args.limit), sort_keys=True))
    elif args.command == "import-hof":
        print(json.dumps(import_hof(sorted(set(args.tasks))), sort_keys=True))
    else:
        print(json.dumps(rebuild_indexes(), sort_keys=True))


if __name__ == "__main__":
    main()
