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
from test_exact_ingest_constants import (
    SCQ, ROOT, BACKLOG, CORPUS, RECEIPTS, SCQ_RUNS, HOF_POINTER,
    GOLDEN42, CATALOG_POINTER,
)
from test_exact_ingest_receipts import (
    command, scheduler_jobs, corpus_db, receipt_dir, local_receipt_dir, validate_and_ingest,
)

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
        except (Exception,) as error:
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
