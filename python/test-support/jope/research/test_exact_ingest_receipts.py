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
