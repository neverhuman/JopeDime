#!/usr/bin/env python3
"""Transactional exactly-once registry for self-play evaluation protocols."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Iterable, Mapping

from test_exact_common import (
    EVALUATOR_PROTOCOL_VERSION,
    EXPECTED_METRIC,
    evaluation_protocol_sha256,
    sha256_file,
    sha256_text,
)

ROOT = Path("/home/ubuntu/jope-flywheel/test_exact_gen0")
DEFAULT_REGISTRY = ROOT / "ops" / "pair-registry.sqlite"
TERMINAL_SEMANTIC_FAILURES = {
    "parse",
    "fit",
    "nonfinite",
    "task-invalid",
    "shape",
}


def now_ms() -> int:
    return int(time.time() * 1000)


def protocol_for_manifest(
    manifest: Mapping[str, object],
    *,
    version: str = EVALUATOR_PROTOCOL_VERSION,
) -> str:
    authority = manifest.get("label_authority") or {}
    if not isinstance(authority, Mapping):
        raise ValueError("manifest label authority is invalid")
    return evaluation_protocol_sha256(
        task=str(manifest["task"]),
        metric=str(authority.get("metric")),
        score_tier=str(authority.get("score_tier")),
        train_sha256=str(manifest["train_sha256"]),
        test_sha256=str(manifest["test_sha256"]),
        version=version,
    )


class PairRegistry:
    def __init__(self, path: Path = DEFAULT_REGISTRY):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path, timeout=60.0)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA busy_timeout=60000")
        for attempt in range(100):
            try:
                self.db.execute("PRAGMA journal_mode=WAL")
                break
            except sqlite3.OperationalError as error:
                if "locked" not in str(error).lower() or attempt == 99:
                    raise
                time.sleep(0.01)
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS pairs(
              task TEXT NOT NULL,
              dataset_hash TEXT NOT NULL,
              semantic_sha256 TEXT NOT NULL,
              protocol_sha256 TEXT NOT NULL,
              canonical_genome TEXT,
              state TEXT NOT NULL,
              reservation_owner TEXT NOT NULL,
              manifest_path TEXT,
              manifest_sha256 TEXT,
              job_id TEXT,
              receipt_sha256 TEXT,
              failure_class TEXT,
              infrastructure_retries INTEGER NOT NULL DEFAULT 0,
              created_at_ms INTEGER NOT NULL,
              updated_at_ms INTEGER NOT NULL,
              PRIMARY KEY(task,dataset_hash,semantic_sha256,protocol_sha256)
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS pair_manifest
              ON pairs(manifest_sha256);
            CREATE TABLE IF NOT EXISTS attempts(
              id INTEGER PRIMARY KEY,
              task TEXT NOT NULL,
              dataset_hash TEXT NOT NULL,
              semantic_sha256 TEXT NOT NULL,
              protocol_sha256 TEXT NOT NULL,
              attempt_no INTEGER NOT NULL,
              state TEXT NOT NULL,
              reservation_owner TEXT NOT NULL,
              manifest_path TEXT,
              manifest_sha256 TEXT,
              job_id TEXT,
              receipt_sha256 TEXT,
              failure_class TEXT,
              detail_json TEXT NOT NULL DEFAULT '{}',
              created_at_ms INTEGER NOT NULL,
              finalized_at_ms INTEGER,
              UNIQUE(task,dataset_hash,semantic_sha256,protocol_sha256,attempt_no),
              FOREIGN KEY(task,dataset_hash,semantic_sha256,protocol_sha256)
                REFERENCES pairs(task,dataset_hash,semantic_sha256,protocol_sha256)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS one_active_attempt_per_pair
              ON attempts(task,dataset_hash,semantic_sha256,protocol_sha256)
              WHERE state IN ('reserved','submitted');
            CREATE INDEX IF NOT EXISTS attempt_manifest
              ON attempts(manifest_sha256);
            CREATE INDEX IF NOT EXISTS attempt_job ON attempts(job_id);
            CREATE TABLE IF NOT EXISTS migrations(
              id INTEGER PRIMARY KEY,
              task TEXT NOT NULL,
              dataset_hash TEXT NOT NULL,
              semantic_sha256 TEXT NOT NULL,
              old_protocol_sha256 TEXT NOT NULL,
              new_protocol_sha256 TEXT NOT NULL,
              old_receipt_sha256 TEXT,
              reason TEXT NOT NULL,
              created_at_ms INTEGER NOT NULL,
              UNIQUE(task,dataset_hash,semantic_sha256,old_protocol_sha256,
                     new_protocol_sha256)
            );
            CREATE TABLE IF NOT EXISTS events(
              id INTEGER PRIMARY KEY,
              kind TEXT NOT NULL,
              task TEXT,
              dataset_hash TEXT,
              semantic_sha256 TEXT,
              protocol_sha256 TEXT,
              detail_json TEXT NOT NULL,
              created_at_ms INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS controls(
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            ) WITHOUT ROWID;
            INSERT OR IGNORE INTO controls VALUES('admission_halted','0');
            INSERT OR IGNORE INTO controls VALUES('admission_paused','0');
            INSERT OR IGNORE INTO controls VALUES('migration_pairs_reused','0');
            DROP INDEX IF EXISTS pair_receipt_unique;
            """
        )
        # Existing registries predate the durable attempt ledger.  Adopt each
        # pair as attempt one without changing its authoritative state.
        self.adopt_missing_attempts()
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def adopt_missing_attempts(self) -> int:
        before = self.db.total_changes
        self.db.execute(
            """
            INSERT OR IGNORE INTO attempts(
              task,dataset_hash,semantic_sha256,protocol_sha256,attempt_no,
              state,reservation_owner,manifest_path,manifest_sha256,job_id,
              receipt_sha256,failure_class,created_at_ms,finalized_at_ms
            )
            SELECT task,dataset_hash,semantic_sha256,protocol_sha256,1,
              state,reservation_owner,manifest_path,manifest_sha256,job_id,
              receipt_sha256,failure_class,created_at_ms,
              CASE WHEN state IN ('reserved','submitted') THEN NULL
                   ELSE updated_at_ms END
            FROM pairs AS pair
            WHERE NOT EXISTS(
              SELECT 1 FROM attempts AS attempt
              WHERE attempt.task=pair.task
                AND attempt.dataset_hash=pair.dataset_hash
                AND attempt.semantic_sha256=pair.semantic_sha256
                AND attempt.protocol_sha256=pair.protocol_sha256
            )
            """
        )
        return self.db.total_changes - before

    def _event(
        self,
        kind: str,
        *,
        task: str | None = None,
        dataset_hash: str | None = None,
        semantic_sha256: str | None = None,
        protocol_sha256: str | None = None,
        detail: Mapping[str, object] | None = None,
    ) -> None:
        self.db.execute(
            "INSERT INTO events(kind,task,dataset_hash,semantic_sha256,"
            "protocol_sha256,detail_json,created_at_ms) VALUES(?,?,?,?,?,?,?)",
            (
                kind,
                task,
                dataset_hash,
                semantic_sha256,
                protocol_sha256,
                json.dumps(detail or {}, sort_keys=True),
                now_ms(),
            ),
        )

    def _halt_for_violation(
        self,
        *,
        task: str,
        dataset_hash: str,
        semantic_sha256: str,
        protocol_sha256: str,
        detail: Mapping[str, object],
    ) -> None:
        self.db.execute(
            "UPDATE controls SET value='1' WHERE key='admission_halted'"
        )
        self._event(
            "uniqueness_violation",
            task=task,
            dataset_hash=dataset_hash,
            semantic_sha256=semantic_sha256,
            protocol_sha256=protocol_sha256,
            detail=detail,
        )

    def admission_halted(self) -> bool:
        row = self.db.execute(
            "SELECT value FROM controls WHERE key='admission_halted'"
        ).fetchone()
        return row is not None and row["value"] == "1"

    def admission_paused(self) -> bool:
        row = self.db.execute(
            "SELECT value FROM controls WHERE key='admission_paused'"
        ).fetchone()
        return row is not None and row["value"] == "1"

    def set_admission_paused(self, paused: bool, *, reason: str) -> bool:
        if not reason.strip():
            raise ValueError("admission pause/resume requires a reason")
        prior = self.admission_paused()
        self.db.execute(
            "UPDATE controls SET value=? WHERE key='admission_paused'",
            ("1" if paused else "0",),
        )
        self._event(
            "admission_paused" if paused else "admission_resumed",
            detail={"reason": reason, "changed": prior != paused},
        )
        self.db.commit()
        return prior != paused

    @staticmethod
    def _identity_values(identity: Mapping[str, object]) -> tuple[str, str, str, str]:
        return (
            str(identity["task"]),
            str(identity["dataset_hash"]),
            str(identity["semantic_sha256"]),
            str(identity["protocol_sha256"]),
        )

    def _append_attempt(
        self,
        identity: Mapping[str, object],
        *,
        state: str,
        owner: str,
        manifest_path: str | None = None,
        manifest_sha256: str | None = None,
        job_id: str | None = None,
        detail: Mapping[str, object] | None = None,
    ) -> int:
        values = self._identity_values(identity)
        attempt_no = int(
            self.db.execute(
                """
                SELECT COALESCE(MAX(attempt_no),0)+1 FROM attempts
                WHERE task=? AND dataset_hash=? AND semantic_sha256=?
                  AND protocol_sha256=?
                """,
                values,
            ).fetchone()[0]
        )
        cursor = self.db.execute(
            """
            INSERT INTO attempts(
              task,dataset_hash,semantic_sha256,protocol_sha256,attempt_no,
              state,reservation_owner,manifest_path,manifest_sha256,job_id,
              detail_json,created_at_ms
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                *values,
                attempt_no,
                state,
                owner,
                manifest_path,
                manifest_sha256,
                job_id,
                json.dumps(detail or {}, sort_keys=True),
                now_ms(),
            ),
        )
        return int(cursor.lastrowid)

    def _active_attempt(self, identity: Mapping[str, object]) -> sqlite3.Row | None:
        return self.db.execute(
            """
            SELECT * FROM attempts
            WHERE task=? AND dataset_hash=? AND semantic_sha256=?
              AND protocol_sha256=? AND state IN ('reserved','submitted')
            """,
            self._identity_values(identity),
        ).fetchone()

    def reserve(
        self,
        *,
        task: str,
        dataset_hash: str,
        semantic_sha256: str,
        protocol_sha256: str,
        owner: str,
        canonical_genome: str | None = None,
        manifest_path: str | None = None,
        manifest_sha256: str | None = None,
    ) -> bool:
        """Reserve a protocol pair atomically; False means authoritative reuse."""

        if canonical_genome is not None and sha256_text(canonical_genome) != semantic_sha256:
            raise ValueError("pair semantic hash does not match its canonical genome")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            if self.admission_halted() or self.admission_paused():
                raise RuntimeError("new pair admission is halted by a uniqueness violation")
            timestamp = now_ms()
            cursor = self.db.execute(
                """
                INSERT OR IGNORE INTO pairs(
                  task,dataset_hash,semantic_sha256,protocol_sha256,
                  canonical_genome,state,reservation_owner,manifest_path,
                  manifest_sha256,created_at_ms,updated_at_ms
                ) VALUES(?,?,?,?,?,'reserved',?,?,?,?,?)
                """,
                (
                    task,
                    dataset_hash,
                    semantic_sha256,
                    protocol_sha256,
                    canonical_genome,
                    owner,
                    manifest_path,
                    manifest_sha256,
                    timestamp,
                    timestamp,
                ),
            )
            if cursor.rowcount == 1:
                self._append_attempt(
                    {
                        "task": task,
                        "dataset_hash": dataset_hash,
                        "semantic_sha256": semantic_sha256,
                        "protocol_sha256": protocol_sha256,
                    },
                    state="reserved",
                    owner=owner,
                    manifest_path=manifest_path,
                    manifest_sha256=manifest_sha256,
                )
                self._event(
                    "reserved",
                    task=task,
                    dataset_hash=dataset_hash,
                    semantic_sha256=semantic_sha256,
                    protocol_sha256=protocol_sha256,
                    detail={"owner": owner},
                )
                self.db.commit()
                return True
            existing = self.db.execute(
                """
                SELECT * FROM pairs WHERE task=? AND dataset_hash=?
                  AND semantic_sha256=? AND protocol_sha256=?
                """,
                (task, dataset_hash, semantic_sha256, protocol_sha256),
            ).fetchone()
            if existing is None:
                raise AssertionError("pair disappeared during reservation")
            if existing["state"] == "available":
                self.db.execute(
                    """
                    UPDATE pairs SET canonical_genome=COALESCE(?,canonical_genome),
                      state='reserved',reservation_owner=?,manifest_path=?,
                      manifest_sha256=?,job_id=NULL,receipt_sha256=NULL,
                      failure_class=NULL,updated_at_ms=?
                    WHERE task=? AND dataset_hash=? AND semantic_sha256=?
                      AND protocol_sha256=? AND state='available'
                    """,
                    (
                        canonical_genome,
                        owner,
                        manifest_path,
                        manifest_sha256,
                        timestamp,
                        task,
                        dataset_hash,
                        semantic_sha256,
                        protocol_sha256,
                    ),
                )
                self._append_attempt(
                    existing,
                    state="reserved",
                    owner=owner,
                    manifest_path=manifest_path,
                    manifest_sha256=manifest_sha256,
                    detail={"reused_after": "superseded_unstarted"},
                )
                self._event(
                    "reserved",
                    task=task,
                    dataset_hash=dataset_hash,
                    semantic_sha256=semantic_sha256,
                    protocol_sha256=protocol_sha256,
                    detail={"owner": owner, "reused_available_pair": True},
                )
                self.db.commit()
                return True
            if (
                canonical_genome is not None
                and existing["canonical_genome"] is not None
                and existing["canonical_genome"] != canonical_genome
            ):
                self._halt_for_violation(
                    task=task,
                    dataset_hash=dataset_hash,
                    semantic_sha256=semantic_sha256,
                    protocol_sha256=protocol_sha256,
                    detail={
                        "existing_owner": existing["reservation_owner"],
                        "attempted_owner": owner,
                        "reason": "semantic_hash_collision",
                    },
                )
                self.db.commit()
                raise RuntimeError("pair registry semantic uniqueness violation")
            self._event(
                "duplicate_prevented",
                task=task,
                dataset_hash=dataset_hash,
                semantic_sha256=semantic_sha256,
                protocol_sha256=protocol_sha256,
                detail={
                    "owner": owner,
                    "existing_owner": existing["reservation_owner"],
                    "state": existing["state"],
                },
            )
            self.db.commit()
            return False
        except BaseException:
            if self.db.in_transaction:
                self.db.rollback()
            raise

    def reserve_batch(
        self,
        pairs: Iterable[Mapping[str, str]],
        *,
        owner: str,
    ) -> list[bool]:
        """Reserve a bounded batch in one transaction, preserving input order."""

        identities = [dict(pair) for pair in pairs]
        self.db.execute("BEGIN IMMEDIATE")
        try:
            if self.admission_halted() or self.admission_paused():
                raise RuntimeError("new pair admission is halted by a uniqueness violation")
            outcomes = []
            timestamp = now_ms()
            for identity in identities:
                canonical = identity.get("canonical_genome")
                semantic = identity["semantic_sha256"]
                if canonical is not None and sha256_text(canonical) != semantic:
                    raise ValueError(
                        "pair semantic hash does not match its canonical genome"
                    )
                values = (
                    identity["task"],
                    identity["dataset_hash"],
                    semantic,
                    identity["protocol_sha256"],
                )
                cursor = self.db.execute(
                    """
                    INSERT OR IGNORE INTO pairs(
                      task,dataset_hash,semantic_sha256,protocol_sha256,
                      canonical_genome,state,reservation_owner,created_at_ms,
                      updated_at_ms
                    ) VALUES(?,?,?,?,?,'reserved',?,?,?)
                    """,
                    (*values, canonical, owner, timestamp, timestamp),
                )
                created = cursor.rowcount == 1
                if created:
                    self._append_attempt(
                        identity,
                        state="reserved",
                        owner=owner,
                    )
                    outcomes.append(True)
                    self._event(
                        "reserved",
                        task=values[0],
                        dataset_hash=values[1],
                        semantic_sha256=values[2],
                        protocol_sha256=values[3],
                        detail={"owner": owner},
                    )
                    continue
                existing = self.db.execute(
                    """
                    SELECT canonical_genome,reservation_owner,state FROM pairs
                    WHERE task=? AND dataset_hash=? AND semantic_sha256=?
                      AND protocol_sha256=?
                    """,
                    values,
                ).fetchone()
                if existing is None:
                    raise AssertionError("pair disappeared during batch reservation")
                if existing["state"] == "available":
                    self.db.execute(
                        """
                        UPDATE pairs SET canonical_genome=COALESCE(?,canonical_genome),
                          state='reserved',reservation_owner=?,manifest_path=NULL,
                          manifest_sha256=NULL,job_id=NULL,receipt_sha256=NULL,
                          failure_class=NULL,updated_at_ms=?
                        WHERE task=? AND dataset_hash=? AND semantic_sha256=?
                          AND protocol_sha256=? AND state='available'
                        """,
                        (canonical, owner, timestamp, *values),
                    )
                    self._append_attempt(
                        identity,
                        state="reserved",
                        owner=owner,
                        detail={"reused_after": "superseded_unstarted"},
                    )
                    outcomes.append(True)
                    self._event(
                        "reserved",
                        task=values[0],
                        dataset_hash=values[1],
                        semantic_sha256=values[2],
                        protocol_sha256=values[3],
                        detail={"owner": owner, "reused_available_pair": True},
                    )
                    continue
                outcomes.append(False)
                if (
                    canonical is not None
                    and existing["canonical_genome"] is not None
                    and existing["canonical_genome"] != canonical
                ):
                    self._halt_for_violation(
                        task=values[0],
                        dataset_hash=values[1],
                        semantic_sha256=values[2],
                        protocol_sha256=values[3],
                        detail={
                            "existing_owner": existing["reservation_owner"],
                            "attempted_owner": owner,
                            "reason": "semantic_hash_collision",
                        },
                    )
                    self.db.commit()
                    raise RuntimeError("pair registry semantic uniqueness violation")
                self._event(
                    "duplicate_prevented",
                    task=values[0],
                    dataset_hash=values[1],
                    semantic_sha256=values[2],
                    protocol_sha256=values[3],
                    detail={
                        "owner": owner,
                        "existing_owner": existing["reservation_owner"],
                        "state": existing["state"],
                    },
                )
            self.db.commit()
            return outcomes
        except BaseException:
            if self.db.in_transaction:
                self.db.rollback()
            raise

    def reserve_manifest(
        self,
        manifest: Mapping[str, object],
        *,
        owner: str,
        manifest_path: str,
        manifest_sha256: str,
        version: str = EVALUATOR_PROTOCOL_VERSION,
    ) -> tuple[list[dict], list[dict]]:
        protocol = protocol_for_manifest(manifest, version=version)
        reserved, reused = [], []
        for raw in manifest.get("candidates") or []:
            candidate = dict(raw)
            identity = {
                "task": str(manifest["task"]),
                "dataset_hash": str(manifest["dataset_hash"]),
                "semantic_sha256": str(candidate["semantic_sha256"]),
                "protocol_sha256": protocol,
            }
            created = self.reserve(
                **identity,
                owner=owner,
                canonical_genome=str(candidate["canonical_genome"]),
                manifest_path=manifest_path,
                manifest_sha256=manifest_sha256,
            )
            (reserved if created else reused).append({**identity, **candidate})
        return reserved, reused

    def manifest_state_counts(self, manifest_sha256: str) -> dict[str, int]:
        return {
            str(row["state"]): int(row["count"])
            for row in self.db.execute(
                "SELECT state,COUNT(*) AS count FROM pairs "
                "WHERE manifest_sha256=? GROUP BY state",
                (manifest_sha256,),
            )
        }

    def authorize_submission(
        self,
        manifest_sha256: str,
        owner: str,
        *,
        job_id: str | None = None,
    ) -> int:
        rows = self.db.execute(
            "SELECT reservation_owner,state,job_id FROM pairs "
            "WHERE manifest_sha256=?",
            (manifest_sha256,),
        ).fetchall()
        if not rows:
            raise RuntimeError("manifest has no protocol reservations")
        owners = {row["reservation_owner"] for row in rows}
        if owners != {owner}:
            raise RuntimeError("manifest submitter does not own every reservation")
        blocked = {
            str(row["state"])
            for row in rows
            if row["state"] not in {"reserved", "infrastructure_retry"}
            and not (
                row["state"] == "submitted"
                and job_id is not None
                and row["job_id"] == job_id
            )
        }
        if blocked:
            raise RuntimeError(
                "manifest contains non-launchable protocol pairs: "
                + ",".join(sorted(blocked))
            )
        return len(rows)

    def manifest_owner(self, manifest_sha256: str) -> str | None:
        rows = self.db.execute(
            "SELECT DISTINCT reservation_owner FROM pairs WHERE manifest_sha256=?",
            (manifest_sha256,),
        ).fetchall()
        owners = {row["reservation_owner"] for row in rows}
        if not owners:
            return None
        if len(owners) != 1:
            raise RuntimeError("manifest reservations have multiple owners")
        return owners.pop()

    def get(
        self,
        *,
        task: str,
        dataset_hash: str,
        semantic_sha256: str,
        protocol_sha256: str,
    ) -> dict | None:
        row = self.db.execute(
            """
            SELECT * FROM pairs WHERE task=? AND dataset_hash=?
              AND semantic_sha256=? AND protocol_sha256=?
            """,
            (task, dataset_hash, semantic_sha256, protocol_sha256),
        ).fetchone()
        return None if row is None else dict(row)

    def record_submission(
        self,
        manifest_sha256: str,
        owner: str,
        job_id: str,
        *,
        commit: bool = True,
    ) -> int:
        rows = self.db.execute(
            "SELECT * FROM pairs WHERE manifest_sha256=?",
            (manifest_sha256,),
        ).fetchall()
        if not rows:
            raise RuntimeError("manifest has no protocol reservations")
        if {row["reservation_owner"] for row in rows} != {owner}:
            raise RuntimeError("manifest submitter does not own every reservation")
        blocked = [
            row
            for row in rows
            if row["state"] not in {"reserved", "infrastructure_retry"}
            and not (row["state"] == "submitted" and row["job_id"] == job_id)
        ]
        if blocked:
            raise RuntimeError(
                "manifest contains protocol pairs already claimed by an "
                "authoritative result or a different submission"
            )
        retry_rows = [row for row in rows if row["state"] == "infrastructure_retry"]
        self.db.execute(
            """
            UPDATE pairs SET state='submitted',job_id=?,updated_at_ms=?
            WHERE manifest_sha256=? AND reservation_owner=?
              AND state IN ('reserved','infrastructure_retry')
            """,
            (job_id, now_ms(), manifest_sha256, owner),
        )
        self.db.execute(
            """
            UPDATE attempts SET state='submitted',job_id=?
            WHERE manifest_sha256=? AND reservation_owner=? AND state='reserved'
            """,
            (job_id, manifest_sha256, owner),
        )
        for row in retry_rows:
            self._append_attempt(
                row,
                state="submitted",
                owner=owner,
                manifest_path=row["manifest_path"],
                manifest_sha256=manifest_sha256,
                job_id=job_id,
                detail={"retry_after_infrastructure_failure": True},
            )
        if commit:
            self.db.commit()
        return len(rows)

    def record_manifest_infrastructure_failure(
        self, manifest_sha256: str, failure_class: str
    ) -> int:
        cursor = self.db.execute(
            """
            UPDATE pairs SET state='infrastructure_retry',failure_class=?,
              infrastructure_retries=infrastructure_retries+1,
              updated_at_ms=?
            WHERE manifest_sha256=? AND state='submitted'
            """,
            (failure_class, now_ms(), manifest_sha256),
        )
        self.db.execute(
            """
            UPDATE attempts SET state='infrastructure_retry',failure_class=?,
              finalized_at_ms=?
            WHERE manifest_sha256=? AND state='submitted'
            """,
            (failure_class, now_ms(), manifest_sha256),
        )
        self._event(
            "infrastructure_retry",
            detail={
                "manifest_sha256": manifest_sha256,
                "failure": failure_class,
                "pairs": cursor.rowcount,
            },
        )
        self.db.commit()
        return cursor.rowcount

    def bind_manifest(
        self,
        *,
        owner: str,
        manifest_path: str,
        manifest_sha256: str,
        pairs: Iterable[Mapping[str, str]],
    ) -> int:
        return self.bind_manifests(
            owner=owner,
            manifests=[
                {
                    "manifest_path": manifest_path,
                    "manifest_sha256": manifest_sha256,
                    "pairs": list(pairs),
                }
            ],
        )

    def bind_manifests(
        self,
        *,
        owner: str,
        manifests: Iterable[Mapping[str, object]],
    ) -> int:
        bindings = [dict(binding) for binding in manifests]
        self.db.execute("BEGIN IMMEDIATE")
        try:
            updated = 0
            for binding in bindings:
                for identity in binding["pairs"]:
                    cursor = self.db.execute(
                        """
                        UPDATE pairs SET manifest_path=?,manifest_sha256=?,
                          updated_at_ms=?
                        WHERE task=? AND dataset_hash=? AND semantic_sha256=?
                          AND protocol_sha256=? AND reservation_owner=?
                        """,
                        (
                            binding["manifest_path"],
                            binding["manifest_sha256"],
                            now_ms(),
                            identity["task"],
                            identity["dataset_hash"],
                            identity["semantic_sha256"],
                            identity["protocol_sha256"],
                            owner,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise RuntimeError(
                            "only the reservation owner may bind a manifest"
                        )
                    self.db.execute(
                        """
                        UPDATE attempts SET manifest_path=?,manifest_sha256=?
                        WHERE task=? AND dataset_hash=? AND semantic_sha256=?
                          AND protocol_sha256=? AND reservation_owner=?
                          AND state='reserved'
                        """,
                        (
                            binding["manifest_path"],
                            binding["manifest_sha256"],
                            identity["task"],
                            identity["dataset_hash"],
                            identity["semantic_sha256"],
                            identity["protocol_sha256"],
                            owner,
                        ),
                    )
                    updated += 1
            self.db.commit()
            return updated
        except BaseException:
            if self.db.in_transaction:
                self.db.rollback()
            raise

    def record_result(
        self,
        *,
        task: str,
        dataset_hash: str,
        semantic_sha256: str,
        protocol_sha256: str,
        receipt_sha256: str,
        failure: str | None = None,
    ) -> None:
        state = (
            "completed"
            if failure is None
            else "terminal_failure"
            if failure in TERMINAL_SEMANTIC_FAILURES
            else "infrastructure_retry"
        )
        retry = 1 if state == "infrastructure_retry" else 0
        identity = {
            "task": task,
            "dataset_hash": dataset_hash,
            "semantic_sha256": semantic_sha256,
            "protocol_sha256": protocol_sha256,
        }
        existing = self.get(**identity)
        if existing is None:
            raise RuntimeError("receipt does not match a reserved protocol pair")
        if existing["state"] in {"completed", "terminal_failure"}:
            if (
                existing["state"] == state
                and existing["receipt_sha256"] == receipt_sha256
                and existing["failure_class"] == failure
            ):
                return
            self._halt_for_violation(
                **identity,
                detail={
                    "reason": "conflicting_authoritative_result",
                    "existing_receipt": existing["receipt_sha256"],
                    "attempted_receipt": receipt_sha256,
                },
            )
            self.db.commit()
            raise RuntimeError("pair already has an authoritative terminal result")
        cursor = self.db.execute(
            """
            UPDATE pairs SET state=?,receipt_sha256=?,
              failure_class=?,infrastructure_retries=infrastructure_retries+?,
              updated_at_ms=?
            WHERE task=? AND dataset_hash=? AND semantic_sha256=?
              AND protocol_sha256=?
            """,
            (
                state,
                None if state == "infrastructure_retry" else receipt_sha256,
                failure,
                retry,
                now_ms(),
                task,
                dataset_hash,
                semantic_sha256,
                protocol_sha256,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("receipt does not match a reserved protocol pair")
        active = self._active_attempt(identity)
        if active is not None:
            self.db.execute(
                """
                UPDATE attempts SET state=?,receipt_sha256=?,failure_class=?,
                  finalized_at_ms=?
                WHERE id=?
                """,
                (
                    state,
                    None if state == "infrastructure_retry" else receipt_sha256,
                    failure,
                    now_ms(),
                    active["id"],
                ),
            )
        elif state != "infrastructure_retry":
            raise RuntimeError("authoritative result has no active attempt")
        self._event(
            "infrastructure_retry" if retry else "authoritative_result",
            task=task,
            dataset_hash=dataset_hash,
            semantic_sha256=semantic_sha256,
            protocol_sha256=protocol_sha256,
            detail={"failure": failure, "receipt_sha256": receipt_sha256},
        )
        self.db.commit()

    def migrate_protocol(
        self,
        *,
        task: str,
        dataset_hash: str,
        semantic_sha256: str,
        old_protocol_sha256: str,
        new_protocol_sha256: str,
        reason: str,
        owner: str,
    ) -> bool:
        if old_protocol_sha256 == new_protocol_sha256:
            raise ValueError("protocol migration must change the protocol hash")
        if not reason.strip():
            raise ValueError("protocol migration requires a correctness reason")
        old = self.db.execute(
            """
            SELECT * FROM pairs WHERE task=? AND dataset_hash=?
              AND semantic_sha256=? AND protocol_sha256=?
            """,
            (task, dataset_hash, semantic_sha256, old_protocol_sha256),
        ).fetchone()
        if old is None:
            raise RuntimeError("old protocol pair is absent")
        created = self.reserve(
            task=task,
            dataset_hash=dataset_hash,
            semantic_sha256=semantic_sha256,
            protocol_sha256=new_protocol_sha256,
            owner=owner,
            canonical_genome=old["canonical_genome"],
        )
        self.db.execute(
            """
            INSERT OR IGNORE INTO migrations(
              task,dataset_hash,semantic_sha256,old_protocol_sha256,
              new_protocol_sha256,old_receipt_sha256,reason,created_at_ms
            ) VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                task,
                dataset_hash,
                semantic_sha256,
                old_protocol_sha256,
                new_protocol_sha256,
                old["receipt_sha256"],
                reason,
                now_ms(),
            ),
        )
        self.db.commit()
        return created

    def supersede_unstarted(
        self,
        manifest_sha256: str,
        job_id: str,
        *,
        retirement_receipt_sha256: str,
    ) -> int:
        """Retire only a reserved or held/unstarted attempt, retaining history."""

        if len(retirement_receipt_sha256) != 64:
            raise ValueError("retirement receipt must be a SHA-256 value")
        rows = self.db.execute(
            """
            SELECT * FROM pairs WHERE manifest_sha256=?
              AND (
                (state='submitted' AND job_id=?)
                OR (state='reserved' AND job_id IS NULL)
              )
            """,
            (manifest_sha256, job_id),
        ).fetchall()
        if not rows:
            prior = self.db.execute(
                """
                SELECT COUNT(*) FROM attempts WHERE manifest_sha256=? AND job_id=?
                  AND state='superseded_unstarted'
                  AND receipt_sha256=?
                """,
                (manifest_sha256, job_id, retirement_receipt_sha256),
            ).fetchone()[0]
            if prior:
                return int(prior)
            raise RuntimeError("stale retirement has no submitted protocol pairs")
        timestamp = now_ms()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            for row in rows:
                active = self._active_attempt(row)
                if (
                    active is None
                    or active["state"] != row["state"]
                    or (
                        row["state"] == "submitted"
                        and active["job_id"] != job_id
                    )
                ):
                    raise RuntimeError("stale retirement active-attempt audit failed")
                self.db.execute(
                    """
                    UPDATE attempts SET state='superseded_unstarted',
                      receipt_sha256=?,job_id=COALESCE(job_id,?),
                      finalized_at_ms=?
                    WHERE id=?
                    """,
                    (
                        retirement_receipt_sha256,
                        job_id,
                        timestamp,
                        active["id"],
                    ),
                )
                self.db.execute(
                    """
                    UPDATE pairs SET state='available',reservation_owner='available',
                      manifest_path=NULL,manifest_sha256=NULL,job_id=NULL,
                      receipt_sha256=NULL,failure_class=NULL,updated_at_ms=?
                    WHERE task=? AND dataset_hash=? AND semantic_sha256=?
                      AND protocol_sha256=?
                      AND (
                        (state='submitted' AND job_id=?)
                        OR (state='reserved' AND job_id IS NULL)
                      )
                    """,
                    (
                        timestamp,
                        row["task"],
                        row["dataset_hash"],
                        row["semantic_sha256"],
                        row["protocol_sha256"],
                        job_id,
                    ),
                )
                self._event(
                    "superseded_unstarted",
                    task=row["task"],
                    dataset_hash=row["dataset_hash"],
                    semantic_sha256=row["semantic_sha256"],
                    protocol_sha256=row["protocol_sha256"],
                    detail={
                        "manifest_sha256": manifest_sha256,
                        "job_id": job_id,
                        "retirement_receipt_sha256": retirement_receipt_sha256,
                    },
                )
            self.db.commit()
            return len(rows)
        except BaseException:
            if self.db.in_transaction:
                self.db.rollback()
            raise

    def attempt_history(
        self,
        *,
        task: str,
        dataset_hash: str,
        semantic_sha256: str,
        protocol_sha256: str,
    ) -> list[dict]:
        return [
            dict(row)
            for row in self.db.execute(
                """
                SELECT * FROM attempts
                WHERE task=? AND dataset_hash=? AND semantic_sha256=?
                  AND protocol_sha256=?
                ORDER BY attempt_no
                """,
                (task, dataset_hash, semantic_sha256, protocol_sha256),
            )
        ]

    def stats(self) -> dict[str, int]:
        event_counts = {
            row["kind"]: int(row["count"])
            for row in self.db.execute(
                "SELECT kind,COUNT(*) AS count FROM events GROUP BY kind"
            )
        }
        migrated_reuse = int(
            self.db.execute(
                "SELECT value FROM controls WHERE key='migration_pairs_reused'"
            ).fetchone()[0]
        )
        return {
            "pairs": int(self.db.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]),
            "attempts": int(
                self.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
            ),
            "active_attempts": int(
                self.db.execute(
                    "SELECT COUNT(*) FROM attempts "
                    "WHERE state IN ('reserved','submitted')"
                ).fetchone()[0]
            ),
            "available_pairs": int(
                self.db.execute(
                    "SELECT COUNT(*) FROM pairs WHERE state='available'"
                ).fetchone()[0]
            ),
            "superseded_unstarted": int(
                self.db.execute(
                    "SELECT COUNT(*) FROM attempts "
                    "WHERE state='superseded_unstarted'"
                ).fetchone()[0]
            ),
            "pairs_reused": migrated_reuse
            + event_counts.get("duplicate_prevented", 0),
            "duplicates_prevented": migrated_reuse
            + event_counts.get("duplicate_prevented", 0),
            "infrastructure_retries": int(
                self.db.execute(
                    "SELECT COALESCE(SUM(infrastructure_retries),0) FROM pairs"
                ).fetchone()[0]
            ),
            "protocol_migrations": int(
                self.db.execute("SELECT COUNT(*) FROM migrations").fetchone()[0]
            ),
            "uniqueness_violations": event_counts.get("uniqueness_violation", 0),
            "admission_halted": int(self.admission_halted()),
            "admission_paused": int(self.admission_paused()),
        }


def import_manifest_paths(
    registry: PairRegistry,
    paths: Iterable[Path],
    *,
    owner: str = "legacy-import",
) -> dict[str, int]:
    manifests = pairs = reused = 0
    timestamp = now_ms()
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        manifest = json.loads(path.read_text())
        if manifest.get("schema") != "JopeFlywheelCorpusEvalShard.v1":
            continue
        digest = sha256_file(path)
        protocol = protocol_for_manifest(manifest)
        values = []
        for candidate in manifest.get("candidates") or []:
            if sha256_text(candidate["canonical_genome"]) != candidate["semantic_sha256"]:
                raise ValueError(f"{path}: candidate semantic hash drift")
            values.append(
                (
                    manifest["task"],
                    manifest["dataset_hash"],
                    candidate["semantic_sha256"],
                    protocol,
                    candidate["canonical_genome"],
                    "reserved",
                    owner,
                    str(path),
                    digest,
                    None,
                    None,
                    None,
                    0,
                    timestamp,
                    timestamp,
                )
            )
        before = registry.db.total_changes
        registry.db.executemany(
            "INSERT OR IGNORE INTO pairs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            values,
        )
        added = registry.db.total_changes - before
        manifests += 1
        pairs += added
        reused += len(values) - added
        if manifests % 500 == 0:
            registry.db.commit()
    registry.adopt_missing_attempts()
    registry.db.commit()
    return {"manifests": manifests, "pairs_imported": pairs, "pairs_reused": reused}


def import_corpus(
    registry: PairRegistry,
    corpus_path: Path,
    catalog_pointer: Path,
    *,
    owner: str = "legacy-import",
) -> dict[str, int]:
    pointer = json.loads(catalog_pointer.read_text())
    catalog = json.loads(Path(pointer["catalog"]).read_text())
    entries = {
        (row["task"], row["dataset_hash"]): row for row in catalog["entries"]
    }
    corpus = sqlite3.connect(f"file:{corpus_path}?mode=ro", uri=True)
    corpus.row_factory = sqlite3.Row
    imported = reused = results = 0
    observations = corpus.execute(
        "SELECT * FROM observations ORDER BY task,dataset_hash,semantic_sha256"
    )
    timestamp = now_ms()
    while True:
        batch = observations.fetchmany(20_000)
        if not batch:
            break
        values = []
        for row in batch:
            protocol = evaluation_protocol_sha256(
                task=row["task"],
                metric=row["metric"],
                score_tier="test_exact",
                train_sha256=row["train_sha256"],
                test_sha256=row["test_sha256"],
            )
            values.append(
                (
                    row["task"],
                    row["dataset_hash"],
                    row["semantic_sha256"],
                    protocol,
                    row["canonical_genome"],
                    "completed",
                    owner,
                    row["receipt_path"],
                    row["manifest_sha256"],
                    None,
                    row["receipt_sha256"],
                    None,
                    0,
                    timestamp,
                    timestamp,
                )
            )
        before = registry.db.total_changes
        registry.db.executemany(
            "INSERT OR IGNORE INTO pairs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            values,
        )
        added = registry.db.total_changes - before
        imported += added
        results += added
        reused += len(values) - added
        registry.db.commit()
    failures = corpus.execute(
        "SELECT * FROM failures ORDER BY task,dataset_hash,semantic_sha256"
    )
    while True:
        batch = failures.fetchmany(20_000)
        if not batch:
            break
        values = []
        for row in batch:
            if row["failure"] not in TERMINAL_SEMANTIC_FAILURES:
                continue
            entry = entries.get((row["task"], row["dataset_hash"]))
            if entry is None:
                continue
            protocol = evaluation_protocol_sha256(
                task=row["task"],
                metric=EXPECTED_METRIC[row["task"]],
                score_tier="test_exact",
                train_sha256=entry["train_sha256"],
                test_sha256=entry["test_sha256"],
            )
            values.append(
                (
                    row["task"],
                    row["dataset_hash"],
                    row["semantic_sha256"],
                    protocol,
                    None,
                    "terminal_failure",
                    owner,
                    None,
                    None,
                    None,
                    row["receipt_sha256"],
                    row["failure"],
                    0,
                    timestamp,
                    timestamp,
                )
            )
        before = registry.db.total_changes
        registry.db.executemany(
            "INSERT OR IGNORE INTO pairs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            values,
        )
        added = registry.db.total_changes - before
        imported += added
        results += added
        reused += len(values) - added
        registry.db.commit()
    registry.adopt_missing_attempts()
    registry.db.commit()
    corpus.close()
    return {
        "corpus_pairs_imported": imported,
        "corpus_pairs_reused": reused,
        "authoritative_results": results,
    }


def reconcile_backlog_claims(
    registry: PairRegistry,
    backlog_path: Path,
) -> dict[str, int]:
    """Adopt legacy checked/submitted claim state without creating new work."""

    if not backlog_path.is_file():
        return {"claims_seen": 0, "claims_adopted": 0, "claims_authoritative": 0}
    backlog = sqlite3.connect(f"file:{backlog_path}?mode=ro", uri=True)
    backlog.row_factory = sqlite3.Row
    seen = adopted = authoritative = 0
    for claim in backlog.execute(
        "SELECT manifest_sha256,job_id,state,reservation_owner FROM claims "
        "WHERE state IN ('checked','submitted') ORDER BY manifest_sha256"
    ):
        seen += 1
        if claim["state"] != "submitted":
            continue
        owner = claim["reservation_owner"] or registry.manifest_owner(
            claim["manifest_sha256"]
        )
        if owner is None:
            raise RuntimeError(
                f"{claim['manifest_sha256']}: submitted claim has no reservations"
            )
        states = registry.manifest_state_counts(claim["manifest_sha256"])
        if set(states).issubset({"completed", "terminal_failure"}):
            authoritative += 1
            continue
        registry.record_submission(
            claim["manifest_sha256"],
            str(owner),
            claim["job_id"],
            commit=False,
        )
        adopted += 1
    registry.db.commit()
    backlog.close()
    return {
        "claims_seen": seen,
        "claims_adopted": adopted,
        "claims_authoritative": authoritative,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    commands = parser.add_subparsers(dest="command", required=True)
    migrate = commands.add_parser("import-manifests")
    migrate.add_argument("roots", nargs="+", type=Path)
    full = commands.add_parser("migrate")
    full.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "corpus" / "strict-test-exact.sqlite",
    )
    full.add_argument(
        "--catalog-pointer",
        type=Path,
        default=ROOT / "catalog" / "current.json",
    )
    full.add_argument(
        "--manifest-root",
        type=Path,
        default=ROOT / "manifests",
    )
    full.add_argument(
        "--backlog",
        type=Path,
        default=ROOT / "ops" / "backlog.sqlite",
    )
    reconcile = commands.add_parser("reconcile-backlog")
    reconcile.add_argument(
        "--backlog",
        type=Path,
        default=ROOT / "ops" / "backlog.sqlite",
    )
    commands.add_parser("status")
    args = parser.parse_args()
    registry = PairRegistry(args.registry)
    try:
        if args.command == "status":
            result = registry.stats()
        elif args.command == "reconcile-backlog":
            result = {
                **reconcile_backlog_claims(registry, args.backlog),
                "registry": str(args.registry),
                "stats": registry.stats(),
            }
        elif args.command == "migrate":
            source = sqlite3.connect(f"file:{args.corpus}?mode=ro", uri=True)
            before = {
                "observations": int(
                    source.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
                ),
                "terminal_semantic_failures": int(
                    source.execute(
                        "SELECT COUNT(*) FROM failures WHERE failure IN "
                        f"({','.join('?' for _ in TERMINAL_SEMANTIC_FAILURES)})",
                        tuple(sorted(TERMINAL_SEMANTIC_FAILURES)),
                    ).fetchone()[0]
                ),
            }
            source.close()
            corpus_result = import_corpus(
                registry, args.corpus, args.catalog_pointer
            )
            manifest_result = import_manifest_paths(
                registry, args.manifest_root.rglob("*.json")
            )
            backlog_result = reconcile_backlog_claims(registry, args.backlog)
            registry.db.execute(
                "UPDATE controls SET value=? WHERE key='migration_pairs_reused'",
                (str(manifest_result["pairs_reused"]),),
            )
            registry.db.commit()
            result = {
                "schema": "JopePairRegistryMigrationReceipt.v1",
                "source_counts": before,
                **corpus_result,
                **manifest_result,
                **backlog_result,
                "registry": str(args.registry),
                "stats": registry.stats(),
            }
            completed = int(
                registry.db.execute(
                    "SELECT COUNT(*) FROM pairs WHERE state='completed'"
                ).fetchone()[0]
            )
            if completed != before["observations"]:
                raise RuntimeError(
                    "pair registry completed count does not match source observations"
                )
            registry.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            result["registry_sha256"] = sha256_file(args.registry)
            receipt = ROOT / "ops" / "pair-registry-migration.json"
            atomic_payload = {**result, "receipt_path": str(receipt)}
            from test_exact_common import atomic_json

            atomic_json(receipt, atomic_payload)
            result = atomic_payload
        else:
            paths = [
                path
                for root in args.roots
                for path in (
                    root.rglob("*.json") if root.is_dir() else [root]
                )
            ]
            result = {
                **import_manifest_paths(registry, paths),
                "registry": str(args.registry),
                "registry_sha256": sha256_file(args.registry),
                "stats": registry.stats(),
            }
        print(json.dumps(result, sort_keys=True))
    finally:
        registry.close()


if __name__ == "__main__":
    main()
