#!/usr/bin/env python3
"""Checkpoint freshness authority for the internal self-play queue.

The supervisor owns stale-epoch decisions under one process lock.  It never
interrupts dispatching/running work and retires a pair only after SCQ, backlog,
manifest, and pair-registry provenance agree that the attempt is held and
unstarted.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import socket
import socketserver
import sqlite3
import subprocess
import threading
import time
from pathlib import Path
from typing import Mapping

from test_exact_common import (
    NON_INTERRUPTIBLE_STATES,
    USEFUL_QUEUE_STATES,
    atomic_json,
    sha256_file,
)
from test_exact_pair_registry import PairRegistry

ROOT = Path("/home/ubuntu/jope-flywheel/test_exact_gen0")
SCQ = Path("/home/ubuntu/smart_cluster/target/release/scq")
SOCKET = ROOT / "ops" / "freshness.sock"
LOCK = ROOT / "ops" / "freshness.lock"
DB = ROOT / "ops" / "freshness.sqlite"
BACKLOG = ROOT / "ops" / "backlog.sqlite"
PREVIEW_MAX_AGE_MS = 5 * 60_000
HOLD_DEADLINE_MS = 30_000
FIRST_JOB_DEADLINE_MS = 5 * 60_000


def now_ms() -> int:
    return int(time.time() * 1000)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def content_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def epoch_for_checkpoint(checkpoint_sha256: str) -> str:
    if len(checkpoint_sha256) != 64:
        raise ValueError("checkpoint epoch requires a SHA-256 checkpoint identity")
    return f"checkpoint-{checkpoint_sha256}"


def manifest_epoch(manifest: Mapping[str, object]) -> str:
    value = manifest.get("checkpoint_epoch")
    if isinstance(value, str) and value:
        return value
    authority = manifest.get("label_authority")
    if isinstance(authority, Mapping):
        checkpoint = authority.get("model_checkpoint_sha256")
        if isinstance(checkpoint, str) and len(checkpoint) == 64:
            return epoch_for_checkpoint(checkpoint)
    return "legacy-jope-testonly"


def freshness_scoped_manifest(manifest: Mapping[str, object]) -> bool:
    """Only adaptive and legacy self-play evaluation may be held or retired."""

    if not manifest:
        return False
    lane = str(manifest.get("evaluation_lane") or "")
    if lane in {"checkpoint_gate", "sealed_audit", "evaluator_canary"}:
        return False
    return (
        manifest.get("schema") == "JopeFlywheelCorpusEvalShard.v1"
        and isinstance(manifest.get("candidates"), list)
    )


def active_epoch_for_root(root: Path = ROOT) -> str | None:
    for relative in ("training/learning.json", "training/accepted.json"):
        path = root / relative
        if not path.is_file():
            continue
        value = json.loads(path.read_text())
        checkpoint = str(value.get("checkpoint_sha256") or "")
        if len(checkpoint) == 64:
            return epoch_for_checkpoint(checkpoint)
    return None


class ScqScheduler:
    def __init__(self, binary: Path = SCQ):
        self.binary = binary

    def _run(self, *args: str) -> dict:
        result = subprocess.run(
            [str(self.binary), "--json", *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout).strip())
        output = result.stdout.strip()
        if not output:
            return {}
        try:
            value = json.loads(output)
        except json.JSONDecodeError:
            return {"output": output}
        return value if isinstance(value, dict) else {"output": value}

    def jobs(self) -> list[dict]:
        return self._run("ls").get("Jobs", {}).get("jobs", [])

    def hold(self, job_id: str) -> None:
        self._run("hold", job_id)

    def cancel(self, job_id: str) -> None:
        self._run("cancel", job_id)


class FreshnessSupervisor:
    def __init__(
        self,
        root: Path = ROOT,
        *,
        scheduler: object | None = None,
        database: Path | None = None,
        backlog: Path | None = None,
        pair_registry: Path | None = None,
    ):
        self.root = root
        self.scheduler = scheduler or ScqScheduler()
        self.database = database or root / "ops" / "freshness.sqlite"
        self.backlog = backlog or root / "ops" / "backlog.sqlite"
        self.pair_registry = pair_registry or root / "ops" / "pair-registry.sqlite"
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.database, timeout=60, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=60000")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS checkpoints(
              epoch TEXT PRIMARY KEY,
              checkpoint_sha256 TEXT NOT NULL UNIQUE,
              activated_at_ms INTEGER NOT NULL,
              superseded_at_ms INTEGER
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS previews(
              receipt_sha256 TEXT PRIMARY KEY,
              kind TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              created_at_ms INTEGER NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS mutations(
              idempotency_key TEXT PRIMARY KEY,
              tool TEXT NOT NULL,
              preview_sha256 TEXT NOT NULL,
              result_json TEXT NOT NULL,
              created_at_ms INTEGER NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS audit_events(
              id INTEGER PRIMARY KEY,
              kind TEXT NOT NULL,
              epoch TEXT,
              job_id TEXT,
              detail_json TEXT NOT NULL,
              created_at_ms INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS controls(
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            ) WITHOUT ROWID;
            INSERT OR IGNORE INTO controls VALUES('active_epoch','');
            INSERT OR IGNORE INTO controls VALUES('active_checkpoint','');
            INSERT OR IGNORE INTO controls VALUES('admission_paused','0');
            """
        )
        self.db.commit()
        self._mutex = threading.RLock()

    def close(self) -> None:
        self.db.close()

    def _control(self, key: str) -> str:
        row = self.db.execute(
            "SELECT value FROM controls WHERE key=?", (key,)
        ).fetchone()
        return "" if row is None else str(row["value"])

    def _set_control(self, key: str, value: str) -> None:
        self.db.execute(
            "INSERT INTO controls VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def _event(
        self,
        kind: str,
        *,
        epoch: str | None = None,
        job_id: str | None = None,
        detail: Mapping[str, object] | None = None,
    ) -> None:
        self.db.execute(
            "INSERT INTO audit_events(kind,epoch,job_id,detail_json,created_at_ms) "
            "VALUES(?,?,?,?,?)",
            (kind, epoch, job_id, json.dumps(detail or {}, sort_keys=True), now_ms()),
        )

    def pointer_checkpoint(self) -> str | None:
        for relative in ("training/learning.json", "training/accepted.json"):
            path = self.root / relative
            if not path.is_file():
                continue
            value = json.loads(path.read_text())
            digest = str(value.get("checkpoint_sha256") or "")
            checkpoint_path = Path(str(value.get("checkpoint") or ""))
            if len(digest) != 64:
                continue
            if checkpoint_path.is_file() and sha256_file(checkpoint_path) != digest:
                raise RuntimeError(f"{relative}: checkpoint pointer hash drift")
            return digest
        return None

    def sync_checkpoint(self, checkpoint_sha256: str | None = None) -> dict:
        with self._mutex:
            checkpoint_sha256 = checkpoint_sha256 or self.pointer_checkpoint()
            if checkpoint_sha256 is None:
                return self.get_active_checkpoint()
            epoch = epoch_for_checkpoint(checkpoint_sha256)
            previous = self._control("active_epoch")
            if previous == epoch:
                return self.get_active_checkpoint()
            timestamp = now_ms()
            if previous:
                self.db.execute(
                    "UPDATE checkpoints SET superseded_at_ms=COALESCE(superseded_at_ms,?) "
                    "WHERE epoch=?",
                    (timestamp, previous),
                )
            self.db.execute(
                "INSERT OR IGNORE INTO checkpoints VALUES(?,?,?,NULL)",
                (epoch, checkpoint_sha256, timestamp),
            )
            self._set_control("active_epoch", epoch)
            self._set_control("active_checkpoint", checkpoint_sha256)
            self._event(
                "checkpoint_advanced",
                epoch=epoch,
                detail={"previous_epoch": previous or None},
            )
            self.db.commit()
            return self.get_active_checkpoint()

    def _claims(self) -> dict[str, dict]:
        if not self.backlog.is_file():
            return {}
        db = sqlite3.connect(f"file:{self.backlog}?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        try:
            return {
                str(row["job_id"]): dict(row)
                for row in db.execute("SELECT * FROM claims")
            }
        finally:
            db.close()

    @staticmethod
    def _job_id(row: Mapping[str, object]) -> str:
        return str(row.get("id") or row.get("job_id") or "")

    def _inventory(self) -> list[dict]:
        claims = self._claims()
        result = []
        seen = set()
        for raw in self.scheduler.jobs():
            job = dict(raw)
            job_id = self._job_id(job)
            claim = claims.get(job_id)
            if claim is None:
                continue
            manifest_path = (
                Path(str(claim["manifest"])) if claim and claim.get("manifest") else None
            )
            manifest = {}
            manifest_sha256 = None
            if manifest_path and manifest_path.is_file():
                manifest_sha256 = sha256_file(manifest_path)
                if manifest_sha256 != claim.get("manifest_sha256"):
                    raise RuntimeError(f"{job_id}: backlog manifest hash drift")
                manifest = json.loads(manifest_path.read_text())
            if not freshness_scoped_manifest(manifest):
                continue
            epoch = manifest_epoch(manifest)
            seen.add(job_id)
            result.append(
                {
                    **job,
                    "job_id": job_id,
                    "epoch": epoch,
                    "manifest": str(manifest_path) if manifest_path else None,
                    "manifest_sha256": manifest_sha256,
                    "claim_state": claim.get("state") if claim else None,
                    "reservation_owner": (
                        claim.get("reservation_owner") if claim else None
                    ),
                }
            )
        for job_id, claim in claims.items():
            if job_id in seen or claim.get("state") != "checked":
                continue
            manifest_path = Path(str(claim.get("manifest") or ""))
            if not manifest_path.is_file():
                continue
            manifest_sha256 = sha256_file(manifest_path)
            if manifest_sha256 != claim.get("manifest_sha256"):
                raise RuntimeError(f"{job_id}: backlog manifest hash drift")
            manifest = json.loads(manifest_path.read_text())
            if not freshness_scoped_manifest(manifest):
                continue
            result.append(
                {
                    "id": job_id,
                    "job_id": job_id,
                    "state": "checked",
                    "epoch": manifest_epoch(manifest),
                    "manifest": str(manifest_path),
                    "manifest_sha256": manifest_sha256,
                    "claim_state": "checked",
                    "reservation_owner": claim.get("reservation_owner"),
                    "outside_scq": True,
                }
            )
        return result

    def get_active_checkpoint(self) -> dict:
        epoch = self._control("active_epoch")
        checkpoint = self._control("active_checkpoint")
        row = self.db.execute(
            "SELECT * FROM checkpoints WHERE epoch=?", (epoch,)
        ).fetchone()
        return {
            "checkpoint_epoch": epoch or None,
            "checkpoint_sha256": checkpoint or None,
            "activated_at_ms": None if row is None else row["activated_at_ms"],
            "admission_paused": self._control("admission_paused") == "1",
        }

    def list_queue_epochs(self) -> dict:
        active = self._control("active_epoch")
        counts: dict[str, dict[str, int]] = {}
        for job in self._inventory():
            state = str(job.get("state") or "unknown")
            epoch = str(job["epoch"])
            counts.setdefault(epoch, {})
            counts[epoch][state] = counts[epoch].get(state, 0) + 1
        return {
            "active_epoch": active or None,
            "epochs": [
                {
                    "epoch": epoch,
                    "active": epoch == active,
                    "states": states,
                    "useful_depth": sum(
                        states.get(state, 0) for state in USEFUL_QUEUE_STATES
                    ),
                    "held": states.get("held", 0),
                    "running": states.get("running", 0),
                    "dispatching": states.get("dispatching", 0),
                }
                for epoch, states in sorted(counts.items())
            ],
        }

    def get_freshness_status(self) -> dict:
        active = self.get_active_checkpoint()
        epochs = self.list_queue_epochs()
        active_epoch = active["checkpoint_epoch"]
        stale = [
            row
            for row in epochs["epochs"]
            if not row["active"] and row["epoch"] != "unknown"
        ]
        activated = active.get("activated_at_ms")
        first_running = None
        for job in self._inventory():
            if job["epoch"] == active_epoch and job.get("state") == "running":
                observed = int(job.get("created_at_ms") or job.get("submitted_at_ms") or 0)
                first_running = observed if first_running is None else min(first_running, observed)
        latency = (
            None
            if activated is None or first_running is None
            else max(0, first_running - int(activated))
        )
        return {
            "schema": "JopeSelfplayFreshnessStatus.v1",
            **active,
            "queue_epochs": epochs["epochs"],
            "stale_epoch_count": len(stale),
            "stale_useful_depth": sum(row["useful_depth"] for row in stale),
            "stale_held": sum(row["held"] for row in stale),
            "checkpoint_to_first_job_latency_ms": latency,
            "first_job_deadline_ms": FIRST_JOB_DEADLINE_MS,
            "healthy": not (
                latency is not None and latency > FIRST_JOB_DEADLINE_MS
            ),
        }

    def explain_job_provenance(self, job_id: str) -> dict:
        job = next(
            (row for row in self._inventory() if row["job_id"] == job_id), None
        )
        if job is None:
            raise KeyError(f"unknown job {job_id}")
        pairs = []
        if job.get("manifest_sha256") and self.pair_registry.is_file():
            registry = PairRegistry(self.pair_registry)
            try:
                pairs = [
                    dict(row)
                    for row in registry.db.execute(
                        "SELECT * FROM pairs WHERE manifest_sha256=? "
                        "ORDER BY task,dataset_hash,semantic_sha256",
                        (job["manifest_sha256"],),
                    )
                ]
            finally:
                registry.close()
        return {
            "job": job,
            "active_epoch": self._control("active_epoch") or None,
            "pair_count": len(pairs),
            "pairs": pairs,
        }

    def _write_preview(self, kind: str, payload: dict) -> dict:
        envelope = {
            "schema": "JopeFreshnessPreview.v1",
            "kind": kind,
            "created_at_ms": now_ms(),
            "payload": payload,
        }
        digest = content_hash(envelope)
        self.db.execute(
            "INSERT OR IGNORE INTO previews VALUES(?,?,?,?)",
            (digest, kind, json.dumps(envelope, sort_keys=True), envelope["created_at_ms"]),
        )
        self.db.commit()
        return {**envelope, "preview_receipt_sha256": digest}

    def preview_stale_retirement(self) -> dict:
        active = self._control("active_epoch")
        actions = []
        preserved = []
        for job in self._inventory():
            state = str(job.get("state") or "")
            if job["epoch"] == active:
                continue
            if state in NON_INTERRUPTIBLE_STATES:
                preserved.append(
                    {"job_id": job["job_id"], "state": state, "reason": "started"}
                )
                continue
            if state not in {
                *USEFUL_QUEUE_STATES,
                "held",
                "cancelled",
                "checked",
            }:
                continue
            action = (
                "finalize_cancelled"
                if state == "cancelled"
                else "retire_checked_unstarted"
                if state == "checked"
                else "cancel_after_hold"
            )
            actions.append(
                {
                    "job_id": job["job_id"],
                    "epoch": job["epoch"],
                    "observed_state": state,
                    "action": action,
                    "manifest": job.get("manifest"),
                    "manifest_sha256": job.get("manifest_sha256"),
                    "claim_state": job.get("claim_state"),
                }
            )
        return self._write_preview(
            "stale_retirement",
            {
                "active_epoch": active or None,
                "actions": sorted(actions, key=lambda row: row["job_id"]),
                "preserved_started": sorted(
                    preserved, key=lambda row: row["job_id"]
                ),
            },
        )

    def _load_preview(self, receipt_sha256: str, kind: str | None = None) -> dict:
        row = self.db.execute(
            "SELECT * FROM previews WHERE receipt_sha256=?", (receipt_sha256,)
        ).fetchone()
        if row is None:
            raise RuntimeError("unknown preview receipt")
        if kind is not None and row["kind"] != kind:
            raise RuntimeError("preview receipt kind mismatch")
        if now_ms() - int(row["created_at_ms"]) > PREVIEW_MAX_AGE_MS:
            raise RuntimeError("preview receipt expired")
        return json.loads(row["payload_json"])

    def _mutate(
        self,
        tool: str,
        preview_sha256: str,
        idempotency_key: str,
        callback,
    ) -> dict:
        if not idempotency_key.strip():
            raise ValueError("mutating freshness tools require an idempotency key")
        prior = self.db.execute(
            "SELECT * FROM mutations WHERE idempotency_key=?",
            (idempotency_key,),
        ).fetchone()
        if prior is not None:
            if prior["tool"] != tool or prior["preview_sha256"] != preview_sha256:
                raise RuntimeError("idempotency key was used for a different mutation")
            return json.loads(prior["result_json"])
        self._load_preview(preview_sha256)
        result = callback()
        self.db.execute(
            "INSERT INTO mutations VALUES(?,?,?,?,?)",
            (
                idempotency_key,
                tool,
                preview_sha256,
                json.dumps(result, sort_keys=True),
                now_ms(),
            ),
        )
        self.db.commit()
        return result

    def apply_stale_retirement(
        self, preview_receipt_sha256: str, idempotency_key: str
    ) -> dict:
        preview = self._load_preview(preview_receipt_sha256, "stale_retirement")

        def apply() -> dict:
            active = self._control("active_epoch")
            inventory = {row["job_id"]: row for row in self._inventory()}
            held = retired = 0
            skipped = []
            registry = PairRegistry(self.pair_registry)
            try:
                for action in preview["payload"]["actions"]:
                    job_id = action["job_id"]
                    current = inventory.get(job_id)
                    if current is None:
                        skipped.append({"job_id": job_id, "reason": "not_active"})
                        continue
                    state = str(current.get("state") or "")
                    if current["epoch"] == active or state in NON_INTERRUPTIBLE_STATES:
                        skipped.append({"job_id": job_id, "reason": "started_or_current"})
                        continue
                    if state in USEFUL_QUEUE_STATES:
                        self.scheduler.hold(job_id)
                        held += 1
                        self._event("stale_job_held", epoch=current["epoch"], job_id=job_id)
                inventory = {row["job_id"]: row for row in self._inventory()}
                for action in preview["payload"]["actions"]:
                    job_id = action["job_id"]
                    current = inventory.get(job_id)
                    if current is None or current.get("state") not in {
                        "held",
                        "cancelled",
                        "checked",
                    }:
                        continue
                    expected_claim = (
                        "checked"
                        if current.get("state") == "checked"
                        else "submitted"
                    )
                    if (
                        current["epoch"] == active
                        or not current.get("manifest_sha256")
                        or current.get("claim_state") != expected_claim
                    ):
                        skipped.append(
                            {"job_id": job_id, "reason": "provenance_audit_failed"}
                        )
                        continue
                    states = registry.manifest_state_counts(
                        current["manifest_sha256"]
                    )
                    if set(states) == {"available"}:
                        self._mark_claim_superseded(
                            job_id, current["manifest_sha256"]
                        )
                        continue
                    expected_pair_state = (
                        "reserved"
                        if current.get("state") == "checked"
                        else "submitted"
                    )
                    if set(states) != {expected_pair_state}:
                        skipped.append(
                            {"job_id": job_id, "reason": "pair_state_audit_failed"}
                        )
                        continue
                    retirement = content_hash(
                        {
                            "preview": preview_receipt_sha256,
                            "job_id": job_id,
                            "manifest_sha256": current["manifest_sha256"],
                        }
                    )
                    # The scheduler state is checked immediately before cancel;
                    # this is the hard guard against interrupting started work.
                    latest = {
                        row["job_id"]: row for row in self._inventory()
                    }.get(job_id)
                    if latest is None or latest.get("state") not in {
                        "held",
                        "cancelled",
                        "checked",
                    }:
                        skipped.append(
                            {"job_id": job_id, "reason": "state_changed_before_cancel"}
                        )
                        continue
                    if latest.get("state") == "held":
                        self.scheduler.cancel(job_id)
                    registry.supersede_unstarted(
                        current["manifest_sha256"],
                        job_id,
                        retirement_receipt_sha256=retirement,
                    )
                    self._mark_claim_superseded(
                        job_id, current["manifest_sha256"]
                    )
                    retired += 1
                    self._event(
                        "stale_job_retired",
                        epoch=current["epoch"],
                        job_id=job_id,
                        detail={"retirement_receipt_sha256": retirement},
                    )
                registry.db.commit()
            finally:
                registry.close()
            return {
                "preview_receipt_sha256": preview_receipt_sha256,
                "held": held,
                "retired": retired,
                "skipped": skipped,
                "running_interrupted": 0,
            }

        return self._mutate(
            "apply_stale_retirement",
            preview_receipt_sha256,
            idempotency_key,
            apply,
        )

    def _mark_claim_superseded(
        self, job_id: str, manifest_sha256: str
    ) -> None:
        if not self.backlog.is_file():
            return
        backlog = sqlite3.connect(self.backlog, timeout=60)
        try:
            backlog.execute(
                """
                UPDATE claims SET state='superseded_unstarted',updated_at_ms=?
                WHERE job_id=? AND manifest_sha256=?
                  AND state IN ('submitted','checked')
                """,
                (now_ms(), job_id, manifest_sha256),
            )
            backlog.commit()
        finally:
            backlog.close()

    def _admission_mutation(
        self,
        paused: bool,
        preview_receipt_sha256: str,
        idempotency_key: str,
    ) -> dict:
        tool = "pause_admission" if paused else "resume_admission"

        def apply() -> dict:
            registry = PairRegistry(self.pair_registry)
            try:
                changed = registry.set_admission_paused(
                    paused, reason=f"freshness:{idempotency_key}"
                )
            finally:
                registry.close()
            self._set_control("admission_paused", "1" if paused else "0")
            return {"admission_paused": paused, "changed": changed}

        return self._mutate(
            tool, preview_receipt_sha256, idempotency_key, apply
        )

    def pause_admission(
        self, preview_receipt_sha256: str, idempotency_key: str
    ) -> dict:
        return self._admission_mutation(
            True, preview_receipt_sha256, idempotency_key
        )

    def resume_admission(
        self, preview_receipt_sha256: str, idempotency_key: str
    ) -> dict:
        return self._admission_mutation(
            False, preview_receipt_sha256, idempotency_key
        )

    def request_micro_wave(
        self,
        preview_receipt_sha256: str,
        idempotency_key: str,
        *,
        reason: str = "operator_request",
    ) -> dict:
        def apply() -> dict:
            request = {
                "schema": "JopeMicroWaveRequest.v1",
                "checkpoint_epoch": self._control("active_epoch"),
                "checkpoint_sha256": self._control("active_checkpoint"),
                "requested_at_ms": now_ms(),
                "reason": reason,
                "idempotency_key": idempotency_key,
            }
            path = self.root / "ops" / "micro-wave-requests"
            path.mkdir(parents=True, exist_ok=True)
            target = path / f"{content_hash(request)}.json"
            atomic_json(target, request)
            return {**request, "request": str(target)}

        return self._mutate(
            "request_micro_wave",
            preview_receipt_sha256,
            idempotency_key,
            apply,
        )

    def _hold_stale_queued(self) -> int:
        active = self._control("active_epoch")
        held = 0
        for job in self._inventory():
            if job["epoch"] == active or job.get("state") not in USEFUL_QUEUE_STATES:
                continue
            self.scheduler.hold(job["job_id"])
            self._event(
                "stale_job_held",
                epoch=job["epoch"],
                job_id=job["job_id"],
                detail={"automatic": True},
            )
            held += 1
        self.db.commit()
        return held

    def cycle(self, *, audit_only: bool | None = None) -> dict:
        self.sync_checkpoint()
        if audit_only is None:
            audit_only = os.environ.get("JOPE_FRESHNESS_AUDIT_ONLY") == "1"
        automatically_held = 0 if audit_only else self._hold_stale_queued()
        preview = self.preview_stale_retirement()
        status = self.get_freshness_status()
        result = {
            **status,
            "latest_preview_receipt_sha256": preview["preview_receipt_sha256"],
            "audit_only": audit_only,
            "automatically_held": automatically_held,
            "loaded_bundle_sha256": os.environ.get("JOPE_LOADED_BUNDLE_SHA256"),
        }
        atomic_json(self.root / "ops" / "freshness-status.json", result)
        return result

    def dispatch(self, method: str, arguments: Mapping[str, object]) -> dict:
        allowed = {
            "get_freshness_status": lambda: self.get_freshness_status(),
            "get_active_checkpoint": lambda: self.get_active_checkpoint(),
            "list_queue_epochs": lambda: self.list_queue_epochs(),
            "explain_job_provenance": lambda: self.explain_job_provenance(
                str(arguments["job_id"])
            ),
            "preview_stale_retirement": lambda: self.preview_stale_retirement(),
            "apply_stale_retirement": lambda: self.apply_stale_retirement(
                str(arguments["preview_receipt_sha256"]),
                str(arguments["idempotency_key"]),
            ),
            "pause_admission": lambda: self.pause_admission(
                str(arguments["preview_receipt_sha256"]),
                str(arguments["idempotency_key"]),
            ),
            "resume_admission": lambda: self.resume_admission(
                str(arguments["preview_receipt_sha256"]),
                str(arguments["idempotency_key"]),
            ),
            "request_micro_wave": lambda: self.request_micro_wave(
                str(arguments["preview_receipt_sha256"]),
                str(arguments["idempotency_key"]),
                reason=str(arguments.get("reason") or "operator_request"),
            ),
        }
        if method not in allowed:
            raise KeyError(f"unsupported freshness method {method}")
        with self._mutex:
            return allowed[method]()


class _SocketHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        line = self.rfile.readline(4 << 20)
        if not line:
            return
        try:
            request = json.loads(line)
            result = self.server.supervisor.dispatch(  # type: ignore[attr-defined]
                str(request["method"]), request.get("arguments") or {}
            )
            response = {"ok": True, "result": result}
        except Exception as error:
            response = {"ok": False, "error": type(error).__name__, "detail": str(error)}
        self.wfile.write(canonical_bytes(response) + b"\n")


class _ThreadedUnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


def socket_request(
    method: str,
    arguments: Mapping[str, object] | None = None,
    *,
    socket_path: Path = SOCKET,
) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(30)
        client.connect(str(socket_path))
        client.sendall(
            canonical_bytes({"method": method, "arguments": arguments or {}}) + b"\n"
        )
        chunks = []
        while True:
            chunk = client.recv(1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
    response = json.loads(b"".join(chunks))
    if not response.get("ok"):
        raise RuntimeError(str(response.get("detail") or response.get("error")))
    return response["result"]


def serve(root: Path, socket_path: Path, interval: float) -> None:
    lock_path = root / "ops" / "freshness.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+")
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    supervisor = FreshnessSupervisor(root)
    server = _ThreadedUnixServer(str(socket_path), _SocketHandler)
    server.supervisor = supervisor  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        while True:
            supervisor.cycle()
            time.sleep(max(1.0, interval))
    finally:
        server.shutdown()
        server.server_close()
        supervisor.close()
        socket_path.unlink(missing_ok=True)
        lock_handle.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    socket_path = args.socket or args.root / "ops" / "freshness.sock"
    if args.once:
        supervisor = FreshnessSupervisor(args.root)
        try:
            print(json.dumps(supervisor.cycle(), sort_keys=True))
        finally:
            supervisor.close()
        return
    serve(args.root, socket_path, args.interval)


if __name__ == "__main__":
    main()
