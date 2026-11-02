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
from test_exact_freshness import (
    BACKLOG,
    DB,
    FIRST_JOB_DEADLINE_MS,
    HOLD_DEADLINE_MS,
    LOCK,
    PREVIEW_MAX_AGE_MS,
    ROOT,
    SCQ,
    SOCKET,
    ScqScheduler,
    active_epoch_for_root,
    canonical_bytes,
    content_hash,
    epoch_for_checkpoint,
    freshness_scoped_manifest,
    manifest_epoch,
    now_ms,
)

class _CheckpointInventoryMixin:
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
