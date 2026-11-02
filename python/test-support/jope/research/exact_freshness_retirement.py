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

class _RetirementMixin:
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
