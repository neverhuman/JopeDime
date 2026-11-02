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

class _AdmissionMixin:
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
