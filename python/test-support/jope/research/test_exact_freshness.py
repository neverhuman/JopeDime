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
import sys
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

if __name__ == "__main__":
    sys.modules.setdefault("test_exact_freshness", sys.modules[__name__])

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


from exact_freshness_inventory import _CheckpointInventoryMixin
from exact_freshness_retirement import _RetirementMixin
from exact_freshness_admission import _AdmissionMixin


class FreshnessSupervisor(
    _CheckpointInventoryMixin,
    _RetirementMixin,
    _AdmissionMixin,
):
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
        except (Exception,) as error:
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
