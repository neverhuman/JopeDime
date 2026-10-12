"""Copy retained inputs first, then open SQLite copies with immutable read-only URIs.

No ResearchHWM/Outbox constructors, S3 clients, evaluator calls or writable live
database handles. WAL-bearing sources must be exported by their owner first.
"""

import hashlib
import json
import os
import sqlite3
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Copies:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.records = {}

    def copy(self, source, *, expected_sha=None, expected_bytes=None):
        source = Path(source).absolute()
        if any(p.is_symlink() for p in (source, *source.parents)):
            raise ValueError("symlink input forbidden")
        if source.is_relative_to(self.root):
            raise ValueError("source and copy roots must be disjoint")
        if source.suffix in (".db", ".sqlite"):
            for suffix in ("-wal", "-journal"):
                sibling = Path(str(source) + suffix)
                if sibling.exists() and sibling.stat().st_size:
                    raise ValueError(
                        "owner must export a quiescent SQLite snapshot first"
                    )
        sha = digest(source)
        size = source.stat().st_size
        if (
            expected_sha is not None
            and expected_sha != sha
            or expected_bytes is not None
            and expected_bytes != size
        ):
            raise ValueError(
                "input hash/length differs from retained artifact reference"
            )
        target = self.root / sha
        if not target.exists():
            # Exclusive creation protects prior evidence. No writable source handles.
            with source.open("rb") as src, target.open("xb") as dst:
                while chunk := src.read(1024**2):
                    dst.write(chunk)
                dst.flush()
                os.fsync(dst.fileno())
            target.chmod(0o444)
        if digest(target) != sha or digest(source) != sha:
            raise ValueError("input changed during copy")
        self.records[str(source)] = {
            "source": str(source),
            "copy": str(target),
            "sha256": sha,
            "bytes": size,
        }
        return target

    def json(self, source, **expected):
        return json.loads(self.copy(source, **expected).read_bytes())

    def artifact(self, ref, bench):
        path = Path(ref["path"])
        if not path.is_absolute() or not path.resolve().is_relative_to(
            Path(bench).resolve() / "runs"
        ):
            raise ValueError("artifact outside read-only benchmark runs")
        return self.json(path, expected_sha=ref["sha256"], expected_bytes=ref["bytes"])

    def database(self, source):
        path = self.copy(source)
        db = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            db.close()
            raise ValueError("copied SQLite integrity check failed")
        return db


def comparison_identity(evidence):
    fields = (
        "context_identity",
        "evaluator_sha256",
        "canonicalizer_sha256",
        "protocol",
        "role_map_sha256",
    )
    if any(not evidence.get(k) for k in fields):
        return None
    return tuple(evidence[k] for k in fields)


def qualified(e):
    return (
        e.get("outcome_class") == "qualified_complete"
        and e.get("qualified") is True
        and e.get("closure_verified") is True
        and e.get("parity_verified") is True
        and type(e.get("q_grid")) is int
    )


def ledger_events(copies, source):
    """Consume merged research_hwm events without instantiating its writer.

    v1 events have no publication clock: preserve event/version but do not assign
    an invented date or count first_label as an improvement. A dated receipt join
    is required before these rows can feed daily G or ΔH.
    """
    db = copies.database(source)
    try:
        rows = [
            dict(r) for r in db.execute("SELECT * FROM events ORDER BY hwm_key,version")
        ]
        for row in rows:
            row["publication"] = json.loads(row["publication"])
        return rows
    finally:
        db.close()


def benchmark_slots(copies, source):
    db = copies.database(source)
    try:
        spec = db.execute(
            "SELECT value FROM metadata WHERE key='spec_sha256'"
        ).fetchone()[0]
        rows = []
        for row in db.execute("SELECT rowid,* FROM slots ORDER BY rowid"):
            payload, evidence = json.loads(row["payload"]), json.loads(row["evidence"])
            if (
                payload["spec_sha256"] != spec
                or payload["context_id"] != row["context"]
            ):
                raise ValueError("slot/source identity mismatch")
            rows.append(
                {
                    "rowid": row["rowid"],
                    "payload": payload,
                    "evidence": evidence,
                    "execution_key": row["execution_key"],
                }
            )
        closed = {r[0] for r in db.execute("SELECT context FROM closed_contexts")}
        return rows, closed
    finally:
        db.close()
