"""Content-bind retained legacy ledgers without promoting their scores to labels.

This is a bounded inventory of explicit controller sources and fenced attempts.
It does not assert that those sources cover every historical fitting namespace.
Every occurrence remains seen/unavailable until original execution is verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def inventory(root, state_path, fence_path, output, seconds):
    root = root.resolve()
    state_bytes, fence_bytes = state_path.read_bytes(), fence_path.read_bytes()
    state, fence = json.loads(state_bytes), json.loads(fence_bytes)
    if not fence.get("legacy_dispatch_fenced") or not fence.get("descendants_reconciled"):
        raise RuntimeError("legacy dispatch fence is incomplete")
    frozen_path = fence_path.parent / "frozen.json"
    frozen_bytes = frozen_path.read_bytes()
    if digest(frozen_bytes) != fence["frozen_sha256"]:
        raise RuntimeError("legacy frozen capture changed")
    paths = {root / path for path in state["teaching_sources"]}
    for item in json.loads(frozen_bytes)["artifacts"]:
        path = Path(item["path"])
        if path.name.endswith(".attempt.json") or path.name in (
            "observations.json",
            "observations.jsonl",
            "search.json",
        ):
            paths.add(path)
    paths = sorted(paths)
    for path in paths:
        if not path.resolve().is_relative_to(root):
            raise RuntimeError("historical source escapes workspace")
    output.mkdir(parents=True, exist_ok=True)
    request = dict(
        schema="parallel.legacy_fit_inventory_request.v1",
        state_sha256=digest(state_bytes),
        fence_sha256=digest(fence_bytes),
        frozen_sha256=digest(frozen_bytes),
        driver_sha256=digest(Path(__file__).read_bytes()),
        source_paths=[str(p.relative_to(root)) for p in paths],
        scope="explicit retained controller teaching sources plus frozen attempts",
        full_namespace_coverage=False,
        numerical_fits_allowed=False,
    )
    request_path = output / "request.json"
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise RuntimeError("inventory resume binding changed")
    else:
        with request_path.open("x") as stream:
            json.dump(request, stream, sort_keys=True, indent=2)
    database = output / "inventory.sqlite"
    db = sqlite3.connect(database)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute(
        "CREATE TABLE IF NOT EXISTS artifacts(path TEXT PRIMARY KEY, sha256 TEXT, bytes INTEGER, status TEXT, rows INTEGER, detail TEXT)"
    )
    db.execute(
        "CREATE TABLE IF NOT EXISTS occurrences(path TEXT, ordinal INTEGER, classification TEXT, program TEXT, training_csv_sha256 TEXT, supplied_identity TEXT, original_json_sha256 TEXT, PRIMARY KEY(path,ordinal)) WITHOUT ROWID"
    )
    began = time.monotonic()
    errors = 0
    for path in paths:
        name = str(path.relative_to(root))
        if db.execute("SELECT 1 FROM artifacts WHERE path=?", (name,)).fetchone():
            continue
        if time.monotonic() - began >= seconds:
            break
        sha, size, rows = None, None, []
        try:
            data = path.read_bytes()
            sha, size = digest(data), len(data)
            if path.suffix == ".jsonl":
                if data and not data.endswith(b"\n"):
                    raise ValueError("torn original journal tail")
                value = [json.loads(line) for line in data.splitlines() if line.strip()]
            else:
                value = json.loads(data)
            rows = value if isinstance(value, list) else [value]
            if not all(isinstance(row, dict) for row in rows):
                raise ValueError("unexpected original record shape")
            occurrences = []
            for number, row in enumerate(rows):
                # Names/digests carried by legacy rows remain unverified
                # provenance. They never become the authority's numerical key.
                program = row.get("genome")
                classification = (
                    "seen_execution_result_unavailable"
                    if isinstance(program, str) and program.strip()
                    else "unresolved_or_conflicting_evidence"
                )
                original = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                occurrences.append(
                    (
                        name,
                        number,
                        classification,
                        program if isinstance(program, str) else None,
                        row.get("training_sha256"),
                        row.get("evaluation_identity_sha256"),
                        digest(original),
                    )
                )
            with db:
                db.executemany("INSERT INTO occurrences VALUES(?,?,?,?,?,?,?)", occurrences)
                db.execute(
                    "INSERT INTO artifacts VALUES(?,?,?,?,?,?)",
                    (
                        name,
                        sha,
                        size,
                        "inventoried",
                        len(rows),
                        "Original process completion and numerical identity not verified; scores informational only",
                    ),
                )
        except (OSError, ValueError, TypeError) as error:
            errors += 1
            with db:
                db.execute(
                    "INSERT INTO artifacts VALUES(?,?,?,?,?,?)",
                    (name, sha, size, "unresolved", 0, str(error)),
                )
        if (db.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] % 500) == 0:
            print(
                json.dumps(
                    dict(
                        artifacts=db.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0],
                        elapsed_seconds=round(time.monotonic() - began, 2),
                    )
                ),
                flush=True,
            )
    counts = dict(db.execute("SELECT classification,COUNT(*) FROM occurrences GROUP BY classification"))
    artifacts = db.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
    errors = db.execute("SELECT COUNT(*) FROM artifacts WHERE status='unresolved'").fetchone()[0]
    copies = db.execute(
        "SELECT COALESCE(SUM(n-1),0) FROM (SELECT COUNT(*) n FROM artifacts WHERE sha256 IS NOT NULL GROUP BY sha256 HAVING COUNT(*)>1)"
    ).fetchone()[0]
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.close()
    result = dict(
        schema="parallel.legacy_fit_inventory.v1",
        request_sha256=digest(request_path.read_bytes()),
        listed_artifacts=len(paths),
        inventoried_artifacts=artifacts,
        unprocessed_artifacts=len(paths) - artifacts,
        classifications=counts,
        unresolved_files=errors,
        copied_artifact_occurrences=copies,
        verified_reusable_completions=0,
        verified_terminal_failures=0,
        elapsed_seconds=time.monotonic() - began,
        full_namespace_coverage=False,
        registry_import_authorized=False,
        required_exclusion="all unproven production namespaces remain blocked",
        database_sha256=digest(database.read_bytes()),
        final_tests_opened=False,
    )
    suffix = str(time.time_ns())
    with (output / ("receipt-" + suffix + ".json")).open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "state", "fence", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=300)
    args = parser.parse_args()
    if not 0 < args.seconds <= 1800:
        parser.error("inventory cap must be within 0..1800 seconds")
    print(json.dumps(inventory(args.root, args.state, args.fence, args.output, args.seconds), sort_keys=True))
