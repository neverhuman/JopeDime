"""Append-only JRB-02 score-history API used by the JRB-01 study adapter."""

from __future__ import annotations

import fcntl
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Mapping

SCHEMA = "jope.score-history/v1"
TASKS = {"regression", "binary"}
REQUIRED = {
    "schema",
    "ts",
    "model",
    "model_sha",
    "task",
    "dataset",
    "family",
    "split",
    "metric",
    "value",
    "higher_is_better",
    "source",
    "round",
    "host",
}


def validate_row(row: Mapping[str, object]) -> dict[str, object]:
    missing = REQUIRED - row.keys()
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    if row["schema"] != SCHEMA:
        raise ValueError("unsupported score schema")
    try:
        timestamp = datetime.fromisoformat(str(row["ts"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("ts must be ISO-8601 UTC") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
        raise ValueError("ts must be ISO-8601 UTC")
    if row["task"] not in TASKS:
        raise ValueError("task must be regression or binary")
    for key in ("model", "dataset", "split", "metric", "source", "host"):
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    value = row["value"]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("value must be finite numeric")
    if not isinstance(row["higher_is_better"], bool):
        raise ValueError("higher_is_better must be boolean")
    if row["family"] is not None and not isinstance(row["family"], str):
        raise ValueError("family must be a string or null")
    if row["model_sha"] is not None and not isinstance(row["model_sha"], str):
        raise ValueError("model_sha must be a string or null")
    round_number = row["round"]
    if round_number is not None and (
        isinstance(round_number, bool) or not isinstance(round_number, int)
    ):
        raise ValueError("round must be an integer or null")
    return dict(row)


def dedupe_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(row[key] for key in ("model", "dataset", "split", "metric", "source"))


def read_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(validate_row(json.loads(line)))
            except Exception as exc:
                raise ValueError(
                    f"invalid score history row {line_number}: {exc}"
                ) from exc
    return rows


def append_rows(rows: Iterable[Mapping[str, object]], path: Path) -> int:
    """Append unseen rows while holding the history file's exclusive lock."""
    candidates = [validate_row(row) for row in rows]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = {dedupe_key(row) for row in read_rows(path)}
        added = []
        for row in candidates:
            key = dedupe_key(row)
            if key not in existing:
                added.append(row)
                existing.add(key)
        if added:
            with path.open("a", encoding="utf-8") as stream:
                for row in added:
                    stream.write(
                        json.dumps(
                            row, sort_keys=True, allow_nan=False, separators=(",", ":")
                        )
                        + "\n"
                    )
                stream.flush()
        return len(added)
