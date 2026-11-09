from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
from typing import Any, BinaryIO


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError, UnicodeError):
        return {}


class JsonlTail:
    """Incremental JSONL reader that tolerates partial writes and rename rotation."""

    def __init__(
        self,
        path: Path,
        *,
        limit: int | None,
        initial_bytes: int | None = 1024 * 1024,
    ) -> None:
        self.path = path
        self.rows: deque[dict[str, Any]] = deque(maxlen=limit)
        self.initial_bytes = initial_bytes
        self.handle: BinaryIO | None = None
        self.inode: tuple[int, int] | None = None
        self.partial = b""

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
        self.handle = None
        self.inode = None
        self.partial = b""

    def _open(self) -> bool:
        try:
            handle = self.path.open("rb")
            stat = os.fstat(handle.fileno())
        except OSError:
            return False
        if self.initial_bytes is not None and stat.st_size > self.initial_bytes:
            handle.seek(stat.st_size - self.initial_bytes)
            handle.readline()
        self.handle = handle
        self.inode = (stat.st_dev, stat.st_ino)
        self.partial = b""
        return True

    def _consume(self) -> None:
        assert self.handle is not None
        data = self.partial + self.handle.read()
        parts = data.split(b"\n")
        self.partial = parts.pop()
        for raw in parts:
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                self.rows.append(value)

    def poll(self) -> list[dict[str, Any]]:
        if self.handle is None and not self._open():
            return list(self.rows)
        assert self.handle is not None
        self._consume()
        try:
            path_stat = self.path.stat()
            current = (path_stat.st_dev, path_stat.st_ino)
        except OSError:
            return list(self.rows)
        if current != self.inode or path_stat.st_size < self.handle.tell():
            # Consume the renamed file to EOF, discard any incomplete old tail,
            # then begin at byte zero of the replacement.
            self._consume()
            self.close()
            if self._open():
                self._consume()
        return list(self.rows)
