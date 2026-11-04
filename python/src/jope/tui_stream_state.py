"""Data models and key input state for the gene stream TUI."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

INITIAL_TAIL_BYTES = 2 * 1024 * 1024


class JsonlTailer:
    """Yield only rows appended since the last poll; reset on rotation (inode/size)."""

    def __init__(self, path: str | Path, *, initial_tail_bytes: int | None = INITIAL_TAIL_BYTES) -> None:
        self.path = Path(path)
        self._offset = 0
        self._inode: int | None = None
        self._buf = ""
        self._initial_tail_bytes = initial_tail_bytes

    def poll(self) -> list[dict[str, Any]]:
        try:
            st = self.path.stat()
        except OSError:
            return []
        # rotation / truncation → reopen from the start
        reset = self._inode is not None and (st.st_ino != self._inode or st.st_size < self._offset)
        if reset:
            self._offset = 0
            self._buf = ""
        if (self._inode is None or reset) and self._initial_tail_bytes is not None:
            self._offset = max(0, st.st_size - max(0, self._initial_tail_bytes))
        self._inode = st.st_ino
        if st.st_size <= self._offset:
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            handle.seek(self._offset)
            chunk = handle.read()
            self._offset = handle.tell()
        self._buf += chunk
        rows: list[dict[str, Any]] = []
        *complete, self._buf = self._buf.split("\n")
        for line in complete:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows


@dataclass
class Gene:
    gene_id: str
    engine: str = ""
    canonical: str = ""
    head: str | None = None
    mu: float | None = None
    sigma: float | None = None
    p_bad: float | None = None
    oof: float | None = None
    test_r2: float | None = None
    state: str = "proposed"  # proposed -> selected -> scored
    total_ms: float | None = None
    seq: int = 0


@dataclass
class StreamState:
    cursor: int = 0
    lang: str = "python"  # or "rust"
    follow: bool = True
    quit: bool = False
    save: bool = False
    order: list[str] = field(default_factory=list)  # gene_ids in arrival order


def handle_key(state: StreamState, key: str, n_genes: int) -> StreamState:
    """Pure key dispatch (no I/O) so the input model is testable without a tty."""
    s = replace(state, save=False)
    if key in ("q", "\x03"):  # q or Ctrl-C
        s.quit = True
    elif key in ("j", "DOWN"):
        s.follow = False
        s.cursor = min(max(0, n_genes - 1), s.cursor + 1)
    elif key in ("k", "UP"):
        s.follow = False
        s.cursor = max(0, s.cursor - 1)
    elif key == "p":
        s.lang = "python"
    elif key == "r":
        s.lang = "rust"
    elif key == "l":
        s.follow = not s.follow
    elif key == "s":
        s.save = True
    return s
