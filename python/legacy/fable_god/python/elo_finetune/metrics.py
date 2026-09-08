"""JSONL metrics writer implementing docs/metrics_schema.md.

One os.write per line on an O_APPEND fd (atomic under 4KB, so a tailing
reader never sees a partial line). Rank 0 only.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path


def new_run_id() -> str:
    return time.strftime("r%Y%m%d") + "-" + secrets.token_hex(3)


class JsonlWriter:
    def __init__(self, path: Path, run_id: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        self.run_id = run_id
        self.seq = 0

    def write(self, kind: str, step: int, fields: dict) -> None:
        record = {
            "run_id": self.run_id,
            "seq": self.seq,
            "t": time.time(),
            "step": step,
            "kind": kind,
            **fields,
        }
        os.write(self.fd, (json.dumps(record) + "\n").encode())
        self.seq += 1

    def event(self, step: int, msg: str, level: str = "info") -> None:
        self.write("event", step, {"msg": msg, "level": level})

    def close(self) -> None:
        os.close(self.fd)


def write_anchors(
    path: Path,
    run_id: str,
    base_elo: dict[str, float],
    base_rank: dict[str, int],
    leaderboard_val: list[dict],
    splits: dict[str, int],
) -> None:
    payload = {
        "run_id": run_id,
        "t": time.time(),
        "algorithm": "tabpfn_v3_regressor",
        "base_elo": base_elo,
        "base_rank": base_rank,
        "leaderboard": {"val": leaderboard_val},
        "splits": splits,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)
