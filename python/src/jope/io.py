from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from .schema import TrainingRow


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict | TrainingRow]) -> int:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            data = row.to_json() if isinstance(row, TrainingRow) else row
            handle.write(json.dumps(data, sort_keys=True) + "\n")
            n += 1
    return n

