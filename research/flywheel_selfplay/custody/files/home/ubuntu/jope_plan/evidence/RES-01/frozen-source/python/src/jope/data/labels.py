from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from jope.schema import TrainingRow


def assign_ranks(rows: list[TrainingRow]) -> list[TrainingRow]:
    grouped: dict[str, list[TrainingRow]] = defaultdict(list)
    for row in rows:
        grouped[row.dataset_id].append(row)
    for group in grouped.values():
        group.sort(key=lambda r: (r.score is None, -(r.score or float("-inf"))))
        for i, row in enumerate(group, start=1):
            row.rank = i
    return rows


def assign_elo(rows: list[TrainingRow], base: float = 1000.0, spread: float = 32.0) -> list[TrainingRow]:
    assign_ranks(rows)
    grouped: dict[str, list[TrainingRow]] = defaultdict(list)
    for row in rows:
        grouped[row.dataset_id].append(row)
    for group in grouped.values():
        n = max(1, len(group) - 1)
        for row in group:
            rank = row.rank or len(group)
            row.elo = base + spread * (len(group) - rank) / n
    return rows


def pairwise_labels(rows: list[TrainingRow]) -> list[dict]:
    assign_ranks(rows)
    grouped: dict[str, list[TrainingRow]] = defaultdict(list)
    for idx, row in enumerate(rows):
        row.metadata.setdefault("_row_index", idx)
        grouped[row.dataset_id].append(row)
    pairs = []
    for dataset_id, group in grouped.items():
        for a, b in combinations(group, 2):
            if a.score is None or b.score is None or a.score == b.score:
                continue
            winner, loser = (a, b) if a.score > b.score else (b, a)
            pairs.append(
                {
                    "dataset_id": dataset_id,
                    "winner": winner.metadata["_row_index"],
                    "loser": loser.metadata["_row_index"],
                    "score_delta": abs((winner.score or 0.0) - (loser.score or 0.0)),
                }
            )
    return pairs
