from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np

from test_exact_common import atomic_json, prime_source_root, sha256_file

try:
    import orjson

    _json_loads = orjson.loads
except ImportError:  # pragma: no cover - stdlib is the portable fallback
    _json_loads = json.loads

PACK_SCHEMA = "PackedCorpus.v3"
LEGACY_PACK_SCHEMA = "PackedCorpus.v2"
PACK_LAYOUT = "content-addressed-label-shards-plus-mmap-columns.v1"
LABEL_SHARD_ROWS = 65_536
# Wire-frozen and append-only (D3): these codes are baked into every packed
# corpus's uint8 `tasks` column, so existing rows must repack byte-identically.
# Never renumber or remove a code; new tasks only ever append the next integer.
TASK_CODE = {"regression": 0, "binary": 1, "timeseries": 2, "multiclass": 3}
SPLIT_CODE = {"train": 0, "validation": 1}


def _jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield _json_loads(line)


def _default_tokenizer(row: dict) -> list[int]:
    prime = prime_source_root()
    if str(prime) not in sys.path:
        sys.path.insert(0, str(prime))
    from jope.policy.model_v3 import encode_row_artifacts
    from jope.policy.tokens import build_prime_action_vocab
    from jope.schema import TrainingRow

    vocab = getattr(_default_tokenizer, "_vocab", None)
    if vocab is None:
        vocab = build_prime_action_vocab()
        _default_tokenizer._vocab = vocab
    tokens, _ = encode_row_artifacts(TrainingRow.from_json(row), vocab)
    return tokens


def _open_array(path: Path, dtype: str, count: int) -> np.memmap:
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(count,))


def _packed_records(manifest: dict, root: Path) -> Iterator[dict]:
    if manifest["schema"] == LEGACY_PACK_SCHEMA:
        yield from _jsonl(root / "observations.jsonl")
        return
    for shard in manifest["label_shards"]:
        yield from _jsonl(root / shard["path"])


class _LabelShardWriter:
    def __init__(
        self,
        temporary: Path,
        shared_root: Path,
        *,
        rows_per_shard: int = LABEL_SHARD_ROWS,
    ):
        self.temporary = temporary
        self.shared_root = shared_root
        self.rows_per_shard = rows_per_shard
        self.shared_root.mkdir(parents=True, exist_ok=True)
        self.handle = None
        self.building = None
        self.rows = 0
        self.first_row = 0
        self.shards: list[dict] = []

    def write(self, record: dict, ordinal: int) -> None:
        if self.handle is None:
            self.first_row = ordinal + 1
            self.building = self.temporary / f".label-shard-{ordinal:012d}.tmp"
            self.handle = self.building.open("wb")
        self.handle.write(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
            + b"\n"
        )
        self.rows += 1
        if self.rows == self.rows_per_shard:
            self._finish(ordinal + 1)

    def close(self, row_count: int) -> list[dict]:
        if self.handle is not None:
            self._finish(row_count)
        return self.shards

    def _finish(self, last_row: int) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        digest = sha256_file(self.building)
        shared = self.shared_root / f"{digest}.jsonl"
        reused = shared.is_file()
        if reused:
            if sha256_file(shared) != digest:
                raise ValueError("shared label shard content drift")
            self.building.unlink()
        else:
            os.replace(self.building, shared)
        name = f"label-shard-{digest}.jsonl"
        os.link(shared, self.temporary / name)
        self.shards.append(
            {
                "first_receipt": self.first_row,
                "last_receipt": last_row,
                "rows": self.rows,
                "path": name,
                "sha256": digest,
                "reused": reused,
                "reused_from_corpus_sha256": None,
            }
        )
        self.handle = None
        self.building = None
        self.rows = 0


def _write_causal_pair_index(path: Path, members: list[dict]) -> dict:
    grouped: dict[str, dict[str, list[dict]]] = {}
    for member in members:
        grouped.setdefault(member["pair_id"], {}).setdefault(
            member["lane"], []
        ).append(member)
    complete = []
    exclusions: Counter[str] = Counter()
    for pair_id, lanes in sorted(grouped.items()):
        if set(lanes) != {"native_dl_direct", "matched_gp_comparator"} or any(
            len(rows) != 1 for rows in lanes.values()
        ):
            exclusions["partial_or_ambiguous_selected_pair"] += 1
            continue
        direct = lanes["native_dl_direct"][0]
        comparator = lanes["matched_gp_comparator"][0]
        identity_fields = (
            "task",
            "dataset_hash",
            "accepted_parent_checkpoint_sha256",
            "wave",
            "pair_index",
        )
        if (
            any(direct[field] != comparator[field] for field in identity_fields)
            or direct["split"] != "train"
            or comparator["split"] != "train"
            or direct["ordinal"] == comparator["ordinal"]
        ):
            exclusions["identity_or_holdout_mismatch"] += 1
            continue
        complete.append(
            {
                "pair_id": pair_id,
                "task": direct["task"],
                "dataset_hash": direct["dataset_hash"],
                "accepted_parent_checkpoint_sha256": direct[
                    "accepted_parent_checkpoint_sha256"
                ],
                "wave": direct["wave"],
                "pair_index": direct["pair_index"],
                "direct_ordinal": direct["ordinal"],
                "comparator_ordinal": comparator["ordinal"],
                "direct_semantic_sha256": direct["semantic_sha256"],
                "comparator_semantic_sha256": comparator["semantic_sha256"],
                "direct_score": direct["score"],
                "comparator_score": comparator["score"],
                "score_gap": direct["score"] - comparator["score"],
            }
        )
    latest_wave = Counter()
    for pair in complete:
        latest_wave[pair["task"]] = max(
            latest_wave[pair["task"]], int(pair["wave"])
        )
    task_counts: Counter[str] = Counter()
    with path.open("w", encoding="utf-8") as output:
        for pair in complete:
            pair["hard"] = abs(float(pair["score_gap"])) <= 0.05 + 1e-12
            pair["novel"] = int(pair["wave"]) >= latest_wave[pair["task"]] - 2
            output.write(json.dumps(pair, sort_keys=True) + "\n")
            task_counts[pair["task"]] += 1
    return {
        "complete_pairs": len(complete),
        "task_counts": dict(task_counts),
        "exclusions": dict(exclusions),
        "fixed_anchor_holdout": "validation rows are excluded from the pair index",
    }


def _write_recipe_e_evidence_index(path: Path, members: list[dict]) -> dict:
    """Write the small, pinned sidecar used to admit descendant-credit training."""

    task_counts: dict[str, Counter[str]] = {}
    descendant_receipts: dict[str, set[str]] = {}
    with path.open("w", encoding="utf-8") as output:
        for member in members:
            task = member["task"]
            lane = member["lane"]
            task_counts.setdefault(task, Counter())[lane] += 1
            if lane == "completed_descendant_trajectories":
                descendant_receipts.setdefault(task, set()).add(
                    member["descendant_credit_receipt_sha256"]
                )
            output.write(json.dumps(member, sort_keys=True) + "\n")
    return {
        "rows": len(members),
        "task_counts": {
            task: {
                **dict(counts),
                "completed_descendant_receipts": len(
                    descendant_receipts.get(task, set())
                ),
            }
            for task, counts in sorted(task_counts.items())
        },
        "censored_descendants_excluded": True,
    }
