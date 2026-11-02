#!/usr/bin/env python3
"""Compile-economy gates and dataset-fitted artifact identities."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def compiled_cache_key(
    *,
    task: str,
    train_sha256: str,
    semantic_sha256: str,
    evaluator_protocol_sha256: str,
    compiler_bundle_sha256: str,
    target_triple: str,
) -> str:
    material = {
        "task": task,
        "dataset_train_sha256": train_sha256,
        "semantic_sha256": semantic_sha256,
        "evaluator_protocol_sha256": evaluator_protocol_sha256,
        "compiler_bundle_sha256": compiler_bundle_sha256,
        "target_triple": target_triple,
    }
    for field in (
        "dataset_train_sha256",
        "semantic_sha256",
        "evaluator_protocol_sha256",
        "compiler_bundle_sha256",
    ):
        value = material[field]
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"invalid compile-cache identity: {field}")
    if task not in {"regression", "binary"} or not target_triple:
        raise ValueError("invalid compile-cache task or target")
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def family_diverse_panel(
    datasets: Iterable[Mapping[str, object]],
    *,
    size: int = 32,
    seed: int = 20260726,
) -> list[dict]:
    by_family: dict[str, list[dict]] = defaultdict(list)
    for raw in datasets:
        row = dict(raw)
        by_family[str(row["source_family"])].append(row)
    ordered_families = sorted(
        by_family,
        key=lambda family: hashlib.sha256(f"{seed}|{family}".encode()).digest(),
    )
    selected = []
    for family in ordered_families:
        values = sorted(
            by_family[family],
            key=lambda row: str(row["dataset_hash"]),
        )
        selected.append(values[0])
        if len(selected) == size:
            break
    if len(selected) < size:
        remainder = [
            row
            for family in ordered_families
            for row in sorted(
                by_family[family][1:], key=lambda value: str(value["dataset_hash"])
            )
        ]
        selected.extend(remainder[: size - len(selected)])
    if len(selected) != size:
        raise ValueError(f"replay panel requires {size} datasets")
    return selected


def compiled_finalists(
    rows: Iterable[Mapping[str, object]],
    incumbents: Mapping[tuple[str, str], float],
) -> list[dict]:
    """Compile strict incumbent beaters or top-two rows, never over two/dataset."""

    by_dataset: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        if row.get("valid") is True and math.isfinite(float(row.get("score", float("nan")))):
            by_dataset[(str(row["task"]), str(row["dataset_hash"]))].append(row)
    result = []
    for identity in sorted(by_dataset):
        ranked = sorted(
            by_dataset[identity],
            key=lambda row: (-float(row["score"]), str(row["semantic_sha256"])),
        )
        incumbent = incumbents.get(identity)
        eligible = [
            row
            for index, row in enumerate(ranked)
            if index < 2
            or (incumbent is not None and float(row["score"]) > incumbent)
        ]
        result.extend(eligible[:2])
    return result


def exact_parity(
    interpreted: Sequence[float],
    compiled: Sequence[float],
    *,
    interpreted_score: float,
    compiled_score: float,
) -> dict:
    predictions_equal = len(interpreted) == len(compiled) and all(
        struct.pack("<d", float(left)) == struct.pack("<d", float(right))
        for left, right in zip(interpreted, compiled, strict=True)
    )
    score_equal = struct.pack("<d", interpreted_score) == struct.pack(
        "<d", compiled_score
    )
    return {
        "prediction_bit_parity": predictions_equal,
        "score_bit_parity": score_equal,
        "qualified": predictions_equal and score_equal,
    }


def cache_status(cache_root: Path) -> dict:
    manifests = list(cache_root.rglob("artifact-manifest.json")) if cache_root.is_dir() else []
    valid = 0
    for manifest in manifests:
        try:
            value = json.loads(manifest.read_text())
            artifact = manifest.parent / value["artifact"]
            if artifact.is_file():
                valid += 1
        except (OSError, KeyError, json.JSONDecodeError):
            continue
    operations = {"hits": 0, "misses": 0}
    operation_path = cache_root / "operations.json"
    if operation_path.is_file():
        try:
            value = json.loads(operation_path.read_text())
            operations = {
                "hits": int(value.get("hits") or 0),
                "misses": int(value.get("misses") or 0),
            }
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    requests = operations["hits"] + operations["misses"]
    return {
        "entries": len(manifests),
        "valid_entries": valid,
        "invalid_entries": len(manifests) - valid,
        **operations,
        "hit_rate": None if requests == 0 else operations["hits"] / requests,
    }
