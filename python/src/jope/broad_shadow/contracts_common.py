"""Shared constants and exact-field validation primitives for broad-shadow contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

BROAD_CATALOG_SCHEMA = "BroadDatasetCatalog.v1"
PROBE_SLATE_SCHEMA = "ProbeSlate.v1"
BROAD_RECEIPT_SCHEMA = "BroadEvaluationReceipt.v1"
BROAD_CORPUS_SCHEMA = "BroadCorpusSnapshot.v1"

DATASET_EMBEDDING_DIMENSIONS = 1536
TARGET_SUMMARY_DIMENSIONS = 512
ANCHOR_COUNT = 4
PILOT_DATASET_COUNT = 2_000
PILOT_GENOMES_PER_DATASET = 10
PILOT_RECEIPT_COUNT = PILOT_DATASET_COUNT * PILOT_GENOMES_PER_DATASET

REAL_SPLIT_COUNTS = {
    "train": 2_090,
    "cold_development": 119,
    "prospective_gate": 64,
    "double_holdout": 49,
}
SYNTHETIC_SPLITS = frozenset({"train", "validation", "test"})
REAL_SPLITS = frozenset(REAL_SPLIT_COUNTS)
FIDELITIES = frozenset({"screening", "exact"})
SLOT_ROLE_COUNTS = {
    "anchor": 4,
    "transfer": 2,
    "policy": 2,
    "diverse": 2,
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DATASET = re.compile(r"^(?:[0-9a-f]{16}|[0-9a-f]{64})$")
_SLUG = re.compile(r"^[a-z0-9]+(?:[-_.][a-z0-9]+)*$")


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _exact_fields(value: Any, expected: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{name} has unknown or missing fields")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _dataset(value: Any, name: str = "dataset_hash") -> str:
    if not isinstance(value, str) or not _DATASET.fullmatch(value):
        raise ValueError(f"{name} must be a 16- or 64-character lowercase dataset hash")
    return value


def _slug(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SLUG.fullmatch(value):
        raise ValueError(f"{name} must be a normalized lowercase slug")
    return value


def _positive_integer(value: Any, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ValueError(f"{name} must be finite")
    return number


def _optional_finite(value: Any, name: str, *, minimum: float | None = None) -> float | None:
    return None if value is None else _finite(value, name, minimum=minimum)


def _content_hash(payload: Mapping[str, Any], field: str, name: str) -> None:
    claimed = _sha256(payload[field], field)
    unhashed = dict(payload)
    unhashed.pop(field)
    if claimed != content_sha256(unhashed):
        raise ValueError(f"{name} content hash differs")


def _safe_relative_path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} must be a normal relative path")
    return value


def robust_score(oof_r2: float, fold_r2: Sequence[float]) -> float:
    """Canonical robust label: clipped OOF R² minus clipped population SD."""

    oof = _finite(oof_r2, "oof_r2")
    folds = [_finite(value, "fold_r2") for value in fold_r2]
    if not folds:
        raise ValueError("fold_r2 must be nonempty")
    population_std = statistics.pstdev(folds) if len(folds) > 1 else 0.0
    return max(oof, -1.0) - 0.25 * min(population_std, 1.0)
