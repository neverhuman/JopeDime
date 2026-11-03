"""Canonical score normalization and receipt ranking helpers."""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .contracts_common import _finite, content_sha256
from .contracts_receipts import validate_broad_receipt


def build_hashed(schema: str, hash_field: str, **fields: Any) -> dict[str, Any]:
    """Build the canonical self-hashed envelope shared by all broad schemas."""

    payload = {"schema": schema, **fields}
    payload[hash_field] = content_sha256(payload)
    return payload


def normalize_score_within_dataset(values: Sequence[float]) -> list[float]:
    """Median/MAD normalization used by anchor signatures and score targets."""

    numbers = [_finite(value, "score") for value in values]
    if not numbers:
        return []
    median = statistics.median(numbers)
    mad = statistics.median(abs(value - median) for value in numbers)
    scale = max(1.4826 * mad, 1e-6)
    return [(value - median) / scale for value in numbers]


def group_receipts_for_ranking(
    receipts: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], list[Mapping[str, Any]]]:
    """Group rankable labels without crossing dataset or evaluator protocol."""

    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for receipt in receipts:
        validate_broad_receipt(receipt)
        if not receipt["valid"]:
            continue
        protocol_hash = content_sha256(receipt["protocol"])
        grouped[(receipt["dataset_hash"], receipt["fidelity"], protocol_hash)].append(receipt)
    return dict(grouped)
