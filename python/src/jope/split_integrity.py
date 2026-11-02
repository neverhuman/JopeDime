"""Frozen dataset split authority for Prime policy training.

The manifest deliberately depends on an explicit ``source_family`` supplied by the
signature inventory.  Dataset hashes are not a safe proxy for lineage families.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SPLIT_MANIFEST_V1 = "PrimeDatasetSplitManifest.v1"
TRAIN_SPLIT = "train"
DEVELOPMENT_SPLIT = "development"
TEMPORAL_AUDIT_SPLIT = "temporal_audit"
REPORT_ONLY_SPLITS = frozenset({"arena", "sealed", "double_holdout", TEMPORAL_AUDIT_SPLIT})
_ASSIGNABLE_SPLITS = frozenset({TRAIN_SPLIT, DEVELOPMENT_SPLIT, TEMPORAL_AUDIT_SPLIT})


class SplitIntegrityError(ValueError):
    """A split cannot safely be created or reused."""


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_families_from_signature_manifest(path: str | Path) -> dict[str, str]:
    """Read only explicit source-family identities; never derive or guess them."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SplitIntegrityError("signature manifest must be a JSON object")
    datasets = raw.get("datasets", raw)
    if not isinstance(datasets, dict):
        raise SplitIntegrityError("signature manifest datasets must be an object")
    families: dict[str, str] = {}
    for dataset, record in datasets.items():
        if not isinstance(dataset, str) or not dataset:
            raise SplitIntegrityError("signature manifest dataset identity is empty")
        if not isinstance(record, dict):
            continue
        family = record.get("source_family")
        if isinstance(family, str) and family.strip():
            families[dataset] = family.strip()
    return families


from .split_integrity_manifest import (
    _authority_split,
    _committed_at_ms,
    _dataset,
    _label_body,
    _manifest_payload,
    build_split_manifest,
    is_training_authority_label,
    validate_split_manifest,
)
from .split_integrity_storage import _atomic_json, load_or_create_split_manifest
from .split_integrity_partition import assignment_for_dataset, partition_training_rows
