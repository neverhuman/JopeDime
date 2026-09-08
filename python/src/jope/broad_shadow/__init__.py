"""Broad performance-map and gated self-play authority.

The package is additive: legacy Dime and ratchet-v1 artifacts remain readable,
but broad-shadow artifacts must pass the strict v1/v2 contracts exported here.
"""

from .contracts import (
    BROAD_CATALOG_SCHEMA,
    BROAD_CORPUS_SCHEMA,
    BROAD_RECEIPT_SCHEMA,
    PROBE_SLATE_SCHEMA,
    canonical_json_bytes,
    content_sha256,
    robust_score,
    validate_broad_catalog,
    validate_broad_corpus_snapshot,
    validate_broad_receipt,
    validate_probe_slate,
)

__all__ = [
    "BROAD_CATALOG_SCHEMA",
    "BROAD_CORPUS_SCHEMA",
    "BROAD_RECEIPT_SCHEMA",
    "PROBE_SLATE_SCHEMA",
    "canonical_json_bytes",
    "content_sha256",
    "robust_score",
    "validate_broad_catalog",
    "validate_broad_corpus_snapshot",
    "validate_broad_receipt",
    "validate_probe_slate",
]
