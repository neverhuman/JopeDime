"""Immutable schemas for the broad-shadow campaign.

All validators are deliberately exact-field and content-hash based.  The
production flag applies campaign-size gates; fixture-sized artifacts still use
the same scientific and leakage boundaries.
"""

from .contracts_common import (
    ANCHOR_COUNT, BROAD_CATALOG_SCHEMA, BROAD_CORPUS_SCHEMA, BROAD_RECEIPT_SCHEMA,
    DATASET_EMBEDDING_DIMENSIONS, FIDELITIES, PILOT_DATASET_COUNT,
    PILOT_GENOMES_PER_DATASET, PILOT_RECEIPT_COUNT, PROBE_SLATE_SCHEMA, REAL_SPLITS,
    REAL_SPLIT_COUNTS, SLOT_ROLE_COUNTS, SYNTHETIC_SPLITS, TARGET_SUMMARY_DIMENSIONS,
    _DATASET, _SHA256, _SLUG, _content_hash, _dataset, _exact_fields, _finite,
    _optional_finite, _positive_integer, _safe_relative_path, _sha256, _slug,
    canonical_json_bytes, content_sha256, robust_score,
)
from .contracts_catalog import (
    _CATALOG_FIELDS, _CATALOG_ROW_FIELDS, _ENCODING_FIELDS, _validate_encoding_ref,
    validate_broad_catalog,
)
from .contracts_slate import (
    _ANCHOR_FIELDS, _PROTOCOL_FIELDS, _SLATE_DATASET_FIELDS, _SLATE_FIELDS, _SLOT_FIELDS,
    _validate_protocol, validate_probe_slate,
)
from .contracts_receipts import (
    _FILE_FIELDS, _RECEIPT_FIELDS, _SNAPSHOT_FIELDS, receipt_dedup_key,
    validate_broad_corpus_snapshot, validate_broad_receipt,
)
from .contracts_ranking import (
    build_hashed, group_receipts_for_ranking, normalize_score_within_dataset,
)
