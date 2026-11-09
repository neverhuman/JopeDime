"""Immutable schemas for the broad-shadow campaign.

All validators are deliberately exact-field and content-hash based.  The
production flag applies campaign-size gates; fixture-sized artifacts still use
the same scientific and leakage boundaries.
"""

from .contracts_catalog import (
    _CATALOG_FIELDS as _CATALOG_FIELDS,
)
from .contracts_catalog import (
    _CATALOG_ROW_FIELDS as _CATALOG_ROW_FIELDS,
)
from .contracts_catalog import (
    _ENCODING_FIELDS as _ENCODING_FIELDS,
)
from .contracts_catalog import (
    _validate_encoding_ref as _validate_encoding_ref,
)
from .contracts_catalog import (
    validate_broad_catalog as validate_broad_catalog,
)
from .contracts_common import (
    _DATASET as _DATASET,
)
from .contracts_common import (
    _SHA256 as _SHA256,
)
from .contracts_common import (
    _SLUG as _SLUG,
)
from .contracts_common import (
    ANCHOR_COUNT as ANCHOR_COUNT,
)
from .contracts_common import (
    BROAD_CATALOG_SCHEMA as BROAD_CATALOG_SCHEMA,
)
from .contracts_common import (
    BROAD_CORPUS_SCHEMA as BROAD_CORPUS_SCHEMA,
)
from .contracts_common import (
    BROAD_RECEIPT_SCHEMA as BROAD_RECEIPT_SCHEMA,
)
from .contracts_common import (
    DATASET_EMBEDDING_DIMENSIONS as DATASET_EMBEDDING_DIMENSIONS,
)
from .contracts_common import (
    FIDELITIES as FIDELITIES,
)
from .contracts_common import (
    PILOT_DATASET_COUNT as PILOT_DATASET_COUNT,
)
from .contracts_common import (
    PILOT_GENOMES_PER_DATASET as PILOT_GENOMES_PER_DATASET,
)
from .contracts_common import (
    PILOT_RECEIPT_COUNT as PILOT_RECEIPT_COUNT,
)
from .contracts_common import (
    PROBE_SLATE_SCHEMA as PROBE_SLATE_SCHEMA,
)
from .contracts_common import (
    REAL_SPLIT_COUNTS as REAL_SPLIT_COUNTS,
)
from .contracts_common import (
    REAL_SPLITS as REAL_SPLITS,
)
from .contracts_common import (
    SLOT_ROLE_COUNTS as SLOT_ROLE_COUNTS,
)
from .contracts_common import (
    SYNTHETIC_SPLITS as SYNTHETIC_SPLITS,
)
from .contracts_common import (
    TARGET_SUMMARY_DIMENSIONS as TARGET_SUMMARY_DIMENSIONS,
)
from .contracts_common import (
    _content_hash as _content_hash,
)
from .contracts_common import (
    _dataset as _dataset,
)
from .contracts_common import (
    _exact_fields as _exact_fields,
)
from .contracts_common import (
    _finite as _finite,
)
from .contracts_common import (
    _optional_finite as _optional_finite,
)
from .contracts_common import (
    _positive_integer as _positive_integer,
)
from .contracts_common import (
    _safe_relative_path as _safe_relative_path,
)
from .contracts_common import (
    _sha256 as _sha256,
)
from .contracts_common import (
    _slug as _slug,
)
from .contracts_common import (
    canonical_json_bytes as canonical_json_bytes,
)
from .contracts_common import (
    content_sha256 as content_sha256,
)
from .contracts_common import (
    robust_score as robust_score,
)
from .contracts_ranking import (
    build_hashed as build_hashed,
)
from .contracts_ranking import (
    group_receipts_for_ranking as group_receipts_for_ranking,
)
from .contracts_ranking import (
    normalize_score_within_dataset as normalize_score_within_dataset,
)
from .contracts_receipts import (
    _FILE_FIELDS as _FILE_FIELDS,
)
from .contracts_receipts import (
    _RECEIPT_FIELDS as _RECEIPT_FIELDS,
)
from .contracts_receipts import (
    _SNAPSHOT_FIELDS as _SNAPSHOT_FIELDS,
)
from .contracts_receipts import (
    receipt_dedup_key as receipt_dedup_key,
)
from .contracts_receipts import (
    validate_broad_corpus_snapshot as validate_broad_corpus_snapshot,
)
from .contracts_receipts import (
    validate_broad_receipt as validate_broad_receipt,
)
from .contracts_slate import (
    _ANCHOR_FIELDS as _ANCHOR_FIELDS,
)
from .contracts_slate import (
    _PROTOCOL_FIELDS as _PROTOCOL_FIELDS,
)
from .contracts_slate import (
    _SLATE_DATASET_FIELDS as _SLATE_DATASET_FIELDS,
)
from .contracts_slate import (
    _SLATE_FIELDS as _SLATE_FIELDS,
)
from .contracts_slate import (
    _SLOT_FIELDS as _SLOT_FIELDS,
)
from .contracts_slate import (
    _validate_protocol as _validate_protocol,
)
from .contracts_slate import (
    validate_probe_slate as validate_probe_slate,
)
