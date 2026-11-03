"""Broad catalog schema and validator."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .contracts_common import (
    BROAD_CATALOG_SCHEMA,
    DATASET_EMBEDDING_DIMENSIONS,
    REAL_SPLIT_COUNTS,
    REAL_SPLITS,
    SYNTHETIC_SPLITS,
    TARGET_SUMMARY_DIMENSIONS,
    _content_hash,
    _dataset,
    _exact_fields,
    _positive_integer,
    _safe_relative_path,
    _sha256,
    _slug,
)

_ENCODING_FIELDS = {"sha256", "path", "offset_bytes", "dimensions", "dtype"}
_CATALOG_ROW_FIELDS = {
    "dataset_hash",
    "train_sha256",
    "domain",
    "source_family",
    "source_family_authority_sha256",
    "generator_lineage",
    "split",
    "rows",
    "features",
    "target_shape",
    "dataset_encoding",
    "target_encoding",
    "foundation_complete",
    "hyperion_complete",
}
_CATALOG_FIELDS = {
    "schema",
    "catalog_sha256",
    "created_at_ms",
    "dataset_embedding_dimensions",
    "target_encoding_dimensions",
    "foundation_cache_count",
    "hyperion_complete_count",
    "real_split_counts",
    "synthetic_split_counts",
    "synthetic_family_split_counts",
    "datasets",
}


def _validate_encoding_ref(value: Any, name: str, expected_dimensions: int) -> None:
    row = _exact_fields(value, _ENCODING_FIELDS, name)
    _sha256(row["sha256"], f"{name}.sha256")
    _safe_relative_path(row["path"], f"{name}.path")
    offset = _positive_integer(row["offset_bytes"], f"{name}.offset_bytes", allow_zero=True)
    if offset % 4:
        raise ValueError(f"{name}.offset_bytes must be float32 aligned")
    if row["dimensions"] != expected_dimensions or row["dtype"] != "float32_le":
        raise ValueError(f"{name} dimensions or dtype differs from frozen authority")


def validate_broad_catalog(payload: Any, *, production: bool = False) -> None:
    manifest = _exact_fields(payload, _CATALOG_FIELDS, "broad dataset catalog")
    if manifest["schema"] != BROAD_CATALOG_SCHEMA:
        raise ValueError("unsupported broad dataset catalog schema")
    _content_hash(manifest, "catalog_sha256", "broad dataset catalog")
    _positive_integer(manifest["created_at_ms"], "created_at_ms")
    if (
        manifest["dataset_embedding_dimensions"] != DATASET_EMBEDDING_DIMENSIONS
        or manifest["target_encoding_dimensions"] != TARGET_SUMMARY_DIMENSIONS
    ):
        raise ValueError("broad catalog requires frozen 1536+512 encodings")
    foundation_count = _positive_integer(
        manifest["foundation_cache_count"],
        "foundation_cache_count",
        allow_zero=True,
    )
    declared_hyperion = _positive_integer(
        manifest["hyperion_complete_count"],
        "hyperion_complete_count",
        allow_zero=True,
    )
    rows = manifest["datasets"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("broad catalog datasets must be a nonempty list")

    seen: set[str] = set()
    previous: str | None = None
    real_counts: Counter[str] = Counter()
    synthetic_counts: Counter[str] = Counter()
    family_split: dict[str, str] = {}
    hyperion_count = 0
    foundation_complete = 0
    for value in rows:
        row = _exact_fields(value, _CATALOG_ROW_FIELDS, "broad catalog row")
        dataset = _dataset(row["dataset_hash"])
        if previous is not None and previous >= dataset:
            raise ValueError("broad catalog datasets must be strictly sorted and unique")
        previous = dataset
        if dataset in seen:
            raise ValueError("broad catalog contains duplicate dataset identities")
        seen.add(dataset)
        _sha256(row["train_sha256"], "train_sha256")
        family = _slug(row["source_family"], "source_family")
        _sha256(
            row["source_family_authority_sha256"],
            "source_family_authority_sha256",
        )
        _positive_integer(row["rows"], "rows")
        _positive_integer(row["features"], "features")
        _slug(row["target_shape"], "target_shape")
        _validate_encoding_ref(
            row["dataset_encoding"],
            "dataset_encoding",
            DATASET_EMBEDDING_DIMENSIONS,
        )
        _validate_encoding_ref(
            row["target_encoding"],
            "target_encoding",
            TARGET_SUMMARY_DIMENSIONS,
        )
        if type(row["foundation_complete"]) is not bool or type(row["hyperion_complete"]) is not bool:
            raise ValueError("catalog completion flags must be booleans")
        foundation_complete += int(row["foundation_complete"])
        hyperion_count += int(row["hyperion_complete"])

        domain, split = row["domain"], row["split"]
        if domain == "real":
            if split not in REAL_SPLITS or row["generator_lineage"] is not None:
                raise ValueError("real datasets require a real split and no generator lineage")
            real_counts[split] += 1
        elif domain == "synthetic":
            if split not in SYNTHETIC_SPLITS:
                raise ValueError("synthetic dataset split is invalid")
            lineage = _slug(row["generator_lineage"], "generator_lineage")
            prior = family_split.setdefault(lineage, split)
            if prior != split:
                raise ValueError("normalized generator lineage crosses synthetic splits")
            synthetic_counts[split] += 1
        else:
            raise ValueError("dataset domain must be real or synthetic")
        if not family:
            raise AssertionError("validated source family unexpectedly empty")

    declared_real = manifest["real_split_counts"]
    declared_synthetic = manifest["synthetic_split_counts"]
    declared_families = manifest["synthetic_family_split_counts"]
    if (
        declared_real != {name: real_counts[name] for name in sorted(REAL_SPLITS)}
        or declared_synthetic != {name: synthetic_counts[name] for name in sorted(SYNTHETIC_SPLITS)}
        or declared_families
        != {
            name: len({family for family, assigned in family_split.items() if assigned == name})
            for name in sorted(SYNTHETIC_SPLITS)
        }
    ):
        raise ValueError("broad catalog declared counts differ from its rows")
    if declared_hyperion != hyperion_count:
        raise ValueError("broad catalog Hyperion count differs from its rows")
    if foundation_count < foundation_complete:
        raise ValueError("foundation cache count is smaller than bound catalog encodings")
    if production:
        if declared_real != REAL_SPLIT_COUNTS:
            raise ValueError("production real split must remain 2090/119/64/49")
        if set(synthetic_counts) != SYNTHETIC_SPLITS or set(family_split.values()) != set(SYNTHETIC_SPLITS):
            raise ValueError("production synthetic authority requires all three splits")
        if foundation_count < 27_875:
            raise ValueError("production catalog must bind the 27,875 foundation cache")
        if hyperion_count < 24_177:
            raise ValueError("pilot cannot start below the frozen Hyperion-ready floor")
