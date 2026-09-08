"""Explicit, content-bound source-family authority for signature datasets.

This module intentionally has no filename, path, or content heuristics for
source-family assignment.  A family is authoritative only when it comes from
the dataset's exact metadata or from a reviewed mapping bound to both dataset
and metadata hashes.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


DATASET_SOURCE_FAMILY_MANIFEST_V1 = "DatasetSourceFamilyManifest.v1"
_DATASET_ID = re.compile(r"^[0-9a-f]{16}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_FAMILY = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def canonical_json_bytes(payload: Any) -> bytes:
    """Return the compact canonical bytes shared with the Rust contracts."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_family(value: Any) -> str:
    if not isinstance(value, str) or not _SOURCE_FAMILY.fullmatch(value):
        raise ValueError("source_family must be a lowercase slug")
    return value


def _metadata_path(
    metadata_root: Path, dataset: str, signature_entry: Mapping[str, Any]
) -> Path:
    referenced = {
        (Path(value).parent / "meta.json").resolve()
        for name, value in signature_entry.items()
        if name
        in {
            "vs_path",
            "train_csv",
            "test_csv",
            "train_target_vs",
            "test_target_vs",
        }
        and isinstance(value, str)
        and value.strip()
    }
    if referenced:
        root = metadata_root.resolve()
        if (
            len(referenced) != 1
            or not next(iter(referenced)).is_relative_to(root)
            or next(iter(referenced)).parent.name != dataset
            or not next(iter(referenced)).is_file()
        ):
            raise ValueError(
                f"dataset {dataset} signature paths do not bind one metadata file"
            )
        return next(iter(referenced))

    candidates = [metadata_root / dataset / "meta.json"]
    candidates.extend(metadata_root.glob(f"*/{dataset}/meta.json"))
    matches = sorted({path.resolve() for path in candidates if path.is_file()})
    if len(matches) != 1:
        raise ValueError(
            f"dataset {dataset} requires exactly one metadata file under metadata_root"
        )
    return matches[0]


def validate_source_family_manifest(payload: Any) -> None:
    expected = {
        "schema",
        "signature_manifest_sha256",
        "status",
        "dataset_count",
        "resolved_dataset_count",
        "unresolved_dataset_count",
        "source_family_count",
        "datasets",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("source-family manifest has unknown or missing fields")
    if payload["schema"] != DATASET_SOURCE_FAMILY_MANIFEST_V1:
        raise ValueError("unsupported source-family manifest schema")
    _require_sha256(payload["signature_manifest_sha256"], "signature_manifest_sha256")
    rows = payload["datasets"]
    if not isinstance(rows, list):
        raise ValueError("source-family datasets must be a list")
    integer_fields = (
        "dataset_count",
        "resolved_dataset_count",
        "unresolved_dataset_count",
        "source_family_count",
    )
    if any(
        type(payload[name]) is not int or payload[name] < 0 for name in integer_fields
    ):
        raise ValueError("source-family manifest counts must be nonnegative integers")
    if payload["dataset_count"] != len(rows):
        raise ValueError("source-family dataset_count differs from its rows")

    previous: str | None = None
    resolved = 0
    unresolved = 0
    families: set[str] = set()
    row_fields = {
        "dataset",
        "dataset_sha256",
        "metadata_sha256",
        "source_family",
        "resolution",
        "review_receipt_sha256",
        "unresolved_reason",
    }
    for row in rows:
        if not isinstance(row, dict) or set(row) != row_fields:
            raise ValueError("source-family row has unknown or missing fields")
        dataset = row["dataset"]
        if not isinstance(dataset, str) or not _DATASET_ID.fullmatch(dataset):
            raise ValueError(
                "dataset identity must be 16 lowercase hexadecimal characters"
            )
        if previous is not None and previous >= dataset:
            raise ValueError(
                "source-family datasets must be strictly sorted and unique"
            )
        previous = dataset
        dataset_sha256 = _require_sha256(row["dataset_sha256"], "dataset_sha256")
        if not dataset_sha256.startswith(dataset):
            raise ValueError("dataset_sha256 must begin with the dataset identity")
        _require_sha256(row["metadata_sha256"], "metadata_sha256")

        resolution = row["resolution"]
        family = row["source_family"]
        receipt = row["review_receipt_sha256"]
        reason = row["unresolved_reason"]
        if resolution == "upstream_metadata":
            _require_family(family)
            if receipt is not None or reason is not None:
                raise ValueError("upstream source-family evidence is inconsistent")
            resolved += 1
            families.add(family)
        elif resolution == "reviewed_mapping":
            _require_family(family)
            _require_sha256(receipt, "review_receipt_sha256")
            if reason is not None:
                raise ValueError("reviewed source-family evidence is inconsistent")
            resolved += 1
            families.add(family)
        elif resolution == "unresolved":
            if (
                family is not None
                or receipt is not None
                or not isinstance(reason, str)
                or not reason.strip()
            ):
                raise ValueError("unresolved source-family evidence is inconsistent")
            unresolved += 1
        else:
            raise ValueError("unsupported source-family resolution")

    if (
        payload["resolved_dataset_count"] != resolved
        or payload["unresolved_dataset_count"] != unresolved
        or payload["source_family_count"] != len(families)
        or resolved + unresolved != len(rows)
    ):
        raise ValueError("source-family manifest counts differ from its rows")
    expected_status = "complete" if unresolved == 0 else "incomplete"
    if payload["status"] != expected_status:
        raise ValueError("source-family status differs from unresolved rows")


def build_source_family_manifest(
    signature_manifest_path: str | Path,
    metadata_root: str | Path,
    reviewed_mappings: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build authority without guessing any missing source family."""
    signature_path = Path(signature_manifest_path)
    signature_bytes = signature_path.read_bytes()
    signature = json.loads(signature_bytes)
    if not isinstance(signature, dict) or not isinstance(
        signature.get("datasets"), dict
    ):
        raise ValueError("signature manifest requires a datasets object")
    datasets = signature["datasets"]
    reviews = dict(reviewed_mappings or {})
    unknown_reviews = set(reviews) - set(datasets)
    if unknown_reviews:
        raise ValueError(
            f"reviewed mappings contain unknown datasets: {sorted(unknown_reviews)}"
        )

    rows: list[dict[str, Any]] = []
    metadata_root = Path(metadata_root)
    for dataset in sorted(datasets):
        if not _DATASET_ID.fullmatch(dataset):
            raise ValueError(f"invalid signature dataset identity: {dataset!r}")
        signature_entry = datasets[dataset]
        if not isinstance(signature_entry, Mapping):
            raise ValueError(f"signature dataset {dataset} must be an object")
        metadata_path = _metadata_path(metadata_root, dataset, signature_entry)
        metadata_bytes = metadata_path.read_bytes()
        metadata = json.loads(metadata_bytes)
        if not isinstance(metadata, dict) or metadata.get("dataset_hash") != dataset:
            raise ValueError(f"metadata identity differs for dataset {dataset}")
        dataset_sha256 = _require_sha256(metadata.get("sha256"), "metadata sha256")
        if not dataset_sha256.startswith(dataset):
            raise ValueError(f"metadata sha256 differs from dataset identity {dataset}")
        metadata_sha256 = _sha256_bytes(metadata_bytes)
        upstream = metadata.get("source_family")
        review = reviews.get(dataset)

        if upstream is not None:
            family = _require_family(upstream)
            if review is not None:
                raise ValueError(
                    f"dataset {dataset} already has upstream source_family"
                )
            row = {
                "dataset": dataset,
                "dataset_sha256": dataset_sha256,
                "metadata_sha256": metadata_sha256,
                "source_family": family,
                "resolution": "upstream_metadata",
                "review_receipt_sha256": None,
                "unresolved_reason": None,
            }
        elif review is not None:
            expected_review_fields = {
                "dataset_sha256",
                "metadata_sha256",
                "source_family",
                "review_receipt_sha256",
            }
            if not isinstance(review, Mapping) or set(review) != expected_review_fields:
                raise ValueError(
                    f"reviewed mapping has invalid fields for dataset {dataset}"
                )
            if (
                review["dataset_sha256"] != dataset_sha256
                or review["metadata_sha256"] != metadata_sha256
            ):
                raise ValueError(f"reviewed mapping is stale for dataset {dataset}")
            row = {
                "dataset": dataset,
                "dataset_sha256": dataset_sha256,
                "metadata_sha256": metadata_sha256,
                "source_family": _require_family(review["source_family"]),
                "resolution": "reviewed_mapping",
                "review_receipt_sha256": _require_sha256(
                    review["review_receipt_sha256"], "review_receipt_sha256"
                ),
                "unresolved_reason": None,
            }
        else:
            row = {
                "dataset": dataset,
                "dataset_sha256": dataset_sha256,
                "metadata_sha256": metadata_sha256,
                "source_family": None,
                "resolution": "unresolved",
                "review_receipt_sha256": None,
                "unresolved_reason": "missing_explicit_source_family",
            }
        rows.append(row)

    resolved = sum(row["source_family"] is not None for row in rows)
    unresolved = len(rows) - resolved
    manifest = {
        "schema": DATASET_SOURCE_FAMILY_MANIFEST_V1,
        "signature_manifest_sha256": _sha256_bytes(signature_bytes),
        "status": "complete" if unresolved == 0 else "incomplete",
        "dataset_count": len(rows),
        "resolved_dataset_count": resolved,
        "unresolved_dataset_count": unresolved,
        "source_family_count": len(
            {row["source_family"] for row in rows if row["source_family"] is not None}
        ),
        "datasets": rows,
    }
    validate_source_family_manifest(manifest)
    return manifest


def source_families_from_authority_manifest(payload: Any) -> dict[str, str]:
    """Return mappings only from complete authority; incomplete is unusable."""
    validate_source_family_manifest(payload)
    if payload["status"] != "complete":
        raise ValueError("incomplete source-family authority cannot be used")
    return {row["dataset"]: row["source_family"] for row in payload["datasets"]}
