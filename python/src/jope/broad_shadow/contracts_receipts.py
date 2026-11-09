"""Evaluation receipt and frozen corpus snapshot contracts."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any

from .contracts_common import (
    BROAD_CORPUS_SCHEMA,
    BROAD_RECEIPT_SCHEMA,
    FIDELITIES,
    PILOT_RECEIPT_COUNT,
    _content_hash,
    _dataset,
    _exact_fields,
    _finite,
    _optional_finite,
    _positive_integer,
    _safe_relative_path,
    _sha256,
    robust_score,
)
from .contracts_slate import _validate_protocol

_RECEIPT_FIELDS = {
    "schema",
    "receipt_sha256",
    "dataset_hash",
    "train_sha256",
    "semantic_sha256",
    "canonical_genome",
    "evaluator_release_sha256",
    "fold_plan_sha256",
    "vocabulary_sha256",
    "fidelity",
    "protocol",
    "valid",
    "failure",
    "fold_r2",
    "oof_r2",
    "robust_score",
    "runtime_seconds",
    "label_authority",
    "test_table_accessed",
    "selection_channel",
    "recorded_at_ms",
}


def receipt_dedup_key(payload: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(payload["dataset_hash"]),
        str(payload["semantic_sha256"]),
        str(payload["evaluator_release_sha256"]),
        str(payload["fold_plan_sha256"]),
    )


def validate_broad_receipt(payload: Any) -> None:
    receipt = _exact_fields(payload, _RECEIPT_FIELDS, "broad evaluation receipt")
    if receipt["schema"] != BROAD_RECEIPT_SCHEMA:
        raise ValueError("unsupported broad evaluation receipt schema")
    _content_hash(receipt, "receipt_sha256", "broad evaluation receipt")
    _dataset(receipt["dataset_hash"])
    _sha256(receipt["train_sha256"], "train_sha256")
    semantic = _sha256(receipt["semantic_sha256"], "semantic_sha256")
    genome = receipt["canonical_genome"]
    if not isinstance(genome, str) or not genome or hashlib.sha256(genome.encode()).hexdigest() != semantic:
        raise ValueError("receipt semantic hash does not bind canonical genome")
    _sha256(receipt["evaluator_release_sha256"], "evaluator_release_sha256")
    _sha256(receipt["fold_plan_sha256"], "fold_plan_sha256")
    _sha256(receipt["vocabulary_sha256"], "vocabulary_sha256")
    fidelity = receipt["fidelity"]
    if fidelity not in FIDELITIES:
        raise ValueError("receipt fidelity must be screening or exact")
    _validate_protocol(receipt["protocol"], fidelity)
    if receipt["label_authority"] != "train_only_oof_r2" or receipt["test_table_accessed"] is not False:
        raise ValueError("test-table or benchmark labels are forbidden")
    if not isinstance(receipt["selection_channel"], str) or not receipt["selection_channel"].strip():
        raise ValueError("selection_channel must be nonempty")
    _positive_integer(receipt["recorded_at_ms"], "recorded_at_ms")
    runtime = _optional_finite(receipt["runtime_seconds"], "runtime_seconds", minimum=0.0)
    if type(receipt["valid"]) is not bool:
        raise ValueError("receipt valid flag must be boolean")
    if receipt["valid"]:
        if receipt["failure"] is not None:
            raise ValueError("valid receipt cannot carry a failure")
        folds = receipt["fold_r2"]
        expected_folds = receipt["protocol"]["folds"]
        if not isinstance(folds, list) or len(folds) != expected_folds:
            raise ValueError("valid receipt fold vector differs from its protocol")
        fold_values = [_finite(value, "fold_r2") for value in folds]
        oof = _finite(receipt["oof_r2"], "oof_r2")
        robust = _finite(receipt["robust_score"], "robust_score")
        expected = robust_score(oof, fold_values)
        if not math.isclose(robust, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("receipt robust score is not canonical")
        if runtime is None:
            raise ValueError("valid receipt requires measured runtime")
    else:
        failure = receipt["failure"]
        if (
            not isinstance(failure, Mapping)
            or set(failure) != {"class", "message"}
            or not isinstance(failure["class"], str)
            or not failure["class"].strip()
            or not isinstance(failure["message"], str)
        ):
            raise ValueError("invalid receipt requires a typed failure")
        if (
            any(receipt[field] is not None for field in ("oof_r2", "robust_score"))
            or receipt["fold_r2"] != []
        ):
            raise ValueError("failure receipts cannot carry finite score labels")


_FILE_FIELDS = {"path", "sha256", "bytes", "dtype", "shape"}
_SNAPSHOT_FIELDS = {
    "schema",
    "snapshot_sha256",
    "catalog_sha256",
    "probe_slate_sha256",
    "vocabulary_sha256",
    "created_at_ms",
    "row_count",
    "valid_row_count",
    "dataset_count",
    "domain_counts",
    "fidelity_counts",
    "files",
}


def validate_broad_corpus_snapshot(payload: Any, *, production: bool = False) -> None:
    snapshot = _exact_fields(payload, _SNAPSHOT_FIELDS, "broad corpus snapshot")
    if snapshot["schema"] != BROAD_CORPUS_SCHEMA:
        raise ValueError("unsupported broad corpus snapshot schema")
    _content_hash(snapshot, "snapshot_sha256", "broad corpus snapshot")
    for field in ("catalog_sha256", "probe_slate_sha256", "vocabulary_sha256"):
        _sha256(snapshot[field], field)
    _positive_integer(snapshot["created_at_ms"], "created_at_ms")
    rows = _positive_integer(snapshot["row_count"], "row_count")
    valid = _positive_integer(snapshot["valid_row_count"], "valid_row_count", allow_zero=True)
    datasets = _positive_integer(snapshot["dataset_count"], "dataset_count")
    if valid > rows or datasets > rows:
        raise ValueError("broad snapshot counts are inconsistent")
    domain_counts = snapshot["domain_counts"]
    fidelity_counts = snapshot["fidelity_counts"]
    if (
        not isinstance(domain_counts, Mapping)
        or set(domain_counts) != {"real", "synthetic", "online"}
        or sum(domain_counts.values()) != rows
        or not isinstance(fidelity_counts, Mapping)
        or set(fidelity_counts) != FIDELITIES
        or sum(fidelity_counts.values()) != rows
    ):
        raise ValueError("broad snapshot domain/fidelity counts differ from rows")
    required_files = {
        "token_ids",
        "offsets",
        "dataset_embeddings",
        "target_summaries",
        "anchor_features",
        "anchor_masks",
        "scores",
        "validity",
        "runtime",
        "metadata",
    }
    files = snapshot["files"]
    if not isinstance(files, Mapping) or set(files) != required_files:
        raise ValueError("broad snapshot binary members differ from the v1 pack")
    for name, value in files.items():
        member = _exact_fields(value, _FILE_FIELDS, f"broad corpus file {name}")
        _safe_relative_path(member["path"], f"{name}.path")
        _sha256(member["sha256"], f"{name}.sha256")
        _positive_integer(member["bytes"], f"{name}.bytes", allow_zero=True)
        if not isinstance(member["dtype"], str) or not member["dtype"]:
            raise ValueError(f"{name}.dtype must be nonempty")
        if (
            not isinstance(member["shape"], list)
            or not member["shape"]
            or any(type(item) is not int or item < 0 for item in member["shape"])
        ):
            raise ValueError(f"{name}.shape is invalid")
    if production and rows < PILOT_RECEIPT_COUNT:
        raise ValueError("pilot corpus requires at least 20,000 receipts")
