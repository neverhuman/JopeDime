"""Immutable schemas for the broad-shadow campaign.

All validators are deliberately exact-field and content-hash based.  The
production flag applies campaign-size gates; fixture-sized artifacts still use
the same scientific and leakage boundaries.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
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


def _optional_finite(
    value: Any, name: str, *, minimum: float | None = None
) -> float | None:
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


def _validate_encoding_ref(
    value: Any, name: str, expected_dimensions: int
) -> None:
    row = _exact_fields(value, _ENCODING_FIELDS, name)
    _sha256(row["sha256"], f"{name}.sha256")
    _safe_relative_path(row["path"], f"{name}.path")
    offset = _positive_integer(
        row["offset_bytes"], f"{name}.offset_bytes", allow_zero=True
    )
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
        if type(row["foundation_complete"]) is not bool or type(
            row["hyperion_complete"]
        ) is not bool:
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
        or declared_synthetic
        != {name: synthetic_counts[name] for name in sorted(SYNTHETIC_SPLITS)}
        or declared_families
        != {
            name: len(
                {family for family, assigned in family_split.items() if assigned == name}
            )
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
        if set(synthetic_counts) != SYNTHETIC_SPLITS or set(family_split.values()) != set(
            SYNTHETIC_SPLITS
        ):
            raise ValueError("production synthetic authority requires all three splits")
        if foundation_count < 27_875:
            raise ValueError("production catalog must bind the 27,875 foundation cache")
        if hyperion_count < 24_177:
            raise ValueError("pilot cannot start below the frozen Hyperion-ready floor")


_ANCHOR_FIELDS = {
    "anchor_index",
    "engine",
    "semantic_sha256",
    "canonical_genome",
    "algorithm_family",
    "performance_quartile",
    "median_runtime_seconds",
    "real_dataset_coverage",
}
_SLOT_FIELDS = {
    "slot",
    "role",
    "engine",
    "semantic_sha256",
    "canonical_genome",
    "source_dataset_hash",
    "anchor_index",
}
_SLATE_DATASET_FIELDS = {"dataset_hash", "synthetic_split", "slots"}
_PROTOCOL_FIELDS = {
    "fidelity",
    "folds",
    "row_cap",
    "timeout_seconds",
    "omp_threads",
    "train_only",
}
_SLATE_FIELDS = {
    "schema",
    "slate_sha256",
    "catalog_sha256",
    "created_at_ms",
    "seed",
    "screening_protocol",
    "exact_protocol",
    "anchors",
    "datasets",
}


def _validate_protocol(value: Any, expected: str) -> None:
    protocol = _exact_fields(value, _PROTOCOL_FIELDS, f"{expected} protocol")
    if (
        protocol["fidelity"] != expected
        or protocol["train_only"] is not True
        or protocol["omp_threads"] != 1
    ):
        raise ValueError(f"{expected} protocol violates train-only/OMP authority")
    if expected == "screening":
        if (
            protocol["folds"] != 3
            or protocol["row_cap"] != 768
            or protocol["timeout_seconds"] != 30
        ):
            raise ValueError("screening protocol must be CV3/768 rows/30 seconds")
    elif (
        protocol["folds"] != 5
        or protocol["row_cap"] is not None
        or protocol["timeout_seconds"] is not None
    ):
        raise ValueError("exact protocol must be full-train CV5 without screening caps")


def validate_probe_slate(payload: Any, *, production: bool = False) -> None:
    slate = _exact_fields(payload, _SLATE_FIELDS, "probe slate")
    if slate["schema"] != PROBE_SLATE_SCHEMA:
        raise ValueError("unsupported probe slate schema")
    _content_hash(slate, "slate_sha256", "probe slate")
    _sha256(slate["catalog_sha256"], "catalog_sha256")
    _positive_integer(slate["created_at_ms"], "created_at_ms")
    if type(slate["seed"]) is not int:
        raise ValueError("probe seed must be an integer")
    _validate_protocol(slate["screening_protocol"], "screening")
    _validate_protocol(slate["exact_protocol"], "exact")

    anchors = slate["anchors"]
    if not isinstance(anchors, list) or len(anchors) != ANCHOR_COUNT:
        raise ValueError("probe slate requires exactly four anchors")
    anchor_semantics: list[str] = []
    anchor_families: set[str] = set()
    for index, value in enumerate(anchors):
        anchor = _exact_fields(value, _ANCHOR_FIELDS, "probe anchor")
        if (
            anchor["anchor_index"] != index
            or anchor["performance_quartile"] != index
        ):
            raise ValueError("anchors must cover ordered performance quartiles 0..3")
        if anchor["engine"] not in {"featmath", "newmath"}:
            raise ValueError("probe anchor engine is unsupported")
        semantic = _sha256(anchor["semantic_sha256"], "anchor semantic_sha256")
        genome = anchor["canonical_genome"]
        if not isinstance(genome, str) or not genome:
            raise ValueError("anchor canonical genome must be nonempty")
        if hashlib.sha256(genome.encode()).hexdigest() != semantic:
            raise ValueError("anchor semantic hash does not bind its genome")
        family = _slug(anchor["algorithm_family"], "algorithm_family")
        if family in anchor_families:
            raise ValueError("anchor top-level algorithm families must be distinct")
        anchor_families.add(family)
        if anchor["real_dataset_coverage"] < 100:
            raise ValueError("anchor must cover at least 100 real datasets")
        _finite(
            anchor["median_runtime_seconds"],
            "anchor median_runtime_seconds",
            minimum=0.0,
        )
        anchor_semantics.append(semantic)
    if len(set(anchor_semantics)) != ANCHOR_COUNT:
        raise ValueError("anchor semantic hashes must be distinct")

    datasets = slate["datasets"]
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("probe slate datasets must be nonempty")
    previous: str | None = None
    for value in datasets:
        row = _exact_fields(value, _SLATE_DATASET_FIELDS, "probe dataset")
        dataset = _dataset(row["dataset_hash"])
        if previous is not None and previous >= dataset:
            raise ValueError("probe datasets must be strictly sorted and unique")
        previous = dataset
        if row["synthetic_split"] not in SYNTHETIC_SPLITS:
            raise ValueError("probe dataset must bind a synthetic family split")
        slots = row["slots"]
        if not isinstance(slots, list) or len(slots) != PILOT_GENOMES_PER_DATASET:
            raise ValueError("each probe dataset requires exactly ten slots")
        roles: Counter[str] = Counter()
        semantics: set[str] = set()
        for index, slot_value in enumerate(slots):
            slot = _exact_fields(slot_value, _SLOT_FIELDS, "probe slot")
            if slot["slot"] != index or slot["role"] not in SLOT_ROLE_COUNTS:
                raise ValueError("probe slots must be ordered and use frozen roles")
            if slot["engine"] not in {"featmath", "newmath"}:
                raise ValueError("probe slot engine is unsupported")
            semantic = _sha256(slot["semantic_sha256"], "slot semantic_sha256")
            genome = slot["canonical_genome"]
            if (
                not isinstance(genome, str)
                or not genome
                or hashlib.sha256(genome.encode()).hexdigest() != semantic
            ):
                raise ValueError("probe slot semantic hash does not bind its genome")
            if semantic in semantics:
                raise ValueError("probe dataset contains a duplicate semantic genome")
            semantics.add(semantic)
            roles[slot["role"]] += 1
            if slot["role"] == "anchor":
                anchor_index = slot["anchor_index"]
                if (
                    anchor_index != index
                    or semantic != anchor_semantics[anchor_index]
                    or slot["engine"] != anchors[anchor_index]["engine"]
                    or slot["source_dataset_hash"] is not None
                ):
                    raise ValueError("anchor slots must exactly match ProbeSlate anchors")
            else:
                if slot["anchor_index"] is not None:
                    raise ValueError("non-anchor slot cannot carry anchor_index")
                source = slot["source_dataset_hash"]
                if slot["role"] == "transfer":
                    _dataset(source, "source_dataset_hash")
                elif source is not None:
                    _dataset(source, "source_dataset_hash")
        if dict(roles) != SLOT_ROLE_COUNTS:
            raise ValueError("probe dataset role counts must be 4/2/2/2")
    if production and len(datasets) != PILOT_DATASET_COUNT:
        raise ValueError("production pilot requires exactly 2,000 synthetic datasets")


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
    if (
        not isinstance(genome, str)
        or not genome
        or hashlib.sha256(genome.encode()).hexdigest() != semantic
    ):
        raise ValueError("receipt semantic hash does not bind canonical genome")
    _sha256(receipt["evaluator_release_sha256"], "evaluator_release_sha256")
    _sha256(receipt["fold_plan_sha256"], "fold_plan_sha256")
    _sha256(receipt["vocabulary_sha256"], "vocabulary_sha256")
    fidelity = receipt["fidelity"]
    if fidelity not in FIDELITIES:
        raise ValueError("receipt fidelity must be screening or exact")
    _validate_protocol(receipt["protocol"], fidelity)
    if (
        receipt["label_authority"] != "train_only_oof_r2"
        or receipt["test_table_accessed"] is not False
    ):
        raise ValueError("test-table or benchmark labels are forbidden")
    if (
        not isinstance(receipt["selection_channel"], str)
        or not receipt["selection_channel"].strip()
    ):
        raise ValueError("selection_channel must be nonempty")
    _positive_integer(receipt["recorded_at_ms"], "recorded_at_ms")
    runtime = _optional_finite(
        receipt["runtime_seconds"], "runtime_seconds", minimum=0.0
    )
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
        if any(
            receipt[field] is not None for field in ("oof_r2", "robust_score")
        ) or receipt["fold_r2"] != []:
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
    valid = _positive_integer(
        snapshot["valid_row_count"], "valid_row_count", allow_zero=True
    )
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
        grouped[
            (receipt["dataset_hash"], receipt["fidelity"], protocol_hash)
        ].append(receipt)
    return dict(grouped)
