"""Probe slate schemas and validator."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

from .contracts_common import (
    ANCHOR_COUNT,
    PILOT_DATASET_COUNT,
    PILOT_GENOMES_PER_DATASET,
    PROBE_SLATE_SCHEMA,
    SLOT_ROLE_COUNTS,
    SYNTHETIC_SPLITS,
    _content_hash,
    _dataset,
    _exact_fields,
    _finite,
    _positive_integer,
    _sha256,
    _slug,
)

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
    if protocol["fidelity"] != expected or protocol["train_only"] is not True or protocol["omp_threads"] != 1:
        raise ValueError(f"{expected} protocol violates train-only/OMP authority")
    if expected == "screening":
        if protocol["folds"] != 3 or protocol["row_cap"] != 768 or protocol["timeout_seconds"] != 30:
            raise ValueError("screening protocol must be CV3/768 rows/30 seconds")
    elif protocol["folds"] != 5 or protocol["row_cap"] is not None or protocol["timeout_seconds"] is not None:
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
        if anchor["anchor_index"] != index or anchor["performance_quartile"] != index:
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
