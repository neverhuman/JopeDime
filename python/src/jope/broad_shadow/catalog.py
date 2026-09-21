"""Deterministic catalog, family, pilot, and semantic-genome splits."""

from __future__ import annotations

import hashlib
import itertools
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .contracts import (
    BROAD_CATALOG_SCHEMA,
    PILOT_DATASET_COUNT,
    REAL_SPLITS,
    SYNTHETIC_SPLITS,
    build_hashed,
    validate_broad_catalog,
)


def _order_key(seed: int, *parts: str) -> bytes:
    body = "\0".join((str(seed), *parts)).encode()
    return hashlib.sha256(body).digest()


def freeze_synthetic_family_splits(
    generator_lineages: Iterable[str],
    *,
    seed: int,
) -> dict[str, str]:
    """Assign normalized generator lineages 80/10/10, never dataset IDs."""

    families = sorted(set(generator_lineages), key=lambda value: _order_key(seed, value))
    if len(families) < 3:
        raise ValueError("synthetic authority needs at least three generator lineages")
    train_end = max(1, min(int(0.8 * len(families)), len(families) - 2))
    validation_end = max(
        train_end + 1,
        min(train_end + int(0.1 * len(families)), len(families) - 1),
    )
    return {
        family: ("train" if index < train_end else "validation" if index < validation_end else "test")
        for index, family in enumerate(families)
    }


def build_broad_catalog(
    rows: Sequence[Mapping[str, Any]],
    *,
    created_at_ms: int,
    foundation_cache_count: int,
    production: bool = False,
) -> dict[str, Any]:
    """Freeze already-reviewed rows into ``BroadDatasetCatalog.v1``."""

    ordered = sorted((dict(row) for row in rows), key=lambda row: row["dataset_hash"])
    real_counts = Counter(row["split"] for row in ordered if row.get("domain") == "real")
    synthetic_counts = Counter(row["split"] for row in ordered if row.get("domain") == "synthetic")
    family_splits: dict[str, str] = {}
    for row in ordered:
        if row.get("domain") == "synthetic":
            family = str(row["generator_lineage"])
            prior = family_splits.setdefault(family, str(row["split"]))
            if prior != row["split"]:
                raise ValueError("generator lineage appears in multiple splits")
    payload = build_hashed(
        BROAD_CATALOG_SCHEMA,
        "catalog_sha256",
        created_at_ms=created_at_ms,
        dataset_embedding_dimensions=1536,
        target_encoding_dimensions=512,
        foundation_cache_count=foundation_cache_count,
        hyperion_complete_count=sum(bool(row["hyperion_complete"]) for row in ordered),
        real_split_counts={split: real_counts[split] for split in sorted(REAL_SPLITS)},
        synthetic_split_counts={split: synthetic_counts[split] for split in sorted(SYNTHETIC_SPLITS)},
        synthetic_family_split_counts={
            split: sum(assigned == split for assigned in family_splits.values())
            for split in sorted(SYNTHETIC_SPLITS)
        },
        datasets=ordered,
    )
    validate_broad_catalog(payload, production=production)
    return payload


def select_balanced_synthetic_pilot(
    catalog_rows: Sequence[Mapping[str, Any]],
    *,
    count: int = PILOT_DATASET_COUNT,
    seed: int,
) -> list[Mapping[str, Any]]:
    """Round-robin families and shape strata into an exact pilot."""

    eligible = [row for row in catalog_rows if row.get("domain") == "synthetic"]
    if len(eligible) < count:
        raise ValueError(f"pilot needs {count} synthetic datasets, found {len(eligible)}")
    buckets: dict[tuple[str, int, int, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in eligible:
        rows = int(row["rows"])
        features = int(row["features"])
        row_stratum = 0 if rows <= 256 else 1 if rows <= 768 else 2
        feature_stratum = 0 if features <= 16 else 1 if features <= 64 else 2
        key = (
            str(row["generator_lineage"]),
            row_stratum,
            feature_stratum,
            str(row["target_shape"]),
            str(row["split"]),
        )
        buckets[key].append(row)
    for key, values in buckets.items():
        values.sort(key=lambda row: _order_key(seed, *map(str, key), str(row["dataset_hash"])))
    keys = sorted(buckets, key=lambda key: _order_key(seed, *map(str, key)))
    selected: list[Mapping[str, Any]] = []
    while len(selected) < count:
        advanced = False
        for key in keys:
            if buckets[key]:
                selected.append(buckets[key].pop())
                advanced = True
                if len(selected) == count:
                    break
        if not advanced:
            raise AssertionError("eligible pilot rows disappeared during balancing")
    return sorted(selected, key=lambda row: str(row["dataset_hash"]))


def select_probe_anchors(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Choose the minimum-runtime distinct-family anchor across four quartiles."""

    by_quartile: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        quartile = candidate.get("performance_quartile")
        if (
            type(quartile) is int
            and quartile in range(4)
            and candidate.get("grammar_valid") is True
            and int(candidate.get("real_dataset_coverage", 0)) >= 100
        ):
            by_quartile[quartile].append(candidate)
    if set(by_quartile) != set(range(4)):
        raise ValueError("anchor authority lacks an eligible candidate in every quartile")

    # Keep only the lowest-runtime representative per family/quartile.  This
    # bounds the cartesian search by reviewed algorithm families.
    compact: dict[int, list[Mapping[str, Any]]] = {}
    for quartile, values in by_quartile.items():
        best: dict[str, Mapping[str, Any]] = {}
        for value in values:
            family = str(value["algorithm_family"])
            key = (float(value["median_runtime_seconds"]), str(value["semantic_sha256"]))
            prior = best.get(family)
            if prior is None or key < (
                float(prior["median_runtime_seconds"]),
                str(prior["semantic_sha256"]),
            ):
                best[family] = value
        compact[quartile] = sorted(
            best.values(),
            key=lambda value: (
                float(value["median_runtime_seconds"]),
                str(value["semantic_sha256"]),
            ),
        )

    feasible = (
        combination
        for combination in itertools.product(*(compact[index] for index in range(4)))
        if len({str(value["algorithm_family"]) for value in combination}) == 4
    )
    try:
        selected = min(
            feasible,
            key=lambda combination: (
                sum(float(value["median_runtime_seconds"]) for value in combination),
                tuple(str(value["semantic_sha256"]) for value in combination),
            ),
        )
    except ValueError as error:
        raise ValueError("four distinct anchor algorithm families are unavailable") from error
    return [
        {
            "anchor_index": index,
            "engine": value["engine"],
            "semantic_sha256": value["semantic_sha256"],
            "canonical_genome": value["canonical_genome"],
            "algorithm_family": value["algorithm_family"],
            "performance_quartile": index,
            "median_runtime_seconds": value["median_runtime_seconds"],
            "real_dataset_coverage": value["real_dataset_coverage"],
        }
        for index, value in enumerate(selected)
    ]


def constrained_semantic_genome_splits(
    receipts: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    minimum_dataset_rows: int = 10,
) -> dict[str, str]:
    """Globally split semantic hashes and repair per-dataset representation.

    A semantic genome has one split everywhere.  The initial assignment is an
    exact global 80/10/10 partition; deterministic repairs only move a genome
    when doing so cannot remove the last representative of its current split
    from another constrained dataset.
    """

    dataset_semantics: dict[str, set[str]] = defaultdict(set)
    for row in receipts:
        dataset_semantics[str(row["dataset_hash"])].add(str(row["semantic_sha256"]))
    semantics = sorted(
        {semantic for values in dataset_semantics.values() for semantic in values},
        key=lambda semantic: _order_key(seed, semantic),
    )
    if len(semantics) < 3:
        raise ValueError("global semantic split needs at least three genomes")
    train_end = max(1, min(int(0.8 * len(semantics)), len(semantics) - 2))
    validation_end = max(
        train_end + 1,
        min(train_end + int(0.1 * len(semantics)), len(semantics) - 1),
    )
    names = ("train", "validation", "test")
    assignment = {
        semantic: ("train" if index < train_end else "validation" if index < validation_end else "test")
        for index, semantic in enumerate(semantics)
    }
    constrained = {
        dataset: values
        for dataset, values in dataset_semantics.items()
        if len(values) >= minimum_dataset_rows
    }
    membership: dict[str, set[str]] = defaultdict(set)
    for dataset, values in constrained.items():
        for semantic in values:
            membership[semantic].add(dataset)

    def counts() -> dict[str, Counter[str]]:
        return {
            dataset: Counter(assignment[semantic] for semantic in values)
            for dataset, values in constrained.items()
        }

    current = counts()
    for _ in range(max(1, len(constrained) * 3)):
        missing = [
            (dataset, split)
            for dataset in sorted(constrained)
            for split in names
            if current[dataset][split] == 0
        ]
        if not missing:
            return assignment
        changed = False
        for dataset, target in missing:
            candidates = []
            for semantic in constrained[dataset]:
                source = assignment[semantic]
                if source == target:
                    continue
                affected = membership[semantic]
                if any(current[member][source] <= 1 for member in affected):
                    continue
                candidates.append(
                    (
                        len(affected),
                        _order_key(seed, dataset, target, semantic),
                        semantic,
                        source,
                    )
                )
            if not candidates:
                continue
            _, _, semantic, source = min(candidates)
            assignment[semantic] = target
            for member in membership[semantic]:
                current[member][source] -= 1
                current[member][target] += 1
            changed = True
        if not changed:
            break
    missing = {
        dataset: [split for split in names if current[dataset][split] == 0]
        for dataset in sorted(constrained)
        if any(current[dataset][split] == 0 for split in names)
    }
    raise ValueError(f"semantic split constraints are infeasible: {missing}")
