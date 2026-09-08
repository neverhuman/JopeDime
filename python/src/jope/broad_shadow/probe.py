"""Exact ten-slot probe construction with deterministic duplicate refill."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import (
    PROBE_SLATE_SCHEMA,
    SLOT_ROLE_COUNTS,
    build_hashed,
    validate_probe_slate,
)


def _candidate_order(
    candidates: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    dataset: str,
    role: str,
) -> list[Mapping[str, Any]]:
    return sorted(
        candidates,
        key=lambda candidate: hashlib.sha256(
            (
                f"{seed}\0{dataset}\0{role}\0"
                f"{candidate['semantic_sha256']}"
            ).encode()
        ).digest(),
    )


def _take_unique(
    candidates: Sequence[Mapping[str, Any]],
    *,
    count: int,
    used: set[str],
    seed: int,
    dataset: str,
    role: str,
) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    for candidate in _candidate_order(
        candidates, seed=seed, dataset=dataset, role=role
    ):
        semantic = str(candidate["semantic_sha256"])
        if semantic in used:
            continue
        used.add(semantic)
        selected.append(candidate)
        if len(selected) == count:
            return selected
    raise ValueError(f"dataset {dataset} cannot fill {count} unique {role} slots")


def build_probe_slate(
    *,
    catalog_sha256: str,
    anchors: Sequence[Mapping[str, Any]],
    pilot_datasets: Sequence[Mapping[str, Any]],
    candidate_pools: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    seed: int,
    created_at_ms: int,
    production: bool = False,
) -> dict[str, Any]:
    datasets: list[dict[str, Any]] = []
    for row in sorted(pilot_datasets, key=lambda value: str(value["dataset_hash"])):
        dataset = str(row["dataset_hash"])
        pools = candidate_pools.get(dataset)
        if pools is None:
            raise ValueError(f"candidate pools are missing dataset {dataset}")
        used = {str(anchor["semantic_sha256"]) for anchor in anchors}
        slots: list[dict[str, Any]] = [
            {
                "slot": index,
                "role": "anchor",
                "engine": anchor["engine"],
                "semantic_sha256": anchor["semantic_sha256"],
                "canonical_genome": anchor["canonical_genome"],
                "source_dataset_hash": None,
                "anchor_index": index,
            }
            for index, anchor in enumerate(anchors)
        ]
        refill = list(pools.get("refill", ()))
        for role in ("transfer", "policy", "diverse"):
            requested = list(pools.get(role, ()))
            count = SLOT_ROLE_COUNTS[role]
            selected: list[Mapping[str, Any]] = []
            for source in (requested, refill):
                needed = count - len(selected)
                if needed == 0:
                    break
                available = [
                    candidate
                    for candidate in source
                    if str(candidate["semantic_sha256"]) not in used
                ]
                if not available:
                    continue
                selected.extend(
                    _take_unique(
                        available,
                        count=min(needed, len(available)),
                        used=used,
                        seed=seed,
                        dataset=dataset,
                        role=role,
                    )
                )
            if len(selected) != count:
                raise ValueError(
                    f"dataset {dataset} cannot deterministically refill {role}"
                )
            for candidate in selected:
                slots.append(
                    {
                        "slot": len(slots),
                        "role": role,
                        "engine": candidate["engine"],
                        "semantic_sha256": candidate["semantic_sha256"],
                        "canonical_genome": candidate["canonical_genome"],
                        "source_dataset_hash": candidate.get("source_dataset_hash"),
                        "anchor_index": None,
                    }
                )
        datasets.append(
            {
                "dataset_hash": dataset,
                "synthetic_split": row["split"],
                "slots": slots,
            }
        )
    slate = build_hashed(
        PROBE_SLATE_SCHEMA,
        "slate_sha256",
        catalog_sha256=catalog_sha256,
        created_at_ms=created_at_ms,
        seed=seed,
        screening_protocol={
            "fidelity": "screening",
            "folds": 3,
            "row_cap": 768,
            "timeout_seconds": 30,
            "omp_threads": 1,
            "train_only": True,
        },
        exact_protocol={
            "fidelity": "exact",
            "folds": 5,
            "row_cap": None,
            "timeout_seconds": None,
            "omp_threads": 1,
            "train_only": True,
        },
        anchors=[dict(anchor) for anchor in anchors],
        datasets=datasets,
    )
    validate_probe_slate(slate, production=production)
    return slate
