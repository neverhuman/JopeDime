"""Build the owner-approved, metadata-only keyed S3 subset and grouped folds."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Iterable, Mapping, Sequence

ALLOWED_TASKS = {"regression", "binary"}
EXPECTED_COUNTS = {"regression": 2720, "binary": 7}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_keyed_subset_manifest(
    items: Sequence[Mapping[str, object]],
    *,
    inventory_sha256: str,
    seed: str = "phase3-keyed-s3-v1",
    sealed_excluded_family_ids: Iterable[str] | None = None,
    expected_counts: Mapping[str, int] | None = EXPECTED_COUNTS,
) -> dict[str, object]:
    """Use only real S3 rows with recorded family keys; never synthesize IDs.

    A supplied sealed exclusion set is consumed read-only and only its hash/count
    are recorded. Manifest `has_meta_test` rows exclude their whole known family.
    """
    if len(inventory_sha256) != 64:
        raise ValueError("inventory_sha256 must be a SHA-256 digest")
    exclusion_roster_supplied = sealed_excluded_family_ids is not None
    excluded = {
        str(f).strip() for f in (sealed_excluded_family_ids or ()) if str(f).strip()
    }
    sealed_families = {
        str(row["family_key"]).strip()
        for row in items
        if row.get("has_meta_test") is True
        and isinstance(row.get("family_key"), str)
        and row["family_key"].strip()
    }
    excluded |= sealed_families

    candidates: list[dict[str, object]] = []
    excluded_sealed_rows = 0
    for row in items:
        task = row.get("task")
        family = row.get("family_key")
        dataset_id = row.get("id")
        if task not in ALLOWED_TASKS or row.get("origin") != "real":
            continue
        if row.get("has_meta_test") is True or (isinstance(family, str) and family.strip() in excluded):
            excluded_sealed_rows += 1
            continue
        if not isinstance(dataset_id, str) or not dataset_id:
            continue
        if not isinstance(family, str) or not family.strip():
            continue
        candidates.append({
            "dataset_id": dataset_id,
            "family_id": family.strip(),
            "task": str(task),
            "origin": "real",
            "manifest_sha256": str(row.get("manifest_sha256", "")),
        })

    ids = [str(row["dataset_id"]) for row in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate dataset IDs in keyed subset inventory")
    counts = Counter(str(row["task"]) for row in candidates)
    if expected_counts is not None and dict(counts) != dict(expected_counts):
        raise ValueError(f"keyed subset coverage differs from owner decision: {dict(counts)}")

    families: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in candidates:
        families[str(row["family_id"])].append(row)
    per_fold = [Counter() for _ in range(5)]
    totals = Counter(str(row["task"]) for row in candidates)
    targets = {task: total / 5 for task, total in totals.items()}
    ordered_families = sorted(
        families,
        key=lambda family: (
            -len(families[family]),
            hashlib.sha256(f"{seed}\0{family}".encode()).hexdigest(),
        ),
    )
    fold_for_family: dict[str, int] = {}
    for family in ordered_families:
        family_counts = Counter(str(row["task"]) for row in families[family])
        def score(fold: int) -> tuple[float, int]:
            error = 0.0
            for candidate_fold in range(5):
                for task, count in totals.items():
                    value = per_fold[candidate_fold][task]
                    if candidate_fold == fold:
                        value += family_counts[task]
                    error += ((value - targets[task]) / max(targets[task], 1.0)) ** 2
            return error, fold
        empty_folds = set(range(5)) - set(fold_for_family.values())
        chosen = min(empty_folds if empty_folds else set(range(5)), key=score)
        fold_for_family[family] = chosen
        per_fold[chosen].update(family_counts)

    for row in candidates:
        row["outer_fold"] = fold_for_family[str(row["family_id"])]
    assignments = sorted(candidates, key=lambda row: str(row["dataset_id"]))
    fold_map_sha256 = _canonical_sha256(assignments)
    family_count_by_task: dict[str, int] = {}
    for task in sorted(ALLOWED_TASKS):
        family_count_by_task[task] = len({
            str(row["family_id"]) for row in candidates if row["task"] == task
        })
    return {
        "schema": "jope.phase3.keyed_s3_fold_manifest.v1",
        "scope": "keyed_subset",
        "provenance": "authoritative_s3_family_roster",
        "sealed_partition_included": False,
        "seed": seed,
        "inventory_sha256": inventory_sha256,
        "family_roster_sha256": _canonical_sha256(sorted(families)),
        "fold_map_sha256": fold_map_sha256,
        "sealed_excluded_applied": exclusion_roster_supplied,
        "sealed_excluded_family_count": len(excluded),
        "sealed_rows_excluded": excluded_sealed_rows,
        "coverage": {
            "regression": f"{counts['regression']}/3603",
            "binary": f"{counts['binary']}/843",
            "datasets": len(assignments),
            "families_by_task": family_count_by_task,
            "datasets_by_fold_and_task": [dict(sorted(row.items())) for row in per_fold],
        },
        "assignments": assignments,
    }


def inventory_digest(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
