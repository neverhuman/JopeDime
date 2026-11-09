"""Fail-closed assignment and training-row boundaries."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .split_integrity import (
    _ASSIGNABLE_SPLITS,
    DEVELOPMENT_SPLIT,
    REPORT_ONLY_SPLITS,
    SPLIT_MANIFEST_V1,
    TEMPORAL_AUDIT_SPLIT,
    TRAIN_SPLIT,
    SplitIntegrityError,
    validate_split_manifest,
)


def assignment_for_dataset(manifest: dict[str, Any], dataset: str) -> dict[str, Any]:
    validate_split_manifest(manifest)
    record = manifest["assignments"].get(dataset)
    if not isinstance(record, dict):
        raise SplitIntegrityError(f"dataset {dataset!r} is absent from frozen split assignments")
    return record


def partition_training_rows(rows: Iterable[Any]) -> dict[str, Any]:
    """Fail-closed row boundary used before any model or checkpoint computation."""
    train: list[Any] = []
    development: list[Any] = []
    report_only: list[Any] = []
    report_counts: dict[str, int] = {}
    dataset_split: dict[str, str] = {}
    family_split: dict[str, str] = {}
    bound_hash: str | None = None
    bound_present = False
    unbound_model_rows = False

    for row in rows:
        metadata = dict(getattr(row, "metadata", None) or {})
        raw_split = metadata.get("split", TRAIN_SPLIT)
        split = str(raw_split).strip().lower().replace("-", "_")
        split = DEVELOPMENT_SPLIT if split == "validation" else split
        authority_hints = {
            str(metadata[key]).strip().lower().replace("-", "_")
            for key in (
                "authority_split",
                "dataset_split",
                "evidence_split",
                "source_split",
            )
            if metadata.get(key) is not None
        }
        tier = str(metadata.get("evidence_tier") or "").lower().replace("-", "_")
        if tier.startswith("arena_"):
            authority_hints.add("arena")
        for report_split in REPORT_ONLY_SPLITS:
            if report_split in tier:
                authority_hints.add(report_split)
        report_hints = authority_hints.intersection(REPORT_ONLY_SPLITS)
        if report_hints:
            if len(report_hints) != 1:
                raise SplitIntegrityError("TrainingRow has conflicting report-only authority")
            split = next(iter(report_hints))
        if split not in _ASSIGNABLE_SPLITS | REPORT_ONLY_SPLITS:
            raise SplitIntegrityError(f"unsupported TrainingRow split {raw_split!r}")
        if split == TEMPORAL_AUDIT_SPLIT or split in REPORT_ONLY_SPLITS:
            report_only.append(row)
            report_counts[split] = report_counts.get(split, 0) + 1
        elif split == TRAIN_SPLIT:
            train.append(row)
        else:
            development.append(row)

        schema = metadata.get("split_manifest_schema")
        if schema is None:
            if split in {TRAIN_SPLIT, DEVELOPMENT_SPLIT}:
                unbound_model_rows = True
            continue
        bound_present = True
        if schema != SPLIT_MANIFEST_V1:
            raise SplitIntegrityError("TrainingRow uses an unsupported split manifest")
        family = metadata.get("source_family")
        assignment = metadata.get("split_assignment")
        digest = metadata.get("split_manifest_sha256")
        if not isinstance(family, str) or not family:
            raise SplitIntegrityError("manifest-bound TrainingRow lacks source_family")
        if assignment != split:
            raise SplitIntegrityError("TrainingRow split differs from its assignment")
        if not isinstance(digest, str) or len(digest) != 64:
            raise SplitIntegrityError("TrainingRow split manifest hash is invalid")
        if bound_hash is None:
            bound_hash = digest
        elif bound_hash != digest:
            raise SplitIntegrityError("TrainingRows mix different split manifests")
        dataset = str(getattr(row, "dataset_id", ""))
        prior_dataset = dataset_split.setdefault(dataset, split)
        if prior_dataset != split:
            raise SplitIntegrityError("dataset is reassigned across splits")
        prior_family = family_split.setdefault(family, split)
        if prior_family != split:
            raise SplitIntegrityError("source family overlaps multiple splits")

    if bound_present and unbound_model_rows:
        raise SplitIntegrityError("manifest-bound training cannot mix unbound train/development rows")
    return {
        "train": train,
        "development": development,
        "report_only": report_only,
        "report_only_counts": report_counts,
        "split_manifest_sha256": bound_hash,
        "compatibility": "historical_unbound" if not bound_present else "bound_v1",
    }
