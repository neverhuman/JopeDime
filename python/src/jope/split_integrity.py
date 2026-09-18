"""Frozen dataset split authority for Prime policy training.

The manifest deliberately depends on an explicit ``source_family`` supplied by the
signature inventory.  Dataset hashes are not a safe proxy for lineage families.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SPLIT_MANIFEST_V1 = "PrimeDatasetSplitManifest.v1"
TRAIN_SPLIT = "train"
DEVELOPMENT_SPLIT = "development"
TEMPORAL_AUDIT_SPLIT = "temporal_audit"
REPORT_ONLY_SPLITS = frozenset(
    {"arena", "sealed", "double_holdout", TEMPORAL_AUDIT_SPLIT}
)
_ASSIGNABLE_SPLITS = frozenset({TRAIN_SPLIT, DEVELOPMENT_SPLIT, TEMPORAL_AUDIT_SPLIT})


class SplitIntegrityError(ValueError):
    """A split cannot safely be created or reused."""


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_families_from_signature_manifest(path: str | Path) -> dict[str, str]:
    """Read only explicit source-family identities; never derive or guess them."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SplitIntegrityError("signature manifest must be a JSON object")
    datasets = raw.get("datasets", raw)
    if not isinstance(datasets, dict):
        raise SplitIntegrityError("signature manifest datasets must be an object")
    families: dict[str, str] = {}
    for dataset, record in datasets.items():
        if not isinstance(dataset, str) or not dataset:
            raise SplitIntegrityError("signature manifest dataset identity is empty")
        if not isinstance(record, dict):
            continue
        family = record.get("source_family")
        if isinstance(family, str) and family.strip():
            families[dataset] = family.strip()
    return families


def _label_body(row: dict[str, Any]) -> dict[str, Any]:
    label = row.get("label")
    return label if isinstance(label, dict) else row


def _dataset(row: dict[str, Any]) -> str:
    label = _label_body(row)
    return str(
        label.get("dataset")
        or label.get("dataset_hash")
        or label.get("dataset_id")
        or ""
    )


def _authority_split(row: dict[str, Any]) -> str:
    value = _label_body(row).get("split")
    return (
        str(value).strip().lower().replace("-", "_") if value is not None else "train"
    )


def _committed_at_ms(row: dict[str, Any]) -> int | None:
    value = row.get("committed_at_ms") or row.get("_committed_at_ms")
    if type(value) is int and value >= 0:
        return value
    return None


def is_training_authority_label(row: dict[str, Any]) -> bool:
    """Only explicit train authority may influence the policy corpus."""
    split = _authority_split(row)
    return split == TRAIN_SPLIT


def _manifest_payload(
    *,
    status: str,
    seed: int,
    development_fraction: float,
    temporal_audit_fraction: float,
    assignments: dict[str, dict[str, Any]],
    missing_source_family: list[str],
    missing_committed_at_ms: list[str],
    incomplete_reasons: list[str],
    legacy_development_dataset_ids: list[str],
) -> dict[str, Any]:
    audit_times = [
        record["first_committed_at_ms"]
        for record in assignments.values()
        if record["split"] == TEMPORAL_AUDIT_SPLIT
    ]
    payload: dict[str, Any] = {
        "schema": SPLIT_MANIFEST_V1,
        "status": status,
        "seed": seed,
        "development_fraction": development_fraction,
        "temporal_audit_fraction": temporal_audit_fraction,
        "temporal_audit_cutoff_ms": min(audit_times) if audit_times else None,
        "assignments": {key: assignments[key] for key in sorted(assignments)},
        "missing_source_family": sorted(set(missing_source_family)),
        "missing_committed_at_ms": sorted(set(missing_committed_at_ms)),
        "incomplete_reasons": sorted(set(incomplete_reasons)),
        "legacy_development_dataset_ids": sorted(set(legacy_development_dataset_ids)),
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    return payload


def validate_split_manifest(manifest: dict[str, Any]) -> None:
    expected = {
        "schema",
        "status",
        "seed",
        "development_fraction",
        "temporal_audit_fraction",
        "temporal_audit_cutoff_ms",
        "assignments",
        "missing_source_family",
        "missing_committed_at_ms",
        "incomplete_reasons",
        "legacy_development_dataset_ids",
        "manifest_sha256",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected:
        raise SplitIntegrityError("split manifest has unknown or missing fields")
    if manifest["schema"] != SPLIT_MANIFEST_V1:
        raise SplitIntegrityError("unsupported split manifest schema")
    if manifest["status"] not in {"complete", "incomplete"}:
        raise SplitIntegrityError("split manifest status is invalid")
    if type(manifest["seed"]) is not int:
        raise SplitIntegrityError("split manifest seed must be an integer")
    for field in ("development_fraction", "temporal_audit_fraction"):
        value = manifest[field]
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise SplitIntegrityError(f"split manifest {field} must be finite")
        if not 0.0 < float(value) < 1.0:
            raise SplitIntegrityError(f"split manifest {field} must be in (0, 1)")
    digest_payload = dict(manifest)
    claimed_digest = digest_payload.pop("manifest_sha256")
    if claimed_digest != canonical_sha256(digest_payload):
        raise SplitIntegrityError("split manifest content hash differs")
    assignments = manifest["assignments"]
    if not isinstance(assignments, dict):
        raise SplitIntegrityError("split manifest assignments must be an object")
    family_split: dict[str, str] = {}
    observed_splits: set[str] = set()
    audit_times: list[int] = []
    for dataset, record in assignments.items():
        if not isinstance(dataset, str) or not dataset:
            raise SplitIntegrityError("split manifest dataset identity is empty")
        if not isinstance(record, dict) or set(record) != {
            "source_family",
            "split",
            "first_committed_at_ms",
        }:
            raise SplitIntegrityError("split assignment has unknown or missing fields")
        family = record["source_family"]
        split = record["split"]
        committed = record["first_committed_at_ms"]
        if not isinstance(family, str) or not family:
            raise SplitIntegrityError("split assignment source_family is empty")
        if split not in _ASSIGNABLE_SPLITS:
            raise SplitIntegrityError("split assignment has an invalid split")
        if type(committed) is not int or committed < 0:
            raise SplitIntegrityError("split assignment timestamp is invalid")
        prior = family_split.setdefault(family, split)
        if prior != split:
            raise SplitIntegrityError("source family overlaps multiple splits")
        observed_splits.add(split)
        if split == TEMPORAL_AUDIT_SPLIT:
            audit_times.append(committed)
    for field in (
        "missing_source_family",
        "missing_committed_at_ms",
        "incomplete_reasons",
        "legacy_development_dataset_ids",
    ):
        value = manifest[field]
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item for item in value
        ):
            raise SplitIntegrityError(f"split manifest {field} must be strings")
        if value != sorted(set(value)):
            raise SplitIntegrityError(f"split manifest {field} is not canonical")
    expected_cutoff = min(audit_times) if audit_times else None
    if manifest["temporal_audit_cutoff_ms"] != expected_cutoff:
        raise SplitIntegrityError("temporal audit cutoff differs from assignments")
    if manifest["status"] == "complete":
        if (
            manifest["incomplete_reasons"]
            or manifest["missing_source_family"]
            or manifest["missing_committed_at_ms"]
        ):
            raise SplitIntegrityError(
                "complete split manifest reports incomplete inputs"
            )
        if observed_splits != _ASSIGNABLE_SPLITS:
            raise SplitIntegrityError(
                "complete split manifest requires all three splits"
            )
    elif assignments:
        raise SplitIntegrityError("incomplete split manifest cannot assign datasets")


def build_split_manifest(
    labels: Iterable[dict[str, Any]],
    *,
    source_families: dict[str, str],
    development_fraction: float,
    temporal_audit_fraction: float,
    seed: int,
    legacy_development_dataset_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Create a deterministic family-disjoint split over current train authority."""
    labels = [row for row in labels if is_training_authority_label(row)]
    datasets = sorted({_dataset(row) for row in labels} - {""})
    missing_family = [dataset for dataset in datasets if dataset not in source_families]
    first_committed: dict[str, int] = {}
    missing_time: list[str] = []
    for dataset in datasets:
        times = [
            value
            for row in labels
            if _dataset(row) == dataset
            for value in [_committed_at_ms(row)]
            if value is not None
        ]
        if times:
            first_committed[dataset] = min(times)
        else:
            missing_time.append(dataset)
    reasons: list[str] = []
    if missing_family:
        reasons.append("missing_source_family")
    if missing_time:
        reasons.append("missing_committed_at_ms")
    if not datasets:
        reasons.append("no_train_authority_datasets")
    legacy_dev = sorted(set(str(value) for value in legacy_development_dataset_ids))
    unknown_legacy = sorted(set(legacy_dev) - set(datasets))
    if unknown_legacy:
        reasons.append("legacy_development_dataset_missing")

    families: dict[str, list[str]] = {}
    if not reasons:
        for dataset in datasets:
            families.setdefault(source_families[dataset], []).append(dataset)
        if len(families) < 3:
            reasons.append("fewer_than_three_source_families")

    development_families: set[str] = set()
    if not reasons and legacy_dev:
        legacy_set = set(legacy_dev)
        for family, members in families.items():
            selected = legacy_set.intersection(members)
            if selected and selected != set(members):
                reasons.append("legacy_development_family_overlap")
            elif selected:
                development_families.add(family)
        if not development_families:
            reasons.append("legacy_development_split_empty")

    if reasons:
        manifest = _manifest_payload(
            status="incomplete",
            seed=seed,
            development_fraction=development_fraction,
            temporal_audit_fraction=temporal_audit_fraction,
            assignments={},
            missing_source_family=missing_family,
            missing_committed_at_ms=missing_time,
            incomplete_reasons=reasons,
            legacy_development_dataset_ids=legacy_dev,
        )
        validate_split_manifest(manifest)
        return manifest

    family_time = {
        family: min(first_committed[dataset] for dataset in members)
        for family, members in families.items()
    }
    audit_candidates = [
        family for family in families if family not in development_families
    ]
    audit_count = max(1, round(temporal_audit_fraction * len(audit_candidates)))
    audit_count = min(audit_count, len(audit_candidates) - 1)
    audit_families = set(
        sorted(audit_candidates, key=lambda family: (family_time[family], family))[
            -audit_count:
        ]
    )
    remaining = [
        family
        for family in sorted(families)
        if family not in audit_families and family not in development_families
    ]
    if not development_families:
        rng = random.Random(seed)
        rng.shuffle(remaining)
        development_count = max(1, round(development_fraction * len(remaining)))
        development_count = min(development_count, len(remaining) - 1)
        development_families = set(remaining[:development_count])
    train_families = set(families) - audit_families - development_families
    if not train_families or not development_families or not audit_families:
        manifest = _manifest_payload(
            status="incomplete",
            seed=seed,
            development_fraction=development_fraction,
            temporal_audit_fraction=temporal_audit_fraction,
            assignments={},
            missing_source_family=[],
            missing_committed_at_ms=[],
            incomplete_reasons=["unable_to_form_three_nonempty_family_splits"],
            legacy_development_dataset_ids=legacy_dev,
        )
        validate_split_manifest(manifest)
        return manifest

    assignments: dict[str, dict[str, Any]] = {}
    for family, members in families.items():
        split = (
            TEMPORAL_AUDIT_SPLIT
            if family in audit_families
            else DEVELOPMENT_SPLIT
            if family in development_families
            else TRAIN_SPLIT
        )
        for dataset in members:
            assignments[dataset] = {
                "source_family": family,
                "split": split,
                "first_committed_at_ms": first_committed[dataset],
            }
    manifest = _manifest_payload(
        status="complete",
        seed=seed,
        development_fraction=development_fraction,
        temporal_audit_fraction=temporal_audit_fraction,
        assignments=assignments,
        missing_source_family=[],
        missing_committed_at_ms=[],
        incomplete_reasons=[],
        legacy_development_dataset_ids=legacy_dev,
    )
    validate_split_manifest(manifest)
    return manifest


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def load_or_create_split_manifest(
    path: str | Path,
    labels: Iterable[dict[str, Any]],
    *,
    source_families: dict[str, str],
    development_fraction: float,
    temporal_audit_fraction: float,
    seed: int,
    strict: bool = True,
) -> dict[str, Any]:
    """Load immutable authority or create it once; reject later assignment drift."""
    path = Path(path)
    labels = list(labels)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(existing, dict) and existing.get("schema") == SPLIT_MANIFEST_V1:
            validate_split_manifest(existing)
            current = build_split_manifest(
                labels,
                source_families=source_families,
                development_fraction=float(existing["development_fraction"]),
                temporal_audit_fraction=float(existing["temporal_audit_fraction"]),
                seed=int(existing["seed"]),
                legacy_development_dataset_ids=existing[
                    "legacy_development_dataset_ids"
                ],
            )
            if existing != current:
                raise SplitIntegrityError(
                    "frozen split assignments differ from current dataset/family inputs"
                )
            manifest = existing
        elif isinstance(existing, dict) and set(existing) == {"dataset_ids"}:
            dataset_ids = existing["dataset_ids"]
            if not isinstance(dataset_ids, list):
                raise SplitIntegrityError("legacy validation state is malformed")
            manifest = build_split_manifest(
                labels,
                source_families=source_families,
                development_fraction=development_fraction,
                temporal_audit_fraction=temporal_audit_fraction,
                seed=seed,
                legacy_development_dataset_ids=dataset_ids,
            )
            _atomic_json(path, manifest)
        else:
            raise SplitIntegrityError("unsupported validation/split state")
    else:
        manifest = build_split_manifest(
            labels,
            source_families=source_families,
            development_fraction=development_fraction,
            temporal_audit_fraction=temporal_audit_fraction,
            seed=seed,
        )
        _atomic_json(path, manifest)
    if strict and manifest["status"] != "complete":
        reasons = ", ".join(manifest["incomplete_reasons"])
        raise SplitIntegrityError(f"split manifest is incomplete: {reasons}")
    return manifest


def assignment_for_dataset(manifest: dict[str, Any], dataset: str) -> dict[str, Any]:
    validate_split_manifest(manifest)
    record = manifest["assignments"].get(dataset)
    if not isinstance(record, dict):
        raise SplitIntegrityError(
            f"dataset {dataset!r} is absent from frozen split assignments"
        )
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
                raise SplitIntegrityError(
                    "TrainingRow has conflicting report-only authority"
                )
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
        raise SplitIntegrityError(
            "manifest-bound training cannot mix unbound train/development rows"
        )
    return {
        "train": train,
        "development": development,
        "report_only": report_only,
        "report_only_counts": report_counts,
        "split_manifest_sha256": bound_hash,
        "compatibility": "historical_unbound" if not bound_present else "bound_v1",
    }
