"""Immutable persistence for dataset split manifests."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .split_integrity import (
    SPLIT_MANIFEST_V1,
    SplitIntegrityError,
    build_split_manifest,
    validate_split_manifest,
)


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
                legacy_development_dataset_ids=existing["legacy_development_dataset_ids"],
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
