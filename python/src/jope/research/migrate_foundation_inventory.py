#!/usr/bin/env python3
"""Publish the legacy V1 foundation inventory as immutable hashed objects.

The legacy files remain byte-for-byte in place for rollback.  Consumers use a
dataset-name hard-link projection, while the authoritative object store and all
published documents are addressed by SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


LEGACY_FOUNDATION = Path("/home/ubuntu/jope/data/processed/foundation_enc_v1")
LEGACY_STANDARDIZATION = Path(
    "/home/ubuntu/jope/data/processed/active_v3_dual_20260706/standardization.json"
)
FOUNDATION_ROOT = Path("/home/ubuntu/jope-flywheel/encodings/foundation-v1")
INVENTORY_SCHEMA = "JopeFoundationInventory.v1"


class FoundationMigrationError(RuntimeError):
    """The immutable destination disagrees with the legacy source identity."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()


def verify_file(path: Path, expected_size: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise FoundationMigrationError(f"immutable artifact is absent: {path}")
    actual_size = path.stat().st_size
    actual_sha256 = sha256_file(path)
    if actual_size != expected_size or actual_sha256 != expected_sha256:
        raise FoundationMigrationError(
            f"immutable artifact collision at {path}: expected "
            f"size={expected_size} sha256={expected_sha256}, found "
            f"size={actual_size} sha256={actual_sha256}"
        )


def publish_copy(source: Path, destination: Path, size: int, digest: str) -> None:
    if destination.exists():
        verify_file(destination, size, digest)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with source.open("rb") as incoming, temporary.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, 1 << 20)
            outgoing.flush()
        verify_file(temporary, size, digest)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            verify_file(destination, size, digest)
        os.chmod(destination, 0o444)
    finally:
        temporary.unlink(missing_ok=True)
    verify_file(destination, size, digest)


def publish_bytes(payload: bytes, destination: Path) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    if destination.name != f"{digest}.json":
        raise FoundationMigrationError(
            f"content-addressed document name mismatch: {destination.name} != {digest}.json"
        )
    if destination.exists():
        if destination.read_bytes() != payload:
            raise FoundationMigrationError(
                f"immutable document collision at {destination}"
            )
        return digest
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != payload:
                raise FoundationMigrationError(
                    f"immutable document collision at {destination}"
                )
        os.chmod(destination, 0o444)
    finally:
        temporary.unlink(missing_ok=True)
    if destination.read_bytes() != payload:
        raise FoundationMigrationError(f"published document changed at {destination}")
    return digest


def publish_dataset_link(
    object_path: Path, dataset_path: Path, size: int, digest: str
) -> None:
    if dataset_path.exists():
        verify_file(dataset_path, size, digest)
        if not os.path.samefile(object_path, dataset_path):
            raise FoundationMigrationError(
                f"dataset identity collision at {dataset_path}: bytes match but it is "
                "not linked to the immutable object"
            )
        return
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(object_path, dataset_path)
    except FileExistsError:
        pass
    verify_file(dataset_path, size, digest)
    if not os.path.samefile(object_path, dataset_path):
        raise FoundationMigrationError(
            f"dataset identity collision at {dataset_path}: destination is not the "
            "content-addressed object"
        )


def migrate(
    legacy_foundation: Path = LEGACY_FOUNDATION,
    legacy_standardization: Path = LEGACY_STANDARDIZATION,
    destination_root: Path = FOUNDATION_ROOT,
) -> dict:
    if not legacy_foundation.is_dir():
        raise FoundationMigrationError(
            f"legacy foundation inventory is absent: {legacy_foundation}"
        )
    if not legacy_standardization.is_file():
        raise FoundationMigrationError(
            f"legacy standardization is absent: {legacy_standardization}"
        )

    objects_root = destination_root / "objects" / "sha256"
    datasets_root = destination_root / "datasets"
    entries = []
    objects: set[str] = set()
    total_bytes = 0
    for source in sorted(legacy_foundation.glob("*.npy")):
        dataset_id = source.stem
        if not dataset_id or source.name != f"{dataset_id}.npy":
            raise FoundationMigrationError(f"invalid legacy dataset identity: {source}")
        size = source.stat().st_size
        digest = sha256_file(source)
        object_path = objects_root / digest[:2] / f"{digest}.npy"
        dataset_path = datasets_root / source.name
        publish_copy(source, object_path, size, digest)
        publish_dataset_link(object_path, dataset_path, size, digest)
        verify_file(source, size, digest)
        verify_file(dataset_path, size, digest)
        entries.append(
            {
                "dataset_id": dataset_id,
                "object": str(object_path),
                "sha256": digest,
                "size_bytes": size,
            }
        )
        objects.add(digest)
        total_bytes += size
    if not entries:
        raise FoundationMigrationError("legacy foundation inventory contains no .npy files")

    legacy_doc = json.loads(legacy_standardization.read_text(encoding="utf-8"))
    foundation = legacy_doc.get("foundation")
    if not isinstance(foundation, dict):
        raise FoundationMigrationError("legacy standardization lacks foundation metadata")
    projection_source = Path(str(foundation.get("projection_file", "")))
    if not projection_source.is_file():
        raise FoundationMigrationError(
            f"legacy foundation projection is absent: {projection_source}"
        )
    projection_size = projection_source.stat().st_size
    projection_sha256 = sha256_file(projection_source)
    projection_path = (
        destination_root
        / "projection"
        / "sha256"
        / f"{projection_sha256}.foundation_projection.npz"
    )
    publish_copy(
        projection_source, projection_path, projection_size, projection_sha256
    )

    standardization_doc = dict(legacy_doc)
    standardization_doc["foundation"] = {
        **foundation,
        "cache_dir": str(datasets_root),
        "cache_sha256": {row["dataset_id"]: row["sha256"] for row in entries},
        "inventory_schema": INVENTORY_SCHEMA,
        "projection_file": str(projection_path),
        "projection_sha256": projection_sha256,
    }
    standardization_payload = canonical_json(standardization_doc)
    standardization_sha256 = hashlib.sha256(standardization_payload).hexdigest()
    standardization_path = (
        destination_root
        / "standardization"
        / "sha256"
        / f"{standardization_sha256}.json"
    )
    publish_bytes(standardization_payload, standardization_path)

    inventory_doc = {
        "schema": INVENTORY_SCHEMA,
        "immutable": True,
        "dataset_count": len(entries),
        "object_count": len(objects),
        "total_dataset_bytes": total_bytes,
        "objects_root": str(objects_root),
        "dataset_projection_root": str(datasets_root),
        "projection": {
            "path": str(projection_path),
            "sha256": projection_sha256,
            "size_bytes": projection_size,
        },
        "standardization": {
            "path": str(standardization_path),
            "sha256": standardization_sha256,
            "size_bytes": len(standardization_payload),
        },
        "legacy_sources": {
            "foundation_root": str(legacy_foundation),
            "standardization_path": str(legacy_standardization),
            "standardization_sha256": sha256_file(legacy_standardization),
            "projection_path": str(projection_source),
            "projection_sha256": projection_sha256,
        },
        "entries": entries,
    }
    inventory_payload = canonical_json(inventory_doc)
    inventory_sha256 = hashlib.sha256(inventory_payload).hexdigest()
    inventory_path = (
        destination_root / "inventory" / "sha256" / f"{inventory_sha256}.json"
    )
    publish_bytes(inventory_payload, inventory_path)

    # A second complete verification catches source mutation during publication
    # and proves the consumer projection is byte-identical to every legacy file.
    for source, row in zip(sorted(legacy_foundation.glob("*.npy")), entries):
        verify_file(source, int(row["size_bytes"]), str(row["sha256"]))
        verify_file(
            datasets_root / f"{row['dataset_id']}.npy",
            int(row["size_bytes"]),
            str(row["sha256"]),
        )
    verify_file(projection_source, projection_size, projection_sha256)
    verify_file(projection_path, projection_size, projection_sha256)

    return {
        "schema": "JopeFoundationMigrationReceipt.v1",
        "dataset_count": len(entries),
        "object_count": len(objects),
        "total_dataset_bytes": total_bytes,
        "inventory": str(inventory_path),
        "inventory_sha256": inventory_sha256,
        "projection": str(projection_path),
        "projection_sha256": projection_sha256,
        "standardization": str(standardization_path),
        "standardization_sha256": standardization_sha256,
        "legacy_preserved": True,
        "source_destination_byte_identity": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-foundation", type=Path, default=LEGACY_FOUNDATION)
    parser.add_argument(
        "--legacy-standardization", type=Path, default=LEGACY_STANDARDIZATION
    )
    parser.add_argument("--destination-root", type=Path, default=FOUNDATION_ROOT)
    args = parser.parse_args()
    receipt = migrate(
        args.legacy_foundation, args.legacy_standardization, args.destination_root
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
