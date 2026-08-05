#!/usr/bin/env python3
"""Freeze the regression+binary train-only context catalog for test-only work."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

from test_exact_common import (
    atomic_json,
    family_disjoint_split,
    prime_source_root,
    sha256_bytes,
    sha256_file,
)

REG_IMPORT_DIR = Path(
    "/home/ubuntu/ratchet-shadow/deployments/"
    "a99b446bfcf9438839dcee1cb8cd1a99ac6b33304dfadf39b0b6881ec70891d8-py128-fw1/"
    "inputs/encoding_import"
)
REG_DATA_ROOT = Path("/home/ubuntu/remote_super/quant/regression")
BINARY_ENCODING_ROOTS = (
    Path("/home/ubuntu/jope-flywheel/encodings/binary"),
    Path("/home/ubuntu/jope-flywheel/encodings/binary42"),
)
STANDARDIZATION = Path(
    "/home/ubuntu/jope-flywheel/encodings/foundation-v1/standardization/sha256/"
    "daac224cf7c6d13a1bf74fc3bb7defd36f33bc3f30db35d3e881c16ede976d1f.json"
)
OUTPUT = Path("/home/ubuntu/jope-flywheel/test_exact_gen0/catalog")
DATASET_DIM = 1536
TARGET_DIM = 512
STRIDE = (DATASET_DIM + TARGET_DIM) * 4


def source_family(dataset_dir: Path) -> str:
    meta_path = dataset_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    explicit = meta.get("source_family")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    if meta.get("source_real_data") or meta.get("source_path") or meta.get("source_file"):
        material = {
            "origin": "real",
            "source": meta.get("source_real_data")
            or meta.get("source_path")
            or meta.get("source_file"),
        }
    else:
        recipe = meta.get("generator_recipe") or {}
        material = {
            "origin": meta.get("origin") or "synthetic",
            "generator": meta.get("generator_name") or "unknown",
            "feature_family_counts": recipe.get("feature_family_counts") or {},
        }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    return f"family-{hashlib.sha256(encoded).hexdigest()[:16]}"


def load_regression() -> tuple[list[dict], bytes, dict]:
    imports = [Path(path) for path in glob.glob(str(REG_IMPORT_DIR / "*.import.json"))]
    bins = [Path(path) for path in glob.glob(str(REG_IMPORT_DIR / "*.f32.bin"))]
    if len(imports) != 1 or len(bins) != 1:
        raise RuntimeError("regression encoding import is not singular")
    meta = json.loads(imports[0].read_text())
    blob = bins[0].read_bytes()
    if sha256_bytes(blob) != meta["data_sha256"] or len(blob) != len(meta["entries"]) * STRIDE:
        raise RuntimeError("regression encoding blob failed its frozen hash/size")
    rows = []
    for entry in meta["entries"]:
        dataset = entry["dataset_hash"]
        root = REG_DATA_ROOT / dataset
        train, test = root / "train.csv", root / "test.csv"
        if not train.is_file() or not test.is_file():
            raise RuntimeError(f"regression dataset {dataset} is missing train/test")
        offset = int(entry["catalog_index"]) * STRIDE
        raw = blob[offset : offset + STRIDE]
        if sha256_bytes(raw) != entry["encoding_sha256"]:
            raise RuntimeError(f"regression encoding slice drift: {dataset}")
        rows.append(
            {
                "dataset_hash": dataset,
                "task": "regression",
                "metric": "r2",
                "train_csv": str(train),
                "test_csv": str(test),
                "train_sha256": sha256_file(train),
                "test_sha256": sha256_file(test),
                "source_family": source_family(root),
                "context_sha256": entry["encoding_sha256"],
                "source_catalog_index": int(entry["catalog_index"]),
            }
        )
    return rows, blob, meta


def load_binary(standardization: dict) -> tuple[list[dict], bytes]:
    sys.path.insert(0, str(prime_source_root()))
    from jope.propose import embedding_from_vs, target_summary_from_vs

    rows: list[dict] = []
    chunks: list[bytes] = []
    seen: set[str] = set()
    directories = sorted(
        child
        for root in BINARY_ENCODING_ROOTS
        for child in root.iterdir()
        if child.is_dir()
    )
    for root in directories:
        dataset = root.name
        if dataset in seen:
            raise RuntimeError(f"binary catalog repeats dataset {dataset}")
        seen.add(dataset)
        train, test = root / "train.csv", root / "test.csv"
        vs_path, target_path = (
            root / "hyperion3train128dim.vs",
            root / "train_target.vs",
        )
        for path in (train, test, vs_path, target_path):
            if not path.is_file():
                raise RuntimeError(f"binary catalog input is missing: {path}")
        context = [
            *embedding_from_vs(vs_path, standardization),
            *target_summary_from_vs(target_path),
        ]
        if len(context) != DATASET_DIM + TARGET_DIM:
            raise RuntimeError(f"binary context has wrong dimensions: {dataset}")
        raw = struct.pack(f"<{len(context)}f", *context)
        source_root = Path("/home/ubuntu/remote_super/quant/binary") / dataset
        rows.append(
            {
                "dataset_hash": dataset,
                "task": "binary",
                "metric": "auc",
                "train_csv": str(train),
                "test_csv": str(test),
                "train_sha256": sha256_file(train),
                "test_sha256": sha256_file(test),
                "source_family": source_family(source_root),
                "context_sha256": sha256_bytes(raw),
                "source_catalog_index": None,
            }
        )
        chunks.append(raw)
    if len(rows) != 542:
        raise RuntimeError(f"expected exactly 542 binary datasets, found {len(rows)}")
    return rows, b"".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    standardization = json.loads(STANDARDIZATION.read_text())
    regression, regression_blob, reg_meta = load_regression()
    binary, binary_blob = load_binary(standardization)
    rows = [*regression, *binary]
    families = {f"{row['task']}:{row['dataset_hash']}": row["source_family"] for row in rows}
    splits = family_disjoint_split(families, validation_fraction=0.1)
    blob = regression_blob + binary_blob
    for index, row in enumerate(rows):
        row["catalog_index"] = index
        row["split"] = splits[f"{row['task']}:{row['dataset_hash']}"]
        raw = blob[index * STRIDE : (index + 1) * STRIDE]
        if sha256_bytes(raw) != row["context_sha256"]:
            raise RuntimeError(f"combined context slice drift: {row['dataset_hash']}")
    manifest = {
        "schema": "JopeTestOnlyContextCatalog.v1",
        "immutable": True,
        "dataset_dimensions": DATASET_DIM,
        "target_dimensions": TARGET_DIM,
        "record_stride_bytes": STRIDE,
        "row_count": len(rows),
        "regression_count": len(regression),
        "binary_count": len(binary),
        "data_sha256": sha256_bytes(blob),
        "standardization_sha256": sha256_file(STANDARDIZATION),
        "regression_source_data_sha256": reg_meta["data_sha256"],
        "source_family_disjoint": not (
            {row["source_family"] for row in rows if row["split"] == "train"}
            & {row["source_family"] for row in rows if row["split"] == "validation"}
        ),
        "entries": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    blob_path = args.output / f"{manifest['data_sha256']}.f32.bin"
    temporary = blob_path.with_name(f".{blob_path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(blob)
    os.replace(temporary, blob_path)
    manifest_path = args.output / f"{sha256_bytes(json.dumps(manifest, sort_keys=True).encode())}.catalog.json"
    atomic_json(manifest_path, manifest)
    atomic_json(
        args.output / "current.json",
        {
            "schema": "JopeTestOnlyContextPointer.v1",
            "catalog": str(manifest_path),
            "catalog_sha256": sha256_file(manifest_path),
            "data": str(blob_path),
            "data_sha256": manifest["data_sha256"],
        },
    )
    print(
        json.dumps(
            {
                "catalog": str(manifest_path),
                "datasets": len(rows),
                "regression": len(regression),
                "binary": len(binary),
                "source_family_disjoint": manifest["source_family_disjoint"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
