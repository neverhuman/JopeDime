#!/usr/bin/env python3
"""Build the private three-row multiclass context-catalog extension.

The request is immutable and content addressed by the Rust controller.  This
worker deliberately writes only to its SCQ output directory: it never rotates
the shared catalog pointer or installs sidecars into the live flywheel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import tempfile
from pathlib import Path


DATASET_DIM = 1536
TARGET_DIM = 512
STRIDE = (DATASET_DIM + TARGET_DIM) * 4
DATASETS = ("81672f160e2b0312", "8347f9a4deae3b56", "0f7f4024a9b28e9c")
SCHEMA = "JopeMulticlassContextExtension.v1"


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise RuntimeError(f"{label} is not a lowercase SHA-256")
    return value


def verify_runtime_bundle(root: Path, expected_identity: str, expected_manifest: str) -> None:
    manifest_path = root / "bundle.json"
    if sha256_file(manifest_path) != expected_manifest:
        raise RuntimeError("pinned runtime-bundle manifest hash drift")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "RuntimeBundle.v3":
        raise RuntimeError("pinned runtime-bundle schema drift")
    if manifest.get("bundle_sha256") != expected_identity or root.name != expected_identity:
        raise RuntimeError("pinned runtime-bundle identity drift")
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("pinned runtime-bundle manifest lacks files")
    identity_rows = [
        {
            "target": row.get("target"),
            "sha256": row.get("sha256"),
            "bytes": row.get("bytes"),
            "python_role": row.get("python_role"),
        }
        for row in rows
    ]
    if sha256_bytes(canonical_bytes(identity_rows)) != expected_identity:
        raise RuntimeError("pinned runtime-bundle manifest identity drift")
    resolved_root = root.resolve()
    for row in rows:
        target = row.get("target")
        if not isinstance(target, str):
            raise RuntimeError("pinned runtime-bundle target is invalid")
        path = (root / target).resolve()
        if not path.is_relative_to(resolved_root) or not path.is_file():
            raise RuntimeError(f"pinned runtime-bundle target is unsafe or missing: {target}")
        if path.stat().st_size != row.get("bytes"):
            raise RuntimeError(f"pinned runtime-bundle target size drift: {target}")
        expected = require_sha256(row.get("sha256"), f"runtime target {target}")
        if sha256_file(path) != expected:
            raise RuntimeError(f"pinned runtime-bundle target hash drift: {target}")


def source_family(meta: dict, dataset: str) -> str:
    explicit = meta.get("source_family")
    if isinstance(explicit, str) and explicit:
        return explicit
    material = {
        "origin": meta.get("origin") or meta.get("synthetic_origin"),
        "generator": meta.get("generator_name"),
        "recipe": meta.get("generator_recipe"),
        "source": meta.get("source_path") or meta.get("source_file"),
        "dataset": dataset,
    }
    return "private-context:" + sha256_bytes(canonical_bytes(material))[:32]


def encode_one(
    dataset: str,
    source: Path,
    standardization: Path,
    temporary: Path,
    art_dir: Path,
    foundation_dir: Path,
) -> tuple[dict, bytes]:
    if not source.is_dir():
        raise RuntimeError(f"source directory is absent: {source}")
    train, test, meta_path = (source / name for name in ("train.csv", "test.csv", "meta.json"))
    for path in (train, test, meta_path):
        if not path.is_file():
            raise RuntimeError(f"{dataset}: missing {path}")

    # Use the pinned full JOPE encoder.  The helper imports the foundation
    # expert, widens its fixed signature timeout, and asks for CUDA explicitly;
    # the resulting sidecars are then projected with the pinned PCA artifact.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_exact_expansion as expansion

    cache = temporary / dataset
    cache.mkdir(parents=True)
    expansion.existing_hyperion_encoder(
        source, cache, "multiclass", art_dir=art_dir
    )
    from test_exact_expansion import encode_train_context

    context = encode_train_context(cache, "multiclass", standardization_path=standardization)
    if len(context) != STRIDE:
        raise RuntimeError(f"{dataset}: context has {len(context)} bytes, expected {STRIDE}")
    standardization_value = json.loads(standardization.read_text())
    from jope.propose import embedding_from_vs

    foundation_values = embedding_from_vs(
        cache / "hyperion3train128dim.vs", standardization_value
    )
    foundation_bytes = struct.pack(f"<{len(foundation_values)}f", *foundation_values)
    meta = json.loads(meta_path.read_text())
    context_sha256 = sha256_bytes(context)
    entry = {
        "task": "multiclass",
        "dataset_hash": dataset,
        "dataset_id": dataset,
        "metric": "macro_f1",
        "train_csv": str(train),
        "test_csv": str(test),
        "meta_json": str(meta_path),
        "train_sha256": sha256_file(train),
        "test_sha256": sha256_file(test),
        "meta_sha256": sha256_file(meta_path),
        "source_family": source_family(meta, dataset),
        "split": "train",
        "n_classes": meta.get("n_classes") or meta.get("num_classes"),
        "n_features": meta.get("n_features"),
        "context_protocol": "jope-dataset1536-target512-f32le-v1",
        "context_protocol_sha256": sha256_bytes(
            canonical_bytes({"protocol": "jope-dataset1536-target512-f32le-v1"})
        ),
        "encoder_protocol": "hyperion-train-context-v3-full-foundation-pca-target-summary",
        "encoder_sidecars": {
            name: sha256_file(cache / name)
            for name in ("hyperion3train128dim.vs", "train3.vs", "train_target.vs")
        },
        "foundation_vector_sha256": sha256_file(
            foundation_dir / f"{dataset}.npy"
        ),
        "foundation_projected_vector_sha256": sha256_bytes(foundation_bytes),
        "derived_standardization_sha256": sha256_file(standardization),
        "target_summary_sha256": sha256_file(cache / "train_target.vs"),
        "context_sha256": context_sha256,
        "catalog_index": None,
    }
    return entry, context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--standardization", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    if request.get("schema") != "JopeMulticlassContextEncodeRequest.v2":
        raise RuntimeError("unsupported context request schema")
    if request.get("test_adaptive") is not True or request.get("benchmark_valid") is not False:
        raise RuntimeError("context request disclosure flags are invalid")
    encoder = request.get("encoder") or {}
    if encoder.get("device") != args.device or encoder.get("train_only") is not True:
        raise RuntimeError("context request encoder binding drift")
    require_sha256(encoder.get("standardization_sha256"), "standardization_sha256")
    if sha256_file(args.standardization) != encoder["standardization_sha256"]:
        raise RuntimeError("standardization artifact hash drift")
    require_sha256(encoder.get("projection_sha256"), "projection_sha256")
    if sha256_file(args.projection) != encoder["projection_sha256"]:
        raise RuntimeError("foundation projection artifact hash drift")
    runtime_bundle = Path(str(encoder["runtime_bundle_pointer"]))
    runtime_python = runtime_bundle / "pytorch"
    bundled_external = runtime_python / "encoding" / "encode_external.py"
    bundled_manifest = runtime_bundle / "bundle.json"
    for path in (runtime_python, bundled_external, bundled_manifest):
        if not path.exists():
            raise RuntimeError(f"pinned runtime-bundle dependency is absent: {path}")
    require_sha256(encoder.get("runtime_bundle_sha256"), "runtime_bundle_sha256")
    runtime_manifest_sha256 = require_sha256(
        encoder.get("runtime_bundle_manifest_sha256"), "runtime_bundle_manifest_sha256"
    )
    verify_runtime_bundle(
        runtime_bundle,
        encoder["runtime_bundle_sha256"],
        runtime_manifest_sha256,
    )
    foundation_expert = Path("/home/ubuntu/remote_super/fable_god")
    os.environ["JOPE_RUNTIME_PYTHONPATH"] = (
        f"{runtime_python}:{foundation_expert}"
    )
    os.environ["JOPE_ENCODE_EXTERNAL_PATH"] = str(bundled_external)
    sys.path.insert(0, str(runtime_python))
    rows = request.get("datasets") or []
    if [row.get("dataset_hash") for row in rows] != list(DATASETS):
        raise RuntimeError("context request dataset order drift")
    for row in rows:
        source = Path(str(row["source_root"]))
        for field, name in (
            ("train_sha256", "train.csv"),
            ("test_sha256", "test.csv"),
            ("meta_sha256", "meta.json"),
        ):
            expected = require_sha256(row.get(field), field)
            path = source / name
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"{row['dataset_hash']}: {name} hash drift")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mc210-context-") as scratch:
        scratch_root = Path(scratch)
        foundation_dir = scratch_root / "foundation-cache"
        foundation_dir.mkdir()
        art_dir = scratch_root / "standardization-art"
        art_dir.mkdir()
        standardization_value = json.loads(args.standardization.read_text())
        foundation_value = dict(standardization_value.get("foundation") or {})
        foundation_value["cache_dir"] = str(foundation_dir)
        foundation_value["projection_file"] = str(args.projection)
        foundation_value["projection_sha256"] = encoder["projection_sha256"]
        standardization_value["foundation"] = foundation_value
        private_standardization = art_dir / "standardization.json"
        private_standardization.write_text(
            json.dumps(standardization_value, sort_keys=True, separators=(",", ":"))
        )
        entries = []
        contexts = []
        for row in rows:
            entry, context = encode_one(
                str(row["dataset_hash"]),
                Path(str(row["source_root"])),
                private_standardization,
                scratch_root,
                art_dir,
                foundation_dir,
            )
            # Bind the worker result to the request's source hashes.
            for field, path_name in (("train_sha256", "train.csv"), ("test_sha256", "test.csv"), ("meta_sha256", "meta.json")):
                if entry[field] != row[field]:
                    raise RuntimeError(f"{row['dataset_hash']}: {path_name} hash drift")
            entries.append(entry)
            contexts.append(context)

    blob = b"".join(contexts)
    data_path = args.output_dir / "contexts.f32.bin"
    data_path.write_bytes(blob)
    request_sha256 = sha256_file(args.request)
    receipt = {
        "schema": SCHEMA,
        "run_id": request["run_id"],
        "test_adaptive": True,
        "benchmark_valid": False,
        "request_sha256": request_sha256,
        "job_id": f"jope-mc210-context-{request_sha256[:16]}-qmc5-v3",
        "encoder": encoder,
        "entries": entries,
        "record_stride_bytes": STRIDE,
        "data_sha256": sha256_file(data_path),
        "data_bytes": len(blob),
        "datasets": len(entries),
    }
    (args.output_dir / "context-extension.json").write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    )
    print(json.dumps({"schema": SCHEMA, "datasets": len(entries), "data_sha256": receipt["data_sha256"]}))


if __name__ == "__main__":
    main()
