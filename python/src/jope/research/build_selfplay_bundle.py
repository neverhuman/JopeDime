#!/usr/bin/env python3
"""Historical Python bundle builder retained for evidence/tests only.

Production must use ``target/release/jope_selfplay_bundle``. This module is
deliberately excluded from the Rust-first runtime bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from test_exact_common import atomic_json, sha256_file

SOURCE = Path("/home/ubuntu/jope-dime")
PRIME = Path("/home/ubuntu/jope-prime")
SCQ_PROJECTS = Path("/home/ubuntu/smart_cluster/etc/projects.toml")


def sources() -> list[tuple[Path, str]]:
    selected = [
        SOURCE / "Cargo.toml",
        SOURCE / "Cargo.lock",
        SOURCE / "src/bin/corpus_eval.rs",
        SOURCE / "target/release/corpus_eval",
        SOURCE / "manifests/test-exact-provenance-correction.v1.json",
        *sorted((SOURCE / "manifests" / "selfplay").glob("*")),
        *sorted((SOURCE / "mcp").glob("*")),
        SCQ_PROJECTS,
        *sorted((SOURCE / "scripts").glob("test_exact_*.py")),
        *sorted((SOURCE / "systemd").glob("jope-selfplay-*.service")),
    ]
    prime_sources = [
        PRIME / "prime.toml",
        *sorted((PRIME / "registry").rglob("*")),
        *sorted((PRIME / "src" / "jope").rglob("*.py")),
    ]
    result = []
    for path in [*selected, *prime_sources]:
        if not path.is_file():
            continue
        if path.is_relative_to(SOURCE):
            target = f"runtime/{path.relative_to(SOURCE)}"
        elif path.is_relative_to(PRIME):
            target = f"jope-prime/{path.relative_to(PRIME)}"
        else:
            target = f"external/{path.name}"
        result.append((path, target))
    return result


def source_bytes(source: Path) -> bytes:
    payload = source.read_bytes()
    if source == PRIME / "prime.toml":
        authority = b'root = "/home/ubuntu/jope-prime"'
        if payload.count(authority) != 1:
            raise RuntimeError("Prime bundle config root authority drift")
        payload = payload.replace(authority, b'root = "."', 1)
    return payload


def build(output_root: Path) -> dict:
    inventory = [
        {
            "source": str(source),
            "target": target,
            "sha256": hashlib.sha256(source_bytes(source)).hexdigest(),
            "bytes": len(source_bytes(source)),
        }
        for source, target in sources()
    ]
    identity = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    final = output_root / identity
    manifest_path = final / "bundle.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        for row in manifest["files"]:
            if sha256_file(final / row["target"]) != row["sha256"]:
                raise RuntimeError(f"existing bundle content drift: {row['target']}")
        return manifest
    temporary = output_root / f".{identity}.tmp-{os.getpid()}"
    output_root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(exist_ok=False)
    try:
        for row in inventory:
            destination = temporary / row["target"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = Path(row["source"])
            if source == PRIME / "prime.toml":
                destination.write_bytes(source_bytes(source))
            else:
                shutil.copy2(source, destination)
            if sha256_file(destination) != row["sha256"]:
                raise RuntimeError(f"bundle copy hash mismatch: {row['source']}")
        manifest = {
            "schema": "JopeSelfplayRuntimeBundle.v1",
            "bundle_sha256": identity,
            "test_adaptive": True,
            "benchmark_valid": False,
            "selection_bias": "test-exact-reused",
            "files": inventory,
        }
        atomic_json(temporary / "bundle.json", manifest)
        os.replace(temporary, final)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/home/ubuntu/jope-flywheel/test_exact_gen0/runtime-bundles"
        ),
    )
    args = parser.parse_args()
    print(json.dumps(build(args.output_root), sort_keys=True))


if __name__ == "__main__":
    main()
