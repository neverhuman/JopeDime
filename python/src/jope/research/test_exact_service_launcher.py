#!/usr/bin/env python3
"""Exec a JOPE supervisor from the verified immutable active runtime bundle."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path("/home/ubuntu/jope-flywheel/test_exact_gen0")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    pointer_path = ROOT / "runtime-bundles" / "active.json"
    pointer = json.loads(pointer_path.read_text())
    manifest_path = Path(pointer["bundle"])
    if (
        sha256_file(manifest_path) != pointer["bundle_manifest_sha256"]
        or manifest_path.parent.name != pointer["bundle_sha256"]
    ):
        raise RuntimeError("active runtime bundle pointer failed verification")
    manifest = json.loads(manifest_path.read_text())
    if manifest["bundle_sha256"] != pointer["bundle_sha256"]:
        raise RuntimeError("active runtime bundle identity drift")
    bundle_root = manifest_path.parent
    freshness = len(sys.argv) > 1 and sys.argv[1] == "freshness"
    entrypoint = (
        "test_exact_freshness.py" if freshness else "test_exact_service.py"
    )
    service = bundle_root / "runtime" / "scripts" / entrypoint
    expected = {
        row["target"]: row["sha256"]
        for row in manifest["files"]
    }.get(f"runtime/scripts/{entrypoint}")
    if expected is None or sha256_file(service) != expected:
        raise RuntimeError("bundled service entrypoint failed verification")
    environment = dict(os.environ)
    environment["JOPE_LOADED_BUNDLE_SHA256"] = pointer["bundle_sha256"]
    environment["JOPE_PRIME_CONFIG"] = str(
        bundle_root / "jope-prime" / "prime.toml"
    )
    environment["PYTHONPATH"] = (
        f"{bundle_root / 'jope-prime' / 'src'}:"
        f"{bundle_root / 'runtime' / 'scripts'}"
    )
    os.chdir(bundle_root / "runtime")
    os.execve(
        sys.executable,
        (
            [sys.executable, str(service), "--root", str(ROOT)]
            if freshness
            else [sys.executable, str(service), *sys.argv[1:]]
        ),
        environment,
    )


if __name__ == "__main__":
    main()
