#!/usr/bin/env python3
"""Verify sidecar integrity for an MCP-05 capture; this does not qualify A5."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CaptureError(ValueError):
    pass


def verify(manifest_path: Path, root: Path) -> dict[str, object]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureError(f"cannot read manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != "jopedime.named-client-transcripts/v1":
        raise CaptureError("unsupported transcript manifest schema")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise CaptureError("manifest has no sidecar inventory")

    verified = 0
    for name, expected in sorted(files.items()):
        if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
            raise CaptureError(f"unsafe sidecar name: {name!r}")
        if not isinstance(expected, dict):
            raise CaptureError(f"invalid sidecar entry: {name}")
        digest = expected.get("sha256")
        size = expected.get("bytes")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise CaptureError(f"invalid SHA-256 for {name}")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise CaptureError(f"invalid byte count for {name}")
        sidecar = root / name
        if sidecar.is_symlink() or not sidecar.is_file():
            raise CaptureError(f"sidecar is missing or not a regular file: {name}")
        actual_size = sidecar.stat().st_size
        if actual_size != size:
            raise CaptureError(f"byte count mismatch for {name}: expected {size}, got {actual_size}")
        actual_digest = hashlib.sha256(sidecar.read_bytes()).hexdigest()
        if actual_digest != digest:
            raise CaptureError(f"SHA-256 mismatch for {name}")
        verified += 1

    a5 = manifest.get("a5")
    label = a5.get("status") if isinstance(a5, dict) else None
    return {"integrity": "PASS", "sidecars_verified": verified,
            "a5_status": f"UNVERIFIED:{label}" if isinstance(label, str) else "UNVERIFIED:missing"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path,
                        help="directory containing sidecars (default: manifest directory)")
    args = parser.parse_args()
    root = args.root if args.root is not None else args.manifest.parent
    try:
        result = verify(args.manifest, root)
    except CaptureError as exc:
        print(json.dumps({"integrity": "FAIL", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
