"""Verify transferred bytes using the sender's manifest before loading tools."""

import hashlib
import json
import sys
from pathlib import Path


def publish(incoming, final, expected):
    incoming, final = Path(incoming), Path(final)
    if incoming.parent != final.parent or final.name != expected["bundle_id"]:
        raise ValueError("Publication namespace mismatch")
    for path in (incoming, final, *final.parents):
        if path.is_symlink():
            raise ValueError("Publication symlink rejected")
    base = final if final.exists() else incoming
    metadata = json.loads((base / "bundle.json").read_text())
    if metadata != expected:
        raise ValueError("Transferred manifest differs from local manifest")
    actual = {}
    for path in base.rglob("*"):
        if path.is_symlink():
            raise ValueError("Transferred symlink rejected")
        if path.is_file() and path != base / "bundle.json":
            actual[str(path.relative_to(base))] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
                "executable": bool(path.stat().st_mode & 0o111),
            }
    if actual != expected["files"]:
        raise ValueError("Transferred file identity mismatch")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(base / "runtime" / metadata["release_id"] / "source/tools"))
    from kernel_bundle import verify_bundle

    if base == incoming:
        verify_bundle(incoming, require_name=False)
        if final.exists():
            raise ValueError("Concurrent publication: refusing overwrite")
        # Same-parent rename works for already read-only source directories.
        incoming.rename(final)
    verified = verify_bundle(final)
    return {
        "bundle_id": verified["bundle_id"],
        "release_id": verified["release_id"],
        "file_count": len(verified["files"]),
        "remote_path": str(final),
        "activation_allowed": False,
    }
