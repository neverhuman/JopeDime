"""Hash local H197 sources relative to the H197 checkout, not an imported parent."""

import hashlib
from pathlib import Path


def verify(root, pins):
    root = Path(root).resolve(strict=True)
    for relative, expected in pins.items():
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError(f"unsafe H197 source path: {relative}")
        path = root / rel
        if path.is_symlink():
            raise RuntimeError(f"symlinked H197 source: {relative}")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise RuntimeError(f"invalid H197 source path: {relative}")
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"H197 source pin mismatch: {relative}")
