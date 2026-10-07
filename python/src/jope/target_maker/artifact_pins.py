"""Integrity checks for executable artifacts from trusted producers."""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path


def artifact_sha256(path: Path | str) -> str:
    """Record the producer's complete artifact identity in its report/manifest."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_joblib_pinned(path: Path | str, expected_sha256: object, *, context: str):
    """Deserialize only the immutable snapshot that matched a required producer pin.

    Pins detect replacement/corruption, not an untrusted producer or manifest.
    """
    path = Path(path)
    if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError(f"{context}: missing or invalid SHA-256 pin for {path}")
    artifact_bytes = path.read_bytes()
    actual_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{context}: SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual_sha256}"
        )
    import joblib

    # Do not reopen the path after verification: it could now refer to other bytes.
    return joblib.load(io.BytesIO(artifact_bytes))
