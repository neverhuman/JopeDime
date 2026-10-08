"""Integrity checks for executable artifacts from trusted producers."""

from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path


def artifact_sha256(path: Path | str) -> str:
    """Record the producer's complete artifact identity in its report/manifest."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class _ProducerAuthority:
    """In-memory authority issued by trusted producer code, never from joblib."""

    pins: frozenset[str]


def _authorize_local_artifact(path: Path | str):
    """Trusted fitting code records its own output, not a consumer-supplied path."""
    pin = artifact_sha256(path)
    return pin, _ProducerAuthority(frozenset([pin]))


def read_trusted_producer_manifest(path: Path | str, trusted_sha256: object):
    """Require an independently trusted caller pin before using bundle pins.

    The caller must obtain this root from an approved producer record outside the
    possibly untrusted bundle. Computing it from that bundle is not admission.
    """
    path = Path(path)
    if not isinstance(trusted_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", trusted_sha256) is None:
        raise ValueError(f"Trusted producer manifest: missing or invalid SHA-256 pin for {path}")
    snapshot = path.read_bytes()
    if hashlib.sha256(snapshot).hexdigest() != trusted_sha256:
        raise ValueError(f"Trusted producer manifest: SHA-256 mismatch for {path}")
    manifest = json.loads(snapshot)
    if not isinstance(manifest, dict):
        raise ValueError("Trusted producer manifest must be a mapping")
    pins = manifest.get("artifact_sha256", {})
    if not isinstance(pins, dict):
        raise ValueError("Trusted producer manifest artifact SHA-256 pins must be a mapping")
    admitted = frozenset(pin for pin in pins.values()
                         if isinstance(pin, str) and re.fullmatch(r"[0-9a-f]{64}", pin))
    return manifest, _ProducerAuthority(admitted)


def load_joblib_pinned(path: Path | str, expected_sha256: object, *, context: str,
                       producer_authority: _ProducerAuthority | None = None):
    """Deserialize only the immutable snapshot that matched a required producer pin.

    The in-memory producer authority must come from a verified outer manifest or
    trusted local fitting code. Artifact/manifest self-reported hashes cannot
    grant it. Privileged Python code is outside this data-admission boundary.
    """
    path = Path(path)
    if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError(f"{context}: missing or invalid SHA-256 pin for {path}")
    if (not isinstance(producer_authority, _ProducerAuthority)
            or expected_sha256 not in producer_authority.pins):
        raise ValueError(f"{context}: artifact has no trusted producer authority for {path}")
    artifact_bytes = path.read_bytes()
    actual_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{context}: SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual_sha256}"
        )
    import joblib

    # Do not reopen the path after verification: it could now refer to other bytes.
    return joblib.load(io.BytesIO(artifact_bytes))
