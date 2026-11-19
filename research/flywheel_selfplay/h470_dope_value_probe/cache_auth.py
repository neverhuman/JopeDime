"""WP01: Content-addressed cache authenticator for DOPE & Stage-1A/1B features.

Computes a canonical SHA-256 manifest over:
  1. dope_weights_sha256
  2. stage1a_mode_and_bridge_sha256
  3. prep_recipe
  4. context_sha256
  5. fit_rows_sha256
  6. conditioning_level
  7. stage1b_sha256

Refuses any cache load whose sidecar manifest, recomputed manifest digest, query-row
count (must be strictly 0), or tensor/array payload bytes disagree with expected pins.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping

import numpy as np

CACHE_AUTH_SCHEMA = "jope.h470.cache-auth/v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_VALID_LEVELS = frozenset({"high", "ultra", "sketch", "nolm_sketch", "packet_high"})


class CacheAuthenticationError(RuntimeError):
    """Raised when a cached representation fails WP01 provenance or digest checks."""


def sha256_bytes(payload: bytes | bytearray | memoryview) -> str:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise CacheAuthenticationError("sha256_bytes requires a bytes-like buffer")
    hasher = hashlib.sha256()
    hasher.update(memoryview(payload))
    return hasher.hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def array_le_sha256(arr: np.ndarray) -> str:
    """Deterministic little-endian IEEE-754 / integer byte digest of a NumPy array."""
    a = np.ascontiguousarray(arr)
    if a.dtype.kind in ("f", "i", "u") and a.dtype.byteorder not in ("<", "|"):
        a = a.astype(a.dtype.newbyteorder("<"), copy=False)
    if a.dtype.kind == "f" and not np.isfinite(a).all():
        raise CacheAuthenticationError("non-finite values refused by cache authenticator")
    h = hashlib.sha256()
    h.update(f"shape={a.shape}|dtype={a.dtype.str}|".encode("utf-8"))
    h.update(a.tobytes(order="C"))
    return h.hexdigest()


def compute_context_inventory_sha256(records: list[dict]) -> str:
    """Order-preserving digest over (context_sha256, family_id, dl_split, train_only_jtt1_sha256)."""
    if not records:
        raise CacheAuthenticationError("empty context inventory")
    items = [
        {
            "cid": str(r["context_sha256"]),
            "family_id": str(r["family_id"]),
            "dl_split": str(r["dl_split"]),
            "jtt1_sha256": str(r["train_only_jtt1_sha256"]),
        }
        for r in records
    ]
    return sha256_bytes(canonical_json_bytes(items))


def compute_fit_rows_sha256(
    ordered_cids: list[str],
    fit_tables: Mapping[str, tuple[np.ndarray, np.ndarray]],
    query_row_counts: Mapping[str, int] | None = None,
) -> str:
    """Digest over FIT-only (X_fit, y_fit) arrays, refusing any nonzero query-row count."""
    if not ordered_cids:
        raise CacheAuthenticationError("empty ordered_cids for fit_rows_sha256")
    h = hashlib.sha256()
    for cid in ordered_cids:
        if query_row_counts is not None and int(query_row_counts.get(cid, 0)) != 0:
            raise CacheAuthenticationError(f"query rows present for context {cid}: refused")
        if cid not in fit_tables:
            raise CacheAuthenticationError(f"missing fit table for context {cid}")
        x_fit, y_fit = fit_tables[cid]
        x_arr = np.ascontiguousarray(x_fit, dtype="<f8")
        y_arr = np.ascontiguousarray(y_fit, dtype="<f8").reshape(-1)
        if x_arr.ndim != 2 or x_arr.shape[0] != y_arr.shape[0] or x_arr.shape[0] < 2:
            raise CacheAuthenticationError(f"malformed fit table shape for context {cid}")
        h.update(f"{cid}:{x_arr.shape[0]}x{x_arr.shape[1]}:".encode("utf-8"))
        h.update(hashlib.sha256(x_arr.tobytes()).digest())
        h.update(hashlib.sha256(y_arr.tobytes()).digest())
    return h.hexdigest()


@dataclass(frozen=True)
class CacheAuthManifest:
    dope_weights_sha256: str
    stage1a_mode_and_bridge_sha256: str
    prep_recipe: str
    context_sha256: str
    fit_rows_sha256: str
    conditioning_level: str
    stage1b_sha256: str

    def validate(self) -> None:
        for field_name in (
            "dope_weights_sha256",
            "stage1a_mode_and_bridge_sha256",
            "context_sha256",
            "fit_rows_sha256",
            "stage1b_sha256",
        ):
            val = getattr(self, field_name)
            if not isinstance(val, str) or not _HEX64.match(val):
                raise CacheAuthenticationError(
                    f"{field_name} must be a lowercase 64-hex SHA-256 digest, got {val!r}"
                )
        if not isinstance(self.prep_recipe, str) or not self.prep_recipe.strip():
            raise CacheAuthenticationError("prep_recipe must be a nonempty string")
        if self.conditioning_level not in _VALID_LEVELS:
            raise CacheAuthenticationError(
                f"conditioning_level {self.conditioning_level!r} not in {sorted(_VALID_LEVELS)}"
            )

    def to_Identity_dict(self) -> dict[str, str]:
        self.validate()
        return {
            "schema": CACHE_AUTH_SCHEMA,
            **asdict(self),
        }

    def manifest_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_Identity_dict()))


def cache_paths_for_manifest(cache_dir: str | Path, tag: str, manifest: CacheAuthManifest) -> tuple[Path, Path]:
    digest = manifest.manifest_sha256()[:24]
    safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("_") or "cache"
    root = Path(cache_dir)
    npz_path = root / f"{safe_tag}_{digest}.npz"
    meta_path = root / f"{safe_tag}_{digest}.manifest.json"
    return npz_path, meta_path


def save_authenticated_npz(
    cache_dir: str | Path,
    tag: str,
    manifest: CacheAuthManifest,
    arrays: Mapping[str, np.ndarray],
) -> tuple[Path, Path]:
    """Atomically write an authenticated .npz cache and its SHA-256 sidecar manifest."""
    manifest.validate()
    if not arrays:
        raise CacheAuthenticationError("cannot save empty array dictionary")
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    npz_path, meta_path = cache_paths_for_manifest(root, tag, manifest)

    array_inventory: dict[str, dict[str, object]] = {}
    clean_arrays: dict[str, np.ndarray] = {}
    for key in sorted(arrays.keys()):
        arr = np.ascontiguousarray(arrays[key])
        digest = array_le_sha256(arr)
        clean_arrays[key] = arr
        array_inventory[key] = {
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "le_sha256": digest,
        }

    tmp_npz = npz_path.with_name(npz_path.name + ".tmp.npz")
    np.savez_compressed(tmp_npz, **clean_arrays)
    npz_file_sha = sha256_file(tmp_npz)
    tmp_npz.replace(npz_path)

    sidecar = {
        "schema": CACHE_AUTH_SCHEMA,
        "manifest": asdict(manifest),
        "manifest_sha256": manifest.manifest_sha256(),
        "npz_file": npz_path.name,
        "npz_file_sha256": npz_file_sha,
        "query_rows_used": 0,
        "arrays": array_inventory,
    }
    tmp_meta = meta_path.with_suffix(".json.tmp")
    tmp_meta.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_meta.replace(meta_path)
    return npz_path, meta_path


def load_authenticated_npz(
    cache_dir: str | Path,
    tag: str,
    expected_manifest: CacheAuthManifest,
    *,
    explicit_npz_path: str | Path | None = None,
    explicit_meta_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Load cached arrays only after full WP01 7-field manifest + payload verification."""
    expected_manifest.validate()
    if explicit_npz_path is not None and explicit_meta_path is not None:
        npz_path = Path(explicit_npz_path)
        meta_path = Path(explicit_meta_path)
    else:
        npz_path, meta_path = cache_paths_for_manifest(cache_dir, tag, expected_manifest)

    if not npz_path.is_file() or not meta_path.is_file():
        raise CacheAuthenticationError(f"authenticated cache miss for {tag} ({npz_path.name})")

    try:
        sidecar = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CacheAuthenticationError(f"corrupt cache manifest {meta_path}: {exc}") from None

    if sidecar.get("schema") != CACHE_AUTH_SCHEMA:
        raise CacheAuthenticationError(f"unsupported cache schema: {sidecar.get('schema')!r}")
    if sidecar.get("query_rows_used") != 0:
        raise CacheAuthenticationError("cache sidecar reports nonzero query_rows_used")

    recorded_fields = sidecar.get("manifest")
    if not isinstance(recorded_fields, dict):
        raise CacheAuthenticationError("missing manifest dictionary in sidecar")

    expected_fields = asdict(expected_manifest)
    mismatches = [
        k for k in sorted(set(expected_fields) | set(recorded_fields))
        if expected_fields.get(k) != recorded_fields.get(k)
    ]
    if mismatches:
        raise CacheAuthenticationError(
            f"cache manifest field mismatch on {mismatches}"
        )

    expected_sha = expected_manifest.manifest_sha256()
    if sidecar.get("manifest_sha256") != expected_sha:
        raise CacheAuthenticationError("sidecar manifest_sha256 does not match recomputed digest")

    actual_npz_sha = sha256_file(npz_path)
    if sidecar.get("npz_file_sha256") != actual_npz_sha:
        raise CacheAuthenticationError("npz file SHA-256 mismatch (tampered or truncated cache)")

    expected_arrays = sidecar.get("arrays")
    if not isinstance(expected_arrays, dict) or not expected_arrays:
        raise CacheAuthenticationError("missing array inventory in sidecar")

    loaded: dict[str, np.ndarray] = {}
    with np.load(npz_path, allow_pickle=False) as archive:
        archive_keys = set(archive.files)
        if archive_keys != set(expected_arrays.keys()):
            raise CacheAuthenticationError("cached archive keys differ from sidecar inventory")
        for key in sorted(archive_keys):
            arr = np.ascontiguousarray(archive[key])
            spec = expected_arrays[key]
            if list(arr.shape) != spec.get("shape") or str(arr.dtype) != spec.get("dtype"):
                raise CacheAuthenticationError(f"array {key} shape/dtype mismatch vs sidecar")
            if array_le_sha256(arr) != spec.get("le_sha256"):
                raise CacheAuthenticationError(f"array {key} content SHA-256 mismatch vs sidecar")
            loaded[key] = arr

    return loaded
