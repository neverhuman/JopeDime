"""Foundation-embedding projection: raw 2048-d pooled hiddens -> 1024-d block.

PCA is fit on TRAIN datasets only (leakage rule), components + train-only
standardization stored in one artifact. The final v3.2 dataset vector is
    z = concat[hyperion_512_standardized, project(foundation_2048)]  -> 1536.

The artifact path is recorded in standardization.json under "foundation" so
`propose.embedding_from_vs` call sites need no signature change.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

PROJ_DIM = 1024
PROJECTION_FILE = "foundation_projection.npz"


def resolve_foundation_paths(standardization: dict, base_dir: Path) -> dict:
    """Resolve relocatable foundation paths against standardization.json."""
    resolved = dict(standardization)
    raw = standardization.get("foundation")
    if not isinstance(raw, dict):
        return resolved
    foundation = dict(raw)
    for key in ("cache_dir", "projection_file"):
        value = foundation.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"foundation {key} is missing")
        path = Path(value)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        foundation[key] = str(path)
    resolved["foundation"] = foundation
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_projection(art_dir: Path, cache_dir: Path) -> Path:
    """Fit PCA(2048->1024) on train-split foundation vectors; write artifact."""
    from sklearn.decomposition import PCA  # noqa: PLC0415

    art_dir, cache_dir = Path(art_dir), Path(cache_dir)
    manifest = json.load(open(art_dir / "split_manifest.json"))
    train = [h for h, v in manifest["datasets"].items() if v["split"] == "train"]
    X = np.stack([np.load(cache_dir / f"{h}.npy") for h in train])
    pca = PCA(n_components=PROJ_DIM, svd_solver="randomized", random_state=0)
    Z = pca.fit_transform(X)
    mu, sd = Z.mean(0), Z.std(0) + 1e-6
    out = art_dir / PROJECTION_FILE
    np.savez(
        out,
        components=pca.components_.astype("float32"),
        pca_mean=pca.mean_.astype("float32"),
        z_mean=mu.astype("float32"),
        z_scale=sd.astype("float32"),
        explained=pca.explained_variance_ratio_.astype("float32"),
    )
    # register in standardization.json so inference finds it without new plumbing
    std_path = art_dir / "standardization.json"
    std = json.load(open(std_path))
    std["foundation"] = {
        "cache_dir": str(cache_dir.resolve()),
        "projection_file": str(out.resolve()),
        "raw_dim": int(X.shape[1]),
        "proj_dim": PROJ_DIM,
    }
    json.dump(std, open(std_path, "w"))
    return out


class FoundationProjector:
    _cache: dict[str, FoundationProjector] = {}

    def __init__(self, spec: dict):
        projection = Path(spec["projection_file"])
        expected_projection = spec.get("projection_sha256")
        if expected_projection and _sha256_file(projection) != expected_projection:
            raise ValueError("foundation projection SHA-256 mismatch")
        art = np.load(projection)
        self.components = art["components"]
        self.pca_mean = art["pca_mean"]
        self.z_mean = art["z_mean"]
        self.z_scale = art["z_scale"]
        self.cache_dir = Path(spec["cache_dir"])
        self.cache_sha256 = dict(spec.get("cache_sha256") or {})

    @classmethod
    def from_spec(cls, spec: dict) -> FoundationProjector:
        key = spec["projection_file"]
        if key not in cls._cache:
            cls._cache[key] = cls(spec)
        return cls._cache[key]

    def block_for_hash(self, dataset_hash: str) -> np.ndarray:
        p = self.cache_dir / f"{dataset_hash}.npy"
        if not p.exists():
            raise FileNotFoundError(
                f"foundation encoding missing for {dataset_hash}: {p} "
                f"(run `jope encode-foundation` for this dataset)"
            )
        expected = self.cache_sha256.get(dataset_hash)
        if expected and _sha256_file(p) != expected:
            raise ValueError(f"foundation encoding SHA-256 mismatch for {dataset_hash}")
        raw = np.load(p)
        z = (raw - self.pca_mean) @ self.components.T
        return ((z - self.z_mean) / self.z_scale).astype("float32")
