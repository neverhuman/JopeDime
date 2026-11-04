"""Dataset conditioning embedders used by the Prime label adapter."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from .prime_label_to_corpus_identity import DUAL_EMBEDDING_DIM, TARGET_SUMMARY_DIM, _sha256_file


# --------------------------------------------------------------------------- #
# Dataset conditioning embedders                                              #
# --------------------------------------------------------------------------- #
class SignatureEmbedder:
    """Real dataset conditioning from ``.vs`` signature files (self_play's path).

    ``manifest`` maps ``dataset_hash -> {"vs_path", "train_target_vs", ...}`` (the
    ``split_manifest.json`` produced by the corpus builder). ``standardization`` is the
    matching ``standardization.json`` (carries the signature dim + foundation projector
    spec so ``embedding_from_vs`` yields the 1536-d dual embedding).
    """

    def __init__(
        self,
        manifest: str | Path,
        standardization: str | Path | None = None,
        *,
        signature_root: str | Path | None = None,
        require_hashes: bool = False,
    ) -> None:

        self._manifest_path = Path(manifest).resolve()
        data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        self._datasets: dict[str, dict] = data.get("datasets", data)
        self._signature_root = Path(signature_root).resolve() if signature_root else None
        self._require_hashes = require_hashes
        if standardization is not None:
            from .encoders.projection import resolve_foundation_paths

            standardization_path = Path(standardization).resolve()
            self._std = resolve_foundation_paths(
                json.loads(standardization_path.read_text(encoding="utf-8")),
                standardization_path.parent,
            )
        else:
            self._std = None
        self._cache: dict[str, tuple[list[float] | None, list[float] | None]] = {}

    def _path(self, dataset_hash: str, entry: dict, field: str) -> Path | None:
        raw = entry.get(field)
        if not raw:
            return None
        source = Path(str(raw))
        if self._signature_root is not None:
            candidate = (self._signature_root / dataset_hash / source.name).resolve()
            try:
                candidate.relative_to(self._signature_root)
            except ValueError as error:
                raise ValueError(f"staged signature path escapes root: {candidate}") from error
        elif source.is_absolute():
            candidate = source
        else:
            candidate = (self._manifest_path.parent / source).resolve()
        if not candidate.is_file():
            raise ValueError(f"staged signature is missing: {candidate}")
        expected = entry.get(f"{field}_sha256")
        if field == "vs_path":
            expected = expected or entry.get("vs_sha256")
        elif field == "train_target_vs":
            expected = expected or entry.get("train_target_sha256")
        if self._require_hashes and not expected:
            raise ValueError(f"{field} for dataset {dataset_hash} has no staged SHA-256")
        if expected:
            digest = _sha256_file(candidate)
            if digest != expected:
                raise ValueError(f"staged signature hash mismatch for {dataset_hash}/{candidate.name}")
        return candidate

    def covers(self, dataset_hash: str) -> bool:
        return dataset_hash in self._datasets

    def __call__(self, dataset_hash: str) -> tuple[list[float] | None, list[float] | None]:
        if dataset_hash in self._cache:
            return self._cache[dataset_hash]
        entry = self._datasets.get(dataset_hash)
        if entry is None or not entry.get("vs_path"):
            out: tuple[list[float] | None, list[float] | None] = (None, None)
        else:
            from .propose import embedding_from_vs, target_summary_from_vs

            vs_path = self._path(dataset_hash, entry, "vs_path")
            target_path = self._path(dataset_hash, entry, "train_target_vs")
            emb = embedding_from_vs(vs_path, self._std)
            tgt = target_summary_from_vs(target_path)
            out = (emb, tgt)
        self._cache[dataset_hash] = out
        return out


class PlaceholderEmbedder:
    """Deterministic per-dataset placeholder embedding (no real signatures available).

    REQUIREMENT for a production checkpoint: replace with ``SignatureEmbedder`` over the
    real per-dataset ``.vs`` Hyperion/foundation signatures. The placeholder is stable
    per ``dataset_hash`` (so the per-dataset listwise grouping and conditioning stay
    self-consistent) but carries no real dataset information — it proves the data path,
    not model quality.
    """

    def __init__(self, dim: int = DUAL_EMBEDDING_DIM, target_dim: int = TARGET_SUMMARY_DIM) -> None:
        self.dim = dim
        self.target_dim = target_dim

    def __call__(self, dataset_hash: str) -> tuple[list[float], list[float]]:
        seed = int(hashlib.sha256(dataset_hash.encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(seed)
        emb = [rng.gauss(0.0, 1.0) for _ in range(self.dim)]
        tgt = [rng.gauss(0.0, 1.0) for _ in range(self.target_dim)]
        return emb, tgt
