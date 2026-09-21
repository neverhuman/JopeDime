"""Feature-matrix assembly with block tags.

Two structural choices, both forced by the arithmetic rather than by taste:

1. The dataset block is stored ONCE (2,078 x ~6,900) and joined by an integer row->dataset index,
   never materialized per row. Per-row materialization is ~18x redundant (median 20 candidates per
   dataset) and would cost 641 MB for what fits in 57 MB.

2. Every column carries a BLOCK TAG. The feature sweep must draw block-stratified windows -- a
   uniformly random 200-column window over this pool would be ~157 columns that are constant within
   a dataset and therefore cannot have a main effect, which corrupts any within-window importance
   normalization. Without the tags that sampling is impossible.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from .parse import ParsedGenome, parse

BLOCK_GENOME = "genome"  # varies within a dataset -- the only block with a main effect
BLOCK_TF = "trainfold"  # train-fold-fitted, also genome-varying
BLOCK_RETRIEVAL = "retrieval"
BLOCK_DATASET = "dataset"  # constant within a dataset -- zero main effect by construction
BLOCK_INTERACT = "interact"


class FeatureSpec:
    """Ordered column names + block tags. Hashable so a matrix can be tied to its definition."""

    def __init__(self) -> None:
        self.names: list[str] = []
        self.blocks: list[str] = []
        self._seen: set[str] = set()

    def add(self, names: list[str], block: str) -> None:
        for n in names:
            if n in self._seen:
                raise ValueError(f"duplicate feature name across modules: {n!r}")
            self._seen.add(n)
            self.names.append(n)
            self.blocks.append(block)

    def __len__(self) -> int:
        return len(self.names)

    def index_of_block(self, block: str) -> np.ndarray:
        return np.array([i for i, b in enumerate(self.blocks) if b == block], dtype=np.int64)

    def hash(self) -> str:
        payload = json.dumps({"names": self.names, "blocks": self.blocks}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_json(self) -> dict:
        return {"names": self.names, "blocks": self.blocks, "hash": self.hash()}


def build_genome_matrix(genomes: np.ndarray, engine: str, modules: list) -> tuple[np.ndarray, FeatureSpec]:
    """Pure-genome features. Parses each DISTINCT genome once and fans the result back to rows.

    featmath has 37,703 rows over 34,419 distinct genomes, so dedup saves only ~9% -- the real win
    is that `parse` is lru_cached and this stays a single pass with no multiprocessing. Measured
    end-to-end extraction for the whole corpus is ~60-90 s warm.
    """
    spec = FeatureSpec()
    for mod in modules:
        spec.add(list(mod.feature_names(engine)), BLOCK_GENOME)
    col = {n: i for i, n in enumerate(spec.names)}

    uniq, inverse = np.unique(genomes, return_inverse=True)
    compact = np.zeros((len(uniq), len(spec)), dtype=np.float32)
    for r, text in enumerate(uniq):
        pg = parse(str(text), engine)
        for mod in modules:
            for k, v in mod.extract(pg, engine).items():
                j = col.get(k)
                if j is not None and np.isfinite(v):
                    compact[r, j] = v
    return compact[inverse], spec


def build_tf_matrix(parses: list[ParsedGenome], fitted) -> tuple[np.ndarray, list[str]]:
    """Train-fold-fitted genome features (G3). `fitted` is an already-fit SynergyFeatures."""
    names = fitted.feature_names()
    col = {n: i for i, n in enumerate(names)}
    out = np.zeros((len(parses), len(names)), dtype=np.float32)
    for i, pg in enumerate(parses):
        for k, v in fitted.extract(pg).items():
            j = col.get(k)
            if j is not None and np.isfinite(v):
                out[i, j] = v
    return out, names


def build_retrieval_matrix(parses, datasets, retrievers, emb_by_space) -> tuple[np.ndarray, list[str]]:
    """G7 across every embedding space. Each retriever must already be fit on train folds only."""
    names: list[str] = []
    for r in retrievers:
        names.extend(r.feature_names())
    col = {n: i for i, n in enumerate(names)}
    out = np.zeros((len(parses), len(names)), dtype=np.float32)
    for i, (pg, d) in enumerate(zip(parses, datasets)):
        for r in retrievers:
            vec = emb_by_space[r.space].get(str(d))
            if vec is None:
                continue
            for k, v in r.extract(pg, str(d), vec).items():
                j = col.get(k)
                if j is not None and np.isfinite(v):
                    out[i, j] = v
    return out, names


def build_interactions(
    genome_mat: np.ndarray,
    genome_cols: list[int],
    ds_mat: np.ndarray,
    ds_cols: list[int],
    gnames: list[str],
    dnames: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Explicit genome x dataset crosses, HARD-BUDGETED.

    The full cross of ~1,950 genome columns against ~6,900 dataset dims is 13.5M products fit on
    2,078 effective groups -- hopeless. The caller must pass a small hand-picked genome subset and a
    compressed (PCA) dataset subset; 40 x 48 = 1,920 columns is the intended scale.
    """
    g = genome_mat[:, genome_cols]
    d = ds_mat[:, ds_cols]
    out = np.einsum("ni,nj->nij", g, d).reshape(len(g), -1).astype(np.float32)
    names = [f"ix_{gnames[i]}__{dnames[j]}" for i in range(len(genome_cols)) for j in range(len(ds_cols))]
    return out, names


def save_matrix(
    path: str,
    genome_block: np.ndarray,
    dataset_block: np.ndarray,
    row_dataset_idx: np.ndarray,
    spec: FeatureSpec,
    meta: dict,
) -> None:
    """Persist the two-table layout plus a manifest tying it to its definition."""
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "genome_block.npy"), genome_block.astype(np.float32))
    np.save(os.path.join(path, "dataset_block.npy"), dataset_block.astype(np.float32))
    np.save(os.path.join(path, "row_dataset_idx.npy"), row_dataset_idx.astype(np.int32))
    with open(os.path.join(path, "feature_spec.json"), "w") as f:
        json.dump(spec.to_json(), f, indent=1)
    with open(os.path.join(path, "manifest.json"), "w") as f:
        json.dump(
            {
                **meta,
                "feature_spec_hash": spec.hash(),
                "genome_block_shape": list(genome_block.shape),
                "dataset_block_shape": list(dataset_block.shape),
            },
            f,
            indent=1,
            sort_keys=True,
        )
