from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from .embeddings import EMBEDDING_DIM, EMBEDDING_VERSION, FrozenContract, sha256_file


def _finite_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _csv_bytes(matrix: np.ndarray) -> bytes:
    output = io.StringIO(newline="")
    for row in matrix:
        output.write(",".join(format(float(value), ".9g") for value in row))
        output.write("\n")
    return output.getvalue().encode("utf-8")


def _npy_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.save(output, array, allow_pickle=False)
    return output.getvalue()


def _write_new_or_equal(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError(f"immutable artifact collision: {path}") from None


def _sidecar(
    dataset: str,
    cohort: str,
    embedding: np.ndarray,
    contract: FrozenContract,
    sources: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "jope-prime.dime-embedding.v1",
        "version": EMBEDDING_VERSION,
        "dataset": dataset,
        "cohort": cohort,
        "shape": [EMBEDDING_DIM],
        "dtype": "float32",
        "embedding_sha256": hashlib.sha256(_npy_bytes(embedding)).hexdigest(),
        "frozen": contract.as_dict(),
        "sources": sources,
        "train_rows_only": True,
        "test_rows_inspected": 0,
    }


def _verify_cached_vector(path: Path, sidecar_path: Path, contract: FrozenContract) -> dict[str, Any]:
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("frozen") != contract.as_dict():
        raise ValueError(f"cached embedding frozen contract mismatch: {sidecar_path}")
    vector = np.load(path, mmap_mode="r", allow_pickle=False)
    if vector.shape != (EMBEDDING_DIM,) or vector.dtype != np.float32 or not np.isfinite(vector).all():
        raise ValueError(f"cached embedding has invalid shape, dtype, or values: {path}")
    if sha256_file(path) != sidecar.get("embedding_sha256"):
        raise ValueError(f"cached embedding SHA-256 mismatch: {path}")
    return sidecar
