from __future__ import annotations

import math
from pathlib import Path

import numpy as np

VS_DIM = 128
DATASET_SIGNATURE_DIM = VS_DIM * 4


def read_vs_matrix(path: str | Path, *, dim: int = VS_DIM) -> np.ndarray:
    """Read a CSV-like .vs matrix and require every row to have `dim` floats."""
    path = Path(path)
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != dim:
                raise ValueError(f"{path}:{line_no}: expected {dim} floats, got {len(parts)}")
            try:
                values = [float(part) for part in parts]
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: non-float value") from exc
            bad = [value for value in values if not math.isfinite(value)]
            if bad:
                raise ValueError(f"{path}:{line_no}: non-finite value")
            rows.append(values)
    if not rows:
        raise ValueError(f"{path}: no .vs rows")
    return np.asarray(rows, dtype=np.float32)


def dataset_vector_from_vs(path: str | Path, *, dim: int = VS_DIM) -> np.ndarray:
    """Collapse row-level Hyperion sidecars into one stable 128-d dataset vector."""
    matrix = read_vs_matrix(path, dim=dim)
    return matrix.mean(axis=0, dtype=np.float64).astype(np.float32)


def dataset_signature_from_vs(path: str | Path, *, dim: int = VS_DIM) -> np.ndarray:
    """Collapse a train-side .vs matrix into mean/std/p10/p90 features."""
    matrix = read_vs_matrix(path, dim=dim).astype(np.float64)
    signature = np.concatenate(
        [
            matrix.mean(axis=0),
            matrix.std(axis=0),
            np.quantile(matrix, 0.10, axis=0),
            np.quantile(matrix, 0.90, axis=0),
        ]
    )
    if signature.shape != (dim * 4,):
        raise ValueError(f"{path}: expected {dim * 4} signature values, got {signature.shape[0]}")
    return signature.astype(np.float32)
