from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np

from jope.prime_config import load_config

TARGET_EMBED_DIM = 64
TARGET_VS_EMBED_DIM = 512
DEFAULT_NEURAL64 = str(load_config().target_encoder_model)


def read_train_targets(train_csv: str | Path) -> np.ndarray:
    rows = _read_csv_floats(train_csv)
    if not rows:
        raise ValueError(f"{train_csv}: no target rows")
    targets = [row[-1] for row in rows if row]
    if not targets:
        raise ValueError(f"{train_csv}: no target column")
    return np.asarray(targets, dtype=np.float32)


def encode_target_embedding(
    train_csv: str | Path,
    *,
    encoder: Literal["none", "summary64", "neural64", "train_target_vs"] = "none",
    model_path: str | Path | None = None,
    train_target_vs: str | Path | None = None,
) -> list[float] | None:
    if encoder == "none":
        return None
    if encoder == "summary64":
        return summary64_embedding(read_train_targets(train_csv)).tolist()
    if encoder == "train_target_vs":
        path = Path(train_target_vs) if train_target_vs is not None else Path(train_csv).with_name("train_target.vs")
        return target_vs_embedding(path).tolist()
    if encoder == "neural64":
        path = Path(model_path or DEFAULT_NEURAL64)
        if not path.exists():
            raise FileNotFoundError(f"neural64 target encoder checkpoint not found: {path}")
        raise NotImplementedError("neural64 target encoder checkpoint format is not wired in Jope yet")
    raise ValueError(f"unknown target encoder: {encoder}")


def summary64_embedding(targets: np.ndarray) -> np.ndarray:
    y = np.asarray(targets, dtype=np.float64)
    y = y[np.isfinite(y)]
    if y.size == 0:
        raise ValueError("target vector has no finite values")
    std = float(y.std())
    centered = (y - float(y.mean())) / (std if std > 1e-12 else 1.0)
    quantiles = np.quantile(y, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    hist, _ = np.histogram(y, bins=16)
    hist = hist.astype(np.float64) / max(1, y.size)
    values = [
        math.log1p(float(y.size)),
        float(y.mean()),
        std,
        float(y.min()),
        float(y.max()),
        float(np.mean(np.abs(centered))),
        float(np.mean(centered**3)),
        float(np.mean(centered**4)),
        *[float(q) for q in quantiles],
        *[float(v) for v in hist],
    ]
    if len(values) < TARGET_EMBED_DIM:
        values.extend([0.0] * (TARGET_EMBED_DIM - len(values)))
    return np.asarray(values[:TARGET_EMBED_DIM], dtype=np.float32)


def target_vs_embedding(path: str | Path, *, dim: int = TARGET_VS_EMBED_DIM) -> np.ndarray:
    matrix = _read_vs_floats(path)
    vector = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    if vector.size < dim:
        vector = np.pad(vector, (0, dim - vector.size), mode="constant")
    return vector[:dim].astype(np.float32)


def _read_vs_floats(path: str | Path) -> np.ndarray:
    path = Path(path)
    rows: list[list[float]] = []
    width: int | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",")]
            if width is None:
                width = len(parts)
            if len(parts) != width:
                raise ValueError(f"{path}:{line_no}: expected {width} floats, got {len(parts)}")
            try:
                row = [float(part) for part in parts]
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: non-float value") from exc
            if any(not math.isfinite(value) for value in row):
                raise ValueError(f"{path}:{line_no}: non-finite value")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no target .vs rows")
    return np.asarray(rows, dtype=np.float32)


def _read_csv_floats(path: str | Path) -> list[list[float]]:
    text = Path(path).read_text(encoding="utf-8")
    records: list[list[str]] = []
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if cells and not all(cell == "" for cell in cells):
            records.append(cells)
    if records and not all(_looks_numeric(cell) for cell in records[0]):
        records = records[1:]
    rows: list[list[float]] = []
    for cells in records:
        rows.append([_parse_cell(cell) for cell in cells])
    return rows


def _looks_numeric(cell: str) -> bool:
    text = cell.strip()
    if not text:
        return True
    if "_" in text:
        return False
    try:
        float(text)
    except ValueError:
        return False
    return True


def _parse_cell(cell: str) -> float:
    text = cell.strip()
    if not text or "_" in text:
        return 0.0
    try:
        value = float(text)
    except ValueError:
        return 0.0
    return value if math.isfinite(value) else 0.0
