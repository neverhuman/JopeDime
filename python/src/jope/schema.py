from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

EMBEDDING_DIM = 512
# v3.2: dataset embeddings may be 512 (Hyperion signature) or 1536
# (concat[Hyperion 512, foundation-expert 1024]). Target embeddings stay 512.
DATASET_EMBEDDING_DIMS = (512, 1536)


def normalize_embedding(
    values: list[float] | tuple[float, ...] | None,
    dims: tuple[int, ...] = DATASET_EMBEDDING_DIMS,
) -> list[float]:
    if values is None:
        return [0.0] * dims[0]
    if len(values) in dims:
        return [float(v) for v in values]
    out = [float(v) for v in values[: dims[0]]]
    if len(out) < dims[0]:
        out.extend([0.0] * (dims[0] - len(out)))
    return out


@dataclass
class TrainingRow:
    dataset_id: str
    source: str
    embedding: list[float]
    genome: str
    target_embedding: list[float] | None = None
    canonical_genome: str | None = None
    valid: bool = True
    metric: str | None = None
    score: float | None = None
    rank: int | None = None
    elo: float | None = None
    cost_ms: float | None = None
    family_bitmap: dict[str, bool] = field(default_factory=dict)
    complexity: int | None = None
    runtime_class: str | None = None
    feedback_origin: str | None = None
    # Which engine grammar/evaluator this genome belongs to ("featmath" | "newmath").
    # Legacy rows (pre-engine) load as featmath via the metadata fallback in from_json.
    engine: str = "featmath"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.embedding = normalize_embedding(self.embedding)
        if self.target_embedding is not None:
            self.target_embedding = normalize_embedding(self.target_embedding, dims=(EMBEDDING_DIM,))

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> TrainingRow:
        return cls(
            dataset_id=str(data.get("dataset_id", "unknown")),
            source=str(data.get("source", "unknown")),
            embedding=normalize_embedding(data.get("embedding")),
            target_embedding=(
                normalize_embedding(data["target_embedding"], dims=(EMBEDDING_DIM,))
                if data.get("target_embedding") is not None
                else None
            ),
            genome=str(data.get("genome") or data.get("canonical_genome") or ""),
            canonical_genome=data.get("canonical_genome"),
            valid=bool(data.get("valid", True)),
            metric=data.get("metric"),
            score=_optional_float(data.get("score")),
            rank=_optional_int(data.get("rank")),
            elo=_optional_float(data.get("elo")),
            cost_ms=_optional_float(data.get("cost_ms")),
            family_bitmap=dict(data.get("family_bitmap") or {}),
            complexity=_optional_int(data.get("complexity")),
            runtime_class=data.get("runtime_class"),
            feedback_origin=data.get("feedback_origin"),
            engine=str(data.get("engine") or (data.get("metadata") or {}).get("engine") or "featmath"),
            metadata=dict(data.get("metadata") or {}),
        )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
