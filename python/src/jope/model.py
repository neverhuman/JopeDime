from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .data.active import (
    ACTIVE_MAX_TRANSFORMS,
    TARGET_SUMMARY_DIM,
    format_param_token,
    format_pipeline_actions,
    parse_param_token,
    parse_pipeline_actions,
)
from .grammar import HEADS, TRANSFORMS
from .schema import EMBEDDING_DIM

TRANSFORM_LIST = list(TRANSFORMS)
HEAD_LIST = list(HEADS)
FAMILY_LIST = [
    "scaling",
    "rank",
    "basis",
    "random_features",
    "interaction",
    "target_aware",
    "rowwise",
    "spectral",
    "kernel",
    "tree",
    "linear",
    "ensemble",
]
NONE_TOKEN = "NONE"
ACTIVE_V2_MODEL_VERSION = "active_v2"

# All loss components the active_v2 model may emit. Used to register task-uncertainty
# weights and to aggregate per-component metrics.
LOSS_COMPONENT_NAMES = [
    "transform_count_ce",
    "slot_op_ce",
    "slot_append_bce",
    "head_ce",
    "param_ce",
    "value_huber",
    "rank_huber",
    "top_candidate_bce",
    "complexity_huber",
    "runtime_huber",
    "family_bce",
    "ranknet",
    "listwise_top_ce",
    "reward_bt",
    "reward_listwise",
]


@dataclass
class JopeBatch:
    embedding: torch.Tensor
    valid: torch.Tensor | None = None
    value: torch.Tensor | None = None
    transform_count: torch.Tensor | None = None
    transform_multi_hot: torch.Tensor | None = None
    head: torch.Tensor | None = None
    family: torch.Tensor | None = None
    complexity: torch.Tensor | None = None
    runtime: torch.Tensor | None = None


class JopeOneShotModel(nn.Module):
    def __init__(self, embedding_dim: int = EMBEDDING_DIM, hidden_dim: int = 256) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.transform_count_head = nn.Linear(hidden_dim, 7)
        self.transform_head = nn.Linear(hidden_dim, len(TRANSFORM_LIST))
        self.head_head = nn.Linear(hidden_dim, len(HEAD_LIST))
        self.valid_head = nn.Linear(hidden_dim, 1)
        self.family_head = nn.Linear(hidden_dim, len(FAMILY_LIST))
        self.complexity_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.runtime_head = nn.Linear(hidden_dim, 1)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)

    def forward(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(embedding.float())
        return {
            "transform_count_logits": self.transform_count_head(z),
            "transform_logits": self.transform_head(z),
            "head_logits": self.head_head(z),
            "valid_logit": self.valid_head(z).squeeze(-1),
            "family_logits": self.family_head(z),
            "complexity": F.softplus(self.complexity_head(z).squeeze(-1)),
            "value": self.value_head(z).squeeze(-1),
            "runtime": F.softplus(self.runtime_head(z).squeeze(-1)),
            "uncertainty": F.softplus(self.uncertainty_head(z).squeeze(-1)),
        }

    def loss(self, batch: JopeBatch) -> torch.Tensor:
        out = self.forward(batch.embedding)
        losses: list[torch.Tensor] = []
        if batch.valid is not None:
            losses.append(F.binary_cross_entropy_with_logits(out["valid_logit"], batch.valid.float()))
        if batch.value is not None:
            losses.append(F.mse_loss(out["value"], batch.value.float()))
        if batch.transform_count is not None:
            losses.append(F.cross_entropy(out["transform_count_logits"], batch.transform_count.long()))
        if batch.transform_multi_hot is not None:
            losses.append(
                F.binary_cross_entropy_with_logits(out["transform_logits"], batch.transform_multi_hot.float())
            )
        if batch.head is not None:
            losses.append(F.cross_entropy(out["head_logits"], batch.head.long()))
        if batch.family is not None:
            losses.append(F.binary_cross_entropy_with_logits(out["family_logits"], batch.family.float()))
        if batch.complexity is not None:
            losses.append(F.mse_loss(out["complexity"], batch.complexity.float()))
        if batch.runtime is not None:
            losses.append(F.mse_loss(torch.log1p(out["runtime"]), torch.log1p(batch.runtime.float())))
        if not losses:
            return out["value"].sum() * 0.0
        return sum(losses)


def rows_to_batch(rows: list, device: str | torch.device = "cpu") -> JopeBatch:
    embedding = torch.tensor([row.embedding for row in rows], dtype=torch.float32, device=device)
    valid = torch.tensor([1.0 if row.valid else 0.0 for row in rows], dtype=torch.float32, device=device)
    value = torch.tensor([_row_value(row) for row in rows], dtype=torch.float32, device=device)
    complexity = torch.tensor(
        [float(row.complexity or 0.0) for row in rows], dtype=torch.float32, device=device
    )
    runtime = torch.tensor([float(row.cost_ms or 0.0) for row in rows], dtype=torch.float32, device=device)
    family = torch.zeros((len(rows), len(FAMILY_LIST)), dtype=torch.float32, device=device)
    transform_multi_hot = torch.zeros((len(rows), len(TRANSFORM_LIST)), dtype=torch.float32, device=device)
    transform_count_values: list[int] = []
    head_values: list[int] = []
    have_actions = True
    for i, row in enumerate(rows):
        for j, name in enumerate(FAMILY_LIST):
            if row.family_bitmap.get(name):
                family[i, j] = 1.0
        actions = (row.metadata or {}).get("actions")
        if not isinstance(actions, dict):
            have_actions = False
            continue
        transforms = actions.get("transforms") or []
        transform_count_values.append(max(0, min(6, int(actions.get("transform_count") or len(transforms)))))
        for stage in transforms:
            op = stage.get("op") if isinstance(stage, dict) else None
            if op in TRANSFORM_LIST:
                transform_multi_hot[i, TRANSFORM_LIST.index(op)] = 1.0
        head = actions.get("head") or {}
        head_op = head.get("op") if isinstance(head, dict) else None
        if head_op not in HEAD_LIST:
            have_actions = False
            continue
        head_values.append(HEAD_LIST.index(head_op))
    transform_count = None
    head = None
    if have_actions and len(transform_count_values) == len(rows) and len(head_values) == len(rows):
        transform_count = torch.tensor(transform_count_values, dtype=torch.long, device=device)
        head = torch.tensor(head_values, dtype=torch.long, device=device)
    else:
        transform_multi_hot = None
    return JopeBatch(
        embedding=embedding,
        valid=valid,
        value=value,
        transform_count=transform_count,
        transform_multi_hot=transform_multi_hot,
        head=head,
        family=family,
        complexity=complexity,
        runtime=runtime,
    )


def _row_value(row) -> float:
    delta = (row.metadata or {}).get("oof_delta_to_incumbent")
    if isinstance(delta, (int, float)):
        return float(delta)
    return float(row.score or 0.0)


from .active_v2 import (
    ActiveV2Batch,
    normalize_active_v2_vocab,
    active_v2_param_specs,
    _GENOME_SCALAR_FEATURES,
    genome_feature_dim,
    genome_features_from_actions,
    JopeActiveModelV2,
    rows_to_active_v2_batch,
    active_v2_labels_to_genome,
    decode_active_v2_output,
    active_v2_validation_metrics,
    _normalize_param_vocab,
    _param_key,
    _safe_key,
    _row_target_summary,
    _row_actions,
    _finite_or_nan,
    _balanced_cross_entropy,
    _ranknet_loss,
    _listwise_top_loss,
    _group_by_dataset,
    _reward_bradley_terry_loss,
    _reward_listwise_loss,
    _select_index,
    _decode_params,
    _labels_to_params,
    _pairwise_accuracy,
    _top1_regret,
    _mae,
    _pearson,
)
