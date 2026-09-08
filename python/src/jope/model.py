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
    parse_pipeline_actions,
    parse_param_token,
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
            losses.append(F.binary_cross_entropy_with_logits(out["transform_logits"], batch.transform_multi_hot.float()))
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
    complexity = torch.tensor([float(row.complexity or 0.0) for row in rows], dtype=torch.float32, device=device)
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


@dataclass
class ActiveV2Batch:
    embedding: torch.Tensor
    target_summary: torch.Tensor
    dataset_ids: list[str]
    transform_count: torch.Tensor
    slot_ops: torch.Tensor
    slot_append: torch.Tensor
    head: torch.Tensor
    param_labels: dict[str, torch.Tensor]
    value: torch.Tensor
    rank_percentile: torch.Tensor
    top_candidate: torch.Tensor
    complexity: torch.Tensor
    runtime: torch.Tensor
    family: torch.Tensor
    genome_features: torch.Tensor | None = None


def normalize_active_v2_vocab(vocab: dict[str, Any] | None) -> dict[str, Any]:
    vocab = vocab or {}
    if "transform_ops" in vocab and "head_ops" in vocab:
        return {
            "syntax": vocab.get("syntax") or "active_jain_regressor_v2",
            "max_transforms": int(vocab.get("max_transforms") or ACTIVE_MAX_TRANSFORMS),
            "transform_ops": list(vocab.get("transform_ops") or [NONE_TOKEN] + sorted(TRANSFORMS)),
            "head_ops": list(vocab.get("head_ops") or sorted(HEADS)),
            "transform_params": vocab.get("transform_params") or {},
            "head_params": vocab.get("head_params") or {},
        }
    raw_transforms = vocab.get("transforms") if isinstance(vocab.get("transforms"), dict) else {}
    raw_heads = vocab.get("heads") if isinstance(vocab.get("heads"), dict) else {}
    transform_ops = sorted(op for op in raw_transforms if op in TRANSFORMS) or sorted(TRANSFORMS)
    head_ops = sorted(op for op in raw_heads if op in HEADS) or sorted(HEADS)
    return {
        "syntax": vocab.get("syntax") or "active_jain_regressor_v2",
        "max_transforms": int(vocab.get("max_transforms") or ACTIVE_MAX_TRANSFORMS),
        "transform_ops": [NONE_TOKEN] + transform_ops,
        "head_ops": head_ops,
        "transform_params": _normalize_param_vocab(raw_transforms, transform_ops),
        "head_params": _normalize_param_vocab(raw_heads, head_ops),
    }


def active_v2_param_specs(vocab: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    transform_params = vocab.get("transform_params") or {}
    for slot in range(ACTIVE_MAX_TRANSFORMS):
        for op, params in sorted(transform_params.items()):
            for param, tokens in sorted(params.items()):
                if not tokens:
                    continue
                specs.append(
                    {
                        "key": _param_key("transform", slot, op, param),
                        "kind": "transform",
                        "slot": slot,
                        "op": op,
                        "param": param,
                        "tokens": list(tokens),
                    }
                )
    head_params = vocab.get("head_params") or {}
    for op, params in sorted(head_params.items()):
        for param, tokens in sorted(params.items()):
            if not tokens:
                continue
            specs.append(
                {
                    "key": _param_key("head", 0, op, param),
                    "kind": "head",
                    "slot": 0,
                    "op": op,
                    "param": param,
                    "tokens": list(tokens),
                }
            )
    return specs


# Extra scalar features appended to the per-op counts in a genome feature vector.
_GENOME_SCALAR_FEATURES = 2


def genome_feature_dim(vocab: dict[str, Any] | None) -> int:
    """Fixed width of the genome feature vector consumed by the reward head."""
    spec = normalize_active_v2_vocab(vocab)
    return (
        len(spec["transform_ops"])
        + len(spec["head_ops"])
        + len(FAMILY_LIST)
        + _GENOME_SCALAR_FEATURES
    )


def genome_features_from_actions(
    actions: dict[str, Any] | None,
    family_bitmap: dict[str, bool] | None,
    vocab: dict[str, Any] | None,
) -> list[float]:
    """Encode a genome (its parsed actions) as a fixed-size vector for the reward head.

    Uses per-op *counts* (not binary) so repeated transforms are represented, a head
    one-hot, the family bitmap, and two shape scalars (transform count, append fraction).
    Order matches ``genome_feature_dim``.
    """
    spec = normalize_active_v2_vocab(vocab)
    transform_ops = spec["transform_ops"]
    head_ops = spec["head_ops"]
    t_index = {op: i for i, op in enumerate(transform_ops)}
    h_index = {op: i for i, op in enumerate(head_ops)}
    actions = actions or {}
    transforms = list(actions.get("transforms") or [])
    tvec = [0.0] * len(transform_ops)
    append_count = 0
    for stage in transforms:
        if not isinstance(stage, dict):
            continue
        op = str(stage.get("op") or NONE_TOKEN)
        if op in t_index:
            tvec[t_index[op]] += 1.0
        if (stage.get("params") or {}).get("append"):
            append_count += 1
    scale = float(max(1, ACTIVE_MAX_TRANSFORMS))
    tvec = [value / scale for value in tvec]
    hvec = [0.0] * len(head_ops)
    head_op = str((actions.get("head") or {}).get("op") or "")
    if head_op in h_index:
        hvec[h_index[head_op]] = 1.0
    fam = [1.0 if (family_bitmap or {}).get(name) else 0.0 for name in FAMILY_LIST]
    scalars = [len(transforms) / scale, append_count / scale]
    return tvec + hvec + fam + scalars


class JopeActiveModelV2(nn.Module):
    def __init__(
        self,
        vocab: dict[str, Any] | None = None,
        *,
        dataset_dim: int = EMBEDDING_DIM,
        target_dim: int = TARGET_SUMMARY_DIM,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab = normalize_active_v2_vocab(vocab)
        self.dataset_dim = dataset_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.dropout = float(dropout)
        self.encoder = nn.Sequential(
            nn.Linear(dataset_dim + target_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.transform_count_head = nn.Linear(hidden_dim, ACTIVE_MAX_TRANSFORMS + 1)
        self.slot_op_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, len(self.vocab["transform_ops"])) for _ in range(ACTIVE_MAX_TRANSFORMS)]
        )
        self.slot_append_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(ACTIVE_MAX_TRANSFORMS)])
        self.head_head = nn.Linear(hidden_dim, len(self.vocab["head_ops"]))
        self.value_head = nn.Linear(hidden_dim, 1)
        self.rank_head = nn.Linear(hidden_dim, 1)
        self.top_head = nn.Linear(hidden_dim, 1)
        self.complexity_head = nn.Linear(hidden_dim, 1)
        self.runtime_head = nn.Linear(hidden_dim, 1)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)
        self.family_head = nn.Linear(hidden_dim, len(FAMILY_LIST))
        self.param_specs = active_v2_param_specs(self.vocab)
        self.param_heads = nn.ModuleDict(
            {spec["key"]: nn.Linear(hidden_dim, len(spec["tokens"])) for spec in self.param_specs}
        )
        # Genome-conditioned reward head: r(dataset_z, genome_features). Unlike the
        # dataset-only value/rank/top heads, this can rank genomes *within* a dataset,
        # which is what the relative-performance (Elo) objective requires.
        self.genome_feature_dim = genome_feature_dim(self.vocab)
        self.genome_encoder = nn.Sequential(
            nn.Linear(self.genome_feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(self.dropout),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(min(0.5, self.dropout * 2.0)),
            nn.Linear(hidden_dim, 1),
        )
        # Kendall & Gal homoscedastic task-uncertainty weights: total loss becomes
        # sum_i exp(-s_i) * L_i + s_i, learned per component so no single term (param_ce
        # today) can dominate a raw unweighted sum.
        self.loss_log_vars = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(())) for name in LOSS_COMPONENT_NAMES}
        )

    def forward(
        self,
        embedding: torch.Tensor,
        target_summary: torch.Tensor | None = None,
        genome_features: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if target_summary is None:
            target_summary = torch.zeros(
                (embedding.shape[0], self.target_dim),
                dtype=embedding.dtype,
                device=embedding.device,
            )
        x = torch.cat([embedding.float(), target_summary.float()], dim=-1)
        z = self.encoder(x)
        param_logits = {key: head(z) for key, head in self.param_heads.items()}
        out = {
            "z": z,
            "transform_count_logits": self.transform_count_head(z),
            "slot_op_logits": torch.stack([head(z) for head in self.slot_op_heads], dim=1),
            "slot_append_logits": torch.stack([head(z).squeeze(-1) for head in self.slot_append_heads], dim=1),
            "head_logits": self.head_head(z),
            "param_logits": param_logits,
            "value": self.value_head(z).squeeze(-1),
            "rank_percentile": torch.sigmoid(self.rank_head(z).squeeze(-1)),
            "top_candidate_logit": self.top_head(z).squeeze(-1),
            "complexity": F.softplus(self.complexity_head(z).squeeze(-1)),
            "runtime": F.softplus(self.runtime_head(z).squeeze(-1)),
            "uncertainty": F.softplus(self.uncertainty_head(z).squeeze(-1)),
            "family_logits": self.family_head(z),
        }
        if genome_features is not None:
            out["reward"] = self._reward_from_z(z, genome_features)
        return out

    def _reward_from_z(self, z: torch.Tensor, genome_features: torch.Tensor) -> torch.Tensor:
        g = self.genome_encoder(genome_features.float())
        return self.reward_head(torch.cat([z, g], dim=-1)).squeeze(-1)

    def score_genomes(
        self,
        embedding: torch.Tensor,
        target_summary: torch.Tensor | None,
        genome_features: torch.Tensor,
    ) -> torch.Tensor:
        """Reward for K genomes against a single dataset embedding.

        ``embedding``/``target_summary`` are a single row (shape ``(1, dim)`` or ``(dim,)``);
        ``genome_features`` is ``(K, genome_feature_dim)``. Returns a ``(K,)`` reward vector.
        """
        emb = embedding.float().reshape(1, -1)
        tgt = None if target_summary is None else target_summary.float().reshape(1, -1)
        if tgt is None:
            tgt = torch.zeros((1, self.target_dim), dtype=emb.dtype, device=emb.device)
        z = self.encoder(torch.cat([emb, tgt], dim=-1))
        z = z.expand(genome_features.shape[0], -1)
        return self._reward_from_z(z, genome_features)

    def loss_components(self, batch: ActiveV2Batch) -> dict[str, torch.Tensor]:
        out = self.forward(batch.embedding, batch.target_summary, batch.genome_features)
        return self.loss_components_from_output(out, batch)

    def loss_components_from_output(self, out: dict[str, Any], batch: ActiveV2Batch) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        losses["transform_count_ce"] = _balanced_cross_entropy(out["transform_count_logits"], batch.transform_count.long())
        flat_slot_logits = out["slot_op_logits"].reshape(-1, out["slot_op_logits"].shape[-1])
        losses["slot_op_ce"] = _balanced_cross_entropy(flat_slot_logits, batch.slot_ops.reshape(-1).long())
        append_mask = batch.slot_append >= 0
        if append_mask.any():
            losses["slot_append_bce"] = (
                F.binary_cross_entropy_with_logits(
                    out["slot_append_logits"][append_mask],
                    batch.slot_append[append_mask].float(),
                )
            )
        losses["head_ce"] = _balanced_cross_entropy(out["head_logits"], batch.head.long())
        param_losses: list[torch.Tensor] = []
        for key, labels in batch.param_labels.items():
            logits = out["param_logits"].get(key)
            if logits is not None and (labels >= 0).any():
                param_losses.append(_balanced_cross_entropy(logits, labels.long(), ignore_index=-100))
        if param_losses:
            losses["param_ce"] = torch.stack(param_losses).sum()
        value_mask = torch.isfinite(batch.value)
        if value_mask.any():
            losses["value_huber"] = F.huber_loss(out["value"][value_mask], batch.value[value_mask].float())
        rank_mask = torch.isfinite(batch.rank_percentile)
        if rank_mask.any():
            losses["rank_huber"] = F.huber_loss(out["rank_percentile"][rank_mask], batch.rank_percentile[rank_mask].float())
        top_mask = torch.isfinite(batch.top_candidate)
        if top_mask.any():
            losses["top_candidate_bce"] = (
                F.binary_cross_entropy_with_logits(
                    out["top_candidate_logit"][top_mask],
                    batch.top_candidate[top_mask].float(),
                )
            )
        complexity_mask = torch.isfinite(batch.complexity)
        if complexity_mask.any():
            losses["complexity_huber"] = F.huber_loss(
                torch.log1p(out["complexity"][complexity_mask]),
                torch.log1p(batch.complexity[complexity_mask].float()),
            )
        runtime_mask = torch.isfinite(batch.runtime)
        if runtime_mask.any():
            losses["runtime_huber"] = F.huber_loss(
                torch.log1p(out["runtime"][runtime_mask]),
                torch.log1p(batch.runtime[runtime_mask].float()),
            )
        losses["family_bce"] = F.binary_cross_entropy_with_logits(out["family_logits"], batch.family.float())
        ranknet = _ranknet_loss(out["value"], batch.value, batch.dataset_ids)
        if ranknet is not None:
            losses["ranknet"] = ranknet
        listwise = _listwise_top_loss(out["top_candidate_logit"], batch.top_candidate, batch.dataset_ids)
        if listwise is not None:
            losses["listwise_top_ce"] = listwise
        reward = out.get("reward")
        if reward is not None:
            reward_bt = _reward_bradley_terry_loss(reward, batch.value, batch.dataset_ids)
            if reward_bt is not None:
                losses["reward_bt"] = reward_bt
            reward_listwise = _reward_listwise_loss(reward, batch.value, batch.dataset_ids)
            if reward_listwise is not None:
                losses["reward_listwise"] = reward_listwise
        return losses

    def weighted_loss(
        self,
        components: dict[str, torch.Tensor],
        profile_weights: dict[str, float] | None = None,
    ) -> torch.Tensor:
        """Combine loss components with learned task-uncertainty weights (Kendall & Gal)."""
        if not components:
            return next(iter(self.parameters())).sum() * 0.0
        terms: list[torch.Tensor] = []
        for name, value in components.items():
            scale = 1.0 if profile_weights is None else float(profile_weights.get(name, 1.0))
            if scale == 0.0:
                continue
            log_var = self.loss_log_vars.get(name)
            scaled = value * scale
            if log_var is None:
                terms.append(scaled)
            else:
                terms.append(torch.exp(-log_var) * scaled + 0.5 * log_var)
        if not terms:
            return next(iter(self.parameters())).sum() * 0.0
        return torch.stack(terms).sum()

    def loss(self, batch: ActiveV2Batch) -> torch.Tensor:
        losses = self.loss_components(batch)
        if not losses:
            return batch.embedding.sum() * 0.0
        return self.weighted_loss(losses)


def rows_to_active_v2_batch(
    rows: list,
    vocab: dict[str, Any],
    device: str | torch.device = "cpu",
) -> ActiveV2Batch:
    spec_vocab = normalize_active_v2_vocab(vocab)
    n = len(rows)
    embedding = torch.tensor([row.embedding[:EMBEDDING_DIM] for row in rows], dtype=torch.float32, device=device)
    target_summary = torch.tensor([_row_target_summary(row) for row in rows], dtype=torch.float32, device=device)
    transform_ops = {op: i for i, op in enumerate(spec_vocab["transform_ops"])}
    head_ops = {op: i for i, op in enumerate(spec_vocab["head_ops"])}
    transform_count = torch.zeros(n, dtype=torch.long, device=device)
    slot_ops = torch.zeros((n, ACTIVE_MAX_TRANSFORMS), dtype=torch.long, device=device)
    slot_append = torch.full((n, ACTIVE_MAX_TRANSFORMS), -1.0, dtype=torch.float32, device=device)
    head = torch.zeros(n, dtype=torch.long, device=device)
    param_specs = active_v2_param_specs(spec_vocab)
    param_labels = {
        spec["key"]: torch.full((n,), -100, dtype=torch.long, device=device)
        for spec in param_specs
    }
    value = torch.full((n,), float("nan"), dtype=torch.float32, device=device)
    rank_percentile = torch.full((n,), float("nan"), dtype=torch.float32, device=device)
    top_candidate = torch.full((n,), float("nan"), dtype=torch.float32, device=device)
    complexity = torch.full((n,), float("nan"), dtype=torch.float32, device=device)
    runtime = torch.full((n,), float("nan"), dtype=torch.float32, device=device)
    family = torch.zeros((n, len(FAMILY_LIST)), dtype=torch.float32, device=device)
    genome_feature_rows: list[list[float]] = []
    specs_by_lookup = {(spec["kind"], spec["slot"], spec["op"], spec["param"]): spec for spec in param_specs}

    for i, row in enumerate(rows):
        actions = _row_actions(row)
        genome_feature_rows.append(genome_features_from_actions(actions, row.family_bitmap, spec_vocab))
        transforms = list(actions.get("transforms") or [])[:ACTIVE_MAX_TRANSFORMS]
        transform_count[i] = len(transforms)
        for slot, stage in enumerate(transforms):
            op = str(stage.get("op") or NONE_TOKEN)
            slot_ops[i, slot] = transform_ops.get(op, 0)
            params = stage.get("params") or {}
            slot_append[i, slot] = 1.0 if bool(params.get("append")) else 0.0
            for param, raw_value in params.items():
                lookup = ("transform", slot, op, param)
                spec = specs_by_lookup.get(lookup)
                if spec is None:
                    continue
                token = format_param_token(raw_value)
                if token in spec["tokens"]:
                    param_labels[spec["key"]][i] = spec["tokens"].index(token)
        head_stage = actions.get("head") or {}
        head_op = str(head_stage.get("op") or "ridge")
        head[i] = head_ops.get(head_op, 0)
        for param, raw_value in (head_stage.get("params") or {}).items():
            spec = specs_by_lookup.get(("head", 0, head_op, param))
            if spec is None:
                continue
            token = format_param_token(raw_value)
            if token in spec["tokens"]:
                param_labels[spec["key"]][i] = spec["tokens"].index(token)
        delta = (row.metadata or {}).get("oof_delta_to_incumbent")
        value[i] = _finite_or_nan(delta if delta is not None else row.score)
        rank_percentile[i] = _finite_or_nan((row.metadata or {}).get("rank_percentile"))
        if (row.metadata or {}).get("candidate_rank") is not None:
            top_candidate[i] = 1.0 if int(row.metadata["candidate_rank"]) == 1 else 0.0
        complexity[i] = _finite_or_nan(row.complexity)
        runtime[i] = _finite_or_nan(row.cost_ms)
        for j, name in enumerate(FAMILY_LIST):
            if row.family_bitmap.get(name):
                family[i, j] = 1.0

    genome_features = torch.tensor(genome_feature_rows, dtype=torch.float32, device=device)
    return ActiveV2Batch(
        embedding=embedding,
        target_summary=target_summary,
        dataset_ids=[str(row.dataset_id) for row in rows],
        transform_count=transform_count,
        slot_ops=slot_ops,
        slot_append=slot_append,
        head=head,
        param_labels=param_labels,
        value=value,
        rank_percentile=rank_percentile,
        top_candidate=top_candidate,
        complexity=complexity,
        runtime=runtime,
        family=family,
        genome_features=genome_features,
    )


def active_v2_labels_to_genome(batch: ActiveV2Batch, vocab: dict[str, Any], index: int = 0) -> str:
    spec_vocab = normalize_active_v2_vocab(vocab)
    transforms = []
    for slot in range(ACTIVE_MAX_TRANSFORMS):
        op_idx = int(batch.slot_ops[index, slot].detach().cpu())
        op = spec_vocab["transform_ops"][max(0, min(op_idx, len(spec_vocab["transform_ops"]) - 1))]
        if op == NONE_TOKEN:
            continue
        params: dict[str, Any] = {}
        if float(batch.slot_append[index, slot].detach().cpu()) > 0.5:
            params["append"] = True
        _labels_to_params(params, batch, spec_vocab, kind="transform", slot=slot, op=op, index=index)
        transforms.append({"op": op, "params": params})
    head_idx = int(batch.head[index].detach().cpu())
    head_op = spec_vocab["head_ops"][max(0, min(head_idx, len(spec_vocab["head_ops"]) - 1))]
    head_params: dict[str, Any] = {}
    _labels_to_params(head_params, batch, spec_vocab, kind="head", slot=0, op=head_op, index=index)
    return format_pipeline_actions(
        {
            "transform_count": len(transforms),
            "transforms": transforms,
            "head": {"op": head_op, "params": head_params},
        }
    )


def decode_active_v2_output(
    output: dict[str, Any],
    vocab: dict[str, Any],
    *,
    row: int = 0,
    temperature: float = 0.0,
    seed: int = 0,
    head_index: int | None = None,
    count_index: int | None = None,
) -> tuple[str, dict[str, Any]]:
    spec_vocab = normalize_active_v2_vocab(vocab)
    generator = torch.Generator(device=output["head_logits"].device)
    generator.manual_seed(seed)
    count_logits = output["transform_count_logits"][row]
    count = count_index if count_index is not None else _select_index(count_logits, temperature, generator)
    count = max(0, min(int(count), ACTIVE_MAX_TRANSFORMS))
    head_logits = output["head_logits"][row]
    head_idx = head_index if head_index is not None else _select_index(head_logits, temperature, generator)
    head_op = spec_vocab["head_ops"][max(0, min(int(head_idx), len(spec_vocab["head_ops"]) - 1))]
    transforms = []
    for slot in range(ACTIVE_MAX_TRANSFORMS):
        if slot >= count:
            continue
        op_idx = _select_index(output["slot_op_logits"][row, slot], temperature, generator)
        op = spec_vocab["transform_ops"][max(0, min(int(op_idx), len(spec_vocab["transform_ops"]) - 1))]
        if op == NONE_TOKEN:
            continue
        params: dict[str, Any] = {}
        if float(output["slot_append_logits"][row, slot].detach().cpu()) > 0.0:
            params["append"] = True
        _decode_params(params, output, spec_vocab, kind="transform", slot=slot, op=op, row=row, temperature=temperature, generator=generator)
        transforms.append({"op": op, "params": params})
    head_params: dict[str, Any] = {}
    _decode_params(head_params, output, spec_vocab, kind="head", slot=0, op=head_op, row=row, temperature=temperature, generator=generator)
    actions = {"transform_count": len(transforms), "transforms": transforms, "head": {"op": head_op, "params": head_params}}
    return format_pipeline_actions(actions), actions


def active_v2_validation_metrics(
    model: JopeActiveModelV2,
    rows: list,
    vocab: dict[str, Any],
    *,
    batch_size: int = 512,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    if not rows:
        return {}
    model.eval()
    exact = 0
    head_ok = 0
    transform_ok = 0
    total = 0
    values: list[float] = []
    reward_preds: list[float] = []
    value_preds: list[float] = []
    dataset_ids: list[str] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            batch = rows_to_active_v2_batch(batch_rows, vocab, device=device)
            out = model(batch.embedding, batch.target_summary, batch.genome_features)
            pred_heads = out["head_logits"].argmax(dim=-1)
            pred_slots = out["slot_op_logits"].argmax(dim=-1)
            for i, row in enumerate(batch_rows):
                genome, _ = decode_active_v2_output(out, vocab, row=i)
                if genome == (row.canonical_genome or row.genome):
                    exact += 1
                head_ok += int(pred_heads[i].item() == batch.head[i].item())
                transform_ok += int(torch.equal(pred_slots[i], batch.slot_ops[i]))
                total += 1
            mask = torch.isfinite(batch.value)
            if mask.any():
                keep = mask.detach().cpu().tolist()
                values.extend(batch.value[mask].detach().cpu().tolist())
                value_preds.extend(out["value"][mask].detach().cpu().tolist())
                reward = out.get("reward")
                reward_preds.extend((reward if reward is not None else out["value"])[mask].detach().cpu().tolist())
                dataset_ids.extend([ds for ds, k in zip(batch.dataset_ids, keep, strict=True) if k])
    return {
        "grammar_exact_match": exact / max(1, total),
        "head_accuracy": head_ok / max(1, total),
        "ordered_transform_accuracy": transform_ok / max(1, total),
        "pairwise_win_accuracy": _pairwise_accuracy(reward_preds, values, dataset_ids),
        "top1_historical_regret": _top1_regret(reward_preds, values, dataset_ids),
        "value_mae": _mae(value_preds, values),
        "reward_value_correlation": _pearson(reward_preds, values),
        "value_correlation": _pearson(value_preds, values),
    }


def _normalize_param_vocab(raw: dict[str, Any], ops: list[str]) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    for op in ops:
        params = raw.get(op) if isinstance(raw.get(op), dict) else {}
        out[op] = {}
        for param, values in sorted(params.items()):
            if isinstance(values, (list, tuple, set)):
                tokens = sorted({str(value) for value in values})
            else:
                tokens = [str(values)]
            out[op][str(param)] = tokens
    return out


def _param_key(kind: str, slot: int, op: str, param: str) -> str:
    return _safe_key(f"{kind}_{slot}_{op}_{param}")


def _safe_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def _row_target_summary(row) -> list[float]:
    values = row.target_embedding[:TARGET_SUMMARY_DIM] if row.target_embedding is not None else []
    out = [float(value) for value in values]
    if len(out) < TARGET_SUMMARY_DIM:
        out.extend([0.0] * (TARGET_SUMMARY_DIM - len(out)))
    return out[:TARGET_SUMMARY_DIM]


def _row_actions(row) -> dict[str, Any]:
    actions = (row.metadata or {}).get("actions")
    if isinstance(actions, dict):
        return actions
    return parse_pipeline_actions(row.canonical_genome or row.genome)


def _finite_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _balanced_cross_entropy(logits: torch.Tensor, target: torch.Tensor, *, ignore_index: int = -100) -> torch.Tensor:
    target = target.long()
    valid = target != ignore_index
    if not valid.any():
        return logits.sum() * 0.0
    classes, counts = torch.unique(target[valid], return_counts=True)
    weights = torch.ones(logits.shape[-1], dtype=logits.dtype, device=logits.device)
    inv = counts.float().rsqrt()
    inv = inv / inv.mean().clamp_min(1e-6)
    weights[classes] = inv.to(logits.device)
    return F.cross_entropy(logits, target, weight=weights, ignore_index=ignore_index)


def _ranknet_loss(pred: torch.Tensor, label: torch.Tensor, dataset_ids: list[str]) -> torch.Tensor | None:
    pairs: list[torch.Tensor] = []
    grouped: dict[str, list[int]] = defaultdict(list)
    finite = torch.isfinite(label).detach().cpu().tolist()
    for idx, dataset_id in enumerate(dataset_ids):
        if finite[idx]:
            grouped[dataset_id].append(idx)
    for indices in grouped.values():
        for a_pos, a in enumerate(indices):
            for b in indices[a_pos + 1 :]:
                if label[a] == label[b]:
                    continue
                sign = 1.0 if label[a] > label[b] else -1.0
                pairs.append(F.softplus(-sign * (pred[a] - pred[b])))
                if len(pairs) >= 256:
                    break
            if len(pairs) >= 256:
                break
        if len(pairs) >= 256:
            break
    if not pairs:
        return None
    return torch.stack(pairs).mean()


def _listwise_top_loss(pred_top: torch.Tensor, label_top: torch.Tensor, dataset_ids: list[str]) -> torch.Tensor | None:
    losses: list[torch.Tensor] = []
    grouped: dict[str, list[int]] = defaultdict(list)
    finite = torch.isfinite(label_top).detach().cpu().tolist()
    for idx, dataset_id in enumerate(dataset_ids):
        if finite[idx]:
            grouped[dataset_id].append(idx)
    for indices in grouped.values():
        positives = [idx for idx in indices if label_top[idx] > 0.5]
        if not positives or len(indices) < 2:
            continue
        logits = pred_top[indices].unsqueeze(0)
        target = torch.tensor([indices.index(positives[0])], dtype=torch.long, device=pred_top.device)
        losses.append(F.cross_entropy(logits, target))
    if not losses:
        return None
    return torch.stack(losses).mean()


def _group_by_dataset(label: torch.Tensor, dataset_ids: list[str]) -> dict[str, list[int]]:
    finite = torch.isfinite(label).detach().cpu().tolist()
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, dataset_id in enumerate(dataset_ids):
        if finite[idx]:
            grouped[dataset_id].append(idx)
    return grouped


def _reward_bradley_terry_loss(
    reward: torch.Tensor,
    value: torch.Tensor,
    dataset_ids: list[str],
    *,
    max_pairs: int = 4096,
) -> torch.Tensor | None:
    """Within-dataset, score-gap-weighted logistic ranking loss on the genome reward.

    P(a beats b) = sigmoid(r_a - r_b); this is the Bradley-Terry / Elo model. Pairs are
    weighted by the true score gap so decisive orderings matter more than ties. Within a
    dataset the incumbent is constant, so ``value`` (score - incumbent) preserves the
    score order and its gaps.
    """
    grouped = _group_by_dataset(value, dataset_ids)
    winners: list[int] = []
    losers: list[int] = []
    gaps: list[float] = []
    values = value.detach().cpu().tolist()
    for indices in grouped.values():
        for pos, a in enumerate(indices):
            for b in indices[pos + 1 :]:
                va, vb = values[a], values[b]
                if va == vb:
                    continue
                if va > vb:
                    winners.append(a)
                    losers.append(b)
                else:
                    winners.append(b)
                    losers.append(a)
                gaps.append(abs(va - vb))
    if not winners:
        return None
    win_t = torch.tensor(winners, device=reward.device)
    lose_t = torch.tensor(losers, device=reward.device)
    weights = torch.tensor(gaps, dtype=reward.dtype, device=reward.device)
    if win_t.numel() > max_pairs:
        sel = torch.linspace(0, win_t.numel() - 1, max_pairs, device=reward.device).round().long()
        win_t, lose_t, weights = win_t[sel], lose_t[sel], weights[sel]
    weights = weights / weights.mean().clamp_min(1e-6)
    diff = reward[win_t] - reward[lose_t]
    return (F.softplus(-diff) * weights).mean()


def _reward_listwise_loss(
    reward: torch.Tensor,
    value: torch.Tensor,
    dataset_ids: list[str],
) -> torch.Tensor | None:
    """ListNet-style listwise loss: match softmax(reward) to softmax of centered scores."""
    grouped = _group_by_dataset(value, dataset_ids)
    losses: list[torch.Tensor] = []
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        idx_t = torch.tensor(indices, device=reward.device)
        centered = value[idx_t].float()
        centered = centered - centered.mean()
        target = torch.softmax(centered, dim=0)
        logp = torch.log_softmax(reward[idx_t], dim=0)
        losses.append(-(target * logp).sum())
    if not losses:
        return None
    return torch.stack(losses).mean()


def _select_index(logits: torch.Tensor, temperature: float, generator: torch.Generator) -> int:
    if temperature <= 0.0:
        return int(torch.argmax(logits).item())
    probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator).item())


def _decode_params(
    params: dict[str, Any],
    output: dict[str, Any],
    vocab: dict[str, Any],
    *,
    kind: str,
    slot: int,
    op: str,
    row: int,
    temperature: float,
    generator: torch.Generator,
) -> None:
    param_space = (vocab.get("transform_params") if kind == "transform" else vocab.get("head_params")) or {}
    for param, tokens in sorted((param_space.get(op) or {}).items()):
        key = _param_key(kind, slot, op, param)
        logits = output["param_logits"].get(key)
        if logits is None or not tokens:
            continue
        idx = _select_index(logits[row], temperature, generator)
        params[param] = parse_param_token(tokens[max(0, min(idx, len(tokens) - 1))])


def _labels_to_params(
    params: dict[str, Any],
    batch: ActiveV2Batch,
    vocab: dict[str, Any],
    *,
    kind: str,
    slot: int,
    op: str,
    index: int,
) -> None:
    param_space = (vocab.get("transform_params") if kind == "transform" else vocab.get("head_params")) or {}
    for param, tokens in sorted((param_space.get(op) or {}).items()):
        key = _param_key(kind, slot, op, param)
        labels = batch.param_labels.get(key)
        if labels is None:
            continue
        label = int(labels[index].detach().cpu())
        if label < 0 or label >= len(tokens):
            continue
        params[param] = parse_param_token(tokens[label])


def _pairwise_accuracy(preds: list[float], values: list[float], dataset_ids: list[str]) -> float:
    wins = 0
    total = 0
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, dataset_id in enumerate(dataset_ids):
        grouped[dataset_id].append(idx)
    for indices in grouped.values():
        for a_pos, a in enumerate(indices):
            for b in indices[a_pos + 1 :]:
                if values[a] == values[b]:
                    continue
                total += 1
                wins += int((preds[a] > preds[b]) == (values[a] > values[b]))
    return wins / total if total else 0.0


def _top1_regret(preds: list[float], values: list[float], dataset_ids: list[str]) -> float:
    regrets: list[float] = []
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, dataset_id in enumerate(dataset_ids):
        grouped[dataset_id].append(idx)
    for indices in grouped.values():
        best = max(values[idx] for idx in indices)
        chosen = max(indices, key=lambda idx: preds[idx])
        regrets.append(best - values[chosen])
    return float(sum(regrets) / len(regrets)) if regrets else 0.0


def _mae(preds: list[float], values: list[float]) -> float:
    if not preds:
        return 0.0
    return float(sum(abs(pred - value) for pred, value in zip(preds, values, strict=True)) / len(preds))


def _pearson(preds: list[float], values: list[float]) -> float:
    if len(preds) < 2 or len(values) < 2:
        return 0.0
    pred_t = torch.tensor(preds, dtype=torch.float32)
    value_t = torch.tensor(values, dtype=torch.float32)
    pred_c = pred_t - pred_t.mean()
    value_c = value_t - value_t.mean()
    denom = pred_c.norm() * value_c.norm()
    if float(denom) <= 1e-12:
        return 0.0
    return float((pred_c * value_c).sum() / denom)
