from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .active_v2_losses import (
    _balanced_cross_entropy,
    _listwise_top_loss,
    _ranknet_loss,
    _reward_bradley_terry_loss,
    _reward_listwise_loss,
)
from .active_v2_vocab import (
    ActiveV2Batch,
    active_v2_param_specs,
    genome_feature_dim,
    normalize_active_v2_vocab,
)
from .data.active import ACTIVE_MAX_TRANSFORMS, TARGET_SUMMARY_DIM
from .model import FAMILY_LIST, LOSS_COMPONENT_NAMES
from .schema import EMBEDDING_DIM


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
        self.slot_append_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(ACTIVE_MAX_TRANSFORMS)]
        )
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
            "slot_append_logits": torch.stack(
                [head(z).squeeze(-1) for head in self.slot_append_heads], dim=1
            ),
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

    def loss_components_from_output(
        self, out: dict[str, Any], batch: ActiveV2Batch
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        losses["transform_count_ce"] = _balanced_cross_entropy(
            out["transform_count_logits"], batch.transform_count.long()
        )
        flat_slot_logits = out["slot_op_logits"].reshape(-1, out["slot_op_logits"].shape[-1])
        losses["slot_op_ce"] = _balanced_cross_entropy(flat_slot_logits, batch.slot_ops.reshape(-1).long())
        append_mask = batch.slot_append >= 0
        if append_mask.any():
            losses["slot_append_bce"] = F.binary_cross_entropy_with_logits(
                out["slot_append_logits"][append_mask],
                batch.slot_append[append_mask].float(),
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
            losses["rank_huber"] = F.huber_loss(
                out["rank_percentile"][rank_mask], batch.rank_percentile[rank_mask].float()
            )
        top_mask = torch.isfinite(batch.top_candidate)
        if top_mask.any():
            losses["top_candidate_bce"] = F.binary_cross_entropy_with_logits(
                out["top_candidate_logit"][top_mask],
                batch.top_candidate[top_mask].float(),
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
