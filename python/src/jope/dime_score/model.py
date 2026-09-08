from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

DATASET_DIM = 1536
D_MODEL = 256
N_LAYERS = 2
N_HEADS = 4
FFN_DIM = 1024


class DimeScoreModel(nn.Module):
    """Causal genome critic whose only dataset input is a 1536-d prefix."""

    def __init__(
        self,
        vocabulary_size: int,
        action_type_count: int,
        max_sequence_length: int,
        *,
        dataset_dim: int = DATASET_DIM,
        d_model: int = D_MODEL,
        n_layers: int = N_LAYERS,
        n_heads: int = N_HEADS,
        ffn_dim: int = FFN_DIM,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dataset_dim = dataset_dim
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ffn_dim = ffn_dim
        self.max_sequence_length = max_sequence_length
        self.dataset_adapter = nn.Sequential(
            nn.Linear(dataset_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.token_emb = nn.Embedding(vocabulary_size, d_model, padding_idx=0)
        self.action_type_emb = nn.Embedding(action_type_count, d_model)
        self.numeric_adapter = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_emb = nn.Embedding(max_sequence_length + 1, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.score_head = nn.Linear(d_model, 1)
        self.validity_head = nn.Linear(d_model, 1)

    def forward(
        self,
        dataset_embedding: torch.Tensor,
        token_ids: torch.Tensor,
        numeric_values: torch.Tensor,
        numeric_mask: torch.Tensor,
        action_type_ids: torch.Tensor,
        lengths: torch.Tensor,
        numeric_mean: torch.Tensor,
        numeric_scale: torch.Tensor,
        *,
        genome_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch, width = token_ids.shape
        if width > self.max_sequence_length:
            raise ValueError(
                f"sequence width {width} exceeds checkpoint grammar contract "
                f"{self.max_sequence_length}; truncation is forbidden"
            )
        if genome_only:
            prefix = torch.zeros(
                (batch, self.d_model),
                dtype=dataset_embedding.dtype,
                device=dataset_embedding.device,
            )
        else:
            prefix = self.dataset_adapter(dataset_embedding.float())
        signed_log = torch.sign(numeric_values) * torch.log1p(torch.abs(numeric_values))
        mean = numeric_mean[token_ids]
        scale = numeric_scale[token_ids].clamp_min(1e-8)
        normalized = ((signed_log - mean) / scale).unsqueeze(-1)
        numeric = self.numeric_adapter(normalized) * numeric_mask.unsqueeze(-1)
        tokens = self.token_emb(token_ids) + self.action_type_emb(action_type_ids) + numeric
        hidden = torch.cat((prefix.unsqueeze(1), tokens), dim=1)
        positions = torch.arange(width + 1, device=token_ids.device)
        hidden = hidden + self.pos_emb(positions).unsqueeze(0)
        causal = torch.triu(
            torch.ones(
                width + 1,
                width + 1,
                dtype=torch.bool,
                device=token_ids.device,
            ),
            diagonal=1,
        )
        token_positions = torch.arange(width, device=token_ids.device).unsqueeze(0)
        padding = token_positions >= lengths.unsqueeze(1)
        key_padding = torch.cat(
            (
                torch.zeros((batch, 1), dtype=torch.bool, device=token_ids.device),
                padding,
            ),
            dim=1,
        )
        hidden = self.decoder(hidden, mask=causal, src_key_padding_mask=key_padding)
        hidden = self.out_norm(hidden)
        # Prefix is position zero; EOS for a sequence of L tokens is at hidden position L.
        final = hidden[torch.arange(batch, device=hidden.device), lengths]
        score_logit = self.score_head(final).squeeze(-1)
        validity_logit = self.validity_head(final).squeeze(-1)
        return {
            "score_logit": score_logit,
            "score": torch.sigmoid(score_logit),
            "validity_logit": validity_logit,
            "validity": torch.sigmoid(validity_logit),
            "representation": final,
        }


def bradley_terry_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    dataset_indices: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    for dataset in torch.unique(dataset_indices):
        indexes = torch.nonzero(
            (dataset_indices == dataset) & valid_mask, as_tuple=False
        ).flatten()
        if len(indexes) < 2:
            continue
        left, right = torch.triu_indices(
            len(indexes), len(indexes), offset=1, device=logits.device
        )
        first, second = indexes[left], indexes[right]
        target_delta = targets[first] - targets[second]
        keep = target_delta != 0
        if not torch.any(keep):
            continue
        direction = torch.sign(target_delta[keep])
        logit_delta = logits[first[keep]] - logits[second[keep]]
        terms.append(F.softplus(-direction * logit_delta).mean())
    if not terms:
        return logits.sum() * 0.0
    return torch.stack(terms).mean()


def score_losses(
    outputs: dict[str, torch.Tensor],
    percentiles: torch.Tensor,
    validities: torch.Tensor,
    dataset_indices: torch.Tensor,
    *,
    validity_positive_weight: float,
    huber_weight: float = 1.0,
    ranking_weight: float = 0.2,
    validity_weight: float = 0.2,
) -> dict[str, torch.Tensor]:
    valid = torch.isfinite(percentiles)
    if torch.any(valid):
        huber = F.huber_loss(
            outputs["score"][valid],
            percentiles[valid],
            delta=0.1,
        )
    else:
        huber = outputs["score"].sum() * 0.0
    ranking = bradley_terry_loss(
        outputs["score_logit"], percentiles, dataset_indices, valid
    )
    positive_weight = torch.as_tensor(
        validity_positive_weight,
        dtype=outputs["validity_logit"].dtype,
        device=outputs["validity_logit"].device,
    )
    validity = F.binary_cross_entropy_with_logits(
        outputs["validity_logit"],
        validities,
        pos_weight=positive_weight,
    )
    total = huber_weight * huber + ranking_weight * ranking + validity_weight * validity
    return {
        "total": total,
        "huber": huber,
        "ranking": ranking,
        "validity_bce": validity,
    }


def partial_warm_start(
    model: DimeScoreModel,
    checkpoint: dict[str, Any],
    tokens: list[str],
) -> dict[str, Any]:
    """Load only exact-shape trunk/position tensors and exact-identity token rows."""
    source = checkpoint.get("model", checkpoint)
    if not isinstance(source, dict):
        raise ValueError("warm-start checkpoint has no model state")
    target = model.state_dict()
    loaded: list[str] = []
    for name, value in source.items():
        if not (
            name.startswith("decoder.")
            or name.startswith("out_norm.")
            or name == "pos_emb.weight"
        ):
            continue
        if name in target and target[name].shape == value.shape:
            target[name] = value.detach().to(dtype=target[name].dtype)
            loaded.append(name)

    source_vocab = checkpoint.get("action_vocab") or checkpoint.get("vocab")
    source_tokens = source_vocab.get("tokens") if isinstance(source_vocab, dict) else None
    token_rows = 0
    if (
        isinstance(source_tokens, list)
        and "token_emb.weight" in source
        and source["token_emb.weight"].ndim == 2
        and source["token_emb.weight"].shape[1] == target["token_emb.weight"].shape[1]
    ):
        source_index = {str(token): index for index, token in enumerate(source_tokens)}
        token_weight = target["token_emb.weight"].clone()
        for target_index, token in enumerate(tokens):
            source_index_value = source_index.get(token)
            if (
                source_index_value is not None
                and source_index_value < source["token_emb.weight"].shape[0]
            ):
                token_weight[target_index] = source["token_emb.weight"][
                    source_index_value
                ].to(dtype=token_weight.dtype)
                token_rows += 1
        target["token_emb.weight"] = token_weight
    model.load_state_dict(target)
    return {
        "loaded_tensors": sorted(loaded),
        "loaded_tensor_count": len(loaded),
        "token_rows": token_rows,
        "source_model_version": checkpoint.get("model_version"),
    }


class DimeBroadCritic(nn.Module):
    """Ordered structural critic for broad screening and exact labels.

    Four independent context prefixes prevent target, anchor, and fidelity
    inputs from being silently folded into the legacy 1536-D adapter.
    """

    context_tokens = 4

    def __init__(
        self,
        vocabulary_size: int,
        action_type_count: int,
        max_sequence_length: int,
        *,
        dataset_dim: int = DATASET_DIM,
        target_dim: int = 512,
        anchor_count: int = 4,
        fidelity_count: int = 2,
        distribution_bins: int = 11,
        d_model: int = D_MODEL,
        n_layers: int = N_LAYERS,
        n_heads: int = N_HEADS,
        ffn_dim: int = FFN_DIM,
        dropout: float = 0.1,
        anchor_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 <= anchor_dropout <= 1.0:
            raise ValueError("anchor_dropout must be in [0, 1]")
        self.dataset_dim = dataset_dim
        self.target_dim = target_dim
        self.anchor_count = anchor_count
        self.fidelity_count = fidelity_count
        self.distribution_bins = distribution_bins
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        self.anchor_dropout = anchor_dropout
        self.dataset_adapter = nn.Sequential(
            nn.Linear(dataset_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.target_adapter = nn.Sequential(
            nn.Linear(target_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        # Three values plus one observed bit per anchor.
        self.anchor_adapter = nn.Sequential(
            nn.Linear(anchor_count * 4, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.fidelity_emb = nn.Embedding(fidelity_count, d_model)
        self.token_emb = nn.Embedding(vocabulary_size, d_model, padding_idx=0)
        self.action_type_emb = nn.Embedding(action_type_count, d_model)
        self.numeric_adapter = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_emb = nn.Embedding(
            max_sequence_length + self.context_tokens,
            d_model,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.score_head = nn.Linear(d_model, 1)
        self.score_scale_head = nn.Linear(d_model, 1)
        self.score_distribution_head = nn.Linear(d_model, distribution_bins)
        self.validity_head = nn.Linear(d_model, 1)
        self.runtime_head = nn.Linear(d_model, 1)

    def forward(
        self,
        dataset_embedding: torch.Tensor,
        target_summary: torch.Tensor,
        anchor_features: torch.Tensor,
        anchor_mask: torch.Tensor,
        fidelity_ids: torch.Tensor,
        token_ids: torch.Tensor,
        numeric_values: torch.Tensor,
        numeric_mask: torch.Tensor,
        action_type_ids: torch.Tensor,
        lengths: torch.Tensor,
        numeric_mean: torch.Tensor,
        numeric_scale: torch.Tensor,
        *,
        anchor_owner: torch.Tensor | None = None,
        zero_shot: bool = False,
        genome_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch, width = token_ids.shape
        if width > self.max_sequence_length:
            raise ValueError(
                f"sequence width {width} exceeds broad grammar contract "
                f"{self.max_sequence_length}; truncation is forbidden"
            )
        if target_summary.shape != (batch, self.target_dim):
            raise ValueError("target summary shape differs from broad critic contract")
        if anchor_features.shape != (batch, self.anchor_count, 3) or anchor_mask.shape != (
            batch,
            self.anchor_count,
        ):
            raise ValueError("anchor signature shape differs from broad critic contract")
        if torch.any((fidelity_ids < 0) | (fidelity_ids >= self.fidelity_count)):
            raise ValueError("fidelity token is outside the broad critic vocabulary")

        observed = anchor_mask.to(dtype=anchor_features.dtype).clone()
        features = anchor_features.clone()
        if anchor_owner is not None:
            valid_owner = (anchor_owner >= 0) & (anchor_owner < self.anchor_count)
            rows = torch.arange(batch, device=anchor_owner.device)[valid_owner]
            owners = anchor_owner[valid_owner].long()
            observed[rows, owners] = 0
            features[rows, owners] = 0
        if zero_shot:
            observed.zero_()
            features.zero_()
        elif self.training and self.anchor_dropout > 0:
            keep = (
                torch.rand(
                    observed.shape,
                    device=observed.device,
                    dtype=features.dtype,
                )
                >= self.anchor_dropout
            ).to(features.dtype)
            observed *= keep
            features *= keep.unsqueeze(-1)

        if genome_only:
            dataset_prefix = torch.zeros(
                (batch, self.d_model),
                dtype=dataset_embedding.dtype,
                device=dataset_embedding.device,
            )
            target_prefix = torch.zeros_like(dataset_prefix)
            anchor_prefix = torch.zeros_like(dataset_prefix)
            fidelity_prefix = torch.zeros_like(dataset_prefix)
        else:
            dataset_prefix = self.dataset_adapter(dataset_embedding.float())
            target_prefix = self.target_adapter(target_summary.float())
            anchor_input = torch.cat(
                (features.float(), observed.float().unsqueeze(-1)), dim=-1
            ).flatten(1)
            anchor_prefix = self.anchor_adapter(anchor_input)
            fidelity_prefix = self.fidelity_emb(fidelity_ids.long())
        context = torch.stack(
            (dataset_prefix, target_prefix, anchor_prefix, fidelity_prefix), dim=1
        )

        signed_log = torch.sign(numeric_values) * torch.log1p(
            torch.abs(numeric_values)
        )
        mean = numeric_mean[token_ids]
        scale = numeric_scale[token_ids].clamp_min(1e-8)
        normalized = ((signed_log - mean) / scale).unsqueeze(-1)
        numeric = self.numeric_adapter(normalized) * numeric_mask.unsqueeze(-1)
        tokens = (
            self.token_emb(token_ids)
            + self.action_type_emb(action_type_ids)
            + numeric
        )
        hidden = torch.cat((context, tokens), dim=1)
        positions = torch.arange(hidden.shape[1], device=token_ids.device)
        hidden = hidden + self.pos_emb(positions).unsqueeze(0)
        # Context may see context. Ordered genome tokens may attend to all
        # context and prior genome tokens, never future genome tokens.
        causal = torch.triu(
            torch.ones(
                hidden.shape[1],
                hidden.shape[1],
                dtype=torch.bool,
                device=token_ids.device,
            ),
            diagonal=1,
        )
        causal[: self.context_tokens, : self.context_tokens] = False
        token_positions = torch.arange(width, device=token_ids.device).unsqueeze(0)
        padding = token_positions >= lengths.unsqueeze(1)
        key_padding = torch.cat(
            (
                torch.zeros(
                    (batch, self.context_tokens),
                    dtype=torch.bool,
                    device=token_ids.device,
                ),
                padding,
            ),
            dim=1,
        )
        hidden = self.transformer(
            hidden,
            mask=causal,
            src_key_padding_mask=key_padding,
        )
        hidden = self.out_norm(hidden)
        final = hidden[
            torch.arange(batch, device=hidden.device),
            lengths + self.context_tokens - 1,
        ]
        score = self.score_head(final).squeeze(-1)
        validity_logit = self.validity_head(final).squeeze(-1)
        return {
            "score": score,
            "score_log_scale": self.score_scale_head(final).squeeze(-1).clamp(-6, 4),
            "score_distribution_logits": self.score_distribution_head(final),
            "validity_logit": validity_logit,
            "validity": torch.sigmoid(validity_logit),
            "log_runtime": self.runtime_head(final).squeeze(-1),
            "representation": final,
        }


def _broad_groups(
    dataset_indices: torch.Tensor,
    fidelity_ids: torch.Tensor,
) -> list[torch.Tensor]:
    groups: list[torch.Tensor] = []
    keys = torch.stack((dataset_indices.long(), fidelity_ids.long()), dim=1)
    for key in torch.unique(keys, dim=0):
        indexes = torch.nonzero(
            (keys[:, 0] == key[0]) & (keys[:, 1] == key[1]),
            as_tuple=False,
        ).flatten()
        if len(indexes):
            groups.append(indexes)
    return groups


def _dataset_balanced_mean(
    values: torch.Tensor,
    groups: list[torch.Tensor],
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    for indexes in groups:
        selected = indexes if mask is None else indexes[mask[indexes]]
        if len(selected):
            terms.append(values[selected].mean())
    return torch.stack(terms).mean() if terms else values.sum() * 0.0


def broad_critic_losses(
    outputs: dict[str, torch.Tensor],
    scores: torch.Tensor,
    validity: torch.Tensor,
    log_runtime: torch.Tensor,
    dataset_indices: torch.Tensor,
    fidelity_ids: torch.Tensor,
    *,
    ranking_weight: float = 0.60,
    score_weight: float = 0.20,
    validity_weight: float = 0.10,
    runtime_weight: float = 0.10,
) -> dict[str, torch.Tensor]:
    """Frozen 60/20/10/10 dataset-balanced objective.

    Ranking groups use both dataset and fidelity, so screening and exact labels
    can never form a pair or list together.
    """

    weights = (ranking_weight, score_weight, validity_weight, runtime_weight)
    if any(weight < 0 for weight in weights) or abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError("broad critic loss weights must be nonnegative and sum to one")
    groups = _broad_groups(dataset_indices, fidelity_ids)
    finite_score = torch.isfinite(scores)
    listwise_terms: list[torch.Tensor] = []
    pairwise_terms: list[torch.Tensor] = []
    for indexes in groups:
        indexes = indexes[finite_score[indexes]]
        if len(indexes) < 2:
            continue
        target = scores[indexes]
        prediction = outputs["score"][indexes]
        listwise_terms.append(
            -(torch.softmax(target, dim=0) * torch.log_softmax(prediction, dim=0)).sum()
        )
        order = torch.argsort(target, stable=True)
        deciles = torch.empty_like(order)
        deciles[order] = torch.div(
            torch.arange(len(indexes), device=order.device) * 10,
            max(len(indexes), 1),
            rounding_mode="floor",
        ).clamp_max(9)
        left, right = torch.triu_indices(
            len(indexes), len(indexes), offset=1, device=indexes.device
        )
        target_delta = target[left] - target[right]
        keep = (target_delta != 0) & (deciles[left] != deciles[right])
        if torch.any(keep):
            direction = torch.sign(target_delta[keep])
            predicted_delta = prediction[left[keep]] - prediction[right[keep]]
            pairwise_terms.append(
                F.softplus(-direction * predicted_delta).mean()
            )
    zero = outputs["score"].sum() * 0.0
    listwise = torch.stack(listwise_terms).mean() if listwise_terms else zero
    pairwise = torch.stack(pairwise_terms).mean() if pairwise_terms else zero
    ranking = 0.5 * (listwise + pairwise)

    huber_values = F.huber_loss(
        outputs["score"],
        torch.nan_to_num(scores),
        delta=0.5,
        reduction="none",
    )
    huber = _dataset_balanced_mean(huber_values, groups, finite_score)
    distribution_targets = (
        ((torch.nan_to_num(scores).clamp(-3, 3) + 3) / 6)
        * outputs["score_distribution_logits"].shape[1]
    ).long().clamp_max(outputs["score_distribution_logits"].shape[1] - 1)
    distribution_values = F.cross_entropy(
        outputs["score_distribution_logits"],
        distribution_targets,
        reduction="none",
    )
    distribution = _dataset_balanced_mean(
        distribution_values, groups, finite_score
    )
    score_loss = 0.75 * huber + 0.25 * distribution

    validity_values = F.binary_cross_entropy_with_logits(
        outputs["validity_logit"],
        validity.float(),
        reduction="none",
    )
    validity_loss = _dataset_balanced_mean(validity_values, groups)
    finite_runtime = torch.isfinite(log_runtime)
    runtime_values = F.huber_loss(
        outputs["log_runtime"],
        torch.nan_to_num(log_runtime),
        delta=0.5,
        reduction="none",
    )
    runtime_loss = _dataset_balanced_mean(
        runtime_values, groups, finite_runtime
    )
    total = (
        ranking_weight * ranking
        + score_weight * score_loss
        + validity_weight * validity_loss
        + runtime_weight * runtime_loss
    )
    return {
        "total": total,
        "ranking": ranking,
        "listwise": listwise,
        "decile_pairwise": pairwise,
        "score": score_loss,
        "score_huber": huber,
        "score_distribution": distribution,
        "validity_bce": validity_loss,
        "runtime_huber": runtime_loss,
    }


def broad_partial_warm_start(
    model: DimeBroadCritic,
    checkpoint: dict[str, Any],
    tokens: list[str],
) -> dict[str, Any]:
    """Load identity-matched policy token rows and reset every other tensor."""

    source = checkpoint.get("model", checkpoint)
    if not isinstance(source, dict):
        raise ValueError("warm-start checkpoint has no model state")
    source_vocab = checkpoint.get("action_vocab") or checkpoint.get("vocab")
    source_tokens = source_vocab.get("tokens") if isinstance(source_vocab, dict) else None
    source_weight = source.get("token_emb.weight")
    token_rows = 0
    if (
        isinstance(source_tokens, list)
        and isinstance(source_weight, torch.Tensor)
        and source_weight.ndim == 2
        and source_weight.shape[1] == model.token_emb.weight.shape[1]
    ):
        source_index = {str(token): index for index, token in enumerate(source_tokens)}
        with torch.no_grad():
            for target_index, token in enumerate(tokens):
                source_row = source_index.get(token)
                if source_row is not None and source_row < source_weight.shape[0]:
                    model.token_emb.weight[target_index].copy_(
                        source_weight[source_row].to(
                            dtype=model.token_emb.weight.dtype,
                            device=model.token_emb.weight.device,
                        )
                    )
                    token_rows += 1
    return {
        "loaded_tensors": ["token_emb.weight"] if token_rows else [],
        "loaded_tensor_count": int(token_rows > 0),
        "token_rows": token_rows,
        "reset_dataset_adapters": True,
        "reset_prediction_heads": True,
        "source_model_version": checkpoint.get("model_version"),
    }
