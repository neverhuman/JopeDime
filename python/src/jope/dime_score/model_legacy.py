from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .model_config import DATASET_DIM, D_MODEL, FFN_DIM, N_HEADS, N_LAYERS


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
        indexes = torch.nonzero((dataset_indices == dataset) & valid_mask, as_tuple=False).flatten()
        if len(indexes) < 2:
            continue
        left, right = torch.triu_indices(len(indexes), len(indexes), offset=1, device=logits.device)
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
    ranking = bradley_terry_loss(outputs["score_logit"], percentiles, dataset_indices, valid)
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
        if not (name.startswith("decoder.") or name.startswith("out_norm.") or name == "pos_emb.weight"):
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
            if source_index_value is not None and source_index_value < source["token_emb.weight"].shape[0]:
                token_weight[target_index] = source["token_emb.weight"][source_index_value].to(
                    dtype=token_weight.dtype
                )
                token_rows += 1
        target["token_emb.weight"] = token_weight
    model.load_state_dict(target)
    return {
        "loaded_tensors": sorted(loaded),
        "loaded_tensor_count": len(loaded),
        "token_rows": token_rows,
        "source_model_version": checkpoint.get("model_version"),
    }
