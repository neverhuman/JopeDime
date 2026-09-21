from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .data import PROFILE_DIM, PROFILE_HIST_BINS


@dataclass
class ModelConfig:
    embedding_dim: int = 64
    hidden_dim: int = 128
    token_dim: int = 128
    attention_heads: int = 4
    attention_layers: int = 2
    histogram_bins: int = PROFILE_HIST_BINS
    rbf_bins: int = 16
    fourier_frequencies: int = 4
    rich_stats: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> ModelConfig:
        if not data:
            return cls()
        known = {field: data[field] for field in cls.__dataclass_fields__ if field in data}
        return cls(**known)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ResidualMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SetBlock(nn.Module):
    def __init__(self, dim: int, heads: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = ResidualMLP(dim, hidden_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        normed = self.norm(x)
        attended, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask, need_weights=False)
        return self.ff(x + attended)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    weights = mask.unsqueeze(-1) if values.ndim == mask.ndim + 1 else mask
    denom = weights.sum(dim=dim).clamp_min(1.0)
    return (values * weights).sum(dim=dim) / denom


def _masked_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = values.masked_fill(mask.unsqueeze(-1) <= 0, -1.0e9)
    return masked.max(dim=1).values


def _histogram(values: torch.Tensor, mask: torch.Tensor, bins: int) -> torch.Tensor:
    clipped = values.clamp(0.0, 1.0)
    centers = torch.linspace(0.0, 1.0, bins, device=values.device, dtype=values.dtype)
    width = 1.0 / max(1, bins - 1)
    soft = torch.clamp(1.0 - torch.abs(clipped.unsqueeze(-1) - centers) / width, min=0.0)
    soft = soft * mask.unsqueeze(-1)
    hist = soft.sum(dim=1)
    return hist / hist.sum(dim=1, keepdim=True).clamp_min(1.0e-6)


RICH_QUANTILE_POSITIONS = (
    0.0,
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
    0.10,
    0.125,
    0.20,
    0.25,
    0.333,
    0.375,
    0.50,
    0.625,
    0.667,
    0.75,
    0.80,
    0.875,
    0.90,
    0.95,
    0.98,
    0.99,
    0.995,
    0.999,
    1.0,
)
RICH_QUANTILE_RANGES = ((0.25, 0.75), (0.10, 0.90), (0.05, 0.95), (0.01, 0.99), (0.001, 0.999))
RICH_THRESHOLDS = (
    0.0,
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.98,
    0.99,
    0.995,
    0.999,
    1.0,
)
RICH_HISTOGRAM_BINS = (4, 8, 16, 32)
RICH_ECF_FREQUENCIES = 16
RICH_RBF_BANDWIDTHS = (0.025, 0.05, 0.10, 0.20)
RICH_RBF_CENTERS = 33
RICH_UNIQUENESS_BANDWIDTHS = (0.001, 0.01, 0.05)


def explicit_stats_dim(config: ModelConfig) -> int:
    base = 2 + 9 + 7 + config.histogram_bins + 3
    if not config.rich_stats:
        return base
    rich = 0
    rich += len(RICH_QUANTILE_POSITIONS) + (len(RICH_QUANTILE_POSITIONS) - 1) + len(RICH_QUANTILE_RANGES)
    rich += 2 * len(RICH_THRESHOLDS) + 4
    rich += sum(bins + 4 for bins in RICH_HISTOGRAM_BINS)
    rich += 3 * RICH_ECF_FREQUENCIES
    rich += len(RICH_RBF_BANDWIDTHS) * RICH_RBF_CENTERS
    rich += 11 + 9 + 8
    rich += 6
    rich += 6
    rich += 3 * len(RICH_UNIQUENESS_BANDWIDTHS)
    return base + rich


def _masked_quantiles(
    clipped: torch.Tensor, mask: torch.Tensor, positions: tuple[float, ...]
) -> torch.Tensor:
    denom = mask.sum(dim=1).clamp_min(1.0)
    sorted_values = clipped.masked_fill(mask <= 0, 1.0).sort(dim=1).values
    pos = torch.tensor(positions, device=clipped.device, dtype=clipped.dtype)
    q_index = ((denom - 1.0).unsqueeze(1) * pos.unsqueeze(0)).round().long()
    return sorted_values.gather(1, q_index.clamp(0, clipped.shape[1] - 1))


def _rich_explicit_stats(
    clipped: torch.Tensor, mask: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    denom = mask.sum(dim=1).clamp_min(1.0)
    batch, width = clipped.shape
    parts: list[torch.Tensor] = []

    rich_q = _masked_quantiles(clipped, mask, RICH_QUANTILE_POSITIONS)
    parts.append(rich_q)
    parts.append(rich_q[:, 1:] - rich_q[:, :-1])
    q_lookup = {position: index for index, position in enumerate(RICH_QUANTILE_POSITIONS)}
    parts.append(
        torch.stack(
            [rich_q[:, q_lookup[right]] - rich_q[:, q_lookup[left]] for left, right in RICH_QUANTILE_RANGES],
            dim=1,
        )
    )

    thresholds = torch.tensor(RICH_THRESHOLDS, device=clipped.device, dtype=clipped.dtype).view(1, 1, -1)
    x3 = clipped.unsqueeze(-1)
    m3 = mask.unsqueeze(-1)
    parts.append(((x3 <= thresholds).float() * m3).sum(dim=1) / denom.unsqueeze(1))
    parts.append(((x3 >= thresholds).float() * m3).sum(dim=1) / denom.unsqueeze(1))
    parts.append(((clipped <= 1.0e-12).float() * mask).sum(dim=1, keepdim=True) / denom.unsqueeze(1))
    parts.append(((clipped >= 1.0 - 1.0e-12).float() * mask).sum(dim=1, keepdim=True) / denom.unsqueeze(1))
    parts.append(
        (((clipped >= 0.40) & (clipped <= 0.60)).float() * mask).sum(dim=1, keepdim=True) / denom.unsqueeze(1)
    )
    parts.append(
        (((clipped <= 0.05) | (clipped >= 0.95)).float() * mask).sum(dim=1, keepdim=True) / denom.unsqueeze(1)
    )

    for bins in RICH_HISTOGRAM_BINS:
        hist = _histogram(clipped, mask, bins)
        entropy = -(hist * torch.log(hist + 1.0e-8)).sum(dim=1, keepdim=True) / math.log(bins)
        simpson = 1.0 - hist.square().sum(dim=1, keepdim=True)
        max_bin = hist.max(dim=1, keepdim=True).values
        empty = (hist <= 1.0e-8).float().mean(dim=1, keepdim=True)
        parts.extend([hist, entropy, simpson, max_bin, empty])

    freqs = torch.arange(1, RICH_ECF_FREQUENCIES + 1, device=clipped.device, dtype=clipped.dtype)
    angles = 2.0 * math.pi * clipped.unsqueeze(-1) * freqs.view(1, 1, -1)
    sin_mean = (torch.sin(angles) * mask.unsqueeze(-1)).sum(dim=1) / denom.unsqueeze(1)
    cos_mean = (torch.cos(angles) * mask.unsqueeze(-1)).sum(dim=1) / denom.unsqueeze(1)
    parts.extend([sin_mean, cos_mean, torch.sqrt(sin_mean.square() + cos_mean.square() + 1.0e-12)])

    centers = torch.linspace(0.0, 1.0, RICH_RBF_CENTERS, device=clipped.device, dtype=clipped.dtype)
    for bandwidth in RICH_RBF_BANDWIDTHS:
        density = torch.exp(
            -((clipped.unsqueeze(-1) - centers.view(1, 1, -1)).square()) / (2.0 * bandwidth * bandwidth)
        )
        parts.append((density * mask.unsqueeze(-1)).sum(dim=1) / denom.unsqueeze(1))

    centered = (clipped - mean.unsqueeze(1)) * mask
    raw = [(clipped.pow(power) * mask).sum(dim=1) / denom for power in range(2, 13)]
    central = [centered.pow(power).sum(dim=1) / denom for power in range(2, 11)]
    standardized = [central[power - 2] / (std.pow(power) + 1.0e-6) for power in range(3, 11)]
    parts.extend([torch.stack(raw, dim=1), torch.stack(central, dim=1), torch.stack(standardized, dim=1)])

    sorted_values = clipped.masked_fill(mask <= 0, 1.0).sort(dim=1).values
    positions = torch.arange(width, device=clipped.device).view(1, -1)
    valid = positions < denom.long().view(-1, 1)
    adj_valid = positions[:, 1:] < denom.long().view(-1, 1)
    adjacent = torch.clamp(sorted_values[:, 1:] - sorted_values[:, :-1], min=0.0) * adj_valid.float()
    adj_denom = (denom - 1.0).clamp_min(1.0)
    gap_mean = adjacent.sum(dim=1) / adj_denom
    gap_std = torch.sqrt(
        ((adjacent - gap_mean.unsqueeze(1)).square() * adj_valid.float()).sum(dim=1) / adj_denom + 1.0e-8
    )
    gap_min = adjacent.masked_fill(~adj_valid, 1.0).min(dim=1).values
    gap_max = adjacent.max(dim=1).values
    left_gap = rich_q[:, 0]
    right_gap = 1.0 - rich_q[:, -1]
    parts.append(torch.stack([gap_mean, gap_std, gap_min, gap_max, left_gap, right_gap], dim=1))

    ecdf_hi = (positions.float() + 1.0) / denom.view(-1, 1)
    ecdf_lo = positions.float() / denom.view(-1, 1)
    ks = (
        torch.maximum((ecdf_hi - sorted_values).abs(), (sorted_values - ecdf_lo).abs())
        .masked_fill(~valid, 0.0)
        .max(dim=1)
        .values
    )
    expected = (positions.float() + 0.5) / denom.view(-1, 1)
    cvm = ((((sorted_values - expected).square()) * valid.float()).sum(dim=1) + 1.0 / (12.0 * denom)) / denom
    wasserstein = ((sorted_values - expected).abs() * valid.float()).sum(dim=1) / denom
    positive_sum = (sorted_values * valid.float()).sum(dim=1).clamp_min(1.0e-8)
    ranks = (positions.float() + 1.0) * valid.float()
    gini = (2.0 * (ranks * sorted_values).sum(dim=1) / (denom * positive_sum)) - ((denom + 1.0) / denom)
    x_safe = clipped.clamp_min(1.0e-12)
    mean_safe = ((x_safe * mask).sum(dim=1) / denom).clamp_min(1.0e-12)
    ratio = x_safe / mean_safe.unsqueeze(1)
    theil = ((ratio * torch.log(ratio)) * mask).sum(dim=1) / denom
    pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
    pair_abs = (clipped.unsqueeze(2) - clipped.unsqueeze(1)).abs()
    pairwise_mean_abs = (pair_abs * pair_mask).sum(dim=(1, 2)) / pair_mask.sum(dim=(1, 2)).clamp_min(1.0)
    parts.append(torch.stack([ks, cvm, wasserstein, gini, theil, pairwise_mean_abs], dim=1))

    for bandwidth in RICH_UNIQUENESS_BANDWIDTHS:
        sim = torch.exp(-(pair_abs.square()) / (2.0 * bandwidth * bandwidth)) * pair_mask
        counts = sim.sum(dim=2).clamp_min(1.0e-6)
        mode_fraction = counts.masked_fill(mask <= 0, -1.0).max(dim=1).values / denom
        soft_unique = ((1.0 / counts) * mask).sum(dim=1) / denom
        probabilities = counts * mask
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        entropy = -(probabilities * torch.log(probabilities + 1.0e-8)).sum(dim=1) / torch.log(denom + 1.0)
        parts.append(torch.stack([mode_fraction, soft_unique, entropy], dim=1))

    return torch.nan_to_num(torch.cat(parts, dim=1), nan=0.0, posinf=0.0, neginf=0.0).view(batch, -1)


def explicit_stats(
    values: torch.Tensor,
    mask: torch.Tensor,
    task: torch.Tensor,
    original_length: torch.Tensor,
    effective_count: torch.Tensor,
    replacement_fraction: torch.Tensor,
    histogram_bins: int,
    rich_stats: bool = False,
) -> torch.Tensor:
    mask = mask.float()
    clipped = values.float().clamp(0.0, 1.0)
    denom = mask.sum(dim=1).clamp_min(1.0)
    mean = (clipped * mask).sum(dim=1) / denom
    centered = (clipped - mean.unsqueeze(1)) * mask
    var = (centered.square().sum(dim=1) / denom).clamp_min(0.0)
    std = torch.sqrt(var + 1.0e-6)
    skew = centered.pow(3).sum(dim=1) / denom / (std.pow(3) + 1.0e-6)
    kurt = centered.pow(4).sum(dim=1) / denom / (std.pow(4) + 1.0e-6)
    masked_hi = clipped.masked_fill(mask <= 0, -1.0e9).max(dim=1).values
    masked_lo = clipped.masked_fill(mask <= 0, 1.0e9).min(dim=1).values

    q = _masked_quantiles(clipped, mask, (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0))

    hist = _histogram(clipped, mask, histogram_bins)
    entropy = -(hist * torch.log(hist + 1.0e-8)).sum(dim=1, keepdim=True) / math.log(histogram_bins)
    low_tail = ((clipped <= 0.05).float() * mask).sum(dim=1, keepdim=True) / denom.unsqueeze(1)
    high_tail = ((clipped >= 0.95).float() * mask).sum(dim=1, keepdim=True) / denom.unsqueeze(1)
    task_one_hot = F.one_hot(task.clamp(0, 1), num_classes=2).float()
    meta = torch.stack(
        [
            torch.log1p(original_length.float()) / 16.0,
            torch.log1p(effective_count.float()) / 16.0,
            replacement_fraction.float(),
            mean,
            std,
            masked_lo,
            masked_hi,
            skew,
            kurt,
        ],
        dim=1,
    )
    parts = [task_one_hot, meta, q, hist, entropy, low_tail, high_tail]
    if rich_stats:
        parts.append(_rich_explicit_stats(clipped, mask, mean, std))
    return torch.cat(parts, dim=1)


class TargetEncoder(nn.Module):
    """Hybrid statistics + neural set encoder that returns one float32[64] vector."""

    def __init__(self, config: ModelConfig | Mapping[str, object] | None = None) -> None:
        super().__init__()
        self.config = config if isinstance(config, ModelConfig) else ModelConfig.from_mapping(config)
        cfg = self.config
        token_input_dim = 1 + cfg.rbf_bins + 2 * cfg.fourier_frequencies + 2 + 1
        centers = torch.linspace(0.0, 1.0, cfg.rbf_bins)
        freqs = torch.arange(1, cfg.fourier_frequencies + 1, dtype=torch.float32)
        self.register_buffer("rbf_centers", centers, persistent=False)
        self.register_buffer("fourier_freqs", freqs, persistent=False)
        self.token_mlp = nn.Sequential(
            nn.Linear(token_input_dim, cfg.token_dim),
            nn.GELU(),
            ResidualMLP(cfg.token_dim, cfg.hidden_dim),
        )
        self.set_blocks = nn.ModuleList(
            [
                SetBlock(cfg.token_dim, cfg.attention_heads, cfg.hidden_dim)
                for _ in range(cfg.attention_layers)
            ]
        )
        stats_dim = explicit_stats_dim(cfg)
        self.stats_mlp = nn.Sequential(
            nn.Linear(stats_dim, cfg.hidden_dim),
            nn.GELU(),
            ResidualMLP(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(cfg.token_dim * 2 + cfg.hidden_dim, cfg.hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden_dim * 2),
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.embedding_dim),
        )
        self.profile_head = nn.Sequential(
            nn.Linear(cfg.embedding_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, PROFILE_DIM),
        )
        self.task_head = nn.Sequential(
            nn.Linear(cfg.embedding_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim // 2, 2),
        )

    def value_features(self, values: torch.Tensor, mask: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        x = values.float().clamp(0.0, 1.0)
        rbf = torch.exp(-48.0 * (x.unsqueeze(-1) - self.rbf_centers.view(1, 1, -1)).square())
        angles = 2.0 * math.pi * x.unsqueeze(-1) * self.fourier_freqs.view(1, 1, -1)
        fourier = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        task_one_hot = (
            F.one_hot(task.clamp(0, 1), num_classes=2).float().unsqueeze(1).expand(-1, x.shape[1], -1)
        )
        return torch.cat([x.unsqueeze(-1), rbf, fourier, task_one_hot, mask.float().unsqueeze(-1)], dim=-1)

    def forward(
        self,
        values: torch.Tensor,
        task: torch.Tensor,
        original_length: torch.Tensor,
        effective_count: torch.Tensor,
        mask: torch.Tensor | None = None,
        replacement_fraction: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if mask is None:
            mask = torch.ones_like(values, dtype=torch.float32)
        if replacement_fraction is None:
            replacement_fraction = torch.zeros(values.shape[0], device=values.device, dtype=values.dtype)
        mask = mask.float()
        token_features = self.value_features(values, mask, task)
        tokens = self.token_mlp(token_features)
        key_padding_mask = mask <= 0
        for block in self.set_blocks:
            tokens = block(tokens, key_padding_mask)
        pooled_mean = _masked_mean(tokens, mask)
        pooled_max = _masked_max(tokens, mask)
        stats = explicit_stats(
            values,
            mask,
            task,
            original_length,
            effective_count,
            replacement_fraction,
            self.config.histogram_bins,
            rich_stats=self.config.rich_stats,
        )
        stats_features = self.stats_mlp(stats)
        embedding = self.fusion(torch.cat([pooled_mean, pooled_max, stats_features], dim=1))
        embedding = F.normalize(embedding.float(), dim=1)
        if not return_aux:
            return embedding
        return {
            "embedding": embedding,
            "profile": self.profile_head(embedding),
            "task_logits": self.task_head(embedding),
            "stats": stats,
        }


def checkpoint_config(checkpoint: Mapping[str, object]) -> ModelConfig:
    return ModelConfig.from_mapping(
        checkpoint.get("model_config") if isinstance(checkpoint, Mapping) else None
    )


def build_model_from_checkpoint(checkpoint: Mapping[str, object]) -> TargetEncoder:
    model = TargetEncoder(checkpoint_config(checkpoint))
    state = checkpoint.get("model")
    if state is None:
        state = checkpoint.get("state_dict")
    if state is None:
        raise KeyError("Checkpoint does not contain a model state dict")
    cleaned = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(cleaned)
    return model
