"""Standalone TabPFN v3 checkpoint-compatible architecture.

This module follows the public PriorLabs TabPFN v3 architecture source pinned
at commit 8e673485d896942ba01695949773c95046d6129f, but keeps the dependency
surface local and small.  The checkpoint-visible module and parameter names are
preserved so the binary v3 classifier checkpoint can be strict-loaded.
"""

from __future__ import annotations

import dataclasses
import argparse
import hashlib
import json
import logging
import math
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

TaskType = Literal["multiclass", "regression"]

_logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PerformanceOptions:
    """Small local equivalent of TabPFN's performance options."""

    save_peak_memory_factor: int | None = None
    force_recompute_layer: bool = False
    use_chunkwise_inference: bool = True
    enable_torch_compile: bool = False


@dataclasses.dataclass
class KVCacheEntry:
    """Unquantized key/value cache entry for one ICL layer."""

    key: torch.Tensor | None = None
    value: torch.Tensor | None = None

    def quantize(self, dtype: torch.dtype = torch.int8) -> QuantizedKVCacheEntry:
        if self.key is None or self.value is None:
            return QuantizedKVCacheEntry(None, None, None, None)
        return QuantizedKVCacheEntry.from_tensors(self.key, self.value, dtype=dtype)


@dataclasses.dataclass
class QuantizedKVCacheEntry:
    """Simple symmetric per-tensor quantized KV cache entry."""

    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    key_scale: torch.Tensor | None = None
    value_scale: torch.Tensor | None = None

    @classmethod
    def from_tensors(
        cls,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        dtype: torch.dtype = torch.int8,
    ) -> QuantizedKVCacheEntry:
        if dtype is not torch.int8:
            raise ValueError("Only torch.int8 quantized cache entries are supported.")
        key_abs = key.detach().abs().amax().clamp(min=1e-8)
        value_abs = value.detach().abs().amax().clamp(min=1e-8)
        key_scale = key_abs / 127
        value_scale = value_abs / 127
        return cls(
            key=(key.detach() / key_scale).round().clamp(-127, 127).to(dtype),
            value=(value.detach() / value_scale).round().clamp(-127, 127).to(dtype),
            key_scale=key_scale,
            value_scale=value_scale,
        )

    def dequantize(self, dtype: torch.dtype) -> KVCacheEntry:
        assert self.key is not None and self.value is not None
        assert self.key_scale is not None and self.value_scale is not None
        return KVCacheEntry(
            key=(self.key.to(torch.float32) * self.key_scale).to(dtype),
            value=(self.value.to(torch.float32) * self.value_scale).to(dtype),
        )


@dataclasses.dataclass
class KVCache:
    """Container for per-layer KV entries."""

    kv: dict[int, KVCacheEntry | QuantizedKVCacheEntry] = dataclasses.field(
        default_factory=dict
    )

    def is_empty(self) -> bool:
        return len(self.kv) == 0

    def _kv_to(
        self, device: torch.device | str
    ) -> dict[int, KVCacheEntry | QuantizedKVCacheEntry]:
        moved: dict[int, KVCacheEntry | QuantizedKVCacheEntry] = {}
        for idx, entry in self.kv.items():
            if isinstance(entry, QuantizedKVCacheEntry):
                moved[idx] = QuantizedKVCacheEntry(
                    key=None if entry.key is None else entry.key.to(device),
                    value=None if entry.value is None else entry.value.to(device),
                    key_scale=(
                        None if entry.key_scale is None else entry.key_scale.to(device)
                    ),
                    value_scale=(
                        None
                        if entry.value_scale is None
                        else entry.value_scale.to(device)
                    ),
                )
            else:
                moved[idx] = KVCacheEntry(
                    key=None if entry.key is None else entry.key.to(device),
                    value=None if entry.value is None else entry.value.to(device),
                )
        return moved

    @staticmethod
    def _dict_of_tensors_to(
        value: dict[str, torch.Tensor] | None,
        device: torch.device | str,
    ) -> dict[str, torch.Tensor] | None:
        if value is None:
            return None
        return {k: v.to(device) for k, v in value.items()}

    @staticmethod
    def _list_of_tensors_to(
        value: list[torch.Tensor] | None,
        device: torch.device | str,
    ) -> list[torch.Tensor] | None:
        if value is None:
            return None
        return [v.to(device) for v in value]


@dataclasses.dataclass
class TabPFNV3Cache(KVCache):
    """Top-level cache container for explicit TabPFN v3 inference caches."""

    train_embeddings: torch.Tensor | None = None
    train_shape: tuple[int, int] = (0, 0)
    scaler_cache: dict[str, torch.Tensor] | None = None
    inducing_hidden: list[torch.Tensor] | None = None

    def to(self, device: torch.device | str) -> TabPFNV3Cache:
        return TabPFNV3Cache(
            kv=self._kv_to(device),
            train_embeddings=(
                None
                if self.train_embeddings is None
                else self.train_embeddings.to(device)
            ),
            train_shape=self.train_shape,
            scaler_cache=self._dict_of_tensors_to(self.scaler_cache, device),
            inducing_hidden=self._list_of_tensors_to(self.inducing_hidden, device),
        )

    def quantize(self, dtype: torch.dtype = torch.int8) -> TabPFNV3Cache:
        return TabPFNV3Cache(
            kv={
                idx: entry.quantize(dtype) if isinstance(entry, KVCacheEntry) else entry
                for idx, entry in self.kv.items()
            },
            train_embeddings=self.train_embeddings,
            train_shape=self.train_shape,
            scaler_cache=self.scaler_cache,
            inducing_hidden=self.inducing_hidden,
        )


@dataclasses.dataclass
class TabPFNV3Config:
    """Configuration for the TabPFN v3 classifier checkpoint."""

    max_num_classes: int = 160
    num_buckets: int = 5000
    name: str = "TabPFN-v3"
    embed_dim: int = 128
    dist_embed_num_blocks: int = 3
    dist_embed_num_heads: int = 8
    dist_embed_num_inducing_points: int = 128
    feature_group_size: int = 3
    feat_agg_num_blocks: int = 3
    feat_agg_num_heads: int = 8
    feat_agg_num_cls_tokens: int = 4
    feat_agg_rope_base: float = 100_000
    use_rope: bool = True
    nlayers: int = 24
    icl_num_heads: int = 8
    icl_num_kv_heads: int | None = None
    icl_num_kv_heads_test: int | None = None
    decoder_head_dim: int = 64
    decoder_num_heads: int = 6
    decoder_use_softmax_scaling: bool = False
    ff_factor: int = 2
    dropout: float = 0.0
    softmax_scaling_mlp_hidden_dim: int = 64
    layernorm_elementwise_affine: bool = True
    use_nan_indicators: bool = True
    inference_row_chunk_size: int = 2048
    inference_col_chunk_size: int = 4

    def __post_init__(self) -> None:
        if self.icl_num_kv_heads is not None:
            if self.icl_num_heads % self.icl_num_kv_heads != 0:
                raise ValueError("icl_num_heads must be divisible by icl_num_kv_heads")
        if self.icl_num_kv_heads_test is not None:
            if self.icl_num_heads % self.icl_num_kv_heads_test != 0:
                raise ValueError(
                    "icl_num_heads must be divisible by icl_num_kv_heads_test"
                )
            effective_kv = self.icl_num_kv_heads or self.icl_num_heads
            if self.icl_num_kv_heads_test > effective_kv:
                raise ValueError("icl_num_kv_heads_test cannot exceed train KV heads")

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> TabPFNV3Config:
        fields = {field.name for field in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in config.items() if k in fields})

    def get_unused_config(self, config: dict[str, Any]) -> dict[str, Any]:
        fields = {field.name for field in dataclasses.fields(self)}
        return {k: v for k, v in config.items() if k not in fields}


class TorchStandardScaler:
    """Minimal train-fitted standard scaler for tensors shaped (rows, batch, cols)."""

    def fit(self, x_RBC: torch.Tensor) -> dict[str, torch.Tensor]:
        mean = x_RBC.mean(dim=0)
        std = x_RBC.std(dim=0, unbiased=False)
        std = torch.where(std > 1e-12, std, torch.ones_like(std))
        return {"mean": mean, "std": std}

    def transform(
        self,
        x_RBC: torch.Tensor,
        *,
        fitted_cache: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        mean = fitted_cache["mean"].to(device=x_RBC.device, dtype=x_RBC.dtype)
        std = fitted_cache["std"].to(device=x_RBC.device, dtype=x_RBC.dtype)
        return (x_RBC - mean.unsqueeze(0)) / std.unsqueeze(0)


def is_oom_error(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "oom" in msg


def chunked_evaluate_maybe_inplace(
    fn: Callable[..., torch.Tensor],
    x: torch.Tensor,
    save_peak_memory_factor: int | None = None,
    *,
    residual: bool,
    batch_dims: int,
    **kwargs: Any,
) -> torch.Tensor:
    """Compatibility shim for TabPFN's chunking helper.

    The upstream helper can fold leading batch dimensions before calling a
    module. This eager version preserves that shape behavior but skips actual
    memory-saving chunks.
    """

    del save_peak_memory_factor
    if batch_dims > 1:
        leading_shape = tuple(x.shape[:batch_dims])
        flat_batch = math.prod(leading_shape)
        x_for_fn = x.reshape(flat_batch, *x.shape[batch_dims:])
        out = fn(x_for_fn, **kwargs)
        out = out.reshape(*leading_shape, *out.shape[1:])
    else:
        out = fn(x, **kwargs)
    return x + out if residual else out


def scaled_dot_product_attention(
    q_BSHD: torch.Tensor,
    k_BJHD: torch.Tensor,
    v_BJHD: torch.Tensor,
) -> torch.Tensor:
    """SDPA wrapper accepting tensors in (batch, sequence, heads, dim) layout."""

    q_heads = q_BSHD.shape[2]
    kv_heads = k_BJHD.shape[2]
    if kv_heads != q_heads:
        if q_heads % kv_heads != 0:
            raise ValueError("KV head count must divide query head count")
        repeat = q_heads // kv_heads
        k_BJHD = k_BJHD.repeat_interleave(repeat, dim=2)
        v_BJHD = v_BJHD.repeat_interleave(repeat, dim=2)

    q = q_BSHD.transpose(1, 2)
    k = k_BJHD.transpose(1, 2)
    v = v_BJHD.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    return out.transpose(1, 2).contiguous()


def apply_rope(
    t: torch.Tensor,
    inv_freq: torch.Tensor,
    *,
    interleaved: bool = False,
) -> torch.Tensor:
    """Apply rotary positional embeddings along the second-to-last dimension."""

    dtype = t.dtype
    seq_len = t.shape[-2]
    positions = torch.arange(seq_len, device=t.device, dtype=inv_freq.dtype)
    freqs = positions[:, None] * inv_freq[None, :]
    cos = freqs.cos()
    sin = freqs.sin()
    if interleaved:
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)
        t_even = t[..., 0::2]
        t_odd = t[..., 1::2]
        t_rotated = torch.stack((-t_odd, t_even), dim=-1).flatten(-2)
    else:
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
        half = t.shape[-1] // 2
        t_rotated = torch.cat((-t[..., half:], t[..., :half]), dim=-1)
    return (t * cos + t_rotated * sin).to(dtype)


class RotaryEmbedding(nn.Module):
    """Compile-friendly rotary positional embedding."""

    def __init__(
        self,
        dim: int,
        *,
        theta: float = 10_000.0,
        interleaved: bool = False,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE head dim must be even, got {dim}")
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.freqs = nn.Parameter(inv_freq, requires_grad=False)
        self.interleaved = interleaved

    def rotate_queries_or_keys(self, t_BHSD: torch.Tensor) -> torch.Tensor:
        return apply_rope(t_BHSD, self.freqs, interleaved=self.interleaved)


class _DtypeMatchingRMSNorm(nn.RMSNorm):
    """RMSNorm that casts its weight to the input dtype under autocast."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.weight is not None and self.weight.dtype != input.dtype:
            return F.rms_norm(
                input,
                self.normalized_shape,
                self.weight.to(input.dtype),
                self.eps,
            )
        return super().forward(input)


class ManyClassDecoder(nn.Module):
    """Attention retrieval decoder for many-class classification."""

    def __init__(
        self,
        max_num_classes: int,
        input_size: int,
        head_dim: int = 64,
        num_heads: int = 6,
        softmax_scaling_layer: nn.Module | None = None,
    ):
        super().__init__()
        self.max_num_classes = max_num_classes
        self.input_size = input_size
        self.attention_size = head_dim * num_heads
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.q_projection = nn.Linear(self.input_size, self.attention_size)
        self.k_projection = nn.Linear(self.input_size, self.attention_size)
        self.softmax_scaling_layer = softmax_scaling_layer

    def forward(
        self,
        train_embeddings: torch.Tensor,
        test_embeddings: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        B, M, _ = test_embeddings.shape
        q_BME = self.q_projection(test_embeddings)
        if train_embeddings.dtype != q_BME.dtype:
            train_embeddings = train_embeddings.to(q_BME.dtype)
        k_BNE = self.k_projection(train_embeddings)

        if M == 0:
            empty = test_embeddings.new_empty((0, B, self.max_num_classes))
            return empty + (q_BME.sum() + k_BNE.sum()) * 0.0

        one_hot_targets_BNT = (
            F.one_hot(targets.long(), num_classes=self.max_num_classes)
            .to(dtype=q_BME.dtype)
            .contiguous()
        )

        q_BMHD = q_BME.view(B, M, self.num_heads, self.head_dim).contiguous()
        k_BNHD = k_BNE.view(B, -1, self.num_heads, self.head_dim).contiguous()
        one_hot_targets_BNHT = (
            one_hot_targets_BNT.unsqueeze(2)
            .expand(-1, -1, self.num_heads, -1)
            .contiguous()
        )
        test_output_BMHT = _chunked_class_attention(
            q_BMHD,
            k_BNHD,
            one_hot_targets_BNHT,
            softmax_scaling_layer=self.softmax_scaling_layer,
        )
        test_output_BMT = test_output_BMHT.mean(2)
        return torch.log(torch.clamp(test_output_BMT.transpose(0, 1), min=1e-5) + 3e-5)


def _chunked_class_attention(
    q_BSHD: torch.Tensor,
    k_BJHD: torch.Tensor,
    v_BJHT: torch.Tensor,
    softmax_scaling_layer: nn.Module | None = None,
) -> torch.Tensor:
    B, S, H, D = q_BSHD.shape
    T = v_BJHT.shape[-1]
    num_chunks = math.ceil(T / D)
    pad = num_chunks * D - T
    if pad > 0:
        v_BJHT = F.pad(v_BJHT, (0, pad))

    J = v_BJHT.shape[1]
    v_folded = (
        v_BJHT.reshape(B, J, H, num_chunks, D)
        .permute(0, 3, 1, 2, 4)
        .reshape(B * num_chunks, J, H, D)
        .contiguous()
    )
    q_folded = (
        q_BSHD.unsqueeze(1)
        .expand(-1, num_chunks, -1, -1, -1)
        .reshape(B * num_chunks, S, H, D)
        .contiguous()
    )
    k_folded = (
        k_BJHD.unsqueeze(1)
        .expand(-1, num_chunks, -1, -1, -1)
        .reshape(B * num_chunks, J, H, D)
        .contiguous()
    )
    out_folded = _batched_scaled_dot_product_attention(
        q_folded,
        k_folded,
        v_folded,
        softmax_scaling_layer=softmax_scaling_layer,
    )
    return (
        out_folded.reshape(B, num_chunks, S, H, D)
        .permute(0, 2, 3, 1, 4)
        .reshape(B, S, H, num_chunks * D)[..., :T]
    )


class TrainableOrthogonalEmbedding(nn.Module):
    """Trainable class embeddings initialized orthogonally."""

    def __init__(self, num_classes: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
        self._init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x.long())

    def _init(self) -> None:
        weight = self.embedding.weight
        num_classes, embed_dim = weight.shape
        k = min(num_classes, embed_dim)
        q, _ = torch.linalg.qr(torch.randn(embed_dim, k))
        with torch.no_grad():
            weight[:k].copy_(q.T)
            if num_classes > embed_dim:
                extra = torch.randn(num_classes - k, embed_dim)
                extra = extra / extra.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                weight[k:].copy_(extra)


class MLP(nn.Sequential):
    """Two-layer GELU feed-forward network with zero-initialized output."""

    def __init__(
        self,
        emsize: int,
        dim_feedforward: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ) -> None:
        kw = {"device": device, "dtype": dtype}
        linear2 = nn.Linear(dim_feedforward, emsize, bias=False, **kw)
        nn.init.zeros_(linear2.weight)
        super().__init__(
            nn.Linear(emsize, dim_feedforward, bias=False, **kw),
            nn.GELU(),
            linear2,
        )


class SoftmaxScalingMLP(nn.Module):
    """Query-aware attention scaling used in the v3 checkpoint."""

    def __init__(self, num_heads: int, head_dim: int, n_hidden: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        base_out_dim = num_heads * head_dim
        self.base_mlp = nn.Sequential(
            nn.Linear(1, n_hidden), nn.GELU(), nn.Linear(n_hidden, base_out_dim)
        )
        self.query_mlp = nn.Sequential(
            nn.Linear(head_dim, n_hidden), nn.GELU(), nn.Linear(n_hidden, head_dim)
        )
        nn.init.zeros_(self.query_mlp[-1].weight)  # type: ignore[index]
        nn.init.zeros_(self.query_mlp[-1].bias)  # type: ignore[index]

    def forward(self, q_BSHD: torch.Tensor, n: int) -> torch.Tensor:
        logn_11 = _safe_log_seqlen(n, q_BSHD.device, q_BSHD.dtype).reshape(1, 1)
        base_scales = self.base_mlp(logn_11).view(1, 1, self.num_heads, self.head_dim)
        modulation = 1 + torch.tanh(self.query_mlp(q_BSHD))
        return q_BSHD * base_scales * modulation


def _batched_scaled_dot_product_attention(
    q_BSHD: torch.Tensor,
    k_BJHD: torch.Tensor,
    v_BJHD: torch.Tensor,
    softmax_scaling_layer: nn.Module | None = None,
) -> torch.Tensor:
    if softmax_scaling_layer is not None:
        q_BSHD = softmax_scaling_layer(q_BSHD, k_BJHD.shape[1])
    return scaled_dot_product_attention(q_BSHD, k_BJHD, v_BJHD)


class Attention(nn.Module):
    """Multi-head self-attention, optionally with RoPE."""

    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        head_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        kw = {"device": device, "dtype": dtype, "bias": False}
        self.q_projection = nn.Linear(embedding_size, head_dim * num_heads, **kw)
        self.k_projection = nn.Linear(embedding_size, head_dim * num_heads, **kw)
        self.v_projection = nn.Linear(embedding_size, head_dim * num_heads, **kw)
        self.out_projection = nn.Linear(head_dim * num_heads, embedding_size, **kw)
        nn.init.xavier_uniform_(self.q_projection.weight)
        nn.init.xavier_uniform_(self.k_projection.weight)
        nn.init.xavier_uniform_(self.v_projection.weight)
        nn.init.zeros_(self.out_projection.weight)

    def forward(
        self,
        x_BSE: torch.Tensor,
        rope: RotaryEmbedding | None = None,
    ) -> torch.Tensor:
        B, S, _ = x_BSE.shape
        q = self.q_projection(x_BSE).view(B, S, -1, self.head_dim)
        k = self.k_projection(x_BSE).view(B, S, -1, self.head_dim)
        v = self.v_projection(x_BSE).view(B, S, -1, self.head_dim)
        if rope is not None:
            q = rope.rotate_queries_or_keys(q.transpose(1, 2)).transpose(1, 2)
            k = rope.rotate_queries_or_keys(k.transpose(1, 2)).transpose(1, 2)
        out = _batched_scaled_dot_product_attention(q, k, v)
        out = out.reshape(B, S, self.head_dim * self.num_heads)
        return self.out_projection(out)


class CrossAttention(nn.Module):
    """Multi-head cross-attention."""

    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        head_dim: int,
        softmax_scaling_layer: nn.Module | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.softmax_scaling_layer = softmax_scaling_layer
        kw = {"device": device, "dtype": dtype, "bias": False}
        self.q_projection = nn.Linear(embedding_size, head_dim * num_heads, **kw)
        self.k_projection = nn.Linear(embedding_size, head_dim * num_heads, **kw)
        self.v_projection = nn.Linear(embedding_size, head_dim * num_heads, **kw)
        self.out_projection = nn.Linear(head_dim * num_heads, embedding_size, **kw)
        nn.init.xavier_uniform_(self.q_projection.weight)
        nn.init.xavier_uniform_(self.k_projection.weight)
        nn.init.xavier_uniform_(self.v_projection.weight)
        nn.init.zeros_(self.out_projection.weight)

    def forward(
        self,
        x_for_query_BQE: torch.Tensor,
        x_for_key_and_value_BVE: torch.Tensor,
    ) -> torch.Tensor:
        B, Q, _ = x_for_query_BQE.shape
        _, V, _ = x_for_key_and_value_BVE.shape
        q = self.q_projection(x_for_query_BQE).view(B, Q, -1, self.head_dim)
        k = self.k_projection(x_for_key_and_value_BVE).view(B, V, -1, self.head_dim)
        v = self.v_projection(x_for_key_and_value_BVE).view(B, V, -1, self.head_dim)
        out = _batched_scaled_dot_product_attention(
            q, k, v, softmax_scaling_layer=self.softmax_scaling_layer
        )
        return self.out_projection(out.reshape(B, Q, self.head_dim * self.num_heads))


class ICLAttention(nn.Module):
    """ICL attention where keys and values are restricted to train rows."""

    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        head_dim: int,
        softmax_scaling_layer: nn.Module | None = None,
        num_kv_heads: int | None = None,
        num_kv_heads_test: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.softmax_scaling_layer = softmax_scaling_layer
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_heads_test = num_kv_heads_test
        kw = {"device": device, "dtype": dtype, "bias": False}
        self.q_projection = nn.Linear(embedding_size, head_dim * num_heads, **kw)
        self.out_projection = nn.Linear(head_dim * num_heads, embedding_size, **kw)
        nn.init.xavier_uniform_(self.q_projection.weight)
        nn.init.zeros_(self.out_projection.weight)
        kv_dim = (num_kv_heads or num_heads) * head_dim
        self.k_projection = nn.Linear(embedding_size, kv_dim, **kw)
        self.v_projection = nn.Linear(embedding_size, kv_dim, **kw)
        nn.init.xavier_uniform_(self.k_projection.weight)
        nn.init.xavier_uniform_(self.v_projection.weight)

    def forward(
        self,
        x_BRE: torch.Tensor,
        single_eval_pos: int,
        *,
        cached_kv: KVCacheEntry | QuantizedKVCacheEntry | None = None,
        return_kv: bool = False,
    ) -> tuple[torch.Tensor, KVCacheEntry | None]:
        B, R, _ = x_BRE.shape
        q = self.q_projection(x_BRE).view(B, R, self.num_heads, self.head_dim)

        if cached_kv is not None:
            if isinstance(cached_kv, QuantizedKVCacheEntry):
                cached_kv = cached_kv.dequantize(q.dtype)
            k = cached_kv.key
            v = cached_kv.value
            assert k is not None and v is not None
            if k.dtype != q.dtype:
                k = k.to(q.dtype)
                v = v.to(q.dtype)
            out = _batched_scaled_dot_product_attention(
                q,
                k,
                v,
                softmax_scaling_layer=self.softmax_scaling_layer,
            )
        else:
            N = R if single_eval_pos is None else single_eval_pos
            x_train = x_BRE[:, :N]
            k = self.k_projection(x_train).view(B, N, self.num_kv_heads, self.head_dim)
            v = self.v_projection(x_train).view(B, N, self.num_kv_heads, self.head_dim)
            if self.num_kv_heads_test is not None and N < R:
                out_train = _batched_scaled_dot_product_attention(
                    q[:, :N],
                    k,
                    v,
                    softmax_scaling_layer=self.softmax_scaling_layer,
                )
                nh_test_heads = self.num_kv_heads_test
                out_test = _batched_scaled_dot_product_attention(
                    q[:, N:],
                    k[:, :, :nh_test_heads],
                    v[:, :, :nh_test_heads],
                    softmax_scaling_layer=self.softmax_scaling_layer,
                )
                out = torch.cat([out_train, out_test], dim=1)
            else:
                out = _batched_scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    softmax_scaling_layer=self.softmax_scaling_layer,
                )

        result = self.out_projection(out.reshape(B, R, self.head_dim * self.num_heads))
        kv_entry: KVCacheEntry | None = None
        if return_kv:
            k_cache, v_cache = k, v
            if self.num_kv_heads_test is not None:
                nh_test_heads = self.num_kv_heads_test
                k_cache = k_cache[:, :, :nh_test_heads].contiguous()
                v_cache = v_cache[:, :, :nh_test_heads].contiguous()
            kv_entry = KVCacheEntry(key=k_cache.detach(), value=v_cache.detach())
        return result, kv_entry


class CrossAttentionBlock(nn.Module):
    """Cross-attention block with pre-norm and MLP."""

    def __init__(
        self,
        *,
        emsize: int,
        nhead: int,
        dim_feedforward: int,
        norm_factory: Callable[[int], nn.Module],
        softmax_scaling_layer: nn.Module | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ) -> None:
        super().__init__()
        kw = {"device": device, "dtype": dtype}
        self.attn = CrossAttention(
            embedding_size=emsize,
            num_heads=nhead,
            head_dim=emsize // nhead,
            softmax_scaling_layer=softmax_scaling_layer,
            **kw,
        )
        self.mlp = MLP(emsize, dim_feedforward, **kw)
        self.layernorm_q = norm_factory(emsize)
        self.layernorm_kv = norm_factory(emsize)
        self.layernorm2 = norm_factory(emsize)

    def forward(self, x_BQE: torch.Tensor, context_BVE: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.layernorm_q(x_BQE), self.layernorm_kv(context_BVE))
        x_BQE = x_BQE + attn_out
        return x_BQE + self.mlp(self.layernorm2(x_BQE))


class TransformerBlock(nn.Module):
    """Transformer block used by ColumnAggregator."""

    def __init__(
        self,
        *,
        emsize: int,
        nhead: int,
        dim_feedforward: int,
        norm_factory: Callable[[int], nn.Module],
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ) -> None:
        super().__init__()
        kw = {"device": device, "dtype": dtype}
        self.attention = Attention(
            embedding_size=emsize,
            num_heads=nhead,
            head_dim=emsize // nhead,
            **kw,
        )
        self.layernorm = norm_factory(emsize)
        self.layernorm_mlp = norm_factory(emsize)
        self.mlp = MLP(emsize, dim_feedforward, **kw)

    def forward(
        self,
        x_BRCE: torch.Tensor,
        rope: RotaryEmbedding | None = None,
        save_peak_memory_factor: int | None = None,
    ) -> torch.Tensor:
        x_BRCE = chunked_evaluate_maybe_inplace(
            lambda x, rope=None: self.attention(self.layernorm(x), rope=rope),
            x_BRCE,
            save_peak_memory_factor=save_peak_memory_factor,
            residual=True,
            batch_dims=2,
            rope=rope,
        )
        return chunked_evaluate_maybe_inplace(
            lambda x: self.mlp(self.layernorm_mlp(x)),
            x_BRCE,
            save_peak_memory_factor=save_peak_memory_factor,
            residual=True,
            batch_dims=3,
        )

    def forward_cross(
        self,
        query_BRQE: torch.Tensor,
        context_BRCE: torch.Tensor,
        rope: RotaryEmbedding | None = None,
    ) -> torch.Tensor:
        B, R, Q, _ = query_BRQE.shape
        _, _, V, E = context_BRCE.shape
        q_flat = self.layernorm(query_BRQE).view(B * R, Q, E)
        c_flat = self.layernorm(context_BRCE).view(B * R, V, E)
        q_proj = self.attention.q_projection(q_flat).view(
            B * R, Q, -1, self.attention.head_dim
        )
        k_flat = self.attention.k_projection(c_flat).view(
            B * R, V, -1, self.attention.head_dim
        )
        v_flat = self.attention.v_projection(c_flat).view(
            B * R, V, -1, self.attention.head_dim
        )
        if rope is not None:
            q_proj = rope.rotate_queries_or_keys(q_proj.transpose(1, 2)).transpose(1, 2)
            k_flat = rope.rotate_queries_or_keys(k_flat.transpose(1, 2)).transpose(1, 2)
        attn_out = _batched_scaled_dot_product_attention(q_proj, k_flat, v_flat)
        attn_out = attn_out.reshape(
            B * R, Q, self.attention.head_dim * self.attention.num_heads
        )
        attn_out = self.attention.out_projection(attn_out).view(B, R, Q, E)
        x_out = query_BRQE + attn_out
        return x_out + self.mlp(self.layernorm_mlp(x_out))


class ICLTransformerBlock(nn.Module):
    """ICL transformer block with train-only keys."""

    def __init__(
        self,
        *,
        emsize: int,
        nhead: int,
        dim_feedforward: int,
        norm_factory: Callable[[int], nn.Module],
        softmax_scaling_layer: nn.Module | None = None,
        num_kv_heads: int | None = None,
        num_kv_heads_test: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ) -> None:
        super().__init__()
        kw = {"device": device, "dtype": dtype}
        self.icl_attention = ICLAttention(
            embedding_size=emsize,
            num_heads=nhead,
            head_dim=emsize // nhead,
            softmax_scaling_layer=softmax_scaling_layer,
            num_kv_heads=num_kv_heads,
            num_kv_heads_test=num_kv_heads_test,
            **kw,
        )
        self.layernorm = norm_factory(emsize)
        self.layernorm_mlp = norm_factory(emsize)
        self.mlp = MLP(emsize, dim_feedforward, **kw)

    def forward(
        self,
        x_BRE: torch.Tensor,
        single_eval_pos: int,
        save_peak_memory_factor: int | None = None,
        *,
        cached_kv: KVCacheEntry | QuantizedKVCacheEntry | None = None,
        return_kv: bool = False,
    ) -> tuple[torch.Tensor, KVCacheEntry | None]:
        kv_entry: KVCacheEntry | None = None
        if return_kv:
            attn_out, kv_entry = self.icl_attention(
                self.layernorm(x_BRE),
                single_eval_pos=single_eval_pos,
                return_kv=True,
            )
            x_BRE = x_BRE + attn_out
        elif cached_kv is not None:
            def _attn_fn_cached(
                x: torch.Tensor, single_eval_pos: int | None = None
            ) -> torch.Tensor:
                out, _ = self.icl_attention(
                    self.layernorm(x),
                    single_eval_pos=single_eval_pos,
                    cached_kv=cached_kv,
                )
                return out

            x_BRE = chunked_evaluate_maybe_inplace(
                _attn_fn_cached,
                x_BRE,
                save_peak_memory_factor=save_peak_memory_factor,
                residual=True,
                batch_dims=1,
                single_eval_pos=single_eval_pos,
            )
        else:
            def _attn_fn(
                x: torch.Tensor, single_eval_pos: int | None = None
            ) -> torch.Tensor:
                out, _ = self.icl_attention(
                    self.layernorm(x),
                    single_eval_pos=single_eval_pos,
                )
                return out

            x_BRE = chunked_evaluate_maybe_inplace(
                _attn_fn,
                x_BRE,
                save_peak_memory_factor=save_peak_memory_factor,
                residual=True,
                batch_dims=1,
                single_eval_pos=single_eval_pos,
            )

        x_BRE = chunked_evaluate_maybe_inplace(
            lambda x: self.mlp(self.layernorm_mlp(x)),
            x_BRE,
            save_peak_memory_factor=save_peak_memory_factor,
            residual=True,
            batch_dims=2,
        )
        return x_BRE, kv_entry


class InducedSelfAttentionBlock(nn.Module):
    """SetTransformer-style induced self-attention block."""

    def __init__(
        self,
        *,
        emsize: int,
        nhead: int,
        num_inducing_points: int,
        dim_feedforward: int,
        norm_factory: Callable[[int], nn.Module],
        softmax_scaling_layer: nn.Module | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ) -> None:
        super().__init__()
        kw = {"device": device, "dtype": dtype}
        block_kw = {
            "emsize": emsize,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "norm_factory": norm_factory,
            **kw,
        }
        self.cross_attn_block1 = CrossAttentionBlock(
            **block_kw, softmax_scaling_layer=softmax_scaling_layer
        )
        self.cross_attn_block2 = CrossAttentionBlock(**block_kw)
        self.num_inducing_points = num_inducing_points
        self.inducing_vectors = nn.Parameter(torch.empty(num_inducing_points, emsize))
        nn.init.trunc_normal_(self.inducing_vectors, std=0.02)

    def _induced_attention(
        self,
        x_BcRE: torch.Tensor,
        single_eval_pos: int | None = None,
        cached_hidden: torch.Tensor | None = None,
        *,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if cached_hidden is not None:
            hidden = cached_hidden.to(x_BcRE.dtype)
        else:
            Bc, R, _ = x_BcRE.shape
            N = R if single_eval_pos is None else single_eval_pos
            ind = self.inducing_vectors.unsqueeze(0).expand(Bc, -1, -1)
            hidden = self.cross_attn_block1(ind, x_BcRE[:, :N])
        out = self.cross_attn_block2(x_BcRE, hidden)
        if return_hidden:
            return out, hidden.detach()
        return out

    def forward(
        self,
        x_BRCE: torch.Tensor,
        single_eval_pos: int | None = None,
        save_peak_memory_factor: int | None = None,
        *,
        cached_hidden: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, R, C, E = x_BRCE.shape
        x_BcRE = x_BRCE.transpose(1, 2).contiguous().reshape(B * C, R, E)
        if return_hidden:
            out_BcRE, hidden = self._induced_attention(
                x_BcRE,
                single_eval_pos=single_eval_pos,
                return_hidden=True,
            )
        else:
            out_BcRE = chunked_evaluate_maybe_inplace(
                self._induced_attention,
                x_BcRE,
                save_peak_memory_factor,
                residual=False,
                batch_dims=1,
                single_eval_pos=single_eval_pos,
                cached_hidden=cached_hidden,
            )
            hidden = None
        out_BCRE = out_BcRE.reshape(B, C, R, E)
        return out_BCRE.transpose(1, 2).contiguous(), hidden


class FeatureDistributionEmbedder(nn.Module):
    """Stack of induced self-attention layers applied per column."""

    def __init__(
        self,
        *,
        emsize: int,
        nhead: int,
        num_inducing_points: int,
        dim_feedforward: int,
        num_layers: int,
        norm_factory: Callable[[int], nn.Module],
        softmax_scaling_layer_factory: Callable[[], nn.Module] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            InducedSelfAttentionBlock(
                emsize=emsize,
                nhead=nhead,
                num_inducing_points=num_inducing_points,
                dim_feedforward=dim_feedforward,
                norm_factory=norm_factory,
                softmax_scaling_layer=(
                    softmax_scaling_layer_factory()
                    if softmax_scaling_layer_factory is not None
                    else None
                ),
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        )

    def forward(
        self,
        x_BRiCE: torch.Tensor,
        num_train_rows: int | None = None,
        save_peak_memory_factor: int | None = None,
        *,
        force_recompute_layer: bool = False,
        cached_hidden: list[torch.Tensor] | None = None,
        return_hidden: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        hidden_states: list[torch.Tensor] | None = [] if return_hidden else None
        if return_hidden and force_recompute_layer:
            raise ValueError("return_hidden is incompatible with force_recompute_layer")
        for i, layer in enumerate(self.layers):
            if force_recompute_layer:
                x_BRiCE, _ = torch.utils.checkpoint.checkpoint(
                    layer,
                    x_BRiCE,
                    num_train_rows,
                    use_reentrant=False,
                    save_peak_memory_factor=save_peak_memory_factor,
                )
            else:
                layer_cached = cached_hidden[i] if cached_hidden is not None else None
                x_BRiCE, h = layer(
                    x_BRiCE,
                    single_eval_pos=num_train_rows,
                    save_peak_memory_factor=save_peak_memory_factor,
                    cached_hidden=layer_cached,
                    return_hidden=return_hidden,
                )
                if hidden_states is not None:
                    assert h is not None
                    hidden_states.append(h)
        return x_BRiCE, hidden_states


class ColumnAggregator(nn.Module):
    """CLS-token feature aggregator."""

    def __init__(
        self,
        emsize: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        num_cls_tokens: int,
        *,
        norm_factory: Callable[[int], nn.Module],
        use_rope: bool = True,
        rope_base: float = 100_000,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = emsize
        self.num_cls_tokens = num_cls_tokens
        kw = {"device": device, "dtype": dtype}
        self.blocks = nn.ModuleList(
            TransformerBlock(
                emsize=emsize,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                norm_factory=norm_factory,
                **kw,
            )
            for _ in range(num_layers)
        )
        self.rope = (
            RotaryEmbedding(dim=emsize // nhead, theta=int(rope_base), interleaved=False)
            if use_rope
            else None
        )
        self.cls_tokens = nn.Parameter(torch.empty(num_cls_tokens, emsize))
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)
        self.out_ln = norm_factory(emsize)

    def forward(
        self,
        x_BRiCE: torch.Tensor,
        save_peak_memory_factor: int | None = None,
        force_recompute_layer: bool = False,
    ) -> torch.Tensor:
        B, Ri, _, E = x_BRiCE.shape
        cls = self.cls_tokens.expand(B, Ri, self.num_cls_tokens, E).to(
            x_BRiCE.device
        )
        x = torch.cat((cls, x_BRiCE), dim=2)
        for block in self.blocks[:-1]:
            if force_recompute_layer:
                x = torch.utils.checkpoint.checkpoint(
                    block,
                    x,
                    self.rope,
                    save_peak_memory_factor,
                    use_reentrant=False,
                )
            else:
                x = block(
                    x,
                    rope=self.rope,
                    save_peak_memory_factor=save_peak_memory_factor,
                )
        last_block = cast(TransformerBlock, self.blocks[-1])
        cls_part = x[..., : self.num_cls_tokens, :]
        if force_recompute_layer:
            cls_out = torch.utils.checkpoint.checkpoint(
                last_block.forward_cross,
                cls_part,
                x,
                self.rope,
                use_reentrant=False,
            )
        else:
            cls_out = last_block.forward_cross(cls_part, x, self.rope)
        return self.out_ln(cls_out)


class TabPFNV3(nn.Module):
    """Single-file TabPFN v3 architecture with checkpoint-compatible keys."""

    def __init__(
        self,
        *,
        config: TabPFNV3Config,
        task_type: TaskType = "multiclass",
        n_out: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ):
        super().__init__()
        self.config = config
        self.ff_factor = config.ff_factor
        self.icl_emsize = config.embed_dim * config.feat_agg_num_cls_tokens
        self.n_out = n_out
        self.task_type: TaskType = task_type
        self.feature_group_size = config.feature_group_size
        self.use_nan_indicators = config.use_nan_indicators
        kw = {"device": device, "dtype": dtype}

        norm_factory = partial(
            _DtypeMatchingRMSNorm,
            elementwise_affine=config.layernorm_elementwise_affine,
            device=device,
            dtype=dtype,
        )

        in_features = config.feature_group_size
        if self.use_nan_indicators:
            in_features *= 2
        self.x_embed = nn.Linear(in_features, config.embed_dim, **kw)

        if task_type == "multiclass":
            self.col_y_encoder = TrainableOrthogonalEmbedding(
                config.max_num_classes,
                config.embed_dim,
            )
        else:
            self.col_y_encoder = nn.Linear(1, config.embed_dim, **kw)

        self.feature_distribution_embedder = FeatureDistributionEmbedder(
            emsize=config.embed_dim,
            nhead=config.dist_embed_num_heads,
            num_layers=config.dist_embed_num_blocks,
            num_inducing_points=config.dist_embed_num_inducing_points,
            dim_feedforward=config.embed_dim * config.ff_factor,
            norm_factory=norm_factory,
            softmax_scaling_layer_factory=lambda: SoftmaxScalingMLP(
                num_heads=config.dist_embed_num_heads,
                head_dim=config.embed_dim // config.dist_embed_num_heads,
                n_hidden=config.softmax_scaling_mlp_hidden_dim,
            ),
            **kw,
        )

        self.column_aggregator = ColumnAggregator(
            emsize=config.embed_dim,
            nhead=config.feat_agg_num_heads,
            num_layers=config.feat_agg_num_blocks,
            num_cls_tokens=config.feat_agg_num_cls_tokens,
            dim_feedforward=config.embed_dim * config.ff_factor,
            norm_factory=norm_factory,
            use_rope=config.use_rope,
            rope_base=config.feat_agg_rope_base,
            **kw,
        )

        if task_type == "multiclass":
            self.icl_y_encoder: nn.Module = TrainableOrthogonalEmbedding(
                config.max_num_classes,
                self.icl_emsize,
            )
        else:
            self.icl_y_encoder = nn.Linear(1, self.icl_emsize, **kw)

        self.icl_blocks = nn.ModuleList(
            ICLTransformerBlock(
                emsize=self.icl_emsize,
                nhead=config.icl_num_heads,
                dim_feedforward=self.icl_emsize * config.ff_factor,
                norm_factory=norm_factory,
                num_kv_heads=config.icl_num_kv_heads,
                num_kv_heads_test=config.icl_num_kv_heads_test,
                softmax_scaling_layer=SoftmaxScalingMLP(
                    num_heads=config.icl_num_heads,
                    head_dim=self.icl_emsize // config.icl_num_heads,
                    n_hidden=config.softmax_scaling_mlp_hidden_dim,
                ),
                **kw,
            )
            for _ in range(config.nlayers)
        )

        self.output_norm = norm_factory(self.icl_emsize)
        if task_type == "multiclass":
            decoder_softmax_scaling = (
                SoftmaxScalingMLP(
                    num_heads=config.decoder_num_heads,
                    head_dim=config.decoder_head_dim,
                    n_hidden=config.softmax_scaling_mlp_hidden_dim,
                )
                if config.decoder_use_softmax_scaling
                else None
            )
            self.many_class_decoder = ManyClassDecoder(
                max_num_classes=config.max_num_classes,
                input_size=self.icl_emsize,
                head_dim=config.decoder_head_dim,
                num_heads=config.decoder_num_heads,
                softmax_scaling_layer=decoder_softmax_scaling,
            )
        else:
            self.output_projection = nn.Sequential(
                nn.Linear(self.icl_emsize, self.icl_emsize * config.ff_factor, **kw),
                nn.GELU(),
                nn.Linear(self.icl_emsize * config.ff_factor, n_out, **kw),
            )

        self.register_buffer(
            "regression_borders",
            _spline_based_regression_borders(config.num_buckets),
        )
        self.standard_scaler = TorchStandardScaler()
        self._nan_safe_output = True
        self.emsize = config.embed_dim
        self.inference_row_chunk_size = config.inference_row_chunk_size
        self.inference_col_chunk_size = config.inference_col_chunk_size

    @property
    def embedding_dim(self) -> int:
        return self.icl_emsize

    def get_default_performance_options(self) -> PerformanceOptions:
        return PerformanceOptions(use_chunkwise_inference=True)

    def forward(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        y: torch.Tensor | dict[str, torch.Tensor] | None,
        *,
        only_return_standard_out: bool = True,
        categorical_inds: list[list[int]] | None = None,
        performance_options: PerformanceOptions | None = None,
        task_type: str | None = None,
        kv_cache: TabPFNV3Cache | None = None,
        return_kv_cache: bool = False,
        x_is_test_only: bool = False,
        test_targets_MB: torch.Tensor | None = None,
    ) -> (
        torch.Tensor
        | dict[str, torch.Tensor]
        | tuple[torch.Tensor | dict[str, torch.Tensor], TabPFNV3Cache | None]
    ):
        del categorical_inds
        del task_type
        del test_targets_MB
        if isinstance(x, dict):
            x = x["main"]
        if isinstance(y, dict):
            y = y["main"]
        if y is None:
            y = torch.zeros(0, device=x.device, dtype=x.dtype)
        if y.dim() == 3 and y.shape[-1] == 1:
            y = y.squeeze(-1)
        if performance_options is None:
            performance_options = self.get_default_performance_options()

        if x_is_test_only and (kv_cache is None or kv_cache.is_empty()):
            raise ValueError("x_is_test_only=True requires a non-empty kv_cache")

        if (
            not self.training
            and self.task_type == "multiclass"
            and y.numel() > 0
            and (y > self.n_out - 1).any()
        ):
            raise ValueError(
                f"Target values must be between 0 and {self.n_out - 1}."
            )

        x_RiBC = x
        B = x_RiBC.shape[1]
        num_train = y.shape[0]
        x_BRiClE, inducing_hidden, scaler_stats = self._stages_0_to_2(
            x_RiBC,
            y,
            performance_options=performance_options,
            return_inducing_hidden=return_kv_cache,
            kv_cache=kv_cache,
            x_is_test_only=x_is_test_only,
        )

        x_BRiD = x_BRiClE.flatten(-2)
        kv_out: dict[int, KVCacheEntry | QuantizedKVCacheEntry] = {}
        if kv_cache is not None and not kv_cache.is_empty():
            for layer_idx, block in enumerate(self.icl_blocks):
                x_BRiD, _ = block(
                    x_BRiD,
                    0,
                    performance_options.save_peak_memory_factor,
                    cached_kv=kv_cache.kv[layer_idx],
                )
        else:
            if num_train > 0:
                y_icl = self._prepare_y(y, num_train, B)
                x_BRiD[:, :num_train] = x_BRiD[:, :num_train] + self._embed_icl_y(
                    y_icl
                )
            if return_kv_cache:
                for layer_idx, block in enumerate(self.icl_blocks):
                    x_BRiD, kv_entry = block(
                        x_BRiD,
                        num_train,
                        performance_options.save_peak_memory_factor,
                        return_kv=True,
                    )
                    assert kv_entry is not None
                    kv_out[layer_idx] = kv_entry
            else:
                for block in self.icl_blocks:
                    if performance_options.force_recompute_layer:
                        x_BRiD, _ = torch.utils.checkpoint.checkpoint(
                            block,
                            x_BRiD,
                            num_train,
                            use_reentrant=False,
                            save_peak_memory_factor=performance_options.save_peak_memory_factor,
                        )
                    else:
                        x_BRiD, _ = block(
                            x_BRiD,
                            num_train,
                            performance_options.save_peak_memory_factor,
                        )

        x_BRiD = self.output_norm(x_BRiD)
        if kv_cache is not None and not kv_cache.is_empty():
            test_emb = x_BRiD
            train_emb = kv_cache.train_embeddings
            assert train_emb is not None
        else:
            test_emb = x_BRiD[:, num_train:]
            train_emb = x_BRiD[:, :num_train]

        built_cache: TabPFNV3Cache | None = None
        if return_kv_cache:
            if kv_cache is not None and not kv_cache.is_empty():
                built_cache = kv_cache
            else:
                assert kv_out
                first_entry = next(iter(kv_out.values()))
                cache_key = (
                    first_entry.dequantize(torch.float32).key
                    if isinstance(first_entry, QuantizedKVCacheEntry)
                    else first_entry.key
                )
                assert cache_key is not None
                built_cache = TabPFNV3Cache(
                    kv=kv_out,
                    train_embeddings=train_emb.detach().to(cache_key.dtype),
                    train_shape=(B, num_train),
                    scaler_cache={k: v.detach() for k, v in scaler_stats.items()},
                    inducing_hidden=(
                        [h.detach() for h in inducing_hidden]
                        if inducing_hidden is not None
                        else None
                    ),
                )

        if self.task_type == "multiclass":
            y_BN = y.transpose(0, 1) if y.dim() == 2 else y.unsqueeze(0)
            test_out = self.many_class_decoder(train_emb, test_emb, y_BN[:, :num_train])
        else:
            test_out = self.output_projection(test_emb.transpose(0, 1))

        if self._nan_safe_output:
            test_out = torch.nan_to_num(test_out, nan=0.0)

        if only_return_standard_out:
            output: torch.Tensor | dict[str, torch.Tensor] = test_out
        else:
            output = {
                "standard": test_out,
                "train_embeddings": train_emb.transpose(0, 1),
                "test_embeddings": test_emb.transpose(0, 1),
            }
        if return_kv_cache:
            return output, built_cache
        return output

    def _prepare_y(
        self,
        y: torch.Tensor,
        num_train: int,
        batch_size: int,
    ) -> torch.Tensor:
        if num_train == 0:
            raise ValueError("No training rows available for target embedding.")
        y_NB1 = _prepare_targets(y, num_train, batch_size)[:num_train]
        y_NB1 = _impute_target_nan_and_inf(y_NB1, self.task_type, num_train)
        return y_NB1.squeeze(-1).transpose(0, 1)

    def _embed_col_y(self, y_BN: torch.Tensor) -> torch.Tensor:
        if self.task_type == "multiclass":
            return self.col_y_encoder(y_BN)
        return self.col_y_encoder(y_BN.unsqueeze(-1))

    def _embed_icl_y(self, y_BN: torch.Tensor) -> torch.Tensor:
        if self.task_type == "multiclass":
            return self.icl_y_encoder(y_BN)
        return self.icl_y_encoder(y_BN.unsqueeze(-1))

    def _preprocess_raw(
        self,
        x_RiBC: torch.Tensor,
        num_train: int,
        scaler_cache: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        nan_ind_BRiC: torch.Tensor | None = None
        if self.use_nan_indicators:
            nan_ind_BRiC = _generate_nan_and_inf_indicator(x_RiBC).transpose(0, 1)
        x_RiBC, _ = _impute_nan_and_inf_with_mean(x_RiBC, num_train, scaler_cache)
        if scaler_cache is None:
            fit_data = x_RiBC[:num_train] if num_train > 0 else x_RiBC
            scaler_cache = self.standard_scaler.fit(fit_data)
        x_RiBC = self.standard_scaler.transform(x_RiBC, fitted_cache=scaler_cache)
        return x_RiBC.transpose(0, 1), nan_ind_BRiC, scaler_cache

    def _group_features(
        self,
        x_BRiC: torch.Tensor,
        nan_ind_BRiC: torch.Tensor | None,
    ) -> torch.Tensor:
        size = self.feature_group_size
        x_grouped = [torch.roll(x_BRiC, shifts=-(2**i), dims=2) for i in range(size)]
        x_grouped_t = torch.stack(x_grouped, dim=-1)
        if nan_ind_BRiC is not None:
            ind_grouped = [
                torch.roll(nan_ind_BRiC, shifts=-(2**i), dims=2)
                for i in range(size)
            ]
            x_grouped_t = torch.cat([x_grouped_t, torch.stack(ind_grouped, dim=-1)], -1)
        return x_grouped_t

    def _compute_all_inducing_hidden(
        self,
        dist_embedder_layers: nn.ModuleList,
        x_grouped_BRiCG: torch.Tensor,
        num_train: int,
        y_col_emb_BNE: torch.Tensor | None,
        col_chunk_size: int,
        *,
        enable_torch_compile: bool,
    ) -> list[torch.Tensor]:
        del enable_torch_compile
        num_columns = x_grouped_BRiCG.shape[2]
        hidden_per_block: list[list[torch.Tensor]] = [
            [] for _ in range(len(dist_embedder_layers))
        ]
        x_grouped_train_BNCG = x_grouped_BRiCG[:, :num_train]
        for c0 in range(0, num_columns, col_chunk_size):
            c1 = min(c0 + col_chunk_size, num_columns)
            chunk_outputs_BCjIE = self._process_col_chunk(
                x_grouped_chunk_BNCjG=x_grouped_train_BNCG[:, :, c0:c1],
                y_col_emb_BNE=y_col_emb_BNE,
                num_train=num_train,
            )
            for blk_idx, h in enumerate(chunk_outputs_BCjIE):
                hidden_per_block[blk_idx].append(h)
        return [torch.cat(chunks, dim=1).flatten(0, 1) for chunks in hidden_per_block]

    def _compiled(self, method: Callable) -> Callable:
        return torch.compile(method, dynamic=True)

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state.pop("_torch_compile_cache", None)
        return state

    def _preprocess_and_group(
        self,
        rows_RiBC: torch.Tensor,
        y: torch.Tensor,
        num_train: int,
        scaler_cache: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        B = rows_RiBC.shape[1]
        x_BRiC, nan_ind_BRiC, scaler_stats = self._preprocess_raw(
            rows_RiBC, num_train, scaler_cache
        )
        y_col_emb_BNE: torch.Tensor | None = None
        if scaler_cache is None and num_train > 0:
            y_col_BN = self._prepare_y(y, num_train, B)
            y_col_emb_BNE = self._embed_col_y(y_col_BN)
        return self._group_features(x_BRiC, nan_ind_BRiC), y_col_emb_BNE, scaler_stats

    def _stages_0_to_2(
        self,
        x_RiBC: torch.Tensor,
        y: torch.Tensor,
        *,
        performance_options: PerformanceOptions,
        return_inducing_hidden: bool,
        kv_cache: TabPFNV3Cache | None,
        x_is_test_only: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None, dict[str, torch.Tensor]]:
        num_train = y.shape[0]
        if performance_options.use_chunkwise_inference and not self.training:
            row_chunk_size = self.inference_row_chunk_size
            col_chunk_size = self.inference_col_chunk_size
        else:
            row_chunk_size = None
            col_chunk_size = None

        if kv_cache is not None and not kv_cache.is_empty():
            rows_RiBC = x_RiBC if x_is_test_only else x_RiBC[num_train:]
            scaler_cache = kv_cache.scaler_cache
            precomputed_hidden = kv_cache.inducing_hidden
            effective_num_train = 0
        else:
            rows_RiBC = x_RiBC
            scaler_cache = None
            precomputed_hidden = None
            effective_num_train = num_train

        preprocess_fn = (
            self._compiled(self._preprocess_and_group)
            if performance_options.enable_torch_compile
            else self._preprocess_and_group
        )
        x_grouped_BRiCG, y_col_emb_BNE, scaler_stats = preprocess_fn(
            rows_RiBC, y, num_train, scaler_cache
        )
        num_rows, C = x_grouped_BRiCG.shape[1], x_grouped_BRiCG.shape[2]
        use_chunks = row_chunk_size is not None and row_chunk_size < num_rows

        if use_chunks and precomputed_hidden is None:
            eff_col_chunk = col_chunk_size if col_chunk_size is not None else C
            while True:
                try:
                    precomputed_hidden = self._compute_all_inducing_hidden(
                        self.feature_distribution_embedder.layers,
                        x_grouped_BRiCG,
                        num_train,
                        y_col_emb_BNE,
                        eff_col_chunk,
                        enable_torch_compile=performance_options.enable_torch_compile,
                    )
                    break
                except RuntimeError as exc:
                    if not is_oom_error(exc) or eff_col_chunk <= 1:
                        raise
                    torch.cuda.empty_cache()
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    eff_col_chunk //= 2
                    self.inference_col_chunk_size = eff_col_chunk
                    _logger.warning("OOM: halving col_chunk_size to %d", eff_col_chunk)

        is_full_path = not use_chunks and precomputed_hidden is None
        effective_chunk_size = row_chunk_size if use_chunks else num_rows
        process_row_chunk = (
            self._compiled(self._process_row_chunk)
            if performance_options.enable_torch_compile
            else self._process_row_chunk
        )
        while True:
            parts: list[torch.Tensor] = []
            inducing_hidden: list[torch.Tensor] | None = None
            try:
                for row_chunk_start in range(0, num_rows, effective_chunk_size):
                    row_chunk_end = min(row_chunk_start + effective_chunk_size, num_rows)
                    row_embedding_chunk, chunk_hidden = process_row_chunk(
                        x_grouped_chunk_BRjCG=x_grouped_BRiCG[
                            :, row_chunk_start:row_chunk_end
                        ],
                        y_col_emb=y_col_emb_BNE,
                        chunk_start=row_chunk_start,
                        chunk_end=row_chunk_end,
                        effective_num_train=effective_num_train,
                        precomputed_hidden=precomputed_hidden,
                        save_peak_memory_factor=performance_options.save_peak_memory_factor,
                        force_recompute_layer=performance_options.force_recompute_layer,
                        return_inducing_hidden=return_inducing_hidden,
                        is_full_path=is_full_path,
                    )
                    if chunk_hidden is not None:
                        inducing_hidden = chunk_hidden
                    parts.append(row_embedding_chunk)
                break
            except RuntimeError as exc:
                if not is_oom_error(exc) or not use_chunks or effective_chunk_size <= 1:
                    raise
                parts.clear()
                torch.cuda.empty_cache()
                effective_chunk_size //= 2
                self.inference_row_chunk_size = effective_chunk_size
                _logger.warning("OOM: halving row_chunk_size to %d", effective_chunk_size)

        if use_chunks:
            inducing_hidden = precomputed_hidden
        x_BRiClE = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
        return x_BRiClE, inducing_hidden, scaler_stats

    def _process_col_chunk(
        self,
        *,
        x_grouped_chunk_BNCjG: torch.Tensor,
        y_col_emb_BNE: torch.Tensor | None,
        num_train: int,
    ) -> list[torch.Tensor]:
        B, _, Cj, _ = x_grouped_chunk_BNCjG.shape
        x_emb_BNCjE = self.x_embed(x_grouped_chunk_BNCjG)
        E = x_emb_BNCjE.shape[-1]
        if y_col_emb_BNE is not None and num_train > 0:
            x_emb_BNCjE = x_emb_BNCjE + y_col_emb_BNE.unsqueeze(2)
        x_flat = x_emb_BNCjE.transpose(1, 2).contiguous().reshape(B * Cj, num_train, E)
        layers = self.feature_distribution_embedder.layers
        chunk_outputs: list[torch.Tensor] = []
        for blk_idx, blk in enumerate(layers):
            ind = blk.inducing_vectors.unsqueeze(0).expand(B * Cj, -1, -1)
            hidden = blk.cross_attn_block1(ind, x_flat)
            chunk_outputs.append(hidden.reshape(B, Cj, -1, E))
            if blk_idx < len(layers) - 1:
                x_flat = blk.cross_attn_block2(x_flat, hidden)
        return chunk_outputs

    def _process_row_chunk(
        self,
        x_grouped_chunk_BRjCG: torch.Tensor,
        y_col_emb: torch.Tensor | None,
        chunk_start: int,
        chunk_end: int,
        effective_num_train: int,
        precomputed_hidden: list[torch.Tensor] | None,
        save_peak_memory_factor: int | None,
        *,
        force_recompute_layer: bool,
        return_inducing_hidden: bool,
        is_full_path: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        row_chunk_range = chunk_end - chunk_start
        num_train_rows = max(0, min(effective_num_train - chunk_start, row_chunk_range))
        x_emb = self.x_embed(x_grouped_chunk_BRjCG)
        if y_col_emb is not None and num_train_rows > 0:
            y_emb = y_col_emb[:, chunk_start : chunk_start + num_train_rows]
            x_emb[:, :num_train_rows] = x_emb[:, :num_train_rows] + y_emb.unsqueeze(2)
        x_emb, chunk_hidden = self.feature_distribution_embedder(
            x_BRiCE=x_emb,
            num_train_rows=num_train_rows,
            cached_hidden=precomputed_hidden,
            save_peak_memory_factor=(save_peak_memory_factor if is_full_path else None),
            force_recompute_layer=force_recompute_layer and is_full_path,
            return_hidden=return_inducing_hidden and is_full_path,
        )
        row_embedding_chunk = self.column_aggregator(
            x_BRiCE=x_emb,
            save_peak_memory_factor=save_peak_memory_factor,
            force_recompute_layer=force_recompute_layer and is_full_path,
        )
        return row_embedding_chunk, chunk_hidden


def parse_config(config: dict[str, Any]) -> tuple[TabPFNV3Config, dict[str, Any]]:
    parsed_config = TabPFNV3Config.from_dict(config)
    return parsed_config, parsed_config.get_unused_config(config)


def get_architecture(
    config: TabPFNV3Config,
    *,
    cache_trainset_representation: bool = False,
) -> TabPFNV3:
    del cache_trainset_representation
    task_type: TaskType = "multiclass" if config.max_num_classes >= 2 else "regression"
    n_out = config.max_num_classes if task_type == "multiclass" else config.num_buckets
    return TabPFNV3(config=config, task_type=task_type, n_out=n_out)


def _prepare_targets(
    y: torch.Tensor,
    num_train_and_test_rows: int,
    batch_size: int,
) -> torch.Tensor:
    num_train_labels = y.shape[0]
    if num_train_labels > num_train_and_test_rows:
        raise ValueError("No test rows provided.")
    target_RBT = y.view(num_train_labels, 1 if y.ndim == 1 else batch_size, -1)
    return F.pad(
        target_RBT,
        (0, 0, 0, 0, 0, num_train_and_test_rows - num_train_labels),
        value=float("nan"),
    )


def _impute_nan_and_inf_with_mean(
    x: torch.Tensor,
    num_train_rows: int,
    scaler_cache: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    is_finite = torch.isfinite(x)
    if num_train_rows == 0 and scaler_cache is None:
        logging.warning("No training rows or scaler cache provided, imputing with 0.")
    if scaler_cache is not None:
        feature_means = scaler_cache["mean"].to(device=x.device, dtype=x.dtype)
    else:
        x_train = torch.where(is_finite[:num_train_rows], x[:num_train_rows], torch.nan)
        feature_means = torch.nan_to_num(torch.nanmean(x_train, dim=0), 0)
    return torch.where(is_finite, x, feature_means.unsqueeze(0).expand_as(x)), is_finite


def _impute_target_nan_and_inf(
    y_NB1: torch.Tensor,
    task_type: TaskType,
    num_train_rows: int,
) -> torch.Tensor:
    y_NB1, is_finite = _impute_nan_and_inf_with_mean(y_NB1, num_train_rows)
    if task_type == "regression":
        return y_NB1
    return torch.where(is_finite, y_NB1, y_NB1.ceil())


_NAN_INDICATOR = -2.0
_INFINITY_INDICATOR = 2.0
_NEG_INFINITY_INDICATOR = 4.0


def _generate_nan_and_inf_indicator(x: torch.Tensor) -> torch.Tensor:
    return (
        torch.isnan(x) * _NAN_INDICATOR
        + torch.isposinf(x) * _INFINITY_INDICATOR
        + torch.isneginf(x) * _NEG_INFINITY_INDICATOR
    ).to(x.dtype)


def _safe_log_seqlen(
    n: int | torch.Tensor, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if isinstance(n, torch.Tensor):
        return n.to(torch.float32).clamp(min=1).log().to(dtype)
    one = torch.ones((), dtype=torch.float32, device=device)
    return (one * n).clamp(min=1).log().to(dtype)


def _spline_based_regression_borders(num_buckets: int) -> torch.Tensor:
    border_reference_points = [
        (0, -128),
        (5, -16.9),
        (20, -13),
        (100, -9.9),
        (200, -8.47),
        (500, -6.48),
        (1000, -4.40),
    ]
    border_reference_points = (
        border_reference_points
        + [(2500, 0)]
        + [(5000 - x, -y) for x, y in border_reference_points[::-1]]
    )
    x_scale = num_buckets / 5000
    xp = np.array([x for x, _ in border_reference_points]) * x_scale
    yp = np.array([y for _, y in border_reference_points])
    return torch.tensor(
        np.interp(x=np.arange(num_buckets + 1), xp=xp, fp=yp),
        dtype=torch.float32,
    )


# ---------------------------------------------------------------------------
# Single-file checkpoint, scratch-training, and CLI utilities
# ---------------------------------------------------------------------------


ARCHITECTURE_NAME = "starlight_core_regressor"


@dataclasses.dataclass(frozen=True)
class CheckpointMetadata:
    """Top-level metadata and state-dict receipt for a checkpoint."""

    path: Path
    sha256: str
    architecture_name: str | None
    config: dict[str, Any]
    tensor_count: int
    parameter_count: int
    dtype_counts: dict[str, int]


@dataclasses.dataclass(frozen=True)
class SyntheticTaskBatch:
    """A batch of independent tabular classification tasks."""

    x: torch.Tensor
    y_train: torch.Tensor
    y_test: torch.Tensor
    num_classes: int


@dataclasses.dataclass
class ScratchTrainConfig:
    """Config for the local train-from-scratch harness."""

    steps: int = 10
    train_rows: int = 16
    test_rows: int = 8
    batch_size: int = 1
    num_features: int = 8
    min_classes: int = 2
    max_classes: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"
    amp: bool = False
    log_every: int = 1
    save_every: int = 0
    output_dir: Path = Path("artifacts/train")
    resume: Path | None = None


def _safe_load(path: str | Path, *, map_location: str | torch.device = "cpu") -> Any:
    """Load a checkpoint without unpickling arbitrary Python objects."""

    return torch.load(path, map_location=map_location, weights_only=True)


def _extract_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    if not isinstance(obj, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(obj)!r}")
    state_dict = obj.get("state_dict", obj.get("model_state_dict"))
    if state_dict is None:
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            state_dict = obj
        else:
            raise KeyError("Checkpoint does not contain state_dict/model_state_dict")
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected state_dict dict, got {type(state_dict)!r}")
    return state_dict


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_state_dict(path: str | Path) -> dict[str, Any]:
    """Return state-dict counts and key/shape details for *path*."""

    obj = _safe_load(path)
    state_dict = _extract_state_dict(obj)
    dtype_counts: dict[str, int] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    parameter_count = 0
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        dtype_counts[str(tensor.dtype)] = dtype_counts.get(str(tensor.dtype), 0) + 1
        shapes[key] = tuple(tensor.shape)
        parameter_count += tensor.numel()
    return {
        "architecture_name": obj.get("architecture_name") if isinstance(obj, dict) else None,
        "tensor_count": len(state_dict),
        "parameter_count": parameter_count,
        "dtype_counts": dtype_counts,
        "shapes": shapes,
    }


def load_checkpoint_metadata(path: str | Path) -> CheckpointMetadata:
    """Load non-mutating checkpoint metadata with a SHA256 receipt."""

    path = Path(path)
    obj = _safe_load(path)
    state_dict = _extract_state_dict(obj)
    dtype_counts: dict[str, int] = {}
    parameter_count = 0
    for tensor in state_dict.values():
        if isinstance(tensor, torch.Tensor):
            dtype_counts[str(tensor.dtype)] = dtype_counts.get(str(tensor.dtype), 0) + 1
            parameter_count += tensor.numel()
    config = obj.get("config", {}) if isinstance(obj, dict) else {}
    if not isinstance(config, dict):
        raise TypeError("Checkpoint config is not a dict")
    return CheckpointMetadata(
        path=path,
        sha256=file_sha256(path),
        architecture_name=obj.get("architecture_name") if isinstance(obj, dict) else None,
        config=config,
        tensor_count=len(state_dict),
        parameter_count=parameter_count,
        dtype_counts=dtype_counts,
    )


def build_model_from_checkpoint(
    path: str | Path,
    *,
    load_weights: bool = True,
    strict: bool = True,
    device: str | torch.device = "cpu",
) -> TabPFNV3:
    """Build a checkpoint-compatible model, optionally strict-loading weights."""

    obj = _safe_load(path, map_location=device)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(obj)!r}")
    config_dict = obj.get("config", {})
    if not isinstance(config_dict, dict):
        raise TypeError("Checkpoint config is not a dict")
    model = get_architecture(TabPFNV3Config.from_dict(config_dict)).to(device)
    if load_weights:
        result = model.load_state_dict(_extract_state_dict(obj), strict=strict)
        if strict and (result.missing_keys or result.unexpected_keys):
            raise RuntimeError(
                "Strict load failed: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )
    return model


def make_checkpoint_payload(
    model: TabPFNV3,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    trained_steps: int = 0,
    trained_epochs: int = 0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a checkpoint payload using the v3 top-level schema."""

    payload: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "scaler_state": None if scaler is None else scaler.state_dict(),
        "config": dataclasses.asdict(model.config),
        "architecture_name": ARCHITECTURE_NAME,
        "inference_config": {},
        "trained_epochs_until_now": trained_epochs,
        "trained_steps_until_now": trained_steps,
        "trained_num_examples_seen_until_now": None,
        "pdr_buffer_state": None,
    }
    if extra:
        payload.update(extra)
    return payload


def save_training_checkpoint(
    path: str | Path,
    model: TabPFNV3,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    trained_steps: int = 0,
    trained_epochs: int = 0,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a trainable checkpoint that can be loaded by this file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        make_checkpoint_payload(
            model,
            optimizer=optimizer,
            scaler=scaler,
            trained_steps=trained_steps,
            trained_epochs=trained_epochs,
            extra=extra,
        ),
        path,
    )


def sample_synthetic_task_batch(
    *,
    train_rows: int,
    test_rows: int,
    batch_size: int,
    num_features: int,
    min_classes: int = 2,
    max_classes: int = 4,
    nan_probability: float = 0.01,
    inf_probability: float = 0.002,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> SyntheticTaskBatch:
    """Sample compatible nonlinear multiclass tabular tasks.

    This is a local training distribution for exercising the full training
    stack. It is not PriorLabs' original private pretraining corpus/curriculum.
    """

    if min_classes < 2:
        raise ValueError("min_classes must be at least 2")
    if max_classes < min_classes:
        raise ValueError("max_classes must be >= min_classes")
    device = torch.device(device)
    num_classes = int(
        torch.randint(
            min_classes,
            max_classes + 1,
            (1,),
            generator=generator,
            device=device,
        ).item()
    )
    rows = train_rows + test_rows
    x = torch.randn(rows, batch_size, num_features, generator=generator, device=device)
    weights = torch.randn(
        batch_size, num_features, num_classes, generator=generator, device=device
    )
    bias = torch.randn(batch_size, num_classes, generator=generator, device=device)
    nonlinear_weights = torch.randn(
        batch_size, num_features, num_classes, generator=generator, device=device
    )
    logits = torch.einsum("rbc,bck->rbk", x, weights) + bias.unsqueeze(0)
    logits = logits + 0.35 * torch.einsum(
        "rbc,bck->rbk", torch.sin(x) + 0.25 * x.square(), nonlinear_weights
    )
    logits = logits + 0.10 * torch.randn(
        rows, batch_size, num_classes, generator=generator, device=device
    )
    labels = logits.argmax(dim=-1)

    if nan_probability > 0:
        nan_mask = torch.rand(x.shape, generator=generator, device=device) < nan_probability
        x = torch.where(nan_mask, torch.full_like(x, float("nan")), x)
    if inf_probability > 0:
        inf_mask = torch.rand(x.shape, generator=generator, device=device) < inf_probability
        signs = torch.where(
            torch.rand(x.shape, generator=generator, device=device) < 0.5,
            torch.full_like(x, -1.0),
            torch.ones_like(x),
        )
        x = torch.where(inf_mask, signs * torch.full_like(x, float("inf")), x)

    return SyntheticTaskBatch(
        x=x,
        y_train=labels[:train_rows],
        y_test=labels[train_rows:],
        num_classes=num_classes,
    )


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def _make_device_generator(device: torch.device, seed: int) -> torch.Generator:
    if device.type == "cpu":
        generator = torch.Generator()
    else:
        generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _loss_from_logits(
    logits_TBC: torch.Tensor,
    targets_TB: torch.Tensor,
    *,
    num_classes: int,
) -> torch.Tensor:
    active_logits = logits_TBC[..., :num_classes].reshape(-1, num_classes)
    return F.cross_entropy(active_logits, targets_TB.reshape(-1).long())


def smoke_train(
    *,
    ckpt: str | Path | None = None,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> dict[str, Any]:
    """Run one small forward/backward/optimizer step."""

    device = torch.device(device)
    torch.manual_seed(seed)
    if ckpt is None:
        config = TabPFNV3Config(decoder_use_softmax_scaling=True)
        model = get_architecture(config).to(device)
    else:
        model = build_model_from_checkpoint(ckpt, load_weights=True, strict=True, device=device)
    model.train()
    generator = _make_device_generator(device, seed)
    batch = sample_synthetic_task_batch(
        train_rows=8,
        test_rows=4,
        batch_size=1,
        num_features=6,
        min_classes=2,
        max_classes=2,
        nan_probability=0.02,
        inf_probability=0.002,
        device=device,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch.x, batch.y_train)
    assert isinstance(logits, torch.Tensor)
    loss = _loss_from_logits(logits, batch.y_test, num_classes=batch.num_classes)
    loss.backward()
    grad_receipts = {
        "decoder_q": _grad_norm(model.many_class_decoder.q_projection.weight),
        "decoder_k": _grad_norm(model.many_class_decoder.k_projection.weight),
        "icl_last_q": _grad_norm(model.icl_blocks[-1].icl_attention.q_projection.weight),
    }
    optimizer.step()
    return {
        "loss": float(loss.detach().cpu()),
        "output_shape": tuple(logits.shape),
        "grad_norms": grad_receipts,
    }


def _grad_norm(parameter: torch.nn.Parameter) -> float:
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().norm().cpu())


def train_from_scratch(train_config: ScratchTrainConfig) -> dict[str, Any]:
    """Train the checkpoint-compatible architecture from random initialization."""

    device = torch.device(train_config.device)
    torch.manual_seed(train_config.seed)
    generator = _make_device_generator(device, train_config.seed)
    dtype = _dtype_from_name(train_config.dtype)
    config = TabPFNV3Config(decoder_use_softmax_scaling=True)
    model = get_architecture(config).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        device=device.type,
        enabled=train_config.amp and device.type == "cuda",
    )
    start_step = 0

    if train_config.resume is not None:
        resume_obj = _safe_load(train_config.resume, map_location=device)
        model.load_state_dict(_extract_state_dict(resume_obj), strict=True)
        if isinstance(resume_obj, dict):
            if resume_obj.get("optimizer_state") is not None:
                optimizer.load_state_dict(resume_obj["optimizer_state"])
            if resume_obj.get("scaler_state") is not None:
                scaler.load_state_dict(resume_obj["scaler_state"])
            start_step = int(resume_obj.get("trained_steps_until_now") or 0)

    train_config.output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = train_config.output_dir / "train_receipts.jsonl"
    model.train()
    last_loss = float("nan")
    started_at = time.time()

    for local_step in range(1, train_config.steps + 1):
        global_step = start_step + local_step
        batch = sample_synthetic_task_batch(
            train_rows=train_config.train_rows,
            test_rows=train_config.test_rows,
            batch_size=train_config.batch_size,
            num_features=train_config.num_features,
            min_classes=train_config.min_classes,
            max_classes=train_config.max_classes,
            device=device,
            generator=generator,
        )
        optimizer.zero_grad(set_to_none=True)
        use_autocast = train_config.amp and device.type in {"cuda", "cpu"}
        autocast_dtype = torch.bfloat16 if dtype is torch.bfloat16 else torch.float16
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
            logits = model(batch.x.to(dtype=dtype), batch.y_train)
            assert isinstance(logits, torch.Tensor)
            loss = _loss_from_logits(logits, batch.y_test, num_classes=batch.num_classes)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {global_step}: {loss.item()}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if train_config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        last_loss = float(loss.detach().cpu())
        if train_config.log_every > 0 and global_step % train_config.log_every == 0:
            receipt = {
                "step": global_step,
                "loss": last_loss,
                "num_classes": batch.num_classes,
                "shape": tuple(logits.shape),
                "elapsed_sec": round(time.time() - started_at, 3),
            }
            with receipt_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(receipt) + "\n")
            print(json.dumps(receipt))
        if train_config.save_every > 0 and global_step % train_config.save_every == 0:
            save_training_checkpoint(
                train_config.output_dir / f"step_{global_step}.ckpt",
                model,
                optimizer=optimizer,
                scaler=scaler,
                trained_steps=global_step,
            )

    final_path = train_config.output_dir / "latest.ckpt"
    save_training_checkpoint(
        final_path,
        model,
        optimizer=optimizer,
        scaler=scaler,
        trained_steps=start_step + train_config.steps,
    )
    return {
        "steps": train_config.steps,
        "trained_steps_until_now": start_step + train_config.steps,
        "last_loss": last_loss,
        "checkpoint": str(final_path),
        "checkpoint_sha256": file_sha256(final_path),
        "receipts": str(receipt_path),
    }


def _parse_train_config(args: argparse.Namespace) -> ScratchTrainConfig:
    return ScratchTrainConfig(
        steps=args.steps,
        train_rows=args.train_rows,
        test_rows=args.test_rows,
        batch_size=args.batch_size,
        num_features=args.num_features,
        min_classes=args.min_classes,
        max_classes=args.max_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        amp=args.amp,
        log_every=args.log_every,
        save_every=args.save_every,
        output_dir=args.output_dir,
        resume=args.resume,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Single-file TabPFN v3 architecture, checkpoint compatibility, "
            "and scratch-training harness."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary_parser = subparsers.add_parser("summary")
    summary_parser.add_argument("--ckpt", required=True, type=Path)
    summary_parser.add_argument("--first-keys", type=int, default=20)

    strict_parser = subparsers.add_parser("strict-load")
    strict_parser.add_argument("--ckpt", required=True, type=Path)
    strict_parser.add_argument("--device", default="cpu")

    smoke_parser = subparsers.add_parser("smoke-train")
    smoke_parser.add_argument("--ckpt", type=Path)
    smoke_parser.add_argument("--device", default="cpu")
    smoke_parser.add_argument("--seed", type=int, default=0)

    train_parser = subparsers.add_parser("train-scratch")
    train_parser.add_argument("--steps", type=int, default=10)
    train_parser.add_argument("--train-rows", type=int, default=16)
    train_parser.add_argument("--test-rows", type=int, default=8)
    train_parser.add_argument("--batch-size", type=int, default=1)
    train_parser.add_argument("--num-features", type=int, default=8)
    train_parser.add_argument("--min-classes", type=int, default=2)
    train_parser.add_argument("--max-classes", type=int, default=4)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-2)
    train_parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
    )
    train_parser.add_argument("--amp", action="store_true")
    train_parser.add_argument("--log-every", type=int, default=1)
    train_parser.add_argument("--save-every", type=int, default=0)
    train_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/train"))
    train_parser.add_argument("--resume", type=Path)

    args = parser.parse_args()
    if args.command == "summary":
        meta = load_checkpoint_metadata(args.ckpt)
        print(f"sha256={meta.sha256}")
        print(f"arch={meta.architecture_name}")
        print(f"tensors={meta.tensor_count}")
        print(f"params={meta.parameter_count}")
        print(f"dtypes={meta.dtype_counts}")
        summary = summarize_state_dict(args.ckpt)
        for key, shape in list(summary["shapes"].items())[: args.first_keys]:
            print(f"{key}: {shape}")
    elif args.command == "strict-load":
        model = build_model_from_checkpoint(args.ckpt, load_weights=True, strict=True, device=args.device)
        print(f"strict_load=ok model_keys={len(model.state_dict())}")
    elif args.command == "smoke-train":
        print(json.dumps(smoke_train(ckpt=args.ckpt, device=args.device, seed=args.seed)))
    elif args.command == "train-scratch":
        print(json.dumps(train_from_scratch(_parse_train_config(args))))
    else:
        raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    _main()
