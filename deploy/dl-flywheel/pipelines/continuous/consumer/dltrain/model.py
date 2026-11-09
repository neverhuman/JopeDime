"""PolicyLM and its canonical identities (PLAN §6.3.1 model, §6.3.2 Optimization, §6.6 checkpoint closure and
determinism controls, §5 P7 tests 1 and 4).

- PolicyLM: a pre-LN causal transformer whose first position is a context prefix token (2-layer context MLP),
  with an LM head tied to ``tok_emb.weight`` and quality / feasibility / runtime heads on ``[h_EOS, z_ctx]``,
  plus an optional context-free head for the baseline-objective arm.
- ``parameter_sha256`` / ``tensor_inventory_sha256``: the closure byte rules (sorted ``named_parameters()``,
  tied alias excluded; inventory algorithm ``sha256-name-nul-dtype-nul-shape-csv-lf-v1``).
- ``param_groups``: the two named AdamW groups (decay 0.01 except norms and biases).
- ``configure_determinism``: the deterministic CPU mode every P7 test and run uses.
"""
from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import dataclass, fields
from typing import Any, Mapping, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from . import common

SAFETENSORS_DTYPE = {
    torch.float32: "F32",
    torch.float64: "F64",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}
INVENTORY_ALGORITHM = "sha256-name-nul-dtype-nul-shape-csv-lf-v1"
SIZES = {"S": {"d_model": 256, "n_layers": 4, "n_heads": 4, "d_ff": 1024},
         "M": {"d_model": 512, "n_layers": 6, "n_heads": 8, "d_ff": 2048}}
INIT_STD = 0.02
_INT_FIELDS = ("vocab_size", "ctx_dim", "d_model", "n_layers", "n_heads", "d_ff", "max_seq", "pad_id")


@dataclass(frozen=True)
class PolicyConfig:
    """Architecture identity; ``max_seq`` is the maximum tokens per program including BOS and EOS."""
    vocab_size: int
    ctx_dim: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    max_seq: int
    dropout: float = 0.0
    context_free_head: bool = False
    pad_id: int = 0

    def __post_init__(self) -> None:
        for name in _INT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"PolicyConfig.{name} must be an int, got {value!r}")
        for name in ("vocab_size", "ctx_dim", "d_model", "n_layers", "n_heads", "d_ff"):
            if getattr(self, name) < 1:
                raise ValueError(f"PolicyConfig.{name} must be >= 1")
        if self.max_seq < 2:
            raise ValueError("PolicyConfig.max_seq must be >= 2 (BOS and EOS)")
        if self.d_model % self.n_heads:
            raise ValueError("PolicyConfig.d_model must be divisible by n_heads")
        if not 0 <= self.pad_id < self.vocab_size:
            raise ValueError("PolicyConfig.pad_id must index the vocabulary")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)):
            raise ValueError(f"PolicyConfig.dropout must be a number, got {self.dropout!r}")
        if not (math.isfinite(self.dropout) and 0.0 <= self.dropout < 1.0):
            raise ValueError("PolicyConfig.dropout must be in [0, 1)")
        if not isinstance(self.context_free_head, bool):
            raise ValueError("PolicyConfig.context_free_head must be a bool")
        object.__setattr__(self, "dropout", float(self.dropout))

    def to_json(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "PolicyConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"unknown PolicyConfig keys: {unknown}")
        try:
            return cls(**dict(d))
        except TypeError as exc:
            raise ValueError(f"bad PolicyConfig: {exc}") from None

    def sha256(self) -> str:
        return common.sha256_bytes(common.canonical(self.to_json()))


def size_config(name: str, *, vocab_size: int, ctx_dim: int, max_seq: int, dropout: float = 0.0,
                context_free_head: bool = False) -> PolicyConfig:
    """The PLAN §6.3.1 sizes: S = d256/L4/H4/FFN1024, M = d512/L6/H8/FFN2048."""
    if name not in SIZES:
        raise ValueError(f"unknown size {name!r}; expected one of {sorted(SIZES)}")
    return PolicyConfig(vocab_size=vocab_size, ctx_dim=ctx_dim, max_seq=max_seq, dropout=dropout,
                        context_free_head=context_free_head, **SIZES[name])


class PolicyOutputs(NamedTuple):
    """``logits`` row i predicts ``tokens[:, i+1]`` (raw, unmasked); ``rt_hat`` is in log1p seconds."""
    logits: torch.Tensor
    q_hat: torch.Tensor
    feas_logits: torch.Tensor
    rt_hat: torch.Tensor
    cf_hat: torch.Tensor | None
    h_eos: torch.Tensor


class CausalSelfAttention(nn.Module):
    """Multi-head causal attention through the explicit MATH SDPA backend (no attention dropout)."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, length, d = x.shape
        head_dim = d // self.n_heads
        q, k, v = (t.view(n, length, self.n_heads, head_dim).transpose(1, 2) for t in self.qkv(x).split(d, dim=-1))
        with sdpa_kernel([SDPBackend.MATH]):
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(n, length, d))


class Block(nn.Module):
    """Pre-LN transformer block: x + drop(attn(ln1 x)), then x + drop(ffn(ln2 x))."""

    def __init__(self, cfg: PolicyConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, cfg.n_heads)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, d))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x)))
        return x + self.drop(self.ffn(self.ln2(x)))


def _head(d_in: int, d_model: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_model), nn.GELU(), nn.Linear(d_model, d_out))


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
        bias = getattr(module, "bias", None)
        if bias is not None:
            nn.init.zeros_(bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class PolicyLM(nn.Module):
    """PLAN §6.3.1 policy model. Dropout modules are the only consumers of the global torch RNG in forward."""

    def __init__(self, cfg: PolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.tok_emb = nn.Embedding(cfg.vocab_size, d)
        self.pos_emb = nn.Embedding(cfg.max_seq + 1, d)
        self.ctx_mlp = nn.Sequential(nn.Linear(cfg.ctx_dim, d), nn.GELU(), nn.LayerNorm(d),
                                     nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d))
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)
        self.q_head = _head(2 * d, d, 1)
        self.feas_head = _head(2 * d, d, 3)
        self.rt_head = _head(2 * d, d, 1)
        self.cf_head = _head(d, d, 1) if cfg.context_free_head else None
        self.apply(_init_weights)
        self.lm_head.weight = self.tok_emb.weight
        assert_tied(self)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor, ctx_z: torch.Tensor,
                seq_ctx: torch.Tensor) -> PolicyOutputs:
        cfg = self.cfg
        if tokens.dim() != 2 or tokens.dtype != torch.long:
            raise ValueError("tokens must be Long[N, T]")
        n, t = tokens.shape
        if not 2 <= t <= cfg.max_seq:
            raise ValueError(f"sequence width {t} outside [2, {cfg.max_seq}]")
        if lengths.shape != (n,) or lengths.dtype != torch.long:
            raise ValueError("lengths must be Long[N]")
        if seq_ctx.shape != (n,) or seq_ctx.dtype != torch.long:
            raise ValueError("seq_ctx must be Long[N]")
        if ctx_z.dim() != 2 or ctx_z.shape[1] != cfg.ctx_dim or not ctx_z.is_floating_point():
            raise ValueError(f"ctx_z must be Float[C, {cfg.ctx_dim}]")
        n_ctx = ctx_z.shape[0]
        if n:
            if int(lengths.min()) < 2 or int(lengths.max()) > t:
                raise ValueError(f"lengths must lie in [2, {t}]")
            if int(seq_ctx.min()) < 0 or int(seq_ctx.max()) >= n_ctx:
                raise ValueError(f"seq_ctx must lie in [0, {n_ctx})")
            if int(tokens.min()) < 0 or int(tokens.max()) >= cfg.vocab_size:
                raise ValueError("token id outside the vocabulary")
        ctx_emb = self.ctx_mlp(ctx_z)[seq_ctx]                                   # [N, d]
        pos = self.pos_emb.weight
        prefix = (ctx_emb + pos[0]).unsqueeze(1)                                 # [N, 1, d]
        body = self.tok_emb(tokens) + pos[1:t + 1]                               # [N, T, d]
        x = self.emb_drop(torch.cat([prefix, body], dim=1))                      # [N, T+1, d]
        for block in self.blocks:
            x = block(x)
        h = self.ln_f(x)
        logits = self.lm_head(h[:, 1:t])                                         # [N, T-1, V]
        h_eos = h[torch.arange(n, device=h.device), lengths]                     # EOS input position
        head_in = torch.cat([h_eos, ctx_emb], dim=-1)
        cf_hat = self.cf_head(h_eos)[:, 0] if self.cf_head is not None else None
        return PolicyOutputs(logits=logits, q_hat=self.q_head(head_in)[:, 0], feas_logits=self.feas_head(head_in),
                             rt_hat=self.rt_head(head_in)[:, 0], cf_hat=cf_hat, h_eos=h_eos)


def assert_tied(model: PolicyLM) -> None:
    """Raise AssertionError unless the LM head shares the token embedding tensor and storage."""
    if model.lm_head.weight is not model.tok_emb.weight or \
            model.lm_head.weight.data_ptr() != model.tok_emb.weight.data_ptr():
        raise AssertionError("lm_head.weight is not tied to tok_emb.weight")


def parameter_names(model: nn.Module) -> list[str]:
    """Sorted ``named_parameters()`` names (the tied ``lm_head.weight`` alias is de-duplicated away)."""
    return sorted(name for name, _ in model.named_parameters())


def _dtype_label(t: torch.Tensor) -> str:
    try:
        return SAFETENSORS_DTYPE[t.dtype]
    except KeyError:
        raise ValueError(f"unsupported dtype {t.dtype}") from None


def _shape_csv(t: torch.Tensor) -> str:
    return ",".join(str(s) for s in t.shape)


def parameter_sha256_tensors(named: Mapping[str, torch.Tensor]) -> str:
    """sha256 over sorted names of ``name NUL dtype NUL shape-csv LF`` followed by the tensor's raw bytes."""
    if sys.byteorder != "little":
        raise RuntimeError("parameter_sha256 is defined on little-endian hosts only")
    h = hashlib.sha256()
    for name in sorted(named):
        t = named[name]
        if t.dtype != torch.float32:
            raise ValueError(f"parameter {name} is {t.dtype}; parameter_sha256 covers float32 only")
        h.update(f"{name}\0{_dtype_label(t)}\0{_shape_csv(t)}\n".encode("utf-8"))
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def parameter_sha256(model: nn.Module) -> str:
    return parameter_sha256_tensors(dict(model.named_parameters()))


def tensor_inventory_sha256(tensors: Mapping[str, torch.Tensor]) -> str:
    """INVENTORY_ALGORITHM: sha256 of ``name NUL dtype NUL shape-csv LF`` over sorted names (no tensor bytes)."""
    text = "".join(f"{name}\0{_dtype_label(tensors[name])}\0{_shape_csv(tensors[name])}\n" for name in sorted(tensors))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Exactly two named AdamW groups: 'decay' and 'no_decay' (biases and every LayerNorm parameter)."""
    wd = float(weight_decay)
    if not math.isfinite(wd) or wd < 0:
        raise ValueError(f"weight_decay must be finite and >= 0, got {weight_decay!r}")
    norm_names = set()
    for module_name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            for pname, _ in module.named_parameters(recurse=False):
                norm_names.add(f"{module_name}.{pname}" if module_name else pname)
    decay: tuple[list, list] = ([], [])
    no_decay: tuple[list, list] = ([], [])
    for name, p in model.named_parameters():
        target = no_decay if name.endswith(".bias") or name in norm_names else decay
        target[0].append(p)
        target[1].append(name)
    return [
        {"params": decay[0], "param_names": decay[1], "group_name": "decay", "weight_decay": wd},
        {"params": no_decay[0], "param_names": no_decay[1], "group_name": "no_decay", "weight_decay": 0.0},
    ]


def count_parameters(model: nn.Module) -> int:
    """Trainable parameter count with the tied embedding counted once."""
    return sum(p.numel() for p in model.parameters())


def configure_determinism(num_threads: int = 4) -> dict:
    """Deterministic CPU mode (PLAN §6.6 determinism controls); idempotent. Returns the run-plan record."""
    if isinstance(num_threads, bool) or not isinstance(num_threads, int) or num_threads < 1:
        raise ValueError(f"num_threads must be a positive int, got {num_threads!r}")
    torch.use_deterministic_algorithms(True)
    if torch.get_num_threads() != num_threads:
        torch.set_num_threads(num_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    observed = (torch.are_deterministic_algorithms_enabled(), torch.get_num_threads(),
                torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                torch.get_float32_matmul_precision())
    if observed != (True, num_threads, False, True, False, False, "highest"):
        raise RuntimeError(f"determinism controls did not take effect: {observed}")
    return {
        "use_deterministic_algorithms": True,
        "num_threads": num_threads,
        "sdpa_backend": "math",
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "allow_tf32_matmul": False,
        "allow_tf32_cudnn": False,
        "float32_matmul_precision": "highest",
        "ce_flatten": "(N*T,V)",
        "device": "cpu",
    }
