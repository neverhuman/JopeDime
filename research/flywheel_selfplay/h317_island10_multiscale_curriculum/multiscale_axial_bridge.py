#!/usr/bin/env python3
"""H317 neural candidate ranker: (DOPE context view features, candidate features) -> 4 heads per candidate.

Inputs
  ctx  [Bc, G, 997]  per-view flat context vector [tokens_10x32 (320) | dope629 (629) | lm48_z (48)] (H58 schema), G views
                     of the same context (G=1 at OOF evaluation: view 0).
  cand [Bc, M, 452]  candidate vector [slog(feat412) (412) | proj16 (16) | sem24 (24)], computable from any DSL
                     (`candidate_vector`). r882 priors are never inputs.
Outputs [Bc, G, M, 4]: (y_s1 = predicted native utility (the ranking score), mean logratio, sd_lr (>= 0), gate logit).

Scales (actual parameter counts: `param_count`, reported by the tests and the timing receipts)
  S  d=32,  4 heads, L=1: per-token context projection, 1 context self-attention block, 1 candidate->context
     cross-attention block (LayerNorm, GELU MLPs), bilinear context-candidate interaction. ~180K params.
  M  d=96,  6 heads, L=3: RMSNorm + QK-norm attention + SwiGLU; per layer a context self-attention block and a
     candidate->context cross-attention block. Full 452-D candidate encoder.
  L  d=160, 8 heads, L=4: axial context-token x view x candidate set transformer. Per layer: token self-attention
     (within a view), view-axis attention (across the G views of a context, per token), candidate->context
     cross-attention, candidate set attention through 16 inducing points (ISAB, so the cost and the slate dependence stay
     bounded at 1,152 candidates); DropPath 0.10 and a learned per-channel gated residual on every sub-block.
Context tokens (all scales): 10 SVD tokens + dope629 cut into 20 coordinate-preserving 32-D chunks + lm48_z as 2 chunks
= 32 raw 32-D tokens, each with its own linear map and a type embedding.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CTX_DIM = 997
CAND_DIM = 452
N_CTX_TOK = 32
N_HEADS_OUT = 4


@dataclass
class ScaleConfig:
    name: str
    d: int
    n_heads: int
    n_layers: int
    cand_hidden: int
    ffn_mult: float
    drop_path: float = 0.0
    n_inducing: int = 16


SCALES: Dict[str, ScaleConfig] = {
    "S": ScaleConfig("S", d=32, n_heads=4, n_layers=1, cand_hidden=228, ffn_mult=4.0),
    "M": ScaleConfig("M", d=96, n_heads=6, n_layers=3, cand_hidden=512, ffn_mult=3.0),
    "L": ScaleConfig(
        "L",
        d=160,
        n_heads=8,
        n_layers=4,
        cand_hidden=256,
        ffn_mult=1.75,
        drop_path=0.10,
    ),
}


# ----------------------------------------------------------------------------------------------------------------------
# Candidate / context feature construction (numpy; shared with the GP screener)
# ----------------------------------------------------------------------------------------------------------------------
def slog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def candidate_vector(
    feat412: np.ndarray, proj16: np.ndarray, sem24: np.ndarray
) -> np.ndarray:
    """[M, 452] float32 = [slog(feat412) | proj16 | sem24] (un-normalized; the model's normalizer z-scores it)."""
    feat412 = np.asarray(feat412, np.float64).reshape(-1, 412)
    proj16 = np.asarray(proj16, np.float64).reshape(-1, 16)
    sem24 = np.asarray(sem24, np.float64).reshape(-1, 24)
    return np.concatenate([slog(feat412), proj16, sem24], axis=1).astype(np.float32)


def complexity_from_feat412(feat412: np.ndarray) -> np.ndarray:
    """`dsl.count("(")` = steps + 1 (feature 411 = 'steps'); verified on all TRAIN champions by the parity check."""
    return np.asarray(feat412, np.float64)[..., 411] + 1.0


# ----------------------------------------------------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def norm_layer(kind: str, d: int) -> nn.Module:
    return RMSNorm(d) if kind == "rms" else nn.LayerNorm(d)


class Attention(nn.Module):
    """Multi-head attention with optional QK RMS-norm and an optional boolean mask (True = may attend)."""

    def __init__(self, d: int, n_heads: int, qk_norm: bool):
        super().__init__()
        self.h, self.dh = n_heads, d // n_heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.qn = RMSNorm(self.dh) if qk_norm else nn.Identity()
        self.kn = RMSNorm(self.dh) if qk_norm else nn.Identity()

    def forward(self, xq, xkv, mask: Optional[torch.Tensor] = None):
        B, Sq, D = xq.shape
        Sk = xkv.shape[1]
        q = self.qn(self.q(xq).view(B, Sq, self.h, self.dh)).transpose(1, 2)
        k = self.kn(self.k(xkv).view(B, Sk, self.h, self.dh)).transpose(1, 2)
        v = self.v(xkv).view(B, Sk, self.h, self.dh).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(out.transpose(1, 2).reshape(B, Sq, D))


class FFN(nn.Module):
    def __init__(self, d: int, mult: float, swiglu: bool):
        super().__init__()
        h = int(round(d * mult))
        self.swiglu = swiglu
        if swiglu:
            self.w1 = nn.Linear(d, h)
            self.w3 = nn.Linear(d, h)
            self.w2 = nn.Linear(h, d)
        else:
            self.w1 = nn.Linear(d, h)
            self.w2 = nn.Linear(h, d)

    def forward(self, x):
        if self.swiglu:
            return self.w2(F.silu(self.w1(x)) * self.w3(x))
        return self.w2(F.gelu(self.w1(x)))


class DropPath(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p <= 0:
            return x
        keep = (
            torch.rand(x.shape[0], *([1] * (x.dim() - 1)), device=x.device) >= self.p
        ).to(x.dtype)
        return x * keep / (1.0 - self.p)


class Residual(nn.Module):
    """Pre-norm residual sub-block; `gated` adds a learned per-channel highway gate sigmoid(g) (init 0.5)."""

    def __init__(self, d: int, norm: str, drop_path: float, gated: bool):
        super().__init__()
        self.norm = norm_layer(norm, d)
        self.dp = DropPath(drop_path)
        self.gate = nn.Parameter(torch.zeros(d)) if gated else None

    def scale(self, y):
        y = self.dp(y)
        return y * torch.sigmoid(self.gate) * 2.0 if self.gate is not None else y


class SelfBlock(nn.Module):
    def __init__(
        self, cfg: ScaleConfig, norm: str, qk_norm: bool, swiglu: bool, gated: bool
    ):
        super().__init__()
        d = cfg.d
        self.r1 = Residual(d, norm, cfg.drop_path, gated)
        self.att = Attention(d, cfg.n_heads, qk_norm)
        self.r2 = Residual(d, norm, cfg.drop_path, gated)
        self.ffn = FFN(d, cfg.ffn_mult, swiglu)

    def forward(self, x, mask=None):
        h = self.r1.norm(x)
        x = x + self.r1.scale(self.att(h, h, mask))
        return x + self.r2.scale(self.ffn(self.r2.norm(x)))


class CrossBlock(nn.Module):
    def __init__(
        self, cfg: ScaleConfig, norm: str, qk_norm: bool, swiglu: bool, gated: bool
    ):
        super().__init__()
        d = cfg.d
        self.r1 = Residual(d, norm, cfg.drop_path, gated)
        self.kv_norm = norm_layer(norm, d)
        self.att = Attention(d, cfg.n_heads, qk_norm)
        self.r2 = Residual(d, norm, cfg.drop_path, gated)
        self.ffn = FFN(d, cfg.ffn_mult, swiglu)

    def forward(self, x, kv):
        x = x + self.r1.scale(self.att(self.r1.norm(x), self.kv_norm(kv)))
        return x + self.r2.scale(self.ffn(self.r2.norm(x)))


class InducedSetBlock(nn.Module):
    """ISAB (Lee et al. 2019): inducing points attend to the candidate set, candidates attend back."""

    def __init__(self, cfg: ScaleConfig, norm: str, gated: bool):
        super().__init__()
        d = cfg.d
        self.inducing = nn.Parameter(torch.randn(cfg.n_inducing, d) * 0.02)
        self.n1 = norm_layer(norm, d)
        self.pool = Attention(d, cfg.n_heads, True)
        self.r = Residual(d, norm, cfg.drop_path, gated)
        self.back = Attention(d, cfg.n_heads, True)

    def forward(self, x):
        h = self.n1(x)
        ind = self.pool(self.inducing.unsqueeze(0).expand(x.shape[0], -1, -1), h)
        return x + self.r.scale(self.back(self.r.norm(x), ind))


# ----------------------------------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------------------------------
class InputNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("ctx_mu", torch.zeros(CTX_DIM))
        self.register_buffer("ctx_sd", torch.ones(CTX_DIM))
        self.register_buffer("cand_mu", torch.zeros(CAND_DIM))
        self.register_buffer("cand_sd", torch.ones(CAND_DIM))

    @torch.no_grad()
    def fit(self, ctx: np.ndarray, cand: np.ndarray) -> None:
        """Feature-only statistics from training-fold views [*, 997] and training-fold candidates [*, 452]."""
        for arr, mu, sd in (
            (ctx, self.ctx_mu, self.ctx_sd),
            (cand, self.cand_mu, self.cand_sd),
        ):
            a = np.asarray(arr, np.float64).reshape(-1, arr.shape[-1])
            m = a.mean(0)
            s = a.std(0)
            mu.copy_(torch.as_tensor(m, dtype=torch.float32))
            sd.copy_(torch.as_tensor(np.where(s > 1e-6, s, 1.0), dtype=torch.float32))

    def ctx(self, x):
        return torch.clamp((x.float() - self.ctx_mu) / self.ctx_sd, -8.0, 8.0)

    def cand(self, x):
        return torch.clamp((x.float() - self.cand_mu) / self.cand_sd, -8.0, 8.0)


def _ctx_raw_tokens(z: torch.Tensor) -> torch.Tensor:
    """[N, 997] normalized -> [N, 32, 32]: 10 SVD tokens, dope629 as 20 chunks (zero-padded to 640), lm48_z as 2."""
    n = z.shape[0]
    tok = z[:, :320].reshape(n, 10, 32)
    dope = F.pad(z[:, 320:949], (0, 11)).reshape(n, 20, 32)
    lm = F.pad(z[:, 949:997], (0, 16)).reshape(n, 2, 32)
    return torch.cat([tok, dope, lm], dim=1)


class MultiScaleAxialResidualBridge(nn.Module):
    def __init__(self, scale: str = "S"):
        super().__init__()
        cfg = SCALES[scale]
        self.cfg = cfg
        d = cfg.d
        big = scale in ("M", "L")
        norm = "rms" if big else "ln"
        self.norm_kind = norm
        self.normalizer = InputNormalizer()
        # per-token context projection: weight [T, 32, d]
        self.ctx_w = nn.Parameter(torch.randn(N_CTX_TOK, 32, d) / math.sqrt(32))
        self.ctx_b = nn.Parameter(torch.zeros(N_CTX_TOK, d))
        self.ctx_type = nn.Parameter(torch.zeros(N_CTX_TOK, d))
        if big:
            self.cand_in = nn.Sequential(
                nn.Linear(CAND_DIM, cfg.cand_hidden),
                nn.SiLU(),
                nn.Linear(cfg.cand_hidden, cfg.cand_hidden),
                nn.SiLU(),
                nn.Linear(cfg.cand_hidden, d),
            )
        else:
            self.cand_in = nn.Sequential(
                nn.Linear(CAND_DIM, cfg.cand_hidden),
                nn.GELU(),
                nn.Linear(cfg.cand_hidden, d),
            )
        kw = dict(norm=norm, qk_norm=big, swiglu=big, gated=(scale == "L"))
        self.ctx_blocks = nn.ModuleList(
            [SelfBlock(cfg, **kw) for _ in range(cfg.n_layers)]
        )
        self.view_blocks = (
            nn.ModuleList([SelfBlock(cfg, **kw) for _ in range(cfg.n_layers)])
            if scale == "L"
            else None
        )
        self.cross_blocks = nn.ModuleList(
            [CrossBlock(cfg, **kw) for _ in range(cfg.n_layers)]
        )
        self.set_blocks = (
            nn.ModuleList(
                [InducedSetBlock(cfg, norm, True) for _ in range(cfg.n_layers)]
            )
            if scale == "L"
            else None
        )
        self.out_norm = norm_layer(norm, d)
        self.ctx_pool_norm = norm_layer(norm, d)
        n_bil = 24
        self.bil_ctx = nn.Linear(d, n_bil, bias=False)
        self.bil_cand = nn.Linear(d, n_bil, bias=False)
        head_in = 2 * d + n_bil
        hh = max(64, 2 * d)
        self.head = nn.Sequential(
            nn.Linear(head_in, hh),
            nn.SiLU() if big else nn.GELU(),
            nn.Linear(hh, N_HEADS_OUT),
        )

    # ------------------------------------------------------------------------------------------------------------------
    def encode_context(self, ctx: torch.Tensor) -> torch.Tensor:
        """[Bc, G, 997] -> [Bc, G, T, d] (raw token projection only)."""
        Bc, G, _ = ctx.shape
        z = self.normalizer.ctx(ctx.reshape(Bc * G, CTX_DIM))
        raw = _ctx_raw_tokens(z)
        tok = torch.einsum("ntc,tcd->ntd", raw, self.ctx_w) + self.ctx_b + self.ctx_type
        return tok.reshape(Bc, G, N_CTX_TOK, -1)

    def forward(
        self,
        ctx: torch.Tensor,
        cand: torch.Tensor,
        mix_lambda: Optional[torch.Tensor] = None,
        view_isolated: bool = False,
    ) -> torch.Tensor:
        """ctx [Bc, G, 997], cand [Bc, M, 452] -> [Bc, G, M, 4].

        mix_lambda [Bc] (G == 2 only): cross-view latent mixup of the two views' context token latents.
        view_isolated: L only, every view attends only to itself on the view axis (exactly the G=1 computation).
        """
        Bc, G, _ = ctx.shape
        M = cand.shape[1]
        d = self.cfg.d
        x = self.encode_context(ctx)
        if mix_lambda is not None:
            if G != 2:
                raise ValueError("cross-view mixup needs G == 2 views")
            lam = mix_lambda.view(Bc, 1, 1, 1).to(x.dtype)
            x = torch.cat(
                [
                    lam * x[:, :1] + (1 - lam) * x[:, 1:],
                    lam * x[:, 1:] + (1 - lam) * x[:, :1],
                ],
                dim=1,
            )
        c = self.cand_in(self.normalizer.cand(cand))  # [Bc, M, d]
        c = c.unsqueeze(1).expand(Bc, G, M, d).reshape(Bc * G, M, d)
        x = x.reshape(Bc * G, N_CTX_TOK, d)
        vmask = None
        if self.view_blocks is not None and (view_isolated and G > 1):
            vmask = torch.eye(G, dtype=torch.bool, device=ctx.device)
        for i in range(self.cfg.n_layers):
            x = self.ctx_blocks[i](x)
            if self.view_blocks is not None:
                xv = (
                    x.reshape(Bc, G, N_CTX_TOK, d)
                    .transpose(1, 2)
                    .reshape(Bc * N_CTX_TOK, G, d)
                )
                xv = self.view_blocks[i](xv, vmask)
                x = (
                    xv.reshape(Bc, N_CTX_TOK, G, d)
                    .transpose(1, 2)
                    .reshape(Bc * G, N_CTX_TOK, d)
                )
            c = self.cross_blocks[i](c, x)
            if self.set_blocks is not None:
                c = self.set_blocks[i](c)
        c = self.out_norm(c)
        pooled = self.ctx_pool_norm(x.mean(dim=1))  # [Bc*G, d]
        bil = self.bil_ctx(pooled).unsqueeze(1) * self.bil_cand(c)
        h = torch.cat([c, pooled.unsqueeze(1).expand_as(c), bil], dim=-1)
        out = self.head(h).reshape(Bc, G, M, N_HEADS_OUT)
        return torch.cat(
            [out[..., :2], F.softplus(out[..., 2:3]), out[..., 3:4]], dim=-1
        )

    def param_groups_muon(self):
        """(muon_params, adamw_params): Muon gets the 2-D hidden matrices of the transformer blocks; AdamW gets the
        input projections/embeddings, heads, norms, gates, biases and inducing points."""
        muon, other = [], []
        for name, p in self.named_parameters():
            hidden = (
                name.startswith(
                    ("ctx_blocks.", "view_blocks.", "cross_blocks.", "set_blocks.")
                )
                and p.ndim == 2
            )
            hidden = hidden and not name.endswith("inducing")
            (muon if hidden else other).append(p)
        return muon, other


def param_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def build_model(scale: str) -> MultiScaleAxialResidualBridge:
    return MultiScaleAxialResidualBridge(scale)


def scale_config_dict(scale: str) -> dict:
    return asdict(SCALES[scale])


# ----------------------------------------------------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------------------------------------------------
LOSS_WEIGHTS = {
    "list": 1.0,
    "pair": 0.5,
    "reg": 0.25,
    "lr": 0.25,
    "sd": 0.10,
    "gate": 0.25,
}


def draw_pairs(
    Bc: int,
    M: int,
    n_pairs: int = 256,
    device=None,
    generator: Optional[torch.Generator] = None,
):
    """Random candidate pairs (i, j), each [Bc, n_pairs]; drawn once per optimizer step (SAM reuses them)."""
    i = torch.randint(0, M, (Bc, n_pairs), device=device, generator=generator)
    j = torch.randint(0, M, (Bc, n_pairs), device=device, generator=generator)
    return i, j


def ranking_losses(
    pred: torch.Tensor,
    tgt: Dict[str, torch.Tensor],
    tau: float,
    kl_weight: float = 0.0,
    n_pairs: int = 256,
    generator: Optional[torch.Generator] = None,
    pairs: Optional[tuple] = None,
) -> Dict[str, torch.Tensor]:
    """pred [Bc, G, M, 4]; tgt: u, lr, lr_mask, sd, sd_mask, gate, each [Bc, M].

    list  ListNet: CE(softmax(u/tau), log_softmax(s/tau)) per view (s = utility head).
    pair  regret-weighted logistic loss on random candidate pairs: |u_i - u_j| * softplus(-sign(u_i - u_j)(s_i - s_j)/tau).
    reg   Huber(s, u); lr Huber on the mean-logratio head (finite |lr| < 4.9); sd Huber on sd_lr (gated candidates);
    gate  BCE on the gate logit; kl: views' slate distributions vs their mean (G >= 2), weighted by kl_weight.
    """
    Bc, G, M, _ = pred.shape
    s = pred[..., 0]
    u = tgt["u"].unsqueeze(1).expand(Bc, G, M)
    p_t = torch.softmax(u / tau, dim=-1)
    logq = torch.log_softmax(s / tau, dim=-1)
    out = {"list": -(p_t * logq).sum(-1).mean()}
    i, j = (
        pairs
        if pairs is not None
        else draw_pairs(Bc, M, n_pairs, pred.device, generator)
    )
    n_pairs = i.shape[1]
    ui, uj = tgt["u"].gather(1, i), tgt["u"].gather(1, j)
    du = ui - uj
    si = s.gather(2, i.unsqueeze(1).expand(Bc, G, n_pairs))
    sj = s.gather(2, j.unsqueeze(1).expand(Bc, G, n_pairs))
    w = du.abs().unsqueeze(1)
    pair = w * F.softplus(-torch.sign(du).unsqueeze(1) * (si - sj) / tau)
    out["pair"] = pair.sum() / w.expand_as(pair).sum().clamp_min(1e-6)
    out["reg"] = F.huber_loss(s, u, delta=0.5)
    lm = tgt["lr_mask"].unsqueeze(1).expand(Bc, G, M)
    out["lr"] = (
        F.huber_loss(
            pred[..., 1],
            tgt["lr"].unsqueeze(1).expand(Bc, G, M),
            delta=0.5,
            reduction="none",
        )
        * lm
    ).sum() / lm.sum().clamp_min(1)
    sm = tgt["sd_mask"].unsqueeze(1).expand(Bc, G, M)
    out["sd"] = (
        F.huber_loss(
            pred[..., 2],
            tgt["sd"].unsqueeze(1).expand(Bc, G, M),
            delta=0.25,
            reduction="none",
        )
        * sm
    ).sum() / sm.sum().clamp_min(1)
    out["gate"] = F.binary_cross_entropy_with_logits(
        pred[..., 3], tgt["gate"].unsqueeze(1).expand(Bc, G, M)
    )
    total = sum(LOSS_WEIGHTS[k] * out[k] for k in LOSS_WEIGHTS)
    if G >= 2 and kl_weight > 0:
        q = logq.exp()
        qbar = q.mean(dim=1, keepdim=True).detach()
        out["kl"] = (qbar * (torch.log(qbar.clamp_min(1e-12)) - logq)).sum(-1).mean()
        total = total + kl_weight * out["kl"]
    out["total"] = total
    return out
