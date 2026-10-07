"""TA_Hyperion_v8: target-aware in-context head on top of frozen Hyperion v7 row embeddings.

Changes vs Hyperion v7 (which pools X-only row embeddings into [mean,std,p10,p90]):
  1. Row tokens = LayerNorm(v7 row embedding 512) -> 256, plus a LABEL CHANNEL: support rows add Emb([y, 1]); query rows add a
     learned [MASK] token (their y is hidden).
  2. 6-layer pre-norm transformer over the context's rows; keys are support rows only, so every row (support or query) reads
     the support set's (x, y) pairs (TabPFN-style in-context learning), and query targets can never leak into the context.
  3. Context embedding (fixed width 1024) = projection of
       a) label-conditioned attention pooling: 8 learned seeds attend over the transformer's support-row states (which mix y),
       b) explicit X->y summaries on the projected row embeddings z (256): per-dimension corr(z_d, y) over support rows,
          per-dimension cov(z_d, residual) of a closed-form in-context ridge fit y ~ z, the ridge fit's LOO R^2, log n.
  4. Row-level embeddings: the transformer's per-row states (256) are returned for every row.
Objective (self-supervised, no GP labels): masked-y regression of held-out query rows inside each context's outer-train set
+ 0.1 * InfoNCE between context embeddings computed from two disjoint support subsets of the same context.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def ridge_summ(z, y, msk, lam=1.0):
    """z [B,N,d], y [B,N], msk [B,N] support mask -> corr [B,d], cov(z, resid) [B,d], loo_r2 [B]. Support rows only."""
    m = msk.float(); n = m.sum(1).clamp(min=2)
    zm = (z * m[..., None]).sum(1) / n[:, None]; ym = (y * m).sum(1) / n
    zc = (z - zm[:, None]) * m[..., None]; yc = (y - ym[:, None]) * m
    zs = torch.sqrt((zc ** 2).sum(1) / n[:, None] + 1e-6); ys = torch.sqrt((yc ** 2).sum(1) / n + 1e-6)
    corr = (zc * yc[..., None]).sum(1) / n[:, None] / (zs * ys[:, None])
    zn = zc / zs[:, None]
    # dual ridge (N x N) on support rows: K = Zn Zn^T / d, alpha = (K + lam I)^-1 y ; LOO residual = alpha / diag((K+lam I)^-1)
    d = z.shape[-1]
    K = zn @ zn.transpose(1, 2) / d
    eye = torch.eye(K.shape[1], device=z.device, dtype=K.dtype)
    A = K + lam * eye + (1 - m)[:, :, None] * eye * 1e3          # padded / query rows decoupled
    Ainv = torch.linalg.inv(A.float()).to(K.dtype)
    alpha = (Ainv @ yc[..., None]).squeeze(-1)
    loo = alpha / torch.diagonal(Ainv, dim1=1, dim2=2).clamp(min=1e-6)
    loo_r2 = 1 - ((loo * m) ** 2).sum(1) / ((yc ** 2).sum(1) + 1e-6)
    fit = yc - lam * alpha
    res = (yc - fit) * m
    cov_res = (zn * res[..., None]).sum(1) / n[:, None]
    return corr, cov_res, loo_r2.clamp(-1, 1)


class TAHyperionV8(nn.Module):
    def __init__(self, d_in=512, d=256, layers=6, heads=8, seeds=8, out_dim=1024):
        super().__init__()
        self.ln_in = nn.LayerNorm(d_in); self.inp = nn.Linear(d_in, d)
        self.yemb = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d)); self.mask_tok = nn.Parameter(torch.zeros(d))
        lay = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(lay, layers, enable_nested_tensor=False)
        self.ln_out = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.seeds = nn.Parameter(torch.randn(seeds, d) * 0.02); self.pma = nn.MultiheadAttention(d, heads, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(seeds * d + 2 * d + 2, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))
        self.cproj = nn.Sequential(nn.Linear(out_dim, 256), nn.GELU(), nn.Linear(256, 128))
        self.out_dim = out_dim

    def forward(self, R, y, sup, valid):
        """R [B,N,512], y [B,N] (standardised with support stats), sup/valid [B,N] bool. Returns yhat [B,N], ctx [B,out], rows [B,N,d]."""
        z = self.inp(self.ln_in(R))
        ych = self.yemb(torch.stack([y * sup, sup.float()], -1))
        tok = z + torch.where(sup[..., None], ych, self.mask_tok.expand_as(z))
        keypad = ~(sup & valid)
        h = self.ln_out(self.enc(tok, src_key_padding_mask=keypad))
        yhat = self.head(h).squeeze(-1)
        q = self.seeds[None].expand(R.shape[0], -1, -1)
        pooled, _ = self.pma(q, h, h, key_padding_mask=keypad)
        with torch.autocast("cuda", enabled=False):
            corr, covr, loo = ridge_summ(z.float(), y.float(), sup & valid)
        n = (sup & valid).float().sum(1)
        feats = torch.cat([pooled.flatten(1), corr.to(pooled.dtype), covr.to(pooled.dtype), loo[:, None].to(pooled.dtype), torch.log1p(n)[:, None].to(pooled.dtype)], 1)
        ctx = self.proj(feats)
        return yhat, ctx, h
