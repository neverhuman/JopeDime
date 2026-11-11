#!/usr/bin/env python3
"""H431 (CW-3 Part B): H419 APEX-ICL model with optional trunk dropout (--dropout, Factor B).

When dropout == 0.0 (the default), parameter tensors, state_dict keys, initialization RNG draws, and forward outputs
are 100% bit-for-bit identical to h419/r9/code/icl_model.py.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SIZES = {
    "S": dict(d_col=96, col_blocks=2, n_ind=24, d_row=96, row_layers=2, n_cls=4, d=256, layers=8, heads=4),
    "M": dict(d_col=128, col_blocks=2, n_ind=24, d_row=128, row_layers=3, n_cls=4, d=384, layers=12, heads=6),
    "L": dict(d_col=128, col_blocks=2, n_ind=24, d_row=160, row_layers=3, n_cls=4, d=768, layers=14, heads=12),
}
N_BINS = 1024
BIN_RANGE = 40.0
CLIP_X = 8.0
FID_SLOTS = 256
FID_DIM = 32


def bin_centers(n_bins=N_BINS, r=BIN_RANGE):
    a = math.asinh(r)
    return torch.sinh(torch.linspace(-a, a, n_bins, dtype=torch.float64)).float()


class Attn(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)

    def forward(self, q, mem):
        B, Lq, D = q.shape
        if B > 32768:
            return torch.cat([self.forward(q[i:i + 32768], mem[i:i + 32768]) for i in range(0, B, 32768)], 0)
        Lk = mem.shape[1]
        h = self.h
        qh = self.q(q).view(B, Lq, h, D // h).transpose(1, 2)
        k, v = self.kv(mem).view(B, Lk, 2, h, D // h).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(qh, k, v)
        return self.o(o.transpose(1, 2).reshape(B, Lq, D))


class Block(nn.Module):
    def __init__(self, d, heads, mult=4, dropout=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.att = Attn(d, heads)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mult * d), nn.GELU(), nn.Linear(mult * d, d))
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x, n_ctx=None, mem=None):
        h = self.n1(x)
        src = mem if mem is not None else (h if n_ctx is None else h[:, :n_ctx])
        x = x + self.drop(self.att(h, src))
        return x + self.drop(self.mlp(self.n2(x)))


def _mlp(i, d):
    return nn.Sequential(nn.Linear(i, d), nn.GELU(), nn.Linear(d, d))


class ICL(nn.Module):
    def __init__(self, size="S", dropout=0.0, **over):
        super().__init__()
        c = dict(SIZES[size]); c.update(over)
        self.cfg = dict(c, size=size)
        if dropout > 0.0:
            self.cfg["dropout"] = float(dropout)
        dc, dr, d = c["d_col"], c["d_row"], c["d"]
        self.n_cls = c["n_cls"]
        self.col_in = _mlp(2, dc)
        self.col_norm = nn.LayerNorm(dc)
        self.ind = nn.Parameter(torch.randn(c["n_ind"], dc) * 0.02)
        self.col_blocks = nn.ModuleList(Block(dc, 4, 2) for _ in range(c["col_blocks"]))
        self.col_out = nn.LayerNorm(dc)
        self.col_scale = nn.Linear(dc, dr)
        self.col_shift = nn.Linear(dc, dr)
        nn.init.zeros_(self.col_scale.weight); nn.init.zeros_(self.col_scale.bias)
        self.cell_in = _mlp(2, dr)
        g = torch.Generator().manual_seed(419)
        self.register_buffer("fid_buf", torch.randn(FID_SLOTS, FID_DIM, generator=g))
        self.fid = nn.Linear(FID_DIM, dr)
        self.cls = nn.Parameter(torch.randn(self.n_cls, dr) * 0.02)
        self.row_blocks = nn.ModuleList(Block(dr, 4, 4) for _ in range(c["row_layers"]))
        self.row_norm = nn.LayerNorm(dr)
        self.row_proj = nn.Linear(self.n_cls * dr, d)
        self.y_lin = nn.Linear(1, d)
        self.y_mlp = _mlp(1, d)
        self.q_emb = nn.Parameter(torch.randn(d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, c["heads"], 4, dropout=dropout) for _ in range(c["layers"]))
        self.out_norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, N_BINS))
        self.register_buffer("centers", bin_centers())

    @staticmethod
    def normalise(x, y_ctx, n_ctx):
        x = x.float(); y_ctx = y_ctx.float()
        miss = ~torch.isfinite(x)
        xc = torch.where(miss[:, :n_ctx], torch.nan, x[:, :n_ctx])
        q = torch.nanquantile(xc, torch.tensor([0.25, 0.5, 0.75], device=x.device), dim=1)
        med, iqr = q[1], (q[2] - q[0]) / 1.349
        mean = torch.nanmean(xc, dim=1)
        sd = torch.sqrt(torch.nanmean((xc - mean[:, None]) ** 2, dim=1))
        mean = torch.where(torch.isfinite(mean), mean, 0.0)
        sd = torch.where(torch.isfinite(sd), sd, 0.0)
        use_iqr = torch.isfinite(iqr) & (iqr > 1e-6 * sd) & (iqr > 0)
        loc = torch.where(use_iqr, med, mean)
        scale = torch.where(use_iqr, iqr, sd)
        scale = torch.where(scale > 0, scale, 1.0)
        z = ((torch.where(miss, 0.0, x) - loc[:, None]) / scale[:, None]).clamp(-CLIP_X, CLIP_X)
        z = torch.where(miss, 0.0, z)
        ym = y_ctx.mean(1, keepdim=True)
        ys = y_ctx.std(1, keepdim=True)
        ys = torch.where(ys > 1e-12 * (ym.abs() + 1.0), ys, 1.0)
        return torch.stack([z, miss.float()], -1), ym, ys

    def forward(self, x, y_ctx, n_ctx, slots=None, amp=True):
        B, N, Fn = x.shape
        dev = x.device
        with torch.autocast(dev.type, enabled=False):
            zm, ym, ys = self.normalise(x, y_ctx, n_ctx)
            yz = ((y_ctx.float() - ym) / ys).clamp(-BIN_RANGE, BIN_RANGE)
        if slots is None:
            slots = torch.arange(Fn, device=dev) % FID_SLOTS
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp and dev.type == "cuda"):
            X = self.col_norm(self.col_in(zm[:, :n_ctx])).permute(0, 2, 1, 3).reshape(B * Fn, n_ctx, -1)
            H = self.ind.unsqueeze(0).expand(B * Fn, -1, -1)
            for blk in self.col_blocks:
                H = blk(H, mem=X)
            c = self.col_out(H.mean(1)).view(B, 1, Fn, -1)
            e = self.cell_in(zm) * (1.0 + self.col_scale(c)) + self.col_shift(c) + self.fid(self.fid_buf[slots])
            t = torch.cat([self.cls.unsqueeze(0).expand(B * N, -1, -1).to(e.dtype), e.reshape(B * N, Fn, -1)], 1)
            for blk in self.row_blocks:
                t = blk(t)
            tok = self.row_proj(self.row_norm(t[:, :self.n_cls]).reshape(B, N, -1))
            yv = yz.unsqueeze(-1).to(tok.dtype)
            tok = torch.cat([tok[:, :n_ctx] + self.y_lin(yv) + self.y_mlp(yv), tok[:, n_ctx:] + self.q_emb], 1)
            for blk in self.blocks:
                tok = blk(tok, n_ctx=n_ctx)
            out = self.out_norm(tok[:, n_ctx:])
        logits = self.head(out.float())
        return logits, ym, ys

    def loss(self, logits, y_q, ym, ys):
        c = self.centers
        z = ((y_q.float() - ym) / ys).clamp(c[0], c[-1])
        hi = torch.searchsorted(c, z.contiguous()).clamp(1, c.numel() - 1)
        lo = hi - 1
        w = (z - c[lo]) / (c[hi] - c[lo])
        lp = F.log_softmax(logits, -1)
        return -(lp.gather(-1, lo.unsqueeze(-1)).squeeze(-1) * (1 - w) + lp.gather(-1, hi.unsqueeze(-1)).squeeze(-1) * w)

    def mean(self, logits, ym, ys):
        return ym + ys * (F.softmax(logits, -1) * self.centers).sum(-1)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
