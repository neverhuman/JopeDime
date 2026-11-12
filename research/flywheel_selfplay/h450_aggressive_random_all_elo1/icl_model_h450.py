#!/usr/bin/env python3
"""H450 Precision-Preserving & Exact-Warp-Integrating ICL (`icl_model_h450.py`).

100% parameter- and checkpoint-compatible (`strict=True`) with `icl_model.ICL` (`S`, `M`, `L` sizes):
  1. `normalise_f64`: Computes context-only median, IQR, mean, std, and `(xd - loc) / scale` in full `float64` BEFORE
     clamping and casting `z` to `float32`. Recovers 15-29 bits of sub-day/sub-second resolution on millisecond Unix
     timestamps (`1.70e12`) and epoch-day columns (`20729.xx`) that `x = x.float()` previously destroyed.
  2. `mean_warped`: Computes the exact Bayes-optimal minimum-MSE expectation `E[y | X] = sum_b p_b * g^{-1}(ym + ys*c_b)`
     directly over all `1,024` softmax bins in `float64` for arbitrary monotone target warps (e.g. Yeo-Johnson).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if "/home/ubuntu/h448-stage/pub-r1/code" not in sys.path and Path("/home/ubuntu/h448-stage/pub-r1/code").is_dir():
    sys.path.append("/home/ubuntu/h448-stage/pub-r1/code")

from icl_model import BIN_RANGE, CLIP_X, FID_SLOTS, ICL, N_BINS  # noqa: E402


class ICLH450(ICL):
    """Exact state_dict-compatible subclass of `ICL` with `float64` context normalization and exact 1,024-bin warp readout."""

    @staticmethod
    def normalise_f64(x: torch.Tensor, y_ctx: torch.Tensor, n_ctx: int):
        xd = x.double()
        yd = y_ctx.double()
        miss = ~torch.isfinite(xd)
        xc = torch.where(miss[:, :n_ctx], torch.nan, xd[:, :n_ctx])
        q_probs = torch.tensor([0.25, 0.5, 0.75], device=xd.device, dtype=torch.float64)
        q = torch.nanquantile(xc, q_probs, dim=1)  # [3, B, F] in float64
        med = q[1]
        iqr = (q[2] - q[0]) / 1.349
        mean = torch.nanmean(xc, dim=1)
        sd = torch.sqrt(torch.nanmean((xc - mean[:, None]) ** 2, dim=1))
        mean = torch.where(torch.isfinite(mean), mean, 0.0)
        sd = torch.where(torch.isfinite(sd), sd, 0.0)
        use_iqr = torch.isfinite(iqr) & (iqr > 1e-8 * sd) & (iqr > 0.0)
        loc = torch.where(use_iqr, med, mean)
        scale = torch.where(use_iqr, iqr, sd)
        scale = torch.where(scale > 0.0, scale, 1.0)
        z64 = ((torch.where(miss, 0.0, xd) - loc[:, None]) / scale[:, None]).clamp(-CLIP_X, CLIP_X)
        z32 = torch.where(miss, 0.0, z64).float()
        ym64 = yd.mean(1, keepdim=True)
        ys64 = yd.std(1, keepdim=True)
        ys64 = torch.where(ys64 > 1e-12 * (ym64.abs() + 1.0), ys64, 1.0)
        return torch.stack([z32, miss.float()], -1), ym64, ys64

    def forward(self, x: torch.Tensor, y_ctx: torch.Tensor, n_ctx: int, slots=None, amp: bool = True):
        B, N, Fn = x.shape
        dev = x.device
        with torch.autocast(dev.type, enabled=False):
            zm, ym64, ys64 = self.normalise_f64(x, y_ctx, n_ctx)
            yz = ((y_ctx.double() - ym64) / ys64).clamp(-BIN_RANGE, BIN_RANGE).float()
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
            tok = self.row_proj(self.row_norm(t[:, : self.n_cls]).reshape(B, N, -1))
            yv = yz.unsqueeze(-1).to(tok.dtype)
            tok = torch.cat([tok[:, :n_ctx] + self.y_lin(yv) + self.y_mlp(yv), tok[:, n_ctx:] + self.q_emb], 1)
            for blk in self.blocks:
                tok = blk(tok, n_ctx=n_ctx)
            out = self.out_norm(tok[:, n_ctx:])
        logits = self.head(out.float())
        return logits, ym64, ys64

    def loss(self, logits: torch.Tensor, y_q: torch.Tensor, ym: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        c = self.centers.double()
        z = ((y_q.double() - ym.double()) / ys.double()).clamp(c[0], c[-1])
        hi = torch.searchsorted(c, z.contiguous()).clamp(1, c.numel() - 1)
        lo = hi - 1
        w = ((z - c[lo]) / (c[hi] - c[lo])).float()
        lp = F.log_softmax(logits, -1)
        return -(lp.gather(-1, lo.unsqueeze(-1)).squeeze(-1) * (1.0 - w) + lp.gather(-1, hi.unsqueeze(-1)).squeeze(-1) * w)

    def mean_f64(self, logits: torch.Tensor, ym: torch.Tensor, ys: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
        """Mean prediction in float64 `[B, Nq]`."""
        probs = F.softmax(logits.double() / max(float(temp), 1e-3), dim=-1)
        return ym.double() + ys.double() * (probs * self.centers.double()).sum(-1)

    def mean_warped(self, logits: torch.Tensor, ym: torch.Tensor, ys: torch.Tensor, inv_fn, temp: float = 1.0) -> torch.Tensor:
        """Exact 1,024-bin probability integration `E[y | X] = sum_b p_b * inv_fn(ym + ys * c_b)` in float64 `[B, Nq]`."""
        probs = F.softmax(logits.double() / max(float(temp), 1e-3), dim=-1)
        z_grid = ym.double().unsqueeze(-1) + ys.double().unsqueeze(-1) * self.centers.double()  # [B, Nq, 1024]
        y_grid = inv_fn(z_grid)
        return (probs * y_grid).sum(-1)


def n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
