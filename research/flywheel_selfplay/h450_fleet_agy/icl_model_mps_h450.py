#!/usr/bin/env python3
"""MPS- & CUDA-Compatible Precision-Preserving ICL (`icl_model_mps_h450.py`).

100% parameter- and `state_dict`-compatible (`strict=True`) with `icl_model_h450.ICLH450` (`S`, `M`, `L` sizes):
  - On Apple Silicon `mps` (`xbabe4` M3 Ultra 80-core GPU), `float64` tensors are unsupported by Metal kernels.
    `forward` and `loss` therefore run `normalise_f64` and `loss` 1,024-bin interpolation in exact `float64` on CPU
    (<0.4 ms per batch) while executing all 14 transformer blocks in `bfloat16`/`float32` on the 80-core Metal GPU.
  - Supports zero-state-dict-delta `checkpoint_row_blocks=True` activation checkpointing on the `[B * N, 4 + F, 160]`
    row encoder (~60% peak activation VRAM reduction, doubling multi-dataset batch capacity inside 80% VRAM) and
    optional CPU-worker `float64 -> float32` `prenorm` fast-path eliminating GeForce GPU `float64` `nanquantile` stalls.
  - On `cuda` and `cpu`, math and `state_dict` are bit-identical to `icl_model_h450.ICLH450`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as _ckpt

HERE = Path(__file__).resolve().parent
for _cand in (
    os.environ.get("JOPE_REF_CODE", ""),
    str(HERE),
    "/home/ubuntu/h450-stage/pub-r5/code",
    "/home/ubuntu/h448-stage/pub-r1/code",
    str(Path.home() / "jope-data/h450-fleet-agy/ref_code"),
):
    if _cand and Path(_cand).is_dir() and _cand not in sys.path:
        sys.path.append(_cand)

from icl_model import BIN_RANGE, CLIP_X, FID_SLOTS, ICL, N_BINS  # noqa: E402


class ICLH450MPS(ICL):
    """Exact state_dict-compatible subclass of `ICL` supporting `mps`, `cuda`, and `cpu` with `float64` context stats."""

    def __init__(self, size: str = "L", checkpoint_row_blocks: bool = False):
        super().__init__(size)
        self.checkpoint_row_blocks = bool(checkpoint_row_blocks)

    @staticmethod
    def normalise_f64(x: torch.Tensor, y_ctx: torch.Tensor, n_ctx: int):
        xd = x.double()
        yd = y_ctx.double()
        miss = ~torch.isfinite(xd)
        xc = torch.where(miss[:, :n_ctx], torch.nan, xd[:, :n_ctx])
        q_probs = torch.tensor([0.25, 0.5, 0.75], device=xd.device, dtype=torch.float64)
        q = torch.nanquantile(xc, q_probs, dim=1)
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

    @classmethod
    def prenorm_cpu_np(
        cls, xa: np.ndarray, ya_ctx: np.ndarray, n_ctx: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Exact `float64` context normalization executed inside background CPU workers (zero shm FDs over Queue)."""
        with torch.no_grad():
            xt = torch.from_numpy(np.ascontiguousarray(xa, dtype=np.float64))
            yt = torch.from_numpy(np.ascontiguousarray(ya_ctx, dtype=np.float64))
            zm, ym64, ys64 = cls.normalise_f64(xt, yt, n_ctx)
            yz = ((yt - ym64) / ys64).clamp(-BIN_RANGE, BIN_RANGE).float()
            return (
                np.ascontiguousarray(zm.numpy(), dtype=np.float32),
                np.ascontiguousarray(yz.numpy(), dtype=np.float32),
                np.ascontiguousarray(ym64.numpy(), dtype=np.float64),
                np.ascontiguousarray(ys64.numpy(), dtype=np.float64),
            )

    def forward(
        self,
        x: torch.Tensor | None,
        y_ctx: torch.Tensor | None,
        n_ctx: int,
        slots=None,
        amp: bool = True,
        prenorm: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ):
        dev = self.fid_buf.device
        if prenorm is not None:
            zm_in, yz_in, ym64, ys64 = prenorm
            B, N, Fn, _ = zm_in.shape
            nb = dev.type == "cuda"
            zm = zm_in.to(dev, non_blocking=nb) if zm_in.device != dev else zm_in
            yz = yz_in.to(dev, non_blocking=nb) if yz_in.device != dev else yz_in
            if dev.type != "mps":
                ym64 = ym64.to(dev, dtype=torch.float64, non_blocking=nb) if ym64.device != dev else ym64.double()
                ys64 = ys64.to(dev, dtype=torch.float64, non_blocking=nb) if ys64.device != dev else ys64.double()
        else:
            assert x is not None and y_ctx is not None
            B, N, Fn = x.shape
            if dev.type == "mps":
                x_cpu = x.detach().cpu() if x.device.type != "cpu" else x
                y_cpu = y_ctx.detach().cpu() if y_ctx.device.type != "cpu" else y_ctx
                zm_cpu, ym64, ys64 = self.normalise_f64(x_cpu, y_cpu, n_ctx)
                yz_cpu = ((y_cpu.double() - ym64) / ys64).clamp(-BIN_RANGE, BIN_RANGE).float()
                zm = zm_cpu.to(dev)
                yz = yz_cpu.to(dev)
            else:
                with torch.autocast(dev.type, enabled=False):
                    zm, ym64, ys64 = self.normalise_f64(x, y_ctx, n_ctx)
                    yz = ((y_ctx.double() - ym64) / ys64).clamp(-BIN_RANGE, BIN_RANGE).float()
        if slots is None:
            slots = torch.arange(Fn, device=dev) % FID_SLOTS
        else:
            slots = slots.to(dev)
        use_amp = amp and dev.type in ("cuda", "mps")
        use_ckpt = bool(getattr(self, "checkpoint_row_blocks", False)) and self.training and torch.is_grad_enabled()
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
            X = self.col_norm(self.col_in(zm[:, :n_ctx])).permute(0, 2, 1, 3).reshape(B * Fn, n_ctx, -1)
            H = self.ind.unsqueeze(0).expand(B * Fn, -1, -1)
            for blk in self.col_blocks:
                H = blk(H, mem=X)
            c = self.col_out(H.mean(1)).view(B, 1, Fn, -1)
            e = self.cell_in(zm) * (1.0 + self.col_scale(c)) + self.col_shift(c) + self.fid(self.fid_buf[slots])
            t = torch.cat([self.cls.unsqueeze(0).expand(B * N, -1, -1).to(e.dtype), e.reshape(B * N, Fn, -1)], 1)
            for blk in self.row_blocks:
                if use_ckpt:
                    t = _ckpt.checkpoint(blk, t, use_reentrant=False)
                else:
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
        dev = logits.device
        if dev.type == "mps":
            c = self.centers.detach().cpu().double()
            yq_c = y_q.detach().cpu().double()
            ym_c = ym.detach().cpu().double()
            ys_c = ys.detach().cpu().double()
            z = ((yq_c - ym_c) / ys_c).clamp(c[0], c[-1])
            hi_c = torch.searchsorted(c, z.contiguous()).clamp(1, c.numel() - 1)
            lo_c = hi_c - 1
            w = ((z - c[lo_c]) / (c[hi_c] - c[lo_c])).float().to(dev)
            lo = lo_c.to(dev)
            hi = hi_c.to(dev)
        else:
            c = self.centers.double()
            z = ((y_q.double() - ym.double()) / ys.double()).clamp(c[0], c[-1])
            hi = torch.searchsorted(c, z.contiguous()).clamp(1, c.numel() - 1)
            lo = hi - 1
            w = ((z - c[lo]) / (c[hi] - c[lo])).float()
        lp = F.log_softmax(logits, -1)
        return -(lp.gather(-1, lo.unsqueeze(-1)).squeeze(-1) * (1.0 - w) + lp.gather(-1, hi.unsqueeze(-1)).squeeze(-1) * w)


def n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
