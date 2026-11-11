#!/usr/bin/env python3
"""H431 (CW-3 Part B, Factor D): Stronger in-context residual base in PyTorch.

Computes 5-fold out-of-fold predictions on context rows [:n_ctx] and full-context predictions on query rows [n_ctx:]
for ridge + 30 rounds of greedy boosted 16-bin feature stumps (learning rate 0.15, L2 leaf shrinkage 5.0).
Everything is fitted strictly on context rows [:n_ctx] (quantile thresholds, ridge weights, stump splits, leaf values).
"""
from __future__ import annotations

import torch

import icl_resid as RES

ROUNDS = 30
LR = 0.15
N_THRESH = 15   # 15 interior quantile thresholds -> 16 bins
LEAF_L2 = 5.0


@torch.no_grad()
def _boosted_stumps_fit_predict(xc: torch.Tensor, rc: torch.Tensor, x_eval: torch.Tensor,
                                rounds: int = ROUNDS, lr: float = LR) -> torch.Tensor:
    """Batched greedy boosted stumps: xc [B, Nc, F], rc [B, Nc], x_eval [B, Ne, F] -> pred_eval [B, Ne] (float64)."""
    B, Nc, Fn = xc.shape
    qs = torch.linspace(1.0 / (N_THRESH + 1), N_THRESH / (N_THRESH + 1), N_THRESH, dtype=xc.dtype, device=xc.device)
    # Thresholds strictly from training context rows: [T, B, F] -> [B, 1, F, T]
    thrs = torch.quantile(xc, qs, dim=1).permute(1, 2, 0).unsqueeze(1)
    ind_c = (xc.unsqueeze(-1) > thrs).to(xc.dtype)          # [B, Nc, F, T]
    ind_e = (x_eval.unsqueeze(-1) > thrs).to(xc.dtype)      # [B, Ne, F, T]

    n_r = ind_c.sum(dim=1)                                  # [B, F, T]
    n_l = float(Nc) - n_r                                   # [B, F, T]
    denom_r = n_r + LEAF_L2
    denom_l = n_l + LEAF_L2

    res = rc.clone()
    pred_e = torch.zeros((B, x_eval.shape[1]), dtype=xc.dtype, device=xc.device)
    b_idx = torch.arange(B, device=xc.device)

    for _ in range(rounds):
        tot = res.sum(dim=1).view(B, 1, 1)                  # [B, 1, 1]
        sum_r = torch.einsum("bn,bnft->bft", res, ind_c)    # [B, F, T]
        sum_l = tot - sum_r
        gain = (sum_r * sum_r) / denom_r + (sum_l * sum_l) / denom_l   # [B, F, T]
        flat = gain.reshape(B, Fn * N_THRESH)
        best = flat.argmax(dim=1)                           # [B]
        best_f = best // N_THRESH
        best_t = best % N_THRESH

        vr = lr * sum_r[b_idx, best_f, best_t] / denom_r[b_idx, best_f, best_t]   # [B]
        vl = lr * sum_l[b_idx, best_f, best_t] / denom_l[b_idx, best_f, best_t]   # [B]

        split_c = ind_c[b_idx, :, best_f, best_t]           # [B, Nc]
        split_e = ind_e[b_idx, :, best_f, best_t]           # [B, Ne]
        step_c = vl[:, None] + (vr - vl)[:, None] * split_c
        step_e = vl[:, None] + (vr - vl)[:, None] * split_e
        res = res - step_c
        pred_e = pred_e + step_e

    return pred_e


@torch.no_grad()
def strong_residual_targets(x: torch.Tensor, y: torch.Tensor, n_ctx: int, gen: torch.Generator | None = None,
                            rounds: int = ROUNDS, lr: float = LR) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same contract as icl_resid.residual_targets: returns (y_res [B,N], used [B] bool, base_q [B,Nq] float64)."""
    # 1. Standard 5-fold ridge base
    oof_r, q_r = RES.ridge_base(x, y[:, :n_ctx], n_ctx, gen)
    xd = torch.nan_to_num(RES._f64(x))
    yd = RES._f64(y)
    yc = yd[:, :n_ctx]
    B, _, _ = xd.shape

    # 2. 5-fold OOF boosted stumps on the ridge residual (context only) + full-context fit on query rows
    order = torch.argsort(torch.rand(B, n_ctx, generator=gen), dim=1)
    fold = torch.empty_like(order)
    fold.scatter_(1, order, (torch.arange(n_ctx) % RES.FOLDS).expand(B, -1))
    fold = fold.to(xd.device)

    rc_full = yc - oof_r
    oof_stump = torch.zeros_like(yc)
    xc = xd[:, :n_ctx]
    xq = xd[:, n_ctx:]

    for k in range(RES.FOLDS):
        # Gather exact 4/5 training rows per task (balanced folds -> fixed n_tr across batch)
        tr_mask = fold != k
        va_mask = fold == k
        # Sort mask per row so gather shapes are uniform across B
        n_va = int(va_mask[0].sum().item())
        n_tr = n_ctx - n_va
        tr_idx = torch.argsort(tr_mask.int(), dim=1, descending=True)[:, :n_tr]
        va_idx = torch.argsort(va_mask.int(), dim=1, descending=True)[:, :n_va]
        xc_tr = xc.gather(1, tr_idx.unsqueeze(-1).expand(-1, -1, xc.shape[-1]))
        rc_tr = rc_full.gather(1, tr_idx)
        xc_va = xc.gather(1, va_idx.unsqueeze(-1).expand(-1, -1, xc.shape[-1]))
        pred_va = _boosted_stumps_fit_predict(xc_tr, rc_tr, xc_va, rounds=rounds, lr=lr)
        oof_stump.scatter_(1, va_idx, pred_va)

    q_stump = _boosted_stumps_fit_predict(xc, rc_full, xq, rounds=rounds, lr=lr)
    oof_tot = oof_r + oof_stump
    q_tot = q_r + q_stump

    rc = yc - oof_tot
    used = rc.std(1) >= RES.MIN_RES_SD * yc.std(1).clamp_min(1e-30)
    used &= torch.isfinite(rc).all(1) & torch.isfinite(q_tot).all(1)
    res = torch.cat([rc, yd[:, n_ctx:] - q_tot], dim=1)
    out = torch.where(used[:, None], res, yd)
    return out.to(dtype=y.dtype, device=y.device), used.to(y.device), q_tot
