#!/usr/bin/env python3
"""H419 addendum 5: an in-context ridge base and residual targets.

The in-context model learns nonlinear structure well and the linear part of a target badly; ridge fits the linear
part exactly. With residual targets the model predicts what ridge leaves over, and the final prediction is
ridge + model. Everything is fitted on context rows only:

  ridge_base        ridge (LAMBDA on features standardised by the context) -> its prediction for every row:
                    context rows get FOLDS-fold out-of-fold predictions, query rows the full-context fit.
  residual_targets  y with those predictions subtracted. A task whose context residual has (almost) no spread
                    (an exactly linear target) keeps its raw target; `used` says which tasks were transformed.

The same code runs in training (icl_train.py --p-resid) and at inference (icl_diag.py, the monitor), so the
out-of-fold / full-fit asymmetry between context and query rows is the same in both.
float64 throughout (on the CPU for the Apple GPU backend, which has none).
"""
import torch

LAMBDA = 1.0        # the monitor's ridge reference (icl_train.ridge_u)
FOLDS = 5
MIN_RES_SD = 1e-3   # residual sd below this share of the target sd: the task keeps its raw target


def _f64(t):
    return t.detach().cpu().double() if t.device.type == "mps" else t.detach().double()


@torch.no_grad()
def ridge_base(x, y_ctx, n_ctx, gen=None, lam=LAMBDA, folds=FOLDS):
    """x [B,N,F] raw (NaN = missing -> 0), y_ctx [B,n_ctx]. Returns (oof [B,n_ctx], q [B,N-n_ctx]), float64.
    gen: a CPU torch.Generator for the fold assignment (None = the global generator)."""
    xd, yc = torch.nan_to_num(_f64(x)), _f64(y_ctx)
    B, _, Fn = xd.shape
    xc = xd[:, :n_ctx]
    mu, sd = xc.mean(1, keepdim=True), xc.std(1, keepdim=True).clamp_min(1e-9)
    z = (xd - mu) / sd
    zc, zq = z[:, :n_ctx], z[:, n_ctx:]
    eye = lam * torch.eye(Fn, dtype=z.dtype, device=z.device)
    G = zc.transpose(1, 2) @ zc                        # [B,F,F]
    s, b, sy = zc.sum(1), (zc * yc.unsqueeze(-1)).sum(1), yc.sum(1)
    ym = sy / n_ctx
    w = torch.linalg.solve(G + eye, (b - ym[:, None] * s).unsqueeze(-1))
    q = (zq @ w).squeeze(-1) + ym[:, None]
    order = torch.argsort(torch.rand(B, n_ctx, generator=gen), 1)
    fold = torch.empty_like(order)
    fold.scatter_(1, order, (torch.arange(n_ctx) % folds).expand(B, -1))   # balanced random folds per task
    fold = fold.to(z.device)
    oof = torch.zeros_like(yc)
    for k in range(folds):
        m = fold == k
        zk = zc * m.unsqueeze(-1)
        n_tr = n_ctx - m.sum(1)
        ymk = (sy - (yc * m).sum(1)) / n_tr
        rhs = (b - (zk * yc.unsqueeze(-1)).sum(1)) - ymk[:, None] * (s - zk.sum(1))
        wk = torch.linalg.solve(G - zk.transpose(1, 2) @ zc + eye, rhs.unsqueeze(-1))
        oof = torch.where(m, (zc @ wk).squeeze(-1) + ymk[:, None], oof)
    return oof, q


@torch.no_grad()
def residual_targets(x, y, n_ctx, gen=None):
    """y [B,N] (context rows first) -> (y_res like y, used [B] bool, base_q [B,Nq] float64).
    y_res = y - ridge_base for the tasks in `used`, y itself for the others; base_q is the ridge prediction for the
    query rows (add it back to the model's residual prediction for the tasks in `used`)."""
    oof, q = ridge_base(x, y[:, :n_ctx], n_ctx, gen)
    yd = _f64(y)
    rc = yd[:, :n_ctx] - oof
    used = rc.std(1) >= MIN_RES_SD * yd[:, :n_ctx].std(1).clamp_min(1e-30)
    used &= torch.isfinite(rc).all(1) & torch.isfinite(q).all(1)
    res = torch.cat([rc, yd[:, n_ctx:] - q], 1)
    out = torch.where(used[:, None], res, yd)
    return out.to(dtype=y.dtype, device=y.device), used.to(y.device), q
