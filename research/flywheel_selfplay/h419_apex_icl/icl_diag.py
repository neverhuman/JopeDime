#!/usr/bin/env python3
"""H419 diagnostic on the real-table monitor (FIT-SEL SEARCH rows, icl_bank.py val): per-context u of a checkpoint's EMA
weights under several inference recipes, with the ridge reference. TRAIN-side only; no test cohort is read.

  icl_diag.py --ckpt CKPT --size L --val fitsel_search_val.npz --out OUT.json [--device cpu|cuda|mps] [--variants a,b,...]

Recipes (all use context rows only for anything that is fitted):
  full            one pass, the whole context (what icl_train.py logs)
  subK_n          K seeded random context subsets of n rows, predictions averaged
  permK           K passes with permuted feature order and random identity slots, averaged
  ridgefeat       one extra feature: a ridge prediction (5-fold out-of-fold on context rows, full-context fit on queries)
  ridgeres        the model predicts the residual of that ridge (context targets = out-of-fold residuals), by
                  icl_resid.residual_targets, the code the trainer uses; ridgeres0 = the first diagnostic version
  top100          contexts wider than 100 features keep the 100 with the largest |correlation| to the context target
Suffix "+permK" on ridgefeat / ridgeres / top100 adds the feature-permutation average.
"""
import argparse
import json
import math
import random

import numpy as np
import torch

import icl_resid as RES
from icl_inject import RealVal
from icl_model import ICL

RIDGE_LAMBDA = 1.0   # the monitor's ridge reference (icl_train.ridge_u)


def u_of(pred, yq, ym):
    r = float(torch.sqrt(((pred - yq) ** 2).mean()))
    r0 = max(float(torch.sqrt(((ym - yq) ** 2).mean())), 1e-30)
    return max(-2.0, min(6.0, -math.log(max(r / r0, 1e-30))))


def ridge_fit_predict(xc, yc, xs):
    """Standardise on xc, ridge with RIDGE_LAMBDA, predict each tensor in xs. float64."""
    mu, sd = xc.mean(0, keepdim=True), xc.std(0, keepdim=True).clamp_min(1e-9)
    a = (xc - mu) / sd
    ym = yc.mean()
    A = a.T @ a + RIDGE_LAMBDA * torch.eye(a.shape[1], dtype=torch.float64)
    w = torch.linalg.solve(A, a.T @ (yc - ym))
    return [((x - mu) / sd) @ w + ym for x in xs]


def ridge_columns(x, y, n_ctx, seed):
    """(out-of-fold ridge prediction on context rows, full-context ridge prediction on query rows), float64."""
    x = torch.nan_to_num(x.double()); y = y.double()
    xc, yc, xq = x[:n_ctx], y[:n_ctx], x[n_ctx:]
    g = torch.Generator().manual_seed(seed)
    fold = torch.randperm(n_ctx, generator=g) % 5
    oof = torch.empty(n_ctx, dtype=torch.float64)
    for k in range(5):
        tr, te = fold != k, fold == k
        oof[te] = ridge_fit_predict(xc[tr], yc[tr], [xc[te]])[0]
    return oof, ridge_fit_predict(xc, yc, [xq])[0]


@torch.no_grad()
def predict(model, x, y_ctx, n_ctx, perms=0, seed=0):
    """Mean prediction for the query rows [Nq] (CPU float64); perms > 0 averages over feature permutations with
    random identity slots. Inputs are CPU tensors; the forward pass runs on the model's device."""
    dev = next(model.parameters()).device
    Fn = x.shape[-1]
    yc = y_ctx.float()[None].to(dev)
    if not perms:
        lg, ym, ys = model(x.float()[None].to(dev), yc, n_ctx)
        return model.mean(lg, ym, ys)[0].float().cpu().double()
    g = torch.Generator().manual_seed(seed)
    acc = 0.0
    for _ in range(perms):
        p = torch.randperm(Fn, generator=g)
        slots = torch.randperm(256, generator=g)[:Fn] if Fn <= 256 else torch.randint(0, 256, (Fn,), generator=g)
        lg, ym, ys = model(x[:, p].float()[None].to(dev), yc, n_ctx, slots=slots.to(dev))
        acc = acc + model.mean(lg, ym, ys)[0].float().cpu().double()
    return acc / perms


def run_variant(model, name, x, y, n_ctx, seed):
    base, _, ens = name.partition("+")
    perms = int(ens[4:]) if ens.startswith("perm") else 0
    yc, xq_all = y[:n_ctx], x[n_ctx:]
    if base == "full":
        return predict(model, x, yc, n_ctx, perms, seed)
    if base.startswith("perm"):
        return predict(model, x, yc, n_ctx, int(base[4:]), seed)
    if base.startswith("sub"):
        k, n = (int(v) for v in base[3:].split("_"))
        n = min(n, n_ctx)
        g = torch.Generator().manual_seed(seed)
        acc = 0.0
        for _ in range(k):
            idx = torch.randperm(n_ctx, generator=g)[:n]
            acc = acc + predict(model, torch.cat([x[idx], xq_all]), yc[idx], n, perms, seed)
        return acc / k
    if base == "top100":
        if x.shape[1] > 100:
            xc = torch.nan_to_num(x[:n_ctx].double())
            c = ((xc - xc.mean(0)) * (yc.double() - yc.double().mean())[:, None]).mean(0) / xc.std(0).clamp_min(1e-12)
            x = x[:, torch.argsort(c.abs(), descending=True)[:100]]
        return predict(model, x, yc, n_ctx, perms, seed)
    if base == "ridgeres":   # the training-time procedure (icl_resid.py)
        yr, used, bq = RES.residual_targets(x[None], y[None], n_ctx, torch.Generator().manual_seed(seed))
        res = predict(model, x, yr[0, :n_ctx], n_ctx, perms, seed)
        return res + bq[0].cpu() if bool(used[0]) else res
    if base in ("ridgefeat", "ridgeres0"):   # ridgeres0: the first version (features standardised per fold)
        oof, rq = ridge_columns(x, y, n_ctx, seed)
        if base == "ridgefeat":
            col = torch.cat([oof, rq]).float()[:, None]
            return predict(model, torch.cat([x, col], 1), yc, n_ctx, perms, seed)
        res = predict(model, x, (yc.double() - oof).float(), n_ctx, perms, seed)
        return rq + res
    raise SystemExit(f"unknown variant {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--size", required=True)
    ap.add_argument("--val", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=16); ap.add_argument("--device", default="cpu")
    ap.add_argument("--variants", default="full,sub4_320,sub4_560,perm4,ridgefeat,ridgeres,top100")
    ap.add_argument("--weights", default="ema", choices=("ema", "model"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    random.seed(0); torch.manual_seed(0)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = ICL(a.size)
    model.load_state_dict(ck[a.weights]); model.eval(); model.to(torch.device(a.device))
    rv = RealVal(a.val, "cpu")
    names = a.variants.split(",")
    rows = []
    for i, (m, n_ctx) in enumerate(rv.items):
        x, y = m[:, :-1], m[:, -1]
        yq, ym = y[n_ctx:].double(), y[:n_ctx].double().mean()
        rq = ridge_fit_predict(torch.nan_to_num(x[:n_ctx].double()), y[:n_ctx].double(),
                               [torch.nan_to_num(x[n_ctx:].double())])[0]
        rec = {"i": i, "family": rv.family[i], "n_ctx": n_ctx, "n_q": int(yq.numel()), "F": int(x.shape[1]),
               "u_ridge": u_of(rq, yq, ym), "u": {}}
        for v in names:
            rec["u"][v] = u_of(run_variant(model, v, x, y, n_ctx, 419_500 + i), yq, ym)
        rows.append(rec)
        print(json.dumps(rec), flush=True)
    med = lambda xs: sorted(xs)[len(xs) // 2]
    summ = {"ridge": {"mean": float(np.mean([r["u_ridge"] for r in rows])), "median": med([r["u_ridge"] for r in rows])}}
    for v in names:
        us = [r["u"][v] for r in rows]
        summ[v] = {"win": float(np.mean([r["u"][v] > r["u_ridge"] for r in rows])), "median": med(us),
                   "mean": float(np.mean(us)),
                   "win_F_le32": float(np.mean([r["u"][v] > r["u_ridge"] for r in rows if r["F"] <= 32])),
                   "win_F_33_100": float(np.mean([r["u"][v] > r["u_ridge"] for r in rows if 32 < r["F"] <= 100])),
                   "win_F_gt100": float(np.mean([r["u"][v] > r["u_ridge"] for r in rows if r["F"] > 100]))}
    out = {"schema": "jope.h419.icl-diag/v1", "ckpt": a.ckpt, "weights": a.weights, "size": a.size,
           "state": ck.get("state"), "n": len(rows), "summary": summ, "rows": rows}
    with open(a.out, "x") as f:
        json.dump(out, f)
    print(json.dumps({"state": ck.get("state"), "summary": summ}, indent=1))


if __name__ == "__main__":
    main()
