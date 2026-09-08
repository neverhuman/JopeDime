"""Fusion trainer: P0 parity -> phased train on the expert cache -> eval vs
best-expert / average / oracle baselines on HELD-OUT datasets.

Fast: trains on the precomputed cache with zero expert forwards.
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from .model import RegressionFusion
from .losses import regression_step_loss, cvar_tail


def rmse_t(pred, y):
    return torch.sqrt(torch.mean((pred - y) ** 2))


def baselines(entries):
    """Per-dataset RMSE for each single expert, their average, and the oracle."""
    E = entries[0]["means"].shape[1]
    exp = np.zeros(E); avg = 0.0; oracle = 0.0
    for e in entries:
        y = e["y"]
        rs = [float(rmse_t(e["means"][:, i], y)) for i in range(E)]
        for i in range(E):
            exp[i] += rs[i]
        avg += float(rmse_t(e["means"].mean(1), y))
        oracle += min(rs)
    n = len(entries)
    return exp / n, avg / n, oracle / n


def evaluate(model, entries, device):
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for e in entries:
            out = model(e["means"].to(device), e["hidden"].to(device),
                        e["gate_meta"].to(device).unsqueeze(0).expand(e["means"].shape[0], -1),
                        e["conf"].to(device))
            tot += float(rmse_t(out["final"], e["y"].to(device)))
    return tot / len(entries)


def parity_check(model, entries, device):
    """P0: forcing gate one-hot + identity calibrator + zero residual reproduces
    each expert's own prediction exactly."""
    e = entries[0]
    worst = 0.0
    with torch.no_grad():
        for i in range(model.n_experts):
            out = model(e["means"].to(device), e["hidden"].to(device),
                        e["gate_meta"].to(device).unsqueeze(0).expand(e["means"].shape[0], -1),
                        e["conf"].to(device), parity_expert=i)
            worst = max(worst, float((out["final"] - e["means"][:, i].to(device)).abs().max()))
    return worst


def anneal(step, total):
    f = step / max(total, 1)
    # distillation to the per-row ORACLE expert is the primary routing teacher
    # (strong throughout, gentle floor); entropy only prevents early collapse.
    return dict(cal=0.2, distill=0.4 * max(0.15, 1 - f),
                ent=0.01 * max(0.0, 1 - 5 * f),                 # off after 20%
                sp=1e-5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from .cache import load_cache
    entries = load_cache(args.cache)
    rng = np.random.default_rng(args.seed); rng.shuffle(entries)
    nval = max(1, int(args.val_frac * len(entries)))
    val, train = entries[:nval], entries[nval:]
    dev = torch.device(args.device)
    E = train[0]["means"].shape[1]
    print(f"cache: {len(entries)} datasets ({len(train)} train / {len(val)} val), {E} experts")

    exp, avg, orac = baselines(val)
    print(f"BASELINE (val): experts RMSE {np.round(exp,4).tolist()} | avg {avg:.4f} | ORACLE {orac:.4f}")

    model = RegressionFusion(n_experts=E, gate_meta_dim=train[0]["gate_meta"].numel(),
                             conf_dim=train[0]["conf"].shape[-1]).to(dev)
    n_train = sum(p.numel() for p in model.fusion_parameters())
    print(f"fusion trainable params: {n_train:,}")
    print(f"P0 PARITY max|Δ| (want ~0): {parity_check(model, entries, dev):.2e}")

    opt = torch.optim.AdamW(model.fusion_parameters(), lr=args.lr, weight_decay=1e-5)
    total_steps = args.epochs * len(train)
    step = 0; best = 1e9
    for ep in range(args.epochs):
        model.train(); order = rng.permutation(len(train))
        for j in order:
            e = train[j]; lam = anneal(step, total_steps)
            M = e["means"].shape[0]
            out = model(e["means"].to(dev), e["hidden"].to(dev),
                        e["gate_meta"].to(dev).unsqueeze(0).expand(M, -1), e["conf"].to(dev))
            best_e = e["means"].to(dev)[torch.arange(M),
                     torch.stack([(e["means"][:, i].to(dev) - e["y"].to(dev)).abs()
                                  for i in range(E)]).argmin(0)]
            loss, parts = regression_step_loss(out, e["y"].to(dev), best_e, lam)
            loss = loss + lam["sp"] * sum((p * p).sum() for p in model.fusion_parameters())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.fusion_parameters(), 1.0); opt.step()
            step += 1
        vr = evaluate(model, val, dev)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"ep {ep:3d} | val RMSE {vr:.4f} (best-exp {min(exp):.4f}, avg {avg:.4f}, "
                  f"oracle {orac:.4f}) | resid_scale {float(model.residual_scale):.3f} "
                  f"| gate_ent {parts['ent']:.2f}")
        best = min(best, vr)
    print(f"\nBEST val RMSE {best:.4f}  vs  best-expert {min(exp):.4f} / avg {avg:.4f} / "
          f"oracle {orac:.4f}  ->  closed {100*(min(exp)-best)/max(min(exp)-orac,1e-9):.0f}% of the oracle gap")


if __name__ == "__main__":
    main()
