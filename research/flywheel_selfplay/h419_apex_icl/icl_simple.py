#!/usr/bin/env python3
"""H419 diagnostic: a checkpoint's EMA weights on synthetic tasks of chosen prior families (plain and residual
readout) against ridge.   icl_simple.py --ckpt C --size M --device mps [--probs simple|v1|v2] [--out OUT.json]"""
import argparse
import json

import torch

import icl_resid as RES
from icl_model import ICL
from icl_prior import KINDS, PROBS, PROBS_SIMPLE, PROBS_V2, Prior

SHAPES = ((512, 6), (512, 20), (1024, 10), (1024, 48), (1024, 80))


def c(t):
    return t.detach().cpu().double()


def u_score(pred, yq, ym):
    r = torch.sqrt(((pred - yq) ** 2).mean(1)); r0 = torch.sqrt(((ym - yq) ** 2).mean(1)).clamp_min(1e-30)
    return (-torch.log((r / r0).clamp_min(1e-30))).clamp(-2, 6)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--size", required=True)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--probs", default="simple")
    ap.add_argument("--batches", type=int, default=3); ap.add_argument("--out")
    a = ap.parse_args()
    dev = torch.device(a.device)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = ICL(a.size); model.load_state_dict(ck["ema"]); model.eval().to(dev)
    probs = {"simple": PROBS_SIMPLE, "v1": PROBS, "v2": PROBS_V2}[a.probs]
    prior = Prior(dev, 419_600_001)
    rows = []
    for N, Fn in SHAPES:
        for b in range(a.batches):
            x, y, kinds = prior.sample(16, N, Fn, probs=probs)
            n = N * 3 // 4
            ymc = c(y[:, :n]).mean(1, keepdim=True)
            lg, ym, ys = model(x, y[:, :n], n)
            u = u_score(c(model.mean(lg, ym, ys)), c(y[:, n:]), ymc)
            base_o, base_q = RES.ridge_base(x, y[:, :n], n, torch.Generator().manual_seed(len(rows)))
            ur = u_score(base_q.cpu(), c(y[:, n:]), ymc)
            yr, used, bq = RES.residual_targets(x, y, n, torch.Generator().manual_seed(len(rows)))
            lg, ym2, ys2 = model(x, yr[:, :n], n)
            ures = u_score(c(model.mean(lg, ym2, ys2)) + torch.where(c(used)[:, None] > 0, bq.cpu(), torch.zeros_like(bq.cpu())), c(y[:, n:]), ymc)
            for i in range(16):
                rows.append((KINDS[int(kinds[i])], float(u[i]), float(ur[i]), float(ures[i])))
    out = {}
    for k in KINDS:
        s = [r for r in rows if r[0] == k]
        if s:
            med = lambda j: sorted(r[j] for r in s)[len(s) // 2]
            out[k] = {"n": len(s), "u_model": sum(r[1] for r in s) / len(s), "u_ridge": sum(r[2] for r in s) / len(s),
                      "u_resid": sum(r[3] for r in s) / len(s), "median_model": med(1), "median_ridge": med(2),
                      "median_resid": med(3), "win_model_vs_ridge": sum(r[1] > r[2] for r in s) / len(s),
                      "win_resid_vs_ridge": sum(r[3] > r[2] for r in s) / len(s)}
            print(k, json.dumps({kk: round(v, 3) for kk, v in out[k].items()}))
    if a.out:
        with open(a.out, "x") as f:
            json.dump({"ckpt": a.ckpt, "state": ck.get("state"), "probs": a.probs, "result": out}, f)


if __name__ == "__main__":
    main()
