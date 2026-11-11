#!/usr/bin/env python3
"""H422 development look 5, DESIGN contexts only: the robust row-wise combiner family found in look 4. Weights stay
the stored SXB weights; only the function that combines the members' predictions of one row changes. No AUDIT target
enters a prediction. In-sample; used to fix the stacker's form before the PREREG.

  explore5_h422.py CACHE_DIR [CACHE_DIR ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_h422 import load, norm, report, u_of  # noqa: E402

CLIP = 0.5


def u_pred(pred, y, b):
    return -np.log(max(float(np.sqrt(np.mean((pred - y) ** 2))) / float(b), 1e-12))


def wquant(P, w, q):
    o = np.argsort(P, axis=1)
    cw = np.cumsum(w[o], axis=1)
    k = (cw >= q).argmax(axis=1)
    return P[np.arange(len(P)), o[np.arange(len(P)), k]]


def wtrim(P, w, t):
    """Weighted mean of the central 1 - 2t of the weight mass of each row."""
    o = np.argsort(P, axis=1)
    ws = w[o]
    cw = np.cumsum(ws, axis=1)
    lo = np.clip(cw - t, 0, None)
    keep = np.minimum(np.minimum(ws, lo), np.clip((1 - t) - (cw - ws), 0, None))
    keep = np.clip(keep, 0, None)
    Ps = np.take_along_axis(P, o, axis=1)
    return (Ps * keep).sum(1) / keep.sum(1)


def main() -> None:
    ctx = load(sys.argv[1:])
    base = np.array([u_of(c, c["w_sxb"]) for c in ctx])
    for c in ctx:
        s = c["w_sxb"] > 0
        c["w"] = norm(c["w_sxb"][s])
        c["A"] = 0.5 * c["BAG"][:, s] + 0.5 * c["REF"][:, s]
        c["A2"] = np.concatenate([c["BAG"][:, s], c["REF"][:, s]], axis=1)
        c["Qs"] = c["Q"][:, s]
        c["mean"] = c["A"] @ c["w"]
        c["med"] = wquant(c["A"], c["w"], 0.5)
        c["med2"] = wquant(c["A2"], np.concatenate([c["w"], c["w"]]) / 2, 0.5)
    for lam in (0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0):
        report(f"median blend {lam}", ctx, base, [u_pred((1 - lam) * c["mean"] + lam * c["med"], c["ya"], c["ba"]) for c in ctx])
    for lam in (0.25, 0.5):
        report(f"median over bag+refit columns {lam}", ctx, base,
               [u_pred((1 - lam) * c["mean"] + lam * c["med2"], c["ya"], c["ba"]) for c in ctx])
    for t in (0.05, 0.1, 0.2, 0.3, 0.4):
        report(f"trimmed mean {t}", ctx, base, [u_pred(wtrim(c["A"], c["w"], t), c["ya"], c["ba"]) for c in ctx])
    # the same change on the SEARCH out-of-fold rows (what a per-context choice from SEARCH data would see)
    g_a, g_s, feats = [], [], []
    for c in ctx:
        ms, md = c["Qs"] @ c["w"], wquant(c["Qs"], c["w"], 0.5)
        sd = c["y"].std()
        us = u_pred(0.5 * ms + 0.5 * md, c["y"], sd) - u_pred(ms, c["y"], sd)
        ua = u_pred(0.5 * c["mean"] + 0.5 * c["med"], c["ya"], c["ba"]) - u_pred(c["mean"], c["ya"], c["ba"])
        g_s.append(us)
        g_a.append(ua)
        disp_a = np.sqrt((((c["A"] - c["mean"][:, None]) ** 2) @ c["w"]).mean())
        disp_s = np.sqrt((((c["Qs"] - ms[:, None]) ** 2) @ c["w"]).mean())
        res_s = np.sqrt(np.mean((ms - c["y"]) ** 2))
        feats.append([np.log(len(c["y"])), u_pred(ms, c["y"], sd), len(c["w"]), 1 / (c["w"] ** 2).sum(),
                      np.log(disp_s / res_s + 1e-12), np.log(disp_a / disp_s + 1e-12),
                      np.log(np.abs(c["med"] - c["mean"]).mean() / (disp_s + 1e-12) + 1e-12)])
    g_a, g_s, F = np.array(g_a), np.array(g_s), np.array(feats)
    ga = np.clip(g_a, -CLIP, CLIP)
    print(f"median blend 0.5: AUDIT gain mean {ga.mean():+.4f}; SEARCH out-of-fold gain mean {g_s.mean():+.4f}; "
          f"corr(AUDIT gain, SEARCH gain) {np.corrcoef(ga, g_s)[0, 1]:+.3f}")
    print(f"  contexts with AUDIT gain > +0.05: {int((ga > 0.05).sum())}, < -0.05: {int((ga < -0.05).sum())}; "
          f"sum of gains above +0.05: {ga[ga > 0.05].sum():+.3f}, below -0.05: {ga[ga < -0.05].sum():+.3f}, rest: {ga[np.abs(ga) <= 0.05].sum():+.3f}")
    names = ["log n_search", "SXB OOF u", "members", "effective members", "log dispersion/residual (SEARCH)",
             "log dispersion AUDIT/SEARCH", "log |median-mean| / dispersion"]
    for j, nm in enumerate(names):
        x = F[:, j]
        q = np.quantile(x, [1 / 3, 2 / 3])
        parts = [ga[x <= q[0]].mean(), ga[(x > q[0]) & (x <= q[1])].mean(), ga[x > q[1]].mean()]
        print(f"  gain by tercile of {nm:36s} {parts[0]:+.4f} {parts[1]:+.4f} {parts[2]:+.4f}   corr {np.corrcoef(x, ga)[0, 1]:+.3f}")
    # per-context choice of the blend from SEARCH out-of-fold rows
    vals = []
    for c in ctx:
        ms, md = c["Qs"] @ c["w"], wquant(c["Qs"], c["w"], 0.5)
        best = min((0.0, 0.25, 0.5, 0.75, 1.0), key=lambda l: np.mean(((1 - l) * ms + l * md - c["y"]) ** 2))
        vals.append(u_pred((1 - best) * c["mean"] + best * c["med"], c["ya"], c["ba"]))
    report("blend chosen per context on SEARCH", ctx, base, vals)


if __name__ == "__main__":
    main()
