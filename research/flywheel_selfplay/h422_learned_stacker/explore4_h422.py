#!/usr/bin/env python3
"""H422 development look 4, DESIGN contexts only: aggregation rules that use the members' predictions on the AUDIT
rows (never the AUDIT targets) - range clipping, robust row-wise aggregation, and down-weighting members whose
disagreement with the blend grows from SEARCH to AUDIT rows. In-sample one-parameter changes; not results.

  explore4_h422.py CACHE_DIR [CACHE_DIR ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_h422 import load, norm, report, u_of  # noqa: E402


def u_pred(c, pred):
    return -np.log(max(float(np.sqrt(np.mean((pred - c["ya"]) ** 2))) / float(c["ba"]), 1e-12))


def wmedian(P, w):
    o = np.argsort(P, axis=1)
    cw = np.cumsum(w[o], axis=1)
    k = (cw >= 0.5).argmax(axis=1)
    return P[np.arange(len(P)), o[np.arange(len(P)), k]]


def main() -> None:
    ctx = load(sys.argv[1:])
    base = np.array([u_of(c, c["w_sxb"]) for c in ctx])
    for c in ctx:
        c["A"] = 0.5 * c["BAG"] + 0.5 * c["REF"]
    for d in (0.0, 0.1, 0.25, 0.5):
        vals = []
        for c in ctx:
            lo, hi = c["y"].min(), c["y"].max()
            r = hi - lo
            vals.append(u_pred(c, np.clip(c["A"], lo - d * r, hi + d * r) @ c["w_sxb"]))
        report(f"clip members to y range +{d}", ctx, base, vals)
    for d in (0.0, 0.25):
        vals = []
        for c in ctx:
            lo, hi = c["y"].min(), c["y"].max()
            r = hi - lo
            vals.append(u_pred(c, np.clip(c["A"] @ c["w_sxb"], lo - d * r, hi + d * r)))
        report(f"clip blend to y range +{d}", ctx, base, vals)
    vals = []
    for c in ctx:
        s = c["w_sxb"] > 0
        vals.append(u_pred(c, wmedian(c["A"][:, s], norm(c["w_sxb"][s]))))
    report("weighted median of members", ctx, base, vals)
    for lam in (0.25, 0.5):
        vals = []
        for c in ctx:
            s = c["w_sxb"] > 0
            vals.append(u_pred(c, lam * wmedian(c["A"][:, s], norm(c["w_sxb"][s])) + (1 - lam) * (c["A"] @ c["w_sxb"])))
        report(f"blend {lam} weighted median", ctx, base, vals)
    for kappa in (0.5, 1.0, 2.0, 4.0):
        vals, moved = [], 0
        for c in ctx:
            w = c["w_sxb"]
            s = w > 0
            ds = np.sqrt(((c["Q"][:, s] - (c["Q"] @ w)[:, None]) ** 2).mean(0)) + 1e-12
            da = np.sqrt(((c["A"][:, s] - (c["A"] @ w)[:, None]) ** 2).mean(0)) + 1e-12
            g = np.log(da / ds)
            g = g - np.median(g)
            w2 = np.zeros_like(w)
            w2[s] = w[s] * np.exp(-kappa * np.maximum(g, 0.0))
            moved += int(np.abs(norm(w2) - w).sum() > 0.05)
            vals.append(u_of(c, norm(w2)))
        report(f"shift-score tilt kappa {kappa} ({moved})", ctx, base, vals)


if __name__ == "__main__":
    main()
