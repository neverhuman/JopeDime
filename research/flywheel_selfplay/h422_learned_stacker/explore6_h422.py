#!/usr/bin/env python3
"""H422 development look 6, DESIGN contexts only: a winsorised mean. Each member's AUDIT-row prediction is pulled
back to within k member-scales of the row's weighted median, where the member-scale is that member's own typical
deviation from the blend on the SEARCH out-of-fold rows; then the stored weighted mean is taken. Rows on which no
member deviates abnormally keep the stored prediction exactly. No AUDIT target enters a prediction. In-sample.

  explore6_h422.py CACHE_DIR [CACHE_DIR ...]      (caches of the dev-s3+ cache builder)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sel_h422 import CLIP, DESIGN_COHORTS, _u, load_cache, wquant  # noqa: E402


def report(name, ctx, g):
    d = np.clip(np.array(g), -CLIP, CLIP)
    pan = np.array([c["panel"] for c in ctx])
    w, l = int((d > 1e-4).sum()), int((d < -1e-4).sum())
    print(f"{name:40s} mean {d.mean():+.4f} se {d.std(ddof=1) / np.sqrt(len(d)):.4f} W/T/L {w}/{len(d) - w - l}/{l} "
          f"net {(w - l) / len(d):+.3f}  " + " ".join(f"{p} {d[pan == p].mean():+.4f}" for p in sorted(set(pan))))


def main() -> None:
    ctx = load_cache(sys.argv[1:], DESIGN_COHORTS)
    for c in ctx:
        s = c["w_sxb"] > 0
        ws = c["w_sxb"][s] / c["w_sxb"][s].sum()
        Qs = c["Q"][:, s]
        ms = Qs @ ws
        c["scale"] = np.sqrt(((Qs - ms[:, None]) ** 2).mean(0)) + 1e-300        # per member, SEARCH rows
        c["scale_med"] = np.sqrt(((Qs - wquant(Qs, ws, 0.5)[:, None]) ** 2).mean(0)) + 1e-300
        c["cols"] = np.concatenate([c["BAG"][:, s], c["REF"][:, s]], axis=1)
        c["w2"] = np.concatenate([ws, ws]) / 2
        c["mean"] = c["cols"] @ c["w2"]
        c["med"] = wquant(c["cols"], c["w2"], 0.5)
        c["u0"] = _u(c["mean"], c["ya"], c["ba"])
    for sc in ("scale", "scale_med"):
        for k in (1.0, 2.0, 3.0, 5.0, 8.0):
            g, changed = [], 0
            for c in ctx:
                s2 = np.concatenate([c[sc], c[sc]])
                W = np.clip(c["cols"], c["med"][:, None] - k * s2, c["med"][:, None] + k * s2)
                p = W @ c["w2"]
                changed += int(np.any(p != c["mean"]))
                g.append(_u(p, c["ya"], c["ba"]) - c["u0"])
            report(f"winsorise k={k} ({sc}; changed {changed})", ctx, g)
    for k in (2.0, 3.0):
        for lam in (0.1, 0.25):
            g = []
            for c in ctx:
                s2 = np.concatenate([c["scale"], c["scale"]])
                p = np.clip(c["cols"], c["med"][:, None] - k * s2, c["med"][:, None] + k * s2) @ c["w2"]
                g.append(_u((1 - lam) * p + lam * c["med"], c["ya"], c["ba"]) - c["u0"])
            report(f"winsorise k={k} + median blend {lam}", ctx, g)
    for lam in (0.1, 0.25, 0.5):
        report(f"median blend {lam}", ctx, [_u((1 - lam) * c["mean"] + lam * c["med"], c["ya"], c["ba"]) - c["u0"] for c in ctx])


if __name__ == "__main__":
    main()
