#!/usr/bin/env python3
"""H422 development look at the DESIGN contexts only (P201e + FIT-SEL): one-parameter changes to the stored stack,
scored on the AUDIT rows of those design contexts. Run before the PREREG to choose the stacker's form; its numbers
are in-sample and are not results. Refuses any cache that is not a design cache.

  explore_h422.py CACHE_DIR [CACHE_DIR ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CLIP = 0.5


def load(dirs):
    ctx = []
    for d in dirs:
        info = json.loads(next(Path(d).glob("cache_*.json")).read_text())
        if not set(info["cohorts"]) <= {"DL-Train", "FIT-SEL"}:
            raise SystemExit(f"{d}: not a design cache ({info['cohorts']})")
        for p in sorted(Path(d).glob("*.npz")):
            z = np.load(p, allow_pickle=False)
            c = {k: z[k] for k in z.files}
            c["cid"], c["panel"] = p.stem, info["tag"]
            for k in ("Q", "BAG", "REF", "CNT"):
                c[k] = c[k].astype(np.float64)
            ctx.append(c)
    return ctx


def u_of(c, w, alpha=0.5, s=1.0):
    pred = alpha * (c["BAG"] @ w) + (1 - alpha) * (c["REF"] @ w)
    pred = float(c["ym"]) + s * (pred - float(c["ym"]))
    r = float(np.sqrt(np.mean((pred - c["ya"]) ** 2)))
    return -np.log(max(r / float(c["ba"]), 1e-12))


def oof_u(c):
    r = np.sqrt(((c["Q"] - c["y"][:, None]) ** 2).mean(0))
    return -np.log(np.maximum(r / c["y"].std(), 1e-12))


def norm(w):
    return w / w.sum()


def report(name, ctx, base, vals):
    d = np.clip(np.array(vals) - base, -CLIP, CLIP)
    pan = np.array([c["panel"] for c in ctx])
    parts = " ".join(f"{p} {d[pan == p].mean():+.4f}" for p in sorted(set(pan)))
    print(f"{name:34s} mean {d.mean():+.4f}  se {d.std(ddof=1) / np.sqrt(len(d)):.4f}  W/L {int((d > 1e-4).sum())}/{int((d < -1e-4).sum())}  {parts}")


def main() -> None:
    ctx = load(sys.argv[1:])
    base = np.array([u_of(c, c["w_sxb"]) for c in ctx])
    print(f"contexts {len(ctx)}  SXB mean u {base.mean():.4f} median {np.median(base):.4f}")
    for a in (0.0, 0.25, 0.75, 1.0):
        report(f"bag share {a}", ctx, base, [u_of(c, c["w_sxb"], alpha=a) for c in ctx])
    for s in (0.90, 0.95, 0.98, 1.02):
        report(f"shrink to SEARCH mean s={s}", ctx, base, [u_of(c, c["w_sxb"], s=s) for c in ctx])
    for g in (0.5, 0.75, 1.25, 1.5, 2.0):
        report(f"temperature {g}", ctx, base, [u_of(c, norm(c["w_sxb"] ** g)) for c in ctx])
    report("unpruned bag mean", ctx, base, [u_of(c, norm(c["CNT"].mean(0))) for c in ctx])
    report("selection frequency", ctx, base, [u_of(c, norm((c["CNT"] > 0).mean(0))) for c in ctx])
    report("frequency >= 0.5 only", ctx, base,
           [u_of(c, norm(np.where((c["CNT"] > 0).mean(0) >= 0.5, c["CNT"].mean(0), 0) + 1e-12)) for c in ctx])
    ou = [oof_u(c) for c in ctx]
    report("best single (OOF)", ctx, base, [u_of(c, np.eye(len(o))[int(np.argmax(o))]) for c, o in zip(ctx, ou)])
    for k in (4, 8, 16):
        for lam in (0.25, 0.5, 1.0):
            vals = []
            for c, o in zip(ctx, ou):
                t = np.zeros(len(o))
                t[np.argsort(-o)[:k]] = 1.0 / k
                vals.append(u_of(c, lam * t + (1 - lam) * c["w_sxb"]))
            report(f"blend {lam} uniform top-{k}", ctx, base, vals)
    # candidate-feature tilts: w * exp(beta * z), z = feature standardised within the context
    feats = {}
    for c, o in zip(ctx, ou):
        fold_u = []
        for f in np.unique(c["lab"]):
            m = c["lab"] == f
            r = np.sqrt(((c["Q"][m] - c["y"][m, None]) ** 2).mean(0))
            fold_u.append(-np.log(np.maximum(r / c["y"][m].std(), 1e-12)))
        c["_f"] = {"oof_u": o, "fold_sd": np.std(fold_u, axis=0), "optimism": c["search_u"] - o,
                   "complexity": c["complexity"], "log_wall": np.log1p(c["wall_s"]),
                   "freq": (c["CNT"] > 0).mean(0)}
    for name in ("oof_u", "fold_sd", "optimism", "complexity", "log_wall", "freq"):
        for beta in (-0.5, 0.5):
            vals = []
            for c in ctx:
                x = np.nan_to_num(c["_f"][name], nan=0.0)
                sel = c["w_sxb"] > 0
                z = (x - x[sel].mean()) / (x[sel].std() + 1e-9)
                vals.append(u_of(c, norm(c["w_sxb"] * np.exp(beta * np.clip(z, -3, 3)))))
            report(f"tilt {name} beta {beta:+.1f}", ctx, base, vals)
    heads = sorted({h for c in ctx for h in c["head"]})
    share = {h: np.mean([c["w_sxb"][c["head"] == h].sum() for c in ctx]) for h in heads}
    for h in heads:
        if share[h] < 0.01:
            continue
        for beta in (-0.5, 0.5):
            report(f"head {h} ({share[h]:.2f}) beta {beta:+.1f}", ctx, base,
                   [u_of(c, norm(c["w_sxb"] * np.exp(beta * (c["head"] == h)))) for c in ctx])
    # ceiling: Caruana on the AUDIT rows themselves (uses AUDIT targets; an upper reference only)
    import meta_h416 as M

    vals = []
    for c in ctx:
        A = 0.5 * c["BAG"] + 0.5 * c["REF"]
        vals.append(u_of(c, norm(M.kcar(A, c["ya"], 100))))
    report("ceiling: Caruana on AUDIT", ctx, base, vals)
    vals = [max(u_of(c, np.eye(c["Q"].shape[1])[j]) for j in range(c["Q"].shape[1])) for c in ctx]
    report("ceiling: best single on AUDIT", ctx, base, vals)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
