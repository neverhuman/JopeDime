#!/usr/bin/env python3
"""H422 development look 2, DESIGN contexts only: remove-only admission rules on the stored candidate columns. For
each rule the stored stack builder (bootstrap Caruana, same row bootstraps) is run on the kept columns and scored on
the AUDIT rows of the design contexts. In-sample; used only to choose the ablation's rule before the PREREG.

  explore2_h422.py CACHE_DIR [CACHE_DIR ...]
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import meta_h416 as M  # noqa: E402
from explore_h422 import load, oof_u, report, u_of  # noqa: E402

TOPK = (8, 12, 16, 24, 32, 48)
CORR = (0.999, 0.99, 0.95)


def bcar_on(c, keep):
    Q, y = c["Q"][:, keep], c["y"]
    n = len(y)
    acc = np.zeros(len(keep))
    for b in range(M.N_BAGS):
        rng = np.random.default_rng(M.stream_seed(M.BAG_SEED, "BCAR", c["cid"], b))
        rows = rng.integers(0, n, size=n)
        acc += M.kcar(Q[rows], y[rows], 100) / 100.0
    w = np.zeros(c["Q"].shape[1])
    w[keep] = M.prune(acc / M.N_BAGS)
    return w


def dedupe(c, o, thr):
    """Greedy by out-of-fold utility: keep a column unless its residual correlates above thr with a kept one."""
    R = c["Q"] - c["y"][:, None]
    R = (R - R.mean(0)) / (R.std(0) + 1e-12)
    keep = []
    for j in np.argsort(-o):
        if all(abs(float(R[:, j] @ R[:, k]) / len(R)) <= thr for k in keep):
            keep.append(int(j))
    return np.array(sorted(keep))


def one(c):
    o = oof_u(c)
    out = {}
    for k in TOPK:
        keep = np.sort(np.argsort(-o)[:k])
        out[f"top-{k} by OOF u"] = u_of(c, bcar_on(c, keep))
    for t in CORR:
        keep = dedupe(c, o, t)
        out[f"dedupe corr>{t}"] = u_of(c, bcar_on(c, keep))
        out[f"_n dedupe {t}"] = len(keep)
    return out


def main() -> None:
    ctx = load(sys.argv[1:])
    base = np.array([u_of(c, c["w_sxb"]) for c in ctx])
    with Pool(46) as pool:
        res = pool.map(one, ctx, chunksize=2)
    for name in [k for k in res[0] if not k.startswith("_")]:
        report(name, ctx, base, [r[name] for r in res])
    for t in CORR:
        n = [r[f"_n dedupe {t}"] for r in res]
        print(f"dedupe {t}: kept columns median {int(np.median(n))} min {min(n)} max {max(n)}")


if __name__ == "__main__":
    main()
