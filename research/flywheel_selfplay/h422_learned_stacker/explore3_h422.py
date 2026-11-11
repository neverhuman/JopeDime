#!/usr/bin/env python3
"""H422 development look 3, FIT-SEL design contexts only: a wider admitted set at no worker cost. The stored variants
of the same run directory (H417 BASE, H420 TGX / TLIB / TBOTH) each admitted their own 64 candidates and evaluated
them on the same SEARCH folds and AUDIT rows; the union of their stored columns is the candidate set a wider
admission rule would have produced. The stored stack builder is run on the union and scored on the AUDIT rows of the
FIT-SEL contexts. In-sample; used only to choose the ablation's rule before the PREREG.

  explore3_h422.py H420_FITSEL_RUN_DIR H417_BASE_RUN_DIR
"""

from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apex_core_meta as AM  # noqa: E402
import meta_h416 as M  # noqa: E402

CLIP = 0.5


def cols(run_dir, tag, cid):
    z = np.load(Path(run_dir) / f"h416_{tag}_npz" / f"{cid}.npz", allow_pickle=False)
    na, m = len(z["ya"]), z["Q"].shape[1]
    mean = np.full(na, float(z["ym"]))
    REF = np.column_stack([z["PA"][:, i] if bool(z["pa_ok"][i]) else mean for i in range(m)])
    BAG = np.column_stack([z["AKF"][:, i] if np.all(np.isfinite(z["AKF"][:, i])) else REF[:, i] for i in range(m)])
    return {"Q": z["Q"], "y": z["y"], "lab": z["lab"], "ya": z["ya"], "ba": float(z["ba"]), "BAG": BAG, "REF": REF,
            "shas": [str(s) for s in z["shas"]]}


def bcar(Q, y, cid):
    n = len(y)
    acc = np.zeros(Q.shape[1])
    for b in range(M.N_BAGS):
        rng = np.random.default_rng(M.stream_seed(M.BAG_SEED, "BCAR", cid, b))
        rows = rng.integers(0, n, size=n)
        acc += M.kcar(Q[rows], y[rows], 100) / 100.0
    return M.prune(acc / M.N_BAGS)


def u(c, w):
    pred = 0.5 * (c["BAG"] @ w) + 0.5 * (c["REF"] @ w)
    return -np.log(max(float(np.sqrt(np.mean((pred - c["ya"]) ** 2))) / c["ba"], 1e-12))


def one(job):
    d20, d17, cid = job
    parts = [cols(d20, "TBOTH", cid), cols(d20, "TLIB", cid), cols(d20, "TGX", cid), cols(d17, "BASE", cid)]
    a = parts[0]
    out = {"same_rows": all(np.array_equal(p["y"], a["y"]) and np.array_equal(p["lab"], a["lab"])
                            and np.array_equal(p["ya"], a["ya"]) for p in parts)}
    out["u_TBOTH"] = u(a, bcar(a["Q"], a["y"], cid))
    seen = list(a["shas"])
    U = {k: [a[k]] for k in ("Q", "BAG", "REF")}
    for p in parts[1:]:
        new = [i for i, s in enumerate(p["shas"]) if s not in seen]
        seen += [p["shas"][i] for i in new]
        for k in U:
            U[k].append(p[k][:, new])
    c = dict(a)
    for k in U:
        c[k] = np.concatenate(U[k], axis=1)
    out["n_union"] = c["Q"].shape[1]
    out["u_union"] = u(c, bcar(c["Q"], c["y"], cid))
    return out


def main() -> None:
    d20, d17 = sys.argv[1:3]
    cids = json.loads((Path(d20) / "config_TBOTH.json").read_text())["cids"]
    rows = AM.rows()
    if {rows[c]["cohort"] for c in cids} != {"FIT-SEL"}:
        raise SystemExit("FIT-SEL contexts only")
    with Pool(46) as pool:
        res = pool.map(one, [(d20, d17, c) for c in cids], chunksize=1)
    d = np.clip(np.array([r["u_union"] - r["u_TBOTH"] for r in res]), -CLIP, CLIP)
    n = [r["n_union"] for r in res]
    print(f"contexts {len(res)} same rows in all variants: {all(r['same_rows'] for r in res)}")
    print(f"union columns median {int(np.median(n))} min {min(n)} max {max(n)}")
    print(f"union vs TBOTH (stored builder on both): mean {d.mean():+.4f} se {d.std(ddof=1) / np.sqrt(len(d)):.4f} "
          f"W/L {int((d > 1e-4).sum())}/{int((d < -1e-4).sum())} median {np.median(d):+.4f}")


if __name__ == "__main__":
    main()
