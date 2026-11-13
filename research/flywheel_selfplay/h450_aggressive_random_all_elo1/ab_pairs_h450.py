#!/usr/bin/env python3
"""H450 pub-r14 per-run dev91 readout (cl_fwriter). The pre-registered eval pools r4_L_A and r4_L_B in every arm, so it
cannot attribute a change to one run. The GPU columns `icl_robridge_a` (L_A + M_init) and `icl_robridge_b`
(L_B + M_init) are the only per-run columns; this script scores each one alone on the dev91 audit rows.

Rule (fixed 2026-11-13 about 16:40Z, before any per-run number was read):
  - per context c and epoch E: u_X(c, E) = mean over the two H418 audit seeds (D1, D2) of
    meta_h416.utility(ref_icl_robridge_X, ya, ba) (higher is better; u = -log(rmse / baseline_rmse));
  - g(c, E) = u_b(c, E) - u_a(c, E) (L_B minus L_A);
  - baseline: mean of g(c, E) over the pure pub-r12 epochs PRE = 7, 9, 11, 12, 13, 15 that are on disk
    (epoch 16 is mixed protocol for L_B and is excluded);
  - effect at a post-switch epoch E >= 17: d(c) = g(c, E) - baseline(c); report mean d over contexts,
    wins/ties/losses, and a two-sided sign-flip p over contexts (200,000 flips, seed 450414);
  - the bag column (bag_icl_robridge_X) is reported the same way as a secondary readout.
y_audit is read only inside meta_h416.utility. CPU only.

  ab_pairs_h450.py --cols-root R4/cols --h418-pulled H418/pulled --epochs 7,9,11,12,13,15,16,18 --out AB.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
if Path("/home/ubuntu/h448-stage/pub-r1/code").is_dir():
    sys.path.append("/home/ubuntu/h448-stage/pub-r1/code")
import meta_h416 as M  # noqa: E402

PRE = (7, 9, 11, 12, 13, 15)
SEED_TAGS = ("D1", "D2")
PAIR = "r4_L_A_r4_L_B"


def _u_cols(npz_p: Path, pulled: Path, cid: str, kind: str):
    z = np.load(npz_p, allow_pickle=False)
    if not bool(z["ok"]):
        return None
    out = []
    for run in ("a", "b"):
        pred = np.asarray(z[f"{kind}_icl_robridge_{run}"], np.float64)
        if not np.all(np.isfinite(pred)):
            return None
        us = []
        for t in SEED_TAGS:
            z16 = np.load(pulled / t / f"h416_{t}_npz" / f"{cid}.npz", allow_pickle=False)
            ya = z16["ya"].astype(np.float64)
            if len(ya) != len(pred):
                return None
            us.append(float(M.utility(pred, ya, float(z16["ba"]))[0]))
        out.append(float(np.mean(us)))
    return out[0], out[1]


def _signflip_p(d: np.ndarray, n: int = 200_000, seed: int = 450414) -> float:
    rng = np.random.default_rng(seed)
    obs = abs(float(np.mean(d)))
    hits = 0
    for i in range(0, n, 10_000):
        s = rng.choice((-1.0, 1.0), size=(min(10_000, n - i), len(d)))
        hits += int(np.sum(np.abs((s * d).mean(axis=1)) >= obs - 1e-15))
    return (hits + 1) / (n + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols-root", required=True)
    ap.add_argument("--h418-pulled", required=True)
    ap.add_argument("--epochs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists (create-once)")
    root, pulled = Path(a.cols_root), Path(a.h418_pulled)
    epochs = sorted({int(x) for x in a.epochs.split(",") if x.strip()})
    cids = None
    per = {}
    for e in epochs:
        d = root / f"g{e:02d}_{PAIR}"
        if not d.is_dir():
            raise SystemExit(f"missing {d}")
        cs = sorted(p.stem for p in d.glob("*.npz"))
        cids = cs if cids is None else sorted(set(cids) & set(cs))
        per[e] = d
    doc = {"schema": "jope.h450.ab_pairs/v1", "pre_epochs": [e for e in PRE if e in per], "epochs": epochs,
           "label": "in-corpus (dev91 in training), canonical_promotion_forbidden", "readouts": {}}
    for kind in ("ref", "bag"):
        U = {}
        for e in epochs:
            for c in cids:
                r = _u_cols(per[e] / f"{c}.npz", pulled, c, kind)
                if r is not None:
                    U[(e, c)] = r
        ok = [c for c in cids if all((e, c) in U for e in epochs)]
        pre = [e for e in PRE if e in per]
        if not pre:
            raise SystemExit("no pre-switch epoch given")
        base = {c: float(np.mean([U[(e, c)][1] - U[(e, c)][0] for e in pre])) for c in ok}
        rows = {}
        for e in epochs:
            ua = np.array([U[(e, c)][0] for c in ok])
            ub = np.array([U[(e, c)][1] for c in ok])
            row = {"mean_u_a": round(float(ua.mean()), 5), "mean_u_b": round(float(ub.mean()), 5),
                   "mean_b_minus_a": round(float((ub - ua).mean()), 5)}
            if e >= 17:
                d = np.array([U[(e, c)][1] - U[(e, c)][0] - base[c] for c in ok])
                row.update({"did_mean": round(float(d.mean()), 5),
                            "did_wtl": [int(np.sum(d > 1e-9)), int(np.sum(np.abs(d) <= 1e-9)), int(np.sum(d < -1e-9))],
                            "did_signflip_p": round(_signflip_p(d), 5)})
            rows[str(e)] = row
        doc["readouts"][kind] = {"contexts": len(ok), "per_epoch": rows}
    doc["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with open(out, "x") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    print(json.dumps(doc, sort_keys=True))


if __name__ == "__main__":
    main()
