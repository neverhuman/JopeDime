#!/usr/bin/env python3
"""Checkpoint probe: score TA context embeddings against the S1 GP fitness HWM with the fast-sweep ridge machinery.

Same units, target, folds and estimator as gp_hwm_probe/fastsweep.py ridge|X|raw (selection_reached contexts with all 10
types; y = S1 GP fitness; outer folds group_folds(family, 5, seed 1000+r); per-block stdz fit on outer-train rows; kernel
ridge with lambda by inner grouped 3-fold CV on outer-train rows). Repeat r=0 (5 outer folds) per checkpoint.
Arms: TA, TA+F (equal-weight kernel concat), TA[y-shuffled-in-context encoding], SHUF:TA (GP labels permuted inside each
outer-train set) and the references F (foundation) and B (tabpfn35), cached. Paired dR2 vs F and vs B with 95% CI from the
family-grouped bootstrap (1000, seed 0) exactly as analyze_fast.py. Appends one JSON line per checkpoint to --out.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "6")
import argparse, fcntl, json, sys, time, zlib
import numpy as np
from scipy.stats import spearmanr

PROBE_CODE = "/home/ubuntu/jopedime_probe_exp/20261007T1710Z/code"


def main():
    ap = argparse.ArgumentParser()
    for k in ("probe-data", "emb", "emb-yshuf", "meta", "model", "out", "cache"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--repeats", default="0")
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, PROBE_CODE)
    from probe import load, group_folds
    import ta_ridge as R
    t0 = time.time()
    D = load(a.probe_data)
    idx = np.where(D["full"] & ~np.isnan(D["fit"]))[0]
    y0 = D["fit"][idx]; grp = D["grp"][idx]; n = len(y0)
    XF = np.load(os.path.join(a.probe_data, "X_foundation.npy"))[idx]; XB = np.load(os.path.join(a.probe_data, "X_tabpfn35.npy"))[idx]
    E = np.load(a.emb); Es = np.load(a.emb_yshuf)
    assert E.shape[0] == n, (E.shape, n)
    reps = [int(r) for r in a.repeats.split(",")]
    os.makedirs(a.cache, exist_ok=True)

    def run_arm(blocks, r, shuf=False, name=""):
        folds = group_folds(grp, 5, 1000 + r); seed = 1000 + r; pred = np.full(n, np.nan)
        rng = np.random.default_rng(seed + zlib.crc32(name.encode()))
        for f in range(5):
            tr = folds != f; tr_idx, te_idx = np.where(tr)[0], np.where(~tr)[0]
            y = y0.copy()
            if shuf: y[tr_idx] = y[rng.permutation(tr_idx)]
            ig = group_folds(grp[tr_idx], 3, seed + f + 7)
            Ks = {}
            for k, X in blocks.items():
                Z = R.stdz(X, tr); Ks[k] = Z @ Z.T
            p, _, _, _ = R.ridge_k(Ks, {k: 1.0 for k in Ks}, y, tr_idx, te_idx, ig)
            pred[te_idx] = p
        return pred, folds

    def ref(name, X, r):
        fn = os.path.join(a.cache, f"{name}_r{r}.npy")
        if os.path.exists(fn): return np.load(fn)
        p, _ = run_arm({name: X}, r, name=name); np.save(fn, p); return p

    ug, gid = np.unique(grp, return_inverse=True); G = len(ug)
    rng = np.random.default_rng(0); Bw = np.array([np.bincount(rng.integers(0, G, G), None, G) for _ in range(1000)], float)
    res = {}
    for r in reps:
        folds = group_folds(grp, 5, 1000 + r); q = np.empty(n)
        for f in range(5): q[folds == f] = y0[folds != f].mean()
        s0 = np.bincount(gid, (y0 - q) ** 2, G)
        preds = {"F": ref("F", XF, r), "B": ref("B", XB, r)}
        preds["TA"], _ = run_arm({"TA": E}, r, name="TA")
        preds["TA+F"], _ = run_arm({"TA": E, "F": XF}, r, name="TA+F")
        preds["TA_yshuf"], _ = run_arm({"TA": Es}, r, name="TA_yshuf")
        preds["SHUF:TA"], _ = run_arm({"TA": E}, r, shuf=True, name="SHUF:TA")
        sse = {k: np.bincount(gid, (y0 - p) ** 2, G) for k, p in preds.items()}
        for k in preds:
            d = res.setdefault(k, {"R2": [], "spearman": [], "bR2": []})
            d["R2"].append(1 - sse[k].sum() / s0.sum()); d["spearman"].append(float(spearmanr(preds[k], y0).correlation))
            d["bR2"].append(1 - (Bw @ sse[k]) / (Bw @ s0))
    out = {}
    for k, d in res.items():
        o = {"R2": float(np.mean(d["R2"])), "spearman": float(np.mean(d["spearman"]))}
        bk = np.mean(d["bR2"], 0)
        for refn, rk in (("foundation", "F"), ("tabpfn35", "B")):
            if k == rk: continue
            br = np.mean(res[rk]["bR2"], 0); dd = bk - br
            o[f"d_vs_{refn}"] = float(np.mean(d["R2"]) - np.mean(res[rk]["R2"])); o[f"ci_vs_{refn}"] = [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]
        out[k] = o
    meta = json.load(open(a.meta))
    line = dict(model=a.model, **meta, repeats=reps, n_ctx=int(n), n_families=int(G), arms=out, probe_s=time.time() - t0)
    with open(a.out, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX); f.write(json.dumps(line) + "\n")
    print(time.strftime("%H:%M:%S"), json.dumps(line), flush=True)


if __name__ == "__main__":
    main()
