#!/usr/bin/env python3
"""External baseline ONLY: cross-fitted LV4 LightGBM (wave-5 head_lgb exactly, inner grouped 3-fold grid + early stopping)
on the same dlhwm bundle and the same outer folds as the DL arms. Bag-K (wave-7 recipe). Saves te preds AND the inner
cross-fitted OOF on outer-train (consumed only by the wave-1 matched anchored/distilled arm; never by a wave-0 pure arm).
usage: lgb_baseline.py --data DIR --out DIR --tasks 'fitness:0:0,...' --K 3 --workers 4   (run at nice 19)"""
import argparse, json, os, sys, time, zlib
import numpy as np
W = "/home/ubuntu/jopedime_probe_exp/ofm-20261008"
def run(t):
    tgt, r, f = t.split(":"); r, f = int(r), int(f); fn = f"{A.out}/{tgt}_r{r}_f{f}.npz"
    if os.path.exists(fn): return t, "skip"
    t0 = time.time(); y = T[tgt].astype(float); tr = (FO[r] != f) & np.isfinite(y); ti = np.where(tr)[0]; te = np.where(FO[r] == f)[0]
    X = LIB.stdz(XL, FO[r] != f)[0]; seed0 = 20261008 + 1000 * r + 10 * f + zlib.crc32((tgt + "lgb").encode()) % 97; P, O, prm = [], [], []
    for k in range(A.K):
        s = seed0 + 7919 * k; ig = W5.inner_cv(ti, s); p, sc, pr, oof = W5.head_lgb(X, y, ti, te, ig, s); P.append(p); O.append(oof); prm.append({**pr, "seed": s, "inner": float(sc)})
    np.savez(fn + ".tmp.npz", te=te, ti=ti, P=np.array(P), OOF=np.array(O), params=json.dumps(prm, default=float), secs=time.time() - t0); os.replace(fn + ".tmp.npz", fn)
    return t, f"ok {time.time() - t0:.0f}s"
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); ap.add_argument("--tasks", required=True)
    ap.add_argument("--K", type=int, default=3); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--threads", default="4"); A = ap.parse_args()
    os.environ["W4D"] = A.data; os.environ["W4_NOYS"] = "1"; os.environ["W5_THREADS"] = A.threads; sys.path.insert(0, f"{W}/w5/code"); sys.path.insert(0, f"{W}/w4/code")
    import w5_run as W5
    LIB = W5.L; GID = np.load(f"{A.data}/gid.npy"); FO = np.load(f"{A.data}/folds.npy"); T = dict(np.load(f"{A.data}/targets.npz"))
    LIB.gid = lambda: GID; LIB.folds = lambda: FO; XL = np.load(f"{A.data}/blocks_L.npy").astype(np.float64); os.makedirs(A.out, exist_ok=True)
    TK = A.tasks.split(","); print(time.strftime("%H:%M:%S"), "tasks", len(TK), "NT", W5.NT, flush=True)
    import multiprocessing as mp
    with mp.get_context("fork").Pool(A.workers, maxtasksperchild=2) as pool:
        for t, m in pool.imap_unordered(run, TK): print(time.strftime("%H:%M:%S"), t, m, flush=True)
    print(time.strftime("%H:%M:%S"), "ALL DONE", flush=True)
