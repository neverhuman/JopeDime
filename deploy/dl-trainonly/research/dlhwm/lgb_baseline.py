#!/usr/bin/env python3
"""External baseline ONLY: cross-fitted LV4 LightGBM on the same dlhwm bundle and the same outer folds as the DL arms.
Vendored verbatim from ofm-20261008 wave-5/wave-4 (w5_run.head_lgb + inner_cv, md5 f4f9416c...; w4_lib.group_folds + stdz, md5 150894a4...)
so the baseline is reproducible from this tree. Bag-K = wave-7 recipe. Saves test preds AND the inner cross-fitted OOF on outer-train
(the OOF is consumed only by the wave-1 matched anchored twin; never by a pure arm). Note: the baseline sees block L only;
lift is reported as "DL (L + clean encoders) vs LightGBM (L)".
usage: lgb_baseline.py --data DIR --out DIR --tasks 'fitness:0:0,...' --K 3 --workers 4 --threads 4   (run at nice 19)"""
import argparse, json, math, os, time, zlib
import numpy as np
NT = 2
def group_folds(groups, k, seed):
    rng = np.random.default_rng(seed); ug, cnt = np.unique(groups, return_counts=True)
    order = rng.permutation(len(ug)); order = order[np.argsort(-cnt[order], kind="stable")]; load_ = np.zeros(k); gf = {}
    for gi in order:
        f = int(np.argmin(load_ + rng.random(k) * 1e-6)); gf[ug[gi]] = f; load_[f] += cnt[gi]
    return np.array([gf[g] for g in groups])
def stdz(A, tr, block=True):
    A = np.array(A, dtype=np.float64); mu = np.nanmean(A[tr], 0); mu = np.where(np.isnan(mu), 0, mu); A = np.where(np.isnan(A), mu, A)
    sd = A[tr].std(0); keep = sd > 1e-8; Z = np.zeros_like(A); Z[:, keep] = np.clip((A[:, keep] - mu[keep]) / sd[keep], -10, 10)
    if block: Z *= math.sqrt(tr.sum()) / (np.linalg.norm(Z[tr]) + 1e-12)
    return Z, keep
def head_lgb(X, y, ti, te, ig, seed):
    import lightgbm as lgb
    grid = [dict(num_leaves=nl, min_child_samples=mc, colsample_bytree=cs) for nl in (7, 15, 31) for mc, cs in ((20, 0.3), (50, 0.7))]
    base = dict(learning_rate=0.03, subsample=0.8, subsample_freq=1, reg_lambda=2.0, n_jobs=NT, deterministic=True, force_col_wise=True, verbose=-1, random_state=seed)
    NE = 2500; best = (-9, None, None, None)
    for g in grid:
        curves = np.zeros((3, NE)); preds = {}
        for k in range(3):
            a, b = ti[ig != k], ti[ig == k]
            m = lgb.LGBMRegressor(n_estimators=NE, **base, **g).fit(X[a], y[a], eval_set=[(X[b], y[b])], eval_metric="l2", callbacks=[lgb.early_stopping(300, verbose=False)])
            c = np.array(m.evals_result_["valid_0"]["l2"]); c = np.concatenate([c, np.full(NE - len(c), c[-1])]); curves[k] = c * len(b); preds[k] = m
        mse = curves.sum(0) / len(ti); it = int(np.argmin(mse)) + 1; sc = 1 - mse[it - 1] / y[ti].var()
        if sc > best[0]:
            oof = np.zeros(len(ti))
            for k in range(3): oof[ig == k] = preds[k].predict(X[ti[ig == k]], num_iteration=it)
            best = (sc, g, it, oof)
    sc, g, it, oof = best; itf = int(it * 1.15)
    m = lgb.LGBMRegressor(n_estimators=itf, **base, **g).fit(X[ti], y[ti]); return m.predict(X[te]), sc, {**g, "n_estimators": itf}, oof
def run(t):
    tgt, r, f = t.split(":"); r, f = int(r), int(f); fn = f"{A.out}/{tgt}_r{r}_f{f}.npz"
    if os.path.exists(fn):
        z = np.load(fn); te_exp = np.where(FO[r] == f)[0]
        if not np.array_equal(z["te"], te_exp) or ("bundle_sha256" in z.files and str(z["bundle_sha256"]) != BSHA): raise SystemExit(f"{fn}: existing result is from another bundle/folds")
        return t, "skip"
    t0 = time.time(); y = T[tgt].astype(float); tr = (FO[r] != f) & np.isfinite(y); ti = np.where(tr)[0]; te = np.where(FO[r] == f)[0]
    X = stdz(XL, FO[r] != f)[0]; seed0 = 20261008 + 1000 * r + 10 * f + zlib.crc32((tgt + "lgb").encode()) % 97; P, O, prm = [], [], []
    for k in range(A.K):
        s = seed0 + 7919 * k; ig = group_folds(GID[ti], 3, s); p, sc, pr, oof = head_lgb(X, y, ti, te, ig, s); P.append(p); O.append(oof); prm.append({**pr, "seed": s, "inner": float(sc)})
    np.savez(fn + ".tmp.npz", bundle_sha256=BSHA, te=te, ti=ti, P=np.array(P), OOF=np.array(O), params=json.dumps(prm, default=float), secs=time.time() - t0); os.replace(fn + ".tmp.npz", fn)
    return t, f"ok {time.time() - t0:.0f}s"
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); ap.add_argument("--tasks", required=True)
    ap.add_argument("--K", type=int, default=3); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--threads", type=int, default=4); A = ap.parse_args()
    NT = A.threads; GID = np.load(f"{A.data}/gid.npy"); FO = np.load(f"{A.data}/folds.npy"); T = dict(np.load(f"{A.data}/targets.npz"))
    XL = np.load(f"{A.data}/blocks_L.npy").astype(np.float64); os.makedirs(A.out, exist_ok=True)
    import hashlib; BSHA = hashlib.sha256(open(f"{A.data}/MANIFEST.json", "rb").read()).hexdigest()
    TK = A.tasks.split(","); print(time.strftime("%H:%M:%S"), "tasks", len(TK), "NT", NT, flush=True)
    import multiprocessing as mp
    with mp.get_context("fork").Pool(A.workers, maxtasksperchild=2) as pool:
        for t, m in pool.imap_unordered(run, TK): print(time.strftime("%H:%M:%S"), t, m, flush=True)
    print(time.strftime("%H:%M:%S"), "ALL DONE", flush=True)
