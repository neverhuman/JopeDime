#!/usr/bin/env python3
"""stats_v4 (wave 8b): richer CHEAP-MODEL DISAGREEMENT / complementarity statistics, the family that carried the stats_v3 lift (wave 8a ablation:
dis_* + ens_gain drop-one -0.022 AUC on beat10). Same contract as stats_v3: the context's own OUTER-TRAIN rows only (ta_common.train_view; outer-test
bytes never parsed; no GP labels); <=2000 rows / <=128 cols (same seeds); 5-fold OOF; scores in GP-fitness units 0.5*log(MSE_oof/MSE_base).
STATS_YSHUF=1 -> y-shuffled twin with the SAME permutation as stats_v2/v3. Ideas: Jailgun wave4 tab 3 (#2 disagreement fingerprints), wave8a ablation.
Learners: ridge, ridge-strong, kNN10 (screened 16), 1-NN (screened 8), tree6, stump, poly2 (screened 8), ExtraTrees(40), HistGB(60).
Usage: stats_v4.py --cids file --workers N  (env STD4 out dir)."""
import argparse, gzip, json, math, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
W = "/home/ubuntu/jopedime_probe_exp/ofm-20261008"
sys.path.insert(0, os.path.join(W, "code")); sys.path.insert(0, os.path.join(W, "code", "ta_code")); sys.path.insert(0, os.path.join(W, "stats", "code"))
STD = os.environ.get("STD4", "/dev/shm/ofm_stats/v4"); PLAN = {}
LOGF = lambda mse, base: 0.5 * math.log(max(mse, 1e-12) / max(base, 1e-12))
from stats_v3 import screen


def stats(row):
    import ta_common as C
    from sklearn.model_selection import KFold
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
    from scipy.stats import spearmanr
    raw = gzip.decompress(open(os.path.join(W, "inputs", row["encoder_jtt1_sha256"] + ".jtt1.gz"), "rb").read())
    X0, y0, h = C.train_view(raw, require_stripped=False)
    if os.environ.get("STATS_YSHUF"): y0 = y0[np.random.default_rng(int(row["context_sha256"][:16], 16) ^ 777).permutation(len(y0))]
    ok = np.isfinite(y0); X, y = X0[ok], y0[ok].astype(float)
    rng = np.random.default_rng(int(row["context_sha256"][:16], 16) ^ 20261008 ^ 3)
    if len(y) > 2000: idx = rng.permutation(len(y))[:2000]; X, y = X[idx], y[idx]
    mu = np.nanmean(X, 0); mu = np.where(np.isnan(mu), 0, mu); X = np.where(np.isnan(X), mu, X); sd = X.std(0); X = X[:, sd > 1e-12]
    if X.shape[1] == 0 or len(y) < 40 or np.ptp(y) <= 1e-12: return {"v4_degenerate": 1.0}
    if X.shape[1] > 128: X = X[:, np.sort(rng.choice(X.shape[1], 128, replace=False))]
    X = (X - X.mean(0)) / X.std(0); X = np.clip(X, -8, 8); y = (y - y.mean()) / y.std(); n, p = X.shape
    L = {"ridge": (lambda: Ridge(alpha=1.0), None, 0), "ridge100": (lambda: Ridge(alpha=100.0), None, 0),
         "knn10": (lambda: KNeighborsRegressor(10, weights="distance"), "screen", 16), "nn1": (lambda: KNeighborsRegressor(1), "screen", 8),
         "tree6": (lambda: DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=0), None, 0),
         "stump": (lambda: DecisionTreeRegressor(max_depth=1, random_state=0), None, 0),
         "poly2": (lambda: make_pipeline(PolynomialFeatures(2, include_bias=False), Ridge(alpha=3.0)), "screen", 8),
         "et": (lambda: ExtraTreesRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=1, random_state=0), None, 0),
         "hgb": (lambda: HistGradientBoostingRegressor(max_iter=60, learning_rate=0.1, max_leaf_nodes=15, early_stopping=False, random_state=0), None, 0)}
    def oof(make, feats, k, seed=0):
        P = np.zeros(n); base = np.zeros(n); fs = []
        for a, b in KFold(5, shuffle=True, random_state=seed).split(X):
            cols = screen(X[a], y[a], min(k, p)) if feats == "screen" else np.arange(p)
            P[b] = make().fit(X[a][:, cols], y[a]).predict(X[b][:, cols]); base[b] = y[a].mean()
            fs.append(LOGF(((y[b] - P[b]) ** 2).mean(), ((y[b] - base[b]) ** 2).mean()))
        return P, LOGF(((y - P) ** 2).mean(), ((y - base) ** 2).mean()), float(np.std(fs))
    names = list(L); P = {}; S = {}; FS = {}
    for nm in names: P[nm], S[nm], FS[nm] = oof(*L[nm])
    sc = np.array([S[k] for k in names]); E = np.column_stack([y - P[k] for k in names]); best = names[int(np.argmin(sc))]
    s = {f"v4_s_{k}": S[k] for k in names}
    weak = ["ridge", "knn10", "nn1", "tree6", "stump"]; strong = ["et", "hgb", "poly2"]
    s.update(v4_best=float(sc.min()), v4_worst=float(sc.max()), v4_score_sd=float(sc.std()),
             v4_strong_minus_weak=float(min(S[k] for k in strong) - min(S[k] for k in weak)),
             v4_hgb_minus_ridge=S["hgb"] - S["ridge"], v4_et_minus_tree=S["et"] - S["tree6"], v4_best_fold_sd=FS[best])
    Cm = np.nan_to_num(np.corrcoef(E.T)); iu = np.triu_indices(len(names), 1); ev = np.clip(np.linalg.eigvalsh(Cm), 0, None)
    s.update(v4_errcorr_mean=float(Cm[iu].mean()), v4_errcorr_min=float(Cm[iu].min()), v4_errcorr_eff_n=float(ev.sum() ** 2 / ((ev ** 2).sum() + 1e-12)))
    s["v4_oracle_gap"] = LOGF(float((E ** 2).min(1).mean()), 1.0) - float(sc.min())
    top3 = [names[i] for i in np.argsort(sc)[:3]]; Pe3 = np.mean([P[k] for k in top3], 0); Pall = np.mean([P[k] for k in names], 0)
    s["v4_ens3_gain"] = LOGF(float(((y - Pe3) ** 2).mean()), 1.0) - float(sc.min()); s["v4_ensall_gain"] = LOGF(float(((y - Pall) ** 2).mean()), 1.0) - float(sc.min())
    Pm = np.column_stack([P[k] for k in names]); dis = Pm.std(1)
    s["v4_dis_mean"] = float(dis.mean()); s["v4_dis_tracks_err"] = float(np.nan_to_num(spearmanr(dis, np.abs(y - Pall)).correlation))
    nl = np.minimum(np.abs(E[:, names.index("et")]), np.abs(E[:, names.index("hgb")])); li = np.minimum(np.abs(E[:, names.index("ridge")]), np.abs(E[:, names.index("ridge100")]))
    s["v4_frac_rows_nonlin_wins"] = float((nl < li).mean())
    _, s2, _ = oof(*L[best], seed=1); s["v4_seed_instab"] = abs(s2 - float(sc.min()))
    s["v4_degenerate"] = 0.0
    return {k: (float(v) if np.isfinite(v) else 0.0) for k, v in s.items()}


def go(c):
    fn = f"{STD}/{c}.json"
    if os.path.exists(fn): return c, "skip"
    try:
        t0 = time.time(); s = stats(PLAN[c]); s["_secs"] = time.time() - t0
        open(fn + ".tmp", "w").write(json.dumps(s)); os.replace(fn + ".tmp", fn); return c, "ok %.1fs" % s["_secs"]
    except Exception as e: return c, "ERR " + repr(e)[:200]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cids", required=True); ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
    os.makedirs(STD, exist_ok=True); want = json.load(open(a.cids)); ws = set(want)
    for l in open(os.path.join(W, "plan/plan_full.jsonl")):
        d = json.loads(l)
        if d["context_sha256"] in ws: PLAN[d["context_sha256"]] = d
    pending_contexts = [c for c in want if c in PLAN and not os.path.exists(f"{STD}/{c}.json")]; print(time.strftime("%H:%M:%S"), "pending", len(pending_contexts), "of", len(want), flush=True)
    import multiprocessing as mp
    with mp.get_context("fork").Pool(a.workers, maxtasksperchild=200) as pool:
        for i, (c, m) in enumerate(pool.imap(go, pending_contexts, chunksize=2)):
            if m.startswith("ERR") or i % 250 == 0: print(time.strftime("%H:%M:%S"), i, c[:12], m, flush=True)
    print(time.strftime("%H:%M:%S"), "ALL DONE", flush=True)
