#!/usr/bin/env python3
"""[2026-10-08 ~11:00Z, wave 9 P1 fix: the three DecisionTreeRegressor landmarkers (lm_stump/lm_tree3/lm_tree6) are now seeded
random_state=0 for determinism GOING FORWARD. Every V block / W6 target computed before this edit used the UNSEEDED version (backup
stats_v2.py.bak-unseeded-20261008T1100Z, md5 e117f864...); those results are NOT rewritten. See w9/AUDIT_P1_canary.md.]
stats_v2: label-free (w.r.t. GP labels) context meta-features, computed ONLY from the context's own OUTER-TRAIN rows (ta_common.train_view
decodes only the outer-train byte range; the outer-test block is never parsed). The context's own target column is part of its training data.
Usage: stats_v2.py --cids file --workers N   (per-context JSON in $STD/<cid>.json; skip if present)."""
import argparse, gzip, json, math, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
W = "/home/ubuntu/jopedime_probe_exp/ofm-20261008"
sys.path.insert(0, os.path.join(W, "code")); sys.path.insert(0, os.path.join(W, "code", "ta_code"))
STD = os.environ.get("STD", "/dev/shm/ofm_stats/v2")
PLAN = {}


def r2cv(model_fn, X, y, k=5, seed=0):
    from sklearn.model_selection import KFold
    p = np.zeros(len(y))
    for a, b in KFold(k, shuffle=True, random_state=seed).split(X):
        p[b] = model_fn().fit(X[a], y[a]).predict(X[b])
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def stats(row):
    import ta_common as C
    from scipy import stats as ss
    from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.feature_selection import mutual_info_regression
    gz = open(os.path.join(W, "inputs", row["encoder_jtt1_sha256"] + ".jtt1.gz"), "rb").read(); raw = gzip.decompress(gz)
    X0, y0, h = C.train_view(raw, require_stripped=False)
    if os.environ.get("STATS_YSHUF"):   # control: y-shuffled context (same seed scheme as the ys encodings: fixed per-context permutation)
        y0 = y0[np.random.default_rng(int(row["context_sha256"][:16], 16) ^ 777).permutation(len(y0))]
    s = {}; n0, p0 = X0.shape
    s["log_n"] = math.log(n0); s["log_p"] = math.log(max(p0, 1)); s["log_n_over_p"] = math.log(n0 / max(p0, 1))
    s["missing_rate"] = float(np.isnan(X0).mean()); s["frac_cols_any_missing"] = float(np.isnan(X0).any(0).mean())
    ok = np.isfinite(y0); X, y = X0[ok], y0[ok].astype(np.float64); s["y_nonfinite_rate"] = float(1 - ok.mean())
    rng = np.random.default_rng(int(row["context_sha256"][:16], 16) ^ 20261008)
    if len(y) > 2048: idx = rng.permutation(len(y))[:2048]; X, y = X[idx], y[idx]
    colmu = np.nanmean(X, 0); colmu = np.where(np.isnan(colmu), 0, colmu); X = np.where(np.isnan(X), colmu, X)
    sd = X.std(0); keep = sd > 1e-12; s["frac_const_cols"] = float(1 - keep.mean()); X = X[:, keep]
    if X.shape[1] == 0 or len(y) < 10 or np.ptp(y) <= 1e-12: return {**s, "degenerate": 1.0}
    if X.shape[1] > 128: X = X[:, np.sort(rng.choice(X.shape[1], 128, replace=False))]
    n, p = X.shape
    # target (context's own y) shape
    ys = (y - y.mean()) / y.std()
    s["y_skew"] = float(ss.skew(y)); s["y_kurt"] = float(ss.kurtosis(y)); s["y_frac_unique"] = float(len(np.unique(y)) / n)
    s["y_is_integer"] = float(np.allclose(y, np.round(y))); s["y_outlier_rate"] = float((np.abs(ys) > 3).mean())
    s["y_cv"] = float(y.std() / (np.abs(y.mean()) + 1e-12)) if abs(y.mean()) > 1e-9 else 0.0; s["y_cv"] = float(np.clip(s["y_cv"], -100, 100))
    hist = np.histogram(ys, 20)[0] / n; s["y_entropy"] = float(-(hist[hist > 0] * np.log(hist[hist > 0])).sum())
    s["y_frac_positive"] = float((y > 0).mean())
    # feature types / shape
    integ = np.array([np.allclose(X[:, j], np.round(X[:, j])) for j in range(p)]); card = np.array([len(np.unique(X[:, j])) for j in range(p)])
    cat = integ & (card <= 10); s["frac_categorical"] = float(cat.mean()); s["frac_binary"] = float((card == 2).mean())
    s["mean_log_card_cat"] = float(np.log(card[cat]).mean()) if cat.any() else 0.0; s["frac_integer"] = float(integ.mean())
    Z = (X - X.mean(0)) / X.std(0)
    s["feat_outlier_rate"] = float((np.abs(Z) > 4).mean()); sk = np.abs(ss.skew(X, 0)); s["feat_mean_abs_skew"] = float(np.nanmean(np.clip(sk, 0, 50)))
    s["feat_mean_kurt"] = float(np.nanmean(np.clip(ss.kurtosis(X, 0), -5, 200)))
    # correlation structure + spectrum
    Cm = np.corrcoef(Z[:, :64].T) if p > 1 else np.ones((1, 1)); iu = np.triu_indices(Cm.shape[0], 1)
    s["ff_mean_abs_corr"] = float(np.nanmean(np.abs(Cm[iu]))) if len(iu[0]) else 0.0; s["ff_max_abs_corr"] = float(np.nanmax(np.abs(Cm[iu]))) if len(iu[0]) else 0.0
    ev = np.clip(np.linalg.eigvalsh(np.cov(Z.T) if p > 1 else np.ones((1, 1)))[::-1], 1e-12, None); cv = np.cumsum(ev) / ev.sum()
    s["pca_frac_comp_90"] = float((np.searchsorted(cv, .9) + 1) / p); s["pca_top1_share"] = float(ev[0] / ev.sum())
    s["pca_participation_ratio"] = float(ev.sum() ** 2 / (ev ** 2).sum() / p)
    k = min(len(ev), 20); s["pca_decay_slope"] = float(np.polyfit(np.log(np.arange(1, k + 1)), np.log(ev[:k]), 1)[0]) if k > 2 else 0.0
    # feature-target relations
    r = np.nan_to_num(Z.T @ ys / n); ar = np.sort(np.abs(r))[::-1]
    s["ft_max_abs_corr"] = float(ar[0]); s["ft_mean_abs_corr"] = float(ar.mean()); s["ft_top5_abs_corr"] = float(ar[:5].mean())
    rs = np.nan_to_num([ss.spearmanr(X[:, j], y)[0] for j in range(min(p, 64))]); s["ft_max_abs_spearman"] = float(np.max(np.abs(rs)))
    s["ft_spearman_minus_pearson"] = float(np.max(np.abs(rs)) - ar[0])
    sub = rng.permutation(n)[:800]; mi = mutual_info_regression(Z[sub][:, :64], ys[sub], random_state=0, n_neighbors=5)
    s["mi_max"] = float(mi.max()); s["mi_mean"] = float(mi.mean()); s["mi_top5"] = float(np.sort(mi)[::-1][:5].mean()); s["mi_frac_zero"] = float((mi < 0.01).mean())
    # landmarkers on the context's own train rows (5-fold CV, <=1000 rows)
    sub = rng.permutation(n)[:1000]; Xl, yl = Z[sub], ys[sub]; top = np.argsort(-np.abs(r))[:32]; Xt = Xl[:, top]
    s["lm_ridge"] = r2cv(lambda: RidgeCV(alphas=np.logspace(-2, 3, 12)), Xl, yl)
    s["lm_1nn"] = r2cv(lambda: KNeighborsRegressor(1), Xt, yl); s["lm_knn10"] = r2cv(lambda: KNeighborsRegressor(10, weights="distance"), Xt, yl)
    s["lm_stump"] = r2cv(lambda: DecisionTreeRegressor(max_depth=1, random_state=0), Xl, yl); s["lm_tree3"] = r2cv(lambda: DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=0), Xl, yl)
    s["lm_tree6"] = r2cv(lambda: DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=0), Xl, yl)
    best_nl = max(s["lm_knn10"], s["lm_tree3"], s["lm_tree6"]); s["lm_nonlinearity_gap"] = float(best_nl - s["lm_ridge"]); s["lm_best"] = float(max(best_nl, s["lm_ridge"]))
    # noise / SNR: Delta-test (nearest-neighbour y differences) in top-feature space
    nn = NearestNeighbors(n_neighbors=2).fit(Xt); d, ii = nn.kneighbors(Xt); s["noise_delta_test"] = float(0.5 * np.mean((yl - yl[ii[:, 1]]) ** 2))
    s["snr_delta"] = float(np.log(max(1e-6, 1 - min(s["noise_delta_test"], 1)) / max(min(s["noise_delta_test"], 1), 1e-6)))
    s["nn_dist_ratio"] = float(np.median(d[:, 1]) / (np.median(np.linalg.norm(Xt - Xt.mean(0), axis=1)) + 1e-12))
    s["degenerate"] = 0.0
    return {k: (float(v) if np.isfinite(v) else 0.0) for k, v in s.items()}


def go(c):
    fn = f"{STD}/{c}.json"
    if os.path.exists(fn): return c, "skip"
    try:
        t0 = time.time(); s = stats(PLAN[c]); s["_secs"] = time.time() - t0
        open(fn + ".tmp", "w").write(json.dumps(s)); os.replace(fn + ".tmp", fn); return c, f"ok {s['_secs']:.1f}s"
    except Exception as e: return c, "ERR " + repr(e)[:200]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cids", required=True); ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
    os.makedirs(STD, exist_ok=True); want = set(json.load(open(a.cids)))
    for l in open(os.path.join(W, "plan/plan_full.jsonl")):
        d = json.loads(l)
        if d["context_sha256"] in want: PLAN[d["context_sha256"]] = d
    pending_contexts = [c for c in PLAN if not os.path.exists(f"{STD}/{c}.json")]; print(time.strftime("%H:%M:%S"), "pending", len(pending_contexts), "of", len(want), flush=True)
    import multiprocessing as mp
    nok = 0
    with mp.get_context("fork").Pool(a.workers, maxtasksperchild=200) as pool:
        for i, (c, m) in enumerate(pool.imap_unordered(go, pending_contexts, chunksize=4)):
            if m.startswith("ERR") or i % 500 == 0: print(time.strftime("%H:%M:%S"), i, c[:12], m, flush=True)
    print(time.strftime("%H:%M:%S"), "ALL DONE", flush=True)
