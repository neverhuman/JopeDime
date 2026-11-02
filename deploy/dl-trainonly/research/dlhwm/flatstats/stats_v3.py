#!/usr/bin/env python3
"""stats_v3: cross-fitted learnability CONTRASTS and failure fingerprints, from the context's own OUTER-TRAIN rows only (ta_common.train_view;
outer-test bytes never parsed; no GP labels). All scores are in GP-fitness units: s = 0.5*log(MSE_oof / MSE_baseline), baseline = fold-train mean
(cross-fitted), i.e. the log RMSE ratio GP fitness is built from (lower = better). Feature screening (top-k by |Spearman|) is done INSIDE each
CV fold on that fold's training rows only. <=2000 rows (seeded), <=128 columns.
Sources (owner ruling 2: cite the Jailgun tab for each):
  [T2-P01] symbolic-expressibility ladder: constant -> linear -> poly2 -> additive spline -> spline+pairwise interactions on fold-screened top-8;
           scores + contrasts (dA_L, dP_L, dI_A, dI_L)                                                    (Jailgun wave4 tab 2, P01)
  [T2-P02] paired learning curves at 32..1024 rows on a fixed 25% holdout with shared denominator: early/late slope, area (ridge & tree)  (tab 2, P02)
  [T2-P03] signal concentration: linear score with top-1/2/4/8 screened features, concentration ratio, fold-screen Jaccard stability     (tab 2, P03)
  [T2-P08/P09] heteroskedasticity corr(|e|, yhat) and tail robustness log(RMSE/MAE) of best cheap model's OOF residuals                (tab 2, P08/P09)
  [T3-1]  residual-shape fingerprint of the best cheap model (bias, median|e|, p90|e|, tail ratio, corr(e, yhat))  -- CPU PROXY for the
          Kumo-L ICL error-shape fingerprint (ICL per-row predictions were not stored; GPU re-inference pending)          (tab 3, #1)
  [T3-2]  cross-model disagreement / error complementarity among ridge, kNN, tree: residual correlations, Spearman(disagreement, |error|),
          ensemble gain over best single model  -- CPU proxy for the cross-ICL-model disagreement fingerprint            (tab 3, #2)
  [T3-L]  effective rank (participation ratio) of the row Gram matrix                                                       (tab 3, lower-ranked)
Usage: stats_v3.py --cids file --workers N  (env STD3 out dir; STATS_YSHUF=1 -> y-shuffled control, same permutation scheme as stats_v2)."""
import argparse, gzip, json, math, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
W = "/home/ubuntu/jopedime_probe_exp/ofm-20261008"
sys.path.insert(0, os.path.join(W, "code")); sys.path.insert(0, os.path.join(W, "code", "ta_code"))
STD = os.environ.get("STD3", "/dev/shm/ofm_stats/v3"); PLAN = {}
LOGF = lambda mse, base: 0.5 * math.log(max(mse, 1e-12) / max(base, 1e-12))


def screen(X, y, k):
    from scipy.stats import rankdata
    yr = rankdata(y); Xr = np.apply_along_axis(rankdata, 0, X); Xr = (Xr - Xr.mean(0)); yr = yr - yr.mean()
    c = np.abs(Xr.T @ yr) / (np.sqrt((Xr ** 2).sum(0) * (yr ** 2).sum()) + 1e-12); return np.argsort(-c)[:k]


def stats(row):
    import ta_common as C
    from sklearn.model_selection import KFold
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    from scipy.stats import spearmanr
    raw = gzip.decompress(open(os.path.join(W, "inputs", row["encoder_jtt1_sha256"] + ".jtt1.gz"), "rb").read())
    X0, y0, h = C.train_view(raw, require_stripped=False)
    if os.environ.get("STATS_YSHUF"): y0 = y0[np.random.default_rng(int(row["context_sha256"][:16], 16) ^ 777).permutation(len(y0))]
    ok = np.isfinite(y0); X, y = X0[ok], y0[ok].astype(float)
    rng = np.random.default_rng(int(row["context_sha256"][:16], 16) ^ 20261008 ^ 3)
    if len(y) > 2000: idx = rng.permutation(len(y))[:2000]; X, y = X[idx], y[idx]
    mu = np.nanmean(X, 0); mu = np.where(np.isnan(mu), 0, mu); X = np.where(np.isnan(X), mu, X); sd = X.std(0); X = X[:, sd > 1e-12]
    if X.shape[1] == 0 or len(y) < 40 or np.ptp(y) <= 1e-12: return {"degenerate": 1.0}
    if X.shape[1] > 128: X = X[:, np.sort(rng.choice(X.shape[1], 128, replace=False))]
    X = (X - X.mean(0)) / X.std(0); X = np.clip(X, -8, 8); y = (y - y.mean()) / y.std(); n, p = X.shape; s = {}
    kf = list(KFold(5, shuffle=True, random_state=0).split(X))
    def oof(make, feats=None, k=8):
        P = np.zeros(n); base = np.zeros(n); sel_sets = []
        for a, b in kf:
            cols = screen(X[a], y[a], min(k, p)) if feats == "screen" else np.arange(p); sel_sets.append(set(cols.tolist()))
            P[b] = make().fit(X[a][:, cols], y[a]).predict(X[b][:, cols]); base[b] = y[a].mean()
        return P, LOGF(((y - P) ** 2).mean(), ((y - base) ** 2).mean()), sel_sets
    from sklearn.pipeline import make_pipeline
    R = lambda: Ridge(alpha=1.0)
    # [T2-P01] ladder (fold-screened top-8)
    _, sL, sets8 = oof(R, "screen")
    _, sP, _ = oof(lambda: make_pipeline(PolynomialFeatures(2, include_bias=False), Ridge(alpha=3.0)), "screen")
    _, sA, _ = oof(lambda: make_pipeline(SplineTransformer(n_knots=5, degree=3), Ridge(alpha=3.0)), "screen")
    _, sI, _ = oof(lambda: make_pipeline(SplineTransformer(n_knots=4, degree=3), PolynomialFeatures(2, interaction_only=True, include_bias=False), Ridge(alpha=10.0)), "screen")
    s.update(lad_lin=sL, lad_poly2=sP, lad_spline=sA, lad_spline_int=sI, lad_dA_L=sA - sL, lad_dP_L=sP - sL, lad_dI_A=sI - sA, lad_dI_L=sI - sL, lad_best=min(sL, sP, sA, sI))
    # [T2-P03] concentration
    sc = {}
    for k in (1, 2, 4, 8): sc[k] = oof(R, "screen", k)[1]
    s.update(conc_top1=sc[1], conc_top2=sc[2], conc_top4=sc[4], conc_top8=sc[8], conc_ratio=(sc[1] / sc[8]) if abs(sc[8]) > 1e-6 else 0.0)
    jac = [len(a & b) / max(len(a | b), 1) for i, a in enumerate(sets8) for b in sets8[i + 1:]]; s["screen_jaccard"] = float(np.mean(jac))
    # [T3-1], [T3-2], [T2-P08/P09]: three cheap learners on all columns
    Pr, sr, _ = oof(lambda: Ridge(alpha=1.0)); Pk, sk, _ = oof(lambda: KNeighborsRegressor(10, weights="distance"), "screen", 16)
    Pt, st, _ = oof(lambda: DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=0))
    best = int(np.argmin([sr, sk, st])); Pb = (Pr, Pk, Pt)[best]; e = y - Pb; ae = np.abs(e)
    s.update(fp_best_score=min(sr, sk, st), fp_bias=float(e.mean()), fp_med_abs=float(np.median(ae)), fp_p90_abs=float(np.quantile(ae, .9)),
             fp_tail_ratio=float(np.quantile(ae, .9) / (np.median(ae) + 1e-9)), fp_corr_e_yhat=float(np.nan_to_num(np.corrcoef(e, Pb)[0, 1])),
             fp_hetero=float(np.nan_to_num(spearmanr(ae, Pb).correlation)), fp_log_rmse_mae=float(math.log(math.sqrt((e ** 2).mean()) / (ae.mean() + 1e-12) + 1e-12)))
    er, ek, et = y - Pr, y - Pk, y - Pt; cc = lambda a, b: float(np.nan_to_num(np.corrcoef(a, b)[0, 1]))
    dis = (np.abs(Pr - Pk) + np.abs(Pr - Pt) + np.abs(Pk - Pt)) / 3; Pe = (Pr + Pk + Pt) / 3
    s.update(dis_rc_rk=cc(er, ek), dis_rc_rt=cc(er, et), dis_rc_kt=cc(ek, et), dis_mean=float(dis.mean()),
             dis_tracks_err=float(np.nan_to_num(spearmanr(dis, np.abs(y - Pe)).correlation)),
             ens_gain=LOGF(((y - Pe) ** 2).mean(), 1.0) - min(sr, sk, st))
    # [T2-P02] paired learning curves on a fixed 25% holdout, shared denominator
    perm = rng.permutation(n); ho = perm[: max(10, n // 4)]; pool = perm[max(10, n // 4):]; den = ((y[ho] - y[pool].mean()) ** 2).mean()
    sizes = [m for m in (32, 64, 128, 256, 512, 1024) if m <= len(pool)]
    for nm, make in (("ridge", lambda: Ridge(alpha=1.0)), ("tree", lambda: DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=0))):
        sc_ = [LOGF(((y[ho] - make().fit(X[pool[:m]], y[pool[:m]]).predict(X[ho])) ** 2).mean(), den) for m in sizes]; lm = np.log(sizes)
        s[f"lc_{nm}_early_slope"] = float(np.polyfit(lm[:3], sc_[:3], 1)[0]) if len(sizes) >= 3 else 0.0
        s[f"lc_{nm}_late_slope"] = float(np.polyfit(lm[-3:], sc_[-3:], 1)[0]) if len(sizes) >= 3 else 0.0
        s[f"lc_{nm}_area"] = float(np.mean(sc_)) if sc_ else 0.0; s[f"lc_{nm}_last"] = float(sc_[-1]) if sc_ else 0.0
    # [T3-L] row Gram effective rank (participation ratio), <=512 rows
    sub = X[rng.permutation(n)[:512]]; ev = np.clip(np.linalg.eigvalsh(sub @ sub.T / p), 0, None); s["gram_eff_rank_frac"] = float(ev.sum() ** 2 / ((ev ** 2).sum() + 1e-12) / len(sub))
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
