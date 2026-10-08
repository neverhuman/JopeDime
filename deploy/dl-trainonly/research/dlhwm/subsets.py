#!/usr/bin/env python3
"""Feature-subset wave report: per subset tag (3-seed ensemble of runs whose summary.subset == tag), outer-fold fitness R2 and beat10 AUC with
paired lineage-cluster bootstrap 95% CIs vs S0 (all clean features, same arch/folds/seeds) and vs LightGBM(L) on the same rows.
S5-auto picks K per outer fold by the inner-validation criterion (mean over seeds of best_val_crit) among the S5 top-K arms -> no outer-test labels used.
usage: subsets.py --data DIR --runs DIR --lgb DIR --figs DIR [--repeat 0]"""
import argparse, glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from report import r2, auc, lgb_oof, boot
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--runs", required=True); ap.add_argument("--lgb", required=True)
    ap.add_argument("--figs", required=True); ap.add_argument("--repeat", type=int, default=0); a = ap.parse_args(); r = a.repeat
    T = np.load(f"{a.data}/targets.npz"); gid = np.load(f"{a.data}/gid.npy"); FO = np.load(f"{a.data}/folds.npy"); n = len(gid); yf, yb = T["fitness"], T["beat10"]
    Lf, Lb = lgb_oof(a.lgb, "fitness", r, n), lgb_oof(a.lgb, "beat10", r, n)
    G = {}
    for d in glob.glob(f"{a.runs}/*/summary.json"):
        S = json.load(open(d)); tag = S.get("subset")
        if not tag or not os.path.exists(os.path.dirname(d) + f"/oofraw_r{r}.npy"): continue
        G.setdefault(tag, []).append((os.path.dirname(d), S))
    P = {}
    for tag, runs in G.items():
        F = np.stack([np.load(f"{x}/oofraw_r{r}.npy")[:, 0] for x, _ in runs]); B = np.stack([np.load(f"{x}/oof_r{r}.npy")[:, 4] for x, _ in runs])
        crit = {k: np.mean([S["folds"].get(f"r{r}f{k}", {}).get("best_val_crit", np.nan) for _, S in runs]) for k in range(5)}
        P[tag] = {"F": F.mean(0), "B": B.mean(0), "done": np.isfinite(F).all(0), "n_seeds": len(runs), "crit": crit, "runs": [os.path.basename(x) for x, _ in runs],
                  "optimistic": any(S.get("optimistic") for _, S in runs)}
    s5 = [t for t in P if t.startswith("S5")]
    if len(s5) >= 2:
        F = np.full(n, np.nan); B = np.full(n, np.nan); pick = {}
        for k in range(5):
            te = FO[r] == k; c = {t: P[t]["crit"][k] for t in s5 if P[t]["done"][te].all()}
            if len(c) < len(s5): continue
            t = max(c, key=c.get); pick[k] = t; F[te] = P[t]["F"][te]; B[te] = P[t]["B"][te]
        P["S5-auto"] = {"F": F, "B": B, "done": np.isfinite(F), "n_seeds": min(P[t]["n_seeds"] for t in s5), "pick": pick, "runs": [], "optimistic": False}
    rep = {"repeat": r, "lgb": {"fit_r2": r2(yf, Lf), "b10_auc": auc(yb, Lb)}, "subsets": {}}
    S0 = P.get("S0")
    for tag in sorted(P):
        p = P[tag]; done = p["done"]; e = {"n_seeds": p["n_seeds"], "oof_rows": int(done.sum()), "runs": p["runs"], "optimistic": p["optimistic"]}
        if "pick" in p: e["pick_per_fold"] = p["pick"]
        if done.sum() > 100:
            e["fit_r2"] = r2(yf[done], p["F"][done]); e["b10_auc"] = auc(yb[done], p["B"][done])
            ok = done & np.isfinite(Lf) & np.isfinite(Lb)
            if ok.sum() > 100: e["vs_lgb"] = {"fit_r2": boot(yf[ok], p["F"][ok], Lf[ok], gid[ok], r2, B=1000), "b10_auc": boot(yb[ok], p["B"][ok], Lb[ok], gid[ok], auc, B=1000)}
            if S0 is not None and tag != "S0":
                ok = done & S0["done"]
                if ok.sum() > 100: e["vs_S0"] = {"fit_r2": boot(yf[ok], p["F"][ok], S0["F"][ok], gid[ok], r2, B=1000), "b10_auc": boot(yb[ok], p["B"][ok], S0["B"][ok], gid[ok], auc, B=1000)}
        rep["subsets"][tag] = e
    os.makedirs(a.figs, exist_ok=True); json.dump(rep, open(f"{a.figs}/subsets_report.json", "w"), indent=1, default=float)
    tags = [t for t in sorted(rep["subsets"]) if "fit_r2" in rep["subsets"][t]]
    fig, ax = plt.subplots(1, 4, figsize=(24, 6))
    for i, (cmp_, met, lab) in enumerate((("vs_S0", "fit_r2", "fitness R2: subset - S0 (all clean)"), ("vs_S0", "b10_auc", "beat10 AUC: subset - S0"),
                                           ("vs_lgb", "fit_r2", "fitness R2: subset DL - LightGBM(L)"), ("vs_lgb", "b10_auc", "beat10 AUC: subset DL - LightGBM(L)"))):
        v = [rep["subsets"][t].get(cmp_, {}).get(met, {"delta": np.nan, "lo": np.nan, "hi": np.nan}) for t in tags]; x = np.arange(len(tags))
        ax[i].bar(x, [z["delta"] for z in v], yerr=[[z["delta"] - z["lo"] for z in v], [z["hi"] - z["delta"] for z in v]], capsize=3,
                  color=["C2" if z["lo"] > 0 else "C3" if z["hi"] < 0 else "C7" for z in v])
        ax[i].axhline(0, color="k", lw=1); ax[i].set_xticks(x); ax[i].set_xticklabels([t + (" (opt.)" if rep["subsets"][t]["optimistic"] else "") for t in tags], rotation=45, ha="right", fontsize=8); ax[i].set_title(lab, fontsize=10)
    fig.suptitle(f"DL feature-subset wave (resmlp, 3-seed ensembles, repeat {r}); LightGBM(L) R2 {rep['lgb']['fit_r2']:.3f} AUC {rep['lgb']['b10_auc']:.3f}; 95% lineage-cluster paired bootstrap")
    fig.tight_layout(); fig.savefig(f"{a.figs}/dashboard_subsets.png", dpi=90); plt.close(fig)
    for t in tags:
        e = rep["subsets"][t]; f = lambda d: f"{d['delta']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]" if d else "-"
        print(f"{t:22s} seeds {e['n_seeds']} rows {e['oof_rows']} R2 {e['fit_r2']:.4f} AUC {e['b10_auc']:.4f} vsS0 R2 {f(e.get('vs_S0', {}).get('fit_r2'))} AUC {f(e.get('vs_S0', {}).get('b10_auc'))} vsLGB R2 {f(e.get('vs_lgb', {}).get('fit_r2'))} AUC {f(e.get('vs_lgb', {}).get('b10_auc'))}")
if __name__ == "__main__": main()
