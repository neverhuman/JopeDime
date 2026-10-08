#!/usr/bin/env python3
"""Clean architecture lift charts from per-repeat report.py outputs (figs/r{0,1,2}/wave0_report.json).
dashboard_wave0_arch.png: the 3 PURE wave-0 archs (+ noTabDPT, y-shuffle controls); dashboard_arch6.png: 3 wave-0 archs + 3 wave-1 arms.
One bar per outer repeat with its 95% paired lineage-cluster bootstrap CI vs LightGBM(L) on the same rows. Writes arch_report.json.
usage: archcharts.py --figs DIR"""
import argparse, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
W0 = ["blockformer", "resmlp", "gatedset"]; CTRL = ["resmlp-noTabDPT", "resmlp-yshuf"]
W1 = ["resmlp-anchored", "resmlp-pca32"]
def load(figs):
    R = {}
    for r in range(3):
        p = f"{figs}/r{r}/wave0_report.json"
        if os.path.exists(p): R[r] = json.load(open(p))
    return R
def mask_key(R):
    ks = {k[4:] for rep in R.values() for k in rep["ensembles"] if "-mask" in k}
    return sorted(ks)[:1]
def entry(R, arm):
    out = {}
    for r, rep in R.items():
        e = rep["ensembles"].get(f"ENS:{arm}")
        if e and e.get("n_seeds", 0) >= 1 and "lift_fit_r2" in e:
            out[r] = {k: e[k] for k in ("n_seeds", "oof_rows", "fit_r2", "b10_auc", "lift_fit_r2", "lift_b10_auc", "lgb_fit_r2_same_rows", "lgb_b10_auc_same_rows")}
    return out
def chart(R, arms, path, title):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5)); w = 0.26
    for i, (m, lab) in enumerate((("lift_fit_r2", "fitness R2 lift vs LightGBM(L)"), ("lift_b10_auc", "beat10 AUC lift vs LightGBM(L)"))):
        for r in range(3):
            d = [entry(R, a).get(r, {}).get(m) for a in arms]
            x = np.arange(len(arms)) + (r - 1) * w
            v = np.array([z["delta"] if z else np.nan for z in d]); lo = np.array([z["lo"] if z else np.nan for z in d]); hi = np.array([z["hi"] if z else np.nan for z in d])
            ax[i].bar(x, v, w, yerr=[v - lo, hi - v], capsize=2, label=f"repeat {r}", color=f"C{r}", alpha=0.85)
        ax[i].axhline(0, color="k", lw=1); ax[i].set_xticks(np.arange(len(arms))); ax[i].set_xticklabels(arms, rotation=25, ha="right", fontsize=9); ax[i].set_title(lab); ax[i].legend(fontsize=8)
    lg = R[min(R)]["lgb"]; fig.suptitle(f"{title}  |  LightGBM(L) r{min(R)}: R2 {lg['fit_r2']:.3f}, AUC {lg['b10_auc']:.3f}  |  3-seed ensembles, 95% paired lineage-cluster bootstrap")
    fig.tight_layout(); fig.savefig(path, dpi=95); plt.close(fig)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--figs", required=True); a = ap.parse_args(); R = load(a.figs)
    arms6 = W0 + W1 + mask_key(R)
    chart(R, W0 + CTRL, f"{a.figs}/dashboard_wave0_arch.png", "DL wave 0: PURE end-to-end archs (+controls)")
    chart(R, arms6, f"{a.figs}/dashboard_arch6.png", "DL architectures: wave 0 (PURE) + wave 1")
    rep = {a_: entry(R, a_) for a_ in W0 + CTRL + arms6}; rep["_lgb"] = {r: R[r]["lgb"] for r in R}
    json.dump(rep, open(f"{a.figs}/arch_report.json", "w"), indent=1, default=float)
    for a_ in W0 + CTRL + arms6:
        for r, e in sorted(rep[a_].items()):
            f = lambda d: f"{d['delta']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]"
            print(f"{a_:20s} r{r} seeds {e['n_seeds']} rows {e['oof_rows']} R2 {e['fit_r2']:.3f} {f(e['lift_fit_r2'])}  AUC {e['b10_auc']:.3f} {f(e['lift_b10_auc'])}")
if __name__ == "__main__": main()
