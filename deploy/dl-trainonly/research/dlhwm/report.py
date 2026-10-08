#!/usr/bin/env python3
"""DL-wave report: per-fold train/val loss curves per arm, dashboard PNG, LIFT vs cross-fitted LV4 LightGBM on the same outer folds
(fitness R2 and beat10 AUC deltas, lineage-cluster paired bootstrap 95% CI), block-ablation (representation usefulness) and gates.
usage: report.py --data DIR --runs DIR --arms run1,run2,... --lgb DIR --figs DIR [--repeat 0]"""
import argparse, glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def r2(y, p): m = np.isfinite(y) & np.isfinite(p); y, p = y[m], p[m]; return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
def auc(y, s):
    m = np.isfinite(y) & np.isfinite(s); y, s = y[m], s[m]
    if len(y) == 0 or y.min() == y.max(): return float("nan")
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def lgb_oof(lgbdir, tgt, r, n):
    P = np.full(n, np.nan)
    for f in range(5):
        fn = f"{lgbdir}/{tgt}_r{r}_f{f}.npz"
        if os.path.exists(fn): d = np.load(fn); P[d["te"]] = d["P"].mean(0)
    return P
def boot(y, a, b, g, fn, B=2000, seed=0):
    m = np.isfinite(y) & np.isfinite(a) & np.isfinite(b); y, a, b, g = y[m], a[m], b[m], g[m]
    ug, inv = np.unique(g, return_inverse=True); idx = [np.where(inv == i)[0] for i in range(len(ug))]; rng = np.random.default_rng(seed); D = []
    for _ in range(B):
        s = np.concatenate([idx[i] for i in rng.integers(0, len(ug), len(ug))]); D.append(fn(y[s], a[s]) - fn(y[s], b[s]))
    D = np.array(D); return {"delta": fn(y, a) - fn(y, b), "lo": float(np.nanpercentile(D, 2.5)), "hi": float(np.nanpercentile(D, 97.5)), "n": int(m.sum())}
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--runs", required=True); ap.add_argument("--arms", required=True)
    ap.add_argument("--lgb", required=True); ap.add_argument("--figs", required=True); ap.add_argument("--repeat", type=int, default=0); ap.add_argument("--tag", default="wave0")
    a = ap.parse_args(); os.makedirs(a.figs, exist_ok=True); r = a.repeat
    T = np.load(f"{a.data}/targets.npz"); gid = np.load(f"{a.data}/gid.npy"); FO = np.load(f"{a.data}/folds.npy"); n = len(gid)
    yf, yb = T["fitness"], T["beat10"]; Lf, Lb = lgb_oof(a.lgb, "fitness", r, n), lgb_oof(a.lgb, "beat10", r, n)
    rep = {"repeat": r, "lgb": {"fit_r2": r2(yf, Lf), "b10_auc": auc(yb, Lb), "folds_done": [os.path.exists(f"{a.lgb}/fitness_r{r}_f{f}.npz") and os.path.exists(f"{a.lgb}/beat10_r{r}_f{f}.npz") for f in range(5)]}, "arms": {}}
    arms = [x for x in a.arms.split(",") if x]
    for run in arms:
        d = f"{a.runs}/{run}"
        if not os.path.exists(f"{d}/summary.json"): continue
        S = json.load(open(f"{d}/summary.json")); hs = sorted(glob.glob(f"{d}/hist_r{r}f*.json"))
        # per-fold loss curves
        if hs:
            fig, ax = plt.subplots(2, len(hs), figsize=(4 * len(hs), 6), squeeze=False)
            for j, h in enumerate(hs):
                H = json.load(open(h)); e = [x["ep"] for x in H]; k = h.split("f")[-1].split(".")[0]
                ax[0, j].plot(e, [x["train_loss"] for x in H], label="train"); ax[0, j].plot(e, [x["val_loss"] for x in H], label="val (inner, EMA)")
                bf = S["folds"].get(f"r{r}f{k}", {}).get("best_ep")
                if bf is not None: ax[0, j].axvline(bf, color="k", ls=":", lw=1)
                ax[0, j].set_title(f"{S['arch']} r{r} f{k}"); ax[0, j].legend(fontsize=7); ax[0, j].set_xlabel("epoch")
                ax[1, j].plot(e, [x["val_r2"] for x in H], label="val fitness R2"); ax[1, j].plot(e, [x["val_auc"] for x in H], label="val beat10 AUC"); ax[1, j].legend(fontsize=7); ax[1, j].set_ylim(0, 1)
            fig.suptitle(f"{run}  (PURE DL: no GBM anchor/distill/input)" if S.get("pure") else run); fig.tight_layout(); fig.savefig(f"{a.figs}/loss_{run}.png", dpi=90); plt.close(fig)
        rec = {"arch": S["arch"], "pure": S.get("pure"), "optimistic": S.get("optimistic", False), "mask_sha256": S.get("mask_sha256"), "gbm_anchor": S.get("gbm_anchor", False), "no_tabdpt": S.get("no_tabdpt"), "yshuffle": S.get("yshuffle"), "folds_done": sorted(S["folds"]),
               "fold_test": {k: [v["test_r2"], v["test_auc"]] for k, v in S["folds"].items()}}
        if os.path.exists(f"{d}/oofraw_r{r}.npy"):
            Pf = np.load(f"{d}/oofraw_r{r}.npy")[:, 0]; Pb = np.load(f"{d}/oof_r{r}.npy")[:, 4]; done = np.isfinite(Pf)
            rec["oof_rows"] = int(done.sum()); rec["fit_r2"] = r2(yf[done], Pf[done]); rec["b10_auc"] = auc(yb[done], Pb[done])
            rec["lgb_same_rows"] = {"fit_r2": r2(yf[done], Lf[done]) if np.isfinite(Lf[done]).sum() > 100 else None, "b10_auc": auc(yb[done], Lb[done]) if np.isfinite(Lb[done]).sum() > 100 else None}
            ok = done & np.isfinite(Lf)
            if ok.sum() > 100:
                rec["lift_fit_r2"] = boot(yf[ok], Pf[ok], Lf[ok], gid[ok], r2)
            ok2 = done & np.isfinite(Lb)
            if ok2.sum() > 100: rec["lift_b10_auc"] = boot(yb[ok2], Pb[ok2], Lb[ok2], gid[ok2], auc, B=1000)
        ab = [v["ablation"] for v in S["folds"].values() if v.get("ablation")]
        if ab: rec["block_ablation"] = {b: {"d_r2": float(np.mean([x[b]["d_r2"] for x in ab])), "d_auc": float(np.nanmean([x[b]["d_auc"] for x in ab]))} for b in ab[0]}
        gt = [v["gates"] for v in S["folds"].values() if v.get("gates")]
        if gt: rec["mean_gates"] = {b: float(np.mean([x[b] for x in gt])) for b in gt[0]}
        rep["arms"][run] = rec
    # seed ensembles: average members sharing (arch, pure, no_tabdpt, yshuffle, mask) on rows every member has scored
    grp = {}
    for run in rep["arms"]:
        S = json.load(open(f"{a.runs}/{run}/summary.json"))
        if not os.path.exists(f"{a.runs}/{run}/oofraw_r{r}.npy"): continue
        key = f"{(S.get('subset') + ':') if S.get('subset') else ''}{S['arch']}{'-anchored' if S.get('gbm_anchor') else ''}{'-noTabDPT' if S.get('no_tabdpt') else ''}{'-yshuf' if S.get('yshuffle', -1) >= 0 else ''}{('-mask' + S['mask_sha256'][:8] + '-optimistic') if S.get('mask_sha256') else ''}{'-pca' + str(S['pca']) if S.get('pca') else ''}"
        grp.setdefault(key, []).append(run)
    rep["ensembles"] = {}
    for key, runs in grp.items():
        F = np.stack([np.load(f"{a.runs}/{x}/oofraw_r{r}.npy")[:, 0] for x in runs]); Bb = np.stack([np.load(f"{a.runs}/{x}/oof_r{r}.npy")[:, 4] for x in runs])
        done = np.isfinite(F).all(0); Pf, Pb = F.mean(0), Bb.mean(0); e = {"members": runs, "n_seeds": len(runs), "oof_rows": int(done.sum())}
        if done.sum() > 100:
            e["fit_r2"] = r2(yf[done], Pf[done]); e["b10_auc"] = auc(yb[done], Pb[done]); ok = done & np.isfinite(Lf); ok2 = done & np.isfinite(Lb)
            if ok.sum() > 100: e["lgb_fit_r2_same_rows"] = r2(yf[ok], Lf[ok]); e["lift_fit_r2"] = boot(yf[ok], Pf[ok], Lf[ok], gid[ok], r2)
            if ok2.sum() > 100: e["lgb_b10_auc_same_rows"] = auc(yb[ok2], Lb[ok2]); e["lift_b10_auc"] = boot(yb[ok2], Pb[ok2], Lb[ok2], gid[ok2], auc, B=1000)
        rep["ensembles"][f"ENS:{key}"] = e
    json.dump(rep, open(f"{a.figs}/{a.tag}_report.json", "w"), indent=1, default=float)
    # dashboard: lift bars + ablation heatmap
    ALL = {**rep["arms"], **rep["ensembles"]}; A = [k for k, v in ALL.items() if "lift_fit_r2" in v]
    nan = {"delta": np.nan, "lo": np.nan, "hi": np.nan}
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
    for i, (key, lab) in enumerate((("lift_fit_r2", "fitness R2 lift: DL(L+enc) vs LightGBM(L)"), ("lift_b10_auc", "beat10 AUC lift: DL(L+enc) vs LightGBM(L)"))):
        v = [ALL[k].get(key, nan) for k in A]; x = np.arange(len(A))
        ax[i].bar(x, [z["delta"] for z in v], yerr=[[z["delta"] - z["lo"] for z in v], [z["hi"] - z["delta"] for z in v]], capsize=4, color=["C2" if z["lo"] > 0 else "C3" if z["hi"] < 0 else "C7" for z in v])
        ax[i].axhline(0, color="k", lw=1); ax[i].set_xticks(x); ax[i].set_xticklabels(A, rotation=30, ha="right", fontsize=7); ax[i].set_title(lab + " (95% lineage-cluster bootstrap)")
    bl = [k for k in rep["arms"] if "block_ablation" in rep["arms"][k]]
    if bl:
        blocks = sorted({b for k in bl for b in rep["arms"][k]["block_ablation"]}); M = np.array([[rep["arms"][k]["block_ablation"].get(b, {}).get("d_r2", np.nan) for b in blocks] for k in bl])
        im = ax[2].imshow(M, cmap="RdBu_r", vmin=-np.nanmax(np.abs(M)), vmax=np.nanmax(np.abs(M)), aspect="auto"); ax[2].set_yticks(range(len(bl))); ax[2].set_yticklabels(bl, fontsize=7)
        ax[2].set_xticks(range(len(blocks))); ax[2].set_xticklabels(blocks, rotation=45, ha="right", fontsize=7); plt.colorbar(im, ax=ax[2]); ax[2].set_title("block ablation at inference: fitness R2 drop")
    fig.suptitle(f"DL {a.tag} r{r}: LV4 LightGBM fitness R2 {rep['lgb']['fit_r2']:.3f}, beat10 AUC {rep['lgb']['b10_auc']:.3f}"); fig.tight_layout(); fig.savefig(f"{a.figs}/dashboard_{a.tag}.png", dpi=90); plt.close(fig)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("fit_r2", "b10_auc", "lift_fit_r2", "lift_b10_auc", "oof_rows", "lgb_same_rows", "n_seeds")} for k, v in ALL.items()}, indent=1, default=float)); print("LGB", rep["lgb"])
if __name__ == "__main__": main()
