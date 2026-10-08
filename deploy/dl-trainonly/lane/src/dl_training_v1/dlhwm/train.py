#!/usr/bin/env python3
"""DL_wave0 trainer: one arm (architecture) x outer folds, PURE end-to-end (no GBM anchor/distill/input).
Per outer fold: preprocessing fit on outer-train only (LV4 rank-gauss, encoder robust-scale, target robust-standardise);
lineage-grouped 10% inner split for early stopping; loss = beta-NLL(fitness) + 0.5 beta-NLL(gap) + BCE(beat10, ls=0.05)
+ 0.2 in-batch pairwise RankNet(fitness); AdamW + warmup + cosine, grad-clip 1, EMA(0.995) weights evaluated; bf16 autocast.
Writes OOF predictions, per-epoch curves, block-ablation deltas, best-EMA checkpoint per fold (create-only S3 + sha256 readback).
usage: train.py --data DIR --arch blockformer --run-id ID --out DIR [--repeats 0] [--no-tabdpt] [--yshuffle SEED] [--s3]"""
import argparse, io, json, math, os, subprocess, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dlhwm.common import PFX, assert_clean, log, put_create_only, sha256b, host
from dlhwm.models import ARCH
def rankgauss_fit(X):
    qs = np.linspace(0, 1, 201); Q = np.nanquantile(X, qs, axis=0); return Q
def rankgauss_apply(X, Q):
    from math import sqrt
    out = np.empty_like(X, dtype=np.float32); qs = np.linspace(0, 1, 201)
    for j in range(X.shape[1]):
        q = Q[:, j]
        if not np.isfinite(q).all() or q[-1] - q[0] < 1e-12: out[:, j] = 0; continue
        u = np.interp(X[:, j], q, qs); u = np.clip(u, 1e-3, 1 - 1e-3)
        out[:, j] = np.sqrt(2) * erfinv(2 * u - 1)
    out[~np.isfinite(X)] = 0; return out
def erfinv(y): return torch.erfinv(torch.as_tensor(y, dtype=torch.float64)).numpy()
def r2(y, p): m = np.isfinite(y) & np.isfinite(p); y, p = y[m], p[m]; return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
def auc(y, s):
    m = np.isfinite(y) & np.isfinite(s); y, s = y[m], s[m]
    if y.min() == y.max(): return float("nan")
    r = np.argsort(np.argsort(s, kind="stable"), kind="stable") + 1.0
    # average ranks for ties
    o = np.argsort(s, kind="stable"); ss = s[o]; rr = np.arange(1, len(s) + 1, dtype=float); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]: j += 1
        rr[i:j + 1] = (i + j + 2) / 2; i = j + 1
    r = np.empty(len(s)); r[o] = rr; n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def bnll(mu, ls, y, beta=0.5):
    m = torch.isfinite(y); y = torch.where(m, y, torch.zeros_like(y)); var = ls.exp()
    l = 0.5 * (ls + (y - mu) ** 2 / var) * var.detach() ** beta
    return (l * m).sum() / m.sum().clamp(min=1)
def ranknet(mu, y, npairs=2048):
    m = torch.isfinite(y)
    if m.sum() < 4: return mu.sum() * 0
    mu, y = mu[m], y[m]; i = torch.randint(0, len(y), (npairs,), device=y.device); j = torch.randint(0, len(y), (npairs,), device=y.device)
    s = torch.sign(y[i] - y[j]); k = s != 0
    return F.softplus(-s[k] * (mu[i] - mu[j])[k]).mean() if k.any() else mu.sum() * 0
def group_folds(groups, k, seed):
    rng = np.random.default_rng(seed); ug, cnt = np.unique(groups, return_counts=True); order = rng.permutation(len(ug))
    order = order[np.argsort(-cnt[order], kind="stable")]; load_ = np.zeros(k); gf = {}
    for gi in order:
        f = int(np.argmin(load_ + rng.random(k) * 1e-6)); gf[ug[gi]] = f; load_[f] += cnt[gi]
    return np.array([gf[g] for g in groups])
CFG = {"blockformer": dict(lr=5e-4, wd=0.05, bs=256, epochs=200, patience=30, kw={}),
       "resmlp": dict(lr=1e-3, wd=0.05, bs=256, epochs=250, patience=35, kw={}),
       "gatedset": dict(lr=7e-4, wd=0.05, bs=256, epochs=250, patience=35, kw={})}
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--arch", required=True); ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--repeats", default="0"); ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--no-tabdpt", action="store_true"); ap.add_argument("--yshuffle", type=int, default=-1); ap.add_argument("--s3", action="store_true")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--git-sha", default=""); ap.add_argument("--mask", default="")
    ap.add_argument("--epochs", type=int, default=0)
    a = ap.parse_args(); out = f"{a.out}/{a.run_id}"; os.makedirs(out, exist_ok=True); cfg = CFG[a.arch]
    if a.epochs: cfg["epochs"] = a.epochs
    torch.backends.cuda.matmul.allow_tf32 = True; dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.load(open(f"{a.data}/meta.json")); T = np.load(f"{a.data}/targets.npz"); FO = np.load(f"{a.data}/folds.npy"); gid = np.load(f"{a.data}/gid.npy")
    blocks = [b for b in meta["blocks"] if b != "L" and not (a.no_tabdpt and b == "tabdpt13")]; assert_clean(["L"] + blocks)
    L = np.load(f"{a.data}/blocks_L.npy").astype(np.float64); Lc = meta["L_cols"]
    tov = np.array(meta["tabdpt_overlap"]); tcols = [j for j, c in enumerate(Lc) if c.startswith("Z:tabdpt13") or c == "ZL:2"]
    if a.no_tabdpt: keep = [j for j in range(L.shape[1]) if j not in tcols]; L = L[:, keep]; Lc = [Lc[j] for j in keep]
    else: L[np.ix_(tov, tcols)] = np.nan          # TabDPT pretraining-overlap rows: TabDPT features marked missing
    E = {b: np.load(f"{a.data}/blocks_{b}.npy") for b in blocks}
    if "tabdpt13" in E: E["tabdpt13"][tov] = np.nan
    mask = json.load(open(a.mask)) if a.mask else None
    if mask:   # encoder_mask_final.json: {"blocks": {enc: {"weight": w, "keep": [dims]}}}
        for b, spec in mask.get("blocks", {}).items():
            if b in E and spec.get("keep") is not None: E[b] = E[b][:, spec["keep"]]
            if b in E and float(spec.get("weight", 1)) == 0: del E[b]; blocks.remove(b)
    n = L.shape[0]; yfit = T["fitness"].astype(np.float64); ygap = T["gap"].astype(np.float64); yb = T["beat10"].astype(np.float64)
    reps = [int(r) for r in a.repeats.split(",")]; folds = [int(k) for k in a.folds.split(",")]
    git = a.git_sha; summary = {"run_id": a.run_id, "arch": a.arch, "pure": True, "gbm_anchor": False, "distill": False, "gbm_input": False,
        "blocks": ["L"] + blocks, "no_tabdpt": a.no_tabdpt, "yshuffle": a.yshuffle, "git_sha": git, "host": host(), "cfg": cfg, "mask": a.mask or "all-dims-fallback",
        "data_manifest_sha256": sha256b(open(f"{a.data}/MANIFEST.json", "rb").read()), "folds": {}, "s3": []}
    for r in reps:
        oof = np.full((n, 5), np.nan); oofraw = np.full((n, 2), np.nan)
        for k in folds:
            t0 = time.time(); te = np.where(FO[r] == k)[0]; tr_all = np.where(FO[r] != k)[0]
            inner = group_folds(gid[tr_all], 10, 777 + 31 * r + k); va = tr_all[inner == 0]; tr = tr_all[inner != 0]
            yf, yg, ybb = yfit.copy(), ygap.copy(), yb.copy()
            if a.yshuffle >= 0:
                p = np.random.default_rng(a.yshuffle * 1000 + r * 10 + k).permutation(tr_all); yf[tr_all], yg[tr_all], ybb[tr_all] = yfit[p], ygap[p], yb[p]
            QL = rankgauss_fit(L[tr]); XL = rankgauss_apply(L, QL); ML = (~np.isfinite(L)).astype(np.float32)
            Xe, Me = [], []
            for b in blocks:
                A = E[b]; med = np.nanmedian(A[tr], 0); iqr = np.nanquantile(A[tr], .75, 0) - np.nanquantile(A[tr], .25, 0)
                sd = np.nanstd(A[tr], 0); sc = np.where(iqr > 1e-9, iqr / 1.349, np.where(sd > 1e-9, sd, 1.0))
                Z = np.clip((A - med) / sc, -8, 8); miss = ~np.isfinite(A).all(1); Z[~np.isfinite(Z)] = 0; Xe.append(Z.astype(np.float32)); Me.append(miss)
            Me = np.stack(Me, 1).astype(np.float32)
            fm, fs = np.nanmedian(yf[tr]), np.nanstd(yf[tr]); gm, gs = np.nanmedian(yg[tr]), np.nanstd(yg[tr])
            Yf = (yf - fm) / fs; Yg = (yg - gm) / gs
            G = lambda x: torch.as_tensor(x, device=dev)
            gXL, gML, gXe, gMe = G(XL), G(ML), [G(x) for x in Xe], G(Me); gYf, gYg, gYb = G(Yf.astype(np.float32)), G(Yg.astype(np.float32)), G(ybb.astype(np.float32))
            torch.manual_seed(a.seed * 100 + r * 10 + k); np.random.seed(a.seed * 100 + r * 10 + k)
            model = ARCH[a.arch](XL.shape[1], [x.shape[1] for x in Xe], **cfg["kw"]).to(dev)
            ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.995))
            opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"], betas=(0.9, 0.98))
            spe = math.ceil(len(tr) / cfg["bs"]); tot = cfg["epochs"] * spe; warm = max(1, int(0.05 * tot))
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / tot))))
            def fwd(m, idx):
                ii = G(idx); return m(gXL[ii], gML[ii], [x[ii] for x in gXe], gMe[ii])
            def loss_of(o, idx):
                ii = G(idx); yfi, ygi, ybi = gYf[ii], gYg[ii], gYb[ii]; mb = torch.isfinite(ybi)
                lb = F.binary_cross_entropy_with_logits(o["b10"][mb].float(), (ybi[mb] * 0.95 + 0.025)) if mb.any() else o["b10"].sum() * 0
                parts = {"fit": bnll(o["fit_mu"].float(), o["fit_ls"].float(), yfi), "gap": bnll(o["gap_mu"].float(), o["gap_ls"].float(), ygi), "b10": lb, "rank": ranknet(o["fit_mu"].float(), yfi)}
                return parts["fit"] + 0.5 * parts["gap"] + parts["b10"] + 0.2 * parts["rank"], parts
            @torch.no_grad()
            def predict(m, idx, me_override=None):
                m.eval(); outs = []
                for s in range(0, len(idx), 1024):
                    ii = G(idx[s:s + 1024]); me = gMe[ii] if me_override is None else me_override(gMe[ii])
                    xl, ml = gXL[ii], gML[ii]
                    if me_override is not None and getattr(me_override, "dropL", False): ml = torch.ones_like(ml)
                    with torch.autocast(dev, dtype=torch.bfloat16): o = m(xl, ml, [x[ii] for x in gXe], me)
                    outs.append(torch.stack([o["fit_mu"], o["fit_ls"], o["gap_mu"], o["gap_ls"], o["b10"]], 1).float().cpu().numpy())
                return np.concatenate(outs)
            def score(P, idx):
                pf = P[:, 0] * fs + fm; return r2(yf[idx], pf), auc(ybb[idx], P[:, 4])
            hist = []; best = (-1e9, -1, None); bad = 0; step = 0
            for ep in range(cfg["epochs"]):
                model.train(); perm = np.random.permutation(tr); tl = []
                for s in range(0, len(perm), cfg["bs"]):
                    idx = perm[s:s + cfg["bs"]]
                    if len(idx) < 16: continue
                    with torch.autocast(dev, dtype=torch.bfloat16): o = fwd(model, idx)
                    l, parts = loss_of(o, idx); opt.zero_grad(set_to_none=True); l.backward()
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); ema.update_parameters(model); tl.append(l.item()); step += 1
                with torch.no_grad():
                    ema.eval(); vo = []
                    for s in range(0, len(va), 1024):
                        idx = va[s:s + 1024]
                        with torch.autocast(dev, dtype=torch.bfloat16): o = fwd(ema.module, idx)
                        vo.append(loss_of(o, idx)[0].item() * len(idx))
                    vl = sum(vo) / len(va)
                Pv = predict(ema.module, va); vr2, vauc = score(Pv, va); crit = vr2 + (vauc if np.isfinite(vauc) else 0)
                hist.append({"ep": ep, "train_loss": float(np.mean(tl)), "val_loss": vl, "val_r2": vr2, "val_auc": vauc, "lr": sched.get_last_lr()[0], "gn": float(gn)})
                if crit > best[0]: best = (crit, ep, {k_: v.detach().clone() for k_, v in ema.module.state_dict().items()}); bad = 0
                else: bad += 1
                if ep % 10 == 0: log(a.arch, f"r{r}f{k} ep{ep} tl {np.mean(tl):.4f} vl {vl:.4f} vR2 {vr2:.4f} vAUC {vauc:.4f}")
                json.dump(hist, open(f"{out}/hist_r{r}f{k}.json", "w"))
                if bad >= cfg["patience"]: break
            ema.module.load_state_dict(best[2]); m = ema.module
            P = predict(m, te); oof[te] = P; oofraw[te, 0] = P[:, 0] * fs + fm; oofraw[te, 1] = P[:, 2] * gs + gm; tr2, tauc = score(P, te)
            # block ablation at inference (representation usefulness): mark one block missing at a time; L ablated via all-missing mask
            abl = {}
            for bi, b in enumerate(["L"] + blocks):
                if b == "L":
                    f_ = lambda me: me; f_.dropL = True
                else:
                    def f_(me, bi=bi - 1): me = me.clone(); me[:, bi] = 1; return me
                Pa = predict(m, te, f_); ar2, aauc = score(Pa, te); abl[b] = {"d_r2": tr2 - ar2, "d_auc": tauc - aauc}
            gates = None
            if a.arch == "gatedset":
                predict(m, te); gates = dict(zip(["L"] + blocks, m.last_gate.mean(0).cpu().numpy().round(4).tolist()))
            fold_rec = {"r": r, "k": k, "n_tr": len(tr), "n_va": len(va), "n_te": len(te), "best_ep": best[1], "epochs_run": len(hist), "test_r2": tr2, "test_auc": tauc,
                        "secs": time.time() - t0, "ablation": abl, "gates": gates, "target_norm": [fm, fs, gm, gs]}
            buf = io.BytesIO(); torch.save({"state_dict": best[2], "arch": a.arch, "kw": cfg["kw"], "dL": XL.shape[1], "dims": [x.shape[1] for x in Xe], "blocks": blocks,
                        "L_cols": Lc, "QL": QL, "target_norm": [fm, fs, gm, gs], "fold": [r, k], "run_id": a.run_id, "git_sha": git}, buf)
            ck = buf.getvalue(); open(f"{out}/model_r{r}f{k}.pt", "wb").write(ck); fold_rec["ckpt_sha256"] = sha256b(ck)
            if a.s3:
                base = f"{PFX}models/hwm-dlwave/{a.run_id}/r{r}f{k}/"
                rec = put_create_only(base + "model.pt", ck); fold_rec["s3_ckpt"] = rec
                rec2 = put_create_only(base + "fold_manifest.json", json.dumps(fold_rec, sort_keys=True, default=float).encode()); summary["s3"] += [rec, rec2]
            summary["folds"][f"r{r}f{k}"] = fold_rec; log(a.arch, f"DONE r{r}f{k} best_ep {best[1]} test R2 {tr2:.4f} AUC {tauc:.4f} {time.time()-t0:.0f}s")
            np.save(f"{out}/oof_r{r}.npy", oof); np.save(f"{out}/oofraw_r{r}.npy", oofraw); json.dump(summary, open(f"{out}/summary.json", "w"), default=float, indent=1)
        summary[f"oof_r{r}"] = {"fit_r2": r2(yfit, oofraw[:, 0]), "gap_r2": r2(ygap, oofraw[:, 1]), "b10_auc": auc(yb, oof[:, 4])}
        json.dump(summary, open(f"{out}/summary.json", "w"), default=float, indent=1)
    if a.s3:
        summary["s3"].append(put_create_only(f"{PFX}models/hwm-dlwave/{a.run_id}/run_manifest.json", json.dumps(summary, sort_keys=True, default=float).encode()))
        json.dump(summary, open(f"{out}/summary.json", "w"), default=float, indent=1)
    log("ALL DONE", a.run_id)
if __name__ == "__main__": main()
