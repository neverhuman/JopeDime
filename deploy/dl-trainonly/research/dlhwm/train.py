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
def b10rank(logit, y, npairs=2048):
    """pairwise logistic AUC surrogate on the beat10 logit: positives should outrank negatives (sweep knob w_b10rank, default off)."""
    m = torch.isfinite(y)
    if m.sum() < 4: return logit.sum() * 0
    logit, y = logit[m], y[m]; pos = torch.where(y > 0.5)[0]; neg = torch.where(y <= 0.5)[0]
    if len(pos) == 0 or len(neg) == 0: return logit.sum() * 0
    i = pos[torch.randint(0, len(pos), (npairs,), device=y.device)]; j = neg[torch.randint(0, len(neg), (npairs,), device=y.device)]
    return F.softplus(-(logit[i] - logit[j])).mean()
def huber_mean(mu, y, delta):
    m = torch.isfinite(y); y = torch.where(m, y, torch.zeros_like(y))
    return (F.smooth_l1_loss(mu, y, reduction="none", beta=delta) * m).sum() / m.sum().clamp(min=1)
class Lion(torch.optim.Optimizer):
    """Lion (Chen et al. 2023): sign of an interpolated momentum, decoupled weight decay."""
    def __init__(s, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))
    @torch.no_grad()
    def step(s):
        for g in s.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None: continue
                st = s.state[p]
                if not st: st["m"] = torch.zeros_like(p)
                m = st["m"]; p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(torch.sign(m * b1 + p.grad * (1 - b1)), alpha=-g["lr"]); m.mul_(b2).add_(p.grad, alpha=1 - b2)
GATES = []
MASK_KEYS = {"block_weight", "weight", "mode", "kept", "keep", "n_pcs", "note", "score", "dims_total", "raw_dims", "kept_dims", "pca_fit"}
def parse_mask(path, blocks, force_mode="", unlisted_policy="keep"):
    """encoder_mask_final.json -> {enc: {weight, mode: dims|pcs, kept, n_pcs}}. Accepts the DL_FINAL_DESIGN schema
    {"encoders": {enc: {"block_weight", "mode": "dims"|"pcs", "kept": [...], "n_pcs": 32}}} and the short form {"blocks": {enc: {"weight", "keep"}}}.
    Fails closed on unknown encoders, keys or modes. PCs are fit per fold on outer-train rows only (exact SVD, variance order)."""
    J = json.load(open(path))
    if "encoders" in J and "blocks" in J: raise SystemExit(f"mask {path}: both 'encoders' and 'blocks' present (ambiguous)")
    top = J.get("encoders", J.get("blocks"))
    if not isinstance(top, dict) or not top: raise SystemExit(f"mask {path}: no 'encoders'/'blocks' map")
    out = {}
    for enc, sp in top.items():
        if enc not in blocks and enc != "tabdpt13": raise SystemExit(f"mask {path}: unknown/refused encoder {enc!r}")
        bad = set(sp) - MASK_KEYS
        if bad: raise SystemExit(f"mask {path}: unknown keys {bad} for {enc}")
        mode = force_mode or sp.get("mode", "dims"); kept = sp.get("kept", sp.get("keep")) if (force_mode or sp.get("mode", "dims")) == sp.get("mode", "dims") else None
        if mode == "dims" and kept is None: kept = sp.get("raw_dims", sp.get("kept_dims"))
        w = float(sp.get("block_weight", sp.get("weight", 1.0)))
        if mode not in ("dims", "pcs"): raise SystemExit(f"mask {path}: mode {mode!r} for {enc}")
        if w < 0 or (w > 0 and (not isinstance(kept, list) or not kept or not all(isinstance(i, int) and i >= 0 for i in kept))): raise SystemExit(f"mask {path}: bad kept/weight for {enc}")
        npc = int(sp.get("n_pcs", 32)) if mode == "pcs" else None
        if mode == "pcs" and w > 0 and max(kept) >= npc: raise SystemExit(f"mask {path}: PC index >= n_pcs for {enc}")
        if enc in blocks: out[enc] = {"weight": w, "mode": mode, "kept": sorted(set(kept or [])), "n_pcs": npc}
    unlisted = [b for b in blocks if b not in top]   # explicit policy (--mask-unlisted), recorded in summary.mask_spec._unlisted
    if unlisted_policy == "drop":
        for b in unlisted: out[b] = {"weight": 0.0, "mode": "dims", "kept": [], "n_pcs": None}
    elif unlisted_policy != "keep": raise SystemExit(f"--mask-unlisted {unlisted_policy!r}")
    out["_unlisted"] = {"policy": unlisted_policy, "encoders": unlisted}
    return out
def pca_exact_gpu(Z, rows, k, dev):
    """exact SVD PCA fit on `rows` (outer-train only), components in variance order with a deterministic sign (largest |loading| > 0), scores standardised on rows."""
    X = torch.as_tensor(Z, device=dev, dtype=torch.float32); R = torch.as_tensor(rows, device=dev); mu = X[R].mean(0)
    _, _, Vh = torch.linalg.svd((X[R] - mu).double(), full_matrices=False); V = Vh[:k].T.float()
    sgn = torch.sign(V.gather(0, V.abs().argmax(0, keepdim=True))).squeeze(0); V = V * sgn; P = (X - mu) @ V
    P = P / P[R].std(0).clamp(min=1e-6); return P.clamp(-8, 8).cpu().numpy().astype(np.float32)
def robust_gpu(A, tr, dev, chunk=1024):
    """per-column median/IQR scaling fit on rows tr (GPU, chunked: torch.nanquantile has an input-size cap). Returns float32 Z (NaN->0, clip 8), row-missing flags."""
    miss = ~np.isfinite(A).all(1); X = torch.as_tensor(A, device=dev, dtype=torch.float32); Xt = X[torch.as_tensor(tr, device=dev)]; out = torch.empty_like(X)
    for s in range(0, X.shape[1], chunk):
        c = Xt[:, s:s + chunk]; q = torch.nanquantile(c, torch.tensor([.25, .5, .75], device=dev), dim=0)
        m_ = torch.nanmean(c, 0); sd = torch.sqrt(torch.nanmean((c - m_) ** 2, 0)); iqr = (q[2] - q[0]) / 1.349
        sc = torch.where(iqr > 1e-9, iqr, torch.where(sd > 1e-9, sd, torch.ones_like(sd))); out[:, s:s + chunk] = ((X[:, s:s + chunk] - q[1]) / sc).clamp(-8, 8)
    out = torch.nan_to_num(out, nan=0.0); return out.cpu().numpy(), miss
def pca_gpu(Z, tr, k, dev):
    X = torch.as_tensor(Z, device=dev); mu = X[torch.as_tensor(tr, device=dev)].mean(0); k = min(k, len(tr) - 1, X.shape[1])
    U, S, V = torch.pca_lowrank(X[torch.as_tensor(tr, device=dev)] - mu, q=k, center=False, niter=4); P = (X - mu) @ V
    P = P / P[torch.as_tensor(tr, device=dev)].std(0).clamp(min=1e-6); return P.clamp(-8, 8).cpu().numpy().astype(np.float32)
def shap_rank(X, yf, yb, seed):
    """mean |TreeSHAP| of a small LightGBM per target (fitness, beat10) on the given (inner-train) rows; shares averaged over the two targets."""
    import lightgbm as lgb
    out = np.zeros(X.shape[1])
    for y in (yf, yb):
        m = np.isfinite(y)
        g = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20, colsample_bytree=0.5, subsample=0.8, subsample_freq=1,
                              n_jobs=3, random_state=seed, verbose=-1).fit(X[m], y[m])
        c = np.abs(g.booster_.predict(X[m], pred_contrib=True)[:, :-1]).mean(0); out += c / max(c.sum(), 1e-12)
    return out / 2
def load_anchor(d, r, k, n, FO, Y, bundle_sha):
    """LightGBM anchor for outer fold (r,k). Fails closed unless the npz rows match THIS bundle's folds exactly (te = outer-test rows,
    ti = outer-train rows with a finite target) and, when present, the npz bundle sha equals this bundle's MANIFEST sha. Rows without an
    anchor (non-finite target on outer-train) get the outer-train mean prediction. Returns (AN [n,3], provenance)."""
    AN = np.full((n, 3), np.nan); prov = {}
    te_exp = np.where(FO[r] == k)[0]
    for j, t in enumerate(("fitness", "gap", "beat10")):
        p = f"{d}/{t}_r{r}_f{k}.npz"; z = np.load(p); ti_exp = np.where((FO[r] != k) & np.isfinite(Y[t]))[0]
        if not (np.array_equal(z["te"], te_exp) and np.array_equal(z["ti"], ti_exp)): raise SystemExit(f"anchor {p}: te/ti do not match this bundle's folds (r{r} f{k})")
        if "bundle_sha256" in z.files and str(z["bundle_sha256"]) != bundle_sha: raise SystemExit(f"anchor {p}: built on another bundle")
        AN[z["te"], j] = z["P"].mean(0); AN[z["ti"], j] = z["OOF"].mean(0)
        miss = ~np.isfinite(AN[:, j]); AN[miss, j] = np.nanmean(AN[z["ti"], j])
        prov[t] = {"file": os.path.abspath(p), "sha256": sha256b(open(p, "rb").read()), "K": int(z["P"].shape[0]), "bundle_sha256": str(z["bundle_sha256"]) if "bundle_sha256" in z.files else None, "filled": int(miss.sum())}
    return AN, prov
class MultiOpt:
    """one or more optimizers stepped together (arch twohead: Lion trunk + AdamW beat10 head); a single optimizer behaves exactly as before."""
    def __init__(s, opts): s.opts = opts
    def zero_grad(s, set_to_none=True):
        for o in s.opts: o.zero_grad(set_to_none=set_to_none)
    def step(s):
        for o in s.opts: o.step()
class MultiSched:
    def __init__(s, scheds): s.scheds = scheds
    def step(s):
        for sc in s.scheds: sc.step()
    def get_last_lr(s): return s.scheds[0].get_last_lr()
def group_folds(groups, k, seed):
    rng = np.random.default_rng(seed); ug, cnt = np.unique(groups, return_counts=True); order = rng.permutation(len(ug))
    order = order[np.argsort(-cnt[order], kind="stable")]; load_ = np.zeros(k); gf = {}
    for gi in order:
        f = int(np.argmin(load_ + rng.random(k) * 1e-6)); gf[ug[gi]] = f; load_[f] += cnt[gi]
    return np.array([gf[g] for g in groups])
CFG = {"blockformer": dict(lr=5e-4, wd=0.05, bs=256, epochs=200, patience=30, kw={}),
       "resmlp": dict(lr=1e-3, wd=0.05, bs=256, epochs=250, patience=35, kw={}),
       "gatedset": dict(lr=7e-4, wd=0.05, bs=256, epochs=250, patience=35, kw={}),
       "twohead": dict(lr=1e-4, wd=0.25, bs=256, epochs=400, patience=60, kw={})}
# sweep knobs (dlhwm/d-sweep); defaults reproduce the frozen d796edc trainer exactly. All selection (early stopping, SWA accept) uses inner-validation rows only.
SWEEP_DEFAULTS = dict(loss="bnll", huber_delta=1.0, w_gap=0.5, w_b10=1.0, w_rank=0.2, w_b10rank=0.0, opt="adamw", ema=0.995, swa_k=0)
TWOHEAD_DEFAULTS = dict(aux_lr=1e-3, aux_wd=0.15)   # arch twohead only: AdamW for the beat10 head (arm (a) settings); trunk uses lr/wd/opt
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--arch", required=True); ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--repeats", default="0"); ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--no-tabdpt", action="store_true"); ap.add_argument("--yshuffle", type=int, default=-1); ap.add_argument("--s3", action="store_true")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--git-sha", default=""); ap.add_argument("--mask", default="")
    ap.add_argument("--epochs", type=int, default=0); ap.add_argument("--mask-mode", default="", choices=["", "dims", "pcs"], help="force the mask's PC or raw-dim lists")
    ap.add_argument("--mask-unlisted", default="keep", choices=["keep", "drop"], help="encoders absent from the mask: keep whole (default) or drop")
    ap.add_argument("--pca", type=int, default=0, help="wave-1: per-block PCA (fit on inner-train rows) to K comps before the model")
    ap.add_argument("--anchor", default="", help="wave-1 matched GBM-anchored twin: lgb_baseline dir; outputs = LightGBM cross-fitted pred + learned residual, GBM preds also as inputs")
    ap.add_argument("--enc", default="all", help="feature-subset wave: comma list of clean encoder blocks to use, 'all' or 'none'")
    ap.add_argument("--lgroups", default="V21,TL,ZL,V3,V4,Z", help="feature-subset wave: LV4 column groups kept in block L")
    ap.add_argument("--topk", type=int, default=0, help="feature-subset wave: keep the top-K of [L cols + 32 fold-fit PCs per used encoder], ranked per fold by LightGBM TreeSHAP on inner-train rows only")
    ap.add_argument("--trunk-enc", default="", help="arch twohead: comma list of encoder blocks fed to the shared trunk (with all L cols); --topk K feeds the beat10 head")
    ap.add_argument("--subset", default="", help="subset tag recorded in summary (e.g. S1)")
    ap.add_argument("--kw", default="{}", help="JSON overrides for the model constructor"); ap.add_argument("--cfg", default="{}", help="JSON overrides for lr/wd/bs/epochs/patience")
    a = ap.parse_args(); out = f"{a.out}/{a.run_id}"; os.makedirs(out, exist_ok=True); cfg = CFG[a.arch]
    if a.epochs: cfg["epochs"] = a.epochs
    cfg = {**SWEEP_DEFAULTS, **(TWOHEAD_DEFAULTS if a.arch == "twohead" else {}), **cfg, **json.loads(a.cfg)}; cfg["kw"] = {**cfg["kw"], **json.loads(a.kw)}
    TH = a.arch == "twohead"
    if TH and (not a.topk or not a.trunk_enc): raise SystemExit("arch twohead needs --topk K (beat10 head) and --trunk-enc (trunk encoder blocks)")
    if cfg["loss"] not in ("bnll", "huber") or cfg["opt"] not in ("adamw", "lion"): raise SystemExit(f"--cfg: unknown loss/opt {cfg['loss']!r}/{cfg['opt']!r}")
    torch.backends.cuda.matmul.allow_tf32 = True; dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(int(os.environ.get("DLHWM_TORCH_THREADS", "3")))   # fixed small CPU footprint; all tensors live on the GPU
    if a.s3 and len(a.git_sha) != 40: raise SystemExit("--s3 requires --git-sha <full 40-hex commit> of the code being run")
    meta = json.load(open(f"{a.data}/meta.json")); T = np.load(f"{a.data}/targets.npz"); FO = np.load(f"{a.data}/folds.npy"); gid = np.load(f"{a.data}/gid.npy")
    blocks = [b for b in meta["blocks"] if b != "L" and not (a.no_tabdpt and b == "tabdpt13")]; assert_clean(["L"] + blocks)
    L = np.load(f"{a.data}/blocks_L.npy").astype(np.float64); Lc = meta["L_cols"]
    tov = np.array(meta["tabdpt_overlap"]); tcols = [j for j, c in enumerate(Lc) if c.startswith("Z:tabdpt13") or c == "ZL:2"]
    if a.no_tabdpt: keep = [j for j in range(L.shape[1]) if j not in tcols]; L = L[:, keep]; Lc = [Lc[j] for j in keep]
    else: L[np.ix_(tov, tcols)] = np.nan          # TabDPT pretraining-overlap rows: TabDPT features marked missing
    if a.enc != "all":
        want = [] if a.enc == "none" else a.enc.split(",")
        bad = [b for b in want if b not in blocks]
        if bad: raise SystemExit(f"--enc: unknown/refused/unavailable blocks {bad}")
        blocks = [b for b in blocks if b in want]
    LG = set(a.lgroups.split(",")); kl = [j for j, c in enumerate(Lc) if c.split(":")[0] in LG]
    if not kl: raise SystemExit("--lgroups keeps no L columns")
    L = L[:, kl]; Lc = [Lc[j] for j in kl]
    E = {b: np.load(f"{a.data}/blocks_{b}.npy") for b in blocks}
    TRUNK = a.trunk_enc.split(",") if TH else []
    if [b for b in TRUNK if b not in blocks]: raise SystemExit(f"--trunk-enc: blocks not available {TRUNK}")
    if "tabdpt13" in E: E["tabdpt13"][tov] = np.nan
    MASK = parse_mask(a.mask, blocks, a.mask_mode, a.mask_unlisted) if a.mask else {}
    for b in [b for b in blocks if MASK.get(b, {}).get("weight", 1.0) == 0.0]: del E[b]; blocks.remove(b)
    n = L.shape[0]; yfit = T["fitness"].astype(np.float64); ygap = T["gap"].astype(np.float64); yb = T["beat10"].astype(np.float64)
    reps = [int(r) for r in a.repeats.split(",")]; folds = [int(k) for k in a.folds.split(",")]
    git = a.git_sha; summary = {"run_id": a.run_id, "arch": a.arch, "pure": not a.anchor, "gbm_anchor": bool(a.anchor), "distill": False, "gbm_input": bool(a.anchor), "pca": a.pca,
        "blocks": ["L"] + blocks, "no_tabdpt": a.no_tabdpt, "yshuffle": a.yshuffle, "git_sha": git, "host": host(), "cfg": cfg, "mask": a.mask or "all-dims-fallback", "mask_sha256": sha256b(open(a.mask, "rb").read()) if a.mask else None,
        "mask_spec": MASK, "optimistic": bool(a.mask), "subset": a.subset, "enc": a.enc, "lgroups": a.lgroups, "topk": a.topk, "trunk_enc": TRUNK, "n_L_cols": len(Lc),   # the mask is chosen on outer-fold OOS scores (owner-accepted overfit): mask arms are optimistic
        "data_manifest_sha256": sha256b(open(f"{a.data}/MANIFEST.json", "rb").read()), "folds": {}, "s3": []}
    for r in reps:
        oof = np.full((n, 5), np.nan); oofraw = np.full((n, 2), np.nan)
        for k in folds:
            t0 = time.time(); te = np.where(FO[r] == k)[0]; tr_all = np.where(FO[r] != k)[0]
            inner = group_folds(gid[tr_all], 10, 777 + 31 * r + k); va = tr_all[inner == 0]; tr = tr_all[inner != 0]
            yf, yg, ybb = yfit.copy(), ygap.copy(), yb.copy()
            if a.yshuffle >= 0:
                p = np.random.default_rng(a.yshuffle * 1000 + r * 10 + k).permutation(tr_all); yf[tr_all], yg[tr_all], ybb[tr_all] = yfit[p], ygap[p], yb[p]
            torch.manual_seed(a.seed * 100 + r * 10 + k); np.random.seed(a.seed * 100 + r * 10 + k)
            QL = rankgauss_fit(L[tr]); XL = rankgauss_apply(L, QL); ML = (~np.isfinite(L)).astype(np.float32)
            Xe, Me = [], []
            for b in blocks:
                Z, miss = robust_gpu(E[b], tr, dev); spec = MASK.get(b)
                if spec and spec["mode"] == "dims": Z = Z[:, spec["kept"]]
                elif spec and spec["mode"] == "pcs": Z = pca_exact_gpu(Z, tr_all[~miss[tr_all]], spec["n_pcs"], dev)[:, spec["kept"]]
                elif a.pca: Z = pca_gpu(Z, tr[~miss[tr]], a.pca, dev)
                if spec: Z = Z * np.float32(spec["weight"])
                Z[miss] = 0
                Xe.append(Z); Me.append(miss)
            Me = np.stack(Me, 1).astype(np.float32) if Me else np.zeros((n, 0), np.float32)
            sel_names = None; ublocks = blocks; XT = MT = None
            if a.topk:   # rank [L cols + 32 PCs per used encoder] on inner-train rows only; the model then sees only the top-K columns (as block L)
                cand = [XL]; cm = [ML]; names = list(Lc)
                for b, Z0, mi in zip(blocks, Xe, Me.T.astype(bool)):
                    P = Z0 if a.pca or MASK.get(b) else pca_gpu(Z0, tr[~mi[tr]], 32, dev); P[mi] = 0; cand.append(P); cm.append(np.repeat(mi[:, None], P.shape[1], 1).astype(np.float32)); names += [f"{b}:pc{i}" for i in range(P.shape[1])]
                C = np.concatenate(cand, 1).astype(np.float32); CM = np.concatenate(cm, 1)
                imp = shap_rank(C[tr], yf[tr], ybb[tr], a.seed * 100 + r * 10 + k); sel = np.argsort(-imp)[:a.topk]
                sel_names = [names[i] for i in sel]
                if TH:   # twohead: top-K columns feed only the beat10 head; the trunk keeps all L cols + the --trunk-enc blocks
                    XT, MT = C[:, sel], CM[:, sel]; ti = [i for i, b in enumerate(blocks) if b in TRUNK]; Xe = [Xe[i] for i in ti]; Me = Me[:, ti]; ublocks = [blocks[i] for i in ti]
                else: XL, ML = C[:, sel], CM[:, sel]; Xe, Me = [], np.zeros((n, 0), np.float32)
            if a.anchor:
                AN, aprov = load_anchor(a.anchor, r, k, n, FO, {"fitness": yfit, "gap": ygap, "beat10": yb}, summary["data_manifest_sha256"])   # outer-train rows = inner cross-fitted OOF; test rows = test preds
                summary.setdefault("anchor_provenance", {})[f"r{r}f{k}"] = aprov
                if a.yshuffle >= 0: raise SystemExit("anchor + yshuffle not supported (the GBM anchor was fit on real labels)")
            fm, fs = np.nanmedian(yf[tr]), np.nanstd(yf[tr]); gm, gs = np.nanmedian(yg[tr]), np.nanstd(yg[tr])
            Yf = (yf - fm) / fs; Yg = (yg - gm) / gs
            if a.anchor:
                aF = (AN[:, 0] - fm) / fs; aG = (AN[:, 1] - gm) / gs; pB = np.clip(AN[:, 2], 0.02, 0.98); aB = np.log(pB / (1 - pB))
                ANC = np.column_stack([aF, aG, aB]).astype(np.float32); XL = np.concatenate([XL, ANC], 1); ML = np.concatenate([ML, np.zeros_like(ANC)], 1)
            else: ANC = np.zeros((n, 3), np.float32)
            G = lambda x: torch.as_tensor(x, device=dev)
            gXL, gML, gXe, gMe, gANC = G(XL), G(ML), [G(x) for x in Xe], G(Me), G(ANC); gXT, gMT = (G(XT), G(MT)) if TH else (None, None)
            X2 = (lambda ii: {"xt": gXT[ii], "mt": gMT[ii]}) if TH else (lambda ii: {}); gYf, gYg, gYb = G(Yf.astype(np.float32)), G(Yg.astype(np.float32)), G(ybb.astype(np.float32))
            torch.manual_seed(a.seed * 100 + r * 10 + k); np.random.seed(a.seed * 100 + r * 10 + k)
            model = ARCH[a.arch](XL.shape[1], [x.shape[1] for x in Xe], **({"dT": XT.shape[1]} if TH else {}), **cfg["kw"]).to(dev)
            ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(cfg["ema"]))
            mk_opt = lambda ps, lr, wd, kind: torch.optim.AdamW(ps, lr=lr, weight_decay=wd, betas=(0.9, 0.98)) if kind == "adamw" else Lion(ps, lr=lr, weight_decay=wd)
            if TH: pt, pa = model.param_split(); opts = [mk_opt(pt, cfg["lr"], cfg["wd"], cfg["opt"]), mk_opt(pa, cfg["aux_lr"], cfg["aux_wd"], "adamw")]
            else: opts = [mk_opt(model.parameters(), cfg["lr"], cfg["wd"], cfg["opt"])]
            opt = MultiOpt(opts)
            spe = math.ceil(len(tr) / cfg["bs"]); tot = cfg["epochs"] * spe; warm = max(1, int(0.05 * tot))
            sched = MultiSched([torch.optim.lr_scheduler.LambdaLR(o_, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / tot)))) for o_ in opts])
            def anch(o, ii):
                if not a.anchor: return o
                o = dict(o); o["fit_mu"] = o["fit_mu"] + gANC[ii, 0]; o["gap_mu"] = o["gap_mu"] + gANC[ii, 1]; o["b10"] = o["b10"] + gANC[ii, 2]; return o
            def fwd(m, idx):
                ii = G(idx); return anch(m(gXL[ii], gML[ii], [x[ii] for x in gXe], gMe[ii], **X2(ii)), ii)
            def loss_of(o, idx):
                ii = G(idx); yfi, ygi, ybi = gYf[ii], gYg[ii], gYb[ii]; mb = torch.isfinite(ybi)
                lb = F.binary_cross_entropy_with_logits(o["b10"][mb].float(), (ybi[mb] * 0.95 + 0.025)) if mb.any() else o["b10"].sum() * 0
                if cfg["loss"] == "huber":   # mean-only robust regression; the log-scale heads are left untrained (unused for scoring)
                    lf, lg = huber_mean(o["fit_mu"].float(), yfi, cfg["huber_delta"]), huber_mean(o["gap_mu"].float(), ygi, cfg["huber_delta"])
                else: lf, lg = bnll(o["fit_mu"].float(), o["fit_ls"].float(), yfi), bnll(o["gap_mu"].float(), o["gap_ls"].float(), ygi)
                parts = {"fit": lf, "gap": lg, "b10": lb, "rank": ranknet(o["fit_mu"].float(), yfi)}
                tot_ = parts["fit"] + cfg["w_gap"] * parts["gap"] + cfg["w_b10"] * parts["b10"] + cfg["w_rank"] * parts["rank"]
                if cfg["w_b10rank"]: parts["b10rank"] = b10rank(o["b10"].float(), ybi); tot_ = tot_ + cfg["w_b10rank"] * parts["b10rank"]
                return tot_, parts
            @torch.no_grad()
            def predict(m, idx, me_override=None):
                m.eval(); outs = []; GATES.clear()
                for s in range(0, len(idx), 1024):
                    ii = G(idx[s:s + 1024]); me = gMe[ii] if me_override is None else me_override(gMe[ii])
                    xl, ml = gXL[ii], gML[ii]
                    if me_override is not None and getattr(me_override, "dropL", False): ml = torch.ones_like(ml)
                    with torch.autocast(dev, dtype=torch.bfloat16): o = anch(m(xl, ml, [x[ii] for x in gXe], me, **X2(ii)), ii)
                    outs.append(torch.stack([o["fit_mu"], o["fit_ls"], o["gap_mu"], o["gap_ls"], o["b10"]], 1).float().cpu().numpy())
                    if getattr(m, "last_gate", None) is not None: GATES.append(m.last_gate.float().cpu().numpy())
                return np.concatenate(outs)
            def score(P, idx):
                pf = P[:, 0] * fs + fm; return r2(yf[idx], pf), auc(ybb[idx], P[:, 4])
            hist = []; best = (-1e9, -1, None); bad = 0; step = 0; topk_snap = []; swa_rec = None
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
                if cfg["swa_k"]:
                    topk_snap.append((crit, ep, {k_: v.detach().clone() for k_, v in ema.module.state_dict().items()})); topk_snap.sort(key=lambda t: -t[0]); del topk_snap[cfg["swa_k"]:]
                if crit > best[0]: best = (crit, ep, {k_: v.detach().clone() for k_, v in ema.module.state_dict().items()}); bad = 0
                else: bad += 1
                if ep % 10 == 0: log(a.arch, f"r{r}f{k} ep{ep} tl {np.mean(tl):.4f} vl {vl:.4f} vR2 {vr2:.4f} vAUC {vauc:.4f}")
                json.dump(hist, open(f"{out}/hist_r{r}f{k}.json", "w"))
                if bad >= cfg["patience"]: break
            if cfg["swa_k"] and len(topk_snap) > 1:   # SWA over the k best inner-val EMA snapshots; kept only if it beats the best single snapshot on inner-val
                avg = {k_: (torch.stack([t[2][k_].float() for t in topk_snap]).mean(0).to(best[2][k_].dtype) if best[2][k_].is_floating_point() else best[2][k_]) for k_ in best[2]}
                ema.module.load_state_dict(avg); Pv = predict(ema.module, va); vr2, vauc = score(Pv, va); c_ = vr2 + (vauc if np.isfinite(vauc) else 0)
                swa_rec = {"k": len(topk_snap), "eps": [t[1] for t in topk_snap], "val_crit": float(c_), "best_single": float(best[0]), "used": bool(c_ > best[0])}
                if c_ > best[0]: best = (c_, best[1], avg)
            ema.module.load_state_dict(best[2]); m = ema.module
            P = predict(m, te); oof[te] = P; oofraw[te, 0] = P[:, 0] * fs + fm; oofraw[te, 1] = P[:, 2] * gs + gm; tr2, tauc = score(P, te)
            # block ablation at inference (representation usefulness): mark one block missing at a time; L ablated via all-missing mask
            abl = {}
            for bi, b in enumerate(["L"] + (ublocks if gMe.shape[1] else [])):
                if b == "L":
                    f_ = lambda me: me; f_.dropL = True
                else:
                    def f_(me, bi=bi - 1): me = me.clone(); me[:, bi] = 1; return me
                Pa = predict(m, te, f_); ar2, aauc = score(Pa, te); abl[b] = {"d_r2": tr2 - ar2, "d_auc": tauc - aauc}
            gates = None
            if a.arch == "gatedset":
                GATES.clear(); predict(m, te); gates = dict(zip(["L"] + (blocks if gMe.shape[1] else []), np.concatenate(GATES).mean(0).round(4).tolist()))
            fold_rec = {"best_val_crit": float(best[0]), "topk_features": sel_names, "r": r, "k": k, "n_tr": len(tr), "n_va": len(va), "n_te": len(te), "best_ep": best[1], "epochs_run": len(hist), "test_r2": tr2, "test_auc": tauc,
                        "secs": time.time() - t0, "swa": swa_rec, "ablation": abl, "gates": gates, "target_norm": [fm, fs, gm, gs]}
            buf = io.BytesIO(); torch.save({"state_dict": best[2], "arch": a.arch, "kw": cfg["kw"], "dL": XL.shape[1], "dims": [x.shape[1] for x in Xe], "blocks": ublocks, "dT": XT.shape[1] if TH else 0, "topk_features": sel_names,
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
