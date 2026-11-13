#!/usr/bin/env python3
"""H450 red-team (reads DL-Train packet + fold_sig + H447 incumbent per-context utilities only; no model, no val, no sealed data).
Per arm/seed: split the pooled OOF paired gain into (a) the incumbent's 19 TRAIN gate failures (utility exactly -2.0),
(b) new candidate gate failures, (c) the remaining passing contexts; recompute gain, W/T/L and Elo on (c) with a
family-bootstrap CI; fold composition of the failures; decision-level seed disagreement. Missing cells are skipped."""
import json, sys, os, hashlib, itertools
from pathlib import Path
import numpy as np
B = Path("/home/ubuntu/jope-data"); SCR = B / "hwave-h450"; SIG = SCR / "results/fold_sig"
sel = json.load(open(SCR / "input/selected.json"))
assert all(r["dl_split"] in ("DL-Train", "DL-Val") for r in sel["train"] + sel["dev"])
train = [r["context_sha256"] for r in sel["train"]]; fam = {r["context_sha256"]: r["family_id"] for r in sel["train"]}
an = json.load(open(B / "hwave-h447/analysis_inc_train.json")); assert an["train"] == train
u = np.array(an["u_inc_train"]); fail = u == -2.0; assert fail.sum() == 19
tr_fams = sorted({fam[c] for c in train}, key=lambda f: hashlib.sha256(f"h444/{f}".encode()).hexdigest())
fs = {f: sum(1 for c in train if fam[c] == f) for f in tr_fams}; fof = {}; load = [0] * 5
for f in sorted(tr_fams, key=lambda f: -fs[f]):
    j = int(np.argmin(load)); fof[f] = j; load[j] += fs[f]
fold = np.array([fof[fam[c]] for c in train]); assert load == [244, 244, 244, 244, 243]
fa = np.array([fam[c] for c in train])
def elo(x):
    x = min(max(x, 1e-4), 1 - 1e-4); return 400 * np.log10(x / (1 - x))
def score(x): return ((x > 0).sum() + 0.5 * (x == 0).sum()) / len(x)
def boot(x, m, B_=2000, seed=0):
    idx = [np.where((fa == f) & m)[0] for f in sorted(set(fa[m]))]; r = np.random.default_rng(seed); g, e = [], []
    for _ in range(B_):
        ix = np.concatenate([idx[j] for j in r.integers(0, len(idx), len(idx))]); g.append(x[ix].mean()); e.append(elo(score(x[ix])))
    return [float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))], [float(np.percentile(e, 2.5)), float(np.percentile(e, 97.5))]
out = {"inc_gatefails_per_fold": [int(fail[fold == k].sum()) for k in range(5)], "fold_sizes": load,
       "inc_util_quantiles_passing": np.percentile(u[~fail], [1, 50, 99]).round(3).tolist(), "cells": {}, "fold1": {}, "seed_disagreement": {}}
X = {}
for arm in ["big_ctrl", "regret_pct", "regret_pct_avg", "regret_pct_gate"]:
    for sd in (1, 2, 3):
        fns = [SIG / f"{arm}_cv{k}_seed{sd}.json" for k in range(5)]
        if not all(p.exists() for p in fns): continue
        x = np.full(len(train), np.nan)
        for k, p in enumerate(fns): x[fold == k] = json.load(open(p))["oof"]
        X[(arm, sd)] = x
        y = u + x; newf = (~fail) & (np.abs(y + 2.0) < 1e-12); rec = fail & (x > 0); pas = (~fail) & (~newf)
        gci, eci = boot(np.where(pas, x, 0.0), pas, seed=sd)
        xp = x[pas]; top = np.sort(x)[::-1]
        out["cells"][f"{arm}:s{sd}"] = {
            "oof_gain_all": float(x.mean()), "recovered_inc_gatefails": int(rec.sum()), "new_gatefails": int(newf.sum()),
            "gain_from_inc_gatefail_ctx": float(x[fail].sum() / len(x)), "gain_from_new_fails": float(x[newf].sum() / len(x)),
            "gain_from_passing_other": float(x[pas].sum() / len(x)),
            "excl_floor": {"n": int(pas.sum()), "mean_gain": float(xp.mean()), "gain_ci95_family_boot": gci,
                           "W_T_L": [int((xp > 0).sum()), int((xp == 0).sum()), int((xp < 0).sum())], "elo": float(elo(score(xp))), "elo_ci95_family_boot": eci},
            "WTL_all": [int((x > 0).sum()), int((x == 0).sum()), int((x < 0).sum())], "elo_all": float(elo(score(x))),
            "top10_share_of_gain": float(top[:10].sum() / x.sum()) if x.sum() else None,
            "fold_gains": [float(x[fold == k].mean()) for k in range(5)],
            "fold_gains_excl_floor": [float(x[(fold == k) & pas].mean()) for k in range(5)],
            "fold1_gain_from_inc_gatefails": float(x[(fold == 1) & fail].sum() / (fold == 1).sum())}
for arm in ["big_ctrl", "regret_pct", "regret_pct_avg", "regret_pct_gate"]:
    ss = [s for s in (1, 2, 3) if (arm, s) in X]
    for a_, b_ in itertools.combinations(ss, 2):
        out["seed_disagreement"][f"{arm}:{a_}v{b_}"] = int((X[(arm, a_)] != X[(arm, b_)]).sum())
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "/dev/stdout", "w"), indent=1)
