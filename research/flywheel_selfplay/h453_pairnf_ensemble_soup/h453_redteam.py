#!/usr/bin/env python3
"""H453 red-team (adapted from h452_redteam.py; no model, no val labels beyond the logged val91 diffs, no sealed data).
Inputs: fold_sig (OOF paired gains per arm/partition/fold/seed, z-logs), val91 per-context diffs of the regression checks,
inc_analysis.json (incumbent per-context TRAIN utility + fold maps written by this wave). Rebuilds both floor-stratified
partitions and asserts they match the logged fold maps and floors per fold (4/4/4/3/4). Reports per arm and seed:
floor-recovery vs non-floor split, new gate failures, non-floor Elo + CI, seed disagreement, partition agreement,
overlap of kept decisions with the control, winner's curse of the max-of-N selection, val91 seed agreement."""
import json, sys, hashlib, itertools, os
from pathlib import Path
import numpy as np
SCR = Path(os.environ.get("WAVE_ROOT", Path(__file__).resolve().parent)); SIG = SCR / "results/fold_sig"
AN = json.load(open(SCR / "inc_analysis.json")); CTRL = os.environ.get("CTRL", "pairnf_z40")
ARMS = os.environ.get("ARMS", "pairnf_z40,ens3_z40,ens3_z60,ens3_agree_z40,soup3_z40").split(","); SEEDS = (1, 2, 3); SALTS = ["h452a/", "h452b/"]
sel = json.load(open(SCR / "input/selected.json"))
assert all(r["dl_split"] in ("DL-Train", "DL-Val") for r in sel["train"] + sel["dev"])
train = [r["context_sha256"] for r in sel["train"]]; assert AN["train"] == train
fam = {r["context_sha256"]: r["family_id"] for r in sel["train"]}; fa = np.array([fam[c] for c in train])
assert not ({fam[c] for c in train} & {r["family_id"] for r in sel["dev"]}), "train/val family overlap"
u = np.array(AN["u_inc_train"]); fail = np.array(AN["inc_fail_train"], bool); assert fail.sum() == 19
assert abs(u.mean() - 1.420532014416241) < 1e-9
ff = {}
for c, f_ in zip(train, fail):
    if f_: ff[fam[c]] = ff.get(fam[c], 0) + 1
FOLD = []
for pi, salt in enumerate(SALTS):
    tf = sorted({fam[c] for c in train}, key=lambda f: hashlib.sha256(f"{salt}{f}".encode()).hexdigest())
    fs = {f: int((fa == f).sum()) for f in tf}; fof, load, fl = {}, [0] * 5, [0] * 5
    for f in sorted([f for f in tf if f in ff], key=lambda f: (-ff[f], -fs[f])):
        j = min(range(5), key=lambda i: (fl[i], load[i])); fof[f] = j; load[j] += fs[f]; fl[j] += ff[f]
    for f in sorted([f for f in tf if f not in ff], key=lambda f: -fs[f]):
        j = int(np.argmin(load)); fof[f] = j; load[j] += fs[f]
    assert load == [244, 244, 244, 244, 243] and fl == [4, 4, 4, 3, 4], (load, fl)
    FOLD.append(np.array([fof[fam[c]] for c in train])); assert (FOLD[-1] == np.array(AN["folds"][pi])).all()
def elo(x):
    x = min(max(x, 1e-4), 1 - 1e-4); return float(400 * np.log10(x / (1 - x)))
def score(x): return ((x > 0).sum() + 0.5 * (x == 0).sum()) / max(len(x), 1)
ufam = sorted(set(fa)); fidx = [np.where(fa == f)[0] for f in ufam]
def boots(B=1000, seed=0):
    r = np.random.default_rng(seed)
    for _ in range(B): yield np.concatenate([fidx[j] for j in r.integers(0, len(fidx), len(fidx))])
X = {}
for arm in ARMS:
    for sd in SEEDS:
        xs = []
        for pi in range(2):
            x = np.full(len(train), np.nan)
            for k in range(5):
                j = json.load(open(SIG / f"{arm}_p{pi}cv{k}_seed{sd}.json")); assert j["partition"] == SALTS[pi]
                x[FOLD[pi] == k] = j["oof"]
            assert not np.isnan(x).any(); xs.append(x)
        X[(arm, sd)] = xs
out = {"inc_gatefails_train": int(fail.sum()), "inc_gatefails_per_fold": [[int(fail[F == k].sum()) for k in range(5)] for F in FOLD],
       "cells": {}, "seed_disagreement": {}, "partition_agreement": {}, "vs_control_decisions": {}, "selection_bootstrap": {}, "val91": {}}
for (arm, sd), xs in X.items():
    c = {}
    for pi, x in enumerate(xs):
        y = u + x; newf = (~fail) & (np.abs(y + 2.0) < 1e-12); rec = fail & (x > 0); pas = (~fail) & (~newf); xp = x[pas]
        es = [elo(score(x[ix][pas[ix]])) for ix in boots(seed=sd * 10 + pi)]
        c[SALTS[pi]] = {"gain_all": float(x.mean()), "gain_from_floors": float(x[fail].sum() / len(x)), "gain_nonfloor_total": float(x[~fail].sum() / len(x)),
                        "recovered": int(rec.sum()), "new_gatefails": int(newf.sum()), "nonfloor_pass_WTL": [int((xp > 0).sum()), int((xp == 0).sum()), int((xp < 0).sum())],
                        "nonfloor_pass_elo": elo(score(xp)), "nonfloor_pass_elo_ci95": np.percentile(es, [2.5, 97.5]).round(2).tolist(),
                        "nonfloor_fold_elo": [elo(score(x[(FOLD[pi] == k) & pas])) for k in range(5)],
                        "fold_gain_wo_floors": [float(x[(FOLD[pi] == k) & ~fail].mean()) for k in range(5)]}
    out["cells"][f"{arm}:s{sd}"] = c
    a_, b_ = xs; ch = (a_ != 0) | (b_ != 0)
    out["partition_agreement"][f"{arm}:s{sd}"] = {"changed_either": int(ch.sum()), "changed_both": int(((a_ != 0) & (b_ != 0)).sum()),
        "same_sign_both": int(((np.sign(a_) == np.sign(b_)) & (a_ != 0)).sum())}
    if arm != CTRL:
        cx = X[(CTRL, sd)]
        out["vs_control_decisions"][f"{arm}:s{sd}"] = [{"arm_changed": int((x != 0).sum()), "ctrl_changed": int((y_ != 0).sum()),
            "both_changed": int(((x != 0) & (y_ != 0)).sum()), "identical_oof": int((x == y_).sum()),
            "arm_only_gain_sum": float(x[(x != 0) & (y_ == 0)].sum()), "ctrl_only_gain_sum": float(y_[(y_ != 0) & (x == 0)].sum())} for x, y_ in zip(xs, cx)]
for arm in ARMS:
    for s1, s2 in itertools.combinations(SEEDS, 2):
        out["seed_disagreement"][f"{arm}:{s1}v{s2}"] = [int((X[(arm, s1)][pi] != X[(arm, s2)][pi]).sum()) for pi in range(2)]
def nfe(arm, sd, ix=None):
    v = []
    for x in X[(arm, sd)]:
        xx, nf = (x, ~fail) if ix is None else (x[ix], ~fail[ix]); v.append(elo(score(xx[nf])))
    return float(np.mean(v))
for sd in SEEDS:
    full = {a: nfe(a, sd) for a in ARMS}; best = max(full, key=full.get); win = 0; pos = 0; se = []
    for ix in boots(seed=100 + sd):
        e = {a: nfe(a, sd, ix) for a in ARMS}; m = max(e, key=e.get); win += m == best; pos += e[m] > 0; se.append(e[m])
    out["selection_bootstrap"][f"s{sd}"] = {"selected": best, "nonfloor_elo": full, "p_selected_is_argmax": win / 1000,
        "p_max_arm_nonfloor_elo_gt_0": pos / 1000, "max_arm_nonfloor_elo_ci95": np.percentile(se, [2.5, 97.5]).round(2).tolist()}
V = {}
for sd in SEEDS:
    fs_ = list(SIG.glob(f"*_full_seed{sd}.json")); assert len(fs_) == 1, fs_
    j = json.load(open(fs_[0])); V[sd] = np.array(j["val_diff"])
    out["val91"][f"s{sd}"] = {"arm": j["arm"], "nonzero_idx": np.where(V[sd] != 0)[0].tolist(), "wins": int((V[sd] > 0).sum()), "losses": int((V[sd] < 0).sum()),
                             "sum_gain": float(V[sd].sum()), "max_abs": float(np.abs(V[sd]).max())}
for s1, s2 in itertools.combinations(SEEDS, 2):
    out["val91"][f"{s1}v{s2}"] = {"differ": int((V[s1] != V[s2]).sum()), "both_changed": int(((V[s1] != 0) & (V[s2] != 0)).sum())}
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "/dev/stdout", "w"), indent=1)
