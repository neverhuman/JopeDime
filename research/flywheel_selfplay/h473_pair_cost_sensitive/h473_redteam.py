#!/usr/bin/env python3
"""H473 red-team (copied from h472_redteam.py, itself from h471/h470_redteam.py; H470 dose block replaced by the pre-registered kept-set overlap /
z-inflation check; adapted from h452_redteam.py; no model, no val labels beyond the logged val91 diffs, no sealed data).
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
ARMS = os.environ.get("ARMS", "pairnf_z40,pgap_z40,pm25_z40").split(","); SEEDS = (1, 2, 3); SALTS = ["h452a/", "h452b/"]
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
# H468: identical fold OOF vectors between seeds (or vs the control) despite distinct adapters: the z >= 0.40 gate keeps the
# same changed set with the same picks, so the discrete utility vector (and the fold gain) repeats exactly.
O = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else None
ident = []
for pi in range(2):
    for k in range(5):
        cells = {}
        for arm in ARMS:
            for sd in SEEDS:
                j = json.load(open(SIG / f"{arm}_p{pi}cv{k}_seed{sd}.json")); cells[(arm, sd)] = j
        for (a1, s1), (a2, s2) in itertools.combinations(cells, 2):
            j1, j2 = cells[(a1, s1)], cells[(a2, s2)]
            if j1["oof"] == j2["oof"] and any(v != 0 for v in j1["oof"]):
                z1 = dict(zip(j1["zlog"]["idx_changed"], j1["zlog"]["z_changed"])); z2 = dict(zip(j2["zlog"]["idx_changed"], j2["zlog"]["z_changed"]))
                com = set(z1) & set(z2)
                ident.append({"cell": f"p{pi}cv{k}", "a": f"{a1}:s{s1}", "b": f"{a2}:s{s2}", "gain": j1["gain"], "same_adapter_sha": j1["adapter_sha"] == j2["adapter_sha"],
                              "n_kept": int(sum(1 for v in j1["oof"] if v != 0)), "changed_sets_equal": set(z1) == set(z2),
                              "max_abs_z_diff_on_common_changed": float(max([abs(z1[i] - z2[i]) for i in com] or [0.0]))})
if O is not None:
    O["identical_fold_oof"] = ident; json.dump(O, open(sys.argv[1], "w"), indent=1)
# H470/H471: equal-pooled-Elo coincidence check (Elo depends only on W - L over N). H471 replaces the H470 gn dose check with the
# pre-registered displacement / z-inflation check: kept counts and median |z| of changed picks vs the control (flag < 0.5x
# gate-starved, > 2x z-inflated), kept-set overlap with the control (subset / superset / extras) and the non-floor W/L of the
# arm-only extras and of the control-only drops.
if O is not None:
    eq = {}; disp = {}
    for sd in SEEDS:
        wl = {a: [int((np.concatenate(X[(a, sd)]) > 0).sum()), int((np.concatenate(X[(a, sd)]) == 0).sum()), int((np.concatenate(X[(a, sd)]) < 0).sum())] for a in ARMS}
        eq[f"s{sd}"] = {"wtl": wl, "equal_net_pairs": [[a1, a2] for a1, a2 in itertools.combinations(ARMS, 2) if wl[a1][0] - wl[a1][2] == wl[a2][0] - wl[a2][2]]}
        d_ = {}
        for a in ARMS:
            ch = 0; kept = 0; zabs = []; sym = 0; ids_diff = 0
            for pi in range(2):
                for k in range(5):
                    j = json.load(open(SIG / f"{a}_p{pi}cv{k}_seed{sd}.json")); jc = json.load(open(SIG / f"{CTRL}_p{pi}cv{k}_seed{sd}.json"))
                    ch += j["zlog"]["n_changed"]; kept += j["zlog"]["n_kept"]; zabs += [abs(z) for z in j["zlog"]["z_changed"]]
                    sym += len(set(j["zlog"]["idx_changed"]) ^ set(jc["zlog"]["idx_changed"])); ids_diff += sum(1 for p_, q_ in zip(j["oof"], jc["oof"]) if p_ != q_)
            ov = {"arm_kept_rows": 0, "ctrl_kept_rows": 0, "both_kept_rows": 0, "extras_nonfloor_WTL": [0, 0, 0], "drops_nonfloor_ctrl_WTL": [0, 0, 0],
                  "extras_gain_sum": 0.0, "drops_ctrl_gain_sum": 0.0}
            for x, y_ in zip(X[(a, sd)], X[(CTRL, sd)]):
                ka = x != 0; kc = y_ != 0; ex = ka & ~kc & ~fail; dr = kc & ~ka & ~fail
                ov["arm_kept_rows"] += int(ka.sum()); ov["ctrl_kept_rows"] += int(kc.sum()); ov["both_kept_rows"] += int((ka & kc).sum())
                ov["extras_nonfloor_WTL"] = [ov["extras_nonfloor_WTL"][0] + int((x[ex] > 0).sum()), ov["extras_nonfloor_WTL"][1] + int((x[ex] == 0).sum()), ov["extras_nonfloor_WTL"][2] + int((x[ex] < 0).sum())]
                ov["drops_nonfloor_ctrl_WTL"] = [ov["drops_nonfloor_ctrl_WTL"][0] + int((y_[dr] > 0).sum()), ov["drops_nonfloor_ctrl_WTL"][1] + int((y_[dr] == 0).sum()), ov["drops_nonfloor_ctrl_WTL"][2] + int((y_[dr] < 0).sum())]
                ov["extras_gain_sum"] += float(x[ka & ~kc].sum()); ov["drops_ctrl_gain_sum"] += float(y_[kc & ~ka].sum())
            ov["subset_of_ctrl"] = ov["both_kept_rows"] == ov["arm_kept_rows"]; ov["superset_of_ctrl"] = ov["both_kept_rows"] == ov["ctrl_kept_rows"]
            d_[a] = {"changed": ch, "kept": kept, "median_abs_z_changed": float(np.median(zabs)) if zabs else 0.0,
                     "share_z_ge_040": float(np.mean(np.array(zabs) >= 0.40)) if zabs else 0.0, "changed_set_symdiff_vs_ctrl": sym,
                     "oof_decisions_differ_vs_ctrl": ids_diff, "kept_overlap_vs_ctrl": ov}
        for a in ARMS:
            if a != CTRL:
                rk = d_[a]["kept"] / max(d_[CTRL]["kept"], 1); rz = d_[a]["median_abs_z_changed"] / max(d_[CTRL]["median_abs_z_changed"], 1e-12)
                d_[a]["kept_ratio_vs_ctrl"] = rk; d_[a]["median_abs_z_ratio_vs_ctrl"] = rz
                d_[a]["gate_starved"] = bool(rk < 0.5); d_[a]["z_inflated"] = bool(rk > 2.0)
        disp[f"s{sd}"] = d_
    O["equal_elo_coincidence"] = eq; O["displacement_check"] = disp; json.dump(O, open(sys.argv[1], "w"), indent=1)
