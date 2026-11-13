#!/usr/bin/env python3
"""H449 red-team diagnostics (reads DL-Train/DL-Val packet + fold_sig only; no model, no sealed data).
(1) regret-curriculum sampling ESS at beta 0.5/1.0/1.5 and the mass it puts on incumbent gate failures;
(2) val91 regcheck losers: is the candidate's pick a gate failure (utility exactly -2.0)?"""
import json, sys, os
from pathlib import Path
import numpy as np
B = Path("/home/ubuntu/jope-data"); SCR = B / "hwave-h449"
LOOP = Path("/home/ubuntu/JopeDime/.agent/worktrees/flywheel-gp-fused/research/flywheel_selfplay")
sys.path.insert(0, str(LOOP)); os.chdir(LOOP.parents[1])
import run_fused_closed_loop as L
selected, pools, tables = L.load_packet(SCR / "input"); del tables
assert all(r["dl_split"] in ("DL-Train", "DL-Val") for r in selected["train"] + selected["dev"])
train = [r["context_sha256"] for r in selected["train"]]; dev = [r["context_sha256"] for r in selected["dev"]]
an = json.load(open(B / "hwave-h447/analysis_inc_train.json")); assert an["train"] == train
u = np.array(an["u_inc_train"]); reg = np.array(an["oracle"]) - u
fail = u == -2.0  # exact: utility() returns -2.0 only on gate failure (passing utility = -fitness can be < -2)
rz = (reg - reg.mean()) / reg.std()
out = {"n_train": len(train), "inc_gatefail_exact": int(fail.sum()), "inc_util_le_-2": int((u <= -2.0 + 1e-9).sum()),
       "regret_quantiles_50_90_99_100": np.percentile(reg, [50, 90, 99, 100]).round(3).tolist(), "rz_max": float(rz.max()), "ess": {}}
for b in (0.25, 0.5, 1.0, 1.5):
    w = np.exp(b * rz); p = w / w.sum(); top = np.sort(p)[::-1]
    out["ess"][str(b)] = {"ess": float(1 / (p ** 2).sum()), "mass_on_inc_gatefails": float(p[fail].sum()), "top3_mass": float(top[:3].sum()), "top20_mass": float(top[:20].sum())}
fit_scale = np.array([np.std([float(f) for f, g in zip(pools[c]["fitness"], pools[c]["gate"]) if g]) if any(pools[c]["gate"]) else np.nan for c in train])
out["passing_fitness_std_quantiles_1_50_99"] = np.nanpercentile(fit_scale, [1, 50, 99]).round(4).tolist()
# val losers
vals = {}
for p in sorted((SCR / "results/fold_sig").glob("*_full_seed*.json")):
    o = json.load(open(p)); d = np.array(o["val_diff"]); rows = []
    for i in np.argsort(d)[:4]:
        c = dev[i]; pl = pools[c]; util = sorted({-float(f) for f, g in zip(pl["fitness"], pl["gate"]) if g})
        # candidate utility y = u_inc + d; if the candidate gate-failed, y = -2 and u_inc = -2 - d must be a passing utility
        x = -2.0 - float(d[i]); cand_fail = any(abs(x - v) < 1e-9 for v in util)
        rows.append({"ctx": c[:12], "diff": round(float(d[i]), 4), "candidate_gatefail_consistent": cand_fail, "frac_programs_failing": round(1 - float(np.mean(pl["gate"])), 3)})
    vals[p.name] = {"mean": float(d.mean()), "sum_of_3_worst": float(np.sort(d)[:3].sum()), "mean_excl_3_worst": float(np.sort(d)[3:].sum() / len(d)), "worst": rows}
out["val_losers"] = vals
# (3) CV OOF split per arm/seed: recovered incumbent gate failures, new candidate gate failures, passing-context W/L
import hashlib
fam = {r["context_sha256"]: r["family_id"] for r in selected["train"]}
tr_fams = sorted({fam[c] for c in train}, key=lambda f: hashlib.sha256(f"h444/{f}".encode()).hexdigest())
fs = {f: sum(1 for c in train if fam[c] == f) for f in tr_fams}; fof = {}; load = [0] * 5
for f in sorted(tr_fams, key=lambda f: -fs[f]):
    j = int(np.argmin(load)); fof[f] = j; load[j] += fs[f]
fold = np.array([fof[fam[c]] for c in train]); cvs = {}
for arm in ["regret_ctrl", "regret_blend", "regret_soup", "regret_soup_blend"]:
    for sd in (1, 2, 3):
        x = np.full(len(train), np.nan)
        for k in range(5):
            x[fold == k] = json.load(open(SCR / f"results/fold_sig/{arm}_cv{k}_seed{sd}.json"))["oof"]
        y = u + x; newf = (~fail) & (np.abs(y + 2.0) < 1e-12); rec = fail & (x > 0)
        cvs[f"{arm}:s{sd}"] = {"recovered_inc_gatefails": int(rec.sum()), "new_gatefails": int(newf.sum()), "gain_from_recovered": float(x[fail].sum() / len(x)),
                               "gain_from_new_fails": float(x[newf].sum() / len(x)), "gain_passing_other": float(x[(~fail) & (~newf)].sum() / len(x)),
                               "W_L_passing": [int((x[~fail] > 0).sum()), int((x[~fail] < 0).sum())],
                               "fold_inc_gatefails": [int(fail[fold == k].sum()) for k in range(5)]}
out["cv_gate_split"] = cvs
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "/dev/stdout", "w"), indent=1)
