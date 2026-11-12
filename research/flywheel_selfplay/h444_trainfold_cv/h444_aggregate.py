#!/usr/bin/env python3
"""Aggregate H444 per-seed results into metrics.json.

Sources: results/seed1 and results/seed3 (summary.json, all three arms + regcheck);
results/seed2 (progress.log only: fbal_accum full CV, swa_cos folds 0-1, then self-abort on the
MemAvailable<15GB guard); results/seed2_hilr (summary.json, hilr_wd only, relaunched).
Utility is higher-is-better (negated fitness, -2.0 on gate failure). Ties never count as wins.
"""
import json, re, sys, statistics as st
from pathlib import Path
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/jope-data/hwave-h444")
R = ROOT / "results"
DOPE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
INC = 1.1389572403748425
MARGIN = 0.002

# sealed / holdout guard: every context in the packet must be DL-Train or DL-Val, families disjoint
sel = json.load(open(ROOT / "input/selected.json"))
rows = sel["train"] + sel["dev"]
splits = sorted({r["dl_split"] for r in rows})
assert set(splits) <= {"DL-Train", "DL-Val"}, splits
assert not ({r["family_id"] for r in sel["train"]} & {r["family_id"] for r in sel["dev"]})

def load(p):
    s = json.load(open(p))
    assert s["dope_before"] == s["dope_after"] == DOPE
    assert s["parity_ok"] and abs(s["parity_incumbent_val"] - INC) < 1e-9 and s["family_overlap"] == 0
    return s
S1, S3, S2h = load(R / "seed1/summary.json"), load(R / "seed3/summary.json"), load(R / "seed2_hilr/summary.json")

# seed2 partial from progress.log
cv2, folds2 = {}, {}
for line in open(R / "seed2/progress.log"):
    m = re.search(r" (\w+) 2 CV (\{.*\})$", line)
    if m: cv2[m.group(1)] = json.loads(m.group(2))
    m = re.search(r" (\w+) 2 fold (\d) gain=([+-][\d.]+)", line)
    if m: folds2.setdefault(m.group(1), []).append(float(m.group(3)))
abort2 = [l.strip() for l in open(R / "seed2/run.log") if "ABORT" in l]

per_seed = {1: dict(S1["results"][0]["cv"]), 3: dict(S3["results"][0]["cv"]), 2: dict(cv2)}
per_seed[2]["hilr_wd"] = S2h["results"][0]["cv"]["hilr_wd"]
per_seed[2]["swa_cos_partial_folds"] = folds2.get("swa_cos", [])

def chk(c):  # recompute strict from stored fields; ties are not wins
    assert c["passes_strict"] == (c["oof_gain_mean"] > MARGIN and c["oof_gain_ci95_family_boot"][0] > 0)
    w, t, l = c["wins"], c["ties"], c["losses"]
    exp = (w + 0.5 * t) / (w + t + l)
    return exp

arm_sum = {}
for arm in ("fbal_accum", "swa_cos", "hilr_wd"):
    seeds = [s for s in (1, 2, 3) if arm in per_seed[s]]
    cs = [per_seed[s][arm] for s in seeds]
    for c in cs: chk(c)
    arm_sum[arm] = {"seeds": seeds,
        "per_seed_oof_gain": [c["oof_gain_mean"] for c in cs],
        "median_oof_gain": st.median(c["oof_gain_mean"] for c in cs),
        "per_seed_ci95": [c["oof_gain_ci95_family_boot"] for c in cs],
        "per_seed_fold_gains": [c["fold_gains"] for c in cs],
        "per_seed_wlt": [[c["wins"], c["ties"], c["losses"]] for c in cs],
        "per_seed_elo_oof": [c["elo_vs_incumbent_oof"] for c in cs],
        "median_elo_oof": st.median(c["elo_vs_incumbent_oof"] for c in cs),
        "seeds_passing_strict": sum(c["passes_strict"] for c in cs),
        "median_beyond_ci_all_seeds": all(c["oof_gain_ci95_family_boot"][0] > 0 for c in cs)}
arm_sum["swa_cos"]["seed2_partial_fold_gains"] = folds2.get("swa_cos", [])

# seed independence red-team
fb = [per_seed[s]["fbal_accum"]["fold_gains"] for s in (1, 2, 3)]
ident = [k for k in range(5) if len({round(f[k], 12) for f in fb}) == 1]

reg = {1: S1["results"][0]["regression_check"], 3: S3["results"][0]["regression_check"],
       2: S2h["results"][0]["regression_check"]}
lift = any(a["seeds_passing_strict"] == len(a["seeds"]) and a["median_beyond_ci_all_seeds"] for a in arm_sum.values())
out = {"wave": "H444", "verdict": "LIFT_CANDIDATE" if lift else "NO_LIFT",
       "metric": "val/OOF utility = -fitness of argmin-predicted pool program (-2.0 if gate fails); higher is better",
       "incumbent": "round-3175 (live best, run-20261030), scratch copy", "incumbent_val_utility": S1["parity_incumbent_val"],
       "incumbent_train_utility": S1["incumbent_train_utility"], "parity_ok_all": True, "dope_before_after": DOPE,
       "trainable_params": 29211, "packet_splits": splits, "n_train": S1["n_train"], "n_train_families": S1["n_train_families"],
       "n_val": S1["n_dev"], "n_val_families": S1["n_dev_families"], "family_overlap": 0, "fold_sizes": S1["fold_sizes"],
       "strict_rule": {"margin": MARGIN, "ci_lower_gt_0": True, "ties_count_as_wins": False},
       "config": {k: v for k, v in S1["config"].items() if k not in ("out", "seeds")}, "arms_cfg": S1["arms_cfg"],
       "peak": {"seed1": S1["peak"], "seed3": S3["peak"], "seed2_hilr": S2h["peak"]},
       "seed2_abort": abort2, "arm_summary": arm_sum,
       "regression_check_per_seed": {str(k): v for k, v in reg.items()},
       "regression_check_note": "seed2 regcheck ran on hilr_wd only (relaunch had a single arm), so it is not a CV-selected config",
       "seed_independence": {"fbal_accum_fold_gains_identical_across_seeds": ident,
            "cause": "restored Adam moments (step 487, |m|=0.45 vs |sqrt v|=0.25) dominate 2 optimizer steps at lr 1e-4 "
                     "(micro 8 / accum 4); the seed only changes minibatch/slate/dropout draws, which move the 10% gradient "
                     "share, and the argmin utility is discrete, so folds 1-4 pick identical programs on every seed"}}
print(json.dumps(out, indent=1))
