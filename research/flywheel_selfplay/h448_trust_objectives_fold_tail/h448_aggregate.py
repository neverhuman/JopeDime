#!/usr/bin/env python3
"""Aggregate H448 per-seed summaries. H448 strict = utility-strict AND Elo > 0 AND family-boot Elo CI low > 0. Asserts DOPE frozen, parity, DL-Train/DL-Val only, strict rule recomputed,
and seed independence (per-arm fold-gain vectors must differ between every pair of seeds)."""
import json, sys, statistics as st, itertools
from pathlib import Path
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/jope-data/hwave-h448"); R = ROOT / "results"
DOPE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"; INC = 1.1389572403748425; MARGIN = 0.002
sel = json.load(open(ROOT / "input/selected.json")); rows = sel["train"] + sel["dev"]
assert {r["dl_split"] for r in rows} <= {"DL-Train", "DL-Val"}
assert not ({r["family_id"] for r in sel["train"]} & {r["family_id"] for r in sel["dev"]})
S = {}
for d in sorted(R.glob("seed*/summary.json")):
    s = json.load(open(d)); assert s["dope_before"] == s["dope_after"] == DOPE
    assert s["parity_ok"] and abs(s["parity_incumbent_val"] - INC) < 1e-9
    S[s["results"][0]["seed"]] = s["results"][0]
seeds = sorted(S); arms = sorted(S[seeds[0]]["cv"])
indep = {}
for a in arms:
    for x, y in itertools.combinations(seeds, 2):
        indep[f"{a}:{x}v{y}"] = S[x]["cv"][a]["fold_gains"] != S[y]["cv"][a]["fold_gains"]
assert all(indep.values()), f"seed independence failed: {indep}"
arm_sum = {}
for a in arms:
    cs = [S[s]["cv"][a] for s in seeds]
    for c in cs:
        us = c["oof_gain_mean"] > MARGIN and c["oof_gain_ci95_family_boot"][0] > 0
        assert c["passes_utility_strict"] == us
        assert c["passes_strict"] == (us and c["elo_vs_incumbent_oof"] > 0 and c["elo_ci95_family_boot"][0] > 0)
    arm_sum[a] = {"per_seed_oof_gain": [c["oof_gain_mean"] for c in cs], "median_oof_gain": st.median(c["oof_gain_mean"] for c in cs),
                  "per_seed_ci95": [c["oof_gain_ci95_family_boot"] for c in cs], "per_seed_fold_gains": [c["fold_gains"] for c in cs],
                  "per_seed_wlt": [[c["wins"], c["ties"], c["losses"]] for c in cs],
                  "per_seed_elo_oof": [c["elo_vs_incumbent_oof"] for c in cs], "median_elo_oof": st.median(c["elo_vs_incumbent_oof"] for c in cs),
                  "per_seed_elo_ci95": [c["elo_ci95_family_boot"] for c in cs], "seeds_passing_utility_strict": sum(c["passes_utility_strict"] for c in cs),
                  "seeds_passing_strict": sum(c["passes_strict"] for c in cs),
                  "ci_lower_gt_0_all_seeds": all(c["oof_gain_ci95_family_boot"][0] > 0 for c in cs)}
reg = {str(s): S[s]["regression_check"] for s in seeds}
# decision-level diversity: how many OOF contexts get a different paired gain between seeds (H445 red-team)
dec = {}
for a in arms:
    tot = {f"{x}v{y}": 0 for x, y in itertools.combinations(seeds, 2)}; changed = {str(s): 0 for s in seeds}; nfold = 0
    while (R.parent / "fold_sig" / f"{a}_cv{nfold}_seed{seeds[0]}.json").exists() or (R / "fold_sig" / f"{a}_cv{nfold}_seed{seeds[0]}.json").exists():
        sd = (R / "fold_sig") if (R / "fold_sig").exists() else (R.parent / "fold_sig")
        o = {s: json.load(open(sd / f"{a}_cv{nfold}_seed{s}.json")) for s in seeds}
        assert len({o[s]["adapter_sha"] for s in seeds}) == len(seeds), f"identical adapters {a} cv{nfold}"
        for x, y in itertools.combinations(seeds, 2):
            tot[f"{x}v{y}"] += sum(1 for p, q in zip(o[x]["oof"], o[y]["oof"]) if p != q)
        for s in seeds: changed[str(s)] += sum(1 for p in o[s]["oof"] if p != 0)
        nfold += 1
    tot_min = min(tot.values()) if tot else 0
    dec[a] = {"seeds_decision_coupled": tot_min == 0, "min_pair_disagreement": tot_min, "folds": nfold, "contexts_changed_vs_incumbent": changed, "contexts_differing_between_seeds": tot}
full_sha = {str(s): reg[str(s)].get("adapter_sha") for s in seeds}
assert len(set(full_sha.values())) == len(seeds), f"regcheck adapters identical: {full_sha}"
lift = any(v["seeds_passing_strict"] == len(seeds) for v in arm_sum.values()) and all(r["not_worse"] for r in reg.values())
s0 = json.load(open(R / f"seed{seeds[0]}/summary.json"))
print(json.dumps({"wave": "H448", "verdict": "LIFT_CANDIDATE" if lift else "NO_LIFT", "seeds": seeds,
    "metric": "utility = -fitness of argmin-predicted pool program (-2.0 on gate fail); higher is better",
    "incumbent_val_utility": INC, "incumbent_train_utility": s0["incumbent_train_utility"], "dope": DOPE, "trainable_params": 29211,
    "n_train": s0["n_train"], "n_val": s0["n_dev"], "fold_sizes": s0["fold_sizes"], "arms_cfg": s0["arms_cfg"],
    "strict_rule": {"margin": MARGIN, "ci_lower_gt_0": True, "elo_gt_0": True, "elo_ci_lower_gt_0": True, "ties_count_as_wins": False, "selection": "max OOF Elo per seed"},
    "seed_independence": indep, "decision_diversity": dec, "regcheck_adapter_sha": full_sha, "arm_summary": arm_sum, "selected_per_seed": {str(s): S[s]["selected"] for s in seeds},
    "regression_check_per_seed": reg, "peak": {str(s): json.load(open(R / f"seed{s}/summary.json"))["peak"] for s in seeds}}, indent=1))
