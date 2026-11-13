#!/usr/bin/env python3
"""Aggregate H452 per-seed summaries. Asserts DOPE frozen before/after, parity, DL-Train/DL-Val only, family-disjoint
partitions, seed independence (per arm, fold-gain vectors differ between every pair of seeds; distinct adapters per
cell). Reports per arm: pooled (2 partitions) OOF gain/Elo with family-bootstrap CIs, non-floor Elo + CI, per-partition
Elo and fold gains, strict passes. Verdict LIFT_CANDIDATE only if the selected arm passes strict on every seed AND
every val91 regression check is not_worse AND the median val91 gain CI low > 0 across seeds is reported separately."""
import json, itertools, statistics as st, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent; R = ROOT / "results"; SD = R / "fold_sig"
DOPE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"; INC = 1.1389572403748425
S = {}; summ = {}
for d in sorted(R.glob("seed*/summary.json")):
    s = json.load(open(d)); assert s["dope_before"] == DOPE and s["dope_after"] == DOPE, d
    assert s["parity_ok"] and abs(s["parity_incumbent_val"] - INC) < 1e-9 and s["family_overlap"] == 0
    r = s["results"][0]; S[r["seed"]] = r; summ[r["seed"]] = s
sel_ = json.load(open(ROOT / "input/selected.json"))
assert all(x["dl_split"] in ("DL-Train", "DL-Val") for x in sel_["train"] + sel_["dev"])
seeds = sorted(S); arms = list(S[seeds[0]]["cv"]); parts = summ[seeds[0]]["partitions"]
indep = {}
for a in arms:
    for x, y in itertools.combinations(seeds, 2):
        fx = [pp["fold_gains"] for pp in S[x]["cv"][a]["per_partition"]]; fy = [pp["fold_gains"] for pp in S[y]["cv"][a]["per_partition"]]
        indep[f"{a}:{x}v{y}"] = fx != fy
    for pi in range(len(parts)):
        for k in range(5):
            shas = [json.load(open(SD / f"{a}_p{pi}cv{k}_seed{s}.json"))["adapter_sha"] for s in seeds if (SD / f"{a}_p{pi}cv{k}_seed{s}.json").exists()]
            assert len(set(shas)) == len(shas), f"identical adapters {a} p{pi}cv{k}"
assert all(indep.values()) or len(seeds) < 2, indep
arm_sum = {}
for a in arms:
    cs = [S[s]["cv"][a] for s in seeds]
    arm_sum[a] = {"per_seed_oof_gain": [c["oof_gain_mean"] for c in cs], "median_oof_gain": st.median(c["oof_gain_mean"] for c in cs),
                  "per_seed_gain_ci95": [c["oof_gain_ci95_family_boot"] for c in cs],
                  "per_seed_elo": [c["elo_vs_incumbent_oof"] for c in cs], "median_elo": st.median(c["elo_vs_incumbent_oof"] for c in cs),
                  "per_seed_elo_ci95": [c["elo_ci95_family_boot"] for c in cs],
                  "per_seed_nonfloor": [c["nonfloor"] for c in cs], "median_nonfloor_elo": st.median(c["nonfloor"]["elo"] for c in cs),
                  "per_seed_partition_elo": [[pp["elo"] for pp in c["per_partition"]] for c in cs],
                  "per_seed_partition_nonfloor_elo": [[pp["nonfloor_elo"] for pp in c["per_partition"]] for c in cs],
                  "per_seed_partition_fold_gains": [[pp["fold_gains"] for pp in c["per_partition"]] for c in cs],
                  "per_seed_partition_nonfloor_fold_gains": [[pp["nonfloor_fold_gains"] for pp in c["per_partition"]] for c in cs],
                  "per_seed_wlt": [[c["wins"], c["ties"], c["losses"]] for c in cs], "per_seed_gate": [c["gate"] for c in cs],
                  "seeds_passing_utility_strict": sum(c["passes_utility_strict"] for c in cs), "seeds_passing_strict": sum(c["passes_strict"] for c in cs)}
reg = {str(s): S[s]["regression_check"] for s in seeds}
full_sha = {k: v["adapter_sha"] for k, v in reg.items()}; assert len(set(full_sha.values())) == len(seeds)
sel = {str(s): S[s]["selected"] for s in seeds}
lift = all(S[s]["selected_passes_strict"] for s in seeds) and all(r["not_worse"] for r in reg.values())
vg = [r["val_gain"] for r in reg.values()]; vlo = [r["val_gain_ci95_family_boot"][0] for r in reg.values()]
print(json.dumps({"wave": "H452", "verdict": "LIFT_CANDIDATE" if lift else "NO_LIFT", "seeds": seeds, "partitions": parts,
    "metric": "utility = -fitness of argmin-predicted pool program (-2.0 on gate fail); higher is better",
    "incumbent_val_utility": INC, "incumbent_train_utility": summ[seeds[0]]["incumbent_train_utility"], "dope": DOPE,
    "n_train": summ[seeds[0]]["n_train"], "n_val": summ[seeds[0]]["n_dev"], "fold_sizes": summ[seeds[0]]["fold_sizes"],
    "arms_cfg": summ[seeds[0]]["arms_cfg"], "derived_cfg": summ[seeds[0]]["derived_cfg"],
    "strict_rule": "pooled OOF gain > 0.002 AND gain CI low > 0 AND pooled Elo > 0 AND Elo CI low > 0 AND non-floor Elo CI low > 0 AND Elo > 0 on each partition; selection = max mean non-floor OOF Elo over partitions",
    "seed_independence": indep, "selected_per_seed": sel, "arm_summary": arm_sum, "regression_check_per_seed": reg,
    "val91_median_gain": st.median(vg), "val91_all_ci_low_gt_0": all(x > 0 for x in vlo), "regcheck_adapter_sha": full_sha,
    "peak": {str(s): summ[s]["peak"] for s in seeds}}, indent=1))
