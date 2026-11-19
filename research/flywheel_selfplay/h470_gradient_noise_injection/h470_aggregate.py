#!/usr/bin/env python3
"""Aggregate H470 per-seed summaries (copied from h469_aggregate.py) (adapted from h452_aggregate.py). Asserts DOPE frozen before/after, parity,
DL-Train/DL-Val only, family-disjoint partitions, seed independence (fold-gain vectors differ between every seed pair,
distinct adapters per cell; an ens arm's adapter_sha hashes all 3 members). Reports per arm: pooled (2 partitions) OOF
gain/Elo with family-bootstrap CIs, non-floor Elo + CI, strict passes, and a PAIRED comparison against the control arm
(pairnf_z40) on the same contexts: arm-minus-control gain with family-bootstrap CI and head-to-head Elo (contexts where
the two differ). Verdict LIFT_CANDIDATE only if the selected arm passes strict on every seed, every val91 regression check
is not_worse, and the selected arm beats the control (paired gain CI low > 0) on every seed."""
import json, itertools, statistics as st, sys, os
from pathlib import Path
import numpy as np
ROOT = Path(os.environ.get("WAVE_ROOT", Path(__file__).resolve().parent)); R = ROOT / "results"; SD = R / "fold_sig"
WAVE = os.environ.get("WAVE", "H470"); CTRL = os.environ.get("CTRL", "pairnf_z40")
DOPE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"; INC = 1.1389572403748425
S = {}; summ = {}
for d in sorted(R.glob("seed*/summary.json")):
    s = json.load(open(d)); assert s["dope_before"] == DOPE and s["dope_after"] == DOPE, d
    assert s["parity_ok"] and abs(s["parity_incumbent_val"] - INC) < 1e-9 and s["family_overlap"] == 0
    r = s["results"][0]; S[r["seed"]] = r; summ[r["seed"]] = s
sel_ = json.load(open(ROOT / "input/selected.json"))
assert all(x["dl_split"] in ("DL-Train", "DL-Val") for x in sel_["train"] + sel_["dev"])
AN = json.load(open(ROOT / "inc_analysis.json")); train = AN["train"]; fam = np.array(AN["family"]); fail = np.array(AN["inc_fail_train"], bool)
FOLD = [np.array(f) for f in AN["folds"]]
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
# H454: an arm whose gate keeps 0 changes everywhere has all-zero fold gains on every seed (identical by construction);
# exempt it from the fold-gain independence check (adapter shas are still asserted distinct above).
degen = {a: all(all(g == 0 for pp in S[s]["cv"][a]["per_partition"] for g in pp["fold_gains"]) for s in seeds) for a in arms}
# H467: an arm whose gate keeps <= 5 of 2,438 OOF decisions on every seed (look4_z40: 1 W / 0 L) is near-degenerate:
# its OOF vectors are legitimately seed-identical, so it is also exempt (reported under degenerate_all_zero_arms).
degen = {a: degen[a] or all(S[s]["cv"][a]["wins"] + S[s]["cv"][a]["losses"] <= 5 for s in seeds) for a in arms}
assert all(v or degen[k.split(":")[0]] for k, v in indep.items()) or len(seeds) < 2, indep
def oof(arm, sd):
    xs = []
    for pi in range(len(parts)):
        x = np.full(len(train), np.nan)
        for k in range(5):
            j = json.load(open(SD / f"{arm}_p{pi}cv{k}_seed{sd}.json")); assert j["partition"] == parts[pi] and j["seed"] == sd
            x[FOLD[pi] == k] = j["oof"]
        assert not np.isnan(x).any(); xs.append(x)
    return np.concatenate(xs)
def elo(p):
    p = min(max(p, 1e-4), 1 - 1e-4); return float(400 * np.log10(p / (1 - p)))
ufam = sorted(set(fam)); n = len(train)
fidx = [np.concatenate([j * n + np.where(fam == f)[0] for j in range(len(parts))]) for f in ufam]
def paired(d, seed):
    r = np.random.default_rng(seed); gs = []; es = []
    for _ in range(2000):
        ix = np.concatenate([fidx[q] for q in r.integers(0, len(fidx), len(fidx))]); x = d[ix]; gs.append(x.mean())
        nz = x[x != 0]; es.append(elo(((nz > 0).sum() + 0.0) / len(nz)) if len(nz) else 0.0)
    nz = d[d != 0]
    return {"gain_vs_ctrl": float(d.mean()), "gain_vs_ctrl_ci95": np.percentile(gs, [2.5, 97.5]).tolist(),
            "n_differ": int(len(nz)), "arm_better": int((nz > 0).sum()), "ctrl_better": int((nz < 0).sum()),
            "h2h_elo_on_differing": elo((nz > 0).sum() / len(nz)) if len(nz) else 0.0,
            "h2h_elo_ci95": np.percentile(es, [2.5, 97.5]).tolist()}
arm_sum = {}
for a in arms:
    cs = [S[s]["cv"][a] for s in seeds]
    vs = {str(s): paired(oof(a, s) - oof(CTRL, s), 100 + s) for s in seeds} if a != CTRL else None
    arm_sum[a] = {"per_seed_oof_gain": [c["oof_gain_mean"] for c in cs], "median_oof_gain": st.median(c["oof_gain_mean"] for c in cs),
                  "per_seed_gain_ci95": [c["oof_gain_ci95_family_boot"] for c in cs],
                  "per_seed_elo": [c["elo_vs_incumbent_oof"] for c in cs], "median_elo": st.median(c["elo_vs_incumbent_oof"] for c in cs),
                  "per_seed_elo_ci95": [c["elo_ci95_family_boot"] for c in cs],
                  "per_seed_nonfloor_elo": [c["nonfloor"]["elo"] for c in cs], "per_seed_nonfloor_elo_ci95": [c["nonfloor"]["elo_ci95_family_boot"] for c in cs],
                  "median_nonfloor_elo": st.median(c["nonfloor"]["elo"] for c in cs),
                  "per_seed_mean_partition_nonfloor_elo": [c["mean_partition_nonfloor_elo"] for c in cs],
                  "per_seed_partition_elo": [[pp["elo"] for pp in c["per_partition"]] for c in cs],
                  "per_seed_partition_fold_gains": [[pp["fold_gains"] for pp in c["per_partition"]] for c in cs],
                  "per_seed_wlt": [[c["wins"], c["ties"], c["losses"]] for c in cs], "per_seed_gate": [c["gate"] for c in cs],
                  "seeds_passing_utility_strict": sum(c["passes_utility_strict"] for c in cs), "seeds_passing_strict": sum(c["passes_strict"] for c in cs),
                  "vs_control_per_seed": vs,
                  "beats_control_all_seeds": bool(vs and all(v["gain_vs_ctrl_ci95"][0] > 0 for v in vs.values()))}
reg = {str(s): S[s]["regression_check"] for s in seeds}
full_sha = {k: v["adapter_sha"] for k, v in reg.items()}; assert len(set(full_sha.values())) == len(seeds)
sel = {str(s): S[s]["selected"] for s in seeds}
beats = all(S[s]["selected"] != CTRL and arm_sum[S[s]["selected"]]["beats_control_all_seeds"] for s in seeds)
lift = all(S[s]["selected_passes_strict"] for s in seeds) and all(r["not_worse"] for r in reg.values()) and beats
vg = [r["val_gain"] for r in reg.values()]; vlo = [r["val_gain_ci95_family_boot"][0] for r in reg.values()]
json.dump({"wave": WAVE, "verdict": "LIFT_CANDIDATE" if lift else "NO_LIFT", "control": CTRL, "seeds": seeds, "partitions": parts,
    "metric": "utility = -fitness of argmin-predicted pool program (-2.0 on gate fail); higher is better",
    "incumbent_val_utility": INC, "incumbent_train_utility": summ[seeds[0]]["incumbent_train_utility"], "dope": DOPE,
    "n_train": summ[seeds[0]]["n_train"], "n_val": summ[seeds[0]]["n_dev"], "fold_sizes": summ[seeds[0]]["fold_sizes"],
    "arms_cfg": summ[seeds[0]]["arms_cfg"], "derived_cfg": summ[seeds[0]]["derived_cfg"],
    "strict_rule": "pooled OOF gain > 0.002 AND gain CI low > 0 AND pooled Elo > 0 AND Elo CI low > 0 AND non-floor Elo CI low > 0 AND Elo > 0 on each partition; selection = max mean non-floor OOF Elo over partitions; counts only if it also beats the control (paired family-bootstrap gain CI low > 0)",
    "seed_independence": indep, "degenerate_all_zero_arms": [a for a in arms if degen[a]], "selected_per_seed": sel, "arm_summary": arm_sum, "regression_check_per_seed": reg,
    "val91_median_gain": st.median(vg), "val91_median_elo": st.median(r["elo_vs_incumbent_val"] for r in reg.values()),
    "val91_all_ci_low_gt_0": all(x > 0 for x in vlo), "regcheck_adapter_sha": full_sha,
    "dope_after_all_seeds": {str(s): summ[s]["dope_after"][:8] for s in seeds},
    "peak": {str(s): summ[s]["peak"] for s in seeds}}, open(sys.argv[1] if len(sys.argv) > 1 else "/dev/stdout", "w"), indent=1)

# H470 addition: cross-seed family bootstrap. Per arm, the per-context OOF gain is averaged over the 3 seeds (both partition
# copies), then families are resampled with all their rows (2,000 draws); same for arm - control and for the pooled val91
# diffs of the per-seed selected arms (19 DL-Val families). Elo = 400 log10(s/(1-s)), s = (W + T/2)/N over seed-pooled rows.
def xboot(vecs, fidx_, seed, B=2000):
    r = np.random.default_rng(seed); M = np.mean(vecs, 0); P_ = np.concatenate(vecs); nn = len(M); gs = []; es = []
    for _ in range(B):
        ix = np.concatenate([fidx_[q] for q in r.integers(0, len(fidx_), len(fidx_))]); gs.append(M[ix].mean())
        px = np.concatenate([v[ix] for v in vecs]); es.append(elo(((px > 0).sum() + 0.5 * (px == 0).sum()) / len(px)))
    return {"gain_seedmean": float(M.mean()), "gain_ci95": np.percentile(gs, [2.5, 97.5]).tolist(),
            "elo_pooled": elo(((P_ > 0).sum() + 0.5 * (P_ == 0).sum()) / len(P_)), "elo_ci95": np.percentile(es, [2.5, 97.5]).tolist(),
            "W_T_L_pooled": [int((P_ > 0).sum()), int((P_ == 0).sum()), int((P_ < 0).sum())]}
cross = {}
for a in arms:
    cross[a] = {"oof_vs_incumbent": xboot([oof(a, s) for s in seeds], fidx, 7)}
    if a != CTRL: cross[a]["oof_vs_control"] = xboot([oof(a, s) - oof(CTRL, s) for s in seeds], fidx, 8)
dfam = [x["family_id"] for x in sel_["dev"]]; udf = sorted(set(dfam)); dfidx = [np.where(np.array(dfam) == f)[0] for f in udf]
VD = [np.array(json.load(open(SD / f"{S[s]['selected']}_full_seed{s}.json"))["val_diff"]) for s in seeds]
val_cross = xboot(VD, dfidx, 9) | {"median_val_gain": st.median(vg), "per_seed": {str(s): [reg[str(s)]["arm"], reg[str(s)]["val_gain"], reg[str(s)]["val_gain_ci95_family_boot"]] for s in seeds}}
val_cross["median_beyond_ci"] = bool(val_cross["median_val_gain"] > val_cross["gain_ci95"][1] or val_cross["median_val_gain"] < val_cross["gain_ci95"][0])
val_cross["median_ci_low_gt_0"] = bool(val_cross["gain_ci95"][0] > 0)
O_ = json.load(open(sys.argv[1])); O_["cross_seed_family_boot"] = cross; O_["val91_cross_seed"] = val_cross
O_["h470_task_verdict_rule"] = "LIFT only if the median val91 gain is beyond 0 with the cross-seed family-bootstrap CI low > 0 AND the pre-registered LIFT rule holds; else NO_LIFT"
O_["h470_task_verdict"] = "LIFT" if (lift and val_cross["median_ci_low_gt_0"]) else "NO_LIFT"
json.dump(O_, open(sys.argv[1], "w"), indent=1)
