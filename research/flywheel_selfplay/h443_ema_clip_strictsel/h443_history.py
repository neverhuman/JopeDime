import json, glob, re, numpy as np, collections, os
R = "/home/ubuntu/jope-data/flywheel-full-20261030/run-20261030"  # read-only
sel = json.load(open("/home/ubuntu/jope-data/hwave-h443/input/selected.json"))
fam = np.array([r["family_id"] for r in sel["dev"]]); uf = sorted(set(fam)); fidx = [np.where(fam == f)[0] for f in uf]
def boot(d, seed):
    r = np.random.default_rng(seed); ms = []
    for _ in range(2000):
        ix = np.concatenate([fidx[j] for j in r.integers(0, len(fidx), len(fidx))]); ms.append(d[ix].mean())
    return np.percentile(ms, 2.5), np.percentile(ms, 97.5)
files = sorted(glob.glob(R + "/round-*.json"), key=lambda p: int(re.search(r"round-(\d+)\.json", p)[1]))
rows = []
for p in files:
    try: d = json.load(open(p))
    except Exception: continue
    c = d.get("candidate_dev_utility"); b = d.get("initial_dev_utility"); adm = d.get("adapter_admission") or {}
    if not c or not b or "per_context" not in c or len(c["per_context"]) != 91: continue
    rows.append((d["round"], np.array(c["per_context"]), np.array(b["per_context"]), bool(adm.get("accepted")), adm.get("previous_best_utility"), c["mean"]))
acc = [r for r in rows if r[3]]
out = {"rounds_parsed": len(rows), "first_round": rows[0][0], "last_round": rows[-1][0], "accepts": []}
for rnd, cu, bu, a, prev, m in acc:
    d = cu - bu; lo, hi = boot(d, rnd)
    out["accepts"].append({"round": rnd, "cand_mean": float(cu.mean()), "prev_best": prev, "gain": float(d.mean()), "ci95": [float(lo), float(hi)],
        "tie": bool(prev is not None and abs(cu.mean() - prev) < 1e-15), "n_ctx_changed": int((d != 0).sum()),
        "max_abs_ctx_delta": float(np.abs(d).max()), "dominant_ctx_share": float(np.abs(d).max() / max(np.abs(d).sum(), 1e-12)),
        "fail_ctx_flips": int(((bu == -2.0) != (cu == -2.0)).sum()), "strict_pass": bool(d.mean() > 0.002 and lo > 0)})
means = [round(r[5], 12) for r in rows[-400:]]
cnt = collections.Counter(means)
out["last400_distinct_candidate_means"] = len(cnt)
out["last400_top_candidate_means"] = cnt.most_common(6)
# lag autocorrelation of candidate-mean repeats (cycling)
lagrep = {L: float(np.mean([means[i] == means[i - L] for i in range(L, len(means))])) for L in range(1, 9)}
out["last400_repeat_rate_by_lag"] = lagrep
out["rejections_since_last_accept"] = len([r for r in rows if r[0] > max([x[0] for x in acc] or [0])])
inc = np.array(json.load(open(R + "/best_dev_checkpoint.json"))["utility"]["per_context"])
out["incumbent_fail_ctx"] = int((inc == -2.0).sum())
out["incumbent_mean"] = float(inc.mean()); out["incumbent_median"] = float(np.median(inc))
out["incumbent_mean_excl_fail"] = float(inc[inc != -2.0].mean())
os.makedirs("/home/ubuntu/jope-data/hwave-h443/history", exist_ok=True)
json.dump(out, open("/home/ubuntu/jope-data/hwave-h443/history/history_replay.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "accepts"}, indent=1)); print(json.dumps(out["accepts"], indent=0)[:4000])
