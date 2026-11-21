#!/usr/bin/env python3
"""H473 close-out supplement: OOF gain with and without -2.0 floor recoveries (incumbent gate-fail contexts), per arm and
seed and cross-seed, with 205-family bootstrap CIs (2,000 draws, families resampled with all rows in both partition
copies), plus the paired arm-minus-control non-floor gain. Reads only results/fold_sig and inc_analysis.json."""
import json, os, statistics as st, sys
from pathlib import Path
import numpy as np
ROOT = Path(os.environ.get("WAVE_ROOT", ".")); SD = ROOT / "results/fold_sig"
AN = json.load(open(ROOT / "inc_analysis.json")); fam = np.array(AN["family"]); fail = np.array(AN["inc_fail_train"], bool)
FOLD = [np.array(f) for f in AN["folds"]]; N = len(fam); ARMS = ["pairnf_z40", "pgap_z40", "pm25_z40"]; SEEDS = (1, 2, 3)
def oof(a, s):
    xs = []
    for pi in range(2):
        x = np.full(N, np.nan)
        for k in range(5): x[FOLD[pi] == k] = json.load(open(SD / f"{a}_p{pi}cv{k}_seed{s}.json"))["oof"]
        assert not np.isnan(x).any(); xs.append(x)
    return np.concatenate(xs)
FL = np.concatenate([fail, fail]); ufam = sorted(set(fam))
fidx = [np.concatenate([j * N + np.where(fam == f)[0] for j in range(2)]) for f in ufam]
rng = np.random.default_rng(473); DRAWS = [rng.integers(0, len(ufam), len(ufam)) for _ in range(2000)]
def boot(v):
    st_ = []
    for d in DRAWS:
        ix = np.concatenate([fidx[i] for i in d]); st_.append(v[ix].mean())
    return [float(np.percentile(st_, 2.5)), float(np.percentile(st_, 97.5))]
G = {a: {s: oof(a, s) for s in SEEDS} for a in ARMS}; out = {"n_rows": 2 * N, "floor_rows": int(FL.sum()), "arms": {}}
for a in ARMS:
    r = {"per_seed": {}}
    for s in SEEDS:
        g = G[a][s]; nf = np.where(FL, 0.0, g); c = G["pairnf_z40"][s]; ncf = np.where(FL, 0.0, c)
        r["per_seed"][s] = {"gain": float(g.mean()), "gain_ci95": boot(g), "floor_part": float(g[FL].sum() / len(g)),
            "nonfloor_gain": float(nf.mean()), "nonfloor_gain_ci95": boot(nf), "floor_recoveries": int(((g > 0) & FL).sum()),
            "nonfloor_vs_control": None if a == "pairnf_z40" else {"diff": float((nf - ncf).mean()), "ci95": boot(nf - ncf)}}
    r["median_gain"] = st.median(r["per_seed"][s]["gain"] for s in SEEDS)
    r["median_nonfloor_gain"] = st.median(r["per_seed"][s]["nonfloor_gain"] for s in SEEDS)
    m = np.mean([G[a][s] for s in SEEDS], 0); mn = np.where(FL, 0.0, m)
    r["cross_seed"] = {"gain": float(m.mean()), "gain_ci95": boot(m), "nonfloor_gain": float(mn.mean()), "nonfloor_gain_ci95": boot(mn)}
    if a != "pairnf_z40":
        mc = np.where(FL, 0.0, np.mean([G["pairnf_z40"][s] for s in SEEDS], 0))
        r["cross_seed"]["nonfloor_vs_control"] = {"diff": float((mn - mc).mean()), "ci95": boot(mn - mc)}
    out["arms"][a] = r
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "nonfloor.json", "w"), indent=1)
for a, r in out["arms"].items():
    print(a, "median gain %.5f nonfloor %.5f | cross-seed gain %.5f %s nonfloor %.5f %s" % (r["median_gain"], r["median_nonfloor_gain"],
          r["cross_seed"]["gain"], np.round(r["cross_seed"]["gain_ci95"], 5), r["cross_seed"]["nonfloor_gain"], np.round(r["cross_seed"]["nonfloor_gain_ci95"], 5)),
          r["cross_seed"].get("nonfloor_vs_control"))
    for s, p in r["per_seed"].items(): print("   s%d gain %.5f %s floors %.5f (rec %d) nonfloor %.5f %s vsctl %s" % (s, p["gain"], np.round(p["gain_ci95"], 5), p["floor_part"], p["floor_recoveries"], p["nonfloor_gain"], np.round(p["nonfloor_gain_ci95"], 5), p["nonfloor_vs_control"]))
