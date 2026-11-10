"""H319 evaluation.

Regression first (phase regress): the archived H315 BASE picks (S3 H315/picks.npz, exact version) must equal the archived
H303 (P0) / H308 (P1, P2) ENS_VOTE2_NESTED picks on 9/9 (partition, seed) cells, and their medians (H303 judge) must
equal the published ones; else REGRESSION_FAIL and no BAG fit.

Decision (pre-registered; ties never advance). BAG_ADVANCE iff
  (a) median over the 9 cells of the paired (BAG - BASE) drop-top-10 gain > 0 AND its paired whole-family bootstrap
      LCB95 > 0 (10,000 draws, RNG 319191; same family resample for both arms; statistic = median over cells);
  (b) BAG median (over cells) family-equal gain >= 0.9 x BASE's;
  (c) BAG median harmed share <= BASE's;
  (d) BAG median seed-Jaccard > BASE's, where seed-Jaccard = for each (partition, outer fold) the mean pairwise
      Jaccard index of the switched-context sets of the 3 root seeds (12 values per arm).
Else BAG_NOT_ADVANCED."""

import hashlib
import json
import os
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ELO_LABEL = "reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN"
PAIRED_SEED = 319191
OBJ = {
    "H315": ("experiments/rd_breakthrough_20261030/H315/picks.npz", "jCll6z9FGwmpNg.cocb305LcVzFYuieR",
             "e9e6b16a7381ff14191d2a4d31ce31d4b623c39297e7383ee6aeb871e08ca615"),
    "H303": ("experiments/rd_breakthrough_20261030/H303/picks.npz", "3CUeOo9dmsTM1rZyuAeS9pF7H39Vv80N",
             "0f8435010390c5bff8d6c95a7650163a2ab4c5952e56264fbee96204784d5ea4"),
    "H308": ("experiments/rd_breakthrough_20261030/H308/picks.npz", "uXsQkA27JSci7XVbEFvbyJvfvzmO3nP0",
             "ac16ccdfd715d000a249dfea87d707159329f189a43475452af038a03517290f"),
}  # fmt: skip
ARCH_NAME = {
    "P0": ("H303", "ENS_VOTE2_NESTED"),
    "P1": ("H308", "ENS_VOTE2_NESTED_P1"),
    "P2": ("H308", "ENS_VOTE2_NESTED_P2"),
}
PUBLISHED_MEDIAN = {"P0": 0.0643163334285782, "P1": 0.04821480030677507, "P2": 0.04360142642936269}


def fetch(name, DATA):
    key, ver, dig = OBJ[name]
    dest = Path(DATA) / "archived" / f"{name}_picks.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        subprocess.run(["aws", "s3api", "get-object", "--bucket", "veox-jopedime-learning-use2", "--key", key, "--version-id", ver, str(dest)],
                       check=True, stdout=subprocess.DEVNULL, env=dict(os.environ, AWS_PROFILE="veox-jepson-prod"))  # fmt: skip
    if hashlib.sha256(dest.read_bytes()).hexdigest() != dig:
        raise RuntimeError("archived picks sha mismatch: " + name)
    with np.load(dest) as z:
        return {k: np.asarray(z[k], dtype=np.int32) for k in z.files}


def base_picks(DATA):
    h315 = fetch("H315", DATA)
    return {p: h315[f"BASE_{p}"] for p in ("P0", "P1", "P2")}


def regress(r303, DATA):
    import evaluate_h303 as ev

    G = r303.G
    fams = [r["family_id"] for r in G["train_rows"]]
    base = base_picks(DATA)
    arch = {n: fetch(n, DATA) for n in ("H303", "H308")}
    rec = {}
    for p, (src, name) in ARCH_NAME.items():
        eq = [bool(np.array_equal(base[p][s], arch[src][name][s])) for s in range(3)]
        u, inv = ev.panel(G, base[p])
        med = ev.judge(G["a302"], u, inv, fams, base[p])["summary"]["median_gain_across_seeds"]
        rec[p] = dict(per_seed_equal=eq, median=med, median_equals_published=med == PUBLISHED_MEDIAN[p])
    ok = all(all(v["per_seed_equal"]) and v["median_equals_published"] for v in rec.values())
    out = dict(wave="H319", step="regression_first", status="PASS" if ok else "REGRESSION_FAIL", cells=rec)
    (Path(DATA) / "regression_receipt.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out), flush=True)
    if not ok:
        sys.exit("REGRESSION_FAIL")


def family_d(u, families):
    fams = sorted(set(families))
    f = np.asarray(families)
    return np.stack(
        [np.array([u[s, f == x, 1].mean() - u[s, f == x, 0].mean() for x in fams]) for s in range(u.shape[0])]
    )


def lofo_min(d):
    """Leave-one-family-out: the minimum over families of the family-equal mean without that family."""
    n = len(d)
    return float(((d.sum() - d) / (n - 1)).min())


def jaccard(a, b):
    a, b = set(a), set(b)
    return 1.0 if not (a | b) else len(a & b) / len(a | b)


def seed_jaccard(picks, fold_of_ctx):
    """Per outer fold: mean pairwise Jaccard of the switched-context index sets of the 3 seeds."""
    out = []
    for k in range(4):
        idx = np.where(np.asarray(fold_of_ctx) == k)[0]
        sets = [set(idx[picks[s, idx] != 0].tolist()) for s in range(picks.shape[0])]
        out.append(float(np.mean([jaccard(x, y) for x, y in combinations(sets, 2)])))
    return out


def decide(paired, gain_bag, gain_base, harm_bag, harm_base, jac_bag, jac_base):
    crit = dict(
        a_paired_drop_top10_median_gt_0_and_lcb95_gt_0=bool(paired["median"] > 0 and paired["lcb95"] > 0),
        b_bag_median_gain_ge_90pct_base=bool(np.median(gain_bag) >= 0.9 * np.median(gain_base)),
        c_bag_median_harmed_le_base=bool(np.median(harm_bag) <= np.median(harm_base)),
        d_bag_median_seed_jaccard_gt_base=bool(np.median(jac_bag) > np.median(jac_base)),
    )
    return crit, ("BAG_ADVANCE" if all(crit.values()) else "BAG_NOT_ADVANCED")


def main(r303, DATA, scored):
    import evaluate_h303 as ev

    sys.path.insert(0, str(HERE.parent / "h315_margin_gated_ens_vote2_robust"))
    import evaluate_h315 as e315
    import run_h319 as R

    G = r303.G
    a302 = G["a302"]
    fams = [r["family_id"] for r in G["train_rows"]]
    reg = json.loads((Path(DATA) / "regression_receipt.json").read_text())
    assert reg["status"] == "PASS"
    base = base_picks(DATA)
    res, D = {}, {"BASE": [], "BAG": []}
    cells = {m: {"BASE": [], "BAG": []} for m in ("gain", "harmed", "lofo_min")}
    jac = {"BASE": [], "BAG": []}
    for p in R.PARTS:
        fold_of_ctx = [R.outer_fold(p, r["family_id"], a302) for r in G["train_rows"]]
        res[p] = {}
        for arm, picks in (("BASE", base[p]), ("BAG", scored[p][0])):
            u, inv = ev.panel(G, picks)
            j = ev.judge(a302, u, inv, fams, picks)
            j["summary"] = {k: v for k, v in j["summary"].items() if k != "harmed_families_by_seed"}
            Dp = family_d(u, fams)
            j["lofo_min_by_seed"] = [lofo_min(d) for d in Dp]
            j["seed_jaccard_by_fold"] = seed_jaccard(picks, fold_of_ctx)
            res[p][arm] = j
            D[arm].extend(Dp)
            cells["gain"][arm].extend(j["summary"]["per_seed_family_mean_gain"])
            cells["harmed"][arm].extend(j["summary"]["harmed_share_by_seed"])
            cells["lofo_min"][arm].extend(j["lofo_min_by_seed"])
            jac[arm].extend(j["seed_jaccard_by_fold"])
    paired = e315.paired_drop_top(np.array(D["BAG"]), np.array(D["BASE"]), seed=PAIRED_SEED)
    crit, verdict = decide(
        paired,
        cells["gain"]["BAG"],
        cells["gain"]["BASE"],
        cells["harmed"]["BAG"],
        cells["harmed"]["BASE"],
        jac["BAG"],
        jac["BASE"],
    )
    full = dict(wave="H319", verdict=verdict, criteria=crit, elo_label=ELO_LABEL, regression=reg, paired_drop_top10=paired,
                cell_medians={m: {a: float(np.median(v)) for a, v in d.items()} for m, d in cells.items()},
                seed_jaccard=dict(values=jac, median={a: float(np.median(v)) for a, v in jac.items()}),
                cells=cells, partitions=res, fit_lock_sha256={p: scored[p][1] for p in scored})  # fmt: skip
    out = Path(DATA) / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics_full.json").write_text(json.dumps(full, indent=1, allow_nan=False) + "\n")
    np.savez_compressed(out / "picks.npz", **{f"BAG_{p}": scored[p][0] for p in scored})
    print(
        json.dumps(
            dict(
                verdict=verdict,
                criteria=crit,
                paired=paired,
                cell_medians=full["cell_medians"],
                jaccard=full["seed_jaccard"]["median"],
            )
        ),
        flush=True,
    )
