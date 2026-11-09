"""H320 evaluation (helpers imported from H319/H315, sha-pinned in manifest.json).

Regression first (phase regress): archived H315 BASE picks (S3, exact version) == archived H303 (P0) / H308 (P1, P2)
ENS_VOTE2_NESTED picks on 9/9 cells and their H303-judge medians == published; else REGRESSION_FAIL, no fit.
In-run check (score phase): the in-run BASE outer picks must also equal the archived BASE picks on 9/9 cells.

Decision (pre-registered; ties never advance). TRIM_ADVANCE iff
  (a) median over the 9 cells of the paired (TRIM - BASE) drop-top-10 gain (raw utility; each arm drops its own top 10)
      > 0 AND its paired whole-family bootstrap LCB95 > 0 (10,000 draws, RNG 320191, statistic = median over cells);
  (b) TRIM median (over cells) family-equal gain >= 0.9 x BASE's;   (c) TRIM median harmed share <= BASE's.
Else TRIM_NOT_ADVANCED. CONFIG_INSENSITIVE is reported alongside if TRIM's (TAU,k,a) == BASE's on >= 30/36 folds."""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ELO_LABEL = "reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN"
PAIRED_SEED = 320191
PARTS = ("P0", "P1", "P2")


def _helpers():
    sys.path.insert(0, str(HERE.parent / "h319_bagged_ens_vote2_members"))
    sys.path.insert(1, str(HERE.parent / "h315_margin_gated_ens_vote2_robust"))
    import evaluate_h315 as e315
    import evaluate_h319 as e319

    return e319, e315


def regress(r303, DATA):
    import evaluate_h303 as ev

    e319, _ = _helpers()
    G = r303.G
    fams = [r["family_id"] for r in G["train_rows"]]
    base = e319.base_picks(DATA)
    arch = {n: e319.fetch(n, DATA) for n in ("H303", "H308")}
    rec = {}
    for p, (src, name) in e319.ARCH_NAME.items():
        eq = [bool(np.array_equal(base[p][s], arch[src][name][s])) for s in range(3)]
        u, inv = ev.panel(G, base[p])
        med = ev.judge(G["a302"], u, inv, fams, base[p])["summary"]["median_gain_across_seeds"]
        rec[p] = dict(per_seed_equal=eq, median=med, median_equals_published=med == e319.PUBLISHED_MEDIAN[p])
    ok = all(all(v["per_seed_equal"]) and v["median_equals_published"] for v in rec.values())
    out = dict(wave="H320", step="regression_first", status="PASS" if ok else "REGRESSION_FAIL", cells=rec)
    (Path(DATA) / "regression_receipt.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out), flush=True)
    if not ok:
        sys.exit("REGRESSION_FAIL")


def decide(paired, gain_t, gain_b, harm_t, harm_b):
    crit = dict(
        a_paired_drop_top10_median_gt_0_and_lcb95_gt_0=bool(paired["median"] > 0 and paired["lcb95"] > 0),
        b_trim_median_gain_ge_90pct_base=bool(np.median(gain_t) >= 0.9 * np.median(gain_b)),
        c_trim_median_harmed_le_base=bool(np.median(harm_t) <= np.median(harm_b)),
    )
    return crit, ("TRIM_ADVANCE" if all(crit.values()) else "TRIM_NOT_ADVANCED")


def main(r303, DATA, scored):
    import evaluate_h303 as ev
    import run_h320 as R

    e319, e315 = _helpers()
    G = r303.G
    a302 = G["a302"]
    fams = [r["family_id"] for r in G["train_rows"]]
    assert json.loads((Path(DATA) / "regression_receipt.json").read_text())["status"] == "PASS"
    archived = e319.base_picks(DATA)
    inrun = {p: [bool(np.array_equal(scored[p][0]["BASE"][s], archived[p][s])) for s in range(3)] for p in PARTS}
    out = Path(DATA) / "out"
    out.mkdir(parents=True, exist_ok=True)
    if not all(all(v) for v in inrun.values()):
        (out / "REGRESSION_FAIL_INRUN.json").write_text(json.dumps(inrun))
        sys.exit("REGRESSION_FAIL: in-run BASE differs from archived BASE")
    res, D = {}, {"BASE": [], "TRIM": []}
    cells = {m: {"BASE": [], "TRIM": []} for m in ("gain", "harmed", "lofo_min")}
    jac = {"BASE": [], "TRIM": []}
    for p in PARTS:
        fold_of_ctx = [R.outer_fold(p, r["family_id"], a302) for r in G["train_rows"]]
        res[p] = {}
        for arm in ("BASE", "TRIM"):
            picks = scored[p][0][arm]
            u, inv = ev.panel(G, picks)
            j = ev.judge(a302, u, inv, fams, picks)
            j["summary"] = {k: v for k, v in j["summary"].items() if k != "harmed_families_by_seed"}
            Dp = e319.family_d(u, fams)
            j["lofo_min_by_seed"] = [e319.lofo_min(d) for d in Dp]
            j["seed_jaccard_by_fold"] = e319.seed_jaccard(picks, fold_of_ctx)
            res[p][arm] = j
            D[arm].extend(Dp)
            cells["gain"][arm].extend(j["summary"]["per_seed_family_mean_gain"])
            cells["harmed"][arm].extend(j["summary"]["harmed_share_by_seed"])
            cells["lofo_min"][arm].extend(j["lofo_min_by_seed"])
            jac[arm].extend(j["seed_jaccard_by_fold"])
    paired = e315.paired_drop_top(np.array(D["TRIM"]), np.array(D["BASE"]), seed=PAIRED_SEED)
    crit, verdict = decide(
        paired, cells["gain"]["TRIM"], cells["gain"]["BASE"], cells["harmed"]["TRIM"], cells["harmed"]["BASE"]
    )
    n_same = int(sum(sum(scored[p][2]) for p in PARTS))
    picks_agree = {
        p: [float(np.mean(scored[p][0]["TRIM"][s] == scored[p][0]["BASE"][s])) for s in range(3)] for p in PARTS
    }
    full = dict(wave="H320", verdict=verdict, config_insensitive=bool(n_same >= 30), configs_same_as_base=f"{n_same}/36",
                criteria=crit, elo_label=ELO_LABEL, inrun_base_equals_archived=inrun, paired_drop_top10=paired,
                cell_medians={m: {a: float(np.median(v)) for a, v in d.items()} for m, d in cells.items()},
                seed_jaccard=dict(values=jac, median={a: float(np.median(v)) for a, v in jac.items()}),
                trim_vs_base_pick_agreement=picks_agree, cells=cells, partitions=res,
                fit_lock_sha256={p: scored[p][1] for p in scored})  # fmt: skip
    (out / "metrics_full.json").write_text(json.dumps(full, indent=1, allow_nan=False) + "\n")
    np.savez_compressed(out / "picks.npz", **{f"TRIM_{p}": scored[p][0]["TRIM"] for p in scored})
    print(json.dumps(dict(verdict=verdict, config_insensitive=full["config_insensitive"], same=full["configs_same_as_base"], criteria=crit,
                          paired=paired, cell_medians=full["cell_medians"], jaccard=full["seed_jaccard"]["median"])), flush=True)  # fmt: skip
