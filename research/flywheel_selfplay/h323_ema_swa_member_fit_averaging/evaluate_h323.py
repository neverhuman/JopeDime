"""H323 evaluation (helpers from H319/H315, sha-pinned). Regression first; in-run BASE check; per-arm paired decision.

Decision (pre-registered, per arm SWA / EMA, Bonferroni over 2 arms; ties or margins below +0.002 never advance):
  ADVANCE iff (a) median over the 9 cells of the paired (ARM - BASE) drop-top-10 gain >= +0.002 AND the paired
  whole-family bootstrap one-sided LCB at 97.5% (0.025 quantile; 10,000 draws, RNG 323191; median over cells) > 0;
  (b) ARM median gain >= 0.9 x BASE's; (c) ARM median harmed <= BASE's; (d) for >= 2 of 3 root seeds, the median over the
  3 partitions of that seed's paired drop-top-10 difference > 0. Else NOT_ADVANCED. CONFIG_INSENSITIVE if the arm's
  (TAU,k,a) equals BASE's on >= 30/36 folds."""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ELO_LABEL = "reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN"
PAIRED_SEED = 323191
MARGIN = 0.002
PARTS = ("P0", "P1", "P2")
DOPE = {"composite_sha256_prefix": "c51a56d0", "loaded": False}


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
    out = dict(wave="H323", step="regression_first", status="PASS" if ok else "REGRESSION_FAIL", cells=rec)
    (Path(DATA) / "regression_receipt.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out), flush=True)
    if not ok:
        sys.exit("REGRESSION_FAIL")


def decide(paired, per_cell, gain_arm, gain_base, harm_arm, harm_base):
    """per_cell: the 9 paired drop-top-10 differences ordered (P0 s1..s3, P1 s1..s3, P2 s1..s3)."""
    pc = np.asarray(per_cell).reshape(3, 3)  # [partition, seed]
    seed_pos = int(sum(np.median(pc[:, s]) > 0 for s in range(3)))
    crit = dict(
        a_paired_drop_top10_median_ge_0_002_and_lcb975_gt_0=bool(paired["median"] >= MARGIN and paired["ci95"][0] > 0),
        b_median_gain_ge_90pct_base=bool(np.median(gain_arm) >= 0.9 * np.median(gain_base)),
        c_median_harmed_le_base=bool(np.median(harm_arm) <= np.median(harm_base)),
        d_paired_drop_top10_positive_on_ge_2_of_3_seeds=bool(seed_pos >= 2),
    )
    return crit, ("ADVANCE" if all(crit.values()) else "NOT_ADVANCED"), seed_pos


def main(r303, DATA, scored):
    import evaluate_h303 as ev
    import run_h323 as R

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
    arms = list(R.ARMS)
    res, D = {}, {a: [] for a in arms}
    cells = {m: {a: [] for a in arms} for m in ("gain", "harmed", "lofo_min")}
    jac = {a: [] for a in arms}
    for p in PARTS:
        fold_of_ctx = [R.outer_fold(p, r["family_id"], a302) for r in G["train_rows"]]
        res[p] = {}
        for arm in arms:
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
    verdicts = {}
    for arm in ("SWA", "EMA"):
        paired = e315.paired_drop_top(np.array(D[arm]), np.array(D["BASE"]), seed=PAIRED_SEED)
        crit, v, seed_pos = decide(
            paired,
            paired["per_cell"],
            cells["gain"][arm],
            cells["gain"]["BASE"],
            cells["harmed"][arm],
            cells["harmed"]["BASE"],
        )
        n_same = int(sum(sum(scored[p][2][arm]) for p in PARTS))
        agree = {p: [float(np.mean(scored[p][0][arm][s] == scored[p][0]["BASE"][s])) for s in range(3)] for p in PARTS}
        verdicts[arm] = dict(verdict=v, criteria=crit, paired_drop_top10=paired, seeds_positive=seed_pos,
                             configs_same_as_base=f"{n_same}/36", config_insensitive=bool(n_same >= 30), pick_agreement_with_base=agree)  # fmt: skip
    overall = (
        "ADVANCE(" + ",".join(a for a in verdicts if verdicts[a]["verdict"] == "ADVANCE") + ")"
        if any(v["verdict"] == "ADVANCE" for v in verdicts.values())
        else "NOT_ADVANCED"
    )
    full = dict(wave="H323", verdict=overall, arms=verdicts, elo_label=ELO_LABEL, dope=DOPE, inrun_base_equals_archived=inrun,
                cell_medians={m: {a: float(np.median(v)) for a, v in d.items()} for m, d in cells.items()},
                seed_jaccard=dict(values=jac, median={a: float(np.median(v)) for a, v in jac.items()}),
                cells=cells, partitions=res, fit_lock_sha256={p: scored[p][1] for p in scored})  # fmt: skip
    (out / "metrics_full.json").write_text(json.dumps(full, indent=1, allow_nan=False) + "\n")
    np.savez_compressed(
        out / "picks.npz", **{f"{arm}_{p}": scored[p][0][arm] for p in scored for arm in ("SWA", "EMA")}
    )
    print(json.dumps(dict(verdict=overall, arms={a: {k: v[k] for k in ("verdict", "criteria", "seeds_positive", "configs_same_as_base")} for a, v in verdicts.items()},
                          paired={a: {k: v["paired_drop_top10"][k] for k in ("median", "ci95")} for a, v in verdicts.items()},
                          cell_medians=full["cell_medians"], jaccard=full["seed_jaccard"]["median"])), flush=True)  # fmt: skip
