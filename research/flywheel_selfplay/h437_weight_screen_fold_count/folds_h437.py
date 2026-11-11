#!/usr/bin/env python3
"""H437: how many folds of rows does the weight rule need? Stored hour-12 FIT-SEL values only; nothing is fitted.

  folds_h437.py stats   --run-dir DIR --tag TBOTH --val VAL_BANK.npz --oof OOF.npz --h435 H435_FITSEL.per_table.jsonl --out FILE
  folds_h437.py readout --stats FILE --screen FITSEL_SCREEN.json --h435-readout H435_FITSEL.readout.json --out FILE
  folds_h437.py selftest

The rule is H434's, as H435 applied it: for a column and the stored blend with RMSE ratio r and error overlap rho on
a set of rows, predicted weight w = (1 - rho r) / (1 + r^2 - 2 rho r) clipped to [0, 1]; keep the table when w >= 0.10.
stats, per FIT-SEL table and for the two hour-12 in-context columns: per stored fold, the row count and the three
sums that give rho and r for any set of folds (sum of squared column errors, of squared blend errors, of their
products). Only SEARCH-side arrays are read (candidates' out-of-fold predictions, SEARCH targets, fold labels);
the stored weights of the record give the blend. The realised weight of each column and the per-table cost model
are H435's stored values. The weight builder is not imported: no weight is fitted in this wave.
readout: the rule on the rows of the first k folds, k = 1 .. 5; verdict on k = 2 and k = 3 (PREREG section 4).
  cost with a k-fold screen = k screening passes over SEARCH rows on every table (extra work, as in H435) + the full
  six passes on the tables kept.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

NEED, BOOT_SEED, N_BOOT = 0.10, 435150, 10_000   # H435's seed: the one-fold block must reproduce H435's intervals
SXB_KEY = "P3+LIB3.SXB"
COLS = {"icl_h12_ridgeres": "ridgeres", "icl_h12_plain": "full"}
TEST_K = (2, 3)


def w_two(rho: float, r: float) -> float:
    d = 1.0 + r * r - 2.0 * rho * r
    return float(np.clip((1.0 - rho * r) / d, 0.0, 1.0)) if d > 1e-12 else 0.5


def from_sums(cc: float, bb: float, cb: float) -> dict:
    r = float(np.sqrt(cc / bb))
    rho = float(cb / np.sqrt(cc * bb)) if cc > 0 and bb > 0 else 0.0
    return {"overlap": rho, "ratio": r, "w_pred": w_two(rho, r)}


def last_result(p: Path) -> dict:
    rs = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    rs = [r for r in rs if r.get("type") == "result"]
    if len(rs) != 1:
        raise SystemExit(f"{p}: {len(rs)} result records")
    return rs[0]


def cmd_stats(a) -> None:
    t0, c0 = time.time(), time.process_time()
    cfg = json.loads((Path(a.run_dir) / f"config_{a.tag}.json").read_text())
    cids = sorted(cfg["cids"])
    h435 = {r["cid"]: r for r in (json.loads(x) for x in Path(a.h435).read_text().splitlines() if x.strip())}
    if sorted(h435) != cids:
        raise SystemExit("H435's per-table file does not hold exactly this panel")
    v = np.load(a.val, allow_pickle=False)
    o = np.load(a.oof, allow_pickle=False)
    if set(v["cohort"].tolist()) != {"FIT-SEL"} or not np.array_equal(v["cids"], o["cids"]) or not np.array_equal(v["shape"], o["shape"]):
        raise SystemExit("the bank is not FIT-SEL or does not match the stored out-of-fold file")
    off = np.concatenate([[0], np.cumsum(v["shape"][:, 0])])
    pos = {str(c): i for i, c in enumerate(v["cids"])}
    rows = []
    for cid in cids:
        sd = Path(a.run_dir) / "stages" / f"H416_{a.tag}"
        rp = sd / f"{cid}__confirm.jsonl"
        rec = last_result(rp if rp.exists() else sd / f"{cid}__dev.jsonl")
        zp = Path(a.run_dir) / f"h416_{a.tag}_npz" / f"{cid}.npz"
        if hashlib.sha256(zp.read_bytes()).hexdigest() != rec["npz"]["sha256"]:
            raise SystemExit(f"{zp}: npz sha differs from its record")
        z = np.load(zp, allow_pickle=False)
        Q, y, lab = z["Q"].astype(np.float64), z["y"].astype(np.float64), z["lab"].astype(np.int64)   # SEARCH-side arrays only
        shas = [str(s) for s in z["shas"]]
        sp = rec["outputs"][SXB_KEY]["spec"]
        ws = np.zeros(len(shas))
        for s, x in zip(sp["members"], sp["w"]):
            ws[shas.index(s)] = x
        eb = Q @ ws - y
        i = pos[cid]
        lo, hi = int(off[i]), int(off[i + 1])
        if not (hi - lo == len(y) and np.array_equal(v["lab"][lo:hi].astype(np.int64), lab) and np.allclose(v["y64"][lo:hi], y)):
            raise SystemExit(f"{cid}: the bank rows do not align with the stored matrix")
        if sorted(set(lab.tolist())) != [0, 1, 2, 3, 4]:
            raise SystemExit(f"{cid}: fold labels are not 0..4")
        h = h435[cid]
        r = {"cid": cid, "n_search": int(len(y)), "fold_rows": [int((lab == f).sum()) for f in range(5)],
             "cost_full_s": h["cost_full_s"], "cost_screen_s": h["cost_screen_s"], "wide": h["wide"], "columns": {}}
        for name, key in COLS.items():
            ec = o[key][lo:hi].astype(np.float64) - y
            s = [[float((ec[lab == f] ** 2).sum()), float((eb[lab == f] ** 2).sum()), float((ec[lab == f] * eb[lab == f]).sum())] for f in range(5)]
            f0, al = from_sums(*s[0]), from_sums(*[sum(x[j] for x in s) for j in range(3)])
            hc = h["columns"][name]
            r["columns"][name] = {"sums": s, "w_real": hc["w_real"],
                                  "replay_abs_diff": max(abs(f0["overlap"] - hc["fold0"]["overlap"]), abs(f0["ratio"] - hc["fold0"]["ratio"]),
                                                         abs(al["overlap"] - hc["all"]["overlap"]), abs(al["ratio"] - hc["all"]["ratio"]))}
        rows.append(r)
    out = {"schema": "jope.h437.stats/v1", "tables": len(rows), "threads": 1, "torch_loaded": "torch" in sys.modules,
           "weight_builder_loaded": "meta_h416" in sys.modules or "cache_h422" in sys.modules,
           "replay_max_abs_diff": float(max(c["replay_abs_diff"] for r in rows for c in r["columns"].values())),
           "wall_s": time.time() - t0, "cpu_s": time.process_time() - c0, "rows": rows}
    with open(a.out, "x") as f:
        json.dump(out, f, sort_keys=True)
    print(json.dumps({k: out[k] for k in out if k != "rows"}))


def score(rows, k: int) -> dict:
    p, t = np.array([x[0] for x in rows]), np.array([x[1] for x in rows])
    cf, cs = np.array([x[2] for x in rows]), np.array([x[3] for x in rows])
    n = len(rows)
    tp, fp, fn, tn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum()), int((~p & ~t).sum())
    return {"decisions": n, "kept": int(p.sum()), "real_weight": int(t.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": (tp + tn) / n, "fn_per_decision": fn / n, "miss_rate_of_real": (fn / (fn + tp)) if (fn + tp) else None,
            "saved_inference": float(1.0 - (k * cs.sum() + cf[p].sum()) / cf.sum()),
            "saved_inference_if_screen_passes_reused": float(1.0 - (k * cs[~p].sum() + cf[p].sum()) / cf.sum())}


def decide(st: dict, folds, names=tuple(COLS)) -> dict:
    """cid -> list of (keep, real, cost_full, cost_screen) for the rule on the rows of `folds`."""
    d = {}
    for r in st["rows"]:
        d[r["cid"]] = []
        for nm in names:
            c = r["columns"][nm]
            e = from_sums(*[sum(c["sums"][f][j] for f in folds) for j in range(3)])
            d[r["cid"]].append((e["w_pred"] >= NEED, float(c["w_real"]) >= NEED, r["cost_full_s"], r["cost_screen_s"]))
    return d


def cmd_readout(a) -> None:
    import restate_h426 as R26

    st = json.loads(Path(a.stats).read_text())
    cids = sorted(r["cid"] for r in st["rows"])
    cl = R26.clusterings(json.loads(Path(a.screen).read_text()), cids)["feature"]
    units = sorted(set(cl.values()))
    by_unit = {u: [c for c in cids if cl[c] == u] for u in units}
    draws = np.random.default_rng(BOOT_SEED).integers(0, len(units), size=(N_BOOT, len(units)))
    q = lambda v: [float(x) for x in np.quantile(v, [0.025, 0.975])]

    def block(folds, names=tuple(COLS), base=None) -> dict:
        k, d = len(folds), decide(st, folds, names)
        s = score([x for c in cids for x in d[c]], k)
        bf, bs, ba, bd = [], [], [], []
        for idx in draws:
            sel = [c for i in idx for c in by_unit[units[i]]]
            b = score([x for c in sel for x in d[c]], k)
            bf.append(b["fn_per_decision"]); bs.append(b["saved_inference"]); ba.append(b["accuracy"])
            if base is not None:
                bd.append(b["fn_per_decision"] - score([x for c in sel for x in base[c]], 1)["fn_per_decision"])
        s.update({"folds": list(folds), "rows_share": float(np.mean([sum(r["fold_rows"][f] for f in folds) / r["n_search"] for r in st["rows"]])),
                  "fn_per_decision_ci95": q(bf), "saved_inference_ci95": q(bs), "accuracy_ci95": q(ba)})
        if base is not None:
            s["fn_per_decision_minus_one_fold_ci95"] = q(bd)
            s["boot_share_fn_not_below_one_fold"] = float(np.mean(np.array(bd) >= 0))
        return s

    one = decide(st, (0,))
    res = {"schema": "jope.h437.readout/v1", "cohort": "FIT-SEL (TRAIN side), stored values", "tables": len(cids), "units": len(units),
           "by_fold_count": {str(k): block(tuple(range(k)), base=(one if k > 1 else None)) for k in range(1, 6)},
           "per_column": {nm: {str(k): block(tuple(range(k)), names=(nm,)) for k in (1, 2, 3)} for nm in COLS},
           "fold_choice_spread": {}}
    for k in range(1, 5):
        sc = [score([x for c in cids for x in decide(st, fs)[c]], k) for fs in itertools.combinations(range(5), k)]
        fn, sv, ac = [x["fn_per_decision"] for x in sc], [x["saved_inference"] for x in sc], [x["accuracy"] for x in sc]
        res["fold_choice_spread"][str(k)] = {"choices": len(sc), "fn_per_decision": [float(min(fn)), float(np.median(fn)), float(max(fn))],
                                             "saved_inference": [float(min(sv)), float(np.median(sv)), float(max(sv))],
                                             "accuracy": [float(min(ac)), float(np.median(ac)), float(max(ac))]}
    b1 = res["by_fold_count"]["1"]
    ok = {str(k): {"fn_le_0.05": bool(res["by_fold_count"][str(k)]["fn_per_decision"] <= 0.05),
                   "fn_upper_le_0.09": bool(res["by_fold_count"][str(k)]["fn_per_decision_ci95"][1] <= 0.09),
                   "saved_ge_0.60": bool(res["by_fold_count"][str(k)]["saved_inference"] >= 0.60),
                   "fewer_fn_than_one_fold": bool(res["by_fold_count"][str(k)]["fn"] < b1["fn"])} for k in TEST_K}
    res["checks"] = ok
    passing = [k for k in TEST_K if ok[str(k)]["fn_le_0.05"] and ok[str(k)]["fn_upper_le_0.09"] and ok[str(k)]["saved_ge_0.60"]]
    res["passing_fold_counts"] = passing
    res["verdict"] = "FOLDS_PASS" if passing else ("FOLDS_BETTER_NOT_ENOUGH" if any(ok[str(k)]["fewer_fn_than_one_fold"] for k in TEST_K) else "FOLDS_FAIL")
    h = json.loads(Path(a.h435_readout).read_text())
    hp, ha = h["primary_pooled_icl_h12"], h["variants_primary"]["estimate_from_all_rows"]
    b5 = res["by_fold_count"]["5"]
    res["replay_h435"] = {"one_fold_counts_equal": bool(all(b1[x] == hp[x] for x in ("tp", "fp", "fn", "tn", "kept", "decisions"))),
                          "one_fold_saved_abs_diff": abs(b1["saved_inference"] - hp["saved_inference"]),
                          "one_fold_ci_abs_diff": float(max(abs(u - w) for u, w in zip(b1["fn_per_decision_ci95"] + b1["saved_inference_ci95"],
                                                                                    hp["fn_per_decision_ci95"] + hp["saved_inference_ci95"]))),
                          "all_rows_counts_equal": bool(all(b5[x] == ha[x] for x in ("tp", "fp", "fn", "tn")))}
    f = lambda s: (f"rows {100 * s['rows_share']:.0f}% kept {s['kept']} TP {s['tp']} FP {s['fp']} FN {s['fn']} TN {s['tn']} acc {s['accuracy']:.3f} "
                   f"FN/dec {s['fn_per_decision']:.3f} {[round(x, 3) for x in s['fn_per_decision_ci95']]} saved {s['saved_inference']:.3f} "
                   f"{[round(x, 3) for x in s['saved_inference_ci95']]} (reused {s['saved_inference_if_screen_passes_reused']:.3f})")
    print(f"== H437 FIT-SEL: {len(cids)} tables in {len(units)} clusters, 212 decisions (two hour-12 columns), real weight on {b1['real_weight']}")
    for k in range(1, 6):
        s = res["by_fold_count"][str(k)]
        extra = "" if k == 1 else f" | FN/dec minus one fold {[round(x, 3) for x in s['fn_per_decision_minus_one_fold_ci95']]} not below in {s['boot_share_fn_not_below_one_fold']:.3f}"
        print(f"   first {k} fold(s){' [TEST]' if k in TEST_K else '       '}: {f(s)}{extra}")
    for nm in COLS:
        for k in ("1", "2", "3"):
            print(f"   {nm:18s} k {k}: {f(res['per_column'][nm][k])}")
    for k, s in res["fold_choice_spread"].items():
        print(f"   any {k} fold(s), {s['choices']} choices: FN/dec min/median/max {[round(x, 3) for x in s['fn_per_decision']]} saved {[round(x, 3) for x in s['saved_inference']]} "
              f"accuracy {[round(x, 3) for x in s['accuracy']]}")
    print(f"   replay {res['replay_h435']}")
    print(f"   checks {ok}  passing {passing}  verdict {res['verdict']}")
    with open(a.out, "x") as fo:
        json.dump(res, fo, sort_keys=True)


def cmd_selftest(a) -> None:
    r = np.random.default_rng(4)
    ec, eb = r.standard_normal(500) * 1.2, r.standard_normal(500)
    lab = np.arange(500) % 5
    s = [[float((ec[lab == f] ** 2).sum()), float((eb[lab == f] ** 2).sum()), float((ec[lab == f] * eb[lab == f]).sum())] for f in range(5)]
    al = from_sums(*[sum(x[j] for x in s) for j in range(3)])
    direct = float(ec @ eb / np.sqrt((ec @ ec) * (eb @ eb)))
    print("selftest: all-row overlap from sums", round(al["overlap"], 12), "direct", round(direct, 12), "ratio", round(al["ratio"], 6),
          "w", round(al["w_pred"], 4), "fold0", {k: round(v, 4) for k, v in from_sums(*s[0]).items()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stats")
    for k in ("--run-dir", "--tag", "--val", "--oof", "--h435", "--out"):
        p.add_argument(k, required=True)
    p.set_defaults(fn=cmd_stats)
    p = sub.add_parser("readout")
    for k in ("--stats", "--screen", "--h435-readout", "--out"):
        p.add_argument(k, required=True)
    p.set_defaults(fn=cmd_readout)
    p = sub.add_parser("selftest")
    p.set_defaults(fn=cmd_selftest)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
