#!/usr/bin/env python3
"""H420 readout: the paired tests fixed in PREREG section 3, from pulled, sha-compared run directories.
No GP stack: train_h416's helpers, numpy, h335_stats.

  readout_h420.py --run-dir DIR --out OUT --h417-run DIR417 [--h417-tag BASE]

Baseline (addendum 1): the stored H417 deploy of the same contexts. It is not recomputed: the H420 run directory was
seeded with H417's evaluation memo, so the variants differ from it only in the programs they add. If the run
directory holds a variant B, that B is the baseline and the stored H417 deploy is reported as a host replicate.

Entrant = P3+LIB3.SXB of each variant; value = AUDIT u; an invalid or missing output scores the variant's single-run
stock value X0.GS (H415 / H416 rule) and is counted. Paired by context, families = contexts.
Primary: TBOTH vs baseline (h335_stats.primary_test). Secondary: TGX and TLIB vs baseline, Holm at 0.025.
The panel's cohort (DEV = the test cohort, FIT-SEL = reported only) is read from the manifest and printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

VARIANTS = ("B", "TGX", "TLIB", "TBOTH")
SXB_KEY, KFB_KEY = "P3+LIB3.SXB", "P3+LIB3.KFB"
ALPHA = 0.025
BOOT_SEED, N_BOOT = 420150, 10_000


def _load(run_dir: Path, tag: str, only=None):
    import train_apex as TA

    T16, cfg, r16, r15 = TA._records(run_dir, tag)
    cids = [c for c in cfg["cids"] if only is None or c in only]
    sxb, kfb, x0s, inv = {}, {}, {}, {"SXB": 0, "KFB": 0, "X0": 0}
    for c in cids:
        gs = r15[c]["outputs"].get("GS.stock")
        ok = gs is not None and not gs.get("invalid") and gs["raw"].get("status") == "completed"
        x0 = float(gs["value"]) if ok else 0.0
        inv["X0"] += not ok
        x0s[c] = x0
        s = r16[c]["outputs"].get(SXB_KEY)
        bad = s is None or s.get("invalid") or r16[c].get("error")
        sxb[c] = x0 if bad else float(s["value"])
        inv["SXB"] += bool(bad)
        v, badk, _ = T16.out_value(r15[c], KFB_KEY, x0)
        kfb[c] = v
        inv["KFB"] += bool(badk)
    ws = {}
    for c in cids:
        for r in (r16[c], r15[c]):
            for k, v in r["worker_s"].items():
                ws[k] = ws.get(k, 0.0) + float(v)
    members = {c: (r16[c]["outputs"].get(SXB_KEY) or {}).get("spec", {}).get("members", []) for c in cids}
    return T16, dict(cfg, cids=cids), {"SXB": sxb, "KFB": kfb, "X0": x0s, "invalid": inv, "worker_s": ws,
                                       "members": members}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--h417-run")
    ap.add_argument("--h417-tag", default="BASE")
    a = ap.parse_args()
    import apex_core_meta as AM
    import lang_h420 as LP

    run_dir = Path(a.run_dir)
    data, cfgs, T16 = {}, {}, None
    for v in VARIANTS:
        if (run_dir / f"config_{v}.json").is_file():
            T16, cfgs[v], data[v] = _load(run_dir, v)
    treat = [v for v in ("TBOTH", "TGX", "TLIB") if v in data]
    if not treat:
        raise SystemExit("no treatment variant in the run directory")
    cids = cfgs[treat[0]]["cids"]
    for v in data:
        if cfgs[v]["cids"] != cids:
            raise SystemExit(f"variant {v} ran on other contexts")
    d17 = None
    if a.h417_run:
        _t, _c17, d17 = _load(Path(a.h417_run), a.h417_tag, only=set(cids))
        if sorted(d17["SXB"]) != sorted(cids):
            raise SystemExit("the stored H417 deploy does not cover these contexts")
    if "B" in data:
        baseline = "variant B of this run directory"
    elif d17 is not None:
        data["B"], baseline = d17, f"stored H417 deploy, tag {a.h417_tag} (not recomputed)"
    else:
        raise SystemExit("no baseline: pass --h417-run")
    np, H = T16._stats_mod()
    rows = AM.rows()
    cohorts = sorted({rows[c]["cohort"] for c in cids})
    fam_of = AM.family_of(cids)
    if len(set(fam_of.values())) != len(cids):
        raise SystemExit("one context per family expected")
    lib_shas = {hashlib.sha256(d.encode()).hexdigest(): n for n, d in LP.LIB_EXT}
    res = {"schema": "jope.h420.readout/v2", "panel": cfgs[treat[0]]["panel"], "cohorts": cohorts, "n": len(cids),
           "baseline": baseline, "role": "test" if cohorts == ["DEV"] else "reported only", "variants": {},
           "vs_B": {}}
    for v, d in data.items():
        with_lib = sum(1 for c in cids if any(m in lib_shas for m in d["members"][c]))
        res["variants"][v] = {"family_mean_u": {k: H.family_mean(d[k], fam_of) for k in ("SXB", "KFB", "X0")},
                              "invalid": d["invalid"], "worker_s": d["worker_s"],
                              "worker_s_total": float(sum(d["worker_s"].values())),
                              "contexts_with_lib_ext_member": with_lib,
                              "x0_max_abs_diff_vs_B": max(abs(d["X0"][c] - data["B"]["X0"][c]) for c in cids)}
    for v in treat:
        res["vs_B"][v] = {"SXB": T16.pair_stats(np, H, data[v]["SXB"], data["B"]["SXB"], fam_of),
                          "KFB": T16.pair_stats(np, H, data[v]["KFB"], data["B"]["KFB"], fam_of)}
    sec = sorted((res["vs_B"][v]["SXB"]["signflip_p"], v) for v in ("TGX", "TLIB") if v in res["vs_B"])
    holm, alive = {}, True
    for i, (p, v) in enumerate(sec):
        alive = alive and p < ALPHA / (len(sec) - i)
        holm[v] = bool(alive and res["vs_B"][v]["SXB"]["pass"])
    res["secondary_holm_pass"] = holm
    prim = res["vs_B"].get("TBOTH", {}).get("SXB")
    res["primary_pass"] = bool(prim and prim["pass"])
    res["verdict"] = ("LANG_CONFIRMED" if res["primary_pass"] else "LANG_PARTIAL" if any(holm.values())
                      else "LANG_NULL") if res["role"] == "test" else "not a test cohort"
    vals = {f"{v}.SXB": data[v]["SXB"] for v in data}
    vals["B.KFB"] = data["B"]["KFB"]
    arms, point, boot, elo = T16.elo_boot(np, H, vals, fam_of, "B.SXB", BOOT_SEED, N_BOOT)
    res["elo_anchor_B.SXB_1500"] = elo
    res["delta_elo_vs_B"] = {v: T16.delta_elo(np, arms, point, boot, f"{v}.SXB", "B.SXB") for v in treat}
    if d17 is not None and baseline.startswith("variant B"):
        diffs = [data["B"]["SXB"][c] - d17["SXB"][c] for c in cids]
        res["host_replicate_B_minus_H417"] = {"n": len(cids), "mean": float(np.mean(diffs)),
                                              "max_abs": float(np.max(np.abs(diffs))),
                                              "exact": int(sum(1 for x in diffs if x == 0.0))}
    print(f"== panel {res['panel']} cohorts {cohorts} n {res['n']} role: {res['role']} baseline: {baseline}")
    for v, o in res["variants"].items():
        m = o["family_mean_u"]
        print(f"   {v:6s} SXB {m['SXB']:.4f} KFB {m['KFB']:.4f} X0.GS {m['X0']:.4f} invalid {o['invalid']} "
              f"worker_s {o['worker_s_total']:.0f} lib-ext member in {o['contexts_with_lib_ext_member']} contexts")
    for v, o in res["vs_B"].items():
        s = o["SXB"]
        print(f"   {v:6s} vs B: gain {s['gain']:+.4f} ({s['pct_rmse']:+.2f}% RMSE) clipped {s['clipped_mean']:+.4f} "
              f"p {s['signflip_p']:.3g} lofo_min {s['lofo_min']:+.4f} median {s['family_median']:+.4f} "
              f"W/T/L {s['wtl']} pass {s['pass']} dElo {res['delta_elo_vs_B'][v]['delta']:+.1f} "
              f"{res['delta_elo_vs_B'][v]['ci95']}")
    print(f"   primary_pass {res['primary_pass']} secondary_holm {holm} verdict {res['verdict']}")
    if "host_replicate_B_minus_H417" in res:
        print(f"   host replicate B - H417: {res['host_replicate_B_minus_H417']}")
    Path(a.out).write_text(json.dumps(res, sort_keys=True))


if __name__ == "__main__":
    main()
