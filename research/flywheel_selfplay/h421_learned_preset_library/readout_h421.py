#!/usr/bin/env python3
"""H421 readout: the tests fixed in PREREG sections 4-5, from a pulled, sha-compared copy of the DEV run directory.
No GP stack: readout_h420's loader, train_h416's helpers, numpy, h335_stats.

  readout_h421.py --run-dir DIR --presets-dir DIR --out OUT

Baseline B* = the stored H420 TBOTH records of the same run directory (never recomputed).
Entrant = P3+LIB3.SXB of each variant; value = AUDIT u; an invalid or missing output scores the variant's single-run
stock value (H415 / H416 rule) and is counted. Paired by context, families = contexts.
  primary   TPORT vs B*     h335_stats.primary_test (clipped family-mean gain > 0, sign-flip p < 0.025 one-sided,
                            leave-one-family-out minimum > 0, median >= 0)
  control   TPORT vs TRAND  the same statistic, reported with its p value
  reported  TRAND vs B*, TCOV vs B*, TCOV vs TPORT; KFB of each; family-bootstrap delta Elo; worker-seconds;
            how many added presets carry stack weight and in how many families
Verdict (section 5): PORT_HARM if TPORT - B* clipped mean < 0 with mirror sign-flip p < 0.025; PORT_NULL if the
primary fails; PORT_CONFIRMED if it passes and the TPORT - TRAND clipped mean is above 0; else PORT_MORE_PRESETS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

BASE = "TBOTH"
TREAT = ("TPORT", "TRAND", "TCOV")
FILES = {"TPORT": "LIB_PORT.json", "TRAND": "LIB_RAND.json", "TCOV": "LIB_COV.json"}
PAIRS = (("TPORT", BASE), ("TPORT", "TRAND"), ("TRAND", BASE), ("TCOV", BASE), ("TCOV", "TPORT"))
BOOT_SEED, N_BOOT = 421150, 10_000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--presets-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import apex_core_meta as AM
    import readout_h420 as R20

    run_dir = Path(a.run_dir)
    data, cfgs, T16 = {}, {}, None
    for v in (BASE,) + TREAT:
        T16, cfgs[v], data[v] = R20._load(run_dir, v)
    cids = cfgs[BASE]["cids"]
    rows = AM.rows()
    if sorted({rows[c]["cohort"] for c in cids}) != ["DEV"]:
        raise SystemExit("the test cohort is DEV")
    for v in TREAT:
        if cfgs[v]["cids"] != cids:
            raise SystemExit(f"variant {v} ran on other contexts")
        raw = (Path(a.presets_dir) / FILES[v]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != cfgs[v]["presets_sha256"]:
            raise SystemExit(f"variant {v} did not run the preset file given here")
    np, H = T16._stats_mod()
    fam_of = AM.family_of(cids)
    if len(set(fam_of.values())) != len(cids):
        raise SystemExit("one context per family expected")
    res = {"schema": "jope.h421.readout/v1", "panel": cfgs[BASE]["panel"], "cohort": "DEV", "n": len(cids),
           "baseline": "stored H420 TBOTH records of this run directory (not recomputed)", "variants": {},
           "pairs": {}, "presets": {}}
    for v, d in data.items():
        res["variants"][v] = {"family_mean_u": {k: H.family_mean(d[k], fam_of) for k in ("SXB", "KFB", "X0")},
                              "invalid": d["invalid"], "worker_s": d["worker_s"],
                              "worker_s_total": float(sum(d["worker_s"].values())),
                              "x0_max_abs_diff_vs_base": max(abs(d["X0"][c] - data[BASE]["X0"][c]) for c in cids)}
    for v in TREAT:   # which added presets carry stack weight (program sha = sha256 of the program text)
        ps = {hashlib.sha256(d.encode()).hexdigest(): n
              for n, d in json.loads((Path(a.presets_dir) / FILES[v]).read_text())}
        per = {n: 0 for n in ps.values()}
        any_ctx = 0
        for c in cids:
            hit = {ps[m] for m in data[v]["members"][c] if m in ps}
            any_ctx += bool(hit)
            for n in hit:
                per[n] += 1
        res["presets"][v] = {"contexts_with_added_member": any_ctx, "presets_with_weight_somewhere": sum(1 for x in per.values() if x),
                             "families_per_preset": per}
    for x, y in PAIRS:
        res["pairs"][f"{x}_vs_{y}"] = {"SXB": T16.pair_stats(np, H, data[x]["SXB"], data[y]["SXB"], fam_of),
                                       "KFB": T16.pair_stats(np, H, data[x]["KFB"], data[y]["KFB"], fam_of)}
    prim = res["pairs"][f"TPORT_vs_{BASE}"]["SXB"]
    ctrl = res["pairs"]["TPORT_vs_TRAND"]["SXB"]
    res["primary_pass"] = bool(prim["pass"])
    res["control_clipped_mean"] = ctrl["clipped_mean"]
    if prim["clipped_mean"] < 0 and prim["mirror_signflip_p"] < 0.025:
        res["verdict"] = "PORT_HARM"
    elif not prim["pass"]:
        res["verdict"] = "PORT_NULL"
    elif ctrl["clipped_mean"] > 0:
        res["verdict"] = "PORT_CONFIRMED"
    else:
        res["verdict"] = "PORT_MORE_PRESETS"
    vals = {f"{v}.SXB": data[v]["SXB"] for v in data}
    arms, point, boot, elo = T16.elo_boot(np, H, vals, fam_of, f"{BASE}.SXB", BOOT_SEED, N_BOOT)
    res[f"elo_anchor_{BASE}.SXB_1500"] = elo
    res["delta_elo"] = {f"{x}_vs_{y}": T16.delta_elo(np, arms, point, boot, f"{x}.SXB", f"{y}.SXB") for x, y in PAIRS}
    print(f"== H421 DEV n {res['n']} baseline: {res['baseline']}")
    for v, o in res["variants"].items():
        m = o["family_mean_u"]
        extra = ""
        if v in res["presets"]:
            p = res["presets"][v]
            extra = (f" added preset in the stack of {p['contexts_with_added_member']} contexts, "
                     f"{p['presets_with_weight_somewhere']} of 20 presets used")
        print(f"   {v:6s} SXB {m['SXB']:.4f} KFB {m['KFB']:.4f} invalid {o['invalid']} worker_s {o['worker_s_total']:.0f}{extra}")
    for k, o in res["pairs"].items():
        s, e = o["SXB"], res["delta_elo"][k]
        print(f"   {k:16s} gain {s['gain']:+.4f} u ({100 * s['pct_rmse']:+.2f}% RMSE) clipped {s['clipped_mean']:+.4f} "
              f"p {s['signflip_p']:.3g} lofo_min {s['lofo_min']:+.4f} median {s['family_median']:+.4f} W/T/L {s['wtl']} "
              f"pass {s['pass']} dElo {e['delta']:+.1f} [{e['ci95'][0]:+.1f}, {e['ci95'][1]:+.1f}]")
    print(f"   primary_pass {res['primary_pass']} control clipped mean {res['control_clipped_mean']:+.4f} verdict {res['verdict']}")
    with open(a.out, "x") as f:
        json.dump(res, f, sort_keys=True)


if __name__ == "__main__":
    main()
