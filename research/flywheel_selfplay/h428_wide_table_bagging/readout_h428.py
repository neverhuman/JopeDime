#!/usr/bin/env python3
"""H428 readout: the test fixed in PREREG section 4.

  readout_h428.py --bag DIR --screen FITSEL_SCREEN.json --out FILE

Units = clusters of related FIT-SEL contexts (H426's `feature` clustering on the stored H425 rehearsal screen of
FIT-SEL), as H426 recommends; the same statistics with single contexts as units are reported beside them.
Per variant (T2 = the queued prompt, T1 = ridge on all features): h335_stats.primary_test of treatment u against
control u (clipped unit-mean gain, exact one-sided sign-flip p, leave-one-unit-out minimum, median).
Two variants, Holm: with p sorted, the smaller must be below 0.0125 and the larger below 0.025. A variant passes if
its clipped gain is above 0, its p meets its Holm threshold, its leave-one-unit-out minimum is above 0 and its median
is not negative.  BAG_CONFIRMED: at least one variant passes.  BAG_HARM: no variant passes and one has a clipped gain
below 0 with mirror p below 0.0125.  BAG_NULL: otherwise.
A wide context without a result (budget) scores the control in both variants (gain 0) and is counted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import restate_h426 as R26  # noqa: E402

N_WIDE = 33


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--screen", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import train_h416 as T16

    _, H = T16._stats_mod()
    st = [json.loads(p.read_text()) for p in sorted(Path(a.bag).glob("bag_status_*.json"))]
    rec = {}
    for p in sorted(Path(a.bag).glob("*.json")):
        if p.name.startswith("bag_status_"):
            continue
        r = json.loads(p.read_text())
        rec[r["cid"]] = r
    screen = json.loads(Path(a.screen).read_text())
    if screen["queries"] != ["FIT-SEL"]:
        raise SystemExit("the screen file is not the FIT-SEL rehearsal")
    all_c = sorted(screen["per_context"])
    wide = sorted(c for c in all_c if screen["per_context"][c]["width"] > 30)
    if len(wide) != N_WIDE or not set(rec) <= set(wide):
        raise SystemExit("the wide FIT-SEL contexts are not the 33 expected")
    cl_all = R26.clusterings(screen, all_c)["feature"]
    missing = [c for c in wide if c not in rec or not rec[c]["finite"]]
    val = {v: {c: (rec[c][f"u_{v}"] if c not in missing else (rec[c]["u_control"] if c in rec else 0.0)) for c in wide}
           for v in ("control", "T1", "T2")}
    for c in missing:   # no result: all three arms share one value, so the gain is 0
        for v in val:
            val[v][c] = rec[c]["u_control"] if c in rec else 0.0
    units = {"clusters": {c: cl_all[c] for c in wide}, "contexts": {c: c for c in wide}}
    res = {"schema": "jope.h428.readout/v1", "cohort": "FIT-SEL (monitor)", "wide_contexts": len(wide), "computed": len(wide) - len(missing),
           "units": {k: len(set(u.values())) for k, u in units.items()},
           "ckpt_sha256": sorted({s["ckpt_sha256"] for s in st}), "mean_u": {v: float(np.mean(list(val[v].values()))) for v in val},
           "mean_u_ridge": float(np.mean([rec[c]["u_ridge"] for c in wide if c in rec])), "tests": {}}
    for uname, u in units.items():
        res["tests"][uname] = {}
        for v in ("T1", "T2"):
            s = T16.pair_stats(np, H, val[v], val["control"], u)
            res["tests"][uname][v] = {k: s[k] for k in ("gain", "clipped_mean", "signflip_p", "mirror_signflip_p", "lofo_min",
                                                         "family_median", "wtl")}
    t = res["tests"]["clusters"]
    order = sorted(("T1", "T2"), key=lambda v: t[v]["signflip_p"])
    thr = {order[0]: 0.0125, order[1]: 0.025}
    passed = {}
    first = t[order[0]]
    ok0 = first["clipped_mean"] > 0 and first["signflip_p"] < thr[order[0]] and first["lofo_min"] > 0 and first["family_median"] >= 0
    passed[order[0]] = bool(ok0)
    second = t[order[1]]
    passed[order[1]] = bool(ok0 and second["clipped_mean"] > 0 and second["signflip_p"] < thr[order[1]]
                            and second["lofo_min"] > 0 and second["family_median"] >= 0)
    res["holm"] = {"order": order, "thresholds": thr, "passed": passed}
    harm = any(t[v]["clipped_mean"] < 0 and t[v]["mirror_signflip_p"] < 0.0125 for v in ("T1", "T2"))
    res["verdict"] = "BAG_CONFIRMED" if any(passed.values()) else "BAG_HARM" if harm else "BAG_NULL"
    bins = {}
    for name, sel in (("F 31-100", [c for c in wide if rec.get(c, {}).get("F", 0) <= 100 and c in rec]),
                      ("F > 100", [c for c in wide if rec.get(c, {}).get("F", 0) > 100])):
        if sel:
            bins[name] = {"n": len(sel), **{f"mean_gain_{v}": float(np.mean([val[v][c] - val["control"][c] for c in sel])) for v in ("T1", "T2")},
                          **{f"wins_losses_{v}": [int(sum(val[v][c] - val["control"][c] > 1e-4 for c in sel)),
                                                  int(sum(val[v][c] - val["control"][c] < -1e-4 for c in sel))] for v in ("T1", "T2")},
                          "mean_control_minus_ridge": float(np.mean([rec[c]["u_control"] - rec[c]["u_ridge"] for c in sel]))}
    res["by_width"] = bins
    res["inference"] = {"processes": len(st), "cpu_s": float(sum(s["cpu_s"] for s in st)), "wall_s_max": float(max(s["wall_s"] for s in st)) if st else None,
                        "budget_stops": int(sum(s["budget_stop"] for s in st)), "devices": sorted({s["device"] for s in st}),
                        "cuda_visible_any": bool(any(s["cuda_visible"] for s in st))}
    print(f"== H428 FIT-SEL monitor, wide contexts {res['wide_contexts']} (computed {res['computed']}), units {res['units']}, mean u {res['mean_u']}")
    for uname in units:
        for v in ("T1", "T2"):
            s = res["tests"][uname][v]
            print(f"   {uname:8s} {v} vs control: gain {s['gain']:+.4f} clipped {s['clipped_mean']:+.4f} p {s['signflip_p']:.3g} "
                  f"lofo_min {s['lofo_min']:+.4f} median {s['family_median']:+.4f} W/T/L {s['wtl']}")
    print(f"   Holm {res['holm']}  by width {bins}  verdict {res['verdict']}")
    with open(a.out, "x") as f:
        json.dump(res, f, sort_keys=True)


if __name__ == "__main__":
    main()
