#!/usr/bin/env python3
"""H424 second read (PREREG addendum 1): the hour-24 checkpoint's column, beside the first read.

  readout2_h424.py --first APPLY.json --second APPLY2.json --audit dup_audit.json --out FILE

Reported, no new verdict (the wave's verdict is the first read's):
  I24 vs B   the first read's test statistics for the hour-24 column (h335_stats.primary_test), with the pre-declared
             secondary (contexts at the +-0.5 clip excluded)
  I24 vs I   does the later checkpoint give a better column (same statistics)
  I vs B     the first read, restated from its apply file
  sensitivity: I vs B and I24 vs B without the DEV contexts the duplicate screen flags (at least half of the distinct
             SEARCH target values also occur in one FIT-ICL training table; audit/dup_audit.py)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stack_h424 as S  # noqa: E402

BOOT_SEED, N_BOOT, FLAG_SHARE = 424250, 10_000, 0.5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first", required=True)
    ap.add_argument("--second", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import train_h416 as T16

    _, H = T16._stats_mod()
    f, s = json.loads(Path(a.first).read_text()), json.loads(Path(a.second).read_text())
    if f["cohort"] != "DEV" or s["cohort"] != "DEV" or sorted(f["per_context"]) != sorted(s["per_context"]):
        raise SystemExit("both apply files must be DEV on the same contexts")
    if f["bank_sha256"] != s["bank_sha256"] or f["ckpt_sha256"] == s["ckpt_sha256"]:
        raise SystemExit("the two reads must share the bank and differ in the checkpoint")
    pf, ps = f["per_context"], s["per_context"]
    cids = sorted(pf)
    if any(abs(pf[c]["B"] - ps[c]["B"]) > 0 for c in cids):
        raise SystemExit("the baseline values differ between the reads")
    fam_of = {c: pf[c]["family"] for c in cids}
    vals = {"B": {c: pf[c]["B"] for c in cids}, "I": {c: pf[c]["I"] for c in cids}, "I24": {c: ps[c]["I"] for c in cids}}
    pairs = (("I24", "B"), ("I24", "I"), ("I", "B"))
    aud = json.loads(Path(a.audit).read_text())["per_context"]
    keep = [c for c in cids if aud[c]["max_share_in_one_fit_icl_table"] < FLAG_SHARE]
    res = {"schema": "jope.h424.readout2/v1", "cohort": "DEV", "n": len(cids), "role": "second read, reported; no new verdict",
           "ckpt_sha256": {"I": f["ckpt_sha256"], "I24": s["ckpt_sha256"]}, "columns_ok": {"I": f["columns_ok"], "I24": s["columns_ok"]},
           "family_mean_u": {x: float(H.family_mean(vals[x], fam_of)) for x in vals},
           "pairs": {f"{x}_vs_{y}": T16.pair_stats(np, H, vals[x], vals[y], fam_of) for x, y in pairs},
           "secondary_clipped_excluded": {f"{x}_vs_{y}": S._excl(np.array([vals[x][c] - vals[y][c] for c in cids])) for x, y in pairs},
           "without_flagged": {"n": len(keep), "flag_share": FLAG_SHARE,
                               **{f"{x}_vs_B": T16.pair_stats(np, H, {c: vals[x][c] for c in keep}, {c: vals["B"][c] for c in keep},
                                                              {c: fam_of[c] for c in keep}) for x in ("I", "I24")}}}
    arms, point, boot, elo = T16.elo_boot(np, H, vals, fam_of, "B", BOOT_SEED, N_BOOT)
    res["delta_elo"] = {f"{x}_vs_{y}": T16.delta_elo(np, arms, point, boot, x, y) for x, y in pairs}
    ok = [c for c in cids if ps[c]["column_ok"]]
    wa = np.array([ps[c]["w_added_I"][0] for c in ok])
    res["column_I24"] = {"contexts_with_weight": int((wa > 0).sum()), "mean_weight": float(wa.mean()),
                         "alone_beats_B": int(sum(ps[c]["alone_ridgeres"] > ps[c]["B"] for c in ok)),
                         "alone_minus_B_mean": float(np.mean([ps[c]["alone_ridgeres"] - ps[c]["B"] for c in ok])),
                         "alone_I24_minus_alone_I_mean": float(np.mean([ps[c]["alone_ridgeres"] - pf[c]["alone_ridgeres"] for c in ok if pf[c]["column_ok"]]))}
    print(f"== H424 second read, DEV n {res['n']}: mean u " + "  ".join(f"{x} {res['family_mean_u'][x]:.4f}" for x in vals)
          + f"   column I24: {res['column_I24']}")
    for k, p in list(res["pairs"].items()) + [(f"{x}_vs_B w/o flagged ({len(keep)})", res["without_flagged"][f"{x}_vs_B"]) for x in ("I", "I24")]:
        e = res["delta_elo"].get(k)
        x = res["secondary_clipped_excluded"].get(k)
        print(f"   {k:28s} gain {p['gain']:+.4f} u clipped {p['clipped_mean']:+.4f} p {p['signflip_p']:.3g} lofo_min {p['lofo_min']:+.4f} "
              f"median {p['family_median']:+.4f} W/T/L {p['wtl']} pass {p['pass']}"
              + (f" dElo {e['delta']:+.1f} [{e['ci95'][0]:+.1f}, {e['ci95'][1]:+.1f}]" if e else "")
              + (f" | clipped excluded: {x['wins']}/{x['ties']}/{x['losses']} share {x['win_share_of_decided']}" if x else ""))
    with open(a.out, "x") as fh:
        json.dump(res, fh, sort_keys=True)


if __name__ == "__main__":
    main()
