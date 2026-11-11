#!/usr/bin/env python3
"""H429 readout: the test fixed in PREREG section 4 (a DEV development read).

  readout_h429.py --h424-apply FILE --h429-apply FILE --bank FILE --screen SCREEN_DEV.json --fallback FILE --out FILE

Arms per DEV context (value = AUDIT u): B = stored stack; I = H424's stored arm (stack + H424's column);
IT = stack + the column as rebuilt here (the T1 readout for tables with more than 30 features, H424's stored column
for the others, so IT = I there). Units = H426's `feature` clusters from the frozen H425 screen.
Primary: IT vs I on the 35 wide tables, h335_stats.primary_test (clipped unit-mean gain above 0, one-sided sign-flip
p < 0.025, leave-one-unit-out minimum above 0, median not negative).
  WIDE_CONFIRMED  the test passes
  WIDE_HARM       clipped gain below 0 with mirror p < 0.025
  WIDE_NULL       anything else
A wide context listed in --fallback (no T1 column before the budget ended) kept H424's column: IT = I, gain 0, counted.
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

N_WIDE, BOOT_SEED, N_BOOT = 35, 429150, 10_000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h424-apply", required=True)
    ap.add_argument("--h429-apply", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--screen", required=True)
    ap.add_argument("--fallback", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import train_h416 as T16

    _, H = T16._stats_mod()
    a4, a9 = json.loads(Path(a.h424_apply).read_text()), json.loads(Path(a.h429_apply).read_text())
    p4, p9 = a4["per_context"], a9["per_context"]
    cids = sorted(p4)
    if a4["cohort"] != "DEV" or a9["cohort"] != "DEV" or sorted(p9) != cids or a4["bank_sha256"] != a9["bank_sha256"]:
        raise SystemExit("the two value files are not the same DEV contexts and bank")
    if a4["ckpt_sha256"] != a9["ckpt_sha256"]:
        raise SystemExit("the two reads must use the same checkpoint")
    z = np.load(a.bank, allow_pickle=False)
    Fn = {str(c): int(s[1]) - 1 for c, s in zip(z["cids"], z["s_shape"])}
    wide = sorted(c for c in cids if Fn[c] > 30)
    if len(wide) != N_WIDE:
        raise SystemExit("the wide DEV contexts are not the 35 expected")
    fallback = [c for c in json.loads(Path(a.fallback).read_text())["fallback"]]
    if not set(fallback) <= set(wide):
        raise SystemExit("a fallback context is not a wide one")
    narrow = [c for c in cids if c not in wide]
    screen = json.loads(Path(a.screen).read_text())
    cl = R26.clusterings(screen, cids)["feature"]
    B = {c: p4[c]["B"] for c in cids}
    I = {c: p4[c]["I"] for c in cids}
    IT = {c: p9[c]["I"] for c in cids}
    if max(abs(p9[c]["B"] - B[c]) for c in cids) > 0:
        raise SystemExit("the baseline values differ between the reads")
    narrow_diff = max(abs(IT[c] - I[c]) for c in narrow)
    fb_diff = max([abs(IT[c] - I[c]) for c in fallback], default=0.0)

    def st(x, y, sub, unit):
        s = T16.pair_stats(np, H, {c: x[c] for c in sub}, {c: y[c] for c in sub}, {c: unit[c] for c in sub})
        return {"units": len({unit[c] for c in sub}), "contexts": len(sub), **{k: s[k] for k in
                ("gain", "clipped_mean", "signflip_p", "mirror_signflip_p", "lofo_min", "family_median", "wtl", "pass")}}

    own = {c: c for c in cids}
    prim = st(IT, I, wide, cl)
    res = {"schema": "jope.h429.readout/v1", "cohort": "DEV (development read)", "wide": len(wide), "narrow": len(narrow),
           "computed": len(wide) - len(fallback), "fallback_kept_h424_column": len(fallback),
           "narrow_max_abs_diff_vs_h424": float(narrow_diff), "fallback_max_abs_diff_vs_h424": float(fb_diff),
           "ckpt_sha256": a9["ckpt_sha256"],
           "primary_IT_vs_I_wide_clusters": prim,
           "reported": {"IT_vs_I_wide_contexts": st(IT, I, wide, own), "IT_vs_B_wide_clusters": st(IT, B, wide, cl),
                        "I_vs_B_wide_clusters": st(I, B, wide, cl), "IT_vs_B_all_clusters": st(IT, B, cids, cl),
                        "I_vs_B_all_clusters": st(I, B, cids, cl), "IT_vs_I_all_clusters": st(IT, I, cids, cl)}}
    arms, point, boot, _ = T16.elo_boot(np, H, {"IT": {c: IT[c] for c in wide}, "I": {c: I[c] for c in wide}},
                                        {c: cl[c] for c in wide}, "I", BOOT_SEED, N_BOOT)
    res["delta_elo_IT_vs_I_wide"] = T16.delta_elo(np, arms, point, boot, "IT", "I")
    w4 = np.array([p4[c]["w_added_I"][0] if p4[c]["column_ok"] else 0.0 for c in wide])
    w9 = np.array([p9[c]["w_added_I"][0] if p9[c]["column_ok"] else 0.0 for c in wide])
    al4 = np.array([p4[c]["alone_ridgeres"] for c in wide])
    al9 = np.array([p9[c]["alone_ridgeres"] for c in wide])
    oo4 = np.array([p4[c]["oof_u_ridgeres"] for c in wide])
    oo9 = np.array([p9[c]["oof_u_ridgeres"] for c in wide])
    bs = np.array([p4[c]["oof_u_best_stored"] for c in wide])
    res["column_wide"] = {"weight_mean_h424": float(w4.mean()), "weight_mean_T1": float(w9.mean()),
                          "used_h424": int((w4 > 0).sum()), "used_T1": int((w9 > 0).sum()),
                          "alone_audit_u_mean_h424": float(al4.mean()), "alone_audit_u_mean_T1": float(al9.mean()),
                          "alone_audit_T1_better": int((al9 > al4).sum()),
                          "oof_u_minus_best_stored_mean_h424": float((oo4 - bs).mean()), "oof_u_minus_best_stored_mean_T1": float((oo9 - bs).mean()),
                          "oof_u_T1_better": int((oo9 > oo4).sum())}
    bins = {}
    for name, sel in (("F 31-100", [c for c in wide if Fn[c] <= 100]), ("F > 100", [c for c in wide if Fn[c] > 100])):
        g = np.array([IT[c] - I[c] for c in sel])
        bins[name] = {"n": len(sel), "mean_gain_IT_minus_I": float(g.mean()), "wins": int((g > 1e-4).sum()), "losses": int((g < -1e-4).sum()),
                      "alone_audit_T1_minus_h424": float(np.mean([p9[c]["alone_ridgeres"] - p4[c]["alone_ridgeres"] for c in sel]))}
    res["by_width"] = bins
    if prim["clipped_mean"] < 0 and prim["mirror_signflip_p"] < 0.025:
        res["verdict"] = "WIDE_HARM"
    elif prim["pass"]:
        res["verdict"] = "WIDE_CONFIRMED"
    else:
        res["verdict"] = "WIDE_NULL"
    print(f"== H429 DEV development read: wide {res['wide']} (computed {res['computed']}, fallback {res['fallback_kept_h424_column']}), "
          f"narrow unchanged (max diff {narrow_diff:.1e})")
    rows = [("PRIMARY IT vs I, wide, clusters", prim)] + list(res["reported"].items())
    for k, s in rows:
        print(f"   {k:32s} units {s['units']:3d} gain {s['gain']:+.4f} clipped {s['clipped_mean']:+.4f} p {s['signflip_p']:.3g} "
              f"lofo_min {s['lofo_min']:+.4f} median {s['family_median']:+.4f} W/T/L {s['wtl']} pass {s['pass']}")
    e = res["delta_elo_IT_vs_I_wide"]
    print(f"   dElo IT vs I (wide) {e['delta']:+.1f} [{e['ci95'][0]:+.1f}, {e['ci95'][1]:+.1f}]  column {res['column_wide']}  by width {bins}")
    print(f"   verdict {res['verdict']}")
    with open(a.out, "x") as f:
        json.dump(res, f, sort_keys=True)


if __name__ == "__main__":
    main()
