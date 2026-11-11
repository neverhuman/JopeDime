#!/usr/bin/env python3
"""H432 part 1: the in-context column on narrow tables only, restated from stored values (PREREG section 3).

  narrow_h432.py --apply H424_DEV.apply.json --apply24 H424_DEV_h024.apply.json --bank bank_DEV.npz
                 --screen SCREEN_DEV.json --out FILE [--selftest]

Arms per DEV context (value = AUDIT u; every number is a stored H424 value, nothing is refitted):
  B  the stored stack, no column (ablation)
  I  H424's arm: the column offered on every table (baseline)
  N  narrow-only: I where the table has at most 30 feature columns, B where it has more
Units = H426's `feature` clusters of the frozen H425 screen (110 units).
  retention R = clipped unit-mean gain of N over B / clipped unit-mean gain of I over B
  NARROW_KEEPS   R >= 0.90, N vs B one-sided sign-flip p < 0.05, and less inference CPU than I
  NARROW_LOSES   R < 0.90
  NARROW_UNCLEAR R >= 0.90 but p >= 0.05
--selftest replaces the stored values by seeded random numbers and writes nothing (code path only).
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

WIDE, CLIP, BOOT_SEED, N_BOOT = 30, 0.5, 432150, 10_000
THRESHOLDS = (10, 20, 30, 50, 100, None)   # None = the column on every table (= I)
KEYS = ("gain", "clipped_mean", "signflip_p", "mirror_signflip_p", "lofo_min", "family_median", "wtl", "pass")


def unit_gain(x, y, unit, cids) -> np.ndarray:
    """Per-unit mean gain, clipped to +-CLIP (the quantity whose mean primary_test reports)."""
    g = {}
    for c in cids:
        g.setdefault(unit[c], []).append(x[c] - y[c])
    return np.array([np.clip(np.mean(g[k]), -CLIP, CLIP) for k in sorted(g)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", required=True)
    ap.add_argument("--apply24", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--screen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    import train_h416 as T16

    _, H = T16._stats_mod()
    a20, a24 = json.loads(Path(a.apply).read_text()), json.loads(Path(a.apply24).read_text())
    p20, p24 = a20["per_context"], a24["per_context"]
    cids = sorted(p20)
    if a20["cohort"] != "DEV" or a24["cohort"] != "DEV" or sorted(p24) != cids or a20["bank_sha256"] != a24["bank_sha256"]:
        raise SystemExit("the two value files are not the same DEV contexts and bank")
    z = np.load(a.bank, allow_pickle=False)
    Fn = {str(c): int(s[1]) - 1 for c, s in zip(z["cids"], z["s_shape"])}
    if sorted(Fn) != cids:
        raise SystemExit("the bank does not hold exactly the contexts of the value files")
    cl = R26.clusterings(json.loads(Path(a.screen).read_text()), cids)["feature"]
    B = {c: float(p20[c]["B"]) for c in cids}
    I = {c: float(p20[c]["I"]) for c in cids}
    I24 = {c: float(p24[c]["I"]) for c in cids}
    if max(abs(p24[c]["B"] - B[c]) for c in cids) > 0:
        raise SystemExit("the baseline values differ between the two reads")
    if a.selftest:
        r = np.random.default_rng(0)
        B = {c: float(r.normal(1.9, 0.5)) for c in cids}
        I = {c: B[c] + float(r.normal(0.005, 0.03)) for c in cids}
        I24 = {c: B[c] + float(r.normal(0.005, 0.03)) for c in cids}

    def narrow(full: dict, thr) -> dict:
        return {c: (full[c] if (thr is None or Fn[c] <= thr) else B[c]) for c in cids}

    def st(x, y) -> dict:
        s = T16.pair_stats(np, H, x, y, cl)
        return {"units": len(set(cl.values())), "contexts": len(cids), **{k: s[k] for k in KEYS}}

    def retention(x, full) -> dict:
        gx, gf = unit_gain(x, B, cl, cids), unit_gain(full, B, cl, cids)
        r = np.random.default_rng(BOOT_SEED)
        idx = r.integers(0, len(gx), size=(N_BOOT, len(gx)))
        bx, bf = gx[idx].mean(1), gf[idx].mean(1)
        ok = bf > 0
        q = np.quantile(bx[ok] / bf[ok], [0.025, 0.975])
        return {"ratio": float(gx.mean() / gf.mean()), "boot_ci95": [float(q[0]), float(q[1])],
                "boot_share_below_0.90": float(np.mean(bx[ok] / bf[ok] < 0.90)), "boot_resamples_with_positive_full_gain": int(ok.sum()),
                "unit_mean_narrow": float(gx.mean()), "unit_mean_full": float(gf.mean())}

    def cpu(per: dict, thr) -> dict:
        t = {c: float(per[c]["infer_wall_s"]) for c in cids}
        keep = [c for c in cids if thr is None or Fn[c] <= thr]
        return {"tables_with_column": len(keep), "seconds_full": float(sum(t.values())), "seconds_narrow_only": float(sum(t[c] for c in keep)),
                "share_saved": float(1.0 - sum(t[c] for c in keep) / sum(t.values()))}

    N, N24 = narrow(I, WIDE), narrow(I24, WIDE)
    prim, full = st(N, B), st(I, B)
    ret = retention(N, I)
    check = abs(ret["unit_mean_narrow"] - prim["clipped_mean"]) + abs(ret["unit_mean_full"] - full["clipped_mean"])
    c20 = cpu(p20, WIDE)
    res = {"schema": "jope.h432.narrow/v1", "cohort": "DEV (development read, stored values)", "selftest": bool(a.selftest),
           "wide_threshold_features": WIDE, "tables": len(cids), "narrow_tables": int(sum(Fn[c] <= WIDE for c in cids)),
           "wide_tables": int(sum(Fn[c] > WIDE for c in cids)), "units": prim["units"],
           "N_vs_B": prim, "I_vs_B": full, "N_vs_I": st(N, I), "retention": ret, "unit_gain_check_abs_diff": float(check),
           "inference_cpu": {"hour20_read": c20, "hour24_read": cpu(p24, WIDE)},
           "hour24_column": {"N24_vs_B": st(N24, B), "I24_vs_B": st(I24, B), "N24_vs_I24": st(N24, I24), "retention": retention(N24, I24)},
           "threshold_sensitivity": []}
    for thr in THRESHOLDS:
        x = narrow(I, thr)
        s, r = st(x, B), retention(x, I)
        res["threshold_sensitivity"].append({"max_features": thr, "tables_with_column": cpu(p20, thr)["tables_with_column"],
                                             "clipped_gain": s["clipped_mean"], "signflip_p": s["signflip_p"], "wtl": s["wtl"],
                                             "retention": r["ratio"], "cpu_share_saved": cpu(p20, thr)["share_saved"]})
    R = ret["ratio"]
    lower_cpu = c20["seconds_narrow_only"] < c20["seconds_full"]
    if R < 0.90:
        res["verdict"] = "NARROW_LOSES"
    elif prim["signflip_p"] < 0.05 and lower_cpu:
        res["verdict"] = "NARROW_KEEPS"
    else:
        res["verdict"] = "NARROW_UNCLEAR"
    res["passes_campaign_rule_0.025"] = bool(prim["pass"])
    print(f"== H432 part 1{' SELFTEST (random values)' if a.selftest else ''}: {res['narrow_tables']} narrow + {res['wide_tables']} wide tables, {res['units']} units")
    for k, s in (("N vs B (narrow-only vs no column)", prim), ("I vs B (H424, column everywhere)", full), ("N vs I", res["N_vs_I"]),
                 ("N24 vs B (hour-24 column)", res["hour24_column"]["N24_vs_B"]), ("I24 vs B", res["hour24_column"]["I24_vs_B"])):
        print(f"   {k:36s} gain {s['gain']:+.4f} clipped {s['clipped_mean']:+.4f} p {s['signflip_p']:.3g} lofo_min {s['lofo_min']:+.4f} "
              f"median {s['family_median']:+.4f} W/T/L {s['wtl']} pass {s['pass']}")
    print(f"   retention {R:.3f} [{ret['boot_ci95'][0]:.3f}, {ret['boot_ci95'][1]:.3f}]  share of resamples below 0.90: {ret['boot_share_below_0.90']:.4f}"
          f"   hour-24 retention {res['hour24_column']['retention']['ratio']:.3f}")
    print(f"   inference CPU: full {c20['seconds_full']:.0f} s, narrow-only {c20['seconds_narrow_only']:.0f} s, saved {100 * c20['share_saved']:.1f}%"
          f" (hour-24 read: {100 * res['inference_cpu']['hour24_read']['share_saved']:.1f}%)")
    for t in res["threshold_sensitivity"]:
        print(f"   column up to {str(t['max_features']):>4s} features: tables {t['tables_with_column']:3d} clipped {t['clipped_gain']:+.4f} "
              f"p {t['signflip_p']:.3g} W/T/L {t['wtl']} retention {t['retention']:.3f} CPU saved {100 * t['cpu_share_saved']:.1f}%")
    print(f"   verdict {res['verdict']}")
    if a.selftest:
        return
    with open(a.out, "x") as f:
        json.dump(res, f, sort_keys=True)


if __name__ == "__main__":
    main()
