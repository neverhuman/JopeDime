#!/usr/bin/env python3
"""H425 sensitivity read: H424's I vs B on DEV with the frozen exclusion list removed. No new verdict; no table, model
or stack is touched: the per-context values are the ones H424 wrote (its apply files).

  sens_h425.py --exclusion EXCLUSION_DEV.json --apply H424_APPLY.json [--apply2 H424_SECOND_APPLY.json] --out FILE

Rows of the table (h335_stats.primary_test statistics of I vs B on each subset):
  all                 the 115 DEV contexts (H424's primary, restated)
  without_copy        tier "copy" removed
  without_feature     tiers "copy" and "possible" removed            <- the pre-declared sensitivity read
  without_any         also tier "target_only" removed (the most conservative list)
  only_feature        the removed contexts of the pre-declared read, on their own (descriptive)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def table(per: dict, tiers: dict, T16, H) -> dict:
    cids = sorted(per)
    sets = {"all": cids,
            "without_copy": [c for c in cids if tiers[c] != "copy"],
            "without_feature": [c for c in cids if tiers[c] not in ("copy", "possible")],
            "without_any": [c for c in cids if tiers[c] == "clear"],
            "only_feature": [c for c in cids if tiers[c] in ("copy", "possible")]}
    out = {}
    for name, sub in sets.items():
        row = {"n": len(sub)}
        if len(sub) >= 5:
            a, b = {c: per[c]["I"] for c in sub}, {c: per[c]["B"] for c in sub}
            row.update(T16.pair_stats(np, H, a, b, {c: per[c]["family"] for c in sub}))
        out[name] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exclusion", required=True)
    ap.add_argument("--apply", required=True)
    ap.add_argument("--apply2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import train_h416 as T16

    _, H = T16._stats_mod()
    ex = json.loads(Path(a.exclusion).read_text())
    tiers = {c: v["tier"] for c, v in ex["contexts"].items()}
    res = {"schema": "jope.h425.sensitivity/v1", "role": "sensitivity read of H424; no new verdict",
           "exclusion_sha256": hashlib.sha256(Path(a.exclusion).read_bytes()).hexdigest(),
           "tier_counts": {t: sum(1 for v in tiers.values() if v == t) for t in ("copy", "possible", "target_only", "clear")}, "reads": {}}
    for label, path in (("first_read_h020", a.apply), ("second_read_h024", a.apply2)):
        if not path:
            continue
        ap_ = json.loads(Path(path).read_text())
        if ap_["cohort"] != "DEV" or sorted(ap_["per_context"]) != sorted(tiers):
            raise SystemExit(f"{path}: not the DEV contexts of the exclusion list")
        res["reads"][label] = {"apply_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                               "ckpt_sha256": ap_["ckpt_sha256"], "table": table(ap_["per_context"], tiers, T16, H)}
        print(f"== {label}")
        for name, r in res["reads"][label]["table"].items():
            if "clipped_mean" in r:
                print(f"   {name:16s} n {r['n']:3d} gain {r['gain']:+.4f} clipped {r['clipped_mean']:+.4f} p {r['signflip_p']:.3g} "
                      f"lofo_min {r['lofo_min']:+.4f} median {r['family_median']:+.4f} W/T/L {r['wtl']} pass {r['pass']}")
            else:
                print(f"   {name:16s} n {r['n']:3d}")
    with open(a.out, "x") as f:
        json.dump(res, f, sort_keys=True)


if __name__ == "__main__":
    main()
