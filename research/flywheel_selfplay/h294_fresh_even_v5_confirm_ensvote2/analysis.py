"""H294 statistics (hw_t0158): ONE primary, ENS_VOTE2 (a deterministic single policy; its 3 "seeds" are bootstrap seeds
294150/294151/294152 on the same picks), against r3175's own slot-0 pick on fresh EVEN cohort v5, at one-sided alpha 0.05.

Per-seed statistics come from H292's analysis.summarize (sha-pinned, unchanged): family-equal mean gain, a 10,000-draw
family bootstrap, one-sided exact sign test on non-tied families (ties |d| <= 1e-4 never wins), harmed share, invalid
counts, Elo. The H292 rule is kept, except that the invalid-pick clause is replaced by GOV's pre-declared
NON-INFERIORITY clause (decided after H292 and before any EVEN-v5 data, so it only counts on this new cohort):

CONFIRMED_FRESH iff ALL hold:
  (1) one-sided 95% bootstrap LCB (the 0.05 quantile) of the family-mean gain > 0 on 3/3 seeds;
  (2) median across seeds of the family-mean gain > +0.010;
  (3) one-sided sign test p < 0.05 on 3/3 seeds;
  (4) harmed families (gain < -0.01) <= 35% on every seed;
  (5) invalid non-inferiority on every seed: (arm_invalid - slot0_invalid) <= max(2, ceil(0.01 * n_families)) AND the
      exact one-sided McNemar test on discordant invalid picks (b = arm invalid & slot 0 valid, c = slot 0 invalid & arm
      valid; p = P(X >= b | b + c, 1/2)) gives p >= 0.05;
  (6) both nulls fail (1)-(5).
Otherwise NOT_CONFIRMED_FRESH; fewer than 100 families built -> UNDERPOWERED (no verdict).
Secondary arms (ENS_VOTE2+HARMVETO, CLIP, BAG_SWITCH) are reported under the same rule but cannot flip the verdict."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "h292_analysis_base",
    HERE.parent / "h292_fresh_v4_odd_confirm_ensvote2_clip" / "analysis.py",
)
base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = base
_spec.loader.exec_module(base)

ALPHA = 0.05
BOOT_SEED = 294150
DETERMINISTIC_BOOT_SEEDS = (294150, 294151, 294152)
VALID_NULL_SEEDS = (294101, 294102, 294103)
NOLABEL_NULL_SEEDS = (294201, 294202, 294203)
MEDIAN_MIN = 0.010
HARM_MAX_SHARE = 0.35
MIN_FAMILIES = 100
PRIMARY = "ENS_VOTE2"

summarize = base.summarize
public = base.public
source_group = base.source_group
sign_test_one_sided = base.sign_test_one_sided
random_valid_slot_picks = base.random_valid_slot_picks
random_slot_no_label_picks = base.random_slot_no_label_picks


def invalid_noninferiority(inv, n_families):
    """inv: (3, contexts, 2) booleans [slot0, arm]. Per seed: excess, tolerance, McNemar discordants and p."""
    tol = max(2, math.ceil(0.01 * n_families))
    out = []
    for v in np.asarray(inv, dtype=bool):
        s0, arm = v[:, 0], v[:, 1]
        b, c = int((arm & ~s0).sum()), int((s0 & ~arm).sum())
        p = sign_test_one_sided(
            b, c
        )  # P(X >= b | n = b + c, 1/2): the arm is worse on invalid picks
        excess = int(arm.sum()) - int(s0.sum())
        out.append(
            dict(
                slot0_invalid=int(s0.sum()),
                arm_invalid=int(arm.sum()),
                excess=excess,
                tolerance=tol,
                mcnemar_b=b,
                mcnemar_c=c,
                mcnemar_p=p,
                ok=bool(excess <= tol and p >= ALPHA),
            )
        )
    return out


def criteria(s, inv, n_families):
    lcb = [float(np.quantile(b, ALPHA)) for b in s["_B"]]
    ni = invalid_noninferiority(inv, n_families)
    return dict(
        lcb_one_sided_95=lcb,
        lcb_positive_3of3=bool(all(x > 0 for x in lcb)),
        median_above_min=bool(s["median_gain_across_seeds"] > MEDIAN_MIN),
        sign_test_3of3=bool(all(p < ALPHA for p in s["sign_test_p_by_seed"])),
        harmed_share_ok=bool(
            all(h <= HARM_MAX_SHARE for h in s["harmed_share_by_seed"])
        ),
        invalid_noninferior=bool(all(x["ok"] for x in ni)),
        invalid_detail=ni,
        strict_invalid_not_above_slot0_descriptive=bool(
            all(x["excess"] <= 0 for x in ni)
        ),
    )


def passes(c):
    return all(
        c[k]
        for k in (
            "lcb_positive_3of3",
            "median_above_min",
            "sign_test_3of3",
            "harmed_share_ok",
            "invalid_noninferior",
        )
    )


def verdict(n_families, arm_c, null_cs):
    if n_families < MIN_FAMILIES:
        return "UNDERPOWERED"
    return (
        "CONFIRMED_FRESH"
        if passes(arm_c) and not any(passes(c) for c in null_cs)
        else "NOT_CONFIRMED_FRESH"
    )
