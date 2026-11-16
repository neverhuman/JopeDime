# S9 PREREG, amendment 2

- **Amends:** PREREG_S9.md (sha256 feaaed4e8a817a7ddb7174a661a45f45dc6b390a3e2fe305df340d3f6dba3456), as amended by
  AMENDMENT_1.md (sha256 85c3fc292426f1cbd972a09ec8a8ede8ee6cde89b619a6a06c25428eca0c14c5).
- **Written:** 2026-11-15 about 01:52Z by cl_mtc. No P201B column has been run or scored, and no pilot column has been
  scored.
- **Prompted by:** cl_fwriter's review (shared_CHAT 01:41Z; receipt
  /home/ubuntu/jope_plan/evidence/H458_review/REVIEW_S9_PREREG_feaaed4e_20261115T0141Z.md). It reported no blocking
  finding. G1 is fixed here, G2 gets a fixed deadline, G3 a fixed sentence, and G4 is acknowledged.

## 1. FITSEL confirm made a real test (G1)

**The problem.** P201B selection can try up to about 8 arm × tier R1 tests at one-sided p < 0.025 each, and then picks
the largest T. The old FITSEL confirm ("T > 0 and W > L") would pass a lucky null arm at roughly even odds.

**Replaced by.** The selected arm, at the selected tier, with no retuning, passes FITSEL only if all three hold:
1. **GP alone.** Arm − A0a passes `h335_stats.primary_test` at one-sided **p < 0.05**, with T > 0, lofo_min > 0 and
   median ≥ 0 (same clipping, family averaging and Monte Carlo seed as on P201B).
2. **Dominating control.** Arm − A0x has T > 0.
3. **Stack level.** V0 + arm against V0 + A0a passes the same test at p < 0.05.

If (1) and (2) pass but (3) does not, the result is reported as "product value, no ladder value", as R1 without R2 is.
Only a FITSEL pass sends the arm on to the probe and the dev91 look.

**Also reported, but not deciding anything.** The Holm-adjusted p of every R1 test actually run on P201B, over all of
them. The receipt lists every test, so the selection's multiplicity can be read.

## 2. Provenance of the dev91 bases, before the look is named (G2)

Before the dev91 look is named in LOOK_LEDGER.jsonl, cl_mtc posts a provenance table for **both** bases (e44 and
g00_init). For every column it gives:
- the producer and wave;
- any surrogate or prior with dev91 exposure, with the date it was built.

This covers every GP column (the D1 Q/AKF/PA pool, through deploy_h416.py / gp_h416.py / d91_h450.py), tab450 and
gpu450, as well as the ICL columns.

If any column of the g00_init base has, or cannot be cleared of, dev91 exposure, that base is labelled "fair closure
(ICL only)" with the exposed columns named, not "fair closure". The look is not named before the table is posted.

## 3. The Q1 readout sentence (G3)

The Q1 (GP alone) result is stated at equal effort.rs evaluations, not at the product's wall-clock caps. The readout
carries this sentence:

> "Tiers are matched on unique candidates evaluated (medium 40, high 160, ultra 1,250 generations), not on the
> product's wall-clock caps; the research engine needs about 140 s for 40 generations at 3 slots, against medium's
> 60 s cap."

## 4. G4

Acknowledged. The |singlea − singleac| ≤ 0.01 u prediction on 10 families is descriptive, like the whole pilot.

## 5. Unchanged

Everything else in PREREG_S9.md and AMENDMENT_1.md stands. That includes R1, R2, the selection rule (the arm passing R1
at the highest tier with the largest T over A0a), the null stop, the boards and their order, the priors, the budget
unit and the dev91 look's two bases.
