# H421 PREREG addendum 2 — the first selection run is void (numerical error); corrected utility and a clipped objective

**Status.** Published on `origin/v1/next` before the corrected selection runs. No DEV or CONFIRM context has been
read or evaluated. The fill (addendum 1) is complete and is not repeated.

## 1. What happened
- **Fill, 00:57–01:19Z:** 291 design contexts, 23,107 worker runs (all SEARCH evaluations), 15,650 memo hits, 14.6
  worker-hours (cap 60), 4 timeouts, no budget stop, 0 guard refusals, every stored stack candidate found in the memo.
- **Selection, 01:21Z (published code of addendum 1):** it reported a gate pass with a family-mean difference of
  +1.15 u, a median of 0 and a baseline mean utility of 5.62. Those values are impossible; the run is **void**.
  Its output is kept (`results/selection`, marked void) and is not used.

## 2. Cause
1. **Numerical error (my bug).** The selection computed the blend's squared error from Gram matrices
   (wᵀGw − 2wᵀc + yᵀy). On contexts that are fitted almost exactly this difference of large numbers is rounding
   noise; where it came out negative it was clamped, giving utilities near 350. Recomputed from the predictions,
   6 of the 291 contexts were wrong by more than 0.01 u, the worst by 334 u; the baseline mean utility is 2.21, not
   5.62.
2. **A weakness of the registered objective that the bug exposed.** Even computed correctly, 9 design contexts have
   a baseline utility above 10 (near-exact fits). An unclipped family mean lets those few contexts decide the
   selection: with the void selection's presets, recomputed correctly, 6 contexts moved by more than 0.5 u (the
   largest by +3.56 u), which is most of an unclipped mean gain of +0.044 u against a median of 0.

**What I looked at before writing this:** only the void selection's presets, recomputed from the predictions on the
design contexts (the numbers above, and: clipped mean gain over the baseline +0.017 u; against the random set
+0.016 u, 119 wins, 113 ties, 59 losses). No corrected selection had been run.

## 3. Corrections (the rules from here on)
1. **Utilities are computed from the predictions**: RMSE of (Q w − y), with the Caruana weights from the stored
   `meta_h416.kcar` itself. The Gram shortcut is removed.
2. **The objective is the family mean of the clipped gain** over the context's own baseline:
   gain_c(S) = clip(u_c(S) − u_c(no added preset), −0.5, +0.5). ±0.5 is the clip of this wave's pre-registered
   test statistic (`h335_stats.primary_test`), so the selection optimises what the DEV test measures and near-exact
   contexts cannot dominate it. The coverage ablation uses the same clip.
3. **The gate uses the same clip:** d_c = clip(u_c(LIB_PORT of the other half) − u_c(LIB_RAND), −0.5, +0.5); it
   passes when the family mean of d is above 0 with one-sided sign-flip p < 0.05 (100,000 flips, seed 421102). The
   unclipped mean is reported beside it.
4. Everything else in addendum 1 stands: candidate-set rule, pool, random set and seed, family halves, K = 20.

## 4. Honest caveats
- **This changes the gate statistic after a (void) gate result was seen.** The void result was a pass; the
  recomputed numbers in §2 also point to a pass. The change is made because the registered statistic is not
  robust, not to rescue a failure. The DEV test and its thresholds are untouched, and the random-preset control on
  DEV remains the check that selection, not preset count, matters.
- Predictions in the fill are stored in single precision, which limits utilities to about 16; with the clip this
  does not matter.
- The selection reads SEARCH rows of design contexts only; it cannot have used DEV.

## 5. Files
`select_h421.py` (rewritten), `stage_h421.sh` (stages this addendum). Other files are unchanged.
