# H129 — GBM-prior blend: replication, lambda 0.25 and a fit-fold harm gate vs r882

**Status: preregistered (CPU); no scoring yet.** [PREREG.md](PREREG.md) is committed and published before any
pool, checkpoint or feature payload is opened for scoring.

## Hypothesis

H124's halfway blend of the GBM context prior toward r882 gained +0.032 U4 (LCB95 +0.017) but harmed 10.7% of
families, against a 10% cap. H129 asks, on a **new** family fold (RNG 129029) with **new** seeds 91–93:

- Does the λ = 0.5 blend replicate?
- Does a smaller λ = 0.25 keep a gain with a thinner harm tail?
- Does gating the λ = 0.5 blend to contexts where the GBM and r882 rankings agree bring harm under the cap? The gate
  threshold is chosen per outer fold from fit-fold predictions only.

All three primaries face the same gates against r882: median gain > 0, Holm-adjusted one-sided bootstrap test
against +0.001 (m = 3), and harm ≤ 10%. Elo against r882 = 1500 is reported with 95% CIs; AUC is not used.

## Implementation

[runner.py](runner.py) reuses the merged H124 module, byte-pinned, for data loading, fitting, the r882 reference pass
and the RSS watchdog. It adds:
- the arms;
- the outcome-free disagreement statistic;
- the fit-fold threshold rule;
- the Holm readout;
- a metrics-size check that runs before writing.

[test_h129.py](test_h129.py) has 8 synthetic controls and reads no data. `--cpu-dry-run` is data-free.

## Plan

One run of 45 fits on xbabe2 at nice 15 with 4 threads; the H124 run took about 2 h 15 min. Closeout gets the
README outcome, a compact `metrics.json` and the index row. The evidence goes to S3 and `jope-rescue`.
