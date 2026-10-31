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

## Amendment A1: resume after host-disk abort

Owner decision (a), hwave overseer `rd_opus_t1340`, 19:40Z. This amendment is committed before any resumed fit and
before any H129 outcome exists.

- **What happened.**
  - The first attempt, `run-20261031T1749Z` (source `3afe028a3`, preregistration `051107312`, merged as
    `ede308c4a`), stopped at the preregistered disk guard.
  - The host filesystem reached 94.0% before fit 30 of 45 (`seed92-exclude12`). Other processes caused the fill.
  - 29 fits had completed. No arm outcome, readout or verdict had been computed, and none has been looked at since.
- **What A1 allows.**
  - Exactly **one resume of that same run root**, under the same fold (129029), sampling RNG, seeds 91–93, arms,
    gate rule, gates, bootstrap RNGs and 45-fit design.
  - The PREREG.md bytes are unchanged; the runner still verifies them against `051107312`.
- **Reuse only by pinned receipts.**
  - The aborted attempt's manifest is pinned by SHA-256 `35404656…`, the copy in S3 at
    `H129/opus-h129-20261031a-partial/`.
  - Before anything runs, every file it lists is verified against its pinned SHA-256: the 29 fit predictions and
    receipts, `split.json`, the r882 reference costs and the DOPE receipt.
  - The resume rebuilds the split and samples deterministically. The split must be byte-identical to the pinned copy.
  - Each reused fit's prediction SHA must equal both its own receipt and the manifest.
  - The r882 reference costs of the aborted attempt are reused from their pinned file; the frozen DOPE pass is not
    repeated. DOPE `c51a56d0` is recorded before and after in that attempt's receipt.
- **The 16 remaining fits run unchanged** with the merged H124 module. The readout runs only after all 45 exist.
- **Run limits.**
  - CPU only, at nice 19 with at most 4 threads.
  - RSS ≤ 6 GiB, MemAvailable ≥ 30 GiB, and the disk guard still requires use below 94%.
  - The resume writes `resume-identity.json`. The runner refuses a second resume and refuses any root whose files
    differ from the manifest.
- **If any guard trips again, H129 is recorded as INCOMPLETE (option c), with no further retries.**
