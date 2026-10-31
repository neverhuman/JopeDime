# H129 — GBM-prior blend: replication, lambda 0.25 and a fit-fold harm gate vs r882

**Status: complete. Verdict: CANDIDATE: BLEND_025.** This is a candidate breakthrough awaiting overseer red-team.
It is not a promotion, and nothing has touched the live flywheel.

BLEND_025 (r882 + 0.25·(GBM − r882)) passes every preregistered gate against r882 on a fresh family fold with new
seeds:
- median seed gain **+0.0155** (95% CI [+0.0065, +0.0273]), one-sided LCB95 +0.0078;
- Holm p = 0.0003 (m = 3);
- harm **12/205 = 5.9%**;
- median-seed Elo **1523.5 [1512.9, 1532.8]** against r882 = 1500.

BLEND_050 replicates H124's gain and also its harm failure (11.7%). HARM_GATED_050 fails Holm.

## Outcome (205 families, 1,219 DL-Train contexts, seeds 91–93, fold 129029; complete 45-fit panel)

| Arm vs r882 | Seed means 91 / 92 / 93 | Median gain [95% CI] | LCB95 | Holm p | Harm (≤ 10%) | Gate |
|---|---|---|---:|---:|---:|---|
| **BLEND_025** | +0.0154 / +0.0155 / +0.0165 | **+0.0155** [+0.0065, +0.0273] | +0.0078 | 0.0003 ✓ | **12 = 5.9%** ✓ | **ADMITTED** |
| BLEND_050 | +0.0256 / +0.0312 / +0.0261 | +0.0261 [+0.0088, +0.0487] | +0.0111 | 0.0009 ✓ | 24 = 11.7% ✗ | fail (harm) |
| HARM_GATED_050 | +0.0038 / +0.0056 / +0.0037 | +0.0038 [−0.0006, +0.0098] | +0.0003 | 0.088 ✗ | 10 = 4.9% | fail (Holm) |
| PURE_GBM (report-only) | +0.056 / +0.073 / +0.057 | +0.057 [+0.028, +0.095] | +0.033 | — | 38 = 18.5% | — |

| Arm | Median-seed Elo vs r882 = 1500 [95% CI] | Per seed 91 / 92 / 93 | Pooled-score Elo (additional) |
|---|---|---|---|
| **BLEND_025** | **1523.5 [1512.9, 1532.8]** | 1524.4 / 1521.0 / 1523.5 | 1525.6 [1513.5, 1537.5] |
| BLEND_050 | 1526.5 [1512.3, 1540.2] | 1526.6 / 1526.5 / 1524.4 | 1534.0 [1518.2, 1550.0] |
| HARM_GATED_050 | 1512.3 [1503.1, 1519.6] | 1514.0 / 1512.3 / 1507.4 | 1521.2 [1510.0, 1532.7] |
| PURE_GBM | 1528.0 [1510.1, 1551.1] | 1525.7 / 1539.7 / 1528.0 | 1558.6 [1534.1, 1584.6] |

Across seeds and outer folds, the HARM_GATED gate chose its fit-fold quantile as 0.25 twelve times, 0.5 twice and
0.75 once. It opened for 32–37% of contexts.

## Reading

- **There is a dose response in λ.**
  - Harm rises with the GBM weight: 5.9% at λ = 0.25, 11.7% at 0.5, 18.5% at 1.0. Gain rises too: +0.016, +0.026,
    +0.057.
  - At λ = 0.25 the blend keeps about a quarter of the pure-GBM gain while the harm tail falls under the cap.
  - BLEND_050's harm failure now replicates across two independent folds and seed sets (H124 10.7%, H129 11.7%).
- **The disagreement gate is too conservative.** It cuts harm (4.9%) but also most of the gain, and it fails Holm.

## Red-team (my own; independent overseer red-team pending)

- **Gates as written.** No rounding, re-thresholding or harm redefinition. BLEND_025 clears every gate by a margin;
  none is a boundary tie.
- **Cherry-picking.** Three primaries were preregistered with Holm, and only one is admitted. λ = 0.25 was fixed in
  the t1140 design before any H129 outcome; H124 never tested it.
- **Leakage.**
  - Family folds are disjoint: 0 families appear in more than one fold.
  - The GBM only sees other families.
  - BLEND_025 uses no gate. HARM_GATED's τ is chosen per outer fold from inner fit-fold predictions only; every
    choice is in `gate_receipts.json`.
  - r882 has earlier exposure to these DL-Train families. This is conditional development evidence, and the
    comparator's in-sample exposure, if anything, favours r882.
- **Seed independence (the H105 lesson).**
  - Per-seed family W/T/L for BLEND_025 differ: 106/53/46, 97/57/51 and 101/53/51.
  - All 45 prediction SHAs are distinct.
  - The seeds share fold, samples and r882, so family deltas correlate at 0.89–0.94 across seeds. The inferential
    basis is the 205-family bootstrap, not three independent replicates.
- **Metric direction.** U4 utility is higher-is-better, and the gain is arm minus r882. 28 families tie in every
  seed. The worst BLEND_025 families are −0.21 and −0.14; they belong in the independent red-team.
- **Ties.** A tie or LCB ≤ 0.001 would be NO_ADVANCE. Neither applies to BLEND_025.
- **Data and frozen state.**
  - 0 DL-Val, sealed, holdout or meta-test payloads, 0 DL-Val cache rows decoded, and no live-loop path.
  - DOPE was `c51a56d0` before and after: the frozen pass ran in the first attempt, and its receipt is SHA-pinned and
    verified at resume.
  - No GPU.
- **Run integrity.**
  - The run was aborted once by the host-disk guard before any outcome existed (see Amendment A1 below).
  - The one authorized resume verified every reused file against the pinned manifest and exited 0.
  - Peak RSS was 3.16 GB on the resume and 4.5 GB on the first attempt, both under the 6 GiB limit.
  - `metrics.json` is 20.6 KB and was size-checked before it was written.

## Next step

Hold for overseer red-team. If it clears, the owner decides on a prospective confirmation: a fresh family fold, new
seeds, and BLEND_025 as the single primary. Only after that would a live-loop ablation be considered.

## Evidence

- **Compact metrics.** [metrics.json](metrics.json).
- **Per-family arrays.** `family_arrays.json`, sha256 recorded in the metrics.
- **Gate receipts, fit receipts and logs.**
  - S3: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H129/opus-h129-20261031a-{partial,final}/`,
    103 objects, each written create-only with a version-pinned readback.
  - Rescue copies: `/home/ubuntu/jope-rescue-20260921/h129-rd_opus-20261031-{partial,final}/`.
- **Candidate receipt.** `~/jope_plan/ns/receipts/H129_WIN.md`.


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
