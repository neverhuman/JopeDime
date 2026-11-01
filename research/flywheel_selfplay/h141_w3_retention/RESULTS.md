# H141 W3 closeout: NO_ADVANCE

All 90 registered cells completed all 3,600 optimizer updates. Both legacy raw-utility primaries failed. The charged GPU total was **520.134 / 7,200 seconds** (490.134 measured lock-wall seconds plus a conservative 30-second startup allowance), with zero new native fits. DOPE remained bit-identical in every cell. The exclusive qualification lock was released and both owned GPU processes exited.

Preregistration [#1265](https://github.com/neverhuman/JopeDime/pull/1265) merged as `45514a5c56592ef468ac3c648e394773918e0692` before CUDA initialization. All ten owned source blobs match the CPU-tested source. The merger also updated only the README Python-file counter. A strict pre-CUDA guard initially refused that extra metadata path; it performed no CUDA/query/optimizer work, and the reviewed launch retained the same budget.

## Registered comparisons

These are conditional retained-cohort family means, averaging all three seeds within each family. Intervals use 10,000 paired family draws, seed20261031; the two primaries use one-sided alpha.025. The observed legacy panel contains 1,213 contexts across all205 families; six contexts remain unknown.

| Contrast | Mean delta | Lower bound | Upper bound | Families losing >.01 | Gate-failure delta | Pass |
|---|---:|---:|---:|---:|---:|---|
| c-b | -0.001137 | -0.002583 | -0.000068 | 4/205 | +0.0000 pp | NO |
| d-a | -0.045796 | -0.141333 | +0.038446 | 106/205 | +0.6821 pp | NO |
| d-d_scalar | -0.046789 | -0.142222 | +0.037427 | 108/205 | +0.6821 pp | NO |
| d-d_shuffled | +0.015574 | -0.016325 | +0.045709 | 66/205 | +0.0678 pp | NO |

d−a also loses the median-family and harm gates:106/205 families lose more than.01. Its gate-failure rate rises by0.6821 percentage points. d does not beat the matched scalar control.

U4 cannot rescue the failed legacy primary. On 1,143 contexts/205families, d−a has mean+0.016979 and lower bound+0.001133, but25/205families lose more than.01 (12.20%, above10%). Its gain is matched by shuffled targets: d−d_shuffled U4 mean−0.000025, lower bound−0.003038. Seventy-six contexts lack a complete four-new-candidate U4 outcome.

## Raw utility and family Elo

Legacy raw utility is pass=−fitness, otherwise−2. Context means and equal-family means are separate. Elo uses the pinned repository solver with r882=1500, prior.01, tie margin.0001 and400iterations. Each track has its own fixed complete contender panel on its declared observed contexts; their Elo scales are not interchangeable. Best-of-revealed is a history-only legacy baseline.

| Arm | Legacy context mean | Legacy family mean | Legacy Elo | U4 family mean | U4 Elo |
|---|---:|---:|---:|---:|---:|
| r882 | 1.364799 | 1.281879 | 1500.00 | 1.806160 | 1500.00 |
| ridge150 | 1.181591 | 1.135353 | 1462.06 | 1.797660 | 1476.35 |
| zero | 1.438280 | 1.427601 | 1543.04 | 1.799758 | 1515.45 |
| best_revealed | 1.796527 | 1.757112 | 1834.61 | — | — |
| random4 | — | — | — | 1.819748 | 1547.61 |
| a | 1.366527 | 1.280175 | 1499.16 | 1.806038 | 1500.24 |
| b | 1.367610 | 1.281522 | 1499.97 | 1.806127 | 1500.97 |
| c | 1.366634 | 1.280384 | 1499.71 | 1.806153 | 1501.23 |
| d | 1.308886 | 1.234379 | 1450.32 | 1.823017 | 1511.95 |
| d_scalar | 1.367529 | 1.281168 | 1500.24 | 1.806080 | 1501.23 |
| d_shuffled | 1.289772 | 1.218805 | 1450.35 | 1.823042 | 1509.18 |

All per-seed utilities/Elo, family harm counts, win/tie/loss counts and paired-family Elo intervals are retained in the exact-version readout artifacts. The bootstrap Elo implementation reproduces the unchanged repository solver for both point panels and six explicit resampled panels.

## Verification and limits

Independent CPU replay verified40,029 raw choices,34,290 U4 outcomes,90 checkpoint hashes and all15 paired-exposure groups. Every cell recorded40 updates, changed parameter tensors and unchanged DOPE hashes. Frozen H113 CPU loss/gradient parity,48 full-model zero-update checks, six real fit-only gradient cases and charged CPU/CUDA parity all passed. A final-source CPU replay also reproduced all1,280 original fit-only pilot rows without changing any recipe or endpoint.

This is the owner-authorized retained TRAIN cohort:31,952 earliest native completions from796 exact-version rounds, not the old1,404,287-slot H50 numerical pool. Forty-three contexts have no unrevealed training candidate after four history anchors. Training192-wide slates used explicitly registered replacement sampling; the top16 statistic is over sampled passing slots and may repeat programs. These are not192 distinct native fits.

r882 and the producer have TRAIN-family exposure; frozen priors/features retain their inherited exposure. The study is conditional development evidence. Six missing legacy contexts,76 missing U4 contexts and all original-pool unknown slots remain explicit. No full-original-panel, independent-confirmation, unbiased off-policy or serving/adoption claim is made.

Controls match backbone/head widths, data, slates and40updates; loss arithmetic and measured wall time differ. Mean cell wall seconds, excluding the initial setup/baseline interval, are: a=3.749, b=5.239, c=5.244, d=5.252, d_scalar=6.559, d_shuffled=5.149. This is not a claim of exact wall-time matching.

There were zero new DL-Val, outer, sealed or holdout payload reads. No owner sources or live-loop files were changed by this lane. No additional checkout/worktree was created. No adoption note is appropriate for NO_ADVANCE.

Exact-version source, input and result bindings are in `manifest.json`, `manifest_custody_binding.json` and `result_manifest.json`. Full numerical rows, checkpoints, scripts, receipts and replay outputs remain outside Git.
