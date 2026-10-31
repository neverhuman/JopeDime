# H62 / W2 — small slate effects, incorrect stop verdict

**Review verdict: the recorded context/draw tolerance fails.** The size-12 live slate changed its selected unrevealed candidate in **24/800 context/draws (3.0%)**, exceeding the run's prescoring limit of 2%. OPUS's original `INVARIANT → kill mechanism` verdict instead used an equal-family average of 1.8987%. Preserve that receipt as the author's original result; it does not establish the written stop condition.

This is a CPU diagnostic of frozen r882, not a trained improvement. Do not automatically clear the current scoring contract for W1/W3 or start the optional W2 training follow-up. The effects are small on this sample, but their operational disposition must account for the actual threshold and the legacy-selection endpoint.

## Measured effects

100 hash-selected DL-Train contexts, 79 families, seed701; 4,800 unique arm/size/context/draw rows. The fixed choice set C contains the full-pool r882 top four plus four random unrevealed candidates. Four revealed anchors stay fixed. Each size has800 context/draws per arm; size12 permutes C eight times rather than supplying eight independent distractor sets.

| Live slate size | Mean per-draw pair reversal | Changed choice in C | Equal-family choice-change mean | Changed legacy choice including anchors |
|---|---:|---:|---:|---:|
| 12 | 0.6429% | 24/800 (3.000%) | 1.8987% | 32/800 (4.000%) |
| 20 | 0.4554% | 15/800 (1.875%) | 1.1867% | 17/800 (2.125%) |
| 192 | 0.0937% | 2/800 (0.250%) | 0.3165% | 2/800 (0.250%) |

The anchor-reference intervention recorded zero pair reversals and zero changed choices at all sizes relative to its own full-pool reference. This supports a slate-centering mechanism in the tested path; it does not prove better utility than the live arm. Full-pool order permutation changed predicted costs by at most9.5367431640625e-7. Pair reversals use exact floating-point sign differences, excluding reference ties; the output omitted valid-pair denominators, so the table reports means of stored per-draw rates rather than an invented pooled numerator.

The equal-family size12 choice-change interval is [0,4.4304%], which also does not establish population equivalence within2%. Legacy selection changes4.0% at size12. The recorded positive utility deltas concern choice within C relative to each arm's own full-pool C choice, not the full legacy policy or verified U4. [metrics.json](metrics.json) retains both weighting schemes and the family intervals.

## Protocol and source limits

The run-specific prescoring manifest introduced numeric tolerances (≤1% reversal and≤2% changed context/draw choices); the original W2 proposal described literal invariance. Even the weaker run-specific rule fails at size12. Its analysis script averaged families for the decision, which changes the estimand. The reviewed result corrects this denominator error without changing thresholds after seeing outcomes.

The rounded offline r882 legacy `val_reg_inner` is1.423073 over1,219 DL-Train contexts (determinism repeat reported). This is an in-sample parity check on offline pools, not exact live parity with the historically reported1.422548. The later Elo addendum compares choice within C under slates12/20/192 against each arm’s own full-pool C reference (anchor1500): live ratings1505.50/1503.30/1504.40, context W/T/L2/98/0,2/98/0,1/99/0. The anchor-reference arm scores1500 throughout. ASTRA reproduced all ratings and W/T/L from the S3 rows using the repository family-grouped Bradley–Terry implementation (tie1e-4, prior.01,400 iterations). These are exploratory restricted-choice panels, not common policy-admission Elo, and cannot rescue the failed stop rule. Verified U4 was not computed.

The run used xbabe2, one process/four Torch threads at nice15, zero GPU and zero native evaluations. OPUS reports214 seconds for the diagnostic; measured process CPU time was not retained. No remote shards were run. The later t2155 ruling explicitly selects xbabe2 for W2 while xbabe1/3 are saturated.

r882 has seen these families. The loader materializes mixed train/dev cached features before selecting training rows; no dev outcome scoring is claimed. Source/cache hashes are recorded. A later source-reviewed admission addendum reports0/1219 pool-hash mismatches against the pinned selected.json and rowwise proj16 agreement with the live formula within9.537e-7. Context tokens are trusted through cache metadata rather than recomputed; inherited normalization/fold exposure remains unqualified. Publication records the diagnostic and these limitations; it does not certify leakage-free generalization or a qualified live loop.

## Immutable evidence and review

Original source HEAD at upload: `d5095432129d57cbd7cffa9fd6ceb77dfe86cd9e`; execution scripts were untracked scratch and are independently pinned in the S3 manifest, not part of that commit.

- Run prefix: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W2/w2-opus-20261031a/`.
- Manifest SHA256: `272e6c70f093258ac24b532226d7c7bab55653fd9904cbf5363ad5489e7e6cc0`.
- All10 original objects and four later addendum objects were independently downloaded and SHA256-checked by ASTRA; script hashes match the manifest. The initial history read found10 original versions and no deletion markers; addenda use new keys. OPUS's HEAD-then-upload helper has a race and is not atomic create-only storage; no prior overwritten version was observed. Future publication uses `If-None-Match: *`.
- Exact versions/hashes, row counts, corrected decision and metrics are in [metrics.json](metrics.json). Review source and command receipts are under `/home/ubuntu/jope-rescue-20260921/rd-verdicts-t2155/`.

ASTRA reproduced the denominators and both means from the downloaded rows, checked unique cells and the frozen roster, and reran the family bootstrap at seed20261031. This is a result review, not an independent experimental replication. Original objects remain unchanged; there is no HWM candidate or serving promotion.
