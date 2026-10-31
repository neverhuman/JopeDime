# H105 — shrinking toward r882 does not rescue the fine-tune direction

**COMPLETE / NO_ADVANCE. Retain r882.** All 12 paired outcome cells finished
at 2026-10-31 18:05:12 UTC (12:05 MDT). Every arm fails the preregistered
improvement and harm gates. The unchanged [native metrics](metrics.json) contain
the complete readout and charged ledger; no checkpoint is promoted.

## Hypothesis and fixed setup

The [preregistered](PREREG.md) hypothesis was that the harmful fine-tune direction
could be rescued by shrinking its displacement toward r882, or by anchoring
training to r882. CONTROL keeps the full displacement, INTERP025 and INTERP050
keep 0.25 and 0.50, and ANCHOR adds the fixed raw-L2 penalty with
lambda 2.8867746553428666. Interpolation is applied after the paired fine-tune;
the CONTROL and interpolation arms share each seed's fine-tuned adapter digest.

- Seeds 42–44; whole-family fold seed 105042, assessment fold 0 of five.
- Real regression DL-Train: 1,006 fitting contexts / 164 families and
  213 assessment contexts / 41 families. Both partitions are within DL-Train.
- Fresh r882 copies; 40 AdamW updates per cell, batch 16, slate 192,
  learning rate and weight decay both 1e-4. Exactly 29,211 adapter parameters
  are trainable; the anchor and DOPE remain frozen.
- Execution source `002b92caeeb313456d6d05cd6049564eaa7d0980`, CPU-readiness
  PR #1075, original implementation PR #1017. Receipts separately record the
  physical checkout HEAD; source guards verify five owned and 13 shared files
  against the execution commit before and after each cell.
- H99/H100 predecessor ordering and FIFO leases were retained. The terminal
  ledger charges **874.769581705099 GPU seconds / 2700**, including one
  40-update timing cell. The 12 outcome cells use 480 updates; including timing,
  520 updates were executed. Timing is excluded from scientific comparisons.

No new run, GPU use, optimizer update or evaluation was performed for closeout.
The earlier [CPU smoke](CPU_VALIDATION.json) and
[pre-run reproducibility handoff](REPRO_BUNDLE.md) remain historical preparation
records; this terminal report supersedes their pending-outcome wording.

## Outcome and admission

Gain is **candidate utility minus r882 utility**, so higher is better.
The evaluator chooses the lowest predicted cost among unrevealed candidates
and uses clipped utility `clip(-fitness, -2, 4)`, with failed gates mapped to -2.
The legacy `train_reg` and `assessment_reg` fields are utilities, not losses.
Contexts are averaged within family before forming the paired family gains.

The primary point estimate is the median of the three seed means. Intervals
resample the same 41 paired whole families using 10,000 draws, RNG 105000;
each draw takes the median across seeds. These are family intervals, not three
independent assessment datasets. Harm counts families whose seed-mean gain
is below -0.01. Elo gives an exact tie half a win; ties remain reported separately.

| Arm | Median gain | 95% family-bootstrap CI | Harmed families | Median Elo vs r882 |
|---|---:|---|---:|---:|
| CONTROL | -0.021952 | [-0.077909, +0.042626] | 14/41 (34.1%) | 1448.8 |
| INTERP025 | -0.025229 | [-0.059983, -0.000466] | 6/41 (14.6%) | 1474.5 |
| INTERP050 | -0.052258 | [-0.101386, -0.012176] | 11/41 (26.8%) | 1474.5 |
| ANCHOR | -0.030529 | [-0.079681, +0.018748] | 9/41 (22.0%) | 1448.8 |

All four Holm improvement tests fail (adjusted tail areas 1.0), all median
gains are negative, and every harm fraction exceeds 10%. Thus every
`admitted` flag is false under the unchanged strict Holm lower bound >0.001,
nonnegative point estimate and harm <=10% rule. The table's two-sided CIs are
the native marginal family intervals, not a new multiplicity-adjusted harm test.

**Even the 0.25 interpolation harms:** its gain CI upper endpoint is
-0.0004656749292895104. Shrinking to this quarter step does not rescue the
fine-tune direction. In this experiment the direction itself is harmful at
that smaller step; step size alone does not explain the failure. This finding
does not establish that every untested interpolation would be harmful.

## Red-team check: equal Elo is not a copied seed result

Seeds 43 and 44 have equal aggregate W/L/T and hence equal Elo in CONTROL,
INTERP050 and ANCHOR. A CPU-only check of all saved adapter tensors reproduced
every native semantic state digest: **12 snapshots, 12 distinct states**.
Each questioned pair differs in 41 parameter tensors. Their continuous utility
vectors and chosen candidates also differ:

| Arm, seeds 43 vs 44 | Equal W/L/T | Adapter pair L2 | Changed context utilities / 213 | Changed family utilities / 41 | Changed family gain signs |
|---|---|---:|---:|---:|---:|
| CONTROL | 12/18/11 | 0.103218 | 15 | 12 | 2 |
| INTERP050 | 10/13/18 | 0.051609 | 10 | 8 | 6 |
| ANCHOR | 6/12/23 | 0.020273 | 4 | 4 | 2 |

Compensating family sign changes preserve the coarse W/L/T totals while
magnitudes and family identities differ. All seed-pair context and family
utility vectors differ, consistent with every `identical_seed_utility_flag`
being false and the singleton `seed_utility_groups` in the metrics.
These are distinct seeded training outcomes on a common family panel, not
independent test datasets or three independent confirmations. The native
`independent_seed_confirmations_claimed` fields remain zero.

## Isolation, integrity and reproducibility

The terminal replay revalidated every receipt hash against the ledger and
reproduced the native summary, CIs, Holm decisions, harms and Elo exactly.
All cells retain the same fold identities, source, preregistration, initial
adapter and r882 reference. Training and assessment have disjoint context and
family sets. The pinned worker loads only fitting-family pools into its update
plan; assessment loading occurs after the fixed 40 updates and after parameters
are frozen and the model enters evaluation mode. Assessment does not choose an
update count, alpha or checkpoint.

All completed receipts report zero DL-Val, outer/holdout payload reads and outer
rows delivered. The source accepts only the approved H50 train/inner input
closure and rejects protected paths and non-DL-Train fitting metadata. The
executed checkpoint guard recomputes the frozen DOPE digest and checks the
adapter-only optimizer inventory around training and final restoration.
DOPE remains `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`.

The unchanged metrics artifact is 46,281 bytes, SHA256
`f4e61d699f673e748a81ee66e7c3ad8d65ea0c3a11ed7bf0fddd9e6b617bd62b`.
Raw terminal receipts, adapters, timing, source proofs and ledger remain at
`/home/ubuntu/jope-data/h105/h105-wave/`; they are not copied into Git or deleted.
The closeout replay, all seed-pair checks and targeted gate receipts are at
`/home/ubuntu/jope_plan/evidence/H105/t1235/`. The terminal log and complete
ledger show operator 3736944 finished normally after all cells.

## Verdict and next step

**NO_ADVANCE; retain r882.** Neither the registered interpolation grid nor the
fixed L2 anchor meets the common gate. The next scientific question is the
utility alignment of the fine-tune direction, rather than treating a smaller
step as a sufficient remedy. Any follow-up requires a new preregistration and
owner authorization. This closeout authorizes no rerun, new seed search,
promotion or serving claim.
