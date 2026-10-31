# H99 — ESH harm-tail follow-up closeout

**Outcome: NO_ADVANCE.** The seven-cell panel and one-time Fold-B scoring are complete. Neither ESH arm met the preregistered admission gates, so H99 does not advance and is not eligible for serving promotion.

## Hypothesis and setup

H99 tested whether increasing H80's family-harm penalty from lambda 1.0 to lambda 2.0 or 4.0 could reduce harmful-family outcomes without losing utility. The fixed candidate arms were ESH-l2 and ESH-l4, each with paired seeds 36–38; NULL seed 36 was a drift diagnostic only. The frozen recipe used 16 generations, population 32, 29,211 adapter parameters, and the fresh seed-9900 family split over the approved DL-Train roster. The protocol and gates are in [PREREG.md](PREREG.md).

All six candidate cells and the NULL diagnostic completed. Fold B was scored once for the six candidates across 41 families. The NULL was not Fold-B scored. The recorded admission receipt reports zero holdout payloads, no validation payloads, and zero outer rows delivered. Source evidence remains in `/home/ubuntu/jope-data/h99/h99-wave/outcomes/`; the compact [metrics.json](metrics.json) contains summary statistics only, with per-family arrays left in custody.

## Outcome

| Arm | Median Elo vs r882 (95% family-cluster CI) | Pooled family median gain | Mean-gain LCB95 | Harm fraction | Admitted |
| --- | ---: | ---: | ---: | ---: | --- |
| ESH-l2 | 1483.04 [1395.25, 1568.67] | 0.000000 | -0.047456 | 39.02% (16/41) | No |
| ESH-l4 | 1448.79 [1366.71, 1534.00] | -0.000288 | -0.048388 | 39.02% (16/41) | No |

Both arms exceed the preregistered 10% harm cap, and each family-median gain interval includes zero. Neither arm has Holm support for mean gain strictly above 0.001. The native admission verdict is **NO_ADVANCE**.

## Readout red-team note

The 17:56Z H99/H100 post-run readout at `~/jope_plan/evidence/H99-H100-readout/t0955/final-H99/H99-readout.json` reports `INVALID` with “H99 cell fold/scoring mismatch,” because it compares each cell's `fold_assignment_sha256` directly with Fold B's hash. The cell hash covers an assignment object containing `train_contexts_a` and `train_contexts_b`; the Fold-B hash covers an object containing `source_split` and `sealed_meta_test_inputs_opened` instead. Those hashes therefore identify different serialized objects. Direct comparison of `fold_a_families`, `fold_b_families`, and `folds` found exact equality for all seven completed cell summaries against Fold B (164 Fold-A families, 41 Fold-B families, five folds). The mismatch is a readout-contract bug, not evidence of a fold-list mismatch or leakage. This closeout records the finding without changing the readout implementation or the underlying receipts.

## Verdict and next step

Close H99 as **NO_ADVANCE**. Do not tune or rescore against this Fold B. Any follow-up needs a new preregistered question, fresh split/evidence, and owner authorization. This result does not authorize or change H100/H105 work and makes no serving-promotion claim.
