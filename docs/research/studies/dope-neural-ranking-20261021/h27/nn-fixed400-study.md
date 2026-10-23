# H27: fresh Dope features and the completed matched neural factorial

All 36 models completed their fixed 400-update endpoint: **14,400 dataset optimizer updates**, with **180 actual TRAIN/internal-monitor snapshots**. Independent raw-statistics, schedule, initialization and full frozen-model OOF replay passed with limits. **The two predeclared comparisons do not establish landmark ranking lift.** Training and internal-monitor learning improved, but that alone is not the ranking claim.

## Predeclared result

The outcome is raw chosen archived GP fitness; lower is better. Each point averages the three fixed-seed utilities, then weights whole components equally. No best seed, score ensemble or checkpoint is selected.

| Real Dope-state comparison | Raw fitness difference | Central 98.75% interval |
|---|---:|---:|
| Five-fold landmarks minus zero landmarks | -0.1055719651 | [-0.7890362249, +0.2869843098] |
| Five-fold landmarks minus one-fold landmarks | +0.1482177800 | [-0.0230595524, +0.4012829990] |

Both intervals include harm. The second point estimate favors the cheaper one-fold representation. The same 100,000 paired component resamples, seed 20261022, support both intervals; the two-comparison familywise coverage is 97.5%. The full result, all seeds and secondary estimates are in [the independent result](nn-fixed400-independent-results.json).

| Fixed arm | Component-equal raw fitness | Failed choices / 2,550 |
|---|---:|---:|
| Real state + zero landmarks | -0.9114288393 | 9 |
| Real state + one-fold landmarks | -1.1652185844 | 7 |
| Real state + five-fold landmarks | -1.0170008044 | 10 |
| Zero state + zero landmarks | -0.9079062799 | 8 |
| Zero state + one-fold landmarks | -0.9468995044 | 9 |
| Zero state + five-fold landmarks | -0.9808861990 | 8 |

These are all six arms, not a selection of favorable runs. State differences and the interaction remain secondary point estimates without a new established uncertainty result. The zero-state arm retains the common Dope input/model features, so it does not remove all Dope information.

## What actually learned

[Every objective curve](nn-all36-objective-curves.svg) and [every raw-fitness curve](nn-all36-raw-fitness-curves.svg) include all 36 models at steps 0, 100, 200, 300 and 400. [The exact aggregate observations](nn-fixed400-training-curves.json) also retain failures and chosen-fitness regret. The plot generator [uses only this public JSON](nn-curve-plot.py).

All 36 models have lower final TRAIN and internal-monitor objectives than their own initial values. However, **31 of 36 internal-monitor objectives are worse at update 400 than at update 100**, while zero of 36 TRAIN objectives worsen over that interval. This is a meaningful generalization/overfitting warning despite successful optimization. Fit-specific objective scales and panel composition differ, so these losses must not be pooled into a single validation curve. Compare each model with its own observations. The fixed endpoint remains update 400: no earlier checkpoint was retrospectively selected, and no OOF learning curve was manufactured.

The two whole-component folds use 73 gradient components, eight internal-monitor components and 81 OOF components each. Fold 0 has 408 gradient / 41 monitor / 401 OOF contexts; fold 1 has 362 / 39 / 449. TRAIN curve panels contain 16 fixed contexts per fold; monitor curves score the whole internal-monitor panel. Monitors contribute no gradients, feature scalers, fitness-scale fitting or recipe selection. All 36 fits closed before final OOF scoring.

## Matched implementation and source

The frozen design is 2 state arms × 3 landmark arms × 3 seeds × 2 component folds. Every fresh model uses the same width-64 M0 architecture (67,522 parameters), candidate dimension 412 and context dimension 376: common240 + padded state128 + landmark8. Genuine candidate/context products and absolute differences supply interactions. Every arm shares capacity, initialization, schedules and 400 updates (128 queries/update, 16-query microbatches; AdamW 0.0003, weight decay 0.1, dropout 0.3, gradient clip 1). Literal control zeros are applied **after gradient-only scaling**.

The one-fold fingerprint reuses the fixed first TRAIN-internal probe fold; five-fold uses the frozen five-fold mean. These fingerprints are not archived candidate target labels. Archived target fitness is **S1 two-fold**. The probe CV5 contract and NN two-component-fold partition are separate identities and are never grafted. Probe canonical/target-program overlap may remain unknown. The frozen basket also contains closely related raw/robust ridge probes; eight names do not establish eight independent signals.

[Fresh Dope qualification](fresh850-dope-qualification.json) covers all 850 contexts: 848 new clean-default native fits and two prior fresh qualification fits reused, zero failed/unknown/unattempted. Legacy S3 router encodings were not reused. The source is explicitly pinned to checkout `098702ae9fb6dd9be198f95e36026c621d3b67f8`. Saved-output reconstruction uses a separate implementation authored by the producer author, with external source/execution review. It is **not independent native retraining or forward replay**. Hash-bound native model/output consistency is useful evidence with that limit.

The independent NN verifier reconstructed raw gradient-only statistics, fitness scale, schedules and matched initialization, then replayed all 36 fixed models: **15,300 OOF choices**, exact chosen indices, maximum score/failure difference 0. It uses the same pinned model kernel and tolerances; it is not a second neural implementation or optimizer rerun. QA performed zero fits or optimizer updates. Source and receipt identities are in [the source receipt](nn-fixed400-source-receipt.json).

## Costs, preserved failures and limits

The two training phases took 651.6721 and 635.2320 seconds of worker wall time; their worker CPU summed to 1,212.1403 seconds and retained unit-tree CPU to 1,261.9730 seconds. Nested per-model costs in [the training completion](nn-fixed400-training.json) overlap these phase totals and must not be added. The sequential controller's 690.7972-second wall time includes passive waiting and is also not additive compute. Independent final QA took 191.1334 seconds wall, 145.6723 worker CPU plus 3.6092 child CPU. The 848-new-encoding worker wall sum is 6,237.9551 seconds across parallel workers, **not global elapsed**; retained unit CPU summed to 3,220.1223 seconds. The earlier two reused native fits were not remeasured here. Saved-QA merge cost is only that merge stage, not the complete preceding saved-QA fleet.

The first data namespace source was prospectively blocked for a writable alias and corrected before access. The first materializer launch stopped before unit dispatch because one declared metadata roster was absent; it read no arrays and performed no fits. A separate code-only correction admitted 97 legitimate empty runtime-code files without weakening numerical/proof lengths. The draft bootstrap drift was corrected prospectively to the frozen 100,000-draw protocol. Two metadata closeout wrapper calls also failed before their separate wrapper correction; those were not native-fit failures. Original sources, failures and costs remain preserved. Unmeasured preparation wall/CPU costs are null, never zero. Actual fit completion is 36/36 with zero unknown optimizer updates; no failed fit was silently replaced.

Retained training properties show 8 GiB memory, swap 0, Tasks64, successful MainPID0 exits, and infinite runtime limit; both units were then stopped. Host/GPU safety and ownership guards remain. There are no research elapsed-time cutoffs.

The 850 contexts, 164 families, 162 components and 979,200 archived candidate rows come from the existing historically exposed curated TRAIN cohort. This is useful component crossfit development evidence, **not new independent-family generalization**. Historical input/A0/preprocessing and source-admission limits remain. No production policy or weight promotion follows. The next recipe requires a new falsifiable mechanism and prospective selection within TRAIN; the null five-fold primary is preserved.
