# H197: per-context gradient-norm clipping on an r882 adapter continuation

Owner: `luna_ns_hbaseline`. Authority: `~/nudges/hwave_t1130_h197_luna.md` (2026-11-01). This is a conditional continuation screen, not independent outer validation or serving qualification. No H197 outcome was accessed when this plan was written.

## Hypothesis and scope

At the initial r882 adapter, a small number of FIT contexts may dominate the squared adapter-gradient norm while carrying little loss mass. Clipping each context's parameter gradient before batch averaging may reduce the influence of an individual failed context and improve held-family utility. The loss, targets, revealed-history inputs, native kernel, and candidate-pool sampling remain unchanged. Only the 47 adapter tensors / 29,211 adapter parameters train; DOPE stays frozen. The live flywheel checkpoint is never opened for writing.

The population is the real S3 regression DL-Train roster (1,219 contexts / 205 families) joined to the pinned source catalog. The exact native r882 replay over DL-Val (91 contexts / 19 families; mean utility `1.1377941898164525`) must pass bit-for-bit before any TRAIN outcome pool is opened. DL-Val is descriptive only. Binary records, sealed meta-test families, holdout, and all protected paths are excluded.

## Frozen design

Three seeds `197101–197103` use independent family-fold seeds `197141–197143`, respectively; the committed `fold_manifest.json` records all five folds and pairwise corresponding-fold Jaccard values. For each seed and each outer fold, A is fold `j`, B is fold `j+1 mod 5`, and FIT is the other three folds (123/41/41 families; FIT=rest). There are 15 cells. FIT batch plans are common across the four trained arms within each cell: 40 updates, eight contexts per batch, four lowest-fitness revealed candidates plus 20 without-replacement remainder candidates per context. The batch stream is fixed from the seed/fold and candidate-order inputs.

All four trained arms start from a fresh in-memory copy of the exact r882 adapter and use the live AdamW recipe (`lr=1e-4`, `weight_decay=1e-4`, `betas=(.9,.999)`), live eval/dropout-off mode, and the unchanged H146 per-context SmoothL1-plus-KL loss. No learning-rate, loss, target, kernel, slate, family, or step tuning is allowed. The reference R882 is zero-update. Fixed candidate step is 40; there is no step selection.

For each cell, before any arm update, a deterministic five-batch FIT-only dry pass is run at the initial r882 adapter on the first five batches of its fixed plan. In the first batch (the step-0 context sample), compute each of its eight context losses and parameter gradients separately. `C` is the median of these eight full adapter-gradient L2 norms; it is cell-specific, frozen before training, and has no floor. A nonfinite or nonpositive C aborts the single attempt. For the same five dry batches, scale each context gradient by `min(1,C/norm)`, average the eight clipped gradients, record that aggregate gradient norm, and set the GCLIP threshold to the median of the five aggregate norms. A nonfinite or nonpositive GCLIP threshold aborts. Dry batches never update weights or use B/A labels.

| Arm | Per-update gradient treatment |
| --- | --- |
| `CTRL40` | Live batch-mean gradient, global L2 clip at the live value 1.0 |
| `PCGC40` (primary) | Compute eight per-context gradients separately; clip each at its cell's frozen C; arithmetic mean of the clipped gradients |
| `GCLIP40` (secondary) | Live batch-mean gradient; global L2 clip at the cell's frozen dry-pass GCLIP threshold |
| `PCGC_SHUF40` (placebo) | Compute the same eight per-context gradients and same multiset of PCGC factors; deterministically permute the factors within each batch before averaging |

All arms get 40 updates in each of 15 cells: 60 trajectories / 2,400 optimizer updates. Training completion and all FIT checkpoint/trace hashes are fixed before any B outcome pool is opened. Evaluate complete B panels for R882 and all arms, then fsync a `pooled_b_selection_lock.json` that binds every B panel and fit state, states fixed step 40/no selection, and authorizes A scoring. B cannot change arm settings or step. Score all fixed-step arms unconditionally on A only after this lock.

## Analysis and decision

Utility is `-fitness` for a valid chosen candidate and `-2` for an invalid chosen candidate. Within each family, average contexts equally. For each seed, pooled A covers all 205 families exactly once across its five rotating folds. The primary is PCGC40 minus R882: report per-seed equal-family gain, median of the three per-seed gains, a 95% paired-family bootstrap interval for the median (10,000 draws; shared family resamples across seeds), positive-seed count, and one-sided bootstrap p-value. This is the sole primary test (Holm `m=1`). Also report the same per-seed/median/CI summaries for PCGC40 minus CTRL40; this is descriptive and does not enter multiplicity.

Harm rate is the fraction of the 205 families whose three-seed-average equal-family PCGC40-minus-R882 utility is below zero. Also report whether the median equal-family PCGC40-minus-PCGC_SHUF40 A gain is strictly positive, exact `-2.0` choice counts per seed/arm, per-context clip rate and factor quantiles, and H148 family-paired Elo/lift vs R882 with 95% family-bootstrap intervals. Report pairwise fold Jaccard. DL-Val candidate results are labeled descriptive. No AUC, speed, or tokens-per-second is reported.

`ADVANCE` only when all four hold: primary PCGC40-vs-R882 median-bootstrap LCB95 is strictly above zero; its one-sided Holm m=1 p is below .05; harm rate is at most 10%; and PCGC40's median A gain is strictly greater than PCGC_SHUF40. Ties are `NO_ADVANCE`. Even on ADVANCE, report only `CANDIDATE` for flywheel adoption and stop; ENG decides any later action. No serving promotion or live-loop mutation is authorized.

## Execution and resource limits

One attempt only on `xbabe2`, CPU-only, `nice 19`, no more than four aggregate threads, aggregate RSS below 6 GB, and a three-hour science cap measured from monitor launch. Prelaunch and running guards require disk use strictly below 93.8%, at least 16 GiB `MemAvailable`, free disk at least 40 GB, host CPU at most 85%, and load below 85% of online cores; running load must be below the limit on three independent five-second windows. Output root must stay below 300 MB. Never use `xbabe0` or `xbabe3`, never fall back to GPU, and never pause or signal another owner’s process. No resume or retry after the single attempt starts. Large arrays and full panels remain outside Git and go to versioned S3 custody.

## Source and delivery

H197 reuses H189/H194 continuation components read-only with exact source/input pins. Open one preregistration PR containing this protocol, runner, data-free tests, independent fold manifest, and index row. Run focused pytest, Ruff, `git diff --check`, and the five jope-records verifiers. Squash-merge that PR under the owner ruling; no remote CI or peer-review wait. Only after verifying the merged source and all prelaunch guards may the one science attempt launch. Then publish a separate small closeout PR with README, compact metrics and the result index update. Append claim, progress, and launch receipts to `/home/ubuntu/jope_plan/ns/receipts/H197_claim.txt`. Preserve and independently restore-verify the lane before cleanup.
