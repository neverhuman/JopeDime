# H337 — Soft bag-vote smoothing

Owner rd_astra; authority hw_t2156_astra_h337_softvote.md. H336 is CLOSED at
6be4064d4c9a3084be1a95062a43d9db9495e545. One fixed SOFTVOTE arm, TRAIN only.
This document is published nonforce origin/v1/next before any fixture/model score.

## Exact code treatment and unchanged decisions

The call site exists at pinned H287 commit213b30ad52e1ce48927bba960ed609bd98ab4dfe,
research/flywheel_selfplay/h287_bagswitch_seed_ensemble_export/bagged.py:59-72,
function vote_counts. Original comparison AST expression:

```python
j_int != 0
```

Replace this single Compare node with `_h337_indicator(delta, margin)`, where
delta is the original candidate-minus-slot0 expression and indicator returns
`sigmoid((delta-margin)/T)` elementwise across the eight bags. The temperature
vector is explicitly bound for the current frozen member. Stable scipy expit is
used; no clipping/tuning/no additional fitted parameters. The single-node inverse
must restore the exact original function AST. The original operation is a
comparison expression, not a keyword: this is a one-call-site inverse, with no
other function-AST edits, not a claim that a literal keyword alone implements it.

Keep the existing zero-initialized int32 vote array, `int(np.sum(...))`, and
switch_by_votes' `int(v)>=int(k)`. For nonnegative sums and integer k5/6/7,
floor(sumsoft)>=k is exactly equivalent to sumsoft>=k. Preserve these original
casts and test their boundary equivalence; save unfloored sums for independent
auditing. Infinity-margin early rejection and slot0 early fallback stay exact.
At zero centered margin each soft contribution is .5; hard votes used strict>0.
No epsilon or alternate tie threshold is introduced. Tests cover strict ties,
near-threshold sums, saturation, per-bag temperatures, infinity and slot0 paths.

All 1080 BASE main/veto/bag files remain byte-identical; no model fits or neural
updates. The main proposer, veto TAU, raw/strict margins, selected rounds,
selected k and agreement a are unchanged. Use the BASE chosen TAU/k/a per cell,
without reselection or grid search. Config agreement is therefore1 by design,
not an empirical discovery. Scoring binds the member's temperature before its
existing frozen scorer and member-pick calls; no proposer or veto function is
patched. Main/veto/raw/bag outputs and soft sums are retained for independent
selection replay without model rescoring.

## Temperature population, fixed before outcomes

There is one T for each (partition,outer fold,root,member index,bag index):
36cells x3members x8bags =864 temperatures. For each cell, use only the archived
H318 inner models at that cell's BASE-selected TAU, four H303 inner folds, and
matching member/bag indices. Exact-version archives and member hashes are
verified. Inner fold assignments are H303's `inner_fold(outer_fold,family)`.
For each inner holdout fold, score its candidates with that fold's corresponding
archived inner bag model, which was trained on the other three inner folds.
These holdout candidates are all within this cell's outer TRAIN partition.
There is no new fit, no final-model resubstitution calibration and no outer-test
calibration. A bag index is calibrated using its four cross-fit counterparts;
those inner models are distinct from the final frozen bag whose T they supply.

Population = every nonbaseline candidate j1..7 for each outer-TRAIN context,
once through its own inner-heldout prediction, original TRAIN context order.
Use raw bag deltas `score[j]-score[0]`, no veto mask, utility weighting, family
weighting, bootstrap duplication or proposed-candidate filtering. Compute
population std (`np.std`, float64, ddof=0) across these candidate deltas, then
T_b=0.5*std. Subtracting the final member's constant finite strict margin does
not change this standard deviation. For an infinite strict margin, compute T
from the same finite deltas for audit but preserve BASE's early zero-vote return;
that T is not applied. This explicitly avoids undefined infinity centering.
Require every T finite and strictly positive. A degenerate/nonfinite T or missing
archive fails closed as INCOMPLETE; do not invent an epsilon, borrow temperatures
or omit a cell. Record all T, populations, fold/model/input identities and raw
calibration delta arrays for independent recalculation. No outcome-based tuning.

## Regression, input closure and locks

First exact-version recover the BASE H318 archive pinned in base_archive_pin.json.
Its BASE is H303 NESTED on H308 P0/P1/P2 x roots30001-30003 x4outerfolds.
Replay unchanged BASE once:36fold/root pick hashes,9pick/rawutility panels and
3medians must match exactly (.0643163334285782,.04821480030677507,
.04360142642936269). Regression failure stops the wave; no BASE refits.
The separate mandatory BASE regression sees TRAIN outer rows only to verify the
archived reference. Its outcome arrays cannot be opened by calibration workers.
The zero-prelock-outer-read assertion concerns new calibration/treatment workers,
not a false claim that the required BASE verification reads no TRAIN rows.

Each calibration worker's physical cache byte ranges and pool opens are limited
to its outer-TRAIN contexts. Other outer contexts, historical outcome arrays,
DL-Val/DL-Test and protected data are outside its closure. After calibration,
fsync36 score locks pinning cell identity, temperatures, source hashes,
all1080final model SHAs and unchanged BASE choices. Audit every lock, native
model identity, temperature calculation and allowed input before any new outer
score. Then once-only36fold/root treatment score calls form9partition/root panels.
Never repeat an outer model score; numerical review uses saved predictions.

## Metrics and frozen decision

Raw utility versus r882slot0;1219contexts/205families. Family means carry equal
weight; pooled gain is median of nine cell mean family gains. Report literal
family medians separately. W/T/L tolerance1e-4; harmed means gain<-.01;
invalid/switch counts are contexts. Drop largest1/5/10family gains per arm;
LOFO minimum, config agreement and seed Jaccard retained. Elo uses the inherited
400-step Bradley-Terry MM recipe, ties omitted, pseudocount.01. Label exactly:
reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.
No AUC or speed verdict. DOPE weights not loaded or modified.

10000 paired whole-family draws RNG337191, shared205family indices across arms
and nine dependent cells. Paired statistic = median9 differences of arm-specific
drop-top10 means. Two-sided95CI quantiles.025/.975; one-sided95LCB quantile.05.
SOFTVOTE_ADVANCE iff paired median>=+.002 AND LCB>0 AND median gain>=.9BASE
AND median harmed<=BASE AND strictly positive per-root paired medians on>=2/3
roots. Exact+.002 satisfies the explicit operator; zero LCB/overall ties fail,
and zero root medians are not positive. Otherwise SOFTVOTE_NOT_ADVANCED.
Adaptive reused TRAIN and dependent cells: intervals condition on frozen models
and do not adjust for wider wave search or rerun model/temperature selection.
Even ADVANCE would require a separate dev91 prereg; none is authorized here.

## Resources and custody

Fixed compute stop00:15Z Nov5, close00:30Z Nov5; no reset, partial-cell score or
model refit. CPUxbabe2nice19numeric1, allowedcores4/5/6/8/33/41; single5s/core+SMT
idle probe selects a core for serial calibration/scoring. At most6cores allowed;
no requirement to occupy all. MemoryMax6GiB/Swap0/RSS6GiB,hostCPU/load<=85,
MemAvailable>=20GiB,scratch<=2000000000B under /home/ubuntu/jope-data/h337-astra.
Stream one inner archive at a time, verify exactversion/allmembers, retain only
needed selected-TAU bag copies, reclaim verified own extraction after calibration.
No foreign process/lane/host changes. S3create-only prefix
experiments/rd_breakthrough_20261030/H337/astra-20261104T215600Z/;
exact-version GET/SHA/everymember proof before own final scratch removal.
Nonforcev1/next only; noPR/main/fastlane/CI underCI_FREEZE/BATCH_WINDOW.
DOPE {"composite_sha256_prefix":"c51a56d0","loaded":false,"scope":"CPU selection layer only"}.
NoGPU/qualification.lock/dev91/DLVal/DLTest/sealed/holdout/meta_test/fresh-cohort,
xbabe0/3, live xbabe1writer or H332/H334/H335 jobs/custody.
