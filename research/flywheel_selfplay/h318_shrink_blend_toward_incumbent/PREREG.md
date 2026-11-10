# H318: convex hard-pick score blend toward r882 slot0

Owner hw_t1125_astra_h318.md; frozen before any H318 fit or alpha selection.
TRAIN-only selection research; no dev91/DL-Val/DL-Test, sealed/holdout/meta-test,
fresh cohorts, GPU or live writer access. DOPE c51a56d0 unchanged.

## Exact executable blend level

Pinned H310 819a45aa5e / H303 run_h303.ens_picks and H287 ensemble.vote_consensus_pick
return an integer candidate slot (0..7), not row-level regression predictions.
Use its executable candidate-action score representation: slot0 score vector e0,
ENS_VOTE2 pick score vector ej (one at selected slot, zero elsewhere). BLEND scores
are (1-alpha)*e0 + alpha*ej; select argmax, with equal scores choosing smaller slot
and hence incumbent0. alpha grid {0,.25,.5,.75,1}. BASE is unchanged ENS_VOTE2,
alpha1. No interpolation of measured utility, no randomized expected-utility
surrogate, and no claim of mixing raw program regression predictions.

This honest hard-pick score interface has only TWO distinct decisions: alpha<=.5
returns0; alpha>.5 returns the ENS pick (all identical when j=0). The grid and
its degeneracy are reported, not hidden as continuous James-Stein smoothing.
A continuous prediction blend would require a different, authorized evaluator
with both programs' sample-level predictions; this wave tests only score blending
at the existing discrete selector output. No new predictor/candidate is synthesized.

## Regression first and pairing

Exact-version GET/SHA the H303(P0) and H308(P1/P2) archived picks and frozen model
archives using their committed s3_manifest pins. Reuse verified archived picks
read-only if present, with the same hashes. Replay all36 frozen outer refits with
H310/H303/H302/H287 byte-pinned code, H308 TRAIN-only loader, roots30001..30003,
partitions P0/P1/P2 and4folds. Compare EVERY fold/root pick hash, per-seed fullpick
hash, family-equal seed gain and partition median. Require exact median
P0=.0643163334285782, P1=.04821480030677507, P2=.04360142642936269.
Mismatch REGRESSION_FAIL: stop before fitting H318. Regression is a separate
process; calibration receives only PASS and hash, never historical outer outcomes.

BASE hyperparameter/member fitting uses unchanged H303 full nested18config grid,
TAU {.05,.10,.15}, member k{5,6,7}, agreement{2,3}, same family folds and member
seeds. Paired BASE and BLEND share exactly the SAME fitted models and chosen BASE
configuration. All fresh BASE picks must again match H303/H308 hashes after locks;
a mismatch invalidates the scientific evaluation. H318 changes only alpha.

For each partition/outerfold/root, use chosen BASE configuration's INNER OOF picks
across4family folds of outer-TRAIN. Each inner-test family is absent from member
fitting; alpha selection sees only these inner-OOF outcomes, never outer-test.
Aggregate raw utility difference from slot0 by context mean within family, then
objective median(family gains) - .5 * mean(family gain < -.01). Choose largest
objective, all alpha values within1e-4 tied toward smallest alpha. BASE config
itself retains H303's mean-gain/conservative tie rule, unchanged. No joint retuning
of BASE configurations for the blend. There are36locked alpha choices, reported
by fourfold cells within each of9 partition/root summaries.

## Input boundary and locks

All source rows are the pinned1219 DL-Train contexts/205families. Mixed H58 NPZ
metadata cids/offsets may be read, but feature rows are read only for allowedTRAIN
contexts using exact uncompressed NPZ row byte ranges. Fitting workers read only
their outer-TRAIN context pools/cost members/cache row ranges; they never prebuild
outer-test packets. A tested allowlist denies an outer-test pool/member/range
before any read. At fit completion fsync fit_lock with all choice tables, alpha,
model hashes, source/input identities and counts: outer_test_rows_read_before_lock0.
Only after all locks and model hashes are verified does a separate scoring process
load each locked outer-test fold ONCE and compute both arms from the same outputs.
Scoring outputs include saved picks and utilities; analysis thereafter reads those
saved outputs. Regression scoring is explicitly separate, not a second H318 test.

## Statistics and decision

Per arm/partition/root: raw context utilities = -fitness for gated candidates,
-2 for gate failures, as frozen H303. Family-equal mean gain over205families;
report literal median family gain too. Pooled primary = median of9 cell family-
equal means, preserving H308's published convention; per-partition medians over3
seeds also reported. Ties within1e-4 never wins, harm family gain<-.01, invalid
counts, drop-top1/5/10 means after removing largest family gains, Elo labelled
exactly "reference = r882 slot0 (DL-Train outer)". r3175 is NOT_MEASURED on TRAIN.

Whole-family bootstrap10000 draws, numpy default_rng(318180), sorted205family
IDs sampled with replacement. The SAME resampled indices apply to both arms,
all partitions and seeds; seed policies are not independent observations.
Report percentile two-sided95% CIs and one-sided95% LCB = .05quantile.
Primary paired drop10 = median over9 of [drop10(BLEND family gains) minus
 drop10(BASE family gains)]. Re-sort/drop10 separately within each arm on EVERY
bootstrap resample, preserving paired family indices. Also report secondary
trim10(BLEND-BASE family gains) to make the alternative interpretation explicit;
it does not change the declared decision.

BLEND_ADVANCE iff primary paired drop10 median>0 AND its one-sided95%LCB>0,
AND BLEND pooled full gain >= .80*BASE pooled full gain,
AND BLEND median harmed-family rate <= BASE. Exact ties never advance.
Otherwise BLEND_NOT_ADVANCED, with all failed clauses. Missing/incomplete evidence
is INCOMPLETE or REGRESSION_FAIL, never a passing/zero-imputed outcome.
Even ADVANCE only proposes one preregistered dev91 shot vs r3175 later; H318
performs none. Reused TRAIN data and adaptive selection limit all inference.

## Resources, publication and custody

xbabe2 CPUonly nice19 numericthreads1, <=6idlecores13,15,38,57,59,125;
check current H315/H316/H317 affinity before launch. systemd MemoryMax6GiB,
aggregateRSS6GiB, hard stop hostCPU>85% or MemAvailable<20GiB. Own scratch
/home/ubuntu/jope-data/h318-astra <=2,000,000,000bytes. Absolute compute stop
2026-11-04T14:15Z, close14:30Z; never reset on resume. Stream verified model
archives to S3 before reclaiming own intermediate model files as needed.
Create-only prefix experiments/rd_breakthrough_20261030/H318/astra-20261104T112505Z/
under veox-jopedime-learning-use2; exact-VersionId GET/SHA/member checks before
own scratch removal. Keep source/receipts under /home/ubuntu/jope-rescue-20260921/h318-astra.
Nonforce origin/v1/next only: own study and one row each wave index. No PR/main/
fastlane/CI, other checkouts/jobs or foreign cleanup. Claims and FINDINGS underlock;
one comms line at claim and close. Metrics <50KB, no models in Git.
