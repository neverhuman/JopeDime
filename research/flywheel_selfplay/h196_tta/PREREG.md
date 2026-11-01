# H196: zero-update test-time augmentation of r882

Status: PREREGISTERED; not executed. Owner: rd_astra / [ASTRA].
Authority: `/home/ubuntu/nudges/hwave_t1105_h196_rdastra.md`.

Hypothesis: averaging incumbent cost predictions after small perturbations of
the frozen adapter input improves family-balanced selection utility. There are
zero optimizer updates. No live files, model promotion, protected data, GPU,
new candidate generation, or adaptive outcome-driven tuning are involved.

## Arms and input identity

The four reported arms, in order, are R882, TTA_NOISE16 (sole primary),
TTA_PERM8 (descriptive), and NOISE1 (descriptive placebo). The checkpoint is a
private read-only copy of H170's preserved r882. The native kernel remains active.
Before treatment computation, native 91-context utility must equal every entry
of the pinned reference vector and its mean must equal 1.1377941898164525 exactly.
The cached-anchor implementation must also reproduce native candidate costs
bit-for-bit. DOPE hash c51a56d0 and the full adapter state hash are checked before
and after every scoring stage. Backward and optimizer construction are forbidden.

Use the inherited, SHA-pinned real S3 regression closure: TRAIN 1,219 contexts /
205 families; DL-Val 91 contexts / 19 distinct families, descriptive only.
Rejoin all 276 dataset IDs against the current owner-specified S3 catalog and
exclude its sealed-name IDs. No sealed/meta-test/holdout object bodies are read.
The inherited token cache contains both TRAIN and DL-Val features; DL-Val
outcomes are used only for the mandated native check and final descriptive report.
The 10x32 inputs are the existing frozen multimodal tokens; not all token roles
should be described as literal DOPE output channels.

## Treatment definition

Sigma is fixed at 0.05. There is no B-selected sigma search.
For each of the 320 token-position/channel coordinates, compute population
standard deviation (ddof=0) over that cell's FIT contexts only. Use only contexts
where that token is active. Fewer than two active observations gives zero scale.
Inactive tokens are unchanged. No clipping, normalization, or checkpoint changes.
Each draw adds float32 Gaussian noise times the float32 FIT scale and 0.05 to the
float32 input. The draw is generated from a SHA-256-derived seed binding wave,
outer seed, outer fold (or VAL), context ID, and draw index. It is independent of
batch order and size. NOISE1 is exactly draw zero of NOISE16.
Average the 16 float32 predicted costs in float64, then use first-index argmin
in the original candidate order. Do not average utility or selected indices.
All draws retain the same candidates, revealed data, masks, anchor and kernel.

PERM8 uses eight independently seeded full candidate-order permutations per
outer seed/context. Permute all candidate-aligned tensors, map each original
revealed index through the inverse permutation, and preserve revealed values
and their history order. Run the full native forward, unpermute its cost vector,
average eight vectors in float64, and apply first-index argmin in original order.
Do not substitute an unpermuted anchor for a permuted full native forward.

The FIT invariance screen uses the first eight lexicographically sorted context
IDs from outer-fold-zero FIT for each seed, with eight permutations each. Compare
full unpermuted cost vectors bitwise against native costs. Stop that seed's screen
on the first mismatch. If all 64 comparisons match, report PERM8=R882 by the
predeclared screen and skip further PERM8 scoring for that seed. This is evidence
on the screened FIT contexts only, not a proof of global permutation invariance.
Record actual comparisons, context IDs, and first mismatch. No evaluation outcome
is used to select the probe or its stopping rule.

## Folds and durable stage order

Outer seeds: 196101, 196102, 196103. Independent family-fold seeds:
196141, 196142, 196143. Reuse the H191/H177 family partition function with private
bindings. Five folds, A=j, B=(j+1) mod 5, FIT=remaining three. Preserve exact-once
pooled A/B coverage and pairwise disjoint families within each cell. Report all
three 5x5 pairwise family-fold Jaccard matrices and their diagonal means.

After the native check and FIT invariance screens, compute all 15 FIT scales and
B panels. Freeze these, the probe decisions, source/input identities, random
streams and all remote cost receipts in one fsynced B lock. Bind its digest and
the complete B panel digest before scoring A. Recheck lock immutability at every
A and DL-Val cell. No B outcomes alter any treatment parameter. A costs for
NOISE16 use that A cell's FIT scale. PERM8 is FIT-scale-independent and may reuse
its per-seed B-stage costs after the lock; B and A roles overlap across cells.
Report this overlap; these reshuffles are not three independent replications.

For descriptive DL-Val only, each seed uses the coordinatewise median of its
five FIT scales, fixed in the B lock. No all-TRAIN scale refit or DL-Val tuning.
Its draws use a distinct VAL stream. Full per-seed A and Val utility panels are
small local evidence; per-candidate cost arrays are not written locally.

## Statistics and decision

Compute each family's mean context utility, then equal-family arm means per
seed. Utility is -original-float64-fitness for a passing gate, otherwise -2;
preserve exact -2 counts without float32 rounding. The estimand is the median
across three seeds of the paired equal-family gain against R882.

Use 10,000 paired-family bootstrap draws, seed 196150, with the same resampled
family indices across all seeds and arms. Each replicate first computes each
seed's equal-family gain then takes the median over seeds. Report percentile
95% CI, positive seed count, and p=(1+count(bootstrap median<=0))/10001. Holm m=1
applies only to NOISE16; PERM8 and NOISE1 are descriptive. Bootstrap conditions
on fixed scales, fold allocations, probe decisions, and random draws; no refit.

For an unambiguous conservative harm definition, a family is harmed when its
seed-averaged primary utility delta is strictly below zero. Report its fraction
and each seed's analogous fraction, plus the supplementary fraction below -0.01.
Admission uses the strictly-negative definition, not the supplementary threshold.
This deliberate choice is stricter than earlier waves' 0.01-loss thresholds.

ADVANCE requires primary LCB95>0 AND Holm p<0.05 AND harm<=0.10. Equal utility
does not count as a positive gain; LCB or p boundary ties reject. Do not import
H177's additional 0.001 threshold or H178 calibration gate. ADVANCE is reported
as CANDIDATE for ENG's decision; no live adoption.

Report per-seed gain, median/CI, positive-seed count, p/Holm, harm, exact -2 counts,
and family-paired Bradley–Terry Elo/lift against fixed R882=1500, using the four-arm
H177 scorer. Elo intervals use 1,000 paired-family draws at seed196151 and medians
across seeds. DL-Val statistics are descriptive. No AUC or speed comparisons.

## Execution and custody

One scientific attempt, only after merged preregistration. CPU xbabe2, nice19,
all numeric libraries one thread, process-tree threads<=4 and RSS<6GB. Three-hour
wall cap. Prelaunch and running guards: host CPU<=85%, MemAvailable>=16GiB,
user-visible disk use<=93.8%, local output root<300,000,000 bytes. No other
host fallback, foreign process changes, GPU, scientific retry or budget reset.

Serialize cost arrays in memory, create unique S3 keys with If-None-Match '*'
under `experiments/rd_breakthrough_20261030/H196/astra-t1105/`. Require a VersionId,
GET that exact version, verify payload and member SHA-256, and retain small
receipts locally. No local candidate-cost dumps. Upload native costs, all noise
draw costs, and non-skipped PERM8 draw costs with context and pool identities.
On any guard/custody/identity failure, record INCOMPLETE and do not infer a verdict.
Reconstruct means, argmin, panels and statistics from exact-version S3 objects
for independent deterministic replay without additional model inference.

Acceptance: data-free targeted pytest; Ruff; all five jope-records gates; full
diff/source review. Small prereg and closeout PRs follow the explicit owner
exception: squash --admin allowed; no wait for remote checks or peer review.
Verify merged source identity. Final README includes hypothesis, setup, complete
three-seed family-grouped results, verdict, limitations, and next step. Preserve
source, receipts, S3 recovery route and stopped-head handoff; remove only the owned
lane and independently verified owned bulk after closeout.
