# H207: frozen r882 TTA with canonical singleton scoring

Status: PREREGISTRATION PROPOSED; NOT RUN; GPU allocation pending.
Owner: rd_astra / [ASTRA]. Authority: `gbw_0305_rd_astra.md`, renumbered by
`n2022_rd_astra_renumber.md` (Jepson via Grok Bot, 8:22 PM MT).

## Why this follow-up

[H196 attempt 4](../h196_tta/README.md) stopped before any completed B or A cell
at a strict native/cached identity check, with zero optimizer updates. Its
[native-only diagnosis](../h196_tta/diagnosis.json) reproduced cross-batch float32
cost differences on an RTX3090 with unchanged weights: 36 batches of eight
passed, the final batch of three differed in 672 values (max 7.62939453125e-6),
and zero of those three argmins changed. Same-batch fresh native/cached,
repeat/order controls, and singleton controls passed. Singleton native91
reproduced every reference utility and mean 1.1377941898164525 exactly.
The original RTX4090 failing vectors were not retained; this reproduction does
not prove their exact discrepancy or establish any TTA efficacy.

The hypothesis remains that averaging costs under small frozen-input
perturbations improves family-balanced selection utility. The harness change is
**one context per forward call everywhere**: native caches, cached-anchor
checks, FIT screens, all NOISE16/NOISE1 draws, and all PERM8 forwards. Candidate
count/order and the full native kernel remain intact. There is no tolerance,
weight update, batch-size search, or outcome-driven retuning. Exact identity
failures remain terminal. Save failing vectors and batch/context identities to
versioned evidence before stopping when custody remains available.

The numerical protocol below is inherited from the pinned H196 preregistration.
Its earlier CPU-only execution and attempt-budget amendments do not apply to
H207. The seeds **and `H196` augmentation seed namespace** stay fixed to preserve
random draws; the wave, branch, folder, receipts and new S3 keys are H207.
Reuse the exact pinned fold manifest; regenerate all predictions, panels and
locks. H196's partial predictions are diagnostic evidence only.

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
float32 input. The draw is generated from a SHA-256-derived seed binding the retained `H196` stream namespace,
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

## Execution, allocation and custody

This is a documentation preregistration, not an implemented or qualified H207
executor. Before launch, implement and review the singleton execution path,
pin its source and package/input closure, and pass data-free protocol tests.
Confirm native/cached, repetition and batch-order identities on the allocated
device; check the complete native91 vector before treatment inference.

One fresh scientific attempt has a proposed **3,600-second GPU wall cap**, no
retry or automatic resume. This cap is part of the proposed protocol, not an
assertion that a GPU has been allocated. The fixed deadline starts with the
scientific supervisor, including native checks and uploads; no budget resets.
The existing diagnosis is separate: three native-only probes totaled 69.209
seconds, zero optimizer/backward calls, and ended before this preregistration.

H207 is **WAITING_GPU_ALLOCATION**. Post and wait under the owner ruling:
xbabe1 RTX4090 is assigned H205, xbabe2 RTX3090 H204, xbabe3 RTX3090 H206.
No claim, reservation, speculative launch, CPU fallback, or displacement of the
live flywheel is authorized by this document. Launch requires this preregistration
merged, a reviewed executor and explicit subsequent GPU allocation. The H206
launch-before-review exception does not extend to H207. One GPU job per GPU.

For the eventual allocated host: nice19, numeric threads=1, aggregate process-tree
threads<=5 (including CUDA helpers), aggregate RSS<6,000,000,000 bytes,
host CPU<=85%, load<=85% of available cores, MemAvailable>=16GiB,
user-visible disk<=93.8%, free disk>=40GB, local output<300,000,000 bytes.
Apply any stricter host guard, including xbabe1's >=150GB free CPU-work floor.
Measure admission before input staging and launch; enforce resources throughout.
Hold the qualification lock and an exclusive owner claim for the admitted GPU.
The aggregate-five cap is proposed here explicitly; the H196 attempt-four
approval is historical and does not by itself authorize a new H207 launch.
No optimizer construction/backward, sealed/holdout reads, serving promotion,
foreign process changes, or live-loop writes.

Use only new create-only keys under
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261101/H207/astra-t0305/attempt1/`.
Require If-None-Match '*', VersionId, exact-version GET and payload/member SHA256.
Stream cost arrays to S3; retain small panels/receipts locally outside Git at
`~/jope-data/h207/attempt1/`. Save native, all noise-draw and non-skipped PERM8
costs with candidate/context identities for deterministic replay without inference.
On guard/custody/identity failure, report INCOMPLETE with no efficacy verdict.
Archive source, exact launch manifest, resources, seeds, folds, metrics and
failure details. No object overwrites or deletion of historical H204 objects.
[diagnosis-custody.json](diagnosis-custody.json) supplies the canonical H207
recovery route and records the superseded pre-renumber H204 namespace.

Acceptance for this docs PR: JSON/pin/relative-link verification, git diff --check,
and all five jope-records gates including paper-trace. The owner waived remote
CI in `squash_now_policy_consolidation_1709_t0215.md`; that is not a CI pass.
Central merger only, exact-head review, no author merge. Future executor changes
need targeted pytest/Ruff and fresh source review before any scientific launch.

## H206 backup duty

Primary owner is gpu_fused_redteam_codex; rd_astra is backup only. H206 launch
requires its prereg PR open, exclusive xbabe3 GPU availability, and disk admission
after reserving roughly 50GB headroom (about <=88% used before launch, search
cache cap40GB). Check owner activity and coordinate the handoff before taking
over. H207 never inherits that device reservation or H206's launch exception.
