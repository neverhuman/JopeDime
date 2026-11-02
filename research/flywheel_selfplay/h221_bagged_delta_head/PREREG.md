# H221 — bagged utility-delta head with a safe gate

Prospective preregistration under `hw_t0530_astra_h221.md`. No H221 GPU outcome
has been observed. Merge the exact source before native91 or training.

## Hypothesis and fixed arms

Averaging family-bootstrap delta heads may reduce the variance of H213's
separate-head predictions and let its unchanged B-only harm gate adopt more
contexts safely. The incumbent is never trained. Fixed arms:

- R882: frozen native reference.
- DELTA_SAFE: one H213 head, its own unchanged B gate; diagnostic reference.
- BAG5_SAFE: mean delta from five bagged heads, one B gate; sole primary.
- BAG5_SHUF_SAFE: five matched bags trained on within-context shuffled utilities,
  their mean delta and their own B gate; diagnostic placebo.

BAG5 is fixed before launch; there is no adaptive reduction to BAG3. Each cell
fits eleven heads (one single, five real, five shuffled), each for 128 updates.
Three seeds × five outer folds × eleven heads = 165 fits /21,120 updates.
There is no extra smoke fit, model search or early stopping.

## Source reuse and prospective executor amendments

[H213 PREREG](../h213_gpu_delta_sustained/PREREG.md) supplies the architecture,
feature transform, optimizer, loss, update recipe and exact B-gate rule.
[manifest.json](manifest.json) pins the parent source and all input identities.
H221 changes seeds, family bootstraps, ensemble averaging and the owner decision
rule. Its single-head architecture/training/gate formulas remain unchanged.

H213 repeatedly decoded pool files and recomputed immutable features inside
training. H221 caches the same outcome-free feature matrices on GPU (5 GiB cap;
the pinned TRAIN roster requires 4,179,158,112 float32 bytes). A role-checked
utility cache avoids repeated pool decompression. Role/allowed-context/global-A-
lock checks run before cache access, including when a value was previously FIT
for another outer cell. Features are tested bit-exact against the H213 formula.
Caching changes no labels, context plan, arithmetic loss, optimizer update or
head feature. Resource sampling retains H213's independent sustained-load
clocks; redundant synchronous samples are throttled to one second, while the
absolute supervisor deadline is checked before every callback return/update.

Use the already exercised merged-only/FIFO launch wrapper. Owner's RSS limit
is now 20 GiB. Retain RAM available >=20 GiB and free disk >=200 GB: byte-based
admission replaces H213's obsolete 93.8% disk stop, incompatible with the owner-
acknowledged 95% disk. These amendments are fixed before outcomes. Numeric CPU
threads remain one; nice19. Swap-free is monitored, not treated as a fabricated
scientific outcome or a new stop threshold.

## Cohort, roles and baseline

Real-S3 DL-Train: 1,219 contexts /205 families. Native regression only:
91 DL-Val contexts /19 disjoint families. Every consumed pool is SHA-verified
against the selected manifest; the approved real catalog and source exclusions
are rechecked. [provenance.json](provenance.json) retains dataset IDs and verified
historical S3 recovery. No synthetic dataset, outer-test, sealed, holdout or
meta-test payload is read. Existing labels/gates remain intact.

Fresh seeds221101–221103 and fold seeds221141–221143. Permute sorted families
with NumPy default_rng(seed+40), split into five groups of41. For outer k,
A=group k, B=group(k+1)%5, FIT=the other three groups (123 families). Each family
appears in A once per seed. [fold_manifest.json](fold_manifest.json) pins A IDs;
[fold_jaccard.json](fold_jaccard.json) reports every fold comparison against
H210/H213/H217/H218/H220, using each wave's actual partition algorithm.

Before any head update, unchanged native r882 must match every ordered native91
float64 utility exactly, including mean1.1377941898164525. Use H213's native
recipe: zero context584/prior/SOS, scale.4208, first four storage-index reveals,
pinned float64 projection then float32, 10×32 tokens/masks, first argmin.
Utility is -fitness for gate pass, otherwise -2; evaluation never clips it.
Only r882 is evaluated on DL-Val. No trained head's DL-Val outcome is computed.

A new read-only r882 copy is SHA-verified before model construction. Every r882
parameter stays frozen. Full model hash and DOPE composite
`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`
are checked before and after separate-head training; persistent input/assets
are rehashed at completion. The frozen model is retained on CPU after native
inference. No live flywheel or serving state changes.

## Head fitting and bag construction

Exact H213 head: Linear(744,64), GELU, Linear(64,1,bias=False), 47,744 parameters.
Final weights start at zero; delta(candidate)=raw(candidate)-raw(r882 choice).
Reference delta is identically zero. Inputs: asinh(412 program features), masked
320 token values,10 mask bits, native cost clipped[-10,10], and native cost minus
reference cost clipped[-10,10]. No outcome/family/dataset ID is a feature.

Each of128 AdamW updates averages four full-pool context losses. LR.001,
weight_decay.0001, default betas(.9,.999), global gradient cap1. Targets are
clip(utility_j-utility_reference,-4,4), float32. H213 balanced SmoothL1 gives
half weight to positive-target and half to nonpositive-target candidate means;
if one group is absent, use the whole-candidate mean. No hidden batch change.

DELTA_SAFE preserves uniform FIT-family then uniform-context sampling. Its RNG
streams are H221/seed/outer/plan and H221/seed/outer/init. For bag member j=0..4,
draw123 family IDs with replacement from the123 FIT families, using stream
H221/seed/outer/j/bootstrap. Sample each training family uniformly over those
123 bootstrap positions, preserving multiplicity, then a context uniformly
inside that family. Its context-plan and initialization streams end in
j/plan and j/init. Initial seeds are reduced modulo2^63-1. All five members have
distinct first-layer initializations; final-layer zero initialization remains.

The shuffled partner of each real bag shares its bootstrap population,
initialization and complete context plan. It permutes true utilities within
context with one fixed H221/seed/outer/j/context/shuffle stream, then constructs
the same delta target. Evaluation uses true outcomes. There are no shared
optimizer states. All165 fits must show actual128 updates, nonzero finite
gradients, changed state hash and a changed future FIT forward on CUDA.
Receipts retain family multiplicities, plan hash, init seed and all losses.

## Ensemble, B gate and global lock

Compute each head's float32 delta vector, then its ordered arithmetic mean in
float64 across five members. Average predictions before picking or gating;
never average utilities or independently selected thresholds. The single head
uses the same mean operation with one member. Reference delta remains zero.

For each of the three trained arms, score B with the fixed grid
0,.001,.01,.05,.1,.2,.5,+infinity. Pick first argmax delta only when its value is
strictly greater than tau; otherwise use r882. Eligible B threshold: equal-family
mean gain strictly >.001 and <=5% families with gain<-.01. Choose largest mean
gain, ties choose larger tau; no eligible threshold means infinity. This is the
unchanged H213 gate. B never fits a head or selects bag members/checkpoints.

All165 terminal heads and45 arm/fold gate decisions must be fsynced and
exact-version S3-recovered under one global lock before any A-role outcome
read. A family may be FIT in another independent cell; this is ordinary cross-
fitting, not a claim that its labels were globally never opened. Current FIT
closures exclude current A and B families. No A result changes a B gate.

## Primary readout and decision

Primary BAG5_SAFE vs R882. Average paired context gains within each family,
then equally across205 families within each seed; take the median of three seed
means. Draw10,000 paired family bootstrap samples, seed221150, sharing each draw
across all seeds and arms. Report two-sided95% percentile CI; LCB denotes its
lower endpoint (the H213 convention). One-sided p=(1+bootstrap medians<=0)/10001.
Holm m=1; DELTA_SAFE and BAG5_SHUF_SAFE cannot substitute for a failed primary.

Owner's ADVANCE rule: LCB>0, p<.05, median gain strictly >+.001, harm<=10%, and
at least2/3 positive seed means. Exact ties reject. Harm counts families with
seed-average paired gain<-.01. Otherwise NO_ADVANCE. This supersedes H213's
primary arm and adds the owner's point-margin and positive-seed requirements;
it does not move the bootstrap null to .001. No across-campaign multiplicity
claim is made. These reused TRAIN development results are not sealed validation.

Report all gates/grids, adoption counts/rates, absolute OOF utilities and every
trained-arm contrast vs r882. H200 family-grouped Bradley–Terry Elo anchors
r882=1500 (seed221151); utility remains decisive. No AUC, speed metric or U4
measurement. No serving promotion, including after ADVANCE.

## Resources, lifecycle and custody

One xbabe2 GPU0 FIFO attempt under qualification.lock, exact UUID in manifest;
one90-minute absolute budget covering setup/native91/training/scoring and worker
finalization. Latest launch15:00Z; hard finish<=16:30Z on2026-11-02. No retry,
resume, budget reset, adaptive bag reduction or zero-filling missing outcomes.
Resource stop: load5>85% effective cores immediately; load1>85% or hostCPU>85%
for300 continuous seconds with independent reset clocks. Admission is immediate.
RAM>=20GiB, free>=200GB, process-tree RSS<20GiB, local output<1GB, feature cache<=5GiB.
All guards and device/exclusive-consumer checks remain active during execution.

Create-only S3 PUT, exact VersionId GET and SHA256 recovery beneath the pinned
H221 prefix. Bulk stays outside Git. Preregistration: focused CPU tests, Ruff,
five records gates, source pins/links, exact-head same-agent review, then the
owner-authorized targeted-check squash merge. Full manual CI is not claimed.
Closeout: README, small metrics, both wave indexes, complete S3 recovery and
reviewed squash merge. Preserve partial evidence and close INCOMPLETE if any
required source/baseline/training/lock/resource/custody condition fails.
