# H211 — train improvement predictions, measure raw and gated adoption

Owner: rd_astra / [ASTRA]. Authority: `gbw_0040_rd_astra.md` (2026-11-02).
Status: **NOT RUN**. One prospective GPU attempt after local acceptance, an open exact-head
preregistration PR and resource admission, explicitly authorized by gbw_0040. No self-merge or serving promotion.

## H209 failure and prospective change

H209 attempt1 used merged d609dab46dc725925008e750e0813bfe40b44ab4.
Native91 parity passed; SIGALRM reentered the buffered resource writer during
TRAIN native-cost generation, stopping after44.374 worker seconds with zero
updates, no fitted head, no B lock and no A scoring. Closeout #1752 retains that
failure and consumed attempt. H211 is a separately authorized new attempt, not
an H209 retry or hidden budget reset. The hypothesis, model, loss, updates,
thresholds, inference and resource limits below remain unchanged. H211 uses
fresh seeds211101-211103/folds211141-211143 and fresh keyed RNG streams.

The only executor fix guards check() against reentry with a try/finally-cleared
flag. A nested SIGALRM returns while the current check performs all original
logging and limits; both two-second timers/supervisor enforcement remain.
A deterministic test delivers SIGALRM from inside the buffered writer: it
reproduces the old exception and passes with the fix. Another test verifies
that a guard failure releases the flag and does not suppress subsequent checks.
The H209 runner receives the same prospective fix; its original d609 execution
and one-attempt claim remain immutable and H209 is not relaunched.

Launch authority differs deliberately: gbw_0040 permits H211 after its PR is
OPEN. launch.py binds a non-draft OPEN PR on rd/h211-gpu-delta to the exact
head SHA and main base, or an already MERGED PR to its exact merge SHA. Closed,
stale, draft and unrelated PRs refuse before the attempt claim. No self-merge.

## Question and limits

H204 reported zero adoption and fallback to r882 for all its width-128 arms.
Its source review at `d8d241a3d6dd2af597b3e033141dfe79711ca758` also found
unconsumed fresh-label receipts, insufficient native-vector verification,
a zero-forward-effect bias and resource/statistical defects. Its subsequently
updated closeout does not by itself qualify those historical runs. H211 treats
zero adoption as a motivating observation, not proof of a width bottleneck.

H200's completed nested blend was NO_ADVANCE: primary mean gain +0.035741,
95% interval [-0.000677,+0.072160], Holm p=0.064794, 71/205 harmed families.
H208 separately preregisters a harm-constrained CPU blend. H211 drops the width
search and tests a new, small CUDA head predicting improvement over r882,
with a fixed harm-constrained selection rule. Raw choices diagnose prediction
quality separately from the selection gate. This is a fixed experiment, not
an adaptive attempt to increase acceptance by weakening live gates.

This uses the historical static real-S3 cohort, **not newly admitted D4 or fresh
Rust-CV labels**. Outcomes and their native gates remain unchanged. Existing
cohort reuse and the validation-selected r882 incumbent limit generalization;
a positive result is a research candidate, never qualification or promotion.

## Inputs, roles and native baseline

[manifest.json](manifest.json) pins the full source, package and input closure;
[provenance.json](provenance.json) preserves the prior actual versioned S3
catalog GET, input recovery receipts, 276 real dataset IDs and zero synthetic
provenance. All 1,310 pool bytes are SHA-checked when consumed against the pinned
selected manifest; catalog and source-manifest exclusions are checked again.
The approved catalog includes both registry-real-v1 and jailgun-regression-hunt;
collection-name equality is not a replacement for the pinned real catalog join.
No outer-test rows, sealed families or excluded payloads enter the worker.

DL-Train: 1,219 contexts / 205 families; DL-Val: 91 / 19 disjoint families.
The 91 DL-Val contexts are used only for the frozen native baseline identity.
Before any optimizer update, singleton CUDA native r882 inference must reproduce
**every ordered float64 utility and mean 1.1377941898164525 exactly**. An equal
mean alone fails. Native recipe: zero context584/prior/SOS, scale .4208,
first four storage-index reveals, pinned float64 projection then float32,
pinned 10x32 tokens/masks, first argmin. Utility is `-fitness` if the existing
native gate passes, otherwise -2. Evaluation never clips utility. Pool decoding
before a role is permitted reads only program features and those four reveals.

All r882/DOPE parameters stay frozen. Full state hashes before/after native
inference must match, in addition to historical DOPE composite c51a56d0….
H211 trains only its separate head. The production code path does not import
H204's training or fresh-label helpers.

For each of seeds 211101, 211102, 211103, permute sorted families with NumPy
`default_rng(seed+40)`, split into five groups of 41. For outer k, A=group k,
B=group(k+1)%5, FIT=other three groups (123 families). Exact A context IDs are
in [fold_manifest.json](fold_manifest.json). Each family appears in A exactly
once per seed. Family siblings stay together. All 30 terminal heads and B
threshold decisions must be durably locked and S3-verified before **any A
endpoint is read**. A context can legitimately be FIT in another outer cell;
model/optimizer state is reset independently in every cell. B influences only
its cell's threshold, never fitting, early stopping or checkpoint choice.

## Actual GPU training

Head: Linear(744,64), GELU, Linear(64,1,bias=False), 47,744 parameters. Final
weights start at zero. A candidate's delta is raw(candidate)-raw(r882 choice).
The reference delta is identically zero; no canceled trainable output bias.
Features: asinh(412 program features), masked 320 token values, 10 mask bits,
native cost clipped [-10,10], native cost minus reference cost clipped [-10,10].
No outcome, dataset ID or family ID is a head feature.

Two fits per cell: REAL and SHUFFLE. Each uses **128 AdamW updates**, lr .001,
weight decay .0001, global gradient norm cap 1. Each update averages four
contexts sampled with replacement: first uniformly among FIT families, then
uniformly among that family's contexts. All candidates in each selected context
are used. The paired fits share initialization and the complete context plan.
Streams are SHA256(`H211:` plus colon-separated seed/outer/label components),
first eight digest bytes little-endian, as defined in science.py; init is reduced
modulo 2^63-1. No early stopping, hyperparameter sweep or validation fitting.

Target: `clip(utility_j-utility_reference,-4,4)` float32. SmoothL1 per candidate,
with 0.5 weight to the positive-target mean and 0.5 to the nonpositive-target
mean; if either group is absent, use the full candidate mean. SHUFFLE permutes
utilities within each FIT context with one fixed per-cell/context RNG stream,
then forms the same targets. This preserves the outcome marginal while
breaking the intended program/outcome association. Evaluation uses true labels.

Total: 3 seeds × 5 outer cells × 2 fits × 128 = **3,840 real CUDA updates**.
Each fit records losses, nonzero data gradients, changed state SHA, actual CUDA
device, and a changed future forward on its first FIT context. Failure of any
proof closes INCOMPLETE. REAL and SHUFFLE have matched architectures, context
plans and update counts; this is not a claim of exact GPU-time matching.

## Selection, arms and inference

Use first argmax delta if its value is strictly greater than threshold tau;
otherwise choose native r882. Ties at zero and infinity always fall back.
For each fitted head, evaluate the complete B-only grid
`0,.001,.01,.05,.1,.2,.5,+infinity`. For each threshold, compute mean utility
gain within each B family, then equally across families (outer average uses
`math.fsum/n`). Eligible: mean gain strictly >.001 and at most 5% of families
with gain < -.01. Choose maximum mean gain, ties choose the larger threshold.
No eligible threshold means infinity. This constraint is fixed before outcomes.

Arms:
- R882: unchanged native reference.
- DELTA_RAW: REAL head, threshold 0, diagnostic.
- DELTA_SAFE: REAL head, B-selected threshold; **sole primary**.
- SHUF_SAFE: SHUFFLE head, its own B-selected threshold, diagnostic placebo.

Report all thresholds/grid results, per-seed raw/safe adoption counts and rates,
absolute OOF utility, native 91-context value, and each arm's contrasts versus
R882. Positive raw adoption with rejected safe adoption localizes the operational
failure to B admission; poor raw utility remains a model failure. These are
descriptive distinctions, not identified causal effects of the historical H204
code. Do not select a new threshold or train more after viewing A.

Primary effect: within-family mean paired utility delta, equally weighted across
205 families, per seed; statistic is median of the three seed means. Bootstrap
10,000 paired family resamples shared across seeds, RNG 211150; recompute median
on every draw. Percentile 95% interval. One-sided p=(1+draws<=0)/10001; Holm m=1.
Harm: fraction of families whose seed-mean paired gain is < -.01.
CANDIDATE iff primary lower interval >0, p<.05 and harm<=10%; otherwise
NO_ADVANCE. Diagnostic arms cannot win the primary gate. No across-campaign
multiplicity claim is made.

Also report Bradley–Terry Elo against r882 anchored at 1500 using the pinned
H200 analysis: utility ties within 1e-4, equal family weights, 400 iterations,
prior .01, 1,000 paired family bootstrap draws, RNG 211151, median across seeds.
These are nested DL-Train held-family development scores, not sealed qualification.
No verified multi-proposal budget is spent; U4 is unmeasured, not zero.

## Execution, custody and stop rules

One attempt, **xbabe2 RTX3090 UUID GPU-44c63d19-492a-ab46-cc6a-56e12a2ef51b**,
GPU index0; no xbabe1 live-loop interaction, no xbabe3 GPU allocation, no xbabe0.
[launch.py](launch.py) refuses a draft, closed, wrong-branch or wrong-source PR, holds
`/home/ubuntu/qualification.lock` throughout, checks UUID and absence of other
GPU consumers, and writes a create-only attempt claim before spawning a worker.
Immutable Git blobs supply all H211 and pinned parent modules; no local source
checkout or additional worktree is created. Source, environment and input hashes
are checked before inference. This is a small GPU job with one numeric CPU
thread at nice19; no multi-host CPU fit is needed.

Bounds, checked before inference and every two seconds by both worker and
supervisor: 1-hour wall from launch (includes input verification), one attempt,
CPU<=85%, 1-minute load<=85% of effective affinity, available RAM>=16GiB,
process-tree RSS<6GB, free disk>=40GB, **user-visible used/(used+available)<=93.8%**,
local output<300MB. No process-affinity reduction hides the real host load.
OMP/MKL/OpenBLAS/torch intraop=1, interop=1, nested numeric parallelism disabled;
measured worker numeric pools must satisfy the shared <=4 numeric-thread policy.
OS/CUDA driver threads are diagnostic and exempt under that merged policy.
Float32, deterministic algorithms, TF32 disabled, CUBLAS workspace :4096:8.

Any budget/resource/input/baseline/update/lock/custody failure stops and closes
INCOMPLETE; no automatic retry, resume or budget reset. Supervisor interrupts
and then kills its own process group only if graceful exit exceeds 20 seconds.
No other process or data is changed. No disk cleanup is assumed for admission.

Create-only, exact-VersionId GET+SHA verified S3 objects beneath
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261101/H211/astra-t0040/`:
preregistration source/pins/test receipts, then attempt1 launch, native score
vectors, all 30 terminal heads and B prediction vectors, complete selection lock,
all A prediction vectors/panels, metrics or failure, resource/terminal evidence.
Local small receipts live outside source checkouts. Retained bulk is removed
only after verified custody; no shared inputs or another owner's files are removed.

Acceptance before launch: focused CPU refusal/training tests, Ruff format/check,
all five records gates and path lint at the stopped source, head-bound implementer review and
an open exact-head PR; then live source/resource/GPU checks. External merge is
not a launch prerequisite under the explicit gbw_0040 owner instruction. CPU fixtures are not GPU
qualification. The parent main's separately reported audit failures remain open;
this research does not certify those capabilities.
