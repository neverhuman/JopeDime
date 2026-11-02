# H208 — harm-constrained nested blend with fixed-length fits

Status: PREREGISTRATION PROPOSED; NOT RUN; executor not implemented.
Owner: rd_astra / [ASTRA]. Authority: `gbw_2107_rd_astra.md` (2026-11-02 04:07Z).
H208 is reserved after checking fresh main and open PRs; H204–H207 remain assigned.

## Reaction to H200

[H200 closeout PR #1725](https://github.com/neverhuman/JopeDime/pull/1725) records
COMPLETE / NO_ADVANCE: primary NEST_L median gain +0.035741, CI
[−0.000677,+0.072160], Holm p0.064794, and 71/205 harmed families (34.63%).
Nested lambda selection maximized inner mean gain without a harm constraint.
Seven fits also ended before300 iterations despite the no-early-stopping prereg.
The monitor recorded disk use above93.8% without enforcing a percentage cap.
These are measured outcomes and deviations, not proof that one repair will help.

H208 tests whether a **FIT-only 5% harm constraint on lambda selection** reduces
out-of-family harm while preserving a positive native gain. Every arm uses fresh
models with explicit `early_stopping=False`, `max_iter=300`, and verified
`n_iter_==300`. A repaired unconstrained nested selector shares exactly those
fits, so its comparison with the primary isolates the selection rule. There is
no attempted comparison against unrepaired H200 as a controlled experiment.
RANK025 is dropped: its H200 median utility gain was negative and harm40%.

This differs from [H203](../h203_blend025_gated/PREREG.md): H203 gates a fixed
lambda0.25 per context using an outcome-free confidence signal. H208 selects
one lambda per outer cell using nested FIT family utility and harm. H203 stays
with its owner; its results are neither assumed nor consulted to alter H208.

## Frozen inputs and outcome boundary

Use the inherited H139/H195 real-S3 regression closure: 1,219 DL-Train contexts
in205 families and91 DL-Val contexts in19 families. [manifest.json](manifest.json)
pins the r882 checkpoint, selected roster, DOPE hash, catalog authority and
parent source files. Before outcome access, validate all declared hashes, join
all context IDs to their family/split metadata, and reject changed/incomplete
rosters, unresolved family aliases and any protected-family overlap. No binary,
sealed, holdout, meta-test or live-loop data enters this worker. If the current
owner-mandated eligibility check cannot validate the inherited roster, stop;
do not silently shrink, augment or substitute the cohort.

Native r882 full-pool first-argmin utility must match the entire pinned91-entry
DL-Val reference exactly and mean **1.1377941898164525** before any TRAIN outcome
pool or treatment computation. DL-Val is baseline-only, never treatment scoring,
selection or a reported lift. Preserve the native kernel and candidates. Check
checkpoint and DOPE hashes before/after each scoring stage. Neural optimizer
construction, backward and adapter/DOPE updates are forbidden. HistGBM fitting
is the specified treatment; “zero updates” refers to the frozen neural model.

Do not read or reuse H200's predictions, selected lambdas, fits or A panels during
H208 execution. The family IDs in [fold_manifest.json](fold_manifest.json) were
regenerated from outcome-free H200 roster metadata, using fresh seeds only.
This remains a repeated analysis of an existing cohort, not an independent new
population or a sealed confirmation experiment.

## Arms, seeds and folds

Report R882; **NEST_HARM (sole primary)**; NEST_REPAIR (unconstrained control);
FIX025 (fixed-lambda control); and SHUF_HARM (descriptive placebo). No rank arm.
Outer seeds208101–208103 use fold seeds208141–208143. Candidate-sampling seed
208140, family-bootstrap seed208150, Elo seed208151, shuffle root208160. The
exact integers were audited against prior source; incidental substrings inside
unrelated floating-point values are not seed uses.

Sort the205 family IDs, permute with NumPy default_rng(fold_seed), array_split
into five equal41-family folds and sort each fold lexically. Freeze this manifest
before outcome access. A=k, B=(k+1)%5, FIT=the other three folds. A/B/FIT families
are disjoint per cell; A covers every TRAIN context exactly once per seed.
B is an exclusion fold only: do not select, score treatments or report a B gain.
Check all identities and memberships before fitting and before each outer score.

## Fixed fits and leakage-free calibration

In each seed fit the H200 plan: five outer models excluding adjacent{A,B}, ten
unique two-fold inner models excluding{A,B,j}, and five one-fold calibrator
models. **60 fits total**. Use HistGradientBoostingRegressor with max_iter300,
learning_rate0.08, max_leaf_nodes63, l2_regularization1.0, early_stopping=False,
random_state=outer seed. Use the pinned1,089 features (program412 + lm48_z +
DOPE629),192 candidates per FIT context, and context-centered utility targets.
No additional hyperparameters, early stop, adaptive fits or outcome-driven seed
changes. Record actual iterations for every fit and stop if any differs from300.

For each outer cell and each inner held-out FIT fold j, its raw predictions come
from the model trained on the other two FIT folds. Calibrate using those other
two folds only: prediction rows for either calibration fold come from the model
trained on the opposite fold, excluding A,B,j and the predicted fold. Fit the
same H124 positive-slope calibration to gate-valid candidate slots4 onward.
Thus neither the GBM nor calibrator for j uses j outcomes. For final outer A
scoring, calibrate using all three cross-fitted FIT predictions and FIT outcomes;
the outer model itself trains on all three FIT folds. No A or B outcome enters
fit, calibration, lambda selection, or a fallback decision.

## Harm-constrained selection

Lambda grid is fixed at{0,.1,.15,.2,.25,.3,.4,.5}. Compute each candidate lambda's
native first-argmin utility on every inner-held-out FIT context. Within each FIT
family, average context utility differences against r882; each family then has
equal weight. Let g(lambda) be mean family delta and h(lambda) the fraction of
FIT families whose delta is **strictly below−0.01**.

NEST_HARM admits a nonzero lambda only if h(lambda)<=0.05 AND
`g(lambda) - g(0) > 0.001`. Choose maximum g among admitted lambdas, with exact
ties choosing the smaller lambda. If none qualify, choose0 and return a copy of
the native r882 cost vector bit-for-bit. Lambda0 is always a feasible fallback;
equality at0.001 is not a gain, equality at−0.01 is not harm. Do not round values
before comparisons. The5% FIT cap is a hypothesis to test, not a guarantee of
10% outer harm. It is fixed before H208 outcomes and is not tuned on A.

NEST_REPAIR applies the identical gain/tie/fallback rule **without the harm
constraint**. FIX025 always uses0.25. SHUF_HARM applies the full harm-constrained
selection to within-context permuted calibrated GBM costs; it uses independent
SHA256-derived streams binding root208160, seed, inner/outer stage, cell and
context ID. Keep each score multiset; permute neither labels nor r882 scores.
All arms share the same fresh60 fits. Mean-centered GBM targets and all native
cost/utility precision rules remain those of the pinned H200 implementation.

Write all15 cells' scales/calibration, per-lambda FIT family deltas, gain/harm,
selected lambdas, fit receipts and source/input identities into a fsynced
immutable selection lock before **any A treatment utility** is computed. Bind
its SHA256 in the launch/result receipts and recheck before every A cell. A
outcomes never alter a lambda, hyperparameter, exclusion or interpretation rule.

## Endpoint, analysis and decision

Utility is−original-fitness at full-pool first argmin if gate-valid, otherwise−2.
Preserve first-index tie breaking, original float64 fitness and exact−2 counts.
For each seed, pool A across five cells and average contexts within family,
then families equally. Primary estimand: median across three seeds of
NEST_HARM minus R882 equal-family gain.

Use10,000 paired-family bootstrap draws, seed208150, resampling the same205
family indices across all seeds and arms. Each replicate averages each seed's
sampled family deltas then takes the three-seed median. Report percentile95% CI,
positive-seed count, and `p=(1+count(bootstrap_median<=0.001))/10001`. Holm m=1
because only NEST_HARM vs R882 is primary. The other arms and NEST_HARM minus
NEST_REPAIR are descriptive, with intervals, not additional discovery claims.
Resampling conditions on fixed fits, folds and lambda choices; it does not refit.
The three reshufflings share families and are not independent replications.

Outer harm is the fraction of families whose **mean over three seed deltas**
is strictly below−0.01. Report per-seed harm, family win/tie/loss and exact ties.
**CANDIDATE requires all three:** primary LCB95>0, p<0.05 and outer harm<=0.10.
Boundary ties for CI or p reject. Otherwise NO_ADVANCE. Any missing cell,
resource/identity/custody failure or iteration-contract failure is INCOMPLETE;
never replace unknown metrics with zero. No early efficacy stop.

Report all seed gains, median/CI/p, harm, chosen lambdas and native fallback
frequency. Include H200-style family-paired Bradley–Terry Elo against r8821500,
with1,000 family draws at208151 and its pinned1e-4 Elo tie tolerance; Elo is
secondary and cannot override native-utility or harm gates. No treatment DL-Val,
AUC or speed/lift inference. CANDIDATE means evidence for later separately
sanctioned confirmation, not serving promotion or a globally corrected claim
across all prior H waves.

## Execution admission and custody

**No executable or launch is supplied by this docs PR.** Before execution,
implement/review the selector and fixed-iteration checks; adopt the merged
numeric-thread admission helper; pin all source, packages and inputs; pass
boundary/fallback/leakage/300-iteration data-free tests. Research tests must set
`PYTHONPATH=$PWD/python/src:$PWD` and verify imported jope belongs to the checkout.
Reject stale user-installed imports. Freeze the actual execution SHA, commands,
package versions (including sklearn, torch, threadpoolctl), host and deadline in
launch.json before a single fresh attempt after preregistration merge.

Planned host is **xbabe3 CPU only**, nice19, one numerical worker, maximum4 CPU
cores. Set OMP_NUM_THREADS=4, MKL/OPENBLAS/NUMEXPR/RAYON threads=1,
OMP_MAX_ACTIVE_LEVELS=1, OMP_NESTED=FALSE before numeric imports; torch intra-op=1.
No simultaneous numerical workers or nested parallelism. Capture and admit
actual worker native-pool/torch thread limits via the reviewed shared numeric
helper at startup and every fit/scoring boundary. Numeric bound<=4; total OS
threads are logged, not mistaken for numeric concurrency. Missing/unknown worker
reports refuse. No CUDA initialization or GPU context.

One4-hour wall attempt, at most16 CPU-hours; no retry/resume or budget reset.
RSS<6GB, MemAvailable>=16GiB, free disk>=40GB, local output<500MB,
host CPU<=85%, load1<=85% of available cores. **User-visible disk must remain
strictly below93.6%**, both before staging/launch and continuously. Stop on a
breach; the original H200 free-space-only monitor is insufficient. Resource
sampling uses the stricter effective CPU affinity, not an inflated host count.
Current lightweight preflight supports CPU preparation only, not future admission.

If CPU admission fails, record **QUEUED_NEXT_FREE_GPU** per the owner instruction;
this queue state grants no GPU access. Any future GPU route needs explicit device
allocation and a reviewed execution amendment, while CPU fits still need CPU
admission. No claims on xbabe1/H205+flywheel, xbabe2/H204 or xbabe3/H206 GPUs; no
foreign process changes. H207's allocation request is separate and unchanged.

Write create-only S3 objects under
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261101/H208/astra-t2107/`.
Use unique attempt keys, If-None-Match '*', required VersionIds and exact-version
GET plus payload/member SHA256 checks. Archive native/raw prediction arrays,
FIT selection evidence, A panels, fold/lock manifests, runtime guards and source
for deterministic replay without fresh inference. Keep small local receipts;
remove only verified owned duplicate bulk. No HWM pointer before a complete
qualified result. Closeout includes every deviation and measured stopping reason.

Documentation acceptance: JSON/family-fold/source-pin/link verification,
`git diff --check`, and five jope-records gates. No outcome-bearing run is part
of this preregistration. Remote CI is owner-waived, not a passing check. Open
PR, exact-head review and central merge; author must not merge it.
