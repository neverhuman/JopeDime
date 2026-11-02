# H215 — pessimistic argmin from fresh ensemble disagreement

Owner: rd_astra / [ASTRA]. Authority: `hw_t0230_astra.md`, 2026-11-02.
**PREREGISTERED, NOT RUN.** One 90-minute xbabe2 GPU0 attempt. Exact open
preregistration PR or merged source is authorized; no serving promotion.
Preregistration by 10:15 UTC, finish by 12:00 UTC. Latest launch 10:30 UTC.

## Question and fixed decisions

H213 completed its trained delta-head test but did not advance: its safe policy
had gain +0.007008, CI [-0.001146, +0.018764], p .076892; raw choices harmed
87/205 families. H215 tests a different mechanism: uncertainty from disagreement
among independently initialized, FIT-trained adapters, added as a pessimism penalty.

For each context and candidate c, compute each member's within-context cost
z score using population standard deviation (denominator floor 1e-12). Sigma(c)
is the population standard deviation across exactly ten such member vectors.
Reference-centering before the across-member variance makes identical members
have exactly zero disagreement without changing the mathematical statistic.
The rule is first argmin of `cost_r882(c) + kappa * sigma(c)` in float64.
At kappa=0, use the original native vector's first argmin directly, preserving
its dtype and tie behavior exactly; this has a dedicated refusal/control test.
No incumbent prediction is included as an eleventh member of sigma.

Arms (the full pool, never a shortlist):

- **PESS_NEST**, sole primary: kappa selected by nested family CV within FIT.
- PESS_FIX: kappa=1, diagnostic.
- PESS_SHUF: sigma permuted within the context, same selected kappa as PESS_NEST;
  permutation is fixed by SHA256 H215/seed/outer/context/placebo.
- OPT: kappa=-1, diagnostic, cannot win.
- R882: unchanged native reference.

The directive calls `{0,.25,.5,1,2}` a FIT-quantile grid. These are fixed numeric
kappa coefficients (2 is not a quantile probability), used exactly as listed;
no additional scaling or data-selected candidate coefficients are introduced.
FIT-only sigma quantiles are recorded descriptively. Grid selection uses the
mean of within-family inner-OOF utility gains over all 123 FIT families. Choose
maximum gain, ties choose smaller kappa; nonpositive best gain falls back to 0.
No B or A outcomes choose kappa, member initialization, updates or checkpoint.

## Roles and nested fits

Use the pinned historical real-S3 H50/H58 cohort: 1,219 DL-Train contexts from
205 families, with 91 DL-Val contexts from 19 disjoint families used only to
verify native identity. No sealed, holdout, meta-test or outer-test rows enter.
This is reused development data, not new labels or sealed qualification.
Every pool is SHA-verified against the selected manifest; catalog/source
exclusions and real provenance are checked. Input arrays preserve existing labels.

Outer seeds 215101–215103 have independent fold seeds 215141–215143. Permute
sorted families with `default_rng(seed+40)` and divide into five groups of 41.
For outer k: A=group k, B=group(k+1)%5, FIT=the other 123 families. Exact A context
IDs are fixed in fold_manifest.json. Each family appears once in B and once
in A per outer seed. Model and optimizer state are reset between all fits.

Within each FIT, independently permute its 123 families with a SHA256-keyed
H215/seed/outer/inner stream and split into three groups of 41. Train ten fresh
members on 82 families; score the held 41 families for every fixed kappa. Repeat
all three inner folds, pool their 123-family OOF gains, then lock kappa. Refit
ten fresh members on all 123 FIT families and score only B at this stage.

There are 3 seeds × 5 outer cells × (3 inner ensembles +1 final ensemble) × 10
members = **600 member trajectories**, each with 40 updates: **24,000 updates**.
The nested fits are included in the same 90-minute cap. No K5 fallback, pilot
budget expansion, fewer seeds/folds or shortened fitting is allowed. A budget
stop is INCOMPLETE and consumes the attempt.

Member seeds are 215201 through 215800, uniquely mapped by
`215201 + (seed_index*5+outer)*40 + phase*10 + member`, phase 0–2 inner, phase 3
final; member 0–9. Each minibatch RNG is `member_seed*1000 + step`, step 0–39.

## H190 adapter recipe and native identity

Only the 47 trainable adapter_10tok tensors  / 29,211 parameters are optimized.
Construct each fresh adapter using H190's constructor and pinned normalization
buffers, on the existing device; never move the shared frozen anchor to CPU.
The original r882 checkpoint copy is read-only and remains unchanged.

Forty AdamW steps, lr 1e-4, weight decay 1e-4, betas(.9,.999), norm clip 1.0;
eval mode, no active dropout. Each step samples 8 FIT contexts without replacement,
then 24 candidates each: four top-fitness reveals and 20 random remaining candidates.
This is H190's training recipe; evaluation still uses the native first four
storage-index reveals and full-pool first argmin. The pinned H146 loss is mean
SmoothL1 plus 0.1 KL at temperature 0.2, using clipped fitness [-4,2] or failure
cost 2. Evaluation utility is unmodified `-fitness` or `-2.0` on failed native gate.
The 41 active  / 6 structural None gradients are checked. Every fit must prove
nonzero data gradients, changed parameter SHA and changed future FIT forward.
Losses, hashes, update count and CUDA device are durably saved per fit, before
another fit starts; every fitted member state goes to create-only S3 custody.

Before any candidate forward or optimizer step, singleton CUDA native r882 must
reproduce the entire ordered 91-utility reference vector bit-exact, including
mean **1.1377941898164525**. Frozen native scoring uses zero context584/prior/SOS,
scale .4208, pinned float64 projection followed by float32, fixed 10x32 tokens and
masks. Mean-only agreement fails. Frozen full state is hashed before/after native
inference. Cached static adapter scoring must then match native full-pool costs
bit-exact on all 1,219 TRAIN contexts before the first fresh member is constructed.

Frozen input tensors, anchor terms and parameter-free static terms are cached
on GPU; they contain no unrevealed outcomes. Label access is role-scoped and A
access refuses before the global lock. Only selected FIT rows supply training
labels; held inner folds never train their own member. Frozen non-adapter state
and DOPE `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`
are rehashed around the run and after every ensemble. All pinned files are
rehashed again at completion. CUDA deterministic algorithms, TF32 off.

## Calibrated pooled selector and final inference

After all 15 cells finish, each arm's B panel has shape(3,1219), covering 205
families exactly once per seed. Import the exact H177 decision rule whose SHA
is 707c4903a77d48ffc8ae14c820a626df5c2858d3d24df35db3f9718d120bafc4 and verify
its committed H178 calibration receipt (four-arm replay; null/harmful rate 0).
Use the one-to-one panel map PESS_NEST→CTRL, PESS_FIX→ACC4,
PESS_SHUF→ACC16, OPT→OC15. These are selector slots only, not claims that H215
uses those earlier treatment mechanisms. All candidates are terminal 40-step fits.

The unmodified B selector requires mean gain>0.001+1e-12, seed-averaged family
bootstrap LCB95>0 and Holm4 p<.05, with 10,000 draws at seed 215150. Write, fsync
and exact-VersionId S3-verify all member proofs, chosen kappas, B panels and
pooled decisions in selection-lock.json before any A outcome role is accessed.
A family may be FIT in a different outer cell; states remain independent.

After the lock, report all four raw A policy panels regardless of B admission;
these are diagnostics if not admitted, not deployed fallback utility. The sole
primary may earn CANDIDATE only if PESS_NEST passed the calibrated B rule AND:

- Median of three seed equal-family A gains is strictly positive.
- Paired 10,000-family bootstrap 95% CI lower bound is strictly positive.
- Holm4-adjusted one-sided bootstrap p is<.05, including all four registered arms.
- At most 10% of families have seed-mean utility gain<-.01.

Otherwise NO_ADVANCE; exact ties reject. A bootstrap resample is shared across
seeds; recompute the median on every draw. p=(1+draws<=0)/10001. No across-wave
multiplicity or generalization claim. Diagnostics and Elo cannot override primary.
Report all gains/CIs, kappas, adoption, absolute OOF utility and H200 Bradley–Terry
Elo with r882=1500,1,000 family draws at 215151; no AUC or speed verdict. U4 is null
with reason: no verified multi-proposal budget.

## Executor, resources and custody

xbabe2 GPU0 UUID GPU-44c63d19-492a-ab46-cc6a-56e12a2ef51b only; hold
`/home/ubuntu/qualification.lock` throughout the worker. Use the shared
`/home/ubuntu/jope_plan/qualification.queue` FIFO ticket ordering; retain tickets
and mark own completion without deleting foreign files. Refuse another GPU
consumer. No live flywheel interaction and no other host's GPU allocation.

Preserve H213's exact guard: load5>85% of effective-affinity cores stops
immediately; load1>85% or CPU>85% uses independent 300-second monotonic clocks,
reset on recovery. Instantaneous: disk used/(used+available)<=93.8%, free>=40GB,
MemAvailable>=16GiB, process-tree RSS<6GB. Record swap-free bytes (including 0),
but use MemAvailable for memory admission. Numeric threads 1, nice 19; driver
threads diagnostic. Worker samples at most once per second plus its 2s timer;
independent supervisor checks every 2s. Reentrant resource logging remains guarded.
Local output<300MB. Claim-to-terminal wall<=5,400s, including input verification.

One create-only attempt claim, exact source/input/package checks, no implicit
resume or retry. Waits for FIFO happen before claim; refuse launch after 10:30Z.
Failures retain actual update count and partial proofs, with null unmeasured
metrics and reasons. Keep all source/evidence; no cleanup assumed for admission.

S3 prefix: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261102/H215/astra-t0230/`.
Every artifact uses create-only writes, exact VersionId GET and SHA verification.
Only small receipts remain locally outside source; no model arrays in Git.
Acceptance: data-free H215 tests (zero-penalty native identity, role/lock order,
nested splits, actual optimizer/forward proof, selectors, FIFO/resource guards),
Ruff, five records gates, exact-head review and open PR. The owner authorizes
squash merge after targeted local checks; no remote CI success is inferred.
