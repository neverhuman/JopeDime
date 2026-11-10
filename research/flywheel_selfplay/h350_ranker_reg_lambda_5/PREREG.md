# H350: ranker reg_lambda 5

Owner sol_v1_hwv; authority hw_t0529_solhwv_h350_reglambda.md, 2026-11-05.
Committed H347 source copied byte-verified from v1/next snapshot
7ee4c9be00e2f3f3500beca099cad676577e6f04, path
research/flywheel_selfplay/h347_ranker_l1_reg_alpha/. Its seven vendor pins and
H287 213b30ad52e1ce48927bba960ed609bd98ab4dfe remain unchanged.
This preregistration must be nonforce-published to v1/next before treatment fits.

## Fixed intervention and hypothesis

Stronger ranker leaf-weight L2 may damp single-context swings and sentinel -2.0
recoveries. ONE arm L2R5 changes the EXISTING ranker constructor reg_lambda1.0
value to5.0 in topk.fit_ranker and bagged.fit_ranker_n (inner, final, all8bags).
Remove H347 reg_alpha=.5; do not add reg_alpha0. Restoring reg_lambda1 reproduces
both complete BASE ASTs exactly. Native archived BASE lambda_l2 must1 or stop
INCOMPLETE_SPEC_MISMATCH before fits. All other constructor fields match BASE:
LambdaRank/ndcg/sigmoid1, labels0..7/default label_gain, minleaf20/leaves7/depth3,
minchildweight.001, row subsample.8/freq1/colsample.8, implicit L1=0, splitgain0,
original LR/tree options/features/seeds, 18-config grid, tie/switch1e-4, sentinel-2.
No tuning, early stopping, calibration, BASE refit or new BASE model predictions.
Veto binary GBDT keeps L1=1/L2=2; all108 final veto SHA256s must equal BASE.
Every actual native fit is audited against every constructor field. Expected
per task819ranker+195veto actualfits,351ranker+39veto saved models; these remain
expectations until recorded execution. No model files in Git.

## Regression and input closure

Unchanged H303 NESTED/H308 code819a45aa5e30e662ab511cd9962b85b2b8c1e44a;
P0/P1/P2 x roots30001..30003 x4 family-grouped outer folds =36tasks/9cells,
1219TRAIN contexts/205families. Fresh exact-version GET frozen BASE read-only:
H316 key experiments/rd_breakthrough_20261030/H316/solhwv-20261104T105633Z/h316_models_locks_outputs.tar.gz,
VersionId W5jY6iZlwoRszA7P9zfNLLMtRtM9ZZkW,
SHA256 4eae1dec792e1c7b4adff9ff9a5226d0c38cfe1bd9605ba07cc2596f4ad7b0b5;
H315 key experiments/rd_breakthrough_20261030/H315/picks.npz,
VersionId jCll6z9FGwmpNg.cocb305LcVzFYuieR,
SHA256 e9e6b16a7381ff14191d2a4d31ce31d4b623c39297e7383ee6aeb871e08ca615.
Verify whole objects and selected member SHAs,9/9panels,36fold-root pick+lock
hashes and medians .0643163334285782/.04821480030677507/.04360142642936269
exact; mismatch REGRESSION_FAIL stops. Hash current1219TRAIN pools, selected,
r882 costs and4mixed-cache TRAIN feature prefixes against archived identities.
No DL-Val feature rows; metadata IDs/offsets may be read without tail features.
Preparation exits before treatment. Fit parent has metadata only; workers admit
only outer-TRAIN context pools and exact cache byte ranges. Historical BASE
outcomes/native copies forbidden during fitting. Physical outer feature reads
before fsynced locks/parent fsync=0. All36locks/guards/native audits/108veto SHAs
must pass before fresh outer scoring. Once-only STARTED/COMPLETE score markers;
no partial scoring, refits/restarts or repeated scores.

## Metrics and decision

Rawutility vs r882slot0, paired frozen BASE. Exact Elo label:
`reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN`.
Report all9cells/18arm records: family-equal mean gain and literal family-median
gain with whole-family bootstrapCI, drop-top1/5/10, LOFOmin, W/T/L, harm gain<-.01,
invalids/switches, per-root paired medians and CIs, Elo/EloCI, config agreement and
seed Jaccard. Ordinary intervals retain RNG302150/10000draws unchanged.
Paired whole-family bootstrap10000draws/RNG350191 uses same resample both arms/all
9cells; each arm drops its own top10 and median9cells is statistic. Per-root
intervals use the same resamples and median3partitions for that root; CI95
quantiles.025/.975, one-sidedLCB95 quantile.05.
ADVANCE iff paired drop10median>=+.002 andLCB95>0, L2R5median fullgain>=.9BASE,
medianharm<=BASE, positive per-root paired drop10median on>=2/3roots.
Otherwise L2R5_NOT_ADVANCED only after COMPLETE scoring. Ties/subthreshold never
advance; incomplete scientific results unknown/null, bootstrapdraws0. No AUC or
speed verdict. A possible one-shot dev91 vs r3175 is outside this authorization.

## One probe, forecasts, fixed budget

xbabe2 CPU only, nice19, numericthreads1, systemd MemoryMax6G/SwapMax0.
ONE fresh5sec per-core+SMT>=85%idle interval, candidates15..26 +SMT79..90.
Take first at most6passing physical cores, minimum4; SAME-interval aggregate
hostCPU<=70% required. No H335cores38..53/102..117, borrowedcores, expansion,
reprobe, fit restart, reset or cutcells. Stop own cgroup only if hostCPU>85%,
load>85, MemAvailable<20GiB, scratch>2GBdecimal or compute deadline.
Initial full36 forecast uses H347 actual29770.57CPU seconds/(workers*.9)*1.15+300.
After first chronological completed fold/root task, reforecast remaining35 x
max(29770.57/36, measured firsttask processCPU)/(workers*.9)*1.15+300 from now,
ongoing work credit0. Corecount/hostCPU failure=>INCOMPLETE_CPU_ADMISSION;
qualified cores+CPU with forecast miss=>INCOMPLETE_RUNTIME_FORECAST (specific
forecast clause governs), never a scientific negative. No timing/calibrationfits.
Compute stop2026-11-05T08:15:00Z; close2026-11-05T08:30:00Z, immutable.
Scratch /home/ubuntu/jope-data/h350-solhwv <=2GB. Create-only custody prefix
experiments/rd_breakthrough_20261030/H350/solhwv-20261105T052900Z/;
exactVersion GET+wholeSHA+everymember verification before own scratch deletion.

## Boundaries and publication

CI_FREEZE+BATCH_WINDOW: private index/objects; nonforce fresh-fetch v1/next only.
No PR/main/fastlane/hostedCI/newlane. Preserve canonical HEAD/index/foreign files.
Own source directory, oneH350row eachwave/overnight index, status/claim/comms and
/home/ubuntu/jope_plan/evidence/H350/solhwv-20261105-t0529/CLOSEOUT.json.
No models inGit. NoGPU/qualification.lock/protecteddata/DL-Val/dev91/DL-Test,
sealed/holdout/meta_test/fresh-cohort, forbiddenhosts, livewriter or foreignjobs.
H335/H348/H349 and all previous waves remain untouched; DOPEc51a56d0 frozen and
unloaded, scope CPU selection layer only. Same-agent exact-head self-review.
