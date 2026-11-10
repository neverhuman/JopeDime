# H355: ranker min_split_gain 0.01

Owner sol_v1_gov03; authority hw_t0915_gov03_h355_minsplitgain.md.
Fixed compute stop 2026-11-05 11:45Z; close 12:00Z. Neither resets.
Copy committed H351 at bf3414cba7918bea721a8b64c04211bc0d687243 from
research/flywheel_selfplay/h351_ranker_min_child_samples_40/, retaining its
seven H303/H308 vendor pins and exact archived BASE identities.
Publish this preregistration non-force to v1/next BEFORE treatment fitting.
H351/H348 and earlier waves remain closed. H354 belongs to another owner.

## Fixed hypothesis and intervention

A split-gain floor may prune low-value splits that follow family noise.
ONE arm MSG01 changes effective ranker min_split_gain from 0 to 0.01 in
pinned H287 topk.fit_ranker and bagged.fit_ranker_n: inner fits, final refits
and all eight bag rankers. The original constructors omit this argument;
its LightGBM default is zero. Add only min_split_gain=0.01. Removing that
keyword restores implicit zero and both complete BASE function ASTs exactly.
An explicit-zero constructor must equal the implicit-zero constructor.
No tuning, calibration, BASE refits or new BASE model predictions.

Restore H351 min_child_samples to BASE20. Retain reg_lambda1/implicit
reg_alpha0, LambdaRank/ndcg, sigmoid1, labels0..7/default label_gain,
num_leaves7/max_depth3, subsample.8/freq1/colsample.8, Hessian floor.001,
original LR/tree options/seeds/features/groups,18-config grid,1e-4 ties and
switches, and sentinel-2.0. Veto binary GBDT stays min_child_samples25,
reg_alpha1/reg_lambda2. All108final veto SHAs must equal BASE. Audit every
native fit against every constructor parameter, with exact integers/seeds.
Expected full36:29484ranker+7020veto fits;12636saved ranker+1404veto bindings.
These counts remain expectations until actual execution.

## BASE first and physical TRAIN closure

Unchanged H303 NESTED/H308819a45aa5e30e662ab511cd9962b85b2b8c1e44a:
P0/P1/P2 x roots30001..30003 x4family-grouped outer folds=36tasks/9cells,
1219TRAIN contexts/205families. First require fresh exact-version BASE GET,
whole-object/memberSHA,9panels/36fold-root pick+lock hashes and exact medians
.0643163334285782/.04821480030677507/.04360142642936269.
Mismatch stops REGRESSION_FAIL. All972BASE rankers must have native
min_gain_to_split0, min_data_in_leaf20, L1zero/L2one; otherwise close
INCOMPLETE_SPEC_MISMATCH before treatment. Verify all108BASE veto identities.
Current TRAIN pool/selected/r882-cost hashes and four consumed TRAIN cache
prefixes must match archived identities before fitting and after scoring.

H316 key experiments/rd_breakthrough_20261030/H316/solhwv-20261104T105633Z/h316_models_locks_outputs.tar.gz;
VersionId W5jY6iZlwoRszA7P9zfNLLMtRtM9ZZkW;
SHA256 4eae1dec792e1c7b4adff9ff9a5226d0c38cfe1bd9605ba07cc2596f4ad7b0b5.
H315 key experiments/rd_breakthrough_20261030/H315/picks.npz;
VersionId jCll6z9FGwmpNg.cocb305LcVzFYuieR;
SHA256 e9e6b16a7381ff14191d2a4d31ce31d4b623c39297e7383ee6aeb871e08ca615.

Preparation exits separately. Fit parent holds metadata only. Workers may
open only outer-TRAIN contexts and allowed feature byte ranges; BASE outcome
and native-model copies are forbidden during fitting. Physical outer feature
reads/packets before fsynced fit lock plus parent fsync must0. All36locks,
guards/native audits and108veto SHA matches must pass before a cold fresh
outer process. Durable STARTED/COMPLETE markers prevent replay. No partial
scoring, repeated fits, cut cells, BASE predictions or design changes.

## Metrics and locked decision

Raw utility versus r882slot0, paired with frozen BASE. Exact Elo label:
`reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN`.
Report family-equal mean and literal family median with whole-family CIs,
drop-top1/5/10, LOFOmin, W/T/L, harmed gain<-.01, invalids, switches,
per-root medians AND paired CIs, Elo/EloCI, seedJaccard/config agreement.
Ordinary bootstrap remains10000draws/RNG302150. Paired bootstrap uses
10000whole-family draws/RNG355191, shared across both arms and all9cells;
each arm redrops its own top10; statistic=median9cells. Per-root CIs use
these same draws and median3partitions at fixed root. CI95quantiles.025/.975;
one-sidedLCB95quantile.05. Root CIs report uncertainty without extra gates.

ADVANCE iff paired drop10median>=+.002 AND LCB95>0, MSG01 median fullgain
>=90%BASE, median harmed share<=BASE, and positive paired root medians on
>=2/3roots. Otherwise MSG01_NOT_ADVANCED only after COMPLETE scoring.
Incomplete outcomes stay UNKNOWN/null. No AUC/speed verdict. Any dev91/r3175
comparison needs a separate owner assignment, outside H355; no serving.

## One probe, two forecasts and custody

xbabe2 CPUonly,nice19,numeric1,systemd MemoryMax6G/SwapMax0. ONE fresh5-second
percore+SMT sample from15..26+sibs79..90; both siblings>=85%idle. Take first
at most8passing cores/min4, with same-interval hostCPU<=70%. Bind controller
threads and workers to admitted cores before fits. No reprobe/expansion.
Exclude H33538..53/102..117 and H352pool0..14/27..37 and siblings.
Initial full36 forecast uses H347actual29770.57CPU/(workers*.9)*1.15+300s.
Cost multiplier1 is an unmeasured planning assumption. First chronological
completed task triggers remaining35*max(29770.57/36,firstactualprocessCPU)/
(workers*.9)*1.15+300s from then; ongoing work credit0. Forecast miss of fixed
11:45Z =>INCOMPLETE_RUNTIME_FORECAST. Insufficient cores/host admission
=>INCOMPLETE_CPU_ADMISSION. No borrowed cores, reset, refit or partial scoring.
Stop only own cgroup for hostCPU>85/load>85/MemAvailable<20GiB/scratch>2GB
or compute stop. Own scratch /home/ubuntu/jope-data/h355-gov03. Create-only S3
experiments/rd_breakthrough_20261030/H355/: exactVersionGET,wholeSHA and
EVERYmember proof precede any own raw-model or scratch deletion.

## Records and boundaries

Private index/objects; no newlane; fresh-fetch nonforce v1/next only under
CI_FREEZE+BATCH_WINDOW. No PR/main/fastlane/hostedCI/model payloads in Git.
Preserve canonical HEAD/foreign index/staging/dirty files. Own H355folder,
README/metrics<50KB,one H355row each wave/overnight index,claim/status and
/home/ubuntu/jope_plan/evidence/H355/gov03-20261105-t0915/CLOSEOUT.json.
Exactly one claim and one close comms line. No H354 file writes occurred.
No xbabe0/xbabe1(any)/xbabe3,GPU/qualification.lock,protected/sealed/holdout/
meta_test/fresh-cohort/DL-Val/dev91/DL-Test rows,live writer or foreign jobs.
H335/H354/H352/H353 lanes/files/units stay untouched. DOPEc51a56d0 stays
unchanged and unloaded, CPU selection layer only.
