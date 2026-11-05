# H351: ranker min_child_samples 40

Owner sol_v1_gov03; authority hw_t0730_gov03_h351_minchild40.md, November5.
Copy the committed H350 harness from origin/v1/next snapshot
15b4b86fe7748f1dc2875302d1719850aea677c2, path
research/flywheel_selfplay/h350_ranker_reg_lambda_5/, with full file SHA inventory.
Seven H303/H308 vendor pins and H287213b30ad52e1ce48927bba960ed609bd98ab4dfe
remain unchanged. H348 and all prior closed waves remain closed.
Publish this preregistration nonforce to v1/next BEFORE any treatment fit.

## Hypothesis and single fixed value

A larger leaf-count floor may pool more contexts and damp single-family swings
and -2.0 recoveries. ONE arm MCS40 changes the EXISTING ranker constructor
min_child_samples20 to40 in topk.fit_ranker and bagged.fit_ranker_n, covering
all inner fits, final refits and eight bag rankers. Restore H350 reg_lambda to
BASE1 and leave ranker reg_alpha implicit BASE0. Restoring min_child_samples20
reproduces both complete BASE ASTs exactly, without adding constructor keywords.
The archived BASE native min_data_in_leaf must20; otherwise stop
INCOMPLETE_SPEC_MISMATCH before treatment. No tuning, alternate value,
calibration fit, BASE refit or new BASE model predictions.

Keep LambdaRank/ndcg, sigmoid1, labels0..7, implicit BASE label_gain, leaves7,
depth3, subsample.8/freq1, colsample.8, reg_alpha0, reg_lambda1, min_split_gain0,
Hessian min_sum_hessian_in_leaf.001, original LR/tree options/seeds/features,
18-config grid, ties/switches1e-4 and sentinel-2.0 unchanged. Veto binary GBDT
keeps its original min_child_samples25, reg_alpha1 and reg_lambda2. All108final
veto SHAs must equal archived BASE. Audit every actual native fit against every
constructor field; native integer settings and seeds match exactly. Expected
per task819ranker+195veto fits and351ranker+39veto saved models; these remain
expectations until actual execution. No model payloads in Git.

## BASE first and physical input closure

Unchanged H303 NESTED/H308819a45aa5e30e662ab511cd9962b85b2b8c1e44a:
P0/P1/P2 x roots30001..30003 x4 family-grouped outer folds =36fit tasks/9cells,
1219TRAIN contexts/205families. First science gate: fresh exact-version frozen
BASE GET read-only, whole-object and selected-member SHA,9/9panels,36fold-root
pick+lock hashes and exact medians .0643163334285782/.04821480030677507/
.04360142642936269. Mismatch=>REGRESSION_FAIL. Current1219TRAIN pool hashes,
selected/r882costs and four consumed mixed-cache TRAIN feature prefixes must
match archived identities. No DL-Val feature rows.

H316 key experiments/rd_breakthrough_20261030/H316/solhwv-20261104T105633Z/h316_models_locks_outputs.tar.gz,
VersionId W5jY6iZlwoRszA7P9zfNLLMtRtM9ZZkW,
SHA256 4eae1dec792e1c7b4adff9ff9a5226d0c38cfe1bd9605ba07cc2596f4ad7b0b5.
H315 key experiments/rd_breakthrough_20261030/H315/picks.npz,
VersionId jCll6z9FGwmpNg.cocb305LcVzFYuieR,
SHA256 e9e6b16a7381ff14191d2a4d31ce31d4b623c39297e7383ee6aeb871e08ca615.

Preparation exits in a separate process before admission. Fit parent has
metadata only; each worker may open only outer-TRAIN context pools and exact
allowed cache byte ranges. Historical BASE outcomes/native copies forbidden
during fit. Require physical outer feature rows/packets before fsynced
fit_lock plus parent-directory fsync0. All36locks/guards/native audits and
108BASE-identical final veto SHAs must pass before fresh cold outer scoring.
Once-only STARTED/COMPLETE markers prevent repeats. No partial scoring,
restarted fits, cut cells or completed-score replay.

## Metrics and locked decision

Raw utility vs r882slot0, paired with frozen BASE. Exact Elo label:
`reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN`.
Report family-equal mean gain and literal family-median gain with whole-family
bootstrapCI, drop-top1/5/10, LOFOmin, W/T/L, harmed gain<-.01, invalids, switches,
per-root medians AND per-root paired CIs, Elo/EloCI, seed Jaccard and config
agreement. Ordinary intervals retain10000draws/RNG302150 unchanged.
Paired10000whole-family draws/RNG351191 use the same resample both arms/all9cells;
each arm drops its own top10, median9cells is statistic. Per-root CIs use the
same draws and median3partitions at fixed root. CI95quantiles.025/.975;
one-sidedLCB95quantile.05. Root intervals report uncertainty without extra gates.

ADVANCE iff paired drop10median>=+.002 AND LCB95>0, MCS40median fullgain>=.9BASE,
medianharm<=BASE, and positive per-root paired drop10medians on>=2/3roots.
Otherwise MCS40_NOT_ADVANCED only after COMPLETE scoring. Ties/subthreshold
never advance. Incomplete scientific outcomes stay UNKNOWN/null; no partial
performance aggregate. No AUC or speed verdict. A possible one-shot dev91 vs
r3175 requires a separate assignment, outside H351. No serving promotion.

## One probe, two forecasts, fixed window

xbabe2 CPU only,nice19,numeric1,systemd MemoryMax6G/MemorySwapMax0. Candidates
15..26 plus SMT79..90. ONE fresh5-second/core+SMT>=85%-idle interval; both
siblings qualify. Take the first at most8passing physical cores, minimum4, AND
require same-interval hostCPU<=70%. Aggregate/proc/stat ticks bracket that same
single mpstat interval, not a second core probe. Controller threads and every
worker bind to admitted cores before fits. No borrowed cores, expansion,
re-probe, fit restart, cut cells or deadline reset. Exclude H33538..53/102..117.

Full36 admission forecast uses H347 actual29770.57CPU seconds /
(workers*.9)*1.15+300seconds reserve. MCS40cost multiplier1 is an unmeasured
planning assumption. Reforecast immediately after first chronological completed
fold/root task: remaining35 x max(29770.57/36,firsttask actual processCPU) /
(workers*.9)*1.15+300 from current time; ongoing work credit0.
Corecount/host admission failure=>INCOMPLETE_CPU_ADMISSION. A full36 forecast
miss=>INCOMPLETE_RUNTIME_FORECAST under the specific forecast requirement.
No extra timing fits or repeated probes.

Fixed compute stop2026-11-05T10:15:00Z; fixed close2026-11-05T10:30:00Z.
Stop only own cgroup if hostCPU>85%, load>85, MemAvailable<20GiB,
scratch>2GBdecimal or compute deadline. Scratch /home/ubuntu/jope-data/h351-gov03.
Create-only S3 experiments/rd_breakthrough_20261030/H351/ custody: exact-version
GET +whole SHA+every-member verification before raw-inner/own-scratch cleanup.
Preserve immutable source/evidence and actual command exits.

## Records and boundaries

Private index/objects, CI_FREEZE+BATCH_WINDOW, nonforce fresh-fetch v1/next only.
No PR/main/fastlane/hostedCI/newlane. Preserve canonical HEAD/current foreign
index/staged files and all other dirty files; read-only Git uses OPTIONAL_LOCKS0.
Own H351source/README/metrics<50KB, one row each wave/overnight index,
claim/status, exactlyone claim+one close comms line, and
/home/ubuntu/jope_plan/evidence/H351/gov03-20261105-t0730/CLOSEOUT.json.
H350R belongs to sol_v1_hwv on xbabe1; all its lanes/files/units/host are offlimits.
No xbabe1 access of any kind, xbabe0/xbabe3, GPU/qualification.lock,
DL-Val/dev91/DL-Test/protected/sealed/holdout/meta_test/fresh-cohort rows,
live writer or foreign-job operations. H335/H350R are left alone.
DOPEc51a56d0 remains unchanged/unloaded; scope CPU selection layer only.
