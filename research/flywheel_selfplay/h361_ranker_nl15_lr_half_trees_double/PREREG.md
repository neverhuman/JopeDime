# H361: NL15 ranker LR half / every iteration limit doubled

Owner sol_v1_gov03; authority hw_t1100_gov03_h361_nl15_lrhalf.md.
Fixed compute stop2026-11-05T14:45:00Z; close15:00Z, never reset.
Copy committed H352 at02dfee2f7e0b8543324494dd4f99d074cd6b93b6,
research/flywheel_selfplay/h352_ranker_num_leaves_15/. NL15 is the primary
frozen TRAIN incumbent; BASE is secondary. No live writer or serving changes.
Publish this preregistration nonforce v1/next BEFORE any treatment fit.

## One coupled intervention

Hypothesis: shrinkage gives a smoother, lower-variance LambdaRank trajectory.
ONE arm LRH halves every ranker learning_rate and doubles every ranker fit,
prediction and locked iteration limit, with NL15 num_leaves15/max_depth3.
The inherited logical rate grid is(.03,.05,.10), so effective rates become
(.015,.025,.05), including nominal .05->.025. Main rankers200->400trees;
selected/grid iteration counts(50,100,200)->(100,200,400), including all8bags.
Preserve the original logical grid labels in seed generation and selection
tie ordering; only effective ranker LR/iteration values change. Stored member
selected_lr/selected_n_estimators record actual half/double values. Inner
prediction and 64-context training reload checks use doubled limits exactly
once; cold scoring consumes those actual locked limits without another scale.

Constructor transformations preserve the exact H352 NL15 AST on inverse.
Two additional narrow AST adapters double the two ranker prediction limits in
veto.run_cell_veto and two ranker reload limits in r302.fit_and_freeze_member;
the latter writes actual selected LR/iteration values. All four complete
function inverses must exactly reproduce H352 ASTs. Veto classifier training
and prediction limits stay unchanged. Test toy constructor grids without any
real LightGBM fit. Audit every actual native fit: inverse LR*2/iterations/2
must restore the logical NL15 configuration, with all other arguments fixed.
Native LR in(.015,.025,.05), requested rounds in(100,200,400); record actual
realized trees/leaves, never infer them from configuration.

Keep LambdaRank/ndcg/sigmoid1/labels0..7/default gains, NL15/depth3,
min_child_samples20, reg_alpha0/reg_lambda1, Hessian.001, min_split_gain0,
subsample.8/freq1/colsample.8, original seeds/features/family bootstrap,
18tau-k-a configs,1e-4ties/switches and-2.0sentinel unchanged. Veto binary
GBDT retains original settings and108/108final model SHAs must equal frozen
BASE and NL15. No BASE/NL15 refits, calibration or new baseline predictions.
Expected36tasks:29484ranker+7020veto nativefits,12636savedranker+1404veto
bindings. These are expectations until execution.

## Regression and input closure first

Unchanged H303/H308 at819a45aa5e30e662ab511cd9962b85b2b8c1e44a and H287 at
213b30ad52e1ce48927bba960ed609bd98ab4dfe. P0/P1/P2 x3roots30001..30003 x4
family-grouped outer folds=36fit tasks/9cells;1219TRAIN contexts/205families.
Fresh exact-version BASE H316/H315 GETs: whole/member SHA,9panels,36fold-root
pick/lock hashes, exact medians .0643163334285782/.04821480030677507/
.04360142642936269; mismatch=>REGRESSION_FAIL. Same VersionIds/keys as H352.
H316 VersionW5jY6iZlwoRszA7P9zfNLLMtRtM9ZZkW, wholeSHA
4eae1dec792e1c7b4adff9ff9a5226d0c38cfe1bd9605ba07cc2596f4ad7b0b5.
H315 VersionjCll6z9FGwmpNg.cocb305LcVzFYuieR, wholeSHA
e9e6b16a7381ff14191d2a4d31ce31d4b623c39297e7383ee6aeb871e08ca615.
Fresh NL15 exact-version manifest/27parts/all2691members: manifestVersion
RtNLAfmRJbXqdXQx7nxZLZ8dWHPcVYQC, SHAd3b47c5fbc1ecc3c6e6d4f89dfef7d6a3f5216f45c6dd1589ce94fb413350f60.
Require NL15 all9panel pick SHAs,36lock+COMPLETEscore marker hashes/rawfamily
arrays and native972ranker NL15/depth3 plus108vetoSHA identities exact.
Current TRAIN pools/selected/r882costs and four consumed TRAIN cache prefixes
must match archived identities before fitting and after once-only scoring.

Preparation exits separately. Fit parent has metadata only; each worker
reads only its outer-TRAIN contexts and permitted cache byte ranges. BASE
and NL15 archived outcomes/model copies are forbidden during fitting.
Physical outer rows/packets before fsynced fit_lock+parent fsync must0.
ALL36locks/guards/native audits/108vetoSHA pass before a fresh cold outer
process. STARTED/COMPLETE markers prevent replay. No partial scoring.

## Metrics, primary decision and multiplicity

Raw utility versus r882slot0. Exact Elo label:
`reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN`.
Report BASE/NL15/LRH family-equal mean and literal family median with CIs,
drop-top1/5/10,LOFOmin,W/T/L,harmed gain<-.01,invalids/switches,seedJaccard,
configuration agreement,Elo/EloCI. All new intervals use10000whole-family
resamples/RNG361191. PRIMARY paired drop10=LRH-NL15; secondary=LRH-BASE.
Same draws shared across both comparison arms/all9cells; each arm drops its
own top10 within each resample; statistic median9cells. Per-root medians/CIs
use median3partitions at fixed root and the same draws; CI95quantiles.025/.975,
one-sidedLCB95quantile.05. Report root uncertainty without extra gates.

ADVANCE iff primary pairedmedian>=+.002 AND LCB95>0, LRH median fullgain
>=90%NL15, LRH median harmed<=NL15, and primary pairedmedian>0 on>=2/3roots.
Else LRH_NOT_ADVANCED only after COMPLETE scoring. Ties/subthreshold never
advance; incomplete outcomes remain UNKNOWN/null. Secondary BASE performance
does not override primary gates. Sequential exploratory sweep H348-H361:
no multiplicity-adjusted confirmatory claim. Any ADVANCE needs a separate
owner-approved one-shot dev91 confirmation, NOT in H361. No AUC/speed verdict.

## One admission probe and fixed resource policy

xbabe2 CPUonly,nice19,numericthreads1,systemd MemoryMax6G/SwapMax0,
CPUQuota<=800%. ONE fresh5sec percore+SMT>=85%-idle sample from15..26+sibs
79..90; take first<=8passing pairs/min4, same-interval hostCPU<=70%.
No other pool/borrowed cores/reprobe/expansion. Bind all controller threads
and workers to admitted cores before fitting.

H352 actualCPU26189.516914seconds is bound to closed receiptSHA
042905333cc018c3baef067f670a86261ef6f0919d732f676c92465c62256eeb and invocation
b43ca85cb1434a7ca8709b588a573ef4. No measured per-role processCPU split exists;
use conservative ranker-share upperbound1.0, doubling TOTAL H352 CPU to
52379.033828seconds. Initial full36 forecast Cref/(workers*.9)*1.15+300s.
After first chronological completed task, remaining35*max(Cref/36,firstactual
processCPU)/(workers*.9)*1.15+300s from then, ongoing workcredit0.
A miss of fixed14:45=>INCOMPLETE_RUNTIME_FORECAST; no cutcells/reset/reprobe.
Insufficient cores/host gate=>INCOMPLETE_CPU_ADMISSION. No timing fits.

CORRECTED OWNER GUARD: stop only own unit for MemAvailable<20GiB, OR hostCPU
>85% averaged over60s, OR load1>108 on3consecutive60s samples, OR ownscratch
>2GB(decimal), plus fixed14:45compute stop. Memory/disk checked every5s;
CPU/load decisions only at completed>=60s windows. Five-second CPU/load are
observations, not the retired instantCPU85/absolute-load85 rejection rule.
Own scratch /home/ubuntu/jope-data/h361-gov03. Before reclaiming any own copy,
create-only S3 H361 exactVersionGET/wholeSHA/EVERYmember proof. Stream baseline
parts and retain necessary locks/picks/metadata, not duplicate weight payloads.
For treatment, archive each inner bundle then reclaim both raw and compressed
local copies after that proof; keep version/member receipts. Final custody
includes refits/inputs/metadata and an exact-version verified component list
for every inner bundle. Do not fabricate one monolithic weight archive.

## Boundaries and records

Private index/objects; preserve canonical HEAD9483d585/indexa78b7cab and all
foreign staging/dirty files; no new worktree. CI_FREEZE+BATCH_WINDOW: fresh
parent/nonforce origin/v1/next only; noPR/main/fastlane/hostedCI/models inGit.
Own H361 folder, README/metrics<50KB, one H361 row each wave/overnight index,
claim/status and /home/ubuntu/jope_plan/evidence/H361/gov03-20261105-t1100/CLOSEOUT.json.
One claim and one close comms line. H355 and earlier waves remain closed.
H354/H357/H359/H360 and foreign lanes/files/units untouched. No xbabe0/1/3,
GPU/qualification.lock,live writer/AUTORESTART markers,protected/sealed/
holdout/meta_test/fresh-cohort/DL-Val/dev91/DL-Test rows. DOPEc51a56d0
unchanged/unloaded, CPU selection layer only.
