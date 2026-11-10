# H344 — ranker LambdaRank sigmoid=0.5

Owner sol_v1_jrb01; authority hw_t2355_jrb01_h344_sigmoid.md, Nov4 23:55Z.
New fixed study/allocation; publish this prereg nonforce to origin/v1/next BEFORE
any treatment fit. H335/H338/H342/H343 and every other lane are outside ownership.

## Hypothesis and single fixed change

BASE is unchanged H303 NESTED on H308 P0/P1/P2, roots30001/30002/30003 and four
family-grouped outer folds (36 fold/root tasks,9 partition/root cells), TRAIN only.
ONE arm SIGMA adds exactly sigmoid=0.5 to pinned H287 topk.fit_ranker and
bagged.fit_ranker_n, including inner,final and all8 bag rankers. This flattens the
logistic pairwise LambdaRank loss; robustness is a hypothesis, not a result.
The fixed setting is owner-selected, not calibrated or tuned. Native BASE
sigmoid=1.0 was read from ALL972 exact-version archived refit rankers (108main,
864bag), LightGBM4.6.0; model-inventory digest
a6cff0c0bdc956c14b24dbda6736fbae7189090122932518156c369296f2cccb.
If BASE native sigmoid differs from1.0, close INCOMPLETE_SPEC_MISMATCH before fit.
No calibration/model predictions/BASE refits were needed to establish defaults.

Deleting only the sigmoid keyword reproduces each entire pinned BASE function
AST exactly. All other code/options/streams stay fixed: LambdaRank/ndcg, labels
0..7, implicit default label_gain, truncation level, min_child_samples20,
min_child_weight/native min_sum_hessian_in_leaf.001, num_leaves7,max_depth3,
subsample.8/freq1,colsample.8,reg_lambda1, original LR/tree grid and original
selected bag tree count, GBDT, seeds,18-config grid,1e-4 tie/switch rule and
-2.0 invalid sentinel. No path_smooth/clip/xendcg/L2/EMA/other intervention.
Veto remains unchanged binary GBDT; all108 saved refit veto SHAs must equal BASE.
Each actual constructor/native fit is audited: ranker keyword delta only
sigmoid[1.0,.5], veto delta empty; every constructor field checked against native
parameters. No speed verdict.

## Frozen BASE regression and physical TRAIN closure

BASE code819a45aa5e30e662ab511cd9962b85b2b8c1e44a; pinned H287 export
213b30ad52e1ce48927bba960ed609bd98ab4dfe. H342 harness copied from committed
v1/next d0bd2d60d8e02bcd0df6db5eec62ac870212f72e, only wave/paths/cores/window/RNG
and ranker intervention adapted. Frozen vendor bytes verified separately.

Exact-version GETs: H316 key
experiments/rd_breakthrough_20261030/H316/solhwv-20261104T105633Z/h316_models_locks_outputs.tar.gz,
VersionId W5jY6iZlwoRszA7P9zfNLLMtRtM9ZZkW,
SHA256 4eae1dec792e1c7b4adff9ff9a5226d0c38cfe1bd9605ba07cc2596f4ad7b0b5;
H315 experiments/rd_breakthrough_20261030/H315/picks.npz,
VersionId jCll6z9FGwmpNg.cocb305LcVzFYuieR,
SHA256 e9e6b16a7381ff14191d2a4d31ce31d4b623c39297e7383ee6aeb871e08ca615.
Whole archive and every selected member verified. BEFORE treatment, regression
PASS9/9 panels,36 fold/root pick hashes+lock hashes and3 partition medians EXACT:
.0643163334285782/.04821480030677507/.04360142642936269. No BASE refits or new
BASE model scores.1219 TRAIN pool hashes/selected/r882 costs and all4 consumed
cache TRAIN prefixes match archived identity9b148003d48ab609b2a796c57fb8b4c0296cc0ee79771f034944f4b8dc064263.
DL-Val feature rows read0. Mismatch means REGRESSION_FAIL and stop.

Preparation is a separate process, terminated before fit admission. Fit parent
loads metadata only; workers physically open only that outer fold's TRAIN
pools and declared TRAIN cache byte ranges. No inherited outer packet cache.
Outer rows/packets opened before each fsynced fit_lock and parent fsync must be0.
After ALL36 locks/model/native audits pass, score outer TRAIN exactly once per
partition/root, with STARTED/COMPLETE markers. Never partial or repeated scoring.
No DL-Val/dev91/DL-Test/sealed/holdout/meta_test/fresh-cohort payloads.
DOPE c51a56d0 is untouched/unloaded; CPU selection layer only.

## Metrics and locked decision

Report raw utility vs r882slot0 paired with archived BASE: family-equal mean gain,
literal family-median gain with whole-family95%CI,drop-top1/5/10,LOFO min,W/T/L,
harmed(gain<-.01),invalid,switches,per-root medians,Elo+EloCI,config agreement.
Exact label: reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.
No AUC or speed headline. Ordinary whole-family bootstrap10000/RNG302150 fixed.

ADVANCE iff paired SIGMA-minus-BASE drop-top10 median over9 cells>=+.002 AND
10000 whole-family bootstrap one-sided LCB95>0 (RNG344191), AND SIGMA median gain
>=90%BASE, AND SIGMA median harmed<=BASE, AND paired drop-top10 median across
partitions is strictly positive on>=2/3 roots. Same family sample for both arms/
all9 cells; each arm redrops its own top10; statistic median9 cells; LCBquantile.05.
Ties or margins<+.002 never advance. Else SIGMA_NOT_ADVANCED, only after complete
scores. If ADVANCE, next is separately authorized one-shot dev91 vs r3175, not H344.

## One capacity gate, two forecasts, immutable window

xbabe2 CPU only,nice19,numeric1,systemd MemoryMax6G/MemorySwapMax0. Assigned
physical cores in order54,56,57,58,59,61,63,16,17,18 and SMT118,120,121,122,123,
125,127,80,81,82. ONE fresh5-second per-core+SMT interval; BOTH siblings >=85%idle.
Take first at most8 passing pairs; require>=4 AND full36 forecast ending BEFORE
2026-11-05T03:45:00Z. H3420-14,H34321-33,H33813/15/60/62,H33538-53 and their SMT
siblings excluded. No second probe/borrowed cores/reset/cut cells/partial scoring.

Forecast H334 measured33343.799676CPU-s/(workers*.9)*1.15+300score-reserve seconds;
SIGMA cost multiplier1 is unmeasured, no speed claim. Reforecast after the FIRST
completed fold/root task: remaining35*max(H334CPU/36,first-task measured CPU)/
(workers*.9)*1.15+300 from current time, partial ongoing work credit0. If either
forecast misses compute stop, close INCOMPLETE_RUNTIME_FORECAST and stop only
own service cgroup. No scientific treatment verdict from partial fits.

Stop hostCPU>85% OR load>85 OR MemAvailable<20GiB OR own scratch>2GB(decimal).
Preexisting swap7/7G does not authorize swap. Scratch /home/ubuntu/jope-data/h344-jrb01
only. Compute stop03:45Z Nov5; close04:00Z Nov5 fixed, late means INCOMPLETE.
Create-only S3 experiments/rd_breakthrough_20261030/H344/; exact-version GET,
whole SHA and every member verified before deleting only own scratch. No models
in Git. Keep actual fit/prereg timestamps and all failed/partial resource costs.

## Publication and red-team

CI_FREEZE+MASTER_MERGE_HOLD: private index/objects, preserve canonical HEAD/index/
foreign files, ordinary nonforce rebase-on-fetch v1/next only; no PR/main/fastlane/
hostedCI/GPU/qualification.lock/live writer/foreign job signals/xbabe0/1/3.
Code/tests/PREREG/README/metrics<50KB and one each H344 wave/overnight index row.
Before close verify prereg publication precedes every treatment start, all36
locks precede once-only outer reads,0protected/dev rows,3root medians/CI/tie bar,
only sigmoid delta per native ranker and108 unchanged veto SHAs, exact regression.
