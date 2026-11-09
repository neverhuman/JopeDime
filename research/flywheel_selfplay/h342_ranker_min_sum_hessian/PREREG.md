# H342: ranker minimum summed Hessian

Owner sol_v1_gov03; authority hw_t2330_gov03_h342_minhess.md, 2026-11-04.
H340 is CLOSED+RELEASED; this is a new intervention/allocation/window.
Publish this preregistration and fixed value nonforce to origin/v1/next BEFORE
any treatment fit. H340/H334/H333/H329/H326 remain closed.

## Hypothesis and fixed value

A curvature-based leaf floor may reduce dependence on small, low-confidence
pair pockets. This is a hypothesis, not an observed robustness or speed result.
ONE arm MINHESS: set only min_child_weight=9.87 (native alias
min_sum_hessian_in_leaf) on the two pinned H287 ranker constructors topk.fit_ranker
and bagged.fit_ranker_n. BASE omits this keyword; every archived BASE ranker's
native effective default is 0.001. Removing the one override
restores both entire BASE function ASTs exactly. Do not pass both aliases.

Value rule fixed by owner: 25th percentile of every archived BASE ranker leaf's
leaf_weight (summed Hessian), then round to three significant figures.
Actual one-time read-only dump_model derivation: 972 BASE refit rankers
(108main+864bag) across all36outer fold/root tasks; 988244 leaves.
Each BASE archived model path counted once; normalized treatment paths and veto
models excluded. All BASE native model bytes match exact-version archive members.
Unrounded percentile=9.865141101181505; linear interpolation;
Python format(.3g) gives MINHESS=9.87, strictly above BASE. No tuning.
Leaf-weight f64le vector SHA256 e412e2091c5ac6fc88c397db49593c3e1a74617825f0f9d04c7b9dd0698f9ac1.
Archive SHA256 4eae1dec792e1c7b4adff9ff9a5226d0c38cfe1bd9605ba07cc2596f4ad7b0b5; full receipt SHA256
a558ee3a55c56747dcc5d6f6fa81986accb37e74b79678931d5123003970376d. [minhess_value.json](minhess_value.json).
All archived BASE rankers are available: calibration probe NOT_NEEDED,0calibration
fits,0model predictions,0treatment fits during derivation. No label/outcome-based
selection. The floor is derived globally from those archived TRAIN models,
not separately inside each outer fold. If value<=BASE, stop
INCOMPLETE_SPEC_MISMATCH before treatment. No replacement value or label transform.

## Exact fixed fit delta

Pinned H287 export213b30ad52e1ce48927bba960ed609bd98ab4dfe. Every inner, final and
all8bag rankers keep original LambdaRank, labels/default label_gain, group/query
construction, features, seeds, GBDT, num_leaves7,max_depth3,min_child_samples20,
subsample/colsample.8,subsample_freq1,reg_lambda1,n_jobs1,deterministic and
force_row_wise and all other arguments. Veto remains binary GBDT with default
min_child_weight.001. No H340 rank_xendcg, path_smooth, clip, LR or other change.
Original200round topk/LR(.03,.05,.10), prediction counts50/100/200 and original
selected count for bagged fits remain. Preserve original18-config grid,
tie1e-4/switch/margin rules and failure sentinel-2.0. No early stopping or tuning.
After every actual fit, audit all native-mappable constructor fields including
Hessian floor; require all other fields unchanged and saved models bound to
member seed/native parameter digests. Expected per outer task819ranker+195veto
fits and351ranker+39veto saved models; expectations become evidence only if run.

## Regression first and physical input closure

Unchanged H303 NESTED/H308 code819a45aa5e30e662ab511cd9962b85b2b8c1e44a,
P0/P1/P2 x roots30001-30003 x4 family-grouped outer folds:36tasks/9cells,
1219TRAIN contexts/205families. First science gate already PASS: exact-version
GET of frozen BASE archive read-only, whole SHA and selected-member SHA verified;
9/9panels,36fold/root pick+lock hashes and medians
.0643163334285782/.04821480030677507/.04360142642936269 exact. No BASE refits
or fresh BASE model predictions. Current1219TRAIN pool hashes/selected/r882costs
and4consumed mixed-cache TRAIN feature prefixes match archived inputs exactly.

Preparation ran in a separate process and ended before fit admission. Fit parent
initializes metadata only, with no inherited packet cache. Worker opens only
outer-TRAIN pools and exact allowed TRAIN cache byte ranges; audit rejects outer
feature packets and BASE outcomes during fitting. Physical outer rows/packets
before fsynced fit_lock and parent fsync must equal0. All36locks/guards/models/
native audits must pass before fresh once-only outer scoring. STARTED/COMPLETE
markers prevent repeats. No partial scoring or repeated completed scores.

## Metrics and locked decision

Raw utility paired with BASE vs r882slot0; exact label:
`reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN`.
Report family-equal mean and literal median/whole-family CI,drop1/5/10,LOFO min,
W/T/L,harmed gain<-.01,invalid,switches,per-root medians,Elo/CI,seed Jaccard and
config agreement. Ordinary whole-family intervals10000draws/RNG302150 unchanged.

ADVANCE iff paired(MINHESS-BASE) drop-top10 median over9cells>=+.002 AND
one-sided LCB95>0; MINHESS median gain>=90%BASE; median harmed<=BASE; and positive
per-root paired drop10 medians across3partitions on>=2/3roots. Otherwise
MINHESS_NOT_ADVANCED only after complete scores. Ties/subthreshold never advance.
Paired bootstrap10000whole-family draws/RNG342191, same family sample botharms/
all9cells, each arm redrops its own top10, statistic median9cells, LCBquantile.05.
No AUC/speed verdict; no H342 dev91. If ADVANCE, a separate owner assignment may
perform one-shot dev91 vs r3175.

## One capacity sample, two forecasts and fixed window

xbabe2 CPU only, nice19,numeric1,MemoryMax6G/MemorySwapMax0. Assigned physical
cores0,2,3,4,5,6,7,10,12,14 plus SMT64,66,67,68,69,70,71,74,76,78.
ONE fresh5-second per-core+SMT>=85%-idle interval, both siblings must qualify;
take the first at most8passing pairs in assigned order. Need>=4cores AND full36
forecast ending BEFORE fixed2026-11-05T03:15:00Z. No second capacity probe.
H334 actual full36 CPU reference33343.799676s; seconds=CPU/(workers*.9)*1.15+300.
A Hessian-floor cost multiplier1.0 is a planning assumption, not measured speed.

Reforecast exactly after first completed outer fold/root task, before any outer
score: remaining35CPU tasks times max(H334actualCPU/36,first completed task
measured processCPU), divided by workers*.9, times1.15 plus300reserve, added to
current time. Ongoing partial work receives0credit (conservative). If forecast
misses stop, record INCOMPLETE_RUNTIME_FORECAST and stop only own service cgroup.
No cut cells/partial scoring/borrowed cores/reprobe/restart/deadline reset.
Record actual treatment first-fit timestamps and bind to published prereg.

Compute stop Nov5 03:15Z; close Nov5 03:30Z immutable. Stop hostCPU>85%,load>85,
MemAvailable<20GiB,scratch>2GBdecimal. Scratch only/home/ubuntu/jope-data/h342-gov03.
Create-only S3 experiments/rd_breakthrough_20261030/H342/ inner/final archives,
exact-version GET+whole SHA+every-member proof before raw inner/own scratch
cleanup. Preserve immutable archive/source/evidence and actual unit exits.
H34321-33,H33813/15/60/62,H33538-53 and all siblings remain offlimits.

## Records and publication

Private index/Git objects, nonforce v1/next only, CI_FREEZE+MASTER_MERGE_HOLD.
No PR/main/fastlane/hosted CI. Final README/metrics<50KB/code/tests, one each
wave/overnight index row, claim/status/claim+close comms/CLOSEOUT receipts.
No models in Git. No GPU/qualification.lock, DL-Val/dev91/DL-Test rows,
protected/sealed/holdout/meta_test/fresh cohort, foreign jobs/hosts or live writer.
DOPEc51a56d0 unchanged/not loaded; CPU selection layer only.
