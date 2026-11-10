# H329: fixed DART tree dropout in ENS_VOTE2 rankers

Owner sol_v1_gov03. Authority `~/nudges/hw_t1735_gov03_h329_dart.md` (Nov4 17:35Z).
H326 is closed and stays closed. This preregistration is published non-force to
`origin/v1/next` before any optimizer fit, including the synthetic timing fixture.

## Hypothesis and unchanged design

Dropping whole boosted trees during ranker fitting may reduce concentrated family
gains. One fixed DART arm versus frozen, unchanged H303 NESTED BASE: H308 partitions
P0/P1/P2 × roots30001–30003 × four family-grouped outer folds, all36 tasks required.
TRAIN only,1,219 contexts/205 families. No dev91/DL-Val feature rows, DL-Test,
sealed/holdout/meta-test/fresh cohorts, GPU or live-writer operation. DOPE c51a56d0
loaded=false, unchanged; scope="CPU selection layer only". Cached TRAIN features
remain allowed. No label-gain smoothing, target/loss changes, extra-trees or bagging
intervention. All original family splits, feature values, raw utility/sentinel-2.0,
labels, LambdaRank, member/bag seeds, three members/root, eight bags/member,
TAU(.05,.10,.15)×k(5,6,7)×agreement(2,3), raw inner mean objective, tie1e-4 ordering,
veto, top4 margin and ENS_VOTE2 switch rules stay fixed.

## Exact constructor delta and round-count call sites

Pinned H287 export213b30ad52e1ce48927bba960ed609bd98ab4dfe:
`topk.py:fit_ranker` and `bagged.py:fit_ranker_n` only. AST transformation changes
the existing `boosting_type="gbdt"` to `"dart"` and adds exactly:

| parameter | BASE | DART |
|---|---|---|
| boosting_type | gbdt | dart |
| drop_rate | unused/default | 0.1 |
| skip_drop | unused/default | 0.5 |
| max_drop | unused/default | 50 |
| uniform_drop | unused/default | False |
| xgboost_dart_mode | unused/default | False |
| drop_seed | unused/default | int(random_state) at that original ranker fit call |

`random_state` itself is unchanged, including each original inner and bag stream.
The owning member seed is recorded separately in constructor audits. The delta
applies to every inner ranker, final ranker and all8 bag rankers. Veto classifiers
keep their original GBDT constructor/function AST and kwargs. Strip the six new
keywords and restore the old boosting value: both complete BASE function ASTs must
be identical. Constructor and native-model audits verify the effective settings.

Every other explicit BASE constructor keyword is retained: objective=lambdarank,
metric=ndcg,truncation=k,max_depth3,num_leaves7,min_child_samples20,learning_rate=lr,
subsample.8,subsample_freq1,colsample_bytree.8,reg_lambda1,n_jobs1,verbosity-1,
deterministic=True,force_row_wise=True. Default ordinal label_gain is unchanged.
`topk.fit_ranker` fits200 rounds. Original `veto.run_cell_veto` tests
LR(.03,.05,.10)×prediction rounds(50,100,200)×the original margin grid, records the
chosen count in `rv['receipt']['n_estimators']`, and original
`r302.fit_and_freeze_member` passes it to `bagged.fit_ranker_n(n_est)`; that function
fits exactly int(n_est). Scoring retains original num_iteration=selected_n_estimators.
There is no best_iteration/early-stopping callback at these call sites. DART's
tree-weight normalization is part of the requested boosting change; the archived
bag function's GBDT-specific prefix-equivalence docstring is not asserted for DART.
Parameter behavior was checked against the installed4.6.0 code and
[official parameters](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html) and
[DART implementation](https://github.com/microsoft/LightGBM/blob/v4.6.0/src/boosting/dart.hpp).

## BASE gate and physical input closure

Exact-version GET archived H316 BASE locks/picks/input identities and H315 picks,
whole-object/member SHA verification, read-only copies under own scratch. Require
all9 panels,36fold/root pick and lock hashes, exact medians
.0643163334285782/.04821480030677507/.04360142642936269, and current TRAIN input
hash/prefix equality. No BASE refits or fresh BASE model scoring. Mismatch stops
REGRESSION_FAIL before DART fits. Regression/preflight exit before fit workers start.

The H325/H322 physical loader initializes metadata only. Each fit worker loads
only its outer-TRAIN pool members, cost members and exact mixed-cache byte ranges.
No full packet cache inherited from a regression/benchmark process. Audit hooks
reject outer pool access and historical BASE outcomes; persist context/range/hash
receipts. Zero physical outer-test reads/packets before the corresponding fsynced
fit lock and parent-directory fsync. Require all36 locks plus model/constructor
audits before one outer score per partition/root. STARTED/COMPLETE markers prevent
repeat scoring. No scoring a partial set of fits, no cut cells or deadline reset.

## Complete-run runtime forecast, before fits

`runtime_forecast_initial.json` was recorded before any fit. Its planning reference
is the complete36 H323 GBDT member fit workload, including extra snapshot scoring:
26,268.587 CPU seconds over4,550.1 wall seconds. Preliminary assumptions:90%ranker
CPU, DART ranker multiplier1.5,6workers,parallel efficiency.9,safety1.15,600seconds
non-fit reserve. At18:05 start this projects20:30:12, provisionally inside20:45.
These are planning assumptions, not measured DART performance or a speed verdict.

After publication and the exact BASE gate, one fixed data-free timing fixture
refines the forecast before ANY campaign task: synthetic RNG329000,
X float32[4096,568],512groups of8, each group's labels a random permutation0..7,
lr.05,200rounds. Three paired fixed seeds329001/2/3, alternating fit order
GBDT-DART/DART-GBDT/GBDT-DART. Six fixture ranker fits total; zero real BASE refits,
zero TRAIN feature/label inputs in the fixture. This is performance control only;
no hyperparameter or dataset tuning and no scientific cells/metrics are taken from it.

Use the WORST of the three paired DART/GBDT time ratios. Forecast complete36 CPU as
reference_CPU×(.1+.9×ratio)/(eligible_workers×.9)×1.15 plus300seconds for full scoring.
Recompute at the actual launch time with fresh eligible assigned cores; require
projected finish<=20:43 (2minute headroom to compute stop). If the refined forecast
misses, record it in the claim/status and close INCOMPLETE_RUNTIME_FORECAST with
zero campaign fits/outer scores. Do not truncate the36-cell design.

## Metrics and decision

All9 cells, raw utility versus r882 slot0: family-equal mean and literal family
median gains with whole-family bootstrap CI, drop-top1/5/10,LOFO minimum,W/T/L,
harmed gain<-.01,invalid,switches,Elo/CI and config agreement. Exact Elo label:
`reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN`.
Ordinary gain/median/Elo intervals use10,000 family draws,RNG302150.

Single-arm ADVANCE iff all: paired DART-minus-BASE drop-top10 median over9cells
>=+.002 and one-sided LCB95>0; DART median full gain>=90%BASE; median harmed<=BASE;
per-root paired drop-top10 median across3partitions>0 on>=2/3roots. Paired bootstrap
10,000 draws,RNG329191,one shared family resample across both arms/all9cells;
each arm re-drops its own top ten;statistic median overcells;LCB quantile.05.
Ties never advance; else DART_NOT_ADVANCED. Any dev91 check requires a separate task.

## Resources, custody and records

xbabe2 CPU only,nice19,numeric1,MemoryMax6GiB/SwapMax0,<=6assigned cores
16,17,18,19,20,22; verify each and SMT80,81,82,83,84,86>=85%idle before launch,
disjoint H327/H328. Stop hostCPU>85%,load>85,MemAvailable<20GiB,scratch>2GB.
Scratch `/home/ubuntu/jope-data/h329-gov03`. Hard compute20:45Z/close21:00Z Nov4.
Retain actual exit via ExecStopPost. No GPU/qualification.lock or forbidden hosts.

Each completed cell's inner archive is create-only S3H329, exact-version GET+SHA
and all member hashes verified before raw inner removal. Final scratch archive
under `experiments/rd_breakthrough_20261030/H329/`, exact-version/member verification
before own scratch deletion. No models in Git. CI_FREEZE+MASTER_MERGE_HOLD:
private index,nonforce origin/v1/next only; no PR/main/fastlane/CI. README,metrics<50KB,
code/tests,both index rows,status/H329.md and claim/comms updates. H326 stays closed.
