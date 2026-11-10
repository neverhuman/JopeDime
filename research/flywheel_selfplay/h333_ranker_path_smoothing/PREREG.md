# H333: fixed ranker path smoothing

Owner sol_v1_gov03. Authority: hw_t2030_gov03_h333_psmooth.md, 2026-11-04.
H329 and H326 are CLOSED and remain closed. This design has no H333 optimizer
fits or outer scores. The resource admission forecast has already failed;
this preregistration records the fixed design and does not authorize a launch.

## Hypothesis and exact intervention

Hierarchical leaf shrinkage toward ancestors may reduce a few small leaves'
dominance without suppressing large supported leaves. One fixed PSMOOTH arm:
add only `path_smooth=20.0` to ranker constructors `topk.fit_ranker` and
`bagged.fit_ranker_n` at pinned H287 export
`213b30ad52e1ce48927bba960ed609bd98ab4dfe`. BASE leaves path_smooth unset
(native default0.0, also observed in an existing H329 synthetic GBDT model).
The value20.0 is the unchanged BASE min_child_samples, fixed by the owner,
without outcome-based selection. No parameter sweep or additional arm.

Keep boosting_type=gbdt, objective=lambdarank, default ordinal label_gain,
all labels/features/seeds, max_depth3, num_leaves7, min_child_samples20,
subsample.8/subsample_freq1, colsample_bytree.8, reg_lambda1, n_jobs1,
deterministic/force_row_wise settings and all other constructor arguments.
Original topk fits200 rounds; the inner LR(.03,.05,.10) and prediction-round
(50,100,200) grids remain, with int(n_est) in original bag fit calls.
No early-stopping callback, DART, cosine, clipping, extra trees or gain change.
Veto classifiers/functions stay byte-for-byte unchanged. Removing the single
keyword must restore both complete original BASE function ASTs. Native fit
parameter audits would require path_smooth20.0 on all inner/final/eight-bag
rankers and unchanged veto/default0.0; no native H333 models are produced here.

## Required design and gates if admitted

Use unchanged H303 NESTED/H308 TRAIN-only loader, P0/P1/P2 × roots30001–30003
× four family-grouped outer folds: all36 tasks and all9 cells mandatory.
First scientific gate: read-only exact-version archived BASE, all9 panels,
36fold/root pick+lock hashes and medians .0643163334285782,
.04821480030677507 and .04360142642936269 exact. No BASE refits; mismatch
REGRESSION_FAIL stops before PSMOOTH. This scientific gate is NOT_RUN in H333
because the earlier resource-admission forecast forbids launching the campaign;
H329's historical PASS is not substituted for a fresh H333 regression result.

Each fit worker physically sees only its outer-TRAIN rows/byte ranges. Require
zero outer-test feature rows/packets before its fit lock, fsync the file and
parent directory, and verify all36 locks before scoring. Score each outer
partition/root once, using STARTED/COMPLETE markers. Retain original inner
18-config TAU(.05,.10,.15) × k(5,6,7) × agreement(2,3), raw inner objective,
member/eight-bag streams, margin/tie/switch rules and failure sentinel−2.0.

Metrics would be raw utility against r882slot0: family-equal mean and literal
median with whole-family CI, drop-top1/5/10, LOFO minimum, W/T/L, harmed gain<−.01,
invalid, switches, Elo/CI and configuration agreement. Elo label exactly:
`reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN`.
Ordinary whole-family bootstrap10,000 draws,RNG302150 as frozen harness.

ADVANCE iff paired(PSMOOTH−BASE) drop-top10 median over9cells≥+.002 AND one-sided
LCB95>0; median full gain≥90%BASE; median harmed≤BASE; and positive per-root
paired drop10 median across3partitions on≥2/3roots. Paired whole-family bootstrap
10,000 draws,RNG333191; one family resample shared across both arms/all9cells;
each arm redrops its own top10; statistic median9cells; LCB quantile.05.
Ties/subthreshold margins fail. Only a completed evaluation can yield ADVANCE
or PSMOOTH_NOT_ADVANCED. The current incomplete task has a null numerical verdict.
Any later dev91/r3175 test is outside H333 and requires separate authorization.

## Resource admission and current disposition

Use one five-second per-core+SMT idle interval, same H329 inclusive≥85% rule,
assigned16/17/18/19/20/22 with siblings80/81/82/83/84/86 only. Observed
20:27:09–14Z: only core22/sibling86 passes. No capacity resampling to seek a pass,
borrowed H331/H332 cores, cut cells, partial scores or deadline reset.

Forecast complete36 from retained H329 synthetic GBDT timing and measured full
workload. H329's three GBDT timing fits took2.4042165620194282,
2.436658366001211 and2.5926205029827543seconds. Its measured DART full-run CPU
30409.446641s, worst paired ratio1.088392827976701 and previously declared90%
ranker CPU-share assumption imply28168.539464114267GBDT-equivalent CPU seconds.
Compare the retained full36 GBDT reference26268.587368CPU seconds from H323
used by H329. Use the larger reference, smoothing multiplier1.0, parallel
 efficiency.9,safety1.15,300score seconds, actual eligible worker count.
This is a planning estimate, not measured PSMOOTH performance. Even the smaller
reference with perfect efficiency/no safety/no scoring reserve misses22:45Z.
See runtime_forecast.json for actual start/finish, inputs and assumptions.

Owner requires forecast miss=>INCOMPLETE. Therefore zero H333 fits, fixtures,
BASE refits, locks or outer scores are permitted in this window. BASE regression,
physical worker qualification, native audits and numerical metrics remain NOT_RUN.
The executable delivered here is the one-keyword patch, data-free AST/forecast
controls and admission evidence, not an instantiated or qualified worker harness.

xbabe2 CPU only,nice19,numeric1,MemoryMax6GiB/SwapMax0; <=6eligible assignedcores.
Stop MemAvailable<20GiB,hostCPU>85%,load>85,scratch>2GB. Fixed compute stop
22:45Z and close23:00Z Nov4. Scratch /home/ubuntu/jope-data/h333-gov03;
create-onlyS3H333 exact-versionGET+whole/memberSHA before own cleanup.
CI_FREEZE+MASTER_MERGE_HOLD: private index,nonforce origin/v1/next only;
noPR/main/fastlane/CI. NoGPU/qualification.lock/protecteddata/DL-Val/dev91/
DL-Test/livewriter/foreignjobs/forbiddenhosts. DOPEc51a56d0 unloaded and unchanged;
scope CPU selection layer only. Record README/metrics<50KB/code/tests/twoH333rows,
claim/status/comms and custody, then release H333.
