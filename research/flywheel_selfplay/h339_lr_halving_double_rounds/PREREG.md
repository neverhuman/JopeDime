# H339 — Learning-rate halving and doubled constructor rounds

Authority: hw_t2225_astra_h339_lrhalf.md; H337 CLOSED0e66b2c5101b.
One fixed LRHALF arm, no tuning or BASE scientific refits. Hypothesis: smaller
ranker steps improve family robustness. This experiment follows the two literal
constructor changes; it does **not** isolate equal effective scored shrinkage.

## Exact pinned changes and source limitation, before outcomes

H287 export213b30ad52e1ce48927bba960ed609bd98ab4dfe:
`topk.fit_ranker` constructs learning_rate=float(lr),n_estimators=200.
`bagged.fit_ranker_n` constructs learning_rate=float(lr),n_estimators=int(n_est).
Change only these keyword values to float(lr)/2 and original_n*2. Apply to
all inner/final ranker fits and all eight bags. Exact AST inverse restores both
original values; no callback, early stopping, label_gain override or extra arm.
BASE LR_GRID=(.03,.05,.10), N_EST_GRID=(50,100,200). Effective constructor
rates=(.015,.025,.05); topk400 rounds; bag100/200/400 rounds.
Other constructor params identical: LambdaRank/ordinal labels/default label_gain,
gbdt/max_depth3/num_leaves7/reg_lambda1.0/colsample_bytree.8/min_child_samples20,
subsample.8/frequency1/seeds/n_jobs1/deterministic/force_row_wise.

**Important source correction:** unchanged `veto.py:90-92,150` and
`ensemble.py:462-485` predict with the selected BASE-grid `num_iteration=n`
(50/100/200), for main rankers and bags. Doubling the constructor's fitted tree
budget does not double these requested prefixes. Full models retain extra trees,
but these decisions use the unchanged prefix. The nominal full-fit lr*T product
is preserved; the same selected prefix has half its former nominal shrinkage.
Selected n may change through unchanged inner selection. No silent doubling of
prediction prefixes, round grid, serialized selected_n_estimators or caller code.
Thus any result belongs to this literal two-keyword treatment, not proof of the
motivating "same total shrinkage" mechanism. Source controls explicitly verify
this limitation. Future effective-prefix doubling would require another spec.

Veto classifier code/params (including LR.05/n70/reg2) unchanged. Features,
families, seeds, 18-config tau/k/agreement grid, original LR/n/margin selection,
ties, hard bag votes, switch rules and -2.0 invalid sentinel unchanged. No H337
soft-vote code installed. Selected settings may change from altered ranker outputs.
No sample weights, neural update, DOPE load or live deployment.

## Regression and input closure

TRAIN1219contexts/205families; H303 NESTED/H308 P0/P1/P2 x roots30001–30003
x4family-grouped outerfolds. Exact-version H318 BASE GET/SHA/member hashes,
read-only final models. Fresh unchanged BASE replay must match all36fold/root
pick hashes,9panels and3medians exactly: .0643163334285782,
.04821480030677507,.04360142642936269. Mismatch=REGRESSION_FAIL; stop.
No BASE scientific refits. Two tiny treatment-only synthetic fits validate native
constructor/tree metadata; they are controls, not scientific cells or BASE fits.

Every fit worker physically admits only its outer-TRAIN pools/costs/cache byte
ranges. Historical BASE outcome files forbidden. No outer packets prebuilt.
Required separate BASE regression reads TRAIN outer rows; the zero-prelock-read
assertion describes treatment workers, not that mandatory reference replay.
Each cell819rankerfits/195unchangedgatefits. Audit all actual ranker fit params,
native requested rounds/LR/tree counts, fit multiset sizes, source/input hashes,
inner18-choice rule and unchanged final veto hashes. If all36finish, require
29484ranker records/972finalrankers/108unchangedgates and36fsynced locks before
36once-only fold/root scores forming9panels. No partial scoring or hidden restart.

## First-cell runtime gate, fixed before fitting

Run exactly P0/fold0/root30001 first, including verified model custody. Only after
that complete cell, forecast remaining35 on the admitted non-overlapping cores:
`ceil(35/workers)*first_cell_wall_seconds*1.15 + 600 seconds`.
Require projected finish strictly before01:45Z Nov5. The 15% reserve covers
different folds;600s reserves audit/outer scoring/analysis. A miss closes
INCOMPLETE_RUNTIME_FORECAST with numerical verdict UNKNOWN/null, preserving the
first-cell evidence and scoring zero outer cells. Do not cut cells, extend the
deadline, lower the reserve or substitute a historical timing after the result.

## Metrics and frozen decision

Raw utility vs r882slot0; context means within families, family equal weighting,
median9cell mean gains; literal family medians also reported. Whole-family paired
bootstrap10000draws, RNG339191, shared indices across arms and dependent cells.
Drop each arm's own top10 gains, subtract LRHALF-BASE then median9; CI.025/.975,
one-sided95LCB.05. Do not substitute difference of unpaired arm medians.

LRHALF_ADVANCE iff paired drop10median>=+.002 AND LCB>0, median gain>=.9BASE,
median harmed-family rate<=BASE, and per-root paired medians>0 on>=2/3roots.
Harm means gain<-.01. Exact+.002 is inclusive; zeroLCB/overallties fail, zero root
median not positive. Otherwise LRHALF_NOT_ADVANCED only after a complete study.
Incomplete outcomes remain unknown. No post-outcome decision changes.

Report gain+CIs/drop1,5,10/LOFO minimum/W,T,L tolerance1e-4/harm/invalid/switches,
config agreement and seed Jaccard. CONFIG_INSENSITIVE>=30/36 matching configs
is diagnostic. Elo uses inherited400-step Bradley-Terry MM, pseudocount.01,
ties omitted; label exactly: **reference = r882 slot0 (DL-Train outer); r3175 not
measured on TRAIN**. Elo is not AUC or a decision substitute. No speed verdict.
Bootstrap conditions on fitted policies, does not repeat selection or adjust for
the adaptive wave search; dependent/reused TRAIN families are not fresh evidence.
If ADVANCE, separate preregistration for dev91 vsr3175; none authorized in H339.

## Resources and custody

CPUxbabe2,nice19,numeric1,MemoryMax6GiB/MemorySwapMax0; RSS<=6GiB,
hostCPU/load1<=85,MemAvailable>=20GiB,scratch<=2,000,000,000bytes at
/home/ubuntu/jope-data/h339-astra. Assigned4/5/6/8/33/41 plus SMT68/69/70/72/97/105.
Exclude41/105 for H335's38–53/102–117 reservation. Fresh5second idle probe of
assigned cores+siblings, each>=85%; use only admitted cores, at most5workers.
Read current peer affinities; do not borrow or interrupt another lane.
Fixed01:45Nov5compute/02:00Nov5close, no reset on resume.

Create-only S3 experiments/rd_breakthrough_20261030/H339/astra-20261104T222500Z/;
exact-versionGET/SHA/every-member verification before own scratch deletion.
Archive each complete cell before reclaiming its inner models; preserve partial
files if incomplete. Original H50/H58/H276 local input S3 versions remain
unestablished; physical range and packet identities retained.

Prereg nonforcev1next before fits. Own study and one each INDEX/overnight row,
statusH339/claim+close/FINDINGS. CI_FREEZE+MASTER_MERGE_HOLD: noPR/main/fastlane/CI.
NoGPU/qualification.lock/xbabe0/3/livewriter/foreignjobs/H334/H335/dev91/DLVal/
DLTest/sealed/holdout/meta_test/fresh. DOPE
`{"composite_sha256_prefix":"c51a56d0","loaded":false,"scope":"CPU selection layer only"}`.
