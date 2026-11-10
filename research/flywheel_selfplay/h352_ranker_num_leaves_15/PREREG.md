# H352 — ranker num_leaves 7 to 15

Owner sol_v1_jrb01; authority hw_t0755_jrb01_h352_numleaves15.md, Nov5 2026.
H349 CLOSED/RELEASED7ee4c9be00; no reopening. Copy COMMITTED H350 source from
v1/next52558cc20966bc7f27dcd6c77da04086e0bbc6bd,
research/flywheel_selfplay/h350_ranker_reg_lambda_5/.
Publish this full source/prereg nonforce to origin/v1/next BEFORE treatment fits.

## Fixed hypothesis, one changed value

Hypothesis: a modest increase in realized tree capacity may reduce underfitting.
ONE arm NL15 changes only the existing num_leaves7->15 value in pinned H287
(topk.fit_ranker and bagged.fit_ranker_n), including all inner/final/eight-bag
rankers. Restoring7 must restore both entire original BASE function ASTs.
No added keyword, tuning, calibration fit, extra arm or BASE refit.

max_depth remains3: binary trees can therefore have at most8 realized leaves,
including when the configured num_leaves limit is15. Record configured/native
limits and observed per-tree leaf counts; never claim15-leaf realized trees.
Record fresh native BASE num_leaves/max_depth before treatment; require7/3 or
stop INCOMPLETE_SPEC_MISMATCH. This is an audit of archived copies, not a fit.

Restore ranker reg_lambda1, implicit reg_alpha0 and min_child_samples20 from
original BASE. Preserve lambdarank, sigmoid1, ndcg, labels0..7/implicit label_gain,
Hessian floor.001, min_split_gain0, subsample.8/freq1,colsample.8,original
learning-rate/tree options, features, all seeds/groups,18-config grid,
tie/switch1e-4 and failure sentinel-2. The original learning-rate inputs and
selection grid (including0.05) remain exact BASE code; no separate LR change.
Veto remains binary GBDT with original L1=1/L2=2/minleaf25; require all108 final
veto SHAs equal BASE. Audit every actual constructor/native parameter record:
ranker delta only num_leaves[7,15], veto delta empty, observed ranker leaves<=8.

## Frozen BASE and physical TRAIN closure

BASE H303 NESTED/H308 P0/P1/P2 x roots30001,30002,30003 x4 family-grouped outer
folds =36 fit tasks/9 panels,1219 TRAIN contexts/205 families. BASE source
819a45aa5e30e662ab511cd9962b85b2b8c1e44a, H287 export
213b30ad52e1ce48927bba960ed609bd98ab4dfe; all seven vendor hashes pinned.
Fresh exact-version GET read-only archived BASE:
H316 experiments/rd_breakthrough_20261030/H316/solhwv-20261104T105633Z/h316_models_locks_outputs.tar.gz,
VersionId W5jY6iZlwoRszA7P9zfNLLMtRtM9ZZkW,
SHA256 4eae1dec792e1c7b4adff9ff9a5226d0c38cfe1bd9605ba07cc2596f4ad7b0b5;
H315 experiments/rd_breakthrough_20261030/H315/picks.npz,
VersionId jCll6z9FGwmpNg.cocb305LcVzFYuieR,
SHA256 e9e6b16a7381ff14191d2a4d31ce31d4b623c39297e7383ee6aeb871e08ca615.
Verify full object/selected member hashes,9panels/36fold-root pick+lock hashes
and3 medians EXACT: .0643163334285782/.04821480030677507/.04360142642936269.
Mismatch stops REGRESSION_FAIL. No BASE refit or new BASE model predictions.
Match current1219 TRAIN pool hashes,selection/r882 costs and4cache TRAIN prefixes
to archived input manifest9b148003d48ab609b2a796c57fb8b4c0296cc0ee79771f034944f4b8dc064263.

Preparation terminates before cold fit admission. Fit parent loads metadata only;
workers physically open only outer-TRAIN pools and declared TRAIN byte ranges.
No inherited outer/preparation packet cache. Record allowed/forbidden contexts,
packet/range hashes; fit-worker outer features/packets before fsynced fit_lock
and parent fsync must be0. All36 locks/native/constructor/physical guards and
108veto identities must pass before cold once-only outer TRAIN scoring.
STARTED/COMPLETE markers prohibit replay. No partial scoring or reduced design.
No DL-Val/dev91/DL-Test/sealed/holdout/meta_test/fresh-cohort payloads.
DOPE c51a56d0 unchanged/unloaded, CPU selection layer only.

## Metrics and locked decision

Raw utility vs r882slot0 paired with archived BASE. Exact Elo label:
reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.
Report family-equal mean/literal family-median gains and whole-family95%CIs,
drop-top1/5/10,LOFO min,W/T/L,harm(gain<-.01),invalids,switches,per-root medians
and paired CIs,Elo/EloCI,config agreement,seed Jaccard. Ordinary bootstrap retains
10000draws/RNG302150. Paired bootstrap10000/RNG352191 uses shared family samples
both arms/all9cells, each arm redrops own top10; median9cells is statistic,
CI95 quantiles.025/.975, one-sidedLCB95 quantile.05. Per-root medians and CIs use
these same draws and median across the3 partitions for each root.
ADVANCE iff paired drop10median>=+.002 andLCB95>0 AND NL15median fullgain>=.9BASE
AND median harmed<=BASE AND paired median strictly positive on>=2/3roots.
Ties/subthreshold never advance. Else NL15_NOT_ADVANCED only after full scores.
Incomplete results unknown/null, never zero or a scientific negative. Report Elo;
no speed verdict. A future one-shot dev91 vs r3175 is outside H352 authorization.

## One capacity gate and immutable window

xbabe2 CPU only,nice19,numeric1,systemd MemoryMax6G/MemorySwapMax0. Two allowed
physical pools: A=0..14(+SMT64..78), B=27..37(+SMT91..101). ONE simultaneous fresh
5-second interval samples both allowed pools, both siblings>=85%idle. Choose ONE
pool with the most eligible pairs, tie=A; take first<=8 eligible within that
pool. Never mix pools or remeasure. Require>=4, same-interval hostCPU<=70%, and
full36 forecast before compute stop. H35115..26/79..90 and H33538..53/102..117
are excluded. Bind controller/monitor threads and fit/score workers to admitted
cores after admission; no borrowed cores or unapproved worker count.

Forecast H347 measured29770.57CPU seconds/(workers*.9)*1.15+300s scoring reserve;
NL15 cost multiplier1 is unmeasured, no speed claim. After FIRST completed
fold/root task reforecast remaining35*max(29770.57/36,actual first-task CPU)/
(workers*.9)*1.15+300 from current time, ongoing credit0. Both forecasts must
end strictly before2026-11-05T11:15:00Z. Otherwise INCOMPLETE_RUNTIME_FORECAST;
core/host rejection means INCOMPLETE_CPU_ADMISSION. No timing fit, reprobe,
restart, reset, cut cells, expanded cores or partial scoring.
Stop only own cgroup for hostCPU>85%,load>85,MemAvailable<20GiB,ownscratch>2GB
(decimal) or fixed compute deadline. Preexisting full swap never authorizes swap.
Scratch /home/ubuntu/jope-data/h352-jrb01 only. Compute stop11:15Z/close11:30Z
Nov5 fixed. Late means INCOMPLETE. Create-only S3
experiments/rd_breakthrough_20261030/H352/jrb01-20261105T075500Z/; exact-version
GET/fullSHA/EVERYmember verification before deleting own scratch. Preserve all
partial states/costs/unknowns; no replay/relabel or models in Git.

## Publication and review

CI_FREEZE+BATCH_WINDOW: private index/objects; preserve canonical HEAD/index and
foreign staged/dirty files. Nonforce fresh-parent origin/v1/next only, no
PR/main/fastlane/hostedCI. Code/tests/PREREG/README/metrics<50KB, oneH352row each
wave/overnight index. Same-agent exact-head review. Before close verify source
and prereg timestamps,0fit-worker outer reads before lock,0protected/dev payloads,
all36 once-only scores,root CIs/tie bar,only num_leaves delta,exact BASE/108veto
SHAs,configured15 versus realized<=8,DOPEc51a56d0unloaded.
No GPU/qualification.lock/foreign jobs or hosts/xbabe0/xbabe1(any)/xbabe3/live
writer changes. H335/H351/H350R and all other owners remain untouched.
