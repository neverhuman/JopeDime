# H349 — rank_xendcg objective rerun

Owner sol_v1_jrb01; authority hw_t0400_jrb01_h349_xendcg.md, 2026-11-05.
H344 is CLOSED/RELEASED at 8e2464435badb0bf3ce801b6b02233823c05f33a.
Copy its committed harness from v1/next6f87ae99ce2752d595663c181aeb2457bd231939;
this reruns H340's unmeasured science under this new fixed allocation. H340 had
zero fits after only three cores admitted; no H340 first-cell timing exists.
Publish this prereg nonforce to origin/v1/next BEFORE any H349 treatment fit.

## Fixed intervention and inverse

Hypothesis: a listwise ranking surrogate may improve robustness to noisy
context outcomes. It is a hypothesis, not a measured robustness/speed claim.
BASE is unchanged H303 NESTED/H308 P0/P1/P2 x roots30001,30002,30003 x four
family-grouped outer folds:36 fit tasks,9 partition/root panels,1219 TRAIN
contexts/205 families. No tuning, calibration fit, alternative arm or changed
labels is permitted.

ONE arm XENDCG changes only the existing ranker objective value lambdarank ->
rank_xendcg in pinned H287 topk.fit_ranker and bagged.fit_ranker_n, including
inner/final/all eight bag rankers. Add objective_seed as an exact AST copy of
that call's original random_state expression. Restoring objective=lambdarank
and removing only objective_seed must restore both entire BASE function ASTs.
Audit every actual constructor/native fit: objective is rank_xendcg and native
objective_seed == native seed == original constructor random_state. All other
arguments/streams stay fixed; veto keyword delta is empty.

BASE native objective=lambdarank and sigmoid=1.0 were verified on all972 exact-
version archived refit rankers, LightGBM4.6.0; native inventory digest
ccd64cd998d3901751ec2c497aefb041418355b98b0d77916342889f849a9e9e.
Mismatch stops INCOMPLETE_SPEC_MISMATCH before fitting. H344's sigmoid override
is absent: native sigmoid remains1.0, Hessian floor.001, path_smooth0,
extra_trees0,max_delta_step0. Preserve ndcg evaluation, labels0..7, implicit
BASE label_gain (ignored by XE-NDCG, never overridden), truncation level,
min_child_samples20,num_leaves7,max_depth3,subsample.8/freq1,colsample.8,
reg_lambda1,original LR/tree grid/selected bag rounds,GBDT,deterministic/
force_row_wise,n_jobs1,all seeds/features/groups,18-config grid,
1e-4 tie/switch rules and -2.0 invalid sentinel. Binary veto GBDT is unchanged;
all108 saved final veto SHAs must match archived BASE. Reject unsupported native
objective/seed/label behavior as INCOMPLETE_SPEC; no substitute or workaround.

## Exact BASE and physical input closure

BASE code819a45aa5e30e662ab511cd9962b85b2b8c1e44a; H287 export
213b30ad52e1ce48927bba960ed609bd98ab4dfe. Immutable vendor hashes are pinned in
manifest.json. Exact-version GET BASE:
H316 experiments/rd_breakthrough_20261030/H316/solhwv-20261104T105633Z/h316_models_locks_outputs.tar.gz,
VersionId W5jY6iZlwoRszA7P9zfNLLMtRtM9ZZkW,
SHA256 4eae1dec792e1c7b4adff9ff9a5226d0c38cfe1bd9605ba07cc2596f4ad7b0b5;
H315 experiments/rd_breakthrough_20261030/H315/picks.npz,
VersionId jCll6z9FGwmpNg.cocb305LcVzFYuieR,
SHA256 e9e6b16a7381ff14191d2a4d31ce31d4b623c39297e7383ee6aeb871e08ca615.
Require full-object/member hashes and archived read-only copies. Regression
must pass9/9 panels,36 fold/root pick+lock hashes and3 medians EXACT:
.0643163334285782/.04821480030677507/.04360142642936269. Otherwise REGRESSION_FAIL.
No BASE refits or fresh BASE model predictions. Match1219 TRAIN pool identities,
selected/r882 costs and all4 consumed TRAIN cache prefixes to archived manifest
9b148003d48ab609b2a796c57fb8b4c0296cc0ee79771f034944f4b8dc064263.

Preparation processes end before the cold fit process starts. Fit parent loads
metadata only; workers physically open only their outer-TRAIN pools/cache byte
ranges. No preparation/outer packet cache is inherited. Audit allowed/forbidden
contexts and bytes; fit-worker outer rows/packets before fsynced fit_lock and
parent-directory fsync must be0. All36 locks, guards, model/native audits and
veto identities pass before cold once-only outer TRAIN scoring. STARTED/COMPLETE
markers prohibit replay. No partial scoring, retries, cut cells or relabeling.
No DL-Val/dev91/DL-Test/sealed/holdout/meta_test/fresh-cohort payloads.
DOPE c51a56d0 remains unchanged/unloaded; CPU selection layer only.

## Metrics and locked decision

Raw utility vs r882slot0 paired with archived BASE; exact Elo label:
reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.
Report family-equal mean and literal median with whole-family95%CIs,
drop-top1/5/10,LOFO minimum,W/T/L,harmed(gain<-.01),invalid,switches,per-root
medians and per-root paired CIs,Elo/EloCI,seed Jaccard and config agreement.
Ordinary bootstrap remains10000 whole-family draws/RNG302150.

Paired drop10 uses10000 whole-family draws/RNG349191, shared families across
both arms/all9 panels, each arm redrops its own top10; statistic median9 panels,
LCB95 quantile.05. Per-root paired medians/CIs use those same draws and the
median across three partitions for that root; descriptive only, no new gate.
ADVANCE iff paired drop10 median>=+.002 AND LCB95>0 AND XENDCG median full gain
>=90%BASE AND median harmed<=BASE AND paired drop10 median across partitions
is strictly positive on>=2/3roots. Ties/subthreshold never advance. Otherwise
XENDCG_NOT_ADVANCED, only after full scoring. No AUC or speed verdict. If ADVANCE,
next is a separately authorized one-shot dev91 vs r3175; NOT part of H349.

## One admission and fixed budget

xbabe2 CPU only,nice19,numeric1,systemd MemoryMax6G/MemorySwapMax0. Assigned
physical cores15..26 and SMT79..90; ONE fresh5-second per-core+SMT interval,
both siblings>=85%idle. Take first<=6 passing pairs, require>=4 AND hostCPU<=70%
in that same interval AND full36 forecast fits. No second probe/borrowed cores.
H3480..14/27..33 and H33538..53, plus all siblings, are excluded.
Controller/fit/score numerical processes bind to admitted cores after gate.

Measured reference H344 full36 CPU26110.639439s (receipt SHA
8d860ca02f90654155bb0f6d52ddae61a44b0ec8c77d52d8798c6956218d8466), H340 first-cell unavailable.
Initial forecast CPU/(workers*.9)*1.15+300s scoring reserve; objective multiplier1
is an unmeasured planning assumption, not a speed result. After FIRST completed
fold/root task, forecast remaining35*max(H344CPU/36,actual first-task CPU)/
(workers*.9)*1.15+300 from current time, ongoing work credit0. Require ending
strictly before2026-11-05T06:15:00Z; otherwise INCOMPLETE_RUNTIME_FORECAST.
Core/host admission failure means INCOMPLETE_CPU_ADMISSION. No cut design,
reprobe, fresh timing fit, restart, deadline extension or partial scientific score.

Stop only own cgroup if hostCPU>85%,load>85,MemAvailable<20GiB or own scratch>2GB
(decimal). No swap despite preexisting full host swap. Scratch only
/home/ubuntu/jope-data/h349-jrb01. Compute stop06:15Z Nov5, close06:30Z fixed.
Late means INCOMPLETE. Create-only S3 experiments/rd_breakthrough_20261030/H349/
exact-version GET/full SHA/every-member proof precedes deleting own scratch.
Keep costs/failures/unknown outcomes truthful; no models in Git.

## Publication and red-team

CI_FREEZE+BATCH_WINDOW: private index/objects preserve canonical HEAD/index/
foreign files; ordinary fresh-parent nonforce origin/v1/next only. No PR/main/
fastlane/hostedCI/GPU/qualification.lock/live writer/foreign jobs/xbabe0/1/3.
Code/tests/PREREG/README/metrics<50KB and oneH349row each wave/overnight index.
Before close verify exact prereg head precedes every fit,0fit-worker outer reads
before lock,0protected/dev rows,all36 once-only scores,per-root mean/median and
paired CIs,tie bar,only objective+its seed delta in every native fit,exact BASE.
