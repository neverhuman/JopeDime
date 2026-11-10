# H382 — NL15 ranker max_delta_step1

Owner sol_v1_api01; authority hw_t1705_api01_h382_nl15_maxdeltastep1_rerun.md, Nov5 2026.
Publish this source/prereg non-force to origin/v1/next BEFORE any treatment fit.
Use a private index/source capsule, preserving canonical HEAD/index/foreign
dirty files. CI_FREEZE+BATCH_WINDOW: no PR/main/fastlane/hosted CI.

## Hypothesis and the sole intervention

Hypothesis: clipping leaf outputs reduces single-context -2.0 noise chasing
and improves paired drop10 over confirmed NL15. Copy committed H3707c99112677,
derived from H35202dfee2f7e; restore NL15 feature_fraction.8 before ONE arm MDS1.
Add only max_delta_step=1.0 (native max_delta_step0->1) to pinned
topk.fit_ranker and bagged.fit_ranker_n, covering inner/final/all8 bag rankers.
No new seed keyword. Removing this one keyword restores both entire
H352 NL15 ASTs. No additional arm, tuning, timing/calibration fit or BASE/NL15 refit.

Keep num_leaves15/max_depth3/min_child_samples20/reg_lambda1/reg_alpha0,
original LR/tree options, bagging_fraction.8/freq1/bagging_by_query0,
feature_fraction.8/extra_trees0, lambdarank/ndcg/sigmoid1, labels0..7/default label_gain,
Hessian floor.001/min_split_gain0, seeds/groups/features/18-config grid,
tie/switch1e-4/failure sentinel-2 unchanged. Depth3 permits at most8 realized
binary leaves. Veto binary GBDT unchanged: extra_trees0/L1=1/L2=2/minleaf25,
feature_fraction.8. Every actual constructor/native fit is audited: ranker
delta vs NL15 only max_delta_step[0,1], veto native delta empty after
resolving original seeds. Require all108 final veto SHAs identical to BASE/NL15.

## Regression first, immutable references and physical input closure

BASE H303 NESTED/H308 P0/P1/P2 x roots30001,30002,30003 x4 family-grouped
outer folds =36tasks/9panels,1219 TRAIN contexts/205 families. Fresh exact-version
GET H316 models/locks/outputs VersionId W5jY6iZlwoRszA7P9zfNLLMtRtM9ZZkW,
SHA4eae1dec792e1c7b4adff9ff9a5226d0c38cfe1bd9605ba07cc2596f4ad7b0b5,
and H315 picks VersionId jCll6z9FGwmpNg.cocb305LcVzFYuieR,
SHAe9e6b16a7381ff14191d2a4d31ce31d4b623c39297e7383ee6aeb871e08ca615.
Require9/9 panels,36 fold-root pick/lock hashes and exact partition medians
.0643163334285782/.04821480030677507/.04360142642936269.

Fresh exact-version GET H352 NL15 closeout manifest at
experiments/rd_breakthrough_20261030/H352/jrb01-20261105T075500Z/closeout/closeout-manifest.json,
VersionId RtNLAfmRJbXqdXQx7nxZLZ8dWHPcVYQC,
SHAd3b47c5fbc1ecc3c6e6d4f89dfef7d6a3f5216f45c6dd1589ce94fb413350f60.
Verify all27 parts/all2691 member hashes,9NL15 panels/36locks/scores/raw arrays.
Check native max_delta_step0 on all972 archived refit rankers and108 veto SHAs.
If archived NL15 native max_delta_step is not0: INCOMPLETE_SPEC_MISMATCH before
fits. Any regression mismatch: REGRESSION_FAIL. No reference refits/new model
predictions. Current TRAIN pools/cache prefixes must match archived input hashes.

Preparation exits before cold fitting. Fit parent loads metadata only; workers
physically open only outer-TRAIN contexts and declared TRAIN byte ranges, with
no inherited preparation/outer packet cache. Guard other contexts and archived
outcome access. Require all36 fit_locks and parent fsyncs, native/constructor/
physical guards and108 veto identities BEFORE outer packet creation/scoring.
Outer test once per partition/root;36 STARTED/COMPLETE fold/root markers prohibit
replay. No partial scoring/reduced design/reset. Zero DL-Val/dev91/DL-Test/
sealed/holdout/meta_test/fresh-cohort payload rows. DOPEc51a56d0 unloaded/
untouched, CPU selection layer only. No writer restart or AUTORESTART mutation.

## Metrics and preregistered decision

PRIMARY paired MDS1 minus archived H352 NL15; secondary paired MDS1 minus BASE.
Report family-equal mean/literal median gains and whole-family95%CIs, drop-top1/
5/10, LOFO minimum,W/T/L,harmed share(gain<-.01),invalids/switches, per-root
medians and CIs(3roots), Elo/EloCI and configuration agreement/seed Jaccard.
Elo label: "reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN".

Bootstrap10000 whole-family draws/RNG382191 shared across both arms/all9 cells.
Each arm independently redrops its own top10. Statistic median of9 cell paired
differences; CI quantiles.025/.975; one-sided LCB95 quantile.05. Per-root median
over3 partition differences and CIs from the same draws. Ordinary gain/Elo
family bootstrap also uses382191.

ADVANCE iff primary paired drop10median>=+.002 AND one-sidedLCB95>0 AND
median full gain>=.9NL15 AND median harmed<=NL15 AND paired per-root drop10
median>0 on>=2/3 roots. Otherwise MDS1_NOT_ADVANCED only after full completion.
Ties/subthreshold effects never advance. Incomplete effects/verdict null/UNKNOWN,
never zero or a scientific negative. Sequential H348-H382 multiplicity remains
uncorrected exploratory evidence. ADVANCE requires separately authorized one-shot
dev91 confirmation, NOT in H382. Elo is not AUC; no speed verdict.

## Single admission, forecast, corrected guard and fixed window

xbabe2 CPU only; primary physical54..63/siblings118..127, fallback4,5/siblings68,69.
ONE fresh5second interval over the UNION; both siblings>=85%idle, primary54..63
then fallback4,5, first<=8 passing physical cores. Require>=4
and same-interval hostCPU<=70%. No reprobe/cores outside the authorized union/otherhost/expanded
count. Bind controller/monitor/workers to admitted cores. nice19/numeric1/
systemd MemoryMax6G/MemorySwapMax0/CPUQuota<=800%. No GPU/qualification.lock.

At admission forecast H352 actual26189.516914CPU/(workers*.9)*1.15+300second
reserve. After FIRST completed task use remaining35*max(26189.516914/36,
first actualCPU)/(workers*.9)*1.15+300 from current time; ongoing credit0.
Both must end strictly before20:15Z Nov5; miss=INCOMPLETE_RUNTIME_FORECAST.
No timing fit, reduced cells, new probe or deadline reset.

CORRECTED GUARD stops ONLY own h382-run.service if MemAvailable<20GiB OR
hostCPU>85% averaged60s OR load1>108 on3 consecutive60s samples OR own scratch
>2GB(decimal), or absolute compute stop. Short CPU samples are diagnostic only;
load85 alone is not a stop. No swapping. Scratch /home/ubuntu/jope-data/h382-api01
<=2GB. Fixed compute stop **2026-11-05T20:15:00Z**, close **20:30Z**. Late means
INCOMPLETE; no partial scientific result. H354/H357/H364/H369/H377/H379/H380 and all
foreign work untouched. Never xbabe0/xbabe1/xbabe3 or foreign pause/kill.

Create-only S3 experiments/rd_breakthrough_20261030/H382/api01-20261105T170500Z/.
Exact-version GET, full-object SHA and EVERY member hash before deleting only
own generated scratch. Preserve partial bytes/costs and source/evidence; no
models in Git. No foreign cleanup or campaign replay.

## Publication and red-team

Non-force fresh-parent/rebase-on-fetch origin/v1/next only. Publish source/tests/
PREREG/README/metrics<50KB with exact DOPE scope; quote prereg commit in README.
One H382 row each wave INDEX.jsonl and overnight-hwave-index.md. Claim
H382_claim.txt under flock, comms at claim+close, status/H382.md and
evidence/H382/api01-20261105-t1705/CLOSEOUT.json. Same-agent exact-head red-team
checks chronology, zero early outer/protected rows, per-root CIs/tie bar,
sole max_delta_step delta, exact BASE/NL15 regressions,108 veto SHAs,
DOPE unloaded and no writer restart. No serving/default adoption in H382.
