# H365 — NL15 ranker extra_trees

Owner sol_v1_api01; authority hw_t1200_api01_h365_nl15_extratrees.md, Nov5 2026.
This full preregistration/code must be published non-force to origin/v1/next
before any treatment fit. Canonical HEAD/index/foreign dirty files are preserved
through a private index and source capsule. No PR/main/fastlane/hosted CI.

## One fixed intervention and hypothesis

Hypothesis: stochastic split-threshold diversity within NL15 bags improves
paired drop-top10 over the confirmed NL15 selection incumbent. Copy committed
H352 harness02dfee2f7e/research/flywheel_selfplay/h352_ranker_num_leaves_15.
Reuse committed H360 comparator, all-native parameter auditing and corrected
resource wiring only, with exact source provenance recorded. ONE arm XT adds
extra_trees=True (native extra_trees1) to ranker constructors in pinned
topk.fit_ranker and bagged.fit_ranker_n, covering inner/final/all8 bag rankers.
Reverting this one added keyword must restore both exact H352 NL15 function
ASTs. No sweep, tuning, timing fit, calibration, extra arm, BASE or NL15 refit.

Keep num_leaves15/max_depth3/min_child_samples20/reg_lambda1/reg_alpha0,
all original learning rates/tree counts, lambdarank/ndcg/sigmoid1, labels0..7,
default label_gain, Hessian floor.001, min_split_gain0, bagging_by_query0,
subsample.8/freq1/colsample.8, variable seeds/groups/features/18-config grid,
tie/switch1e-4/invalid sentinel-2 exactly. Depth3 bounds realized binary leaves
to8 even with configured num_leaves15. No explicit extra_seed or seed changes.
Veto remains the original binary GBDT extra_trees0/L1=1/L2=2/minleaf25.
Audit every constructor and every native fit: ranker native delta vs NL15 is
only extra_trees[0,1], veto native delta empty after resolving original seeds.
All108 final veto SHAs must equal archived BASE/H352. Record realized leaves.

## Regression first and TRAIN input closure

Archived BASE H303 NESTED/H308 P0/P1/P2 x roots30001..30003 x4 grouped outer
folds gives36tasks/9panels,1219 TRAIN contexts/205 families. Exact-version GET
H316 models/locks/outputs VersionId W5jY6iZlwoRszA7P9zfNLLMtRtM9ZZkW,
SHA4eae1dec792e1c7b4adff9ff9a5226d0c38cfe1bd9605ba07cc2596f4ad7b0b5,
and H315 picks VersionId jCll6z9FGwmpNg.cocb305LcVzFYuieR,
SHAe9e6b16a7381ff14191d2a4d31ce31d4b623c39297e7383ee6aeb871e08ca615.
Require9/9 panels,36 fold-root pick/lock hashes and exact partition medians
.0643163334285782/.04821480030677507/.04360142642936269.

Fresh exact-version GET archived H352 NL15 closeout manifest at
experiments/rd_breakthrough_20261030/H352/jrb01-20261105T075500Z/closeout/closeout-manifest.json,
VersionId RtNLAfmRJbXqdXQx7nxZLZ8dWHPcVYQC,
SHAd3b47c5fbc1ecc3c6e6d4f89dfef7d6a3f5216f45c6dd1589ce94fb413350f60.
Verify all27 part hashes and every2691 member; retain only outcome/lock metadata.
Require all9 NL15 panels/36 locks and once-only score hashes/raw family gains
exact; inspect all972 archived refit ranker configurations and108 veto SHAs.
Mismatch is REGRESSION_FAIL. No archived model is newly predicted or refitted.
Verify current TRAIN identities/cache prefixes against archived BASE manifest.

Preparation exits before the cold fitting process. Parent loads metadata only;
each fit worker opens only its outer-TRAIN contexts and declared TRAIN byte
ranges, with no inherited preparation/outer packet cache. Guard excluded
contexts and archived comparator outcomes. All36 fit_lock files, parent fsyncs,
native/constructor/input guards and108 veto identities must pass BEFORE outer
feature packets or once-only treatment scoring. One outer test pass per
partition/root, recorded by36 STARTED/COMPLETE fold/root score markers.
No partial scoring/replay/reset/reduced design. Zero DL-Val/dev91/DL-Test,
sealed/holdout/meta_test/fresh-cohort payload rows. DOPE c51a56d0 unloaded and
untouched; CPU selection layer only. No flywheel writer restart/mutation;
writer intentionally remains stopped at round5049.

## Metrics and locked advance rule

PRIMARY paired comparison XT minus H352 NL15; secondary XT minus BASE.
Raw utilities reference r882slot0. Report family-equal mean and literal family
median gains with whole-family95%CIs, drop-top1/5/10, LOFO minimum, family W/T/L,
harmed share gain<-.01, invalids/switches, per-root medians and CIs, Elo/EloCI,
configuration agreement and seed Jaccard. Exact Elo label:
"reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN".

Whole-family bootstrap10000 draws/RNG365191, shared family resamples over all9
cells and both paired arms. Each arm independently redrops its own top10;
statistic median of9 cell differences. CI quantiles.025/.975; one-sided LCB95
quantile.05. Per-root statistic median over its3 partition differences and CIs
from the same shared draws. Ordinary gain/Elo family bootstrap also RNG365191.

ADVANCE iff primary paired drop10median>=+.002 AND one-sidedLCB95>0 AND XT
median full gain>=.9*NL15 AND XT median harmed<=NL15 AND per-root paired
drop10median strictly positive on>=2/3 roots. Ties/subthreshold never advance.
Otherwise XT_NOT_ADVANCED only after the entire design finishes. Incomplete
metrics/verdict remain null/UNKNOWN, never zero/scientific negative. Sequential
sweep H348-H365 multiplicity is uncorrected exploratory evidence. Any ADVANCE
requires separately authorized one-shot dev91 confirmation, outside H365.
Elo is not AUC and no speed verdict is authorized.

## Single capacity gate, forecast, resources and immutable window

xbabe2 CPU only; pool54..63 plus siblings118..127. ONE fresh simultaneous
5second interval; require both siblings>=85%idle. First<=8 passing physical
cores; >=4 needed and same-interval hostCPU<=70%. No second probe/reprobe,
borrowed cores, other host or expanded count. Bind controller/monitor/workers
to admitted cores. nice19/numeric threads1/systemd-run --user MemoryMax6G,
MemorySwapMax0/CPUQuota<=800%. No GPU/qualification.lock.

At admission forecast H352 actual26189.516914 CPU seconds/(workers*.9)*1.15
+300second scoring reserve. After FIRST completed task, remaining35*
max(26189.516914/36,first actualCPU)/(workers*.9)*1.15+300 from current time;
ongoing work credit0. Both must end strictly before15:45Z Nov5, else
INCOMPLETE_RUNTIME_FORECAST. No timing fit, cut cells or deadline reset.

CORRECTED GUARD stops ONLY own h365-run.service if MemAvailable<20GiB OR
host CPU>85% averaged over60seconds OR load1>108 on3 consecutive60second
samples OR own scratch>2GB(decimal), or fixed compute stop. Five-second CPU
samples are diagnostic only; load85 alone is not a stop. Preexisting swap
does not permit swapping. Own scratch /home/ubuntu/jope-data/h365-api01<=2GB.
Fixed compute stop **2026-11-05T15:45:00Z**; close **16:00Z**. Late is
INCOMPLETE, with no partial scientific scoring. No foreign job pause/kill.
H354/H357/H364/H360/H361/H362/H363 are untouched. Never xbabe0/xbabe1/xbabe3.

Create-only S3 experiments/rd_breakthrough_20261030/H365/api01-20261105T120000Z/.
Exact-version GET, full-object SHA and EVERY archive member hash must verify
before removing only owned generated scratch. Preserve partial work/costs and
all pinned source/evidence. No models in Git. No foreign cleanup.

## Publication and red-team

Non-force fresh-parent/rebase-on-fetch origin/v1/next only under CI_FREEZE and
BATCH_WINDOW. Publish code/tests/PREREG/README/metrics<50KB; one H365 row each
in wave INDEX.jsonl and overnight-hwave-index.md. Quote prereg commit in final
README. Claim H365_claim.txt under flock, comms line at claim+close, status
master/status/H365.md and evidence/H365/api01-20261105-t1200/CLOSEOUT.json.
Same-agent red-team binds exact published head, prereg before first fit,
0 outer packet reads before locks/0 protected rows, per-root CIs/tie bar,
only extra_trees delta, exact BASE+NL15 regressions,108 veto hashes, unchanged
DOPE/unloaded and stopped writer. No release/serving/default adoption in H365.
