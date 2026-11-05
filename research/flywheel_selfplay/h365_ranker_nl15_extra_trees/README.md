# H365 — NL15 ranker extra_trees

**XT_NOT_ADVANCED**. All36 fold/root tasks and9 partition/root panels completed on TRAIN only.

Hypothesis: randomized split thresholds in NL15 rankers may add useful diversity. ONE preregistered XT arm enables extra_trees=True for inner/final/all8 bag rankers. num_leaves15/max_depth3/MCS20/L2=1/L1=0, LR/tree options, labels/seeds/groups/grid/ties/failure sentinel and veto GBDT stay fixed. Depth3 permits at most8 realized binary leaves. Reverting the one added keyword restores exact committed H352 NL15 ASTs in both ranker functions.

Preregistration commit `4b2086bc571514cefa26a6b8d61996f304ef0733`, published 2026-11-05T12:10:49.560834+00:00; first treatment fit 2026-11-05T12:12:26.468497+00:00. Full specification in [PREREG.md](PREREG.md). Exact-version archived BASE9panels/36hashes/3medians and NL15ninepanels/36locks/scores/rawarrays passed before fitting; no BASE/NL15 refits or new archived-model predictions.

PRIMARY XT minus NL15 paired drop-top10: -0.011603, whole-family CI95 [-0.031141, +0.001337], one-sided LCB95 -0.026189. Advance requires at least+0.002 and positive LCB95 plus gain-retention/harm/root gates; ties and subthreshold effects never advance.
Secondary XT minus BASE paired drop-top10: -0.003199, CI95 [-0.014489, +0.008677], LCB95 -0.012360.

| Median over nine cells | BASE | NL15 | XT |
|---|---:|---:|---:|
| Family-equal mean gain | +0.048215 | +0.058405 | +0.039469 |
| Literal family-median gain | +0.003624 | +0.007409 | +0.006174 |
| Mean-gain whole-family CI95 | [+0.024790, +0.075667] | [+0.033755, +0.092441] | [+0.014595, +0.068131] |
| Harmed share (gain<−0.01) | 0.092683 | 0.092683 | 0.107317 |
| Elo | 1725.66 | 1734.47 | 1715.57 |

**Elo reference:** reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN. Elo is not AUC.

| Root seed | Paired drop10 median | CI95 | LCB95 |
|---|---:|---|---:|
| 30001 | -0.000815 | [-0.015748, +0.012663] | -0.013105 |
| 30002 | -0.017638 | [-0.038786, +0.001264] | -0.034242 |
| 30003 | -0.000738 | [-0.014094, +0.007089] | -0.012098 |

Configuration agreement XT/NL15: 6/36 (CONFIG_SENSITIVE). Full27cell mean/median and Elo CIs, drop-top1/5/10, LOFO minimum, W/T/L, harm, invalids, switches and seed Jaccard are in [metrics.json](metrics.json). Whole-family bootstrap10000/RNG365191 shares family draws across all9 cells and both arms; each arm drops its own top10. Root CIs retain uncertainty even where medians are positive.

Decision gates: paired_drop_top10_at_least_002=False, gain_retention=False, harmed_nonincrease=False, positive_on_two_seeds=False.

All36 fsynced fit locks, native/constructor guards,108 identical final veto SHAs and input closures passed before once-only outer scoring. Native fits: {'LGBMClassifier': 7020, 'LGBMRanker': 29484}. Every ranker native delta vs NL15 is extra_trees[0,1], every veto native delta empty. The copied archive metadata field ranker_bbq is a legacy role label; actual native bagging_by_query0/extra_trees1 is checked for every ranker. It does not indicate a query-bagging intervention.

Resources: xbabe2 CPU only, admitted cores [54, 55, 56, 57, 58, 59], nice19/math1/systemd MemoryMax6G/SwapMax0/CPUQuota800%. One5second admission gate; initial and first-task forecasts use H352 actual26189.516914CPU/.9eff/1.15safety/+300s reserve. Actual cgroupCPU 26621.577354s; no speed verdict. Corrected guard uses hostCPU>85% over60s, load1>108 on3 consecutive60s samples, MemAvailable<20GiB or ownscratch>2GB. Fixed compute stop15:45Z/close16:00Z Nov5 unchanged.
Verification: 49 data-free controls and11 custody controls pass. Initial copied-report test label failure was corrected before prereg/fits; failed evidence retained. Same-agent red-team binds source/prereg chronology, all native parameters, physical input ranges, root CIs, tie bar and both regressions. Zero fit-worker outer payload reads before locks; zero DL-Val/dev91/DL-Test/sealed/holdout/meta_test payload rows. DOPEc51a56d0 unloaded/untouched; no neural optimizer updates, GPU, qualification.lock, writer restart or foreign-job action.
Custody: create-only s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H365/api01-20261105T120000Z/closeout/closeout-manifest.json, exact VersionId `pdrIfIX30GDvcE.3blYjHSSEWW42f2Kg`, SHA256 `535245ab9e5ae84b590a23cbc288fefcd76d54d99be1d2dbef777d82c5d938b7`; 2846 files/28 parts, exact-version GET plus every member verified before own scratch cleanup. No models in Git. Evidence and executed source/vendor retained at `/home/ubuntu/jope_plan/evidence/H365/api01-20261105-t1200`.

Multiplicity: exploratory sequential H348-H365 sweep on the same folds/roots, without correction. Any ADVANCE needs separately authorized one-shot dev91 confirmation; none is performed here. Next: retain confirmed NL15; do not adopt XT from this study.
