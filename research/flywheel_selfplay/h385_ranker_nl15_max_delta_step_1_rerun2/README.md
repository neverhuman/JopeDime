# H385 — NL15 ranker max_delta_step1

**MDS1_NOT_ADVANCED**. All36 fold/root tasks and9 partition/root panels completed on TRAIN only.

Hypothesis: clipping NL15 ranker leaf outputs may reduce single-context noise chasing. ONE preregistered MDS1 arm adds ranker max_delta_step=1.0 for inner/final/all8 bag rankers. num_leaves15/max_depth3/MCS20/L2=1/L1=0, LR/tree options, labels/seeds/groups/grid/ties/failure sentinel and veto GBDT stay fixed. Depth3 permits at most8 realized binary leaves. Reverting the one added keyword restores exact committed H352 NL15 ASTs in both ranker functions.

Preregistration commit `6e6bbc072733d401d396d46f4252dd7bd01f0751`, published 2026-11-05T17:44:04.817426+00:00; first treatment fit 2026-11-05T17:44:29.782429+00:00. Full specification in [PREREG.md](PREREG.md). Exact-version archived BASE9panels/36hashes/3medians and NL15ninepanels/36locks/scores/rawarrays passed before fitting; no BASE/NL15 refits or new archived-model predictions.

PRIMARY MDS1 minus NL15 paired drop-top10: -0.004410, whole-family CI95 [-0.013792, -0.000054], one-sided LCB95 -0.012635. Advance requires at least+0.002 and positive LCB95 plus gain-retention/harm/root gates; ties and subthreshold effects never advance.
Secondary MDS1 minus BASE paired drop-top10: +0.002378, CI95 [-0.005184, +0.007241], LCB95 -0.004170.

| Median over nine cells | BASE | NL15 | MDS1 |
|---|---:|---:|---:|
| Family-equal mean gain | +0.048215 | +0.058405 | +0.049369 |
| Literal family-median gain | +0.003624 | +0.007409 | +0.004783 |
| Mean-gain whole-family CI95 | [+0.024906, +0.076212] | [+0.033914, +0.092347] | [+0.025891, +0.092804] |
| Harmed share (gain<−0.01) | 0.092683 | 0.092683 | 0.087805 |
| Elo | 1725.66 | 1734.47 | 1736.23 |

**Elo reference:** reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN. Elo is not AUC.

| Root seed | Paired drop10 median | CI95 | LCB95 |
|---|---:|---|---:|
| 30001 | -0.009588 | [-0.017561, +0.001759] | -0.016166 |
| 30002 | -0.003994 | [-0.016618, +0.005894] | -0.014685 |
| 30003 | -0.004410 | [-0.020208, +0.002684] | -0.017774 |

Configuration agreement MDS1/NL15: 8/36 (CONFIG_SENSITIVE). Full27cell mean/median and Elo CIs, drop-top1/5/10, LOFO minimum, W/T/L, harm, invalids, switches and seed Jaccard are in [metrics.json](metrics.json). Whole-family bootstrap10000/RNG385191 shares family draws across all9 cells and both arms; each arm drops its own top10. Per-root confidence intervals are reported independently of the aggregate advancement gate.

Decision gates: paired_drop_top10_at_least_002=False, gain_retention=False, harmed_nonincrease=True, positive_on_two_seeds=False.

All36 fsynced fit locks, native/constructor guards,108 identical final veto SHAs and input closures passed before once-only outer scoring. Native fits: {'LGBMClassifier': 7020, 'LGBMRanker': 29484}. Every ranker native delta vs NL15 is max_delta_step[0,1], every veto native delta empty. Ranker feature_fraction0.8, extra_trees0, bagging_by_query0, bagging_fraction0.8 and all other native defaults remain pinned.

Resources: xbabe2 CPU only, eligible cores [38, 39, 40, 41, 42, 43, 44, 45], nice19/math1/systemd MemoryMax6G/SwapMax0/CPUQuota800%. One5second admission gate; initial and first-task forecasts use H352 actual26189.516914CPU/.9eff/1.15safety/+300s reserve. Actual cgroupCPU 27785.309508s; no speed verdict. Corrected guard uses hostCPU>85% over60s, load1>108 on3 consecutive60s samples, MemAvailable<20GiB or ownscratch>2GB. Fixed compute stop21:45Z/close22:00Z Nov5 unchanged.
Verification: 54 data-free controls and11 custody controls pass. Same-agent red-team binds source/prereg chronology, all native parameters, physical input ranges, root CIs, tie bar and both regressions. Zero fit-worker outer payload reads before locks; zero DL-Val/dev91/DL-Test/sealed/holdout/meta_test payload rows. DOPEc51a56d0 unloaded/untouched; no neural optimizer updates, GPU, qualification.lock, writer restart or foreign-job action.
Custody: create-only s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H385/api01-20261105T173000Z/closeout/closeout-manifest.json, exact VersionId `dF0VUavtNp3VcupJNuA1NGgL_xuiDxJR`, SHA256 `87e35b4a739b18c040a4e8eba7644bd7b712335c4d70dcd4e0ccaa92d69c433c`; 2854 files/27 parts, exact-version GET plus every member verified before own scratch cleanup. No models in Git. Evidence and executed source/vendor retained at `/home/ubuntu/jope_plan/evidence/H385/api01-20261105-t1730`.

Multiplicity: exploratory sequential H348-H385 sweep on the same folds/roots, without correction. Any ADVANCE needs separately authorized one-shot dev91 confirmation; none is performed here. Next: retain confirmed NL15; do not adopt MDS1 from this study.
