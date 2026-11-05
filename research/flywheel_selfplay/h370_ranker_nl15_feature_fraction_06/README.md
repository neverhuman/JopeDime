# H370 — NL15 ranker feature fraction0.6

**FF06_NOT_ADVANCED**. All36 fold/root tasks and9 partition/root panels completed on TRAIN only.

Hypothesis: stronger column subsampling in NL15 rankers may add useful diversity. ONE preregistered FF06 arm changes ranker colsample_bytree0.8->0.6 for inner/final/all8 bag rankers. num_leaves15/max_depth3/MCS20/L2=1/L1=0, LR/tree options, labels/seeds/groups/grid/ties/failure sentinel and veto GBDT stay fixed. Depth3 permits at most8 realized binary leaves. Reverting the one existing keyword value restores exact committed H352 NL15 ASTs in both ranker functions.

Preregistration commit `7c991126779b6cb1fee21273c6fb0e448b5a5c17`, published 2026-11-05T14:17:39.226165+00:00; first treatment fit 2026-11-05T14:18:32.162728+00:00. Full specification in [PREREG.md](PREREG.md). Exact-version archived BASE9panels/36hashes/3medians and NL15ninepanels/36locks/scores/rawarrays passed before fitting; no BASE/NL15 refits or new archived-model predictions.

PRIMARY FF06 minus NL15 paired drop-top10: -0.005557, whole-family CI95 [-0.014907, +0.000737], one-sided LCB95 -0.013257. Advance requires at least+0.002 and positive LCB95 plus gain-retention/harm/root gates; ties and subthreshold effects never advance.
Secondary FF06 minus BASE paired drop-top10: +0.001748, CI95 [-0.005157, +0.008620], LCB95 -0.004046.

| Median over nine cells | BASE | NL15 | FF06 |
|---|---:|---:|---:|
| Family-equal mean gain | +0.048215 | +0.058405 | +0.051374 |
| Literal family-median gain | +0.003624 | +0.007409 | +0.004151 |
| Mean-gain whole-family CI95 | [+0.024800, +0.075507] | [+0.033824, +0.092477] | [+0.025808, +0.086610] |
| Harmed share (gain<−0.01) | 0.092683 | 0.092683 | 0.107317 |
| Elo | 1725.66 | 1734.47 | 1716.03 |

**Elo reference:** reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN. Elo is not AUC.

| Root seed | Paired drop10 median | CI95 | LCB95 |
|---|---:|---|---:|
| 30001 | -0.009264 | [-0.019952, +0.001132] | -0.017817 |
| 30002 | -0.001769 | [-0.010773, +0.011578] | -0.008892 |
| 30003 | -0.018368 | [-0.038587, +0.000943] | -0.034407 |

Configuration agreement FF06/NL15: 6/36 (CONFIG_SENSITIVE). Full27cell mean/median and Elo CIs, drop-top1/5/10, LOFO minimum, W/T/L, harm, invalids, switches and seed Jaccard are in [metrics.json](metrics.json). Whole-family bootstrap10000/RNG370191 shares family draws across all9 cells and both arms; each arm drops its own top10. Per-root confidence intervals are reported independently of the aggregate advancement gate.

Decision gates: paired_drop_top10_at_least_002=False, gain_retention=False, harmed_nonincrease=False, positive_on_two_seeds=False.

All36 fsynced fit locks, native/constructor guards,108 identical final veto SHAs and input closures passed before once-only outer scoring. Native fits: {'LGBMClassifier': 7020, 'LGBMRanker': 29484}. Every ranker native delta vs NL15 is feature_fraction[0.8,0.6], every veto native delta empty. Ranker extra_trees0, bagging_by_query0, bagging_fraction0.8 and all other native defaults remain pinned.

Resources: xbabe2 CPU only, admitted cores [57, 60, 61, 62, 63], nice19/math1/systemd MemoryMax6G/SwapMax0/CPUQuota800%. One5second admission gate; initial and first-task forecasts use H352 actual26189.516914CPU/.9eff/1.15safety/+300s reserve. Actual cgroupCPU 30413.907269s; no speed verdict. Corrected guard uses hostCPU>85% over60s, load1>108 on3 consecutive60s samples, MemAvailable<20GiB or ownscratch>2GB. Fixed compute stop16:45Z/close17:00Z Nov5 unchanged.
Verification: 51 data-free controls and11 custody controls pass. Same-agent red-team binds source/prereg chronology, all native parameters, physical input ranges, root CIs, tie bar and both regressions. Zero fit-worker outer payload reads before locks; zero DL-Val/dev91/DL-Test/sealed/holdout/meta_test payload rows. DOPEc51a56d0 unloaded/untouched; no neural optimizer updates, GPU, qualification.lock, writer restart or foreign-job action.
Custody: create-only s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H370/api01-20261105T140000Z/closeout/closeout-manifest.json, exact VersionId `OH.cVz.5_9S71c8xlulDVcqT4.fvM6Eh`, SHA256 `674daf3647135e403a4f4006acade6e5993333f88fce254b19976bda41472871`; 2846 files/27 parts, exact-version GET plus every member verified before own scratch cleanup. No models in Git. Evidence and executed source/vendor retained at `/home/ubuntu/jope_plan/evidence/H370/api01-20261105-t1400`.

Multiplicity: exploratory sequential H348-H370 sweep on the same folds/roots, without correction. Any ADVANCE needs separately authorized one-shot dev91 confirmation; none is performed here. Next: retain confirmed NL15; do not adopt FF06 from this study.
