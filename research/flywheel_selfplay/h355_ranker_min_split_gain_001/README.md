# H355: ranker min_split_gain 0.01

**COMPLETE — MSG01_NOT_ADVANCED.** All 36 fit tasks and once-only outer-score markers completed.

## Hypothesis and setup

A minimum split-gain floor may prune low-value splits that follow family noise. The sole intervention changes effective ranker min_split_gain from 0 to 0.01 in pinned topk.fit_ranker and bagged.fit_ranker_n, including every inner/final/eight-bag fit. The original constructors omit that keyword and default to zero; removing the added min_split_gain=0.01 restores both complete BASE function ASTs exactly. Explicit zero and implicit-zero constructors match. H351 min_child_samples is restored from 40 to BASE 20; reg_lambda remains BASE 1 and reg_alpha implicit BASE 0. LambdaRank/ndcg, sigmoid 1, labels 0..7/default label_gain, Hessian 0.001, leaves 7/depth 3, sampling, LR/tree options, seeds/features, original 18-config grid, 1e-4 ties/switches and sentinel -2 remain unchanged.

Veto binary GBDT retains min_child_samples 25, reg_alpha 1, reg_lambda 2. All 108/108 final veto SHAs matched frozen BASE. All 972 archived BASE rankers verified native min_gain_to_split 0 / min_data_in_leaf 20 / L1 zero / L2 one; no BASE refits, calibration fits or fresh BASE model predictions.

Copied committed H351 source at `bf3414cba7918bea721a8b64c04211bc0d687243`, path `research/flywheel_selfplay/h351_ranker_min_child_samples_40/`. [PREREG.md](PREREG.md) commit `3dc9f92317c0de55f5fefaae1f5d82e5107afe2a` was published 2026-11-05T09:24:13.578336+00:00, before first actual fit 2026-11-05T09:25:23.953294+00:00. Source/manifest/prereg/vendor pins remain exact.

H303 NESTED/H308 TRAIN-only design: P0/P1/P2 × roots 30001–30003 × 4 family-grouped outer folds = 36 tasks / 9 cells, 1219 contexts / 205 families. BASE first exact-version GET/whole+member SHA, 9 panels, 36 fold-root pick+lock hashes and exact medians 0.0643163334285782 / 0.04821480030677507 / 0.04360142642936269 passed. Current TRAIN pools, selected programs and r882 costs and four consumed cache prefixes matched before and after the run.

## Results

Gains below are medians across nine evaluation cells; intervals are whole-family bootstrap 95% CIs.

| Arm | Family-equal mean gain [95% CI] | Literal family median [95% CI] | Harmed share | Median Elo |
|---|---|---|---|---|
| BASE | +0.048215 [+0.025157, +0.076452] | +0.003624 [+0.000299, +0.007726] | 9.27% | 1725.66 |
| MSG01 | +0.056049 [+0.023872, +0.094483] | +0.005371 [+0.001190, +0.008580] | 11.22% | 1714.65 |

`reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN`

Paired drop-top-10 median **+0.000541**, 95% CI [-0.005069, +0.004449], one-sided LCB95 **-0.004177**. Gain retention 116.25%; positive root medians 2/3; configuration agreement 8/36 (CONFIG_SENSITIVE).

| Partition | Root | BASE gain | MSG01 gain | Paired drop10 | BASE harmed | MSG01 harmed |
|---|---|---|---|---|---|---|
| P0 | 30001 | +0.041289 | +0.032027 | -0.000236 | 0.063415 | 0.039024 |
| P0 | 30002 | +0.064316 | +0.059229 | +0.002074 | 0.087805 | 0.102439 |
| P0 | 30003 | +0.072258 | +0.056049 | -0.004899 | 0.092683 | 0.102439 |
| P1 | 30001 | +0.048215 | +0.044752 | +0.003975 | 0.112195 | 0.068293 |
| P1 | 30002 | +0.048658 | +0.032821 | -0.011054 | 0.102439 | 0.117073 |
| P1 | 30003 | +0.038294 | +0.059672 | +0.002934 | 0.082927 | 0.126829 |
| P2 | 30001 | +0.043601 | +0.026583 | -0.011799 | 0.107317 | 0.136585 |
| P2 | 30002 | +0.031840 | +0.059775 | +0.003711 | 0.078049 | 0.112195 |
| P2 | 30003 | +0.054721 | +0.064999 | +0.000541 | 0.107317 | 0.121951 |

Elo by evaluation cell (95% whole-family bootstrap CIs):

| Partition | Root | BASE Elo [95% CI] | MSG01 Elo [95% CI] |
|---|---|---|---|
| P0 | 30001 | 1725.66 [1652.45, 1812.97] | 1786.31 [1700.58, 1901.96] |
| P0 | 30002 | 1722.48 [1656.71, 1799.21] | 1714.65 [1653.66, 1786.92] |
| P0 | 30003 | 1727.10 [1665.65, 1798.70] | 1706.20 [1644.99, 1778.19] |
| P1 | 30001 | 1697.13 [1636.28, 1768.50] | 1786.02 [1713.53, 1881.57] |
| P1 | 30002 | 1693.84 [1633.76, 1761.95] | 1687.82 [1628.10, 1756.17] |
| P1 | 30003 | 1734.68 [1668.06, 1812.70] | 1668.51 [1612.46, 1730.57] |
| P2 | 30001 | 1716.88 [1656.11, 1787.68] | 1692.30 [1633.76, 1761.24] |
| P2 | 30002 | 1740.78 [1674.77, 1820.09] | 1719.13 [1655.45, 1792.28] |
| P2 | 30003 | 1726.57 [1665.97, 1796.09] | 1717.59 [1658.41, 1785.63] |

| Root | Paired drop10 median | 95% CI | One-sided LCB95 |
|---|---|---|---|
| 30001 | -0.000236 | [-0.009173, +0.007256] | -0.007510 |
| 30002 | +0.002074 | [-0.008234, +0.008683] | -0.006505 |
| 30003 | +0.000541 | [-0.008491, +0.008166] | -0.007203 |

Gates: `{"gain_retention": true, "harmed_nonincrease": false, "paired_drop_top10_at_least_002": false, "positive_on_two_seeds": true}`.

ADVANCE requires paired drop-top-10 median >= +0.002 AND LCB95 > 0, median full gain >= 90% of BASE, median harmed share <= BASE and positive paired root medians on >= 2/3 roots. Ties/subthreshold never advance. Paired 10k whole-family RNG355191 draws are shared across both arms / all 9 cells; each arm redrops its own top 10 and the statistic is median of 9 cells. Per-root intervals use the same draws and median of 3 partitions for that root; CI95 quantiles 0.025/0.975 and LCB95 quantile 0.05. Root intervals add no decision gate. Ordinary intervals use RNG302150/10k unchanged. [metrics.json](metrics.json) retains every scientific field: all per-cell intervals/drop1/5/10/LOFO/W-T-L/harm gain<-0.01/invalids/switches/Elo+CI/Jaccard/configurations/lock hashes. No AUC or speed verdict.

## Verification and resources

All 29,484 ranker + 7,020 veto native fits and 12,636 saved ranker + 1,404 veto bindings passed exact constructor/native/integer/seed checks. Physical outer rows/packets before fsynced lock and parent fsync were 0. All 36 locks/guards/native audits and 108 veto SHA matches passed before cold scoring; all 36 once-only markers COMPLETE. Every inner archive passed exact-version GET/whole-file and every-member SHA before raw-inner cleanup.

All 43 data-free controls passed, 0 failed/skipped. Actual campaign exited 0. Independent saved-family arithmetic and paired/global/root bootstrap replay passed with no new fits, model predictions or outer feature reads. These are same-agent reviews. Read-only preparation exited 0; BASE and current TRAIN identities remained exact. Hosted CI NOT_RUN under CI_FREEZE+BATCH_WINDOW.

One 5-second/core+SMT >= 85%-idle sample admitted [15, 16, 17, 18, 19, 20, 21, 22]; first at most 8 passing pairs / minimum 4; same-interval host CPU 12.755981% <= 70. Controller threads/all workers bound before fits. Initial H347 29770.57 CPU / (workers × 0.9) × 1.15 + 300s reserve forecast 2026-11-05T10:49:21.940436+00:00; first-task conservative remaining-35 forecast 2026-11-05T10:58:34.657306+00:00. Both passed fixed Nov5 11:45Z compute / 12:00Z close. No reprobe/borrowed cores/expansion/restart/cut cells/partial scoring/window reset.

Actual campaign Thu 2026-11-05 09:25:00 UTC to Thu 2026-11-05 10:27:12 UTC: nice 19, numeric threads 1, 8 workers, MemoryMax 6 GiB / SwapMax 0. Peak memory 4.138 GiB, swap 0 bytes, scratch 1.525 GB. Retained preparation/run samples: host CPU max 38.55%, load max 56.03, MemAvailable min 72.02GiB. Timing is resource accounting only.

## Custody and next step

Retain frozen BASE. MSG01 failed at least one locked gate; no dev91 evaluation is triggered. Any further intervention needs a new owner assignment and preregistration.

Create-only archive `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H355/gov03-closeout-20261105T102734Z/h355-scratch.tar`; VersionId `uCRH9P4g0F56SZo3uNRa4o7NXwOcLmIc`; SHA256 `66430d2c3abac7157b17b011858b1a4c4c7cfc478891c71388594b0786993503`. Exact-version GET/whole SHA/all 2,745 file members and all 36 inner gzip CRCs passed before own scratch (1,414,919,439 bytes) and generated-restore cleanup. Fresh consumer check: 0. Receipt VersionId `E3PxwEUDdbvPbuitog.flXcXwaw6.4IZ`. Immutable local archive/source/evidence retained at `/home/ubuntu/jope_plan/evidence/H355/gov03-20261105-t0915`.

DOPE c51a56d0 unchanged/unloaded, CPU selection layer only. No DL-Val/dev91/DL-Test/protected rows, GPU/qualification.lock, live writer, foreign job or forbidden host actions. No xbabe1 access; H335/H354/H352/H353 lanes/files/units were left alone. Private index/nonforcev1next only; canonical HEAD/current foreign index/staged/dirty files preserved. No models in Git or PR/main/fastlane/hostedCI operations. H351/H348 and earlier closed waves stay closed.
