# H351: ranker min_child_samples 40

**COMPLETE — MCS40_NOT_ADVANCED.** All 36 fit tasks and once-only outer-score markers completed.

## Hypothesis and setup

A larger leaf-count floor may reduce noisy small leaves and family swings. The sole intervention changes the existing ranker `min_child_samples` from 20 to 40 in pinned topk.fit_ranker and bagged.fit_ranker_n, including every inner/final/eight-bag fit. It sets native `min_data_in_leaf=40`; it does not establish a distinct-context or family-count guarantee. Restoring 20 reproduces both complete BASE function ASTs exactly with no added keywords. Ranker reg_lambda is restored from H350 to BASE 1 and reg_alpha remains implicit BASE 0. LambdaRank/ndcg, sigmoid 1, labels 0..7/default label_gain, Hessian 0.001, leaves 7/depth 3, row/column sampling, LR/tree options, seeds/features, original 18-config grid, 1e-4 ties/switches and sentinel -2 remain unchanged.

Veto binary GBDT retains min_child_samples 25, reg_alpha 1, reg_lambda 2. All 108/108 final veto SHAs matched frozen BASE. All 972 archived BASE rankers verified native min_data_in_leaf 20 / L1 zero / L2 one; no BASE refits, calibration fits or fresh BASE model predictions.

Copied committed H350 source at `15b4b86fe7748f1dc2875302d1719850aea677c2`, path `research/flywheel_selfplay/h350_ranker_reg_lambda_5/`. [PREREG.md](PREREG.md) commit `0f818a0361953b4e671c9de857eb82712b6bef01` was published 2026-11-05T07:37:43.200628+00:00, before first actual fit 2026-11-05T07:38:09.371009+00:00. Source/manifest/prereg/vendor pins remain exact.

H303 NESTED/H308 TRAIN-only design: P0/P1/P2 × roots 30001–30003 × 4 family-grouped outer folds = 36 tasks / 9 cells, 1219 contexts / 205 families. BASE first exact-version GET/whole+member SHA, 9 panels, 36 fold-root pick+lock hashes and exact medians 0.0643163334285782 / 0.04821480030677507 / 0.04360142642936269 passed. Current TRAIN pools, selected programs and r882 costs and four consumed cache prefixes matched before and after the run.

## Results

Gains below are medians across nine evaluation cells; intervals are whole-family bootstrap 95% CIs.

| Arm | Family-equal mean gain [95% CI] | Literal family median [95% CI] | Harmed share | Median Elo |
|---|---|---|---|---|
| BASE | +0.048215 [+0.025157, +0.076452] | +0.003624 [+0.000299, +0.007726] | 9.27% | 1725.66 |
| MCS40 | +0.047988 [+0.025479, +0.083979] | +0.007684 [+0.002637, +0.011287] | 11.22% | 1722.49 |

`reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN`

Paired drop-top-10 median **+0.000395**, 95% CI [-0.004536, +0.009060], one-sided LCB95 **-0.003311**. Gain retention 99.53%; positive root medians 2/3; configuration agreement 10/36 (CONFIG_SENSITIVE).

| Partition | Root | BASE gain | MCS40 gain | Paired drop10 | BASE harmed | MCS40 harmed |
|---|---|---|---|---|---|---|
| P0 | 30001 | +0.041289 | +0.041578 | -0.001391 | 0.063415 | 0.087805 |
| P0 | 30002 | +0.064316 | +0.047988 | -0.001856 | 0.087805 | 0.136585 |
| P0 | 30003 | +0.072258 | +0.066694 | +0.006173 | 0.092683 | 0.112195 |
| P1 | 30001 | +0.048215 | +0.063305 | +0.011395 | 0.112195 | 0.117073 |
| P1 | 30002 | +0.048658 | +0.038305 | -0.003191 | 0.102439 | 0.078049 |
| P1 | 30003 | +0.038294 | +0.043410 | +0.000395 | 0.082927 | 0.087805 |
| P2 | 30001 | +0.043601 | +0.038747 | +0.002612 | 0.107317 | 0.112195 |
| P2 | 30002 | +0.031840 | +0.055765 | +0.010602 | 0.078049 | 0.117073 |
| P2 | 30003 | +0.054721 | +0.058885 | -0.001688 | 0.107317 | 0.097561 |

| Root | Paired drop10 median | 95% CI | One-sided LCB95 |
|---|---|---|---|
| 30001 | +0.002612 | [-0.011376, +0.017801] | -0.008740 |
| 30002 | -0.001856 | [-0.010674, +0.009181] | -0.008739 |
| 30003 | +0.000395 | [-0.012133, +0.012198] | -0.009603 |

Gates: `{"gain_retention": true, "harmed_nonincrease": false, "paired_drop_top10_at_least_002": false, "positive_on_two_seeds": true}`.

ADVANCE requires paired drop-top-10 median >= +0.002 AND LCB95 > 0, median full gain >= 90% of BASE, median harmed share <= BASE and positive paired root medians on >= 2/3 roots. Ties/subthreshold never advance. Paired 10k whole-family RNG351191 draws are shared across both arms / all 9 cells; each arm redrops its own top 10 and the statistic is median of 9 cells. Per-root intervals use the same draws and median of 3 partitions for that root; CI95 quantiles 0.025/0.975 and LCB95 quantile 0.05. Root intervals add no decision gate. Ordinary intervals use RNG302150/10k unchanged. [metrics.json](metrics.json) retains every scientific field: all per-cell intervals/drop1/5/10/LOFO/W-T-L/harm gain<-0.01/invalids/switches/Elo+CI/Jaccard/configurations/lock hashes. No AUC or speed verdict.

## Verification and resources

All 29,484 ranker + 7,020 veto native fits and 12,636 saved ranker + 1,404 veto bindings passed exact constructor/native/integer/seed checks. Physical outer rows/packets before fsynced lock and parent fsync were 0. All 36 locks/guards/native audits and 108 veto SHA matches passed before cold scoring; all 36 once-only markers COMPLETE. Every inner archive passed exact-version GET/whole-file and every-member SHA before raw-inner cleanup.

All 42 data-free controls passed, 0 failed/skipped. Actual campaign exited 0. Independent saved-family arithmetic and paired/global/root bootstrap replay passed with no new fits, model predictions or outer feature reads. These are same-agent reviews. Read-only preparation exited 0; BASE and current TRAIN identities remained exact. Hosted CI NOT_RUN under CI_FREEZE+BATCH_WINDOW.

One 5-second/core+SMT >= 85%-idle sample admitted [15, 16, 18, 24, 25, 26]; first at most 8 passing pairs / minimum 4; same-interval host CPU 31.995334% <= 70. Controller threads/all workers bound before fits. Initial H347 29770.57 CPU / (workers × 0.9) × 1.15 + 300s reserve forecast 2026-11-05T09:28:30.510249+00:00; first-task conservative remaining-35 forecast 2026-11-05T09:38:06.542685+00:00. Both passed fixed Nov5 10:15Z compute / 10:30Z close. No reprobe/borrowed cores/expansion/restart/cut cells/partial scoring/window reset.

Actual campaign Thu 2026-11-05 07:37:44 UTC to Thu 2026-11-05 08:55:38 UTC: nice 19, numeric threads 1, 6 workers, MemoryMax 6 GiB / SwapMax 0. Peak memory 3.153 GiB, swap 0 bytes, scratch 1.523 GB. Retained preparation/run samples: host CPU max 57.99%, load max 65.20, MemAvailable min 68.14 GiB. Timing is resource accounting only.

## Custody and next step

Retain frozen BASE. MCS40 failed at least one locked gate; no dev91 evaluation is triggered. Any further intervention needs a new owner assignment and preregistration.

Create-only archive `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H351/gov03-closeout-20261105T085638Z/h351-scratch.tar`; VersionId `DVuYLTXrOCn.IDX0ca7ASwEqt_Z2dZen`; SHA256 `c772002da6c1ad5a103b8fd6cf5e764a6e9f9b36ac657da111708f7cf110cab3`. Exact-version GET/whole SHA/all 2,776 file members and all 36 inner gzip CRCs passed before own scratch (1,396,231,743 bytes) and generated-restore cleanup. Fresh consumer check: 0. Receipt VersionId `mVdRVA4_jcOJZB9JA3tEZ0mpE.sibiyG`. Immutable local archive/source/evidence retained at `/home/ubuntu/jope_plan/evidence/H351/gov03-20261105-t0730`.

DOPE c51a56d0 unchanged/unloaded, CPU selection layer only. No DL-Val/dev91/DL-Test/protected rows, GPU/qualification.lock, live writer, foreign job or forbidden host actions. No xbabe1 access; H335/H350R lanes/files/units were left alone. Private index/nonforcev1next only; canonical HEAD/current foreign index/staged/dirty files preserved. No models in Git or PR/main/fastlane/hostedCI operations. H348/H347 and earlier closed waves stay closed.
