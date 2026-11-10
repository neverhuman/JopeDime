# H342: ranker minimum summed Hessian

**COMPLETE — MINHESS_NOT_ADVANCED.** All 36 fit tasks and once-only outer score markers completed. Retain frozen BASE; this result triggers no dev91 evaluation.

## Hypothesis and setup

A curvature-based leaf floor may reduce dependence on small, low-confidence pair pockets. The single treatment sets `min_child_weight=9.87`, the sklearn alias for native `min_sum_hessian_in_leaf`, on every inner, final and eight-bag ranker. All 972 archived BASE rankers confirm an effective floor of 0.001. Removing this override restores both complete pinned H287 constructor-function ASTs exactly.

LambdaRank, default label gains, labels, groups, features, seeds, GBDT settings, `min_child_samples=20`, regularization, rounds, the original 18-config grid, ties, switches and the −2.0 sentinel retain their original definitions. Veto classifiers retain binary GBDT and Hessian floor 0.001.

The owner fixed the value rule before treatment: take the 25th percentile of every archived BASE ranker leaf’s `leaf_weight`, then round to three significant figures. Read-only `dump_model` yielded **988,244 leaves from 972 models** (108 main rankers and 864 bag rankers), each verified against its exact-version archive member. The percentile was **9.865141101181505**, giving **9.87**. Derivation pools archived TRAIN models across all folds. It required zero calibration fits or model predictions. [minhess_value.json](minhess_value.json) records the method and hashes.

Preregistration: `d65a79f737f5f9b2985a72ea37adb6f95715cfc8` ([PREREG.md](PREREG.md)), published at 2026-11-04T23:38:03.641830+00:00. The first actual fit began at 2026-11-04T23:38:48.167868+00:00.

H303 NESTED/H308 design: P0/P1/P2 × roots 30001–30003 × four family-grouped outer folds, giving 36 fit tasks and nine evaluation cells across 1,219 TRAIN contexts and 205 families. Frozen BASE matched all nine panels, 36 fold/root pick and lock hashes, and the three medians 0.0643163334285782 / 0.04821480030677507 / 0.04360142642936269 exactly. Current TRAIN pools, selected/r882 costs and four consumed cache prefixes matched archived input hashes before and after the campaign. BASE required no refits or new model predictions.

## Results

Aggregate gains are medians across the nine cells; brackets show whole-family bootstrap 95% confidence intervals.

| Arm | Family-equal mean gain [95% CI] | Literal family median [95% CI] | Harmed share | Median Elo |
|---|---|---|---|---|
| BASE | +0.048215 [+0.025157, +0.076452] | +0.003624 [+0.000299, +0.007726] | 9.27% | 1725.66 |
| MINHESS | +0.053871 [+0.023470, +0.094150] | +0.006326 [+0.002802, +0.010838] | 12.20% | 1707.78 |

`reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN`

Paired drop-top-10 median: **+0.000602**, 95% CI [-0.009353, +0.006607], one-sided LCB95 **-0.007940**. Gain retention is 111.73%; positive root medians are 1/3; configuration agreement is 5/36 (CONFIG_SENSITIVE).

| Partition | Root | BASE gain | MINHESS gain | Paired drop10 | BASE harmed | MINHESS harmed |
|---|---|---|---|---|---|---|
| P0 | 30001 | +0.041289 | +0.060001 | +0.001351 | 0.063415 | 0.102439 |
| P0 | 30002 | +0.064316 | +0.043067 | -0.005175 | 0.087805 | 0.141463 |
| P0 | 30003 | +0.072258 | +0.057466 | -0.014891 | 0.092683 | 0.117073 |
| P1 | 30001 | +0.048215 | +0.059830 | +0.013567 | 0.112195 | 0.082927 |
| P1 | 30002 | +0.048658 | +0.049215 | +0.000602 | 0.102439 | 0.126829 |
| P1 | 30003 | +0.038294 | +0.061076 | +0.006723 | 0.082927 | 0.131707 |
| P2 | 30001 | +0.043601 | +0.053337 | +0.010321 | 0.107317 | 0.121951 |
| P2 | 30002 | +0.031840 | +0.032557 | -0.014910 | 0.078049 | 0.156098 |
| P2 | 30003 | +0.054721 | +0.053871 | -0.020809 | 0.107317 | 0.121951 |

Per-root paired medians for 30001 / 30002 / 30003: +0.010321 / -0.005175 / -0.014891. Gates: `{"gain_retention": true, "harmed_nonincrease": false, "paired_drop_top10_at_least_002": false, "positive_on_two_seeds": false}`.

ADVANCE requires paired drop10 median ≥ +0.002 and LCB95 > 0, median gain ≥ 90% of BASE, median harmed share ≤ BASE, and positive paired root medians on at least two of three roots. MINHESS fails the paired, harm and root gates.

The paired bootstrap uses 10,000 whole-family draws, RNG 342191, one shared family sample for both arms and all nine cells, each arm dropping its own top ten, and the median across nine cells as the statistic; LCB95 is quantile 0.05. Ordinary gain, literal-median and Elo intervals use 10,000 draws and RNG 302150. [metrics.json](metrics.json) retains every scientific value, including per-cell intervals, drop1/5/10, LOFO, W/T/L, harm below −0.01, invalids, switches, Elo intervals, seed Jaccard, configurations and lock hashes.

## Verification and resources

During treatment fitting, physical outer feature rows and packets read before the fsynced lock and parent-directory fsync were zero. All 36 locks, guards, input identities, models and native audits passed before cold outer scoring. All 36 score markers are COMPLETE. Audits checked every native-mappable constructor argument after 29,484 ranker and 7,020 veto fits, and bound 12,636 saved rankers and 1,404 veto models to their member-native records. Each inner archive passed exact-version GET, whole SHA and every-member checks before raw-inner removal.

All 37 data-free controls passed, with zero failures or skips. The campaign exited 0. Separate arithmetic replay reproduced the family metrics and paired bootstrap without fitting or predicting again. These are same-agent reviews. Source, manifest and preregistration hashes remain exact. Hosted CI was NOT_RUN under CI_FREEZE + MASTER_MERGE_HOLD.

One five-second per-core/SMT ≥85%-idle gate admitted cores **4, 7, 12, 14**. The H334 CPU reference forecast ended at 2026-11-05T02:41:03.295611+00:00; the required first-completed-task forecast ended at 2026-11-05T02:48:11.134111+00:00. The second forecast used 35 remaining tasks, zero credit for ongoing work, and per-task CPU = max(H334 CPU/36, first task actual CPU). Both passed the fixed November 5 03:15Z compute stop. Close was fixed at 03:30Z. The single gate and both deadlines were preserved.

Actual campaign: Wed 2026-11-04 23:38:25 UTC to Thu 2026-11-05 01:27:32 UTC, nice 19, one numeric thread, four workers, MemoryMax 6 GiB and SwapMax 0. Peak memory was 1.922 GiB, swap 0 bytes and scratch 1.440 GB. Sampled host CPU peaked at 37.82%, load at 60.39, and available memory reached a minimum of 70.16 GiB. Timing supports resource accounting and admission.

The admission controller initially retained a ten-candidate affinity mask. Fit processes used the four admitted cores from launch. At 23:44:19Z both controller threads were restricted to those same four; own-cgroup readback confirmed every process on 4/7/12/14. The correction preserved scientific source, value, fit mask and workers, with no restart or new capacity sample. Receipts are retained in metrics and lifecycle evidence.

## Custody and next step

Create-only archive: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H342/gov03-closeout-20261105T012844Z/h342-scratch.tar`. VersionId `XfkKTEUFbJBUJuxoSXrO5BYheQ7ENg9U`; SHA256 `151624223c8e55e2f014cf4709344297124240516ec174f6d08d1edd7fb4d273`. Exact-version GET, whole SHA and all **2,601 file members** passed verification before deleting 1,337,807,629 bytes of own scratch. Fresh consumer checks found zero references. The generated restore copy was reclaimed; the immutable local archive and source/evidence remain at `/home/ubuntu/jope_plan/evidence/H342/gov03-20261104-t2330`. Archive-receipt VersionId: `qyiomKhBvVYuu4v3ZjDBTFZoTxUy7jLJ`.

Retain BASE. Any future intervention requires a separate owner assignment and preregistration. H342 performs no dev91 evaluation or serving promotion.

DOPE c51a56d0 remains unchanged and unloaded; scope is CPU selection only. No DL-Val/dev91/DL-Test or protected rows, GPU/qualification.lock, live writer, foreign jobs or forbidden hosts were used. Publication uses a private index and nonforce v1/next; there are no model files in Git or PR/main/fastlane/CI operations. H340/H334/H333/H329/H326 remain closed.
