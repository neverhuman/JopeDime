# H380: NL15 ranker bagging_freq = 5

**COMPLETE — BFQ5_NOT_ADVANCED.** The primary comparison is BFQ5 against frozen H352 NL15; the secondary comparison is against original BASE.

## Hypothesis and setup

The hypothesis is that reusing each subsampled bag for five rounds reduces bag-churn variance and improves NL15 selection. BFQ5 changes only ranker `bagging_freq` from 1 to 5 through the existing `subsample_freq` value, for inner fits, final refits and all eight bags. Restoring that value to 1 restores the exact H352 function ASTs; no keyword or seed keyword is added. The veto classifier is unchanged; all 108 final veto hashes match BASE and NL15.

The H352 configuration remains: 15 leaves, depth 3, minimum child count 20, L1 = 0, L2 = 1, minimum Hessian 0.001 and sigmoid 1. Learning rates (0.03, 0.05, 0.10), rounds (50, 100, 200), main rounds 200, bagging_fraction 0.8, feature_fraction 0.8, path_smooth 0, and the original 18-configuration grid, sampling, labels, groups, features, seed expressions, prediction and lock logic, 1e-4 ties and sentinel -2 are unchanged. There is no calibration, BASE/NL15 refit or new baseline model prediction.

[PREREG.md](PREREG.md) and the 35 frozen source files were published at `b16b7354e87e394784ec17ace5894b5a766aad54` on 2026-11-05T16:40:59.908941+00:00, before the first treatment fit on 2026-11-05T16:42:01.030849+00:00. This copies frozen H374 transport `e8a8f67263a58cbb2dd0e120ccb42d9ee2bfc356` and manifest `a5b18c71a2b3b2aced904a786fc4fd402e12115505c6336a3a0c152e4b10fea5`, adapting wave identities/paths, RNG 380191, window and the sole BFQ5 intervention. H374 path_smooth = 20 is fully removed; H352 path_smooth = 0 is restored. H371 and H374 remain closed; H380 is a separate owner allocation.

TRAIN only: 1,219 contexts from 205 families; P0/P1/P2 × roots 30001/30002/30003 × four family-grouped outer folds give 36 tasks and nine evaluation cells. Fresh exact-Version GETs, whole-object and every-member hashes verified BASE 9 panels / 36 pick-lock pairs / exact medians 0.0643163334285782, 0.04821480030677507, 0.04360142642936269. Fresh frozen NL15 verification covered 27 parts / 2,691 members, nine panels, 36 locks and COMPLETE score markers, raw family arrays, 972 native rankers and 108 veto identities. Explicit native bagging_freq = 1, path_smooth = 0 and bagging_fraction = 0.8 were verified before treatment.

## Outcome

Point summaries below are medians across nine evaluation cells. Intervals use whole-family resampling.

| Arm | Family-equal mean gain [95% CI] | Literal family median [95% CI] | Harmed share | Median Elo |
|---|---|---|---|---|
| BASE | +0.048215 [+0.025506, +0.076282] | +0.003624 [+0.000299, +0.007509] | 9.27% | 1725.66 |
| NL15 | +0.058405 [+0.034449, +0.092471] | +0.007409 [+0.002351, +0.011715] | 9.27% | 1734.47 |
| BFQ5 | +0.052369 [+0.024432, +0.095400] | +0.006981 [+0.002620, +0.010888] | 9.76% | 1710.06 |

`reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN`

PRIMARY paired drop-top-10, BFQ5 − NL15: **-0.008467**, 95% CI [-0.020867, +0.001974], one-sided LCB95 **-0.018310**. SECONDARY, BFQ5 − BASE: -0.000102, 95% CI [-0.006078, +0.007156], LCB95 -0.004817. Full-gain retention versus NL15 is 89.67%; positive primary root medians: 0/3. Configuration agreement is 15/36 against NL15 and 8/36 against BASE.

| Comparator | Root | Paired drop-top-10 median | 95% CI | One-sided LCB95 |
|---|---|---|---|---|
| NL15 | 30001 | -0.008467 | [-0.025079, +0.000830] | -0.022651 |
| NL15 | 30002 | -0.005980 | [-0.019601, +0.007477] | -0.016461 |
| NL15 | 30003 | -0.008704 | [-0.027508, +0.006723] | -0.023232 |
| BASE | 30001 | -0.002774 | [-0.014074, +0.007630] | -0.012207 |
| BASE | 30002 | +0.004416 | [-0.004221, +0.013116] | -0.002912 |
| BASE | 30003 | +0.002271 | [-0.009282, +0.011947] | -0.007440 |

Gates: `{"gain_retention": false, "harmed_nonincrease": false, "paired_drop_top10_at_least_002": false, "positive_on_two_seeds": false}`.

ADVANCE requires primary paired drop-top-10 median ≥ +0.002 AND LCB95 > 0; full gain ≥ 90% of NL15; harmed share no greater than NL15; and positive primary point medians on at least two of three roots. Ties never advance. Root intervals add no gate. All new ordinary and paired intervals use 10,000 shared whole-family draws, RNG 380191, across all three arms and nine cells. Each arm drops its own top ten after every resample. The global statistic is the median of nine cells; each root uses the median of its three partitions and the same draws. CI95 uses quantiles 0.025/0.975, LCB95 quantile 0.05.

[metrics.json](metrics.json) retains all 27 cell means and literal medians, intervals, drop-top-1/5/10, leave-one-family-out, wins/ties/losses, harmed share below −0.01, invalids, switches, Elo intervals, Jaccard and chosen configurations, plus both paired comparisons and root intervals. Full regression and lock inventories are hash-bound to the original raw metrics in verified custody. This is part of the sequential exploratory H348–H380 sweep, without a multiplicity-adjusted confirmatory claim. Elo is not AUC; there is no speed verdict.

## Verification and resources

All 36 fsynced locks, parent fsyncs, native/physical guards and veto checks preceded the fresh cold outer-score process. All 36 once-only score markers are COMPLETE; physical outer rows and packets read before lock are zero. All 36,504 native fits (29,484 ranker / 7,020 veto) and 14,040 saved bindings (12,636 ranker / 1,404 veto) passed constructor, native integer/seed and observed-leaf checks. Every native fit header was compared to frozen H352 at the same unchanged grid choice; only ranker bagging_freq 1→5 differs.

The 48 preregistered data-free controls passed, with zero failures or skips. Two separate synthetic terminal-verifier controls passed, including rejection of a corrupted secondary root interval. Science exited 0. Independent saved-family arithmetic verified every ordinary and paired global/root interval without new fits, model predictions or outer feature reads. TRAIN input identities matched before and after. Reviews are by the same agent. Hosted CI was NOT_RUN under CI_FREEZE+BATCH_WINDOW.

One five-second both-SMT ≥85%-idle admission gate over the authorized union admitted cores [15, 16, 18, 20, 21, 23, 24, 25], the first at most eight passing pairs, with at least four required. Same-interval host CPU was 20.863275% (limit 70%). Initial forecast: 2026-11-05T17:56:26.301306+00:00; mandatory first-task / remaining-35 forecast: 2026-11-05T18:13:21.178556+00:00. Both passed the fixed November 5 19:15Z compute stop; close deadline 19:30Z. The reference was actual H352 CPU 26189.516914 seconds, efficiency 0.9, safety 1.15, reserve 300 seconds, and no ongoing-work credit at the first-task forecast. There was no second probe, borrowed core pool, restart, deadline reset, cut cell or partial scoring.

nice 19; math threads 1; CPUQuota 800%; MemoryMax 6 GiB / MemorySwapMax 0. Actual peak memory: 3.081 GiB; swap: 0 bytes; scratch peak: 0.967 GB; maximum completed 60-second host CPU: 46.67%; minimum available memory: 66.24 GiB. Corrected stop guard: available memory <20 GiB OR host CPU >85% averaged over 60 seconds OR load1 >108 on three consecutive 60-second samples OR scratch >2 GB. The former instantaneous-CPU85 / absolute-load85 guard is retired. Unit ran Thu 2026-11-05 16:41:36 UTC to Thu 2026-11-05 17:49:31 UTC; timing is resource accounting only.

## Custody and next step

Create-only main archive: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H380/gov03-closeout-20261105T175344Z/h380-scratch.tar`, VersionId `I4t6tzUJQqOEM6oObue0iFvm.F0Fxe1a`, SHA256 `ca7e3379e087ee8ed88b6de11092be7c3f575c4ff5e4bb71c5a17eff801c00a2`. Exact-Version GET, whole-object hash and all 2864 members verified. Inner weights reside in 36 separate component objects (12,960 members), independently verified by exact Version, whole/member hashes, native parameters, observed leaves and gzip CRC. The complete recovery manifest is `provenance/component-archive-verified.json` in the main archive. The main archive preserves final refits, inputs, locks, audits, results and source without duplicating reclaimed inner weights. Receipt object VersionId: `JtP3LtPkhCzVtWIX6AXYC9byDGCClEYp`. Own scratch (656,357,485 bytes) and generated restoration were removed only after verification and zero active consumers. Immutable local main archive, source and receipts remain at `/home/ubuntu/jope_plan/evidence/H380/gov03-20261105-t1629`.

Retain frozen NL15. BFQ5 failed at least one locked gate. No dev91 evaluation is triggered; any next intervention needs a new owner assignment and preregistration.

DOPE c51a56d0 is unchanged and unloaded, CPU selection layer only. No DL-Val/dev91/DL-Test/sealed/protected rows, GPU, qualification.lock, live writer or AUTORESTART changes; no forbidden-host access or foreign-job actions. Foreign H354/H357/H364/H369/H377/H378/H379/H376 were untouched. Canonical HEAD, foreign index, 62 staged whitepaper paths and other dirty files were preserved. Publication uses the private index and non-force origin/v1/next only: no PR, main, fastlane, hosted CI or models in Git. Earlier waves remain closed.
