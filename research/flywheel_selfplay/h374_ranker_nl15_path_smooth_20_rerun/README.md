# H374: NL15 ranker path_smooth = 20 rerun

**COMPLETE — PS20_NOT_ADVANCED.** The primary comparison is PS20 against frozen H352 NL15; the secondary comparison is against original BASE.

## Hypothesis and setup

The hypothesis is that hierarchical shrinkage of leaf values improves NL15 selection. PS20 changes only ranker `path_smooth` from 0 to 20, for inner fits, final refits and all eight bags. Removing the added keyword restores the exact H352 function ASTs. The veto classifier is unchanged; all 108 final veto hashes match BASE and NL15.

The H352 configuration remains: 15 leaves, depth 3, minimum child count 20, L1 = 0, L2 = 1, minimum Hessian 0.001 and sigmoid 1. Learning rates (0.03, 0.05, 0.10), rounds (50, 100, 200), main rounds 200, the original 18-configuration grid, sampling, labels, groups, features, seed expressions, prediction and lock logic, 1e-4 ties and sentinel -2 are unchanged. There is no calibration, BASE/NL15 refit or new baseline model prediction.

[PREREG.md](PREREG.md) and the 35 frozen source files were published at `e8a8f67263a58cbb2dd0e120ccb42d9ee2bfc356` on 2026-11-05T14:42:23.128056+00:00, before the first treatment fit on 2026-11-05T14:43:10.579665+00:00. This copies frozen H371 `74eb11380472134d23acb1dc2261bd68034d4ec5` and binding `0efc8cbcabc642e2a670936513dacad6308578c50adb0a7737eb2175fbc4c55e`, adapting only wave identities/paths, RNG 374191, window and authorized core union. H371 was closed with zero fits after CPU admission failed; H374 is a separate owner allocation.

TRAIN only: 1,219 contexts from 205 families; P0/P1/P2 × roots 30001/30002/30003 × four family-grouped outer folds give 36 tasks and nine evaluation cells. Fresh exact-Version GETs, whole-object and every-member hashes verified BASE 9 panels / 36 pick-lock pairs / exact medians 0.0643163334285782, 0.04821480030677507, 0.04360142642936269. Fresh frozen NL15 verification covered 27 parts / 2,691 members, nine panels, 36 locks and COMPLETE score markers, raw family arrays, 972 native rankers and 108 veto identities. Explicit native path_smooth = 0 was verified before treatment.

## Outcome

Point summaries below are medians across nine evaluation cells. Intervals use whole-family resampling.

| Arm | Family-equal mean gain [95% CI] | Literal family median [95% CI] | Harmed share | Median Elo |
|---|---|---|---|---|
| BASE | +0.048215 [+0.025265, +0.076216] | +0.003624 [+0.000299, +0.007726] | 9.27% | 1725.66 |
| NL15 | +0.058405 [+0.033881, +0.092400] | +0.007409 [+0.002351, +0.011715] | 9.27% | 1734.47 |
| PS20 | +0.054565 [+0.022904, +0.096017] | +0.008509 [+0.003577, +0.011715] | 12.20% | 1710.84 |

`reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN`

PRIMARY paired drop-top-10, PS20 − NL15: **-0.008428**, 95% CI [-0.017972, -0.002127], one-sided LCB95 **-0.016494**. SECONDARY, PS20 − BASE: -0.002420, 95% CI [-0.009384, +0.004329], LCB95 -0.008111. Full-gain retention versus NL15 is 93.43%; positive primary root medians: 0/3. Configuration agreement is 16/36 against NL15 and 5/36 against BASE.

| Comparator | Root | Paired drop-top-10 median | 95% CI | One-sided LCB95 |
|---|---|---|---|---|
| NL15 | 30001 | -0.006758 | [-0.018149, +0.003905] | -0.016377 |
| NL15 | 30002 | -0.012862 | [-0.024847, +0.000557] | -0.022794 |
| NL15 | 30003 | -0.008428 | [-0.019753, +0.001497] | -0.017972 |
| BASE | 30001 | -0.002420 | [-0.014584, +0.010037] | -0.011957 |
| BASE | 30002 | +0.002266 | [-0.011697, +0.011297] | -0.009897 |
| BASE | 30003 | -0.004348 | [-0.014261, +0.006284] | -0.012398 |

Gates: `{"gain_retention": true, "harmed_nonincrease": false, "paired_drop_top10_at_least_002": false, "positive_on_two_seeds": false}`.

ADVANCE requires primary paired drop-top-10 median ≥ +0.002 AND LCB95 > 0; full gain ≥ 90% of NL15; harmed share no greater than NL15; and positive primary point medians on at least two of three roots. Ties never advance. Root intervals add no gate. All new ordinary and paired intervals use 10,000 shared whole-family draws, RNG 374191, across all three arms and nine cells. Each arm drops its own top ten after every resample. The global statistic is the median of nine cells; each root uses the median of its three partitions and the same draws. CI95 uses quantiles 0.025/0.975, LCB95 quantile 0.05.

[metrics.json](metrics.json) retains all 27 cell means and literal medians, intervals, drop-top-1/5/10, leave-one-family-out, wins/ties/losses, harmed share below −0.01, invalids, switches, Elo intervals, Jaccard and chosen configurations, plus both paired comparisons and root intervals. Full regression and lock inventories are hash-bound to the original raw metrics in verified custody. This is part of the sequential exploratory H348–H374 sweep, without a multiplicity-adjusted confirmatory claim. Elo is not AUC; there is no speed verdict.

## Verification and resources

All 36 fsynced locks, parent fsyncs, native/physical guards and veto checks preceded the fresh cold outer-score process. All 36 once-only score markers are COMPLETE; physical outer rows and packets read before lock are zero. All 36,504 native fits (29,484 ranker / 7,020 veto) and 14,040 saved bindings (12,636 ranker / 1,404 veto) passed constructor, native integer/seed and observed-leaf checks. Every native fit header was compared to frozen H352 at the same unchanged grid choice; only ranker path_smooth 0→20 differs.

The 48 preregistered data-free controls passed, with zero failures or skips. Two separate synthetic terminal-verifier controls passed, including rejection of a corrupted secondary root interval. Science exited 0. Independent saved-family arithmetic verified every ordinary and paired global/root interval without new fits, model predictions or outer feature reads. TRAIN input identities matched before and after. Reviews are by the same agent. Hosted CI was NOT_RUN under CI_FREEZE+BATCH_WINDOW.

One five-second both-SMT ≥85%-idle admission gate over the authorized union admitted cores [15, 20, 21, 23, 24, 25, 26, 48], the first at most eight passing pairs, with at least four required. Same-interval host CPU was 17.684207% (limit 70%). Initial forecast: 2026-11-05T15:57:35.985018+00:00; mandatory first-task / remaining-35 forecast: 2026-11-05T16:34:01.182433+00:00. Both passed the fixed November 5 17:30Z compute stop; close deadline 17:45Z. The reference was actual H352 CPU 26189.516914 seconds, efficiency 0.9, safety 1.15, reserve 300 seconds, and no ongoing-work credit at the first-task forecast. There was no second probe, borrowed core pool, restart, deadline reset, cut cell or partial scoring.

nice 19; math threads 1; CPUQuota 800%; MemoryMax 6 GiB / MemorySwapMax 0. Actual peak memory: 3.075 GiB; swap: 0 bytes; scratch peak: 0.964 GB; maximum completed 60-second host CPU: 46.03%; minimum available memory: 66.16 GiB. Corrected stop guard: available memory <20 GiB OR host CPU >85% averaged over 60 seconds OR load1 >108 on three consecutive 60-second samples OR scratch >2 GB. The former instantaneous-CPU85 / absolute-load85 guard is retired. Unit ran Thu 2026-11-05 14:42:46 UTC to Thu 2026-11-05 16:10:35 UTC; timing is resource accounting only.

## Custody and next step

Create-only main archive: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H374/gov03-closeout-20261105T161458Z/h374-scratch.tar`, VersionId `yAR5SaqmJN5d6mBiTv5Kdmj8A0deKK7c`, SHA256 `c50a3400101e72f9e94992225658a9a2e1e679f056f7a9eb3810bebe39c08ec8`. Exact-Version GET, whole-object hash and all 2866 members verified. Inner weights reside in 36 separate component objects (12,960 members), independently verified by exact Version, whole/member hashes, native parameters, observed leaves and gzip CRC. The complete recovery manifest is `provenance/component-archive-verified.json` in the main archive. The main archive preserves final refits, inputs, locks, audits, results and source without duplicating reclaimed inner weights. Receipt object VersionId: `JIylTYJ.agP0mTrumDGjPU0_GZlR9sA1`. Own scratch (654,558,185 bytes) and generated restoration were removed only after verification and zero active consumers. Immutable local main archive, source and receipts remain at `/home/ubuntu/jope_plan/evidence/H374/gov03-20261105-t1430`.

Retain frozen NL15. PS20 failed at least one locked gate. No dev91 evaluation is triggered; any next intervention needs a new owner assignment and preregistration.

DOPE c51a56d0 is unchanged and unloaded, CPU selection layer only. No DL-Val/dev91/DL-Test/sealed/protected rows, GPU, qualification.lock, live writer or AUTORESTART changes; no forbidden-host access or foreign-job actions. Foreign H354/H357/H364/H369/H367/H368/H370/H372/H373 were untouched. Canonical HEAD, foreign index, 62 staged whitepaper paths and other dirty files were preserved. Publication uses the private index and non-force origin/v1/next only: no PR, main, fastlane, hosted CI or models in Git. Earlier waves remain closed.
