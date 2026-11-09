# H361: NL15 ranker learning rate half, trees double

**COMPLETE — LRH_NOT_ADVANCED.** Primary comparator: frozen H352 NL15. Secondary comparator: original BASE.

The LRH hypothesis tests a smoother LambdaRank trajectory. Ranker learning rate is halved and every ranker fit/predict/lock iteration count is doubled, including all eight bag rankers. The inherited logical LR grid (0.03,0.05,0.10) becomes (0.015,0.025,0.05), and rounds (50,100,200) become (100,200,400); main fit 200 becomes 400. Logical grid labels in seed generation and tie ordering stay unchanged. Complete inverse AST checks reproduce H352. Leaves 15/depth 3, minimum child count 20, L1 zero/L2 one, Hessian 0.001, sigmoid 1, labels/groups/features/seeds/sampling, original 18-config grid, 1e-4 ties and sentinel -2 remain unchanged. The unchanged veto retains original settings, and all 108 final veto SHAs match both BASE and NL15. No calibration or BASE/NL15 refits/new baseline predictions.

Copied committed H352 `02dfee2f7e0b8543324494dd4f99d074cd6b93b6`. [PREREG.md](PREREG.md) source commit `e125117fac0826a66b7f724fbc961d33d0337b92` published 2026-11-05T11:25:39.542749+00:00, before first treatment fit 2026-11-05T11:26:49.035449+00:00. The 34-file scientific source remains frozen.

TRAIN only: 1,219 contexts and 205 families, P0/P1/P2 × roots30001/30002/30003 × 4 family-grouped outer folds = 36 tasks and 9 evaluation cells. Fresh exact-Version GET/whole+everymember SHA regression verified BASE 9 panels/36 fold-root pick+lock pairs and exact medians 0.0643163334285782/0.04821480030677507/0.04360142642936269. Frozen NL15 9 panels, 36 locks and COMPLETE score markers, raw family arrays, 972 native rankers and 108 veto identities passed 27 exact-version parts and all 2,691 members.

The first read-only NL15 preparation attempt failed because the checker incorrectly applied the ranker depth constraint to veto models. The corrected read-only attempt passed. Both attempts and the repair receipt are retained; neither attempt fitted models, predicted with models or consumed the admission probe.

## Results

Point summaries are medians across nine evaluation cells; 95% intervals use whole-family resampling.

| Arm | Family-equal mean gain [95% CI] | Literal family median [95% CI] | Harmed share | Median Elo |
|---|---|---|---|---|
| BASE | +0.048215 [+0.025571, +0.075964] | +0.003624 [+0.000299, +0.007782] | 9.27% | 1725.66 |
| NL15 | +0.058405 [+0.034059, +0.092199] | +0.007409 [+0.002351, +0.011715] | 9.27% | 1734.47 |
| LRH | +0.050921 [+0.028062, +0.078722] | +0.010072 [+0.005663, +0.013746] | 11.71% | 1707.88 |

`reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN`

PRIMARY LRH−NL15 paired drop10 **-0.003934**, 95% CI [-0.013603,+0.002829], one-sided LCB95 **-0.011671**. SECONDARY LRH−BASE +0.000155, 95% CI [-0.005841,+0.010376], LCB95 -0.004653. NL15 full-gain retention 87.19%;positive primary root medians 0/3;configuration agreement vs NL15 9/36, vs BASE 7/36.

| Partition | Root | BASE gain | NL15 gain | LRH gain | Primary paired drop10 |
|---|---|---|---|---|---|
| P0 | 30001 | +0.041289 | +0.050668 | +0.061653 | +0.004644 |
| P0 | 30002 | +0.064316 | +0.047569 | +0.074212 | +0.008038 |
| P0 | 30003 | +0.072258 | +0.073628 | +0.050921 | -0.015218 |
| P1 | 30001 | +0.048215 | +0.076944 | +0.057197 | -0.003934 |
| P1 | 30002 | +0.048658 | +0.047437 | +0.050706 | -0.006984 |
| P1 | 30003 | +0.038294 | +0.073295 | +0.045290 | -0.003333 |
| P2 | 30001 | +0.043601 | +0.047241 | +0.034401 | -0.014248 |
| P2 | 30002 | +0.031840 | +0.063382 | +0.039272 | -0.022310 |
| P2 | 30003 | +0.054721 | +0.058405 | +0.052894 | +0.005005 |

| Comparator | Root | Paired drop10 median | 95% CI | One-sided LCB95 |
|---|---|---|---|---|
| NL15 | 30001 | -0.003934 | [-0.014240,+0.008083] | -0.012333 |
| NL15 | 30002 | -0.006984 | [-0.026984,+0.009229] | -0.023621 |
| NL15 | 30003 | -0.003333 | [-0.014077,+0.005284] | -0.012231 |
| BASE | 30001 | +0.002551 | [-0.020711,+0.019377] | -0.016746 |
| BASE | 30002 | -0.000612 | [-0.011365,+0.016224] | -0.008913 |
| BASE | 30003 | +0.000155 | [-0.007398,+0.008022] | -0.006175 |

Gates: `{"gain_retention": false, "harmed_nonincrease": false, "paired_drop_top10_at_least_002": false, "positive_on_two_seeds": false}`.

ADVANCE requires primary paired drop10 median >= +0.002 AND one-sided LCB95 > 0, full gain >= 90% of NL15, harmed share <= NL15, and positive primary paired root point medians on >= 2/3 roots. Ties/subthreshold never advance. Root CIs add no gate. All new ordinary and paired bootstraps use 10,000 draws/RNG 361191; shared whole-family draws span all arms/nine cells. Each arm redrops its own top 10 after each resample; the statistic is median of 9 cells. Per-root uses the same draws and median of 3 partitions for that root. CI95 quantiles 0.025/0.975; LCB95 quantile 0.05.

[metrics.json](metrics.json) retains all 27 cell means/medians/CIs/drop1/5/10/LOFO/W-T-L/harm<-0.01/invalids/switches/Elo+CI/Jaccard/configurations, both paired comparisons and root intervals. Full regression and lock-hash inventories are hash-bound to raw metrics in verified custody. Sequential exploratory sweep H348–H361; no multiplicity-adjusted confirmatory claim. Elo is not AUC; no speed verdict.

## Verification and resources

All 36 fsynced locks/guards/108 veto matches preceded the fresh cold once-only outer score process; all 36 score markers COMPLETE. Physical outer rows/packets read before lock and parent fsync: 0. All 36,504 native fits (29,484 ranker/7,020 veto) and 14,040 saved bindings (12,636 ranker/1,404 veto) passed constructor/native/integer/seed checks, with every serialized header field compared to frozen H352 inner and final headers, allowing only ranker LR/iteration deltas; observed leaves were verified from actual trees. 48 preregistered data-free controls PASS, 0 failed/skipped. Two separate synthetic terminal-verifier checks PASS, including rejection of a corrupted secondary root CI. Actual science exited 0; independent saved-family arithmetic and all ordinary/paired/global/root CIs PASS, 0 new fits/model predictions/outer feature reads. TRAIN input identities matched before and after. Reviews are same-agent. Hosted CI NOT_RUN under CI_FREEZE+BATCH_WINDOW.

One 5-second both-SMT >= 85%-idle gate admitted [15, 20, 21, 22, 23, 24, 25, 26], first <= 8/minimum 4; same-interval host CPU 9.843160%<=70. Initial forecast 2026-11-05T13:50:59.233098+00:00; first completed-task remaining-35 forecast 2026-11-05T14:05:14.563639+00:00, both before fixed Nov5 14:45Z compute stop/15:00Z close. H352 measured CPU 26189.516914 s doubled conservatively for the entire CPU bound 52379.033828 s; efficiency 0.9/safety 1.15/reserve 300 s; ongoing-work credit 0 at the first task. No second probe, borrowed cores, restart, reset, cut cells or partial scoring.

nice 19, numeric threads 1, CPUQuota 800%, MemoryMax 6 GiB/SwapMax 0. Actual memory peak 3.423 GiB, swap 0 bytes, sampled scratch peak 1.146 GB, 60s host CPU max 53.99%, MemAvailable min 65.65 GiB. Corrected stop guard: MemAvailable < 20 GiB OR CPU > 85% averaged over 60s OR load1 > 108 on 3 consecutive 60s samples OR scratch > 2 GB. The previous instantaneous CPU85/absolute-load85 guard is retired. Unit Thu 2026-11-05 11:26:27 UTC to Thu 2026-11-05 13:23:59 UTC; timing is resource accounting only.

## Custody and next step

Retain frozen NL15. LRH failed all four locked gates. No dev91 evaluation is triggered; any next intervention requires a new owner assignment and preregistration.

Create-only main archive `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H361/gov03-closeout-20261105T133208Z/h361-scratch.tar`, VersionId `92CeM27ibQgCA4WCpYp4goNGL4MjbGvs`, SHA256 `a2b09a44c5427900b5b2421028dcf0fa6493ff4102fdea469c7955b6201b6d60`; exact-version GET/whole SHA/all 1,785 members verified. Inner weights live in 36 separate component objects (12,960 members), independently re-GET by exact version and whole/member/native/observed-leaf/gzipCRC verified; their complete recovery manifest is `provenance/component-archive-verified.json` in the main archive. The main archive holds final refits, inputs, locks, audits, results and source, and does not duplicate reclaimed inner-weight payloads. Receipt object VersionId `iWJVa0C06sF_La8csRUIFzxmaB2un5Eg`. Only after both recovery routes passed and zero consumers were own scratch (608,794,146 bytes) and generated restore reclaimed. Immutable local main archive/source/receipts remain at `/home/ubuntu/jope_plan/evidence/H361/gov03-20261105-t1100`.

DOPE c51a56d0 unchanged/unloaded, CPU selection layer only. No DL-Val/dev91/DL-Test/sealed/protected rows, GPU/qualification.lock/livewriter/AUTORESTART changes, forbidden host or foreign job actions. H354/H357/H359/H360 files/units untouched. Canonical HEAD/foreign index/62 staged whitepaper paths and other dirty files preserved; own private index/nonforce origin/v1/next only. No PR/main/fastlane/hostedCI or models in Git. Earlier waves stay closed.
