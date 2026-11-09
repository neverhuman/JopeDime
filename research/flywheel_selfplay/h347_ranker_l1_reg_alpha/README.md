# H347: ranker L1 regularization

**COMPLETE — L1_NOT_ADVANCED.** All 36 fit tasks and once-only outer score markers completed.

## Hypothesis and setup

L1 leaf-output regularization may reduce sensitivity to small noisy corrections. The sole treatment adds `reg_alpha=0.5` to every inner, final and eight-bag ranker. All 972 archived BASE rankers confirm native L1 zero. Stripping this override restores both complete pinned H287 function ASTs exactly. LambdaRank/ndcg, sigmoid 1, labels 0..7, implicit label gains, Hessian floor 0.001, min_child_samples 20, depth/leaves, subsampling, L2, LR/tree options, seeds, features, the original 18-config grid, 1e-4 ties/switches and the −2 sentinel retain their original definitions.

Veto classifiers retain their pinned binary GBDT configuration, including reg_alpha=1 and reg_lambda=2. Every final veto file matched its archived BASE SHA: **108/108**. BASE inputs and native models were checked read-only, with zero BASE refits or calibration fits.

Copied the committed H342 harness from origin/v1/next at `167a527e2889b6b4e4be69736e03a0507bd43f4f`. Preregistration `6e08f967487a81a9be1c95b32a75565dba222a4e` ([PREREG.md](PREREG.md)) was published at 2026-11-05T02:14:27.876684+00:00, before first actual fit 2026-11-05T02:15:20.775901+00:00.

H303 NESTED/H308 TRAIN-only design: P0/P1/P2 × roots 30001–30003 × four family-grouped outer folds, 36 tasks and nine evaluation cells, 1,219 contexts and 205 families. Exact-version BASE GET, whole/member hashes, all nine panels, 36 fold/root pick+lock hashes and three medians 0.0643163334285782 / 0.04821480030677507 / 0.04360142642936269 passed. Current TRAIN pools, selected/r882 costs and four consumed cache prefixes matched archived input hashes before and after the run. No fresh BASE model predictions.

## Results

Aggregate gains are medians across nine cells; brackets are whole-family bootstrap 95% confidence intervals.

| Arm | Family-equal mean gain [95% CI] | Literal family median [95% CI] | Harmed share | Median Elo |
|---|---|---|---|---|
| BASE | +0.048215 [+0.025157, +0.076452] | +0.003624 [+0.000299, +0.007726] | 9.27% | 1725.66 |
| L1 | +0.056299 [+0.027275, +0.093454] | +0.006552 [+0.002858, +0.011286] | 11.22% | 1714.01 |

`reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN`

Paired drop-top-10 median: **+0.003644**, 95% CI [-0.004728, +0.010292], one-sided LCB95 **-0.003531**. Gain retention 116.77%; positive root medians 2/3; configuration agreement 10/36 (CONFIG_SENSITIVE).

| Partition | Root | BASE gain | L1 gain | Paired drop10 | BASE harmed | L1 harmed |
|---|---|---|---|---|---|---|
| P0 | 30001 | +0.041289 | +0.078876 | +0.009033 | 0.063415 | 0.117073 |
| P0 | 30002 | +0.064316 | +0.044087 | +0.003644 | 0.087805 | 0.078049 |
| P0 | 30003 | +0.072258 | +0.051822 | -0.001385 | 0.092683 | 0.117073 |
| P1 | 30001 | +0.048215 | +0.057750 | +0.001827 | 0.112195 | 0.112195 |
| P1 | 30002 | +0.048658 | +0.044123 | -0.002984 | 0.102439 | 0.073171 |
| P1 | 30003 | +0.038294 | +0.056299 | +0.006423 | 0.082927 | 0.073171 |
| P2 | 30001 | +0.043601 | +0.055105 | +0.006909 | 0.107317 | 0.136585 |
| P2 | 30002 | +0.031840 | +0.056464 | +0.008590 | 0.078049 | 0.102439 |
| P2 | 30003 | +0.054721 | +0.061340 | -0.012439 | 0.107317 | 0.136585 |

Per-root paired medians for 30001 / 30002 / 30003: +0.006909 / +0.003644 / -0.001385. Gates: `{"gain_retention": true, "harmed_nonincrease": false, "paired_drop_top10_at_least_002": false, "positive_on_two_seeds": true}`.

| Root | Paired drop10 median | 95% CI |
|---|---|---|
| 30001 | +0.006909 | [-0.001959, +0.016194] |
| 30002 | +0.003644 | [-0.007675, +0.014312] |
| 30003 | -0.001385 | [-0.015927, +0.012058] |

Each root interval uses the median across its three partitions and the same 10,000 shared whole-family resamples. All three intervals cross zero. This saved-gain reporting adds no fits, model predictions, or decision gates.

ADVANCE requires paired drop10 median ≥ +0.002 and LCB95 > 0, full median gain ≥ 90% of BASE, median harmed share ≤ BASE, and positive paired root medians on at least two of three roots. Paired bootstrap: 10,000 whole-family draws, RNG 347191, one family sample shared across both arms/all nine cells, each arm dropping its own top ten, median across nine cells as the statistic; LCB95 is quantile 0.05. Ordinary gain/literal-median/Elo intervals use 10,000 draws and RNG 302150. [metrics.json](metrics.json) preserves every scientific value, including per-cell intervals, drop1/5/10, LOFO, W/T/L, harm below −0.01, invalids, switches, Elo intervals, Jaccard, configurations and lock hashes.

## Verification and resources

During treatment fitting, physical outer feature rows and packets read before the fsynced lock and parent-directory fsync were zero. All 36 guards/locks/input identities/native audits and 108 final-veto SHA matches passed before cold scoring. Native audits checked all 29,484 ranker and 7,020 veto fits, and bound 12,636 saved rankers and 1,404 veto models to member-native records. Every inner archive passed exact-version GET, whole SHA and member verification before raw-inner removal.

All 39 data-free controls passed with zero failures or skips. The campaign exited 0 with all 36 score markers COMPLETE. Separate saved-family arithmetic and bootstrap replay passed without fitting or predicting again. These are same-agent reviews; source/manifest/prereg hashes remain exact. Hosted CI was NOT_RUN under CI_FREEZE + MASTER_MERGE_HOLD.

The first read-only native verifier exited 1 after BASE regression/input checks passed because it applied the ranker L2 assertion to veto. Before any treatment fit or admission probe, it was corrected to the unchanged pinned veto L1=1/L2=2; a separate read-only native check exited 0 and a real archived-veto regression control passed. Original source and failed-command receipts remain preserved. No data or treatment value changed.

ONE five-second/core+SMT ≥85%-idle sample admitted [0, 1, 2, 3, 8, 9], the first at most six passing pairs; same-interval host CPU was 10.207% ≤70%. Controller threads and all workers were bound to those admitted cores before fitting. Initial H342 actual-CPU forecast ended at 2026-11-05T03:48:57.008520+00:00; required first-completed-task forecast at 2026-11-05T04:07:22.475611+00:00. Both precede the fixed November 5 05:45Z compute stop. Close is fixed at 06:00Z. There was no re-probe, worker expansion, borrowed core, partial scoring, restart or deadline reset.

Actual campaign: Thu 2026-11-05 02:14:59 UTC to Thu 2026-11-05 03:41:32 UTC, nice 19, one numeric thread, 6 workers, MemoryMax 6 GiB and SwapMax 0. Peak memory 3.097 GiB, swap 0 bytes, scratch 1.532 GB. Sampled host CPU maximum 25.86%, load maximum 46.03, minimum available memory 73.14 GiB. Timing supports resource admission/accounting.

## Custody and next step

Retain frozen BASE. L1 failed at least one locked gate; no dev91 evaluation is triggered. Any future intervention needs a separate owner assignment and preregistration.

Create-only archive `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H347/gov03-closeout-20261105T034236Z/h347-scratch.tar`; VersionId `DRLwbi1GrR8YYuEH3nXKq_ASJl457jr_`; SHA256 `53b4d4abf2ebab76bd5d6e5a8d3fa3beb7b59ba6131c2ea11cf86383650dec9e`. Exact-version GET, whole SHA and all 2,751 file members passed before deleting 1,399,431,250 bytes of own scratch. Fresh consumer checks found zero references. Generated restore reclaimed; immutable local archive/source/evidence retained at `/home/ubuntu/jope_plan/evidence/H347/gov03-20261105-t0200`. Receipt VersionId `b18mqNytFKL5oqEWLr_ehvCww361rThD`.

DOPE c51a56d0 remains unchanged and unloaded; scope is CPU selection only. No DL-Val/dev91/DL-Test/protected rows, GPU/qualification.lock, live writer, foreign jobs or forbidden hosts were used. H345R and H335 jobs were left alone. Private index, nonforce v1/next only; no models in Git or PR/main/fastlane/CI operations. Prior H342/H340/H334/H333/H329/H326 remain closed.
