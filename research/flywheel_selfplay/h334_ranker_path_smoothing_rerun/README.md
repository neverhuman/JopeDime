# H334: ranker path smoothing on a new allocation

**Verdict: PSMOOTH_NOT_ADVANCED.** All 36 fold/root fits and once-only score markers completed successfully. Paired drop-top-10 median was −0.005699 with one-sided LCB95 −0.010495. Median harmed share increased from 9.27% to 11.71%, and none of the three root medians was positive. Gain retention passed; robustness, harm and root-consistency gates failed.

## Hypothesis and setup

Hierarchical shrinkage toward parent nodes may reduce reliance on a few small leaves fitted to high-residual families. The one fixed treatment adds only `path_smooth=20.0` to every inner, final and eight-bag ranker. BASE leaves it unset (native default 0.0); the owner fixed 20.0 equal to unchanged min_child_samples=20. Rankers retain GBDT and LambdaRank/default label_gain; veto classifiers retain GBDT/default path_smooth=0. Labels, features, seeds, rounds, the nested 18-config grid, objective and tie/switch/failure-sentinel rules are unchanged. Removing the one keyword restores both complete original function ASTs. Original topk fits 200 rounds; selected 50/100/200 prediction counts and original bag fit counts remain, without early stopping.

H303 NESTED/H308 TRAIN-only P0/P1/P2 × roots 30001–30003 × four family-grouped outer folds: 1,219 contexts, 205 families, all 36 tasks/9 cells. Frozen BASE matches all nine panels, 36 fold/root pick and lock hashes, and published medians 0.0643163334285782 / 0.04821480030677507 / 0.04360142642936269 exactly. No BASE refits or new BASE model scores. Current TRAIN identities matched the archive before fitting and after completed scoring: all 1,219 pool hashes, four consumed feature prefixes, selected.json and r882 TRAIN costs.

Preregistered before any H334 optimizer fit at `9a002956844f3825e2555a874de1a7b7f11c5f49`; [PREREG.md](PREREG.md). Parent H333 preregistration is `0ab56940ee96bafd959d29c8fa9e6c4b850fc56b`. Its treatment and inverse-AST tests were copied byte-exact. H333 remains CLOSED with zero fits and unknown numerical outcome; H334 uses a separately assigned allocation/window.

## Outcome and metrics

| arm | median family-equal mean gain [95% CI] | literal family median [95% CI] | median harmed | median Elo |
|---|---|---|---|---|
| BASE | +0.048215 [+0.025157, +0.076452] | +0.003624 [+0.000299, +0.007726] | 0.092683 | 1725.66 |
| PSMOOTH | +0.045952 [+0.021722, +0.086625] | +0.006326 [+0.002917, +0.010888] | 0.117073 | 1707.23 |

**Elo label: reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.**

Paired drop-top-10 median **-0.005699**, 95% CI [-0.012184, +0.004997], one-sided LCB95 **-0.010495**. Full gain retention 95.31%; positive root medians 0/3; config agreement 6/36 (CONFIG_SENSITIVE).

| partition | root | BASE gain | PSMOOTH gain | paired drop10 | BASE harmed | PSMOOTH harmed |
|---|---|---|---|---|---|---|
| P0 | 30001 | +0.041289 | +0.059618 | -0.001857 | 0.063415 | 0.112195 |
| P0 | 30002 | +0.064316 | +0.031806 | -0.005699 | 0.087805 | 0.146341 |
| P0 | 30003 | +0.072258 | +0.045952 | -0.017065 | 0.092683 | 0.117073 |
| P1 | 30001 | +0.048215 | +0.063011 | +0.011231 | 0.112195 | 0.073171 |
| P1 | 30002 | +0.048658 | +0.043645 | -0.012739 | 0.102439 | 0.102439 |
| P1 | 30003 | +0.038294 | +0.039772 | +0.003102 | 0.082927 | 0.102439 |
| P2 | 30001 | +0.043601 | +0.034187 | -0.006044 | 0.107317 | 0.126829 |
| P2 | 30002 | +0.031840 | +0.047479 | +0.009234 | 0.078049 | 0.117073 |
| P2 | 30003 | +0.054721 | +0.066677 | -0.007780 | 0.107317 | 0.131707 |

Per-root paired medians (30001/30002/30003): -0.001857, -0.005699, -0.007780. Gates: `{"gain_retention": true, "harmed_nonincrease": false, "paired_drop_top10_at_least_002": false, "positive_on_two_seeds": false}`.

ADVANCE requires paired nine-cell drop10 median ≥ +.002 and LCB95 > 0, full median gain ≥ 90% of BASE, median harmed ≤ BASE, and positive paired root medians on ≥ 2/3 roots. Paired bootstrap: 10,000 whole-family draws, RNG 334191; one resample shared across both arms/all nine cells; each arm redrops its own top 10; statistic median nine cells, LCB quantile .05. Ordinary gain/literal median/Elo intervals use 10,000 family draws, RNG 302150. [metrics.json](metrics.json) preserves every scientific value, including per-cell gain/CI, drop1/5/10, LOFO min, W/T/L, harmed > .01, invalid, switches, Elo/CI, seed Jaccard, chosen configs and fit-lock hashes.

## Verification and resources

Physical outer feature rows and packets before each fsynced fit lock: 0. All 36 locks/guards/input byte identities passed before scoring. Native audits validated every native-mappable constructor argument after all 29,484 ranker and 7,020 veto fits. The terminal review also checks integer settings and seeds exactly. Saved-model review verified 12,636 PSMOOTH rankers and 1,404 veto models with unchanged GBDT parameters against their member’s native fit records. All 36 inner archives passed exact-version GET, SHA and every member check before raw-inner removal.

All 31 data-free controls passed, with no failures or skips. The campaign exited 0 with 36 COMPLETE score markers. Saved-family arithmetic and the paired bootstrap were independently recomputed without model predictions or outer feature reads. Source, manifest and preregistration pins remain exact. These are same-agent reviews; no independent or native approval is claimed. Hosted CI was NOT_RUN under CI_FREEZE and BATCH_WINDOW.

One five-second per-core+SMT idle gate admitted all eight assigned cores: 1, 2, 3, 7, 9, 10, 12, 14. Both members of each core/sibling pair were ≥85% idle; no re-probe or borrowed core. The complete-run forecast ended at 22:35:47Z and was recorded before any H334 fit. The actual unit ran 21:14:37Z–22:36:29Z on 4 November 2026, nice 19, one numerical thread, MemoryMax 6 GiB/SwapMax 0. Peak memory was 3.340 GiB with zero swap; scratch peak 1.339 GB. Host CPU peaked at 49.12%, load at 74.13, and available memory remained at least 67.26 GiB. Fixed compute stop 23:59Z Nov4 and close 00:15Z Nov5 were never reset. These timings are resource accounting and admission evidence; no speed verdict is made.

## Verdict, next step and custody

PSMOOTH does not meet every preregistered gate. Retain frozen BASE; no dev91 evaluation is triggered. Any new intervention requires a new preregistration.

Create-only archive: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H334/gov03-closeout-20261104T223818Z/h334-scratch.tar`. Exact VersionId `yJ3totlF.ftvnT55TqYQ3LRvdntpTuID`, SHA256 `684e8d5f36047dc934f78943f58c35f90c7c346bf60c8835bc76a7a315b95f51`. Exact-version GET and all 1,562 file hashes passed before deleting only `/home/ubuntu/jope-data/h334-gov03` (1,237,836,130 bytes). Fresh actual-path consumer checks passed. Generated restore copy reclaimed; immutable archive and source refs retained. Archive-receipt VersionId `BkXUqiPBzTgddRpp8Zlu1iOBZE3wetHq`. Lifecycle evidence: `/home/ubuntu/jope_plan/evidence/H334/gov03-20261104-t2105`. No models in Git.

DOPE c51a56d0 was unchanged and not loaded; scope CPU selection layer only. No GPU, DL-Val feature rows, dev91/DL-Test/protected cohorts, live writer, foreign jobs or forbidden hosts were used. Publication is nonforce origin/v1/next via private index; no PR/main/fastlane/CI dispatch. H333/H329/H326 remain closed.
