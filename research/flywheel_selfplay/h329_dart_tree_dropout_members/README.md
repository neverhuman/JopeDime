# H329: DART tree dropout in ENS_VOTE2 rankers

**Verdict: DART_NOT_ADVANCED.** All 36 fold/root fits and once-only outer score
markers completed successfully. The paired drop-top-10 median was −0.000167,
with one-sided LCB95 −0.005541. Median harmed share increased from 9.27% to 12.68%,
and only one of three root medians was positive. These fail the preregistered
robustness, harm and root-consistency gates despite higher full mean gain.

## Hypothesis and setup

Dropping complete boosted trees may reduce concentration of gains in the
strongest families. One fixed arm changes every inner, final and bag ranker to
`boosting_type="dart"`, `drop_rate=0.1`, `skip_drop=0.5`, `max_drop=50`,
`uniform_drop=False`, `xgboost_dart_mode=False`, and
`drop_seed=int(original per-fit random_state)`. Veto classifiers retain GBDT.
Labels, LambdaRank/default label_gain, features, member and bag seeds, the nested
18-config grid, rounds, tie/switch rules and failure sentinel −2.0 are unchanged.
Stripping the override restores both complete original ranker function ASTs.
Original topk rankers fit 200 rounds; inner-selected 50/100/200 prediction counts
and original bag fit counts are retained, without early stopping.

The design is H308 TRAIN-only P0/P1/P2 × roots 30001–30003 × four family-grouped
outer folds: 1,219 contexts and 205 families. Frozen H303 NESTED BASE reproduces
all nine panels, all 36 fold/root pick and lock hashes, and published partition
medians 0.0643163334285782 / 0.04821480030677507 / 0.04360142642936269 exactly.
BASE was retrieved read-only by exact archive version; no BASE refits or fresh
BASE model scores were performed.

[PREREG.md](PREREG.md) was published before any optimizer fit at
`b77b549d3ea31d934781129f825d3c6eba2e41bf`. The resource-only correction
`b7f44b9e879c4b3039ed9cf28347b88ba14a75ff` preserves scientific code and completed
controls. The pre-score [report-field correction](EVALUATOR_FIELD_CORRECTION.md)
`f6f442f1aad936cf1b695b140aa5e60963ac9a85` renames two inherited XT fields to DART;
all numerical ASTs are unchanged. Existing workers continued without restart,
and no fits, timing fixtures or model predictions were repeated.

## Outcome and metrics

| arm | median family-equal mean gain [95% CI] | literal family median [95% CI] | median harmed | median Elo |
|---|---|---|---|---|
| BASE | +0.048215 [+0.025157, +0.076452] | +0.003624 [+0.000299, +0.007726] | 0.092683 | 1725.66 |
| DART | +0.058526 [+0.024616, +0.110201] | +0.006299 [+0.002436, +0.010521] | 0.126829 | 1693.56 |

**Elo label: reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.**

Paired drop-top-10 median **-0.000167**, 95% CI [-0.006514, +0.007384], one-sided LCB95 **-0.005541**. Gain retention 121.39%; positive root medians 1/3; configuration agreement 6/36 (CONFIG_SENSITIVE).

| partition | root | BASE gain | DART gain | paired drop10 | BASE harmed | DART harmed |
|---|---|---|---|---|---|---|
| P0 | 30001 | +0.041289 | +0.069910 | +0.005840 | 0.063415 | 0.117073 |
| P0 | 30002 | +0.064316 | +0.055055 | +0.002074 | 0.087805 | 0.141463 |
| P0 | 30003 | +0.072258 | +0.058526 | -0.006448 | 0.092683 | 0.087805 |
| P1 | 30001 | +0.048215 | +0.048189 | +0.007307 | 0.112195 | 0.126829 |
| P1 | 30002 | +0.048658 | +0.030164 | -0.001795 | 0.102439 | 0.073171 |
| P1 | 30003 | +0.038294 | +0.066234 | -0.000139 | 0.082927 | 0.126829 |
| P2 | 30001 | +0.043601 | +0.040281 | -0.013868 | 0.107317 | 0.131707 |
| P2 | 30002 | +0.031840 | +0.058582 | -0.000167 | 0.078049 | 0.151220 |
| P2 | 30003 | +0.054721 | +0.068887 | -0.000784 | 0.107317 | 0.131707 |

Per-root paired medians (30001/30002/30003): +0.005840, -0.000167, -0.000784. Gates: `{"gain_retention": true, "harmed_nonincrease": false, "paired_drop_top10_at_least_002": false, "positive_on_two_seeds": false}`.

ADVANCE requires paired nine-cell drop10 median ≥ +.002 and LCB95 > 0, median full gain ≥ 90% of BASE, median harmed ≤ BASE and positive paired drop10 medians on ≥ 2/3 roots. Paired bootstrap: 10,000 whole-family draws, RNG 329191, one resample shared across arms/all nine cells, each arm redrops its own top 10; LCB quantile .05. Ordinary gain/median/Elo intervals use 10,000 whole-family draws, RNG 302150. All per-cell gain/literal median/CI, drop1/5/10, LOFO minimum, W/T/L, harmed > .01, invalid, switches, Elo/CI, configs and exact lock hashes remain in [metrics.json](metrics.json).

## Verification and resources

All 36 physically scoped input receipts confirm zero outer feature rows or
packets read before the corresponding fsynced fit lock. Every lock and guard
passed review before scoring began. Native model review verified 12,636 DART
rankers and 1,404 unchanged GBDT veto models; constructor audits verified 29,484
DART ranker and 7,020 GBDT classifier calls. All 36 inner archives passed exact
version, SHA and member verification before raw inner models were removed.

The actual campaign exited 0 with 36 COMPLETE score markers. Saved-family
arithmetic and the paired bootstrap were independently recomputed without model
predictions or outer feature reads. All 23 data-free controls passed, with no
failures or skips; the original report fixture reproduced `KeyError: DART`
before its correction. These are same-agent reviews, with no independent or
native approval claimed. Hosted CI was NOT_RUN under the freeze.

The campaign ran on xbabe2 from 18:19:45Z to 19:48:03Z, with six assigned cores
16/17/18/19/20/22, nice 19, one numerical thread, MemoryMax 6 GiB and SwapMax 0.
Peak memory was 2.479 GiB with zero swap; maximum
scratch was 1.146 GB. Recorded host CPU peaked at
28.33%, load at 40.34, and available memory
remained at least 70.93 GiB. Full accounting and
the 1,088 resource samples are retained in the receipts and archive.

An assumption-based forecast preceded all fits. Six fixed synthetic timing fits
refined the DART/GBDT multiplier to 1.088392827976701. The first capacity probe
used the minimum of three one-second readings and stopped before any campaign
fit; its exit 1, logs and forecast are preserved. One corrected five-second
interval per core and SMT sibling admitted all six assigned cores at 92–96%
idle, retaining the ≥85% threshold. The full-run forecast ended at 20:05:31Z.
The fixed 20:45Z compute stop and 21:00Z close deadline were never reset.

## Verdict, next step and custody

Retain frozen BASE. DART does not advance, and no dev91 evaluation is triggered.
Any further intervention requires a new preregistration. H329 used no DL-Val
feature rows, dev91, DL-Test or protected cohort data. DOPE `c51a56d0` was
unchanged and not loaded; scope: CPU selection layer only.

Create-only archive:
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H329/gov03-closeout-20261104T194852Z/h329-scratch.tar`.
Exact VersionId: `f.MjXZSOhYawRNNrzuRy6jLiHzZg4ldI`.
SHA256: `8cf8e5a979d055c4fb68c830ea9fcb726cd386f020cb3291a5f4ec4ad2f42da9`.
Exact-version GET and all 1,545 file hashes passed before removal of
only `/home/ubuntu/jope-data/h329-gov03` (1,011,611,627 bytes).
The generated restore tar was reclaimed; the immutable local archive and source
refs remain. Archive-receipt VersionId: `rqP8hFWu7rAaJw0QMp4Fy9Hd53d7RdQ_`.
Lifecycle evidence: `/home/ubuntu/jope_plan/evidence/H329/gov03-20261104-t1735`. No models enter Git.

[metrics.json](metrics.json) retains every scientific value, source identity,
actual exit, resource summary and custody receipt in fewer than 50 KB.
Publication is a nonforce push to `origin/v1/next` through a private index under
CI_FREEZE and MASTER_MERGE_HOLD. No PR, main, fastlane, CI dispatch, GPU, live
writer or foreign-job action was taken. H326 remains closed.
