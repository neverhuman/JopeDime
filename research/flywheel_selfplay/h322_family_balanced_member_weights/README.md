# H322 — family-balanced LightGBM member weights

**FAMW_NOT_ADVANCED — TRAIN-only research.**
BASE pooled gain **+0.048214800**, 95% CI **[0.025146061127176742, 0.07529224349179893]**;
FAMW **+0.032460501**, 95% CI **[0.013417581432634117, 0.05266750028869769]**.
Paired drop-top-10 difference **-0.004816675**, CI **[-0.01495739279675088, 0.0016860761326157117]**,
one-sided 95% LCB **-0.013102759**.
Paired drop-top-10 by cell (P0 roots30001–30003, thenP1, thenP2):
`[-0.012213528977026584, 0.0014031878727228091, -0.004816674554508538, -0.00013208226485066955, -0.011187641493794268, -0.00443214829376774, -0.00889459985541069, 0.001633800731813781, -0.02285299614907655]`.
Criteria: `{'gain_retention_90percent': False, 'harmed_nonincrease': False, 'paired_drop10_positive': False}`. Failed clauses: `['paired_drop10_positive', 'gain_retention_90percent', 'harmed_nonincrease']`.
Median harmed-family rates BASE **0.092683** /
FAMW **0.097561**.
No dev91, r3175 TRAIN evaluation, neural optimizer update or serving change.

DOPE declaration required by the15:00Z closeout directive:
`dope = {"composite_sha256_prefix": "c51a56d0", "loaded": false, "scope": "CPU selection layer only; DOPE weights not loaded or modified"}`.
[Metadata amendment](dope_metadata_amendment.json) proves all pre-existing metrics
are identical; the numerical review was repeated on the saved panel after this
metadata-only addition. No model was rescored.

## Hypothesis and single intervention

Families with more contexts may dominate member fitting. H322 tests whether
balancing their fitting weights spreads gains across families while retaining
most of BASE's gain. This is a hypothesis, not an established cause of the dev91
tail. H298 concerned a neural adapter, not these LightGBM members.

Within EACH actual fitting multiset C of N context occurrences and F distinct
original families, count n_f occurrences from family f. Set
`w_c = N / (F * n_family(c))`, repeating that weight on all K=8 candidate slots.
Weights have mean one; each represented family has total slot weight K*N/F.
Recompute on each inner training set, final refit and bag sample. Bootstrap
copies count as occurrences: normalization cancels their within-fit multiplicity
while retaining the original bootstrap draw, seed and represented-family set.
Original family IDs remain intact across internal validation folds.

Weights reach inner rankers, inner invalid/veto classifiers, final ranker and
classifier refits, and all eight bag rankers. The H303 18-config TAU/k/agreement
grid, family folds, member seeds, model parameter recipe, features, labels,
utilities, strict margin selection and inner mean-family-gain objective are
unchanged. Selected hyperparameters can change as a consequence of weighted fits.
No H318 alpha gate, H320 trimmed objective or H321 target normalization is applied.

[weights.py](weights.py) adds only sample-weight arguments at five explicit
call sites and three LightGBM fit boundaries. [Transformation receipt](weight_transform_preview.json)
binds original and transformed functions. Removing those arguments restores
identical original ASTs. Per-fit formula and boundary receipts are independently
reconstructed from metadata, family splits and seeds: **36,504
weight vectors** match their context order and hashes exactly. All 36 cells have
1,014 model fits each. Final model hashes differing from BASE:
**1080/1080**. This records fitted model
changes; it is not a neural-learning or serving claim.

## Preregistration, BASE regression and physical input closure

[PREREG.md](PREREG.md) published at **b6311d03fd7d4e0621a98efc6fcb23ed3732cb8c** before any H322 fit.
Full fitting source **a67d4905b85772ee4df43e6da89947dcf4f7c8d0**; score/statistics source
**a845feb1ecae724d8f1c0046f77a55b7ab1303d7**, both before their respective execution.
Template H318 **fcc1baa75bada859929b288d590a8024b5eeac2f**; byte-pinned parent
H310 **819a45aa5e30e662ab511cd9962b85b2b8c1e44a** and unchanged H303/H302/H287
modules: [vendor pins](vendor_pins.json).

Recover H318's BASE refits, locks and panel by exact S3 VersionId GET, whole-object
SHA256 and member hashes: [input receipt](input_receipt.json),
[archive pin](base_archive_pin.json). Replay every frozen BASE model before FAMW
fitting, with the TRAIN-only physical range loader. All 36 fold/root pick hashes,
nine full pick panels and raw utility arrays match exactly. Partition medians:
P0 **+.0643163334285782**, P1 **+.04821480030677507**, P2 **+.04360142642936269**.
[Regression receipt](regression_receipt.json) records the proof. BASE is reused
read-only after this replay; it is not refitted in H322.

Each FAMW worker sees only its own outer-TRAIN pools, per-context cost members and
allowed byte ranges of the mixed H58 cache. Held-out outer packets are not
prebuilt. Each packet hash matches the corresponding unchanged BASE input.
**outer_test_rows_read_before_lock=0**, **outer_test_packets_built_before_lock=0**.
Fsync all 36 fit locks, chosen configs and model/source/input/weight identities.
The [pre-score review](lock_review.json) independently checks all weight vectors,
input identities, config tie rules and chosen inner gains. [Lock receipt](lock_receipt.json)
verifies 1080 final model hashes before the separate scoring process accesses
outer features. Exactly 36 FAMW fold/root scores produce the paired panel;
[score receipt](score_receipt.json) binds it. Later analysis reads only saved outputs.
Historical BASE regression is explicitly separate from this once-only evaluation.

## Metrics and uncertainty

1219 TRAIN contexts / 205 families; P0/P1/P2 × roots 30001–30003 × four outer folds.
Raw utility is -fitness on gate pass, otherwise -2. Within-family context means
receive equal family weight. The pooled gain is the median of nine cell mean
family gains, preserving H308's convention; literal median family gains are also
reported. Nine cells share families and are not independent replications.
W/T/L tolerance 1e-4; harmed means family gain < -.01. Invalid picks are context
counts. Drop-top-1/5/10 removes the largest family gains. LOFO minimum equals
removing the best family, so LOFO and drop-top-1 are not independent evidence.

Paired whole-family bootstrap: 10,000 draws, RNG 322191, the SAME sampled family
indices across both arms and all cells. Re-sort and trim each arm separately in
each draw; take the median of nine paired trimmed-mean differences. Two-sided
percentile CI uses .025/.975; decision LCB uses .05. Intervals condition on fitted
policies and do not repeat model selection or training. A separately reported
trim-of-paired-deltas statistic has no decision authority.

Advance requires positive paired drop10 and positive LCB, at least 90% of BASE's
pooled full gain, and no increase in median harmed-family rate. Ties do not
advance the paired improvement. No endpoint was changed after outcomes.

**reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN**.
Elo uses the unchanged two-arm Bradley–Terry MM recipe (400 iterations,
pseudocount .01, ties omitted). Median Elo BASE **1725.66** /
FAMW **1729.20**.

### BASE

| Partition/root | Mean family gain | Median family gain | 95% CI | W/T/L | Harmed | Invalid | Switches | Drop1/5/10 | LOFO min | Top10 share | Elo |
|---|---:|---:|---|---|---:|---:|---:|---|---:|---:|---:|
| P0/30001 | +0.041289223 | +0.000000000 | [+0.021885681, +0.063924774] | 88/93/24 | 13 | 13 | 219 | +0.034482/+0.022905/+0.013174 | +0.034482 | 0.696508 | 1725.66 |
| P0/30002 | +0.064316333 | +0.001588084 | [+0.015837506, +0.131071915] | 108/67/30 | 18 | 14 | 300 | +0.037100/+0.018970/+0.008633 | +0.037100 | 0.872323 | 1722.48 |
| P0/30003 | +0.072257657 | +0.006984224 | [+0.023137152, +0.140082463] | 122/50/33 | 19 | 11 | 320 | +0.045081/+0.027514/+0.016621 | +0.045081 | 0.781203 | 1727.10 |
| P1/30001 | +0.048214800 | +0.002999623 | [+0.018306340, +0.091584362] | 112/57/36 | 23 | 17 | 327 | +0.030657/+0.018311/+0.010198 | +0.030657 | 0.798815 | 1697.13 |
| P1/30002 | +0.048657946 | +0.003624258 | [+0.021819501, +0.083278572] | 116/51/38 | 21 | 15 | 275 | +0.035309/+0.023340/+0.014063 | +0.035309 | 0.725074 | 1693.84 |
| P1/30003 | +0.038294238 | +0.003253142 | [+0.016114776, +0.060918439] | 112/64/29 | 17 | 16 | 285 | +0.033333/+0.020954/+0.012702 | +0.033333 | 0.684486 | 1734.68 |
| P2/30001 | +0.043601426 | +0.006326451 | [+0.017478474, +0.071469848] | 122/48/35 | 22 | 16 | 315 | +0.037280/+0.021223/+0.010678 | +0.037280 | 0.767036 | 1716.88 |
| P2/30002 | +0.031839978 | +0.003624258 | [+0.010195141, +0.053698449] | 112/65/28 | 16 | 17 | 292 | +0.026502/+0.014887/+0.008774 | +0.026502 | 0.737886 | 1740.78 |
| P2/30003 | +0.054720630 | +0.008144684 | [+0.033244921, +0.079060965] | 129/41/35 | 22 | 15 | 335 | +0.048595/+0.033993/+0.023605 | +0.048595 | 0.589665 | 1726.57 |

### FAMW

| Partition/root | Mean family gain | Median family gain | 95% CI | W/T/L | Harmed | Invalid | Switches | Drop1/5/10 | LOFO min | Top10 share | Elo |
|---|---:|---:|---|---|---:|---:|---:|---|---:|---:|---:|
| P0/30001 | +0.033132445 | +0.000000000 | [+0.002034010, +0.066782922] | 95/84/26 | 15 | 15 | 185 | +0.021893/+0.009431/+0.000960 | +0.021893 | 0.972438 | 1725.05 |
| P0/30002 | +0.030944142 | +0.010809190 | [+0.004796098, +0.054137438] | 125/55/25 | 17 | 15 | 351 | +0.026444/+0.017609/+0.010036 | +0.026444 | 0.691494 | 1779.53 |
| P0/30003 | +0.050625899 | +0.003658849 | [+0.018275474, +0.084056072] | 116/58/31 | 20 | 14 | 278 | +0.042558/+0.023066/+0.011804 | +0.042558 | 0.778215 | 1729.20 |
| P1/30001 | +0.040485986 | +0.003867208 | [+0.017779807, +0.065153680] | 122/52/31 | 21 | 13 | 302 | +0.034713/+0.021122/+0.010065 | +0.034713 | 0.763512 | 1737.96 |
| P1/30002 | +0.028934108 | +0.000000000 | [+0.005439546, +0.052794823] | 100/68/37 | 23 | 17 | 289 | +0.022609/+0.010965/+0.002876 | +0.022609 | 0.905460 | 1672.69 |
| P1/30003 | +0.026040550 | +0.000000000 | [+0.007647168, +0.043861727] | 88/100/17 | 8 | 17 | 167 | +0.021517/+0.013850/+0.008270 | +0.021517 | 0.697916 | 1785.53 |
| P2/30001 | +0.032460501 | +0.007288601 | [-0.002120214, +0.062819297] | 118/45/42 | 28 | 14 | 395 | +0.027286/+0.013743/+0.001784 | +0.027286 | 0.947726 | 1679.43 |
| P2/30002 | +0.034812517 | +0.003577225 | [+0.012866054, +0.056946195] | 118/57/30 | 18 | 16 | 254 | +0.030170/+0.019834/+0.010407 | +0.030170 | 0.715625 | 1737.86 |
| P2/30003 | +0.018551516 | +0.001684134 | [-0.000346257, +0.036165031] | 106/67/32 | 20 | 20 | 287 | +0.015953/+0.007038/+0.000752 | +0.015953 | 0.961427 | 1708.02 |

Concentration is the signed sum of the ten highest family gains divided by total
signed gain when that total is positive; otherwise it is undefined. Cancellation
can make the ratio exceed one. Gross-positive concentration, numerators and
denominators are also retained. Median signed top10 share BASE
**0.737886** / FAMW **0.778215**.

Seed-Jaccard compares switched-context sets for each root pair within each
partition/outer fold (36 comparisons per arm; empty/empty=1). Median BASE
**0.451380** / FAMW
**0.410494**. Candidate-action Jaccard is
separately labelled. Exact paired config agreement **0.083333**;
parameter agreement `{'a': 1.0, 'k': 0.3888888888888889, 'tau': 0.3333333333333333}`. [metrics.json](metrics.json),
37,416 bytes, contains all nine cells, partition
CIs, all Jaccard comparisons and all 36 paired config choices.

## Verification, custody and limitations

27 focused tests passed; exact BASE replay, full metadata/weight/input/config
review and independent numerical review passed. [Numerical receipt](numerical_review.json)
recomputes counts, LOFO, concentration, Jaccard, config agreement, partial-sort
bootstrap and decision. Same-agent review is identified honestly. The first
synthetic full-member fixture had three columns while frozen reload requires
568; the fixture was corrected and its failed/passing receipts retained.
Full repository/hosted CI NOT_RUN under CI_FREEZE/MASTER_MERGE_HOLD.

CPU xbabe2, nice19, one numeric thread, cores13/15/57/59/60/62; their SMT siblings
were disjoint from H317/H320/H321 reservations at launch. Actual cgroup readback:
MemoryMax=6442450944, MemorySwapMax=0. Aggregate RSS cap6GiB, scratch cap2GB,
host CPU cap85%, MemAvailable floor20GiB. Peak sampled RSS through analysis
**3,902,636,032 bytes**, host CPU
**49.1%**; no guard violations. Absolute compute
15:15Z / close15:30Z Nov4 retained. [Resource receipt](resource_receipt.json) and
[dependency pins](dependency_pins.json) preserve execution details.

36 create-only model archives have exact-version GET/SHA/member verification;
[model archive index](model_archive_index.json) binds
14,220 members. Final evidence,
manifest and lifecycle objects use
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H322/astra-20261104T132627Z/`.
[Manifest](manifest.json) and [S3 receipt](s3_manifest.json) identify exact versions.
Own scratch is reclaimed only after every file has a verified recovery mapping
and no active consumer. Foreign files, jobs and checkouts remain untouched.

[TRAIN provenance](train_input_provenance.json) records selected/cache metadata,
TRAIN cost hash and permitted range/packet identities. Original H50/H58/H276
local-input S3 versions are not independently established. No whole mixed-cache
feature hash is recomputed, because that would read forbidden DL-Val rows.
The custody archive supplies source/, vendor/, receipts/ and outputs/; restore
outputs at the declared H322 data root. Numerical replay uses the archived panel;
full refitting also needs those separately retained original inputs. Published
code is an exact run artifact, not a standalone checkout CLI. No new replay
budget is implied. Local custody: `/home/ubuntu/jope-rescue-20260921/h322-astra`.

Cross-critique: H318's negative median-minus-harm objective was observed, not a
theorem about every switching policy; some inner family medians were positive.
H319 relabelled bootstrap copies before internal family folds; H322 preserves
original family identities and counts multiplicity only for weights. Negative
results from specific H315/H316/H318/H319 recipes do not rule out every selector
or fitting method. TRAIN drop10 removes 10/205 families versus 10/19 on dev91;
those contrasts also change cohort/incumbent and do not isolate a single cause.
H320 also closed TRIM_NOT_ADVANCED: its paired drop10 was negative in all nine
cells and it retained 57% of BASE gain. That is evidence against its specific
trimmed inner objective, not proof that every selector fails or that pool
composition uniquely causes the dev91 tail. Its inner trim removes about10/37
families, a different fraction from outer10/205 and dev10/19. Its report declares
outer observations present in memory with zero guarded accesses; H322 separately
requires physical worker input exclusion. [Peer review](h320_crosscritique.json)
binds the reviewed H320 report. H322 endpoints and fitting code were unchanged.
Do not advance this weighted member recipe to dev91. Keep unchanged ENS_VOTE2 as the reference; this result does not rule out every training-weight or selector method.
