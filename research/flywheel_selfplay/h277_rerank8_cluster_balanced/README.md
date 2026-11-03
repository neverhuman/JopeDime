# H277 — Cluster-balanced RERANK8

Hypothesis: equal cluster mass during RERANK8 training/selection, plus a nested H273 veto threshold, improves utility on entirely held-out family clusters without extra raw−2 picks or harmed families. Outcome: **NO_ADVANCE**. Arm decisions: `{"A": {"advance": false, "gates": {"adjusted_cluster_lcb_every_partition": false, "harmed_not_above_REF": false, "placebo_fails": true, "raw_invalid_not_worse": false, "secondary_three_positive": true, "three_positive_every_partition": true}}, "B": {"advance": false, "gates": {"adjusted_cluster_lcb_every_partition": true, "harmed_not_above_REF": false, "placebo_fails": true, "raw_invalid_not_worse": true, "secondary_three_positive": true, "three_positive_every_partition": true}}}`. Failed gates: `{"A": ["adjusted_cluster_lcb_every_partition", "harmed_not_above_REF", "raw_invalid_not_worse"], "B": ["harmed_not_above_REF"]}`. PLACEBO would pass the frozen qualification: `False`. No serving promotion follows automatically.

Setup: exact H272 topk/core reference at K8; H276 outcome-blind three k-means partitions and Ward k5 reused by exact hash. Fresh model seeds277101–277103; bootstrap277150.60 outer leave-cluster-out cells ×4 fitted policies=240policy receipts, zero neural updates. A applies inverse-cluster-family-count weights with within-family context normalization, giving every present cluster equal mass. B uses A's rankers and chooses tau from.05/.10/.15/.20 by nested inner-family CV. H273's selected-candidate veto is preserved, including p>tau, baseline immunity and class weights; its thread count is1. REF is descriptive, not a tested candidate. PLACEBO permutes cluster inverse weights within FIT only.

All-clustering primary results versus BASE, higher utility better. Per-seed gain is the equal-family mean; the reported statistic is the median across three seeds. No partition was selected after outcomes.

| Clustering | Arm | Median gain | Per-seed gains | 95% cluster CI | Holm-compatible LCB | Positive seeds | MM BT Elo |
|---|---|---:|---|---|---|---|---:|
| KM276140 | BASE | 0 | 0,0,0 | 0,0 | reference | 0/3 | 1500.000 |
| KM276140 | REF | +0.049328703 | +0.048124616, +0.071262523, +0.049328703 | +0.020754580, +0.087740085 | descriptive | 3/3 | 1546.929 |
| KM276140 | A | +0.049864033 | +0.018162443, +0.049864033, +0.054699250 | +0.005408066, +0.066809611 | -0.003125490 | 3/3 | 1548.286 |
| KM276140 | B | +0.043444018 | +0.036963520, +0.043444018, +0.060957024 | +0.017486244, +0.078534281 | +0.014710562 | 3/3 | 1543.115 |
| KM276140 | PLACEBO | +0.046495695 | +0.046495695, +0.067513876, +0.044540377 | +0.016433446, +0.103622579 | descriptive | 3/3 | 1547.146 |
| KM276141 | BASE | 0 | 0,0,0 | 0,0 | reference | 0/3 | 1500.000 |
| KM276141 | REF | +0.057332728 | +0.057332728, +0.071841880, +0.045773752 | +0.012679834, +0.115298312 | descriptive | 3/3 | 1549.841 |
| KM276141 | A | +0.034398025 | +0.065809453, +0.034398025, +0.023191391 | -0.023525373, +0.090107427 | -0.029962550 | 3/3 | 1550.212 |
| KM276141 | B | +0.061735276 | +0.061735276, +0.056972750, +0.064224491 | +0.031379067, +0.106856761 | +0.030137144 | 3/3 | 1551.145 |
| KM276141 | PLACEBO | +0.021642363 | +0.030236936, +0.021642363, +0.003860562 | +0.000987751, +0.059115934 | descriptive | 3/3 | 1520.160 |
| KM276142 | BASE | 0 | 0,0,0 | 0,0 | reference | 0/3 | 1500.000 |
| KM276142 | REF | +0.056950888 | +0.056950888, +0.047169997, +0.071871887 | +0.025678953, +0.101267986 | descriptive | 3/3 | 1546.206 |
| KM276142 | A | +0.054514626 | +0.054514626, +0.033408231, +0.064366866 | +0.033273223, +0.076436514 | +0.029504858 | 3/3 | 1549.395 |
| KM276142 | B | +0.061856401 | +0.061856401, +0.056551492, +0.068362305 | +0.031516279, +0.080107881 | +0.030772246 | 3/3 | 1550.168 |
| KM276142 | PLACEBO | +0.038377562 | +0.047011037, +0.036084978, +0.038377562 | +0.012577052, +0.068467176 | descriptive | 3/3 | 1537.529 |
| WARD | BASE | 0 | 0,0,0 | 0,0 | reference | 0/3 | 1500.000 |
| WARD | REF | +0.056154284 | +0.065794205, +0.056154284, +0.040484435 | +0.020966977, +0.087124157 | descriptive | 3/3 | 1553.516 |
| WARD | A | +0.019083838 | +0.011186987, +0.019083838, +0.026929063 | +0.000330624, +0.040337475 | +0.000000000 | 3/3 | 1522.370 |
| WARD | B | +0.052969679 | +0.052969679, +0.040311277, +0.061503762 | +0.021755564, +0.079170097 | +0.020611875 | 3/3 | 1543.641 |
| WARD | PLACEBO | +0.036096263 | +0.052841095, -0.001455759, +0.036096263 | -0.003175475, +0.069684521 | descriptive | 2/3 | 1530.165 |

The paired10k cluster bootstrap resamples five whole clusters with replacement and retains their member-family multiplicities, then takes the median of three seed effects. Holm m2 covers A/B within each partition. The reported lower bound is the conservative simultaneous97.5% bound; it is not an exact inverted Holm interval. ADVANCE requires success in EVERY clustering, an intersection-union condition. All gates were frozen before outcomes: three strictly positive seed effects (>1e-12) in each partition; positive adjusted cluster lower bound in each; raw−2 no greater than BASE per partition pooled across seeds; harmed-family count no greater than REF for each partition/seed; three positive DL-Val effects; PLACEBO fails its analogous qualification. Ties and fallback identity are not improvement or equivalence. Placebo gates: `{"adjusted_cluster_lcb_every_partition": false, "harmed_not_above_REF": false, "raw_invalid_not_worse": false, "secondary_three_positive": true, "three_positive_every_partition": false}`.

| Clustering | Arm | Raw−2 picks | Harmed families per seed (>1e-4 loss) | Family W/T/L per seed |
|---|---|---|---|---|
| KM276140 | BASE | 51/3657 | [0, 0, 0] | 0/205/0; 0/205/0; 0/205/0 |
| KM276140 | REF | 45/3657 | [55, 41, 49] | 135/15/55; 140/24/41; 122/34/49 |
| KM276140 | A | 47/3657 | [62, 58, 58] | 135/8/62; 135/12/58; 141/6/58 |
| KM276140 | B | 42/3657 | [48, 52, 50] | 116/41/48; 138/15/52; 139/16/50 |
| KM276140 | PLACEBO | 51/3657 | [65, 53, 68] | 134/6/65; 135/17/53; 130/7/68 |
| KM276141 | BASE | 51/3657 | [0, 0, 0] | 0/205/0; 0/205/0; 0/205/0 |
| KM276141 | REF | 52/3657 | [49, 43, 62] | 138/18/49; 137/25/43; 130/13/62 |
| KM276141 | A | 58/3657 | [69, 60, 68] | 134/2/69; 124/21/60; 127/10/68 |
| KM276141 | B | 38/3657 | [68, 60, 60] | 133/4/68; 138/7/60; 138/7/60 |
| KM276141 | PLACEBO | 54/3657 | [39, 37, 47] | 82/84/39; 74/94/37; 58/100/47 |
| KM276142 | BASE | 51/3657 | [0, 0, 0] | 0/205/0; 0/205/0; 0/205/0 |
| KM276142 | REF | 45/3657 | [55, 51, 58] | 132/18/55; 134/20/51; 136/11/58 |
| KM276142 | A | 53/3657 | [63, 46, 54] | 139/3/63; 109/50/46; 139/12/54 |
| KM276142 | B | 32/3657 | [57, 49, 45] | 144/4/57; 131/25/49; 149/11/45 |
| KM276142 | PLACEBO | 50/3657 | [50, 48, 43] | 132/23/50; 96/61/48; 112/50/43 |
| WARD | BASE | 51/3657 | [0, 0, 0] | 0/205/0; 0/205/0; 0/205/0 |
| WARD | REF | 47/3657 | [47, 43, 46] | 135/23/47; 138/24/43; 127/32/46 |
| WARD | A | 49/3657 | [20, 33, 36] | 37/148/20; 85/87/33; 72/97/36 |
| WARD | B | 38/3657 | [52, 44, 59] | 134/19/52; 124/37/44; 139/7/59 |
| WARD | PLACEBO | 54/3657 | [34, 45, 33] | 113/58/34; 67/93/45; 91/81/33 |

DL-Val91 is heavily reused and is **supporting only**. It averages the five KM276140 outer-policy utilities per context/seed, then averages families equally. It is not independent external validation.

| Arm | Median family gain | Per-seed gains | 95% family CI | Positive seeds | MM BT Elo | Raw−2 picks |
|---|---:|---|---|---|---:|---|
| REF | +0.023315711 | +0.039809713, +0.023315711, +0.019028347 | +0.011552473, +0.038963563 | 3/3 | 1558.281 | 0/1365 |
| A | +0.043441205 | +0.047704361, +0.043441205, +0.034585103 | +0.021766852, +0.065775705 | 3/3 | 1608.841 | 0/1365 |
| B | +0.030574704 | +0.021165241, +0.036425131, +0.030574704 | +0.015438801, +0.046498201 | 3/3 | 1592.371 | 0/1365 |
| PLACEBO | +0.035300854 | +0.045262052, +0.035300854, +0.029507924 | +0.020564772, +0.053212540 | 3/3 | 1587.355 | 0/1365 |

Elo anchors BASE1500 in the five-arm competition; all per-cluster gains, Holm p-values, family W/T/L and harms remain in [metrics.json](metrics.json). These policy-utility averages do not define one deployable model or a prediction ensemble. No AUC or speed verdict.

Self red-team: H276 outcomes were not read before preregistration; its assignments are fixed, including the singleton KM276141 cluster. Five clusters give weak and discrete uncertainty, especially with that outlier; fresh seeds do not make reused families independent. Cluster descriptors use global outcome-blind TRAIN features, not held-out labels. Shared feature archives decode both visible cohorts; actual FIT input/weight receipts and family-split checks bind training to TRAIN rows. Tau, ranker hyperparameters and margins use inner OOF only, never outer or DL-Val labels. Class-weighted veto scores are not claimed calibrated probabilities. PLACEBO preserves labels/features and can improve; passing it blocks both candidate arms rather than triggering a revised null. No H276/H277 post-outcome retry or variant is used. No sealed/holdout/meta_test or live-worker data path was opened. Native91 reproduced91/91 and1.1377941898164525 before fitting. DOPE `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` and r882 `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64` match before/after; checkpoint copy only, zero neural updates. All60 selections were locked before trained validation; cached baseline retains native support conditioning and no extra test-time reveals were added.

Verification:19 precompute tests (8H272+11H277, including exact H276 manifest identity, real synthetic weighted LightGBM/classifier fits, outer-label noninterference, gate boundaries and resource rules), Ruff, exact-source/JSON/index/scope checks. Saved-evidence replay: 1576 FIT input/weight receipts, 180 weighted inner grids, 300 outer/VAL models, exact panels and frozen statistic recomputation. The summarizer is reused for recomputation, not described as an independent implementation. GuardPASS, peak RSS756465664bytes, monitored threads3, CPU48,50–52/nice19/numeric1. Swap-in pressure, memory/load/RSS/disk and23:30Z deadline were monitored without controlling foreign processes.

Execution source `1e8af0396cc4ac368b2c7505853ea9262ce378a6`. Full outputs/source/folds preserved before compact 34975-byte metrics. [Cluster manifest](folds.json), runtime `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H277/astra-t2055/closeout/runtime.tar.gz`, VersionId `_x4Lh97cR0MbklaN0OpJjBfTNig161tQ`, SHA256 `53410a49efdf8b24d1cbe2ab562a279cea7ef261f30c7f64ff7ee50e63c599fa`; all525 members exact-version GET verified. Same-agent review at the exact published head; hostedCI NOT_RUN under split/hold; v1/next only.

Next step: retain the failed qualification, make no adoption claim, and do not reroll seeds or thresholds from these outcomes.

## Deviations
None to the frozen scientific design. H273 n_jobs2→1 is the preregistered resource-only adjustment. Full outputs precede compact packaging; operational receipts preserve any resource pauses. See [PREREG.md](PREREG.md) and [manifest.json](manifest.json).
