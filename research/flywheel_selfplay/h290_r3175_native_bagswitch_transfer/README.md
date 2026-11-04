# H290: r3175-native BAG_SWITCH transfer

**COMPLETE — NOT_TRANSFERRED_SEMI_FRESH.** The r3175-native model does not qualify under the frozen transfer gate. No serving change or fresh-confirmation qualification was earned.

## Hypothesis and setup

The hypothesis was that refitting BAG_SWITCH on r3175's own candidate distribution would improve transfer relative to the frozen r882-fit H284 reference. Git and S3 H280–H289 inventory found no full-DL-Train r3175-native export: H280 retained outer-fold models; H284/H285 exports were r882-fit references. H290 therefore fitted all 1,219 DL-Train contexts / 205 families using H280's recorded CPU top-8 packets and exact 568-column features.

H279 `bagged.py`, H278 `topk.py`, H265 core and H273 veto were reused byte-identically. B=8 family-bootstrap rankers, q=6, tau=0.10 and the strict H265 margin/inner-selection procedure are unchanged. Seeds290101–103 used grouped four-fold inner training CV. Native and within-family label-permuted placebo proposer/voters were fitted once and every serialization reloaded bit-exactly on all training contexts. The placebo shares the truthful training-only veto and hyperparameter selection, as H279 specifies; it is a conditional permutation control.

The CPU DL-Val91 regression reproduced **1.1366829009431036 exactly**. H280 cell KM276140_280101_0 reproduced its selected parameters, picks, ranker and gate hashes exactly. This authorized old DL-Val regression preceded fitting; H286/H283 evaluation packets remained unopened until the fit-lock was written and archived. Fit-lock SHA256: `9c1405fb1509cce8c104626cff39e4da65d27c3a3634483a824ebe7fccbd3abe`.

The [preregistration](PREREG.md) was published at `53df9ffc0f0c3d0da6a445a2947ae813f4f39123` before the single scoring attempt. PRIMARY H286 v2 has **200/200 families**. SECONDARY H283 has **29/60 built families**, with 31 already unbuilt in its published archive. That structural coverage clarification was fixed after fit-lock and before scoring; the [original training protocol](FIT_PROTOCOL.md) is retained. Missing secondary pools were not assigned zero outcomes. Both cohorts were opened by earlier studies: these results are **SEMI_FRESH**.

All evaluated features and top-8 orders reproduced bit-exactly from the published GPU costs and recorded pools. Both parent GPU receipts recorded exact r3175 DL-Val91 regression, 91/91 picks and utilities. H290 also reproduced each cohort's slot0 utility and found no training context/dataset overlap. No ODD/v3/reserved-remainder packets were opened. CPU-training/GPU-test drift remains a limitation: no matched CPU/GPU packet pairs exist for these test contexts, so affected-feature/pick fractions are **NOT_MEASURABLE**, not zero. No new forwards were made to estimate them.

## Primary outcome: H286 v2, 200 families

Average contexts within family, then families equally. Report the median of three model-seed gains and a 10,000-draw paired family-bootstrap CI (seed290150); the bootstrap takes the median of the three seed means within each draw. Model seeds are not independent datasets. Elo is the six-arm MM Bradley–Terry comparison, BASE=1500, 400 iterations, prior0.01 and tie band1e-4; its CI uses the same family draws.

| Arm | Median gain vs BASE | 95% CI | Family-median gain | MM BT Elo [95% CI] | Harmed >0.01, per seed | Raw −2, per seed |
| --- | ---: | --- | ---: | --- | --- | --- |
| BASE | +0.000000 | [+0.000000, +0.000000] | +0.000000 | 1500.000 [1500.000, 1500.000] | [0, 0, 0] | [4, 4, 4] |
| NATIVE | +0.041404 | [+0.006211, +0.107237] | +0.000000 | 1524.114 [1500.580, 1550.316] | [24, 32, 19] | [4, 2, 2] |
| R882_FIT | +0.062995 | [+0.018938, +0.116735] | +0.000000 | 1528.212 [1503.505, 1545.334] | [21, 7, 17] | [2, 2, 4] |
| PLACEBO | -0.010018 | [-0.036784, +0.025284] | +0.000000 | 1495.342 [1488.958, 1503.497] | [8, 6, 5] | [4, 4, 3] |
| NULL_RV | +0.027027 | [-0.034317, +0.092788] | +0.000000 | 1505.223 [1470.555, 1540.169] | [63, 69, 65] | [0, 0, 0] |
| NULL_LF | -0.054744 | [-0.152300, +0.026012] | +0.000000 | 1472.076 [1437.802, 1509.566] | [83, 69, 77] | [7, 4, 8] |

NATIVE's seed gains are **+0.040632 / +0.083613 / +0.041404**. Their CIs are **[−0.023172,+0.105666] / [+0.036529,+0.139850] / [+0.001693,+0.090130]**: positive LCB on **2/3**, not the required 3/3. Non-tied family sign-test p-values are **0.238344 / 0.007875 / 0.048977**, also **2/3** below0.05. Family W/T/L is **39/129/32**, **67/92/41**, **38/138/24**. Ties never count as wins.

Harmed-family counts **24/32/19** exceed the matched R882_FIT counts **21/7/17** on every seed. NATIVE's raw−2 counts **4/2/2** do satisfy BASE's **4/4/4** bound. Every primary arm's family-median gain is zero; a positive mean gain is not evidence of broad-family improvement.

The paired **NATIVE−R882_FIT** gains are **−0.028542 / +0.020618 / +0.005848**; their median is **+0.005848**, CI **[−0.026143,+0.029348]**. This predeclared median of paired differences is not the difference between the two marginal seed medians (+0.041404 versus +0.062995). The contrast is uncertain and does not establish that candidate-distribution mismatch alone caused prior transfer failures.

PLACEBO, validity-informed NULL_RV and label-free NULL_LF all fail the same primary transfer gate. NULL_RV's positive median is accompanied by CIs spanning zero, nonsignificant signs and more harmed families. NULL_LF does not use validity labels. No null draw was selected or repeated after observing utility.

## Secondary outcome: H283, 29 of 60 requested families

This incomplete secondary coverage is descriptive and never changes the primary verdict.

| Arm | Median gain | 95% CI | MM BT Elo [95% CI] | Harmed >0.01, per seed | Raw −2, per seed |
| --- | ---: | --- | --- | --- | --- |
| BASE | +0.000000 | [+0.000000, +0.000000] | 1500.000 [1500.000, 1500.000] | [0, 0, 0] | [0, 0, 0] |
| NATIVE | -0.085405 | [-0.286124, +0.090250] | 1584.140 [1510.037, 1658.235] | [1, 8, 3] | [0, 1, 1] |
| R882_FIT | +0.063616 | [+0.004972, +0.137694] | 1555.170 [1508.024, 1620.965] | [2, 1, 3] | [0, 0, 0] |
| PLACEBO | +0.000000 | [-0.000594, +0.000000] | 1500.000 [1483.589, 1500.000] | [0, 0, 1] | [0, 0, 0] |
| NULL_RV | -0.008553 | [-0.128384, +0.137295] | 1579.965 [1466.492, 1662.075] | [13, 9, 8] | [0, 0, 0] |
| NULL_LF | -0.217602 | [-0.368848, -0.029254] | 1524.296 [1407.548, 1607.282] | [11, 14, 12] | [1, 1, 2] |

NATIVE introduces raw−2 picks on two secondary seeds where BASE has none. Elo counts pairwise wins/ties and does not measure utility magnitude; its favorable secondary rating does not overturn the negative utility result or the primary gate failures.

## Verdict and next step

**NOT_TRANSFERRED_SEMI_FRESH**: the native model fails the all-seed LCB gate, all-seed sign gate and harm bound. The positive pooled CI, passing controls and positive median paired contrast cannot replace those requirements. Keep the native export as a shadow research artifact; do not advance or adopt it. These results motivate studying family harm before any separately scoped fresh confirmation. They do not prove either generalization or overfitting as the sole explanation of the transfer gap.

## Verification, deviations and custody

Six synthetic tests passed, including family weighting, ties, raw−2 versus gate-invalid accounting, control gates and exact MM equivalence to the inherited implementation. The saved-result audit independently checks all4,122 selected-slot/outcome joins, 87 unchanged model hashes, both 10,000-draw bootstrap index arrays, family CIs and exact binomial sign tests. Training code and models were unchanged after fit-lock. Audit/review are same-agent verification, not an external review.

Operational corrections occurred before scoring: the archive reader was adjusted for the published `fresh/` member prefix and H283's distinct rescore-receipt name; import formatting/lint fixes were completed before publication. Original receipts and code snapshots remain archived. There was one scientific fit attempt and one frozen scoring attempt, no outcome retry, no model retuning and no post-outcome gate change.

CPU48,50–52, nice19, numeric threads1; fit/scoring peaks were822,456,320 /660,905,984 bytes RSS and3 aggregate threads. Guards enforced RSS<=4GiB, aggregate<=4 and abort on MemAvailable<10GiB. **Zero neural forwards, zero neural updates, no GPU or checkpoint loads.** DOPE `c51a56d0`, writer1402163 and observer2849436 were untouched. Hosted CI is **NOT_RUN** under the owner research split; local tests/checks are not represented as hosted CI.

[Metrics](metrics.json), [manifest](manifest.json), [fit summary](fit-summary.json), [fit custody](fit-custody.json), [prepared input custody](input-custody.json), [output custody](output-custody.json) and [audit](audit.json) preserve the exact source, model and data identities. Full outcomes and diagnostics were archived with exact-VersionId GET and member-hash verification before producing compact metrics. Inputs/model artifacts are transitively recoverable through their immutable manifests and VersionIds. All new keys are create-only beneath `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H290/astra-t0030/`.

Publication is non-force to `origin/v1/next` only, with both H290 index rows `GIT_COMMITTED`. Final lifecycle receipts and recovery/reclamation evidence are retained under `lifecycle/` and local custody `/home/ubuntu/jope-rescue-20260921/h290-astra/`. No main/fastlane merge or CI dispatch.
