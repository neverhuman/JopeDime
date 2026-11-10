# H271 — CTXNORM confirmation

Hypothesis: the H268 gain replicates under fresh splits/seeds and beats a shuffled-label model when strict fallback is disabled. Setup: byte-identical H268 feature builder, label map, fitter and selector; three new family partitions × five outer folds × three model seeds × three fitted arms = 135 cells. Seven reporting panels reuse those fits. Data: DL-Train 1,219 contexts /205 families and DL-Val 91 contexts /19 families. Neural updates: 0.

Outcome: **NOT_CONFIRMED / NO_ADVANCE**. Failed frozen gates: primary_holm_lcb. The H268-style strict gate set fails; the additional unconditional-placebo gate has median gain +0.054716015 and 3/3 positive seeds. This is a confirmation attempt under new randomization of the same visible families, not independent external-data validation.

| Contrast | Median family gain | Per-seed family gains | 95% CI | Conservative Holm-compatible LCB |
|---|---:|---|---|---:|
| PRIMARY: CTXNORM - CONTROL | +0.021430455 | +0.015051461, +0.021430455, +0.034701355 | -0.000373664, +0.042736542 | -0.003456893 |
| SECONDARY: CTXNORM - CONTROL | +0.024809534 | +0.025868642, +0.024809534, +0.024089499 | +0.010130200, +0.046124468 | +0.008603774 |
| PRIMARY strict placebo: CTXNORM - CTXNORM_SHUF | +0.025781122 | +0.015051461, +0.025781122, +0.034591622 | +0.000638307, +0.046753598 | — |
| PRIMARY unconditional placebo: CTXNORM_UNC - CTXNORM_SHUF_UNC | +0.054716015 | +0.041744186, +0.054716015, +0.055217467 | +0.024903159, +0.088997352 | — |
| SECONDARY unconditional placebo (descriptive): CTXNORM_UNC - CTXNORM_SHUF_UNC | +0.041448106 | +0.041448106, +0.025374846, +0.041964320 | +0.015984851, +0.065551611 | — |
| PRIMARY reference (descriptive): CTXNORM - RERANK4 | +0.011304465 | +0.003123752, +0.011304465, +0.019055790 | -0.000061694, +0.021597431 | — |

Holm m=2 covers only PRIMARY and SECONDARY CTXNORM−CONTROL. Adjusted one-sided bootstrap p: PRIMARY 0.027797, SECONDARY 0.000200. The bounds use H268's conservative simultaneous 97.5% interval, not a newly inverted exact Holm interval. The frozen secondary gate requires three positive gains; the unconditional-placebo gate requires a positive median and three positive gains, without an added CI gate. All gates: `{"placebo_positive": true, "primary_holm_lcb": false, "primary_margin": true, "primary_three_positive": true, "secondary_three_positive": true, "unconditional_placebo_positive": true, "unconditional_placebo_three_positive": true}`.

| Cohort | Arm | Median gain vs CONTROL | 95% CI | Seed-gain range | Harmed families by seed (>1e-4 loss) | Raw −2 picks | Median MM BT Elo |
|---|---|---:|---|---:|---|---|---:|
| PRIMARY | RERANK4 | +0.011927708 | -0.005844268, +0.031366996 | 0.005519575 | [69, 78, 68] | 157/10971 | 1513.039 |
| PRIMARY | CTXNORM | +0.021430455 | -0.000373664, +0.042736542 | 0.019649894 | [71, 78, 82] | 153/10971 | 1528.074 |
| PRIMARY | CTXNORM_SHUF | +0.000000000 | -0.000204423, +0.000166445 | 0.004460399 | [0, 17, 6] | 153/10971 | 1500.000 |
| PRIMARY | RERANK4_UNC | +0.026289928 | -0.002974993, +0.057925840 | 0.017743463 | [71, 76, 77] | 145/10971 | 1527.599 |
| PRIMARY | CTXNORM_UNC | +0.034701355 | +0.003302525, +0.076054838 | 0.012116802 | [76, 71, 82] | 145/10971 | 1534.164 |
| PRIMARY | CTXNORM_SHUF_UNC | -0.009758498 | -0.028951742, +0.001857887 | 0.011612641 | [110, 101, 89] | 186/10971 | 1478.719 |
| SECONDARY | RERANK4 | +0.012692957 | +0.005813371, +0.019535364 | 0.005644461 | [4, 2, 3] | 0/1365 | 1543.120 |
| SECONDARY | CTXNORM | +0.024809534 | +0.010130200, +0.046124468 | 0.001779142 | [4, 3, 3] | 0/1365 | 1608.876 |
| SECONDARY | CTXNORM_SHUF | +0.000000000 | +0.000000000, +0.000000000 | 0.004781649 | [0, 5, 0] | 0/1365 | 1500.000 |
| SECONDARY | RERANK4_UNC | +0.020757511 | +0.008553112, +0.036165291 | 0.002131583 | [4, 3, 5] | 0/1365 | 1568.743 |
| SECONDARY | CTXNORM_UNC | +0.029889980 | +0.011625171, +0.051922888 | 0.007843599 | [4, 2, 3] | 0/1365 | 1610.908 |
| SECONDARY | CTXNORM_SHUF_UNC | -0.009515008 | -0.020615180, +0.003378242 | 0.022389954 | [10, 6, 13] | 2/1365 | 1492.532 |

CONTROL raw invalid counts: PRIMARY 153/10,971; SECONDARY 0/1,365. Counts above retain every repeat/seed/fold policy evaluation; invalid counts on averaged panels have a different denominator and are preserved separately. Elo is anchored to CONTROL = 1500 and fitted jointly over the seven reporting panels; it is not a direct comparison to H268's four-arm competition. All contrasts are paired per-seed differences; their medians need not equal differences of individual-arm medians. The strict secondary shuffled placebo has a zero median/CI because two seed utility panels equal CONTROL exactly, not because all policies are equivalent. The positive ordinary 95% lower bound for CTXNORM_UNC−CONTROL is descriptive and does not replace the strict-policy confirmation endpoint. Compact [metrics.json](metrics.json) retains per-seed gains, CIs, Elo, harm counts and raw invalid arrays for all contrasts. Full family/context detail remains in S3.

Strictly admitted cells (out of 45 per fitted arm): `{"CTXNORM": 39, "CTXNORM_SHUF": 3, "RERANK4": 34}`. UNCONDITIONAL disables only the strict admission fallback; it retains the same model, inner-selected tree count and finite margin. It may still choose r882 naturally. No extra fitted model or outcome-based retry was used. Primary scores average each context's OOF utility over three partitions; secondary averages utility over five outer policies for fold seed 271140. Neither is one deployable checkpoint or a prediction ensemble. Fallback identity is NO_ADVANCE, not equivalence.

Native91 reproduced 91/91 utilities and mean 1.1377941898164525 before fitting. r882 and DOPE hashes remained unchanged. All 135 selections were globally locked before trained DL-Val scoring. Cached r882 scores retain native support conditioning; no additional test-time reveals were introduced. Shared feature archives contain both visible cohorts, but fitting selects only TRAIN rows; the lock does not mean no validation feature bytes were decoded earlier.

Validation: 19 tests (5 H265, 8 H268, 6 H271), lint/format, exact-source/JSON/index/scope checks; scientific dependencies byte-identical to H268 execution source. Independent replay checked 135 selectors, 1850 fit-input/seed receipts and 74016 strict/unconditional picks, plus 180 saved-model replays. All 20 paired inference contrasts and confirmation arithmetic independently reproduced. Resource guard PASS: peak RSS 847609856 bytes, aggregate threads 3, CPU 48,50–52 / nice 19 / numeric 1, H239-A1, absolute 21:30Z stop. No AUC or speed verdict.

Execution source `68f517d63ab22c31902cfa0c0e0f837a49aeb255`. Full fold assignments are [version-pinned in S3](folds.json), not Git. Complete outputs were saved and verified before packaging the 39,759-byte compact metrics. Runtime archive `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H271/astra-t1905/closeout/runtime.tar.gz`, VersionId `PfchBITqRoJVxP24ZAWM1JFnesBX05hE`, SHA256 `95af6f62c4653bebaae2ebc3a09fefbf7c03b310b70a0ecbeac766f20e9c171e`; all 571 members verified from an exact-version GET. Same-agent exact-head review; hosted CI NOT_RUN under CI_FREEZE / MASTER_MERGE_HOLD. Publication is only to v1/next and own two index rows.

Next step: do not adopt H268 CTXNORM on this evidence; preserve the failed confirmation and await a separately authorized hypothesis, without outcome-selected retries. See [PREREG.md](PREREG.md) and [manifest.json](manifest.json).
