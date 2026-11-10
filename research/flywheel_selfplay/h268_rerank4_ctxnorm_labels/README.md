# H268 — Context-normalized reranker labels

Hypothesis: robust per-context relevance labels reduce H265 seed instability while improving equal-family single-pick utility. Setup: byte-identical H265 features/model/selector; CONTROL, RERANK4, CTXNORM and within-family CTXNORM_SHUF. Three fresh family partitions × five outer folds × three model seeds × three fitted arms = 135 cells. DL-Train: 1,219 contexts / 205 families; DL-Val: 91 contexts / 19 families. Neural updates: 0. Cached r882 scores retain native support conditioning; the reranker adds no new test-time outcome reveals.

Outcome: **ADVANCE**. Native91 reproduced 91/91 and mean 1.1377941898164525 exactly; r882/DOPE unchanged. All 135 decisions were globally locked before trained DL-Val scoring. Fallback identity is NO_ADVANCE, not equivalence. The primary averages OOF utilities over three partitions; secondary averages five selected-cell utilities for fold seed 268140. These are multiple single-pick policies, not one deployable checkpoint or a prediction ensemble. Shared feature archives contain both visible cohorts; fitting selects only TRAIN rows. The lock governs trained validation scoring, not an absence of decoded validation feature bytes.

| Contrast | Median family gain | Per-seed family gains | 95% CI | Conservative Holm-compatible LCB | Holm p |
|---|---:|---|---|---:|---:|
| PRIMARY CTXNORM−CONTROL | +0.029964208 | +0.043631901, +0.029964208, +0.026404904 | +0.003086479, +0.064087310 | +0.000010358 | 0.025197 |
| SECONDARY CTXNORM−CONTROL | +0.013746892 | +0.019654378, +0.011222366, +0.013746892 | +0.001842384, +0.029271451 | -0.000233593 | 0.025197 |
| PRIMARY CTXNORM−RERANK4 (descriptive) | +0.022547095 | +0.027485343, +0.022547095, +0.014123738 | +0.006021751, +0.038197268 | — | — |
| PRIMARY CTXNORM−CTXNORM_SHUF | +0.029964208 | +0.045969537, +0.029964208, +0.026314909 | +0.003510342, +0.064087310 | — | — |

The Holm-compatible bounds use the preregistered conservative simultaneous 97.5% interval, not a newly inverted exact Holm interval. Confirmatory family comprises only PRIMARY and SECONDARY CTXNORM−CONTROL. Gates: `{"placebo_positive": true, "primary_holm_lcb": true, "primary_margin": true, "primary_three_positive": true, "secondary_three_positive": true}`. Strictly admitted cells: `{"CTXNORM": 42, "CTXNORM_SHUF": 5, "RERANK4": 28}`. No AUC or speed verdict.

| Cohort | CTXNORM harmed families by seed (>1e-4 loss) | CTXNORM harmed families by seed (>0.01 loss) | CTXNORM median MM BT Elo (CONTROL = 1500) |
|---|---|---|---:|
| PRIMARY | [70, 77, 64] | [49, 49, 39] | 1539.712 |
| SECONDARY | [5, 3, 3] | [2, 3, 3] | 1584.764 |

The qualification is narrow: PRIMARY's conservative adjusted lower bound is only +0.000010358. SECONDARY's corresponding lower bound is negative (−0.000233593); the frozen rule required three positive secondary seed gains, not a positive secondary adjusted bound. The independently recomputed bootstrap intervals and gate arithmetic match the saved result exactly.

The intended reduction in seed spread was **not observed** against the matched RERANK4 reference:

| Cohort | CTXNORM seed-gain range | RERANK4 seed-gain range |
|---|---:|---:|
| PRIMARY | 0.017226998 | 0.008729445 |
| SECONDARY | 0.008432011 | 0.003504947 |

Primary harmed-family counts also do not improve uniformly: CTXNORM [70,77,64] versus RERANK4 [81,65,63]. Across all three primary partitions and three seeds, raw invalid (−2) picks are CTXNORM 162, RERANK4 160 and CONTROL 153 out of 10,971 picks per arm. Secondary invalid picks are 0/1,365 for every arm. These descriptive shortcomings do not alter the frozen gain gate, but they limit the robustness claim.

Full per-family/per-seed harm tables, raw -2 counts, per-context choices, inner grids and selected models are in the verified S3 archive; compact [metrics.json](metrics.json) retains all contrasts, CIs, seed gains, Elo and raw invalid-count summaries. Invalid counts on an averaged panel mean all constituent policies invalid; raw per-repeat/per-fold counts are reported separately.

Validation: 13 data-free tests (all 5 H265 regressions plus 8 H268 tests), lint/format, exact Git source/JSON/index/scope checks. Independent replay verified 135 selectors, 1846 fitting input/seed receipts and 37008 choices, plus all 180 saved-model assessment/validation replays. Resource guard PASS; peak RSS 849575936 bytes, aggregate threads 3. CPU 48,50–52 / nice 19, numeric 1, H239-A1 and absolute 20:40Z stop. No process resume or second scientific attempt.

Execution source `ee85f557b4050bb696ac857e93b5ef3a383331ea`. Runtime archive `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H268/astra-t1812/closeout/runtime.tar.gz`, VersionId `XXFrHUTC5kkC6KGMOm5.3lwg_lUsEBMT`, SHA256 `52e46fc386bc01021ac08c043c47ee1635d48179ab75261671915c779ff2ff56`. Exact-version GET and every archive member verified. Full outputs were saved BEFORE the 50 KB compact-metrics packaging check. Same-agent exact-head review; hosted CI NOT_RUN per CI_FREEZE split. Published only to v1/next; no main/fastlane/PR2071 or release-critical changes.

Next step: consider a separately preregistered external confirmation of the label intervention; no serving promotion follows this result. Compare the matched RERANK4 reference before attributing any seed differences to normalization; the intervention also changes the available LightGBM relevance grades.

See [PREREG.md](PREREG.md), [manifest.json](manifest.json) and [fold_manifest.json](fold_manifest.json) for the frozen design and input identities.
