# H274 — Unconditional selector decomposition

Hypothesis: prospectively selected CTXNORM_UNC improves over r882 and beats both label- and feature-permutation placebos while respecting raw-invalid and family-harm budgets. Setup: H271 execution source68f517d63's byte-identical H268 scientific closure, fresh family/model/bootstrap seeds; three partitions ×five outer folds ×three model seeds ×four fitted arms =180 cells. Nine reporting panels, zero neural updates. DL-Train1,219 contexts/205families; DL-Val91/19families. Feature placebo permutes training-only context blocks within each family while retaining true labels, candidate rank and original evaluation features.

Outcome: **NOT_CONFIRMED / NO_ADVANCE**. Failed frozen gates: raw_invalid_not_worse. Higher utility is better. Primary and secondary below are CTXNORM_UNC−CONTROL, named before the sole run. DL-Val91 has been reused heavily across H26x; **SECONDARY is supporting only**. Fresh randomization of these same visible families is not independent external-data validation.

| Endpoint | Median family gain | Per-seed gains | 95% CI | Conservative Holm-compatible LCB | Holm one-sided p |
|---|---:|---|---|---:|---:|
| PRIMARY | +0.036263281 | +0.036263281, +0.035444530, +0.037447223 | +0.007297622, +0.070678397 | +0.003021491 | 0.007399 |
| SECONDARY | +0.021130936 | +0.021130936, +0.025789967, +0.020324628 | +0.008713070, +0.038128013 | +0.007514369 | 0.000200 |

Holm m=2 covers these two endpoints only. The bound inherits the conservative simultaneous97.5% interval, not an exact inverted Holm interval. Both placebo gates require positive median and all three gains>0; SECONDARY requires3/3positive. Raw-invalid budget pools all9 TRAIN policy repetitions; harm budget compares medians of the three per-seed family-harm counts after averaging partitions. CTXNORM_UNC harms [66, 70, 70] versus strict CTXNORM [67, 64, 67], permitted median increase10. Frozen gates: `{"feature_signal_positive": true, "harmed_family_budget": true, "label_signal_positive": true, "primary_holm_lcb": true, "primary_margin": true, "primary_three_positive": true, "raw_invalid_not_worse": false, "secondary_three_positive": true}`. No added p-value, placebo-CI or secondary-CI gate.

Prespecified decomposition, descriptive except the two primary placebo median/seed gates:

| Cohort | Paired contrast | Median | Per-seed gains | 95% CI |
|---|---|---:|---|---|
| PRIMARY | machinery: CTXNORM_SHUF_UNC - CONTROL | -0.018586465 | -0.004788334, -0.026617330, -0.018586465 | -0.034494031, -0.003584247 |
| SECONDARY | machinery: CTXNORM_SHUF_UNC - CONTROL | -0.011183322 | -0.011183322, -0.008980221, -0.017061143 | -0.028343550, +0.000017905 |
| PRIMARY | label signal: CTXNORM_UNC - CTXNORM_SHUF_UNC | +0.056033688 | +0.041051615, +0.062061860, +0.056033688 | +0.024339167, +0.084630025 |
| SECONDARY | label signal: CTXNORM_UNC - CTXNORM_SHUF_UNC | +0.034770188 | +0.032314258, +0.034770188, +0.037385770 | +0.015056031, +0.057431081 |
| PRIMARY | feature signal: CTXNORM_UNC - CTXNORM_FSHUF_UNC | +0.052763081 | +0.045536305, +0.052763081, +0.059009925 | +0.026945834, +0.081104409 |
| SECONDARY | feature signal: CTXNORM_UNC - CTXNORM_FSHUF_UNC | +0.027004597 | +0.023992465, +0.039141546, +0.027004597 | +0.007919308, +0.054567283 |
| PRIMARY | CTXNORM selector cost: CTXNORM_UNC - CTXNORM | +0.008110675 | +0.003440932, +0.008110675, +0.011980968 | +0.000126384, +0.017094632 |
| SECONDARY | CTXNORM selector cost: CTXNORM_UNC - CTXNORM | +0.002208033 | +0.000000000, +0.011685800, +0.002208033 | +0.000000000, +0.005163930 |
| PRIMARY | RERANK4 selector cost: RERANK4_UNC - RERANK4 | +0.010737539 | -0.000616982, +0.017480917, +0.010737539 | -0.000794162, +0.023593459 |
| SECONDARY | RERANK4 selector cost: RERANK4_UNC - RERANK4 | +0.005653422 | +0.006827667, +0.003519974, +0.005653422 | -0.001734985, +0.011967385 |
| PRIMARY | RERANK4 shared label null: RERANK4_UNC - CTXNORM_SHUF_UNC | +0.049530829 | +0.024102541, +0.055655997, +0.049530829 | +0.019957969, +0.078408443 |
| SECONDARY | RERANK4 shared label null: RERANK4_UNC - CTXNORM_SHUF_UNC | +0.031035183 | +0.031035183, +0.025762522, +0.035362093 | +0.009819030, +0.054763033 |
| PRIMARY | RERANK4 shared feature null: RERANK4_UNC - CTXNORM_FSHUF_UNC | +0.046357218 | +0.028587230, +0.046357218, +0.052507066 | +0.019597613, +0.070535093 |
| SECONDARY | RERANK4 shared feature null: RERANK4_UNC - CTXNORM_FSHUF_UNC | +0.024980920 | +0.022713390, +0.030133880, +0.024980920 | +0.003917051, +0.052147825 |

RERANK4 uses shared CTXNORM placebo arms with unmatched relevance labels; those comparisons are not a clean RERANK4-specific causal decomposition. No extra null fits were added. Medians of paired differences need not equal differences of arm medians. H271's +0.0547 CTXNORM_UNC−SHUF_UNC meant treatment beat the placebo, not the reverse.

| Cohort | Arm | Median gain vs CONTROL | 95% CI | Family W/T/L per seed | Harmed families per seed (>1e-4) | Raw−2 picks | Median MM BT Elo |
|---|---|---:|---|---|---|---|---:|
| PRIMARY | CONTROL | +0.000000000 | +0.000000000, +0.000000000 | 0/205/0; 0/205/0; 0/205/0 | [0, 0, 0] | 153/10971 | 1500.000 |
| PRIMARY | RERANK4 | +0.019931188 | -0.003835837, +0.039418549 | 116/13/76; 125/13/67; 117/16/72 | [76, 67, 72] | 159/10971 | 1520.205 |
| PRIMARY | CTXNORM | +0.027333856 | +0.005012933, +0.055506651 | 127/11/67; 127/14/64; 131/7/67 | [67, 64, 67] | 168/10971 | 1532.980 |
| PRIMARY | CTXNORM_SHUF | +0.000000000 | +0.000000000, +0.000000000 | 0/205/0; 6/197/2; 0/205/0 | [0, 2, 0] | 153/10971 | 1500.000 |
| PRIMARY | CTXNORM_FSHUF | +0.001121249 | -0.000205262, +0.002400205 | 13/185/7; 1/201/3; 7/194/4 | [7, 3, 4] | 153/10971 | 1500.625 |
| PRIMARY | RERANK4_UNC | +0.029038667 | -0.000654835, +0.058498739 | 121/5/79; 125/3/77; 124/3/78 | [79, 77, 78] | 153/10971 | 1532.236 |
| PRIMARY | CTXNORM_UNC | +0.036263281 | +0.007297622, +0.070678397 | 132/7/66; 127/8/70; 130/5/70 | [66, 70, 70] | 165/10971 | 1536.351 |
| PRIMARY | CTXNORM_SHUF_UNC | -0.018586465 | -0.034494031, -0.003584247 | 76/44/85; 73/19/113; 73/35/97 | [85, 113, 97] | 179/10971 | 1484.720 |
| PRIMARY | CTXNORM_FSHUF_UNC | -0.017318551 | -0.035193717, +0.000451413 | 87/34/84; 88/23/94; 86/13/106 | [84, 94, 106] | 177/10971 | 1493.265 |
| SECONDARY | CONTROL | +0.000000000 | +0.000000000, +0.000000000 | 0/19/0; 0/19/0; 0/19/0 | [0, 0, 0] | 0/1365 | 1500.000 |
| SECONDARY | RERANK4 | +0.013024193 | +0.002530941, +0.024456933 | 14/0/5; 13/0/6; 14/1/4 | [5, 6, 4] | 0/1365 | 1554.776 |
| SECONDARY | CTXNORM | +0.018116595 | +0.007134782, +0.033402079 | 15/1/3; 16/1/2; 13/1/5 | [3, 2, 5] | 0/1365 | 1571.051 |
| SECONDARY | CTXNORM_SHUF | +0.000000000 | +0.000000000, +0.000000000 | 0/19/0; 0/19/0; 0/19/0 | [0, 0, 0] | 0/1365 | 1500.000 |
| SECONDARY | CTXNORM_FSHUF | +0.000000000 | +0.000000000, +0.000000000 | 0/19/0; 0/19/0; 0/19/0 | [0, 0, 0] | 0/1365 | 1500.000 |
| SECONDARY | RERANK4_UNC | +0.018300950 | +0.002943237, +0.036323105 | 14/0/5; 14/0/5; 15/1/3 | [5, 5, 3] | 0/1365 | 1584.980 |
| SECONDARY | CTXNORM_UNC | +0.021130936 | +0.008713070, +0.038128013 | 15/1/3; 16/0/3; 16/0/3 | [3, 3, 3] | 0/1365 | 1588.021 |
| SECONDARY | CTXNORM_SHUF_UNC | -0.011183322 | -0.028343550, +0.000017905 | 8/0/11; 11/1/7; 9/0/10 | [11, 7, 10] | 3/1365 | 1514.722 |
| SECONDARY | CTXNORM_FSHUF_UNC | -0.006679969 | -0.021798285, +0.007515561 | 8/2/9; 9/0/10; 11/0/8 | [9, 10, 8] | 4/1365 | 1511.052 |

Elo uses canonical MM Bradley-Terry with CONTROL1500, equal-family outcomes and ties1e-4, jointly fitted over all nine panels. It is not directly comparable to a competition with a different arm set. Raw counts retain each repetition/seed/fold evaluation; averaged-panel invalid counts have a different denominator and remain separately in metrics. Compact [metrics.json](metrics.json) preserves all17 paired contrasts per cohort, per-seed gains/CIs, W/T/L, Elo, harms, raw-invalid arrays and selector summary. Full cell/family data are in S3.

| Fitted arm | Admitted | Rejected | False reject / rejected (rate) | False accept / admitted (rate) | Zero outer gains |
|---|---:|---:|---|---|---:|
| CTXNORM | 39 | 6 | 6/6 (1.000000) | 7/39 (0.179487) | 0 |
| CTXNORM_FSHUF | 5 | 40 | 12/40 (0.300000) | 2/5 (0.400000) | 1 |
| CTXNORM_SHUF | 1 | 44 | 12/44 (0.272727) | 0/1 (0.000000) | 2 |
| RERANK4 | 33 | 12 | 10/12 (0.833333) | 8/33 (0.242424) | 0 |

False reject means a rejected cell has positive outer equal-family UNC−CONTROL gain; false accept means an admitted cell has negative outer gain. Exact zeros are separate. All cell identifiers and seed/repeat breakdowns remain in the archived full metrics. These ASSESS diagnostics never affect fit, selection, retries or stopping.

UNC retains each model's inner-selected tree count and finite margin; only strict admission fallback is disabled. It may naturally retain r882. Prediction ties retain r882; fallback identity is NO_ADVANCE, never equivalence. Primary averages OOF utility across three partitions; secondary averages utility across five outer policies from foldseed274140. Each policy makes one pick; these aggregate estimands are neither one deployable checkpoint nor a prediction ensemble.

Self red-team: shared feature archives decode both visible cohorts, but fit input hashes and donor-map replay establish TRAIN-only fitting. Donors cannot cross inner SELECT or outer ASSESS families; labels stay at recipients. FSHUF preserves within-family information and candidate-rank structure; its contrast tests context-specific feature alignment, not removal of every predictive feature. Outer results never retune the frozen selector. All180 selections are globally locked before trained DL-Val scoring; Native91 baseline is the declared prior exception. Fresh seeds address randomization sensitivity but not repeated-dataset selection bias; all seed values and intervals are reported. UNC is confirmatory only because this preregistration named it, not because of current outcomes. CTXNORM changes the relevance package including available default exponential gains, not just normalization. No sealed/holdout/meta_test data were read; provenance pins276 real S3 dataset IDs with disjoint excluded IDs. Native91 reproduced91/91 and mean1.1377941898164525; checkpoint/DOPE remain unchanged, zero neural updates, read-only copy, no live worker. Cached baseline retains native support conditioning; no additional test-time reveals. OOM/disk/load/swap-in-pressure guards monitored only our own processes; no foreign pause/kill or swap manipulation. Guard transitions and timing remain in full receipts.

Verification:29 outcome-free tests (5H265+8H268+6H271+10H274), Ruff, exact Git/source/JSON/index/scope checks. Replay checked180 complete selector grids/2000-draw CIs,2458 fitted-input/seed receipts,98688 strict/UNC picks and240 saved-model predictions. All34 paired inference contrasts, Holm arithmetic, raw-invalid counts, harms/W/T/L and selector-rate arithmetic independently reproduced. Guard statusPASS; peak RSS849354752bytes, aggregate threads3; CPU48,50–52/nice19/numeric1, absolute22:30Z stop. No AUC or speed verdict.

Execution source `6f1470e3361bf6a7be2f64849fe7ddf60904e6e1`. Full folds are [version-pinned in S3](folds.json); complete outputs were stored and verified before packaging the39,960-byte compact metrics. Runtime `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H274/astra-t1955/closeout/runtime.tar.gz`, VersionId `ZT83GXPF8XpOccGfx.Gv44t2L6URWAdl`, SHA256 `9d7ee1b6aa4c67e0d0238d20b4e3e9de3b83312a28dcfb60a6af2ca19224346d`; all777 members verified by exact-version GET. Same-agent exact-head review; hostedCI NOT_RUN under CI_FREEZE/MASTER_MERGE_HOLD; only v1/next/own index rows.

Next step: retain the failed qualification, make no adoption or equivalence claim, and do not rerun based on these outcomes. See [PREREG.md](PREREG.md) and [manifest.json](manifest.json).

## Deviations

Resource guard amendment authorized by `hw_t2030_astra_h274_swapguard.md`; recorded2026-11-03T20:26:52.308144+00:00, applied2026-11-03T20:28:42.224426+00:00. Reason: Swap occupancy has remained full with negligible swap-in despite ample MemAvailable; occupancy alone was a false pause trigger. Original rule: pause if swap_free <=max(1MiB,0.1%total), resume only >=64MiB. Replacement: pause if psutil.swap_memory().sin delta / elapsed_seconds / OS page_size >2000 pages/s sustained30s; recover when rate<=2000. The rest of the H239-A1 memory, RSS4GiB, load108 pause, disk300GB, CPU/numeric/aggregate limits and22:30Z compute stop remain unchanged; closeout deadline23:15Z.

The paused monitor/runner were restarted within **the same attempt**, under the owner's explicit allowance, before any Native91 result, fitted cell or outcome artifact existed. Original empty directories and process/launch/resource/stdout receipts were preserved. Reason: Original monitor has no hot-reload interface. A fresh parent monitor must launch the child to retain its actual exit status; latest owner amendment explicitly permits same-attempt relaunch before outcomes. This is **not a post-outcome retry**. No outcomes were read to make the resource change; no scientific function, feature, label, seed, fold, model or decision gate changed. Scientific execution source remains `6f1470e3361bf6a7be2f64849fe7ddf60904e6e1`; amended resource-monitor SHA256 `53c0970368923d330e0f816a6af414cac4249a1f18f590f2baf0774eac2ff99c`. Six focused resource-rule tests pass in addition to the29 original tests. `metrics.json.resource_guard_change` preserves the authority, exact rules, timestamps, process continuity and hashes; complete custody includes the amended monitor and original receipts.
