# H293 — robust family-balanced switch fitting

**ROBUST_NOT_ADVANCED**. Complete single CPU attempt; frozen primary exported either way.

Hypothesis: family-balanced weights and per-context normalized rank targets improve family-median gain and harm while retaining >=80% of CLIP mean gain. Setup: 205 DL-Train families / 1,219 contexts, model seeds 293101–293103, H287 fourfold family splits, fixed H287 tree/veto/eight-bag policy. HUBER delta selected inside each outer training set only. The primary is FAMBAL_RANKNORM; all other fitted arms are descriptive and PLACEBO/NULL_LF are controls.

H287 CLIP seed 287101 regression reproduced all 1,219 held-out utilities bit exactly: equal-family gain 0.09355105605170717. All 75 policies were fit and saved in 750 SHA-locked booster files (including shared-veto copies) before H293 outer scoring. Prereg commit `4713d1a4460732449a952c7887b31c6294542d36` was published at 2026-11-04T01:50:38.943936+00:00; scoring began at epoch 1793757048.941008. All three primary full-training exports (30 boosters) passed actual in-memory versus disk reload checks on their own training features.

Outcome: primary median equal-family gain is **+0.071449 [0.030418, 0.116774]**, versus CLIP **+0.065795 [0.029855, 0.115752]** (108.6% retained). The failures are **median harmed families 35 versus 30** and **median top-four share 0.433449 versus 0.351050**. Positive family medians and positive LCBs occur on all three primary seeds, but CLIP already has positive family medians on all three seeds too. Both controls fail their gates. These observations do not establish the hypothesized harm/concentration improvement.

| Arm | Median family-equal gain [95% CI] | Family-median gains by seed | Harmed families by seed | Top-four shares by seed | Median Elo [95% CI] |
|---|---|---|---|---|---|
| slot0 | 0.000000 [0.000000, 0.000000] | 0.000000, 0.000000, 0.000000 | 0, 0, 0 | undefined, undefined, undefined | 1500.000000 [1500.000000, 1500.000000] |
| CLIP | 0.065795 [0.029855, 0.115752] | 0.012636, 0.000931, 0.013635 | 30, 22, 34 | 0.337901, 0.418497, 0.351050 | 1664.369916 [1618.610312, 1699.696154] |
| FAMBAL | 0.043964 [0.020266, 0.076499] | 0.006984, 0.000000, 0.008957 | 38, 11, 27 | 0.154240, 0.268712, 0.333634 | 1625.038668 [1588.511314, 1656.846212] |
| RANKNORM | 0.085338 [0.040186, 0.148021] | 0.018016, 0.010521, 0.019193 | 37, 28, 21 | 0.324136, 0.373804, 0.361761 | 1687.507405 [1644.380739, 1734.285387] |
| HUBER | 0.057229 [0.028588, 0.109634] | 0.003424, 0.000000, 0.006122 | 28, 16, 18 | 0.484070, 0.420100, 0.408400 | 1615.248924 [1588.828997, 1653.874737] |
| FAMBAL_RANKNORM | 0.071449 [0.030418, 0.116774] | 0.012653, 0.004345, 0.013822 | 40, 32, 35 | 0.250684, 0.571070, 0.433449 | 1649.271433 [1605.722391, 1694.644406] |
| PLACEBO | -0.021902 [-0.039112, 0.008807] | 0.000000, 0.000000, 0.000000 | 51, 24, 50 | undefined, 1.143782, undefined | 1483.385235 [1468.432445, 1510.814041] |
| NULL_LF | -0.013740 [-0.051172, 0.024418] | 0.000000, 0.000000, 0.000000 | 76, 58, 60 | undefined, 3.822972, undefined | 1508.398356 [1477.752850, 1537.932710] |

Primary gates: median_gain_retains_80pct_clip=True; lcb_positive_3of3=True; family_median_positive_2of3=True; harmed_median_lower=False; top4_share_median_lower=False; raw_minus2_no_worse_3of3=True. PLACEBO gate=False; NULL_LF gate=False.

All per-seed CIs, trimmed means, sign tests excluding ties, exact raw -2 counts, switches and Elo are in [metrics.json](metrics.json). The bootstrap uses 10,000 paired family draws (seed 293150), sharing draws across arms and seeds; the median CI is the bootstrap distribution of the three-seed median. Elo is MM Bradley–Terry, slot0=1500, tie band 1e-4, symmetric .01 smoothing. Concentration is H289's signed sum for the four largest absolute context changes divided by net context gain, not a family-weighted share. Nonpositive totals are undefined.

Code-level limitation: H287 CLIP clips selection utilities but leaves ordinal fit targets unchanged. Here rank-normalized and Huber-score-clipped targets enter LambdaRank through explicit linear gains; HUBER is target winsorization, not a Huber regression loss. Family-total-one weighting also normalizes bootstrap multiplicity and changes regularization scale, while score margins remain fixed. H287 configurations were historically selected on DL-Train and reused; the outer estimates are internal robustness evidence, not untouched fresh tests. See [PREREG.md](PREREG.md) for the frozen definitions.

Next: retain this negative robustness result and frozen exports; no fresh-family launch or adoption is earned. A different objective or margin/weight-scale treatment requires a new preregistered wave.

No new neural forwards/updates, no GPU or writer changes, and no forbidden fresh/validation/test/holdout payloads. Numeric 1/aggregate<=4/RSS<=4GiB/MemAvailable>=10GiB guards passed. Main, fastlane and hosted CI remain held; this publication changes only the three authorized study files and two H293 index rows.

Source/model lock SHA256 `712a1e3ea6360ffe09774220573292d411c1495a5bcbd938d91ee1b95a8a73c6`; exact S3 VersionIds, SHA256s and member manifests are in [metrics.json](metrics.json). Full source/models: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H293/astra-t0125/fit-source-models.tar.gz`, VersionId `qscjKf9iKiExpjuMIwEj5aLGvRLFPUhA`. Full results: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H293/astra-t0125/results-full.tar.gz`, VersionId `TXpIJpnVdPMoSocUAmRhC5KqZ7Enui0t`. All archives were restored through exact-version GET and every member SHA checked.

Verification precision note: the initial regression receipt/prereg quoted 0.09355105605170722 from averaging context differences. H287's original reduction averages selected and baseline utilities first and yields **0.09355105605170717**, a 4.16e-17 difference. The 1,219 selected utilities were bit-exact; the original reduction is reproduced in the versioned verification supplement. Independently recomputed H293 family means differ by at most 8.881784197001252e-16 across reduction orders (tolerance1e-12). No H293 metric, model, source file or decision changed.
