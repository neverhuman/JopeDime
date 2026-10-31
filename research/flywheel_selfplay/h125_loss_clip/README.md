# H125 — per-context loss clipping and normalization

**NO_ADVANCE**. CPU-only offline r882 fine-tuning under [PREREG.md](PREREG.md), published in [PR #1083](https://github.com/neverhuman/JopeDime/pull/1083) before execution. Status: COMPLETE; optimizer updates: 360. No serving promotion.

The fresh family partition contains 164 fitting families/978 contexts and 41 assessment families/241 contexts, all from DL-Train. Seeds 67–69 are paired across CONTROL, CLIP and NORM. Each cell starts from its own SHA-verified r882 copy; only the 29,211 adapter parameters train. H97/H105 batch 16, slate 192, 20 rounds × 2 AdamW updates and the H61 unrevealed-candidate evaluator are unchanged.

| Arm | Seed assessment utilities (67,68,69) | Median gain vs r882 | Family-bootstrap 95% CI | Median Elo (r882=1500), 95% CI | Harm fraction | Holm | Verdict |
|---|---|---:|---|---|---:|---|---|
| CONTROL | 0.805985079, 0.806394923, 0.811887793 | -0.063180126 | [-0.126558, 0.002110] | 1422.5, [1336.4, 1517.0] | 14.94% | n/a | DIAGNOSTIC |
| CLIP | 0.808773924, 0.809357874, 0.807233804 | -0.060801125 | [-0.127817, 0.004082] | 1404.4, [1314.8, 1500.0] | 17.01% | fail | NO_ADVANCE |
| NORM | 0.808545030, 0.814521531, 0.813844966 | -0.055730083 | [-0.123460, 0.008802] | 1376.4, [1267.0, 1457.4] | 19.92% | fail | NO_ADVANCE |

Fresh r882 assessment utility: **0.869575049160**. Full-precision values, seed gains, ties and Elo limits are in [metrics.json](metrics.json). All nine cells are included; ties do not pass the strict-positive gain/LCB gates. CONTROL is diagnostic and excluded from the two-arm Holm test.

Descriptively, median paired gains versus trained CONTROL were +0.002788845 for CLIP and +0.002559951 for NORM. This does not change their failed primary comparisons against unchanged r882. The trained CONTROL itself declined by 0.063180126 median utility; the preregistered regression gate concerned unchanged checkpoint copies before optimization.

Unchanged-r882 regression passed with zero prediction and utility differences across two copied checkpoints, and zero CONTROL loss/gradient differences against H59; [regression.json](regression.json). The frozen CPU tokens are shared across cells because all their generating weights are frozen. Native landmark feature extraction is part of the existing recipe; no new GP candidate fitness evaluation or search was run.

Before the first update, the numeric [calibration addendum](CALIBRATION_PREREG.md) and its full fit-only loss/slate/denominator ledger were created in S3 and read back by exact VersionId/SHA. CLIP q90=8.061488056182863 (98/978 baseline contexts above it); no NORM denominator needed the 0.1 floor. The builder leaves the model in eval mode; this preserves the actual H105 recipe while adapter gradients remain enabled.

External monitoring: peak process-tree RSS **4.130GB**, elapsed **1123.7s**. Four math/Torch threads were configured; seven to eight process threads observed, effective nice 19 (at least the requested nice 15). Owner stops: 8 GB RSS, disk>95%, load>86.7/102; added MemAvailable floor 16 GiB. Monitor disposition `WORKER_EXITED`, worker exit `0`. No CUDA, GPU lease or qualification.lock; audited reads restricted to the admitted DL-Train pool/table IDs. Original r882 and input files were not written or deleted.

Interpretation: r882 was previously exposed to the full DL-Train cohort. The new family partition excludes assessment families from H125 updates and calibration; it does not establish unseen-family generalization. Family-bootstrap lower-tail areas are the preregistered exploratory Holm inputs, not a separately calibrated null-test proof. Repeated seed vectors are explicitly flagged. All 257 dataset IDs match origin=real in exact-version S3 catalog v246 and the catalog exclusion checks pass; this is not new original-source-row-lineage qualification.

Validation: 10 synthetic tests cover whole-context clipping gradients, detached normalization, strict ties, harm veto, complete-cell requirements, family partition and resource boundaries; all pass. Real regression and actual optimizer/DOPE/checkpoint/read-audit evidence are archived. No historical mixed-result summary values were used; the initial seed search accidentally displayed some historical summary lines, disclosed in the archived search-boundary note, before narrowing to runner/prereg/protocol files.

Evidence: [custody.json](custody.json), [manifest.json](manifest.json), [REPRO_BUNDLE.md](REPRO_BUNDLE.md). Create-only manifest `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H125/astra-t0955/manifest-v2.json`, VersionId `YxMNr.xPc8JYEWyREIA7_a5IBNJXSyxN`, SHA256 `50cc7893b3d7285f968865f0700c3777c2f7a8d8023d5395d6542e71276cffb7`. Archive has 86 exact-version restored/hash-verified members. Bulk outputs remain outside Git; no checkpoint/dataset/receipt deletion.
