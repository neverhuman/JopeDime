# H191 — nested kernel coefficient selection

**NO_ADVANCE.** One zero-update CPU attempt completed. All adapter/DOPE weights remained byte-identical; no serving adoption. [metrics.json](metrics.json) preserves the exact run result.

H186's grid stopped at its upper edge without passing inference. [PREREG.md](PREREG.md) fixed six kernel multipliers, three literal family-fold reshuffles and 15 equal-family B-only choices. Every choice was locked before A/treatment DL-Val. KXSEL is the sole primary, tested on pooled A through imported H177/H178 with actual Holm m=1 and an additional strict median-margin/LCB gate. Fixed doses and DL-Val cannot rescue a failed primary.

| Policy | Role | A per-seed gains | A median gain [95% CI] | A Elo [95% CI] | DL-Val median gain [95% CI] | DL-Val Elo [95% CI] |
|---|---|---|---|---|---|---|
| KXSEL | primary | +0.01892532, +0.01675411, +0.00289565 | +0.01675411 [-0.01354413, +0.04260473] | 1500.13 [1492.05, 1510.74] | +0.00197912 [-0.01577595, +0.01982649] | 1528.58 [1491.23, 1573.68] |
| KX2.5 | descriptive dose only | +0.02169144, +0.02169144, +0.02169144 | +0.02169144 [-0.00793249, +0.05185675] | 1500.93 [1490.96, 1511.75] | +0.00214802 [-0.01682848, +0.02068857] | 1528.58 [1486.08, 1575.91] |
| KX3.0 | descriptive dose only | +0.01964067, +0.01964067, +0.01964067 | +0.01964067 [-0.01062796, +0.05047603] | 1503.49 [1492.16, 1515.49] | +0.00197912 [-0.01671204, +0.02005970] | 1531.52 [1491.23, 1579.36] |
| KX4.0 | descriptive dose only | +0.01624380, +0.01624380, +0.01624380 | +0.01624380 [-0.01756145, +0.04898681] | 1502.85 [1489.21, 1516.87] | +0.00696066 [-0.01609999, +0.02982967] | 1538.84 [1493.40, 1587.97] |

Imported H177 primary admission: `False`; Holm-m1 p `0.15958404159584041`. Combined median confirmation: `False`. Full decision and H178 receipt disposition are in metrics. R882 anchors Elo at 1500 in the five-policy panel; utility magnitude and win frequency differ.

| Reshuffle seed | B choices by fold 0–4 | Locked DL-Val mode |
|---|---|---|
| 191101 | 2.5, 1.0, 2.5, 3.0, 4.0 | 2.5 |
| 191102 | 4.0, 3.0, 1.5, 3.0, 2.5 | 3.0 |
| 191103 | 1.5, 4.0, 1.0, 2.0, 3.0 | 1.0 |

Selection counts across 15 cells: `{'1.0': 2, '1.5': 2, '2.0': 1, '2.5': 3, '3.0': 4, '4.0': 3}`. Modal DL-Val policies use only the five B choices for that seed, with smaller-coefficient ties.

| Policy | A context utilities by seed | DL-Val context utilities by seed (val_reg) | A exact -2 counts | DL-Val exact -2 counts |
|---|---|---|---|---|
| R882 | 1.423073435, 1.423073435, 1.423073435 | 1.137794190, 1.137794190, 1.137794190 | [17, 17, 17] | [0, 0, 0] |
| KXSEL | 1.433204004, 1.430053267, 1.425193709 | 1.139894404, 1.139718073, 1.137794190 | [12, 12, 13] | [0, 0, 0] |
| KX2.5 | 1.432692714, 1.432692714, 1.432692714 | 1.139894404, 1.139894404, 1.139894404 | [11, 11, 11] | [0, 0, 0] |
| KX3.0 | 1.430440363, 1.430440363, 1.430440363 | 1.139718073, 1.139718073, 1.139718073 | [12, 12, 12] | [0, 0, 0] |
| KX4.0 | 1.430744751, 1.430744751, 1.430744751 | 1.144918585, 1.144918585, 1.144918585 | [11, 11, 11] | [0, 0, 0] |

Context means are descriptive and differ from equal-family primary effects. Gate-invalid counts are retained separately. Fixed-grid seed panels identical: `True`; selected A panels identical: `False`. Seeds reshuffle families, not fitted models: three positive seeds would not be independent replication. Each A cell excludes its own B families, but families recur across other roles and were exposed by H186/r882 development. The paired family CI conditions on locked policies and does not reselect B within bootstrap draws. H178 calibrated original m4, not nested H191 m1; no new empirical FWER guarantee or sealed qualification is claimed.

Exact native91 baseline passed first (1.1377941898164525 bit-exact). Adapter SHA before=after `9c3ba940d9eb60aa2a39dd57c229c0952927ea1c2f22c372ae9764f43baa8669`; DOPE `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`; frozen-state hash `8d1070faf574ee1cd2a4e630887e1662b43189b3f89b04982143ef2704818aae`. Raw reconstruction verified 45522 grid cost/choice/label checks, all15 B decisions and projections, baseline and resource limits; the registered statistical implementation was re-executed for one primary selector and eight contrasts/Elo, not independently reimplemented. Replay executed zero model forwards/updates.

Next: retain r882; no eligible serving promotion or alternative-primary rescue is claimed.

Run source `5df97ad920ed630e86129599c80b5feb6efd5876`, prereg [PR #1581](https://github.com/neverhuman/JopeDime/pull/1581). Targeted16tests/Ruff/five records/source pins and audited zero-score constructor smoke passed. rd_argon exact-head PASS review5728918370 and author review are retained. CPU xbabe2 nice19, aggregate <=4 threads/RSS<6GB, CPU85/load108.8/Mem16GiB/disk93.8%/free40GB terminal guards, one3600s cap and complete scientific output root <500MB. No hosted full-CI/product qualification claim, GPU, protected data or live-loop writes. Python audit is application evidence, not an OS sandbox.

Versioned custody: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H191/astra-t0835/manifest.json`, VersionId `QfW6dn8jErBrdyv5soyuCQsydX6rxgWn`, SHA256 `5a98bfc75da4520bd8e6628023c0fccc7d4567caf41af62e390a35a4c94bbb87`. [custody.json](custody.json) binds full exact-version payload recovery and 1521 archive-member hashes. Archives were constructed/recovered in memory to avoid duplicate disk outputs. Inputs, raw costs/choices, lock, manifests and receipts preserved; source remains Git-pinned. Shared inputs untouched; only verified owned bulk eligible for reclamation after closeout merge.

Metadata clarification: the shared summarizer labels KXSEL `role=primary` and emits `holm_p_m1` inside both readouts. Inside `DL_Val_descriptive` those are descriptive policy labels/statistics only and never enter admission. The imported H177 p-value uses seed-averaged family gains; the separately reported median-bootstrap p-value uses the seed median. Both are retained exactly, with no favorable substitution.
