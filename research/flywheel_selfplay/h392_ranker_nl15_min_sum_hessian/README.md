# H392 — NL15 ranker Hessian floor

**MINHESS_NOT_ADVANCED**. MINHESS changes only the ranker leaf Hessian floor from **0.001** to **7.65**, through sklearn `min_child_weight` / native `min_sum_hessian_in_leaf`. The primary comparator is archived H352 NL15; BASE is secondary. Elo does not override the preregistered family-level gates.

[PREREG.md](PREREG.md) was published at 2026-11-05T20:39:50.574573+00:00 in commit `5c08e5ae72223513b549bc33830863ac81c4f836`, before the first actual fit at 2026-11-05T20:41:25.923638+00:00. H383 remains closed as L1A05_NOT_ADVANCED, with its original source and CLOSEOUT unchanged.

## One fixed, archive-derived value

The value was computed once, before treatment, from read-only `Booster(model_str).dump_model()` leaf_weight values in all **12636 archived NL15 rankers**: 972 final models and 11,664 inner models from 36 compressed archives. BASE copies and all veto classifiers were excluded. All 12967896 leaves across 1821200 trees received equal weight. Raw Q25 was **7.6547720804810515** using `numpy.quantile(method='linear')`; rounding to three significant figures fixed **7.65**. Every archived ranker floor was 0.001. The derived value exceeded baseline, so the mismatch stop did not apply. Calibration fits, optimizer updates, and new predictions during derivation were all zero.

[hessian_value.json](hessian_value.json) pins the archive versions, whole/member checks, pooled leaf sequence hash, and per-model inventory hash. Full inventory rows are retained in S3 custody. No alternate value was selected after outcomes were observed.

## Sole scientific intervention

Both complete H352 ranker function ASTs invert exactly to actual committed H352 source after removing the one added `min_child_weight` keyword. It applies to every inner fit, final refit, and all eight bag rankers. No seed keyword was added. Fixed settings remain leaves 15, depth 3, L2 1, L1 0, min_child_samples 20, row fraction 0.8, bagging frequency 1, column fraction 0.8, path_smooth 0, and max_bin 255. Labels, gain mapping, LambdaRank/ndcg defaults, sigmoid, truncation, metric positions, learning rates, rounds, features, groups, derived seeds, all 18 configurations, sentinel -2, and tie rule 1e-4 remain unchanged.

Veto classifiers retain their own L1 1/L2 2 and baseline Hessian floor. Full qualification requires all 108 final veto hashes to equal BASE. Native constructor/default/seed, physical Dataset, and ordered tree-depth/leaf evidence covers every completed fit. Both model roles use fresh raw-numpy max_bin-255 Datasets without prebuilt/reference bins. Ranker depth 3 bounds actual leaves to eight despite configured leaves 15. Telemetry never changes selection.

## References and evaluation

Fresh exact-Version BASE H316/H315 GETs verified whole hashes and selected members. BASE reproduced nine panels, 36 fold/root pick and lock hashes, and three exact partition medians. Fresh H352 manifest/27 parts verified all 2,691 members and reproduced nine panels, 36 locks/scores, and raw family arrays exactly. The H352 full native trace verifies all 29,484 ranker and 7,020 veto fit headers. Current selected inputs, 1,219 TRAIN pools, r882 costs, and four consumed TRAIN feature prefixes match their archived identities. No comparator was refitted or used for new model predictions. [regression_receipt.json](regression_receipt.json), [nl15_regression_receipt.json](nl15_regression_receipt.json), and [run_receipt.json](run_receipt.json) retain these checks.

P0/P1/P2 × roots 30001–30003 × four family-disjoint outer folds gives 36 tasks and nine panels over 205 TRAIN families. Fit parents use metadata only. Worker hooks admit outer-TRAIN pools and exact TRAIN feature ranges, and deny archived comparator outcomes and models. All 36 locks/directories must be fsynced and their native, physical input, veto, and custody reviews pass before cold outer packets and once-only scoring. No partial scoring is used. This is historically exposed TRAIN evidence, without independent confirmation. **reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN**.

| Arm | Median Elo | Median family mean gain | Literal family median | Median harmed |
|---|---:|---:|---:|---:|
| BASE | 1725.655944 | +0.048214800 | +0.003624258 | 9.2683% |
| NL15 | 1734.469531 | +0.058404671 | +0.007409439 | 9.2683% |
| MINHESS | 1721.680102 | +0.046639425 | +0.006236274 | 10.2439% |

PRIMARY vs archived H352 NL15: median **-0.006282420660**, CI95[-0.01604804654301786, 0.0012619381139801977], LCB95 **-0.014303682685**.

| Root | Paired median | CI95 | LCB95 |
|---:|---:|---|---:|
| 30001 | -0.001090685 | [-0.01778700047538923, 0.011661180620676222] | -0.015016703 |
| 30002 | -0.004527174 | [-0.01732783371873925, 0.008697078397470828] | -0.015104473 |
| 30003 | -0.009389275 | [-0.025683517376930476, -0.0005514080249329899] | -0.022883519 |

Secondary vs BASE: median **+0.001844320111**, CI95[-0.007499331801294911, 0.007196796697052684], LCB95 **-0.006282379812**.

| Root | Paired median | CI95 | LCB95 |
|---:|---:|---|---:|
| 30001 | -0.002809006 | [-0.013854102995903011, 0.010595272081503463] | -0.011558805 |
| 30002 | +0.005741285 | [-0.002818968179008562, 0.012647326756160631] | -0.001688293 |
| 30003 | -0.009739998 | [-0.01963645261202813, 0.0013836721306711764] | -0.017690493 |

Four primary gates: `{"paired_drop_top10_at_least_002": false, "gain_retention": false, "harmed_nonincrease": false, "positive_on_two_seeds": false}`. The secondary BASE comparison adds no gate.

| Part | Root | Arm | Mean | Literal median | Drop10 | W/T/L | Harm | Invalid | Switches | Elo | Elo CI95 |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| P0 | 30001 | BASE | +0.041289 | +0.000000 | +0.013174 | 88/93/24 | 6.341% | 13 | 219 | 1725.66 | [1652.447886376557, 1812.9700690664945] |
| P0 | 30002 | BASE | +0.064316 | +0.001588 | +0.008633 | 108/67/30 | 8.780% | 14 | 300 | 1722.48 | [1656.7079288809393, 1799.2122929353282] |
| P0 | 30003 | BASE | +0.072258 | +0.006984 | +0.016621 | 122/50/33 | 9.268% | 11 | 320 | 1727.10 | [1665.6454632422033, 1798.7037721418976] |
| P0 | 30001 | NL15 | +0.050668 | +0.004840 | +0.011081 | 115/60/30 | 7.317% | 13 | 257 | 1733.39 | [1668.6813747437993, 1811.1881388441423] |
| P0 | 30002 | NL15 | +0.047569 | +0.000000 | +0.012995 | 94/82/29 | 9.268% | 16 | 227 | 1704.25 | [1637.3491989417423, 1784.415165468581] |
| P0 | 30003 | NL15 | +0.073628 | +0.007409 | +0.028384 | 118/62/25 | 7.317% | 12 | 285 | 1769.52 | [1700.7429336578732, 1857.2955733217061] |
| P0 | 30001 | MINHESS | +0.066957 | +0.004749 | +0.020013 | 114/55/36 | 11.707% | 14 | 307 | 1700.21 | [1638.6937657353535, 1769.9200265211048] |
| P0 | 30002 | MINHESS | +0.052241 | +0.008220 | +0.015097 | 129/41/35 | 9.268% | 15 | 321 | 1726.57 | [1665.681155196094, 1797.9729406428341] |
| P0 | 30003 | MINHESS | +0.040005 | +0.006236 | +0.006881 | 118/50/37 | 12.195% | 15 | 323 | 1701.44 | [1641.2626333286298, 1770.2877253566094] |
| P1 | 30001 | BASE | +0.048215 | +0.003000 | +0.010198 | 112/57/36 | 11.220% | 17 | 327 | 1697.13 | [1636.2756543932148, 1768.502029971351] |
| P1 | 30002 | BASE | +0.048658 | +0.003624 | +0.014063 | 116/51/38 | 10.244% | 15 | 275 | 1693.84 | [1633.762844611895, 1761.9483371747197] |
| P1 | 30003 | BASE | +0.038294 | +0.003253 | +0.012702 | 112/64/29 | 8.293% | 16 | 285 | 1734.68 | [1668.0582230695036, 1812.7033860359509] |
| P1 | 30001 | NL15 | +0.076944 | +0.016096 | +0.029863 | 141/32/32 | 10.244% | 12 | 431 | 1757.59 | [1696.1259403623662, 1831.3772390386723] |
| P1 | 30002 | NL15 | +0.047437 | +0.003867 | +0.020435 | 112/64/29 | 6.829% | 12 | 288 | 1734.68 | [1669.1805216399953, 1812.7290314140926] |
| P1 | 30003 | NL15 | +0.073295 | +0.009586 | +0.022337 | 129/41/35 | 11.707% | 13 | 369 | 1726.57 | [1664.4745655416996, 1798.6980188537896] |
| P1 | 30001 | MINHESS | +0.025991 | +0.000669 | +0.001636 | 105/71/29 | 7.805% | 18 | 242 | 1723.47 | [1657.3104732319473, 1804.4871493889048] |
| P1 | 30002 | MINHESS | +0.059823 | +0.002251 | +0.015908 | 105/66/34 | 10.732% | 16 | 257 | 1695.85 | [1634.29457313176, 1768.0452156163988] |
| P1 | 30003 | MINHESS | +0.042417 | +0.006226 | +0.016055 | 124/50/31 | 9.756% | 14 | 305 | 1740.78 | [1677.621270554157, 1816.3863183025085] |
| P2 | 30001 | BASE | +0.043601 | +0.006326 | +0.010678 | 122/48/35 | 10.732% | 16 | 315 | 1716.88 | [1656.1058566632537, 1787.6763019641999] |
| P2 | 30002 | BASE | +0.031840 | +0.003624 | +0.008774 | 112/65/28 | 7.805% | 17 | 292 | 1740.78 | [1674.769204976847, 1820.0941275524494] |
| P2 | 30003 | BASE | +0.054721 | +0.008145 | +0.023605 | 129/41/35 | 10.732% | 15 | 335 | 1726.57 | [1665.9655865631883, 1796.0904217803327] |
| P2 | 30001 | NL15 | +0.047241 | +0.003000 | +0.008960 | 110/62/33 | 10.244% | 14 | 291 | 1709.11 | [1647.1518552896166, 1782.6547018715517] |
| P2 | 30002 | NL15 | +0.063382 | +0.011913 | +0.023902 | 136/36/33 | 9.268% | 12 | 360 | 1745.97 | [1684.26328616166, 1818.2987002068173] |
| P2 | 30003 | NL15 | +0.058405 | +0.010809 | +0.018755 | 135/35/35 | 12.195% | 17 | 389 | 1734.47 | [1674.0214585562887, 1805.3137043261177] |
| P2 | 30001 | MINHESS | +0.046639 | +0.006326 | +0.007869 | 121/51/33 | 10.244% | 15 | 296 | 1725.67 | [1664.0445138157652, 1798.7037721418976] |
| P2 | 30002 | MINHESS | +0.041206 | +0.012165 | +0.014515 | 127/42/36 | 10.244% | 16 | 393 | 1718.97 | [1659.8914236702215, 1788.9972557305082] |
| P2 | 30003 | MINHESS | +0.059544 | +0.009117 | +0.009365 | 129/40/36 | 12.683% | 16 | 362 | 1721.68 | [1661.9009533037415, 1789.5367289837013] |

Configuration agreement: BASE/MINHESS 11/36, NL15/MINHESS 5/36. All 27 cells retain mean and median CIs, drop-top-1/5/10, LOFO minimum, W/T/L, harmed share, invalid selections, switches, Elo CI, pick and lock hashes, and seed Jaccard agreement.


[metrics.json](metrics.json) is below 50 KB and retains the exact frozen/unloaded DOPE block and SHA-linked receipts. Shared whole-family paired bootstraps use 10,000 draws and RNG 392191; ordinary reporting retains RNG 302150. Each arm drops its own best ten families. The paired summary takes the median of nine cell differences; each root takes the median across three partitions with its paired CI and LCB.

Advancement requires all four preregistered gates: primary paired NL15 drop10 median at least +0.002 with LCB95 above zero; median full gain at least 90% of NL15; median harmed share no higher than NL15; positive paired medians on at least two of three roots. Ties never advance. BASE adds no gate. A completed failed gate yields MINHESS_NOT_ADVANCED; incomplete work stays UNKNOWN. Multiplicity across H348–H392 is uncorrected. Any advance needs separately authorized one-shot dev91 confirmation outside H392. No speed or serving conclusion is made.

## Verification and operational history

All 95 final source/fixture tests passed locally with zero failures or skips and zero optimizer fixture fits. They cover the complete committed AST inverse, native defaults and alias, seeds, percentile/rounding fixtures, Dataset and veto identity, physical outer-lock barriers, poisoned feature tails, decision ties and root CIs, union admission, affinity order, and full-minute guards. An earlier fixture explicitly passed reg_alpha=0 and was correctly refused by the no-extra-keyword guard; the fixture was corrected to the actual H352 implicit default. The guard was not loosened. That preliminary failure remains in custody.

The initial read-only derivation guard daemon rejected the new preparation unit name. Before treatment, the owned whitelist was corrected and an external own-unit guard was attached to the same calculation. The percentile was not restarted or recomputed. [prep_guard_receipt.json](prep_guard_receipt.json) preserves the exception and correction. The early interval is **not retrospectively counted as fully guarded**; systemd one-core/nice19/math1/CPU100/6GiB/Swap0/fixed-deadline limits were enforced. Scientific execution uses the corrected frozen guard and a fresh resource log.

Actual treatment qualification: **PASS**. Available evidence includes 36504 completed native records, 14040 saved-model reviews, 36 locks, 36 score markers, and 108 matching final veto hashes. [actual_tree_summary.json](actual_tree_summary.json) summarizes structure; full ordered native/Dataset/tree rows remain archived. The same implementing agent reviewed the work, with no independent or hosted-CI claim and no new fits or predictions during review.

## Resources and custody

Exactly one five-second union admission sample selected [30, 34, 35, 37, 12, 13, 14], with both siblings of every pair at least 85% idle and host CPU 21.9865%. Primary cores 27–37 precede fallback 12–14. At least four and at most eight cores were allowed; no retry, borrowed core, restart, or reset occurred. Execution used nice19, numeric threads1, MemoryMax6GiB, SwapMax0, and CPUQuota800%. Its actual invocation was `546b08e89d3d475580f58e451325e868`, with exit 0 at 2026-11-05T21:55:55.086277+00:00.

Only owned units could be stopped. Guards were MemAvailable below20GiB, scratch above2GB, host CPU above85% averaged over a full60-second window, load above108 for three consecutive60-second samples, or the fixed 00:45Z Nov6 compute cutoff. Exactly85% CPU does not stop; load at/below108 resets its streak. Closure is fixed at01:00Z. Root free space is telemetry.

Observed maximum full-minute CPU was 39.05605090702369%; maximum short CPU sample 61.56937193482044%; maximum load 53.65380859375; high-load streak 0; minimum available memory 74185248768 bytes; peak scratch 1246542706 bytes. Initial and first-task forecasts are retained in run_receipt.json. They use H352 CPU26189.516914 seconds, efficiency0.9, safety1.15, scoring reserve300s; the remaining35-task forecast uses max(reference/36,actual first-task CPU), with zero ongoing-work credit. The fixed deadline was never reset.

Actual environment was xbabe2, kernel 6.8.0-147-generic, Python 3.12.3, NumPy 1.26.4, sklearn 1.7.2, LightGBM 4.6.0, native library SHA `6a12e1a7b13824d14c252d2a4d2e36af467383331a114eb0e7448eecbf02d7b1`. The inherited manifest environment is a historical template record; these separately observed runtime identities are authoritative.

The owned create-only prefix is `experiments/rd_breakthrough_20261030/H392/hwv-20261105T203000Z/`. Before fitting, 2,232 historical copies (576,553,824 bytes) were rehashed and pruned only after whole-object and every-member exact-Version verification of their reference archive. Inner raw/cache removal requires its own exact-Version whole/member proof. [s3_manifest.json](s3_manifest.json) pins reference, main, and inner archives. Final cleanup requires exact file-set/every-file SHA agreement with verified custody. Models never enter Git.

Publication uses a private index and fresh-parent non-force `origin/v1/next` pushes, with one H392 row in each required index and all other rows byte-preserved. Canonical HEAD/index, foreign files, and H383 evidence remain intact. Owner CI_FREEZE/BATCH_WINDOW means no PR, main/fastlane change, hosted CI, FINDINGS edit, or new lane. DOPE c51a56d0 remains frozen and unloaded, CPU selection layer only. No GPU/qualification lock, forbidden host access, protected data, dev91, sealed/holdout/meta-test/DL-Test, foreign process signal, or flywheel/writer/AUTORESTART mutation occurred. Explicit ownership release follows final closeout; no further H392 model work is authorized afterward.
