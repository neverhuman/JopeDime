# H383 — NL15 ranker L1 regularization on xbabe2

**L1A05_NOT_ADVANCED**. This TRAIN-only experiment changes ranker `reg_alpha` from 0 to 0.5 on the archived H352 NL15 configuration. The primary comparator is H352 NL15; BASE is secondary. The higher median Elo does not override the preregistered family-level advancement gates.

[PREREG.md](PREREG.md) was published at 2026-11-05T17:16:19.877296+00:00 in commit `c3f954b192d54bc2680c6f67654c35f3d5c932b9`. The first actual fit began at 2026-11-05T17:30:03.637382+00:00. H376 remains closed as `INCOMPLETE_RESOURCE_GUARD`, with zero fits and its original source and receipts preserved.

## Frozen science

FROZEN-SCIENCE-EQUIVALENCE passed for all 16 complete scientific AST groups against H376 after the declared wave, path, host, RNG (383191), and window substitutions. All 45 original source files retained their frozen hashes throughout execution.

The sole scientific change from H352 NL15 is ranker `reg_alpha=0.5` / native `lambda_l1=0.5`, applied to inner fits, final refits, and all eight bag rankers. Fixed settings remain leaves 15, depth 3, L2 1, minimum leaf samples 20, row fraction 0.8 with frequency 1, column fraction 0.8, and max bin 255. Removing the added keyword restores both complete H352 function ASTs. No seed keyword was added. Labels, groups, gain mapping, LambdaRank defaults, metric settings, search grid, rounds, features, sentinel (-2), and tie rule (1e-4) remain unchanged.

Veto classifiers retain L1 1 and L2 2; all 108 final veto hashes match BASE. Every completed fit has native parameter, H352 default/seed, physical Dataset, and ordered tree-depth/leaf evidence. Depth 3 bounds actual ranker leaves to eight even though configured leaves are 15. Dataset max bin is 255 for both model roles; no prebuilt reference bins were supplied.

## Admission and failed startup

The only capacity probe followed a read-only observation that `h377-run` was inactive at 17:21:56Z. The command returned status 4 for a garbage-collected transient unit. One five-second interval covered primary cores 27–37, then fallback 12–14, and their SMT siblings. It admitted four cores in priority order `[31, 12, 13, 14]`; both siblings of every pair were at least 85% idle, with host CPU 24.4157%.

The original startup compared sorted kernel affinity with that priority-ordered list and exited 1 before any forecast, constructor, optimizer fit, lock, or score. The failed invocation and logs remain evidence of failure. [RESOURCE_AMENDMENT.md](RESOURCE_AMENDMENT.md) was published at 2026-11-05T17:27:59.167610+00:00 in commit `b8d80c834f8e9b3067e1e6949312183fe7984558`, before every actual fit. It documents a reviewed set-comparison continuation using the same allocation, frozen source, RNG, and fixed deadline. Admission had passed; no second probe, new cores, scientific replay, or budget reset occurred.

Three operator tests passed: nonascending admission order, ascending order, and refusal of a different core set. [startup_continuation_receipt.json](startup_continuation_receipt.json) binds the tested entrypoint hash. The original failed invocation was `20c72e3931ce4f159807e274cdcc3691`; the actual scientific execution was `596abd11eadd4ab6a19a9c8a5837832c`, which exited 0 at 2026-11-05T19:55:10.581792+00:00. The no-restart rule for an admission, forecast, or guard miss remains binding.

## Inputs and evaluation

Fresh exact-Version S3 reads verified BASE H316/H315 as whole objects and members, and the H352 manifest plus all 27 parts / 2,691 members. BASE reproduced nine panels, 36 fold/root pick and lock hashes, and three exact partition medians. H352 NL15 reproduced nine panels, 36 locks/scores, and raw family arrays exactly on xbabe2 before treatment fits. Archived H352 headers cover 972 saved rankers and 29,484 actual fit records, proving original ranker L1 zero and max bin 255. No comparator was refitted or used for new model predictions.

Selected inputs, all 1,219 TRAIN pools, r882 cost data, and four consumed TRAIN feature prefixes passed identity checks. [regression_receipt.json](regression_receipt.json), [nl15_regression_receipt.json](nl15_regression_receipt.json), and [run_receipt.json](run_receipt.json) retain the evidence. Protected feature tails were excluded.

The design is P0/P1/P2 × roots 30001–30003 × four outer folds: 36 tasks, 1,219 TRAIN contexts, and 205 families. Fit parents consumed metadata only; worker hooks admitted outer-TRAIN ranges and denied comparator outcomes and native copies. All 36 locks and directories were fsynced and their native, physical input, veto, and custody reviews passed before cold outer packets and once-only scoring. This is historically exposed TRAIN evidence, with no fresh independent confirmation. **reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN**.

| Arm | Median Elo | Median family mean gain | Literal family median | Median harmed |
|---|---:|---:|---:|---:|
| BASE | 1725.655944 | +0.048214800 | +0.003624258 | 9.2683% |
| NL15 | 1734.469531 | +0.058404671 | +0.007409439 | 9.2683% |
| L1A05 | 1742.095508 | +0.053550758 | +0.006223684 | 9.7561% |

PRIMARY vs archived H352 NL15: median **-0.004754850503**, CI95[-0.012546446375461281, 0.0016978291999303458], LCB95 **-0.011238094931**.

| Root | Paired median | CI95 | LCB95 |
|---:|---:|---|---:|
| 30001 | +0.003712076 | [-0.00436559551613009, 0.013562022578438387] | -0.003199074 |
| 30002 | -0.004754851 | [-0.013157519883430627, 0.004737509124799805] | -0.011572204 |
| 30003 | -0.014986835 | [-0.027154046513839833, -0.0018713081390023792] | -0.024696767 |

Secondary vs BASE: median **+0.001619317896**, CI95[-0.004643046785812533, 0.009521895498483554], LCB95 **-0.003511394334**.

| Root | Paired median | CI95 | LCB95 |
|---:|---:|---|---:|
| 30001 | +0.005551715 | [-0.0039161262475135225, 0.021040593637754488] | -0.002070142 |
| 30002 | +0.002620381 | [-0.004430394945657266, 0.013296738162797096] | -0.003165535 |
| 30003 | -0.005351567 | [-0.023363321176634047, 0.0003230086907055751] | -0.021011274 |

Four primary gates: `{"paired_drop_top10_at_least_002": false, "gain_retention": true, "harmed_nonincrease": false, "positive_on_two_seeds": false}`. The secondary BASE comparison adds no gate.

| Part | Root | Arm | Mean | Literal median | Drop10 | W/T/L | Harm | Invalid | Switches | Elo | Elo CI95 |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| P0 | 30001 | BASE | +0.041289 | +0.000000 | +0.013174 | 88/93/24 | 6.341% | 13 | 219 | 1725.66 | [1652.447886376557, 1812.9700690664945] |
| P0 | 30002 | BASE | +0.064316 | +0.001588 | +0.008633 | 108/67/30 | 8.780% | 14 | 300 | 1722.48 | [1656.7079288809393, 1799.2122929353282] |
| P0 | 30003 | BASE | +0.072258 | +0.006984 | +0.016621 | 122/50/33 | 9.268% | 11 | 320 | 1727.10 | [1665.6454632422033, 1798.7037721418976] |
| P0 | 30001 | NL15 | +0.050668 | +0.004840 | +0.011081 | 115/60/30 | 7.317% | 13 | 257 | 1733.39 | [1668.6813747437993, 1811.1881388441423] |
| P0 | 30002 | NL15 | +0.047569 | +0.000000 | +0.012995 | 94/82/29 | 9.268% | 16 | 227 | 1704.25 | [1637.3491989417423, 1784.415165468581] |
| P0 | 30003 | NL15 | +0.073628 | +0.007409 | +0.028384 | 118/62/25 | 7.317% | 12 | 285 | 1769.52 | [1700.7429336578732, 1857.2955733217061] |
| P0 | 30001 | L1A05 | +0.062928 | +0.008580 | +0.014793 | 123/53/29 | 9.268% | 12 | 293 | 1750.96 | [1684.7763698444555, 1830.9325615738558] |
| P0 | 30002 | L1A05 | +0.063337 | +0.006224 | +0.007258 | 119/49/37 | 11.707% | 16 | 329 | 1702.91 | [1642.0309600562873, 1770.2877253566094] |
| P0 | 30003 | L1A05 | +0.050113 | +0.000204 | +0.012901 | 103/79/23 | 7.805% | 14 | 238 | 1760.39 | [1687.4724057518818, 1849.9495423194305] |
| P1 | 30001 | BASE | +0.048215 | +0.003000 | +0.010198 | 112/57/36 | 11.220% | 17 | 327 | 1697.13 | [1636.2756543932148, 1768.502029971351] |
| P1 | 30002 | BASE | +0.048658 | +0.003624 | +0.014063 | 116/51/38 | 10.244% | 15 | 275 | 1693.84 | [1633.762844611895, 1761.9483371747197] |
| P1 | 30003 | BASE | +0.038294 | +0.003253 | +0.012702 | 112/64/29 | 8.293% | 16 | 285 | 1734.68 | [1668.0582230695036, 1812.7033860359509] |
| P1 | 30001 | NL15 | +0.076944 | +0.016096 | +0.029863 | 141/32/32 | 10.244% | 12 | 431 | 1757.59 | [1696.1259403623662, 1831.3772390386723] |
| P1 | 30002 | NL15 | +0.047437 | +0.003867 | +0.020435 | 112/64/29 | 6.829% | 12 | 288 | 1734.68 | [1669.1805216399953, 1812.7290314140926] |
| P1 | 30003 | NL15 | +0.073295 | +0.009586 | +0.022337 | 129/41/35 | 11.707% | 13 | 369 | 1726.57 | [1664.4745655416996, 1798.6980188537896] |
| P1 | 30001 | L1A05 | +0.072216 | +0.011044 | +0.026885 | 133/39/33 | 9.756% | 11 | 322 | 1742.10 | [1679.700727354095, 1814.9333175686554] |
| P1 | 30002 | L1A05 | +0.050109 | +0.004783 | +0.016684 | 114/61/30 | 8.780% | 12 | 290 | 1731.87 | [1666.9867189013976, 1809.8074811083952] |
| P1 | 30003 | L1A05 | +0.035555 | +0.003583 | +0.007350 | 112/59/34 | 10.244% | 15 | 313 | 1707.06 | [1645.5064333750322, 1780.9165279341546] |
| P2 | 30001 | BASE | +0.043601 | +0.006326 | +0.010678 | 122/48/35 | 10.732% | 16 | 315 | 1716.88 | [1656.1058566632537, 1787.6763019641999] |
| P2 | 30002 | BASE | +0.031840 | +0.003624 | +0.008774 | 112/65/28 | 7.805% | 17 | 292 | 1740.78 | [1674.769204976847, 1820.0941275524494] |
| P2 | 30003 | BASE | +0.054721 | +0.008145 | +0.023605 | 129/41/35 | 10.732% | 15 | 335 | 1726.57 | [1665.9655865631883, 1796.0904217803327] |
| P2 | 30001 | NL15 | +0.047241 | +0.003000 | +0.008960 | 110/62/33 | 10.244% | 14 | 291 | 1709.11 | [1647.1518552896166, 1782.6547018715517] |
| P2 | 30002 | NL15 | +0.063382 | +0.011913 | +0.023902 | 136/36/33 | 9.268% | 12 | 360 | 1745.97 | [1684.26328616166, 1818.2987002068173] |
| P2 | 30003 | NL15 | +0.058405 | +0.010809 | +0.018755 | 135/35/35 | 12.195% | 17 | 389 | 1734.47 | [1674.0214585562887, 1805.3137043261177] |
| P2 | 30001 | L1A05 | +0.053551 | +0.008220 | +0.016230 | 121/55/29 | 9.268% | 15 | 280 | 1748.11 | [1683.7830363772848, 1827.185055324599] |
| P2 | 30002 | L1A05 | +0.069291 | +0.006985 | +0.019147 | 126/48/31 | 9.756% | 15 | 322 | 1743.56 | [1678.8273197760768, 1820.583296421622] |
| P2 | 30003 | L1A05 | +0.026437 | +0.003000 | +0.006506 | 111/51/43 | 14.634% | 18 | 297 | 1664.72 | [1605.8455258010197, 1730.254906258501] |

Configuration agreement: BASE/L1A05 7/36, NL15/L1A05 10/36. All 27 cells retain mean and median CIs, drop-top-1/5/10, LOFO minimum, W/T/L, harmed share, invalid selections, switches, Elo CI, pick and lock hashes, and seed Jaccard agreement.


[metrics.json](metrics.json) is below 50 KB and includes the exact frozen/unloaded DOPE block and SHA-linked receipts. Paired whole-family bootstraps use 10,000 shared draws and RNG 383191; ordinary reporting retains RNG 302150. Each arm drops its own best ten families. The paired summary is the median of nine cell differences; each root summary takes the median over three partitions and includes its CI.

Advancement requires all four preregistered gates: paired NL15 drop-top-10 median at least +0.002 with LCB95 above zero; median full gain at least 90% of NL15; harmed share no higher than NL15; and positive root medians on at least two of three roots. Ties never advance. BASE adds no gate. A completed failed gate yields `L1A05_NOT_ADVANCED`; incomplete work stays UNKNOWN. Multiplicity across H348–H383 is uncorrected. Any TRAIN advance would require separately authorized one-shot dev91 confirmation outside H383. No speed or serving conclusion is made.

## Verification and resources

All 89 source/fixture tests and three startup operator tests passed on xbabe2, with zero failures or skips. Fixtures performed no optimizer fits. Tests cover the full AST inverse, native defaults and seeds, Dataset and veto identity, physical outer-lock barriers, poisoned input tails, decision ties and root CIs, admission, release gating, and guard windows. Actual treatment qualification is **PASS**: 36504 completed native fit records, 14040 saved-model reviews, 36 locks, 36 complete score markers, and 108 matching final veto hashes. [actual_tree_summary.json](actual_tree_summary.json) summarizes model structure; full ordered native/Dataset/tree records are retained in S3. Review used the same implementing agent and performed no new fits or predictions.

Execution used nice 19, one numeric thread per worker, the same four admitted cores, MemoryMax 6 GiB, SwapMax 0, and CPUQuota 800%. Guards applied only to the owned unit: MemAvailable below 20 GiB, scratch above 2 GB, host CPU above 85% over a full 60-second window, load above 108 for three consecutive 60-second samples, or the fixed 20:45Z cutoff. Root disk free space was telemetry, without the old xbabe1 guard. Closure was fixed at 21:00Z.

Observed maxima: 60-second CPU 40.6929%, load 96.9995, scratch 1102724857 bytes. Minimum available memory was 68927483904 bytes. The maximum short CPU sample was 92.9574%; it did not cross the declared full-minute guard. The high-load streak remained 0. Admission and first-completed-task forecasts passed, ending at 2026-11-05T19:54:09.595494+00:00 and 2026-11-05T20:22:53.548425+00:00. They used H352 CPU 26189.516914 seconds, efficiency 0.9, safety 1.15, and 300 seconds of scoring reserve. The second forecast used max(reference/36, first-task CPU) for 35 remaining tasks with zero credit for ongoing work. Startup delay counted against the original window.

## Custody and publication

Only the owned H383 S3 prefix `experiments/rd_breakthrough_20261030/H383/hwv-xb2-20261105T170500Z/` was written, create-only. Before fitting, 2,232 historical native copies (576,553,824 bytes) were rehashed and pruned only after whole-object and every-member exact-Version verification of their archive. Comparator arrays, locks, scores, native JSON, and TRAIN inputs remained available. Inner raw/cache files were removed only after their own exact-Version whole/member proofs.

The final archive preserves models, native/Dataset/tree records, locks, scores, inputs, guards, admission, forecasts, release gating, both startup invocations, exits, tests, source, and control code. [s3_manifest.json](s3_manifest.json) pins reference, main, and inner archives. Cleanup requires exact file-set and every-file SHA agreement with verified custody. Models are excluded from Git.

Publication uses a private index and fresh-parent non-force pushes to `origin/v1/next`, with one H383 row in each index and other rows retained byte-for-byte. Canonical HEAD/index and foreign files are preserved. Owner CI_FREEZE and BATCH_WINDOW prohibit PRs, main/fastlane changes, hosted CI, FINDINGS edits, and a new lane. DOPE c51a56d0 remains frozen and unloaded (CPU selection layer only). No GPU, qualification lock, forbidden host access, protected data, dev91, sealed/holdout/meta-test/DL-Test rows, foreign process signals, or flywheel/writer mutation occurred. All H383 ownership is released at closeout; no further H383 model work is authorized afterward.
