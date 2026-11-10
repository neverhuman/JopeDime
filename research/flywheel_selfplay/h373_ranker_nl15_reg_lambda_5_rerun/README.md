# H373 — NL15 ranker reg_lambda 5

**L2R5_NOT_ADVANCED**. The fixed L2R5 treatment completed all 36 fold/root tasks and nine partition/root panels on TRAIN only.
Paired drop-top-10 gain: -0.001821; whole-family 95% CI [-0.010218, +0.004041]; one-sided LCB95 -0.008998. The advance bar is +0.002 with positive LCB95 and all three retention/harm/root gates.

Hypothesis: stronger L2 on NL15 may damp leaf weights and improve paired drop10. This was one fixed setting, selected before fitting; there was no tuning.

Primary frozen reference: H352 NL15 from committed02dfee2f7e0b8543324494dd4f99d074cd6b93b6. Secondary reference: archived BASE. P0/P1/P2 ×roots30001–30003 ×four family-grouped folds,1219 TRAIN contexts/205 families. Only existing ranker reg_lambda1→5/native lambda_l2_1→5 changed, in both pinned ranker functions and all inner/final/eight bag rankers. Retain configured num_leaves15/max_depth3 (at most8 realized binary leaves), original learning-rate/tree options and adaptive selection logic, labels/groups/seeds and every veto option. Restoring1 restores each entire H352 AST. Every native fitted model passed the full H352 config-only native contract; the learning-rate/tree-count fields are normalized to values selected by the unchanged grid, all other fields including derived seeds compare exactly except the prescribed ranker L2.

Preregistration commit: `3e6c55318d82100253e624d5044e2fb92cff806c`, published 2026-11-05T14:39:22.877496+00:00; first treatment fit 2026-11-05T14:45:20.270865+00:00. See [PREREG.md](PREREG.md) and [metrics.json](metrics.json).

| Aggregate over nine partition/root cells | BASE | H352 NL15 | L2R5 |
|---|---:|---:|---:|
| Median family-equal mean gain | +0.048215 | +0.058405 | +0.057075 |
| Whole-family 95% CI | [+0.025140, +0.075508] | [+0.033857, +0.092482] | [+0.029065, +0.095703] |
| Literal family-median gain | +0.003624 | +0.007409 | +0.006326 |
| Literal median 95% CI | [+0.000299, +0.007722] | [+0.002351, +0.011715] | [+0.002565, +0.009700] |
| Median harmed share (gain < −0.01) | 0.092683 | 0.092683 | 0.097561 |
| Median Elo | 1725.66 | 1734.47 | 1726.98 |

**Elo reference:** reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.

| Partition | Root | Arm | Family-equal gain [95% CI] | Literal family-median gain [95% CI] | Elo [95% CI] |
|---|---:|---|---:|---:|---:|
| P0 | 30001 | BASE | +0.041289 [+0.021035, +0.063927] | +0.000000 [+0.000000, +0.000000] | 1725.66 [1653.45, 1814.98] |
| P0 | 30002 | BASE | +0.064316 [+0.016099, +0.134289] | +0.001588 [+0.000000, +0.007380] | 1722.48 [1657.49, 1799.80] |
| P0 | 30003 | BASE | +0.072258 [+0.023188, +0.140846] | +0.006984 [+0.003011, +0.013091] | 1727.10 [1666.87, 1801.26] |
| P0 | 30001 | NL15 | +0.050668 [+0.017525, +0.087182] | +0.004840 [+0.000000, +0.010407] | 1733.39 [1669.27, 1809.73] |
| P0 | 30002 | NL15 | +0.047569 [+0.022199, +0.080559] | +0.000000 [+0.000000, +0.000704] | 1704.25 [1636.50, 1783.13] |
| P0 | 30003 | NL15 | +0.073628 [+0.040566, +0.119865] | +0.007409 [+0.001190, +0.011900] | 1769.52 [1700.74, 1857.02] |
| P0 | 30001 | L2R5 | +0.070920 [+0.028728, +0.139556] | +0.002222 [+0.000000, +0.007444] | 1723.10 [1658.30, 1799.80] |
| P0 | 30002 | L2R5 | +0.063770 [+0.024229, +0.112796] | +0.005911 [+0.001291, +0.011913] | 1731.01 [1666.87, 1805.85] |
| P0 | 30003 | L2R5 | +0.045189 [+0.020230, +0.076077] | +0.002777 [+0.000000, +0.007423] | 1697.13 [1635.18, 1768.50] |
| P1 | 30001 | BASE | +0.048215 [+0.018638, +0.092754] | +0.003000 [+0.000000, +0.012335] | 1697.13 [1635.18, 1767.55] |
| P1 | 30002 | BASE | +0.048658 [+0.022000, +0.084309] | +0.003624 [+0.000000, +0.007782] | 1693.84 [1634.02, 1762.61] |
| P1 | 30003 | BASE | +0.038294 [+0.015749, +0.060634] | +0.003253 [+0.000000, +0.010809] | 1734.68 [1669.27, 1811.19] |
| P1 | 30001 | NL15 | +0.076944 [+0.039749, +0.129058] | +0.016096 [+0.008624, +0.021135] | 1757.59 [1696.01, 1833.40] |
| P1 | 30002 | NL15 | +0.047437 [+0.029016, +0.067903] | +0.003867 [+0.000000, +0.008983] | 1734.68 [1667.81, 1812.63] |
| P1 | 30003 | NL15 | +0.073295 [+0.033529, +0.134025] | +0.009586 [+0.003187, +0.017012] | 1726.57 [1665.36, 1797.97] |
| P1 | 30001 | L2R5 | +0.036861 [+0.019860, +0.055629] | +0.002299 [+0.000000, +0.007087] | 1734.46 [1666.31, 1815.70] |
| P1 | 30002 | L2R5 | +0.057686 [+0.027734, +0.099116] | +0.006552 [+0.002399, +0.012113] | 1711.45 [1650.24, 1782.19] |
| P1 | 30003 | L2R5 | +0.045681 [+0.024800, +0.069179] | +0.008895 [+0.004749, +0.015954] | 1726.98 [1668.91, 1797.30] |
| P2 | 30001 | BASE | +0.043601 [+0.016640, +0.071602] | +0.006326 [+0.000781, +0.010515] | 1716.88 [1655.97, 1786.63] |
| P2 | 30002 | BASE | +0.031840 [+0.010255, +0.054314] | +0.003624 [+0.000000, +0.006960] | 1740.78 [1673.94, 1821.54] |
| P2 | 30003 | BASE | +0.054721 [+0.033195, +0.079110] | +0.008145 [+0.003611, +0.013831] | 1726.57 [1665.09, 1798.70] |
| P2 | 30001 | NL15 | +0.047241 [+0.016395, +0.090504] | +0.003000 [+0.000000, +0.009627] | 1709.11 [1647.73, 1783.82] |
| P2 | 30002 | NL15 | +0.063382 [+0.035594, +0.094107] | +0.011913 [+0.007555, +0.018370] | 1745.97 [1685.35, 1819.79] |
| P2 | 30003 | NL15 | +0.058405 [+0.027867, +0.093728] | +0.010809 [+0.005804, +0.016901] | 1734.47 [1673.64, 1805.07] |
| P2 | 30001 | L2R5 | +0.047383 [+0.027432, +0.069413] | +0.006326 [+0.000620, +0.011715] | 1728.12 [1664.98, 1802.54] |
| P2 | 30002 | L2R5 | +0.071364 [+0.032543, +0.128396] | +0.010574 [+0.006731, +0.018370] | 1748.51 [1686.83, 1821.42] |
| P2 | 30003 | L2R5 | +0.057075 [+0.021847, +0.105441] | +0.007555 [+0.003583, +0.016847] | 1690.82 [1632.37, 1756.02] |

Root-wise paired drop-top-10 medians across the three partitions: 30001: +0.006826 CI95 [-0.010665, +0.016976] LCB95 -0.008770, 30002: -0.001405 CI95 [-0.011985, +0.011484] LCB95 -0.009796, 30003: -0.005692 CI95 [-0.019479, +0.002034] LCB95 -0.017579.
Configuration agreement: 8/36 fold/root choices (CONFIG_SENSITIVE). Full family mean/literal median CIs, drop-top-1/5/10, LOFO, W/T/L, harm, invalid, switches, root results and Elo CIs are retained in metrics.json. Ordinary and paired bootstrap:10,000 whole-family draws/RNG373191. Primary paired comparator H352 NL15; secondary BASE retained separately in metrics.json.

Decision gates: paired_drop_top10_at_least_002=False; gain_retention=True; harmed_nonincrease=False; positive_on_two_seeds=False.

Verification: exact-version archived BASE regression passed 9/9 panels, all 36 fold/root pick and lock hashes, and all three exact medians. Frozen H352 NL15 regression also passed all9panels/36locks/36scores/pick hashes/raw family arrays exactly. No BASE/NL15 refits or new reference predictions. Current TRAIN identities matched. Preparation processes ended before the cold fit process launched; no preparation packet cache was inherited. All 36 fsynced fit locks and audited models preceded once-only treatment outer scoring. Native audits checked 29,484 ranker fits (only lambda_l2 delta vs H352) and 7,020 unchanged veto fits; all 108 final veto hashes equal BASE. Zero fit-worker outer payload reads before their locks; zero DL-Val/dev91/DL-Test/sealed payload rows. All 55 data-free harness controls and seven adversarial custody controls passed. Same-agent red-team recomputed the paired bootstrap and verdict exactly.

Resources: xbabe2 CPU only, cores [2, 5, 6, 7, 9, 10, 11, 12], 8 workers, numeric threads1, nice19, MemoryMax6G/MemorySwapMax0. One admission probe; initial and first-completed-task forecasts passed. First-task conservative forecast ended 2026-11-05T16:16:38.547558+00:00. Study cgroup CPU 31961.383s; peak memory 3.231GiB. Observed host CPU60s max 49.48% (5s diagnostic max 95.21%), load max 90.59 with at most0 consecutive60s samples above108, MemAvailable min 66.08GiB, own scratch max 1453972846 bytes. Corrected guard uses CPU>85 averaged60s/load1>108 on3consecutive60s/MemAvailable20GiB/scratch2GB. Compute deadline17:30Z and close17:45Z Nov5 were fixed. No speed verdict.

Custody: create-only `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H373/jrb01-20261105T143000Z/closeout/closeout-manifest.json`, VersionId `VvSEbF9NcK27nNMWsvkynTP9RMGoCGgM`, SHA256 `6be5fcd6add93e3cd45574d98a547e4ccbdca13293ca023b1ef40b938c237116`. The manifest binds 1826 files in 25 parts; every exact-version GET, whole-object hash and member hash was verified before removing only H373 scratch. Execution source/vendor and detailed local receipts remain in `/home/ubuntu/jope_plan/evidence/H373/jrb01-20261105-t1430/`. No models are in Git.

Observed ranker tree leaf-count histogram across recorded fits: {'4': 8713, '5': 207301, '6': 601831, '7': 1362464, '8': 3012491}. Configured15/depth3 therefore permits only up to8 realized leaves; this is diagnostic, not another intervention.

Sequential exploratory sweep H348–H374 carries multiplicity/selection bias. No live adoption; any ADVANCE requires a separately authorized one-shot dev91 confirmation. H373 reads zero dev91 rows.

DOPE `c51a56d0` was not loaded or changed; this was a CPU selection-layer experiment.

Acceptance command, after reconstructing the archived original source/vendor layout in approved scratch: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 python3 -B -m unittest discover -s /home/ubuntu/jope-data/h373-jrb01/source -p "test_*.py" -v`. These controls perform no model fits. The published manifest pins all executed fitting code and vendor hashes; detailed execution and custody receipts bind the actual run.

Next: retain H352 NL15; do not advance L2R5. Any further intervention requires a new preregistered study.
