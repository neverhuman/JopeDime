# H386 — NL15 ranker path_smooth 5

**PS5_NOT_ADVANCED**. The fixed PS5 treatment completed all 36 fold/root tasks and nine partition/root panels on TRAIN only.
Paired drop-top-10 gain: -0.011896; whole-family 95% CI [-0.021459, -0.000000]; one-sided LCB95 -0.019151. The advance bar is +0.002 with positive LCB95 and all three retention/harm/root gates.

Hypothesis: mild path smoothing may reduce gate-cliff chasing and improve paired drop10. This was one fixed setting, selected before fitting; there was no tuning.

Primary frozen reference: H352 NL15 from committed02dfee2f7e0b8543324494dd4f99d074cd6b93b6. Secondary reference: archived BASE. P0/P1/P2 ×roots30001–30003 ×four family-grouped folds,1219 TRAIN contexts/205 families. Only ranker path_smooth0→5 changed, in both pinned ranker functions and all inner/final/eight bag rankers. Retain min_child_samples20/reg_lambda1/implicitreg_alpha0/bagging_freq1, configured num_leaves15/max_depth3 (at most8 realized binary leaves), original learning-rate/tree options and adaptive selection logic, labels/groups/seeds and every veto option. Removing path_smooth5 restores each entire H352 AST. Every native fitted model passed the full H352 config-only native contract; the learning-rate/tree-count fields are normalized to values selected by the unchanged grid, all other fields including derived seeds compare exactly except the prescribed ranker path smoothing.

Preregistration commit: `06f4d3a7f023c905ca22a36198cd0c3d0cb09100`, published 2026-11-05T18:22:30.667602+00:00; first treatment fit 2026-11-05T18:24:27.464621+00:00. See [PREREG.md](PREREG.md) and [metrics.json](metrics.json).

| Aggregate over nine partition/root cells | BASE | H352 NL15 | PS5 |
|---|---:|---:|---:|
| Median family-equal mean gain | +0.048215 | +0.058405 | +0.045339 |
| Whole-family 95% CI | [+0.025407, +0.075396] | [+0.034732, +0.091688] | [+0.023169, +0.079857] |
| Literal family-median gain | +0.003624 | +0.007409 | +0.005658 |
| Literal median 95% CI | [+0.000299, +0.007722] | [+0.002460, +0.011715] | [+0.001190, +0.009219] |
| Median harmed share (gain < −0.01) | 0.092683 | 0.092683 | 0.102439 |
| Median Elo | 1725.66 | 1734.47 | 1746.87 |

**Elo reference:** reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.

| Partition | Root | Arm | Family-equal gain [95% CI] | Literal family-median gain [95% CI] | Elo [95% CI] |
|---|---:|---|---:|---:|---:|
| P0 | 30001 | BASE | +0.041289 [+0.021463, +0.064902] | +0.000000 [+0.000000, +0.000000] | 1725.66 [1652.89, 1813.21] |
| P0 | 30002 | BASE | +0.064316 [+0.016030, +0.132053] | +0.001588 [+0.000000, +0.007380] | 1722.48 [1659.13, 1798.32] |
| P0 | 30003 | BASE | +0.072258 [+0.024335, +0.139339] | +0.006984 [+0.003011, +0.013091] | 1727.10 [1664.72, 1800.82] |
| P0 | 30001 | NL15 | +0.050668 [+0.017725, +0.086819] | +0.004840 [+0.000000, +0.010407] | 1733.39 [1668.23, 1811.19] |
| P0 | 30002 | NL15 | +0.047569 [+0.022251, +0.079816] | +0.000000 [+0.000000, +0.000704] | 1704.25 [1636.94, 1783.33] |
| P0 | 30003 | NL15 | +0.073628 [+0.040448, +0.119041] | +0.007409 [+0.001190, +0.011900] | 1769.52 [1700.74, 1855.96] |
| P0 | 30001 | PS5 | +0.063982 [+0.036699, +0.097545] | +0.006224 [+0.000000, +0.011715] | 1746.87 [1681.43, 1827.74] |
| P0 | 30002 | PS5 | +0.057299 [+0.019591, +0.104542] | +0.005658 [+0.000401, +0.012758] | 1749.84 [1684.44, 1830.92] |
| P0 | 30003 | PS5 | +0.041398 [+0.011956, +0.070563] | +0.003360 [+0.000000, +0.008220] | 1692.38 [1631.92, 1762.61] |
| P1 | 30001 | BASE | +0.048215 [+0.018534, +0.090866] | +0.003000 [+0.000000, +0.012335] | 1697.13 [1635.51, 1768.93] |
| P1 | 30002 | BASE | +0.048658 [+0.022116, +0.083460] | +0.003624 [+0.000000, +0.007782] | 1693.84 [1634.02, 1762.61] |
| P1 | 30003 | BASE | +0.038294 [+0.016052, +0.060466] | +0.003253 [+0.000000, +0.010809] | 1734.68 [1669.27, 1812.70] |
| P1 | 30001 | NL15 | +0.076944 [+0.039689, +0.128536] | +0.016096 [+0.008624, +0.021135] | 1757.59 [1697.58, 1831.07] |
| P1 | 30002 | NL15 | +0.047437 [+0.028784, +0.068303] | +0.003867 [+0.000000, +0.008983] | 1734.68 [1669.17, 1815.48] |
| P1 | 30003 | NL15 | +0.073295 [+0.034513, +0.132477] | +0.009586 [+0.003187, +0.017012] | 1726.57 [1666.54, 1799.49] |
| P1 | 30001 | PS5 | +0.057073 [+0.021199, +0.108207] | +0.005863 [+0.001190, +0.011474] | 1709.12 [1648.70, 1779.57] |
| P1 | 30002 | PS5 | +0.045339 [+0.013801, +0.082708] | +0.004783 [+0.000000, +0.010461] | 1686.31 [1625.85, 1754.68] |
| P1 | 30003 | PS5 | +0.039205 [+0.011454, +0.065270] | +0.004637 [+0.000000, +0.009791] | 1758.24 [1689.20, 1841.05] |
| P2 | 30001 | BASE | +0.043601 [+0.017178, +0.071578] | +0.006326 [+0.000781, +0.010521] | 1716.88 [1656.11, 1788.32] |
| P2 | 30002 | BASE | +0.031840 [+0.010212, +0.054012] | +0.003624 [+0.000000, +0.006960] | 1740.78 [1673.43, 1822.09] |
| P2 | 30003 | BASE | +0.054721 [+0.033395, +0.078998] | +0.008145 [+0.003611, +0.013831] | 1726.57 [1665.36, 1797.97] |
| P2 | 30001 | NL15 | +0.047241 [+0.016520, +0.088292] | +0.003000 [+0.000000, +0.009627] | 1709.11 [1645.39, 1782.68] |
| P2 | 30002 | NL15 | +0.063382 [+0.036425, +0.094487] | +0.011913 [+0.007555, +0.018370] | 1745.97 [1685.48, 1818.60] |
| P2 | 30003 | NL15 | +0.058405 [+0.027617, +0.093055] | +0.010809 [+0.005804, +0.016901] | 1734.47 [1674.65, 1806.51] |
| P2 | 30001 | PS5 | +0.053873 [+0.028316, +0.086342] | +0.010809 [+0.006927, +0.019626] | 1757.34 [1694.91, 1833.18] |
| P2 | 30002 | PS5 | +0.037951 [+0.009140, +0.066631] | +0.008220 [+0.001470, +0.012021] | 1747.87 [1683.61, 1822.65] |
| P2 | 30003 | PS5 | +0.038384 [+0.013159, +0.064335] | +0.003000 [+0.000000, +0.008960] | 1686.19 [1625.53, 1754.22] |

Root-wise paired drop-top-10 medians across the three partitions: 30001: +0.011683 CI95 [-0.003175, +0.019281] LCB95 -0.001593, 30002: -0.013326 CI95 [-0.027043, +0.000611] LCB95 -0.023834, 30003: -0.011896 CI95 [-0.034022, -0.000080] LCB95 -0.030068.
Configuration agreement: 14/36 fold/root choices (CONFIG_SENSITIVE). Full family mean/literal median CIs, drop-top-1/5/10, LOFO, W/T/L, harm, invalid, switches, root results and Elo CIs are retained in metrics.json. Ordinary and paired bootstrap:10,000 whole-family draws/RNG386191. Primary paired comparator H352 NL15; secondary BASE retained separately in metrics.json.

Decision gates: paired_drop_top10_at_least_002=False; gain_retention=False; harmed_nonincrease=False; positive_on_two_seeds=False.

Verification: exact-version archived BASE regression passed 9/9 panels, all 36 fold/root pick and lock hashes, and all three exact medians. Frozen H352 NL15 regression also passed all9panels/36locks/36scores/pick hashes/raw family arrays exactly. No BASE/NL15 refits or new reference predictions. Current TRAIN identities matched. Preparation processes ended before the cold fit process launched; no preparation packet cache was inherited. All 36 fsynced fit locks and audited models preceded once-only treatment outer scoring. Native audits checked 29,484 ranker fits (only path_smooth delta vs H352) and 7,020 unchanged veto fits; all 108 final veto hashes equal BASE. Zero fit-worker outer payload reads before their locks; zero DL-Val/dev91/DL-Test/sealed payload rows. All 55 data-free harness controls and seven adversarial custody controls passed. Same-agent red-team recomputed the paired bootstrap and verdict exactly.

Resources: xbabe2 CPU only, cores [5, 9, 10, 11], 4 workers, numeric threads1, nice19, MemoryMax6G/MemorySwapMax0. One admission probe; initial and first-completed-task forecasts passed. First-task conservative forecast ended 2026-11-05T21:36:34.121348+00:00. Study cgroup CPU 37545.436s; peak memory 2.243GiB. Observed host CPU60s max 36.12% (5s diagnostic max 64.74%), load max 53.65 with at most0 consecutive60s samples above108, MemAvailable min 65.31GiB, own scratch max 1416092497 bytes. Corrected guard uses CPU>85 averaged60s/load1>108 on3consecutive60s/MemAvailable20GiB/scratch2GB. Compute deadline22:15Z and close22:30Z Nov5 were fixed. No speed verdict.

Custody: create-only `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H386/jrb01-20261105T180500Z/closeout/closeout-manifest.json`, VersionId `WxbhEHmZ1KSleWBCk0j5CsKfNQ06mjC3`, SHA256 `10602e49967f85f27815c1197c1be990a817296585c59f74e64a368b0ef454fb`. The manifest binds 1842 files in 25 parts; every exact-version GET, whole-object hash and member hash was verified before removing only H386 scratch. Execution source/vendor and detailed local receipts remain in `/home/ubuntu/jope_plan/evidence/H386/jrb01-20261105-t1805/`. No models are in Git.

Observed ranker tree leaf-count histogram across recorded fits: {'4': 46406, '5': 483797, '6': 833570, '7': 1519399, '8': 2345228}. Configured15/depth3 therefore permits only up to8 realized leaves; this is diagnostic, not another intervention.

Sequential exploratory sweep H348–H386 carries multiplicity/selection bias. No live adoption; any ADVANCE requires a separately authorized one-shot dev91 confirmation. H386 reads zero dev91 rows.

DOPE `c51a56d0` was not loaded or changed; this was a CPU selection-layer experiment.

Acceptance command, after reconstructing the archived original source/vendor layout in approved scratch: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 python3 -B -m unittest discover -s /home/ubuntu/jope-data/h386-jrb01/source -p "test_*.py" -v`. These controls perform no model fits. The published manifest pins all executed fitting code and vendor hashes; detailed execution and custody receipts bind the actual run.

Next: retain H352 NL15; do not advance PS5. Any further intervention requires a new preregistered study.
