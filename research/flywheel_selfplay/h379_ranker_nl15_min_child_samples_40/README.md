# H379 — NL15 ranker min_child_samples 40

**MCS40_NOT_ADVANCED**. The fixed MCS40 treatment completed all 36 fold/root tasks and nine partition/root panels on TRAIN only.
Paired drop-top-10 gain: -0.007260; whole-family 95% CI [-0.012589, -0.000755]; one-sided LCB95 -0.011558. The advance bar is +0.002 with positive LCB95 and all three retention/harm/root gates.

Hypothesis: a stronger leaf occupancy floor may damp sparse-leaf updates and improve paired drop10. This was one fixed setting, selected before fitting; there was no tuning.

Primary frozen reference: H352 NL15 from committed02dfee2f7e0b8543324494dd4f99d074cd6b93b6. Secondary reference: archived BASE. P0/P1/P2 ×roots30001–30003 ×four family-grouped folds,1219 TRAIN contexts/205 families. Only existing ranker min_child_samples20→40/native min_data_in_leaf20→40 changed, in both pinned ranker functions and all inner/final/eight bag rankers. Retain reg_lambda1/implicitreg_alpha0, configured num_leaves15/max_depth3 (at most8 realized binary leaves), original learning-rate/tree options and adaptive selection logic, labels/groups/seeds and every veto option. Restoring20 restores each entire H352 AST. Every native fitted model passed the full H352 config-only native contract; the learning-rate/tree-count fields are normalized to values selected by the unchanged grid, all other fields including derived seeds compare exactly except the prescribed ranker leaf floor.

Preregistration commit: `d6be93f670ca07808695051a1bb96af7ffee2392`, published 2026-11-05T16:37:56.652729+00:00; first treatment fit 2026-11-05T16:39:07.104671+00:00. See [PREREG.md](PREREG.md) and [metrics.json](metrics.json).

| Aggregate over nine partition/root cells | BASE | H352 NL15 | MCS40 |
|---|---:|---:|---:|
| Median family-equal mean gain | +0.048215 | +0.058405 | +0.053501 |
| Whole-family 95% CI | [+0.025433, +0.076460] | [+0.034564, +0.091940] | [+0.026097, +0.098240] |
| Literal family-median gain | +0.003624 | +0.007409 | +0.005637 |
| Literal median 95% CI | [+0.000434, +0.007782] | [+0.002351, +0.011715] | [+0.000000, +0.008580] |
| Median harmed share (gain < −0.01) | 0.092683 | 0.092683 | 0.102439 |
| Median Elo | 1725.66 | 1734.47 | 1714.81 |

**Elo reference:** reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.

| Partition | Root | Arm | Family-equal gain [95% CI] | Literal family-median gain [95% CI] | Elo [95% CI] |
|---|---:|---|---:|---:|---:|
| P0 | 30001 | BASE | +0.041289 [+0.021514, +0.064666] | +0.000000 [+0.000000, +0.000000] | 1725.66 [1652.90, 1813.08] |
| P0 | 30002 | BASE | +0.064316 [+0.015445, +0.132601] | +0.001588 [+0.000000, +0.007380] | 1722.48 [1656.71, 1801.79] |
| P0 | 30003 | BASE | +0.072258 [+0.022938, +0.140072] | +0.006984 [+0.003011, +0.013091] | 1727.10 [1662.97, 1800.33] |
| P0 | 30001 | NL15 | +0.050668 [+0.017012, +0.086858] | +0.004840 [+0.000000, +0.010407] | 1733.39 [1668.68, 1809.81] |
| P0 | 30002 | NL15 | +0.047569 [+0.022039, +0.080562] | +0.000000 [+0.000000, +0.000704] | 1704.25 [1637.35, 1784.92] |
| P0 | 30003 | NL15 | +0.073628 [+0.040875, +0.119350] | +0.007409 [+0.001190, +0.011900] | 1769.52 [1698.90, 1855.96] |
| P0 | 30001 | MCS40 | +0.053501 [+0.010395, +0.109003] | +0.000728 [+0.000000, +0.006552] | 1679.50 [1618.23, 1751.32] |
| P0 | 30002 | MCS40 | +0.068559 [+0.025893, +0.134133] | +0.007186 [+0.001968, +0.012909] | 1714.81 [1653.27, 1785.86] |
| P0 | 30003 | MCS40 | +0.061588 [+0.030947, +0.103933] | +0.005637 [+0.001379, +0.012269] | 1763.37 [1696.97, 1846.04] |
| P1 | 30001 | BASE | +0.048215 [+0.018693, +0.092345] | +0.003000 [+0.000000, +0.012335] | 1697.13 [1635.96, 1768.50] |
| P1 | 30002 | BASE | +0.048658 [+0.022236, +0.085026] | +0.003624 [+0.000000, +0.007782] | 1693.84 [1633.40, 1762.72] |
| P1 | 30003 | BASE | +0.038294 [+0.016389, +0.060871] | +0.003253 [+0.000000, +0.010809] | 1734.68 [1670.34, 1814.21] |
| P1 | 30001 | NL15 | +0.076944 [+0.039574, +0.130406] | +0.016096 [+0.008624, +0.021135] | 1757.59 [1696.38, 1833.18] |
| P1 | 30002 | NL15 | +0.047437 [+0.029535, +0.068632] | +0.003867 [+0.000000, +0.008983] | 1734.68 [1669.27, 1812.63] |
| P1 | 30003 | NL15 | +0.073295 [+0.033887, +0.133865] | +0.009586 [+0.003187, +0.017012] | 1726.57 [1666.27, 1799.99] |
| P1 | 30001 | MCS40 | +0.070544 [+0.033001, +0.123529] | +0.007636 [+0.002396, +0.016997] | 1705.42 [1645.71, 1774.50] |
| P1 | 30002 | MCS40 | +0.027899 [+0.007490, +0.048687] | +0.000000 [+0.000000, +0.000266] | 1798.02 [1716.38, 1907.38] |
| P1 | 30003 | MCS40 | +0.045390 [+0.027445, +0.066615] | +0.000000 [+0.000000, +0.004637] | 1742.50 [1672.32, 1827.42] |
| P2 | 30001 | BASE | +0.043601 [+0.017288, +0.071770] | +0.006326 [+0.000781, +0.010521] | 1716.88 [1656.11, 1787.37] |
| P2 | 30002 | BASE | +0.031840 [+0.010141, +0.054392] | +0.003624 [+0.000000, +0.006960] | 1740.78 [1673.94, 1822.97] |
| P2 | 30003 | BASE | +0.054721 [+0.033314, +0.079754] | +0.008145 [+0.003611, +0.013831] | 1726.57 [1665.97, 1799.22] |
| P2 | 30001 | NL15 | +0.047241 [+0.016076, +0.089471] | +0.003000 [+0.000000, +0.009627] | 1709.11 [1645.00, 1782.68] |
| P2 | 30002 | NL15 | +0.063382 [+0.036380, +0.094651] | +0.011913 [+0.007555, +0.018370] | 1745.97 [1684.11, 1819.13] |
| P2 | 30003 | NL15 | +0.058405 [+0.028106, +0.093878] | +0.010809 [+0.005804, +0.016901] | 1734.47 [1676.06, 1804.11] |
| P2 | 30001 | MCS40 | +0.041817 [+0.011464, +0.076890] | +0.006691 [+0.001726, +0.011657] | 1701.17 [1641.62, 1768.93] |
| P2 | 30002 | MCS40 | +0.057138 [+0.025633, +0.105264] | +0.005663 [+0.000000, +0.010809] | 1711.65 [1650.45, 1782.68] |
| P2 | 30003 | MCS40 | +0.043542 [+0.019949, +0.075009] | +0.000000 [+0.000000, +0.005804] | 1739.03 [1667.85, 1825.07] |

Root-wise paired drop-top-10 medians across the three partitions: 30001: -0.006599 CI95 [-0.015566, +0.003125] LCB95 -0.014090, 30002: -0.007319 CI95 [-0.020289, +0.003399] LCB95 -0.017756, 30003: -0.007260 CI95 [-0.015889, +0.001838] LCB95 -0.014266.
Configuration agreement: 10/36 fold/root choices (CONFIG_SENSITIVE). Full family mean/literal median CIs, drop-top-1/5/10, LOFO, W/T/L, harm, invalid, switches, root results and Elo CIs are retained in metrics.json. Ordinary and paired bootstrap:10,000 whole-family draws/RNG379191. Primary paired comparator H352 NL15; secondary BASE retained separately in metrics.json.

Decision gates: paired_drop_top10_at_least_002=False; gain_retention=True; harmed_nonincrease=False; positive_on_two_seeds=False.

Verification: exact-version archived BASE regression passed 9/9 panels, all 36 fold/root pick and lock hashes, and all three exact medians. Frozen H352 NL15 regression also passed all9panels/36locks/36scores/pick hashes/raw family arrays exactly. No BASE/NL15 refits or new reference predictions. Current TRAIN identities matched. Preparation processes ended before the cold fit process launched; no preparation packet cache was inherited. All 36 fsynced fit locks and audited models preceded once-only treatment outer scoring. Native audits checked 29,484 ranker fits (only min_data_in_leaf delta vs H352) and 7,020 unchanged veto fits; all 108 final veto hashes equal BASE. Zero fit-worker outer payload reads before their locks; zero DL-Val/dev91/DL-Test/sealed payload rows. All 55 data-free harness controls and seven adversarial custody controls passed. Same-agent red-team recomputed the paired bootstrap and verdict exactly.

Resources: xbabe2 CPU only, cores [0, 1, 2, 3, 6, 7, 8, 9], 8 workers, numeric threads1, nice19, MemoryMax6G/MemorySwapMax0. One admission probe; initial and first-completed-task forecasts passed. First-task conservative forecast ended 2026-11-05T18:04:05.557644+00:00. Study cgroup CPU 27970.587s; peak memory 3.513GiB. Observed host CPU60s max 51.17% (5s diagnostic max 95.55%), load max 97.00 with at most0 consecutive60s samples above108, MemAvailable min 66.30GiB, own scratch max 1395588410 bytes. Corrected guard uses CPU>85 averaged60s/load1>108 on3consecutive60s/MemAvailable20GiB/scratch2GB. Compute deadline19:15Z and close19:30Z Nov5 were fixed. No speed verdict.

Custody: create-only `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H379/jrb01-20261105T162900Z/closeout/closeout-manifest.json`, VersionId `.tCUoOn3oxEuTuNSyiL6geltFdnG8uF1`, SHA256 `792f7340c6c8b01556db4ab9203c386cd69e790cb128db87a6059fbb6e0dde3a`. The manifest binds 1842 files in 24 parts; every exact-version GET, whole-object hash and member hash was verified before removing only H379 scratch. Execution source/vendor and detailed local receipts remain in `/home/ubuntu/jope_plan/evidence/H379/jrb01-20261105-t1629/`. No models are in Git.

Observed ranker tree leaf-count histogram across recorded fits: {'4': 136748, '5': 824038, '6': 1076589, '7': 1627954, '8': 1538671}. Configured15/depth3 therefore permits only up to8 realized leaves; this is diagnostic, not another intervention.

Sequential exploratory sweep H348–H379 carries multiplicity/selection bias. No live adoption; any ADVANCE requires a separately authorized one-shot dev91 confirmation. H379 reads zero dev91 rows.

DOPE `c51a56d0` was not loaded or changed; this was a CPU selection-layer experiment.

Acceptance command, after reconstructing the archived original source/vendor layout in approved scratch: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 python3 -B -m unittest discover -s /home/ubuntu/jope-data/h379-jrb01/source -p "test_*.py" -v`. These controls perform no model fits. The published manifest pins all executed fitting code and vendor hashes; detailed execution and custody receipts bind the actual run.

Next: retain H352 NL15; do not advance MCS40. Any further intervention requires a new preregistered study.
