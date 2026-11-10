# H363 — NL15 ranker min_child_samples 10

**MCS10_NOT_ADVANCED**. The fixed MCS10 treatment completed all 36 fold/root tasks and nine partition/root panels on TRAIN only.
Paired drop-top-10 gain: -0.008176; whole-family 95% CI [-0.015762, +0.001286]; one-sided LCB95 -0.014009. The advance bar is +0.002 with positive LCB95 and all three retention/harm/root gates.

Hypothesis: lowering the NL15 ranker leaf floor may resolve smaller context groups. This was one fixed setting, selected before fitting; there was no tuning.

Primary frozen reference: H352 NL15 from committed02dfee2f7e0b8543324494dd4f99d074cd6b93b6. Secondary reference: archived BASE. P0/P1/P2 ×roots30001–30003 ×four family-grouped folds,1219 TRAIN contexts/205 families. Only existing ranker min_child_samples20→10/native min_data_in_leaf20→10 changed, in both pinned ranker functions and all inner/final/eight bag rankers. Retain configured num_leaves15/max_depth3 (at most8 realized binary leaves), original learning-rate/tree options and adaptive selection logic, labels/groups/seeds and every veto option. Restoring20 restores each entire H352 AST. Every native fitted model passed the full H352 config-only native contract; the learning-rate/tree-count fields are normalized to values selected by the unchanged grid, all other fields including derived seeds compare exactly except the prescribed ranker leaf floor.

Preregistration commit: `7899a399269112164a9af857eea02e480c11f17b`, published 2026-11-05T11:50:06.947825+00:00; first treatment fit 2026-11-05T11:51:53.427251+00:00. See [PREREG.md](PREREG.md) and [metrics.json](metrics.json).

| Aggregate over nine partition/root cells | BASE | H352 NL15 | MCS10 |
|---|---:|---:|---:|
| Median family-equal mean gain | +0.048215 | +0.058405 | +0.053981 |
| Whole-family 95% CI | [+0.025426, +0.076536] | [+0.034410, +0.092178] | [+0.025814, +0.085032] |
| Literal family-median gain | +0.003624 | +0.007409 | +0.005706 |
| Literal median 95% CI | [+0.000296, +0.007509] | [+0.002351, +0.011715] | [+0.001923, +0.010809] |
| Median harmed share (gain < −0.01) | 0.092683 | 0.092683 | 0.092683 |
| Median Elo | 1725.66 | 1734.47 | 1734.68 |

**Elo reference:** reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.

| Partition | Root | Arm | Family-equal gain [95% CI] | Literal family-median gain [95% CI] | Elo [95% CI] |
|---|---:|---|---:|---:|---:|
| P0 | 30001 | BASE | +0.041289 [+0.021447, +0.064626] | +0.000000 [+0.000000, +0.000000] | 1725.66 [1654.44, 1811.17] |
| P0 | 30002 | BASE | +0.064316 [+0.015215, +0.133088] | +0.001588 [+0.000000, +0.007380] | 1722.48 [1656.59, 1800.75] |
| P0 | 30003 | BASE | +0.072258 [+0.023396, +0.140718] | +0.006984 [+0.003011, +0.013091] | 1727.10 [1664.04, 1799.48] |
| P0 | 30001 | NL15 | +0.050668 [+0.017757, +0.086589] | +0.004840 [+0.000000, +0.010407] | 1733.39 [1669.17, 1812.51] |
| P0 | 30002 | NL15 | +0.047569 [+0.022129, +0.079545] | +0.000000 [+0.000000, +0.000704] | 1704.25 [1636.09, 1786.33] |
| P0 | 30003 | NL15 | +0.073628 [+0.040735, +0.119414] | +0.007409 [+0.001190, +0.011900] | 1769.52 [1700.46, 1852.50] |
| P0 | 30001 | MCS10 | +0.055106 [+0.018867, +0.097988] | +0.005706 [+0.000296, +0.011835] | 1736.38 [1672.52, 1813.81] |
| P0 | 30002 | MCS10 | +0.053981 [+0.027368, +0.084899] | +0.004674 [+0.000669, +0.011715] | 1726.65 [1663.34, 1800.82] |
| P0 | 30003 | MCS10 | +0.075248 [+0.029818, +0.142688] | +0.006552 [+0.001777, +0.012485] | 1695.11 [1636.34, 1761.24] |
| P1 | 30001 | BASE | +0.048215 [+0.018442, +0.092487] | +0.003000 [+0.000000, +0.012335] | 1697.13 [1635.96, 1768.04] |
| P1 | 30002 | BASE | +0.048658 [+0.021945, +0.084337] | +0.003624 [+0.000000, +0.007782] | 1693.84 [1635.08, 1763.37] |
| P1 | 30003 | BASE | +0.038294 [+0.016233, +0.060410] | +0.003253 [+0.000000, +0.010809] | 1734.68 [1669.79, 1811.19] |
| P1 | 30001 | NL15 | +0.076944 [+0.040358, +0.129781] | +0.016096 [+0.008624, +0.021135] | 1757.59 [1696.25, 1833.40] |
| P1 | 30002 | NL15 | +0.047437 [+0.029234, +0.067425] | +0.003867 [+0.000000, +0.008983] | 1734.68 [1669.72, 1812.63] |
| P1 | 30003 | NL15 | +0.073295 [+0.033909, +0.133826] | +0.009586 [+0.003187, +0.017012] | 1726.57 [1664.47, 1796.72] |
| P1 | 30001 | MCS10 | +0.047263 [+0.021823, +0.082828] | +0.000847 [+0.000000, +0.005611] | 1745.71 [1677.05, 1829.47] |
| P1 | 30002 | MCS10 | +0.039289 [+0.015445, +0.063931] | +0.002802 [+0.000000, +0.010340] | 1739.22 [1673.18, 1820.09] |
| P1 | 30003 | MCS10 | +0.035572 [+0.004503, +0.071118] | +0.002436 [+0.000000, +0.005760] | 1734.68 [1668.23, 1812.70] |
| P2 | 30001 | BASE | +0.043601 [+0.017482, +0.071605] | +0.006326 [+0.000781, +0.010521] | 1716.88 [1655.57, 1787.68] |
| P2 | 30002 | BASE | +0.031840 [+0.010301, +0.053756] | +0.003624 [+0.000000, +0.006960] | 1740.78 [1673.94, 1822.09] |
| P2 | 30003 | BASE | +0.054721 [+0.033349, +0.079426] | +0.008145 [+0.003611, +0.013831] | 1726.57 [1665.09, 1797.30] |
| P2 | 30001 | NL15 | +0.047241 [+0.015935, +0.090094] | +0.003000 [+0.000000, +0.009627] | 1709.11 [1646.59, 1783.82] |
| P2 | 30002 | NL15 | +0.063382 [+0.036222, +0.093948] | +0.011913 [+0.007555, +0.018370] | 1745.97 [1685.21, 1817.38] |
| P2 | 30003 | NL15 | +0.058405 [+0.027507, +0.094235] | +0.010809 [+0.005804, +0.016901] | 1734.47 [1676.06, 1805.06] |
| P2 | 30001 | MCS10 | +0.036953 [+0.003094, +0.067815] | +0.008385 [+0.002565, +0.012991] | 1727.52 [1665.09, 1799.22] |
| P2 | 30002 | MCS10 | +0.058279 [+0.023767, +0.108337] | +0.010698 [+0.006326, +0.016504] | 1750.03 [1686.64, 1825.68] |
| P2 | 30003 | MCS10 | +0.071887 [+0.037643, +0.116252] | +0.011210 [+0.006607, +0.019908] | 1714.38 [1655.57, 1780.73] |

Root-wise paired drop-top-10 medians across the three partitions: 30001: -0.003204 CI95 [-0.021062, +0.008181] LCB95 -0.018223, 30002: -0.009166 CI95 [-0.020328, +0.003670] LCB95 -0.017662, 30003: -0.008176 CI95 [-0.020964, +0.004208] LCB95 -0.018987.
Configuration agreement: 7/36 fold/root choices (CONFIG_SENSITIVE). Full family mean/literal median CIs, drop-top-1/5/10, LOFO, W/T/L, harm, invalid, switches, root results and Elo CIs are retained in metrics.json. Ordinary and paired bootstrap:10,000 whole-family draws/RNG363191. Primary paired comparator H352 NL15; secondary BASE retained separately in metrics.json.

Decision gates: paired_drop_top10_at_least_002=False; gain_retention=True; harmed_nonincrease=True; positive_on_two_seeds=False.

Verification: exact-version archived BASE regression passed 9/9 panels, all 36 fold/root pick and lock hashes, and all three exact medians. Frozen H352 NL15 regression also passed all9panels/36locks/36scores/pick hashes/raw family arrays exactly. No BASE/NL15 refits or new reference predictions. Current TRAIN identities matched. Preparation processes ended before the cold fit process launched; no preparation packet cache was inherited. All 36 fsynced fit locks and audited models preceded once-only treatment outer scoring. Native audits checked 29,484 ranker fits (only min_data_in_leaf delta vs H352) and 7,020 unchanged veto fits; all 108 final veto hashes equal BASE. Zero fit-worker outer payload reads before their locks; zero DL-Val/dev91/DL-Test/sealed payload rows. All 55 data-free harness controls and seven adversarial custody controls passed. Same-agent red-team recomputed the paired bootstrap and verdict exactly.

Resources: xbabe2 CPU only, cores [1, 2, 3, 4, 5, 6, 7, 8], 8 workers, numeric threads1, nice19, MemoryMax6G/MemorySwapMax0. One admission probe; initial and first-completed-task forecasts passed. First-task conservative forecast ended 2026-11-05T13:23:02.955521+00:00. Study cgroup CPU 35427.905s; peak memory 3.194GiB. Observed host CPU60s max 54.28% (5s diagnostic max 59.28%), load max 116.30 with at most2 consecutive60s samples above108, MemAvailable min 65.60GiB, own scratch max 1500160473 bytes. Corrected guard uses CPU>85 averaged60s/load1>108 on3consecutive60s/MemAvailable20GiB/scratch2GB. Compute deadline15:15Z and close15:30Z Nov5 were fixed. No speed verdict.

Custody: create-only `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H363/jrb01-20261105T112500Z/closeout/closeout-manifest.json`, VersionId `AuL.EVrj_6rKi20.EZKtFywSck86aevH`, SHA256 `5cfb0c3744d2d4644a66497c6bbdbaca5c9b72ed29bf43bc70dc1c6646e7b712`. The manifest binds 1850 files in 26 parts; every exact-version GET, whole-object hash and member hash was verified before removing only H363 scratch. Execution source/vendor and detailed local receipts remain in `/home/ubuntu/jope_plan/evidence/H363/jrb01-20261105-t1125/`. No models are in Git.

Observed ranker tree leaf-count histogram across recorded fits: {'4': 18063, '5': 275531, '6': 542493, '7': 1323204, '8': 3084309}. Configured15/depth3 therefore permits only up to8 realized leaves; this is diagnostic, not another intervention.

Sequential exploratory sweep H348–H363 carries multiplicity/selection bias. No live adoption; any ADVANCE requires a separately authorized one-shot dev91 confirmation. H363 reads zero dev91 rows.

DOPE `c51a56d0` was not loaded or changed; this was a CPU selection-layer experiment.

Acceptance command, after reconstructing the archived original source/vendor layout in approved scratch: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 python3 -B -m unittest discover -s /home/ubuntu/jope-data/h363-jrb01/source -p "test_*.py" -v`. These controls perform no model fits. The published manifest pins all executed fitting code and vendor hashes; detailed execution and custody receipts bind the actual run.

Next: retain H352 NL15; do not advance MCS10. Any further intervention requires a new preregistered study.
