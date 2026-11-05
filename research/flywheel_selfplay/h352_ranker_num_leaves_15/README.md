# H352 — ranker num_leaves 15

**ADVANCE**. The fixed NL15 treatment completed all 36 fold/root tasks and nine partition/root panels on TRAIN only.
Paired drop-top-10 gain: +0.006371; whole-family 95% CI [+0.000059, +0.015092]; one-sided LCB95 +0.000974. The advance bar is +0.002 with positive LCB95 and all three retention/harm/root gates.

Hypothesis: modestly increasing realized leaf capacity may reduce underfitting. This was one fixed setting, selected before fitting; there was no tuning.

BASE is unchanged archived H303 NESTED on H308 P0/P1/P2, roots 30001–30003, four family-grouped outer folds; 1,219 TRAIN contexts from 205 families. Only the existing ranker `num_leaves` value changed from 7 to 15 in both pinned H287 ranker functions, covering inner/final/all eight bag rankers. BASE native limits were `num_leaves=7`, `max_depth=3` on all 972 archived refit rankers; `max_depth=3` remains, so realized binary ranker trees have at most 8 leaves, and BASE regularization/LambdaRank/sigmoid1 remain. Every other option, labels, seeds, 18-config grid, veto, tie rules and invalid sentinel stayed fixed. Restoring only `num_leaves=7` restores each entire BASE function AST.

Preregistration commit: `934693e421009a8580bdb49c5b8984c77d10d442`, published 2026-11-05T08:14:28.715488+00:00; first treatment fit 2026-11-05T08:17:15.322843+00:00. See [PREREG.md](PREREG.md) and [metrics.json](metrics.json).

| Aggregate over nine partition/root cells | BASE | NL15 |
|---|---:|---:|
| Median family-equal mean gain | +0.048215 | +0.058405 |
| Whole-family 95% CI | [+0.025157, +0.076452] | [+0.034267, +0.092537] |
| Literal family-median gain | +0.003624 | +0.007409 |
| Literal median 95% CI | [+0.000299, +0.007726] | [+0.002460, +0.011715] |
| Median harmed share (gain < −0.01) | 0.092683 | 0.092683 |
| Median Elo | 1725.66 | 1734.47 |

**Elo reference:** reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.

| Partition | Root | Arm | Family-equal gain [95% CI] | Literal family-median gain [95% CI] | Elo [95% CI] |
|---|---:|---|---:|---:|---:|
| P0 | 30001 | BASE | +0.041289 [+0.021971, +0.064441] | +0.000000 [+0.000000, +0.000000] | 1725.66 [1652.45, 1812.97] |
| P0 | 30002 | BASE | +0.064316 [+0.015120, +0.133592] | +0.001588 [+0.000000, +0.007380] | 1722.48 [1656.71, 1799.21] |
| P0 | 30003 | BASE | +0.072258 [+0.022549, +0.142169] | +0.006984 [+0.003011, +0.013091] | 1727.10 [1665.65, 1798.70] |
| P0 | 30001 | NL15 | +0.050668 [+0.016880, +0.088093] | +0.004840 [+0.000000, +0.010407] | 1733.39 [1668.68, 1811.19] |
| P0 | 30002 | NL15 | +0.047569 [+0.022390, +0.080198] | +0.000000 [+0.000000, +0.000704] | 1704.25 [1637.35, 1784.42] |
| P0 | 30003 | NL15 | +0.073628 [+0.040405, +0.121147] | +0.007409 [+0.001190, +0.011900] | 1769.52 [1700.74, 1857.30] |
| P1 | 30001 | BASE | +0.048215 [+0.018017, +0.093011] | +0.003000 [+0.000000, +0.012335] | 1697.13 [1636.28, 1768.50] |
| P1 | 30002 | BASE | +0.048658 [+0.022191, +0.084701] | +0.003624 [+0.000000, +0.007782] | 1693.84 [1633.76, 1761.95] |
| P1 | 30003 | BASE | +0.038294 [+0.016006, +0.060059] | +0.003253 [+0.000000, +0.010809] | 1734.68 [1668.06, 1812.70] |
| P1 | 30001 | NL15 | +0.076944 [+0.040063, +0.131361] | +0.016096 [+0.008624, +0.021135] | 1757.59 [1696.13, 1831.38] |
| P1 | 30002 | NL15 | +0.047437 [+0.029741, +0.067875] | +0.003867 [+0.000000, +0.008983] | 1734.68 [1669.18, 1812.73] |
| P1 | 30003 | NL15 | +0.073295 [+0.033450, +0.134862] | +0.009586 [+0.003187, +0.017012] | 1726.57 [1664.47, 1798.70] |
| P2 | 30001 | BASE | +0.043601 [+0.017676, +0.071963] | +0.006326 [+0.000781, +0.010521] | 1716.88 [1656.11, 1787.68] |
| P2 | 30002 | BASE | +0.031840 [+0.010274, +0.053433] | +0.003624 [+0.000000, +0.006960] | 1740.78 [1674.77, 1820.09] |
| P2 | 30003 | BASE | +0.054721 [+0.033837, +0.080179] | +0.008145 [+0.003611, +0.013831] | 1726.57 [1665.97, 1796.09] |
| P2 | 30001 | NL15 | +0.047241 [+0.016001, +0.090062] | +0.003000 [+0.000000, +0.009627] | 1709.11 [1647.15, 1782.65] |
| P2 | 30002 | NL15 | +0.063382 [+0.036040, +0.094375] | +0.011913 [+0.007555, +0.018370] | 1745.97 [1684.26, 1818.30] |
| P2 | 30003 | NL15 | +0.058405 [+0.027179, +0.094988] | +0.010809 [+0.005804, +0.016901] | 1734.47 [1674.02, 1805.31] |

Root-wise paired drop-top-10 medians across the three partitions: 30001: -0.001718 CI95 [-0.011203, +0.017082] LCB95 -0.008836, 30002: +0.006371 CI95 [-0.002078, +0.020691] LCB95 -0.000382, 30003: +0.009635 CI95 [-0.004104, +0.018981] LCB95 -0.002325.
Two of three root medians are positive; root 30001 is negative, and every per-root 95% interval crosses zero. The preregistered aggregate gate passes; the root intervals retain that uncertainty.

Configuration agreement: 9/36 fold/root choices (CONFIG_SENSITIVE). Full family mean/literal median CIs, drop-top-1/5/10, LOFO, W/T/L, harm, invalid, switches, root results and Elo CIs are retained in metrics.json. Ordinary bootstrap: 10,000 whole-family draws/RNG302150; paired decision: 10,000/RNG352191.

Decision gates: paired_drop_top10_at_least_002=True; gain_retention=True; harmed_nonincrease=True; positive_on_two_seeds=True.

Verification: exact-version archived BASE regression passed 9/9 panels, all 36 fold/root pick and lock hashes, and all three exact medians. No BASE refits. Current TRAIN identities matched. Preparation processes ended before the cold fit process launched; no preparation packet cache was inherited. All 36 fsynced fit locks and audited models preceded once-only treatment outer scoring. Native audits checked 29,484 ranker fits (only num_leaves delta) and 7,020 unchanged veto fits; all 108 final veto hashes equal BASE. Zero fit-worker outer payload reads before their locks; zero DL-Val/dev91/DL-Test/sealed payload rows. All 43 data-free harness controls and seven adversarial custody controls passed. Same-agent red-team recomputed the paired bootstrap and verdict exactly.

Resources: xbabe2 CPU only, cores [1, 2, 3, 4, 5, 6, 9, 10], 8 workers, numeric threads1, nice19, MemoryMax6G/MemorySwapMax0. One admission probe; initial and first-completed-task forecasts passed. First-task conservative forecast ended 2026-11-05T09:51:21.574078+00:00. Study cgroup CPU 26189.517s; peak memory 3.608GiB. Observed host CPU max 58.75%, load max 65.20, MemAvailable min 68.19GiB, own scratch max 1587025974 bytes. Compute deadline11:15Z and close11:30Z Nov5 were fixed. No speed verdict.

Custody: create-only `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H352/jrb01-20261105T075500Z/closeout/closeout-manifest.json`, VersionId `RtNLAfmRJbXqdXQx7nxZLZ8dWHPcVYQC`, SHA256 `d3b47c5fbc1ecc3c6e6d4f89dfef7d6a3f5216f45c6dd1589ce94fb413350f60`. The manifest binds 2691 files in 27 parts; every exact-version GET, whole-object hash and member hash was verified before removing only H352 scratch. Execution source/vendor and detailed local receipts remain in `/home/ubuntu/jope_plan/evidence/H352/jrb01-20261105-t0755/`. No models are in Git.

Observed ranker tree leaf-count histogram across recorded fits: {'4': 48478, '5': 492692, '6': 824961, '7': 1545694, '8': 2278975}. Configured `num_leaves=15` with `max_depth=3` permits at most 8 realized leaves; this is diagnostic, not another intervention.

DOPE `c51a56d0` was not loaded or changed; this was a CPU selection-layer experiment.

Acceptance command, after reconstructing the archived original source/vendor layout in approved scratch: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 python3 -B -m unittest discover -s /home/ubuntu/jope-data/h352-jrb01/source -p "test_*.py" -v`. These controls perform no model fits. The published manifest pins all executed fitting code and vendor hashes; detailed execution and custody receipts bind the actual run.

Next: a separately authorized one-shot dev91 comparison against r3175. It was not run in H352.
