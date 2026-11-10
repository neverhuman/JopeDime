# H344 — LambdaRank sigmoid=0.5

**SIGMA_NOT_ADVANCED**. The fixed sigmoid treatment completed all 36 fold/root tasks and nine partition/root panels on TRAIN only.
Paired drop-top-10 gain: -0.002442; whole-family 95% CI [-0.008506, +0.001976]; one-sided LCB95 -0.007626. The advance bar is +0.002 with positive LCB95 and all three retention/harm/root gates.

Hypothesis: flattening LambdaRank’s pairwise logistic loss may improve robustness. This was one fixed setting, selected before fitting; there was no tuning.

BASE is unchanged archived H303 NESTED on H308 P0/P1/P2, roots 30001–30003, four family-grouped outer folds; 1,219 TRAIN contexts from 205 families. Only `sigmoid=0.5` was added to both pinned H287 ranker functions, covering inner/final/all eight bag rankers. BASE native sigmoid was 1.0 on all 972 archived refit rankers. Every other option, labels, seeds, 18-config grid, veto, tie rules and invalid sentinel stayed fixed. Removing this keyword restores each entire BASE function AST.

Preregistration commit: `66ef7b3e75b69bdbb0098a9322473e4799c63d90`, published 2026-11-05T00:16:08.901925+00:00; first treatment fit 2026-11-05T00:20:55.999194+00:00. See [PREREG.md](PREREG.md) and [metrics.json](metrics.json).

| Aggregate over nine partition/root cells | BASE | SIGMA |
|---|---:|---:|
| Median family-equal mean gain | +0.048215 | +0.044539 |
| Whole-family 95% CI | [+0.025157, +0.076452] | [+0.021322, +0.076293] |
| Literal family-median gain | +0.003624 | +0.002436 |
| Literal median 95% CI | [+0.000299, +0.007726] | [+0.000000, +0.006704] |
| Median harmed share (gain < −0.01) | 0.092683 | 0.102439 |
| Median Elo | 1725.66 | 1717.59 |

**Elo reference:** reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.

| Partition | Root | Arm | Family-equal gain [95% CI] | Literal family-median gain [95% CI] | Elo [95% CI] |
|---|---:|---|---:|---:|---:|
| P0 | 30001 | BASE | +0.041289 [+0.021971, +0.064441] | +0.000000 [+0.000000, +0.000000] | 1725.66 [1652.45, 1812.97] |
| P0 | 30002 | BASE | +0.064316 [+0.015120, +0.133592] | +0.001588 [+0.000000, +0.007380] | 1722.48 [1656.71, 1799.21] |
| P0 | 30003 | BASE | +0.072258 [+0.022549, +0.142169] | +0.006984 [+0.003011, +0.013091] | 1727.10 [1665.65, 1798.70] |
| P0 | 30001 | SIGMA | +0.064614 [+0.033034, +0.108441] | +0.001957 [+0.000000, +0.006984] | 1734.46 [1667.23, 1815.70] |
| P0 | 30002 | SIGMA | +0.062085 [+0.012430, +0.131393] | +0.006830 [+0.002565, +0.011988] | 1717.59 [1656.23, 1787.37] |
| P0 | 30003 | SIGMA | +0.059347 [+0.011016, +0.128563] | +0.001280 [+0.000000, +0.006984] | 1681.43 [1622.06, 1751.31] |
| P1 | 30001 | BASE | +0.048215 [+0.018017, +0.093011] | +0.003000 [+0.000000, +0.012335] | 1697.13 [1636.28, 1768.50] |
| P1 | 30002 | BASE | +0.048658 [+0.022191, +0.084701] | +0.003624 [+0.000000, +0.007782] | 1693.84 [1633.76, 1761.95] |
| P1 | 30003 | BASE | +0.038294 [+0.016006, +0.060059] | +0.003253 [+0.000000, +0.010809] | 1734.68 [1668.06, 1812.70] |
| P1 | 30001 | SIGMA | +0.043715 [+0.019073, +0.076001] | +0.002436 [+0.000000, +0.006985] | 1719.97 [1654.63, 1793.81] |
| P1 | 30002 | SIGMA | +0.044539 [+0.010014, +0.088744] | +0.000000 [+0.000000, +0.003583] | 1699.63 [1635.51, 1776.48] |
| P1 | 30003 | SIGMA | +0.031418 [+0.016228, +0.048177] | +0.000000 [+0.000000, +0.000669] | 1748.49 [1674.26, 1841.44] |
| P2 | 30001 | BASE | +0.043601 [+0.017676, +0.071963] | +0.006326 [+0.000781, +0.010521] | 1716.88 [1656.11, 1787.68] |
| P2 | 30002 | BASE | +0.031840 [+0.010274, +0.053433] | +0.003624 [+0.000000, +0.006960] | 1740.78 [1674.77, 1820.09] |
| P2 | 30003 | BASE | +0.054721 [+0.033837, +0.080179] | +0.008145 [+0.003611, +0.013831] | 1726.57 [1665.97, 1796.09] |
| P2 | 30001 | SIGMA | +0.038639 [+0.013089, +0.066104] | +0.002822 [+0.000000, +0.006975] | 1714.46 [1650.24, 1788.75] |
| P2 | 30002 | SIGMA | +0.048447 [+0.023111, +0.074179] | +0.012283 [+0.006370, +0.018916] | 1725.67 [1665.42, 1796.72] |
| P2 | 30003 | SIGMA | +0.035273 [+0.005414, +0.064553] | +0.006552 [+0.002955, +0.013051] | 1696.66 [1637.85, 1764.77] |

Root-wise paired drop-top-10 medians across the three partitions: 30001: -0.000634, 30002: -0.002442, 30003: -0.011378.
Configuration agreement: 7/36 fold/root choices (CONFIG_SENSITIVE). Full family mean/literal median CIs, drop-top-1/5/10, LOFO, W/T/L, harm, invalid, switches, root results and Elo CIs are retained in metrics.json. Ordinary bootstrap: 10,000 whole-family draws/RNG302150; paired decision: 10,000/RNG344191.

Decision gates: paired_drop_top10_at_least_002=False; gain_retention=True; harmed_nonincrease=False; positive_on_two_seeds=False.

Verification: exact-version archived BASE regression passed 9/9 panels, all 36 fold/root pick and lock hashes, and all three exact medians. No BASE refits. Current TRAIN identities matched. Preparation processes ended before the cold fit process launched; no preparation packet cache was inherited. All 36 fsynced fit locks and audited models preceded once-only treatment outer scoring. Native audits checked 29,484 ranker fits (only sigmoid delta) and 7,020 unchanged veto fits; all 108 final veto hashes equal BASE. Zero fit-worker outer payload reads before their locks; zero DL-Val/dev91/DL-Test/sealed payload rows. All 34 data-free harness controls and seven adversarial custody controls passed. Same-agent red-team recomputed the paired bootstrap and verdict exactly.

Resources: xbabe2 CPU only, cores [54, 56, 57, 58, 61, 63, 16, 17], eight workers, numeric threads1, nice19, MemoryMax6G/MemorySwapMax0. One admission probe; initial and first-completed-task forecasts passed. First-task conservative forecast ended 2026-11-05T02:03:56.542284+00:00. Study cgroup CPU 26110.639s; peak memory 3.304GiB. Observed host CPU max 37.47%, load max 56.34, MemAvailable min 69.84GiB, own scratch max 1524019863 bytes. Compute deadline03:45Z and close04:00Z Nov5 were fixed. No speed verdict.

Custody: create-only `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H344/jrb01-20261104T235500Z/closeout/closeout-manifest.json`, VersionId `DbdfoEFAm03uf1GGjwH0FFfGZarmM6Qa`, SHA256 `bda6bf5560b4fdefc783f6f8bef16841ee3f067bf5b09836a9fa2b2ecd9b2186`. The manifest binds 2544 files in 25 parts; every exact-version GET, whole-object hash and member hash was verified before removing only H344 scratch. Execution source/vendor and detailed local receipts remain in `/home/ubuntu/jope_plan/evidence/H344/jrb01-20261104-t2355/`. No models are in Git.

DOPE `c51a56d0` was not loaded or changed; this was a CPU selection-layer experiment.

Acceptance command, after reconstructing the archived original source/vendor layout in approved scratch: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 python3 -B -m unittest discover -s /home/ubuntu/jope-data/h344-jrb01/source -p "test_*.py" -v`. These controls perform no model fits. The published manifest pins all executed fitting code and vendor hashes; detailed execution and custody receipts bind the actual run.

Next: retain BASE for this comparison; do not advance sigmoid=0.5. Any further intervention requires a new preregistered study.
