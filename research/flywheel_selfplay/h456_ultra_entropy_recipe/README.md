# Wave H456 — Ultra-Entropy `104M` Neural-GP Flywheel Recipe (`h456_ultra_entropy_recipe`)

> [!IMPORTANT]
> **In-Corpus `dev91` Disclosure & Quarantine:** All `dev91` metrics reported in the H454–H456 flywheel studies are **in-corpus** (`canonical_promotion_forbidden = true`). Startup and runtime quarantine checks enforce `0` `CONFIRM`, `0` `FMGAP` (Parts B/C or any FMGAP cohort), `0` `HOLDOUT`, `0` `SEALED`, and `0` `DL-VAL` opens.

## Summary of Recipe Upgrades over Wave H455
1. **Exact `1 / 2,389` Uniform S3 Source Mixing (`mmap_bank_h456.py`):**
   - `s3_source_exact_uniform` assigns exact `1 / (2,389 * |family|)` probability mass (`max/min` family mass = **`1.000000x`**, eliminating H455's `1.50x` soft-2-cap skew and `5%` regret tilt).
   - Variance-reduced **shuffled `2,389`-family permutation deck** (`sample_s3_exact_deck_table_idx`) visits every single S3 source family once per cycle before any source repeats, paired with exact `1 / N_shard_families` hierarchical Exact-GP teacher shard sampling.
2. **Mandatory Simultaneous Feature Removal + 28-Family Zero-Leakage Injection (`ultra_entropy_ops_h456.py`):**
   - Reserves `k_inj = round(U(0.10, 0.70) * F_use)` synthetic/distractor slots *before* raw column subsetting so wide tables (`p_raw >= F_use`, `F_use in [4..80]`) also undergo simultaneous column drop (`[0.05..0.90]`, `drop <= 0.75` on teacher KD/residual streams) and synthetic replacement.
   - Expands strictly context-fitted (`Xc, yc`) synthetic/distractor generation from 22 to **28 zero-leakage families**, adding OOF multi-feature Ridge-GCV, depth-2 oblivious decision-stump ensembles, context quantile hinge splines, copula rank-differences, adversarial **spurious context-overfit / pseudo-ID distractors**, and MNAR censored twins.
3. **Continuous 4-Regime Varied Row Sampling (`N_use in [32..2048]`, `p_ctx in [0.05..0.96]`):**
   - Covers small-N `[32..95]` episodes (`60%` log-uniform + `40%` uniform `N_use`) with OOM-exact `B * N_use^2` batch sizing and 4 fresh full-table row regimes (`40%` uniform, `30%` feature-quantile stratified, `15%` covariate-projection slice, `15%` soft leverage-tilted).
4. **Stronger Feature Permute / Scale / Noise & Monotone Target Warps (`0.0` Query Leakage):**
   - `70%` unbiased uniform random column subsetting, 3-mode `FID_SLOTS` permutation, `50%` sign flips, `exp(U(-4, 4))` scale, signed-power `[0.35, 2.5]`, `K`-level (`[3..32]`) context-quantile discretization, `2%` Cauchy spikes, heteroscedastic + rank-1 correlated context-IQR noise, `exp(+-0.45)` post-norm affine jitter, and monotone target warps on `45%` of non-KD episodes.

## Stage 1 In-Corpus `dev91` Evaluation (`fleet_M_x2_h455_maxent` Valleys 001 & 002 + Cross-Scale Soups)

Evaluated via memory-capped pure-CPU distributed MKL + `openblas64` scoring (`score_h455_distributed_cpu.py`, `steal_distributed_tails_cpu.py`, `CUDA_VISIBLE_DEVICES=""`, `0` GPU/driver touch, `0` trainer interruptions, `in_corpus_dev91_disclosure = true`, `canonical_promotion_forbidden = true`; full receipt in `stage1_m_valley_progression_summary.json`):

| Stage 1 Wave (`in-corpus dev91`) | `pure_104m_cv7` | `neural_104m_all` | `hyb_gpresid_cv7` | `2-Tower ugp_2xcv7` | `stack_ugp_2xcv7` | `stack_ugp_cons_post` | `stack_ugp_cons_joint` (`u`) | Best Arm (`stack_no_ugp_post`) | 95% Bootstrap CI | $\Delta$ vs `1859.63` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Prior Wave-2 Cooled Soup (`H455`) | `1681.21` | `1829.48` | `1821.35` | `1819.04` | `1853.18` | `1858.64` | `1858.88` (`1.7223`) | `1859.08` (`W2A+` `1859.63`) | `[1728.31, 2042.16]` | — |
| `m_x2_h455_valley001_soup50` | `1684.16` | `1835.25` | `1832.46` | **`1829.62`** | `1861.03` | `1861.45` | `1861.55` (`1.7234`) | **`1864.96`** | `[1731.99, 2051.18]` | **`+5.33`** |
| `m_x2_h455_valley002_soup50` | **`1686.14`** | `1833.55` | **`1834.51`** | `1825.43` | `1859.46` | `1860.22` | `1861.52` (`1.7242`) | **`1864.14`** | `[1731.32, 2052.88]` | **`+4.51`** |
| **`m_x2_h455_valley001_002_avg`** | `1684.18` | `1833.52` | `1833.75` | `1827.74` | **`1861.42`** | **`1861.84`** | **`1861.92`** (`1.7239`) | **`1865.36`** | `[1732.68, 2052.12]` | **`+5.73`** |
| `lw2_cooled_plus_m_h455_v2_80_20` | `1681.65` | `1831.16` | `1824.30` | `1824.83` | `1855.65` | `1859.90` | `1859.26` (`1.7228`) | **`1860.76`** | `[1729.28, 2046.53]` | **`+1.13`** |
| `lw2_cooled_plus_m_h455_v2_65_35` | `1681.34` | **`1836.72`** | `1827.07` | `1828.32` | `1856.22` | `1860.10` | `1860.20` (`1.7238`) | **`1861.34`** | `[1730.43, 2047.49]` | **`+1.71`** |

All 5/5 Stage 1 matrices exceed the prior `1859.63` Wave-2A+ Cooled Soup best. Stage 2 (`fleet_L_x1_h455_maxent_v2` `ckpt_valley_001.pt` + cross-scale `L+M` soups) is running on `xbabe0` (`24` MKL shards inside `MemoryHigh=48G/MemoryMax=58G`) + `xbabe2` (`2` persistent reverse-depth tail-stealers inside `MemoryHigh=10G/MemoryMax=14G`).

## Verification
```bash
CUDA_VISIBLE_DEVICES="" python3 research/flywheel_selfplay/h456_ultra_entropy_recipe/selftest_h456.py
```
