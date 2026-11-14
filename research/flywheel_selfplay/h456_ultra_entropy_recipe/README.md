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

## Verification
```bash
CUDA_VISIBLE_DEVICES="" python3 research/flywheel_selfplay/h456_ultra_entropy_recipe/selftest_h456.py
```
