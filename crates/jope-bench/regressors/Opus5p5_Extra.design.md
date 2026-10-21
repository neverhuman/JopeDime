# CBRS: Cross-fitted Boosted-Ridge Stack — design rationale

Section 13.3 of the General Regressor Challenge submission whose source is
`Opus5p5_Extra.rs` (section 13.1) and whose manifest is `Opus5p5_Extra.submission.json`
(section 13.2). This rationale was written after the source was frozen and before it was
compiled or executed.

## 1. The learning procedure

Training table `T = [X | y]` with `C` columns; `P = C - 1` features; `y` is always column `C-1`.

**Target scale.** With `ymin`, `ymax` the training range, `u = (y - c)/h` with
`c = (ymin+ymax)/2`, `h = (ymax-ymin)/2`, so `u ∈ [-1, 1]`; then `s = (u - μ_u)/σ_u`. Every
model is fitted to `s`. A prediction `ŝ` is mapped back through `u = μ_u + σ_u ŝ`, clipped to
`[-1.2, 1.2]`, and returned as `c + h u`. Outputs therefore stay within 10% of the training
range beyond either end, and are finite whenever the inputs are finite.

**Working rows.** If `n > 400,000`, a seeded random subset of 400,000 rows (kept in original
order) is used for all fitting; otherwise all rows.

**Feature statistics (unsupervised).** From up to `S = clamp(10⁸/P, 2,000, 200,000)` evenly
spaced working rows, each column gets:

- *Tree bins:* if it has at most 255 distinct observed values, one bin per value with
  thresholds at midpoints; otherwise up to 255 equal-frequency bins. Thresholds are stored as
  raw-unit cut points `x ≤ t`. Missing values get a dedicated extra bin.
- *Linear transform:* clip to the 0.1% and 99.9% sample quantiles (falling back to min/max), map
  that interval to `[-1, 1]`, then standardize by the clipped sample mean and standard
  deviation. Missing becomes 0 (the clipped mean). A column with missing entries also
  contributes a standardized missing indicator.

Columns with no observed value, or one distinct value and no missing entries, are excluded from
trees and from the linear model.

**Cross-fitting.** Working rows are split into `K` target-stratified folds (`K = 5` up to
20,000 rows, 4 up to 60,000, 3 up to 150,000, else 2): rows are sorted by `s` and each
consecutive block of `K` receives a random permutation of fold labels. For each fold `k`, three
predictors are fitted on the other folds' rows only:

1. **Ridge.** On the linear candidates, the fold computes a centered Gram matrix and solves
   `(A + λ·tr(A)/d · I) β = b` by Cholesky for 14 relative penalties
   `λ ∈ {10⁻⁶, …, 100}`. When there are more than `min(300, budget cap)` candidates, the fold
   keeps the top candidates by squared marginal correlation with `s`, computed on that fold's
   training rows only. One `λ` is selected for all folds by the pooled out-of-fold squared
   error over every fold's held-out rows.
2. **Leaf-wise gradient boosting (LW).** Squared-loss boosting from the fold-training mean.
   Each tree grows best-first to at most 31 leaves and depth 12, with leaf value
   `η·ΣR/(n_leaf + 1)` and split gain `G_L²/(n_L+1) + G_R²/(n_R+1) − G²/(n+1)`. Histograms use
   the subtraction trick (build the smaller child, subtract from the parent). Missing values
   are tried on both sides of every threshold; when a node saw none, unseen missing goes to the
   larger child. Each round samples 80% of the fold-training rows and 80% of the tree features
   (at most 512). The minimum child size is `clamp(round(n_fit/40), 2, 20)`.
3. **Ridge-initialized oblivious boosting (OB).** Squared-loss boosting starting from the
   fold's ridge predictions, so it models the ridge residual. Each tree is symmetric: one
   (feature, threshold, missing-direction) per level, chosen by the summed gain over every node
   at that level. Its depth is up to 6, limited so each leaf has about four rows on average,
   with `λ = 3`. It uses all fold-training rows and up to 256 features per tree.

Both boosting families stop early on the fold's held-out rows, using patience
`clamp(round(5/η), 20, 150)`, and are truncated to their best iteration. The learning rate
follows the planned number of rounds `R`: `η = clamp(0.05·√(1000/R), 0.05, 0.3)`.

**Blend.** Out-of-fold predictions from LW, OB, ridge and the fold-training mean form four
columns. Simplex weights (nonnegative, summing to 1) are chosen to minimize out-of-fold squared
error. The search is exact over the lattice with step 1/20, using the 4×4 Gram matrix of
those columns (1,771 candidate points).

**Prediction.** For each component, average its `K` fold models (cross-fit bagging), combine
with the blend weights, and map back to target units. A component whose weight is 0 is
discarded at fit time.

## 2. Why it should be broadly competitive

Without data, the safest prior is a portfolio covering the dominant regimes of tabular
regression, with selection performed on held-out rows inside each task.

- Histogram gradient-boosted trees are invariant to monotone feature transforms, need no
  scaling, handle missing values natively, and capture interactions and thresholds. That makes
  them the most reliable single family on heterogeneous tabular data.
- Many targets are dominated by a smooth, nearly additive linear trend. Trees approximate that
  with staircases and extrapolate badly, which is where ridge is strong. Initializing the
  oblivious booster from ridge gives a "linear trend + nonlinear correction" model. Oblivious
  trees are strongly regularized (one split per level), which suits small and noisy data.
- Leaf-wise trees are the high-capacity complement for larger data with sharp interactions.
- The mean component and simplex constraint let the blend fall back to shrinkage when nothing
  generalizes. The blend cannot do worse out of fold than its best vertex, except for the
  lattice resolution.
- Averaging the `K` fold models reduces variance at no extra fitting cost. Each held-out row
  receives a prediction from a model that never saw it, so selection uses honest residuals.

## 3. What is learned and what is fixed

**Learned from the current training call:** target range, mean and scale; working-row subset;
bin thresholds; clip bounds; per-column means and scales; missing rates; the tree-feature
screen (only on very wide or tall data); fold assignment; per-fold ridge screen, coefficients
and intercept; the global ridge penalty; every tree's structure, thresholds, missing directions
and leaf values; the number of trees per fold per family (early stopping); and blend weights.

**Fixed in source:** the algorithm structure; 255 bins; caps on rows, features and work; fold
counts by size; tree shapes (31 leaves, depth 12; oblivious depth 6); regularization constants
(λ = 1 and 3); subsampling fractions; the ridge penalty grid; learning-rate and patience
schedules; blend lattice step; and the 10%-of-range output margin. These are general numerical
choices, not values tuned to any dataset.

## 4. Governing objective

Every fitted parameter minimizes squared error on standardized targets on that component's
training rows, with explicit regularization: ridge penalty, the λ terms in leaf values,
shrinkage `η`, row and feature subsampling, minimum child size, and depth or leaf limits.
Every internal decision (ridge `λ`, number of boosting rounds, blend weights) minimizes
squared error on held-out fold rows. That is the same quantity as the competition's RMSE, up
to the affine target map. The work allocation (rounds, subsample sizes) follows deterministic
cost formulas from the data shape and is never chosen by observed error.

## 5. Missing values, degeneracies, stability and units

- **Missing features.** Trees route missing values with a learned per-split direction (larger
  child if unseen). Ridge imputes the clipped mean and, for columns with observed missingness,
  adds an indicator. Fully missing columns are ignored.
- **Zero features (`C = 1`), one row, or constant target.** The model is a constant: the
  training mean, or exactly `ymin` for a constant target. Prediction with `P = 0` returns
  `n_rows` copies.
- **Constant or duplicate columns** are excluded or harmless: duplicate trees tie
  deterministically, and ridge is regularized.
- **Tiny `n`.** `K ≤ n`, and boosting is skipped when a fold has fewer than two training rows.
- **Stability.** The ridge penalty is relative to `tr(A)/d`, so the solve is scale-free and
  strictly positive-definite. A failed factorization falls back to zero coefficients
  (intercept only). No division can reach zero: counts plus λ, guarded scales and checked sums.
  The output map replaces non-finite values and clips to range.
- **Units.** Feature units do not matter to trees (ranks) or to ridge (robust standardization).
  Target units are removed by the affine map and restored exactly.
- **Validation.** `fit` checks `n_rows ≥ 1`, `n_cols ≥ 1`, length with checked multiplication,
  finite targets and no infinite features. `predict` checks fitted state, width, checked
  length and no infinite features. Violations return `Err`.

## 6. Cost

Notation: `n` training rows, `P` features, `n_w ≤ 400k` working rows, `p_t ≤ 2,000` tree
features, `B ≤ 256` bins, `K ≤ 5` folds, `R ≤ 4,000` rounds, `d ≤ 300` ridge columns.

- **Validation and target map:** `O(nP)`.
- **Statistics:** `O(P·S log S)` for sorting the sample.
- **Binning:** `O(n_w p_t log B)`, using `n_w·p_t ≤ 6·10⁸` bytes.
- **Ridge:** a screen `O(n_w·2P)` when needed, Gram matrices `O(K n_w d²)`, bounded by a fixed
  budget of `4·10⁹` multiply-adds, and solves `O(14 K d³)`.
- **Boosting:** each round costs histogram construction over sampled rows and features,
  split scans over bins, and updates for out-of-bag and held-out rows. A deterministic cost
  model splits a fixed budget of `4·10¹⁰` work units equally over 2 families × `K` folds and
  derives rounds, row and feature subsample sizes from it. Prediction cost caps rounds at
  about `5·10⁹ / n` node visits per row in total.
- **Blend:** `O(n_w)` plus 1,771 evaluations of a 4×4 quadratic form.
- **Prediction:** `O(m(P + K(T·D + d)))`.
- **Memory:** the input; one byte per binned cell; `O(n_w)` work vectors; pooled histograms;
  and stored trees, `O(K R L)`.

The budget constants are design targets. Their wall-clock behavior under the 600 s and 60 s
limits has not been measured.

## 7. Determinism, isolation, target handling

All randomness comes from a SplitMix64 generator seeded by `derive_seed(seed, tag)` with a
fixed tag per use (row subset, folds, each fold and family). No hash maps, clocks or global
state are used. Sorts use total ordering with index tie-breaks where order matters. `fit`
clears any previous model first, rebuilds everything from the stored constructor seed and the
current table, and leaves the instance unfitted on error. `predict` takes `&self`, mutates
nothing, and computes each row with the same arithmetic order regardless of batch size or
position (block processing only interleaves rows). Feature indices run over `0..P` only. The
target column is read only into `y`, never into features, bins or summaries. Prediction
receives exactly `P` columns and never drops one.

## 8. Weaknesses

- **Squared error only.** There is no robust loss or target transform (log, rank) for
  heavy-tailed or skewed targets, so a few extreme labels can dominate. The clipping margin
  prevents extrapolation more than 10% beyond the training range, which hurts genuinely
  trending targets.
- **Optimistic selection.** Out-of-fold estimates used for blending come from the same rows
  that chose each fold's early-stopping point and the ridge `λ`. They are slightly optimistic,
  most for the boosted components.
- **Label leakage in tree screening.** When tree-feature screening is active (more than 2,000
  eligible features, or `n_w·p_t > 6·10⁸`), the adjusted correlation-ratio screen uses
  labels from all working rows, including each fold's held-out rows. In that regime, the
  held-out estimates for the tree components are not fully clean. Screening is marginal, so
  features useful only in interactions can be dropped.
- **Mismatch between selection and prediction.** Blend weights are chosen for single-fold
  models but applied to `K`-model averages, which have lower variance than what was evaluated.
- **Data-type limits.** Categorical codes are treated as ordered numbers. Codes with more than
  255 levels are merged into quantile bins. There are no kernel, nearest-neighbor or smooth
  basis components for low-noise smooth functions in many dimensions.
- **Unmeasured resource behavior.** The work budget and the prediction cap, which assumes a
  held-out set comparable in size to the training set, are untested. A held-out set much
  larger than the training set, or extreme shapes, could exceed the time limits.

## 9. Compilation and experiments

Authoring mode was **reasoning-only**. The source (13.1) and manifest (13.2) were frozen
without compiling, running or testing any part of them. No dataset, synthetic data,
benchmark result, external code or other model was used to design, select or tune the
algorithm. No data experiments were run. Any later compilation or contract checking is a
post-freeze organizer action. It is recorded separately and did not feed back into this
design.
