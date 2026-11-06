# Ridge Additive Boost

## 1. Learning procedure

The loss is squared error in the original target units. For a training table with `C` columns, the target is column `C - 1` and `P = C - 1` is the feature count. With training mean `mu` and population standard deviation `s`, the standardized target is `y' = (y - mu) / s`. The predictor is

```text
y_hat = mu + s * (b + z·beta + sum_j u[j, bin_j(x)] + sum_t Tree_t(x))
```

`mu` and `s` use every training target. If `s` is below `1e-12 * max(1, |mu|)`, or if `P = 0`, the model is the constant `mu`. That constant is the squared-error minimizer among constant functions.

Otherwise a deterministic subset `L` of at most 32,768 rows is drawn from the seed. Screening rows `S` are a subset of `L` whose size is `min(|L|, max(400, 40_000_000 / P))`. On `S`, each feature is ranked by the squared correlation of `asinh(x)` with `asinh(y)`, and by the squared correlation of the centered square of `asinh(x)` with `asinh(y)`. The `asinh` map is used only to keep the ranking sums finite. A column is kept when it has at least 8 observed screen rows and `n_obs * r^2` exceeds 6 for the linear term or 8 for the quadratic term. At most 24 columns are kept, and fewer when `|L|` is small: at most 2 below 16 rows, at most 6 below 64, and never more than `|L| / 4`. Ties prefer the smaller column index.

A separate missingness indicator is kept for up to 8 columns whose missing rate on `S` lies in `[0.02, 0.98]`, with at least 8 missing and 8 observed rows, using the same factor-6 gate.

On `L`, each kept column is median-imputed and scaled by `1.4826 * MAD`, or by its standard deviation when the MAD is below `1e-12`. Scaled values are clipped to `[-6, 6]` and centered. Coordinate descent, at most 16 epochs, minimizes squared error plus a ridge penalty that multiplies an orthogonal coefficient by `|L| / (|L| + 4 M_r)`. If at least two numeric columns exist and `|L|` is at least 25 times the expanded width, the pairwise products of the top four centered columns (at most six), themselves centered, enter the same system.

Two Gauss–Seidel passes then fit a shrunk univariate bin table on the residual, strongest column first. Finite bins are `clamp(|L| / 20, 2, 16)`. Few distinct values get a midpoint cut between adjacent values; otherwise the cuts are empirical quantiles. Bin 0 is missing. A finite value lies in the left bin when `x <= edge`. The bin value is `sum / (count + lambda)` with `lambda = max(1, 40 K / |L|)`, minus the count-weighted mean of those values.

If `|L| >= 48` and a column can still be split, gradient-boosted trees fit the remaining residual. The schedule is depth 2 and 12 trees below 300 rows (learning rate 0.12), depth 3 and 24 trees below 3,000 (rate 0.08), and depth 4 and 36 trees otherwise (rate 0.07). Each tree uses a fresh generator mixed from the instance seed and the tree index, a 75 percent row subset once `|L| >= 200`, and about three quarters of the splittable columns. Split gain is the regularized reduction `s_L^2/(c_L+lambda) + s_R^2/(c_R+lambda) - s^2/(c+lambda)`. The minimum leaf size is `clamp(|subset| / 64, 1, 200)`, and at most half the current node. Missing values try both sides. A split must beat `1e-6` times the node count. A root with no legal split is discarded. Leaf contributions are stored after multiplication by the learning rate. The final `b` is the mean residual on `L`.

## 2. Why this should travel

Squared error matches RMSE. The mean is the right constant. Robust scaling and quantile cuts do not require features to share units. The three additive pieces cover the structures that show up across unrelated tables: a global linear effect, a nonlinear main effect, and a low-order interaction. Shrinkage, the feature cap, and the row cap grow with the sample and stay small when it is tiny. A contract-valid table still produces a finite prediction, because a finished mean model beats a timeout or an explicit failure under the stated comparison rule.

## 3. What is learned, what is fixed

Learned from the current table only: `mu`, `s`, the screened column set, medians, scales, centers, ridge coefficients, bin edges and bin values, tree structure and leaf values, and `b`.

Fixed in the source: the gates, caps, clip width, ridge multiplier, epoch limit, bin rule, tree schedule, learning rates, and the seed-to-subset map. Nothing is chosen by looking at another dataset or at a held-out test label. There is no internal validation search. The schedule is a function of `|L|` and of the screening scores.

## 4. Objective

Every fitted number is a squared-error step on `y'`. Ridge is penalized coordinate descent. Bin values and leaves are shrunk means, which is the closed form for squared error plus a quadratic penalty on the bin or leaf. The constant `mu` minimizes training squared error among constants. The final bias removes the average residual left on the fitting rows.

## 5. Degenerate inputs and units

NaN features are missing. Infinite features and nonfinite targets are errors. A missing entry uses the training median in the linear part and bin 0 in the tables and trees. Trees may send missing values left or right; the side is the one with larger regularized gain. The sentinels `+infinity` and `-infinity` appear only as internal split thresholds that separate missing from finite values. They are not predictions.

`P = 0`, a constant target, a single row, and a screen that keeps nothing all predict `mu`. Constant finite columns are dropped. A column that is constant except for missingness can still contribute a missing bin and, if it passes the gate, a linear indicator. Duplicate rows need no special case. Prediction rows are not used to rescale or to choose a split. If a component is nonfinite, that contribution is replaced by zero; if the final prediction is nonfinite, it is replaced by `mu`.

Target units are restored by `mu + s * F`. Feature units are absorbed by the MAD or standard-deviation scale and, for the trees and bins, by quantile cuts. Labels are not winsorized, because the scoring loss is squared error.

## 6. Cost

Let `n` be training rows, `C` the training width, `P = C - 1`, `m` the prediction rows, `n_l = min(n, 32768)`, `n_s` the screen-row count, `M <= 24`, `M_r <= 38`, `E <= 16`, `K <= 16`, `T <= 36`, and `D <= 4`.

Fit cost is a `Theta(n C)` validation scan, plus `O(n_s P)` screening arithmetic, plus `O(n_l M_r E + n_l M K + T D n_l M)`. Predict cost is `O(m (M_r + M log K + T D))`. Peak extra memory is an `O(P)` moment table and an `O(n_l M_r)` working matrix. The stored model is `O(M K + T 2^D)`. The moment table and the working matrix are dropped when `fit` returns.

The row, feature, and tree caps are the timeout policy. The full scan can itself exceed 600 seconds on an extreme shape, because every entry must be checked for an infinite feature. That limit is shared by any implementation that honors the input contract.

## 7. Determinism, isolation, and the column contract

`new(seed)` stores the seed and leaves the model unfitted. `fit` clears any previous model first and draws subsets from that original seed, not from a stream left by an earlier fit. A failed fit leaves the instance unfitted. `predict` takes `&self` and does not resample. The same source, seed, training rows, and runtime repeat the same decisions. Subsets use SplitMix64 followed by xorshift64*, with rejection sampling for the index range. No clock, hash map, or process identifier is used.

Feature formulas never include the target. At prediction the row has exactly the fitted `P` columns. A different width, a bad length, an infinite feature, or a call before a successful fit returns `Err(String)`. A valid empty prediction batch returns an empty vector. Each output depends only on that row and the fitted state.

## 8. Weaknesses

A feature with no linear or quadratic marginal signal is never offered to the trees, so a pure interaction can be missed. At most 24 columns are modeled, so a wide table of many weak signals is truncated. Depth 4 cannot represent a high-order interaction. Numeric order is a poor code for an unordered category with more than 16 levels. Patterns that occur only outside the 32,768-row subset are invisible to the coefficients and the trees, though the mean still uses every target. The fixed schedule does not stop early on a quiet residual, and it does not add trees when the residual is still structured. Collinear copies share one shrunk coefficient rather than being removed. Very smooth low-noise curves are approximated by bins and shallow trees, not by a dedicated smoother.

## 9. Authoring

No compilation was performed. The regressor was not constructed, fitted, or called. No dataset, synthetic matrix, benchmark table, or other entrant source was read. A directory listing showed that other submission filenames exist in the workspace; their contents were not opened and were not used. `compiled_during_design` is false, `experiments_run_during_design` is false, and `authoring_mode` is `reasoning_only`.
