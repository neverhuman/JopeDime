### 13.3 `design.md`

#### 1. Mathematical Learning Procedure



**OmniStack-6** is a sovereign, multi-paradigm tabular regression system engineered from first principles to maximize comparative Elo across thousands of unseen datasets. Given a training matrix $T_{\text{train}} = [X \mid y] \in \mathbb{R}^{n \times (P+1)}$, the algorithm isolates the first $P$ feature columns from the final target column $y$, evaluates adaptive target distribution transformations, builds dual feature representations with out-of-fold Bayesian categorical target encoding, and trains six distinct model families across target-stratified $F$-fold cross-validation.

1. **Adaptive Target Distribution Normalization & Smearing Correction (`TargetTransform`)**:
Tabular targets frequently exhibit severe positive skewness (e.g., prices, counts, reaction rates) or heavy-tailed leptokurtic distributions. Fitting squared-error learners directly on raw skewed targets causes extreme values to dominate gradient steps. OmniStack-6 computes empirical skewness $\gamma_1$ and kurtosis $\gamma_2$ of $y$ and constructs a candidate transformation:
* **Shifted Logarithmic**: $\tilde{y} = \ln(y + s)$ when $y_{\min} \ge 0$ and $\gamma_1 > 1.75$.
* **Inverse Hyperbolic Sine**: $\tilde{y} = \operatorname{asinh}((y - \operatorname{med}(y)) / \sigma_{\text{IQR}})$ when $\gamma_2 > 9.0$ or $\vert{}\gamma_1\vert{} > 2.2$.
* **Standardized Identity**: $\tilde{y} = (y - \mu_y) / \sigma_y$ otherwise.
To guarantee that a nonlinear target transformation never degrades original-unit RMSE, an internal 80/20 stratified holdout probe compares original-unit RMSE of the nonlinear transform (inverted with Duan's smearing correction $\hat{S} = \frac{1}{n_{\text{tr}}} \sum \exp(\hat{r}_i)$ to eliminate Jensen's inequality bias) against the identity transform, adopting the nonlinear transform only if held-out original-unit MSE improves by at least $3\%$.


2. **Multi-Basis Feature Engineering & Out-of-Fold Bayesian Target Encoding (`Preprocessor` & `FoldTargetEncoder`)**:
* **Exact Normal Quantile Transformation**: Maps each feature value $x_{ij}$ through its empirical cumulative distribution function $\hat{F}_j(x_{ij})$ on 127 quantiles, followed by a high-precision rational inverse Gaussian CDF $\Phi^{-1}(\hat{F}_j(x))$, rendering parametric and distance learners invariant to arbitrary monotonic feature distortions.
* **Natural Piecewise-Linear Splines**: For tables with $P \le 96$, expands each standardized feature $z_j$ with 3 hinge knots $\max(0, z_j - \kappa_{jk})$ at the 25th, 50th, and 75th percentiles.
* **Out-of-Fold Empirical Bayes Target Encoding**: Automatically detects discrete/low-cardinality numeric columns ($3 \le U_j \le 256$ unique values with $3 U_j \le n$) and computes both frequency encoding and fold-isolated Empirical Bayes target statistics:

$$\text{TE}_j(v) = \frac{\sum_{i \in \mathcal{I}_{\text{tr}}, x_{ij}=v} \tilde{y}_i + m \bar{y}_{\text{tr}}}{n_{jv} + m} - \bar{y}_{\text{tr}}, \qquad m = 12.0$$



strictly inside each fold's training set to prevent target leakage.
* **Interaction & Row Summary Synthesis**: Augments features with 8 row-level distributional statistics (sum, mean, $L_1$, $L_2$, min, max, std, missing rate) and top bivariate interactions including products $z_a z_b$, differences $z_a - z_b$, and damped ratios $z_a / (\vert{}z_b\vert{} + 1)$.




3. **Six-Family Cross-Fitted Base Learners**:
Across each stratified fold $f \in \{1, \dots, F\}$:
* **`ElasticNetModel` ($L_1 + L_2$ Penalized Spline/Basis Regressor)**: Evaluates both exact Cholesky Ridge solutions across a 8-point logarithmic $\lambda$ grid and proximal coordinate descent with soft-thresholding $S_{\lambda_1}(\rho) / (\Vert{}x_j\Vert{}_2^2 + \lambda_2)$ to recover sparse linear and additive spline laws.
* **`ObliviousGBDTModel` (CatBoost-Style Symmetric Decision Trees)**: Grows depth-6 oblivious trees where all $2^d$ nodes at depth $d$ share the same split feature and threshold, evaluated on 127-bin histograms. Oblivious trees act as regularized global decision tables that resist overfitting on noisy tabular benchmarks.
* **`gbdt_shallow` (Depth-4 Standard $L_2$ Histogram GBDT)**: Captures low-order localized interactions with stochastic row bagging ($85\%$) and column subsampling ($80\%$).
* **`gbdt_deep_huber` (Depth-7 Smooth $C^2$ Pseudo-Huber Histogram GBDT)**: Uses exact first and second derivatives of the Pseudo-Huber loss $L_\delta(r) = \delta^2 (\sqrt{1 + (r/\delta)^2} - 1)$:

$$g_i = \frac{r_i}{\sqrt{1 + (r_i/\delta)^2}}, \qquad h_i = \left(1 + \left(\frac{r_i}{\delta}\right)^2\right)^{-3/2}$$



allowing deep interaction splits while automatically down-weighting high-residual outliers in the Newton leaf step.
* **`ResMLPModel` (Residual Skip-Connection SiLU Neural Network)**: A neural network combining a direct linear skip pathway $w_{\text{skip}}^\top x$ with a SiLU hidden layer $W_2 \sigma_{\text{SiLU}}(W_1 x + b_1)$, trained with cosine-annealed AdamW and validation checkpointing.
* **`KNNModel` (Metric-Adaptive $L_1 / L_2$ Relevance-Weighted Kernel Regressor)**: Selects both distance metric ($L_1$ Manhattan vs. $L_2$ Euclidean) and kernel bandwidth $h^*$ on the fold validation set using squared-correlation feature weights.


4. **Simplex Stacking & Shrinkage Calibration**:
Out-of-fold predictions $\hat{Y}_{\text{OOF}} \in \mathbb{R}^{n \times 6}$ are combined via projected gradient descent onto the 6-simplex $\Delta^5$, followed by regularized affine calibration $\hat{z}_{\text{cal}} = \beta_1 (\hat{Y}_{\text{OOF}} \alpha) + \beta_0$.

---

#### 2. Why This Design Is Broadly Competitive Across Unknown Datasets



Winning head-to-head Elo across thousands of diverse datasets requires dominating both classic GBDT territory and regimes where standard decision trees fail:

* **Symmetric vs. Asymmetric Tree Topologies**: Pairing CatBoost-style oblivious trees (`ObliviousGBDTModel`) with standard depth-4 and depth-7 trees (`GBDTModel`) captures both global conditional tables and localized axis-aligned pockets.
* **Continuous, Additive, and Physical Laws**: `ElasticNetModel` (equipped with normal quantile scores, $\operatorname{asinh}$, piecewise-linear splines, and damped ratios) and `ResMLPModel` (with explicit linear skip connections) achieve near-zero approximation error on smooth manifolds where step-function trees lose Elo.
* **High-Cardinality Categoricals & Discrete Identifiers**: Out-of-fold Bayesian target encoding plus frequency encoding allows both linear/neural and tree learners to extract signal from encoded categorical columns immediately.
* **Skewed Targets & Outliers**: Adaptive target transformation with Duan's smearing correction plus $C^2$ Pseudo-Huber Newton boosting prevents catastrophic RMSE inflation on skewed or noisy targets.

---

#### 3. What Is Learned from Training Data vs. What Remains Fixed



* Learned exclusively from `train` inside `fit`

: Target transform selection and smearing factor; feature screening scores; median, IQR scale, 127 empirical quantiles, spline knots, and frequency maps; fold-isolated Bayesian target encoding maps; row-summary and pairwise interaction moments; ElasticNet regularization $(\lambda_1, \lambda_2)$ and weights; Oblivious and standard GBDT tree structures, split thresholds, NaN routing directions, leaf values, and early-stopped tree counts; ResMLP weights at best validation epoch; KNN metric ($L_1$ vs. $L_2$) and bandwidth; 6-simplex ensemble weights and post-stacking calibration $(\beta_1, \beta_0)$.
* Fixed in source code: Hyperparameter search grids, maximum feature cap ($1,024$), histogram bin resolution ($127$), tree depths ($4, 6, 7$), and resource-adaptive epoch/tree caps.



---

#### 4. Governing Objectives and Internal Decision Principles



Every internal decision minimizes held-out squared error:

* Target transformation is selected by held-out original-unit MSE on an internal stratified split.
* ElasticNet penalties, GBDT stopping rounds, ResMLP epochs, and KNN metric/bandwidth are chosen by minimizing MSE on each fold's validation partition.
* Ensemble weights solve $\min_{\alpha \in \Delta^5} \Vert{}\hat{Y}_{\text{OOF}} \alpha - \tilde{y}\Vert{}_2^2$ on out-of-fold predictions.

---

#### 5. Handling Missing Values, Degenerate Inputs, Zero Features, Numerical Stability, and Varying Units



* **Missing Values (`f64::NAN`)**: Imputed to median ($z=0, q=0$) with standardized missingness indicators for parametric/neural/kernel models; routed natively via learned `nan_go_left` booleans (`bin = 255`) in both Oblivious and standard GBDTs; included as a dedicated category key (`u64::MAX`) in Bayesian target encoding.
* **Degenerate Inputs & Zero Features**: $P=0$, $n=1$, constant targets, or all-NaN columns are handled cleanly without out-of-bounds indexing or division by zero; $m=0$ prediction calls return `Ok( vec![] )` after validating fitted state and `n_features`.
* **Numerical Stability & Units**: Normal quantile scores, IQR scaling, Cholesky positivity guards ($>10^{-13}$), clamped exponentials, and target range clamping $[\min(y)-3S_y, \max(y)+3S_y]$ guarantee finite predictions across arbitrary scales.



---

#### 6. Computational and Memory Complexity



With training rows $n$, features $P$, prediction rows $m$, expanded linear dimension $d_{\text{lin}} \le 5,200$, and tree dimension $d_{\text{tree}} \le 1,140$:

* **`fit` Time**: $\mathcal{O}(n P \log n + F (n d_{\text{lin}} + \min(d_{\text{lin}}^3, n d_{\text{lin}}) + T 2^D n d_{\text{tree}} + E n H d_{\text{lin}} + n_{\text{val}} n_{\text{ref}} d_{\text{lin}}))$, completing well within the 600s single-core limit.


* **`predict` Time**: $\mathcal{O}(m (P + F (d_{\text{lin}} + T D + H d_{\text{lin}} + n_{\text{ref}} d_{\text{lin}})))$.
* **Memory**: $\mathcal{O}(n (P + d_{\text{lin}} + d_{\text{tree}}) + m P)$, strictly bounded well below 8 GiB.



---

#### 7. Determinism, State Isolation, Target Handling, and Batch-Independent Inference



* **Determinism & Isolation**: All randomness derives from `XorShift64` initialized with `self.seed`; `BTreeMap` is used for deterministic iteration without OS-seeded hashing. `fit` clears `self.state = None` immediately on entry.


* **Target Contract**: Column `n_cols - 1` is strictly isolated as the target in `fit`; `predict` receives `P = n_cols - 1` features only and predicts each row independently without batch coupling.



---

#### 8. Important Weaknesses and Expected Failure Modes



1. **Ultra-Wide Combinatorial Parity ($P \gg 10,000$)**: Univariate screening to 1,024 features may miss pure high-order parity interactions with zero marginal correlation.
2. **Extreme Unbounded Extrapolation**: Predictions are safely clamped to $[\min(y) - 3 S_y, \max(y) + 3 S_y]$, which bounds catastrophic errors but caps extreme out-of-distribution trend extrapolation.

---

#### 9. Compilation and Zero-Experimentation Confirmation



Authored strictly in **`reasoning_only`** mode: zero compilations (`compiled_during_design: false`), zero executions, and zero data experiments were performed.

---
