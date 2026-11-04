use super::*;

/// Stable per-kind key: also the base of the emitted function names.
pub fn op_fn_name(op: &PlannedOp) -> &'static str {
    match op {
        // B05–B07: scalar-v2 named registration.
        PlannedOp::ScalarV2 { .. } => "scalar_expression_v2",
        // B13: a goertzel_bank stage keys its own helper; B12 keys stay unchanged.
        PlannedOp::RowVectorV2 {
            state: crate::kernel::row_vector::FittedStage::Goertzel(_),
        } => "row_vector_goertzel_v2",
        // B12: version-two FeatureWithinRow stage.
        PlannedOp::RowVectorV2 { .. } | PlannedOp::RowVectorInputV2 { .. } => "row_vector_v2",
        PlannedOp::Standardize => "standard_scaler",
        PlannedOp::RobustScale => "robust_scaler",
        PlannedOp::Rank => "mid_rank_transform",
        PlannedOp::Blom => "blom_normal_scores",
        PlannedOp::QuantileBins { .. } => "quantile_bin_indexer",
        PlannedOp::BinOnehot { .. } => "quantile_onehot_encoder",
        PlannedOp::Hinge { .. } => "hinge_basis",
        PlannedOp::WinsorClip { .. } => "winsor_limiter",
        PlannedOp::Cheb { .. } => "chebyshev_basis",
        PlannedOp::PolyInteract => "pairwise_products",
        PlannedOp::RffCos { .. } => "random_fourier_cosine_features",
        PlannedOp::RffSin { .. } => "random_fourier_sine_features",
        PlannedOp::RandProj { .. } => "gaussian_random_projection",
        PlannedOp::LandmarkDist { .. } => "gaussian_landmark_features",
        PlannedOp::ObliqueHinge { .. } => "oblique_hinge_features",
        PlannedOp::RowStats => "row_summary_statistics",
        PlannedOp::PairProd { .. } => "correlation_screened_products",
        PlannedOp::TargetBinEncode { .. } => "smoothed_target_bin_encoder",
        PlannedOp::Sigmoid => "logistic_squash",
        PlannedOp::Step { .. } => "step_indicator",
        PlannedOp::NormRow => "unit_norm_rows",
        PlannedOp::Polar => "polar_coordinates",
        PlannedOp::Lag { .. } => "shift_columns",
        PlannedOp::Roll { .. } => "rotate_columns",
        PlannedOp::FftMag { .. } => "fourier_magnitudes",
        PlannedOp::RowSpectral => "row_spectral_summary",
        PlannedOp::RowDiffStats => "row_difference_statistics",
        PlannedOp::SvdProject { .. } => "principal_component_projection",
        PlannedOp::Nystrom { .. } => "nystrom_landmark_features",
        PlannedOp::MapExpr { .. } => "elementwise_formula",
        PlannedOp::DeriveExpr { .. } => "derived_column",
        PlannedOp::Symlog { .. } => "signed_log",
        PlannedOp::SelectCorr { .. } => "column_select",
        PlannedOp::Impute { .. } => "zero_imputer",
        PlannedOp::PairRatio { .. } => "ratio_screened_pairs",
        PlannedOp::Spline { .. } => "natural_cubic_spline",
    }
}

/// Rust fitted-state struct name (None for stateless transforms).
pub fn op_struct_name(op: &PlannedOp) -> Option<&'static str> {
    match op {
        // B05–B07: scalar-v2 named registration.
        PlannedOp::ScalarV2 { .. } => None,
        // B12: version-two FeatureWithinRow stage.
        PlannedOp::RowVectorV2 { .. } | PlannedOp::RowVectorInputV2 { .. } => None,
        PlannedOp::Standardize => Some("StandardScaler"),
        PlannedOp::RobustScale => Some("RobustScaler"),
        PlannedOp::Rank => Some("MidRankTransform"),
        PlannedOp::Blom => Some("BlomNormalScores"),
        PlannedOp::QuantileBins { .. } => Some("QuantileBinIndexer"),
        PlannedOp::BinOnehot { .. } => Some("QuantileOnehotEncoder"),
        PlannedOp::Hinge { .. } => Some("HingeBasis"),
        PlannedOp::WinsorClip { .. } => Some("WinsorLimiter"),
        PlannedOp::Cheb { .. } => None,
        PlannedOp::PolyInteract => Some("PairwiseProducts"),
        PlannedOp::RffCos { .. } => Some("RandomFourierCosineFeatures"),
        PlannedOp::RffSin { .. } => Some("RandomFourierSineFeatures"),
        PlannedOp::RandProj { .. } => Some("GaussianRandomProjection"),
        PlannedOp::LandmarkDist { .. } => Some("GaussianLandmarkFeatures"),
        PlannedOp::ObliqueHinge { .. } => Some("ObliqueHingeFeatures"),
        PlannedOp::RowStats => None,
        PlannedOp::PairProd { .. } => Some("CorrelationScreenedProducts"),
        PlannedOp::TargetBinEncode { .. } => Some("SmoothedTargetBinEncoder"),
        PlannedOp::Sigmoid
        | PlannedOp::Symlog { .. }
        | PlannedOp::Step { .. }
        | PlannedOp::NormRow
        | PlannedOp::Polar
        | PlannedOp::Lag { .. }
        | PlannedOp::Roll { .. }
        | PlannedOp::RowSpectral
        | PlannedOp::RowDiffStats
        | PlannedOp::MapExpr { .. }
        | PlannedOp::DeriveExpr { .. } => None,
        // Fitted state is the resolved frequency count.
        PlannedOp::FftMag { .. } => Some("usize"),
        PlannedOp::SvdProject { .. } => Some("PrincipalComponentProjection"),
        PlannedOp::Nystrom { .. } => Some("NystromLandmarkFeatures"),
        PlannedOp::SelectCorr { .. } => Some("ColumnSelect"),
        PlannedOp::Impute { .. } => Some("ZeroImputer"),
        PlannedOp::PairRatio { .. } => Some("RatioScreenedPairs"),
        PlannedOp::Spline { .. } => Some("NaturalCubicSpline"),
    }
}

/// One-sentence math description with a literature anchor, used in docs.
pub fn op_doc(op: &PlannedOp) -> String {
    match op {
        // B05–B07: scalar-v2 named registration.
        PlannedOp::ScalarV2 { .. } => "evaluates the bounded scalar-v2 program in row or elementwise scope, retaining its numerical policy and per-row faults.".into(),
        // B13: version-two sparse Fourier power bank.
        PlannedOp::RowVectorV2 {
            state: crate::kernel::row_vector::FittedStage::Goertzel(_),
        } => "computes, for each frozen frequency, the squared magnitude of the row's discrete Fourier sum with a second-order recurrence (Goertzel, 1958) and a sum-of-squares finalization.".into(),
        // B12: version-two FeatureWithinRow stage.
        PlannedOp::RowVectorV2 { .. } | PlannedOp::RowVectorInputV2 { .. } => "applies a width-bound feature-position FIR with frozen coefficient bits; clipped boundary windows reduce polynomial degree.".into(),
        PlannedOp::Standardize => "centers each column by its training mean and divides by the \
            standard deviation (1/n convention, floored at 1e-9), clamping the score to [-20, 20]."
            .to_string(),
        PlannedOp::RobustScale => "centers each column at its training median and divides by the \
            interquartile range, the outlier-resistant scaling of exploratory data analysis \
            (Tukey, 1977)."
            .to_string(),
        PlannedOp::Rank => "replaces each value by its mid-rank within the training column, i.e. \
            the average of the strict and non-strict ranks scaled to [0, 1], an empirical \
            distribution-function transform."
            .to_string(),
        PlannedOp::Blom => "maps mid-ranks to standard normal quantiles through the plotting \
            position (r + 1/8)/(n + 1/4), a normal-scores transform in the tradition of Blom \
            (1958), computed with Acklam's rational probit approximation."
            .to_string(),
        PlannedOp::QuantileBins { bins } => format!(
            "discretizes each column into {bins} equal-frequency bins at training quantiles and \
             reports the bin index scaled to [0, 1]."
        ),
        PlannedOp::BinOnehot { bins } => format!(
            "one-hot encodes {bins} equal-frequency quantile bins per column (an indicator basis)."
        ),
        PlannedOp::Hinge { knots } => format!(
            "expands each column into {knots} hinge functions max(x - t, 0) with knots at training \
             quantiles, the basis of multivariate adaptive regression splines (Friedman, 1991)."
        ),
        PlannedOp::WinsorClip { q } => format!(
            "winsorizes each column by clipping to its [{q}, 1-{q}] training quantiles, limiting \
             the leverage of extreme values (Tukey & McLaughlin, 1963).",
            q = crate::mathexpr::fmt_f64(*q)
        ),
        PlannedOp::Cheb { degree } => format!(
            "evaluates Chebyshev polynomials of the first kind T_1..T_{degree} on u = clip(2x - 1) \
             via the recurrence T_(k+1) = 2u T_k - T_(k-1)."
        ),
        PlannedOp::PolyInteract => "forms degree-two polynomial features: the square of every \
            column plus pairwise products, capped at 512 outputs."
            .to_string(),
        PlannedOp::RffCos { n, gamma, .. } => format!(
            "computes {n} random Fourier features sqrt(2/n)*cos(w.x + b) with Gaussian frequencies \
             scaled by sqrt(2*{g}) and uniform phases; such features approximate a shift-invariant \
             Gaussian kernel (Rahimi & Recht, 2007).",
            g = crate::mathexpr::fmt_f64(*gamma)
        ),
        PlannedOp::RffSin { n, gamma, .. } => format!(
            "computes {n} random Fourier features sqrt(2/n)*sin(w.x + b) with Gaussian frequencies \
             scaled by sqrt(2*{g}) and uniform phases, the sine counterpart of the Rahimi & Recht \
             (2007) construction.",
            g = crate::mathexpr::fmt_f64(*gamma)
        ),
        PlannedOp::RandProj { k, .. } => format!(
            "projects each row onto {k} random Gaussian directions scaled by 1/sqrt(k), preserving \
             geometry in expectation (Johnson & Lindenstrauss, 1984)."
        ),
        PlannedOp::LandmarkDist { k, gamma, .. } => format!(
            "measures Gaussian kernel similarity exp(-{g}*||x - l||^2) to {k} fixed random \
             landmarks, a landmark feature map in the spirit of Nystrom approximations \
             (Williams & Seeger, 2001).",
            g = crate::mathexpr::fmt_f64(*gamma)
        ),
        PlannedOp::ObliqueHinge { n, .. } => format!(
            "emits {n} oblique hinge units max(w.x - t, 0) whose random directions are scaled by \
             1/sqrt(p) and whose thresholds sit at the median training projection."
        ),
        PlannedOp::RowStats => "summarizes each row by ten distribution statistics: mean, \
            standard deviation, min, max, median, lower and upper quartiles, skewness, excess \
            kurtosis, and mean energy."
            .to_string(),
        PlannedOp::PairProd { topk } => format!(
            "multiplies pairs among the {topk} columns most correlated (absolute Pearson) with the \
             target, a screened interaction expansion."
        ),
        PlannedOp::TargetBinEncode { bins, prior } => format!(
            "replaces each value with the smoothed mean of the target inside its quantile bin \
             ({bins} bins, additive prior {p} toward the global mean), following Micci-Barreca \
             (2001).",
            p = crate::mathexpr::fmt_f64(*prior)
        ),
        PlannedOp::Sigmoid => "squashes every value through the logistic function \
            1/(1 + exp(-v)) with the argument clipped to [-30, 30]."
            .to_string(),
        PlannedOp::Step { t } => format!(
            "replaces each value with the hard indicator 1.0 when the value exceeds the threshold \
             {t} and 0.0 otherwise (a Heaviside step evaluated as an explicit conditional).",
            t = crate::mathexpr::fmt_f64(*t)
        ),
        PlannedOp::NormRow => "scales each row to unit Euclidean length (the norm is floored at \
            1e-9)."
            .to_string(),
        PlannedOp::Polar => "rewrites each consecutive column pair (a, b) in polar coordinates: \
            radius sqrt(a^2 + b^2) then angle atan2(b, a); an odd trailing column passes through \
            unchanged."
            .to_string(),
        PlannedOp::Lag { k } => format!(
            "shifts each row {k} columns to the right, filling vacated positions with zeros (a \
             within-row lag)."
        ),
        PlannedOp::Roll { k } => {
            format!("rotates each row {k} columns to the right (a circular shift).")
        }
        PlannedOp::FftMag { k } => format!(
            "computes magnitudes of the discrete Fourier transform of each row at frequencies \
             1..{k} (capped at half the width), scaled by 1/d — the direct O(d^2) sum for exact \
             reproducibility."
        ),
        PlannedOp::RowSpectral => "summarizes the spectrum of each row over DFT frequencies \
            1..d/2 by spectral entropy, first and second spectral moments, peak frequency, and \
            peak magnitude."
            .to_string(),
        PlannedOp::RowDiffStats => "summarizes the first differences of each row by mean, \
            standard deviation, min, max, mean absolute change, sign-change rate, lag-1 \
            autocorrelation of the original row, and mean squared change."
            .to_string(),
        PlannedOp::SvdProject { k, iters, .. } => format!(
            "projects mean-centered rows onto the top {k} principal directions found by {iters} \
             rounds of deterministic power iteration with Gram-Schmidt deflation (Pearson, 1901; \
             Golub & Van Loan, 2013)."
        ),
        PlannedOp::Nystrom { k, gamma, .. } => format!(
            "measures Gaussian kernel similarity exp(-{g}*||x - l||^2) to {k} training rows \
             chosen by a seeded partial Fisher-Yates shuffle, the Nystrom landmark construction \
             (Williams & Seeger, 2001).",
            g = crate::mathexpr::fmt_f64(*gamma)
        ),
        PlannedOp::MapExpr { source, .. } => format!(
            "applies the scalar formula f(x, row) = {source} to every cell, where x is the cell \
             value and row is the full row, with guarded arithmetic."
        ),
        PlannedOp::DeriveExpr { source, .. } => format!(
            "reduces each row to the single derived column g(row) = {source}, with guarded \
             arithmetic."
        ),
        PlannedOp::Symlog { c } => format!(
            "compresses heavy tails through the signed shifted logarithm asinh(x / {c}) = \
             sign(u) * ln(|u| + sqrt(u^2 + 1)) with u = x / {c}, a smooth monotone transform \
             that is linear near zero, logarithmic in the tails, and extrapolates beyond the \
             training range.",
            c = crate::mathexpr::fmt_f64(*c)
        ),
        PlannedOp::SelectCorr { k, by_var } => {
            if *by_var {
                format!(
                    "keeps the {k} columns with the largest training variance (ties to the \
                     lower index, original column order preserved), an unsupervised screen \
                     that narrows wide inputs."
                )
            } else {
                format!(
                    "keeps the {k} columns with the largest absolute Pearson correlation to \
                     the training target (ties to the lower index, original column order \
                     preserved), the sure-independence-screening step of Fan & Lv (2008)."
                )
            }
        }
        PlannedOp::Impute {
            strategy,
            add_indicator,
            ..
        } => {
            let fill = match strategy {
                0 => "the mean of the column's non-zero training values",
                2 => "a fixed constant",
                _ => "the median of the column's non-zero training values",
            };
            if *add_indicator {
                format!(
                    "treats exact-zero cells as missing and replaces them with {fill}, \
                     appending one 0/1 missingness indicator per column that contained a \
                     zero in training (Rubin, 1976)."
                )
            } else {
                format!(
                    "treats exact-zero cells as missing and replaces them with {fill} \
                     (Rubin, 1976)."
                )
            }
        }
        PlannedOp::PairRatio { topk } => format!(
            "forms the {topk} regularized ratios x_a * x_b / (x_b^2 + eps) with the strongest \
             absolute Pearson correlation to the training target, screening pairs over the 32 \
             most target-correlated columns; eps = max(1e-12, (0.01 * training std of x_b)^2) \
             keeps every ratio finite and smooth."
        ),
        PlannedOp::Spline { knots } => format!(
            "expands each column in the natural cubic spline basis with {knots} knots at the \
             interior training quantiles i/({knots}+1) (Hastie, Tibshirani & Friedman, ESL \
             §5.2.1): the raw value plus {m} truncated-power basis functions constrained to \
             extrapolate linearly beyond the boundary knots; a near-constant column falls back \
             to its raw value.",
            m = knots - 2
        ),
    }
}

/// Short phrase for titles.
pub fn op_short(op: &PlannedOp) -> &'static str {
    match op {
        // B05–B07: scalar-v2 named registration.
        PlannedOp::ScalarV2 { .. } => "scalar expression",
        // B13: version-two sparse Fourier power bank.
        PlannedOp::RowVectorV2 {
            state: crate::kernel::row_vector::FittedStage::Goertzel(_),
        } => "Fourier power bank",
        // B12: version-two FeatureWithinRow stage.
        PlannedOp::RowVectorV2 { .. } | PlannedOp::RowVectorInputV2 { .. } => {
            "feature-position filter"
        }
        PlannedOp::Standardize => "standard scaling",
        PlannedOp::RobustScale => "robust scaling",
        PlannedOp::Rank => "mid-rank transform",
        PlannedOp::Blom => "normal scores",
        PlannedOp::QuantileBins { .. } => "quantile binning",
        PlannedOp::BinOnehot { .. } => "quantile one-hot encoding",
        PlannedOp::Hinge { .. } => "hinge basis",
        PlannedOp::WinsorClip { .. } => "winsorizing",
        PlannedOp::Cheb { .. } => "Chebyshev basis",
        PlannedOp::PolyInteract => "pairwise products",
        PlannedOp::RffCos { .. } => "random Fourier cosine features",
        PlannedOp::RffSin { .. } => "random Fourier sine features",
        PlannedOp::RandProj { .. } => "Gaussian random projection",
        PlannedOp::LandmarkDist { .. } => "Gaussian landmark features",
        PlannedOp::ObliqueHinge { .. } => "oblique hinge features",
        PlannedOp::RowStats => "row summary statistics",
        PlannedOp::PairProd { .. } => "correlation-screened products",
        PlannedOp::TargetBinEncode { .. } => "smoothed target bin encoding",
        PlannedOp::Sigmoid => "logistic squash",
        PlannedOp::Step { .. } => "step indicator",
        PlannedOp::NormRow => "unit row scaling",
        PlannedOp::Polar => "polar coordinates",
        PlannedOp::Lag { .. } => "column shift",
        PlannedOp::Roll { .. } => "column rotation",
        PlannedOp::FftMag { .. } => "Fourier magnitudes",
        PlannedOp::RowSpectral => "row spectral summary",
        PlannedOp::RowDiffStats => "row difference statistics",
        PlannedOp::SvdProject { .. } => "principal component projection",
        PlannedOp::Nystrom { .. } => "Nystrom landmark features",
        PlannedOp::MapExpr { .. } => "an elementwise formula",
        PlannedOp::DeriveExpr { .. } => "a derived column",
        PlannedOp::Symlog { .. } => "signed log compression",
        PlannedOp::SelectCorr { .. } => "column screening",
        PlannedOp::Impute { .. } => "zero imputation",
        PlannedOp::PairRatio { .. } => "ratio features",
        PlannedOp::Spline { .. } => "natural cubic splines",
    }
}
