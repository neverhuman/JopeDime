//! Public descriptions for feature-stage operations.

use super::PublicName;

/// Public descriptions for every operation handled by `crate::ops` and every
/// transform in the governed operator registry.
pub const OPERATIONS: &[PublicName] = &[
    PublicName {
        key: "bin_onehot",
        public_name: "one-hot bins",
        purpose: "represent each value by its quantile bin",
    },
    PublicName {
        key: "blom",
        public_name: "normal score transform",
        purpose: "map feature ranks to approximate normal values",
    },
    PublicName {
        key: "cheb",
        public_name: "polynomial features",
        purpose: "add smooth curves for nonlinear patterns",
    },
    PublicName {
        key: "count_sketch",
        public_name: "compact feature sketch",
        purpose: "summarize many features in a smaller space",
    },
    PublicName {
        key: "derive",
        public_name: "derived features",
        purpose: "calculate new values from simple feature expressions",
    },
    PublicName {
        key: "fastfood_rbf",
        public_name: "fast kernel approximation",
        purpose: "approximate smooth similarity with compact projections",
    },
    PublicName {
        key: "fft_mag",
        public_name: "frequency magnitudes",
        purpose: "measure repeating patterns within each row",
    },
    PublicName {
        key: "hinge",
        public_name: "hinge features",
        purpose: "add piecewise linear changes around learned knots",
    },
    PublicName {
        key: "impute",
        public_name: "missing-value replacements",
        purpose: "fill missing values and optionally mark them",
    },
    PublicName {
        key: "lag",
        public_name: "lagged values",
        purpose: "add values from earlier positions in a row",
    },
    PublicName {
        key: "landmark_dist",
        public_name: "landmark distances",
        purpose: "measure distance from learned reference points",
    },
    PublicName {
        key: "map",
        public_name: "mapped features",
        purpose: "apply a simple expression to each feature value",
    },
    PublicName {
        key: "norm_features",
        public_name: "standardized features",
        purpose: "center features and scale them to comparable ranges",
    },
    PublicName {
        key: "norm_row",
        public_name: "row-normalized features",
        purpose: "divide each row by its length",
    },
    PublicName {
        key: "nystrom",
        public_name: "kernel landmark features",
        purpose: "approximate nonlinear similarity from sampled rows",
    },
    PublicName {
        key: "nystrom_leverage",
        public_name: "leverage-based kernel features",
        purpose: "select informative reference rows for a kernel approximation",
    },
    PublicName {
        key: "oblique_hinge",
        public_name: "projected hinge features",
        purpose: "add thresholds along learned feature combinations",
    },
    PublicName {
        key: "pair_prod",
        public_name: "feature products",
        purpose: "combine selected feature pairs to capture nonlinear relationships",
    },
    PublicName {
        key: "pair_ratio",
        public_name: "pairwise ratios",
        purpose: "represent selected feature ratios with protected denominators",
    },
    PublicName {
        key: "polar",
        public_name: "polar coordinates",
        purpose: "describe paired values by radius and angle",
    },
    PublicName {
        key: "poly_interact",
        public_name: "pairwise products",
        purpose: "combine feature pairs to express interactions",
    },
    PublicName {
        key: "quantile_bins",
        public_name: "quantile bins",
        purpose: "group values into bins with similar counts",
    },
    PublicName {
        key: "rand_proj",
        public_name: "random projections",
        purpose: "reduce feature width while preserving broad distances",
    },
    PublicName {
        key: "rank",
        public_name: "ranked features",
        purpose: "replace values with their relative order",
    },
    PublicName {
        key: "rff_cos",
        public_name: "smooth periodic features",
        purpose: "approximate similarity with cosine curves",
    },
    PublicName {
        key: "rff_halton",
        public_name: "low-discrepancy kernel features",
        purpose: "approximate smooth similarity with evenly spread projections",
    },
    PublicName {
        key: "rff_orthogonal",
        public_name: "orthogonal kernel features",
        purpose: "approximate smooth similarity with perpendicular projections",
    },
    PublicName {
        key: "rff_sin",
        public_name: "smooth periodic features",
        purpose: "approximate similarity with sine curves",
    },
    PublicName {
        key: "robust_scale",
        public_name: "robustly scaled features",
        purpose: "center by the median and reduce sensitivity to extreme values",
    },
    PublicName {
        key: "roll",
        public_name: "rolling summaries",
        purpose: "summarize nearby values across a moving window",
    },
    PublicName {
        key: "row_diff_stats",
        public_name: "difference summaries",
        purpose: "describe changes between neighboring values",
    },
    PublicName {
        key: "row_spectral",
        public_name: "spectral summaries",
        purpose: "summarize frequency energy within each row",
    },
    PublicName {
        key: "row_stats",
        public_name: "row summaries",
        purpose: "summarize each row with basic statistics",
    },
    PublicName {
        key: "select_corr",
        public_name: "correlation-based feature selection",
        purpose: "keep features most associated with the outcome or their spread",
    },
    PublicName {
        key: "sigmoid",
        public_name: "bounded curve transform",
        purpose: "map values through a smooth bounded curve",
    },
    PublicName {
        key: "spline",
        public_name: "spline features",
        purpose: "add smooth curves joined at learned knots",
    },
    PublicName {
        key: "standardize",
        public_name: "standardized features",
        purpose: "center features and scale them to comparable ranges",
    },
    PublicName {
        key: "step",
        public_name: "threshold indicators",
        purpose: "mark values above a chosen threshold",
    },
    PublicName {
        key: "svd_project",
        public_name: "low-rank projection",
        purpose: "retain the leading directions in the data",
    },
    PublicName {
        key: "symlog",
        public_name: "symmetric log transform",
        purpose: "compress large magnitudes while preserving their sign",
    },
    PublicName {
        key: "target_bin_encode",
        public_name: "target-guided bin values",
        purpose: "replace each bin with a smoothed outcome average",
    },
    PublicName {
        key: "tensor_sketch",
        public_name: "compact interaction sketch",
        purpose: "represent higher-order feature combinations in less space",
    },
    PublicName {
        key: "winsor_clip",
        public_name: "winsorized features",
        purpose: "limit extreme values at chosen quantiles",
    },
    PublicName {
        key: "winsor_moments",
        public_name: "winsorized features",
        purpose: "limit extreme values at chosen quantiles",
    },
];
