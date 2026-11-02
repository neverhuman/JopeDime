use super::*;

pub(crate) fn op_label(op: &PlannedOp) -> String {
    match op {
        PlannedOp::Standardize => "standardize".into(),
        PlannedOp::RobustScale => "robust_scale".into(),
        PlannedOp::Rank => "rank".into(),
        PlannedOp::Blom => "blom".into(),
        PlannedOp::Sigmoid => "sigmoid".into(),
        PlannedOp::Step { .. } => "step".into(),
        PlannedOp::WinsorClip { .. } => "winsor_clip".into(),
        PlannedOp::NormRow => "norm_row".into(),
        PlannedOp::RffCos { .. } => "rff_cos".into(),
        PlannedOp::RffSin { .. } => "rff_sin".into(),
        PlannedOp::Cheb { .. } => "cheb".into(),
        PlannedOp::Hinge { .. } => "hinge".into(),
        PlannedOp::PolyInteract => "poly_interact".into(),
        PlannedOp::QuantileBins { .. } => "quantile_bins".into(),
        PlannedOp::BinOnehot { .. } => "bin_onehot".into(),
        PlannedOp::PairProd { .. } => "pair_prod".into(),
        PlannedOp::RandProj { .. } => "rand_proj".into(),
        PlannedOp::RowStats => "row_stats".into(),
        PlannedOp::RowDiffStats => "row_diff_stats".into(),
        PlannedOp::ObliqueHinge { .. } => "oblique_hinge".into(),
        PlannedOp::LandmarkDist { .. } => "landmark_dist".into(),
        PlannedOp::TargetBinEncode { .. } => "target_bin_encode".into(),
        other => format!("{other:?}"),
    }
}

pub(crate) fn head_label(head: &PlannedHead) -> String {
    match head {
        PlannedHead::Ridge { .. } => "ridge".into(),
        PlannedHead::ElasticNet { .. } => "elasticnet".into(),
        other => format!("{other:?}"),
    }
}

pub(crate) fn head_supported(head: &PlannedHead) -> bool {
    matches!(
        head,
        PlannedHead::Ridge { .. } | PlannedHead::ElasticNet { .. }
    )
}

/// The emitted struct + fit + predict definition for a supported head.
pub(crate) fn head_src_cpp(head: &PlannedHead) -> &'static str {
    match head {
        PlannedHead::Ridge { .. } => HEAD_RIDGE,
        PlannedHead::ElasticNet { .. } => HEAD_ELASTICNET,
        _ => unreachable!("head_src_cpp only reached for a supported head"),
    }
}

/// `(fit-call over x/y, predict-call over the `head` member, C++ type name)`.
pub(crate) fn head_calls_cpp(head: &PlannedHead) -> (String, String, &'static str) {
    match head {
        PlannedHead::Ridge { lambda_frac } => (
            format!("ridge_regression_fit(x, y, {})", fmt_f64(*lambda_frac)),
            "ridge_regression_predict(head, x)".into(),
            "RidgeRegression",
        ),
        PlannedHead::ElasticNet {
            alpha,
            l1_ratio,
            sweeps,
        } => (
            format!(
                "elastic_net_fit(x, y, {}, {}, {sweeps})",
                fmt_f64(*alpha),
                fmt_f64(*l1_ratio)
            ),
            "elastic_net_predict(head, x)".into(),
            "ElasticNet",
        ),
        _ => unreachable!("head_calls_cpp only reached for a supported head"),
    }
}

/// C++ body for a shared helper. Helpers outside this set return "" — they
/// cannot appear because the op gate rejects any op that would need them.
/// `RngUniform`/`RngRange`/`RngGaussian` return "" because C++ cannot reopen a
/// struct across definitions the way Rust's `impl` blocks do, so those draw
/// methods are bundled into the single `Pcg32` definition.
pub(crate) fn helper_cpp(helper: Helper, partition_point_bounds: bool) -> &'static str {
    match helper {
        Helper::Sanitize => H_SANITIZE,
        Helper::FinishStage => H_FINISH_STAGE,
        Helper::StdFit => H_STD_FIT,
        Helper::CholeskySolve => H_CHOLESKY,
        Helper::SortedColumn => H_SORTED_COLUMN,
        Helper::Quantile => H_QUANTILE,
        Helper::LowerBound => {
            if partition_point_bounds {
                H_LOWER_BOUND_PARTITION_POINT
            } else {
                H_LOWER_BOUND
            }
        }
        Helper::UpperBound => {
            if partition_point_bounds {
                H_UPPER_BOUND_PARTITION_POINT
            } else {
                H_UPPER_BOUND
            }
        }
        Helper::MidRank => H_MID_RANK,
        Helper::Probit => H_PROBIT,
        Helper::MathSigmoid => H_MATH_SIGMOID,
        Helper::Pcg32 => H_PCG32,
        Helper::StageRng => H_STAGE_RNG,
        Helper::Dot => H_DOT,
        Helper::QuantileEdges => H_QUANTILE_EDGES,
        Helper::Pearson => H_PEARSON,
        Helper::SoftThreshold => H_SOFT_THRESHOLD,
        Helper::RbfRow => H_RBF_ROW,
        _ => "",
    }
}

/// `(fn-name, definition)` for a supported op, or `None` if unsupported. The
/// name deduplicates repeated ops (the definition is emitted once).
pub(crate) fn op_cpp(op: &PlannedOp) -> Option<(&'static str, &'static str)> {
    match op {
        // B13: the goertzel_bank helper, only for a bank-bearing program.
        PlannedOp::RowVectorV2 {
            state: crate::kernel::row_vector::FittedStage::Goertzel(_),
        } => Some((
            "row_vector_goertzel_v2",
            crate::kernel::row_vector::emit::CPP_GOERTZEL_HELPER,
        )),
        // B12: explicit v2 row-vector recipe.
        PlannedOp::RowVectorV2 { .. } | PlannedOp::RowVectorInputV2 { .. } => {
            Some(("row_vector_v2", crate::kernel::row_vector::emit::CPP_HELPER))
        }
        PlannedOp::Standardize => Some(("standard_scaler", OP_STANDARDIZE)),
        PlannedOp::RobustScale => Some(("robust_scaler", OP_ROBUST_SCALE)),
        PlannedOp::Rank => Some(("mid_rank_transform", OP_RANK)),
        PlannedOp::Blom => Some(("blom_normal_scores", OP_BLOM)),
        PlannedOp::Sigmoid => Some(("logistic_squash", OP_SIGMOID)),
        PlannedOp::Step { .. } => Some(("step_indicator", OP_STEP)),
        PlannedOp::WinsorClip { .. } => Some(("winsor_limiter", OP_WINSOR)),
        PlannedOp::NormRow => Some(("unit_norm_rows", OP_NORM_ROW)),
        PlannedOp::RffCos { .. } => Some(("random_fourier_cosine_features", OP_RFF_COS)),
        PlannedOp::RffSin { .. } => Some(("random_fourier_sine_features", OP_RFF_SIN)),
        PlannedOp::Cheb { .. } => Some(("chebyshev_basis", OP_CHEB)),
        PlannedOp::Hinge { .. } => Some(("hinge_basis", OP_HINGE)),
        PlannedOp::PolyInteract => Some(("pairwise_products", OP_POLY_INTERACT)),
        PlannedOp::QuantileBins { .. } => Some(("quantile_bin_indexer", OP_QUANTILE_BINS)),
        PlannedOp::BinOnehot { .. } => Some(("quantile_onehot_encoder", OP_BIN_ONEHOT)),
        PlannedOp::PairProd { .. } => Some(("correlation_screened_products", OP_PAIR_PROD)),
        PlannedOp::RandProj { .. } => Some(("gaussian_random_projection", OP_RAND_PROJ)),
        PlannedOp::RowStats => Some(("row_summary_statistics", OP_ROW_STATS)),
        PlannedOp::RowDiffStats => Some(("row_difference_statistics", OP_ROW_DIFF_STATS)),
        PlannedOp::ObliqueHinge { .. } => Some(("oblique_hinge_features", OP_OBLIQUE_HINGE)),
        PlannedOp::LandmarkDist { .. } => Some(("gaussian_landmark_features", OP_LANDMARK_DIST)),
        PlannedOp::TargetBinEncode { .. } => {
            Some(("smoothed_target_bin_encoder", OP_TARGET_BIN_ENCODE))
        }
        _ => None,
    }
}

/// C++ struct type holding a stage's fitted state, or `None` if stateless.
pub(crate) fn op_struct_cpp(op: &PlannedOp) -> Option<&'static str> {
    match op {
        PlannedOp::Standardize => Some("StandardScaler"),
        PlannedOp::RobustScale => Some("RobustScaler"),
        PlannedOp::Rank => Some("MidRankTransform"),
        PlannedOp::Blom => Some("BlomNormalScores"),
        PlannedOp::WinsorClip { .. } => Some("WinsorLimiter"),
        PlannedOp::RffCos { .. } => Some("RandomFourierCosineFeatures"),
        PlannedOp::RffSin { .. } => Some("RandomFourierSineFeatures"),
        PlannedOp::Hinge { .. } => Some("HingeBasis"),
        PlannedOp::PolyInteract => Some("PairwiseProducts"),
        PlannedOp::QuantileBins { .. } => Some("QuantileBinIndexer"),
        PlannedOp::BinOnehot { .. } => Some("QuantileOnehotEncoder"),
        PlannedOp::PairProd { .. } => Some("CorrelationScreenedProducts"),
        PlannedOp::RandProj { .. } => Some("GaussianRandomProjection"),
        PlannedOp::ObliqueHinge { .. } => Some("ObliqueHingeFeatures"),
        PlannedOp::LandmarkDist { .. } => Some("GaussianLandmarkFeatures"),
        PlannedOp::TargetBinEncode { .. } => Some("SmoothedTargetBinEncoder"),
        _ => None,
    }
}

/// `(fit-call, apply-template)` for a stage. `fit-call` is `None` for stateless
/// ops. The apply template uses `{state}`/`{append}` placeholders.
pub(crate) fn op_calls_cpp(op: &PlannedOp) -> (Option<String>, String) {
    match op {
        // B12: explicit v2 row-vector recipe.
        PlannedOp::RowVectorInputV2 { width } => {
            (None, format!("row_vector_v2_input(x, {width}, {{append}})"))
        }
        PlannedOp::RowVectorV2 { state } => (
            None,
            crate::kernel::row_vector::emit::apply_call(
                state,
                crate::kernel::row_vector::emit::RowRecipeLanguage::Cpp17,
            ),
        ),
        PlannedOp::Standardize => (
            Some("standard_scaler_fit(x)".into()),
            "standard_scaler_apply({state}, x, {append})".into(),
        ),
        PlannedOp::RobustScale => (
            Some("robust_scaler_fit(x)".into()),
            "robust_scaler_apply({state}, x, {append})".into(),
        ),
        PlannedOp::Rank => (
            Some("mid_rank_transform_fit(x)".into()),
            "mid_rank_transform_apply({state}, x, {append})".into(),
        ),
        PlannedOp::Blom => (
            Some("blom_normal_scores_fit(x)".into()),
            "blom_normal_scores_apply({state}, x, {append})".into(),
        ),
        PlannedOp::WinsorClip { q } => (
            Some(format!("winsor_limiter_fit(x, {})", fmt_f64(*q))),
            "winsor_limiter_apply({state}, x, {append})".into(),
        ),
        PlannedOp::Sigmoid => (None, "logistic_squash_apply(x, {append})".into()),
        PlannedOp::Step { t } => (
            None,
            format!("step_indicator_apply(x, {}, {{append}})", fmt_f64(*t)),
        ),
        PlannedOp::NormRow => (None, "unit_norm_rows_apply(x, {append})".into()),
        PlannedOp::RffCos { n, gamma, seed } => (
            Some(format!(
                "random_fourier_cosine_features_fit(x, {n}, {}, {seed}ULL)",
                fmt_f64(*gamma)
            )),
            "random_fourier_cosine_features_apply({state}, x, {append})".into(),
        ),
        PlannedOp::RffSin { n, gamma, seed } => (
            Some(format!(
                "random_fourier_sine_features_fit(x, {n}, {}, {seed}ULL)",
                fmt_f64(*gamma)
            )),
            "random_fourier_sine_features_apply({state}, x, {append})".into(),
        ),
        PlannedOp::Cheb { degree } => (
            None,
            format!("chebyshev_basis_apply(x, {degree}, {{append}})"),
        ),
        PlannedOp::Hinge { knots } => (
            Some(format!("hinge_basis_fit(x, {knots})")),
            "hinge_basis_apply({state}, x, {append})".into(),
        ),
        PlannedOp::PolyInteract => (
            Some("pairwise_products_fit(x)".into()),
            "pairwise_products_apply({state}, x, {append})".into(),
        ),
        PlannedOp::QuantileBins { bins } => (
            Some(format!("quantile_bin_indexer_fit(x, {bins})")),
            format!("quantile_bin_indexer_apply({{state}}, x, {bins}, {{append}})"),
        ),
        PlannedOp::BinOnehot { bins } => (
            Some(format!("quantile_onehot_encoder_fit(x, {bins})")),
            format!("quantile_onehot_encoder_apply({{state}}, x, {bins}, {{append}})"),
        ),
        PlannedOp::PairProd { topk } => (
            Some(format!("correlation_screened_products_fit(x, y, {topk})")),
            "correlation_screened_products_apply({state}, x, {append})".into(),
        ),
        PlannedOp::RandProj { k, seed } => (
            Some(format!("gaussian_random_projection_fit(x, {k}, {seed}ULL)")),
            "gaussian_random_projection_apply({state}, x, {append})".into(),
        ),
        PlannedOp::RowStats => (None, "row_summary_statistics_apply(x, {append})".into()),
        PlannedOp::RowDiffStats => (None, "row_difference_statistics_apply(x, {append})".into()),
        PlannedOp::ObliqueHinge { n, seed } => (
            Some(format!("oblique_hinge_features_fit(x, {n}, {seed}ULL)")),
            "oblique_hinge_features_apply({state}, x, {append})".into(),
        ),
        PlannedOp::LandmarkDist { k, gamma, seed } => (
            Some(format!("gaussian_landmark_features_fit(x, {k}, {seed}ULL)")),
            format!(
                "gaussian_landmark_features_apply({{state}}, x, {}, {{append}})",
                fmt_f64(*gamma)
            ),
        ),
        PlannedOp::TargetBinEncode { bins, prior } => (
            Some(format!(
                "smoothed_target_bin_encoder_fit(x, y, {bins}, {})",
                fmt_f64(*prior)
            )),
            "smoothed_target_bin_encoder_apply({state}, x, {append})".into(),
        ),
        other => unreachable!("op_calls_cpp reached for unsupported op {other:?}"),
    }
}
