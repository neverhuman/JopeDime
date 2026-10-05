//! C++ emitter: a single std-only `.cpp` file that compiles with
//! `g++ -O2 -std=c++17 -ffp-contract=off -fno-fast-math model.cpp` and mirrors
//! the Rust and Python numeric behavior on the certified operator set.
//!
//! This is the deterministic CPU rung beneath the optional CUDA build: the same
//! source, compiled without fused-multiply-add contraction so every `a*b + c`
//! stays two separately rounded IEEE-754 operations, exactly as the reference
//! runtime evaluates it. Transcendental ops (`cos`/`ln`/`exp`) reach the same
//! glibc libm as the Rust artifact, applied to arguments built in the identical
//! fixed order, so the three languages agree to the last bit. Coverage grows one
//! certified op at a time; unsupported ops are reported so a broad
//! half-implemented surface never ships silently.

use crate::mathexpr::fmt_f64;
use crate::spec::{helper_set, Helper, Plan, PlannedHead, PlannedOp, PlannedStage};
// B05–B07: explicit scalar-v2 recipe assembly.
use crate::kernel::scalar_program::emit::{self as scalar_emit, ScalarLanguage};
use crate::transpile::{EmitOptions, MathKernel};

/// Emit a standalone C++ program for `plan`, or list the elements this rung does
/// not yet support (mirrors `supports()` semantics for the Rust/Python paths).
pub fn emit_cpp(plan: &Plan, opts: &EmitOptions) -> Result<String, Vec<String>> {
    emit_cpp_inner(plan, opts, false)
}

impl EmitOptions {
    pub(crate) fn cpp_expression_recipe(&self, plan: &Plan) -> Result<String, Vec<String>> {
        emit_cpp_inner(plan, self, true)
    }
}

fn emit_cpp_inner(
    plan: &Plan,
    opts: &EmitOptions,
    expressions: bool,
) -> Result<String, Vec<String>> {
    let (stages, head) = match plan {
        Plan::Pipeline {
            target: None,
            stages,
            head,
        } => (stages, head),
        Plan::Pipeline {
            target: Some(_), ..
        } => return Err(vec!["target transforms".to_string()]),
        Plan::Symbolic { expr, affine, .. } if expressions => {
            let model = symbolic_cpp(expr, *affine)?;
            return Ok([
                PREAMBLE,
                crate::FittedCppArtifact::expression_helpers(),
                H_SANITIZE,
                EXPRESSION_SHAPE,
                &model,
                IO_BLOCK,
            ]
            .join("\n\n")
                + "\n");
        }
        Plan::Symbolic { .. } => return Err(vec!["symbolic formula programs".to_string()]),
    };

    let mut unsupported: Vec<String> = Vec::new();
    for stage in stages {
        if op_cpp(&stage.op).is_none()
            && !(expressions
                && matches!(
                    stage.op,
                    PlannedOp::MapExpr { .. }
                        | PlannedOp::DeriveExpr { .. }
                        | PlannedOp::ScalarV2 { .. }
                ))
        {
            unsupported.push(format!("transform {}", op_label(&stage.op)));
        }
    }
    if !head_supported(head) {
        unsupported.push(format!("head {}", head_label(head)));
    }
    if !unsupported.is_empty() {
        return Err(unsupported);
    }

    if scalar_emit::raw_input(plan) && opts.math_kernel == MathKernel::Portable {
        return Err(vec!["scalar-v2 old operations require PlatformLibm".into()]);
    }
    let mut sections: Vec<String> = Vec::new();
    sections.push(PREAMBLE.to_string());
    if expressions {
        sections.push(crate::FittedCppArtifact::expression_helpers().to_string());
        sections.push(EXPRESSION_SHAPE.to_string());
    }
    if opts.math_kernel == MathKernel::Portable {
        sections.push(PORTABLE_WARNING_BEGIN.to_string());
        sections.push(PORTABLE_MATH.to_string());
        sections.push(PORTABLE_WARNING_END.to_string());
    }
    for helper in helper_set(plan) {
        if expressions && matches!(helper, Helper::MathSigmoid) {
            // The shared expression helper block supplies this same scalar.
            continue;
        }
        let body = helper_cpp(helper);
        if !body.is_empty() {
            sections.push(body.to_string());
        }
    }
    let scalar_programs = scalar_emit::programs(plan);
    if !scalar_programs.is_empty() {
        sections.push(
            scalar_emit::emit_v2_helpers(&scalar_programs, ScalarLanguage::Cpp17)
                .map_err(|e| vec![e.to_string()])?,
        );
    }
    let mut seen: Vec<&'static str> = Vec::new();
    for (index, stage) in stages.iter().enumerate() {
        if expressions {
            match &stage.op {
                PlannedOp::ScalarV2 {
                    program,
                    carrier,
                    width,
                } => {
                    sections.push(
                        scalar_emit::emit_stage(
                            program,
                            *carrier,
                            *width,
                            index + 1,
                            ScalarLanguage::Cpp17,
                        )
                        .map_err(|e| vec![e.to_string()])?,
                    );
                }
                PlannedOp::MapExpr { expr, .. } | PlannedOp::DeriveExpr { expr, .. } => {
                    sections.push(expression_stage_cpp(
                        index + 1,
                        expr,
                        matches!(stage.op, PlannedOp::MapExpr { .. }),
                    )?);
                }
                _ => {}
            }
        }
        if let Some((name, src)) = op_cpp(&stage.op) {
            if !seen.contains(&name) {
                seen.push(name);
                sections.push(src.to_string());
            }
        }
    }
    sections.push(head_src_cpp(head).to_string());
    sections.push(model_cpp(stages, head, expressions));
    sections.push(IO_BLOCK.to_string());
    let mut source = sections.join("\n\n") + "\n";
    if opts.math_kernel == MathKernel::Portable {
        for (platform, portable) in [
            ("std::exp(", "xexp("),
            ("std::log(", "xln("),
            ("std::sin(", "xsin("),
            ("std::cos(", "xcos("),
        ] {
            source = source.replace(platform, portable);
        }
    }
    Ok(source)
}

fn op_label(op: &PlannedOp) -> String {
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

fn head_label(head: &PlannedHead) -> String {
    match head {
        PlannedHead::Ridge { .. } => "ridge".into(),
        PlannedHead::ElasticNet { .. } => "elasticnet".into(),
        other => format!("{other:?}"),
    }
}

fn head_supported(head: &PlannedHead) -> bool {
    matches!(
        head,
        PlannedHead::Ridge { .. } | PlannedHead::ElasticNet { .. }
    )
}

/// The emitted struct + fit + predict definition for a supported head.
fn head_src_cpp(head: &PlannedHead) -> &'static str {
    match head {
        PlannedHead::Ridge { .. } => HEAD_RIDGE,
        PlannedHead::ElasticNet { .. } => HEAD_ELASTICNET,
        _ => unreachable!("head_src_cpp only reached for a supported head"),
    }
}

/// `(fit-call over x/y, predict-call over the `head` member, C++ type name)`.
fn head_calls_cpp(head: &PlannedHead) -> (String, String, &'static str) {
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
fn helper_cpp(helper: Helper) -> &'static str {
    match helper {
        Helper::Sanitize => H_SANITIZE,
        Helper::FinishStage => H_FINISH_STAGE,
        Helper::StdFit => H_STD_FIT,
        Helper::CholeskySolve => H_CHOLESKY,
        Helper::SortedColumn => H_SORTED_COLUMN,
        Helper::Quantile => H_QUANTILE,
        Helper::LowerBound => H_LOWER_BOUND,
        Helper::UpperBound => H_UPPER_BOUND,
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
fn op_cpp(op: &PlannedOp) -> Option<(&'static str, &'static str)> {
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
fn op_struct_cpp(op: &PlannedOp) -> Option<&'static str> {
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
fn op_calls_cpp(op: &PlannedOp) -> (Option<String>, String) {
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

/// Assemble the `Model` struct with its `fit`/`predict` methods from the plan.
fn model_cpp(stages: &[PlannedStage], head: &PlannedHead, expressions: bool) -> String {
    let (head_fit, head_predict, head_ty) = head_calls_cpp(head);

    let mut fields: Vec<String> = Vec::new();
    let mut fit_body: Vec<String> = Vec::new();
    let mut assigns: Vec<String> = Vec::new();
    let mut predict_body: Vec<String> = Vec::new();

    for (i, stage) in stages.iter().enumerate() {
        let n = i + 1;
        let append = if stage.append { "true" } else { "false" };
        let (fit_call, apply_template) = match &stage.op {
            PlannedOp::ScalarV2 { .. } if expressions => {
                (None, format!("scalar_stage{n}_apply(x, {{append}})"))
            }
            PlannedOp::MapExpr { .. } | PlannedOp::DeriveExpr { .. } if expressions => {
                (None, format!("expression_stage{n}(x, {{append}})"))
            }
            _ => op_calls_cpp(&stage.op),
        };
        let apply = apply_template
            .replace("{state}", &format!("stage{n}"))
            .replace("{append}", append);
        if let Some(ty) = op_struct_cpp(&stage.op) {
            fields.push(format!("    {ty} stage{n};"));
        }
        if let Some(fit_call) = fit_call {
            fit_body.push(format!("    auto stage{n} = {fit_call};"));
            assigns.push(format!("    m.stage{n} = std::move(stage{n});"));
        }
        fit_body.push(format!("    x = {apply};"));
        predict_body.push(format!("    x = {apply};"));
    }

    let mut out = String::new();
    out.push_str("// Feature stages applied in order, then the ridge regression head.\n");
    out.push_str("struct Model {\n");
    if expressions {
        out.push_str("    std::size_t input_width=0;\n");
    }
    for field in &fields {
        out.push_str(field);
        out.push('\n');
    }
    out.push_str(&format!("    {head_ty} head;\n\n"));
    out.push_str(
        "    static Model fit(const std::vector<std::vector<double>>& x_in, const std::vector<double>& y);\n",
    );
    out.push_str(
        "    std::vector<double> predict(const std::vector<std::vector<double>>& x_in) const;\n",
    );
    // B05–B07: public prediction faults on the typed one-stage recipe.
    if stages.len() == 1
        && matches!(
            stages.first().map(|s| &s.op),
            Some(PlannedOp::ScalarV2 { .. })
        )
    {
        out.push_str("    std::pair<std::vector<double>,std::vector<std::uint32_t>> predict_with_faults(const std::vector<std::vector<double>>& rows) const {auto features=scalar_stage1_values(rows);return {predict(rows),std::move(features.second)};}\n");
    }
    out.push_str("};\n\n");

    out.push_str(
        "Model Model::fit(const std::vector<std::vector<double>>& x_in, const std::vector<double>& y) {\n",
    );
    out.push_str("    if (x_in.empty() || y.size() < x_in.size()) {\n");
    out.push_str(
        "        throw std::invalid_argument(\"training data is empty or target is short\");\n",
    );
    out.push_str("    }\n");
    if expressions {
        out.push_str("    const std::size_t input_width = expression_width(x_in);\n");
    }
    let raw_scalar = matches!(
        stages.first().map(|s| &s.op),
        Some(PlannedOp::ScalarV2 { .. })
    );
    if raw_scalar {
        out.push_str("    if(x_in.empty()||y.size()!=x_in.size())throw std::invalid_argument(\"scalar training shape\");for(double v:y)if(!std::isfinite(v)||std::abs(v)>1e12)throw std::invalid_argument(\"scalar training labels\");\n");
    }
    // B12: preserve the explicit v2 fit contract across native recipes.
    if stages
        .iter()
        .any(|s| matches!(s.op, PlannedOp::RowVectorInputV2 { .. }))
    {
        out.push_str("    if(x_in.empty() || y.size()!=x_in.size())throw std::invalid_argument(\"row-vector training shape\");for(double v:y)if(!std::isfinite(v)||std::abs(v)>1e12)throw std::invalid_argument(\"row-vector training labels\");\n");
    }
    out.push_str(if raw_scalar {
        "    std::vector<std::vector<double>> x = x_in;\n"
    } else {
        "    std::vector<std::vector<double>> x = sanitize(x_in);\n"
    });
    for line in &fit_body {
        out.push_str(line);
        out.push('\n');
    }
    out.push_str(&format!("    {head_ty} head = {head_fit};\n"));
    if raw_scalar && matches!(head, PlannedHead::Ridge { .. }) {
        out.push_str("    if(std::all_of(y.begin(),y.end(),[](double v){return v==0.0&&std::signbit(v);}))head.w[0]=-0.0;\n");
    }

    out.push_str("    Model m;\n");
    if expressions {
        out.push_str("    m.input_width = input_width;\n");
    }
    for line in &assigns {
        out.push_str(line);
        out.push('\n');
    }
    out.push_str("    m.head = std::move(head);\n");
    out.push_str("    return m;\n");
    out.push_str("}\n\n");

    out.push_str(
        "std::vector<double> Model::predict(const std::vector<std::vector<double>>& x_in) const {\n",
    );
    if expressions {
        out.push_str("    if (!x_in.empty() && expression_width(x_in) != input_width) throw std::invalid_argument(\"prediction width\");\n");
    }
    out.push_str(if raw_scalar {
        "    std::vector<std::vector<double>> x = x_in;\n"
    } else {
        "    std::vector<std::vector<double>> x = sanitize(x_in);\n"
    });
    for line in &predict_body {
        out.push_str(line);
        out.push('\n');
    }
    out.push_str(&format!("    return {head_predict};\n"));
    out.push('}');
    out
}

fn expression_stage_cpp(
    n: usize,
    expr: &crate::mathexpr::MathAst,
    map: bool,
) -> Result<String, Vec<String>> {
    let expression = crate::math_expr::MathExpr::from_ast(expr.clone())
        .and_then(|expr| expr.to_cpp17(map))
        .map_err(|e| vec![e.to_string()])?;
    let values = if map {
        format!("for(double x:values) {{ (void)x; output.push_back(math_safe({expression})); }}")
    } else {
        format!("output.push_back(math_safe({expression}));")
    };
    Ok(format!(
        r#"static std::vector<std::vector<double>> expression_stage{n}(const std::vector<std::vector<double>>& input, bool append) {{
    std::vector<std::vector<double>> out;
    out.reserve(input.size());
    for(const auto& values:input) {{
        const double* row=values.data(); const std::size_t row_len=values.size();
        (void)row; (void)row_len;
        std::vector<double> output;
        {values}
        out.push_back(std::move(output));
    }}
    return finish_stage(input, out, append);
}}"#
    ))
}

fn symbolic_cpp(expr: &crate::mathexpr::MathAst, affine: bool) -> Result<String, Vec<String>> {
    let expression = crate::math_expr::MathExpr::from_ast(expr.clone())
        .and_then(|expr| expr.to_cpp17(false))
        .map_err(|e| vec![e.to_string()])?;
    let fit = if affine {
        "double fm=0.0,ym=0.0; for(std::size_t i=0;i<x.size();++i){fm+=expression_value(x[i]);ym+=y[i];} fm/=static_cast<double>(x.size());ym/=static_cast<double>(x.size());double cov=0.0,var=0.0;for(std::size_t i=0;i<x.size();++i){double d=expression_value(x[i])-fm;cov+=d*(y[i]-ym);var+=d*d;}if(var>1e-24){m.a=cov/var;m.b=ym-m.a*fm;}else{m.a=0.0;m.b=ym;}"
    } else {
        "(void)y;"
    };
    Ok(format!(
        r#"static double expression_value(const std::vector<double>& values) {{
    const double* row=values.data(); const std::size_t row_len=values.size();
    (void)row; (void)row_len;
    return math_safe({expression});
}}
struct Model {{
    double a=1.0,b=0.0;
    std::size_t input_width=0;
    static Model fit(const std::vector<std::vector<double>>& input,const std::vector<double>& y) {{
        Model m; m.input_width=expression_width(input);
        if(y.size()<input.size()) throw std::invalid_argument("short target");
        const auto x=sanitize(input);
        {fit}
        return m;
    }}
    std::vector<double> predict(const std::vector<std::vector<double>>& input) const {{
        if(!input.empty() && expression_width(input)!=input_width) throw std::invalid_argument("prediction width");
        std::vector<double> out;
        for(const auto& row:sanitize(input)) {{double value=a*expression_value(row)+b;out.push_back(std::isfinite(value)?clamp_range(value,-60.0,60.0):0.0);}}
        return out;
    }}
}};"#
    ))
}

const EXPRESSION_SHAPE: &str = r#"static std::size_t expression_width(const std::vector<std::vector<double>>& rows) {
    if(rows.empty() || rows.front().empty()) throw std::invalid_argument("empty numeric table");
    const std::size_t width=rows.front().size();
    for(const auto& row:rows) if(row.size()!=width) throw std::invalid_argument("ragged numeric table");
    return width;
}"#;

const PREAMBLE: &str = r#"// Standalone regression model — self-contained, depends only on the C++
// standard library, and accepts all data through its C ABI.
//   Build: g++ -O2 -std=c++17 -ffp-contract=off -fno-fast-math -c model.cpp
//
// All arithmetic uses IEEE-754 double precision in a fixed evaluation order, so
// repeated runs on the same input produce identical results. Build without
// fused-multiply-add contraction (-ffp-contract=off) to keep each multiply and
// add separately rounded.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

// Clamp to [lo, hi] with a fixed branch order; a NaN input passes through.
static double clamp_range(double x, double lo, double hi) {
    if (x < lo) x = lo;
    if (x > hi) x = hi;
    return x;
}

// Larger of two values, ignoring NaN.
static double max_num(double a, double b) {
    if (std::isnan(a)) return b;
    if (std::isnan(b)) return a;
    if (a < b) return b;
    return a;
}

// IEEE-754 total order, so sorts and score-ranking are deterministic (matches the
// reference tie/sign handling). Unused by pipelines with no ordering stage.
[[maybe_unused]] static bool total_less(double a, double b) {
    std::uint64_t ua;
    std::uint64_t ub;
    std::memcpy(&ua, &a, sizeof(double));
    std::memcpy(&ub, &b, sizeof(double));
    std::int64_t la = static_cast<std::int64_t>(ua);
    std::int64_t lb = static_cast<std::int64_t>(ub);
    la ^= static_cast<std::int64_t>(static_cast<std::uint64_t>(la >> 63) >> 1);
    lb ^= static_cast<std::int64_t>(static_cast<std::uint64_t>(lb >> 63) >> 1);
    return la < lb;
}"#;

const H_SANITIZE: &str = r#"// Replace non-finite values with 0.0 in every row.
static std::vector<std::vector<double>> sanitize(const std::vector<std::vector<double>>& x) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> r;
        r.reserve(row.size());
        for (double v : row) r.push_back(std::isfinite(v) ? v : 0.0);
        out.push_back(std::move(r));
    }
    return out;
}"#;

const H_FINISH_STAGE: &str = r#"// Stage epilogue: optionally keep the inputs, cap width at 512, zero non-finite cells.
static std::vector<std::vector<double>> finish_stage(const std::vector<std::vector<double>>& x, std::vector<std::vector<double>> out, bool append) {
    if (append) {
        for (std::size_t i = 0; i < out.size(); ++i) {
            std::vector<double> merged = x[i];
            merged.insert(merged.end(), out[i].begin(), out[i].end());
            out[i] = std::move(merged);
        }
    }
    for (auto& row : out) {
        if (row.size() > 512) row.resize(512);
        for (double& v : row) {
            if (!std::isfinite(v)) v = 0.0;
        }
        if (row.empty()) row.push_back(0.0);
    }
    return out;
}"#;

const H_STD_FIT: &str = r#"// Per-column mean and standard deviation (1/n convention), floored at 1e-9.
static void std_fit(const std::vector<std::vector<double>>& x, std::vector<double>& means, std::vector<double>& scales) {
    double n = static_cast<double>(std::max<std::size_t>(x.size(), 1));
    std::size_t p = x[0].size();
    means.assign(p, 0.0);
    scales.assign(p, 1.0);
    for (std::size_t j = 0; j < p; ++j) {
        double s = 0.0;
        for (const auto& row : x) s += row[j];
        means[j] = s / n;
        double v = 0.0;
        for (const auto& row : x) {
            double d = row[j] - means[j];
            v += d * d;
        }
        scales[j] = max_num(std::sqrt(v / n), 1e-9);
    }
}"#;

const H_CHOLESKY: &str = r#"// Solve the symmetric positive-definite system A w = b (A row-major, overwritten
// in place as L L^T, then forward and backward triangular solves).
static std::vector<double> cholesky_solve(std::vector<double>& a, const std::vector<double>& b, std::size_t p) {
    for (std::size_t i = 0; i < p; ++i) {
        for (std::size_t j = 0; j < i + 1; ++j) {
            double s = a[i * p + j];
            for (std::size_t k = 0; k < j; ++k) s -= a[i * p + k] * a[j * p + k];
            if (i == j) {
                a[i * p + j] = (s > 1e-12) ? std::sqrt(s) : 1e-6;
            } else {
                a[i * p + j] = s / a[j * p + j];
            }
        }
    }
    std::vector<double> z(p, 0.0);
    for (std::size_t i = 0; i < p; ++i) {
        double s = b[i];
        for (std::size_t k = 0; k < i; ++k) s -= a[i * p + k] * z[k];
        z[i] = s / a[i * p + i];
    }
    std::vector<double> w(p, 0.0);
    for (std::size_t ii = 0; ii < p; ++ii) {
        std::size_t i = p - 1 - ii;
        double s = z[i];
        for (std::size_t k = i + 1; k < p; ++k) s -= a[k * p + i] * w[k];
        w[i] = s / a[i * p + i];
    }
    return w;
}"#;

const H_SORTED_COLUMN: &str = r#"// Column j of x in ascending order (IEEE total order via total_less).
static std::vector<double> sorted_column(const std::vector<std::vector<double>>& x, std::size_t j) {
    std::vector<double> v;
    v.reserve(x.size());
    for (const auto& row : x) v.push_back(row[j]);
    std::sort(v.begin(), v.end(), total_less);
    return v;
}"#;

const H_QUANTILE: &str = r#"// Linearly interpolated quantile of an ascending slice (the numpy 'linear' rule).
static double quantile(const std::vector<double>& sorted_values, double q) {
    if (sorted_values.empty()) return 0.0;
    double pos = q * static_cast<double>(sorted_values.size() - 1);
    std::size_t lo = static_cast<std::size_t>(pos);
    std::size_t hi = std::min(lo + 1, sorted_values.size() - 1);
    double frac = pos - static_cast<double>(lo);
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac;
}"#;

const H_LOWER_BOUND: &str = r#"// Count of elements strictly less than v (hand-rolled binary search).
static std::size_t lower_bound(const std::vector<double>& sorted_values, double v) {
    std::size_t lo = 0;
    std::size_t hi = sorted_values.size();
    while (lo < hi) {
        std::size_t mid = (lo + hi) / 2;
        if (sorted_values[mid] < v) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}"#;

const H_UPPER_BOUND: &str = r#"// Count of elements less than or equal to v (hand-rolled binary search).
static std::size_t upper_bound(const std::vector<double>& sorted_values, double v) {
    std::size_t lo = 0;
    std::size_t hi = sorted_values.size();
    while (lo < hi) {
        std::size_t mid = (lo + hi) / 2;
        if (sorted_values[mid] <= v) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}"#;

const H_MID_RANK: &str = r#"// Average of the strict and non-strict ranks of v, scaled to [0, 1].
static double mid_rank(const std::vector<double>& sorted_values, double v) {
    std::size_t lb = lower_bound(sorted_values, v);
    std::size_t ub = upper_bound(sorted_values, v);
    return static_cast<double>(lb + ub) / (2.0 * static_cast<double>(sorted_values.size()));
}"#;

const H_PROBIT: &str = r#"// Standard normal quantile function via Acklam's rational approximation.
static double probit(double p) {
    const double a[6] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
        1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
    const double b[5] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
        6.680131188771972e+01, -1.328068155288572e+01};
    const double c[6] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
        -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
    const double d[4] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
        3.754408661907416e+00};
    p = clamp_range(p, 1e-12, 1.0 - 1e-12);
    if (p < 0.02425) {
        double q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if (p <= 1.0 - 0.02425) {
        double q = p - 0.5;
        double r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    } else {
        double q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
}"#;

const H_MATH_SIGMOID: &str = r#"// Logistic function 1/(1+exp(-v)) with the argument clipped to [-30, 30].
static double m_sigmoid(double value) {
    return 1.0 / (1.0 + std::exp(-clamp_range(value, -30.0, 30.0)));
}"#;

const H_PCG32: &str = r#"// 32-bit rotate right (rotate by 0 is identity).
static std::uint32_t rotr32(std::uint32_t x, std::uint32_t r) {
    return (x >> r) | (x << ((32 - r) & 31));
}

// Permuted congruential generator, XSH-RR 64/32 (O'Neill, 2014), with the
// uniform/range/normal draws layered on top. Mirrors the reference RNG bit-for-bit.
struct Pcg32 {
    std::uint64_t state;
    std::uint64_t inc;

    std::uint32_t next_u32() {
        std::uint64_t old = state;
        state = old * 6364136223846793005ULL + inc;
        std::uint32_t xorshifted = static_cast<std::uint32_t>((((old >> 18) ^ old) >> 27));
        std::uint32_t rot = static_cast<std::uint32_t>(old >> 59);
        return rotr32(xorshifted, rot);
    }

    // Uniform in [0, 1) from 53 random bits: ((hi<<32|lo) >> 11) * 2^-53.
    double next_f64() {
        std::uint64_t hi = static_cast<std::uint64_t>(next_u32()) << 32;
        std::uint64_t lo = static_cast<std::uint64_t>(next_u32());
        return static_cast<double>((hi | lo) >> 11) * (1.0 / 9007199254740992.0);
    }

    double range_f64(double a, double b) {
        return a + (b - a) * next_f64();
    }

    // Standard normal via the Box-Muller cosine branch (u1 floored at 1e-12).
    double gaussian() {
        double u1 = max_num(next_f64(), 1e-12);
        double u2 = next_f64();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.141592653589793 * u2);
    }
};

static Pcg32 pcg32_new(std::uint64_t seed, std::uint64_t stream) {
    Pcg32 rng;
    rng.state = 0;
    rng.inc = (stream << 1) | 1ULL;
    rng.next_u32();
    rng.state = rng.state + seed;
    rng.next_u32();
    return rng;
}"#;

const H_STAGE_RNG: &str = r#"// Independent random stream keyed by a stage's input and output widths.
static Pcg32 stage_rng(std::uint64_t seed, std::size_t in_cols, std::size_t n_out) {
    return pcg32_new(seed, (static_cast<std::uint64_t>(in_cols) << 32) ^ static_cast<std::uint64_t>(n_out));
}"#;

const H_DOT: &str = r#"// Inner product over the shorter of the two vectors.
static double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double total = 0.0;
    std::size_t m = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < m; ++i) total += a[i] * b[i];
    return total;
}"#;

const OP_STANDARDIZE: &str = r#"struct StandardScaler {
    std::vector<double> means;
    std::vector<double> scales;
};

// Centers each column by its training mean and divides by its standard deviation.
static StandardScaler standard_scaler_fit(const std::vector<std::vector<double>>& x) {
    StandardScaler state;
    std_fit(x, state.means, state.scales);
    return state;
}

static std::vector<std::vector<double>> standard_scaler_apply(const StandardScaler& state, const std::vector<std::vector<double>>& x, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> r;
        r.reserve(row.size());
        for (std::size_t j = 0; j < row.size(); ++j) {
            r.push_back(clamp_range((row[j] - state.means[j]) / state.scales[j], -20.0, 20.0));
        }
        out.push_back(std::move(r));
    }
    return finish_stage(x, std::move(out), append);
}"#;

const OP_ROBUST_SCALE: &str = r#"struct RobustScaler {
    std::vector<double> medians;
    std::vector<double> scales;
};

// Centers each column by its median and divides by its interquartile range.
static RobustScaler robust_scaler_fit(const std::vector<std::vector<double>>& x) {
    RobustScaler state;
    for (std::size_t j = 0; j < x[0].size(); ++j) {
        std::vector<double> s = sorted_column(x, j);
        state.medians.push_back(quantile(s, 0.5));
        double iqr = quantile(s, 0.75) - quantile(s, 0.25);
        state.scales.push_back(max_num(iqr, 1e-9));
    }
    return state;
}

static std::vector<std::vector<double>> robust_scaler_apply(const RobustScaler& state, const std::vector<std::vector<double>>& x, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> r;
        r.reserve(row.size());
        for (std::size_t j = 0; j < row.size(); ++j) {
            r.push_back((row[j] - state.medians[j]) / state.scales[j]);
        }
        out.push_back(std::move(r));
    }
    return finish_stage(x, std::move(out), append);
}"#;

const OP_RANK: &str = r#"struct MidRankTransform {
    std::vector<std::vector<double>> sorted_cols;
};

// Maps each value to its mid-rank within its training column, in [0, 1].
static MidRankTransform mid_rank_transform_fit(const std::vector<std::vector<double>>& x) {
    MidRankTransform state;
    for (std::size_t j = 0; j < x[0].size(); ++j) state.sorted_cols.push_back(sorted_column(x, j));
    return state;
}

static std::vector<std::vector<double>> mid_rank_transform_apply(const MidRankTransform& state, const std::vector<std::vector<double>>& x, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> r;
        r.reserve(row.size());
        for (std::size_t j = 0; j < row.size(); ++j) r.push_back(mid_rank(state.sorted_cols[j], row[j]));
        out.push_back(std::move(r));
    }
    return finish_stage(x, std::move(out), append);
}"#;

const OP_BLOM: &str = r#"struct BlomNormalScores {
    std::vector<std::vector<double>> sorted_cols;
};

// Maps each value to a normal quantile of its mid-rank (Blom, 1958).
static BlomNormalScores blom_normal_scores_fit(const std::vector<std::vector<double>>& x) {
    BlomNormalScores state;
    for (std::size_t j = 0; j < x[0].size(); ++j) state.sorted_cols.push_back(sorted_column(x, j));
    return state;
}

static std::vector<std::vector<double>> blom_normal_scores_apply(const BlomNormalScores& state, const std::vector<std::vector<double>>& x, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> values;
        values.reserve(row.size());
        for (std::size_t j = 0; j < row.size(); ++j) {
            double n = static_cast<double>(state.sorted_cols[j].size());
            double mid = mid_rank(state.sorted_cols[j], row[j]) * n;
            values.push_back(probit((mid + 0.125) / (n + 0.25)));
        }
        out.push_back(std::move(values));
    }
    return finish_stage(x, std::move(out), append);
}"#;

const OP_SIGMOID: &str = r#"// Applies the logistic squashing function to every value.
static std::vector<std::vector<double>> logistic_squash_apply(const std::vector<std::vector<double>>& x, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> r;
        r.reserve(row.size());
        for (double v : row) r.push_back(m_sigmoid(v));
        out.push_back(std::move(r));
    }
    return finish_stage(x, std::move(out), append);
}"#;

const OP_STEP: &str = r#"// Per-column hard indicator: 1.0 where the value exceeds the threshold, else 0.0.
static std::vector<std::vector<double>> step_indicator_apply(const std::vector<std::vector<double>>& x, double t, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> r;
        r.reserve(row.size());
        for (double v : row) r.push_back(v > t ? 1.0 : 0.0);
        out.push_back(std::move(r));
    }
    return finish_stage(x, std::move(out), append);
}"#;

const OP_WINSOR: &str = r#"struct WinsorLimiter {
    std::vector<double> lo;
    std::vector<double> hi;
};

// Clamps each column to its lower/upper training quantiles.
static WinsorLimiter winsor_limiter_fit(const std::vector<std::vector<double>>& x, double q) {
    WinsorLimiter state;
    for (std::size_t j = 0; j < x[0].size(); ++j) {
        std::vector<double> s = sorted_column(x, j);
        double low = quantile(s, q);
        double high = max_num(quantile(s, 1.0 - q), low);
        state.lo.push_back(low);
        state.hi.push_back(high);
    }
    return state;
}

static std::vector<std::vector<double>> winsor_limiter_apply(const WinsorLimiter& state, const std::vector<std::vector<double>>& x, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> r;
        r.reserve(row.size());
        for (std::size_t j = 0; j < row.size(); ++j) r.push_back(clamp_range(row[j], state.lo[j], state.hi[j]));
        out.push_back(std::move(r));
    }
    return finish_stage(x, std::move(out), append);
}"#;

const OP_NORM_ROW: &str = r#"// Scales each row to unit L2 norm.
static std::vector<std::vector<double>> unit_norm_rows_apply(const std::vector<std::vector<double>>& x, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        double s = 0.0;
        for (double v : row) s += v * v;
        double inv = 1.0 / max_num(std::sqrt(s), 1e-9);
        std::vector<double> r;
        r.reserve(row.size());
        for (double v : row) r.push_back(v * inv);
        out.push_back(std::move(r));
    }
    return finish_stage(x, std::move(out), append);
}"#;

const OP_RFF_COS: &str = r#"struct RandomFourierCosineFeatures {
    std::vector<std::vector<double>> w;
    std::vector<double> b;
};

// Random Fourier cosine features (Rahimi & Recht, 2007): cosines of random projections.
static RandomFourierCosineFeatures random_fourier_cosine_features_fit(const std::vector<std::vector<double>>& x, std::size_t n_features, double gamma, std::uint64_t seed) {
    std::size_t cols = x[0].size();
    Pcg32 rng = stage_rng(seed, cols, n_features);
    double scale = std::sqrt(2.0 * gamma);
    std::vector<std::vector<double>> w;
    w.reserve(n_features);
    for (std::size_t i = 0; i < n_features; ++i) {
        std::vector<double> row;
        row.reserve(cols);
        for (std::size_t c = 0; c < cols; ++c) {
            row.push_back(rng.gaussian() * scale);
        }
        w.push_back(std::move(row));
    }
    std::vector<double> b;
    b.reserve(n_features);
    for (std::size_t i = 0; i < n_features; ++i) {
        b.push_back(rng.range_f64(0.0, 2.0 * 3.141592653589793));
    }
    RandomFourierCosineFeatures state;
    state.w = std::move(w);
    state.b = std::move(b);
    return state;
}

static std::vector<std::vector<double>> random_fourier_cosine_features_apply(const RandomFourierCosineFeatures& state, const std::vector<std::vector<double>>& x, bool append) {
    double scale = std::sqrt(2.0 / static_cast<double>(state.w.size()));
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> values;
        values.reserve(state.w.size());
        for (std::size_t i = 0; i < state.w.size(); ++i) {
            values.push_back(scale * std::cos(dot(state.w[i], row) + state.b[i]));
        }
        out.push_back(std::move(values));
    }
    return finish_stage(x, std::move(out), append);
}"#;

const OP_RFF_SIN: &str = r#"struct RandomFourierSineFeatures {
    std::vector<std::vector<double>> w;
    std::vector<double> b;
};

// Random Fourier sine features (Rahimi & Recht, 2007): sines of random projections.
static RandomFourierSineFeatures random_fourier_sine_features_fit(const std::vector<std::vector<double>>& x, std::size_t n_features, double gamma, std::uint64_t seed) {
    std::size_t cols = x[0].size();
    Pcg32 rng = stage_rng(seed, cols, n_features);
    double scale = std::sqrt(2.0 * gamma);
    std::vector<std::vector<double>> w;
    w.reserve(n_features);
    for (std::size_t i = 0; i < n_features; ++i) {
        std::vector<double> row;
        row.reserve(cols);
        for (std::size_t c = 0; c < cols; ++c) {
            row.push_back(rng.gaussian() * scale);
        }
        w.push_back(std::move(row));
    }
    std::vector<double> b;
    b.reserve(n_features);
    for (std::size_t i = 0; i < n_features; ++i) {
        b.push_back(rng.range_f64(0.0, 2.0 * 3.141592653589793));
    }
    RandomFourierSineFeatures state;
    state.w = std::move(w);
    state.b = std::move(b);
    return state;
}

static std::vector<std::vector<double>> random_fourier_sine_features_apply(const RandomFourierSineFeatures& state, const std::vector<std::vector<double>>& x, bool append) {
    double scale = std::sqrt(2.0 / static_cast<double>(state.w.size()));
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> values;
        values.reserve(state.w.size());
        for (std::size_t i = 0; i < state.w.size(); ++i) {
            values.push_back(scale * std::sin(dot(state.w[i], row) + state.b[i]));
        }
        out.push_back(std::move(values));
    }
    return finish_stage(x, std::move(out), append);
}"#;

const H_QUANTILE_EDGES: &str = r#"// Interior quantile cut points (1/bins .. (bins-1)/bins) for every column.
static std::vector<std::vector<double>> quantile_edges(const std::vector<std::vector<double>>& x, std::size_t bins) {
    std::vector<std::vector<double>> edges;
    for (std::size_t j = 0; j < x[0].size(); ++j) {
        std::vector<double> s = sorted_column(x, j);
        std::vector<double> col;
        for (std::size_t b = 1; b < bins; ++b) {
            col.push_back(quantile(s, static_cast<double>(b) / static_cast<double>(bins)));
        }
        edges.push_back(std::move(col));
    }
    return edges;
}"#;

const H_PEARSON: &str = r#"// Pearson correlation, 0.0 for degenerate inputs.
static double pearson(const std::vector<double>& a, const std::vector<double>& b) {
    std::size_t n = std::min(a.size(), b.size());
    if (n < 2) return 0.0;
    double ma = 0.0;
    for (std::size_t i = 0; i < n; ++i) ma += a[i];
    ma /= static_cast<double>(n);
    double mb = 0.0;
    for (std::size_t i = 0; i < n; ++i) mb += b[i];
    mb /= static_cast<double>(n);
    double sab = 0.0;
    double saa = 0.0;
    double sbb = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double da = a[i] - ma;
        double db = b[i] - mb;
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    if (saa <= 0.0 || sbb <= 0.0) return 0.0;
    return sab / std::sqrt(saa * sbb);
}"#;

const OP_CHEB: &str = r#"// Chebyshev polynomial basis (T_1..T_degree) of each column, mapped to [-1, 1].
static std::vector<std::vector<double>> chebyshev_basis_apply(const std::vector<std::vector<double>>& x, std::size_t degree, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> values;
        for (double v : row) {
            double u = clamp_range(2.0 * v - 1.0, -1.0, 1.0);
            double t_prev = 1.0;
            double t_cur = u;
            for (std::size_t d = 0; d < degree; ++d) {
                values.push_back(t_cur);
                double t_next = 2.0 * u * t_cur - t_prev;
                t_prev = t_cur;
                t_cur = t_next;
            }
        }
        out.push_back(std::move(values));
    }
    return finish_stage(x, std::move(out), append);
}"#;

const OP_HINGE: &str = r#"struct HingeBasis {
    std::vector<std::vector<double>> knots;
};

// Piecewise-linear hinge features at per-column interior quantile knots.
static HingeBasis hinge_basis_fit(const std::vector<std::vector<double>>& x, std::size_t knots) {
    HingeBasis state;
    for (std::size_t j = 0; j < x[0].size(); ++j) {
        std::vector<double> s = sorted_column(x, j);
        std::vector<double> fitted;
        for (std::size_t i = 1; i < knots + 1; ++i) {
            fitted.push_back(quantile(s, static_cast<double>(i) / static_cast<double>(knots + 1)));
        }
        state.knots.push_back(std::move(fitted));
    }
    return state;
}

static std::vector<std::vector<double>> hinge_basis_apply(const HingeBasis& state, const std::vector<std::vector<double>>& x, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> values;
        for (std::size_t j = 0; j < row.size(); ++j) {
            for (double t : state.knots[j]) {
                values.push_back(max_num(row[j] - t, 0.0));
            }
        }
        out.push_back(std::move(values));
    }
    return finish_stage(x, std::move(out), append);
}"#;

const OP_POLY_INTERACT: &str = r#"struct PairwiseProducts {
    std::vector<std::pair<std::size_t, std::size_t>> pairs;
};

// Products of every column with itself and each other column, capped at 512.
static PairwiseProducts pairwise_products_fit(const std::vector<std::vector<double>>& x) {
    std::size_t cols = x[0].size();
    PairwiseProducts state;
    for (std::size_t i = 0; i < cols; ++i) state.pairs.push_back(std::make_pair(i, i));
    bool full = false;
    for (std::size_t i = 0; i < cols; ++i) {
        if (full) break;
        for (std::size_t j = i + 1; j < cols; ++j) {
            if (state.pairs.size() >= 512) {
                full = true;
                break;
            }
            state.pairs.push_back(std::make_pair(i, j));
        }
    }
    return state;
}

static std::vector<std::vector<double>> pairwise_products_apply(const PairwiseProducts& state, const std::vector<std::vector<double>>& x, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> values;
        values.reserve(state.pairs.size());
        for (const auto& pr : state.pairs) values.push_back(row[pr.first] * row[pr.second]);
        out.push_back(std::move(values));
    }
    return finish_stage(x, std::move(out), append);
}"#;

const OP_QUANTILE_BINS: &str = r#"struct QuantileBinIndexer {
    std::vector<std::vector<double>> edges;
};

// Replaces each value with its quantile bin index, scaled to [0, 1].
static QuantileBinIndexer quantile_bin_indexer_fit(const std::vector<std::vector<double>>& x, std::size_t bins) {
    QuantileBinIndexer state;
    state.edges = quantile_edges(x, bins);
    return state;
}

static std::vector<std::vector<double>> quantile_bin_indexer_apply(const QuantileBinIndexer& state, const std::vector<std::vector<double>>& x, std::size_t bins, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> r;
        r.reserve(row.size());
        for (std::size_t j = 0; j < row.size(); ++j) {
            r.push_back(static_cast<double>(lower_bound(state.edges[j], row[j])) / (static_cast<double>(bins) - 1.0));
        }
        out.push_back(std::move(r));
    }
    return finish_stage(x, std::move(out), append);
}"#;

const OP_BIN_ONEHOT: &str = r#"struct QuantileOnehotEncoder {
    std::vector<std::vector<double>> edges;
};

// One-hot encodes each value's quantile bin (bins columns per input column).
static QuantileOnehotEncoder quantile_onehot_encoder_fit(const std::vector<std::vector<double>>& x, std::size_t bins) {
    QuantileOnehotEncoder state;
    state.edges = quantile_edges(x, bins);
    return state;
}

static std::vector<std::vector<double>> quantile_onehot_encoder_apply(const QuantileOnehotEncoder& state, const std::vector<std::vector<double>>& x, std::size_t bins, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> encoded(row.size() * bins, 0.0);
        for (std::size_t j = 0; j < row.size(); ++j) {
            std::size_t b = std::min(lower_bound(state.edges[j], row[j]), bins - 1);
            encoded[j * bins + b] = 1.0;
        }
        out.push_back(std::move(encoded));
    }
    return finish_stage(x, std::move(out), append);
}"#;

const OP_PAIR_PROD: &str = r#"struct CorrelationScreenedProducts {
    std::vector<std::pair<std::size_t, std::size_t>> pairs;
};

// Products of the top-k columns most correlated with the target (by |Pearson|).
static CorrelationScreenedProducts correlation_screened_products_fit(const std::vector<std::vector<double>>& x, const std::vector<double>& y, std::size_t top_k) {
    std::size_t cols = x[0].size();
    top_k = std::min(top_k, cols);
    std::vector<std::pair<std::size_t, double>> scored;
    scored.reserve(cols);
    for (std::size_t j = 0; j < cols; ++j) {
        std::vector<double> col;
        col.reserve(x.size());
        for (const auto& row : x) col.push_back(row[j]);
        scored.push_back(std::make_pair(j, std::fabs(pearson(col, y))));
    }
    std::sort(scored.begin(), scored.end(),
        [](const std::pair<std::size_t, double>& a, const std::pair<std::size_t, double>& b) {
            if (total_less(b.second, a.second)) return true;
            if (total_less(a.second, b.second)) return false;
            return a.first < b.first;
        });
    std::vector<std::size_t> selected;
    selected.reserve(top_k);
    for (std::size_t i = 0; i < top_k; ++i) selected.push_back(scored[i].first);
    std::sort(selected.begin(), selected.end());
    CorrelationScreenedProducts state;
    for (std::size_t a = 0; a < selected.size(); ++a) {
        for (std::size_t b = a + 1; b < selected.size(); ++b) {
            state.pairs.push_back(std::make_pair(selected[a], selected[b]));
        }
    }
    return state;
}

static std::vector<std::vector<double>> correlation_screened_products_apply(const CorrelationScreenedProducts& state, const std::vector<std::vector<double>>& x, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> values;
        values.reserve(state.pairs.size());
        for (const auto& pr : state.pairs) values.push_back(row[pr.first] * row[pr.second]);
        out.push_back(std::move(values));
    }
    return finish_stage(x, std::move(out), append);
}"#;

const H_RBF_ROW: &str = r#"// Gaussian kernel similarities exp(-gamma * ||row - l||^2) to each landmark.
static std::vector<double> rbf_row(const std::vector<std::vector<double>>& landmarks, const std::vector<double>& row, double gamma) {
    std::vector<double> values;
    values.reserve(landmarks.size());
    for (const auto& landmark : landmarks) {
        double d2 = 0.0;
        std::size_t lim = std::min(landmark.size(), row.size());
        for (std::size_t j = 0; j < lim; ++j) {
            double d = row[j] - landmark[j];
            d2 += d * d;
        }
        values.push_back(std::exp(-gamma * d2));
    }
    return values;
}"#;

const OP_RAND_PROJ: &str = r#"struct GaussianRandomProjection {
    std::vector<std::vector<double>> w;
};

// Projects each row onto k random Gaussian directions, scaled by 1/sqrt(k) (Johnson & Lindenstrauss, 1984).
static GaussianRandomProjection gaussian_random_projection_fit(const std::vector<std::vector<double>>& x, std::size_t k, std::uint64_t seed) {
    std::size_t cols = x[0].size();
    Pcg32 rng = stage_rng(seed, cols, k);
    GaussianRandomProjection state;
    state.w.reserve(k);
    for (std::size_t i = 0; i < k; ++i) {
        std::vector<double> row;
        row.reserve(cols);
        for (std::size_t j = 0; j < cols; ++j) {
            row.push_back(rng.gaussian());
        }
        state.w.push_back(std::move(row));
    }
    return state;
}

static std::vector<std::vector<double>> gaussian_random_projection_apply(const GaussianRandomProjection& state, const std::vector<std::vector<double>>& x, bool append) {
    double inv = 1.0 / std::sqrt(static_cast<double>(state.w.size()));
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> values;
        values.reserve(state.w.size());
        for (std::size_t i = 0; i < state.w.size(); ++i) {
            values.push_back(dot(state.w[i], row) * inv);
        }
        out.push_back(std::move(values));
    }
    return finish_stage(x, std::move(out), append);
}"#;

const OP_ROW_STATS: &str = r#"// Per-row summary [mean, std, min, max, median, q25, q75, skewness, excess kurtosis, mean energy].
static std::vector<std::vector<double>> row_summary_statistics_apply(const std::vector<std::vector<double>>& x, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        double n = static_cast<double>(row.size());
        double mean = 0.0;
        for (double v : row) mean += v;
        mean /= n;
        double m2 = 0.0;
        double m3 = 0.0;
        double m4 = 0.0;
        double energy = 0.0;
        double mn = std::numeric_limits<double>::infinity();
        double mx = -std::numeric_limits<double>::infinity();
        for (double v : row) {
            double d = v - mean;
            m2 += d * d;
            m3 += d * d * d;
            m4 += d * d * d * d;
            energy += v * v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        m2 /= n;
        m3 /= n;
        m4 /= n;
        energy /= n;
        double sd = std::sqrt(m2);
        double skew = (m2 > 1e-24) ? m3 / (m2 * sd) : 0.0;
        double kurt = (m2 > 1e-24) ? m4 / (m2 * m2) - 3.0 : 0.0;
        std::vector<double> s = row;
        std::sort(s.begin(), s.end(), total_less);
        out.push_back({mean, sd, mn, mx, quantile(s, 0.5), quantile(s, 0.25), quantile(s, 0.75), skew, kurt, energy});
    }
    return finish_stage(x, std::move(out), append);
}"#;

const OP_ROW_DIFF_STATS: &str = r#"// Per-row first-difference summary [mean, std, min, max, mean |diff|, sign-change rate, lag-1 autocorrelation, mean energy].
static std::vector<std::vector<double>> row_difference_statistics_apply(const std::vector<std::vector<double>>& x, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::size_t d = row.size();
        if (d < 2) {
            out.push_back(std::vector<double>(8, 0.0));
            continue;
        }
        std::size_t nd = d - 1;
        std::vector<double> diffs;
        diffs.reserve(nd);
        for (std::size_t j = 0; j < nd; ++j) diffs.push_back(row[j + 1] - row[j]);
        double mean = 0.0;
        for (double v : diffs) mean += v;
        mean /= static_cast<double>(nd);
        double m2 = 0.0;
        double mean_abs = 0.0;
        double energy = 0.0;
        double mn = std::numeric_limits<double>::infinity();
        double mx = -std::numeric_limits<double>::infinity();
        std::size_t signs = 0;
        for (std::size_t j = 0; j < nd; ++j) {
            double v = diffs[j];
            double dd = v - mean;
            m2 += dd * dd;
            mean_abs += std::fabs(v);
            energy += v * v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            if (j > 0 && diffs[j - 1] * v < 0.0) signs += 1;
        }
        double sd = std::sqrt(m2 / static_cast<double>(nd));
        std::vector<double> a(row.begin(), row.begin() + nd);
        std::vector<double> b(row.begin() + 1, row.end());
        double autoc = pearson(a, b);
        out.push_back({mean, sd, mn, mx, mean_abs / static_cast<double>(nd),
            static_cast<double>(signs) / static_cast<double>(nd), autoc, energy / static_cast<double>(nd)});
    }
    return finish_stage(x, std::move(out), append);
}"#;

const OP_OBLIQUE_HINGE: &str = r#"struct ObliqueHingeFeatures {
    std::vector<std::vector<double>> w;
    std::vector<double> t;
};

// Rectified random projections thresholded at each direction's median training projection.
static ObliqueHingeFeatures oblique_hinge_features_fit(const std::vector<std::vector<double>>& x, std::size_t n_features, std::uint64_t seed) {
    std::size_t cols = x[0].size();
    Pcg32 rng = stage_rng(seed, cols, n_features);
    double inv = 1.0 / std::sqrt(static_cast<double>(cols));
    ObliqueHingeFeatures state;
    state.w.reserve(n_features);
    for (std::size_t i = 0; i < n_features; ++i) {
        std::vector<double> row;
        row.reserve(cols);
        for (std::size_t j = 0; j < cols; ++j) {
            row.push_back(rng.gaussian() * inv);
        }
        state.w.push_back(std::move(row));
    }
    for (const auto& wi : state.w) {
        std::vector<double> proj;
        proj.reserve(x.size());
        for (const auto& row : x) proj.push_back(dot(wi, row));
        std::sort(proj.begin(), proj.end(), total_less);
        state.t.push_back(quantile(proj, 0.5));
    }
    return state;
}

static std::vector<std::vector<double>> oblique_hinge_features_apply(const ObliqueHingeFeatures& state, const std::vector<std::vector<double>>& x, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> values;
        values.reserve(state.w.size());
        for (std::size_t i = 0; i < state.w.size(); ++i) {
            values.push_back(max_num(dot(state.w[i], row) - state.t[i], 0.0));
        }
        out.push_back(std::move(values));
    }
    return finish_stage(x, std::move(out), append);
}"#;

const OP_LANDMARK_DIST: &str = r#"struct GaussianLandmarkFeatures {
    std::vector<std::vector<double>> landmarks;
};

// Gaussian kernel similarity to k fixed random landmarks (Williams & Seeger, 2001).
static GaussianLandmarkFeatures gaussian_landmark_features_fit(const std::vector<std::vector<double>>& x, std::size_t k, std::uint64_t seed) {
    std::size_t cols = x[0].size();
    Pcg32 rng = stage_rng(seed, cols, k);
    GaussianLandmarkFeatures state;
    state.landmarks.reserve(k);
    for (std::size_t i = 0; i < k; ++i) {
        std::vector<double> row;
        row.reserve(cols);
        for (std::size_t j = 0; j < cols; ++j) {
            row.push_back(rng.range_f64(0.0, 1.0));
        }
        state.landmarks.push_back(std::move(row));
    }
    return state;
}

static std::vector<std::vector<double>> gaussian_landmark_features_apply(const GaussianLandmarkFeatures& state, const std::vector<std::vector<double>>& x, double gamma, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) out.push_back(rbf_row(state.landmarks, row, gamma));
    return finish_stage(x, std::move(out), append);
}"#;

const OP_TARGET_BIN_ENCODE: &str = r#"struct SmoothedTargetBinEncoder {
    std::vector<std::vector<double>> edges;
    std::vector<std::vector<double>> enc;
};

// Smoothed target (mean-of-y) encoding per quantile bin (Micci-Barreca, 2001).
static SmoothedTargetBinEncoder smoothed_target_bin_encoder_fit(const std::vector<std::vector<double>>& x, const std::vector<double>& y, std::size_t bins, double prior) {
    SmoothedTargetBinEncoder state;
    state.edges = quantile_edges(x, bins);
    double y_mean = 0.0;
    for (double v : y) y_mean += v;
    y_mean /= static_cast<double>(std::max<std::size_t>(y.size(), 1));
    for (std::size_t j = 0; j < x[0].size(); ++j) {
        std::vector<double> sums(bins, 0.0);
        std::vector<double> counts(bins, 0.0);
        for (std::size_t r = 0; r < x.size(); ++r) {
            std::size_t b = std::min(lower_bound(state.edges[j], x[r][j]), bins - 1);
            sums[b] += y[r];
            counts[b] += 1.0;
        }
        std::vector<double> col;
        col.reserve(bins);
        for (std::size_t b = 0; b < bins; ++b) {
            col.push_back((sums[b] + prior * y_mean) / max_num(counts[b] + prior, 1e-9));
        }
        state.enc.push_back(std::move(col));
    }
    return state;
}

static std::vector<std::vector<double>> smoothed_target_bin_encoder_apply(const SmoothedTargetBinEncoder& state, const std::vector<std::vector<double>>& x, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> values;
        values.reserve(row.size());
        for (std::size_t j = 0; j < row.size(); ++j) {
            std::size_t bins = state.enc[j].size();
            values.push_back(state.enc[j][std::min(lower_bound(state.edges[j], row[j]), bins - 1)]);
        }
        out.push_back(std::move(values));
    }
    return finish_stage(x, std::move(out), append);
}"#;

const H_SOFT_THRESHOLD: &str = r#"// Shrink v toward zero by t (the lasso proximal step).
static double soft_threshold(double v, double t) {
    if (v > t) return v - t;
    if (v < -t) return v + t;
    return 0.0;
}"#;

const HEAD_ELASTICNET: &str = r#"struct ElasticNet {
    std::vector<double> means;
    std::vector<double> scales;
    double w0;
    std::vector<double> w;
};

// Elastic net regression (Zou & Hastie, 2005) via cyclic coordinate descent with
// soft-thresholding (Friedman, Hastie & Tibshirani, 2010); predictions clipped to [-60, 60].
static ElasticNet elastic_net_fit(const std::vector<std::vector<double>>& x, const std::vector<double>& y, double alpha, double l1_ratio, std::size_t sweeps) {
    std::size_t n = std::min(x.size(), y.size());
    std::size_t p = x[0].size();
    std::vector<double> means;
    std::vector<double> scales;
    std_fit(x, means, scales);
    std::vector<std::vector<double>> xs;
    xs.reserve(n);
    for (std::size_t r = 0; r < n; ++r) {
        const auto& row = x[r];
        std::vector<double> sr;
        sr.reserve(p);
        for (std::size_t j = 0; j < p; ++j) {
            sr.push_back(clamp_range((row[j] - means[j]) / scales[j], -20.0, 20.0));
        }
        xs.push_back(std::move(sr));
    }
    double y_mean = 0.0;
    for (std::size_t i = 0; i < n; ++i) y_mean += y[i];
    y_mean /= static_cast<double>(std::max<std::size_t>(n, 1));
    double l1 = alpha * l1_ratio;
    double l2 = alpha * (1.0 - l1_ratio);
    std::vector<double> col_sq(p, 0.0);
    for (const auto& row : xs) {
        for (std::size_t j = 0; j < p; ++j) col_sq[j] += row[j] * row[j];
    }
    std::vector<double> w(p, 0.0);
    std::vector<double> resid(n, 0.0);
    for (std::size_t r = 0; r < n; ++r) resid[r] = y[r] - y_mean;
    std::size_t max_sweeps = std::max<std::size_t>(sweeps, 1);
    for (std::size_t sweep = 0; sweep < max_sweeps; ++sweep) {
        double max_delta = 0.0;
        for (std::size_t j = 0; j < p; ++j) {
            if (col_sq[j] <= 1e-12) continue;
            double rho = 0.0;
            for (std::size_t r = 0; r < n; ++r) rho += xs[r][j] * resid[r];
            rho = rho / static_cast<double>(n) + w[j] * col_sq[j] / static_cast<double>(n);
            double denom = col_sq[j] / static_cast<double>(n) + l2;
            double new_w = soft_threshold(rho, l1) / denom;
            double delta = new_w - w[j];
            if (delta != 0.0) {
                for (std::size_t r = 0; r < n; ++r) resid[r] -= delta * xs[r][j];
                w[j] = new_w;
                double ad = std::fabs(delta);
                if (ad > max_delta) max_delta = ad;
            }
        }
        if (max_delta < 1e-9) break;
    }
    ElasticNet state;
    state.means = std::move(means);
    state.scales = std::move(scales);
    state.w0 = y_mean;
    state.w = std::move(w);
    return state;
}

static std::vector<double> elastic_net_predict(const ElasticNet& state, const std::vector<std::vector<double>>& x) {
    std::size_t p = state.means.size();
    std::vector<double> predictions;
    predictions.reserve(x.size());
    for (const auto& row : x) {
        double z = state.w0;
        std::size_t lim = std::min(p, row.size());
        for (std::size_t j = 0; j < lim; ++j) {
            double xs = clamp_range((row[j] - state.means[j]) / state.scales[j], -20.0, 20.0);
            z += state.w[j] * xs;
        }
        predictions.push_back(clamp_range(z, -60.0, 60.0));
    }
    return predictions;
}"#;

const HEAD_RIDGE: &str = r#"struct RidgeRegression {
    std::vector<double> means;
    std::vector<double> scales;
    std::vector<double> w;
};

// Ridge regression on standardized features via the Cholesky normal equations.
static RidgeRegression ridge_regression_fit(const std::vector<std::vector<double>>& x, const std::vector<double>& y, double lambda_frac) {
    std::size_t n = std::min(x.size(), y.size());
    std::size_t p = x[0].size();
    std::vector<double> means;
    std::vector<double> scales;
    std_fit(x, means, scales);
    double lam = max_num(lambda_frac * static_cast<double>(n), 1e-3);
    double y_mean = 0.0;
    for (std::size_t i = 0; i < n; ++i) y_mean += y[i];
    y_mean /= static_cast<double>(std::max<std::size_t>(n, 1));
    std::vector<double> ata(p * p, 0.0);
    std::vector<double> atb(p, 0.0);
    std::vector<double> xs_row(p, 0.0);
    for (std::size_t r = 0; r < n; ++r) {
        const auto& row = x[r];
        for (std::size_t j = 0; j < p; ++j) {
            xs_row[j] = clamp_range((row[j] - means[j]) / scales[j], -20.0, 20.0);
        }
        double yr = y[r] - y_mean;
        for (std::size_t i = 0; i < p; ++i) {
            double xi = xs_row[i];
            atb[i] += xi * yr;
            for (std::size_t j = i; j < p; ++j) {
                ata[i * p + j] += xi * xs_row[j];
            }
        }
    }
    for (std::size_t i = 0; i < p; ++i) {
        ata[i * p + i] += lam;
        for (std::size_t j = 0; j < i; ++j) {
            ata[i * p + j] = ata[j * p + i];
        }
    }
    std::vector<double> coeffs = cholesky_solve(ata, atb, p);
    std::vector<double> w;
    w.reserve(p + 1);
    w.push_back(y_mean);
    for (double c : coeffs) w.push_back(c);
    RidgeRegression state;
    state.means = std::move(means);
    state.scales = std::move(scales);
    state.w = std::move(w);
    return state;
}

static std::vector<double> ridge_regression_predict(const RidgeRegression& state, const std::vector<std::vector<double>>& x) {
    std::size_t p = state.means.size();
    std::vector<double> predictions;
    predictions.reserve(x.size());
    for (const auto& row : x) {
        double z = state.w[0];
        std::size_t lim = std::min(p, row.size());
        for (std::size_t j = 0; j < lim; ++j) {
            double xs = clamp_range((row[j] - state.means[j]) / state.scales[j], -20.0, 20.0);
            z += state.w[j + 1] * xs;
        }
        predictions.push_back(clamp_range(z, -60.0, 60.0));
    }
    return predictions;
}"#;

// Verified portable-scalar-kernel-v1 C++ implementation. This is the same
// defined-arithmetic body exercised by tests/portable/full_ref.cpp; the
// generated source contains no fixture harness or external access.
const PORTABLE_WARNING_BEGIN: &str = r#"#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif"#;

const PORTABLE_WARNING_END: &str = r#"#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif"#;

const PORTABLE_MATH: &str = r####"static inline uint64_t to_bits(double x){ uint64_t b; std::memcpy(&b,&x,8); return b; }
static inline double from_bits(uint64_t b){ double x; std::memcpy(&x,&b,8); return x; }
static inline uint32_t high_word(double x){ return (uint32_t)(to_bits(x) >> 32); }
static inline uint32_t low_word(double x){ return (uint32_t)to_bits(x); }
static inline double with_high_word(double x, uint32_t hi){ return from_bits(((uint64_t)hi<<32)|(to_bits(x)&0xffffffffULL)); }
static inline double with_low_word(double x, uint32_t lo){ return from_bits((to_bits(x)&0xffffffff00000000ULL)|(uint64_t)lo); }
static inline double fabs_(double x){ return from_bits(to_bits(x)&0x7fffffffffffffffULL); }
static inline double canonical_nan(){ return from_bits(0x7ff8000000000000ULL); }
static inline double two_pow(int32_t n){ return from_bits((uint64_t)((int64_t)(0x3ff+n))<<52); }

static double p_scalbn(double x, int32_t n){
    double p1023=two_pow(1023), pm1022=two_pow(-1022), p53=two_pow(53); double y=x;
    if (n>1023){ y*=p1023; n-=1023; if(n>1023){ y*=p1023; n-=1023; if(n>1023) n=1023; } }
    else if (n<-1022){ y*=pm1022*p53; n+=1022-53; if(n<-1022){ y*=pm1022*p53; n+=1022-53; if(n<-1022) n=-1022; } }
    return y*two_pow(n);
}

static double xexp(double x){
    if (x!=x) return canonical_nan();
    const double HALF[2]={0.5,-0.5};
    const double LN2HI[2]={6.9314718036912381649e-1,-6.9314718036912381649e-1};
    const double LN2LO[2]={1.9082149292705877000e-10,-1.9082149292705877000e-10};
    const double INVLN2=1.4426950408889633870;
    const double P1=1.66666666666666019037e-1,P2=-2.77777777770155933842e-3,P3=6.61375632143793436117e-5,P4=-1.65339022054652515390e-6,P5=4.13813679705723846039e-8;
    const double HUGE=1.0e300,TWOM1000=9.33263618503218878990e-302,O_THRESHOLD=7.09782712893383973096e2,U_THRESHOLD=-7.45133219101941108420e2;
    uint32_t hx=high_word(x); size_t xsb=(hx>>31)&1; hx&=0x7fffffff;
    if (hx>=0x40862E42){
        if (hx>=0x7ff00000){ if(((hx&0xfffff)|low_word(x))!=0) return x+x; return xsb==0?x:0.0; }
        if (x>O_THRESHOLD) return HUGE*HUGE;
        if (x<U_THRESHOLD) return TWOM1000*TWOM1000;
    }
    int32_t k=0; double hi=0.0,lo=0.0,xr;
    if (hx>0x3fd62e42){
        if (hx<0x3FF0A2B2){ hi=x-LN2HI[xsb]; lo=LN2LO[xsb]; k=1-(int32_t)xsb-(int32_t)xsb; }
        else { k=(int32_t)(INVLN2*x+HALF[xsb]); double t=(double)k; hi=x-t*LN2HI[0]; lo=t*LN2LO[0]; }
        xr=hi-lo;
    } else if (hx<0x3e300000){ return 1.0+x; } else { xr=x; }
    double t=xr*xr; double c=xr-t*(P1+t*(P2+t*(P3+t*(P4+t*P5))));
    if (k==0) return 1.0-((xr*c)/(c-2.0)-xr);
    double y=1.0-((lo-(xr*c)/(2.0-c))-hi);
    return p_scalbn(y,k);
}

static double xln(double x){
    if (x!=x) return canonical_nan();
    const double LN2_HI=6.9314718036912381649e-1,LN2_LO=1.9082149292705877000e-10,TWO54=1.80143985094819840000e16;
    const double LG1=6.666666666666735130e-1,LG2=3.999999999940941908e-1,LG3=2.857142874366239149e-1,LG4=2.222219843214978396e-1,LG5=1.818357216161805012e-1,LG6=1.531383769920937332e-1,LG7=1.479819860511658591e-1;
    int32_t hx=(int32_t)high_word(x); uint32_t lx=low_word(x); int32_t k=0; double xx=x;
    if (hx<0x00100000){
        if (((hx&0x7fffffff)|(int32_t)lx)==0) return -TWO54/0.0;
        if (hx<0) return canonical_nan();
        k-=54; xx*=TWO54; hx=(int32_t)high_word(xx);
    }
    if (hx>=0x7ff00000) return x+x;
    k+=(hx>>20)-1023; hx&=0x000fffff;
    int32_t i=(hx+0x95f64)&0x100000;
    xx=with_high_word(xx,(uint32_t)(hx|(i^0x3ff00000))); k+=i>>20;
    double f=xx-1.0;
    if ((0x000fffff&(2+hx))<3){
        if (f==0.0){ if(k==0) return 0.0; double dk=(double)k; return dk*LN2_HI+dk*LN2_LO; }
        double r=f*f*(0.5-0.33333333333333333*f);
        if (k==0) return f-r;
        double dk=(double)k; return dk*LN2_HI-((r-dk*LN2_LO)-f);
    }
    double s=f/(2.0+f); double dk=(double)k; double z=s*s,w=z*z;
    double t1=w*(LG2+w*(LG4+w*LG6)); double t2=z*(LG1+w*(LG3+w*(LG5+w*LG7))); double r=t2+t1;
    int32_t i2=hx-0x6147a, j2=0x6b851-hx;
    if ((i2|j2)>0){ double hfsq=0.5*f*f; if(k==0) return f-(hfsq-s*(hfsq+r)); return dk*LN2_HI-((hfsq-(s*(hfsq+r)+dk*LN2_LO))-f); }
    else if (k==0) return f-s*(f-r);
    return dk*LN2_HI-((s*(f-r)-dk*LN2_LO)-f);
}

static double xlog1p(double x){
    if (x!=x) return canonical_nan();
    const double LN2_HI=6.9314718036912381649e-1,LN2_LO=1.9082149292705877000e-10;
    const double LG1=6.666666666666735130e-1,LG2=3.999999999940941908e-1,LG3=2.857142874366239149e-1,LG4=2.222219843214978396e-1,LG5=1.818357216161805012e-1,LG6=1.531383769920937332e-1,LG7=1.479819860511658591e-1;
    uint32_t hx=high_word(x); int32_t k=1; double f=0.0,c=0.0;
    if (hx<0x3fda827a || (hx>>31)!=0){
        if (hx>=0xbff00000){ if(x==-1.0) return x/0.0; return canonical_nan(); }
        if ((hx<<1) < (0x3ca00000u<<1)) return x;
        if (hx<=0xbfd2bec4){ k=0; c=0.0; f=x; }
    } else if (hx>=0x7ff00000){ return x; }
    if (k!=0){
        double u=1.0+x; uint32_t hu=high_word(u);
        hu=hu+(0x3ff00000u-0x3fe6a09eu);
        k=(int32_t)(hu>>20)-0x3ff;
        if (k<54){ c=(k>=2)?(1.0-(u-x)):(x-(u-1.0)); c/=u; } else c=0.0;
        hu=(hu&0x000fffff)+0x3fe6a09eu;
        f=with_high_word(u,hu)-1.0;
    }
    double hfsq=0.5*f*f, s=f/(2.0+f), z=s*s, w=z*z;
    double t1=w*(LG2+w*(LG4+w*LG6)); double t2=z*(LG1+w*(LG3+w*(LG5+w*LG7))); double r=t2+t1;
    double dk=(double)k;
    return s*(hfsq+r)+(dk*LN2_LO+c)-hfsq+f+dk*LN2_HI;
}

static double xexpm1(double x){
    if (x!=x) return canonical_nan();
    const double LN2_HI=6.9314718036912381649e-1,LN2_LO=1.90821492927058770002e-10,INVLN2=1.4426950408889633870;
    const double Q1=-3.33333333333331315428e-2,Q2=1.58730158725481460165e-3,Q3=-7.93650757867487942473e-5,Q4=4.00821782732936239552e-6,Q5=-2.01099218183624371326e-7;
    uint32_t hx=high_word(x)&0x7fffffff; bool sign=(high_word(x)>>31)!=0;
    double c=0.0; int32_t k; double xx=x;
    if (hx>=0x4043687A){ if(sign) return -1.0; if(hx>=0x40862E42) return x*two_pow(1023); }
    if (hx>0x3fd62e42){
        double hi,lo;
        if (hx<0x3FF0A2B2){ if(!sign){ hi=x-LN2_HI; lo=LN2_LO; k=1; } else { hi=x+LN2_HI; lo=-LN2_LO; k=-1; } }
        else { k=(int32_t)(INVLN2*x+(sign?-0.5:0.5)); double t=(double)k; hi=x-t*LN2_HI; lo=t*LN2_LO; }
        xx=hi-lo; c=(hi-xx)-lo;
    } else if (hx<0x3c900000){ return x; } else { k=0; }
    double hfx=0.5*xx, hxs=xx*hfx;
    double r1=1.0+hxs*(Q1+hxs*(Q2+hxs*(Q3+hxs*(Q4+hxs*Q5))));
    double t=3.0-r1*hfx; double e=hxs*((r1-t)/(6.0-xx*t));
    if (k==0) return xx-(xx*e-hxs);
    e=xx*(e-c)-c; e-=hxs;
    if (k==-1) return 0.5*(xx-e)-0.5;
    if (k==1){ if(xx<-0.25) return -2.0*(e-(xx+0.5)); return 1.0+2.0*(xx-e); }
    double twopk=two_pow(k);
    if (!(k>=0 && k<=56)){ double y=(xx-e+1.0)*twopk; return y-1.0; }
    double inv=two_pow(-k);
    if (k<20) return (xx-e+(1.0-inv))*twopk;
    return (xx-e-inv+1.0)*twopk;
}

static double xpow(double x, double y){
    const double BP[2]={1.0,1.5}, DP_H[2]={0.0,5.84962487220764160156e-1}, DP_L[2]={0.0,1.35003920212974897128e-8};
    const double TWO53=9007199254740992.0, HUGE=1.0e300, TINY=1.0e-300;
    const double L1=5.99999999999994648725e-1,L2=4.28571428578550184252e-1,L3=3.33333329818377432918e-1,L4=2.72728123808534006489e-1,L5=2.30660745775561366331e-1,L6=2.06975017800338417784e-1;
    const double P1=1.66666666666666019037e-1,P2=-2.77777777770155933842e-3,P3=6.61375632143793436117e-5,P4=-1.65339022054652515390e-6,P5=4.13813679705723846039e-8;
    const double LG2=6.93147180559945286227e-1,LG2_H=6.93147182464599609375e-1,LG2_L=-1.90465429995776804525e-9,OVT=8.0085662595372944372e-17;
    const double CP=9.61796693925975554329e-1,CP_H=9.61796700954437255859e-1,CP_L=-7.02846165095275826516e-9;
    const double IVLN2=1.4426950408889633870,IVLN2_H=1.44269502162933349609e0,IVLN2_L=1.92596299112661746887e-8;
    int32_t hx=(int32_t)high_word(x); uint32_t lx=low_word(x); int32_t hy=(int32_t)high_word(y); uint32_t ly=low_word(y);
    int32_t ix=hx&0x7fffffff, iy=hy&0x7fffffff;
    if ((iy|(int32_t)ly)==0) return 1.0;
    if (x!=x||y!=y) return canonical_nan();
    if (ix>0x7ff00000||(ix==0x7ff00000&&lx!=0)||iy>0x7ff00000||(iy==0x7ff00000&&ly!=0)) return canonical_nan();
    int32_t yisint=0;
    if (hx<0){
        if (iy>=0x43400000) yisint=2;
        else if (iy>=0x3ff00000){ int32_t k=(iy>>20)-0x3ff;
            if (k>20){ uint32_t j=ly>>(52-k); if((j<<(52-k))==ly) yisint=2-(int32_t)(j&1); }
            else if (ly==0){ int32_t j=iy>>(20-k); if((j<<(20-k))==iy) yisint=2-(j&1); } }
    }
    if (ly==0){
        if (iy==0x7ff00000){ if(((ix-0x3ff00000)|(int32_t)lx)==0) return 1.0; else if(ix>=0x3ff00000) return hy>=0?y:0.0; else return hy<0?-y:0.0; }
        if (iy==0x3ff00000) return hy<0?1.0/x:x;
        if (hy==0x40000000) return x*x;
        if (hy==0x3fe00000 && hx>=0) return std::sqrt(x);
    }
    double ax=fabs_(x);
    if (lx==0 && (ix==0x7ff00000||ix==0||ix==0x3ff00000)){
        double z=ax; if(hy<0) z=1.0/z;
        if (hx<0){ if(((ix-0x3ff00000)|yisint)==0) z=canonical_nan(); else if(yisint==1) z=-z; }
        return z;
    }
    int32_t n=(hx>>31)+1;
    if ((n|yisint)==0) return canonical_nan();
    double s=1.0; if((n|(yisint-1))==0) s=-1.0;
    double t1,t2;
    if (iy>0x41e00000){
        if (iy>0x43f00000){ if(ix<=0x3fefffff) return hy<0?HUGE*HUGE:TINY*TINY; if(ix>=0x3ff00000) return hy>0?HUGE*HUGE:TINY*TINY; }
        if (ix<0x3fefffff) return hy<0?s*HUGE*HUGE:s*TINY*TINY;
        if (ix>0x3ff00000) return hy>0?s*HUGE*HUGE:s*TINY*TINY;
        double t=ax-1.0; double w=(t*t)*(0.5-t*(0.333333333333333333333-t*0.25));
        double u=IVLN2_H*t; double v=t*IVLN2_L-w*IVLN2; double t1v=with_low_word(u+v,0); t1=t1v; t2=v-(t1v-u);
    } else {
        n=0;
        if (ix<0x00100000){ ax*=TWO53; n-=53; ix=(int32_t)high_word(ax); }
        n+=(ix>>20)-0x3ff; int32_t j0=ix&0x000fffff; ix=j0|0x3ff00000; int32_t k;
        if (j0<=0x3988E) k=0; else if(j0<0xBB67A) k=1; else { k=0; n+=1; ix-=0x00100000; }
        ax=with_high_word(ax,(uint32_t)ix);
        double u=ax-BP[k]; double v=1.0/(ax+BP[k]); double ss=u*v; double s_h=with_low_word(ss,0);
        double t_h=with_high_word(0.0,(uint32_t)(((ix>>1)|0x20000000)+0x00080000+(k<<18)));
        double t_l=ax-(t_h-BP[k]); double s_l=v*((u-s_h*t_h)-s_h*t_l);
        double s2=ss*ss; double r=s2*s2*(L1+s2*(L2+s2*(L3+s2*(L4+s2*(L5+s2*L6))))); r+=s_l*(s_h+ss);
        double s2b=s_h*s_h; double t_h2=with_low_word(3.0+s2b+r,0); double t_l2=r-((t_h2-3.0)-s2b);
        double u2=s_h*t_h2; double v2=s_l*t_h2+t_l2*ss; double p_h=with_low_word(u2+v2,0); double p_l=v2-(p_h-u2);
        double z_h=CP_H*p_h; double z_l=CP_L*p_h+p_l*CP+DP_L[k];
        double t=(double)n; double t1v=with_low_word(((z_h+z_l)+DP_H[k])+t,0); t1=t1v; t2=z_l-(((t1v-t)-DP_H[k])-z_h);
    }
    double y1=with_low_word(y,0); double p_l=(y-y1)*t1+y*t2; double p_h=y1*t1; double z=p_l+p_h;
    int32_t j=(int32_t)high_word(z); uint32_t i=low_word(z);
    if (j>=0x40900000){ if(((j-0x40900000)|(int32_t)i)!=0) return s*HUGE*HUGE; if(p_l+OVT>z-p_h) return s*HUGE*HUGE; }
    else if ((j&0x7fffffff)>=0x4090cc00){ if(((int32_t)((uint32_t)j-0xc090cc00u)|(int32_t)i)!=0) return s*TINY*TINY; if(p_l<=z-p_h) return s*TINY*TINY; }
    int32_t iabs=j&0x7fffffff; int32_t k=(iabs>>20)-0x3ff; int32_t nn=0;
    if (iabs>0x3fe00000){
        nn=j+(0x00100000>>(k+1)); k=((nn&0x7fffffff)>>20)-0x3ff;
        double t=with_high_word(0.0,(uint32_t)(nn&~(0x000fffff>>k)));
        nn=((nn&0x000fffff)|0x00100000)>>(20-k); if(j<0) nn=-nn; p_h-=t;
    }
    double t=with_low_word(p_l+p_h,0); double u=t*LG2_H; double v=(p_l-(t-p_h))*LG2+t*LG2_L;
    double z2=u+v; double w=v-(z2-u); double tt=z2*z2; double t1b=z2-tt*(P1+tt*(P2+tt*(P3+tt*(P4+tt*P5))));
    double r=(z2*t1b)/(t1b-2.0)-(w+z2*w); z2=1.0-(r-z2);
    j=(int32_t)((uint32_t)high_word(z2)+((uint32_t)nn<<20));
    if ((j>>20)<=0) z2=p_scalbn(z2,nn); else z2=with_high_word(z2,(uint32_t)j);
    return s*z2;
}

static double xtanh(double x){
    if (x!=x) return canonical_nan();
    bool sign=(to_bits(x)>>63)!=0; double ax=from_bits(to_bits(x)&0x7fffffffffffffffULL);
    uint32_t w=(uint32_t)(to_bits(ax)>>32); double t;
    if (w>0x3fe193ea){ if(w>0x40340000){ t=1.0-0.0/ax; } else { double e=xexpm1(2.0*ax); t=1.0-2.0/(e+2.0); } }
    else if (w>0x3fd058ae){ double e=xexpm1(2.0*ax); t=e/(e+2.0); }
    else if (w>=0x00100000){ double e=xexpm1(-2.0*ax); t=-e/(e+2.0); }
    else { t=ax; }
    return sign?-t:t;
}

static double kernel_sin(double x, double y, int32_t iy){
    const double S1=-1.66666666666666324348e-1,S2=8.33333333332248946124e-3,S3=-1.98412698298579493134e-4,S4=2.75573137070700676789e-6,S5=-2.50507602534068634195e-8,S6=1.58969099521155010221e-10;
    double z=x*x,w=z*z; double r=S2+z*(S3+z*S4)+z*w*(S5+z*S6); double v=z*x;
    if (iy==0) return x+v*(S1+z*r);
    return x-((z*(0.5*y-v*r)-y)-v*S1);
}
static double kernel_cos(double x, double y){
    const double C1=4.16666666666666019037e-2,C2=-1.38888888888741095749e-3,C3=2.48015872894767294178e-5,C4=-2.75573143513906633035e-7,C5=2.08757232129817482790e-9,C6=-1.13596475577881948265e-11;
    double z=x*x,w=z*z; double r=z*(C1+z*(C2+z*C3))+w*w*(C4+z*(C5+z*C6)); double hz=0.5*z; double w2=1.0-hz;
    return w2+(((1.0-w2)-hz)+(z*r-x*y));
}
static void rem_pio2(double x, int32_t* nout, double* y0out, double* y1out){
    const double INVPIO2=6.36619772367581382433e-1,PIO2_1=1.57079632673412561417e0,PIO2_1T=6.07710050650619224932e-11,PIO2_2=6.07710050630396597660e-11,PIO2_2T=2.02226624879595063154e-21,PIO2_3=2.02226624871116645580e-21,PIO2_3T=8.47842766036889956997e-32;
    const double INV_TWO_PI=1.59154943091895335769e-1,TWO_PI_HI=6.28318530717958623199e0,TWO_PI_LO=2.44929359829470641471e-16;
    uint32_t ix=high_word(x)&0x7fffffff;
    if (ix>=0x7ff00000){ double z=canonical_nan(); *nout=0; *y0out=z; *y1out=z; return; }
    double xr=x;
    while ((high_word(xr)&0x7fffffff)>=0x413921fb){ double k=std::round(xr*INV_TWO_PI); xr=(xr-k*TWO_PI_HI)-k*TWO_PI_LO; }
    uint32_t ixr=high_word(xr)&0x7fffffff;
    if (ixr<=0x3fe921fb){ *nout=0; *y0out=xr; *y1out=0.0; return; }
    double half=xr>0.0?0.5:-0.5; int32_t n=(int32_t)(xr*INVPIO2+half); double fnn=(double)n;
    double r=xr-fnn*PIO2_1; double w=fnn*PIO2_1T; int32_t j=(int32_t)(ixr>>20);
    double y0=r-w; int32_t hi=(int32_t)(high_word(y0)>>20)&0x7ff; int32_t i=j-hi;
    if (i>16){ double t=r; w=fnn*PIO2_2; r=t-w; w=fnn*PIO2_2T-((t-r)-w); y0=r-w; hi=(int32_t)(high_word(y0)>>20)&0x7ff; i=j-hi;
        if (i>49){ double t2=r; w=fnn*PIO2_3; r=t2-w; w=fnn*PIO2_3T-((t2-r)-w); y0=r-w; } }
    double y1=(r-y0)-w; *nout=n&3; *y0out=y0; *y1out=y1;
}
static double xsin(double x){
    if (x!=x) return canonical_nan();
    uint32_t ix=high_word(x)&0x7fffffff;
    if (ix<=0x3fe921fb){ if(ix<0x3e500000) return x; return kernel_sin(x,0.0,0); }
    if (ix>=0x7ff00000) return canonical_nan();
    int32_t n; double y0,y1; rem_pio2(x,&n,&y0,&y1);
    switch(n){ case 0: return kernel_sin(y0,y1,1); case 1: return kernel_cos(y0,y1); case 2: return -kernel_sin(y0,y1,1); default: return -kernel_cos(y0,y1); }
}
static double xcos(double x){
    if (x!=x) return canonical_nan();
    uint32_t ix=high_word(x)&0x7fffffff;
    if (ix<=0x3fe921fb){ if(ix<0x3e46a09e) return 1.0; return kernel_cos(x,0.0); }
    if (ix>=0x7ff00000) return canonical_nan();
    int32_t n; double y0,y1; rem_pio2(x,&n,&y0,&y1);
    switch(n){ case 0: return kernel_cos(y0,y1); case 1: return -kernel_sin(y0,y1,1); case 2: return -kernel_cos(y0,y1); default: return kernel_sin(y0,y1,1); }
}
static double xatan(double x){
    if (x!=x) return canonical_nan();
    const double ATANHI[4]={4.63647609000806093515e-1,7.85398163397448278999e-1,9.82793723247329054082e-1,1.57079632679489655800e0};
    const double ATANLO[4]={2.26987774529616870924e-17,3.06161699786838301793e-17,1.39033110312309984516e-17,6.12323399573676603587e-17};
    const double AT[11]={3.33333333333329318027e-1,-1.99999999998764832476e-1,1.42857142725034663711e-1,-1.11111104054623557880e-1,9.09088713343650656196e-2,-7.69187620504482999495e-2,6.66107313738753120669e-2,-5.83357013379057348645e-2,4.97687799461593236017e-2,-3.65315727442169155270e-2,1.62858201153657823623e-2};
    int32_t hx=(int32_t)high_word(x); int32_t ix=hx&0x7fffffff;
    if (ix>=0x44100000){ if(ix>0x7ff00000||(ix==0x7ff00000&&low_word(x)!=0)) return canonical_nan(); return hx>0?ATANHI[3]+ATANLO[3]:-ATANHI[3]-ATANLO[3]; }
    int32_t id; double xr=x;
    if (ix<0x3fdc0000){ if(ix<0x3e400000) return x; id=-1; }
    else { xr=fabs_(x);
        if (ix<0x3ff30000){ if(ix<0x3fe60000){ id=0; xr=(2.0*xr-1.0)/(2.0+xr); } else { id=1; xr=(xr-1.0)/(xr+1.0); } }
        else if (ix<0x40038000){ id=2; xr=(xr-1.5)/(1.0+1.5*xr); } else { id=3; xr=-1.0/xr; } }
    double z=xr*xr,w=z*z;
    double s1=z*(AT[0]+w*(AT[2]+w*(AT[4]+w*(AT[6]+w*(AT[8]+w*AT[10])))));
    double s2=w*(AT[1]+w*(AT[3]+w*(AT[5]+w*(AT[7]+w*AT[9]))));
    if (id<0) return xr-xr*(s1+s2);
    double z2=ATANHI[id]-((xr*(s1+s2)-ATANLO[id])-xr);
    return hx<0?-z2:z2;
}
static double xatan2(double y, double x){
    if (x!=x||y!=y) return canonical_nan();
    const double TINY=1.0e-300,PI_O_4=7.85398163397448278990e-1,PI_O_2=1.57079632679489655800e0,PI=3.14159265358979311600e0,PI_LO=1.22464679914735317722e-16;
    int32_t hx=(int32_t)high_word(x); int32_t ix=hx&0x7fffffff; uint32_t lx=low_word(x);
    int32_t hy=(int32_t)high_word(y); int32_t iy=hy&0x7fffffff; uint32_t ly=low_word(y);
    if ((ix|(int32_t)((lx|(0u-lx))>>31))>0x7ff00000 || (iy|(int32_t)((ly|(0u-ly))>>31))>0x7ff00000) return canonical_nan();
    if (hx==0x3ff00000 && lx==0) return xatan(y);
    int32_t m=((hy>>31)&1)|((hx>>30)&2);
    if ((iy|(int32_t)ly)==0){ switch(m){ case 0: case 1: return y; case 2: return PI+TINY; default: return -PI-TINY; } }
    if ((ix|(int32_t)lx)==0) return hy<0?-PI_O_2-TINY:PI_O_2+TINY;
    if (ix==0x7ff00000){
        if (iy==0x7ff00000){ switch(m){ case 0: return PI_O_4+TINY; case 1: return -PI_O_4-TINY; case 2: return 3.0*PI_O_4+TINY; default: return -3.0*PI_O_4-TINY; } }
        switch(m){ case 0: return 0.0; case 1: return -0.0; case 2: return PI+TINY; default: return -PI-TINY; }
    }
    if (iy==0x7ff00000) return hy<0?-PI_O_2-TINY:PI_O_2+TINY;
    int32_t k=(iy-ix)>>20; double z;
    if (k>60) z=PI_O_2+0.5*PI_LO; else if(hx<0&&k<-60) z=0.0; else z=xatan(fabs_(y/x));
    switch(m){ case 0: return z; case 1: return -z; case 2: return PI-(z-PI_LO); default: return (z-PI_LO)-PI; }
}
"####;

#[cfg(test)]
pub(crate) fn portable_math_matches_verified_fixture() -> bool {
    let fixture = include_str!("../tests/portable/full_ref.cpp");
    let Some((prefix, _)) = fixture.split_once("static std::vector<double> grid(){") else {
        return false;
    };
    let Some(start) = prefix.find("static inline uint64_t") else {
        return false;
    };
    PORTABLE_MATH.trim() == prefix[start..].trim()
}

const IO_BLOCK: &str = r#"// ---- caller-owned C ABI ----------------------------------------------------

extern "C" {

struct model_handle {
    Model model;
    std::size_t width;

    model_handle(Model&& fitted, std::size_t fitted_width)
        : model(std::move(fitted)), width(fitted_width) {}
};

std::uint32_t model_abi_version(void) noexcept {
    return 1U;
}

std::size_t model_workspace_bytes(const model_handle*, std::size_t) noexcept {
    return 0U;
}

int model_fit_numeric(
    const double* values,
    std::size_t values_len,
    std::size_t rows,
    std::size_t columns,
    const double* targets,
    std::size_t target_len,
    model_handle** out
) noexcept {
    if (out == nullptr) return 1;
    *out = nullptr;
    if (values == nullptr || targets == nullptr || rows == 0U || columns == 0U) return 1;
    if (target_len != rows) return 1;
    if (rows > std::numeric_limits<std::size_t>::max() / columns) return 1;
    if (values_len < rows * columns) return 1;
    try {
        std::vector<std::vector<double>> x;
        x.reserve(rows);
        std::vector<double> y;
        y.reserve(rows);
        for (std::size_t row = 0; row < rows; ++row) {
            std::vector<double> cells;
            cells.reserve(columns);
            for (std::size_t column = 0; column < columns; ++column) {
                cells.push_back(values[row * columns + column]);
            }
            if (!std::isfinite(targets[row])) return 1;
            x.push_back(std::move(cells));
            y.push_back(targets[row]);
        }
        Model fitted = Model::fit(x, y);
        *out = new model_handle(std::move(fitted), columns);
        return 0;
    } catch (...) {
        return 2;
    }
}

int model_predict_numeric_batch(
    const model_handle* handle,
    const double* values,
    std::size_t values_len,
    std::size_t rows,
    std::size_t columns,
    void* workspace,
    std::size_t workspace_bytes,
    double* out,
    std::size_t out_len
) noexcept {
    (void)workspace;
    (void)workspace_bytes;
    if (handle == nullptr || columns == 0U || columns != handle->width) return 1;
    if (rows == 0U) return values_len == 0U ? 0 : 1;
    if (rows > std::numeric_limits<std::size_t>::max() / columns) return 1;
    if (values == nullptr || values_len < rows * columns || out == nullptr || out_len < rows) return 1;
    try {
        std::vector<std::vector<double>> x;
        x.reserve(rows);
        for (std::size_t row = 0; row < rows; ++row) {
            std::vector<double> cells;
            cells.reserve(columns);
            for (std::size_t column = 0; column < columns; ++column) {
                cells.push_back(values[row * columns + column]);
            }
            x.push_back(std::move(cells));
        }
        std::vector<double> predictions = handle->model.predict(x);
        if (predictions.size() != rows) return 2;
        for (std::size_t row = 0; row < rows; ++row) out[row] = predictions[row];
        return 0;
    } catch (...) {
        return 2;
    }
}

void model_unload(model_handle* handle) noexcept {
    delete handle;
}

}  // extern "C"
"#;
