//! Fitted, std-only Rust over the existing recipe kernels. No prediction call fits a model.
//!
//! [`emit`] accepts encoded numeric rows. [`emit_pipeline`] also carries the version-two
//! math-pipeline input contract (raw cells, selected columns and target units). Categorical
//! encodings are not present in that contract; unsupported model forms fail closed.

use crate::calibration::OutputTransform;
use crate::fitted_js::ExportCalibration;
use crate::math_pipeline::{InputContract, MathPipeline, F64_PREDICTION_VERSION};
use anyhow::{bail, ensure, Context, Result};
use feat_math::genome::GenomeIr;
use feat_math::heads::FittedHead;
use feat_math::model::Model;
use feat_math::ops::FittedOp;
use feat_math::spec::{self, Plan, PlannedHead, PlannedOp, PlannedStage, TargetKind};
use serde_json::{json, Value};

pub const EMITTER_VERSION: &str = "jope-fitted-rust/v1";

/// Engine output for one caller-supplied parity row.
#[derive(Clone, Debug)]
pub struct CheckRow {
    pub inputs: Vec<f64>,
    pub expected: f64,
}

/// Emit a single Rust module with `predict_numeric(&[Vec<f64>]) -> Result<Vec<f64>, String>`.
/// Rows must have exactly `width` finite cells. Constants round-trip every fitted f64 bit.
pub fn emit(genome: &GenomeIr, model: &Model, width: usize) -> Result<String> {
    emit_with_platt(genome, model, width, &ExportCalibration::default())
}

/// Emit a predictor with up to 64 caller-supplied engine check rows.
pub fn emit_with_checks(
    genome: &GenomeIr,
    model: &Model,
    width: usize,
    checks: &[CheckRow],
) -> Result<String> {
    emit_with_checks_and_platt(genome, model, width, &ExportCalibration::default(), checks)
}

/// Emit std-only Rust with the shared optional regression and Platt calibration.
pub fn emit_with_platt(
    genome: &GenomeIr,
    model: &Model,
    width: usize,
    calibration: &ExportCalibration,
) -> Result<String> {
    emit_with_checks_and_platt(genome, model, width, calibration, &[])
}

/// Emit Rust with shared calibration and caller-supplied engine check rows.
pub fn emit_with_checks_and_platt(
    genome: &GenomeIr,
    model: &Model,
    width: usize,
    calibration: &ExportCalibration,
    checks: &[CheckRow],
) -> Result<String> {
    let mut source =
        emit_calibrated_with_checks(genome, model, width, calibration.output.as_ref(), checks)?;
    if let Some(platt) = &calibration.platt {
        platt.validate().context("invalid Platt calibration")?;
        let sigmoid = &platt.sigmoid;
        source.push_str(&format!(
            "\npub const PLATT_CLASS_ORDER: [&str; 2] = [{:?}, {:?}];\n\
             pub fn fitted_platt(score: f64) -> Result<[f64; 2], String> {{\n\
                 if !score.is_finite() || !(0.0..=1.0).contains(&score) {{ return Err(\"invalid Platt score\".into()); }}\n\
                 let z = if {} == 0.0 {{ {} }} else {{ {} + {} * (score / {} - {}) }};\n\
                 let p = if z >= 0.0 {{ 1.0 / (1.0 + (-z).exp()) }} else {{ let e = z.exp(); e / (1.0 + e) }};\n\
                 Ok([1.0 - p, p])\n\
             }}\n",
            platt.class_order[0],
            platt.class_order[1],
            rs_number(sigmoid.slope),
            rs_number(sigmoid.intercept),
            rs_number(sigmoid.intercept),
            rs_number(sigmoid.slope),
            rs_number(sigmoid.scale),
            rs_number(sigmoid.center),
        ));
    }
    lint(&source)?;
    Ok(source)
}

fn rs_number(value: f64) -> String {
    format!("{value:?}")
}

/// Emit std-only Rust with the optional fitted regression output transform.
pub fn emit_calibrated(
    genome: &GenomeIr,
    model: &Model,
    width: usize,
    output_transform: Option<&OutputTransform>,
) -> Result<String> {
    emit_calibrated_with_checks(genome, model, width, output_transform, &[])
}

/// Emit std-only Rust with calibration and caller-supplied engine check rows.
pub fn emit_calibrated_with_checks(
    genome: &GenomeIr,
    model: &Model,
    width: usize,
    output_transform: Option<&OutputTransform>,
    checks: &[CheckRow],
) -> Result<String> {
    ensure!(
        checks.len() <= 64 && checks.iter().all(|row| row.inputs.len() == width),
        "invalid Rust self-check rows"
    );
    ensure!(width > 0, "fitted Rust requires a positive input width");
    feat_math::kernel::width::check_input_width(model, width).context("fitted Rust input width")?;
    let plan = spec::build_plan(genome)
        .map_err(|missing| anyhow::anyhow!("fitted Rust unsupported: {}", missing.join(", ")))?;
    let Plan::Pipeline {
        target,
        stages,
        head,
    } = &plan
    else {
        bail!("fitted Rust requires a pipeline recipe");
    };
    let Model::Pipeline {
        target: fitted_target,
        ops,
        head: fitted_head,
    } = model
    else {
        bail!("fitted Rust requires a fitted pipeline");
    };
    // Serialize once so serde's non-finite-to-null behavior cannot lose an invalid coefficient.
    let state = serde_json::to_value(model)?;
    ensure!(
        no_null(&state["Pipeline"]["ops"]),
        "non-finite fitted operation state"
    );
    ensure!(
        no_null(&state["Pipeline"]["target"]) || fitted_target.is_none(),
        "non-finite fitted target state"
    );
    let mut source =
        fitted_module_header(feat_math::emit_rs::emit_rust(&plan, &Default::default()))?;
    // The recipe's final CSV training CLI is not an entry point in a fitted module.
    let end = source
        .find("\nfn main() {")
        .context("recipe has no main boundary")?;
    source.truncate(end);
    let mut fields = pipeline_fields(&source, stages, ops, head, fitted_head)?;
    match (target, fitted_target) {
        (None, None) => {}
        (Some(planned), Some(fitted)) => {
            use feat_math::target_transform::FittedTarget as F;
            let ty = match (planned.kind, fitted) {
                (TargetKind::RankGauss, F::RankGauss { .. }) => Some("NormalScoresTarget"),
                (TargetKind::Log1p, F::Log1p) => None,
                (TargetKind::Winsor, F::Winsor { .. }) => Some("WinsorizedTarget"),
                (TargetKind::ZScore, F::ZScore { .. }) => Some("StandardizedTarget"),
                _ => bail!("fitted Rust target differs from recipe"),
            };
            if let Some(ty) = ty {
                let value = serde_json::to_value(fitted)?;
                let mut value = value
                    .as_object()
                    .context("fitted target variant")?
                    .values()
                    .next()
                    .context("fitted target")?
                    .clone();
                if ty == "StandardizedTarget" {
                    value["lo"] = value["clamp_lo"].clone();
                    value["hi"] = value["clamp_hi"].clone();
                }
                fields.push(format!("target: {}", literal(&source, ty, &value)?));
            }
        }
        _ => bail!("fitted Rust target differs from recipe"),
    }
    let output_transform = output_transform
        .copied()
        .filter(|t| !is_identity_transform(t));
    let calibrated_prediction = if let Some(transform) = output_transform {
        ensure!(
            transform.intercept.is_finite()
                && transform.slope.is_finite()
                && transform.lower.is_none_or(f64::is_finite),
            "non-finite regression output transform"
        );
        // Debug keeps an f64 literal for integral values (`0.0`, not `0`), which rustc requires.
        let code = if let Some(lower) = transform.lower {
            format!(
                "let predictions: Vec<f64> = predictions.into_iter().map(|pred| ({} + {} * pred).max({})).collect();",
                rs_number(transform.intercept),
                rs_number(transform.slope),
                rs_number(lower)
            )
        } else {
            format!(
                "let predictions: Vec<f64> = predictions.into_iter().map(|pred| {} + {} * pred).collect();",
                rs_number(transform.intercept),
                rs_number(transform.slope)
            )
        };
        code
    } else {
        String::new()
    };
    source.push_str(&format!(
        "\n// Fitted state: prediction only. Emitter {EMITTER_VERSION}.\n\
         fn fitted_model() -> Model {{ Model {{ {} }} }}\n\
         pub fn predict_numeric(rows: &[Vec<f64>]) -> Result<Vec<f64>, String> {{\n\
         if rows.iter().any(|r| r.len() != {width} || r.iter().any(|v| !v.is_finite())) {{ return Err(\"invalid numeric row\".into()); }}\n\
         if rows.is_empty() {{ return Ok(Vec::new()); }}\n\
         let predictions = fitted_model().predict(rows);\n{calibrated_prediction}\n\
         if predictions.len() != rows.len() || predictions.iter().any(|v| !v.is_finite()) {{ return Err(\"prediction failed: non-finite fitted prediction\".into()); }}\n\
         Ok(predictions)\n}}\n", fields.join(", "), calibrated_prediction=calibrated_prediction
    ));
    let rows = checks
        .iter()
        .map(|row| {
            let inputs = row
                .inputs
                .iter()
                .map(|v| format!("f64::from_bits(0x{:016x})", v.to_bits()))
                .collect::<Vec<_>>()
                .join(",");
            format!(
                "(vec![{inputs}],f64::from_bits(0x{:016x}))",
                row.expected.to_bits()
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    source.push_str(&format!(r#"
const SELF_CHECK_ABS_TOL: f64 = 1e-12;
const SELF_CHECK_REL_TOL: f64 = 1e-10;
#[test]
fn self_check() {{
    let rows: Vec<(Vec<f64>, f64)> = vec![{rows}];
    for (index, (inputs, expected)) in rows.iter().enumerate() {{
        let actual = predict_numeric(std::slice::from_ref(inputs)).unwrap()[0];
        assert!((expected.is_nan() && actual.is_nan()) || (actual.is_finite() && (actual-expected).abs() <= SELF_CHECK_ABS_TOL + SELF_CHECK_REL_TOL*expected.abs()), "self_check failed at row {{index}}");
    }}
}}
"#));
    lint(&source)?;
    Ok(source)
}

fn fitted_module_header(source: String) -> Result<String> {
    const CODE_START: &str = "\n#![forbid(unsafe_code)]";
    let code_start = source
        .find(CODE_START)
        .context("recipe has no module code boundary")?;
    let mut lines = source[..code_start].lines();
    lines.next(); // The recipe title describes the old training program.

    let mut first_paragraph_started = false;
    let mut preserving = false;
    let mut operation_docs = Vec::new();
    for line in lines {
        if !preserving {
            if first_paragraph_started && line == "//!" {
                preserving = true;
            } else if line != "//!" {
                first_paragraph_started = true;
            }
            continue;
        }
        if line.starts_with("//! Build:")
            || line.starts_with("//! Runtime:")
            || line.starts_with("//! Usage:")
        {
            continue;
        }
        operation_docs.push(line);
    }

    while operation_docs.first() == Some(&"//!") {
        operation_docs.remove(0);
    }
    while operation_docs.last() == Some(&"//!") {
        operation_docs.pop();
    }

    let mut header = String::from(
        "//! Fitted, std-only Rust prediction module generated from an already fitted model.\n\
         //!\n\
         //! Save this artifact as `predictor.rs`, then import it with `mod predictor;`. Call\n\
         //! `predictor::predict_numeric(&rows)` with encoded numeric rows of the fitted width.\n\
         //! The module embeds fitted state and predicts without fitting or reading files.\n\
         //! When emitted with a version-two input contract, use `predict_cells` or\n\
         //! `predict_table` to predict from raw cells or named table columns.\n\
         //!\n",
    );
    if !operation_docs.is_empty() {
        header.push_str(&operation_docs.join("\n"));
        header.push('\n');
    }
    header.push_str(&source[code_start..]);
    Ok(header)
}

/// Emit version-two raw-cell prediction without changing the stored object or legacy semantics.
/// `predict_cells` reads contract order; `predict_table` matches headers by name and ignores extras.
pub fn emit_pipeline(pipeline: &MathPipeline) -> Result<String> {
    let input = &pipeline.input;
    ensure!(
        input.prediction_version == F64_PREDICTION_VERSION,
        "fitted Rust requires prediction_version=2"
    );
    // Reuse the public constructor's complete input validation, including mutable public fields.
    InputContract::new(
        input.columns.clone(),
        input.selected.clone(),
        input.target.clone(),
    )?;
    ensure!(
        input.task == "regression",
        "fitted Rust input task must be regression"
    );
    let genome = GenomeIr::parse(&pipeline.state.dsl)?;
    // Every exported math-pipeline carries 64 engine-derived numeric self-checks.
    let rows: Vec<Vec<f64>> = (0..64)
        .map(|i| {
            (0..input.selected.len())
                .map(|j| ((i + 17 * j) % 64) as f64 / 63.0)
                .collect()
        })
        .collect();
    let expected = pipeline.state.model.predict(&rows)?;
    let checks: Vec<CheckRow> = rows
        .into_iter()
        .zip(expected)
        .map(|(inputs, expected)| CheckRow { inputs, expected })
        .collect();
    let mut source = emit_with_checks(
        &genome,
        &pipeline.state.model,
        input.selected.len(),
        &checks,
    )?;
    let names: Vec<_> = input
        .columns
        .iter()
        .map(|c| format!("{:?}", c.name))
        .collect();
    let scalers: Vec<_> = input
        .columns
        .iter()
        .map(|c| format!("({:?}, {:?}, {})", c.min, c.max, c.constant))
        .collect();
    source.push_str(&format!(
        "\nconst MODEL_COLUMNS: &[&str] = &[{}];\n\
         const INPUT_SCALERS: &[(f64, f64, bool)] = &[{}];\n\
         const INPUT_SELECTED: &[usize] = &{:?};\n\
         const TARGET_MIN: f64 = {:?};\nconst TARGET_SCALE: f64 = {:?};\n{}",
        names.join(", "),
        scalers.join(", "),
        input.selected,
        input.target.min,
        input.target.scale,
        INPUT_ADAPTER
    ));
    // Input names are user data, not generated public vocabulary.
    Ok(source)
}

fn lint(source: &str) -> Result<()> {
    let hits = feat_math::transpile::lint_shipped_text(source);
    ensure!(
        hits.is_empty(),
        "banned vocabulary in Rust artifact: {}",
        hits.join(", ")
    );
    Ok(())
}

fn no_null(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Array(items) => items.iter().all(no_null),
        Value::Object(fields) => fields.values().all(no_null),
        _ => true,
    }
}

fn is_identity_transform(transform: &OutputTransform) -> bool {
    transform.intercept == 0.0 && transform.slope == 1.0 && transform.lower.is_none()
}

fn pipeline_fields(
    source: &str,
    stages: &[PlannedStage],
    ops: &[FittedOp],
    planned_head: &PlannedHead,
    head: &FittedHead,
) -> Result<Vec<String>> {
    ensure!(
        stages.len() == ops.len(),
        "fitted Rust stage count differs from recipe"
    );
    let mut fields = Vec::new();
    for (i, (stage, op)) in stages.iter().zip(ops).enumerate() {
        if let Some(value) = stage_literal(source, stage, op)? {
            fields.push(format!("stage{}: {value}", i + 1));
        }
    }
    fields.push(format!(
        "head: {}",
        head_literal(source, planned_head, head)?
    ));
    Ok(fields)
}

fn stage_literal(source: &str, stage: &PlannedStage, op: &FittedOp) -> Result<Option<String>> {
    use FittedOp as F;
    use PlannedOp as P;
    let value = serde_json::to_value(op)?;
    let state = value
        .as_object()
        .context("fitted operation variant")?
        .values()
        .next()
        .context("fitted operation")?;
    ensure!(
        state["append"].as_bool() == Some(stage.append),
        "fitted Rust append differs from recipe"
    );
    // Prediction-time literals must agree. Fit-only seeds, regularizers and knot counts need
    // not: their learned output is carried in the constants instead of being recomputed.
    let matches = match (&stage.op, op) {
        (P::Standardize, F::Standardize { .. })
        | (P::RobustScale, F::RobustScale { .. })
        | (P::Rank, F::Rank { .. })
        | (P::Blom, F::Blom { .. })
        | (P::Hinge { .. }, F::Hinge { .. })
        | (P::WinsorClip { .. }, F::WinsorClip { .. })
        | (P::PolyInteract, F::PolyInteract { .. })
        | (P::RffCos { .. }, F::RffCos { .. })
        | (P::RffSin { .. }, F::RffSin { .. })
        | (P::RandProj { .. }, F::RandProj { .. })
        | (P::ObliqueHinge { .. }, F::ObliqueHinge { .. })
        | (P::RowStats, F::RowStats { .. })
        | (P::PairProd { .. }, F::PairProd { .. })
        | (P::TargetBinEncode { .. }, F::TargetBinEncode { .. })
        | (P::Sigmoid, F::Sigmoid { .. })
        | (P::NormRow, F::NormRow { .. })
        | (P::Polar, F::Polar { .. })
        | (P::FftMag { .. }, F::FftMag { .. })
        | (P::RowSpectral, F::RowSpectral { .. })
        | (P::RowDiffStats, F::RowDiffStats { .. })
        | (P::SvdProject { .. }, F::SvdProject { .. })
        | (P::SelectCorr { .. }, F::SelectCorr { .. })
        | (P::Impute { .. }, F::Impute { .. })
        | (P::PairRatio { .. }, F::PairRatio { .. })
        | (P::Spline { .. }, F::Spline { .. }) => true,
        (P::QuantileBins { bins: a }, F::QuantileBins { bins: b, .. })
        | (P::BinOnehot { bins: a }, F::BinOnehot { bins: b, .. }) => a == b,
        (P::Cheb { degree: a }, F::Cheb { degree: b, .. }) => a == b,
        (P::LandmarkDist { gamma: a, .. }, F::LandmarkDist { gamma: b, .. })
        | (P::Nystrom { gamma: a, .. }, F::Nystrom { gamma: b, .. })
        | (P::Symlog { c: a }, F::Symlog { c: b, .. })
        | (P::Step { t: a }, F::Step { t: b, .. }) => a.to_bits() == b.to_bits(),
        (P::Lag { k: a }, F::Lag { k: b, .. }) | (P::Roll { k: a }, F::Roll { k: b, .. }) => a == b,
        (
            P::MapExpr { source: a, .. },
            F::MapExpr {
                source: b, expr, ..
            },
        )
        | (
            P::DeriveExpr { source: a, .. },
            F::DeriveExpr {
                source: b, expr, ..
            },
        ) => {
            a == b
                && serde_json::to_value(feat_math::math_expr::MathExpr::parse(a)?)?
                    == serde_json::to_value(expr)?
        }
        _ => false,
    };
    ensure!(
        matches,
        "fitted Rust operation or prediction parameter differs from recipe: {:?}",
        stage.op
    );
    let Some(ty) = spec::op_struct_name(&stage.op) else {
        return Ok(None);
    };
    let state = if let F::FftMag { k, .. } = op {
        json!(k)
    } else {
        state.clone()
    };
    Ok(Some(literal(source, ty, &state)?))
}

fn head_literal(source: &str, planned: &PlannedHead, head: &FittedHead) -> Result<String> {
    use FittedHead as F;
    use PlannedHead as P;
    let serialized = serde_json::to_value(head)?;
    let state = serialized
        .as_object()
        .context("fitted head variant")?
        .values()
        .next()
        .context("fitted head")?;
    let (ty, state) = match (planned, head) {
        (P::Ridge { .. }, F::Ridge(_)) => ("RidgeRegression", state.clone()),
        (P::RidgeGcv, F::RidgeGcv(_)) => ("GcvRidgeRegression", state.clone()),
        (P::ElasticNet { .. }, F::ElasticNet(_)) => ("ElasticNet", state.clone()),
        (P::Gbm(_), F::Gbm(_)) => ("GradientBoostedTrees", state.clone()),
        (P::Blend { .. }, F::Ensemble { ridge, gbm, w }) => (
            "BlendedRegression",
            json!({"linear":ridge, "boosted":gbm, "weight":w}),
        ),
        (P::HistGbm2 { .. }, F::HistGbm2(_)) => ("LeafwiseBoostedTrees", state.clone()),
        (P::Krr { .. }, F::Krr(_)) => ("KernelRidgeRegression", state.clone()),
        (P::Gpr { .. }, F::Gpr(_)) => ("GaussianProcessArdRegression", state.clone()),
        (P::GprKernel { spec, .. }, F::GprKernel(fitted))
            if serde_json::to_value(spec)? == serde_json::to_value(&fitted.spec)? =>
        {
            ("GaussianProcessCompositeKernelRegression", state.clone())
        }
        (P::Ksmooth { k, gamma }, F::Ksmooth(fitted))
            if *k == fitted.k && gamma.to_bits() == fitted.gamma.to_bits() =>
        {
            ("NadarayaWatsonRegression", state.clone())
        }
        (P::Stack, F::Stack(_)) => ("GreedyEnsembleSelection", state.clone()),
        (P::Portfolio(plan), F::Portfolio(fitted)) => {
            return portfolio_literal(source, plan, fitted)
        }
        _ => bail!("fitted Rust head or prediction parameter differs from recipe"),
    };
    literal(source, ty, &state)
}

fn portfolio_literal(
    source: &str,
    plan: &spec::PortfolioPlan,
    fitted: &feat_math::portfolio::PortfolioModel,
) -> Result<String> {
    use feat_math::portfolio::FittedMemberState as F;
    ensure!(
        fitted.members.len() == fitted.weights.len(),
        "fitted Rust ensemble weights differ from members"
    );
    let entries = fitted.members.iter().zip(&fitted.weights).map(|(member, weight)| -> Result<String> {
        let (index, planned) = plan.members.iter().enumerate().find(|(_, p)| p.name == member.name).context("fitted Rust ensemble member absent from recipe")?;
        ensure!(member.proj_view == planned.proj_view, "fitted Rust ensemble member view differs");
        let state = match (&planned.kind, &member.state) {
            (_, F::Mean | F::Skipped) => "None".to_owned(),
            (spec::PlannedMemberKind::Pipeline { stages, head }, F::Fitted(model)) => {
                let Model::Pipeline { target: None, ops, head: fitted_head } = model.as_ref() else { bail!("fitted Rust ensemble component is not a plain pipeline"); };
                let fields = pipeline_fields(source, stages, ops, head, fitted_head)?;
                format!("Some(ComponentState::C{}(Component{} {{ {} }}))", index + 1, index + 1, fields.join(", "))
            }
            (spec::PlannedMemberKind::GroupMean, F::GroupMean(model)) => {
                let value = serde_json::to_value(model)?;
                let table = value["table"].as_array().context("fitted Rust group table")?;
                let mapped = json!({"keys":table.iter().map(|v| v[0].clone()).collect::<Vec<_>>(), "values":table.iter().map(|v| v[1].clone()).collect::<Vec<_>>(), "fallback":value["g_mean"]});
                format!("Some(ComponentState::C{}({}))", index + 1, literal(source, "DuplicateRowAverage", &mapped)?)
            }
            _ => bail!("fitted Rust ensemble component differs from recipe"),
        };
        ensure!(weight.is_finite(), "non-finite fitted ensemble weight");
        Ok(format!("EnsembleEntry {{ projected: {}, weight: {weight:?}, state: {state} }}", member.proj_view))
    }).collect::<Result<Vec<_>>>()?;
    let proj = match &fitted.proj {
        None => "None".to_owned(),
        Some(op @ FittedOp::SvdProject { append: false, .. }) => {
            let v = serde_json::to_value(op)?;
            format!(
                "Some({})",
                literal(source, "PrincipalComponentProjection", &v["SvdProject"])?
            )
        }
        _ => bail!("fitted Rust ensemble projection differs"),
    };
    let state = serde_json::to_value(fitted)?;
    let fields = ["mean_only", "y_mean", "width", "keep", "clip_lo", "clip_hi"];
    let types = ["bool", "f64", "usize", "Vec<usize>", "f64", "f64"];
    let mut fields = fields
        .iter()
        .zip(types)
        .map(|(name, ty)| Ok(format!("{name}: {}", literal(source, ty, &state[name])?)))
        .collect::<Result<Vec<_>>>()?;
    fields.push(format!("proj: {proj}"));
    fields.push(format!("entries: vec![{}]", entries.join(", ")));
    Ok(format!(
        "InternallyValidatedEnsemble {{ {} }}",
        fields.join(", ")
    ))
}

/// Render the small field-type grammar declared by emit_rs, never a Debug dump of fitted state.
/// This binds constructors to the emitter's actual fields and refuses unrecognized template drift.
fn literal(source: &str, ty: &str, value: &Value) -> Result<String> {
    if let Some(inner) = ty.strip_prefix("Vec<").and_then(|s| s.strip_suffix('>')) {
        let items = value
            .as_array()
            .with_context(|| format!("fitted Rust expected {ty}"))?;
        let values = items
            .iter()
            .map(|v| literal(source, inner, v))
            .collect::<Result<Vec<_>>>()?;
        return Ok(format!("vec![{}]", values.join(", ")));
    }
    if let Some(inner) = ty.strip_prefix("Option<").and_then(|s| s.strip_suffix('>')) {
        return if value.is_null() {
            Ok("None".into())
        } else {
            Ok(format!("Some({})", literal(source, inner, value)?))
        };
    }
    if ty == "(usize, usize)" {
        let pair = value
            .as_array()
            .filter(|v| v.len() == 2)
            .context("fitted Rust expected index pair")?;
        return Ok(format!(
            "({}, {})",
            literal(source, "usize", &pair[0])?,
            literal(source, "usize", &pair[1])?
        ));
    }
    match ty {
        "f64" => {
            return value
                .as_f64()
                .filter(|v| v.is_finite())
                .map(|v| format!("{v:?}"))
                .context("non-finite fitted Rust coefficient")
        }
        "usize" | "u64" => {
            return value
                .as_u64()
                .map(|v| v.to_string())
                .context("invalid fitted Rust unsigned integer")
        }
        "i64" => {
            return value
                .as_i64()
                .map(|v| v.to_string())
                .context("invalid fitted Rust signed integer")
        }
        "bool" => {
            return value
                .as_bool()
                .map(|v| v.to_string())
                .context("invalid fitted Rust boolean")
        }
        _ => {}
    }
    let mut value = value.clone();
    if matches!(
        ty,
        "RidgeRegression"
            | "GcvRidgeRegression"
            | "ElasticNet"
            | "KernelRidgeRegression"
            | "GaussianProcessArdRegression"
            | "GaussianProcessCompositeKernelRegression"
            | "NadarayaWatsonRegression"
    ) {
        let means = value["means"].as_array().context("missing fitted means")?;
        ensure!(
            value["scales"]
                .as_array()
                .is_some_and(|v| v.len() == means.len()),
            "fitted Rust scale width differs from means"
        );
        if matches!(ty, "RidgeRegression" | "GcvRidgeRegression" | "ElasticNet") {
            let bias = usize::from(ty != "ElasticNet");
            ensure!(
                value["w"]
                    .as_array()
                    .is_some_and(|v| v.len() == means.len() + bias),
                "fitted Rust coefficient width differs from means"
            );
        }
    }
    if matches!(ty, "TreeNode" | "BoostNode") {
        value = if let Some(leaf) = value.get("Leaf") {
            json!({"feature":-1,"threshold":0,"left":0,"right":0,"value":leaf})
        } else if let Some(split) = value.get("Split") {
            json!({"feature":split["feature"],"threshold":split.get("thr_bin").unwrap_or(&split["threshold"]),"left":split["left"],"right":split["right"],"value":0.0})
        } else {
            bail!("invalid fitted Rust split node");
        };
    }
    if ty == "GradientBoostedTrees" {
        value["fallback"] = value["constant_value"].clone();
    }
    let marker = format!("struct {ty} {{\n");
    let body = source
        .split_once(&marker)
        .with_context(|| format!("recipe has no fitted type {ty}"))?
        .1
        .split_once("\n}")
        .context("incomplete fitted type")?
        .0;
    let fields = body
        .lines()
        .map(|line| -> Result<String> {
            let (name, field_type) = line
                .trim()
                .strip_suffix(',')
                .and_then(|s| s.split_once(": "))
                .context("unsupported fitted Rust field declaration")?;
            let value = value
                .get(name)
                .with_context(|| format!("missing fitted Rust field {ty}.{name}"))?;
            Ok(format!("{name}: {}", literal(source, field_type, value)?))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(format!("{ty} {{ {} }}", fields.join(", ")))
}

const INPUT_ADAPTER: &str = r##"
fn encode_cell(cell: Option<&str>, column: usize) -> Result<f64, String> {
    let (min, max, constant) = INPUT_SCALERS[column];
    let value = match cell {
        None => return Ok(0.0),
        Some(cell) => {
            let text = cell.trim();
            match text.to_ascii_lowercase().as_str() {
                "" | "na" | "n/a" | "null" | "none" | "nan" | "?" | "#n/a" => return Ok(0.0),
                "true" | "yes" => 1.0,
                "false" | "no" => 0.0,
                _ => match text.parse::<f64>() {
                    Ok(v) if v.is_finite() => v,
                    _ => return Ok(0.0),
                },
            }
        }
    };
    if constant { return Ok(0.0); }
    let value = ((value - min) / (max - min)).clamp(0.0, 1.0);
    if !value.is_finite() { return Err("prediction failed: non-finite normalized input".into()); }
    Ok(value)
}

pub fn predict_cells(rows: &[Vec<Option<&str>>]) -> Result<Vec<f64>, String> {
    let encoded = rows.iter().map(|row| {
        if row.len() != MODEL_COLUMNS.len() { return Err("invalid raw row width".to_owned()); }
        INPUT_SELECTED.iter().map(|&j| encode_cell(row[j], j)).collect()
    }).collect::<Result<Vec<Vec<f64>>, String>>()?;
    predict_numeric(&encoded)?.into_iter().map(|v| {
        let value = if TARGET_SCALE <= 1e-12 { TARGET_MIN } else { TARGET_MIN + v * TARGET_SCALE };
        if !value.is_finite() { return Err("prediction failed: non-finite target prediction".into()); }
        Ok(value)
    }).collect()
}

pub fn predict_table(headers: &[&str], rows: &[Vec<Option<&str>>]) -> Result<Vec<f64>, String> {
    let at = MODEL_COLUMNS.iter().map(|name| headers.iter().position(|h| h == name)
        .ok_or_else(|| format!("input lacks column {:?}", name))).collect::<Result<Vec<_>, _>>()?;
    let ordered = rows.iter().map(|row| {
        if row.len() != headers.len() { return Err("invalid table row width".to_owned()); }
        Ok(at.iter().map(|&i| row[i]).collect())
    }).collect::<Result<Vec<_>, String>>()?;
    predict_cells(&ordered)
}
"##;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math_pipeline::{
        NumericColumn, PipelineState, TargetScale, INPUT_PART, PIPELINE_PART,
    };
    use std::collections::BTreeMap;
    use std::process::Command;

    // Real catalog v246 dataset b4c8807a75fc190f, TRAIN blob only. No outer-test rows.
    // s3://veox-jopedime-datasets-use2/blobs/sha256/782acd17eed2bfc7acfa4438be076fb63026763e6578661cfc284c2f3f9a21e1
    const TRAIN: &str = r#"0.392630965,0.430275381,0.430894703,0.12725848,0.360910535,0.64234072,0.693609476,0.579903841,0.505381107,0.69527173,0.582644641,0.5773651,0.405602932
0.532239497,0.324855298,0,0.272702694,0.0432448462,0.122006744,0.130229235,0.423766315,0.0293887053,0.75315696,0,0.880756497,1
0.88658607,0.473776519,0.739836991,0.290895343,0.0643178821,0.822988451,0.687003195,0.984539568,0.761445224,0.0731142461,0.648760319,0.719476581,0.565164447
0.114277489,0.382564455,0.752032399,0.090887025,0.526813567,0.742386162,0.539474308,0.938393354,0.617167652,0.662819445,0.859504104,0.347767889,0.422655284
0.408175021,0.558849335,0.829268336,0.45453915,0.316212088,0.782663882,0.434365064,0.881171227,0.740591943,0.642971337,0.561983466,0.1910301,0.339829475
0.189407021,0.309770226,0.367886305,0.363631308,0.296831459,0.308776051,0.355381072,0.305160969,0.416478246,0.835775375,0.512396693,0.261981696,0.420219243
0.102763385,0.174179971,0.778455317,0.21818018,0.535733581,0.665172696,0.484818637,0.710756838,0.655283988,0.277088135,0.921487629,0.450724959,0.369062126
0.822107077,0.773197711,0.908536494,0.45453915,0.0365236998,0.84244597,0.688308239,0.603560925,0.82356292,0.79825902,0.863636374,0.326938272,0.397076726
0.282383412,0.517277658,0.556910694,0.599983335,0.402099937,0.568185449,0.65837121,0.641646266,0.561009467,0.542055309,0.648760319,0.671366394,0.393422663
0.305123776,0.213997543,1,0.145451143,0.197167844,0.708773315,0.541624963,0.768193126,0.693888068,0.192443401,0.719008267,0.130075127,0
1,1,0.833333492,1,0.0193282682,1,0.898360729,1,1,0.308720618,0.454545468,0.283314794,0.237515226
0.160621762,0.23031047,0.249999806,0.254523903,0.601590395,0.516470432,0.582751989,0.466739893,0.437724054,0.839140177,0.446280986,0.878900409,0.311814874
0.900978684,0.281880379,0.855690539,0.309081078,0,0.909259915,0.753536522,0.814094543,0.958069861,0.260106772,0.797520638,0.372082263,0.258221686
0.205526769,0.0280652512,0.530487716,0.799998641,0.173617244,0.570154846,0.518912315,0.64190495,0.488903433,0.474720091,0.516528904,0.456496716,0.428745419
0.158606797,0.0387651287,0.182927072,0.399995834,1,0.114793263,0.0579076372,0.418282688,0.169862658,0.563431859,0.533057868,0.209798709,0.345919609
0.204951063,0.103841431,0.792682946,0.490903676,0.523491383,0.665682733,0.645152509,0.370122582,0.776554227,0.904096305,0.950413227,0.599517882,0.278928131
0.387449622,0.30503419,0.78861779,0.309074134,0.509860694,0.608530283,0.451485544,0.662896037,0.708410084,0.553323686,1,0.650197566,0.464068204
0.367012084,0.286265582,0.924796224,0.109079689,0.0870680586,0.902512968,0.63376081,0.859470963,0.862631083,0.747068584,0.657024801,0.153011963,0.274056017
0.486183077,0.551482201,0.699187219,0.927264035,0.161958084,0.781150758,0.752020597,0.474855214,0.877674222,0.557997108,0.859504104,0.347767889,0.293544471
0.263385147,0.140852481,0.723577201,0.181822598,0.391995251,0.663917363,0.493468851,0.502018988,0.718742073,0.730278254,0.834710717,0.4485493,0.401948839
0.873344839,0.554113328,0.959349453,0.618176043,0.207016751,0.925042927,1,0.939120531,0.890241802,0,0.719008267,0.327552319,0.313032895
0.828727663,0.402210146,0.61382097,0.745455325,0.0333997682,0.759799778,0.896631718,0.580347061,0.621512294,0.533041358,0.871900797,0.394919485,0.381242394
0.36557284,0.109279074,0.152438968,0.0363645181,0.389253587,0.037192557,0.201275483,0,0.265883029,0.495052397,0.425619841,0.515321374,0.54202193
0.24582614,0.469566733,0.743902087,0,0.317109227,0.78819716,0.579324722,0.863871396,0.563951254,1,0.690082669,0.35882771,0.414129108
0.374208391,0.231362924,0.0955286622,0.363631308,0.0852978751,0.31474036,0.239869252,0.429203749,0.290933728,0.98757869,0.119834714,1,0.651644349
"#;

    fn training() -> (Vec<Vec<f64>>, Vec<f64>) {
        assert_eq!(
            crate::sha256_hex(TRAIN.as_bytes()),
            "782acd17eed2bfc7acfa4438be076fb63026763e6578661cfc284c2f3f9a21e1"
        );
        let rows: Vec<Vec<f64>> = TRAIN
            .lines()
            .map(|l| l.split(',').map(|v| v.parse().unwrap()).collect())
            .collect();
        let y = rows.iter().map(|r| *r.last().unwrap()).collect();
        (
            rows.into_iter()
                .map(|mut r| {
                    r.pop();
                    r
                })
                .collect(),
            y,
        )
    }

    fn fitted(dsl: &str) -> (GenomeIr, Model) {
        let (x, y) = training();
        let genome = GenomeIr::parse(dsl).unwrap();
        let model = feat_math::model::fit(&genome, &x, &y).unwrap();
        (genome, model)
    }

    fn run_rust(name: &str, source: &str) {
        // One reused lane target; preserve generated source and executables as proof artifacts.
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/fitted-rs-proof")
            .join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let program = dir.join("predictor.rs");
        let executable = dir.join("predictor");
        std::fs::write(&program, source).unwrap();
        let version = Command::new("rustc").arg("--version").output().unwrap();
        eprintln!("{}", String::from_utf8_lossy(&version.stdout));
        let compile = Command::new("rustc")
            .args([
                "--edition=2021",
                "-C",
                "codegen-units=1",
                "-C",
                "opt-level=0",
                "-D",
                "warnings",
            ])
            .arg(&program)
            .arg("-o")
            .arg(&executable)
            .output()
            .unwrap();
        assert!(
            compile.status.success(),
            "{name}: rustc failed: {}",
            String::from_utf8_lossy(&compile.stderr)
        );
        let output = Command::new(&executable).output().unwrap();
        assert!(
            output.status.success(),
            "{name}: predictor failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        eprintln!("{name}: {}", String::from_utf8_lossy(&output.stdout));
    }

    #[test]
    fn fitted_module_docs_compile_and_run_as_a_plain_rust_module() {
        let (genome, model) = fitted("(pipe X (standardize) (ridge))");
        let source = emit(&genome, &model, 12).unwrap();
        assert!(source.starts_with("//! Fitted, std-only Rust prediction module"));
        assert!(source.contains("`mod predictor;`"));
        assert!(source.contains("predictor::predict_numeric(&rows)"));
        assert!(source.contains("centers each column by its training mean"));
        assert!(!source.contains("train.csv"));
        assert!(!source.contains("test_features.csv"));
        assert!(!source.contains("Usage: ./model"));
        assert!(!source.contains("fn main()"));

        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/fitted-rs-proof/module-usage");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("predictor.rs"), source).unwrap();
        let consumer = dir.join("consumer.rs");
        std::fs::write(
            &consumer,
            "mod predictor;\nfn main() {\n    let rows = vec![vec![0.5; 12]];\n    let predictions = predictor::predict_numeric(&rows).expect(\"fitted prediction\");\n    assert_eq!(predictions.len(), 1);\n    assert!(predictions[0].is_finite());\n}\n",
        )
        .unwrap();
        let executable = dir.join("consumer");
        let compile = Command::new("rustc")
            .args(["--edition=2021"])
            .arg(&consumer)
            .arg("-o")
            .arg(&executable)
            .output()
            .unwrap();
        assert!(
            compile.status.success(),
            "plain rustc consumer compile failed: {}",
            String::from_utf8_lossy(&compile.stderr)
        );
        let output = Command::new(executable).output().unwrap();
        assert!(
            output.status.success(),
            "plain rustc consumer failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn numeric_case(index: usize, dsl: &str, test: &[Vec<f64>]) -> String {
        let (genome, model) = fitted(dsl);
        let expected = model.predict(&test.to_vec()).unwrap();
        let source = emit(&genome, &model, 12).unwrap_or_else(|e| panic!("{dsl}: {e:#}"));
        assert_eq!(
            source,
            emit(&genome, &model, 12).unwrap(),
            "deterministic {dsl}"
        );
        assert!(!source.contains("fn main()"));
        let rows = literal("", "Vec<Vec<f64>>", &json!(test)).unwrap();
        let expected = literal("", "Vec<f64>", &json!(expected)).unwrap();
        format!("mod c{index} {{\n{source}\n}}\nfn check{index}() {{ let rows = {rows}; let expected: Vec<f64> = {expected}; let got = c{index}::predict_numeric(&rows).unwrap(); assert_eq!(got.len(), expected.len()); for (i,(a,b)) in got.iter().zip(&expected).enumerate() {{ assert!((a-b).abs() <= 1e-12+1e-10*b.abs(), \"case {index} row {{i}}: {{a}} vs {{b}}\"); }} assert!(c{index}::predict_numeric(&[]).unwrap().is_empty()); assert!(c{index}::predict_numeric(&[vec![0.0;11]]).is_err()); assert!(c{index}::predict_numeric(&[vec![f64::NAN;12]]).is_err()); }}\n")
    }

    #[test]
    fn rustc_replays_every_legacy_recipe_operation_head_and_target() {
        // Inventory is cross-checked against EXP-01 below. Every legacy recipe branch occurs.
        let operations = [
            "standardize",
            "robust_scale",
            "rank",
            "blom",
            "quantile_bins bins=4",
            "bin_onehot bins=3",
            "hinge knots=3",
            "winsor_clip",
            "cheb degree=3",
            "poly_interact",
            "rff_cos n=5",
            "rff_sin n=5",
            "rand_proj k=4",
            "landmark_dist k=4 gamma=0.8",
            "oblique_hinge n=4",
            "row_stats",
            "pair_prod topk=4",
            "target_bin_encode bins=4",
            "sigmoid",
            "step t=0.5",
            "norm_row",
            "polar",
            "lag k=2",
            "roll k=2",
            "fft_mag k=4",
            "row_spectral",
            "row_diff_stats",
            "svd_project k=4 iters=3",
            "nystrom k=4 gamma=0.8",
            "symlog c=0.4",
            "select_corr k=4",
            "impute strategy=mean add_indicator=true",
            "pair_ratio topk=4",
            "spline knots=3",
            "map expr=\"sin(x)\"",
            "derive expr=\"c0*c1\"",
        ];
        let heads = [
            "ridge",
            "ridge_gcv",
            "elasticnet sweeps=4",
            "gbm rounds=4 depth=2 min_leaf=2",
            "histgbm2 rounds=4 leaves=4 min_leaf=2",
            "ensemble rounds=4 min_leaf=2",
            "krr m=50",
            "gpr m=50 steps=1",
            "gpr m=50 steps=1 kernel=\"rbf+linear\"",
            "ksmooth k=4 gamma=0.8",
            "stack",
            "portfolio preset=core",
        ];
        let mut dsls = vec!["(pipe X (ridge))".to_owned()];
        // Test both replacement and appending, because the recipe carries that flag, not state.
        for op in operations {
            for append in [false, true] {
                dsls.push(format!("(pipe X ({op} append={append}) (ridge))"));
            }
        }
        dsls.extend(heads.into_iter().map(|h| format!("(pipe X ({h}))")));
        dsls.extend(
            ["rank_gauss", "log1p", "winsor", "zscore"]
                .map(|kind| format!("(pipe X (target kind={kind}) (standardize) (ridge))")),
        );
        dsls.push(
            "(pipe X (robust_scale append=true) (hinge knots=3 append=true) (elasticnet sweeps=4))"
                .into(),
        );
        dsls.extend(
            [
                "(pipe X (norm_features) (ridge))",
                "(pipe X (winsor_moments) (ridge))",
                "(pipe X (ensemble_ridge_gbm rounds=4 min_leaf=2))",
                "(pipe X (histgbm rounds=4 min_leaf=2))",
                "(pipe X (logistic))",
            ]
            .map(str::to_owned),
        );
        let mut covered = std::collections::BTreeSet::new();
        for dsl in &dsls {
            if let GenomeIr::Pipeline(pipeline) = GenomeIr::parse(dsl).unwrap() {
                covered.extend(pipeline.stages.into_iter().map(|s| s.op));
            }
        }
        let registry = feat_math::evolve::api::capability::Registry::current();
        use feat_math::evolve::api::capability::{CapabilityKind, Language};
        let admitted: Vec<_> = registry
            .entries
            .iter()
            .filter(|row| {
                matches!(row.kind, CapabilityKind::Operation | CapabilityKind::Head)
                    && row.supported[&Language::Rust]
            })
            .map(|row| row.name)
            .collect();
        eprintln!("Rust registry: {admitted:?}");
        let missing: Vec<_> = admitted
            .iter()
            .filter(|name| !covered.contains(**name))
            .collect();
        assert!(missing.is_empty(), "registry cases missing: {missing:?}");
        let (mut test, _) = training();
        test.extend([
            vec![-0.0; 12],
            vec![0.0; 12],
            vec![1.5; 12],
            vec![-0.5; 12],
            vec![0.5; 12],
        ]);
        let mut program = String::new();
        for (i, dsl) in dsls.iter().enumerate() {
            program.push_str(&numeric_case(i, dsl, &test));
        }
        program.push_str("fn main() {\n");
        for i in 0..dsls.len() {
            program.push_str(&format!("check{i}();\n"));
        }
        program.push_str(&format!(
            "println!(\"{} recipes, {} prediction rows, abs1e-12+rel1e-10\"); }}",
            dsls.len(),
            dsls.len() * test.len()
        ));
        run_rust("all-recipes", &program);
    }

    fn pipeline() -> MathPipeline {
        let (_, model) = fitted("(pipe X (robust_scale append=true) (ridge))");
        let mut names = (0..14)
            .map(|i| NumericColumn {
                name: format!("x{i}"),
                min: -2.0,
                max: 2.0,
                constant: false,
            })
            .collect::<Vec<_>>();
        names[0].name = "unicode\"\\\nλ\u{2028}".into();
        names[4].constant = true;
        names[4].max = names[4].min;
        let input = InputContract::new(
            names,
            vec![13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            TargetScale {
                name: "label".into(),
                min: 1e10 + 0.25,
                scale: 1e-5,
            },
        )
        .unwrap();
        MathPipeline::decode(
            &BTreeMap::from([
                (INPUT_PART.into(), serde_json::to_vec(&input).unwrap()),
                (
                    PIPELINE_PART.into(),
                    serde_json::to_vec(&PipelineState {
                        dsl: "(pipe X (robust_scale append=true) (ridge))".into(),
                        model,
                    })
                    .unwrap(),
                ),
            ]),
            PIPELINE_PART,
        )
        .unwrap()
    }

    #[test]
    fn raw_cells_match_f64_engine_with_missing_tokens_extremes_and_named_columns() {
        let mut p = pipeline();
        let tokens = [
            None,
            Some(""),
            Some("NA"),
            Some("n/a"),
            Some("null"),
            Some("none"),
            Some("NaN"),
            Some("?"),
            Some("#N/A"),
            Some("true"),
            Some("yes"),
            Some("false"),
            Some("no"),
            Some("inf"),
            Some("-inf"),
            Some("1e999"),
            Some("unknown category"),
            Some("-0.0"),
            Some("1.0000000000000002"),
            Some("1.7976931348623157e308"),
            Some("-1.7976931348623157e308"),
            Some("\u{2003}0.12345678901234567\u{2003}"),
            Some("\u{feff}1"),
            Some("0x1"),
            Some("1_0"),
        ];
        let rows: Vec<_> = tokens.iter().map(|v| vec![*v; 14]).collect();
        let mut program = String::new();
        for i in 0..3 {
            if i == 1 {
                p.input.target.scale = 1.0;
                p.input.target.min = -0.0;
            }
            if i == 2 {
                p.input.target.scale = 1e-13;
                p.input.target.min = -0.0;
            }
            let source = emit_pipeline(&p).unwrap();
            let expected = p.predict_cells(&rows).unwrap();
            let rows_source = format!(
                "vec![{}]",
                tokens
                    .iter()
                    .map(|v| format!("vec![{v:?};14]"))
                    .collect::<Vec<_>>()
                    .join(",")
            );
            let expected_source = literal("", "Vec<f64>", &json!(expected)).unwrap();
            program.push_str(&format!("mod c{i} {{\n{source}\n}}\nfn check{i}() {{ let rows={rows_source};let expected:Vec<f64>={expected_source};let got=c{i}::predict_cells(&rows).unwrap();assert_eq!(got.len(),expected.len());for (a,b) in got.iter().zip(&expected) {{assert!((a-b).abs()<=1e-12+1e-10*b.abs());assert_eq!(a.to_bits(),b.to_bits());}} let names=&[{}];let mut reversed=names.to_vec();reversed.reverse();let reordered:Vec<_>=rows.iter().map(|r|r.iter().copied().rev().collect()).collect();assert_eq!(c{i}::predict_table(&reversed,&reordered).unwrap(),got);assert!(c{i}::predict_cells(&[vec![None;13]]).is_err());assert!(c{i}::predict_table(&[],&rows).is_err());assert!(c{i}::predict_cells(&[]).unwrap().is_empty()); }}\n",p.input.columns.iter().map(|c|format!("{:?}",c.name)).collect::<Vec<_>>().join(",")));
        }
        program.push_str("fn main(){check0();check1();check2();println!(\"75 raw rows: exact f64 bits, signed zero, selected order, constant target and named columns\");}");
        run_rust("raw-input", &program);
    }

    #[test]
    fn invalid_pairings_legacy_input_and_nonfinite_constants_are_refused() {
        let (g, mut m) = fitted("(pipe X (step t=0.5) (ridge))");
        assert!(emit(&g, &m, 0).is_err());
        assert!(emit(&g, &m, 11).is_err());
        assert!(emit(
            &GenomeIr::parse("(pipe X (step t=0.25) (ridge))").unwrap(),
            &m,
            12
        )
        .is_err());
        assert!(emit(
            &GenomeIr::parse("(pipe X (sigmoid) (ridge))").unwrap(),
            &m,
            12
        )
        .is_err());
        assert!(emit(
            &GenomeIr::parse("(pipe X (step t=0.5 append=true) (ridge))").unwrap(),
            &m,
            12
        )
        .is_err());
        assert!(emit(&GenomeIr::parse("(pipe X (ridge))").unwrap(), &m, 12).is_err());
        assert!(emit(
            &GenomeIr::parse("(pipe X (step t=0.5) (elasticnet))").unwrap(),
            &m,
            12
        )
        .is_err());
        if let Model::Pipeline { head, .. } = &mut m {
            if let FittedHead::Ridge(r) = head.as_mut() {
                r.w[0] = f64::NAN;
            }
        }
        assert!(emit(&g, &m, 12).is_err());
        let (valid, mut malformed) = fitted("(pipe X (ridge))");
        if let Model::Pipeline { head, .. } = &mut malformed {
            if let FittedHead::Ridge(r) = head.as_mut() {
                r.w.pop();
            }
        }
        assert!(emit(&valid, &malformed, 12).is_err());
        let mut p = pipeline();
        p.input.prediction_version = 1;
        assert!(emit_pipeline(&p).is_err());
        p.input.prediction_version = 2;
        p.input.selected[0] = 999;
        assert!(emit_pipeline(&p).is_err());
    }

    #[test]
    fn generated_prediction_refuses_nonfinite_output_instead_of_zero() {
        let (g, mut m) = fitted("(pipe X (ridge))");
        if let Model::Pipeline { head, .. } = &mut m {
            if let FittedHead::Ridge(r) = head.as_mut() {
                r.w.fill(f64::MAX);
                r.w[2] = -f64::MAX;
                r.scales.fill(1.0);
                r.means.fill(0.0);
            }
        }
        let source = emit(&g, &m, 12).unwrap();
        run_rust("nonfinite", &format!("{source}\nfn main() {{assert!(predict_numeric(&[vec![f64::MAX;12]]).is_err());println!(\"nonfinite output refused\");}}"));
    }
    #[test]
    fn tree_decision_flip_rate_is_zero_for_golden_and_random_rows() {
        use feat_math::boost2::BoostNode;
        use feat_math::heads::GbmNode;
        let (mut rows, _) = training();
        rows.extend([vec![-0.0; 12], vec![1.5; 12], vec![-0.5; 12]]);
        let mut seed = 731u64;
        rows.extend((0..1000).map(|_| {
            (0..12)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (seed >> 11) as f64 / ((1u64 << 53) as f64) * 2.0 - 0.5
                })
                .collect::<Vec<_>>()
        }));
        let mut program = String::new();
        let mut cases = 0;
        for dsl in [
            "(pipe X (gbm rounds=4 depth=2 min_leaf=2))",
            "(pipe X (histgbm2 rounds=4 leaves=4 min_leaf=2))",
        ] {
            let (genome, model) = fitted(dsl);
            let Model::Pipeline { head, .. } = &model else {
                unreachable!()
            };
            // Probe each real fitted tree separately, replacing only leaf values with their
            // node indices. This makes the actual engine/emitted routing decisions observable;
            // it is not a score/model-quality evaluation or a second routing implementation.
            let probes: Vec<FittedHead> = match head.as_ref() {
                FittedHead::Gbm(fitted) => fitted
                    .trees
                    .iter()
                    .map(|tree| {
                        let mut fitted = fitted.clone();
                        let mut tree = tree.clone();
                        for (i, node) in tree.iter_mut().enumerate() {
                            if let GbmNode::Leaf(value) = node {
                                *value = i as f64;
                            }
                        }
                        fitted.init = 0.0;
                        fitted.lr = 1.0;
                        fitted.trees = vec![tree];
                        FittedHead::Gbm(fitted)
                    })
                    .collect(),
                FittedHead::HistGbm2(fitted) => fitted
                    .bags
                    .iter()
                    .flatten()
                    .map(|tree| {
                        let mut fitted = fitted.clone();
                        let mut tree = tree.clone();
                        for (i, node) in tree.iter_mut().enumerate() {
                            if let BoostNode::Leaf(value) = node {
                                *value = i as f64;
                            }
                        }
                        fitted.init = 0.0;
                        fitted.lr = 1.0;
                        fitted.bags = vec![vec![tree]];
                        FittedHead::HistGbm2(fitted)
                    })
                    .collect(),
                _ => unreachable!(),
            };
            assert!(!probes.is_empty());
            for head in probes {
                let model = Model::Pipeline {
                    target: None,
                    ops: Vec::new(),
                    head: Box::new(head),
                };
                let source = emit(&genome, &model, 12).unwrap();
                let expected = model.predict(&rows).unwrap();
                let x = literal("", "Vec<Vec<f64>>", &json!(rows)).unwrap();
                let y = literal("", "Vec<f64>", &json!(expected)).unwrap();
                program.push_str(&format!("mod c{cases} {{\n{source}\n}}\nfn check{cases}() {{let rows={x};let expected:Vec<f64>={y};assert_eq!(c{cases}::predict_numeric(&rows).unwrap(),expected,\"leaf decision flip\");}}\n"));
                cases += 1;
            }
        }
        program.push_str("fn main(){");
        for i in 0..cases {
            program.push_str(&format!("check{i}();"));
        }
        program.push_str(&format!("println!(\"{cases} trees: 0 golden / 0 randomized row decision flips ({} row-tree decisions)\");}}",cases*rows.len()));
        run_rust("tree-decisions", &program);
    }

    #[test]
    fn emitted_self_checks_replay_caller_rows_and_reject_perturbed_outputs() {
        let (train, targets) = training();
        let genome = GenomeIr::parse("(pipe X (ridge))").unwrap();
        let model = feat_math::model::fit(&genome, &train, &targets).unwrap();
        let inputs = train[0].clone();
        let expected = model.predict(&vec![inputs.clone()]).unwrap()[0];
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/fitted-self-check-proof");
        std::fs::create_dir_all(&root).unwrap();

        let js_checks = [crate::fitted_js::CheckRow {
            inputs: inputs.clone(),
            expected,
        }];
        let js = crate::fitted_js::emit_with_checks(&genome, &model, 12, &js_checks).unwrap();
        let js_path = root.join("predictor.mjs");
        std::fs::write(&js_path, js).unwrap();
        assert!(Command::new("node")
            .arg(&js_path)
            .arg("--self-check")
            .status()
            .unwrap()
            .success());
        let bad_js = crate::fitted_js::emit_with_checks(
            &genome,
            &model,
            12,
            &[crate::fitted_js::CheckRow {
                inputs: inputs.clone(),
                expected: expected + 10.0,
            }],
        )
        .unwrap();
        std::fs::write(&js_path, bad_js).unwrap();
        assert!(!Command::new("node")
            .arg(&js_path)
            .arg("--self-check")
            .status()
            .unwrap()
            .success());

        let py_checks = [crate::fitted_py::CheckRow {
            inputs: inputs.clone(),
            expected,
        }];
        let py = crate::fitted_py::emit_with_checks(&genome, &model, 12, &py_checks).unwrap();
        let py_path = root.join("predictor.py");
        std::fs::write(&py_path, py).unwrap();
        assert!(Command::new("python3")
            .arg(&py_path)
            .arg("--self-check")
            .status()
            .unwrap()
            .success());
        let bad_py = crate::fitted_py::emit_with_checks(
            &genome,
            &model,
            12,
            &[crate::fitted_py::CheckRow {
                inputs: inputs.clone(),
                expected: expected + 10.0,
            }],
        )
        .unwrap();
        std::fs::write(&py_path, bad_py).unwrap();
        assert!(!Command::new("python3")
            .arg(&py_path)
            .arg("--self-check")
            .status()
            .unwrap()
            .success());

        let rs_checks = [CheckRow {
            inputs: inputs.clone(),
            expected,
        }];
        let rs = emit_with_checks(&genome, &model, 12, &rs_checks).unwrap();
        let rs_path = root.join("predictor.rs");
        let rs_test = root.join("predictor-test");
        std::fs::write(&rs_path, rs).unwrap();
        assert!(Command::new("rustc")
            .args(["--edition=2021", "--test"])
            .arg(&rs_path)
            .arg("-o")
            .arg(&rs_test)
            .status()
            .unwrap()
            .success());
        assert!(Command::new(&rs_test).status().unwrap().success());
        let bad_rs = emit_with_checks(
            &genome,
            &model,
            12,
            &[CheckRow {
                inputs,
                expected: expected + 10.0,
            }],
        )
        .unwrap();
        std::fs::write(&rs_path, bad_rs).unwrap();
        assert!(Command::new("rustc")
            .args(["--edition=2021", "--test"])
            .arg(&rs_path)
            .arg("-o")
            .arg(&rs_test)
            .status()
            .unwrap()
            .success());
        assert!(!Command::new(&rs_test).status().unwrap().success());
    }

    #[test]
    fn calibrated_emitters_match_engine_and_binding_floor() {
        let (train, targets) = training();
        let genome = GenomeIr::parse("(pipe X (ridge))").unwrap();
        let model = feat_math::model::fit(&genome, &train, &targets).unwrap();
        let raw = model.predict(&train).unwrap();
        let affine: Vec<_> = raw.iter().map(|v| 0.7 + 1.2 * v).collect();
        let (low_index, low_value) = affine
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        let (high_index, high_value) = affine
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        let lower = (low_value + high_value) / 2.0;
        let transform = OutputTransform {
            intercept: 0.7,
            slope: 1.2,
            lower: Some(lower),
        };
        let input_rows = [train[low_index].clone(), train[high_index].clone()];
        let checks_js: Vec<_> = input_rows
            .iter()
            .zip([low_index, high_index])
            .map(|(inputs, index)| crate::fitted_js::CheckRow {
                inputs: inputs.clone(),
                expected: transform.apply(raw[index]),
            })
            .collect();
        let checks_py: Vec<_> = input_rows
            .iter()
            .zip([low_index, high_index])
            .map(|(inputs, index)| crate::fitted_py::CheckRow {
                inputs: inputs.clone(),
                expected: transform.apply(raw[index]),
            })
            .collect();
        let checks_rs: Vec<_> = input_rows
            .iter()
            .zip([low_index, high_index])
            .map(|(inputs, index)| CheckRow {
                inputs: inputs.clone(),
                expected: transform.apply(raw[index]),
            })
            .collect();
        assert!(transform.apply(raw[low_index]) == lower);
        assert!(transform.apply(raw[high_index]) > lower);

        assert_eq!(
            crate::fitted_js::emit(&genome, &model, 12).unwrap(),
            crate::fitted_js::emit_calibrated(
                &genome,
                &model,
                12,
                Some(&OutputTransform::IDENTITY)
            )
            .unwrap()
        );
        assert_eq!(
            crate::fitted_py::emit(&genome, &model, 12).unwrap(),
            crate::fitted_py::emit_calibrated(
                &genome,
                &model,
                12,
                Some(&OutputTransform::IDENTITY)
            )
            .unwrap()
        );
        assert_eq!(
            emit(&genome, &model, 12).unwrap(),
            emit_calibrated(&genome, &model, 12, Some(&OutputTransform::IDENTITY)).unwrap()
        );

        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/fitted-calibration-proof");
        std::fs::create_dir_all(&root).unwrap();
        let js_path = root.join("predictor.mjs");
        std::fs::write(
            &js_path,
            crate::fitted_js::emit_calibrated_with_checks(
                &genome,
                &model,
                12,
                Some(&transform),
                &checks_js,
            )
            .unwrap(),
        )
        .unwrap();
        assert!(Command::new("node")
            .arg(&js_path)
            .arg("--self-check")
            .status()
            .unwrap()
            .success());
        let py_path = root.join("predictor.py");
        std::fs::write(
            &py_path,
            crate::fitted_py::emit_calibrated_with_checks(
                &genome,
                &model,
                12,
                Some(&transform),
                &checks_py,
            )
            .unwrap(),
        )
        .unwrap();
        assert!(Command::new("python3")
            .arg(&py_path)
            .arg("--self-check")
            .status()
            .unwrap()
            .success());
        let rs_path = root.join("predictor.rs");
        let rs_test = root.join("predictor-test");
        std::fs::write(
            &rs_path,
            emit_calibrated_with_checks(&genome, &model, 12, Some(&transform), &checks_rs).unwrap(),
        )
        .unwrap();
        assert!(Command::new("rustc")
            .args(["--edition=2021", "--test"])
            .arg(&rs_path)
            .arg("-o")
            .arg(&rs_test)
            .status()
            .unwrap()
            .success());
        assert!(Command::new(&rs_test).status().unwrap().success());
    }

    #[test]
    fn integral_calibration_constants_compile_and_match_engine() {
        let (train, targets) = training();
        let genome = GenomeIr::parse("(pipe X (ridge))").unwrap();
        let model = feat_math::model::fit(&genome, &train, &targets).unwrap();
        let raw = model.predict(&train).unwrap();
        let mut sorted = raw.clone();
        sorted.sort_by(f64::total_cmp);
        // An integral intercept that puts the zero floor inside the prediction range.
        let centre = -(4.0 * sorted[sorted.len() / 2]).round();
        let transforms = [
            // calibration::floor_only for a log-error objective: identity plus a zero floor.
            OutputTransform {
                intercept: 0.0,
                slope: 1.0,
                lower: Some(0.0),
            },
            // The shift family keeps a unit slope.
            OutputTransform {
                intercept: -1.0,
                slope: 1.0,
                lower: Some(-2.0),
            },
            OutputTransform {
                intercept: centre,
                slope: 4.0,
                lower: Some(0.0),
            },
            OutputTransform {
                intercept: 3.0,
                slope: 1.0,
                lower: None,
            },
            OutputTransform {
                intercept: -0.0,
                slope: 2.0,
                lower: Some(-0.0),
            },
        ];
        let binding = raw.iter().filter(|p| centre + 4.0 * **p < 0.0).count();
        assert!(
            binding > 0 && binding < raw.len(),
            "zero floor must bind on some rows"
        );
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/fitted-calibration-integral-proof");
        std::fs::create_dir_all(&root).unwrap();
        for (index, transform) in transforms.iter().enumerate() {
            let checks: Vec<_> = train
                .iter()
                .zip(&raw)
                .map(|(inputs, prediction)| CheckRow {
                    inputs: inputs.clone(),
                    expected: transform.apply(*prediction),
                })
                .collect();
            let source =
                emit_calibrated_with_checks(&genome, &model, 12, Some(transform), &checks).unwrap();
            let rs_path = root.join(format!("predictor-{index}.rs"));
            let rs_test = root.join(format!("predictor-{index}-test"));
            std::fs::write(&rs_path, &source).unwrap();
            assert!(
                Command::new("rustc")
                    .args(["--edition=2021", "--test"])
                    .arg(&rs_path)
                    .arg("-o")
                    .arg(&rs_test)
                    .status()
                    .unwrap()
                    .success(),
                "{transform:?} does not compile"
            );
            assert!(
                Command::new(&rs_test).status().unwrap().success(),
                "{transform:?} self-check failed"
            );
            if index == 0 {
                assert!(source.contains("(0.0 + 1.0 * pred).max(0.0)"));
            }
        }
    }

    #[test]
    fn platt_emitters_match_engine_probabilities_in_node_python_and_rustc() {
        let (genome, model) = fitted("(pipe X (ridge))");
        let mut calibration = crate::model_object::calibration_platt::cross_fit(
            &[0.1, 0.2, 0.8, 0.9],
            &["cold", "cold", "hot", "hot"],
            ["cold", "hot"],
            &[0, 1, 0, 1],
        )
        .unwrap()
        .calibration;
        calibration.sigmoid.scale = 0.7;
        calibration.sigmoid.center = 0.1;
        let scores = [0.0, 0.2, 0.75, 1.0];
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/fitted-platt-proof");
        std::fs::create_dir_all(&dir).unwrap();
        let no_calibration = ExportCalibration::default();
        assert_eq!(
            crate::fitted_js::emit_with_platt(&genome, &model, 12, &no_calibration).unwrap(),
            crate::fitted_js::emit(&genome, &model, 12).unwrap()
        );
        assert_eq!(
            crate::fitted_py::emit_with_platt(&genome, &model, 12, &no_calibration).unwrap(),
            crate::fitted_py::emit(&genome, &model, 12).unwrap()
        );
        assert_eq!(
            crate::fitted_rs::emit_with_platt(&genome, &model, 12, &no_calibration).unwrap(),
            crate::fitted_rs::emit(&genome, &model, 12).unwrap()
        );

        for (intercept, slope) in [(-0.35, 2.4), (1000.0, 2.4), (-1000.0, 0.0)] {
            calibration.sigmoid.intercept = intercept;
            calibration.sigmoid.slope = slope;
            let expected = scores
                .iter()
                .map(|score| calibration.apply(*score).unwrap())
                .collect::<Vec<_>>();
            let export = ExportCalibration {
                output: None,
                platt: Some(calibration.clone()),
            };

            let mut js = crate::fitted_js::emit_with_platt(&genome, &model, 12, &export).unwrap();
            assert!(js.contains("const PLATT_CLASS_ORDER=Object.freeze([\"cold\",\"hot\"])"));
            js.push_str(
                "\nfor(const s of [0,0.2,0.75,1]) console.log(JSON.stringify(fittedPlatt(s)));\n",
            );
            let js_path = dir.join("predictor.js");
            std::fs::write(&js_path, js).unwrap();
            let js_output = Command::new("node").arg(&js_path).output().unwrap();
            assert!(
                js_output.status.success(),
                "{}",
                String::from_utf8_lossy(&js_output.stderr)
            );
            assert_probability_pairs(&String::from_utf8_lossy(&js_output.stdout), &expected);

            let py = crate::fitted_py::emit_with_platt(&genome, &model, 12, &export).unwrap();
            assert!(py.contains("PLATT_CLASS_ORDER = [\"cold\",\"hot\"]"));
            let py_path = dir.join("predictor.py");
            std::fs::write(&py_path, py).unwrap();
            let py_output = Command::new("python3")
                .args([
                    "-c",
                    "import runpy,sys; m=runpy.run_path(sys.argv[1],run_name='platt_test'); [print(repr(m['fitted_platt'](s))) for s in [0,0.2,0.75,1]]",
                ])
                .arg(&py_path)
                .output()
                .unwrap();
            assert!(
                py_output.status.success(),
                "{}",
                String::from_utf8_lossy(&py_output.stderr)
            );
            assert_probability_pairs(&String::from_utf8_lossy(&py_output.stdout), &expected);

            let mut rs = crate::fitted_rs::emit_with_platt(&genome, &model, 12, &export).unwrap();
            assert!(rs.contains("pub const PLATT_CLASS_ORDER: [&str; 2] = [\"cold\", \"hot\"]"));
            rs.push_str("\nfn main() { for s in [0.0,0.2,0.75,1.0] { let p=fitted_platt(s).unwrap(); println!(\"{:?}\",p); } }\n");
            let rs_path = dir.join("predictor.rs");
            let rs_binary = dir.join("predictor");
            std::fs::write(&rs_path, rs).unwrap();
            let compile = Command::new("rustc")
                .args(["--edition=2021", "-D", "warnings"])
                .arg(&rs_path)
                .arg("-o")
                .arg(&rs_binary)
                .output()
                .unwrap();
            assert!(
                compile.status.success(),
                "{}",
                String::from_utf8_lossy(&compile.stderr)
            );
            let rs_output = Command::new(&rs_binary).output().unwrap();
            assert!(rs_output.status.success());
            assert_probability_pairs(&String::from_utf8_lossy(&rs_output.stdout), &expected);
        }
    }

    fn assert_probability_pairs(output: &str, expected_positive: &[f64]) {
        let rows = output.lines().collect::<Vec<_>>();
        assert_eq!(rows.len(), expected_positive.len(), "{output}");
        for (line, expected) in rows.iter().zip(expected_positive) {
            let values: Vec<f64> = line
                .trim_matches(|ch| ch == '[' || ch == ']' || ch == '(' || ch == ')')
                .split(|ch: char| ch == ',' || ch.is_whitespace())
                .filter(|part| !part.is_empty())
                .map(|part| part.parse().unwrap())
                .collect();
            assert_eq!(values.len(), 2, "{line}");
            assert!((values[0] - (1.0 - expected)).abs() < 1e-14, "{line}");
            assert!((values[1] - expected).abs() < 1e-14, "{line}");
        }
    }
}
