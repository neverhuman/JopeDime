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

#[cfg(test)]
mod tests;

mod pipeline;
pub use pipeline::emit_pipeline;
