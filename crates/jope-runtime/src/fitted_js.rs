//! Fitted JavaScript for a feat-math pipeline: the program's ES2020 recipe from
//! `feat_math::transpile::emit_javascript`, followed by the fitted state and a
//! `modelFittedHandle()` for the recipe's own `modelPredictNumericBatch`, so no caller retrains.
//!
//! The recipe declares its stages in `MODEL_STAGES` and its head in `MODEL_HEAD`; each fitted
//! stage must carry the same kind in the same position, and is written in the shape that kind's
//! `applyStage` branch reads. Anything outside the recipe's slice is refused by name.

use crate::error::Failure;
use anyhow::Result;
use feat_math::genome::GenomeIr;
use feat_math::heads::FittedHead;
use feat_math::model::Model;
use feat_math::ops::FittedOp;
use feat_math::transpile::{emit_javascript, lint_shipped_text, EmitOptions};
use serde_json::{json, Value};

/// Emit standalone ES2020 that predicts with `model`, fitted from `genome` on rows `width` wide.
pub fn emit(genome: &GenomeIr, model: &Model, width: usize) -> Result<String> {
    let Model::Pipeline {
        target: None,
        ops,
        head,
    } = model
    else {
        return Err(Failure::backend(
            "fitted JavaScript needs a pipeline without a target transform",
        ));
    };
    feat_math::kernel::width::check_input_width(model, width)
        .map_err(|e| Failure::integrity(format!("fitted JavaScript input width: {e}")))?;
    let mut source = emit_javascript(genome, &EmitOptions::default())
        .map_err(|e| Failure::backend(format!("{e:#}")))?;
    let planned_stages = declared_kinds(&source, "const MODEL_STAGES=")?;
    let planned_head = declared_kinds(&source, "const MODEL_HEAD=")?;
    if planned_stages.len() != ops.len() {
        return Err(Failure::backend(format!(
            "fitted JavaScript refused: {} fitted stages for {} planned stages",
            ops.len(),
            planned_stages.len()
        )));
    }
    let mut states = Vec::with_capacity(ops.len());
    let mut problems = Vec::new();
    for (planned, op) in planned_stages.iter().zip(ops) {
        match stage_state(op) {
            Some((kind, state)) => {
                if kind != planned {
                    problems.push(format!("fitted {kind} where the plan has {planned}"));
                }
                states.push(state);
            }
            None => problems.push(format!(
                "no JavaScript state for the fitted stage at {planned}"
            )),
        }
    }
    let head_state = match (head.as_ref(), planned_head.as_slice()) {
        (FittedHead::Ridge(r), [kind]) if kind == "ridge" => {
            json!({"means": r.means, "scales": r.scales, "w": r.w})
        }
        (FittedHead::ElasticNet(e), [kind]) if kind == "elasticnet" => {
            json!({"means": e.means, "scales": e.scales, "w0": e.w0, "w": e.w})
        }
        _ => {
            problems.push(format!(
                "fitted head does not match planned head {planned_head:?}"
            ));
            Value::Null
        }
    };
    if !problems.is_empty() {
        return Err(Failure::backend(format!(
            "fitted JavaScript refused: {}",
            problems.join(", ")
        )));
    }
    validate_prediction_parameters(&source, ops)?;
    // serde_json writes the shortest round-trip form, so every f64 parses back bit-exactly. It
    // writes a non-finite f64 as null, so a null anywhere inside a state is refused; only a whole
    // stateless stage is null.
    if !states.iter().all(|s| s.is_null() || all_finite(s)) || !all_finite(&head_state) {
        return Err(Failure::integrity("non-finite fitted state"));
    }
    source.push_str(&format!(
        "\n// Fitted state. Prediction only: these constants are what the pipeline learned.\n\
         const FITTED_STATES=Object.freeze({});\n\
         const FITTED_HEAD=Object.freeze({});\n\
         function modelFittedHandle(){{return {{model:new Model(FITTED_STATES,FITTED_HEAD),width:{width}}};}}\n",
        serde_json::to_string(&Value::Array(states))?,
        serde_json::to_string(&head_state)?,
    ));
    let hits = lint_shipped_text(&source);
    if !hits.is_empty() {
        return Err(Failure::integrity(format!(
            "banned vocabulary in JavaScript artifact: {}",
            hits.join(", ")
        )));
    }
    Ok(source)
}

/// Check every recipe literal read by applyStage during prediction. Learned arrays
/// come from the fitted model; fit-only parameters (seed, regularization, etc.) do
/// not affect prediction and cannot be recovered from every fitted operator.
fn validate_prediction_parameters(source: &str, ops: &[FittedOp]) -> Result<()> {
    let line = source
        .lines()
        .find(|line| line.starts_with("const MODEL_STAGES="))
        .ok_or_else(|| Failure::integrity("recipe has no MODEL_STAGES line"))?;
    // stage_js emits flat object literals. Read its resolved defaults/clamps,
    // rather than duplicating the planner's parameter normalization here.
    let specs: Vec<_> = line
        .split('{')
        .skip(1)
        .map(|tail| {
            tail.split('}')
                .next()
                .unwrap_or("")
                .split(',')
                .filter_map(|field| field.split_once(':'))
                .collect::<std::collections::BTreeMap<_, _>>()
        })
        .collect();
    if specs.len() != ops.len() {
        return Err(Failure::integrity("recipe stage literals are incomplete"));
    }
    for (index, (spec, op)) in specs.iter().zip(ops).enumerate() {
        use FittedOp as F;
        let append = match op {
            F::Standardize { append, .. }
            | F::RobustScale { append, .. }
            | F::Rank { append, .. }
            | F::Blom { append, .. }
            | F::QuantileBins { append, .. }
            | F::BinOnehot { append, .. }
            | F::Hinge { append, .. }
            | F::WinsorClip { append, .. }
            | F::Cheb { append, .. }
            | F::PolyInteract { append, .. }
            | F::RffCos { append, .. }
            | F::RffSin { append, .. }
            | F::RandProj { append, .. }
            | F::LandmarkDist { append, .. }
            | F::ObliqueHinge { append, .. }
            | F::RowStats { append }
            | F::PairProd { append, .. }
            | F::TargetBinEncode { append, .. }
            | F::Sigmoid { append }
            | F::Step { append, .. }
            | F::NormRow { append }
            | F::RowDiffStats { append } => *append,
            _ => return Err(Failure::backend("unsupported fitted JavaScript stage")),
        };
        if spec.get("append").copied() != Some(if append { "true" } else { "false" }) {
            return Err(Failure::backend(format!(
                "fitted JavaScript refused: stage {index} append parameter differs from fitted state"
            )));
        }
        let parameter = match op {
            F::QuantileBins { bins, .. } | F::BinOnehot { bins, .. } => {
                Some(("bins", *bins as f64))
            }
            F::Cheb { degree, .. } => Some(("degree", *degree as f64)),
            F::LandmarkDist { gamma, .. } => Some(("gamma", *gamma)),
            F::Step { t, .. } => Some(("t", *t)),
            _ => None,
        };
        if let Some((name, fitted)) = parameter {
            let planned = spec.get(name).and_then(|value| value.parse::<f64>().ok());
            if !fitted.is_finite()
                || planned.is_none_or(|value| value.to_bits() != fitted.to_bits())
            {
                return Err(Failure::backend(format!(
                    "fitted JavaScript refused: stage {index} {name} parameter differs from fitted state"
                )));
            }
        }
    }
    Ok(())
}

/// The `kind:"…"` values, in order, on the recipe line that starts with `prefix`.
fn declared_kinds(source: &str, prefix: &str) -> Result<Vec<String>> {
    let line = source
        .lines()
        .find(|line| line.starts_with(prefix))
        .ok_or_else(|| Failure::integrity(format!("recipe has no {prefix:?} line")))?;
    Ok(line
        .split("kind:\"")
        .skip(1)
        .filter_map(|rest| rest.split('"').next())
        .map(str::to_owned)
        .collect())
}

/// The kind the recipe names a fitted stage by, and the state its `applyStage` reads.
fn stage_state(op: &FittedOp) -> Option<(&'static str, Value)> {
    use FittedOp as F;
    Some(match op {
        F::Standardize { means, scales, .. } => {
            ("standardize", json!({"means": means, "scales": scales}))
        }
        F::RobustScale {
            medians, scales, ..
        } => (
            "robust_scale",
            json!({"medians": medians, "scales": scales}),
        ),
        F::Rank { sorted_cols, .. } => ("rank", json!({"sorted": sorted_cols})),
        F::Blom { sorted_cols, .. } => ("blom", json!({"sorted": sorted_cols})),
        F::QuantileBins { edges, .. } => ("quantile_bins", json!({"edges": edges})),
        F::BinOnehot { edges, .. } => ("bin_onehot", json!({"edges": edges})),
        F::Hinge { knots, .. } => ("hinge", json!({"knots": knots})),
        F::WinsorClip { lo, hi, .. } => ("winsor_clip", json!({"lo": lo, "hi": hi})),
        F::Cheb { .. } => ("cheb", Value::Null),
        F::PolyInteract { pairs, .. } => ("poly_interact", json!({"pairs": pairs})),
        F::RffCos { w, b, .. } => ("rff_cos", json!({"w": w, "b": b})),
        F::RffSin { w, b, .. } => ("rff_sin", json!({"w": w, "b": b})),
        F::RandProj { w, .. } => ("rand_proj", json!({"w": w})),
        F::LandmarkDist { landmarks, .. } => ("landmark_dist", json!({"landmarks": landmarks})),
        F::ObliqueHinge { w, t, .. } => ("oblique_hinge", json!({"w": w, "t": t})),
        F::RowStats { .. } => ("row_stats", Value::Null),
        F::PairProd { pairs, .. } => ("pair_prod", json!({"pairs": pairs})),
        F::TargetBinEncode { edges, enc, .. } => {
            ("target_bin_encode", json!({"edges": edges, "enc": enc}))
        }
        F::Sigmoid { .. } => ("sigmoid", Value::Null),
        F::Step { .. } => ("step", Value::Null),
        F::NormRow { .. } => ("norm_row", Value::Null),
        F::RowDiffStats { .. } => ("row_diff_stats", Value::Null),
        _ => return None,
    })
}

fn all_finite(value: &Value) -> bool {
    match value {
        Value::Number(n) => n.as_f64().is_some_and(f64::is_finite),
        Value::Array(items) => items.iter().all(all_finite),
        Value::Object(map) => map.values().all(all_finite),
        Value::Null => false,
        _ => true,
    }
}
