//! Fitted JavaScript for a feat-math pipeline: the program's ES2020 recipe from
//! `feat_math::transpile::emit_javascript`, followed by the fitted state and a
//! `modelFittedHandle()` for the recipe's own `modelPredictNumericBatch`, so no caller retrains.
//!
//! The recipe declares its stages in `MODEL_STAGES` and its head in `MODEL_HEAD`; each fitted
//! stage must carry the same kind in the same position, and is written in the shape that kind's
//! `applyStage` branch reads. Anything outside the recipe's slice is refused by name.

use crate::calibration::OutputTransform;
use crate::error::Failure;
use anyhow::Result;
use feat_math::genome::GenomeIr;
use feat_math::heads::FittedHead;
use feat_math::model::Model;
use feat_math::ops::FittedOp;
use feat_math::transpile::{emit_javascript, lint_shipped_text, EmitOptions};
use serde_json::{json, Value};

/// Calibration options shared by the fitted language emitters.
///
/// The output transform is used for regression predictions; Platt maps native
/// positive-class probabilities and retains their explicit class order.
#[derive(Clone, Debug, Default)]
pub struct ExportCalibration {
    pub output: Option<OutputTransform>,
    pub platt: Option<crate::model_object::calibration_platt::Calibration>,
}

/// Engine output for one caller-supplied parity row.
#[derive(Clone, Debug)]
pub struct CheckRow {
    pub inputs: Vec<f64>,
    pub expected: f64,
}

/// Emit standalone ES2020 that predicts with `model`, fitted from `genome` on rows `width` wide.
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

/// Emit standalone JavaScript with the shared optional regression and Platt calibration.
pub fn emit_with_platt(
    genome: &GenomeIr,
    model: &Model,
    width: usize,
    calibration: &ExportCalibration,
) -> Result<String> {
    emit_with_checks_and_platt(genome, model, width, calibration, &[])
}

/// Emit JavaScript with shared calibration and caller-supplied engine check rows.
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
        platt
            .validate()
            .map_err(|e| Failure::integrity(format!("invalid Platt calibration: {e:#}")))?;
        let sigmoid = &platt.sigmoid;
        source.push_str(&format!(
            "\nconst PLATT_CLASS_ORDER=Object.freeze({});\n\
             function fittedPlatt(score) {{\n\
               if(!Number.isFinite(score) || score<0 || score>1) throw new Error(\"invalid Platt score\");\n\
               const z={}===0 ? {} : {} + {}*(score/{} - {});\n\
               let p; if(z>=0) p=1/(1+Math.exp(-z)); else {{const e=Math.exp(z); p=e/(1+e);}}\n\
               return [1-p,p];\n\
             }}\n",
            serde_json::to_string(&platt.class_order)?,
            js_number(sigmoid.slope),
            js_number(sigmoid.intercept),
            js_number(sigmoid.intercept),
            js_number(sigmoid.slope),
            js_number(sigmoid.scale),
            js_number(sigmoid.center),
        ));
    }
    let hits = lint_shipped_text(&source);
    if !hits.is_empty() {
        return Err(Failure::integrity(format!(
            "banned vocabulary in JavaScript artifact: {}",
            hits.join(", ")
        )));
    }
    Ok(source)
}

fn js_number(value: f64) -> String {
    serde_json::to_string(&value).expect("validated finite Platt coefficient")
}

/// Emit standalone JavaScript with the optional fitted regression output transform.
pub fn emit_calibrated(
    genome: &GenomeIr,
    model: &Model,
    width: usize,
    output_transform: Option<&OutputTransform>,
) -> Result<String> {
    emit_calibrated_with_checks(genome, model, width, output_transform, &[])
}

/// Emit standalone JavaScript with calibration and caller-supplied engine check rows.
pub fn emit_calibrated_with_checks(
    genome: &GenomeIr,
    model: &Model,
    width: usize,
    output_transform: Option<&OutputTransform>,
    checks: &[CheckRow],
) -> Result<String> {
    if checks.len() > 64 || checks.iter().any(|row| row.inputs.len() != width) {
        return Err(Failure::integrity("invalid JavaScript self-check rows"));
    }
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
    let (mut source, use_extended_head) = match emit_javascript(genome, &EmitOptions::default()) {
        Ok(src) => (src, false),
        Err(first_err) => match stage_only_ridge_genome(genome) {
            Some(stage_genome) if supports_extended_js_head(head) => {
                let src = emit_javascript(&stage_genome, &EmitOptions::default())
                    .map_err(|e| Failure::backend(format!("{e:#}")))?;
                (src, true)
            }
            _ => return Err(Failure::backend(format!("{first_err:#}"))),
        },
    };
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
    let head_state = if use_extended_head {
        match extended_js_head_state(genome, head) {
            Ok(st) => st,
            Err(msg) => {
                problems.push(msg);
                Value::Null
            }
        }
    } else {
        match (head.as_ref(), planned_head.as_slice()) {
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
    if use_extended_head {
        source.push_str(EXTENDED_JS_HEAD_RUNTIME);
    }
    let model_ctor = if use_extended_head {
        "new ExtendedFittedModel(FITTED_STATES,FITTED_HEAD)"
    } else {
        "new Model(FITTED_STATES,FITTED_HEAD)"
    };
    let output_transform = output_transform
        .copied()
        .filter(|t| !is_identity_transform(t));
    if let Some(transform) = output_transform {
        if !transform.intercept.is_finite()
            || !transform.slope.is_finite()
            || transform.lower.is_some_and(|lower| !lower.is_finite())
        {
            return Err(Failure::integrity("non-finite regression output transform"));
        }
        let lower = transform
            .lower
            .map_or("null".to_owned(), |v| format!("{v:?}"));
        source.push_str(&format!(r#"
// Fitted state: prediction only. Regression output calibration.
const FITTED_STATES=Object.freeze({});
const FITTED_HEAD=Object.freeze({});
function modelFittedHandle(){{const base={model_ctor};return {{model:{{predict(rows){{return base.predict(rows).map(pred=>{{const value={intercept:?}+{slope:?}*pred;const calibrated={lower}===null?value:(Number.isNaN(value)?{lower}:Math.max(value,{lower}));if(!Number.isFinite(calibrated))throw new Error("calibrated prediction is not finite");return calibrated;}});}}}},width:{width}}};}}
"#, serde_json::to_string(&Value::Array(states))?, serde_json::to_string(&head_state)?, model_ctor=model_ctor, intercept=transform.intercept, slope=transform.slope, lower=lower, width=width));
    } else {
        source.push_str(&format!(
            "\n// Fitted state. Prediction only: these constants are what the pipeline learned.\n\
         const FITTED_STATES=Object.freeze({});\n\
         const FITTED_HEAD=Object.freeze({});\n\
         function modelFittedHandle(){{return {{model:{model_ctor},width:{width}}};}}\n",
            serde_json::to_string(&Value::Array(states))?,
            serde_json::to_string(&head_state)?,
            model_ctor = model_ctor,
        ));
    }
    let rows = checks
        .iter()
        .map(|row| {
            let inputs = row
                .inputs
                .iter()
                .map(|v| format!("0x{:016x}n", v.to_bits()))
                .collect::<Vec<_>>()
                .join(",");
            format!(
                "{{inputs:[{inputs}],expected:0x{:016x}n}}",
                row.expected.to_bits()
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    source.push_str(&format!(r#"
const SELF_CHECK_ROWS=[{rows}];
const SELF_CHECK_ABS_TOL=1e-9, SELF_CHECK_REL_TOL=1e-7;
function self_check() {{
  if(SELF_CHECK_ROWS.length===0) return true;
  const buffer=new ArrayBuffer(8), view=new DataView(buffer);
  const fromBits=(bits)=>{{view.setBigUint64(0,bits);return view.getFloat64(0);}};
  const h=modelFittedHandle(), flat=new Float64Array(SELF_CHECK_ROWS.flatMap(r=>r.inputs.map(fromBits)));
  const out=new Float64Array(SELF_CHECK_ROWS.length);
  const status=modelPredictNumericBatch(h,flat,flat.length,SELF_CHECK_ROWS.length,{width},null,0,out,out.length);
  for(let i=0;i<SELF_CHECK_ROWS.length;i++){{
    const expected=fromBits(SELF_CHECK_ROWS[i].expected), actual=out[i];
    if(status!==0 || !(Number.isNaN(expected)&&Number.isNaN(actual)) &&
       !(Number.isFinite(actual)&&Math.abs(actual-expected)<=SELF_CHECK_ABS_TOL+SELF_CHECK_REL_TOL*Math.abs(expected)))
      throw new Error(`self_check failed at row ${{i}}`);
  }}
  return true;
}}
  if(typeof process!=="undefined" && process.argv?.[2]==="--self-check") {{ self_check(); }}
"#, width = width));
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

fn is_identity_transform(transform: &OutputTransform) -> bool {
    transform.intercept == 0.0 && transform.slope == 1.0 && transform.lower.is_none()
}

pub(crate) fn stage_only_ridge_genome(genome: &GenomeIr) -> Option<GenomeIr> {
    let GenomeIr::Pipeline(pipe) = genome else {
        return None;
    };
    let mut clone = pipe.clone();
    let last = clone.stages.last_mut()?;
    *last = feat_math::dsl::Stage {
        op: "ridge".to_owned(),
        params: std::collections::BTreeMap::new(),
    };
    Some(GenomeIr::Pipeline(clone))
}

fn supports_extended_js_head(head: &FittedHead) -> bool {
    matches!(
        head,
        FittedHead::Gbm(_)
            | FittedHead::HistGbm2(_)
            | FittedHead::Ensemble { .. }
            | FittedHead::RidgeGcv(_)
            | FittedHead::Krr(_)
            | FittedHead::Gpr(_)
            | FittedHead::GprKernel(_)
            | FittedHead::Ksmooth(_)
            | FittedHead::Stack(_)
            | FittedHead::Hurdle2(_)
    )
}

fn serialize_gbm_json(g: &feat_math::heads::GbmModel) -> Value {
    use feat_math::heads::GbmNode;
    let trees: Vec<Value> = g
        .trees
        .iter()
        .map(|t| {
            Value::Array(
                t.iter()
                    .map(|n| match n {
                        GbmNode::Leaf(v) => json!([-1, 0, 0, 0, *v]),
                        GbmNode::Split {
                            feature,
                            thr_bin,
                            left,
                            right,
                        } => json!([*feature, usize::from(*thr_bin), *left, *right, 0.0]),
                    })
                    .collect(),
            )
        })
        .collect();
    json!({
        "kind": "gbm",
        "edges": g.edges,
        "trees": trees,
        "init": g.init,
        "lr": g.lr,
        "n_feat": g.n_feat,
        "constant_value": g.constant_value,
    })
}

fn serialize_histgbm2_json(h: &feat_math::boost2::HistGbm2Model) -> Value {
    use feat_math::boost2::BoostNode;
    let bags: Vec<Value> = h
        .bags
        .iter()
        .map(|bag| {
            Value::Array(
                bag.iter()
                    .map(|tree| {
                        Value::Array(
                            tree.iter()
                                .map(|n| match n {
                                    BoostNode::Leaf(v) => json!([-1, 0.0, 0, 0, *v]),
                                    BoostNode::Split {
                                        feature,
                                        threshold,
                                        left,
                                        right,
                                    } => json!([*feature, *threshold, *left, *right, 0.0]),
                                })
                                .collect(),
                        )
                    })
                    .collect(),
            )
        })
        .collect();
    json!({
        "kind": "histgbm2",
        "init": h.init,
        "lr": h.lr,
        "bags": bags,
    })
}

fn serialize_gpr_kernel_json(
    g: &feat_math::gpr::GprKernelModel,
) -> std::result::Result<Value, String> {
    use feat_math::gpr::KernelPrim;
    let width = g.means.len();
    let mut offset = 0usize;
    let mut terms = Vec::with_capacity(g.spec.len());
    for term in &g.spec {
        let amp = *g
            .theta
            .get(offset)
            .ok_or_else(|| "gpr_kernel theta truncated at amplitude".to_owned())?;
        let signal = (2.0 * amp).exp();
        offset += 1;
        let mut factors = Vec::with_capacity(term.len());
        for prim in term {
            match prim {
                KernelPrim::Rbf => {
                    let mut inv = Vec::with_capacity(width);
                    for d in 0..width {
                        let l = g
                            .theta
                            .get(offset + d)
                            .ok_or_else(|| "gpr_kernel theta truncated at rbf".to_owned())?
                            .exp();
                        inv.push(1.0 / (l * l));
                    }
                    offset += width;
                    factors.push(json!({ "op": "rbf", "inv": inv }));
                }
                KernelPrim::Rq => {
                    let mut inv = Vec::with_capacity(width);
                    for d in 0..width {
                        let l = g
                            .theta
                            .get(offset + d)
                            .ok_or_else(|| "gpr_kernel theta truncated at rq".to_owned())?
                            .exp();
                        inv.push(1.0 / (l * l));
                    }
                    let shape = g
                        .theta
                        .get(offset + width)
                        .ok_or_else(|| "gpr_kernel theta truncated at rq alpha".to_owned())?
                        .exp();
                    offset += width + 1;
                    factors.push(json!({ "op": "rq", "inv": inv, "shape": shape }));
                }
                KernelPrim::Periodic => {
                    let period = g
                        .theta
                        .get(offset)
                        .ok_or_else(|| "gpr_kernel theta truncated at period".to_owned())?
                        .exp();
                    let l = g
                        .theta
                        .get(offset + 1)
                        .ok_or_else(|| "gpr_kernel theta truncated at periodic ls".to_owned())?
                        .exp();
                    offset += 2;
                    factors
                        .push(json!({ "op": "periodic", "period": period, "inv": 1.0 / (l * l) }));
                }
                KernelPrim::Linear => {
                    factors.push(json!({ "op": "linear", "width": width.max(1) }));
                }
            }
        }
        terms.push(json!({ "signal": signal, "factors": factors }));
    }
    if g.theta.get(offset).is_none() {
        return Err("gpr_kernel theta truncated at observation noise".to_owned());
    }
    Ok(json!({
        "kind": "gpr_kernel",
        "means": g.means,
        "scales": g.scales,
        "basis": g.basis,
        "alpha": g.alpha,
        "y_mean": g.y_mean,
        "clamp_lo": g.clamp_lo,
        "clamp_hi": g.clamp_hi,
        "terms": terms,
    }))
}

pub(crate) fn extended_js_head_state(
    genome: &GenomeIr,
    head: &FittedHead,
) -> std::result::Result<Value, String> {
    use feat_math::heads::ConditionalRidge;
    use feat_math::spec::{build_plan, Plan, PlannedHead};
    let plan = build_plan(genome).map_err(|m| m.join("; "))?;
    let Plan::Pipeline {
        head: planned_head, ..
    } = &plan
    else {
        return Err("expected pipeline plan".to_owned());
    };
    match (planned_head, head) {
        (PlannedHead::Gbm(_), FittedHead::Gbm(g)) => Ok(serialize_gbm_json(g)),
        (PlannedHead::HistGbm2 { .. }, FittedHead::HistGbm2(h)) => Ok(serialize_histgbm2_json(h)),
        (PlannedHead::Blend { .. }, FittedHead::Ensemble { ridge, gbm, w }) => Ok(json!({
            "kind": "ensemble",
            "w": *w,
            "ridge": { "means": ridge.means, "scales": ridge.scales, "w": ridge.w },
            "gbm": serialize_gbm_json(gbm),
        })),
        (PlannedHead::RidgeGcv, FittedHead::RidgeGcv(r)) => Ok(json!({
            "kind": "ridge_gcv",
            "cols": r.cols,
            "means": r.means,
            "scales": r.scales,
            "w": r.w,
        })),
        (PlannedHead::Krr { .. }, FittedHead::Krr(k)) => Ok(json!({
            "kind": "krr",
            "means": k.means,
            "scales": k.scales,
            "basis": k.basis,
            "alpha": k.alpha,
            "gamma": k.gamma,
            "y_mean": k.y_mean,
        })),
        (PlannedHead::Gpr { .. }, FittedHead::Gpr(g)) => Ok(json!({
            "kind": "gpr",
            "means": g.means,
            "scales": g.scales,
            "basis": g.basis,
            "alpha": g.alpha,
            "ls": g.ls,
            "sf2": g.sf2,
            "y_mean": g.y_mean,
            "clamp_lo": g.clamp_lo,
            "clamp_hi": g.clamp_hi,
        })),
        (PlannedHead::GprKernel { .. }, FittedHead::GprKernel(g)) => serialize_gpr_kernel_json(g),
        (PlannedHead::Ksmooth { k, gamma }, FittedHead::Ksmooth(m)) => {
            if *k != m.k || !(m.gamma.is_finite() && m.gamma.to_bits() == gamma.to_bits()) {
                return Err("ksmooth fitted parameters do not match planned head".to_owned());
            }
            Ok(json!({
                "kind": "ksmooth",
                "means": m.means,
                "scales": m.scales,
                "z": m.z,
                "ys": m.ys,
                "k": m.k,
                "gamma": m.gamma,
                "y_mean": m.y_mean,
            }))
        }
        (PlannedHead::Stack, FittedHead::Stack(s)) => {
            let mut members = Vec::new();
            let slots: [(Option<Value>, usize); 5] = [
                (
                    s.ridge_a.as_ref().map(|r| {
                        json!({ "kind": "ridge", "means": r.means, "scales": r.scales, "w": r.w })
                    }),
                    0,
                ),
                (
                    s.ridge_b.as_ref().map(|r| {
                        json!({ "kind": "ridge", "means": r.means, "scales": r.scales, "w": r.w })
                    }),
                    1,
                ),
                (
                    s.net.as_ref().map(|e| {
                        json!({ "kind": "elasticnet", "means": e.means, "scales": e.scales, "w0": e.w0, "w": e.w })
                    }),
                    2,
                ),
                (s.trees.as_ref().map(serialize_histgbm2_json), 3),
                (
                    s.kernel.as_ref().map(|k| {
                        json!({
                            "kind": "krr",
                            "means": k.means,
                            "scales": k.scales,
                            "basis": k.basis,
                            "alpha": k.alpha,
                            "gamma": k.gamma,
                            "y_mean": k.y_mean,
                        })
                    }),
                    4,
                ),
            ];
            for (opt_m, idx) in slots {
                if let Some(m_val) = opt_m {
                    let w = s.weights.get(idx).copied().unwrap_or(0.0);
                    members.push(json!({ "weight": w, "model": m_val }));
                }
            }
            Ok(json!({ "kind": "stack", "members": members }))
        }
        (_, FittedHead::Hurdle2(h)) => {
            let enc_cond = |c: &ConditionalRidge| -> Value {
                match c {
                    ConditionalRidge::Ridge(r) => {
                        json!({ "kind": "ridge", "means": r.means, "scales": r.scales, "w": r.w })
                    }
                    ConditionalRidge::Mean(v) => json!({ "kind": "mean", "value": *v }),
                }
            };
            let comps: Vec<Value> = h
                .components
                .iter()
                .map(|c| {
                    json!({
                        "gate": { "means": c.gate.means, "scales": c.gate.scales, "w": c.gate.w },
                        "low": enc_cond(&c.low),
                        "high": enc_cond(&c.high),
                    })
                })
                .collect();
            Ok(json!({
                "kind": "hurdle2",
                "fallback_mean": h.fallback_mean,
                "components": comps,
            }))
        }
        _ => Err("fitted head does not match planned head".to_owned()),
    }
}

const EXTENDED_JS_HEAD_RUNTIME: &str = r#"
function predictGbmHead(state,x){
  if(state.n_feat===0)return x.map(()=>state.constant_value);
  return x.map(row=>{
    const bins=new Uint16Array(state.n_feat);
    for(let f=0;f<state.n_feat;f++){
      const v=f<row.length?row[f]:0,edges=state.edges[f];
      let lo=0,hi=edges.length;
      while(lo<hi){const mid=Math.trunc((lo+hi)/2);if(edges[mid]<v)lo=mid+1;else hi=mid;}
      bins[f]=lo;
    }
    let acc=state.init;
    for(const tree of state.trees){
      let idx=0;
      while(true){
        const node=tree[idx];
        if(node[0]<0){acc+=state.lr*node[4];break;}
        idx=bins[node[0]]<=node[1]?node[2]:node[3];
      }
    }
    return acc;
  });
}
function predictHistGbm2Head(state,x){
  return x.map(row=>{
    if(state.bags.length===0)return state.init;
    let total=0;
    for(const bag of state.bags){
      let s=0;
      for(const tree of bag){
        let idx=0;
        while(true){
          const node=tree[idx];
          if(node[0]<0){s+=node[4];break;}
          const v=node[0]<row.length?row[node[0]]:0;
          idx=v<=node[1]?node[2]:node[3];
        }
      }
      total+=s;
    }
    return state.init+state.lr*total/state.bags.length;
  });
}
function predictRidgeClamped(state,x){
  const p=state.means.length;
  return x.map(row=>{
    let z=state.w[0];
    const lim=Math.min(p,row.length);
    for(let j=0;j<lim;j++){
      const raw=(row[j]-state.means[j])/state.scales[j];
      const xs=raw<-20?-20:(raw>20?20:raw);
      z+=state.w[j+1]*xs;
    }
    return z<-60?-60:(z>60?60:z);
  });
}
function predictExtendedHead(state,x){
  if(state.kind==="gbm")return predictGbmHead(state,x);
  if(state.kind==="histgbm2")return predictHistGbm2Head(state,x);
  if(state.kind==="ensemble"){
    const r=predictRidgeClamped(state.ridge,x),g=predictGbmHead(state.gbm,x);
    return r.map((rv,i)=>state.w*rv+(1-state.w)*g[i]);
  }
  if(state.kind==="ridge_gcv"){
    return x.map(row=>{
      let z=state.w[0];
      for(let k=0;k<state.cols.length;k++){
        const j=state.cols[k];
        if(j<row.length)z+=state.w[k+1]*((row[j]-state.means[k])/state.scales[k]);
      }
      return z<-60?-60:(z>60?60:z);
    });
  }
  if(state.kind==="krr"){
    return x.map(row=>{
      if(state.basis.length===0)return state.y_mean<-60?-60:(state.y_mean>60?60:state.y_mean);
      const p=Math.min(state.means.length,row.length),zRow=new Array(p);
      for(let j=0;j<p;j++)zRow[j]=(row[j]-state.means[j])/state.scales[j];
      let s=state.y_mean;
      for(let i=0;i<state.basis.length;i++){
        let d2=0;const b=state.basis[i],lim=Math.min(zRow.length,b.length);
        for(let j=0;j<lim;j++){const d=zRow[j]-b[j];d2+=d*d;}
        s+=state.alpha[i]*mExp(-state.gamma*d2);
      }
      return s<-60?-60:(s>60?60:s);
    });
  }
  if(state.kind==="gpr"){
    const invLs2=state.ls.map(l=>1/(l*l));
    return x.map(row=>{
      const lo=state.clamp_lo,hi=state.clamp_hi;
      if(state.basis.length===0)return state.y_mean<lo?lo:(state.y_mean>hi?hi:state.y_mean);
      const p=Math.min(state.means.length,row.length),zRow=new Array(p);
      for(let j=0;j<p;j++)zRow[j]=(row[j]-state.means[j])/state.scales[j];
      let s=state.y_mean;
      for(let i=0;i<state.basis.length;i++){
        let r2=0;const b=state.basis[i],lim=Math.min(zRow.length,b.length);
        for(let j=0;j<lim;j++){const d=zRow[j]-b[j];r2+=d*d*invLs2[j];}
        s+=state.alpha[i]*state.sf2*mExp(-0.5*r2);
      }
      return s<lo?lo:(s>hi?hi:s);
    });
  }
  if(state.kind==="ksmooth"){
    const p=state.means.length,pf=p;
    return x.map(row=>{
      if(state.z.length===0)return state.y_mean<-60?-60:(state.y_mean>60?60:state.y_mean);
      const limP=Math.min(p,row.length),zRow=new Array(limP);
      for(let j=0;j<limP;j++)zRow[j]=(row[j]-state.means[j])/state.scales[j];
      const dists=state.z.map((b,idx)=>{
        let d2=0;const lim=Math.min(zRow.length,b.length);
        for(let j=0;j<lim;j++){const d=zRow[j]-b[j];d2+=d*d;}
        return [d2,idx];
      });
      dists.sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
      const kk=Math.min(state.k,dists.length),base=dists[0][0];
      let num=0,den=0;
      for(let i=0;i<kk;i++){
        const w=mExp(-state.gamma*(dists[i][0]-base)/pf);
        num+=w*state.ys[dists[i][1]];den+=w;
      }
      const out=num/den;
      return out<-60?-60:(out>60?60:out);
    });
  }
  if(state.kind==="gpr_kernel"){
    return x.map(row=>{
      const lo=state.clamp_lo,hi=state.clamp_hi;
      if(state.basis.length===0)return state.y_mean<lo?lo:(state.y_mean>hi?hi:state.y_mean);
      const p=Math.min(state.means.length,row.length),zRow=new Array(p);
      for(let j=0;j<p;j++)zRow[j]=(row[j]-state.means[j])/state.scales[j];
      let s=state.y_mean;
      for(let i=0;i<state.basis.length;i++){
        const b=state.basis[i],lim=Math.min(zRow.length,b.length);
        let kSum=0;
        for(const t of state.terms){
          let term=t.signal;
          for(const f of t.factors){
            if(f.op==="rbf"){
              let dist=0;for(let d=0;d<lim;d++){const q=zRow[d]-b[d];dist+=q*q*f.inv[d];}
              term*=mExp(-0.5*dist);
            }else if(f.op==="rq"){
              let dist=0;for(let d=0;d<lim;d++){const q=zRow[d]-b[d];dist+=q*q*f.inv[d];}
              term*=mExp(-f.shape*Math.log(1+dist/(2*f.shape)));
            }else if(f.op==="periodic"){
              let s2=0;for(let d=0;d<lim;d++){const sn=Math.sin(Math.PI*(zRow[d]-b[d])/f.period);s2+=sn*sn;}
              term*=mExp(-2*s2*f.inv);
            }else if(f.op==="linear"){
              let prod=0;for(let d=0;d<lim;d++)prod+=zRow[d]*b[d];
              term*=prod/f.width;
            }
          }
          kSum+=term;
        }
        s+=state.alpha[i]*kSum;
      }
      return s<lo?lo:(s>hi?hi:s);
    });
  }
  if(state.kind==="ridge")return predictRidgeClamped(state,x);
  if(state.kind==="elasticnet"){
    const p=state.means.length;
    return x.map(row=>{
      let z=state.w0;const lim=Math.min(p,row.length);
      for(let j=0;j<lim;j++){
        const raw=(row[j]-state.means[j])/state.scales[j];
        z+=state.w[j]*(raw<-20?-20:(raw>20?20:raw));
      }
      return z<-60?-60:(z>60?60:z);
    });
  }
  if(state.kind==="stack"){
    if(state.members.length===0)return x.map(()=>0);
    const acc=new Array(x.length).fill(0);
    for(const m of state.members){
      const preds=predictExtendedHead(m.model,x);
      for(let i=0;i<x.length;i++)acc[i]+=m.weight*preds[i];
    }
    return acc;
  }
  if(state.kind==="hurdle2"){
    if(state.components.length===0)return x.map(()=>state.fallback_mean);
    const evalCond=(c,row)=>{
      if(c.kind==="mean")return c.value;
      let z=c.w[0];const lim=Math.min(c.means.length,row.length);
      for(let j=0;j<lim;j++){
        const raw=(row[j]-c.means[j])/c.scales[j];
        z+=c.w[j+1]*(raw<-20?-20:(raw>20?20:raw));
      }
      return z;
    };
    return x.map(row=>{
      let v=0;
      for(const c of state.components){
        let gz=c.gate.w[0];const lim=Math.min(c.gate.means.length,row.length);
        for(let j=0;j<lim;j++){
          const raw=(row[j]-c.gate.means[j])/c.gate.scales[j];
          gz+=c.gate.w[j+1]*(raw<-20?-20:(raw>20?20:raw));
        }
        const gzc=gz<-60?-60:(gz>60?60:gz);
        const prob=gzc>=0?1/(1+mExp(-gzc)):(mExp(gzc)/(1+mExp(gzc)));
        v+=prob*evalCond(c.high,row)+(1-prob)*evalCond(c.low,row);
      }
      v/=state.components.length;
      return Number.isFinite(v)?v:state.fallback_mean;
    });
  }
  throw new Error("unsupported fitted head kind");
}
class ExtendedFittedModel {
  constructor(states,head){this.states=states;this.head=head;}
  predict(xIn){
    let x=sanitize(xIn);
    for(let i=0;i<MODEL_STAGES.length;i++)x=applyStage(MODEL_STAGES[i],this.states[i],x);
    return predictExtendedHead(this.head,x);
  }
}
"#;
