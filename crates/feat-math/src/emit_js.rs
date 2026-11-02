//! Standalone ES2020 emitter for the independently certified fitted-model slice.

#[path = "emit_js/core_runtime.rs"]
mod core_runtime;
use core_runtime::CORE;

use crate::mathexpr::fmt_f64;
use crate::spec::{Plan, PlannedHead, PlannedOp, PlannedStage};
use crate::transpile::{EmitOptions, MathKernel};

/// Emit a self-contained ES2020 model, or return every unsupported plan element.
pub fn emit_js(plan: &Plan, opts: &EmitOptions) -> Result<String, Vec<String>> {
    let (stages, head) = match plan {
        Plan::Pipeline {
            target: None,
            stages,
            head,
        } => (stages, head),
        Plan::Pipeline {
            target: Some(_), ..
        } => return Err(vec!["target transforms".to_string()]),
        Plan::Symbolic { .. } => return Err(vec!["symbolic formula programs".to_string()]),
    };

    let mut unsupported = Vec::new();
    for stage in stages {
        if !op_supported(&stage.op) {
            unsupported.push(format!("transform {}", op_label(&stage.op)));
        }
    }
    if !head_supported(head) {
        unsupported.push(format!("head {}", head_label(head)));
    }
    if !unsupported.is_empty() {
        return Err(unsupported);
    }

    let mut source = String::with_capacity(48_000);
    source.push_str(PREAMBLE);
    source.push('\n');
    if opts.math_kernel == MathKernel::Portable {
        let Some(body) = portable_math_body() else {
            return Err(vec![
                "verified Portable JavaScript helper fixture".to_string()
            ]);
        };
        source.push_str(body);
        source.push('\n');
        source.push_str(PORTABLE_MATH_BINDINGS);
    } else {
        source.push_str(PLATFORM_MATH_BINDINGS);
    }
    source.push('\n');
    if opts.partition_point_quantile_bounds {
        let needle = "function lowerBound(s,v){let lo=0,hi=s.length;while(lo<hi){const mid=Math.trunc((lo+hi)/2);if(s[mid]<v)lo=mid+1;else hi=mid;}return lo;}\nfunction upperBound(s,v){let lo=0,hi=s.length;while(lo<hi){const mid=Math.trunc((lo+hi)/2);if(s[mid]<=v)lo=mid+1;else hi=mid;}return lo;}";
        assert_eq!(
            CORE.matches(needle).count(),
            1,
            "CORE template must contain the legacy lowerBound/upperBound needle exactly once"
        );
        let core = CORE.replacen(
            needle,
            "function lowerBound(s,v){let size=s.length;if(size===0)return 0;let base=0;while(size>1){const half=Math.trunc(size/2),mid=base+half;if(s[mid]<v)base=mid;size-=half;}return base+(s[base]<v?1:0);}\nfunction upperBound(s,v){let size=s.length;if(size===0)return 0;let base=0;while(size>1){const half=Math.trunc(size/2),mid=base+half;if(s[mid]<=v)base=mid;size-=half;}return base+(s[base]<=v?1:0);}",
            1,
        );
        source.push_str(&core);
    } else {
        source.push_str(CORE);
    }
    source.push('\n');
    source.push_str(&model_js(stages, head));
    source.push('\n');
    source.push_str(NUMERIC_BOUNDARY);
    source.push('\n');
    Ok(source)
}

fn op_supported(op: &PlannedOp) -> bool {
    matches!(
        op,
        PlannedOp::Standardize
            | PlannedOp::RobustScale
            | PlannedOp::Rank
            | PlannedOp::Blom
            | PlannedOp::QuantileBins { .. }
            | PlannedOp::BinOnehot { .. }
            | PlannedOp::Hinge { .. }
            | PlannedOp::WinsorClip { .. }
            | PlannedOp::Cheb { .. }
            | PlannedOp::PolyInteract
            | PlannedOp::RffCos { .. }
            | PlannedOp::RffSin { .. }
            | PlannedOp::RandProj { .. }
            | PlannedOp::LandmarkDist { .. }
            | PlannedOp::ObliqueHinge { .. }
            | PlannedOp::RowStats
            | PlannedOp::PairProd { .. }
            | PlannedOp::TargetBinEncode { .. }
            | PlannedOp::Sigmoid
            | PlannedOp::Step { .. }
            | PlannedOp::NormRow
            | PlannedOp::RowDiffStats
    )
}

fn op_label(op: &PlannedOp) -> &'static str {
    match op {
        // B05–B07: JS remains outside the scalar-v2 backend slice.
        PlannedOp::ScalarV2 { .. } => "scalar_expression_v2",
        PlannedOp::RowVectorV2 { .. } | PlannedOp::RowVectorInputV2 { .. } => "feature_row_v2",
        PlannedOp::Standardize => "standardize",
        PlannedOp::RobustScale => "robust_scale",
        PlannedOp::Rank => "rank",
        PlannedOp::Blom => "blom",
        PlannedOp::QuantileBins { .. } => "quantile_bins",
        PlannedOp::BinOnehot { .. } => "bin_onehot",
        PlannedOp::Hinge { .. } => "hinge",
        PlannedOp::WinsorClip { .. } => "winsor_clip",
        PlannedOp::Cheb { .. } => "cheb",
        PlannedOp::PolyInteract => "poly_interact",
        PlannedOp::RffCos { .. } => "rff_cos",
        PlannedOp::RffSin { .. } => "rff_sin",
        PlannedOp::RandProj { .. } => "rand_proj",
        PlannedOp::LandmarkDist { .. } => "landmark_dist",
        PlannedOp::ObliqueHinge { .. } => "oblique_hinge",
        PlannedOp::RowStats => "row_stats",
        PlannedOp::PairProd { .. } => "pair_prod",
        PlannedOp::TargetBinEncode { .. } => "target_bin_encode",
        PlannedOp::Sigmoid => "sigmoid",
        PlannedOp::Step { .. } => "step",
        PlannedOp::NormRow => "norm_row",
        PlannedOp::RowDiffStats => "row_diff_stats",
        PlannedOp::Polar => "polar",
        PlannedOp::Lag { .. } => "lag",
        PlannedOp::Roll { .. } => "roll",
        PlannedOp::FftMag { .. } => "fft_mag",
        PlannedOp::RowSpectral => "row_spectral",
        PlannedOp::SvdProject { .. } => "svd_project",
        PlannedOp::Nystrom { .. } => "nystrom",
        PlannedOp::MapExpr { .. } => "map_expr",
        PlannedOp::DeriveExpr { .. } => "derive_expr",
        PlannedOp::Symlog { .. } => "symlog",
        PlannedOp::SelectCorr { .. } => "select_corr",
        PlannedOp::Impute { .. } => "impute",
        PlannedOp::PairRatio { .. } => "pair_ratio",
        PlannedOp::Spline { .. } => "spline",
    }
}

fn head_supported(head: &PlannedHead) -> bool {
    matches!(
        head,
        PlannedHead::Ridge { .. } | PlannedHead::ElasticNet { .. }
    )
}

fn head_label(head: &PlannedHead) -> &'static str {
    match head {
        PlannedHead::Ridge { .. } => "ridge",
        PlannedHead::RidgeGcv => "ridge_gcv",
        PlannedHead::ElasticNet { .. } => "elasticnet",
        PlannedHead::Gbm(_) => "gbm",
        PlannedHead::Blend { .. } => "blend",
        PlannedHead::HistGbm2 { .. } => "histgbm2",
        PlannedHead::Krr { .. } => "krr",
        PlannedHead::Gpr { .. } => "gpr",
        PlannedHead::GprKernel { .. } => "gpr_kernel",
        PlannedHead::Ksmooth { .. } => "ksmooth",
        PlannedHead::Stack => "stack",
        PlannedHead::Portfolio(_) => "portfolio",
    }
}

fn stage_js(stage: &PlannedStage) -> String {
    let append = if stage.append { "true" } else { "false" };
    match &stage.op {
        PlannedOp::Standardize => format!(r#"{{kind:"standardize",append:{append}}}"#),
        PlannedOp::RobustScale => format!(r#"{{kind:"robust_scale",append:{append}}}"#),
        PlannedOp::Rank => format!(r#"{{kind:"rank",append:{append}}}"#),
        PlannedOp::Blom => format!(r#"{{kind:"blom",append:{append}}}"#),
        PlannedOp::QuantileBins { bins } => {
            format!(r#"{{kind:"quantile_bins",bins:{bins},append:{append}}}"#)
        }
        PlannedOp::BinOnehot { bins } => {
            format!(r#"{{kind:"bin_onehot",bins:{bins},append:{append}}}"#)
        }
        PlannedOp::Hinge { knots } => format!(r#"{{kind:"hinge",knots:{knots},append:{append}}}"#),
        PlannedOp::WinsorClip { q } => format!(
            r#"{{kind:"winsor_clip",q:{},append:{append}}}"#,
            fmt_f64(*q)
        ),
        PlannedOp::Cheb { degree } => format!(r#"{{kind:"cheb",degree:{degree},append:{append}}}"#),
        PlannedOp::PolyInteract => format!(r#"{{kind:"poly_interact",append:{append}}}"#),
        PlannedOp::RffCos { n, gamma, seed } => format!(
            r#"{{kind:"rff_cos",n:{n},gamma:{},seed:{seed}n,append:{append}}}"#,
            fmt_f64(*gamma)
        ),
        PlannedOp::RffSin { n, gamma, seed } => format!(
            r#"{{kind:"rff_sin",n:{n},gamma:{},seed:{seed}n,append:{append}}}"#,
            fmt_f64(*gamma)
        ),
        PlannedOp::RandProj { k, seed } => {
            format!(r#"{{kind:"rand_proj",k:{k},seed:{seed}n,append:{append}}}"#)
        }
        PlannedOp::LandmarkDist { k, gamma, seed } => format!(
            r#"{{kind:"landmark_dist",k:{k},gamma:{},seed:{seed}n,append:{append}}}"#,
            fmt_f64(*gamma)
        ),
        PlannedOp::ObliqueHinge { n, seed } => {
            format!(r#"{{kind:"oblique_hinge",n:{n},seed:{seed}n,append:{append}}}"#)
        }
        PlannedOp::RowStats => format!(r#"{{kind:"row_stats",append:{append}}}"#),
        PlannedOp::PairProd { topk } => {
            format!(r#"{{kind:"pair_prod",topk:{topk},append:{append}}}"#)
        }
        PlannedOp::TargetBinEncode { bins, prior } => format!(
            r#"{{kind:"target_bin_encode",bins:{bins},prior:{},append:{append}}}"#,
            fmt_f64(*prior)
        ),
        PlannedOp::Sigmoid => format!(r#"{{kind:"sigmoid",append:{append}}}"#),
        PlannedOp::Step { t } => format!(r#"{{kind:"step",t:{},append:{append}}}"#, fmt_f64(*t)),
        PlannedOp::NormRow => format!(r#"{{kind:"norm_row",append:{append}}}"#),
        PlannedOp::RowDiffStats => format!(r#"{{kind:"row_diff_stats",append:{append}}}"#),
        _ => unreachable!("stage_js called after the support gate"),
    }
}

fn model_js(stages: &[PlannedStage], head: &PlannedHead) -> String {
    let stages = stages.iter().map(stage_js).collect::<Vec<_>>().join(",");
    let head = match head {
        PlannedHead::Ridge { lambda_frac } => {
            format!(r#"{{kind:"ridge",lambdaFrac:{}}}"#, fmt_f64(*lambda_frac))
        }
        PlannedHead::ElasticNet {
            alpha,
            l1_ratio,
            sweeps,
        } => format!(
            r#"{{kind:"elasticnet",alpha:{},l1Ratio:{},sweeps:{sweeps}}}"#,
            fmt_f64(*alpha),
            fmt_f64(*l1_ratio)
        ),
        _ => unreachable!("model_js called after the support gate"),
    };
    format!(
        r#"const MODEL_STAGES=Object.freeze([{stages}]);
const MODEL_HEAD=Object.freeze({head});

class Model {{
  constructor(states,head){{ this.states=states; this.head=head; }}
  static fit(xIn,y){{
    if(xIn.length===0||y.length!==xIn.length) throw new RangeError("invalid training shape");
    let x=sanitize(xIn),states=[];
    for(const spec of MODEL_STAGES){{ const fitted=fitStage(spec,x,y); states.push(fitted[0]); x=fitted[1]; }}
    return new Model(states,fitHead(MODEL_HEAD,x,y));
  }}
  predict(xIn){{
    let x=sanitize(xIn);
    for(let i=0;i<MODEL_STAGES.length;i++) x=applyStage(MODEL_STAGES[i],this.states[i],x);
    return predictHead(MODEL_HEAD,this.head,x);
  }}
}}"#
    )
}

const VERIFIED_PORTABLE_JS: &str = include_str!("../tests/portable/full_ref.js");

fn portable_math_body() -> Option<&'static str> {
    let (prefix, _) = VERIFIED_PORTABLE_JS.split_once("\nfunction grid(){")?;
    let start = prefix.find("'use strict';")?;
    Some(prefix[start..].trim())
}

#[cfg(test)]
pub(crate) fn portable_math_matches_verified_fixture() -> bool {
    let Some(body) = portable_math_body() else {
        return false;
    };
    body.starts_with("'use strict';")
        && !body.contains("function grid()")
        && !body.contains("process.stdout")
}

const PREAMBLE: &str = r#"// Standalone regression model. All input and output crosses typed numeric boundaries.
'use strict';
"#;

const PLATFORM_MATH_BINDINGS: &str = r#"function mExp(x){return Math.exp(x);}
function mLog(x){return Math.log(x);}
function mSin(x){return Math.sin(x);}
function mCos(x){return Math.cos(x);}"#;

const PORTABLE_MATH_BINDINGS: &str = r#"function mExp(x){return xexp(x);}
function mLog(x){return xln(x);}
function mSin(x){return xsin(x);}
function mCos(x){return xcos(x);}"#;

const NUMERIC_BOUNDARY: &str = r#"
function shapeIsSafe(n){return Number.isSafeInteger(n)&&n>=0;}
function exactFlatShape(values,valuesLen,rows,columns){
  return values instanceof Float64Array&&shapeIsSafe(valuesLen)&&shapeIsSafe(rows)&&shapeIsSafe(columns)&&columns>0&&
    rows<=Number.MAX_SAFE_INTEGER/columns&&valuesLen===rows*columns&&values.length===valuesLen;
}
function flatRows(values,rows,columns){const x=[];for(let r=0;r<rows;r++){const row=[];for(let c=0;c<columns;c++)row.push(values[r*columns+c]);x.push(row);}return x;}
function modelAbiVersion(){return 1;}
function modelWorkspaceBytes(_handle,_rows){return 0;}
function modelFitNumeric(values,valuesLen,rows,columns,targets,targetLen){
  if(!exactFlatShape(values,valuesLen,rows,columns)||rows===0||!(targets instanceof Float64Array)||!shapeIsSafe(targetLen)||targetLen!==rows||targets.length!==targetLen)return {status:1,handle:null};
  for(let i=0;i<targetLen;i++)if(!Number.isFinite(targets[i]))return {status:1,handle:null};
  try{return {status:0,handle:{model:Model.fit(flatRows(values,rows,columns),Array.from(targets)),width:columns}};}catch(_error){return {status:2,handle:null};}
}
function modelPredictNumericBatch(handle,values,valuesLen,rows,columns,_workspace,workspaceBytes,out,outLen){
  if(handle===null||typeof handle!=="object"||!shapeIsSafe(columns)||columns===0||columns!==handle.width||!shapeIsSafe(workspaceBytes))return 1;
  if(!(out instanceof Float64Array)||!shapeIsSafe(outLen)||outLen!==rows||out.length!==outLen)return 1;
  if(rows===0)return values instanceof Float64Array&&valuesLen===0&&values.length===0?0:1;
  if(!exactFlatShape(values,valuesLen,rows,columns))return 1;
  try{const predicted=handle.model.predict(flatRows(values,rows,columns));if(predicted.length!==rows)return 2;for(let r=0;r<rows;r++)out[r]=predicted[r];return 0;}catch(_error){return 2;}
}
function modelUnload(handle){if(handle!==null&&typeof handle==="object"){handle.model=null;handle.width=0;}}
"#;
