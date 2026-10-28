//! Standalone ES2020 emitter for the independently certified fitted-model slice.

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
    let mut core = if opts.partition_point_quantile_bounds {
        let needle = "function lowerBound(s,v){let lo=0,hi=s.length;while(lo<hi){const mid=Math.trunc((lo+hi)/2);if(s[mid]<v)lo=mid+1;else hi=mid;}return lo;}\nfunction upperBound(s,v){let lo=0,hi=s.length;while(lo<hi){const mid=Math.trunc((lo+hi)/2);if(s[mid]<=v)lo=mid+1;else hi=mid;}return lo;}";
        assert_eq!(
            CORE.matches(needle).count(),
            1,
            "CORE template must contain the legacy lowerBound/upperBound needle exactly once"
        );
        CORE.replacen(
            needle,
            "function lowerBound(s,v){let size=s.length;if(size===0)return 0;let base=0;while(size>1){const half=Math.trunc(size/2),mid=base+half;if(s[mid]<v)base=mid;size-=half;}return base+(s[base]<v?1:0);}\nfunction upperBound(s,v){let size=s.length;if(size===0)return 0;let base=0;while(size>1){const half=Math.trunc(size/2),mid=base+half;if(s[mid]<=v)base=mid;size-=half;}return base+(s[base]<=v?1:0);}",
            1,
        )
    } else {
        CORE.to_owned()
    };
    if matches!(
        head,
        PlannedHead::RidgeGcv | PlannedHead::Gbm(_) | PlannedHead::Ksmooth { .. }
    ) {
        let head_dispatch_needle = r#"function fitHead(spec,x,y){if(spec.kind==="ridge")return ridgeFit(x,y,spec.lambdaFrac);if(spec.kind==="elasticnet")return elasticFit(x,y,spec.alpha,spec.l1Ratio,spec.sweeps);throw new RangeError("unsupported fitted head");}
function predictHead(spec,state,x){if(spec.kind==="ridge")return ridgePredict(state,x);if(spec.kind==="elasticnet")return elasticPredict(state,x);throw new RangeError("unsupported applied head");}"#;
        assert_eq!(
            core.matches(head_dispatch_needle).count(),
            1,
            "CORE template must contain the legacy fitHead/predictHead needle exactly once"
        );
        core = core.replacen(head_dispatch_needle, EXTRA_HEADS_JS, 1);
    }
    source.push_str(&core);
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
        PlannedHead::Ridge { .. }
            | PlannedHead::RidgeGcv
            | PlannedHead::ElasticNet { .. }
            | PlannedHead::Gbm(_)
            | PlannedHead::Ksmooth { .. }
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
        PlannedHead::RidgeGcv => r#"{kind:"ridge_gcv"}"#.to_owned(),
        PlannedHead::ElasticNet {
            alpha,
            l1_ratio,
            sweeps,
        } => format!(
            r#"{{kind:"elasticnet",alpha:{},l1Ratio:{},sweeps:{sweeps}}}"#,
            fmt_f64(*alpha),
            fmt_f64(*l1_ratio)
        ),
        PlannedHead::Gbm(g) => format!(
            r#"{{kind:"gbm",rounds:{},depth:{},lr:{},lambda:{},bins:{},minLeaf:{},maxFeatures:{}}}"#,
            g.rounds,
            g.depth,
            fmt_f64(g.lr),
            fmt_f64(g.lambda),
            g.bins,
            g.min_leaf,
            g.max_features
        ),
        PlannedHead::Ksmooth { k, gamma } => format!(
            r#"{{kind:"ksmooth",k:{k},gamma:{}}}"#,
            fmt_f64(*gamma)
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

const CORE: &str = r#"
function clampRange(x,lo,hi){if(x<lo)x=lo;if(x>hi)x=hi;return x;}
function maxNum(a,b){if(Number.isNaN(a))return b;if(Number.isNaN(b))return a;return a<b?b:a;}
function totalCompare(a,b){
  if(a<b)return -1;if(a>b)return 1;
  if(Object.is(a,-0)&&Object.is(b,0))return -1;
  if(Object.is(a,0)&&Object.is(b,-0))return 1;
  return 0;
}
function sanitize(x){return x.map(row=>Array.from(row,v=>Number.isFinite(v)?v:0.0));}
function finishStage(x,out,append){
  for(let r=0;r<out.length;r++){
    let row=append?x[r].concat(out[r]):out[r];
    if(row.length>512)row.length=512;
    for(let j=0;j<row.length;j++)if(!Number.isFinite(row[j]))row[j]=0.0;
    if(row.length===0)row.push(0.0);
    out[r]=row;
  }
  return out;
}
function stdFit(x){
  const n=Math.max(x.length,1),p=x[0].length,means=new Array(p).fill(0.0),scales=new Array(p).fill(1.0);
  for(let j=0;j<p;j++){
    let s=0.0;for(const row of x)s+=row[j];means[j]=s/n;
    let v=0.0;for(const row of x){const d=row[j]-means[j];v+=d*d;}
    scales[j]=maxNum(Math.sqrt(v/n),1e-9);
  }
  return {means,scales};
}
function choleskySolve(a,b,p){
  for(let i=0;i<p;i++)for(let j=0;j<=i;j++){
    let s=a[i*p+j];for(let k=0;k<j;k++)s-=a[i*p+k]*a[j*p+k];
    a[i*p+j]=i===j?(s>1e-12?Math.sqrt(s):1e-6):s/a[j*p+j];
  }
  const z=new Array(p).fill(0.0);
  for(let i=0;i<p;i++){let s=b[i];for(let k=0;k<i;k++)s-=a[i*p+k]*z[k];z[i]=s/a[i*p+i];}
  const w=new Array(p).fill(0.0);
  for(let ii=0;ii<p;ii++){const i=p-1-ii;let s=z[i];for(let k=i+1;k<p;k++)s-=a[k*p+i]*w[k];w[i]=s/a[i*p+i];}
  return w;
}
function sortedColumn(x,j){const out=x.map(row=>row[j]);out.sort(totalCompare);return out;}
function quantile(s,q){if(s.length===0)return 0.0;const pos=q*(s.length-1),lo=Math.trunc(pos),hi=Math.min(lo+1,s.length-1),f=pos-lo;return s[lo]*(1.0-f)+s[hi]*f;}
function lowerBound(s,v){let lo=0,hi=s.length;while(lo<hi){const mid=Math.trunc((lo+hi)/2);if(s[mid]<v)lo=mid+1;else hi=mid;}return lo;}
function upperBound(s,v){let lo=0,hi=s.length;while(lo<hi){const mid=Math.trunc((lo+hi)/2);if(s[mid]<=v)lo=mid+1;else hi=mid;}return lo;}
function midRank(s,v){return (lowerBound(s,v)+upperBound(s,v))/(2.0*s.length);}
function probit(p){
  const a=[-3.969683028665376e1,2.209460984245205e2,-2.759285104469687e2,1.383577518672690e2,-3.066479806614716e1,2.506628277459239],
    b=[-5.447609879822406e1,1.615858368580409e2,-1.556989798598866e2,6.680131188771972e1,-1.328068155288572e1],
    c=[-7.784894002430293e-3,-3.223964580411365e-1,-2.400758277161838,-2.549732539343734,4.374664141464968,2.938163982698783],
    d=[7.784695709041462e-3,3.224671290700398e-1,2.445134137142996,3.754408661907416];
  p=clampRange(p,1e-12,1.0-1e-12);
  if(p<0.02425){const q=Math.sqrt(-2.0*mLog(p));return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);}
  if(p<=1.0-0.02425){const q=p-0.5,r=q*q;return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);}
  const q=Math.sqrt(-2.0*mLog(1.0-p));return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
}
function mSigmoid(v){return 1.0/(1.0+mExp(-clampRange(v,-30.0,30.0)));}
function dot(a,b){let s=0.0;for(let i=0;i<Math.min(a.length,b.length);i++)s+=a[i]*b[i];return s;}
function pearson(a,b){
  const n=Math.min(a.length,b.length);if(n<2)return 0.0;
  let ma=0.0,mb=0.0;for(let i=0;i<n;i++){ma+=a[i];mb+=b[i];}ma/=n;mb/=n;
  let sab=0.0,saa=0.0,sbb=0.0;for(let i=0;i<n;i++){const da=a[i]-ma,db=b[i]-mb;sab+=da*db;saa+=da*da;sbb+=db*db;}
  return saa<=0.0||sbb<=0.0?0.0:sab/Math.sqrt(saa*sbb);
}
function quantileEdges(x,bins){const out=[];for(let j=0;j<x[0].length;j++){const s=sortedColumn(x,j),e=[];for(let b=1;b<bins;b++)e.push(quantile(s,b/bins));out.push(e);}return out;}

const PCG_MASK64=0xffffffffffffffffn;
function rotr32(x,r){return ((x>>>r)|(x<<((32-r)&31)))>>>0;}
class Pcg32{
  constructor(seed,stream){this.state=0n;this.inc=((stream<<1n)|1n)&PCG_MASK64;this.nextU32();this.state=(this.state+seed)&PCG_MASK64;this.nextU32();}
  nextU32(){const old=this.state;this.state=(old*6364136223846793005n+this.inc)&PCG_MASK64;const x=Number((((old>>18n)^old)>>27n)&0xffffffffn)>>>0,r=Number((old>>59n)&31n);return rotr32(x,r);}
  nextF64(){const bits=(BigInt(this.nextU32())<<32n)|BigInt(this.nextU32());return Number(bits>>11n)*(1.0/9007199254740992.0);}
  rangeF64(a,b){return a+(b-a)*this.nextF64();}
  gaussian(){const u1=maxNum(this.nextF64(),1e-12),u2=this.nextF64();return Math.sqrt(-2.0*mLog(u1))*mCos(2.0*3.141592653589793*u2);}
}
function stageRng(seed,inCols,nOut){return new Pcg32(seed,(BigInt(inCols)<<32n)^BigInt(nOut));}

function standardizeFit(x){return stdFit(x);}
function standardizeApply(s,x,a){return finishStage(x,x.map(row=>row.map((v,j)=>clampRange((v-s.means[j])/s.scales[j],-20.0,20.0))),a);}
function robustScaleFit(x){const medians=[],scales=[];for(let j=0;j<x[0].length;j++){const s=sortedColumn(x,j);medians.push(quantile(s,0.5));scales.push(maxNum(quantile(s,0.75)-quantile(s,0.25),1e-9));}return {medians,scales};}
function robustScaleApply(s,x,a){return finishStage(x,x.map(row=>row.map((v,j)=>(v-s.medians[j])/s.scales[j])),a);}
function rankFit(x){const sorted=[];for(let j=0;j<x[0].length;j++)sorted.push(sortedColumn(x,j));return {sorted};}
function rankApply(s,x,a){return finishStage(x,x.map(row=>row.map((v,j)=>midRank(s.sorted[j],v))),a);}
function blomApply(s,x,a){return finishStage(x,x.map(row=>row.map((v,j)=>{const n=s.sorted[j].length,mid=midRank(s.sorted[j],v)*n;return probit((mid+0.125)/(n+0.25));})),a);}
function winsorFit(x,q){const lo=[],hi=[];for(let j=0;j<x[0].length;j++){const s=sortedColumn(x,j),low=quantile(s,q);lo.push(low);hi.push(maxNum(quantile(s,1.0-q),low));}return {lo,hi};}
function winsorApply(s,x,a){return finishStage(x,x.map(row=>row.map((v,j)=>clampRange(v,s.lo[j],s.hi[j]))),a);}
function sigmoidApply(x,a){return finishStage(x,x.map(row=>row.map(mSigmoid)),a);}
function stepApply(x,t,a){return finishStage(x,x.map(row=>row.map(v=>v>t?1.0:0.0)),a);}
function normRowApply(x,a){return finishStage(x,x.map(row=>{let s=0.0;for(const v of row)s+=v*v;const inv=1.0/maxNum(Math.sqrt(s),1e-9);return row.map(v=>v*inv);}),a);}
function chebApply(x,degree,a){return finishStage(x,x.map(row=>{const out=[];for(const v of row){const u=clampRange(2.0*v-1.0,-1.0,1.0);let prev=1.0,cur=u;for(let d=0;d<degree;d++){out.push(cur);const next=2.0*u*cur-prev;prev=cur;cur=next;}}return out;}),a);}
function rffFit(x,n,gamma,seed){const rng=stageRng(seed,x[0].length,n),scale=Math.sqrt(2.0*gamma),w=[],b=[];for(let i=0;i<n;i++){const row=[];for(let j=0;j<x[0].length;j++)row.push(rng.gaussian()*scale);w.push(row);}for(let i=0;i<n;i++)b.push(rng.rangeF64(0.0,2.0*3.141592653589793));return {w,b};}
function rffApply(s,x,a,trig){const scale=Math.sqrt(2.0/s.w.length);return finishStage(x,x.map(row=>s.w.map((w,i)=>scale*trig(dot(w,row)+s.b[i]))),a);}

function hingeFit(x,knots){const fitted=[];for(let j=0;j<x[0].length;j++){const s=sortedColumn(x,j),col=[];for(let i=1;i<=knots;i++)col.push(quantile(s,i/(knots+1)));fitted.push(col);}return {knots:fitted};}
function hingeApply(s,x,a){return finishStage(x,x.map(row=>{const out=[];for(let j=0;j<row.length;j++)for(const t of s.knots[j])out.push(maxNum(row[j]-t,0.0));return out;}),a);}
function polyFit(x){const pairs=[],cols=x[0].length;for(let i=0;i<cols;i++)pairs.push([i,i]);let full=false;for(let i=0;i<cols&&!full;i++)for(let j=i+1;j<cols;j++){if(pairs.length>=512){full=true;break;}pairs.push([i,j]);}return {pairs};}
function pairsApply(s,x,a){return finishStage(x,x.map(row=>s.pairs.map(pair=>row[pair[0]]*row[pair[1]])),a);}
function quantileBinsApply(s,x,bins,a){return finishStage(x,x.map(row=>row.map((v,j)=>lowerBound(s.edges[j],v)/(bins-1.0))),a);}
function onehotApply(s,x,bins,a){return finishStage(x,x.map(row=>{const out=new Array(row.length*bins).fill(0.0);for(let j=0;j<row.length;j++)out[j*bins+Math.min(lowerBound(s.edges[j],row[j]),bins-1)]=1.0;return out;}),a);}
function pairProdFit(x,y,topk){
  const scored=[],cols=x[0].length,k=Math.min(topk,cols);
  for(let j=0;j<cols;j++)scored.push([j,Math.abs(pearson(x.map(row=>row[j]),y))]);
  scored.sort((a,b)=>{if(totalCompare(b[1],a[1])<0)return -1;if(totalCompare(a[1],b[1])<0)return 1;return a[0]-b[0];});
  const selected=scored.slice(0,k).map(v=>v[0]).sort((a,b)=>a-b),pairs=[];
  for(let i=0;i<selected.length;i++)for(let j=i+1;j<selected.length;j++)pairs.push([selected[i],selected[j]]);
  return {pairs};
}
function randProjFit(x,k,seed){const rng=stageRng(seed,x[0].length,k),w=[];for(let i=0;i<k;i++){const row=[];for(let j=0;j<x[0].length;j++)row.push(rng.gaussian());w.push(row);}return {w};}
function randProjApply(s,x,a){const inv=1.0/Math.sqrt(s.w.length);return finishStage(x,x.map(row=>s.w.map(w=>dot(w,row)*inv)),a);}
function rowStatsApply(x,a){return finishStage(x,x.map(row=>{
  const n=row.length;let mean=0.0;for(const v of row)mean+=v;mean/=n;
  let m2=0.0,m3=0.0,m4=0.0,energy=0.0,mn=Infinity,mx=-Infinity;
  for(const v of row){const d=v-mean;m2+=d*d;m3+=d*d*d;m4+=d*d*d*d;energy+=v*v;if(v<mn)mn=v;if(v>mx)mx=v;}
  m2/=n;m3/=n;m4/=n;energy/=n;const sd=Math.sqrt(m2),skew=m2>1e-24?m3/(m2*sd):0.0,kurt=m2>1e-24?m4/(m2*m2)-3.0:0.0,s=Array.from(row);s.sort(totalCompare);
  return [mean,sd,mn,mx,quantile(s,0.5),quantile(s,0.25),quantile(s,0.75),skew,kurt,energy];
}),a);}
function rowDiffApply(x,a){return finishStage(x,x.map(row=>{
  const d=row.length;if(d<2)return new Array(8).fill(0.0);const nd=d-1,diffs=[];for(let j=0;j<nd;j++)diffs.push(row[j+1]-row[j]);
  let mean=0.0;for(const v of diffs)mean+=v;mean/=nd;let m2=0.0,meanAbs=0.0,energy=0.0,mn=Infinity,mx=-Infinity,signs=0;
  for(let j=0;j<nd;j++){const v=diffs[j],dd=v-mean;m2+=dd*dd;meanAbs+=Math.abs(v);energy+=v*v;if(v<mn)mn=v;if(v>mx)mx=v;if(j>0&&diffs[j-1]*v<0.0)signs++;}
  return [mean,Math.sqrt(m2/nd),mn,mx,meanAbs/nd,signs/nd,pearson(row.slice(0,nd),row.slice(1)),energy/nd];
}),a);}
function obliqueFit(x,n,seed){
  const rng=stageRng(seed,x[0].length,n),inv=1.0/Math.sqrt(x[0].length),w=[],t=[];
  for(let i=0;i<n;i++){const row=[];for(let j=0;j<x[0].length;j++)row.push(rng.gaussian()*inv);w.push(row);}
  for(const wi of w){const p=x.map(row=>dot(wi,row));p.sort(totalCompare);t.push(quantile(p,0.5));}return {w,t};
}
function obliqueApply(s,x,a){return finishStage(x,x.map(row=>s.w.map((w,i)=>maxNum(dot(w,row)-s.t[i],0.0))),a);}
function landmarkFit(x,k,seed){const rng=stageRng(seed,x[0].length,k),landmarks=[];for(let i=0;i<k;i++){const row=[];for(let j=0;j<x[0].length;j++)row.push(rng.rangeF64(0.0,1.0));landmarks.push(row);}return {landmarks};}
function landmarkApply(s,x,gamma,a){return finishStage(x,x.map(row=>s.landmarks.map(mark=>{let d2=0.0;for(let j=0;j<Math.min(mark.length,row.length);j++){const d=row[j]-mark[j];d2+=d*d;}return mExp(-gamma*d2);})),a);}
function targetBinFit(x,y,bins,prior){
  const edges=quantileEdges(x,bins),enc=[];let yMean=0.0;for(const v of y)yMean+=v;yMean/=Math.max(y.length,1);
  for(let j=0;j<x[0].length;j++){const sums=new Array(bins).fill(0.0),counts=new Array(bins).fill(0.0);for(let r=0;r<x.length;r++){const b=Math.min(lowerBound(edges[j],x[r][j]),bins-1);sums[b]+=y[r];counts[b]+=1.0;}const col=[];for(let b=0;b<bins;b++)col.push((sums[b]+prior*yMean)/maxNum(counts[b]+prior,1e-9));enc.push(col);}
  return {edges,enc};
}
function targetBinApply(s,x,a){return finishStage(x,x.map(row=>row.map((v,j)=>{const bins=s.enc[j].length;return s.enc[j][Math.min(lowerBound(s.edges[j],v),bins-1)];})),a);}

function softThreshold(v,t){if(v>t)return v-t;if(v< -t)return v+t;return 0.0;}
function elasticFit(x,y,alpha,l1Ratio,sweeps){
  const n=Math.min(x.length,y.length),p=x[0].length,{means,scales}=stdFit(x),xs=[];
  for(let r=0;r<n;r++){const row=[];for(let j=0;j<p;j++)row.push(clampRange((x[r][j]-means[j])/scales[j],-20.0,20.0));xs.push(row);}
  let yMean=0.0;for(let i=0;i<n;i++)yMean+=y[i];yMean/=Math.max(n,1);
  const l1=alpha*l1Ratio,l2=alpha*(1.0-l1Ratio),colSq=new Array(p).fill(0.0),w=new Array(p).fill(0.0),resid=new Array(n);
  for(const row of xs)for(let j=0;j<p;j++)colSq[j]+=row[j]*row[j];for(let r=0;r<n;r++)resid[r]=y[r]-yMean;
  for(let sweep=0;sweep<Math.max(sweeps,1);sweep++){let maxDelta=0.0;for(let j=0;j<p;j++){
    if(colSq[j]<=1e-12)continue;let rho=0.0;for(let r=0;r<n;r++)rho+=xs[r][j]*resid[r];rho=rho/n+w[j]*colSq[j]/n;
    const next=softThreshold(rho,l1)/(colSq[j]/n+l2),delta=next-w[j];if(delta!==0.0){for(let r=0;r<n;r++)resid[r]-=delta*xs[r][j];w[j]=next;const ad=Math.abs(delta);if(ad>maxDelta)maxDelta=ad;}
  }if(maxDelta<1e-9)break;}
  return {means,scales,w0:yMean,w};
}
function elasticPredict(s,x){return x.map(row=>{let z=s.w0;for(let j=0;j<Math.min(s.means.length,row.length);j++)z+=s.w[j]*clampRange((row[j]-s.means[j])/s.scales[j],-20.0,20.0);return clampRange(z,-60.0,60.0);});}
function ridgeFit(x,y,lambdaFrac){
  const n=Math.min(x.length,y.length),p=x[0].length,{means,scales}=stdFit(x),lam=maxNum(lambdaFrac*n,1e-3),ata=new Array(p*p).fill(0.0),atb=new Array(p).fill(0.0),xs=new Array(p).fill(0.0);
  let yMean=0.0;for(let i=0;i<n;i++)yMean+=y[i];yMean/=Math.max(n,1);
  for(let r=0;r<n;r++){for(let j=0;j<p;j++)xs[j]=clampRange((x[r][j]-means[j])/scales[j],-20.0,20.0);const yr=y[r]-yMean;for(let i=0;i<p;i++){const xi=xs[i];atb[i]+=xi*yr;for(let j=i;j<p;j++)ata[i*p+j]+=xi*xs[j];}}
  for(let i=0;i<p;i++){ata[i*p+i]+=lam;for(let j=0;j<i;j++)ata[i*p+j]=ata[j*p+i];}
  return {means,scales,w:[yMean].concat(choleskySolve(ata,atb,p))};
}
function ridgePredict(s,x){return x.map(row=>{let z=s.w[0];for(let j=0;j<Math.min(s.means.length,row.length);j++)z+=s.w[j+1]*clampRange((row[j]-s.means[j])/s.scales[j],-20.0,20.0);return clampRange(z,-60.0,60.0);});}

function fitStage(spec,x,y){
  let state;
  switch(spec.kind){
    case "standardize":state=standardizeFit(x);break;
    case "robust_scale":state=robustScaleFit(x);break;
    case "rank":case "blom":state=rankFit(x);break;
    case "quantile_bins":case "bin_onehot":state={edges:quantileEdges(x,spec.bins)};break;
    case "hinge":state=hingeFit(x,spec.knots);break;
    case "winsor_clip":state=winsorFit(x,spec.q);break;
    case "poly_interact":state=polyFit(x);break;
    case "rff_cos":case "rff_sin":state=rffFit(x,spec.n,spec.gamma,spec.seed);break;
    case "rand_proj":state=randProjFit(x,spec.k,spec.seed);break;
    case "landmark_dist":state=landmarkFit(x,spec.k,spec.seed);break;
    case "oblique_hinge":state=obliqueFit(x,spec.n,spec.seed);break;
    case "pair_prod":state=pairProdFit(x,y,spec.topk);break;
    case "target_bin_encode":state=targetBinFit(x,y,spec.bins,spec.prior);break;
    case "cheb":case "row_stats":case "sigmoid":case "step":case "norm_row":case "row_diff_stats":state=null;break;
    default:throw new RangeError("unsupported fitted stage");
  }
  return [state,applyStage(spec,state,x)];
}
function applyStage(spec,state,x){
  switch(spec.kind){
    case "standardize":return standardizeApply(state,x,spec.append);
    case "robust_scale":return robustScaleApply(state,x,spec.append);
    case "rank":return rankApply(state,x,spec.append);
    case "blom":return blomApply(state,x,spec.append);
    case "quantile_bins":return quantileBinsApply(state,x,spec.bins,spec.append);
    case "bin_onehot":return onehotApply(state,x,spec.bins,spec.append);
    case "hinge":return hingeApply(state,x,spec.append);
    case "winsor_clip":return winsorApply(state,x,spec.append);
    case "cheb":return chebApply(x,spec.degree,spec.append);
    case "poly_interact":return pairsApply(state,x,spec.append);
    case "rff_cos":return rffApply(state,x,spec.append,mCos);
    case "rff_sin":return rffApply(state,x,spec.append,mSin);
    case "rand_proj":return randProjApply(state,x,spec.append);
    case "landmark_dist":return landmarkApply(state,x,spec.gamma,spec.append);
    case "oblique_hinge":return obliqueApply(state,x,spec.append);
    case "row_stats":return rowStatsApply(x,spec.append);
    case "pair_prod":return pairsApply(state,x,spec.append);
    case "target_bin_encode":return targetBinApply(state,x,spec.append);
    case "sigmoid":return sigmoidApply(x,spec.append);
    case "step":return stepApply(x,spec.t,spec.append);
    case "norm_row":return normRowApply(x,spec.append);
    case "row_diff_stats":return rowDiffApply(x,spec.append);
    default:throw new RangeError("unsupported applied stage");
  }
}
function fitHead(spec,x,y){if(spec.kind==="ridge")return ridgeFit(x,y,spec.lambdaFrac);if(spec.kind==="elasticnet")return elasticFit(x,y,spec.alpha,spec.l1Ratio,spec.sweeps);throw new RangeError("unsupported fitted head");}
function predictHead(spec,state,x){if(spec.kind==="ridge")return ridgePredict(state,x);if(spec.kind==="elasticnet")return elasticPredict(state,x);throw new RangeError("unsupported applied head");}
"#;

const EXTRA_HEADS_JS: &str = r#"function meanOf(values,n){const lim=Math.min(values.length,n);if(lim===0)return 0.0;let total=0.0;for(let i=0;i<lim;i++)total+=values[i];return total/lim;}
function jacobiEigen(a,p){
  const v=new Array(p*p).fill(0.0);for(let i=0;i<p;i++)v[i*p+i]=1.0;
  let scale=0.0;for(let i=0;i<p;i++)scale+=a[i*p+i];const limit=(scale*1e-12)*(scale*1e-12);
  for(let sweep=0;sweep<20;sweep++){
    let off=0.0;for(let i=0;i<p;i++)for(let j=i+1;j<p;j++)off+=a[i*p+j]*a[i*p+j];
    if(off<=limit)break;
    for(let i=0;i<p;i++)for(let j=i+1;j<p;j++){
      const apq=a[i*p+j];if(apq===0.0)continue;
      const theta=(a[j*p+j]-a[i*p+i])/(2.0*apq);
      const t=theta>=0.0?1.0/(theta+Math.sqrt(theta*theta+1.0)):-1.0/(Math.sqrt(theta*theta+1.0)-theta);
      const c=1.0/Math.sqrt(t*t+1.0),s=t*c;
      for(let k=0;k<p;k++)if(k!==i&&k!==j){
        const aki=a[k*p+i],akj=a[k*p+j];
        a[k*p+i]=c*aki-s*akj;a[i*p+k]=a[k*p+i];
        a[k*p+j]=s*aki+c*akj;a[j*p+k]=a[k*p+j];
      }
      const aii=a[i*p+i],ajj=a[j*p+j];
      a[i*p+i]=c*c*aii-2.0*s*c*apq+s*s*ajj;
      a[j*p+j]=s*s*aii+2.0*s*c*apq+c*c*ajj;
      a[i*p+j]=0.0;a[j*p+i]=0.0;
      for(let k=0;k<p;k++){const vki=v[k*p+i],vkj=v[k*p+j];v[k*p+i]=c*vki-s*vkj;v[k*p+j]=s*vki+c*vkj;}
    }
  }
  const e=new Array(p).fill(0.0);for(let i=0;i<p;i++)e[i]=a[i*p+i];
  return [e,v];
}
function ridgeGcvFit(x,y){
  const n=Math.min(x.length,y.length);let p=x[0].length;
  let {means,scales}=stdFit(x);
  for(let j=0;j<scales.length;j++)if(scales[j]<=1e-9)scales[j]=1.0;
  let yMean=0.0;for(let i=0;i<n;i++)yMean+=y[i];yMean/=Math.max(n,1);
  let yVar=0.0;for(let i=0;i<n;i++){const d=y[i]-yMean;yVar+=d*d;}
  let cols=new Array(p);for(let j=0;j<p;j++)cols[j]=j;
  if(n<2||p===0||Math.sqrt(yVar/Math.max(n,1))<1e-12){
    const w=new Array(p+1).fill(0.0);w[0]=yMean;return {cols,means,scales,w};
  }
  if(Math.min(n,p)>384){
    const score=new Array(p).fill(0.0);
    for(let r=0;r<n;r++){const row=x[r],yr=y[r]-yMean;for(let j=0;j<p;j++)score[j]+=((row[j]-means[j])/scales[j])*yr;}
    for(let j=0;j<p;j++)if(score[j]<0.0)score[j]=-score[j];
    const used=new Array(p).fill(false),sel=[];
    for(let k=0;k<384;k++){
      let bestJ=0,bestS=-1.0;
      for(let j=0;j<p;j++)if(!used[j]&&score[j]>bestS){bestS=score[j];bestJ=j;}
      used[bestJ]=true;sel.push(bestJ);
    }
    sel.sort((a,b)=>a-b);cols=sel;p=cols.length;
    means=cols.map(j=>means[j]);scales=cols.map(j=>scales[j]);
  }
  const lambdas=[
    1e-05,2.1544346900318823e-05,4.641588833612782e-05,0.0001,0.00021544346900318823,
    0.00046415888336127773,0.001,0.002154434690031882,0.004641588833612777,0.01,
    0.021544346900318822,0.046415888336127774,0.1,0.21544346900318823,0.46415888336127725,
    1.0,2.154434690031882,4.641588833612772,10.0,21.54434690031882,
    46.41588833612773,100.0,215.44346900318823,464.1588833612773,1000.0
  ];
  const nf=n;
  if(n<p){
    const xs=new Array(n*p).fill(0.0);
    for(let r=0;r<n;r++){const row=x[r];for(let j=0;j<p;j++)xs[r*p+j]=(row[cols[j]]-means[j])/scales[j];}
    const yc=new Array(n).fill(0.0);let yty=0.0;
    for(let r=0;r<n;r++){yc[r]=y[r]-yMean;yty+=yc[r]*yc[r];}
    const km=new Array(n*n).fill(0.0);
    for(let r=0;r<n;r++)for(let s=r;s<n;s++){let acc=0.0;for(let j=0;j<p;j++)acc+=xs[r*p+j]*xs[s*p+j];km[r*n+s]=acc;}
    for(let i=0;i<n;i++)for(let j=0;j<i;j++)km[i*n+j]=km[j*n+i];
    const [e,q]=jacobiEigen(km,n);
    let eMax=0.0;for(let i=0;i<n;i++)if(e[i]>eMax)eMax=e[i];
    for(let i=0;i<n;i++)if(e[i]<eMax*1e-12)e[i]=0.0;
    const u=new Array(n).fill(0.0);
    for(let i=0;i<n;i++){let s=0.0;for(let r=0;r<n;r++)s+=q[r*n+i]*yc[r];u[i]=s;}
    let bestLambda=lambdas[0],bestGcv=Infinity;
    for(const lam of lambdas){
      let rss=yty,tr=0.0;
      for(let i=0;i<n;i++)if(e[i]>0.0){const c=u[i]/(e[i]+lam),f=e[i]*c;rss+=f*f-2.0*f*u[i];tr+=e[i]/(e[i]+lam);}
      if(rss<0.0)rss=0.0;
      let denom=1.0-tr/nf;if(denom<1e-3)denom=1e-3;
      const gcv=(rss/nf)/(denom*denom);
      if(gcv<bestGcv){bestGcv=gcv;bestLambda=lam;}
    }
    const t=new Array(n).fill(0.0);
    for(let r=0;r<n;r++){let s=0.0;for(let i=0;i<n;i++)if(e[i]>0.0)s+=q[r*n+i]*(u[i]/(e[i]+bestLambda));t[r]=s;}
    const w=[yMean];
    for(let j=0;j<p;j++){let s=0.0;for(let r=0;r<n;r++)s+=xs[r*p+j]*t[r];w.push(s);}
    return {cols,means,scales,w};
  }
  const g=new Array(p*p).fill(0.0),b=new Array(p).fill(0.0),xsRow=new Array(p).fill(0.0);
  let yty=0.0;
  for(let r=0;r<n;r++){
    const row=x[r];for(let j=0;j<p;j++)xsRow[j]=(row[cols[j]]-means[j])/scales[j];
    const yr=y[r]-yMean;yty+=yr*yr;
    for(let i=0;i<p;i++){const xi=xsRow[i];b[i]+=xi*yr;for(let j=i;j<p;j++)g[i*p+j]+=xi*xsRow[j];}
  }
  for(let i=0;i<p;i++)for(let j=0;j<i;j++)g[i*p+j]=g[j*p+i];
  const [e,v]=jacobiEigen(g,p);
  let eMax=0.0;for(let i=0;i<p;i++)if(e[i]>eMax)eMax=e[i];
  for(let i=0;i<p;i++)if(e[i]<eMax*1e-12)e[i]=0.0;
  const bt=new Array(p).fill(0.0);
  for(let i=0;i<p;i++){let s=0.0;for(let j=0;j<p;j++)s+=v[j*p+i]*b[j];bt[i]=s;}
  let bestLambda=lambdas[0],bestGcv=Infinity;
  for(const lam of lambdas){
    let rss=yty,tr=0.0;
    for(let i=0;i<p;i++)if(e[i]>0.0){const c=bt[i]/(e[i]+lam);rss+=c*c*e[i]-2.0*c*bt[i];tr+=e[i]/(e[i]+lam);}
    if(rss<0.0)rss=0.0;
    let denom=1.0-tr/nf;if(denom<1e-3)denom=1e-3;
    const gcv=(rss/nf)/(denom*denom);
    if(gcv<bestGcv){bestGcv=gcv;bestLambda=lam;}
  }
  const w=[yMean];
  for(let j=0;j<p;j++){let s=0.0;for(let i=0;i<p;i++)if(e[i]>0.0)s+=v[j*p+i]*(bt[i]/(e[i]+bestLambda));w.push(s);}
  return {cols,means,scales,w};
}
function ridgeGcvPredict(s,x){
  return x.map(row=>{
    let z=s.w[0];
    for(let k=0;k<s.cols.length;k++){const j=s.cols[k];if(j<row.length)z+=s.w[k+1]*((row[j]-s.means[k])/s.scales[k]);}
    return clampRange(z,-60.0,60.0);
  });
}
function treeBinEdges(x,f,n,bins){
  const vals=new Array(n);for(let r=0;r<n;r++)vals[r]=x[r][f];
  vals.sort(totalCompare);
  const edges=[];
  for(let q=1;q<bins;q++){
    const idx=Math.trunc(vals.length*(q/bins));
    const v=vals[Math.min(idx,vals.length-1)];
    if(edges.length===0||edges[edges.length-1]!==v)edges.push(v);
  }
  return edges;
}
function treeBinRows(x,rows,nFeat,edges){
  const out=new Int32Array(rows*nFeat);
  for(let r=0;r<rows;r++)for(let f=0;f<nFeat;f++){
    const v=f<x[r].length?x[r][f]:0.0;
    out[r*nFeat+f]=lowerBound(edges[f],v);
  }
  return out;
}
function growTree(bins,grad,hess,rows,nFeat,nbins,maxDepth,minLeaf,lam,depth,tree,leafOf){
  let g=0.0,h=0.0;
  for(const i of rows){g+=grad[i];h+=hess[i];}
  const leafValue=-g/(h+lam);
  if(depth>=maxDepth||rows.length<2*minLeaf){
    const idx=tree.length;
    tree.push({feature:-1,threshold:0,left:0,right:0,value:leafValue});
    for(const i of rows)leafOf[i]=idx;
    return idx;
  }
  let bestGain=0.0,bestFeat=-1,bestThr=0;
  const parent=g*g/(h+lam);
  for(let f=0;f<nFeat;f++){
    const gh=new Array(nbins).fill(0.0),hh=new Array(nbins).fill(0.0);
    for(const i of rows){const b=bins[i*nFeat+f];gh[b]+=grad[i];hh[b]+=hess[i];}
    let gl=0.0,hl=0.0;
    for(let b=0;b<nbins-1;b++){
      gl+=gh[b];hl+=hh[b];
      const gr=g-gl,hr=h-hl;
      if(hl<1.0||hr<1.0)continue;
      const gain=gl*gl/(hl+lam)+gr*gr/(hr+lam)-parent;
      if(gain>bestGain){bestGain=gain;bestFeat=f;bestThr=b;}
    }
  }
  if(bestFeat<0||bestGain<=1e-9){
    const idx=tree.length;
    tree.push({feature:-1,threshold:0,left:0,right:0,value:leafValue});
    for(const i of rows)leafOf[i]=idx;
    return idx;
  }
  const leftRows=[],rightRows=[];
  for(const i of rows){if(bins[i*nFeat+bestFeat]<=bestThr)leftRows.push(i);else rightRows.push(i);}
  if(leftRows.length<minLeaf||rightRows.length<minLeaf){
    const idx=tree.length;
    tree.push({feature:-1,threshold:0,left:0,right:0,value:leafValue});
    for(const i of rows)leafOf[i]=idx;
    return idx;
  }
  const idx=tree.length;
  tree.push({feature:-1,threshold:0,left:0,right:0,value:0.0});
  const left=growTree(bins,grad,hess,leftRows,nFeat,nbins,maxDepth,minLeaf,lam,depth+1,tree,leafOf);
  const right=growTree(bins,grad,hess,rightRows,nFeat,nbins,maxDepth,minLeaf,lam,depth+1,tree,leafOf);
  tree[idx]={feature:bestFeat,threshold:bestThr,left,right,value:0.0};
  return idx;
}
function routeTree(tree,bins,r,nFeat){
  let idx=0;
  for(;;){
    const node=tree[idx];
    if(node.feature<0)return node.value;
    idx=bins[r*nFeat+node.feature]<=node.threshold?node.left:node.right;
  }
}
function gbmFit(x,y,rounds,maxDepth,lr,lam,bins,minLeaf,maxFeatures){
  const n=Math.min(x.length,y.length),cols=x[0].length,nFeat=Math.min(cols,maxFeatures);
  if(n<4||nFeat===0){
    const m=meanOf(y,Math.min(Math.max(n,1),y.length));
    return {edges:[],trees:[],init:m,lr,nFeat:0,fallback:m};
  }
  const edges=new Array(nFeat);for(let f=0;f<nFeat;f++)edges[f]=treeBinEdges(x,f,n,bins);
  const trainBins=treeBinRows(x,n,nFeat,edges),init=meanOf(y,n),fTrain=new Array(n).fill(init),trees=[];
  const allRows=new Array(n);for(let i=0;i<n;i++)allRows[i]=i;
  for(let r=0;r<rounds;r++){
    const grad=new Array(n);for(let i=0;i<n;i++)grad[i]=fTrain[i]-y[i];
    const hess=new Array(n).fill(1.0),tree=[],trainLeaf=new Array(n).fill(0);
    growTree(trainBins,grad,hess,allRows,nFeat,bins,maxDepth,minLeaf,lam,0,tree,trainLeaf);
    for(let i=0;i<n;i++)fTrain[i]+=lr*tree[trainLeaf[i]].value;
    trees.push(tree);
  }
  return {edges,trees,init,lr,nFeat,fallback:init};
}
function gbmPredict(s,x){
  if(s.nFeat===0)return new Array(x.length).fill(s.fallback);
  const bins=treeBinRows(x,x.length,s.nFeat,s.edges),out=new Array(x.length);
  for(let r=0;r<x.length;r++){
    let val=s.init;
    for(const tree of s.trees)val+=s.lr*routeTree(tree,bins,r,s.nFeat);
    out[r]=val;
  }
  return out;
}
function ksmoothFit(x,y){
  const n=Math.min(x.length,y.length),p=x[0].length,{means,scales}=stdFit(x);
  let yMean=0.0;for(let i=0;i<n;i++)yMean+=y[i];yMean/=Math.max(n,1);
  if(n<8||p===0)return {means,scales,z:[],ys:[],yMean};
  const z=new Array(n);
  for(let r=0;r<n;r++){const row=new Array(p);for(let j=0;j<p;j++)row[j]=(x[r][j]-means[j])/scales[j];z[r]=row;}
  return {means,scales,z,ys:y.slice(0,n),yMean};
}
function ksmoothPredict(s,x,k,gamma){
  const p=s.means.length;
  if(s.z.length===0)return new Array(x.length).fill(clampRange(s.yMean,-60.0,60.0));
  const pf=p,kk=Math.min(k,s.z.length);
  return x.map(row=>{
    const lim=Math.min(p,row.length),zr=new Array(lim);
    for(let j=0;j<lim;j++)zr[j]=(row[j]-s.means[j])/s.scales[j];
    const d2s=new Array(s.z.length);
    for(let i=0;i<s.z.length;i++){
      const zb=s.z[i],dLim=Math.min(zr.length,zb.length);
      let d2=0.0;for(let d=0;d<dLim;d++){const t=zr[d]-zb[d];d2+=t*t;}
      d2s[i]=d2;
    }
    const order=new Array(s.z.length);for(let i=0;i<s.z.length;i++)order[i]=i;
    order.sort((a,b)=>{const c=totalCompare(d2s[a],d2s[b]);return c!==0?c:a-b;});
    const base=d2s[order[0]];
    let num=0.0,den=0.0;
    for(let q=0;q<kk;q++){
      const i=order[q],w=mExp(-gamma*(d2s[i]-base)/pf);
      num+=w*s.ys[i];den+=w;
    }
    return clampRange(num/den,-60.0,60.0);
  });
}
function fitHead(spec,x,y){
  if(spec.kind==="ridge")return ridgeFit(x,y,spec.lambdaFrac);
  if(spec.kind==="ridge_gcv")return ridgeGcvFit(x,y);
  if(spec.kind==="elasticnet")return elasticFit(x,y,spec.alpha,spec.l1Ratio,spec.sweeps);
  if(spec.kind==="gbm")return gbmFit(x,y,spec.rounds,spec.depth,spec.lr,spec.lambda,spec.bins,spec.minLeaf,spec.maxFeatures);
  if(spec.kind==="ksmooth")return ksmoothFit(x,y);
  throw new RangeError("unsupported fitted head");
}
function predictHead(spec,state,x){
  if(spec.kind==="ridge")return ridgePredict(state,x);
  if(spec.kind==="ridge_gcv")return ridgeGcvPredict(state,x);
  if(spec.kind==="elasticnet")return elasticPredict(state,x);
  if(spec.kind==="gbm")return gbmPredict(state,x);
  if(spec.kind==="ksmooth")return ksmoothPredict(state,x,spec.k,spec.gamma);
  throw new RangeError("unsupported applied head");
}"#;

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
