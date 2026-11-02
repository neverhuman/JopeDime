//! Direct, fitted `BoundModel` to C++17 lowering.
//!
//! This is deliberately separate from the frozen V1 emitter.  The generated
//! program owns no files and performs no parsing: a persistent host supplies a
//! typed row-major table.  Numeric cells carry their parsed `f64`; categorical
//! cells carry the fitted vocabulary index (`-1` means unknown).  The parallel
//! missing bitmap distinguishes an unknown category from a missing category.
//!
//! The emitter revision, `FITTED_CPP_REVISION`, names what the emitted code
//! computes. Every source carries it in its first line (`emitter-revision=`),
//! and the typed-table codec and the fitted-model container in their
//! `emitter_revision` field. Revision 3 (kernel2 X3) checks each fitted column
//! index a stage reads against the row it reads, and answers status 8 instead of
//! reading out of range (see `emit_abi`); divides the mean and the standard
//! deviation of an empty row by one, as the runtime does; pads a stage's empty
//! output row to `[0.0]` on the legacy-predictor paths, as `ops::finish` does;
//! and encodes a `piecewise_linear` interval no wider than 1e-15 as a step, as
//! the runtime does. Everything else computes revision 2's bits.

use super::calibration::CalibrationModel;
use super::heads::{ExtraNode, V3Head};
use super::input::{
    CategoricalMode, FittedColumn, FittedRowOrder, HistoricalTargetState, SemanticKey,
};
use super::pipeline::{BoundModel, BoundPredictor};
use super::schema::{ColumnKind, RowSemantics};
use super::transforms::{FittedTableOrder, GroupTail, SemanticKey as RowSemanticKey, V3Op};
use super::types::{text_hash, TaskKind};
use crate::boost2::BoostNode;
use crate::gpr::KernelPrim;
use crate::heads::ConditionalRidge;
use crate::heads::{FittedHead, GbmNode};
use crate::ops::FittedOp;
use crate::portfolio::FittedMemberState;
use crate::target_transform::FittedTarget;
use anyhow::{bail, Context, Result};
use serde::Serialize;

#[derive(Clone, Debug)]
pub struct FittedCppArtifact {
    pub source: String,
    pub source_sha256: String,
    pub codec: String,
    pub codec_sha256: String,
}

#[derive(Serialize)]
struct Codec<'a> {
    schema: &'static str,
    emitter_revision: u32,
    schema_sha256: &'a str,
    vocabulary_sha256: &'a str,
    columns: Vec<CodecColumn<'a>>,
    categorical_unknown: i32,
    missing_bitmap: &'static str,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum CodecColumn<'a> {
    Numeric,
    Categorical { vocabulary: &'a [String] },
}

/// Exhaustive admission check.  Every fitted enum is matched explicitly so a
/// newly-added registry variant fails compilation until its C++ semantics are
/// deliberately classified.
pub fn supports_fitted_cpp17(model: &BoundModel) -> Result<()> {
    model.validate_binding()?;
    match &model.schema.row_semantics {
        RowSemantics::Iid | RowSemantics::Ordered { .. } => {}
    }
    match &model.predictor {
        BoundPredictor::Pipeline {
            target,
            ops,
            head,
            calibration,
            ..
        } => {
            check_target(target);
            check_calibration(calibration);
            for op in ops {
                check_op(op)?;
            }
            check_head(head)
        }
        BoundPredictor::Legacy(crate::model::Model::Pipeline { target, ops, head }) => {
            check_target(target);
            for op in ops {
                check_op(&V3Op::Legacy(op.clone()))?;
            }
            check_head(&V3Head::Legacy((**head).clone()))
        }
        BoundPredictor::Legacy(crate::model::Model::Symbolic { expr, .. }) => {
            expr.to_cpp17(false).map(|_| ())
        }
        BoundPredictor::Legacy(crate::model::Model::Portfolio(model)) => check_portfolio(model),
    }
}

fn check_target(target: &Option<FittedTarget>) {
    match target {
        None
        | Some(FittedTarget::RankGauss { .. })
        | Some(FittedTarget::Log1p)
        | Some(FittedTarget::Winsor { .. })
        | Some(FittedTarget::ZScore { .. }) => {}
    }
}

fn check_calibration(calibration: &CalibrationModel) {
    match calibration {
        CalibrationModel::None
        | CalibrationModel::Platt { .. }
        | CalibrationModel::Isotonic { .. } => {}
    }
}

fn check_op(op: &V3Op) -> Result<()> {
    match op {
        V3Op::Legacy(op) => match op {
            FittedOp::Standardize { .. }
            | FittedOp::RobustScale { .. }
            | FittedOp::Rank { .. }
            | FittedOp::Blom { .. }
            | FittedOp::QuantileBins { .. }
            | FittedOp::BinOnehot { .. }
            | FittedOp::Hinge { .. }
            | FittedOp::WinsorClip { .. }
            | FittedOp::Cheb { .. }
            | FittedOp::PolyInteract { .. }
            | FittedOp::RffCos { .. }
            | FittedOp::RffSin { .. }
            | FittedOp::RandProj { .. }
            | FittedOp::LandmarkDist { .. }
            | FittedOp::ObliqueHinge { .. }
            | FittedOp::PairProd { .. }
            | FittedOp::TargetBinEncode { .. }
            | FittedOp::Sigmoid { .. }
            | FittedOp::Step { .. }
            | FittedOp::NormRow { .. }
            | FittedOp::Polar { .. }
            | FittedOp::Lag { .. }
            | FittedOp::Roll { .. }
            | FittedOp::SvdProject { .. }
            | FittedOp::Nystrom { .. }
            | FittedOp::Symlog { .. }
            | FittedOp::SelectCorr { .. }
            | FittedOp::Impute { .. }
            | FittedOp::PairRatio { .. }
            | FittedOp::Spline { .. } => Ok(()),
            FittedOp::RowStats { .. }
            | FittedOp::FftMag { .. }
            | FittedOp::RowSpectral { .. }
            | FittedOp::RowDiffStats { .. }
            | FittedOp::MapExpr { .. }
            | FittedOp::DeriveExpr { .. } => Ok(()),
        },
        V3Op::MaxAbs { .. }
        | V3Op::YeoJohnson { .. }
        | V3Op::PiecewiseLinear { .. }
        | V3Op::CorrScreen { .. }
        | V3Op::PlsProject { .. } => Ok(()),
        V3Op::Lag { .. } | V3Op::Roll { .. } => Ok(()),
    }
}

fn check_head(head: &V3Head) -> Result<()> {
    match head {
        V3Head::Legacy(head) => match head {
            FittedHead::Ridge(_)
            | FittedHead::RidgeGcv(_)
            | FittedHead::ElasticNet(_)
            | FittedHead::Gbm(_)
            | FittedHead::Ensemble { .. }
            | FittedHead::HistGbm2(_)
            | FittedHead::Krr(_)
            | FittedHead::Gpr(_)
            | FittedHead::Ksmooth(_)
            | FittedHead::Hurdle2(_)
            | FittedHead::Stack(_) => Ok(()),
            FittedHead::GprKernel(_) => Ok(()),
            FittedHead::Portfolio(model) => check_portfolio(model),
        },
        V3Head::LogitRidge(_)
        | V3Head::TabBoost(_)
        | V3Head::ExtraTrees(_)
        | V3Head::GamBoost(_)
        | V3Head::Pls(_) => Ok(()),
        V3Head::ClassificationPortfolio(model) => {
            for member in &model.members {
                check_head(member)?;
            }
            Ok(())
        }
        V3Head::MulticlassRidge(_)
        | V3Head::MulticlassOvrLogistic(_)
        | V3Head::MulticlassDiagonalLda(_)
        | V3Head::MulticlassNearestCentroid(_) => {
            bail!("multiclass probability matrices are not supported by the scalar C++ ABI")
        }
    }
}

fn check_legacy_model(model: &crate::model::Model) -> Result<()> {
    match model {
        crate::model::Model::Pipeline { target, ops, head } => {
            check_target(target);
            for op in ops {
                check_op(&V3Op::Legacy(op.clone()))?;
            }
            check_head(&V3Head::Legacy((**head).clone()))
        }
        crate::model::Model::Symbolic { expr, .. } => expr.to_cpp17(false).map(|_| ()),
        crate::model::Model::Portfolio(model) => check_portfolio(model),
    }
}

fn check_portfolio(model: &crate::portfolio::PortfolioModel) -> Result<()> {
    if let Some(projection) = &model.proj {
        check_op(&V3Op::Legacy(projection.clone()))?;
    }
    for member in &model.members {
        match &member.state {
            FittedMemberState::Fitted(model) => check_legacy_model(model)?,
            FittedMemberState::GroupMean(_)
            | FittedMemberState::Mean
            | FittedMemberState::Skipped => {}
        }
    }
    Ok(())
}

pub fn emit_fitted_cpp17(model: &BoundModel) -> Result<FittedCppArtifact> {
    supports_fitted_cpp17(model)?;
    let codec = serde_json::to_string(&Codec {
        schema: "jope-dime.fitted-cpp-typed-table.v1",
        emitter_revision: FITTED_CPP_REVISION,
        schema_sha256: &model.schema_sha256,
        vocabulary_sha256: &model.vocabulary_sha256,
        columns: model
            .schema
            .columns
            .iter()
            .zip(&model.input.columns)
            .map(|(schema, fitted)| match (schema.kind, fitted) {
                (ColumnKind::Numeric, FittedColumn::Numeric { .. }) => CodecColumn::Numeric,
                (ColumnKind::Categorical, FittedColumn::Categorical { vocabulary, .. }) => {
                    CodecColumn::Categorical { vocabulary }
                }
                _ => unreachable!("validated fitted input/schema correspondence"),
            })
            .collect(),
        categorical_unknown: -1,
        missing_bitmap: "one byte per raw cell; nonzero means missing",
    })?;

    let (head_functions, transform, predict, missing_aware, table_aware) = match &model.predictor {
        BoundPredictor::Pipeline {
            target,
            ops,
            head,
            calibration,
            missing_aware,
        } => {
            let mut functions = String::new();
            emit_head_function("head_0", head, &mut functions)?;
            let table_aware = ops
                .iter()
                .any(|op| matches!(op, V3Op::Lag { .. } | V3Op::Roll { .. }));
            if table_aware {
                (
                    functions,
                    emit_table_transforms(model, ops)?,
                    emit_predict_transformed(target, calibration, model.task.kind),
                    *missing_aware,
                    true,
                )
            } else {
                (
                    functions,
                    emit_transforms(ops)?,
                    emit_predict(target, calibration, model.task.kind),
                    *missing_aware,
                    false,
                )
            }
        }
        BoundPredictor::Legacy(crate::model::Model::Pipeline { target, ops, head }) => {
            let mut functions = String::new();
            emit_legacy_head("head_0", head, &mut functions)?;
            (
                functions,
                emit_legacy_transforms(ops)?,
                emit_predict(target, &CalibrationModel::None, model.task.kind),
                false,
                false,
            )
        }
        BoundPredictor::Legacy(crate::model::Model::Symbolic { expr, a, b }) => {
            let expression = expr.to_cpp17(false)?;
            let functions = format!(
                "static double head_0(const V& values){{const double* row=values.data();const std::size_t row_len=values.size();(void)row;(void)row_len;double value=math_safe({expression});double result={a}*value+{b};return std::isfinite(result)?clampv(result,-60.0,60.0):0.0;}}\n",
                a = f64_lit(*a),
                b = f64_lit(*b),
            );
            (
                functions,
                "static V transform(V x,V){return x;}\n".to_string(),
                emit_predict(&None, &CalibrationModel::None, model.task.kind),
                false,
                false,
            )
        }
        BoundPredictor::Legacy(crate::model::Model::Portfolio(portfolio)) => {
            let mut functions = String::new();
            emit_portfolio("head_0", portfolio, &mut functions)?;
            (
                functions,
                "static V transform(V x,V){return x;}\n".to_string(),
                emit_predict(&None, &CalibrationModel::None, model.task.kind),
                false,
                false,
            )
        }
    };
    let abi = if table_aware {
        emit_table_abi(model.schema.columns.len())
    } else {
        emit_abi(model.schema.columns.len())
    };
    let source = format!(
        "{PRELUDE}\n{head_functions}\n{encode}\n{transform}\n{predict}\n{abi}",
        encode = emit_input(model, missing_aware)?,
    );
    // The caller's column names and categorical vocabularies are masked before the lint runs
    // (`emit::user_text`), so the codec's vocabulary arrays cannot trip it on an innocent name.
    let user = super::emit::user_text(model);
    let portfolio = matches!(
        &model.predictor,
        BoundPredictor::Pipeline {
            head: V3Head::ClassificationPortfolio(_),
            ..
        }
    );
    for (what, text) in [
        ("fitted C++17 artifact", &source),
        ("fitted C++17 codec", &codec),
    ] {
        let hits = super::emit::lint_shipped_masked(text, &user, portfolio);
        if !hits.is_empty() {
            bail!("banned vocabulary in {what}: {}", hits.join(", "));
        }
    }
    Ok(FittedCppArtifact {
        source_sha256: text_hash(&source),
        codec_sha256: text_hash(&codec),
        source,
        codec,
    })
}

fn f64_lit(value: f64) -> String {
    format!("f64_from_bits(0x{:016x}ULL)", value.to_bits())
}

fn f64_vec(values: &[f64]) -> String {
    format!(
        "V{{{}}}",
        values
            .iter()
            .map(|value| f64_lit(*value))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn usize_vec(values: &[usize]) -> String {
    format!(
        "U{{{}}}",
        values
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn f64_matrix(values: &[Vec<f64>]) -> String {
    format!(
        "M{{{}}}",
        values
            .iter()
            .map(|row| f64_vec(row))
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// The emitter revision this module writes (see the module doc). Emitted code that computes
/// anything differently needs the next revision: the fitted-model container compares a stored
/// source of the current revision with this emitter's, and regenerates older revisions before
/// loading them (`fitted_cpp_runtime`). The PRELUDE's first line carries the same number.
pub(super) const FITTED_CPP_REVISION: u32 = 3;

macro_rules! expression_common {
    () => {
        r#"#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>
using V = std::vector<double>;
using U = std::vector<std::size_t>;
using M = std::vector<V>;
using B = std::vector<std::uint64_t>;
struct MathDomainError {};
struct FittedWidthError {};
[[maybe_unused]] static std::size_t fitted_index(std::size_t index,std::size_t size){if(index>=size)throw FittedWidthError{};return index;}
[[maybe_unused]] static double f64_from_bits(std::uint64_t bits) noexcept { double value; std::memcpy(&value,&bits,sizeof(value)); return value; }
[[maybe_unused]] static double clampv(double value,double lo,double hi) noexcept { if(value<lo)return lo; if(value>hi)return hi; return value; }
[[maybe_unused]] static double finitev(double value) noexcept { return std::isfinite(value)?value:0.0; }
[[maybe_unused]] static double sigmoid(double value) noexcept { if(value>=0.0)return 1.0/(1.0+std::exp(-std::min(value,40.0))); double z=std::exp(std::max(value,-40.0)); return z/(1.0+z); }
"#
    };
}

macro_rules! expression_math {
    () => {
        r#"[[maybe_unused]] static double row_sum(const double* row,std::size_t n){double value=0.0;for(std::size_t i=0;i<n;++i)value+=row[i];return value;}
[[maybe_unused]] static double row_mean(const double* row,std::size_t n){return row_sum(row,n)/static_cast<double>(std::max(n,std::size_t{1}));}
[[maybe_unused]] static double row_l1(const double* row,std::size_t n){double value=0.0;for(std::size_t i=0;i<n;++i)value+=std::abs(row[i]);return value;}
[[maybe_unused]] static double row_l2(const double* row,std::size_t n){double value=0.0;for(std::size_t i=0;i<n;++i)value+=row[i]*row[i];return std::sqrt(value);}
[[maybe_unused]] static double row_min(const double* row,std::size_t n){double value=std::numeric_limits<double>::infinity();for(std::size_t i=0;i<n;++i)if(row[i]<value)value=row[i];return value;}
[[maybe_unused]] static double row_max(const double* row,std::size_t n){double value=-std::numeric_limits<double>::infinity();for(std::size_t i=0;i<n;++i)if(row[i]>value)value=row[i];return value;}
[[maybe_unused]] static double row_std(const double* row,std::size_t n){double mean=row_mean(row,n),value=0.0;for(std::size_t i=0;i<n;++i){double delta=row[i]-mean;value+=delta*delta;}return std::sqrt(value/static_cast<double>(std::max(n,std::size_t{1})));}
[[maybe_unused]] static double m_clamp(double value,double lo,double hi){return clampv(value,lo,hi);}
[[maybe_unused]] static double m_div(double left,double right){return std::abs(right)<1e-12?0.0:left/right;}
[[maybe_unused]] static double m_pow(double left,double right){return std::pow(left,right);}
[[maybe_unused]] static double m_sqrt(double value){return std::sqrt(std::fmax(value,0.0));}
[[maybe_unused]] static double m_log1p(double value){return value<=-1.0?0.0:std::log1p(value);}
[[maybe_unused]] static double m_sin(double value){return std::sin(value);}
[[maybe_unused]] static double m_cos(double value){return std::cos(value);}
[[maybe_unused]] static double m_tanh(double value){return std::tanh(value);}
[[maybe_unused]] static double m_sigmoid(double value){return 1.0/(1.0+std::exp(-clampv(value,-30.0,30.0)));}
[[maybe_unused]] static double m_clip(double value,double a,double b){double lo=std::fmin(a,b),hi=std::fmax(a,b);if(std::isnan(lo)||std::isnan(hi))throw MathDomainError{};return clampv(value,lo,hi);}
[[maybe_unused]] static double m_exp(double value){return std::exp(clampv(value,-30.0,30.0));}
[[maybe_unused]] static double m_log(double value){return value<=1e-300?0.0:std::log(value);}
[[maybe_unused]] static double m_asinh(double value){double a=std::abs(value),s=value>0.0?1.0:(value<0.0?-1.0:0.0);return s*std::log(a+std::sqrt(value*value+1.0));}
[[maybe_unused]] static double m_atan(double value){return std::atan(value);}
[[maybe_unused]] static double m_erf(double value){double a=std::abs(value),s=value>0.0?1.0:(value<0.0?-1.0:0.0),t=1.0/(1.0+0.3275911*a);double poly=t*(0.254829592+t*(-0.284496736+t*(1.421413741+t*(-1.453152027+t*1.061405429))));return s*(1.0-poly*std::exp(-(a*a)));}
[[maybe_unused]] static double m_sign(double value){return value>0.0?1.0:(value<0.0?-1.0:0.0);}
[[maybe_unused]] static double m_round(double value){return value>=0.0?std::floor(value+0.5):std::ceil(value-0.5);}
[[maybe_unused]] static double math_safe(double value){return std::isfinite(value)?clampv(value,-1e12,1e12):0.0;}
"#
    };
}

const PRELUDE: &str = concat!(
    "// jope-dime fitted-cpp-typed-table-v1 emitter-revision=3\n",
    expression_common!(),
    r#"// Match Rust slice::partition_point's fixed-iteration binary-search path.
// Fitted interpolated quantiles can cross by one ULP; preserving the search
// path is required to execute those immutable legacy states bit-for-bit.
[[maybe_unused]] static std::size_t lower(const V& values,double value) noexcept { if(values.empty())return 0;std::size_t size=values.size(),base=0;while(size>1){const std::size_t half=size/2,mid=base+half;if(values[mid]<value)base=mid;size-=half;}return base+static_cast<std::size_t>(values[base]<value); }
[[maybe_unused]] static std::size_t upper(const V& values,double value) noexcept { if(values.empty())return 0;std::size_t size=values.size(),base=0;while(size>1){const std::size_t half=size/2,mid=base+half;if(values[mid]<=value)base=mid;size-=half;}return base+static_cast<std::size_t>(values[base]<=value); }
[[maybe_unused]] static double mid_rank(const V& values,double value) noexcept { return static_cast<double>(lower(values,value)+upper(values,value))/(2.0*static_cast<double>(values.size())); }
[[maybe_unused]] static double probit(double p) noexcept {
  const V a={-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.38357751867269e+02,-3.066479806614716e+01,2.506628277459239e+00};
  const V b={-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01};
  const V c={-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00};
  const V e={7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00};
  p=clampv(p,1e-12,1.0-1e-12); const double p_low=0.02425;
  if(p<p_low){const double q=std::sqrt(-2.0*std::log(p));return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((e[0]*q+e[1])*q+e[2])*q+e[3])*q+1.0);}
  if(p<=1.0-p_low){const double q=p-0.5,r=q*q;return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);}
  const double q=std::sqrt(-2.0*std::log(1.0-p));return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((e[0]*q+e[1])*q+e[2])*q+e[3])*q+1.0);
}
[[maybe_unused]] static double dot(const V& a,const V& b) noexcept { double s=0.0; for(std::size_t i=0;i<std::min(a.size(),b.size());++i)s+=a[i]*b[i]; return s; }
[[maybe_unused]] static void finish_block(const V& input,const V& input_mask,V output,V output_mask,bool append,V& x,V& mask){ for(std::size_t i=0;i<output.size();++i)if(output_mask[i]!=0.0||!std::isfinite(output[i]))output[i]=0.0;if(append){x=input;mask=input_mask;x.insert(x.end(),output.begin(),output.end());mask.insert(mask.end(),output_mask.begin(),output_mask.end());}else{x=std::move(output);mask=std::move(output_mask);}if(x.size()>512){x.resize(512);mask.resize(512);} }
[[maybe_unused]] static std::uint64_t bits(double value) noexcept {std::uint64_t result;std::memcpy(&result,&value,8);return result;}
[[maybe_unused]] static double piece(double value,double lo,double hi) noexcept { const double width=hi-lo; if(std::abs(width)<=1e-15)return value>lo?1.0:0.0; return clampv((value-lo)/width,0.0,1.0); }
[[maybe_unused]] static double yj(double value,double lambda) noexcept { if(value>=0.0){ if(std::abs(lambda)<1e-8)return std::log1p(value); return (std::pow(value+1.0,lambda)-1.0)/lambda; } double q=2.0-lambda; if(std::abs(q)<1e-8)return -std::log1p(-value); return -(std::pow(1.0-value,q)-1.0)/q; }
struct GNode { bool leaf; double value; std::size_t feature; std::uint16_t threshold; std::size_t left; std::size_t right; };
struct Block { V x; V mask; };
struct KCell { bool present; double value; };
using Key = std::vector<KCell>;
[[maybe_unused]] static bool operator==(const KCell& a,const KCell& b) noexcept {return a.present==b.present&&(!a.present||a.value==b.value);}
[[maybe_unused]] static bool operator<(const KCell& a,const KCell& b) noexcept {if(a.present!=b.present)return !a.present;if(!a.present)return false;if(a.value<b.value)return true;if(a.value>b.value)return false;return false;}
[[maybe_unused]] static bool key_less(const Key& a,const Key& b) noexcept {return std::lexicographical_compare(a.begin(),a.end(),b.begin(),b.end());}
[[maybe_unused]] static bool key_equal(const Key& a,const Key& b) noexcept {return a==b;}
[[maybe_unused]] static Key raw_key(const double* raw,const std::uint8_t* missing,const U& columns){Key key;key.reserve(columns.size());for(std::size_t column:columns)key.push_back(KCell{missing[column]==0,raw[column]});return key;}
[[maybe_unused]] static double route_gbm(const std::vector<GNode>& tree,const M& edges,const V& x) noexcept { std::size_t at=0; for(;;){const GNode& n=tree[at];if(n.leaf)return n.value;double value=n.feature<x.size()?x[n.feature]:0.0;std::size_t bin=lower(edges[n.feature],value);at=bin<=n.threshold?n.left:n.right;} }
[[maybe_unused]] static double rank_inverse(double value,const V& scores,const V& values) noexcept { if(scores.empty()||values.empty())return 0.0;if(!(value>scores[0]))return values[0];if(value>=scores.back())return values.back();std::size_t hi=lower(scores,value),lo=hi-1;double width=scores[hi]-scores[lo];return width>0.0?values[lo]+(value-scores[lo])/width*(values[hi]-values[lo]):values[lo]; }
[[maybe_unused]] static double quantile_linear(const V& sorted,double q) noexcept { if(sorted.empty())return 0.0;double position=q*static_cast<double>(sorted.size()-1);std::size_t lo=static_cast<std::size_t>(position),hi=std::min(lo+1,sorted.size()-1);double fraction=position-static_cast<double>(lo);return sorted[lo]*(1.0-fraction)+sorted[hi]*fraction; }
[[maybe_unused]] static V dft_mags(const V& row,std::size_t k){V out;out.reserve(k);double width=static_cast<double>(row.size());for(std::size_t f=1;f<=k;++f){double re=0.0,im=0.0;for(std::size_t j=0;j<row.size();++j){double angle=(2.0*3.14159265358979323846264338327950288)*static_cast<double>(f)*static_cast<double>(j)/width;re+=row[j]*std::cos(angle);im+=row[j]*std::sin(angle);}out.push_back(std::sqrt(re*re+im*im)/width);}return out;}
[[maybe_unused]] static V row_stats(const V& row){double n=static_cast<double>(row.size()),mean=0.0;for(double value:row)mean+=value;mean/=n;double m2=0.0,m3=0.0,m4=0.0,energy=0.0,mn=std::numeric_limits<double>::infinity(),mx=-std::numeric_limits<double>::infinity();for(double value:row){double delta=value-mean;m2+=delta*delta;m3+=delta*delta*delta;m4+=delta*delta*delta*delta;energy+=value*value;if(value<mn)mn=value;if(value>mx)mx=value;}m2/=n;m3/=n;m4/=n;energy/=n;double stddev=std::sqrt(m2),skew=m2>1e-24?m3/(m2*stddev):0.0,kurt=m2>1e-24?m4/(m2*m2)-3.0:0.0;V sorted=row;std::sort(sorted.begin(),sorted.end());return V{mean,stddev,mn,mx,quantile_linear(sorted,0.5),quantile_linear(sorted,0.25),quantile_linear(sorted,0.75),skew,kurt,energy};}
[[maybe_unused]] static V row_spectral(const V& row){std::size_t count=row.size()/2;if(count==0)return V(5,0.0);V mags=dft_mags(row,count);double total=0.0;for(double value:mags)total+=value;double denominator=std::max(total,1e-12),entropy=0.0,m1=0.0,m2=0.0;std::size_t peak=0;for(std::size_t i=0;i<mags.size();++i){double probability=mags[i]/denominator;if(probability>1e-12)entropy-=probability*std::log(probability);double frequency=static_cast<double>(i+1);m1+=frequency*probability;m2+=frequency*frequency*probability;if(mags[i]>mags[peak])peak=i;}return V{entropy,m1,m2,static_cast<double>(peak+1),mags[peak]};}
[[maybe_unused]] static double pearson_shift(const V& row){std::size_t n=row.size()-1;if(n<2)return 0.0;double ma=0.0,mb=0.0;for(std::size_t i=0;i<n;++i){ma+=row[i];mb+=row[i+1];}ma/=static_cast<double>(n);mb/=static_cast<double>(n);double sab=0.0,saa=0.0,sbb=0.0;for(std::size_t i=0;i<n;++i){double da=row[i]-ma,db=row[i+1]-mb;sab+=da*db;saa+=da*da;sbb+=db*db;}return saa<=0.0||sbb<=0.0?0.0:sab/std::sqrt(saa*sbb);}
[[maybe_unused]] static V row_diff_stats(const V& row){if(row.size()<2)return V(8,0.0);std::size_t n=row.size()-1;double nf=static_cast<double>(n);V diffs;diffs.reserve(n);for(std::size_t i=0;i<n;++i)diffs.push_back(row[i+1]-row[i]);double mean=0.0;for(double value:diffs)mean+=value;mean/=nf;double m2=0.0,mean_abs=0.0,energy=0.0,mn=std::numeric_limits<double>::infinity(),mx=-std::numeric_limits<double>::infinity();std::size_t signs=0;for(std::size_t i=0;i<n;++i){double value=diffs[i],delta=value-mean;m2+=delta*delta;mean_abs+=std::abs(value);energy+=value*value;if(value<mn)mn=value;if(value>mx)mx=value;if(i>0&&diffs[i-1]*value<0.0)++signs;}return V{mean,std::sqrt(m2/nf),mn,mx,mean_abs/nf,static_cast<double>(signs)/nf,pearson_shift(row),energy/nf};}
"#,
    expression_math!(),
);

impl FittedCppArtifact {
    // B05–B07: reuse the existing fitted ridge lowerer for scalar-v2 features.
    pub(crate) fn scalar_ridge_function(model: &crate::heads::RidgeModel) -> String {
        let mut source = String::new();
        emit_linear(
            "scalar_head_v2",
            &model.means,
            &model.scales,
            &model.w,
            None,
            -20.0,
            20.0,
            false,
            &mut source,
        );
        source
    }

    /// Share the frozen scalar lowering helpers with the opt-in training recipe.
    pub(crate) fn expression_helpers() -> &'static str {
        concat!(expression_common!(), expression_math!())
    }
}

fn emit_input(model: &BoundModel, missing_aware: bool) -> Result<String> {
    let mut body = String::from(
        "static bool encode_row(const double* raw,const std::uint8_t* missing,V& x,V& mask){\n  x.clear(); mask.clear();\n",
    );
    for (column, fitted) in model.input.columns.iter().enumerate() {
        match fitted {
            FittedColumn::Numeric { fill, indicator } => {
                body.push_str(&format!(
                    "  const bool absent_{column}=missing[{column}]!=0||!std::isfinite(raw[{column}]); const double value_{column}=absent_{column}?{fill}:raw[{column}]; x.push_back(value_{column}); mask.push_back(absent_{column}?1.0:0.0);\n",
                    fill = f64_lit(*fill),
                ));
                if *indicator {
                    body.push_str(&format!(
                        "  x.push_back(absent_{column}?1.0:0.0); mask.push_back(0.0);\n"
                    ));
                }
            }
            FittedColumn::Categorical {
                mode,
                vocabulary,
                counts,
                sums,
                total,
                global,
                onehot,
            } => {
                body.push_str(&format!(
                    "  const bool absent_{column}=missing[{column}]!=0; (void)absent_{column}; const double encoded_{column}=raw[{column}]; if(!std::isfinite(encoded_{column})||std::floor(encoded_{column})!=encoded_{column}||encoded_{column} < -1.0||encoded_{column}>={vocab}.0)return false; const std::ptrdiff_t index_{column}=static_cast<std::ptrdiff_t>(encoded_{column});\n",
                    vocab = vocabulary.len(),
                ));
                match mode {
                    CategoricalMode::Native => body.push_str(&format!(
                        "  x.push_back(static_cast<double>(index_{column})); mask.push_back(absent_{column}?1.0:0.0);\n"
                    )),
                    CategoricalMode::Frequency => body.push_str(&format!(
                        "  {{const V counts={counts}; x.push_back(index_{column}<0?0.0:counts[static_cast<std::size_t>(index_{column})]/{total}.0); mask.push_back(absent_{column}?1.0:0.0);}}\n",
                        counts = f64_vec(&counts.iter().map(|v| *v as f64).collect::<Vec<_>>()),
                    )),
                    CategoricalMode::Onehot => {
                        for category in onehot {
                            body.push_str(&format!(
                                "  x.push_back(index_{column}=={category}?1.0:0.0); mask.push_back(0.0);\n"
                            ));
                        }
                    }
                    CategoricalMode::Ordered | CategoricalMode::Hybrid => {
                        let ordered = match model.input.historical.get(column).and_then(Option::as_ref) {
                            Some(state) => emit_historical_ordered(model, state, column)?,
                            None => format!(
                                "const V counts={counts};const V sums={sums};double ordered=index_{column}<0?{global}:(sums[static_cast<std::size_t>(index_{column})]+{prior}*{global})/(counts[static_cast<std::size_t>(index_{column})]+{prior});",
                                counts = f64_vec(&counts.iter().map(|v| *v as f64).collect::<Vec<_>>()),
                                sums = f64_vec(sums),
                                global = f64_lit(*global),
                                prior = f64_lit(model.input.config.prior),
                            ),
                        };
                        body.push_str(&format!(
                            "  {{{ordered}",
                        ));
                        if *mode == CategoricalMode::Hybrid {
                            body.push_str(&format!(
                                "x.push_back(index_{column}<0?0.0:counts[static_cast<std::size_t>(index_{column})]/{total}.0);mask.push_back(absent_{column}?1.0:0.0);"
                            ));
                        }
                        body.push_str(&format!(
                            "x.push_back(ordered);mask.push_back({});}}\n",
                            if *mode == CategoricalMode::Ordered {
                                format!("absent_{column}?1.0:0.0")
                            } else {
                                "0.0".to_string()
                            }
                        ));
                    }
                }
            }
        }
    }
    if missing_aware {
        body.push_str(
            "  for(std::size_t i=0;i<x.size();++i)if(mask[i]!=0.0)x[i]=std::numeric_limits<double>::quiet_NaN();\n",
        );
    }
    body.push_str("  return true;\n}\n");
    Ok(body)
}

fn emit_historical_ordered(
    model: &BoundModel,
    state: &HistoricalTargetState,
    target_column: usize,
) -> Result<String> {
    let FittedRowOrder::Ordered { group_columns, .. } = &model.input.row_order else {
        bail!("historical target state is attached to IID row semantics");
    };
    let global = if state.count == 0 {
        0.0
    } else {
        state.sum / state.count as f64
    };
    let mut body = format!("double ordered={};", f64_lit(global));
    for group in &state.groups {
        if group.key.len() != group_columns.len() {
            bail!("historical group key width differs from fitted row semantics");
        }
        let mut conditions = Vec::new();
        for ((column, kind), key) in group_columns.iter().zip(&group.key) {
            let condition = match (kind, key) {
                (_, SemanticKey::Missing) => format!("missing[{column}]!=0"),
                (ColumnKind::Numeric, SemanticKey::Numeric(value)) => format!(
                    "missing[{column}]==0&&raw[{column}]=={}",
                    f64_lit(f64::from_bits(value.0))
                ),
                (ColumnKind::Categorical, SemanticKey::Categorical(value)) => {
                    let FittedColumn::Categorical { vocabulary, .. } =
                        &model.input.columns[*column]
                    else {
                        bail!("historical categorical key references a numeric fitted column");
                    };
                    let index = vocabulary.binary_search(value).map_err(|_| {
                        anyhow::anyhow!(
                            "historical categorical key is absent from fitted vocabulary"
                        )
                    })?;
                    format!("missing[{column}]==0&&raw[{column}]=={index}.0")
                }
                _ => bail!("historical group key kind differs from fitted schema"),
            };
            conditions.push(condition);
        }
        let condition = if conditions.is_empty() {
            "true".to_string()
        } else {
            conditions.join("&&")
        };
        let fallback = if group.count == 0 {
            global
        } else {
            group.sum / group.count as f64
        };
        body.push_str(&format!(
            "if({condition}){{const double fallback={};ordered=fallback;",
            f64_lit(fallback)
        ));
        for category in &group.categories {
            let statistic = if category.count == 0 && model.input.config.prior == 0.0 {
                fallback
            } else {
                (category.sum + model.input.config.prior * fallback)
                    / (category.count as f64 + model.input.config.prior)
            };
            body.push_str(&format!(
                "if(index_{target_column}=={}.0)ordered={};",
                category.category,
                f64_lit(statistic)
            ));
        }
        body.push('}');
    }
    Ok(body)
}

fn legacy_append(op: &FittedOp) -> bool {
    match op {
        FittedOp::Standardize { append, .. }
        | FittedOp::RobustScale { append, .. }
        | FittedOp::Rank { append, .. }
        | FittedOp::Blom { append, .. }
        | FittedOp::QuantileBins { append, .. }
        | FittedOp::BinOnehot { append, .. }
        | FittedOp::Hinge { append, .. }
        | FittedOp::WinsorClip { append, .. }
        | FittedOp::Cheb { append, .. }
        | FittedOp::PolyInteract { append, .. }
        | FittedOp::RffCos { append, .. }
        | FittedOp::RffSin { append, .. }
        | FittedOp::RandProj { append, .. }
        | FittedOp::LandmarkDist { append, .. }
        | FittedOp::ObliqueHinge { append, .. }
        | FittedOp::RowStats { append }
        | FittedOp::PairProd { append, .. }
        | FittedOp::TargetBinEncode { append, .. }
        | FittedOp::Sigmoid { append }
        | FittedOp::Step { append, .. }
        | FittedOp::NormRow { append }
        | FittedOp::Polar { append }
        | FittedOp::Lag { append, .. }
        | FittedOp::Roll { append, .. }
        | FittedOp::FftMag { append, .. }
        | FittedOp::RowSpectral { append }
        | FittedOp::RowDiffStats { append }
        | FittedOp::SvdProject { append, .. }
        | FittedOp::Nystrom { append, .. }
        | FittedOp::MapExpr { append, .. }
        | FittedOp::DeriveExpr { append, .. }
        | FittedOp::Symlog { append, .. }
        | FittedOp::SelectCorr { append, .. }
        | FittedOp::Impute { append, .. }
        | FittedOp::PairRatio { append, .. }
        | FittedOp::Spline { append, .. } => *append,
    }
}

fn emit_transforms(ops: &[V3Op]) -> Result<String> {
    emit_transforms_named(ops, "transform", false)
}

fn emit_transforms_named(ops: &[V3Op], name: &str, block: bool) -> Result<String> {
    let signature = if block { "Block" } else { "V" };
    let mut body = format!("static {signature} {name}(V x,V mask){{\n  (void)mask;\n");
    for op in ops {
        match op {
            V3Op::Legacy(op) => emit_legacy_op(&mut body, op)?,
            V3Op::MaxAbs { scales, append } => body.push_str(&format!(
                "  {{V input=x,input_mask=mask,g;const V scales={};for(std::size_t j=0;j<std::min(x.size(),scales.size());++j)g.push_back(x[j]/scales[j]);V gm=mask;finish_block(input,input_mask,std::move(g),std::move(gm),{},x,mask);}}\n",
                f64_vec(scales), append
            )),
            V3Op::YeoJohnson { lambdas, append } => body.push_str(&format!(
                "  {{V input=x,input_mask=mask,g;const V lambdas={};for(std::size_t j=0;j<std::min(x.size(),lambdas.size());++j)g.push_back(yj(x[j],lambdas[j]));V gm=mask;finish_block(input,input_mask,std::move(g),std::move(gm),{},x,mask);}}\n",
                f64_vec(lambdas), append
            )),
            V3Op::PiecewiseLinear { edges, append } => {
                body.push_str(&format!("  {{V input=x,input_mask=mask,g,gm;const M edges={};for(std::size_t j=0;j<std::min(x.size(),edges.size());++j)for(std::size_t k=1;k<edges[j].size();++k){{g.push_back(piece(x[j],edges[j][k-1],edges[j][k]));gm.push_back(mask[j]);}}finish_block(input,input_mask,std::move(g),std::move(gm),{},x,mask);}}\n", f64_matrix(edges), append));
            }
            V3Op::CorrScreen { indices, append } => body.push_str(&format!(
                "  {{V input=x,input_mask=mask,g,gm;const U indices={};for(std::size_t j:indices){{g.push_back(x[fitted_index(j,x.size())]);gm.push_back(mask[fitted_index(j,mask.size())]);}}finish_block(input,input_mask,std::move(g),std::move(gm),{},x,mask);}}\n",
                usize_vec(indices), append
            )),
            V3Op::PlsProject {
                means,
                weights,
                append,
            } => body.push_str(&format!(
                "  {{V input=x,input_mask=mask,g;const V means={};const M weights={};for(const V& w:weights){{double s=0.0;for(std::size_t j=0;j<std::min(x.size(),std::min(means.size(),w.size()));++j)s+=w[j]*(x[j]-means[j]);g.push_back(s);}}V gm(g.size(),0.0);finish_block(input,input_mask,std::move(g),std::move(gm),{},x,mask);}}\n",
                f64_vec(means), f64_matrix(weights), append
            )),
            V3Op::Lag { .. } | V3Op::Roll { .. } => {
                unreachable!("table-aware stages are emitted at the batch boundary")
            }
        }
    }
    body.push_str(if block {
        "  return Block{std::move(x),std::move(mask)};\n}\n"
    } else {
        "  return x;\n}\n"
    });
    Ok(body)
}

fn emit_legacy_transforms(ops: &[FittedOp]) -> Result<String> {
    emit_legacy_transforms_named(ops, "transform", false)
}

fn emit_legacy_transforms_named(ops: &[FittedOp], name: &str, block: bool) -> Result<String> {
    let signature = if block { "Block" } else { "V" };
    let mut body = format!("static {signature} {name}(V x,V mask){{\n  (void)mask;\n");
    for op in ops {
        emit_legacy_op_padded(&mut body, op)?;
    }
    body.push_str(if block {
        "  return Block{std::move(x),std::move(mask)};\n}\n"
    } else {
        "  return x;\n}\n"
    });
    Ok(body)
}

fn encoded_row_key_value(
    model: &BoundModel,
    column: usize,
    kind: ColumnKind,
    key: &RowSemanticKey,
) -> Result<String> {
    match (kind, key) {
        (_, RowSemanticKey::Missing) => Ok("KCell{false,0.0}".to_string()),
        (ColumnKind::Numeric, RowSemanticKey::Numeric(value)) => {
            Ok(format!("KCell{{true,{}}}", f64_lit(value.value())))
        }
        (ColumnKind::Categorical, RowSemanticKey::Categorical(value)) => {
            let FittedColumn::Categorical { vocabulary, .. } = &model.input.columns[column] else {
                bail!("row-operation categorical key references a numeric fitted column");
            };
            let index = vocabulary.binary_search(value).map_err(|_| {
                anyhow::anyhow!("row-operation categorical key is absent from fitted vocabulary")
            })?;
            Ok(format!("KCell{{true,{index}.0}}"))
        }
        _ => bail!("row-operation key kind differs from fitted schema"),
    }
}

fn row_key_literal(
    model: &BoundModel,
    columns: &[(usize, ColumnKind)],
    key: &[RowSemanticKey],
) -> Result<String> {
    if columns.len() != key.len() {
        bail!("row-operation stored key width mismatch");
    }
    Ok(format!(
        "Key{{{}}}",
        columns
            .iter()
            .zip(key)
            .map(|((column, kind), key)| encoded_row_key_value(model, *column, *kind, key))
            .collect::<Result<Vec<_>>>()?
            .join(",")
    ))
}

fn tail_rows_literal(tail: &GroupTail) -> String {
    format!(
        "std::vector<Block>{{{}}}",
        tail.rows
            .iter()
            .map(|row| format!(
                "Block{{{},{}}}",
                f64_vec(&row.values),
                f64_vec(
                    &row.missing
                        .iter()
                        .map(|missing| if *missing { 1.0 } else { 0.0 })
                        .collect::<Vec<_>>()
                )
            ))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn emit_row_operation_stage(
    model: &BoundModel,
    k: usize,
    append: bool,
    order: &FittedTableOrder,
    tails: &[GroupTail],
    roll: bool,
) -> Result<String> {
    let group_columns = order
        .group_columns
        .iter()
        .map(|(column, _)| *column)
        .collect::<Vec<_>>();
    let order_columns = order
        .order_columns
        .iter()
        .map(|(column, _)| *column)
        .collect::<Vec<_>>();
    let mut initializers = String::new();
    for tail in tails {
        let group = row_key_literal(model, &order.group_columns, &tail.key)?;
        let cutoff = row_key_literal(model, &order.order_columns, &tail.last_order)?;
        initializers.push_str(&format!(
            "if(key_equal(active_group,{group})){{history={};cutoff={cutoff};has_cutoff=true;}}",
            tail_rows_literal(tail)
        ));
    }
    let generation = if roll {
        format!(
            "for(std::size_t feature=0;feature<width;++feature){{double sum=0.0;std::size_t count=0;std::size_t start=history.size()>{window}?history.size()-{window}:0;for(std::size_t prior=start;prior<history.size();++prior)if(history[prior].mask[feature]==0.0&&std::isfinite(history[prior].x[feature])){{sum+=history[prior].x[feature];++count;}}if(masks[row][feature]==0.0&&std::isfinite(table[row][feature])){{sum+=table[row][feature];++count;}}if(count>0){{generated[feature]=sum/static_cast<double>(count);generated_mask[feature]=0.0;}}}}",
            window = k.saturating_sub(1)
        )
    } else {
        format!(
            "if(history.size()>={k}){{const Block& prior=history[history.size()-{k}];generated=prior.x;generated_mask=prior.mask;}}"
        )
    };
    let tail_limit = if roll { k.saturating_sub(1) } else { k };
    Ok(format!(
        "{{const U group_columns={groups};const U order_columns={orders};std::vector<Key> group_keys(rows),order_keys(rows);for(std::size_t row=0;row<rows;++row){{group_keys[row]=raw_key(raw+row*columns,missing+row*columns,group_columns);order_keys[row]=raw_key(raw+row*columns,missing+row*columns,order_columns);}}U chronological(rows);for(std::size_t row=0;row<rows;++row)chronological[row]=row;std::sort(chronological.begin(),chronological.end(),[&](std::size_t left,std::size_t right){{if(key_less(group_keys[left],group_keys[right]))return true;if(key_less(group_keys[right],group_keys[left]))return false;if(key_less(order_keys[left],order_keys[right]))return true;if(key_less(order_keys[right],order_keys[left]))return false;return left<right;}});Key active_group;Key cutoff;bool active=false,has_cutoff=false;std::vector<Block> history;for(std::size_t row:chronological){{if(!active||!key_equal(active_group,group_keys[row])){{active=true;active_group=group_keys[row];has_cutoff=false;history.clear();{initializers}}}if(has_cutoff&&!key_less(cutoff,order_keys[row]))return false;const std::size_t width=table[row].size();if(masks[row].size()!=width)throw FittedWidthError{{}};for(const Block& prior:history)if(prior.x.size()!=width||prior.mask.size()!=width)throw FittedWidthError{{}};V generated(width,0.0),generated_mask(width,1.0);{generation}Block current{{table[row],masks[row]}};history.push_back(std::move(current));if(history.size()>{tail_limit})history.erase(history.begin(),history.begin()+(history.size()-{tail_limit}));V input=table[row],input_mask=masks[row];finish_block(input,input_mask,std::move(generated),std::move(generated_mask),{append},table[row],masks[row]);}}}}\n",
        groups = usize_vec(&group_columns),
        orders = usize_vec(&order_columns),
    ))
}

fn emit_table_transforms(model: &BoundModel, ops: &[V3Op]) -> Result<String> {
    let mut helpers = String::new();
    let mut body = String::from(
        "static bool transform_table(M& table,M& masks,const double* raw,const std::uint8_t* missing,std::size_t rows,std::size_t columns){\n",
    );
    let mut segment = Vec::new();
    let mut segment_index = 0usize;
    let flush = |segment: &mut Vec<V3Op>,
                 segment_index: &mut usize,
                 helpers: &mut String,
                 body: &mut String|
     -> Result<()> {
        if segment.is_empty() {
            return Ok(());
        }
        let name = format!("table_segment_{}", *segment_index);
        helpers.push_str(&emit_transforms_named(segment, &name, true)?);
        body.push_str(&format!("for(std::size_t row=0;row<rows;++row){{Block block={name}(std::move(table[row]),std::move(masks[row]));table[row]=std::move(block.x);masks[row]=std::move(block.mask);}}\n"));
        segment.clear();
        *segment_index += 1;
        Ok(())
    };
    for op in ops {
        match op {
            V3Op::Lag {
                k,
                append,
                order,
                tails,
            } => {
                flush(&mut segment, &mut segment_index, &mut helpers, &mut body)?;
                body.push_str(&emit_row_operation_stage(
                    model, *k, *append, order, tails, false,
                )?);
            }
            V3Op::Roll {
                k,
                append,
                order,
                tails,
            } => {
                flush(&mut segment, &mut segment_index, &mut helpers, &mut body)?;
                body.push_str(&emit_row_operation_stage(
                    model, *k, *append, order, tails, true,
                )?);
            }
            other => segment.push(other.clone()),
        }
    }
    flush(&mut segment, &mut segment_index, &mut helpers, &mut body)?;
    body.push_str("return true;}\n");
    Ok(format!("{helpers}{body}"))
}

/// A legacy stage of a legacy predictor, whose runtime is `Model::predict`: `ops::finish` pads an
/// empty output row to `[0.0]`, so the emitted stage does too. The padded cell's mask is 0.0, since
/// the legacy model has no missing mask, and the mask is reset with it, since it may be longer
/// than the row. The V3 pipeline path emits `emit_legacy_op` alone: its runtime
/// (`finish_generated_from_sets`) does not pad.
fn emit_legacy_op_padded(body: &mut String, op: &FittedOp) -> Result<()> {
    emit_legacy_op(body, op)?;
    body.push_str("  if(x.empty()){x.assign(1,0.0);mask.assign(1,0.0);}\n");
    Ok(())
}

fn emit_legacy_op(body: &mut String, op: &FittedOp) -> Result<()> {
    let append = legacy_append(op);
    let generated_mask = match op {
        FittedOp::Standardize { .. }
        | FittedOp::RobustScale { .. }
        | FittedOp::WinsorClip { .. }
        | FittedOp::TargetBinEncode { .. }
        | FittedOp::Sigmoid { .. }
        | FittedOp::Step { .. }
        | FittedOp::NormRow { .. } => "V gm=mask;".to_string(),
        FittedOp::SelectCorr { indices, .. } => format!(
            "V gm;const U mask_indices={};for(std::size_t j:mask_indices)gm.push_back(mask[fitted_index(j,mask.size())]);",
            usize_vec(indices)
        ),
        FittedOp::Impute { .. } => "V gm(g.size(),0.0);".to_string(),
        _ => "V gm(g.size(),0.0);".to_string(),
    };
    let generated = match op {
        FittedOp::Standardize { means, scales, .. } => format!(
            "const V a={};const V b={};for(std::size_t j=0;j<std::min(x.size(),std::min(a.size(),b.size()));++j)g.push_back(clampv((x[j]-a[j])/b[j],-20.0,20.0));",
            f64_vec(means), f64_vec(scales)
        ),
        FittedOp::RobustScale {
            medians, scales, ..
        } => format!(
            "const V a={};const V b={};for(std::size_t j=0;j<std::min(x.size(),std::min(a.size(),b.size()));++j)g.push_back((x[j]-a[j])/b[j]);",
            f64_vec(medians), f64_vec(scales)
        ),
        FittedOp::Rank { sorted_cols, .. } => format!(
            "const M a={};for(std::size_t j=0;j<std::min(x.size(),a.size());++j)g.push_back(mid_rank(a[j],x[j]));",
            f64_matrix(sorted_cols)
        ),
        FittedOp::Blom { sorted_cols, .. } => format!(
            "static const M a={};for(std::size_t j=0;j<std::min(x.size(),a.size());++j){{const double n=static_cast<double>(a[j].size());const double mid=mid_rank(a[j],x[j])*n;g.push_back(probit((mid+0.125)/(n+0.25)));}}",
            f64_matrix(sorted_cols)
        ),
        FittedOp::QuantileBins { edges, bins, .. } => format!(
            "static const M a={};for(std::size_t j=0;j<std::min(x.size(),a.size());++j)g.push_back(static_cast<double>(lower(a[j],x[j]))/({}.0-1.0));",
            f64_matrix(edges), bins
        ),
        FittedOp::BinOnehot { edges, bins, .. } => format!(
            "const M a={};for(std::size_t j=0;j<std::min(x.size(),a.size());++j){{const std::size_t bin=std::min(lower(a[j],x[j]),static_cast<std::size_t>({}-1));for(std::size_t q=0;q<{};++q)g.push_back(q==bin?1.0:0.0);}}",
            f64_matrix(edges), bins, bins
        ),
        FittedOp::Hinge { knots, .. } => format!(
            "const M a={};for(std::size_t j=0;j<std::min(x.size(),a.size());++j)for(double knot:a[j])g.push_back(std::max(x[j]-knot,0.0));",
            f64_matrix(knots)
        ),
        FittedOp::WinsorClip { lo, hi, .. } => format!(
            "const V a={};const V b={};for(std::size_t j=0;j<std::min(x.size(),std::min(a.size(),b.size()));++j)g.push_back(clampv(x[j],a[j],b[j]));",
            f64_vec(lo), f64_vec(hi)
        ),
        FittedOp::TargetBinEncode { edges, enc, .. } => format!(
            "const M a={};const M b={};for(std::size_t j=0;j<std::min(x.size(),std::min(a.size(),b.size()));++j){{if(b[j].empty())throw FittedWidthError{{}};g.push_back(b[j][std::min(lower(a[j],x[j]),b[j].size()-1)]);}}",
            f64_matrix(edges), f64_matrix(enc)
        ),
        FittedOp::Sigmoid { .. } => "for(double v:x)g.push_back(1.0/(1.0+std::exp(-clampv(v,-30.0,30.0))));".to_string(),
        FittedOp::Step { t, .. } => format!("for(double v:x)g.push_back(v>{}?1.0:0.0);", f64_lit(*t)),
        FittedOp::NormRow { .. } => "double s=0.0;for(double v:x)s+=v*v;double inv=1.0/std::max(std::sqrt(s),1e-9);for(double v:x)g.push_back(v*inv);".to_string(),
        FittedOp::RowStats { .. } => "g=row_stats(x);".to_string(),
        FittedOp::FftMag { k, .. } => format!("g=dft_mags(x,{k});"),
        FittedOp::RowSpectral { .. } => "g=row_spectral(x);".to_string(),
        FittedOp::RowDiffStats { .. } => "g=row_diff_stats(x);".to_string(),
        FittedOp::Cheb { degree, .. } => format!("for(double v:x){{const double u=clampv(2.0*v-1.0,-1.0,1.0);double previous=1.0,current=u;for(std::size_t q=0;q<{};++q){{g.push_back(current);const double next=2.0*u*current-previous;previous=current;current=next;}}}}", degree),
        FittedOp::SelectCorr { indices, .. } => format!("const U a={};for(std::size_t j:a)g.push_back(x[fitted_index(j,x.size())]);", usize_vec(indices)),
        FittedOp::Impute { fill, indicator_cols, .. } => format!("const V a={};const U cols={};for(std::size_t j=0;j<x.size();++j)g.push_back(x[j]==0.0?a[fitted_index(j,a.size())]:x[j]);for(std::size_t j:cols)g.push_back(x[fitted_index(j,x.size())]==0.0?1.0:0.0);", f64_vec(fill), usize_vec(indicator_cols)),
        FittedOp::PairProd { pairs, .. } | FittedOp::PolyInteract { pairs, .. } => {
            let pairs = pairs.iter().map(|(a,b)| format!("{{{a},{b}}}")).collect::<Vec<_>>().join(",");
            format!("static const std::vector<std::pair<std::size_t,std::size_t>> pairs{{{pairs}}};for(const auto& p:pairs)g.push_back(x[fitted_index(p.first,x.size())]*x[fitted_index(p.second,x.size())]);")
        }
        FittedOp::PairRatio { pairs, eps, .. } => {
            let pairs = pairs.iter().map(|(a,b)| format!("{{{a},{b}}}")).collect::<Vec<_>>().join(",");
            format!("const std::vector<std::pair<std::size_t,std::size_t>> pairs{{{pairs}}};const V eps={};for(std::size_t i=0;i<pairs.size();++i){{double a=x[fitted_index(pairs[i].first,x.size())],b=x[fitted_index(pairs[i].second,x.size())];g.push_back(a*b/(b*b+eps[fitted_index(i,eps.size())]));}}", f64_vec(eps))
        }
        FittedOp::RffCos { w, b, .. } => format!("const M a={};const V b={};const double scale=std::sqrt(2.0/static_cast<double>(a.size()));for(std::size_t i=0;i<a.size();++i)g.push_back(scale*std::cos(dot(a[i],x)+b[fitted_index(i,b.size())]));", f64_matrix(w), f64_vec(b)),
        FittedOp::RffSin { w, b, .. } => format!("const M a={};const V b={};const double scale=std::sqrt(2.0/static_cast<double>(a.size()));for(std::size_t i=0;i<a.size();++i)g.push_back(scale*std::sin(dot(a[i],x)+b[fitted_index(i,b.size())]));", f64_matrix(w), f64_vec(b)),
        FittedOp::RandProj { w, .. } => format!("const M a={};double inv=1.0/std::sqrt(static_cast<double>(a.size()));for(const V& row:a)g.push_back(dot(row,x)*inv);", f64_matrix(w)),
        FittedOp::LandmarkDist { landmarks, gamma, .. } | FittedOp::Nystrom { landmarks, gamma, .. } => format!("const M a={};for(const V& row:a){{double s=0.0;for(std::size_t j=0;j<std::min(row.size(),x.size());++j){{double q=x[j]-row[j];s+=q*q;}}g.push_back(std::exp(-{}*s));}}", f64_matrix(landmarks), f64_lit(*gamma)),
        FittedOp::ObliqueHinge { w, t, .. } => format!("const M a={};const V b={};for(std::size_t i=0;i<a.size();++i)g.push_back(std::max(dot(a[i],x)-b[fitted_index(i,b.size())],0.0));", f64_matrix(w), f64_vec(t)),
        FittedOp::SvdProject { means, components, .. } => format!("const V a={};const M b={};for(const V& row:b){{double s=0.0;for(std::size_t j=0;j<std::min(x.size(),std::min(a.size(),row.size()));++j)s+=(x[j]-a[j])*row[j];g.push_back(s);}}", f64_vec(means), f64_matrix(components)),
        FittedOp::Symlog { c, .. } => format!("for(double v:x){{const double q=v/{};const double a=std::abs(q),s=q>0.0?1.0:(q<0.0?-1.0:0.0);g.push_back(s*std::log(a+std::sqrt(q*q+1.0)));}}", f64_lit(*c)),
        FittedOp::Polar { .. } => "for(std::size_t j=0;j+1<x.size();j+=2){const double a=x[j],b=x[j+1];g.push_back(std::sqrt(a*a+b*b));g.push_back(std::atan2(b,a));}if(x.size()%2==1)g.push_back(x.back());".to_string(),
        FittedOp::Lag { k, .. } => format!("for(std::size_t j=0;j<x.size();++j)g.push_back(j>={0}?x[j-{0}]:0.0);", k),
        FittedOp::Roll { k, .. } => format!("if(x.empty())throw FittedWidthError{{}};const std::size_t shift={}%x.size();for(std::size_t j=0;j<x.size();++j)g.push_back(x[(j+x.size()-shift)%x.size()]);", k),
        FittedOp::Spline { knots, .. } => format!("const M knots={};for(std::size_t j=0;j<std::min(x.size(),knots.size());++j){{const V& xi=knots[j];const std::size_t n=xi.size();if(n<2)throw FittedWidthError{{}};const double v=x[j];if(xi[n-1]-xi[0]<1e-9){{g.push_back(v);continue;}}g.push_back(v);const double pl=std::max(v-xi[n-1],0.0),pk=std::max(v-xi[n-2],0.0);const double dlast=(pk*pk*pk-pl*pl*pl)/std::max(xi[n-1]-xi[n-2],1e-9);for(std::size_t q=0;q<n-2;++q){{const double pt=std::max(v-xi[q],0.0);const double d=(pt*pt*pt-pl*pl*pl)/std::max(xi[n-1]-xi[q],1e-9);g.push_back(d-dlast);}}}}", f64_matrix(knots)),
        FittedOp::MapExpr { expr, .. } => {
            let expression = expr.to_cpp17(true)?;
            format!("const double* row=x.data();const std::size_t row_len=x.size();(void)row;(void)row_len;for(double element:x){{double x=element;(void)x;g.push_back(math_safe({expression}));}}")
        }
        FittedOp::DeriveExpr { expr, .. } => {
            let expression = expr.to_cpp17(false)?;
            format!("const double* row=x.data();const std::size_t row_len=x.size();(void)row;(void)row_len;g.push_back(math_safe({expression}));")
        }
    };
    body.push_str(&format!(
        "  {{V input=x,input_mask=mask,g;{generated}{generated_mask}finish_block(input,input_mask,std::move(g),std::move(gm),{append},x,mask);}}\n"
    ));
    Ok(())
}

fn emit_head_function(name: &str, head: &V3Head, out: &mut String) -> Result<()> {
    match head {
        V3Head::Legacy(head) => emit_legacy_head(name, head, out),
        V3Head::LogitRidge(model) => {
            emit_linear(
                name,
                &model.means,
                &model.scales,
                &model.weights,
                None,
                -30.0,
                30.0,
                true,
                out,
            );
            Ok(())
        }
        V3Head::TabBoost(model) => {
            let mut body = format!(
                "static double {name}(const V& x){{double value={};",
                f64_lit(model.base)
            );
            for stump in &model.stumps {
                body.push_str(&format!("{{const double cell=x[{feature}];const bool left=std::isfinite(cell)?cell<={threshold}:{missing};value+={lr}*(left?{left}:{right});}}", feature=stump.feature,threshold=f64_lit(stump.threshold),missing=stump.missing_left,lr=f64_lit(model.learning_rate),left=f64_lit(stump.left),right=f64_lit(stump.right)));
            }
            body.push_str(if model.binary {
                "return sigmoid(value);}\n"
            } else {
                "return finitev(value);}\n"
            });
            out.push_str(&body);
            Ok(())
        }
        V3Head::ExtraTrees(model) => {
            let terms = model.trees.iter().map(extra_expr).collect::<Vec<_>>();
            let sum = if terms.is_empty() {
                "0.0".to_string()
            } else {
                terms.join("+")
            };
            let finish = if model.binary {
                format!("clampv(({sum})/{}.0,0.0,1.0)", terms.len().max(1))
            } else {
                format!("finitev(({sum})/{}.0)", terms.len().max(1))
            };
            out.push_str(&format!(
                "static double {name}(const V& x){{return {finish};}}\n"
            ));
            Ok(())
        }
        V3Head::GamBoost(model) => {
            let mut body = format!(
                "static double {name}(const V& x){{double value={};",
                f64_lit(model.base)
            );
            for term in &model.terms {
                body.push_str(&format!("{{const V edges={};const V values={};const std::size_t bin=std::min(lower(edges,x[{}]),values.size()-1);value+={}*values[bin];}}", f64_vec(&term.edges),f64_vec(&term.values),term.feature,f64_lit(model.learning_rate)));
            }
            body.push_str(if model.binary {
                "return sigmoid(value);}\n"
            } else {
                "return finitev(value);}\n"
            });
            out.push_str(&body);
            Ok(())
        }
        V3Head::Pls(model) => {
            let V3Op::PlsProject { means, weights, .. } = &model.projection else {
                unreachable!("fitted PLS head stores a PLS projection")
            };
            out.push_str(&format!("static double {name}(const V& x){{const V center={};const M projection={};V p;for(const V& w:projection){{double s=0.0;for(std::size_t j=0;j<std::min(x.size(),std::min(center.size(),w.size()));++j)s+=w[j]*(x[j]-center[j]);p.push_back(s);}}const V means={};const V scales={};const V weights={};double value=weights[0];for(std::size_t j=0;j<means.size();++j)value+=weights[j+1]*clampv((p[j]-means[j])/scales[j],-20.0,20.0);return clampv(value,-60.0,60.0);}}\n", f64_vec(means),f64_matrix(weights),f64_vec(&model.ridge.means),f64_vec(&model.ridge.scales),f64_vec(&model.ridge.w)));
            Ok(())
        }
        V3Head::ClassificationPortfolio(model) => {
            let mut terms = Vec::new();
            for (index, (member, weight)) in model.members.iter().zip(&model.weights).enumerate() {
                let child = format!("{name}_{index}");
                emit_head_function(&child, member, out)?;
                terms.push(format!("{}*{}(x)", f64_lit(*weight), child));
            }
            out.push_str(&format!(
                "static double {name}(const V& x){{return clampv({},0.0,1.0);}}\n",
                if terms.is_empty() {
                    "0.0".to_string()
                } else {
                    terms.join("+")
                }
            ));
            Ok(())
        }
        V3Head::MulticlassRidge(_)
        | V3Head::MulticlassOvrLogistic(_)
        | V3Head::MulticlassDiagonalLda(_)
        | V3Head::MulticlassNearestCentroid(_) => {
            bail!("multiclass probability matrices are not supported by the scalar C++ ABI")
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_linear(
    name: &str,
    means: &[f64],
    scales: &[f64],
    weights: &[f64],
    columns: Option<&[usize]>,
    lo: f64,
    hi: f64,
    probability: bool,
    out: &mut String,
) {
    let mapping = columns.map_or_else(
        || "const std::size_t source=j;".to_string(),
        |columns| {
            format!(
                "const U columns={};const std::size_t source=columns[j];",
                usize_vec(columns)
            )
        },
    );
    let finish = if probability {
        "sigmoid(value)"
    } else {
        "clampv(value,-60.0,60.0)"
    };
    out.push_str(&format!("static double {name}(const V& x){{const V means={};const V scales={};const V weights={};double value=weights[0];for(std::size_t j=0;j<means.size();++j){{{mapping}if(source<x.size())value+=weights[j+1]*clampv((x[source]-means[j])/scales[j],{},{});}}return {finish};}}\n",f64_vec(means),f64_vec(scales),f64_vec(weights),f64_lit(lo),f64_lit(hi)));
}

fn emit_legacy_head(name: &str, head: &FittedHead, out: &mut String) -> Result<()> {
    match head {
        FittedHead::Ridge(model) => {
            emit_linear(
                name,
                &model.means,
                &model.scales,
                &model.w,
                None,
                -20.0,
                20.0,
                false,
                out,
            );
            Ok(())
        }
        FittedHead::RidgeGcv(model) => {
            emit_linear(
                name,
                &model.means,
                &model.scales,
                &model.w,
                Some(&model.cols),
                f64::NEG_INFINITY,
                f64::INFINITY,
                false,
                out,
            );
            Ok(())
        }
        FittedHead::ElasticNet(model) => {
            out.push_str(&format!("static double {name}(const V& x){{static const V means={};static const V scales={};static const V weights={};double value={};for(std::size_t j=0;j<means.size();++j)value+=weights[j]*clampv((x[j]-means[j])/scales[j],-20.0,20.0);return clampv(value,-60.0,60.0);}}\n",f64_vec(&model.means),f64_vec(&model.scales),f64_vec(&model.w),f64_lit(model.w0)));
            Ok(())
        }
        FittedHead::Gbm(model) => emit_gbm(name, model, out),
        FittedHead::HistGbm2(model) => emit_hist_gbm2(name, model, out),
        FittedHead::Ensemble { ridge, gbm, w } => {
            let ridge_name = format!("{name}_ridge");
            let gbm_name = format!("{name}_gbm");
            emit_linear(
                &ridge_name,
                &ridge.means,
                &ridge.scales,
                &ridge.w,
                None,
                -20.0,
                20.0,
                false,
                out,
            );
            emit_gbm(&gbm_name, gbm, out)?;
            out.push_str(&format!("static double {name}(const V& x){{return {}*{ridge_name}(x)+(1.0-{})*{gbm_name}(x);}}\n",f64_lit(*w),f64_lit(*w)));
            Ok(())
        }
        FittedHead::Krr(model) => {
            out.push_str(&format!("static double {name}(const V& x){{const V means={};const V scales={};const M basis={};const V alpha={};if(basis.empty())return clampv({},-60.0,60.0);V z;for(std::size_t j=0;j<std::min(x.size(),means.size());++j)z.push_back((x[j]-means[j])/scales[j]);double value={};for(std::size_t i=0;i<basis.size();++i){{double r2=0.0;for(std::size_t j=0;j<std::min(z.size(),basis[i].size());++j){{double q=z[j]-basis[i][j];r2+=q*q;}}value+=alpha[i]*std::exp(-{}*r2);}}return clampv(value,-60.0,60.0);}}\n",f64_vec(&model.means),f64_vec(&model.scales),f64_matrix(&model.basis),f64_vec(&model.alpha),f64_lit(model.y_mean),f64_lit(model.y_mean),f64_lit(model.gamma)));
            Ok(())
        }
        FittedHead::Gpr(model) => {
            let inverse = model
                .ls
                .iter()
                .map(|length| 1.0 / (length * length))
                .collect::<Vec<_>>();
            out.push_str(&format!("static double {name}(const V& x){{const V means={};const V scales={};const M basis={};const V alpha={};const V inv={};if(basis.empty())return clampv({},-60.0,60.0);V z;for(std::size_t j=0;j<std::min(x.size(),means.size());++j)z.push_back((x[j]-means[j])/scales[j]);double value={};for(std::size_t i=0;i<basis.size();++i){{double r2=0.0;for(std::size_t j=0;j<std::min(z.size(),basis[i].size());++j){{double q=z[j]-basis[i][j];r2+=q*q*inv[j];}}value+=alpha[i]*{}*std::exp(-0.5*r2);}}return clampv(value,-60.0,60.0);}}\n",f64_vec(&model.means),f64_vec(&model.scales),f64_matrix(&model.basis),f64_vec(&model.alpha),f64_vec(&inverse),f64_lit(model.y_mean),f64_lit(model.y_mean),f64_lit(model.sf2)));
            Ok(())
        }
        FittedHead::GprKernel(model) => emit_gpr_kernel(name, model, out),
        FittedHead::Ksmooth(model) => {
            out.push_str(&format!("static double {name}(const V& x){{const V means={};const V scales={};const M basis={};const V ys={};if(basis.empty())return clampv({},-60.0,60.0);V z;for(std::size_t j=0;j<std::min(x.size(),means.size());++j)z.push_back((x[j]-means[j])/scales[j]);std::vector<std::pair<double,std::size_t>> order;for(std::size_t i=0;i<basis.size();++i){{double r2=0.0;for(std::size_t j=0;j<std::min(z.size(),basis[i].size());++j){{double q=z[j]-basis[i][j];r2+=q*q;}}order.push_back({{r2,i}});}}std::sort(order.begin(),order.end(),[](const auto&a,const auto&b){{return a.first<b.first||(a.first==b.first&&a.second<b.second);}});const double base=order[0].first;double num=0.0,den=0.0;for(std::size_t q=0;q<std::min<std::size_t>({},order.size());++q){{double w=std::exp(-{}*(order[q].first-base)/static_cast<double>(means.size()));num+=w*ys[order[q].second];den+=w;}}return clampv(num/den,-60.0,60.0);}}\n",f64_vec(&model.means),f64_vec(&model.scales),f64_matrix(&model.z),f64_vec(&model.ys),f64_lit(model.y_mean),model.k,f64_lit(model.gamma)));
            Ok(())
        }
        FittedHead::Hurdle2(model) => emit_hurdle(name, model, out),
        FittedHead::Stack(model) => emit_stack(name, model, out),
        FittedHead::Portfolio(model) => emit_portfolio(name, model, out),
    }
}

fn emit_unclamped_ridge(name: &str, model: &crate::heads::RidgeModel, out: &mut String) {
    out.push_str(&format!(
        "static double {name}(const V& x){{const V means={};const V scales={};const V weights={};double value=weights[0];for(std::size_t j=0;j<std::min(x.size(),means.size());++j)value+=weights[j+1]*clampv((x[j]-means[j])/scales[j],-20.0,20.0);return value;}}\n",
        f64_vec(&model.means),
        f64_vec(&model.scales),
        f64_vec(&model.w),
    ));
}

type DecodedGprKernel = (Vec<f64>, Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>, Vec<Vec<f64>>);

fn decode_gpr_kernel(model: &crate::gpr::GprKernelModel) -> Result<DecodedGprKernel> {
    let width = model.means.len();
    let mut signal = Vec::new();
    let mut inverse = Vec::new();
    let mut alpha = Vec::new();
    let mut period = Vec::new();
    let mut offset = 0usize;
    for term in &model.spec {
        let amplitude = *model
            .theta
            .get(offset)
            .context("gpr kernel theta is truncated at amplitude")?;
        signal.push((2.0 * amplitude).exp());
        offset += 1;
        let mut inverse_term = Vec::new();
        let mut alpha_term = Vec::new();
        let mut period_term = Vec::new();
        for primitive in term {
            match primitive {
                KernelPrim::Rbf => {
                    let mut values = Vec::with_capacity(width);
                    for dimension in 0..width {
                        let length = model
                            .theta
                            .get(offset + dimension)
                            .context("gpr kernel theta is truncated at RBF lengthscale")?
                            .exp();
                        values.push(1.0 / (length * length));
                    }
                    offset += width;
                    inverse_term.push(values);
                    alpha_term.push(0.0);
                    period_term.push(0.0);
                }
                KernelPrim::Rq => {
                    let mut values = Vec::with_capacity(width);
                    for dimension in 0..width {
                        let length = model
                            .theta
                            .get(offset + dimension)
                            .context("gpr kernel theta is truncated at RQ lengthscale")?
                            .exp();
                        values.push(1.0 / (length * length));
                    }
                    let shape = model
                        .theta
                        .get(offset + width)
                        .context("gpr kernel theta is truncated at RQ alpha")?
                        .exp();
                    offset += width + 1;
                    inverse_term.push(values);
                    alpha_term.push(shape);
                    period_term.push(0.0);
                }
                KernelPrim::Periodic => {
                    let cycle = model
                        .theta
                        .get(offset)
                        .context("gpr kernel theta is truncated at period")?
                        .exp();
                    let length = model
                        .theta
                        .get(offset + 1)
                        .context("gpr kernel theta is truncated at periodic lengthscale")?
                        .exp();
                    offset += 2;
                    inverse_term.push(vec![1.0 / (length * length)]);
                    alpha_term.push(0.0);
                    period_term.push(cycle);
                }
                KernelPrim::Linear => {
                    inverse_term.push(Vec::new());
                    alpha_term.push(0.0);
                    period_term.push(0.0);
                }
            }
        }
        inverse.push(inverse_term);
        alpha.push(alpha_term);
        period.push(period_term);
    }
    if model.theta.get(offset).is_none() {
        bail!("gpr kernel theta is truncated at observation noise");
    }
    Ok((signal, inverse, alpha, period))
}

fn emit_gpr_kernel(name: &str, model: &crate::gpr::GprKernelModel, out: &mut String) -> Result<()> {
    if model.basis.is_empty() {
        out.push_str(&format!(
            "static double {name}(const V&){{return clampv({},-60.0,60.0);}}\n",
            f64_lit(model.y_mean)
        ));
        return Ok(());
    }
    let (signal, inverse, alpha, period) = decode_gpr_kernel(model)?;
    let mut kernel = String::from("double kernel=0.0;");
    for (term_index, term) in model.spec.iter().enumerate() {
        kernel.push_str(&format!("{{double term={};", f64_lit(signal[term_index])));
        for (factor_index, primitive) in term.iter().enumerate() {
            match primitive {
                KernelPrim::Rbf => kernel.push_str(&format!(
                    "{{const V inv={};double distance=0.0;for(std::size_t d=0;d<std::min(z.size(),basis_row.size());++d){{double delta=z[d]-basis_row[d];distance+=delta*delta*inv[d];}}term*=std::exp(-0.5*distance);}}",
                    f64_vec(&inverse[term_index][factor_index])
                )),
                KernelPrim::Rq => kernel.push_str(&format!(
                    "{{const V inv={};double distance=0.0;for(std::size_t d=0;d<std::min(z.size(),basis_row.size());++d){{double delta=z[d]-basis_row[d];distance+=delta*delta*inv[d];}}const double shape={};const double scaled=distance/(2.0*shape);term*=std::exp(-shape*std::log(1.0+scaled));}}",
                    f64_vec(&inverse[term_index][factor_index]),
                    f64_lit(alpha[term_index][factor_index]),
                )),
                KernelPrim::Periodic => kernel.push_str(&format!(
                    "{{const double period={};const double inv={};double sine2=0.0;for(std::size_t d=0;d<std::min(z.size(),basis_row.size());++d){{double angle={}*(z[d]-basis_row[d])/period;double sine=std::sin(angle);sine2+=sine*sine;}}term*=std::exp(-2.0*sine2*inv);}}",
                    f64_lit(period[term_index][factor_index]),
                    f64_lit(inverse[term_index][factor_index][0]),
                    f64_lit(std::f64::consts::PI),
                )),
                KernelPrim::Linear => kernel.push_str(&format!(
                    "{{double product=0.0;for(std::size_t d=0;d<std::min(z.size(),basis_row.size());++d)product+=z[d]*basis_row[d];term*=product/{}.0;}}",
                    model.means.len()
                )),
            }
        }
        kernel.push_str("kernel+=term;}");
    }
    out.push_str(&format!(
        "static double {name}(const V& x){{const V means={};const V scales={};const M basis={};const V coefficients={};if(basis.empty())return clampv({},-60.0,60.0);V z;for(std::size_t j=0;j<std::min(x.size(),means.size());++j)z.push_back((x[j]-means[j])/scales[j]);double value={};for(std::size_t row_index=0;row_index<basis.size();++row_index){{const V& basis_row=basis[row_index];{kernel}value+=coefficients[row_index]*kernel;}}return clampv(value,-60.0,60.0);}}\n",
        f64_vec(&model.means),
        f64_vec(&model.scales),
        f64_matrix(&model.basis),
        f64_vec(&model.alpha),
        f64_lit(model.y_mean),
        f64_lit(model.y_mean),
    ));
    Ok(())
}

fn emit_hurdle(name: &str, model: &crate::heads::Hurdle2Model, out: &mut String) -> Result<()> {
    if model.components.is_empty() {
        out.push_str(&format!(
            "static double {name}(const V&){{return {};}}\n",
            f64_lit(model.fallback_mean)
        ));
        return Ok(());
    }
    let mut terms = Vec::new();
    for (index, component) in model.components.iter().enumerate() {
        let gate = format!("{name}_gate_{index}");
        let low = format!("{name}_low_{index}");
        let high = format!("{name}_high_{index}");
        emit_linear(
            &gate,
            &component.gate.means,
            &component.gate.scales,
            &component.gate.w,
            None,
            -20.0,
            20.0,
            true,
            out,
        );
        emit_conditional_ridge(&low, &component.low, out);
        emit_conditional_ridge(&high, &component.high, out);
        terms.push(format!(
            "{{double probability={gate}(x);value+=probability*{high}(x)+(1.0-probability)*{low}(x);}}"
        ));
    }
    out.push_str(&format!(
        "static double {name}(const V& x){{double value=0.0;{}value*=1.0/{}.0;return std::isfinite(value)?value:{};}}\n",
        terms.join(""),
        model.components.len(),
        f64_lit(model.fallback_mean),
    ));
    Ok(())
}

fn emit_conditional_ridge(name: &str, model: &ConditionalRidge, out: &mut String) {
    match model {
        ConditionalRidge::Ridge(model) => emit_unclamped_ridge(name, model, out),
        ConditionalRidge::Mean(value) => out.push_str(&format!(
            "static double {name}(const V&){{return {};}}\n",
            f64_lit(*value)
        )),
    }
}

fn emit_stack(name: &str, model: &crate::stack::StackModel, out: &mut String) -> Result<()> {
    let mut terms = Vec::new();
    let members = [
        model
            .ridge_a
            .as_ref()
            .map(|model| FittedHead::Ridge(model.clone())),
        model
            .ridge_b
            .as_ref()
            .map(|model| FittedHead::Ridge(model.clone())),
        model
            .net
            .as_ref()
            .map(|model| FittedHead::ElasticNet(model.clone())),
        model
            .trees
            .as_ref()
            .map(|model| FittedHead::HistGbm2(model.clone())),
        model
            .kernel
            .as_ref()
            .map(|model| FittedHead::Krr(model.clone())),
    ];
    for (index, member) in members.into_iter().enumerate() {
        let Some(member) = member else { continue };
        let child = format!("{name}_{index}");
        emit_legacy_head(&child, &member, out)?;
        let weight = model.weights.get(index).copied().unwrap_or(0.0);
        terms.push(format!("{}*{child}(x)", f64_lit(weight)));
    }
    out.push_str(&format!(
        "static double {name}(const V& x){{return {};}}\n",
        if terms.is_empty() {
            "0.0".to_string()
        } else {
            terms.join("+")
        }
    ));
    Ok(())
}

fn emit_legacy_model(name: &str, model: &crate::model::Model, out: &mut String) -> Result<()> {
    match model {
        crate::model::Model::Pipeline { target, ops, head } => {
            let head_name = format!("{name}_head");
            let transform_name = format!("{name}_transform");
            emit_legacy_head(&head_name, head, out)?;
            out.push_str(&format!(
                "static V {transform_name}(V x){{V mask(x.size(),0.0);\n"
            ));
            for op in ops {
                emit_legacy_op_padded(out, op)?;
            }
            out.push_str("return x;}\n");
            out.push_str(&format!(
                "static double {name}(const V& source){{V x={transform_name}(source);double value={head_name}(x);{}return finitev(value);}}\n",
                emit_target_inverse_value(target)
            ));
            Ok(())
        }
        crate::model::Model::Symbolic { expr, a, b } => {
            let expression = expr.to_cpp17(false)?;
            out.push_str(&format!(
                "static double {name}(const V& values){{const double* row=values.data();const std::size_t row_len=values.size();(void)row;(void)row_len;double value=math_safe({expression});double result={a}*value+{b};return std::isfinite(result)?clampv(result,-60.0,60.0):0.0;}}\n",
                a = f64_lit(*a),
                b = f64_lit(*b),
            ));
            Ok(())
        }
        crate::model::Model::Portfolio(model) => emit_portfolio(name, model, out),
    }
}

fn emit_target_inverse_value(target: &Option<FittedTarget>) -> String {
    match target {
        None => String::new(),
        Some(FittedTarget::RankGauss {
            table_values,
            table_scores,
            ..
        }) => format!(
            "value=clampv(rank_inverse(value,{},{}),-60.0,60.0);",
            f64_vec(table_scores),
            f64_vec(table_values)
        ),
        Some(FittedTarget::Log1p) => {
            "value=clampv(std::exp(clampv(value,-745.0,700.0))-1.0,-60.0,60.0);".to_string()
        }
        Some(FittedTarget::Winsor { .. }) => "value=clampv(value,-60.0,60.0);".to_string(),
        Some(FittedTarget::ZScore {
            mu,
            sigma,
            clamp_lo,
            clamp_hi,
        }) => format!(
            "value=clampv(value*{}+{},{},{});",
            f64_lit(*sigma),
            f64_lit(*mu),
            f64_lit(*clamp_lo),
            f64_lit(*clamp_hi)
        ),
    }
}

fn u64_vec(values: &[u64]) -> String {
    format!(
        "B{{{}}}",
        values
            .iter()
            .map(|value| format!("0x{value:016x}ULL"))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn emit_group_mean(name: &str, model: &crate::portfolio::GroupMeanModel, out: &mut String) {
    let entries = model
        .table
        .iter()
        .map(|(key, value)| {
            format!(
                "std::pair<B,double>{{{},{}}}",
                u64_vec(key),
                f64_lit(*value)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    out.push_str(&format!(
        "static double {name}(const V& x){{B key;key.reserve(x.size());for(double value:x){{double rounded=std::floor(value*1e9+0.5)/1e9+0.0;key.push_back(bits(rounded));}}const std::vector<std::pair<B,double>> table{{{entries}}};auto found=std::lower_bound(table.begin(),table.end(),key,[](const auto& item,const B& wanted){{return item.first<wanted;}});return found!=table.end()&&found->first==key?found->second:{};}}\n",
        f64_lit(model.g_mean)
    ));
}

fn emit_portfolio(
    name: &str,
    model: &crate::portfolio::PortfolioModel,
    out: &mut String,
) -> Result<()> {
    if model.mean_only {
        out.push_str(&format!(
            "static double {name}(const V&){{return {};}}\n",
            f64_lit(model.y_mean)
        ));
        return Ok(());
    }
    let projection_name = format!("{name}_project");
    if let Some(projection) = &model.proj {
        out.push_str(&format!(
            "static V {projection_name}(V x){{V mask(x.size(),0.0);\n"
        ));
        emit_legacy_op_padded(out, projection)?;
        out.push_str("return x;}\n");
    }
    let mut member_terms = Vec::new();
    for (index, member) in model.members.iter().enumerate() {
        let weight = model.weights.get(index).copied().unwrap_or(0.0);
        if weight <= 0.0 {
            continue;
        }
        let child = format!("{name}_member_{index}");
        match &member.state {
            FittedMemberState::Fitted(model) => emit_legacy_model(&child, model, out)?,
            FittedMemberState::GroupMean(model) => emit_group_mean(&child, model, out),
            FittedMemberState::Mean | FittedMemberState::Skipped => continue,
        }
        let view = if member.proj_view && model.proj.is_some() {
            "projected"
        } else {
            "raw"
        };
        member_terms.push(format!(
            "{{double member={child}({view});if(!std::isfinite(member))member=y_mean;member=clampv(member,clip_lo,clip_hi);value+={weight}*(member-y_mean);}}",
            weight = f64_lit(weight)
        ));
    }
    let projection = if model.proj.is_some() {
        format!("V projected={projection_name}(raw);")
    } else {
        "V projected;".to_string()
    };
    out.push_str(&format!(
        "static double {name}(const V& source){{V resized=source;resized.resize({},0.0);for(double& cell:resized)if(!std::isfinite(cell))cell=0.0;const U keep={};V raw;raw.reserve(keep.size());for(std::size_t column:keep)raw.push_back(resized[column]);{projection}const double y_mean={},clip_lo={},clip_hi={};double value=y_mean;{}if(!std::isfinite(value))value=y_mean;return clampv(value,clip_lo,clip_hi);}}\n",
        model.width,
        usize_vec(&model.keep),
        f64_lit(model.y_mean),
        f64_lit(model.clip_lo),
        f64_lit(model.clip_hi),
        member_terms.join(""),
    ));
    Ok(())
}

fn extra_expr(node: &ExtraNode) -> String {
    match node {
        ExtraNode::Leaf(value) => f64_lit(*value),
        ExtraNode::Split {
            feature,
            threshold,
            missing_left,
            left,
            right,
        } => format!(
            "((std::isfinite(x[{feature}])?x[{feature}]<={}:{} )?({}):({}))",
            f64_lit(*threshold),
            missing_left,
            extra_expr(left),
            extra_expr(right)
        ),
    }
}

fn emit_gbm(name: &str, model: &crate::heads::GbmModel, out: &mut String) -> Result<()> {
    let mut trees = Vec::new();
    for tree in &model.trees {
        let nodes = tree
            .iter()
            .map(|node| match node {
                GbmNode::Leaf(value) => format!("GNode{{true,{},0,0,0,0}}", f64_lit(*value)),
                GbmNode::Split {
                    feature,
                    thr_bin,
                    left,
                    right,
                } => format!("GNode{{false,0.0,{feature},{thr_bin},{left},{right}}}"),
            })
            .collect::<Vec<_>>()
            .join(",");
        trees.push(format!("std::vector<GNode>{{{nodes}}}"));
    }
    out.push_str(&format!("static double {name}(const V& x){{if({}==0)return {};const M edges={};const std::vector<std::vector<GNode>> trees{{{}}};double value={};for(const auto& tree:trees)value+={}*route_gbm(tree,edges,x);return value;}}\n",model.n_feat,f64_lit(model.constant_value),f64_matrix(&model.edges),trees.join(","),f64_lit(model.init),f64_lit(model.lr)));
    Ok(())
}

fn emit_hist_gbm2(
    name: &str,
    model: &crate::boost2::HistGbm2Model,
    out: &mut String,
) -> Result<()> {
    if model.bags.is_empty() {
        out.push_str(&format!(
            "static double {name}(const V&){{return {};}}\n",
            f64_lit(model.init)
        ));
        return Ok(());
    }
    let mut bag_terms = Vec::new();
    for bag in &model.bags {
        let mut tree_terms = Vec::new();
        for tree in bag {
            tree_terms.push(boost_expr(tree, 0, 0)?);
        }
        bag_terms.push(if tree_terms.is_empty() {
            "0.0".to_string()
        } else {
            format!("({})", tree_terms.join("+"))
        });
    }
    out.push_str(&format!(
        "static double {name}(const V& x){{double acc={};return {}+{}*acc/{}.0;}}\n",
        bag_terms.join("+"),
        f64_lit(model.init),
        f64_lit(model.lr),
        model.bags.len(),
    ));
    Ok(())
}

fn boost_expr(tree: &[BoostNode], index: usize, depth: usize) -> Result<String> {
    if depth > tree.len() {
        bail!("histgbm2 fitted tree contains a cycle");
    }
    match tree.get(index) {
        Some(BoostNode::Leaf(value)) => Ok(f64_lit(*value)),
        Some(BoostNode::Split {
            feature,
            threshold,
            left,
            right,
        }) => Ok(format!(
            "(x[{feature}]<={}?({}):({}))",
            f64_lit(*threshold),
            boost_expr(tree, *left, depth + 1)?,
            boost_expr(tree, *right, depth + 1)?,
        )),
        None => bail!("histgbm2 fitted tree references an invalid node"),
    }
}

fn emit_predict(
    target: &Option<FittedTarget>,
    calibration: &CalibrationModel,
    task: TaskKind,
) -> String {
    let steps = emit_prediction_steps(target, calibration, task);
    format!(
        "static double predict_one(V x,V mask){{x=transform(std::move(x),std::move(mask));double value=head_0(x);\n{steps}  return value;\n}}\n"
    )
}

fn emit_predict_transformed(
    target: &Option<FittedTarget>,
    calibration: &CalibrationModel,
    task: TaskKind,
) -> String {
    let steps = emit_prediction_steps(target, calibration, task);
    format!(
        "static double predict_transformed(const V& x){{double value=head_0(x);\n{steps}  return value;\n}}\n"
    )
}

fn emit_prediction_steps(
    target: &Option<FittedTarget>,
    calibration: &CalibrationModel,
    task: TaskKind,
) -> String {
    let calibrated = match calibration {
        CalibrationModel::None => String::new(),
        CalibrationModel::Platt { slope, intercept } => format!(
            "  {{double p=clampv(value,1e-12,1.0-1e-12);value=sigmoid({}*std::log(p/(1.0-p))+{});}}\n",
            f64_lit(*slope),
            f64_lit(*intercept)
        ),
        CalibrationModel::Isotonic { upper, values } => format!(
            "  {{const V upper={};const V levels={};double p=clampv(value,1e-12,1.0-1e-12);if(!levels.empty())value=levels[std::min(lower(upper,p),levels.size()-1)];value=clampv(value,0.0,1.0);}}\n",
            f64_vec(upper),
            f64_vec(values)
        ),
    };
    let inverse = match target {
        None => String::new(),
        Some(FittedTarget::RankGauss {
            table_values,
            table_scores,
            ..
        }) => format!(
            "  value=clampv(rank_inverse(value,{},{}),-60.0,60.0);\n",
            f64_vec(table_scores),
            f64_vec(table_values)
        ),
        Some(FittedTarget::Log1p) => {
            "  value=clampv(std::exp(clampv(value,-745.0,700.0))-1.0,-60.0,60.0);\n".to_string()
        }
        Some(FittedTarget::Winsor { .. }) => "  value=clampv(value,-60.0,60.0);\n".to_string(),
        Some(FittedTarget::ZScore {
            mu,
            sigma,
            clamp_lo,
            clamp_hi,
        }) => format!(
            "  value=clampv(value*{}+{}, {}, {});\n",
            f64_lit(*sigma),
            f64_lit(*mu),
            f64_lit(*clamp_lo),
            f64_lit(*clamp_hi)
        ),
    };
    let final_task = match task {
        TaskKind::Regression | TaskKind::Timeseries => "  value=finitev(value);\n",
        TaskKind::Binary => "  value=std::isfinite(value)?clampv(value,0.0,1.0):0.5;\n",
        TaskKind::Multiclass => "  value=finitev(value);\n",
    };
    format!("{calibrated}{inverse}{final_task}")
}

/// The C ABI of a fitted model. `jope_fitted_predict` returns one status:
///
/// | Status | Meaning |
/// |---|---|
/// | 0 | every row is predicted |
/// | 1 | `columns` is not the fitted column count |
/// | 2 | `rows * columns` overflows |
/// | 3 | a buffer length does not fit `rows` and `columns` |
/// | 4 | a buffer is null while there are rows to predict |
/// | 5 | a categorical cell is neither -1 nor a fitted vocabulary index |
/// | 6 | any other C++ exception: the closed-form math's domain error, an allocation failure |
/// | 7 | a row of a table-aware model does not follow its group's fitted history (`emit_table_abi` only) |
/// | 8 | a stage's fitted state does not fit the row it reads (revision 3): a fitted column index, or the index of a fitted table entry, at or past the end of what it indexes, or `roll` on an empty row |
///
/// Status 8 comes from `FittedWidthError`, which `fitted_index` throws before any out-of-range
/// read; it is caught ahead of status 6 in both ABIs. The host Rust runtime reports model
/// statuses directly while preserving the same fail-closed behavior.
fn emit_abi(columns: usize) -> String {
    format!(
        r#"extern "C" std::size_t jope_fitted_columns() noexcept {{ return {columns}; }}
extern "C" int jope_fitted_predict(
    const double* values,std::size_t values_len,
    const std::uint8_t* missing,std::size_t missing_len,
    std::size_t rows,std::size_t columns,
    double* output,std::size_t output_len) noexcept {{
  if(columns!={columns})return 1;
  if(rows>std::numeric_limits<std::size_t>::max()/columns)return 2;
  const std::size_t cells=rows*columns;
  if(values_len!=cells||missing_len!=cells||output_len<rows)return 3;
  if(rows!=0&&(!values||!missing||!output))return 4;
  try {{
    V x,mask;
    for(std::size_t row=0;row<rows;++row){{
      const std::size_t offset=row*columns;
      if(!encode_row(values+offset,missing+offset,x,mask))return 5;
      output[row]=predict_one(x,mask);
    }}
    return 0;
  }} catch(const FittedWidthError&) {{ return 8; }} catch(...) {{ return 6; }}
}}
"#
    )
}

/// The C ABI of a table-aware fitted model (lag, roll): the statuses of `emit_abi`, with 7.
fn emit_table_abi(columns: usize) -> String {
    format!(
        r#"extern "C" std::size_t jope_fitted_columns() noexcept {{ return {columns}; }}
extern "C" int jope_fitted_predict(
    const double* values,std::size_t values_len,
    const std::uint8_t* missing,std::size_t missing_len,
    std::size_t rows,std::size_t columns,
    double* output,std::size_t output_len) noexcept {{
  if(columns!={columns})return 1;
  if(rows>std::numeric_limits<std::size_t>::max()/columns)return 2;
  const std::size_t cells=rows*columns;
  if(values_len!=cells||missing_len!=cells||output_len<rows)return 3;
  if(rows!=0&&(!values||!missing||!output))return 4;
  try {{
    M table(rows),masks(rows);
    for(std::size_t row=0;row<rows;++row){{
      const std::size_t offset=row*columns;
      if(!encode_row(values+offset,missing+offset,table[row],masks[row]))return 5;
    }}
    if(!transform_table(table,masks,values,missing,rows,columns))return 7;
    for(std::size_t row=0;row<rows;++row)output[row]=predict_transformed(table[row]);
  }} catch(const FittedWidthError&) {{ return 8; }} catch(...) {{ return 6; }}
  return 0;
}}
"#
    )
}

/// B12: export the explicit v2 FeatureWithinRow model through its fitted route.
/// Historical BoundModel artifacts retain their frozen emission bytes.
pub fn emit_row_vector_cpp17(
    model: &crate::kernel::row_vector::Model,
) -> crate::kernel::Result<String> {
    crate::kernel::row_vector::emit::fitted_cpp17(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_prelude_marker_names_the_emitter_revision() {
        let marker = format!(
            "// jope-dime fitted-cpp-typed-table-v1 emitter-revision={FITTED_CPP_REVISION}"
        );
        assert_eq!(PRELUDE.lines().next(), Some(marker.as_str()));
    }

    #[test]
    fn only_the_legacy_predictor_paths_pad_an_empty_row() {
        let pad = "  if(x.empty()){x.assign(1,0.0);mask.assign(1,0.0);}\n";
        let ops = [
            FittedOp::Sigmoid { append: false },
            FittedOp::Sigmoid { append: true },
        ];
        // The legacy predictor's transform pads after every stage, as `ops::finish` does.
        let legacy = emit_legacy_transforms(&ops).unwrap();
        assert_eq!(legacy.matches(pad).count(), ops.len(), "{legacy}");
        let mut stage = String::new();
        emit_legacy_op_padded(&mut stage, &ops[0]).unwrap();
        assert!(stage.ends_with(pad), "{stage}");
        // The V3 pipeline path emits the same stages without padding, as its runtime does not pad.
        let v3 = emit_transforms(&ops.map(V3Op::Legacy)).unwrap();
        assert!(!v3.contains("x.assign("), "{v3}");
    }
}
