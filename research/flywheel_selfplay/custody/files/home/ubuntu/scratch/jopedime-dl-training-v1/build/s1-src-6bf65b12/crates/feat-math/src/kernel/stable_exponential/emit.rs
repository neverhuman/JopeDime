//! Standalone portable helper twins; existing recipe emitters never select them.
use crate::kernel::{Error, Result};
use crate::math_expr::FaultSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StableExponentialLanguage {
    Rust,
    Python,
    Cpp17,
}

/// Reuse the existing immutable portable arithmetic bodies at compile time.
/// No fixture files are read by a running helper or a deployed library. These
/// sources return value and fault bits; unknown IDs refuse before input access.
/// Complete training and prediction recipe integration remains a separate gate.
pub fn emit_helpers(language: StableExponentialLanguage) -> Result<String> {
    let (body, wrapper) = match language {
        StableExponentialLanguage::Rust => {
            let (_, math) = include_str!("../../portable_math.rs")
                .split_once("// ------------------------------- bit helpers -------------------------------")
                .ok_or_else(|| Error::Unsupported("portable bit helper boundary".into()))?;
            let (math, _) = math
                .split_once("// ---------------------------------- pow ------------------------------------")
                .ok_or_else(|| Error::Unsupported("portable exponential boundary".into()))?;
            (math, RUST)
        }
        StableExponentialLanguage::Python => (
            include_str!("../../../tests/portable/full_ref.py")
                .split_once("\ndef xpow(")
                .ok_or_else(|| Error::Unsupported("portable Python exponential boundary".into()))?
                .0,
            PYTHON,
        ),
        StableExponentialLanguage::Cpp17 => (
            include_str!("../../../tests/portable/full_ref.cpp")
                .split_once("\nstatic double xpow(")
                .ok_or_else(|| Error::Unsupported("portable C++ exponential boundary".into()))?
                .0,
            CPP,
        ),
    };
    let wrapper = wrapper
        .replace("@INPUT@", &FaultSet::NONFINITE_INPUT.bits().to_string())
        .replace("@ARG@", &FaultSet::EXP_ARG_CLAMPED.bits().to_string())
        .replace("@DOMAIN@", &FaultSet::LOG_DOMAIN.bits().to_string())
        .replace(
            "@NONFINITE@",
            &FaultSet::NONFINITE_SANITIZED.bits().to_string(),
        )
        .replace(
            "@MAGNITUDE@",
            &FaultSet::MAGNITUDE_CLAMPED.bits().to_string(),
        );
    Ok(format!("{body}\n{wrapper}"))
}

const RUST: &str = r#"
fn stable_exponential_v2(id:u16,x:f64)->Result<(f64,u32),&'static str> {
    if id!=21 && id!=22 {return Err("unsupported stable exponential function");}
    if !x.is_finite() {return Ok((0.0,@INPUT@));}
    let mut faults=0;
    let value=if id==21 {
        let q=x.clamp(-30.0,30.0);
        if q!=x {faults|=@ARG@;}
        expm1(q)
    } else {
        if x>=0.0 {return Ok((0.0,@DOMAIN@));}
        if x<f64::from_bits(0xbfe62e42fefa39ef) {log1p(-exp(x))} else {ln(-expm1(x))}
    };
    if !value.is_finite() {return Ok((0.0,faults|@NONFINITE@));}
    if value.abs()>1e12 {faults|=@MAGNITUDE@;}
    Ok((value.clamp(-1e12,1e12),faults))
}
"#;

const PYTHON: &str = r#"
def stable_exponential_v2(op,x):
    if op not in (21,22):
        raise ValueError('unsupported stable exponential function')
    if not math.isfinite(x):
        return (0.0,@INPUT@)
    faults=0
    if op==21:
        q=max(-30.0,min(30.0,x))
        if q!=x: faults|=@ARG@
        result=xexpm1(q)
    else:
        if x>=0.0: return (0.0,@DOMAIN@)
        result=xlog1p(-xexp(x)) if x<from_bits(0xbfe62e42fefa39ef) else xln(-xexpm1(x))
    if not math.isfinite(result): return (0.0,faults|@NONFINITE@)
    if abs(result)>1e12: faults|=@MAGNITUDE@
    return (max(-1e12,min(1e12,result)),faults)
"#;

const CPP: &str = r#"
#include <algorithm>
#include <stdexcept>
#include <utility>
static std::pair<double,std::uint32_t> stable_exponential_v2(unsigned id,double x) {
    if(id!=21 && id!=22) throw std::invalid_argument("unsupported stable exponential function");
    if(!std::isfinite(x)) return {0.0,@INPUT@};
    std::uint32_t faults=0;
    double value;
    if(id==21) {
        const double q=std::clamp(x,-30.0,30.0);
        if(q!=x) faults|=@ARG@;
        value=xexpm1(q);
    } else {
        if(x>=0.0) return {0.0,@DOMAIN@};
        value=x<from_bits(0xbfe62e42fefa39efULL)?xlog1p(-xexp(x)):xln(-xexpm1(x));
    }
    if(!std::isfinite(value)) return {0.0,faults|@NONFINITE@};
    if(std::abs(value)>1e12) faults|=@MAGNITUDE@;
    return {std::clamp(value,-1e12,1e12),faults};
}
"#;
