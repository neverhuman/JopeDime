//! Helper twins for the new numerical feature; no existing emitter selects them.
use super::{A, B, C, D};
use crate::math_expr::FaultSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Language {
    Rust,
    Python,
    Cpp17,
}

/// Standalone helper source, not a training recipe or a fitted model artifact.
/// Every helper returns both value and fault bits. Unknown function IDs refuse.
/// Same-host PlatformLibm parity is tested with contraction disabled in C++.
pub fn emit_helpers(language: Language) -> String {
    let template = match language {
        Language::Rust => RUST,
        Language::Python => PYTHON,
        Language::Cpp17 => CPP,
    };
    let mut source = template.to_owned();
    for (marker, values) in [("@A@", A), ("@B@", B), ("@C@", C), ("@D@", D)] {
        let numbers = values
            .iter()
            .map(|value| format!("{value:?}"))
            .collect::<Vec<_>>()
            .join(",");
        source = source.replace(marker, &numbers);
    }
    source
        .replace("@DOMAIN@", &FaultSet::NONFINITE_INPUT.bits().to_string())
        .replace("@CLAMP@", &FaultSet::PROBABILITY_CLAMPED.bits().to_string())
}

const RUST: &str = r#"
fn inverse_link_poly_v2(a: &[f64], x: f64) -> f64 {
    a[1..].iter().fold(a[0], |v,c| v*x+c)
}
fn inverse_link_v2(id: u16, p: f64) -> Result<(f64,u32), &'static str> {
    if id != 19 && id != 20 {return Err("unsupported inverse-link function");}
    if !p.is_finite() {return Ok((0.0,@DOMAIN@));}
    let q=p.clamp(f64::from_bits(0x3ca0000000000000),f64::from_bits(0x3fefffffffffffff));
    let faults=if q==p {0} else {@CLAMP@};
    let value=if id==20 {
        if (0.25..=0.75).contains(&q) {
            let d=(q-0.5)*2.0;
            d.ln_1p()-(-d).ln_1p()
        } else {q.ln()-(-q).ln_1p()}
    } else {
        let a: &[f64]=&[@A@]; let b: &[f64]=&[@B@];
        let c: &[f64]=&[@C@]; let d: &[f64]=&[@D@];
        if q<0.02425 {
            let x=(-2.0*q.ln()).sqrt();
            inverse_link_poly_v2(c,x)/inverse_link_poly_v2(d,x)
        } else if q<=1.0-0.02425 {
            let x=q-0.5; let r=x*x;
            inverse_link_poly_v2(a,r)*x/inverse_link_poly_v2(b,r)
        } else {
            let x=(-2.0*(1.0-q).ln()).sqrt();
            -inverse_link_poly_v2(c,x)/inverse_link_poly_v2(d,x)
        }
    };
    Ok((value,faults))
}
"#;

const PYTHON: &str = r#"
import math
def inverse_link_poly_v2(a,x):
    value=a[0]
    for coefficient in a[1:]:
        value=value*x+coefficient
    return value
def inverse_link_v2(op,p):
    if op not in (19,20):
        raise ValueError('unsupported inverse-link function')
    if not math.isfinite(p):
        return (0.0,@DOMAIN@)
    q=max(float.fromhex('0x1p-53'),min(float.fromhex('0x1.fffffffffffffp-1'),p))
    faults=0 if q==p else @CLAMP@
    if op==20:
        if 0.25<=q<=0.75:
            d=(q-0.5)*2.0
            return (math.log1p(d)-math.log1p(-d),faults)
        return (math.log(q)-math.log1p(-q),faults)
    a=[@A@]; b=[@B@]; c=[@C@]; d=[@D@]
    if q<0.02425:
        x=math.sqrt(-2.0*math.log(q))
        value=inverse_link_poly_v2(c,x)/inverse_link_poly_v2(d,x)
    elif q<=1.0-0.02425:
        x=q-0.5; r=x*x
        value=inverse_link_poly_v2(a,r)*x/inverse_link_poly_v2(b,r)
    else:
        x=math.sqrt(-2.0*math.log(1.0-q))
        value=-inverse_link_poly_v2(c,x)/inverse_link_poly_v2(d,x)
    return (value,faults)
"#;

const CPP: &str = r#"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <utility>
static double inverse_link_poly_v2(std::initializer_list<double> a,double x) {
    auto it=a.begin(); double value=*it++;
    for(;it!=a.end();++it) value=value*x+*it;
    return value;
}
static std::pair<double,std::uint32_t> inverse_link_v2(unsigned id,double p) {
    if(id!=19 && id!=20) throw std::invalid_argument("unsupported inverse-link function");
    if(!std::isfinite(p)) return {0.0,@DOMAIN@};
    const double q=std::clamp(p,0x1p-53,0x1.fffffffffffffp-1);
    const std::uint32_t faults=q==p?0:@CLAMP@;
    if(id==20) {
        if(0.25<=q && q<=0.75) {
            const double d=(q-0.5)*2.0;
            return {std::log1p(d)-std::log1p(-d),faults};
        }
        return {std::log(q)-std::log1p(-q),faults};
    }
    const auto a={@A@},b={@B@},c={@C@},d={@D@};
    double value;
    if(q<0.02425) {
        const double x=std::sqrt(-2.0*std::log(q));
        value=inverse_link_poly_v2(c,x)/inverse_link_poly_v2(d,x);
    } else if(q<=1.0-0.02425) {
        const double x=q-0.5,r=x*x;
        value=inverse_link_poly_v2(a,r)*x/inverse_link_poly_v2(b,r);
    } else {
        const double x=std::sqrt(-2.0*std::log(1.0-q));
        value=-inverse_link_poly_v2(c,x)/inverse_link_poly_v2(d,x);
    }
    return {value,faults};
}
"#;
