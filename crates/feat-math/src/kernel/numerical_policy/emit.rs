//! Standalone numerical twins, not train-predict or fitted model artifacts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumericalLanguage {
    Rust,
    Python,
    Cpp17,
}

pub fn emit_helpers(language: NumericalLanguage) -> String {
    match language {
        NumericalLanguage::Rust => include_str!("core.rs"),
        NumericalLanguage::Python => PYTHON,
        NumericalLanguage::Cpp17 => CPP,
    }
    .to_owned()
}

const PYTHON: &str = r#"
import math
def numerical_finish_v2(value):
    if not math.isfinite(value):
        return (0.0,1<<8)
    if value>1e12:
        return (1e12,1<<9)
    if value< -1e12:
        return (-1e12,1<<9)
    return (value,0)
def numerical_asinh_v2(value):
    if not math.isfinite(value):
        return (0.0,1<<13)
    a=abs(value)
    if a<float.fromhex('0x1p-28'):
        return (value,0)
    if a>float.fromhex('0x1p28'):
        result=math.log(a)+float.fromhex('0x1.62e42fefa39efp-1')
    elif a>2.0:
        result=math.log(2.0*a+1.0/(math.sqrt(a*a+1.0)+a))
    else:
        square=a*a
        result=math.log1p(a+square/(1.0+math.sqrt(1.0+square)))
    return numerical_finish_v2(math.copysign(result,value))
def numerical_neumaier_v2(row,absolute,divisor,scaling_loss):
    total=0.0
    correction=0.0
    for cell in row:
        original=abs(cell) if absolute else cell
        divided=original/divisor
        value=original-divided*divisor if scaling_loss else divided
        next_value=total+value
        if abs(total)>=abs(value):
            correction+=(total-next_value)+value
        else:
            correction+=(value-next_value)+total
        total=next_value
    return total+correction
def numerical_reduce_v2(kind,row):
    if kind not in ('sum','l1','mean','l2'):
        raise ValueError('unsupported numerical reduction')
    if any(not math.isfinite(cell) for cell in row):
        return (0.0,1<<13)
    if not row:
        return (0.0,(1<<14) if kind=='mean' else 0)
    if kind=='l2':
        scale=0.0
        squares=1.0
        for cell in row:
            a=abs(cell)
            if a==0.0:
                continue
            if scale<a:
                ratio=scale/a
                squares=1.0+squares*ratio*ratio
                scale=a
            else:
                ratio=a/scale
                squares+=ratio*ratio
        root=math.sqrt(squares)
        if scale>1e12/root:
            return (1e12,1<<9)
        return numerical_finish_v2(scale*root)
    value=numerical_neumaier_v2(row,kind=='l1',1.0,False)
    if kind=='mean':
        count=float(len(row))
        if math.isfinite(value):
            value=value/count
        else:
            scale=2.0
            while scale<count*2.0:
                scale*=2.0
            scaled=numerical_neumaier_v2(row,False,scale,False)
            if abs(scaled)>float.fromhex('0x1.fffffffffffffp+1023')/scale:
                return (math.copysign(1e12,scaled),1<<9)
            loss=numerical_neumaier_v2(row,False,scale,True)
            value=(scaled*scale+loss)/count
    return numerical_finish_v2(value)
"#;

const CPP: &str = r#"
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
static std::pair<double,std::uint32_t> numerical_finish_v2(double value) {
    if(!std::isfinite(value)) return {0.0,1u<<8};
    if(value>1e12) return {1e12,1u<<9};
    if(value< -1e12) return {-1e12,1u<<9};
    return {value,0};
}
static std::pair<double,std::uint32_t> numerical_asinh_v2(double value) {
    if(!std::isfinite(value)) return {0.0,1u<<13};
    const double a=std::abs(value);
    if(a<0x1p-28) return {value,0};
    double result;
    if(a>0x1p28) result=std::log(a)+0x1.62e42fefa39efp-1;
    else if(a>2.0) result=std::log(2.0*a+1.0/(std::sqrt(a*a+1.0)+a));
    else {
        const double square=a*a;
        result=std::log1p(a+square/(1.0+std::sqrt(1.0+square)));
    }
    return numerical_finish_v2(std::copysign(result,value));
}
static double numerical_neumaier_v2(const std::vector<double>& row,bool absolute,double divisor,bool scaling_loss) {
    double total=0.0,correction=0.0;
    for(const double cell:row) {
        const double original=absolute?std::abs(cell):cell;
        const double divided=original/divisor;
        const double value=scaling_loss?original-divided*divisor:divided;
        const double next=total+value;
        correction+=std::abs(total)>=std::abs(value)?(total-next)+value:(value-next)+total;
        total=next;
    }
    return total+correction;
}
static std::pair<double,std::uint32_t> numerical_reduce_v2(const std::string& kind,const std::vector<double>& row) {
    if(kind!="sum" && kind!="l1" && kind!="mean" && kind!="l2")
        throw std::invalid_argument("unsupported numerical reduction");
    for(const double cell:row) if(!std::isfinite(cell)) return {0.0,1u<<13};
    if(row.empty()) return {0.0,kind=="mean"?1u<<14:0u};
    if(kind=="l2") {
        double scale=0.0,squares=1.0;
        for(const double cell:row) {
            const double a=std::abs(cell);
            if(a==0.0) continue;
            if(scale<a) {
                const double ratio=scale/a;
                squares=1.0+squares*ratio*ratio;
                scale=a;
            } else {
                const double ratio=a/scale;
                squares+=ratio*ratio;
            }
        }
        const double root=std::sqrt(squares);
        if(scale>1e12/root) return {1e12,1u<<9};
        return numerical_finish_v2(scale*root);
    }
    double value=numerical_neumaier_v2(row,kind=="l1",1.0,false);
    if(kind=="mean") {
        const double count=static_cast<double>(row.size());
        if(std::isfinite(value)) value/=count;
        else {
            double scale=2.0;
            while(scale<count*2.0) scale*=2.0;
            const double scaled=numerical_neumaier_v2(row,false,scale,false);
            if(std::abs(scaled)>std::numeric_limits<double>::max()/scale)
                return {std::copysign(1e12,scaled),1u<<9};
            const double loss=numerical_neumaier_v2(row,false,scale,true);
            value=(scaled*scale+loss)/count;
        }
    }
    return numerical_finish_v2(value);
}
"#;
