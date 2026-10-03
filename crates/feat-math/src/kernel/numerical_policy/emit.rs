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
def numerical_neumaier_v2(row,absolute):
    total=0.0
    correction=0.0
    for cell in row:
        value=abs(cell) if absolute else cell
        next_value=total+value
        if abs(total)>=abs(value):
            correction+=(total-next_value)+value
        else:
            correction+=(value-next_value)+total
        total=next_value
    return total+correction
def numerical_add_partial_v2(partials,value):
    kept=0
    for part in partials:
        if abs(value)<abs(part):
            value,part=part,value
        high=value+part
        low=part-(high-value)
        if low!=0.0:
            partials[kept]=low
            kept+=1
        value=high
    del partials[kept:]
    if value!=0.0:
        partials.append(value)
def numerical_mean_v2(row):
    count=float(len(row))
    scale=2.0
    while scale<count*2.0:
        scale*=2.0
    scaled=[]
    for value in row:
        numerical_add_partial_v2(scaled,value/scale)
    head=scaled[-1] if scaled else 0.0
    if abs(head)>2.0*1e12*count/scale:
        return (math.copysign(1e12,head),1<<9)
    restored=[]
    for part in scaled:
        numerical_add_partial_v2(restored,part*scale)
    for value in row:
        numerical_add_partial_v2(restored,value-(value/scale)*scale)
    head=restored[-1] if restored else 0.0
    if head==0.0:
        return (0.0,0)
    sign=math.copysign(1.0,head)
    distance=[part*sign for part in restored]
    remaining=len(row)
    place=1.0
    while remaining!=0:
        chunk=float(remaining&0x1fff)
        numerical_add_partial_v2(distance,-(chunk*1e12)*place)
        remaining>>=13
        place*=8192.0
    if (distance[-1] if distance else 0.0)>0.0:
        return (math.copysign(1e12,head),1<<9)
    return (max(-1e12,min(1e12,head/count)),0)
def numerical_reduce_v2(kind,row):
    if kind not in ('sum','l1','mean','l2'):
        raise ValueError('unsupported numerical reduction')
    if any(not math.isfinite(cell) for cell in row):
        return (0.0,1<<13)
    if not row:
        return (0.0,(1<<14) if kind=='mean' else 0)
    if kind=='mean':
        return numerical_mean_v2(row)
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
    return numerical_finish_v2(numerical_neumaier_v2(row,kind=='l1'))
"#;

const CPP: &str = r#"
#include <cmath>
#include <cstdint>
#include <algorithm>
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
static double numerical_neumaier_v2(const std::vector<double>& row,bool absolute) {
    double total=0.0,correction=0.0;
    for(const double cell:row) {
        const double value=absolute?std::abs(cell):cell;
        const double next=total+value;
        correction+=std::abs(total)>=std::abs(value)?(total-next)+value:(value-next)+total;
        total=next;
    }
    return total+correction;
}
static void numerical_add_partial_v2(std::vector<double>& partials,double value) {
    std::size_t kept=0;
    for(std::size_t index=0;index<partials.size();++index) {
        double part=partials[index];
        if(std::abs(value)<std::abs(part)) std::swap(value,part);
        const double high=value+part;
        const double low=part-(high-value);
        if(low!=0.0) partials[kept++]=low;
        value=high;
    }
    partials.resize(kept);
    if(value!=0.0) partials.push_back(value);
}
static std::pair<double,std::uint32_t> numerical_mean_v2(const std::vector<double>& row) {
    const double count=static_cast<double>(row.size());
    double scale=2.0;
    while(scale<count*2.0) scale*=2.0;
    std::vector<double> scaled;
    for(const double value:row) numerical_add_partial_v2(scaled,value/scale);
    double head=scaled.empty()?0.0:scaled.back();
    if(std::abs(head)>2.0*1e12*count/scale) return {std::copysign(1e12,head),1u<<9};
    std::vector<double> restored;
    for(const double part:scaled) numerical_add_partial_v2(restored,part*scale);
    for(const double value:row) numerical_add_partial_v2(restored,value-(value/scale)*scale);
    head=restored.empty()?0.0:restored.back();
    if(head==0.0) return {0.0,0};
    const double sign=std::copysign(1.0,head);
    std::vector<double> distance;
    for(const double part:restored) distance.push_back(part*sign);
    std::size_t remaining=row.size();
    double place=1.0;
    while(remaining!=0) {
        const double chunk=static_cast<double>(remaining&0x1fff);
        numerical_add_partial_v2(distance,-(chunk*1e12)*place);
        remaining>>=13;
        place*=8192.0;
    }
    if((distance.empty()?0.0:distance.back())>0.0) return {std::copysign(1e12,head),1u<<9};
    return {std::clamp(head/count,-1e12,1e12),0};
}
static std::pair<double,std::uint32_t> numerical_reduce_v2(const std::string& kind,const std::vector<double>& row) {
    if(kind!="sum" && kind!="l1" && kind!="mean" && kind!="l2")
        throw std::invalid_argument("unsupported numerical reduction");
    for(const double cell:row) if(!std::isfinite(cell)) return {0.0,1u<<13};
    if(row.empty()) return {0.0,kind=="mean"?1u<<14:0u};
    if(kind=="mean") return numerical_mean_v2(row);
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
    return numerical_finish_v2(numerical_neumaier_v2(row,kind=="l1"));
}
"#;
