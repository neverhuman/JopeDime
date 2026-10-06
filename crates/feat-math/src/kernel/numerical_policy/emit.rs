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
import struct
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
def numerical_expansion_v2(row,absolute,mean):
    count=float(len(row))
    divisor=count if mean else 1.0
    scale=2.0
    while scale<count*2.0:
        scale*=2.0
    scaled=[]
    for cell in row:
        value=abs(cell) if absolute else cell
        numerical_add_partial_v2(scaled,value/scale)
    ordered=[]
    for part in scaled:
        numerical_add_partial_v2(ordered,part)
    scaled=ordered
    head=scaled[-1] if scaled else 0.0
    if abs(head)>2.0*1e12*divisor/scale:
        return (math.copysign(1e12,head),1<<9)
    restored=[]
    for part in scaled:
        numerical_add_partial_v2(restored,part*scale)
    for cell in row:
        value=abs(cell) if absolute else cell
        numerical_add_partial_v2(restored,value-(value/scale)*scale)
    head=restored[-1] if restored else 0.0
    if head==0.0:
        return (0.0,0)
    sign=math.copysign(1.0,head)
    distance=[part*sign for part in restored]
    remaining=len(row) if mean else 1
    place=1.0
    while remaining!=0:
        chunk=float(remaining&0x1fff)
        numerical_add_partial_v2(distance,-(chunk*1e12)*place)
        remaining>>=13
        place*=8192.0
    if (distance[-1] if distance else 0.0)>0.0:
        return (math.copysign(1e12,head),1<<9)
    return (max(-1e12,min(1e12,head/divisor)),0)
def numerical_mean_v2(row):
    return numerical_expansion_v2(row,False,True)
def numerical_add_word_v2(bins,word,offset):
    mask=(1<<26)-1
    while word:
        carry=(word&mask)<<(offset%26)
        index=offset//26
        while carry:
            total=bins[index]+carry
            bins[index]=total&mask
            carry=total>>26
            index+=1
        word>>=26
        offset+=26
def numerical_add_square_v2(bins,value):
    bits=struct.unpack('>Q',struct.pack('>d',value))[0]&0x7fffffffffffffff
    exponent=(bits>>52)&0x7ff
    mantissa=(bits&((1<<52)-1))|((1<<52) if exponent else 0)
    low=mantissa&((1<<26)-1)
    high=mantissa>>26
    offset=2*exponent-2 if exponent else 0
    numerical_add_word_v2(bins,low*low,offset)
    numerical_add_word_v2(bins,2*low*high,offset+26)
    numerical_add_word_v2(bins,high*high,offset+52)
def numerical_square_bit_v2(bins,bit):
    return (bins[bit//26]>>(bit%26))&1
def numerical_square_sticky_v2(bins,cutoff):
    for index,word in enumerate(bins):
        start=index*26
        if start<cutoff and word&((1<<min(cutoff-start,26))-1):
            return True
    return False
def numerical_square_root_v2(bins):
    index=next((i for i in range(85,-1,-1) if bins[i]),None)
    if index is None:
        return 0.0
    top_bit=index*26+bins[index].bit_length()-1
    leading=0
    for position in range(53):
        leading<<=1
        if position<=top_bit:
            leading|=numerical_square_bit_v2(bins,top_bit-position)
    if top_bit>52:
        discarded=top_bit-52
        if numerical_square_bit_v2(bins,discarded-1) and (leading&1 or numerical_square_sticky_v2(bins,discarded-1)):
            leading+=1
    normalized=float(leading)*(2.220446049250313e-16 if top_bit%2==0 else 4.440892098500626e-16)
    scale=math.ldexp(1.0,top_bit//2-1074)
    return math.sqrt(normalized)*scale
def numerical_l2_v2(row):
    threshold=[0]*86
    numerical_add_square_v2(threshold,1e12)
    squares=[0]*86
    for value in row:
        if abs(value)>1e12:
            return (1e12,1<<9)
        numerical_add_square_v2(squares,value)
        if squares[::-1]>threshold[::-1]:
            return (1e12,1<<9)
    return (min(1e12,numerical_square_root_v2(squares)),0)
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
        return numerical_l2_v2(row)
    if not math.isfinite(numerical_neumaier_v2(row,kind=='l1')):
        return (0.0,1<<8)
    return numerical_expansion_v2(row,kind=='l1',False)
"#;

const CPP: &str = r#"
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <array>
#include <cstring>
static std::pair<double,std::uint32_t> numerical_finish_v2(double value) {
    if(!std::isfinite(value)) return {0.0,1u<<8};
    if(value>1e12) return {1e12,1u<<9};
    if(value< -1e12) return {-1e12,1u<<9};
    return {value,0};
}
[[maybe_unused]] static std::pair<double,std::uint32_t> numerical_asinh_v2(double value) {
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
static std::pair<double,std::uint32_t> numerical_expansion_v2(const std::vector<double>& row,bool absolute,bool mean) {
    const double count=static_cast<double>(row.size());
    const double divisor=mean?count:1.0;
    double scale=2.0;
    while(scale<count*2.0) scale*=2.0;
    std::vector<double> scaled;
    for(const double cell:row) numerical_add_partial_v2(scaled,(absolute?std::abs(cell):cell)/scale);
    std::vector<double> ordered;
    for(const double part:scaled) numerical_add_partial_v2(ordered,part);
    scaled.swap(ordered);
    double head=scaled.empty()?0.0:scaled.back();
    if(std::abs(head)>2.0*1e12*divisor/scale) return {std::copysign(1e12,head),1u<<9};
    std::vector<double> restored;
    for(const double part:scaled) numerical_add_partial_v2(restored,part*scale);
    for(const double cell:row) {
        const double value=absolute?std::abs(cell):cell;
        numerical_add_partial_v2(restored,value-(value/scale)*scale);
    }
    head=restored.empty()?0.0:restored.back();
    if(head==0.0) return {0.0,0};
    const double sign=std::copysign(1.0,head);
    std::vector<double> distance;
    for(const double part:restored) distance.push_back(part*sign);
    std::size_t remaining=mean?row.size():1;
    double place=1.0;
    while(remaining!=0) {
        const double chunk=static_cast<double>(remaining&0x1fff);
        numerical_add_partial_v2(distance,-(chunk*1e12)*place);
        remaining>>=13;
        place*=8192.0;
    }
    if((distance.empty()?0.0:distance.back())>0.0) return {std::copysign(1e12,head),1u<<9};
    return {std::clamp(head/divisor,-1e12,1e12),0};
}
static std::pair<double,std::uint32_t> numerical_mean_v2(const std::vector<double>& row) {
    return numerical_expansion_v2(row,false,true);
}
using NumericalSquaresV2=std::array<std::uint64_t,86>;
static void numerical_add_word_v2(NumericalSquaresV2& bins,std::uint64_t word,std::size_t offset) {
    const std::uint64_t mask=(1ull<<26)-1;
    while(word) {
        std::uint64_t carry=(word&mask)<<(offset%26);
        std::size_t index=offset/26;
        while(carry) {
            const std::uint64_t total=bins[index]+carry;
            bins[index]=total&mask;carry=total>>26;++index;
        }
        word>>=26;offset+=26;
    }
}
static void numerical_add_square_v2(NumericalSquaresV2& bins,double value) {
    std::uint64_t bits;std::memcpy(&bits,&value,sizeof(bits));bits&=0x7fffffffffffffffull;
    const std::size_t exponent=(bits>>52)&0x7ff;
    const std::uint64_t mantissa=(bits&((1ull<<52)-1))|(exponent?1ull<<52:0);
    const std::uint64_t low=mantissa&((1ull<<26)-1),high=mantissa>>26;
    const std::size_t offset=exponent?2*exponent-2:0;
    numerical_add_word_v2(bins,low*low,offset);
    numerical_add_word_v2(bins,2*low*high,offset+26);
    numerical_add_word_v2(bins,high*high,offset+52);
}
static std::uint64_t numerical_square_bit_v2(const NumericalSquaresV2& bins,std::size_t bit) {
    return (bins[bit/26]>>(bit%26))&1;
}
static bool numerical_square_sticky_v2(const NumericalSquaresV2& bins,std::size_t cutoff) {
    for(std::size_t index=0;index<bins.size();++index) {
        const std::size_t start=index*26;
        if(start<cutoff && (bins[index]&((1ull<<std::min(cutoff-start,std::size_t(26)))-1))) return true;
    }
    return false;
}
static double numerical_square_root_v2(const NumericalSquaresV2& bins) {
    int index=85;while(index>=0 && bins[index]==0) --index;
    if(index<0) return 0.0;
    std::uint64_t word=bins[index];std::size_t top_bit=index*26;
    while(word>>1) {word>>=1;++top_bit;}
    std::uint64_t leading=0;
    for(std::size_t position=0;position<53;++position) {
        leading<<=1;
        if(position<=top_bit) leading|=numerical_square_bit_v2(bins,top_bit-position);
    }
    if(top_bit>52) {
        const std::size_t discarded=top_bit-52;
        if(numerical_square_bit_v2(bins,discarded-1) && ((leading&1) || numerical_square_sticky_v2(bins,discarded-1))) ++leading;
    }
    const double normalized=static_cast<double>(leading)*(top_bit%2==0?2.220446049250313e-16:4.440892098500626e-16);
    const double scale=std::ldexp(1.0,static_cast<int>(top_bit/2)-1074);
    return std::sqrt(normalized)*scale;
}
static std::pair<double,std::uint32_t> numerical_l2_v2(const std::vector<double>& row) {
    NumericalSquaresV2 threshold{},squares{};
    numerical_add_square_v2(threshold,1e12);
    for(const double value:row) {
        if(std::abs(value)>1e12) return {1e12,1u<<9};
        numerical_add_square_v2(squares,value);
        if(std::lexicographical_compare(threshold.rbegin(),threshold.rend(),squares.rbegin(),squares.rend())) return {1e12,1u<<9};
    }
    return {std::min(1e12,numerical_square_root_v2(squares)),0};
}
[[maybe_unused]] static std::pair<double,std::uint32_t> numerical_reduce_v2(const std::string& kind,const std::vector<double>& row) {
    if(kind!="sum" && kind!="l1" && kind!="mean" && kind!="l2")
        throw std::invalid_argument("unsupported numerical reduction");
    for(const double cell:row) if(!std::isfinite(cell)) return {0.0,1u<<13};
    if(row.empty()) return {0.0,kind=="mean"?1u<<14:0u};
    if(kind=="mean") return numerical_mean_v2(row);
    if(kind=="l2") {
        return numerical_l2_v2(row);
    }
    if(!std::isfinite(numerical_neumaier_v2(row,kind=="l1"))) return {0.0,1u<<8};
    return numerical_expansion_v2(row,kind=="l1",false);
}
"#;
