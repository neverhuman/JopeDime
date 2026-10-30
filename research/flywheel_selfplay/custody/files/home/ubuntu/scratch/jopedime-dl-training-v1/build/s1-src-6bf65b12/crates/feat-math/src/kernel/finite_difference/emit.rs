//! Standalone arithmetic twins used by the ordered recipes and fitted export.
use crate::math_expr::FaultSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DifferenceLanguage {
    Rust,
    Python,
    Cpp17,
}

pub fn emit_helpers(language: DifferenceLanguage) -> String {
    let source = match language {
        DifferenceLanguage::Rust => {
            let arithmetic = include_str!("../finite_difference.rs")
                .split_once("// arithmetic-begin\n")
                .unwrap()
                .1
                .split_once("// arithmetic-end")
                .unwrap()
                .0;
            format!("{RUST}\n{arithmetic}")
        }
        DifferenceLanguage::Python => PYTHON.to_owned(),
        DifferenceLanguage::Cpp17 => CPP.to_owned(),
    };
    source
        .replace("@WIDTH@", &crate::MAX_WIDTH.to_string())
        .replace("@BOUNDARY@", &super::BOUNDARY_HISTORY.to_string())
        .replace("@INPUT@", &FaultSet::NONFINITE_INPUT.bits().to_string())
        .replace(
            "@NONFINITE@",
            &FaultSet::NONFINITE_SANITIZED.bits().to_string(),
        )
        .replace(
            "@MAGNITUDE@",
            &FaultSet::MAGNITUDE_CLAMPED.bits().to_string(),
        )
}

const RUST: &str = r#"
const BOUNDARY_HISTORY:u32=@BOUNDARY@;
const NONFINITE_INPUT:u32=@INPUT@;
const NONFINITE_SANITIZED:u32=@NONFINITE@;
const MAGNITUDE_CLAMPED:u32=@MAGNITUDE@;
fn finite_difference_v2(version:u8,wire:u16,order:u8,step:f64,rows:&[Vec<f64>])->Result<(Vec<f64>,Vec<u32>),&'static str> {
    if version!=2 || wire!=59 {return Err("unsupported difference version/ID");}
    if !(1..=4).contains(&order) || !step.is_finite() || step<=0.0 {return Err("invalid difference parameters");}
    if rows.is_empty() || rows.len()>usize::from(order)+1 {return Err("invalid difference history");}
    let width=rows[0].len();
    if width==0 || width>@WIDTH@ || rows.iter().any(|row|row.len()!=width) {return Err("invalid difference width");}
    Ok(difference_values(order,step,rows))
}
"#;

const PYTHON: &str = r#"
import math,struct,sys
def finite_difference_v2(version,wire,order,step,rows):
    if version!=2 or wire!=59: raise ValueError('unsupported difference version/ID')
    if not isinstance(order,int) or not 1<=order<=4 or not math.isfinite(step) or step<=0.0: raise ValueError('invalid difference parameters')
    if not rows or len(rows)>order+1: raise ValueError('invalid difference history')
    width=len(rows[0])
    if width==0 or width>@WIDTH@ or any(len(row)!=width for row in rows): raise ValueError('invalid difference width')
    values=[];faults=[]
    for column in range(width):
        work=[0.0]*5;bits=0
        for i,row in enumerate(rows):
            x=row[column]
            if math.isfinite(x): work[i]=x
            else: bits|=@INPUT@
        if len(rows)<=order:
            values.append(0.0);faults.append(bits|@BOUNDARY@);continue
        for level in range(order):
            for i in range(order-level):
                a=work[i+1];b=work[i];difference=a-b
                work[i]=a/step-b/step if not math.isfinite(difference) and math.isfinite(a) and math.isfinite(b) else difference/step
        value=work[0]
        if not math.isfinite(value):
            values.append(0.0);faults.append(bits|@NONFINITE@)
        else:
            if abs(value)>1e12: bits|=@MAGNITUDE@
            values.append(max(-1e12,min(1e12,value)));faults.append(bits)
    return values,faults
"#;

const CPP: &str = r#"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>
static std::pair<std::vector<double>,std::vector<std::uint32_t>> finite_difference_v2(unsigned version,unsigned wire,unsigned order,double step,const std::vector<std::vector<double>>& rows) {
    if(version!=2 || wire!=59) throw std::invalid_argument("unsupported difference version/ID");
    if(order<1 || order>4 || !std::isfinite(step) || step<=0.0) throw std::invalid_argument("invalid difference parameters");
    if(rows.empty() || rows.size()>order+1) throw std::invalid_argument("invalid difference history");
    const auto width=rows[0].size();
    if(width==0 || width>@WIDTH@ || std::any_of(rows.begin(),rows.end(),[width](const auto& row){return row.size()!=width;})) throw std::invalid_argument("invalid difference width");
    std::vector<double> values;std::vector<std::uint32_t> faults;
    for(std::size_t column=0;column<width;column++) {
        double work[5]={};std::uint32_t bits=0;
        for(std::size_t i=0;i<rows.size();i++) {const auto x=rows[i][column];if(std::isfinite(x)) work[i]=x;else bits|=@INPUT@;}
        if(rows.size()<=order) {values.push_back(0.0);faults.push_back(bits|@BOUNDARY@);continue;}
        for(unsigned level=0;level<order;level++) for(unsigned i=0;i<order-level;i++) {
            const auto a=work[i+1],b=work[i];const auto difference=a-b;
            work[i]=!std::isfinite(difference) && std::isfinite(a) && std::isfinite(b)?a/step-b/step:difference/step;
        }
        const auto value=work[0];
        if(!std::isfinite(value)) {values.push_back(0.0);faults.push_back(bits|@NONFINITE@);}
        else {if(std::abs(value)>1e12) bits|=@MAGNITUDE@;values.push_back(std::clamp(value,-1e12,1e12));faults.push_back(bits);}
    }
    return {values,faults};
}
"#;
