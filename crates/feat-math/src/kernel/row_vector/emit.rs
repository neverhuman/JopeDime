//! Feature-specific additions to the existing recipe emitters. FIR constants
//! are emitted from frozen u64 bits; multiply then add is never contracted.
use super::{FittedStage, Model};
use crate::kernel::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowRecipeLanguage {
    Rust,
    Python,
    Cpp17,
}

pub const RUST_HELPER: &str = r#"
fn row_vector_v2_input(x: &[Vec<f64>], width: usize, append: bool) -> Vec<Vec<f64>> {
    assert!(!append); assert!(x.iter().all(|r|r.len()==width),"row-vector input width");
    x.iter().map(|row|row.iter().map(|&v|if !v.is_finite(){0.0}else{v.clamp(-1e12,1e12)}).collect()).collect()
}
fn row_vector_v2_values(x: &[Vec<f64>], starts: &[usize], weights: &[Vec<u64>], boundary: &[u32]) -> (Vec<Vec<f64>>, Vec<Vec<u32>>) {
    let mut rows=Vec::with_capacity(x.len()); let mut faults=Vec::with_capacity(x.len());
    for row in x {
        assert_eq!(row.len(),starts.len(),"row-vector fitted width");
        let mut values=Vec::with_capacity(starts.len()); let mut bits=Vec::with_capacity(starts.len());
        for j in 0..starts.len() {
            let mut value=0.0; let mut fault=boundary[j];
            for k in 0..weights[j].len() {
                let raw=row[starts[j]+k];
                let v=if !raw.is_finite() {fault|=8192;0.0} else if raw.abs()>1e12 {fault|=512;raw.clamp(-1e12,1e12)} else {raw};
                let term=f64::from_bits(weights[j][k])*v; value+=term;
            }
            if !value.is_finite() {fault|=256;value=0.0;} else if value.abs()>1e12 {fault|=512;value=value.clamp(-1e12,1e12);}
            values.push(value);bits.push(fault);
        }
        rows.push(values);faults.push(bits);
    }
    (rows,faults)
}
fn row_vector_v2_apply(x: &[Vec<f64>], starts: &[usize], weights: &[Vec<u64>], boundary: &[u32], append: bool) -> Vec<Vec<f64>> {
    assert!(!append,"v2 row stages replace features");row_vector_v2_values(x,starts,weights,boundary).0
}
"#;

pub const PYTHON_HELPER: &str = r#"
import struct

def row_vector_v2_input(x, width, append):
    if append or any(len(row)!=width for row in x): raise ValueError("row-vector input width")
    return [[max(-1e12,min(1e12,v)) if math.isfinite(v) else 0.0 for v in row] for row in x]

def row_vector_v2_values(x, starts, weights, boundary):
    rows, faults = [], []
    for row in x:
        if len(row) != len(starts): raise ValueError("row-vector fitted width")
        values, bits = [], []
        for j in range(len(starts)):
            value, fault = 0.0, boundary[j]
            for k in range(len(weights[j])):
                raw = row[starts[j]+k]
                if not math.isfinite(raw): fault |= 8192; v = 0.0
                elif abs(raw)>1e12: fault |= 512; v = max(-1e12,min(1e12,raw))
                else: v = raw
                coefficient = struct.unpack('<d', struct.pack('<Q',weights[j][k]))[0]
                term = coefficient*v
                value += term
            if not math.isfinite(value): fault |= 256; value = 0.0
            elif abs(value)>1e12: fault |= 512; value = max(-1e12,min(1e12,value))
            values.append(value); bits.append(fault)
        rows.append(values); faults.append(bits)
    return rows, faults

def row_vector_v2_apply(x, starts, weights, boundary, append):
    if append: raise ValueError("v2 row stages replace features")
    return row_vector_v2_values(x,starts,weights,boundary)[0]
"#;

pub const CPP_HELPER: &str = r#"
static std::vector<std::vector<double>> row_vector_v2_input(const std::vector<std::vector<double>>& x,std::size_t width,bool append){
    if(append)throw std::invalid_argument("v2 row stage append");for(const auto& row:x)if(row.size()!=width)throw std::invalid_argument("row-vector input width");
    auto out=x;for(auto& row:out)for(auto& v:row)v=std::isfinite(v)?std::max(-1e12,std::min(1e12,v)):0.0;return out;
}
static double row_vector_v2_bits(std::uint64_t bits){double value;std::memcpy(&value,&bits,8);return value;}
static std::pair<std::vector<std::vector<double>>,std::vector<std::vector<std::uint32_t>>> row_vector_v2_values(const std::vector<std::vector<double>>& x,const std::vector<std::size_t>& starts,const std::vector<std::vector<std::uint64_t>>& weights,const std::vector<std::uint32_t>& boundary){
    std::vector<std::vector<double>> rows;std::vector<std::vector<std::uint32_t>> faults;
    for(const auto& row:x){
        if(row.size()!=starts.size())throw std::invalid_argument("row-vector fitted width");
        std::vector<double> values;std::vector<std::uint32_t> bits;
        for(std::size_t j=0;j<starts.size();++j){
            double value=0.0;auto fault=boundary[j];
            for(std::size_t k=0;k<weights[j].size();++k){
                const double raw=row[starts[j]+k];double v;
                if(!std::isfinite(raw)){fault|=8192;v=0.0;}
                else if(std::abs(raw)>1e12){fault|=512;v=std::max(-1e12,std::min(1e12,raw));}else{v=raw;}
                const double term=row_vector_v2_bits(weights[j][k])*v;value+=term;
            }
            if(!std::isfinite(value)){fault|=256;value=0.0;}else if(std::abs(value)>1e12){fault|=512;value=std::max(-1e12,std::min(1e12,value));}
            values.push_back(value);bits.push_back(fault);
        }
        rows.push_back(values);faults.push_back(bits);
    }
    return {rows,faults};
}
static std::vector<std::vector<double>> row_vector_v2_apply(const std::vector<std::vector<double>>& x,const std::vector<std::size_t>& starts,const std::vector<std::vector<std::uint64_t>>& weights,const std::vector<std::uint32_t>& boundary,bool append){
    if(append)throw std::invalid_argument("v2 row stages replace features");return row_vector_v2_values(x,starts,weights,boundary).first;
}
"#;

pub fn helper(language: RowRecipeLanguage) -> &'static str {
    match language {
        RowRecipeLanguage::Rust => RUST_HELPER,
        RowRecipeLanguage::Python => PYTHON_HELPER,
        RowRecipeLanguage::Cpp17 => CPP_HELPER,
    }
}

pub fn arguments(stage: &FittedStage, language: RowRecipeLanguage) -> String {
    let FittedStage::Savgol(state) = stage;
    let list = |items: Vec<String>, nested: bool| -> String {
        let items = items.join(",");
        match language {
            RowRecipeLanguage::Rust if nested => format!("vec![{items}]"),
            RowRecipeLanguage::Rust | RowRecipeLanguage::Python => format!("[{items}]"),
            RowRecipeLanguage::Cpp17 => format!("{{{items}}}"),
        }
    };
    let starts = list(
        state
            .stencils()
            .iter()
            .map(|s| s.start.to_string())
            .collect(),
        false,
    );
    let weights = list(
        state
            .stencils()
            .iter()
            .map(|s| {
                list(
                    s.bits
                        .iter()
                        .map(|b| {
                            if language == RowRecipeLanguage::Cpp17 {
                                format!("{b}ULL")
                            } else {
                                b.to_string()
                            }
                        })
                        .collect(),
                    true,
                )
            })
            .collect(),
        false,
    );
    let boundary = list(
        state
            .stencils()
            .iter()
            .map(|s| {
                if usize::from(state.config().derivative()) >= s.bits.len() {
                    crate::kernel::savgol::BOUNDARY_DEGREE.to_string()
                } else {
                    "0".into()
                }
            })
            .collect(),
        false,
    );
    let borrow = if language == RowRecipeLanguage::Rust {
        "&"
    } else {
        ""
    };
    format!("{borrow}{starts},{borrow}{weights},{borrow}{boundary}")
}

pub fn apply_call(stage: &FittedStage, language: RowRecipeLanguage) -> String {
    let x = if language == RowRecipeLanguage::Rust {
        "&x"
    } else {
        "x"
    };
    format!(
        "row_vector_v2_apply({x},{},{{append}})",
        arguments(stage, language)
    )
}

pub fn fitted_cpp17(model: &Model) -> Result<String> {
    model.validate()?;
    let mut source = String::from("// feat-math.feature-row.fitted.v2\n#include <algorithm>\n#include <cmath>\n#include <cstring>\n#include <cstdint>\n#include <stdexcept>\n#include <utility>\n#include <vector>\n");
    source.push_str(CPP_HELPER);
    source.push_str(
        "static std::vector<double> row_vector_predict(std::vector<std::vector<double>> x){\n",
    );
    source.push_str(&format!("for(const auto& row:x)if(row.size()!={})throw std::invalid_argument(\"row-vector input width\");\n",model.program().input_width()));
    for stage in model.fitted_stages() {
        source.push_str(&format!(
            "x={};\n",
            apply_call(stage, RowRecipeLanguage::Cpp17).replace("{append}", "false")
        ));
    }
    let floats = |v: &[f64]| {
        format!(
            "{{{}}}",
            v.iter()
                .map(|v| format!("row_vector_v2_bits({}ULL)", v.to_bits()))
                .collect::<Vec<_>>()
                .join(",")
        )
    };
    let ridge = model.ridge();
    source.push_str(&format!(
        "const std::vector<double> means={},scales={},w={};\n",
        floats(&ridge.means),
        floats(&ridge.scales),
        floats(&ridge.w)
    ));
    source.push_str("std::vector<double> predictions;for(const auto& row:x){double value=w[0];for(std::size_t j=0;j<means.size();++j){double raw=row[j];if(!std::isfinite(raw))raw=0.0;raw=std::max(-1e12,std::min(1e12,raw));double score=(raw-means[j])/scales[j];score=std::max(-20.0,std::min(20.0,score));const double term=w[j+1]*score;value+=term;}predictions.push_back(std::max(-60.0,std::min(60.0,value)));}return predictions;}\n");
    Ok(source)
}
