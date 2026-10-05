//! Reuse the existing symbolic planner/helpers, then append an explicit fitted
//! library entry point. Legacy recipe/CLI entry points are not this route.
use super::{LinearScaledWinner, Result, PROFILE};
use crate::transpile::{emit, EmitOptions};

#[derive(Clone, Debug)]
pub struct ScaledArtifacts {
    pub profile: &'static str,
    pub winner_identity: String,
    /// Library API: scaled_predict(rows), not the inherited legacy CSV main.
    pub rust: String,
    pub python: String,
    pub cpp17: String,
}

impl LinearScaledWinner {
    /// Emits the stored winner: no labels, fitting, or coefficient replacement
    /// at inference. Rust is compiled as a library; Python is imported; C++17
    /// calls scaled_predict. Compile C++ with -ffp-contract=off -fno-fast-math.
    pub fn emit(&self) -> Result<ScaledArtifacts> {
        let super::Program::LegacyV1(genome) = self.raw_program()?;
        let opts = EmitOptions::default();
        let artifacts =
            emit(&genome, &opts).map_err(|e| super::Error::Unsupported(e.to_string()))?;
        let raw_rust = rust_scalar_library(&artifacts.rust)?;
        let raw_python = python_scalar_library(&artifacts.python);
        let cpp = super::super::expression_recipe::emit_cpp17(&genome, &opts)?;
        let raw_cpp = cpp_scalar_library(&cpp.source);
        let [a, b] = self.coefficients().map(f64::to_bits);
        let width = self.input_width;
        let rust = format!("{}\n// {PROFILE}\npub fn scaled_predict(rows: &[Vec<f64>]) -> Result<Vec<f64>, &'static str> {{\n    if rows.iter().any(|row| row.len() != {width}) {{ return Err(\"linear-scaling row width\"); }}\n    if rows.iter().flatten().any(|v| !v.is_finite()) {{ return Err(\"linear-scaling input\"); }}\n    rows.iter().map(|row| {{\n        let value = f64::from_bits(0x{a:016x}) * expression_value(row) + f64::from_bits(0x{b:016x});\n        if value.is_finite() {{ Ok(value) }} else {{ Err(\"linear-scaled prediction\") }}\n    }}).collect()\n}}\n", raw_rust);
        let python = format!("{}\n# {PROFILE}\ndef scaled_predict(rows):\n    import struct\n    if any(len(row) != {width} for row in rows):\n        raise ValueError('linear-scaling row width')\n    if any(not math.isfinite(v) for row in rows for v in row):\n        raise ValueError('linear-scaling input')\n    a = struct.unpack('>d', bytes.fromhex('{a:016x}'))[0]\n    b = struct.unpack('>d', bytes.fromhex('{b:016x}'))[0]\n    out = []\n    for row in rows:\n        value = a * expression_value(row) + b\n        if not math.isfinite(value):\n            raise ValueError('linear-scaled prediction')\n        out.append(value)\n    return out\n", raw_python);
        let cpp17 = format!("{}\n// {PROFILE}\nstd::vector<double> scaled_predict(const std::vector<std::vector<double>>& rows) {{\n    for (const auto& row : rows) {{\n        if (row.size() != {width}) throw std::invalid_argument(\"linear-scaling row width\");\n        for (double v : row) if (!std::isfinite(v)) throw std::invalid_argument(\"linear-scaling input\");\n    }}\n    const std::uint64_t ab = 0x{a:016x}ULL, bb = 0x{b:016x}ULL;\n    double a, b; std::memcpy(&a, &ab, 8); std::memcpy(&b, &bb, 8);\n    std::vector<double> out;\n    for (const auto& row : rows) {{\n        const double value = a * expression_value(row) + b;\n        if (!std::isfinite(value)) throw std::invalid_argument(\"linear-scaled prediction\");\n        out.push_back(value);\n    }}\n    return out;\n}}\n", raw_cpp);
        for source in [&rust, &python, &cpp17] {
            let hits = crate::transpile::lint_shipped_text(source);
            if !hits.is_empty() {
                return Err(super::Error::Unsupported(format!(
                    "artifact vocabulary: {}",
                    hits.join(", ")
                )));
            }
        }
        Ok(ScaledArtifacts {
            profile: PROFILE,
            winner_identity: self
                .identity()
                .map_err(|e| super::Error::Unsupported(e.to_string()))?,
            rust,
            python,
            cpp17,
        })
    }
}

// Type the scalar binding and Step literals in these new version-2 libraries.
// Method receivers need a concrete type even when the result binding is typed.
// The frozen legacy emitter remains untouched, including its original bytes.
pub(super) fn rust_scalar_library(source: &str) -> Result<String> {
    let marker = "fn expression_value(row: &[f64]) -> f64 {";
    let (before, body) = source
        .split_once(marker)
        .ok_or_else(|| super::Error::Unsupported("linear-scaling scalar library binding".into()))?;
    let mut library = format!(
        "{before}{marker}{}",
        body.replacen("    let value = ", "    let value: f64 = ", 1)
            .replace("{ 1.0 } else { 0.0 }", "{ 1.0_f64 } else { 0.0_f64 }")
    );
    library = library
        .replace("m = m.min(v);", "m = if m < v { m } else { v };")
        .replace("m = m.max(v);", "m = if m > v { m } else { v };");
    // f64 min/max leave zero ties unspecified to the optimizer. These private
    // expression methods freeze the evaluator's observed right-operand ties.
    let methods = "\n    trait ScalarExt { fn scalar_min(self,b:f64)->f64; fn scalar_max(self,b:f64)->f64; }\n    impl ScalarExt for f64 {\n        fn scalar_min(self,b:f64)->f64 { if self.is_nan(){b}else if b.is_nan() || self<b{self}else{b} }\n        fn scalar_max(self,b:f64)->f64 { if self.is_nan(){b}else if b.is_nan() || self>b{self}else{b} }\n    }\n";
    library = library.replacen(marker, &format!("{marker}{methods}"), 1);
    library = library
        .lines()
        .map(|line| {
            if line.starts_with("    let value: f64 = ") {
                line.replace(".min(", ".scalar_min(")
                    .replace(".max(", ".scalar_max(")
            } else {
                line.to_owned()
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    for name in ["row_sum", "row_mean"] {
        library = library.replace(
            &format!("fn {name}(row: &[f64]) -> f64 {{\n    let mut total = 0.0;"),
            &format!("fn {name}(row: &[f64]) -> f64 {{\n    let mut total = -0.0;"),
        );
    }
    // A constant exponent must use the same runtime pow as the evaluator,
    // rather than an optimizer's separately rounded reciprocal rewrite.
    let mut end = library.len();
    while let Some(start) = library[..end].rfind(".powf(") {
        let argument = start + ".powf(".len();
        let mut depth = 1;
        let close = library[argument..]
            .char_indices()
            .find_map(|(i, c)| {
                if c == '(' {
                    depth += 1;
                }
                if c == ')' {
                    depth -= 1;
                }
                (depth == 0).then_some(argument + i)
            })
            .ok_or_else(|| super::Error::Unsupported("linear-scaling power binding".into()))?;
        library.insert(close, ')');
        library.insert_str(argument, "std::hint::black_box(");
        end = start;
    }
    Ok(library)
}

// Adapt only these new libraries; the legacy emitters remain byte-identical.
pub(super) fn python_scalar_library(source: &str) -> String {
    let mut library = source.to_owned();
    for name in ["_row_sum", "_row_mean"] {
        let start = format!("def {name}(row):");
        if let Some((before, body)) = library.split_once(&start) {
            library = format!(
                "{before}{start}{}",
                body.replacen("total = 0.0", "total = -0.0", 1)
            );
        }
    }
    library
        .replace(
            "return float(math.floor(v))",
            "return math.copysign(float(math.floor(v)), v) if v == 0.0 else float(math.floor(v))",
        )
        .replace(
            "return float(math.ceil(v - 0.5))",
            "return math.copysign(float(math.ceil(v - 0.5)), -1.0)",
        )
}

pub(super) fn cpp_scalar_library(source: &str) -> String {
    let helpers = "\nstatic double scalar_min(double a,double b){if(std::isnan(a))return b;if(std::isnan(b))return a;return a<b?a:b;}\nstatic double scalar_max(double a,double b){if(std::isnan(a))return b;if(std::isnan(b))return a;return a>b?a:b;}\n";
    source
        .replacen("using V =", &format!("{helpers}\nusing V ="), 1)
        .replace("std::fmin(", "scalar_min(")
        .replace("std::fmax(", "scalar_max(")
        .replace(
            "row_sum(const double* row,std::size_t n){double value=0.0;",
            "row_sum(const double* row,std::size_t n){double value=-0.0;",
        )
        .replace(
            "if(row[i]<value)value=row[i];",
            "value=scalar_min(value,row[i]);",
        )
        .replace(
            "if(row[i]>value)value=row[i];",
            "value=scalar_max(value,row[i]);",
        )
        .replace(
            "return std::pow(left,right);",
            "volatile double exponent=right;return std::pow(left,exponent);",
        )
}
