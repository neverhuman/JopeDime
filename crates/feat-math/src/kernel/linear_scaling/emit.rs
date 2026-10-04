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
        let cpp = super::super::expression_recipe::emit_cpp17(&genome, &opts)?;
        let [a, b] = self.coefficients().map(f64::to_bits);
        let width = self.input_width;
        let rust = format!("{}\n// {PROFILE}\npub fn scaled_predict(rows: &[Vec<f64>]) -> Result<Vec<f64>, &'static str> {{\n    if rows.iter().any(|row| row.len() != {width}) {{ return Err(\"linear-scaling row width\"); }}\n    if rows.iter().flatten().any(|v| !v.is_finite()) {{ return Err(\"linear-scaling input\"); }}\n    rows.iter().map(|row| {{\n        let value = f64::from_bits(0x{a:016x}) * expression_value(row) + f64::from_bits(0x{b:016x});\n        if value.is_finite() {{ Ok(value) }} else {{ Err(\"linear-scaled prediction\") }}\n    }}).collect()\n}}\n", artifacts.rust);
        let python = format!("{}\n# {PROFILE}\ndef scaled_predict(rows):\n    import struct\n    if any(len(row) != {width} for row in rows):\n        raise ValueError('linear-scaling row width')\n    if any(not math.isfinite(v) for row in rows for v in row):\n        raise ValueError('linear-scaling input')\n    a = struct.unpack('>d', bytes.fromhex('{a:016x}'))[0]\n    b = struct.unpack('>d', bytes.fromhex('{b:016x}'))[0]\n    out = []\n    for row in rows:\n        value = a * expression_value(row) + b\n        if not math.isfinite(value):\n            raise ValueError('linear-scaled prediction')\n        out.append(value)\n    return out\n", artifacts.python);
        let cpp17 = format!("{}\n// {PROFILE}\nstd::vector<double> scaled_predict(const std::vector<std::vector<double>>& rows) {{\n    for (const auto& row : rows) {{\n        if (row.size() != {width}) throw std::invalid_argument(\"linear-scaling row width\");\n        for (double v : row) if (!std::isfinite(v)) throw std::invalid_argument(\"linear-scaling input\");\n    }}\n    const std::uint64_t ab = 0x{a:016x}ULL, bb = 0x{b:016x}ULL;\n    double a, b; std::memcpy(&a, &ab, 8); std::memcpy(&b, &bb, 8);\n    std::vector<double> out;\n    for (const auto& row : rows) {{\n        const double value = a * expression_value(row) + b;\n        if (!std::isfinite(value)) throw std::invalid_argument(\"linear-scaled prediction\");\n        out.push_back(value);\n    }}\n    return out;\n}}\n", cpp.source);
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
