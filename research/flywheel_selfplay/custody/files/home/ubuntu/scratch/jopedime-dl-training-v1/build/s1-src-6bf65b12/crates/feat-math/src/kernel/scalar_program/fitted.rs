//! Immutable scalar-model export through the existing fitted C++ ridge lowerer.
use super::emit::{emit_stage, emit_v2_helpers, ScalarLanguage};
use super::ScalarModel;
use crate::kernel::{Error, Result};
use crate::FittedCppArtifact;

impl ScalarModel {
    /// A self-contained C++17 translation unit. `scalar_predict_v2` returns
    /// predictions and the native per-row fault union; it never trains.
    pub fn emit_fitted_cpp17(&self) -> Result<String> {
        self.validate()?;
        let mut source = format!(
            "// feat-math.scalar.ridge.fitted.v2; policy={}; width={}; model={}\n#include <stdexcept>\n",
            self.program().numerical_policy().name(),
            self.input_width(),
            self.identity().map_err(|e|Error::Unsupported(e.to_string()))?,
        );
        source.push_str(FittedCppArtifact::expression_helpers());
        source.push_str(&emit_v2_helpers(&[self.program()], ScalarLanguage::Cpp17)?);
        source.push_str(&emit_stage(
            self.program(),
            self.carrier(),
            self.input_width(),
            1,
            ScalarLanguage::Cpp17,
        )?);
        source.push_str(&FittedCppArtifact::scalar_ridge_function(self.ridge()));
        source.push_str(
            "static std::pair<V,std::vector<std::uint32_t>> scalar_predict_v2(const M& rows){auto result=scalar_stage1_values(rows);V predictions;predictions.reserve(rows.size());for(const V& row:result.first)predictions.push_back(scalar_head_v2(row));return {predictions,result.second};}\n",
        );
        super::recipe::admit_source(&source)?;
        Ok(source)
    }
}

impl ScalarModel {
    /// The established fitted artifact container carries strict scalar-model
    /// JSON as its numeric codec; hashes bind source and immutable fit state.
    pub fn emit_fitted_artifact(&self) -> Result<FittedCppArtifact> {
        use sha2::{Digest, Sha256};
        let source = self.emit_fitted_cpp17()?;
        let codec = self
            .to_json()
            .map_err(|e| Error::Unsupported(e.to_string()))?;
        Ok(FittedCppArtifact {
            source_sha256: format!("{:x}", Sha256::digest(source.as_bytes())),
            codec_sha256: format!("{:x}", Sha256::digest(codec.as_bytes())),
            source,
            codec,
        })
    }
}
