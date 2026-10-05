//! Bind strict producer provenance through the existing generic scalar fit.
use super::*;
use crate::kernel::scalar_program::producer::ScalarProducerProfile;
use crate::kernel::wire_v2::{VersionError, VersionResult, VersionedModel, VersionedProgram};

impl ScalarProducerProfile {
    /// Admit the complete declared scope before running the existing fit.
    /// Prediction and the serialized reader retain this exact declaration.
    pub fn fit_training(
        &self,
        program: &ScalarExpressionProgram,
        training: &Rows,
        labels: &[f64],
        lambda: f64,
    ) -> VersionResult<ScalarModel> {
        self.validate_program(program, self.carrier(), self.input_width())
            .map_err(|error| VersionError::Malformed(format!("scalar producer: {error}")))?;
        if training.iter().any(|row| row.len() != self.input_width()) {
            return Err(VersionError::Malformed(
                "scalar producer training width".into(),
            ));
        }
        // Use the shipped generic scalar fit dispatch, with no parallel head.
        let VersionedModel::ScalarV2(model) = VersionedProgram::ScalarV2(program.clone())
            .fit_scalar(training, labels, self.carrier(), lambda)?
        else {
            return Err(VersionError::UnsupportedVersion(
                "scalar producer fit route".into(),
            ));
        };
        let mut model = *model;
        model.producer = Some(self.clone());
        model.validate()?;
        Ok(model)
    }
}

impl ScalarModel {
    /// Reuse artifact containers and actual token/hash accounting. Bound sources
    /// add only their declaration identity; unbound format-2 artifacts stay exact.
    pub fn emit_recipes(
        &self,
    ) -> Result<(crate::transpile::Emitted, crate::transpile::Cpp17Emitted)> {
        use crate::kernel::scalar_program::ScalarLanguage;
        use crate::tokens::{count_tokens, Lang};
        use sha2::{Digest, Sha256};
        self.validate()?;
        let start = std::time::Instant::now();
        let (mut recipes, mut cpp) =
            self.program()
                .emit_recipes(self.carrier(), self.input_width(), self.lambda())?;
        if let Some(producer) = &self.producer {
            recipes.python = self.emit_recipe(ScalarLanguage::Python)?;
            recipes.rust = self.emit_recipe(ScalarLanguage::Rust)?;
            cpp.source = self.emit_recipe(ScalarLanguage::Cpp17)?;
            recipes.py_tokens = count_tokens(&recipes.python, Lang::Python);
            recipes.rs_tokens = count_tokens(&recipes.rust, Lang::Rust);
            cpp.tokens = count_tokens(&cpp.source, Lang::Cpp);
            let hash = |bytes: &[u8]| format!("{:x}", Sha256::digest(bytes));
            cpp.source_sha256 = hash(cpp.source.as_bytes());
            let mut manifest: serde_json::Value = serde_json::from_str(&recipes.manifest)
                .map_err(|error| Error::Unsupported(error.to_string()))?;
            manifest["format_version"] = 3.into();
            manifest["min_reader_version"] = 3.into();
            manifest["producer"] = serde_json::json!({
                "declaration":producer.canonical_json(),"sha256":producer.sha256(),
            });
            manifest["model_sha256"] = self
                .identity()
                .map_err(|error| Error::Unsupported(error.to_string()))?
                .into();
            manifest["py_sha256"] = hash(recipes.python.as_bytes()).into();
            manifest["rs_sha256"] = hash(recipes.rust.as_bytes()).into();
            manifest["cpp17_sha256"] = cpp.source_sha256.clone().into();
            manifest["py_tokens"] = serde_json::to_value(&recipes.py_tokens)
                .map_err(|error| Error::Unsupported(error.to_string()))?;
            manifest["rs_tokens"] = serde_json::to_value(&recipes.rs_tokens)
                .map_err(|error| Error::Unsupported(error.to_string()))?;
            manifest["cpp17_tokens"] = serde_json::to_value(&cpp.tokens)
                .map_err(|error| Error::Unsupported(error.to_string()))?;
            recipes.manifest = serde_json::to_string(&manifest)
                .map_err(|error| Error::Unsupported(error.to_string()))?;
            recipes.timing.emit_ns = start.elapsed().as_nanos().max(1);
            recipes.timing.py_tokens = recipes.py_tokens.code_tokens + recipes.py_tokens.doc_tokens;
            recipes.timing.rs_tokens = recipes.rs_tokens.code_tokens + recipes.rs_tokens.doc_tokens;
            recipes.timing.tokens_per_sec = (recipes.timing.py_tokens + recipes.timing.rs_tokens)
                as f64
                / (recipes.timing.emit_ns as f64 / 1e9);
        }
        Ok((recipes, cpp))
    }
}
