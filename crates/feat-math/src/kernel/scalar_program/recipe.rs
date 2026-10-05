//! Scalar v2 stages use the existing recipe plan, head fitter and emitters.
use super::emit::ScalarLanguage;
use super::{ScalarCarrier, ScalarExpressionProgram, ScalarModel};
use crate::kernel::{Error, Result};
use crate::spec::{Plan, PlannedHead, PlannedOp, PlannedStage};
use crate::transpile::EmitOptions;

impl ScalarExpressionProgram {
    pub fn plan_training(&self, carrier: ScalarCarrier, width: usize, lambda: f64) -> Result<Plan> {
        super::model::validate_config(self, carrier, width, lambda)?;
        Ok(Plan::Pipeline {
            target: None,
            stages: vec![PlannedStage {
                op: PlannedOp::ScalarV2 {
                    program: self.clone(),
                    carrier,
                    width,
                },
                append: false,
            }],
            head: PlannedHead::Ridge {
                lambda_frac: lambda,
            },
        })
    }

    /// A complete standalone recipe, fitting exclusively on supplied training rows.
    /// The typed scalar route fixes the existing PlatformLibm old-operation contract.
    pub fn emit_recipe(
        &self,
        language: ScalarLanguage,
        carrier: ScalarCarrier,
        width: usize,
        lambda: f64,
    ) -> Result<String> {
        let plan = self.plan_training(carrier, width, lambda)?;
        let opts = EmitOptions::default();
        let source = match language {
            ScalarLanguage::Rust => crate::emit_rs::emit_rust(&plan, &opts),
            ScalarLanguage::Python => crate::emit_py::emit_python(&plan, &opts),
            ScalarLanguage::Cpp17 => opts
                .cpp_expression_recipe(&plan)
                .map_err(|errors| Error::Unsupported(errors.join("; ")))?,
        };
        let marker = if language == ScalarLanguage::Python {
            "#"
        } else {
            "//"
        };
        let source = format!("{marker} feat-math.scalar.ridge.recipe.v2; policy={}; grammar={}; features={}; profile={}\n{source}",
            self.numerical_policy().name(),super::model::grammar_sha256(),self.required_features().map_err(|e| Error::Unsupported(e.to_string()))?,super::MODEL_PROFILE);
        admit_source(&source)?;
        Ok(source)
    }
}

impl ScalarExpressionProgram {
    pub fn emit_recipe_counted(
        &self,
        language: ScalarLanguage,
        carrier: ScalarCarrier,
        width: usize,
        lambda: f64,
    ) -> Result<(String, crate::transpile::TokenCount)> {
        let source = self.emit_recipe(language, carrier, width, lambda)?;
        let lang = match language {
            ScalarLanguage::Rust => crate::tokens::Lang::Rust,
            ScalarLanguage::Python => crate::tokens::Lang::Python,
            ScalarLanguage::Cpp17 => crate::tokens::Lang::Cpp,
        };
        let count = crate::tokens::count_tokens(&source, lang);
        Ok((source, count))
    }
}

pub(super) fn admit_source(source: &str) -> Result<()> {
    if !crate::transpile::lint_shipped_text(source).is_empty() {
        return Err(Error::Unsupported("scalar artifact vocabulary".into()));
    }
    Ok(())
}

impl ScalarModel {
    pub fn emit_recipe(&self, language: ScalarLanguage) -> Result<String> {
        self.validate()?;
        self.program()
            .emit_recipe(language, self.carrier(), self.input_width(), self.lambda())
    }
}

impl ScalarExpressionProgram {
    /// Reuse the existing recipe artifact containers and token accounting. The
    /// sidecar binds all three sources to the same program/policy/fit contract.
    pub fn emit_recipes(
        &self,
        carrier: ScalarCarrier,
        width: usize,
        lambda: f64,
    ) -> Result<(crate::transpile::Emitted, crate::transpile::Cpp17Emitted)> {
        use crate::transpile::{Cpp17Emitted, EmitTiming, Emitted};
        use sha2::{Digest, Sha256};
        let hash = |bytes: &[u8]| format!("{:x}", Sha256::digest(bytes));
        let start = std::time::Instant::now();
        let (python, py_tokens) =
            self.emit_recipe_counted(ScalarLanguage::Python, carrier, width, lambda)?;
        let (rust, rs_tokens) =
            self.emit_recipe_counted(ScalarLanguage::Rust, carrier, width, lambda)?;
        let emit_ns = start.elapsed().as_nanos().max(1);
        let py_total = py_tokens.code_tokens + py_tokens.doc_tokens;
        let rs_total = rs_tokens.code_tokens + rs_tokens.doc_tokens;
        let (source, tokens) =
            self.emit_recipe_counted(ScalarLanguage::Cpp17, carrier, width, lambda)?;
        let source_sha256 = hash(source.as_bytes());
        let program = self
            .to_bytes()
            .map_err(|e| Error::Unsupported(e.to_string()))?;
        let manifest=serde_json::to_string(&serde_json::json!({
            "schema":"feat-math.scalar-recipe.v2", "format_version":2,"min_reader_version":2,
            "program":program,"program_sha256":hash(&program),
            "required_features":self.required_features().map_err(|e| Error::Unsupported(e.to_string()))?,"numerical_policy":self.numerical_policy().name(),
            "profile":super::MODEL_PROFILE,"grammar_sha256":super::model::grammar_sha256(),
            "emitter_feature_identity":"feat-math.scalar.ridge.recipe.v2","output_schema":["prediction:f64"],
            "carrier":carrier,"input_width":width,"lambda_bits":lambda.to_bits(),
            "py_sha256":hash(python.as_bytes()),"rs_sha256":hash(rust.as_bytes()),"cpp17_sha256":source_sha256,
            "py_tokens":py_tokens,"rs_tokens":rs_tokens,"cpp17_tokens":tokens,
        })).map_err(|e| Error::Unsupported(e.to_string()))?;
        Ok((
            Emitted {
                python,
                rust,
                manifest,
                py_tokens,
                rs_tokens,
                timing: EmitTiming {
                    emit_ns,
                    py_tokens: py_total,
                    rs_tokens: rs_total,
                    tokens_per_sec: (py_total + rs_total) as f64 / (emit_ns as f64 / 1e9),
                },
            },
            Cpp17Emitted {
                source,
                source_sha256,
                tokens,
                math_policy: "scalar-v2/platform-libm",
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math_expr::Var;
    use crate::transpile::MathKernel;
    #[test]
    fn scalar_old_operation_contract_refuses_portable_recipes() {
        let program =
            ScalarExpressionProgram::new(super::super::ScalarNode::Var(Var::Col(0))).unwrap();
        let plan = program.plan_training(ScalarCarrier::Row, 1, 0.02).unwrap();
        let options = EmitOptions {
            math_kernel: MathKernel::Portable,
            ..EmitOptions::default()
        };
        assert!(options.cpp_expression_recipe(&plan).is_err());
        assert!(std::panic::catch_unwind(|| crate::emit_rs::emit_rust(&plan, &options)).is_err());
        assert!(std::panic::catch_unwind(|| crate::emit_py::emit_python(&plan, &options)).is_err());
    }
}
