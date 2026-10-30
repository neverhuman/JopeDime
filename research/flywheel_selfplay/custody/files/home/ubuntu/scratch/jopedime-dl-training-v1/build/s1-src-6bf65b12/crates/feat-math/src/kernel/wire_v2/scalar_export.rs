//! Explicit scalar export routes reuse the native recipe and fitted lowerers.
use super::{
    ExportReport, FittedBasis, Route, VersionError, VersionResult, VersionedModel, VersionedProgram,
};
use crate::kernel::scalar_program::{ScalarCarrier, ScalarLanguage};

impl VersionedProgram {
    pub fn emit_scalar_recipe(
        &self,
        language: ScalarLanguage,
        carrier: ScalarCarrier,
        width: usize,
        lambda: f64,
    ) -> VersionResult<String> {
        match self {
            Self::ScalarV2(program) => Ok(program.emit_recipe(language, carrier, width, lambda)?),
            _ => Err(VersionError::UnsupportedVersion(
                "program has no scalar recipe route".into(),
            )),
        }
    }

    /// Support is scoped to the same carrier, width and regularizer as fitting.
    /// Fitted support is declared here; it still requires a trained scalar model.
    pub fn scalar_export_report(
        &self,
        carrier: ScalarCarrier,
        width: usize,
        lambda: f64,
    ) -> ExportReport {
        let support = match self {
            Self::ScalarV2(program) => match program.plan_training(carrier, width, lambda) {
                Ok(_) => Route::Ok,
                Err(error) => Route::Unsupported(vec![error.to_string()]),
            },
            _ => Route::Unsupported(vec!["program has no scalar export route".into()]),
        };
        ExportReport {
            rust_recipe: support.clone(),
            python_recipe: support.clone(),
            cpp17_recipe: support.clone(),
            cpp17_fitted: support,
            cpp17_fitted_basis: FittedBasis::Declared,
            javascript: Route::Unsupported(vec!["scalar JavaScript route unavailable".into()]),
        }
    }
}

impl VersionedModel {
    pub fn emit_scalar_recipe(&self, language: ScalarLanguage) -> VersionResult<String> {
        match self {
            Self::ScalarV2(model) => Ok(model.emit_recipe(language)?),
            _ => Err(VersionError::UnsupportedVersion(
                "model has no scalar recipe route".into(),
            )),
        }
    }

    pub fn emit_scalar_fitted_cpp17(&self) -> VersionResult<crate::FittedCppArtifact> {
        match self {
            Self::ScalarV2(model) => Ok(model.emit_fitted_artifact()?),
            _ => Err(VersionError::UnsupportedVersion(
                "model has no scalar fitted route".into(),
            )),
        }
    }
}
