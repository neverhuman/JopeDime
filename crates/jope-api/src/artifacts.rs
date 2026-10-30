use crate::{decimal::DecimalU64, ids::*, Language};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Public source and immutable provenance, never an internal genome (§5.1, §8).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Algorithm {
    pub algorithm_id: AlgorithmId,
    pub summary: String,
    pub metric: String,
    pub cv_value: f64,
    pub model_artifact: Option<ArtifactId>,
    pub model_digest: String,
    pub conditioning_digest: String,
    pub bundle_digest: String,
    pub revision: DecimalU64,
    pub source: Vec<AlgorithmSource>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AlgorithmSource {
    pub language: Language,
    pub source: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ExportKind {
    Model,
    Source,
    Bundle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Refit {
    Fit,
    FitHoldout,
}

/// Client output paths are deliberately absent: the engine writes only its store.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ExportRequest {
    pub algorithm_id: AlgorithmId,
    pub kind: ExportKind,
    pub languages: Vec<Language>,
    pub require_all_languages: bool,
    pub refit: Refit,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum LanguageExport {
    Exported {
        language: Language,
        path: String,
    },
    Unsupported {
        language: Language,
        operations: Vec<String>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ParityStatus {
    Passed,
    Failed,
    Unavailable,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ExportManifest {
    pub artifact_id: ArtifactId,
    pub algorithm_id: AlgorithmId,
    pub kind: ExportKind,
    pub model_digest: String,
    pub emitter_version: String,
    pub qualified_set_id: String,
    pub refit: Refit,
    pub retains_training_data: bool,
    pub languages: Vec<LanguageExport>,
    pub parity_status: ParityStatus,
    pub parity_receipt: Option<ArtifactId>,
}

/// Registered input avoids exposing a client-side output path to the engine.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PredictRequest {
    pub model_artifact: ArtifactId,
    pub dataset: DatasetId,
}
