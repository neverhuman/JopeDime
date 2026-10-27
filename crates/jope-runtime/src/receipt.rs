//! The run receipt, `jope.appliance-run/v1` (SPECIFICATION.md §13.3): what ran, on what, with
//! which inputs, budget and outputs. Dataset values and targets never appear in it.
use serde::Serialize;

pub const SCHEMA: &str = "jope.appliance-run/v1";

#[derive(Clone, Debug, Serialize)]
pub struct Receipt {
    pub schema: &'static str,
    pub command: String,
    pub product_version: &'static str,
    pub runtime_abi: &'static str,
    pub source_sha40: String,
    pub package: Option<PackageRecord>,
    pub device: crate::backend::DeviceChoice,
    pub task: Option<TaskRecord>,
    pub method: Option<MethodRecord>,
    pub inputs: Vec<FileRecord>,
    pub outputs: Vec<FileRecord>,
    pub ignored_columns: Vec<String>,
    pub budget: crate::budget::BudgetRecord,
    pub timings: Timings,
    pub selection: Option<crate::automl::Selection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub landmark_encoding: Option<FileRecord>,
    pub outcome: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorRecord>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PackageRecord {
    pub profile: String,
    pub scope: String,
    pub manifest_sha256: String,
    pub signer: String,
    pub signer_role: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct TaskRecord {
    pub task: String,
    pub reason: String,
    pub target: String,
    pub classes: usize,
    pub rows: usize,
    pub features: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct MethodRecord {
    pub requested: String,
    pub selected: String,
    pub recipe_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct FileRecord {
    pub role: &'static str,
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize, Default)]
pub struct Timings {
    pub read_seconds: f64,
    pub select_seconds: f64,
    pub fit_seconds: f64,
    pub predict_seconds: f64,
    pub write_seconds: f64,
    pub total_seconds: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ErrorRecord {
    pub kind: crate::FailureKind,
    pub exit_code: u8,
    pub message: String,
}
