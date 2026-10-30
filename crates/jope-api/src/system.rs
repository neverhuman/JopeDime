use crate::{decimal::DecimalU64, ids::DatasetId, Language, MetricSample, Task, Warning};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SystemInfo {
    pub version: String,
    pub api_version: String,
    pub health: String,
    pub local_paths: bool,
    pub capabilities: Capabilities,
    pub warnings: Vec<Warning>,
    pub telemetry: Vec<MetricSample>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Capabilities {
    pub tasks: Vec<Task>,
    pub languages: Vec<Language>,
    pub devices: Vec<String>,
    pub learned_guidance: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct DatasetProfile {
    pub dataset_id: DatasetId,
    pub name: String,
    pub bytes: DecimalU64,
    pub rows: DecimalU64,
    pub columns: Vec<ColumnProfile>,
    pub target_suggestions: Vec<TargetSuggestion>,
    pub warnings: Vec<Warning>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ColumnProfile {
    pub name: String,
    pub data_type: String,
    pub missing: DecimalU64,
    pub distinct: DecimalU64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TargetSuggestion {
    pub column: String,
    pub task: Task,
    pub reason: String,
}
