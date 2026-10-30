use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Warning code vocabulary from spec §9.3. Runtime state is owned by jope-service.
pub const WARNING_CODES: &[&str] = &[
    "GPU_NOT_DETECTED",
    "BACKEND_UNAVAILABLE",
    "GPU_SENSOR_UNAVAILABLE",
    "MEMORY_PRESSURE",
    "DISK_LOW",
    "DATASET_LARGE",
    "WORKER_LIMIT",
    "LANGUAGE_CONSTRAINT",
    "GUIDANCE_HEURISTIC",
    "OBJECTIVE_SURROGATE_MISMATCH",
    "MODEL_INTEGRITY_FAILED",
    "NO_VALID_MODEL",
    "EVENT_LAG",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Info,
    Warn,
    Error,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WarningState {
    Raised,
    Cleared,
}

/// Public warning fields from §9.3. TEL-06 owns the warning catalog.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Warning {
    pub code: String,
    pub severity: Severity,
    pub state: WarningState,
    pub host: bool,
    pub value: Option<f64>,
    pub limit: Option<f64>,
    pub message: String,
    pub remedy: String,
}
