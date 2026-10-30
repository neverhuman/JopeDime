use crate::decimal::DecimalU64;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// §9.1 includes both numeric measurements and the rust_timed_alone boolean.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum MetricValue {
    Number(f64),
    Boolean(bool),
    /// Exact byte counts follow §4.1 even above JavaScript's integer range.
    Decimal(DecimalU64),
}

/// Null is unavailable, never zero. Scope and availability are extensible strings
/// (`device:<uuid>` and `unavailable:<reason>` respectively).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MetricSample {
    pub name: String,
    pub value: Option<MetricValue>,
    pub unit: String,
    pub scope: String,
    pub window_ms: u32,
    pub age_ms: u32,
    pub availability: String,
}
