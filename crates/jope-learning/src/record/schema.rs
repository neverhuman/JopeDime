use super::*;

/// The schemars derivation of [`HwmRecord`] (tests compare it with the committed file).
pub fn derived_hwm_schema() -> serde_json::Value {
    schemars::schema_for!(HwmRecord).to_value()
}

/// The schemars derivation of [`ScoreRecord`].
pub fn derived_score_schema() -> serde_json::Value {
    schemars::schema_for!(ScoreRecord).to_value()
}
