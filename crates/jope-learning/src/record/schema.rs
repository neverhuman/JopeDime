use super::*;

fn add_direction_alias(mut schema: serde_json::Value) -> serde_json::Value {
    let Some(direction) = schema
        .get("$defs")
        .and_then(|definitions| definitions.get("MetricDirection"))
        .cloned()
    else {
        return schema;
    };
    if let Some(definitions) = schema
        .get_mut("$defs")
        .and_then(serde_json::Value::as_object_mut)
    {
        definitions.insert("Direction".to_owned(), direction);
    }
    schema
}

/// The schemars derivation of [`HwmRecord`] (tests compare it with the committed file).
pub fn derived_hwm_schema() -> serde_json::Value {
    add_direction_alias(schemars::schema_for!(HwmRecord).to_value())
}

/// The schemars derivation of [`ScoreRecord`].
pub fn derived_score_schema() -> serde_json::Value {
    add_direction_alias(schemars::schema_for!(ScoreRecord).to_value())
}
