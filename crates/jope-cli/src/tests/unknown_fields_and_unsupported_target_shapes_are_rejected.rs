use super::*;

#[test]
fn unknown_fields_and_unsupported_target_shapes_are_rejected() {
    let bytes = serialize(&dataset(Task::Regression, 0)).unwrap();
    let original: Value = serde_json::from_slice(&bytes).unwrap();
    for value in [
        json!([1, 2]),
        json!(null),
        json!(1.5),
        json!("1"),
        json!({"value": 1}),
    ] {
        let mut changed = original.clone();
        changed["rows"][0]["target"] = value;
        assert!(serde_json::from_value::<Dataset>(changed).is_err());
    }
    let mut changed = original;
    changed["unexpected_schema_field"] = json!(true);
    assert!(serde_json::from_value::<Dataset>(changed).is_err());
}
