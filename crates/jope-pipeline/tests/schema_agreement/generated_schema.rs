use super::*;

#[test]
fn schema_only_consumers_require_every_new_v2_field() {
    let schema = schema();
    for field in [
        "budget_policy",
        "compared_cuts",
        "coverage",
        "seeding",
        "islands",
    ] {
        assert!(schema["$defs"]["Search"]["required"]
            .as_array()
            .unwrap()
            .contains(&json!(field)));
        let mut v = fixture();
        v["search"].as_object_mut().unwrap().remove(field);
        let errors = jope_datasets::schema::validate(&schema, &v);
        assert!(errors.iter().any(|e| e.contains(field)), "{errors:?}");
    }
    assert!(schema["$defs"]["CutSearch"]["required"]
        .as_array()
        .unwrap()
        .contains(&json!("accounting")));
    let mut v = fixture();
    v["search"]["cuts"][0]
        .as_object_mut()
        .unwrap()
        .remove("accounting");
    assert!(jope_datasets::schema::validate(&schema, &v)
        .iter()
        .any(|e| e.contains("accounting")));
}

#[test]
fn committed_schema_is_generated_from_code_exactly() {
    assert_eq!(schema(), derived_schema());
}

#[test]
fn the_objective_receipt_is_v3_plus_the_objective_generated_from_code() {
    let committed = objective_schema();
    assert_eq!(committed, derived_objective_schema());
    let schemas = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../schemas");
    assert_eq!(
        std::fs::read_to_string(schemas.join("jope.pipeline-objective-receipt.v1.schema.json"))
            .unwrap(),
        OBJECTIVE_SCHEMA_JSON
    );
    assert_eq!(committed["title"], OBJECTIVE_SCHEMA_ID);
    assert_eq!(
        committed["properties"]["schema"]["const"],
        OBJECTIVE_SCHEMA_ID
    );
    // Whatever v3 says, the objective receipt says identically: it only adds.
    let v3 = schema();
    let mut reduced = committed.clone();
    for pointer in [
        "/$defs/Best/properties/validation_objective",
        "/$defs/CutSearch/properties/best_validation_objective",
        "/$defs/Search/properties/objective",
        "/$defs/Objective",
    ] {
        let (parent, key) = pointer.rsplit_once('/').unwrap();
        let parent = reduced
            .pointer_mut(parent)
            .unwrap()
            .as_object_mut()
            .unwrap();
        assert!(parent.remove(key).is_some(), "{pointer}");
    }
    reduced["$id"] = v3["$id"].clone();
    reduced["title"] = v3["title"].clone();
    reduced["properties"]["schema"]["const"] = v3["properties"]["schema"]["const"].clone();
    assert_eq!(reduced, v3);
}

#[test]
fn only_the_objective_receipt_records_an_objective() {
    let mut ranked = fixture();
    ranked["search"]["objective"] = json!({
        "name": "pinball", "direction": "minimize", "fitness_sha256": "0".repeat(64),
        "specification": {"metric": "pinball", "quantile": 0.9}});
    // The frozen v3 schema has no place for it.
    assert!(validate_value(&ranked).is_err());
    // An objective receipt without its objective says nothing about how it ranked.
    let mut bare = fixture();
    bare["schema"] = OBJECTIVE_SCHEMA_ID.into();
    let errors = validate_value(&bare).unwrap_err();
    assert!(
        errors.iter().any(|e| e.contains("lacks an objective")),
        "{errors:?}"
    );
    // With its objective, each best program must carry the loss it was best by.
    ranked["schema"] = OBJECTIVE_SCHEMA_ID.into();
    let errors = validate_value(&ranked).unwrap_err();
    for missing in [
        "lacks its validation_objective",
        "lacks its best_validation_objective",
    ] {
        assert!(errors.iter().any(|e| e.contains(missing)), "{errors:?}");
    }
}

#[test]
fn asm_contract_schemas_match_generated_types_exactly() {
    use jope_pipeline::experiment::{EvaluationRequest, EvaluationResult, Experiment, Panel};
    let schemas = [
        (
            "jope.experiment.v1",
            serde_json::to_value(schemars::schema_for!(Experiment)).unwrap(),
        ),
        (
            "jope.benchmark-panel.v1",
            serde_json::to_value(schemars::schema_for!(Panel)).unwrap(),
        ),
        (
            "jope.evaluation-request.v1",
            serde_json::to_value(schemars::schema_for!(EvaluationRequest)).unwrap(),
        ),
        (
            "jope.evaluation-result.v1",
            serde_json::to_value(schemars::schema_for!(EvaluationResult)).unwrap(),
        ),
        (
            "jope.evaluation-result.v2",
            serde_json::to_value(schemars::schema_for!(
                jope_pipeline::evaluation_result::EvaluationResult
            ))
            .unwrap(),
        ),
    ];
    for (name, generated) in schemas {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../schemas")
            .join(format!("{name}.schema.json"));
        let committed: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(committed, generated, "{name}");
    }
}
