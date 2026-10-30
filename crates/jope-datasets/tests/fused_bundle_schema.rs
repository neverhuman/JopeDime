use jope_datasets::schema;
use serde_json::{json, Value};

const FUSED_BUNDLE_SCHEMA: &str = include_str!("../../../schemas/jope.fused-bundle.v1.schema.json");

fn example_manifest() -> Value {
    json!({
        "schema": "jope.fused-bundle/v1",
        "architecture": "dope-fused-ranker/h40b",
        "bundle_digest": format!("sha256:{}", "a".repeat(64)),
        "dope_core_rev": "15e12aa3dc33851655580f6aa7c9c66353143f23",
        "dope_weights_sha256": "d0b6960328d1955386632ff78b512e3add53a9b0256db7bbf067e2e0be5f0c6c",
        "jope_weights_sha256": "b".repeat(64),
        "reference_source_sha256": {
            "cpu_exact_fused.py": "b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab"
        },
        "constants": {
            "scale": 0.4208,
            "ctx_feat_dim": 584,
            "revealed": 4,
            "proposals": 8,
            "proposals_explore": 16,
            "admit": 4,
            "elite_pool": 48
        },
        "surrogate_recipe": "declared by exporter",
        "trained_objectives": ["regression"],
        "landmark_recipe": "full-high/v1",
        "landmark_sample_rows": 1024,
        "qualified": [
            { "task": "regression", "landmarks": "full", "backend": "cpu-f32" }
        ],
        "asset_policy": "redistributable"
    })
}

fn assert_valid(instance: &Value) {
    let schema: Value = serde_json::from_str(FUSED_BUNDLE_SCHEMA).unwrap();
    let errors = schema::validate(&schema, instance);
    assert!(errors.is_empty(), "manifest rejected: {errors:?}");
}

fn assert_invalid(instance: &Value) {
    let schema: Value = serde_json::from_str(FUSED_BUNDLE_SCHEMA).unwrap();
    let errors = schema::validate(&schema, instance);
    assert!(
        !errors.is_empty(),
        "manifest unexpectedly accepted: {instance}"
    );
}

#[test]
fn fused_bundle_manifest_validates_and_rejects_invalid_variants() {
    let valid = example_manifest();
    assert_valid(&valid);

    let mut missing_field = valid.clone();
    missing_field
        .as_object_mut()
        .unwrap()
        .remove("architecture");
    assert_invalid(&missing_field);

    let mut bad_hash = valid.clone();
    bad_hash["jope_weights_sha256"] = json!("A".repeat(64));
    assert_invalid(&bad_hash);

    let mut unknown_task = valid.clone();
    unknown_task["qualified"][0]["task"] = json!("clustering");
    assert_invalid(&unknown_task);

    let mut extra_top_level = valid.clone();
    extra_top_level["unexpected"] = json!(true);
    assert_invalid(&extra_top_level);

    let mut wrong_type = valid;
    wrong_type["constants"]["scale"] = json!("0.4208");
    assert_invalid(&wrong_type);
}
