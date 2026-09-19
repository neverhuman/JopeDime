//! The committed schemas are the schemars derivations of the Rust types (plus `$id`, title and
//! description), and real records from every source validate against them.

mod common;

use jope_datasets::schema::validate;
use jope_learning::record::{derived_hwm_schema, derived_score_schema};
use jope_learning::sources::{champions, genebank, sotarena};
use serde_json::Value;

fn body(mut v: Value) -> Value {
    for k in ["$schema", "$id", "title", "description"] {
        v.as_object_mut().unwrap().remove(k);
    }
    v
}

#[test]
fn committed_schemas_are_the_derived_schemas() {
    let hwm: Value = serde_json::from_str(jope_learning::HWM_SCHEMA_JSON).unwrap();
    assert_eq!(hwm["title"], "jope.hwm/v1");
    assert_eq!(body(hwm), body(derived_hwm_schema()));
    let score: Value = serde_json::from_str(jope_learning::SCORE_SCHEMA_JSON).unwrap();
    assert_eq!(score["title"], "jope.score/v1");
    assert_eq!(body(score), body(derived_score_schema()));
}

#[test]
fn hwm_schema_pins_the_plan_record() {
    let hwm: Value = serde_json::from_str(jope_learning::HWM_SCHEMA_JSON).unwrap();
    let required: Vec<&str> = hwm["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    for k in [
        "schema",
        "identity",
        "identity_sha256",
        "genome",
        "score",
        "cost",
        "evaluator",
        "benchmark_valid",
        "sealed",
        "replayable",
        "provenance",
        "prev_event_sha256",
        "improvement_number",
        "ts_unix",
    ] {
        assert!(required.contains(&k), "{k} must be required");
    }
    assert!(!required.contains(&"legacy_identity_sha256"));
    assert_eq!(hwm["additionalProperties"], false);
    let id = &hwm["$defs"]["HwmIdentity"]["required"];
    assert_eq!(
        id.as_array().unwrap().len(),
        6,
        "identity has exactly the six hashed fields"
    );
    let metrics: Vec<&str> = hwm["$defs"]["Metric"]["enum"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(
        metrics,
        [
            "r2",
            "oof_r2",
            "auc",
            "macro_f1",
            "negative_mase",
            "rmse",
            "logloss",
            "mase"
        ]
    );
    assert_eq!(
        hwm["$defs"]["Direction"]["enum"],
        serde_json::json!(["higher", "lower"])
    );
}

#[test]
fn real_records_validate_against_the_committed_schemas() {
    let hwm: Value = serde_json::from_str(jope_learning::HWM_SCHEMA_JSON).unwrap();
    let score: Value = serde_json::from_str(jope_learning::SCORE_SCHEMA_JSON).unwrap();
    // Champions (chain fields copied, bound evaluator, one sealed).
    let (cands, scan) = champions::scan(&common::fixtures().join("champions"), "fixture").unwrap();
    assert_eq!((scan.files, scan.mapped, scan.sealed), (10, 10, 2));
    for c in &cands {
        let v = serde_json::to_value(&c.record).unwrap();
        let errors = validate(&hwm, &v);
        assert!(errors.is_empty(), "{errors:?}\n{v}");
    }
    // Genebank (unbound, no chain).
    let row = genebank::Row {
        task: "binary".into(),
        dataset_hash: "0259e0b6af043f58".into(),
        score_tier: "test_exact".into(),
        metric: "auc".into(),
        score: 0.91,
        semantic_sha256: Some(
            "4d86203118a32b2182ac63b67bb1c03c32cb63c47f3105262dfc5385e2fb95e5".into(),
        ),
        canonical_genome: "(pipe X (logit_ridge l2=0.5))".into(),
        achieved_at_ms: 1_784_970_730_252,
        receipt_ref: None,
    };
    let c = genebank::map_row(&row, "db").unwrap();
    let v = serde_json::to_value(&c.record).unwrap();
    assert!(v["prev_event_sha256"].is_null());
    let errors = validate(&hwm, &v);
    assert!(errors.is_empty(), "{errors:?}");
    // A corrupted evaluator hash is rejected by the schema.
    let mut bad = v.clone();
    bad["evaluator"]["sha256"] = "not-a-hash".into();
    assert!(!validate(&hwm, &bad).is_empty());
    // SOTArena score rows.
    let csv = "track_id,dataset_id,method_id,protocol,metric,value,status,benchmark_valid,source_kind,source_revision,source_sha256,source_uri,notes\n\
tabular-regression-v2,019de5d76a718163,ada_boost,public-development,rmse,0.0733947653603558,ok,True,reproducible-run-report,2026-06-30T00:40:20+00:00,c1c0c56758be19f97ee4d716bab1c3627f6bef689afdae3fc108f4d440cb6dfc,results/external-references/golden42/019de5d76a718163/ada_boost.json,Exact Golden42 report; method identity is the report filename.\n";
    let (rows, scan) =
        sotarena::parse_scores(csv.as_bytes(), "tabular-regression-v2", "unbound", 0).unwrap();
    assert_eq!((scan.rows, scan.mapped), (1, 1));
    let v = serde_json::to_value(&rows[0]).unwrap();
    let errors = validate(&score, &v);
    assert!(errors.is_empty(), "{errors:?}");
    assert_eq!(rows[0].ts_unix, 1_782_780_020);
    assert!(rows[0].benchmark_valid);
}
