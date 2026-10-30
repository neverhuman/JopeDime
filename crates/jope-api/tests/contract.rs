use jope_api::*;
use serde::{de::DeserializeOwned, Serialize};
use serde_json::{json, Value};
use std::fmt::Debug;

mod support;

fn round_trip<T: Serialize + DeserializeOwned + PartialEq + Debug>(value: Value) {
    let first: T = serde_json::from_value(value.clone()).unwrap();
    let encoded = serde_json::to_value(&first).unwrap();
    assert_eq!(
        encoded, value,
        "no example field may disappear or change kind"
    );
    let second: T = serde_json::from_value(encoded).unwrap();
    assert_eq!(first, second);
}

fn plan() -> Value {
    serde_json::from_str(include_str!("fixtures/plan.json")).unwrap()
}

fn warning() -> Value {
    json!({"code":"MEMORY_PRESSURE", "severity":"warn", "state":"raised",
        "host":true, "value":8.0, "limit":10.0, "message":"Available RAM is low",
        "remedy":"lower effort or close other programs"})
}

fn sample() -> Value {
    json!({"name":"cpu.cores", "value":1.0, "unit":"core-equivalents",
        "scope":"engine", "window_ms":1000, "age_ms":0, "availability":"measured"})
}

#[test]
fn normative_run_spec_round_trip() {
    round_trip::<RunSpec>(serde_json::from_str(include_str!("fixtures/run_spec.json")).unwrap());
}

#[test]
fn normative_plan_round_trip() {
    round_trip::<Plan>(plan());
}

#[test]
fn normative_run_snapshot_round_trip() {
    round_trip::<RunSnapshot>(
        serde_json::from_str(include_str!("fixtures/run_snapshot.json")).unwrap(),
    );
}

#[test]
fn normative_reasoning_event_round_trip() {
    round_trip::<PublicEvent>(
        serde_json::from_str(include_str!("fixtures/reasoning_event.json")).unwrap(),
    );
}

#[test]
fn every_appendix_b_durable_body_round_trips() {
    let configuration = json!({"revision":"17", "effort":"medium",
        "budget":{"evaluations":"128", "wall_seconds":"60"}});
    let cases = [
        (
            "run.state",
            json!({"state":"pausing", "phase":"searching",
            "pending":"round commit", "reason":null}),
        ),
        (
            "run.config",
            json!({"before":configuration, "after":configuration}),
        ),
        ("plan", plan()),
        (
            "experiment.started",
            json!({"experiment_id":"exp_1",
            "hypothesis":"Try log-scaled income"}),
        ),
        (
            "experiment.finished",
            json!({"experiment_id":"exp_1", "outcome":"failed",
            "metric":"rmse", "value":null,
            "failure":{"code":"FITNESS_UNDEFINED", "reason":"candidate predictions were non-finite"}}),
        ),
        (
            "algorithm.accepted",
            json!({"algorithm_id":"alg_0007", "metric":"rmse",
            "cv_value":0.4871, "previous_value":0.512, "summary":"New best",
            "model_artifact":"art_3be100000000000000000000", "excerpt":"fn predict() {}"}),
        ),
        (
            "reasoning",
            json!({"stage":"accepted", "algorithm_id":"alg_0007",
            "summary":"New best", "evidence":{"metric":"rmse", "scope":"cv_fit_5fold",
                "before":0.512, "after":0.487, "comparable":true}}),
        ),
        ("warning", warning()),
        (
            "export.completed",
            json!({"artifact_id":"art_3be100000000000000000000",
            "kind":"bundle", "languages":["rust", "python"], "parity_status":"passed"}),
        ),
        (
            "run.finished",
            json!({"completion_reason":"budget_evaluations", "champion":null,
            "scores":{"cv":null,"holdout":null,"test":null},
            "budget_used":{"evaluations":"128","wall_seconds":"23.4"},
            "nondeterminism":{"count":"1","experiment_ids":["exp_1"]}}),
        ),
    ];
    assert_eq!(cases.len(), 10);
    for (kind, body) in cases {
        round_trip::<PublicEvent>(json!({"run_id":"run_01JBD8Q6N4V2T5Y8K3M7P9R1S0",
            "sequence":"214", "time":"2026-10-29T18:31:07.412Z", "kind":kind, "body":body}));
    }
}

#[test]
fn remaining_top_level_types_round_trip() {
    round_trip::<Algorithm>(json!({"algorithm_id":"alg_0007", "summary":"New best",
        "metric":"rmse", "cv_value":0.4871, "model_artifact":null,
        "model_digest":"sha256:model", "conditioning_digest":"sha256:conditioning",
        "bundle_digest":"sha256:bundle", "revision":"17",
        "source":[{"language":"rust","source":"fn predict() {}"}]}));
    round_trip::<ExportRequest>(json!({"algorithm_id":"alg_0007", "kind":"bundle",
        "languages":["python","rust"], "require_all_languages":true, "refit":"fit_holdout"}));
    round_trip::<ExportManifest>(json!({"artifact_id":"art_3be100000000000000000000",
        "algorithm_id":"alg_0007", "kind":"bundle", "model_digest":"sha256:model",
        "emitter_version":"1.0", "qualified_set_id":"v1", "refit":"fit_holdout",
        "retains_training_data":false,
        "languages":[{"status":"exported","language":"rust","path":"src/rust/predictor.rs"},
            {"status":"unsupported","language":"python","operations":["example_operation"]}],
        "parity_status":"passed", "parity_receipt":null}));
    round_trip::<PredictRequest>(json!({"model_artifact":"art_3be100000000000000000000",
        "dataset":"ds_7f3a9c0e1b2d4a5f6e7d8c9b"}));
    round_trip::<SystemInfo>(json!({"version":"0.4.1", "api_version":"1.0",
        "health":"ok", "local_paths":true, "capabilities":{"tasks":["regression","binary"],
            "languages":["rust","python"],"devices":["cpu"],"learned_guidance":false},
        "warnings":[warning()], "telemetry":[sample()]}));
    round_trip::<DatasetProfile>(json!({"dataset_id":"ds_7f3a9c0e1b2d4a5f6e7d8c9b",
        "name":"train.csv", "bytes":"9007199254740993", "rows":"20640",
        "columns":[{"name":"target","data_type":"numeric","missing":"0","distinct":"3842"}],
        "target_suggestions":[{"column":"target","task":"regression","reason":"numeric target"}],
        "warnings":[]}));
    round_trip::<Warning>(warning());
    round_trip::<MetricSample>(sample());
}

#[test]
fn telemetry_is_ephemeral_and_null_is_preserved() {
    let mut unavailable = sample();
    unavailable["value"] = Value::Null;
    unavailable["availability"] = json!("unavailable:permission_denied");
    round_trip::<MetricSample>(unavailable.clone());
    let mut boolean = sample();
    boolean["name"] = json!("throughput.tokens_per_s.rust_timed_alone");
    boolean["value"] = json!(true);
    boolean["unit"] = json!("bool");
    round_trip::<MetricSample>(boolean.clone());
    let mut bytes = sample();
    bytes["name"] = json!("store.bytes");
    bytes["value"] = json!("9007199254740993");
    bytes["unit"] = json!("bytes");
    round_trip::<MetricSample>(bytes.clone());
    let ephemeral = json!({"run_id":"run_01JBD8Q6N4V2T5Y8K3M7P9R1S0",
        "time":"2026-10-29T18:31:07.412Z", "kind":"telemetry", "body":[unavailable,boolean,bytes]});
    round_trip::<TelemetryEvent>(ephemeral.clone());
    assert!(serde_json::from_value::<PublicEvent>(ephemeral).is_err());
    let mut durable_shape = json!({"run_id":"run_01JBD8Q6N4V2T5Y8K3M7P9R1S0",
        "time":"2026-10-29T18:31:07.412Z", "sequence":"1", "kind":"telemetry", "body":[]});
    assert!(serde_json::from_value::<PublicEvent>(durable_shape.clone()).is_err());
    durable_shape["kind"] = json!("future.event");
    assert!(serde_json::from_value::<PublicEvent>(durable_shape).is_err());
}

#[test]
fn event_kind_selects_its_body_and_cursors_stay_strings() {
    let valid: Value = serde_json::from_str(include_str!("fixtures/reasoning_event.json")).unwrap();
    let mut mismatched = valid.clone();
    mismatched["kind"] = json!("experiment.finished");
    assert!(serde_json::from_value::<PublicEvent>(mismatched).is_err());
    let mut numeric = valid.clone();
    numeric["sequence"] = json!(214);
    assert!(serde_json::from_value::<PublicEvent>(numeric).is_err());
    for sequence in ["0", "9007199254740993", "18446744073709551615"] {
        let mut event = valid.clone();
        event["sequence"] = json!(sequence);
        round_trip::<PublicEvent>(event);
    }
}

#[test]
fn unknown_vocabularies_and_wrong_field_kinds_are_rejected() {
    for bad in [json!("extreme"), json!(1)] {
        let mut spec: Value = serde_json::from_str(include_str!("fixtures/run_spec.json")).unwrap();
        spec["effort"] = bad;
        assert!(serde_json::from_value::<RunSpec>(spec).is_err());
    }
    let mut spec: Value = serde_json::from_str(include_str!("fixtures/run_spec.json")).unwrap();
    spec["objective"] = json!({"version":1,"metric":"rmse"});
    round_trip::<RunSpec>(spec);
    for state in [
        "running",
        "pausing",
        "paused",
        "completed",
        "cancelled",
        "failed",
    ] {
        round_trip::<RunState>(json!(state));
    }
}

#[test]
fn every_top_level_schema_matches_the_committed_snapshot() {
    let expected: Value = serde_json::from_str(include_str!("snapshots/schemas.json")).unwrap();
    assert_eq!(
        support::schemas(),
        expected,
        "review schema changes, then regenerate with the schema_snapshots example"
    );
}
