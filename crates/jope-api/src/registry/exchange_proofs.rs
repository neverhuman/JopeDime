use super::*;
use serde::{de::DeserializeOwned, Serialize};
use serde_json::{json, Value};

fn exchange_snapshot_path() -> std::path::PathBuf {
    std::env::var_os("JOPE_EXCHANGE_SNAPSHOT_OUTPUT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/snapshots/exchanges.json")
        })
}
use std::collections::BTreeMap;

fn positives() -> BTreeMap<&'static str, Vec<Value>> {
    let profile = json!({"dataset_id":"ds_00112233445566778899aabb","name":"training fixture","bytes":"42","rows":"3","columns":[],"target_suggestions":[],"warnings":[]});
    let snapshot: Value =
        serde_json::from_str(include_str!("../../tests/fixtures/run_snapshot.json")).unwrap();
    let event: Value =
        serde_json::from_str(include_str!("../../tests/fixtures/reasoning_event.json")).unwrap();
    let algorithm = json!({"algorithm_id":"alg_0001","summary":"Public fitted algorithm","metric":"rmse","cv_value":0.25,"model_artifact":null,"model_digest":"sha256:model","conditioning_digest":"sha256:conditioning","bundle_digest":"sha256:bundle","revision":"1","source":[]});
    let feedback = json!({"algorithm":"opt-in evaluation description","features":[0.5],"scores":{"rmse":0.25,"test":null},"failure":null,"conditioning_tokens":vec![vec![0.0;32];10],"conditioning_mask":vec![true;10]});
    BTreeMap::from([
        (
            "DatasetRegistration",
            vec![
                json!({"path":"data/train.csv","name":"training"}),
                json!({"csv_text":"x,y\n1,2\n"}),
            ],
        ),
        (
            "DatasetPage",
            vec![json!({"datasets":[profile],"next_after":null})],
        ),
        (
            "RunPage",
            vec![json!({"runs":[snapshot],"next_after":"opaque-continuation"})],
        ),
        (
            "AlgorithmPage",
            vec![json!({"algorithms":[algorithm],"next_after":null})],
        ),
        ("DeleteReceipt", vec![json!({"deleted":true})]),
        (
            "ObjectiveValidationRequest",
            vec![
                json!({"spec":"rmse","task":"regression"}),
                json!({"spec":{"name":"weighted_error","column":"weight"}}),
            ],
        ),
        (
            "ObjectiveValidation",
            vec![
                json!({"resolved":{"name":"rmse"},"digest":"sha256:reviewed-objective","direction":"minimize","required_columns":["target"],"errors":[]}),
                json!({"errors":[{"code":"INVALID_INPUT","field":"spec","message":"Unknown objective"}]}),
            ],
        ),
        (
            "RunControlRequest",
            vec![
                json!({"action":"pause"}),
                json!({"action":"continue","expected_revision":"7"}),
                json!({"action":"cancel"}),
                json!({"action":"set_effort","effort":"high"}),
                json!({"action":"set_effort","budget":{"evaluations":"20"},"expected_revision":"9007199254740993"}),
            ],
        ),
        (
            "RunForkRequest",
            vec![
                json!({"seed":"same"}),
                json!({"seed":"new","warm_start":true,"overrides":{"effort":"fast","test_dataset":null,"budget":{"wall_seconds":null}}}),
            ],
        ),
        (
            "EventPage",
            vec![
                json!({"events":[event.clone()],"next_after":event["sequence"]}),
                json!({"events":[],"next_after":"0"}),
            ],
        ),
        (
            "TrainingFeedback",
            vec![json!({"records":[]}), json!({"records":[feedback]})],
        ),
        (
            "GcRequest",
            vec![json!({"dry_run":true}), json!({"dry_run":false})],
        ),
        (
            "GcReport",
            vec![
                json!({"dry_run":true,"blobs":[{"sha256":"a".repeat(64),"bytes":"9007199254740993"}],"bytes":"9007199254740993"}),
            ],
        ),
    ])
}
fn negatives() -> BTreeMap<&'static str, Vec<Value>> {
    BTreeMap::from([
        (
            "DatasetRegistration",
            vec![
                json!({}),
                json!({"path":"a","csv_text":"b"}),
                json!({"path":""}),
            ],
        ),
        (
            "DatasetPage",
            vec![json!({"datasets":{},"next_after":null})],
        ),
        ("RunPage", vec![json!({"runs":{},"next_after":null})]),
        (
            "AlgorithmPage",
            vec![json!({"algorithms":{},"next_after":null})],
        ),
        ("DeleteReceipt", vec![json!({"deleted":false})]),
        (
            "ObjectiveValidationRequest",
            vec![
                json!({"spec":false}),
                json!({"spec":"rmse","task":"unknown"}),
            ],
        ),
        (
            "ObjectiveValidation",
            vec![
                json!({"errors":"bad"}),
                json!({"errors":[{"code":"NEW_UNKNOWN_CODE","message":"bad"}]}),
            ],
        ),
        (
            "RunControlRequest",
            vec![
                json!({"action":"stop"}),
                json!({"action":"pause","effort":"fast"}),
                json!({"action":"set_effort"}),
                json!({"action":"set_effort","effort":null,"budget":null}),
                json!({"action":"set_effort","budget":{}}),
            ],
        ),
        (
            "RunForkRequest",
            vec![
                json!({"seed":"invalid"}),
                json!({"seed":"same","overrides":{"seed":"5"}}),
                json!({"seed":"same","overrides":{"dry_run":true}}),
            ],
        ),
        (
            "EventPage",
            vec![
                json!({"events":[],"next_after":7}),
                json!({"events":[],"next_after":"01"}),
            ],
        ),
        (
            "TrainingFeedback",
            vec![json!({"records":[],"rows":[[1,2]]})],
        ),
        (
            "GcRequest",
            vec![
                json!({"dry_run":"true"}),
                json!({"dry_run":true,"force":true}),
            ],
        ),
        (
            "GcReport",
            vec![
                json!({"dry_run":true,"blobs":[],"bytes":42}),
                json!({"dry_run":true,"blobs":[{"sha256":"A".repeat(64),"bytes":"1"}],"bytes":"1"}),
            ],
        ),
    ])
}
fn document() -> Value {
    let mut schemas = BTreeMap::new();
    for operation in OPERATIONS {
        for exchange in operation
            .request_types
            .iter()
            .chain(operation.response_types)
        {
            if let Some(schema) = exchange.json_schema {
                let value = serde_json::to_value(schema()).unwrap();
                if let Some(previous) = schemas.insert(exchange.name, value.clone()) {
                    assert_eq!(previous, value);
                }
            }
        }
    }
    json!({"schemas":schemas,"positive":positives(),"negative":negatives()})
}
fn round_trip<T: DeserializeOwned + Serialize>(value: &Value) {
    let dto: T = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(serde_json::to_value(dto).unwrap(), *value);
}
fn typed_cases<T: DeserializeOwned + Serialize>(name: &str) {
    for value in &positives()[name] {
        round_trip::<T>(value);
    }
    for value in &negatives()[name] {
        assert!(
            serde_json::from_value::<T>(value.clone()).is_err(),
            "{name} accepted {value}"
        );
    }
}
#[test]
fn all_thirteen_exchange_examples_and_refusals_round_trip() {
    typed_cases::<DatasetRegistration>("DatasetRegistration");
    typed_cases::<DatasetPage>("DatasetPage");
    typed_cases::<RunPage>("RunPage");
    typed_cases::<AlgorithmPage>("AlgorithmPage");
    typed_cases::<DeleteReceipt>("DeleteReceipt");
    typed_cases::<ObjectiveValidationRequest>("ObjectiveValidationRequest");
    typed_cases::<ObjectiveValidation>("ObjectiveValidation");
    typed_cases::<RunControlRequest>("RunControlRequest");
    typed_cases::<RunForkRequest>("RunForkRequest");
    typed_cases::<EventPage>("EventPage");
    typed_cases::<TrainingFeedback>("TrainingFeedback");
    typed_cases::<GcRequest>("GcRequest");
    typed_cases::<GcReport>("GcReport");
}
#[test]
fn csv_limit_counts_utf8_bytes_at_the_inclusive_boundary() {
    for (text, valid) in [
        ("x".repeat(1024 * 1024), true),
        ("x".repeat(1024 * 1024 + 1), false),
        ("é".repeat(512 * 1024), true),
        ("é".repeat(512 * 1024 + 1), false),
    ] {
        assert_eq!(
            serde_json::from_value::<DatasetRegistration>(json!({"csv_text":text})).is_ok(),
            valid
        );
    }
}
#[test]
fn list_pages_and_event_pages_enforce_256_and_cursor_order() {
    let samples = positives();
    for (name, key) in [
        ("DatasetPage", "datasets"),
        ("RunPage", "runs"),
        ("AlgorithmPage", "algorithms"),
        ("EventPage", "events"),
    ] {
        let item = samples[name][0][key][0].clone();
        for count in [256, 257] {
            let mut page = samples[name][0].clone();
            page[key] = json!(vec![item.clone(); count]);
            if name == "EventPage" {
                for (index, event) in page[key].as_array_mut().unwrap().iter_mut().enumerate() {
                    event["sequence"] = json!((index + 1).to_string());
                }
                page["next_after"] = json!(count.to_string());
            }
            let valid = match name {
                "DatasetPage" => serde_json::from_value::<DatasetPage>(page).is_ok(),
                "RunPage" => serde_json::from_value::<RunPage>(page).is_ok(),
                "AlgorithmPage" => serde_json::from_value::<AlgorithmPage>(page).is_ok(),
                _ => serde_json::from_value::<EventPage>(page).is_ok(),
            };
            assert_eq!(valid, count == 256, "{name}/{count}");
        }
    }
    let event = samples["EventPage"][0]["events"][0].clone();
    assert!(serde_json::from_value::<EventPage>(
        json!({"events":[event.clone(),event.clone()],"next_after":event["sequence"]})
    )
    .is_err());
    assert!(
        serde_json::from_value::<EventPage>(json!({"events":[event],"next_after":"0"})).is_err()
    );
}
#[test]
fn fork_patch_preserves_omission_null_and_partial_budgets() {
    let missing: RunOverrides = serde_json::from_value(json!({})).unwrap();
    let clear: RunOverrides = serde_json::from_value(json!({"test_dataset":null})).unwrap();
    assert_eq!(missing.test_dataset, None);
    assert_eq!(clear.test_dataset, Some(None));
    assert_eq!(serde_json::to_value(missing).unwrap(), json!({}));
    assert_eq!(
        serde_json::to_value(clear).unwrap(),
        json!({"test_dataset":null})
    );
    round_trip::<RunControlRequest>(&json!({"action":"set_effort","budget":{"wall_seconds":null}}));
}
#[test]
fn feedback_has_fixed_conditioning_and_no_raw_rows() {
    let valid = positives()["TrainingFeedback"][1].clone();
    for field in ["conditioning_tokens", "conditioning_mask"] {
        let mut bad = valid.clone();
        bad["records"][0][field].as_array_mut().unwrap().pop();
        assert!(serde_json::from_value::<TrainingFeedback>(bad).is_err());
    }
    let mut raw = valid;
    raw["records"][0]["rows"] = json!([[1, 2]]);
    assert!(serde_json::from_value::<TrainingFeedback>(raw).is_err());
}
#[test]
fn generated_exchange_schemas_and_examples_are_current() {
    let path = exchange_snapshot_path();
    let expected: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(
        document(),
        expected,
        "regenerate through the explicit owning proof test"
    );
}
#[test]
fn regenerate_exchange_proofs() {
    if std::env::var_os("JOPE_EXCHANGE_SNAPSHOT_OUTPUT").is_none() {
        return;
    }
    let path = exchange_snapshot_path();
    std::fs::write(
        path,
        format!("{}\n", serde_json::to_string_pretty(&document()).unwrap()),
    )
    .unwrap();
}
