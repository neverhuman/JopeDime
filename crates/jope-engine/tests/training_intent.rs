#![cfg(feature = "neural")]

use jope_parallel::{
    manifest::hash,
    task_contract,
    training::{
        inputs::{freeze_intent_closure, SCHEMA as INPUTS_SCHEMA},
        intent::{self, ParentState},
    },
};
use serde_json::{json, Value};
use std::collections::BTreeMap;

const INTENT: &str = include_str!("fixtures/training-intent-v1/intent.json");
const INCOMPATIBLE_PARENT: &str =
    include_str!("fixtures/training-intent-v1/incompatible-parent.json");
const INCOMPATIBLE_CORPUS: &str =
    include_str!("fixtures/training-intent-v1/incompatible-corpus.json");
const PARENT: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const EPOCH: &str = "2222222222222222222222222222222222222222222222222222222222222222";
const CORPUS: &str = "1111111111111111111111111111111111111111111111111111111111111111";

fn intent() -> Value {
    serde_json::from_str(INTENT).unwrap()
}

fn parent() -> ParentState<'static> {
    let parameters = BTreeMap::from([
        ("layer.weight".to_owned(), vec![2, 3]),
        ("layer.bias".to_owned(), vec![2]),
    ]);
    // Leak these tiny test maps to make the borrowed state concise.
    let parameters = Box::leak(Box::new(parameters));
    ParentState {
        checkpoint_sha256: PARENT,
        optimizer_epoch_sha256: EPOCH,
        optimizer_step: 12,
        parameter_shapes: parameters,
        first_moment_shapes: parameters,
        second_moment_shapes: parameters,
    }
}

fn frozen_inputs() -> Value {
    json!({
        "schema": "jope.training_intent_inputs/v1",
        "base_inputs": {
            "schema": INPUTS_SCHEMA,
            "corpus": {"index_sha256": CORPUS}
        },
        "base_inputs_sha256": hash(b"base-inputs").as_str(),
        "per_head_eligibility": {"head-a": "5555555555555555555555555555555555555555555555555555555555555555"},
        "fresh_occurrences": ["fresh-1"],
        "replay_occurrences": ["replay-1"]
    })
}

#[test]
fn compatible_intent_preflights_and_returns_stable_digests() {
    let frozen = frozen_inputs();
    let first = intent::validate(&intent(), &frozen, &parent()).unwrap();
    let second = intent::validate(&intent(), &frozen, &parent()).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.initial_step, 12);
    assert_eq!(first.target_step, 20);
}

#[test]
fn unsupported_intent_version_is_refused() {
    let mut input = intent();
    input["schema"] = json!("jope.training-intent/v2");
    assert!(intent::validate(&input, &frozen_inputs(), &parent()).is_err());
}

#[test]
fn parent_checkpoint_mismatch_is_refused() {
    let input: Value = serde_json::from_str(INCOMPATIBLE_PARENT).unwrap();
    assert!(intent::validate(&input, &frozen_inputs(), &parent()).is_err());
}

#[test]
fn optimizer_epoch_mismatch_is_refused_before_any_launch_state_exists() {
    let mut input = intent();
    input["optimizer_epoch_sha256"] =
        json!("9999999999999999999999999999999999999999999999999999999999999999");
    // validate has no CUDA, filesystem, journal, or optimizer handles.
    assert!(intent::validate(&input, &frozen_inputs(), &parent()).is_err());
}

#[test]
fn parent_step_mismatch_is_refused() {
    let mut input = intent();
    input["initial_step"] = json!(11);
    assert!(intent::validate(&input, &frozen_inputs(), &parent()).is_err());
}

#[test]
fn first_moment_shape_mismatch_is_refused() {
    let params = BTreeMap::from([("weight".to_owned(), vec![2, 3])]);
    let first = BTreeMap::from([("weight".to_owned(), vec![3, 2])]);
    let second = params.clone();
    let state = ParentState {
        checkpoint_sha256: PARENT,
        optimizer_epoch_sha256: EPOCH,
        optimizer_step: 12,
        parameter_shapes: &params,
        first_moment_shapes: &first,
        second_moment_shapes: &second,
    };
    assert!(intent::validate(&intent(), &frozen_inputs(), &state).is_err());
}

#[test]
fn second_moment_key_mismatch_is_refused() {
    let params = BTreeMap::from([("weight".to_owned(), vec![2, 3])]);
    let first = params.clone();
    let second = BTreeMap::new();
    let state = ParentState {
        checkpoint_sha256: PARENT,
        optimizer_epoch_sha256: EPOCH,
        optimizer_step: 12,
        parameter_shapes: &params,
        first_moment_shapes: &first,
        second_moment_shapes: &second,
    };
    assert!(intent::validate(&intent(), &frozen_inputs(), &state).is_err());
}

#[test]
fn corpus_change_requires_matching_frozen_input_closure() {
    let input: Value = serde_json::from_str(INCOMPATIBLE_CORPUS).unwrap();
    assert!(intent::validate(&input, &frozen_inputs(), &parent()).is_err());
}

#[test]
fn incomplete_protocol_identity_is_refused() {
    let mut input = intent();
    input["protocol"]
        .as_object_mut()
        .unwrap()
        .remove("support_sha256");
    assert!(intent::validate(&input, &frozen_inputs(), &parent()).is_err());
}

#[test]
fn original_gpu_grant_hold_refuses_public_route() {
    let error = task_contract::require_original_gpu_grant("training-intent-test").unwrap_err();
    assert!(format!("{error:#}").contains("grant verifier unavailable"));
}

#[test]
fn new_outcome_creates_a_later_closure_without_changing_the_old_one() {
    let base = json!({
        "schema": INPUTS_SCHEMA,
        "corpus": {"index_sha256": CORPUS}
    });
    let eligibility =
        json!({"head-a": "5555555555555555555555555555555555555555555555555555555555555555"});
    let before = freeze_intent_closure(
        &base,
        &eligibility,
        &["fresh-1".into()],
        &["replay-1".into()],
    )
    .unwrap();
    let before_digest = hash(&serde_json::to_vec(&before).unwrap());
    let later = freeze_intent_closure(
        &base,
        &eligibility,
        &["fresh-1".into(), "outcome-2".into()],
        &["replay-1".into()],
    )
    .unwrap();
    assert_eq!(hash(&serde_json::to_vec(&before).unwrap()), before_digest);
    assert_ne!(hash(&serde_json::to_vec(&later).unwrap()), before_digest);
    assert!(later["fresh_occurrences"]
        .as_array()
        .unwrap()
        .iter()
        .any(|x| x == "outcome-2"));
}

#[test]
fn occurrence_cannot_be_reclassified_or_duplicated_in_place() {
    let base = json!({"schema": INPUTS_SCHEMA, "corpus": {"index_sha256": CORPUS}});
    let eligibility =
        json!({"head-a": "5555555555555555555555555555555555555555555555555555555555555555"});
    assert!(
        freeze_intent_closure(&base, &eligibility, &["same".into()], &["same".into()]).is_err()
    );
    assert!(freeze_intent_closure(
        &base,
        &eligibility,
        &["same".into(), "same".into()],
        &["replay".into()]
    )
    .is_err());
}
