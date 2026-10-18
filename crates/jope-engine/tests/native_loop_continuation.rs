//! W6 follow-up (c): the native loop's materialized training input and its
//! continuation-mode declaration (default features, CPU, no worker).
//!
//! Before this change a materialized input (`task.input.json`) had one schema
//! version, so a declared and an undeclared training input carried the same
//! label, and adding a declaration changed the bytes of every request frozen
//! before it (reproduction `w6-pre-nlcont-*`, which records the defect).
//!
//! Now `jope.smartcluster_input.v2` is a training input that declares its
//! lineage mode, and v1 never declares. The materializer re-materializes an
//! input under the version it was first frozen with
//! (`InputManifest::declare_training`), so earlier requests keep their exact
//! bytes. The worker (`smartcluster_worker::verify_task_inputs`) and the
//! native graph refuse any other pairing (`InputManifest::validate_version`).

use jope_parallel::{
    smartcluster_worker::{
        ClosureManifest, InputManifest, Operation, INPUT_SCHEMA_V1, INPUT_SCHEMA_V2,
    },
    task_contract::{classify_continuation, ContinuationMode},
};
use serde_json::json;
use std::collections::BTreeMap;

#[test]
fn v3_uses_the_descendant_epoch_without_changing_the_four_mode_strings() {
    use jope_parallel::native_loop::warm_start::{self, LOOP_V1, LOOP_V2, LOOP_V3};
    for (schema, base, expected) in [
        (LOOP_V1, 60000, (60000, 63000, None)),
        (LOOP_V2, 60000, (0, 3000, Some(60000))),
        (LOOP_V3, 6000, (6000, 9000, None)),
    ] {
        let window = warm_start::window(schema, base, 0).unwrap();
        assert_eq!(
            (window.initial, window.target, window.warm_start_parent_step),
            expected
        );
    }
    for (mode, name) in [
        (ContinuationMode::ColdInitialization, "cold_initialization"),
        (ContinuationMode::WeightWarmStart, "weight_warm_start"),
        (
            ContinuationMode::NewCorpusTransition,
            "new_corpus_transition",
        ),
        (ContinuationMode::ExactResume, "exact_resume"),
    ] {
        assert_eq!(serde_json::to_string(&mode).unwrap(), format!("\"{name}\""));
        assert_eq!(ContinuationMode::parse(name).unwrap(), mode);
    }
}

fn input(schema: &str, mode: Option<ContinuationMode>) -> InputManifest {
    InputManifest {
        schema: schema.into(),
        root: "/root".into(),
        operation: Operation::Train {
            corpus: "corpus".into(),
            run: "run".into(),
            launch: "launch".into(),
            micro: 256,
            checkpoint_steps: 1000,
            initialize: "parent".into(),
            replay_policy: "policy.json".into(),
            initial_optimizer_step: 60000,
            requested_step_increment: 3000,
            target_optimizer_step: 63000,
            continuation_mode: mode,
        },
        input_artifacts: BTreeMap::new(),
        closure: Some(ClosureManifest {
            path: "closure.jsonl".into(),
            sha256: "a".repeat(64),
        }),
        output_receipt: "result.json".into(),
    }
}

/// The exact bytes the materializer froze for a training input before the
/// declaration existed.
fn frozen_v1_bytes() -> String {
    format!(
        "{{\"schema\":\"jope.smartcluster_input.v1\",\"root\":\"/root\",\"operation\":\
         {{\"kind\":\"Train\",\"corpus\":\"corpus\",\"run\":\"run\",\"launch\":\"launch\",\
         \"micro\":256,\"checkpoint_steps\":1000,\"initialize\":\"parent\",\
         \"replay_policy\":\"policy.json\",\"initial_optimizer_step\":60000,\
         \"requested_step_increment\":3000,\"target_optimizer_step\":63000}},\
         \"input_artifacts\":{{}},\"closure\":{{\"path\":\"closure.jsonl\",\"sha256\":\"{}\"}},\
         \"output_receipt\":\"result.json\"}}",
        "a".repeat(64)
    )
}

/// What the materializer builds before declaring: the undeclared v1 form.
fn undeclared() -> InputManifest {
    input(INPUT_SCHEMA_V1, None)
}

/// The mode the native loop's replay policy classifies to.
fn native_mode() -> ContinuationMode {
    classify_continuation(false, true, Some(&json!({"optimizer":"carry"}))).unwrap()
}

#[test]
fn a_request_frozen_as_v1_re_materializes_byte_for_byte() {
    let frozen = frozen_v1_bytes();
    // The undeclared form is exactly the bytes frozen before the change.
    assert_eq!(serde_json::to_string(&undeclared()).unwrap(), frozen);
    let again = undeclared()
        .declare_training(Some(frozen.as_bytes()), native_mode())
        .unwrap();
    assert_eq!(serde_json::to_string(&again).unwrap(), frozen);
    again.validate_version().unwrap();
    let Operation::Train {
        continuation_mode, ..
    } = &again.operation
    else {
        panic!("not a training input")
    };
    assert_eq!(*continuation_mode, None, "v1 keeps its implicit meaning");
    // A frozen v1 input still verifies at the worker boundary.
    let parsed: InputManifest = serde_json::from_str(&frozen).unwrap();
    parsed.validate_version().unwrap();
}

#[test]
fn a_new_request_is_frozen_as_v2_with_its_declaration_and_stays_stable() {
    assert_eq!(native_mode(), ContinuationMode::NewCorpusTransition);
    let first = undeclared().declare_training(None, native_mode()).unwrap();
    assert_eq!(first.schema, INPUT_SCHEMA_V2);
    let Operation::Train {
        continuation_mode, ..
    } = &first.operation
    else {
        panic!("not a training input")
    };
    assert_eq!(
        *continuation_mode,
        Some(ContinuationMode::NewCorpusTransition)
    );
    first.validate_version().unwrap();
    let bytes = serde_json::to_vec(&first).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["schema"], INPUT_SCHEMA_V2);
    assert_eq!(
        value["operation"]["continuation_mode"],
        "new_corpus_transition"
    );
    // Re-materializing a v2 input reproduces its bytes.
    let again = undeclared()
        .declare_training(Some(&bytes), native_mode())
        .unwrap();
    assert_eq!(serde_json::to_vec(&again).unwrap(), bytes);
    // The declaration is the only difference from the v1 form.
    let mut v1: serde_json::Value = serde_json::from_str(&frozen_v1_bytes()).unwrap();
    v1["schema"] = json!(INPUT_SCHEMA_V2);
    v1["operation"]["continuation_mode"] = json!("new_corpus_transition");
    assert_eq!(value, v1);
}

#[test]
fn only_v2_declares_and_v2_always_declares_a_lineage_mode() {
    type Case = (&'static str, InputManifest);
    let refused: Vec<Case> = vec![
        (
            "v1 with a declaration",
            input(INPUT_SCHEMA_V1, Some(ContinuationMode::NewCorpusTransition)),
        ),
        ("v2 without a declaration", input(INPUT_SCHEMA_V2, None)),
        (
            "v2 declaring exact resume",
            input(INPUT_SCHEMA_V2, Some(ContinuationMode::ExactResume)),
        ),
        ("unknown schema", input("jope.smartcluster_input.v3", None)),
        (
            "v2 for a non-training operation",
            InputManifest {
                operation: Operation::Verify,
                ..input(INPUT_SCHEMA_V2, None)
            },
        ),
    ];
    for (case, manifest) in refused {
        assert!(manifest.validate_version().is_err(), "{case}: accepted");
    }
    for mode in [
        ContinuationMode::ColdInitialization,
        ContinuationMode::WeightWarmStart,
        ContinuationMode::NewCorpusTransition,
    ] {
        input(INPUT_SCHEMA_V2, Some(mode))
            .validate_version()
            .unwrap();
    }
    // Non-training v1 inputs are unchanged.
    InputManifest {
        operation: Operation::Verify,
        ..undeclared()
    }
    .validate_version()
    .unwrap();
}

#[test]
fn declaring_refuses_anything_but_an_undeclared_training_input_and_a_known_version() {
    let mode = native_mode();
    for (case, existing) in [
        (
            "unknown frozen version",
            br#"{"schema":"jope.smartcluster_input.v3"}"#.as_slice(),
        ),
        ("unreadable frozen input", b"not json".as_slice()),
        ("frozen input without a schema", b"{}".as_slice()),
    ] {
        assert!(
            undeclared().declare_training(Some(existing), mode).is_err(),
            "{case}: accepted"
        );
    }
    // Only the undeclared v1 training form is declared.
    assert!(input(INPUT_SCHEMA_V2, Some(mode))
        .declare_training(None, mode)
        .is_err());
    assert!(input(INPUT_SCHEMA_V1, Some(mode))
        .declare_training(None, mode)
        .is_err());
    assert!(InputManifest {
        operation: Operation::Verify,
        ..undeclared()
    }
    .declare_training(None, mode)
    .is_err());
    // An exact-resume declaration is never frozen.
    assert!(undeclared()
        .declare_training(None, ContinuationMode::ExactResume)
        .is_err());
}
