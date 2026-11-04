use super::run_support::needed;
use super::runner::{may_submit, normal};
use super::*;
#[test]
fn interrupted_export_reenters_pool_recovery_without_retrying_terminal_results() {
    assert!(may_submit(&json!("unattempted")));
    assert!(may_submit(&json!("unknown_started_without_finished")));
    for status in [
        "qualified",
        "failed",
        "failed_parity",
        "unavailable",
        "unsupported",
    ] {
        assert!(!may_submit(&json!(status)));
    }
}

#[test]
fn export_child_closure_omits_reference_predictions_and_targets() {
    let value = serde_json::to_value(WorkerInput {
        schema: "jope.fitted-export-worker/v1".into(),
        files: BTreeMap::new(),
        commands: vec![vec!["python3".into(), "qualification.py".into()]],
        runtimes: BTreeMap::new(),
    })
    .unwrap();
    assert_eq!(value.as_object().unwrap().len(), 4);
    assert!(value.get("expected").is_none());
    assert!(value.get("campaign").is_none());
}
#[test]
fn export_generated_filenames_cannot_escape_or_be_nested() {
    for name in ["../model.py", "/model.py", "a/b.py", ".", ""] {
        assert!(!normal(name));
    }
    assert!(normal("model.py"));
}
#[test]
fn generated_executable_is_not_a_toolchain_dependency() {
    assert_eq!(
        needed(&[
            vec!["rustc".into(), "qualification.rs".into()],
            vec!["./qualification".into()]
        ])
        .collect::<Vec<_>>(),
        vec!["rustc"]
    );
}
