use jope_cli::research_view::{evaluate_receipts, ExecutionEvidence, Freshness, MachineFailure};
use serde_json::json;
use std::process::ExitCode;

#[test]
fn unavailable_status_has_a_stable_machine_reason_and_nonzero_code() {
    let failure = evaluate_receipts(&[], "2026-10-18T00:00:00Z", None).unwrap_err();

    assert_eq!(
        failure,
        MachineFailure::unavailable("receipt_backed_status")
    );
    assert_eq!(failure.exit_code, 2);
    assert_eq!(failure.exit_status(), ExitCode::from(2));
    assert_eq!(failure.reason, "capability_unavailable");
    assert_eq!(
        serde_json::to_value(failure).unwrap(),
        json!({
            "ok": false,
            "exit_code": 2,
            "reason": "capability_unavailable",
            "capability": "receipt_backed_status"
        })
    );
}

#[test]
fn configured_command_is_not_reported_as_executed_proof() {
    let receipt = json!({
        "task_id": "I61",
        "status": "configured",
        "verification": {"command": "cargo test --locked -p jope-cli"}
    });
    let report = evaluate_receipts(&[receipt], "2026-10-18T00:00:00Z", None).unwrap();

    assert_eq!(
        report.receipts[0].execution_evidence,
        ExecutionEvidence::ConfiguredOnly
    );
}

#[test]
fn stale_receipt_keeps_its_original_date_and_source_identity() {
    let receipt = json!({
        "task_id": "I45",
        "stage": "science",
        "status": "worker_implementation_committed_and_pushed_pending_integration",
        "head": "0123456789abcdef0123456789abcdef01234567",
        "verification": {"recorded_at_utc": "2026-10-16T12:00:00.000Z"}
    });
    let report = evaluate_receipts(
        &[receipt],
        "2026-10-18T00:00:00Z",
        Some("2026-10-16T12:00:00.000500Z"),
    )
    .unwrap();
    let status = &report.receipts[0];

    assert_eq!(status.freshness, Freshness::Dated);
    assert_eq!(status.stage.as_deref(), Some("science"));
    assert_eq!(
        status.recorded_at_utc.as_deref(),
        Some("2026-10-16T12:00:00.000Z")
    );
    assert_eq!(
        status.source_sha.as_deref(),
        Some("0123456789abcdef0123456789abcdef01234567")
    );
}
