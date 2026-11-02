use super::*;

fn payload(seed: u64) -> DimeScientificPayloadV1 {
    DimeScientificPayloadV1 {
        parent_intent_sha256: digest_label("parent"),
        runtime_bundle_sha256: digest_label("runtime"),
        binary_sha256: digest_label("binary"),
        configuration_sha256: digest_label("config"),
        train_data_sha256: digest_label("train"),
        protocol_sha256: digest_label("protocol"),
        metric_sha256: digest_label("metric"),
        seed,
        inputs: vec![DimeNamedDigestV1 {
            name: "frozen".to_owned(),
            sha256: digest_label("frozen"),
        }],
    }
}

fn receipt(attempt_number: u64, prior: Option<String>) -> DimeInfrastructureAttemptV1 {
    DimeInfrastructureAttemptV1::new_terminal(
        DimeRoleV1::CohortProposer,
        payload(101),
        attempt_number,
        format!("proposer-attempt-{attempt_number}"),
        Some(format!("proposer-attempt-{}", attempt_number + 1)),
        prior,
        attempt_number as u128,
        attempt_number as u128 + retry_backoff_ms(attempt_number),
        DimeTerminalEvidenceV1 {
            terminal_state: "cancelled".to_owned(),
            class: DimeTerminalClassV1::InfrastructureFailure,
            evidence_sha256: digest_label(&format!("terminal-{attempt_number}")),
            observed_at_ms: attempt_number as u128,
            foreign_cancellation: true,
        },
    )
    .unwrap()
}

#[test]
fn retry_five_can_append_attempt_six_without_changing_science() {
    let directory = tempfile::tempdir().unwrap();
    let mut prior = None;
    for attempt in 1..=6 {
        let value = receipt(attempt, prior.clone());
        append_infrastructure_attempt(directory.path(), &value).unwrap();
        prior = Some(value.content_sha256.clone());
    }
    let last = latest_attempt_receipt_sha256(
        directory.path(),
        DimeRoleV1::CohortProposer,
        &payload(101).identity_sha256().unwrap(),
    )
    .unwrap();
    assert_eq!(last, prior);
}

#[test]
fn append_is_idempotent_but_number_reuse_and_science_drift_fail() {
    let directory = tempfile::tempdir().unwrap();
    let one = receipt(1, None);
    let path = append_infrastructure_attempt(directory.path(), &one).unwrap();
    assert_eq!(
        append_infrastructure_attempt(directory.path(), &one).unwrap(),
        path
    );
    let mut drift = one.clone();
    drift.terminal.evidence_sha256 = digest_label("different-terminal");
    drift.content_sha256 = drift.computed_content_sha256().unwrap();
    assert!(append_infrastructure_attempt(directory.path(), &drift).is_err());
    let mut changed_science = payload(101);
    changed_science.seed = 202;
    assert!(ensure_same_scientific_payload(&payload(101), &changed_science).is_err());
}

#[test]
fn only_an_unexpired_exact_drain_suppresses_recovery() {
    let directory = tempfile::tempdir().unwrap();
    let payload = payload(101);
    let intent = DimeDrainIntentV1::new(
        DimeRoleV1::CohortProposer,
        payload.identity_sha256().unwrap(),
        payload.parent_intent_sha256.clone(),
        1_000,
        2_000,
        "dime-supervisor".to_owned(),
        "planned maintenance".to_owned(),
    )
    .unwrap();
    write_drain_intent(directory.path(), &intent).unwrap();
    assert!(matching_drain_intent(
        directory.path(),
        DimeRoleV1::CohortProposer,
        &payload,
        1_500
    )
    .unwrap()
    .is_some());
    assert!(matching_drain_intent(
        directory.path(),
        DimeRoleV1::CohortProposer,
        &payload,
        2_000
    )
    .unwrap()
    .is_none());
    assert!(matching_drain_intent(
        directory.path(),
        DimeRoleV1::AlternatingTrainer,
        &payload,
        1_500
    )
    .unwrap()
    .is_none());
}

#[test]
fn infrastructure_backoff_is_exact_and_then_caps_at_fifteen_minutes() {
    assert_eq!(
        (1..=7).map(retry_backoff_ms).collect::<Vec<_>>(),
        vec![30_000, 60_000, 120_000, 240_000, 900_000, 900_000, 900_000]
    );
}

#[test]
fn cancellation_ignores_even_a_stray_scientific_artifact() {
    assert_eq!(
        terminal_classification("cancelled", Some(&digest_label("stray"))).unwrap(),
        DimeTerminalClassV1::InfrastructureFailure
    );
    assert_eq!(
        terminal_classification("failed", Some(&digest_label("bound"))).unwrap(),
        DimeTerminalClassV1::ScientificFailure
    );
}

#[test]
fn receipt_append_crash_rebuilds_the_recorded_next_identity_without_duplication() {
    let directory = tempfile::tempdir().unwrap();
    let payload = payload(101);
    let receipt = receipt(1, None);
    append_infrastructure_attempt(directory.path(), &receipt).unwrap();

    let rebuilt = load_or_rebuild_recovery_projection(
        directory.path(),
        DimeRoleV1::CohortProposer,
        &payload,
        "proposer-attempt-1",
    )
    .unwrap();
    assert_eq!(rebuilt.current_job_identity, "proposer-attempt-2");
    assert_eq!(rebuilt.current_attempt_number, 2);
    assert_eq!(rebuilt.retry_not_before_ms, receipt.retry_not_before_ms);
    assert_eq!(
        rebuilt.last_attempt_receipt_sha256,
        Some(receipt.content_sha256.clone())
    );

    let replayed = load_or_rebuild_recovery_projection(
        directory.path(),
        DimeRoleV1::CohortProposer,
        &payload,
        "proposer-attempt-1",
    )
    .unwrap();
    assert_eq!(replayed, rebuilt);
    assert_eq!(
        infrastructure_attempts(
            directory.path(),
            DimeRoleV1::CohortProposer,
            &payload.identity_sha256().unwrap(),
        )
        .unwrap()
        .len(),
        1
    );
}

#[test]
fn three_foreign_cancellations_open_contention_and_force_half_open_cap() {
    let mut projection = DimeRecoveryProjectionV1::initial(
        DimeRoleV1::CohortProposer,
        payload(101),
        "proposer-attempt-1".to_owned(),
    )
    .unwrap();
    let mut prior = None;
    for attempt in 1..=3 {
        let observed = attempt as u128 * 60_000;
        let value = DimeInfrastructureAttemptV1::new_terminal(
            DimeRoleV1::CohortProposer,
            payload(101),
            attempt,
            format!("proposer-attempt-{attempt}"),
            Some(format!("proposer-attempt-{}", attempt + 1)),
            prior.clone(),
            observed,
            observed + retry_backoff_ms(attempt),
            DimeTerminalEvidenceV1 {
                terminal_state: "cancelled".to_owned(),
                class: DimeTerminalClassV1::InfrastructureFailure,
                evidence_sha256: digest_label(&format!("cancelled-{attempt}")),
                observed_at_ms: observed,
                foreign_cancellation: true,
            },
        )
        .unwrap();
        projection.apply_receipt(&value).unwrap();
        prior = Some(value.content_sha256);
    }
    assert!(projection.contention_alarm_open);
    assert_eq!(projection.next_backoff_ms(180_000), 15 * 60 * 1_000);
}
