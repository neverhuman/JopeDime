#[test]
fn q1_all_failed_attempts_commit_terminal_replayable_actual_counts() {
    let (directory, service) = q1_service_fixture();
    let request = q1_request(directory.path(), &service.store);
    let bytes = service
        .q1_evaluate_using(request.clone(), || Ok(Q1AllFailed))
        .unwrap();
    let receipt: EvaluationReceipt = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(receipt.charged, 24);
    assert_eq!(receipt.attempts.len(), 24);
    assert!(receipt.champion_program.is_none());
    assert!(receipt
        .attempts
        .iter()
        .enumerate()
        .all(|(number, a)| a.attempt == number as u64
            && a.outcome == EvaluationAttemptOutcome::Failed
            && a.failure.is_some()
            && a.loss.is_none()
            && a.fold_scores.is_empty()));
    let command = service
        .store
        .get(&q1_command_key(&receipt.request_sha256))
        .unwrap()
        .unwrap();
    assert_eq!(command.data["status"], "completed");
    let run: jope_api::ids::RunId = command.data["run_id"].as_str().unwrap().parse().unwrap();
    let snapshot = service.journal.snapshot(&run).unwrap().unwrap();
    assert_eq!(snapshot.state, RunState::Failed);
    assert!(snapshot.pending.is_none());
    assert_eq!(snapshot.budget_used.evaluations.0, 24);
    let replay = service
        .q1_evaluate_using(request, || -> anyhow::Result<Q1AllFailed> {
            panic!("terminal replay must never invoke an executor factory")
        })
        .unwrap();
    assert_eq!(bytes, replay);
}
#[test]
fn q1_deterministic_search_stop_commits_only_actual_attempts() {
    let (directory, service) = q1_service_fixture();
    let request_value = q1_request(directory.path(), &service.store);
    let request: Q1Request = serde_json::from_value(request_value).unwrap();
    let manifest = load_q1_manifest(directory.path(), &request.pins["protocol_sha256"]).unwrap();
    let dataset_binding_sha256 = manifest.datasets[&request.dataset.id]
        .dataset_binding_sha256
        .clone();
    let run_id = jope_api::ids::RunId::new();
    assert!(service
        .reserve_q1_request(&request, &run_id)
        .unwrap()
        .is_none());
    let attempt = jope_api::EvaluationAttempt {
        attempt: 0,
        outcome: EvaluationAttemptOutcome::Failed,
        fold_scores: vec![],
        loss: None,
        program_sha256: None,
        failure: Some(jope_api::EvaluationFailure {
            code: "non_finite".into(),
            reason: "actual evaluator attempt failed".into(),
        }),
    };
    let receipt = EvaluationReceipt {
        schema: "jope.q1-evaluation-receipt/v1".into(),
        request_sha256: request.request_sha256.clone(),
        identity: EvaluationIdentity {
            source_commit: request.source_commit.clone(),
            plan_sha256: request.plan_sha256.clone(),
            pins: request.pins.clone(),
            dataset_id: request.dataset.id.clone(),
            dataset_sha256: request.dataset.dataset_sha256.clone(),
            cut_sha256: request.dataset.cut_sha256.clone(),
            guidance: request.guidance.clone(),
            metric: request.metric.clone(),
            seed: request.seed,
            requested_budget: request.budget,
            dataset_binding_sha256,
        },
        charged: 1,
        attempts: vec![attempt],
        champion_program: None,
        adapter: "test".into(),
    };
    let bytes = serde_json::to_vec(&receipt).unwrap();
    service
        .commit_q1_receipt(
            &request,
            &run_id,
            receipt.clone(),
            bytes.clone(),
            Q1ReceiptCompletion {
                attempt_events: vec![],
                elapsed: std::time::Duration::from_millis(5),
                terminal_failure: Some("synthetic deterministic discovery stop".into()),
            },
        )
        .unwrap();

    let snapshot = service.journal.snapshot(&run_id).unwrap().unwrap();
    assert_eq!(snapshot.state, RunState::Failed);
    assert_eq!(snapshot.pending, None);
    assert_eq!(snapshot.budget_used.evaluations.0, 1);
    assert_eq!(snapshot.evaluation_receipt, Some(receipt));
    let command = service
        .store
        .get(&q1_command_key(&request.request_sha256))
        .unwrap()
        .unwrap();
    assert_eq!(command.data["status"], "completed");
    assert_eq!(
        command.data["terminal_failure"],
        "synthetic deterministic discovery stop"
    );
    assert_eq!(
        command.data["receipt_bytes"],
        String::from_utf8(bytes.clone()).unwrap()
    );
    assert_eq!(q1_replay_or_unknown(command.data).unwrap(), Some(bytes));
}
