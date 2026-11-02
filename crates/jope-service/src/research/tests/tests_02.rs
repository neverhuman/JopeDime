use super::*;

#[cfg(unix)]
#[test]
fn budget_truncation_preserves_policy_and_is_not_complete_training_data() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("truncated.jsonl");
    let mut writer = RecordWriter::create(&path).unwrap();
    writer.append(&record(0, true)).unwrap();
    let mut truncated = record(1, true);
    truncated.dispatched.truncate(2);
    truncated.outcomes.truncate(2);
    truncated.capture_complete = false;
    assert_eq!(truncated.policy.admitted.len(), 4);
    writer.append(&truncated).unwrap();
    assert!(!reader(&path, true).status.success());
    let script = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../python/src/jope/north_star/record_reader.py");
    let output = Command::new("python3")
        .arg(script)
        .arg(&path)
        .args([
            "--allow-synthetic",
            "--allow-incomplete",
            "--detailed-counts",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let summary: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        summary,
        json!({
            "rounds":2, "proposals":10, "admissions":6, "dispatches":4,
            "outcomes":4, "successes":2, "failures":2, "incomplete_rounds":1
        })
    );
    truncated.capture_complete = true;
    assert!(truncated.validate().is_err());
}

#[cfg(unix)]
#[test]
fn python_rejects_malformed_or_truncated_streams() {
    let zero = serde_json::to_value(record(0, true)).unwrap();
    let one = serde_json::to_value(record(1, true)).unwrap();
    let line = |v: &Value| format!("{}\n", serde_json::to_string(v).unwrap());
    let mut cases = vec![
        String::new(),
        line(&zero).trim_end().into(),
        format!("{}{}", line(&zero), line(&zero)),
        line(&one),
    ];
    for partition in ["val", "sealed", "holdout", "test"] {
        let mut bad = zero.clone();
        bad["identity"]["partition"] = json!(partition);
        cases.push(line(&bad));
    }
    let mut bad = one.clone();
    bad["outcomes"].as_array_mut().unwrap().remove(0);
    cases.push(format!("{}{}", line(&zero), line(&bad)));
    let mut bad = one.clone();
    bad["seed"] = json!(30);
    cases.push(format!("{}{}", line(&zero), line(&bad)));
    let mut bad = zero.clone();
    bad["tokens"][0][0] = json!(1e39);
    cases.push(line(&bad));
    let mut bad = zero.clone();
    bad["seed"] = json!(false);
    cases.push(line(&bad));
    let mut bad = zero.clone();
    bad["extra"] = json!(0);
    cases.push(line(&bad));
    cases.push(line(&zero).replacen("\"schema\":", "\"seed\":29,\"schema\":", 1));
    cases.push(line(&zero).replacen("0.0", "NaN", 1));
    for (i, bytes) in cases.iter().enumerate() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.jsonl");
        fs::write(&path, bytes).unwrap();
        let output = reader(&path, true);
        assert!(!output.status.success(), "Python negative case {i}");
        assert!(
            output.stdout.is_empty(),
            "no partial success summary: case {i}"
        );
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("encoded.jsonl");
    reject_non_utf8_jsonl(&path, &line(&zero), &line(&one));
}

#[cfg(not(unix))]
#[test]
fn non_unix_private_file_creation_is_explicitly_unavailable() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("records.jsonl");
    match RecordWriter::create(&path) {
        Err(error) => assert!(error.to_string().contains("not qualified")),
        Ok(_) => panic!("unqualified private-file creation unexpectedly succeeded"),
    }
    assert!(!path.exists());
}

#[test]
fn private_capture_envelope_refuses_unknown_fields_and_oversized_cas_references() {
    use capture::{Binding, CaptureState};
    let home = tempfile::tempdir().unwrap();
    let store = crate::store::Store::open(home.path()).unwrap();
    let r = record(0, true);
    let binding = Binding {
        config: CaptureConfig {
            max_attempts: 16,
            max_rounds: 16,
            max_bytes: 1024 * 1024,
            evaluate_all_proposals: true,
            seed_protocol_sha256: "2".repeat(64),
        },
        identity: r.identity,
        seed: r.seed,
        tokens: r.tokens,
        mask: r.mask,
        matrix_sha256: "3".repeat(64),
    };
    let mut value = serde_json::to_value(&binding).unwrap();
    value["arbitrary_import"] = json!(true);
    let reference = store
        .publish_blob(std::io::Cursor::new(serde_json::to_vec(&value).unwrap()))
        .unwrap();
    assert!(CaptureState::read::<Binding>(&store, &reference).is_err());
    let mut too_large = reference;
    too_large.bytes = u64::MAX;
    assert!(CaptureState::read::<Binding>(&store, &too_large)
        .unwrap_err()
        .to_string()
        .contains("ceiling"));
}

#[test]
fn auxiliary_journal_requires_readable_results_and_matching_attempts() {
    use crate::worker::{Attempt, Limits, Request, Response};
    use capture::{AuxiliaryAttempt, AuxiliaryResult, Binding, CaptureState};

    let home = tempfile::tempdir().unwrap();
    let store = crate::store::Store::open(home.path()).unwrap();
    let mut row = record(0, true);
    row.identity.run_id = jope_api::ids::RunId::new().to_string();
    let binding = Binding {
        config: CaptureConfig {
            max_attempts: 16,
            max_rounds: 16,
            max_bytes: 1024 * 1024,
            evaluate_all_proposals: true,
            seed_protocol_sha256: "2".repeat(64),
        },
        identity: row.identity,
        seed: row.seed,
        tokens: row.tokens,
        mask: row.mask,
        matrix_sha256: "3".repeat(64),
    };
    let bytes = serde_json::to_vec(&binding).unwrap();
    let mut state = CaptureState {
        version: 1,
        binding: store.publish_blob(std::io::Cursor::new(&bytes)).unwrap(),
        config: binding.config.clone(),
        charged: 0,
        bytes: bytes.len() as u64,
        label_counter: 0,
        tainted: false,
        rounds: Default::default(),
        auxiliary: vec![],
    };
    let request = Request {
        attempt: Attempt {
            run: binding.identity.run_id.parse().unwrap(),
            round: 0,
            slot: 0,
            attempt: 0,
        },
        matrix_path: "synthetic.jmat".into(),
        fold_spec: serde_json::json!({}),
        dsl: "synthetic_control".into(),
        objective: serde_json::json!("rmse"),
        seed: 29,
        limits: Limits {
            wall_ms: 100,
            rss_bytes: 1024,
        },
    };
    let response = Response {
        attempt: request.attempt.clone(),
        scores: vec![1.0, 1.0, 1.0],
        fold_scores: vec![],
        predictions_digest: "4".repeat(64),
        wall_ms: 2,
        failure: None,
    };
    let request_ref = state.publish(&store, &request).unwrap();
    state.auxiliary.push(AuxiliaryAttempt {
        request: request_ref,
        result: None,
        model: None,
    });
    state.validate_storage(&store).unwrap();
    assert!(
        state.unknown(),
        "pending attempts remain available for inspection"
    );

    let valid = state
        .publish(
            &store,
            &AuxiliaryResult::Evaluation {
                response: response.clone(),
            },
        )
        .unwrap();
    state.auxiliary[0].result = Some(valid);
    state.validate_storage(&store).unwrap();
    assert!(!state.unknown());

    let unreadable = state.publish(&store, &serde_json::Value::Null).unwrap();
    state.auxiliary[0].result = Some(unreadable);
    assert!(
        state.validate_storage(&store).is_err(),
        "valid-hash null is not a known result"
    );

    let mut wrong_attempt = response.clone();
    wrong_attempt.attempt.attempt += 1;
    let reference = state
        .publish(
            &store,
            &AuxiliaryResult::Evaluation {
                response: wrong_attempt,
            },
        )
        .unwrap();
    state.auxiliary[0].result = Some(reference);
    assert!(
        state.validate_storage(&store).is_err(),
        "response must match its reservation"
    );

    for failure in [WorkerFailure::Timeout, WorkerFailure::StaleAttempt] {
        let reference = state
            .publish(
                &store,
                &AuxiliaryResult::Evaluation {
                    response: Response::failed(request.attempt.clone(), failure, 2),
                },
            )
            .unwrap();
        state.auxiliary[0].result = Some(reference);
        assert_eq!(
            state.validate_storage(&store).is_ok(),
            failure == WorkerFailure::Timeout
        );
    }

    for worker in [
        None,
        Some(Response::failed(
            request.attempt.clone(),
            WorkerFailure::Timeout,
            2,
        )),
    ] {
        let reference = state
            .publish(&store, &AuxiliaryResult::Failure { worker })
            .unwrap();
        state.auxiliary[0].result = Some(reference);
        state.validate_storage(&store).unwrap();
    }
    for worker in [
        response.clone(),
        Response::failed(request.attempt.clone(), WorkerFailure::StaleAttempt, 2),
    ] {
        let reference = state
            .publish(
                &store,
                &AuxiliaryResult::Failure {
                    worker: Some(worker),
                },
            )
            .unwrap();
        state.auxiliary[0].result = Some(reference);
        assert!(state.validate_storage(&store).is_err());
    }
    let mut invalid = response.clone();
    invalid.scores.clear();
    let reference = state
        .publish(&store, &AuxiliaryResult::Evaluation { response: invalid })
        .unwrap();
    state.auxiliary[0].result = Some(reference);
    assert!(state.validate_storage(&store).is_err());

    let fit = state.publish(&store, &AuxiliaryResult::Fit).unwrap();
    state.auxiliary[0].result = Some(fit);
    assert!(
        state.validate_storage(&store).is_err(),
        "successful fit requires its model root"
    );
    state.auxiliary[0].model = Some(state.publish(&store, &vec![1_u8, 2, 3]).unwrap());
    state.validate_storage(&store).unwrap();

    let mut foreign = request;
    foreign.attempt.run = jope_api::ids::RunId::new();
    state.auxiliary[0].request = state.publish(&store, &foreign).unwrap();
    assert!(
        state.validate_storage(&store).is_err(),
        "request must belong to the bound run"
    );
}
