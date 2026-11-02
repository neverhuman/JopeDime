#[test]
fn q1_cut_hash_matches_canonical_cross_language_control() {
    let binding = Q1DatasetBinding {
        id: "0000000000000001".into(),
        task: "regression".into(),
        dataset_sha256: "a".repeat(64),
        family: "f".repeat(40),
        family_authoritative: true,
        rows: 4,
        features: 2,
        cut_sha256: String::new(),
        holdout_group: "development_fit".into(),
        catalog_version: "v1".into(),
        catalog_record_sha256: "c".repeat(64),
        dataset_binding_sha256: String::new(),
        target: "y".into(),
        feature_columns: vec!["x1".into(), "x2".into()],
        train_rows: 4,
        train_blob_sha256: "b".repeat(64),
        train_blob_bytes: 1,
        folds: vec![
            Q1Fold {
                repeat: 0,
                train: vec![0, 1],
                validation: vec![2, 3],
            },
            Q1Fold {
                repeat: 0,
                train: vec![2, 3],
                validation: vec![0, 1],
            },
        ],
    };
    assert_eq!(
        q1_cut_sha256(&binding).unwrap(),
        "0c87e715ccb1a01fe05d4bc4e8b81f62f978d8aed0126248d97b866686acf249"
    );
    let identity = Q1DatasetIdentity {
        id: binding.id.clone(),
        task: binding.task.clone(),
        dataset_sha256: binding.dataset_sha256.clone(),
        family: binding.family.clone(),
        family_authoritative: binding.family_authoritative,
        rows: binding.rows,
        features: binding.features,
        cut_sha256: binding.cut_sha256.clone(),
        holdout_group: binding.holdout_group.clone(),
    };
    let mut mismatched = binding;
    mismatched.cut_sha256 = "d".repeat(64);
    let mut binding_value = serde_json::to_value(&mismatched).unwrap();
    binding_value
        .as_object_mut()
        .unwrap()
        .remove("dataset_binding_sha256");
    mismatched.dataset_binding_sha256 = hash(&serde_json::to_vec(&binding_value).unwrap());
    assert!(validate_q1_binding(&mismatched, &identity).is_err());
}
#[test]
#[ignore = "Q1 worker subprocess target; invoked by the worker process supervisor"]
fn q1_worker_helper() {
    use std::io::Write as _;

    assert_eq!(std::env::var("JOPE_WORKER_FIXTURE").unwrap(), "worker");
    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "WORKER_FIXTURE_READY").unwrap();
    stdout.flush().unwrap();
    drop(stdout);
    crate::worker::serve(1, |request, matrix| {
        let request_seed = request
            .seed
            .wrapping_sub(request.attempt.attempt.saturating_sub(1));
        if request_seed == Q1_FAIL_ALL_TEST_SEED {
            Err(crate::worker::WorkerFailure::InvalidRequest)
        } else {
            crate::discover::r#loop::evaluate_native(request, matrix)
        }
    })
    .unwrap();
}
#[test]
fn learned_seed_schedule_cycles_until_successful_measurements_exist() {
    let seeds = vec![
        "seed-a".to_owned(),
        "seed-b".to_owned(),
        "seed-c".to_owned(),
    ];
    let scheduled = (0..24)
        .map(|attempt| q1_learned_seed(&seeds, attempt, &[]))
        .collect::<Vec<_>>();
    assert_eq!(scheduled.len(), 24);
    assert!(scheduled
        .iter()
        .enumerate()
        .all(|(attempt, seed)| *seed == seeds[attempt % seeds.len()]));
    let measured = vec![("seed-b".to_owned(), 1.0)];
    assert_eq!(q1_learned_seed(&seeds, 1, &measured), "seed-c");
}
#[test]
fn q1_attempt_timeout_is_bounded_by_one_total_wall_budget() {
    assert_eq!(
        q1_remaining_wall_ms(std::time::Duration::ZERO),
        Some(600_000)
    );
    assert_eq!(
        q1_remaining_wall_ms(std::time::Duration::from_millis(1250)),
        Some(598_750)
    );
    assert_eq!(
        q1_remaining_wall_ms(Q1_WALL_BUDGET - std::time::Duration::from_millis(1)),
        Some(1)
    );
    assert_eq!(q1_remaining_wall_ms(Q1_WALL_BUDGET), None);
    assert_eq!(
        q1_remaining_wall_ms(Q1_WALL_BUDGET + std::time::Duration::from_secs(1)),
        None
    );
}
#[test]
fn q1_expired_policy_never_invokes_the_evaluator() {
    use crate::discover::r#loop::Executor;
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let executor = Q1PinnedExecutor {
        inner: InlineQ1Native {
            calls: Arc::clone(&calls),
        },
        folds: json!({}),
        started_at: std::time::Instant::now() - Q1_WALL_BUDGET,
        attempts: Mutex::new(vec![]),
    };
    let request = crate::worker::Request {
        attempt: crate::worker::Attempt {
            run: jope_api::ids::RunId::new(),
            round: 0,
            slot: 0,
            attempt: 1,
        },
        matrix_path: PathBuf::new(),
        fold_spec: json!({}),
        dsl: "(pipe X (ridge))".into(),
        objective: json!("rmse"),
        seed: 0,
        limits: crate::worker::Limits {
            wall_ms: 600_000,
            rss_bytes: WORKER_RAM,
        },
    };
    let response = executor.evaluate(&request);
    assert_eq!(
        response.failure,
        Some(crate::worker::WorkerFailure::Timeout)
    );
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 0);
    let attempts = executor.attempts.into_inner().unwrap();
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].1.outcome, EvaluationAttemptOutcome::Failed);
    assert!(attempts[0].1.loss.is_none() && attempts[0].1.fold_scores.is_empty());
}
#[test]
fn q1_development_fit_scope_keeps_request_identity_validation() {
    let (directory, service) = q1_service_fixture();
    let mut request = q1_request(directory.path(), &service.store);
    request["dataset"]["holdout_group"] = json!("development_fit");
    request.as_object_mut().unwrap().remove("request_sha256");
    request["request_sha256"] = json!(canonical_sha256(&request).unwrap());
    let parsed: Q1Request = serde_json::from_value(request.clone()).unwrap();
    validate_q1_request(&parsed, &request).unwrap();
    request["dataset"]["holdout_group"] = json!("unapproved_group");
    request.as_object_mut().unwrap().remove("request_sha256");
    request["request_sha256"] = json!(canonical_sha256(&request).unwrap());
    let parsed: Q1Request = serde_json::from_value(request.clone()).unwrap();
    assert!(validate_q1_request(&parsed, &request).is_err());
}
#[test]
fn learned_q1_plan_and_folds_are_derived_from_the_pinned_manifest() {
    let (directory, engine, _) = fixture();
    let raw = q1_request(directory.path(), engine.store());
    let request: Q1Request = serde_json::from_value(raw.clone()).unwrap();
    let manifest = load_q1_manifest(
        directory.path(),
        request.pins.get("protocol_sha256").unwrap(),
    )
    .unwrap();
    let binding = manifest.datasets.get(&request.dataset.id).unwrap();

    let run_spec =
        q1_run_spec(&request_with_guidance(request.clone(), "learned"), binding).unwrap();
    assert_eq!(run_spec.guidance, Guidance::Learned);
    assert_eq!(run_spec.landmarks, LandmarkMode::Full);
    assert_eq!(run_spec.device, "cpu");
    validate_q1_folds(binding).unwrap();

    let fold_spec = q1_fold_spec(binding, &"a".repeat(64), true);
    assert_eq!(fold_spec["version"], 2);
    assert_eq!(fold_spec["target"], binding.target);
    assert_eq!(fold_spec["features"], json!(binding.feature_columns));
    assert_eq!(fold_spec["folds"], json!(binding.folds));
    assert!(fold_spec.get("refit_output").is_none());

    let mut caller_controlled = raw;
    caller_controlled["target"] = json!("attacker_target");
    let mut unsigned = caller_controlled.clone();
    unsigned.as_object_mut().unwrap().remove("request_sha256");
    caller_controlled["request_sha256"] = json!(canonical_sha256(&unsigned).unwrap());
    assert!(serde_json::from_value::<Q1Request>(caller_controlled).is_err());
}
