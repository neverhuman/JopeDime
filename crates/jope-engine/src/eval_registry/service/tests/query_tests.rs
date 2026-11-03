use super::*;

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn query_client_deadline_bounds_unserved_authority_without_acceptance() {
    let (fixture, mut inputs) = retained_query_fixture();
    let service = Service::bind(fixture.config.clone()).unwrap();
    // Warm executable verification before timing the intentionally unserved IPC.
    verified_executable(&ProcessIdentity::current().unwrap()).unwrap();
    let before = fs::read(fixture.root.join("dispatch.jsonl")).unwrap();
    inputs.deadline_monotonic_ns = request_monotonic_ns().unwrap() + 100_000_000;
    let started = Instant::now();
    let error =
        query_evaluate(&fixture.socket, &fixture.root, &fixture.witness, &inputs).unwrap_err();
    assert!(started.elapsed() < Duration::from_secs(1), "{error:#}");
    assert!(started.elapsed() >= Duration::from_millis(50), "{error:#}");
    assert!(error.to_string().contains("deadline"), "{error:#}");
    assert_eq!(
        before,
        fs::read(fixture.root.join("dispatch.jsonl")).unwrap()
    );
    // The service is bound but not running, so no worker can be spawned.
    assert!(service.socket().exists());
    assert!(query_events(&fixture.root).is_empty());
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn query_expired_descriptor_rejects_before_access_or_acceptance() {
    let (fixture, mut inputs) = retained_query_fixture();
    let (stop, task) = fixture.launch();
    inputs.deadline_monotonic_ns = request_monotonic_ns().unwrap() - 1;
    let (mut stream, _) =
        connect_approved(&fixture.socket, &fixture.root, &fixture.witness).unwrap();
    let bytes = serde_json::to_vec(&inputs).unwrap();
    let descriptor = sealed(&bytes).unwrap();
    send_json(
        &mut stream,
        &Request::QueryEvaluate {
            descriptor_sha256: sha(&bytes),
        },
        Duration::from_secs(1),
    )
    .unwrap();
    send_fd(&stream, descriptor.as_raw_fd(), Duration::from_secs(1)).unwrap();
    let response: Value = receive_json(&mut stream, Duration::from_secs(1)).unwrap();
    assert_eq!(response["ok"], false);
    assert!(response["error"].as_str().unwrap().contains("deadline"));
    assert!(query_events(&fixture.root).is_empty());
    stop.store(true, Ordering::SeqCst);
    assert_eq!(task.join().unwrap().unwrap().worker_spawns, 0);
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn query_original_service_completion_reuses_after_restart_without_fit_starts() {
    let (fixture, inputs) = retained_query_fixture();
    let (stop, task) = fixture.launch();
    let first = query_evaluate(&fixture.socket, &fixture.root, &fixture.witness, &inputs).unwrap();
    assert!(!first.reused);
    assert_eq!(
        serde_json::from_slice::<Vec<f64>>(&fs::read(&first.predictions_path).unwrap()).unwrap(),
        vec![16., 18.]
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&fs::read(&first.metrics_path).unwrap()).unwrap()["rmse"],
        0.
    );
    let original_events = query_events(&fixture.root);
    assert_eq!(original_events.len(), 2);
    let reused = query_evaluate(&fixture.socket, &fixture.root, &fixture.witness, &inputs).unwrap();
    assert!(reused.reused);
    assert_eq!(first.record.receipt_sha256, reused.record.receipt_sha256);
    assert_eq!(original_events, query_events(&fixture.root));
    stop.store(true, Ordering::SeqCst);
    let result = task.join().unwrap().unwrap();
    assert_eq!(result.worker_spawns, 0);
    assert_eq!(result.committed, 0);
    let (stop, task) = fixture.launch();
    let replay = query_evaluate(&fixture.socket, &fixture.root, &fixture.witness, &inputs).unwrap();
    assert!(replay.reused);
    assert_eq!(first.record.receipt_sha256, replay.record.receipt_sha256);
    assert_eq!(original_events, query_events(&fixture.root));
    stop.store(true, Ordering::SeqCst);
    assert_eq!(task.join().unwrap().unwrap().worker_spawns, 0);
    let restored = Authority::recovery_inspect(&fixture.root, &fixture.witness).unwrap();
    assert!(restored["retained_artifact_errors"]
        .as_array()
        .unwrap()
        .is_empty());
    assert!(restored["blockers"].as_array().unwrap().iter().any(|b| b
        .as_str()
        .unwrap()
        .contains("production activation verifiers unavailable")));
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn query_fitting_scope_does_not_authorize_outcome_access() {
    let (mut fixture, inputs) = retained_query_fixture();
    fixture.config.synthetic_query_access.clear();
    let (stop, task) = fixture.launch();
    let error =
        query_evaluate(&fixture.socket, &fixture.root, &fixture.witness, &inputs).unwrap_err();
    assert!(format!("{error:#}").contains("fitting scope grants no query access"));
    assert!(query_events(&fixture.root).is_empty());
    stop.store(true, Ordering::SeqCst);
    assert_eq!(task.join().unwrap().unwrap().worker_spawns, 0);
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn query_changed_descriptor_cannot_extend_original_deadline_or_replace_result() {
    let (mut fixture, inputs) = retained_query_fixture();
    let mut changed = inputs.clone();
    changed.deadline_monotonic_ns += 1;
    fixture
        .config
        .synthetic_query_access
        .insert(changed.descriptor_sha256().unwrap());
    let (stop, task) = fixture.launch();
    query_evaluate(&fixture.socket, &fixture.root, &fixture.witness, &inputs).unwrap();
    let events = query_events(&fixture.root);
    let error =
        query_evaluate(&fixture.socket, &fixture.root, &fixture.witness, &changed).unwrap_err();
    assert!(format!("{error:#}").contains("descriptor changed"));
    assert_eq!(events, query_events(&fixture.root));
    stop.store(true, Ordering::SeqCst);
    assert_eq!(task.join().unwrap().unwrap().worker_spawns, 0);
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn query_predictor_timeout_preserves_unavailable_acceptance_without_refitting() {
    let (mut fixture, mut inputs) = retained_query_fixture();
    fixture.config.query_worker_sleep_ms = Some(300_000);
    // Includes original service startup/hash verification, then a bounded
    // predictor deadline. No fitted production or synthetic model is trained.
    inputs.deadline_monotonic_ns = request_monotonic_ns().unwrap() + 60_000_000_000;
    fixture.config.synthetic_query_access.clear();
    fixture
        .config
        .synthetic_query_access
        .insert(inputs.descriptor_sha256().unwrap());
    let (stop, task) = fixture.launch();
    let started = Instant::now();
    let error =
        query_evaluate(&fixture.socket, &fixture.root, &fixture.witness, &inputs).unwrap_err();
    assert!(format!("{error:#}").contains("deadline"), "{error:#}");
    assert!(started.elapsed() < Duration::from_secs(65));
    stop.store(true, Ordering::SeqCst);
    assert_eq!(task.join().unwrap().unwrap().worker_spawns, 0);
    let accepted = query_events(&fixture.root);
    assert_eq!(accepted.len(), 1);
    let recovery = Authority::recovery_inspect(&fixture.root, &fixture.witness).unwrap();
    assert_eq!(
        recovery["unresolved_original_queries"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(accepted, query_events(&fixture.root));
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn query_retrieval_preserves_expired_original_and_never_accepts_new_work() {
    let (mut fixture, mut inputs) = retained_query_fixture();
    verified_executable(&ProcessIdentity::current().unwrap()).unwrap();
    // Use the same bounded original window as retained_query_fixture: the first
    // query still starts an isolated child and verifies its runtime. Only after
    // that immutable execution deadline expires do we exercise retrieval.
    inputs.deadline_monotonic_ns = request_monotonic_ns().unwrap() + 120_000_000_000;
    let mut changed = inputs.clone();
    changed.deadline_monotonic_ns += 1;
    let mut absent = inputs.clone();
    absent.task_id.push_str("-never-accepted");
    fixture.config.synthetic_query_access.clear();
    for authorized in [&inputs, &changed, &absent] {
        fixture
            .config
            .synthetic_query_access
            .insert(authorized.descriptor_sha256().unwrap());
    }
    let service = Service::bind(fixture.config.clone()).unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let signal = stop.clone();
    let task = std::thread::spawn(move || service.run(signal));
    let first = query_evaluate(&fixture.socket, &fixture.root, &fixture.witness, &inputs).unwrap();
    let original = query_events(&fixture.root);
    while request_monotonic_ns().unwrap() <= inputs.deadline_monotonic_ns {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(query_evaluate(&fixture.socket, &fixture.root, &fixture.witness, &inputs).is_err());
    let deadline = request_monotonic_ns().unwrap() + 30_000_000_000;
    let retrieved = query_retrieve(
        &fixture.socket,
        &fixture.root,
        &fixture.witness,
        &inputs,
        deadline,
    )
    .unwrap();
    assert!(retrieved.reused);
    assert_eq!(retrieved.record.receipt_sha256, first.record.receipt_sha256);
    assert_eq!(
        retrieved.receipt.deadline_monotonic_ns,
        inputs.deadline_monotonic_ns
    );
    assert!(query_retrieve(
        &fixture.socket,
        &fixture.root,
        &fixture.witness,
        &changed,
        deadline
    )
    .is_err());
    assert!(query_retrieve(
        &fixture.socket,
        &fixture.root,
        &fixture.witness,
        &absent,
        deadline
    )
    .is_err());
    assert!(query_retrieve(&fixture.socket, &fixture.root, &fixture.witness, &inputs, 1).is_err());
    assert_eq!(
        original,
        query_events(&fixture.root),
        "retrieval mutated query history"
    );
    stop.store(true, Ordering::SeqCst);
    assert_eq!(task.join().unwrap().unwrap().worker_spawns, 0);
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn query_ambiguous_acceptance_stays_unavailable_after_restart() {
    let (mut fixture, inputs) = retained_query_fixture();
    fixture.config.query_fail_after_accept = true;
    let (stop, task) = fixture.launch();
    let error =
        query_evaluate(&fixture.socket, &fixture.root, &fixture.witness, &inputs).unwrap_err();
    assert!(format!("{error:#}").contains("failure after durable acceptance"));
    let accepted = query_events(&fixture.root);
    assert_eq!(accepted.len(), 1);
    assert!(query_retrieve(
        &fixture.socket,
        &fixture.root,
        &fixture.witness,
        &inputs,
        request_monotonic_ns().unwrap() + 30_000_000_000
    )
    .is_err());
    assert_eq!(accepted, query_events(&fixture.root));
    stop.store(true, Ordering::SeqCst);
    assert_eq!(task.join().unwrap().unwrap().worker_spawns, 0);
    fixture.config.query_fail_after_accept = false;
    let (stop, task) = fixture.launch();
    let error =
        query_evaluate(&fixture.socket, &fixture.root, &fixture.witness, &inputs).unwrap_err();
    assert!(format!("{error:#}").contains("evaluation cannot repeat"));
    assert_eq!(accepted, query_events(&fixture.root));
    stop.store(true, Ordering::SeqCst);
    assert_eq!(task.join().unwrap().unwrap().worker_spawns, 0);
    let recovery = Authority::recovery_inspect(&fixture.root, &fixture.witness).unwrap();
    assert_eq!(
        recovery["unresolved_original_queries"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn query_missing_original_predictions_blocks_recovery() {
    let (fixture, inputs) = retained_query_fixture();
    let (stop, task) = fixture.launch();
    let result = query_evaluate(&fixture.socket, &fixture.root, &fixture.witness, &inputs).unwrap();
    stop.store(true, Ordering::SeqCst);
    assert_eq!(task.join().unwrap().unwrap().worker_spawns, 0);
    fs::remove_file(&result.predictions_path).unwrap();
    let recovery = Authority::recovery_inspect(&fixture.root, &fixture.witness).unwrap();
    assert!(!recovery["blockers"].as_array().unwrap().is_empty());
    assert!(Authority::recovery_reconcile(&fixture.root, &fixture.witness).is_err());
}
