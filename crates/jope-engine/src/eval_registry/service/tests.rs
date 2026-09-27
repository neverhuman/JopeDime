use super::*;
use std::sync::Barrier;

fn inputs(alpha: u32) -> ServiceInputs {
    ServiceInputs {
        program: format!("(pipe X (ridge alpha={alpha}))"),
        x: vec![vec![1.], vec![2.]],
        y: vec![2., 3.],
        preprocessing: b"preprocess".to_vec(),
        feature_view: b"raw".to_vec(),
    }
}
fn provenance() -> RequestProvenance {
    RequestProvenance {
        scope: "synthetic".into(),
        task_id: "synthetic-service-task".into(),
        details: json!({"synthetic_only":true,"deadline_monotonic_ns":request_monotonic_ns().unwrap()+120_000_000_000u64}),
    }
}
struct Fixture {
    temp: tempfile::TempDir,
    _socket_custody: Option<tempfile::TempDir>,
    root: PathBuf,
    witness: PathBuf,
    socket: PathBuf,
    config: ServiceConfig,
}
impl Fixture {
    fn new() -> Self {
        Self::new_at(None)
    }
    fn new_at(evidence: Option<&Path>) -> Self {
        let mut temp = if let Some(evidence) = evidence {
            fs::create_dir_all(evidence).unwrap();
            tempfile::tempdir_in(evidence).unwrap()
        } else {
            tempfile::tempdir().unwrap()
        };
        if evidence.is_some() {
            temp.disable_cleanup(true);
            println!(
                "retained original synthetic query fixture: {}",
                temp.path().display()
            );
        }
        let root = temp.path().join("authority");
        let witness = temp.path().join("witness");
        // Retained evidence paths can exceed sockaddr_un's pathname limit.
        // Socket custody remains machine-local and is cleaned independently.
        let socket_custody = evidence.map(|_| tempfile::tempdir().unwrap());
        let socket = socket_custody
            .as_ref()
            .map_or(temp.path(), |dir| dir.path())
            .join("service.sock");
        let i = inputs(0);
        // Explicit test-only coverage and an unrelated original synthetic model;
        // no numerical fit is used to prepare an eligible service fixture.
        super::super::synthetic_completion_for_test(
            &root,
            &witness,
            "(sym raw \"col(0)\")",
            i.x,
            i.y,
            i.preprocessing,
            i.feature_view,
            0,
        )
        .unwrap();
        let mut config = ServiceConfig::regression(
            root.clone(),
            witness.clone(),
            socket.clone(),
            std::env::current_exe().unwrap(),
            104,
        );
        config.worker_arguments = vec![
            "--exact".into(),
            "eval_registry::service::tests::synthetic_worker".into(),
            "--ignored".into(),
            "--nocapture".into(),
        ];
        config.query_worker_arguments = vec![
            "--exact".into(),
            "eval_registry::service::tests::synthetic_query_worker".into(),
            "--ignored".into(),
            "--nocapture".into(),
        ];
        config.handshake_timeout = Duration::from_secs(60);
        Self {
            temp,
            _socket_custody: socket_custody,
            root,
            witness,
            socket,
            config,
        }
    }
    fn launch(
        &self,
    ) -> (
        Arc<AtomicBool>,
        std::thread::JoinHandle<Result<ServiceStats>>,
    ) {
        let service = Service::bind(self.config.clone()).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let signal = stop.clone();
        (stop, std::thread::spawn(move || service.run(signal)))
    }
}

#[test]
fn invalid_cpu_evaluator_binding_rejects_before_opening_authority() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("unopened-authority");
    let witness = temp.path().join("unopened-witness");
    let socket = temp.path().join("unopened.sock");
    let other = temp.path().join("other-evaluator");
    fs::write(&other, b"#!/bin/sh\nexit 99\n").unwrap();
    fs::set_permissions(&other, fs::Permissions::from_mode(0o700)).unwrap();
    for (path, expected) in [
        (
            PathBuf::from("relative-evaluator"),
            "CPU evaluator binding must be an absolute path",
        ),
        (
            temp.path().to_path_buf(),
            "CPU evaluator binding must name an executable regular file",
        ),
        (
            other,
            "service and explicitly bound CPU evaluator executable differ",
        ),
    ] {
        let result = Service::bind(ServiceConfig::regression(
            root.clone(),
            witness.clone(),
            socket.clone(),
            path,
            1,
        ));
        let error = result.err().expect("invalid evaluator binding accepted");
        let expected = if cfg!(feature = "neural") {
            "CPU fit authority requires a non-neural evaluator build"
        } else {
            expected
        };
        assert!(error.to_string().contains(expected), "{error:#}");
        assert!(!root.exists());
        assert!(!witness.exists());
        assert!(!socket.exists());
    }
}

#[test]
#[ignore = "internal synthetic service-owned executor; never runs numerical fits"]
fn synthetic_worker() {
    let (fit, permit, output) = await_start_from_environment().unwrap();
    permit.verify().unwrap();
    let program = &fit.identity().canonical_program;
    if program.contains("alpha=2") {
        std::process::exit(9);
    }
    if program.contains("alpha=3") {
        unsafe {
            libc::raise(libc::SIGTERM);
        }
    }
    if program.contains("alpha=4") {
        std::thread::sleep(Duration::from_secs(30));
    }
    if program.contains("alpha=5") {
        let old = permit.permission.custody_path.with_extension("old");
        fs::rename(&permit.permission.custody_path, &old).unwrap();
        fs::write(&permit.permission.custody_path, b"replacement").unwrap();
        assert!(permit.verify().is_err());
        std::process::exit(11);
    }
    let model = feat_math::Model::Symbolic {
        expr: feat_math::math_expr::MathExpr::parse("col(0)").unwrap(),
        a: 1.,
        b: 0.,
    };
    write_new(
        &output,
        &serde_json::to_vec(&json!({"status":"success","model":model,"synthetic_only":true}))
            .unwrap(),
    )
    .unwrap();
    permit.verify().unwrap();
}

/// The original service observes a worker that serializes a directly
/// constructed symbolic model. This fixture never calls numerical fitting.
fn retained_query_fixture() -> (Fixture, query::QueryInputs) {
    let evidence = std::env::var_os("JOPE_QUERY_TEST_EVIDENCE").map(PathBuf::from);
    let mut fixture = Fixture::new_at(evidence.as_deref());
    let mut fit_inputs = inputs(0);
    fit_inputs.preprocessing = serde_json::to_vec(&crate::prepare::Preprocessing {
        schema: "parallel.preprocessing.v1".into(),
        columns: vec![crate::prepare::Column {
            kind: "numeric".into(),
            impute: 0.,
            mean: 0.,
            scale: 1.,
            categories: vec![],
        }],
        labels: vec![],
        target_mean: 10.,
        target_scale: 2.,
    })
    .unwrap();
    fit_inputs.feature_view = serde_json::to_vec(&json!({"protocol":"raw_training_fold_v1",
        "training_sha256":sha(b"synthetic training csv"),
        "preprocessing_sha256":sha(&fit_inputs.preprocessing)}))
    .unwrap();
    let (stop, task) = fixture.launch();
    let original = submit_to(
        &fixture.socket,
        &fixture.root,
        &fixture.witness,
        &fit_inputs,
        provenance(),
    )
    .unwrap();
    assert_eq!(original.receipt.outcome, ExecutionOutcome::Success);
    stop.store(true, Ordering::SeqCst);
    assert_eq!(task.join().unwrap().unwrap().worker_spawns, 1);
    let query = query::QueryInputs {
        schema: "parallel.query_inputs.v1".into(),
        completion_sha256: original.completion_sha256,
        scope: "synthetic".into(),
        task_id: "synthetic-query-service-task".into(),
        rows: vec![vec!["3".into()], vec!["4".into()]],
        targets: vec![16., 18.],
        preprocessing: fit_inputs.preprocessing,
        feature_view: fit_inputs.feature_view,
        contract: query::QueryContract::regression_rmse(),
        access_receipt_sha256: sha(b"exact private query capability"),
        deadline_monotonic_ns: request_monotonic_ns().unwrap() + 120_000_000_000,
    };
    fixture
        .config
        .synthetic_query_access
        .insert(query.descriptor_sha256().unwrap());
    (fixture, query)
}

fn query_events(root: &Path) -> Vec<Value> {
    fs::read_to_string(root.join("dispatch.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .filter(|event| event["kind"].as_str().unwrap().starts_with("query_"))
        .collect()
}

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
    assert_eq!(service.state.stats().worker_spawns, 0);
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
    let (fixture, mut inputs) = retained_query_fixture();
    let mut service = Service::bind(fixture.config.clone()).unwrap();
    verified_executable(&ProcessIdentity::current().unwrap()).unwrap();
    // Use the same bounded original window as retained_query_fixture: the first
    // query still starts an isolated child and verifies its runtime. Only after
    // that immutable execution deadline expires do we exercise retrieval.
    inputs.deadline_monotonic_ns = request_monotonic_ns().unwrap() + 120_000_000_000;
    let state = Arc::get_mut(&mut service.state).unwrap();
    state.config.synthetic_query_access.clear();
    state
        .config
        .synthetic_query_access
        .insert(inputs.descriptor_sha256().unwrap());
    let mut changed = inputs.clone();
    changed.deadline_monotonic_ns += 1;
    state
        .config
        .synthetic_query_access
        .insert(changed.descriptor_sha256().unwrap());
    let mut absent = inputs.clone();
    absent.task_id.push_str("-never-accepted");
    state
        .config
        .synthetic_query_access
        .insert(absent.descriptor_sha256().unwrap());
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

#[test]
fn sealed_fd_and_absolute_partial_frame_validation() {
    let file = sealed(b"verified bytes").unwrap();
    assert_eq!(read_sealed(&file, 100).unwrap(), b"verified bytes");
    assert_eq!(unsafe { libc::ftruncate(file.as_raw_fd(), 0) }, -1);
    let temp = tempfile::tempfile().unwrap();
    assert!(read_sealed(&temp, 100).is_err());
    let (mut a, mut b) = UnixStream::pair().unwrap();
    b.write_all(&[0, 0]).unwrap();
    let began = Instant::now();
    assert!(receive_json::<Value>(&mut a, Duration::from_millis(20)).is_err());
    assert!(began.elapsed() < Duration::from_secs(1));
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn service_reuses_one_connection_and_104_clients_start_once() {
    let fixture = Fixture::new();
    let (stop, service) = fixture.launch();
    // Warm the immutable executable authentication cache before contention.
    assert_eq!(
        stats(&fixture.socket, &fixture.root, &fixture.witness)
            .unwrap()
            .authority_opens,
        1
    );
    let barrier = Arc::new(Barrier::new(104));
    let clients: Vec<_> = (0..104)
        .map(|_| {
            let socket = fixture.socket.clone();
            let root = fixture.root.clone();
            let witness = fixture.witness.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                submit_to(&socket, &root, &witness, &inputs(1), provenance())
            })
        })
        .collect();
    let mut completed = Vec::new();
    for client in clients {
        if let Ok(result) = client.join().unwrap() {
            completed.push(result);
        }
    }
    assert!(!completed.is_empty());
    let original = &completed[0];
    assert_eq!(original.receipt.outcome, ExecutionOutcome::Success);
    assert!(original.model_bytes().is_ok());
    let reused = submit_to(
        &fixture.socket,
        &fixture.root,
        &fixture.witness,
        &inputs(1),
        provenance(),
    )
    .unwrap();
    assert!(reused.reused);
    assert_eq!(reused.completion_sha256, original.completion_sha256);
    let verified = verify_original(
        &fixture.socket,
        &fixture.root,
        &fixture.witness,
        &original.completion_sha256,
        "synthetic",
    )
    .unwrap();
    assert_eq!(verified.fit_key, original.fit_key);
    assert!(verify_original(
        &fixture.socket,
        &fixture.root,
        &fixture.witness,
        &original.completion_sha256,
        "sealed_test"
    )
    .is_err());
    stop.store(true, Ordering::SeqCst);
    let stats = service.join().unwrap().unwrap();
    assert_eq!(stats.authority_opens, 1);
    assert_eq!(stats.worker_spawns, 1);
    assert_eq!(stats.committed, 1);
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn lost_submit_response_still_commits_original_without_replay() {
    let fixture = Fixture::new();
    let (stop, service) = fixture.launch();
    let (mut client, _) =
        connect_approved(&fixture.socket, &fixture.root, &fixture.witness).unwrap();
    let bytes = serde_json::to_vec(&inputs(1)).unwrap();
    let fd = sealed(&bytes).unwrap();
    send_json(
        &mut client,
        &Request::Submit {
            provenance: provenance(),
            descriptor_sha256: sha(&bytes),
        },
        Duration::from_secs(5),
    )
    .unwrap();
    send_fd(&client, fd.as_raw_fd(), Duration::from_secs(5)).unwrap();
    drop(client);
    let deadline = Instant::now() + Duration::from_secs(100);
    loop {
        let stats = stats(&fixture.socket, &fixture.root, &fixture.witness).unwrap();
        if stats.committed == 1 {
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(20));
    }
    let reused = submit_to(
        &fixture.socket,
        &fixture.root,
        &fixture.witness,
        &inputs(1),
        provenance(),
    )
    .unwrap();
    assert!(reused.reused);
    stop.store(true, Ordering::SeqCst);
    let stats = service.join().unwrap().unwrap();
    assert_eq!(stats.worker_spawns, 1);
    assert_eq!(stats.committed, 1);
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn failed_signaled_timed_out_and_replaced_custody_are_terminal() {
    let mut fixture = Fixture::new();
    // Leave bounded room for debug-binary authentication and custody transfer.
    // The timeout worker deliberately sleeps 30 seconds, so a two-second cap
    // still exercises numerical timeout rather than a pre-consumption refusal.
    fixture.config.fit_deadline = Duration::from_secs(2);
    let (stop, service) = fixture.launch();
    for alpha in [2, 3, 4, 5] {
        let original = submit_to(
            &fixture.socket,
            &fixture.root,
            &fixture.witness,
            &inputs(alpha),
            provenance(),
        )
        .unwrap();
        assert_ne!(original.receipt.outcome, ExecutionOutcome::Success);
        if alpha == 4 {
            assert_eq!(original.receipt.outcome, ExecutionOutcome::Timeout);
        }
        let reused = submit_to(
            &fixture.socket,
            &fixture.root,
            &fixture.witness,
            &inputs(alpha),
            provenance(),
        )
        .unwrap();
        assert!(reused.reused);
        assert_eq!(reused.completion_sha256, original.completion_sha256);
    }
    stop.store(true, Ordering::SeqCst);
    let stats = service.join().unwrap().unwrap();
    assert_eq!(stats.worker_spawns, 4);
    assert_eq!(stats.committed, 4);
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn lost_start_permission_remains_consumed_and_terminal() {
    let mut fixture = Fixture::new();
    fixture.config.drop_start_reply = true;
    let (stop, service) = fixture.launch();
    let result = submit_to(
        &fixture.socket,
        &fixture.root,
        &fixture.witness,
        &inputs(1),
        provenance(),
    )
    .unwrap();
    assert_ne!(result.receipt.outcome, ExecutionOutcome::Success);
    let reused = submit_to(
        &fixture.socket,
        &fixture.root,
        &fixture.witness,
        &inputs(1),
        provenance(),
    )
    .unwrap();
    assert!(reused.reused);
    stop.store(true, Ordering::SeqCst);
    assert_eq!(service.join().unwrap().unwrap().worker_spawns, 1);
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn missing_authority_and_replaced_socket_cannot_start_service() {
    let fixture = Fixture::new();
    let (stop, service) = fixture.launch();
    assert!(Service::bind(fixture.config.clone()).is_err());
    let old = fixture.socket.with_extension("old");
    fs::rename(&fixture.socket, &old).unwrap();
    let _replacement = UnixListener::bind(&fixture.socket).unwrap();
    assert!(stats(&fixture.socket, &fixture.root, &fixture.witness).is_err());
    stop.store(true, Ordering::SeqCst);
    assert!(service.join().unwrap().is_err());
    let absent = fixture.temp.path().join("missing-root");
    let mut config = fixture.config.clone();
    config.authority_root = absent.clone();
    assert!(Service::bind(config).is_err());
    assert!(!absent.exists());
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn forged_executor_ticket_does_not_consume_any_fit() {
    let fixture = Fixture::new();
    let (stop, service) = fixture.launch();
    let (mut client, _) =
        connect_approved(&fixture.socket, &fixture.root, &fixture.witness).unwrap();
    send_json(
        &mut client,
        &Request::Consume {
            ticket: "forged".into(),
            fit_key: "made-up".into(),
        },
        Duration::from_secs(5),
    )
    .unwrap();
    let response: Value = receive_json(&mut client, Duration::from_secs(5)).unwrap();
    assert_eq!(response["ok"], false);
    stop.store(true, Ordering::SeqCst);
    let stats = service.join().unwrap().unwrap();
    assert_eq!(stats.worker_spawns, 0);
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn backlog_connect_and_invalid_request_deadlines_are_bounded() {
    let fixture = Fixture::new();
    let busy_path = fixture.temp.path().join("busy.sock");
    let busy = UnixListener::bind(&busy_path).unwrap();
    assert_eq!(unsafe { libc::listen(busy.as_raw_fd(), 0) }, 0);
    let _queued = connect_bounded(&busy_path, Duration::from_millis(100)).unwrap();
    let began = Instant::now();
    assert!(connect_bounded(&busy_path, Duration::from_millis(25)).is_err());
    assert!(began.elapsed() < Duration::from_secs(1));
    let (stop, service) = fixture.launch();
    for deadline in [Value::Null, json!(0), json!("123"), json!(-1)] {
        let mut request = provenance();
        request.details["deadline_monotonic_ns"] = deadline;
        assert!(submit_to(
            &fixture.socket,
            &fixture.root,
            &fixture.witness,
            &inputs(1),
            request
        )
        .is_err());
    }
    stop.store(true, Ordering::SeqCst);
    let result = service.join().unwrap().unwrap();
    assert_eq!(result.worker_spawns, 0);
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn socket_loss_during_worker_readiness_fences_authorization_and_restart() {
    let mut fixture = Fixture::new();
    let pause = Arc::new(TestPause::default());
    fixture.config.before_authorize = Some(pause.clone());
    let (stop, service) = fixture.launch();
    let socket = fixture.socket.clone();
    let root = fixture.root.clone();
    let witness = fixture.witness.clone();
    let client =
        std::thread::spawn(move || submit_to(&socket, &root, &witness, &inputs(1), provenance()));
    let deadline = Instant::now() + Duration::from_secs(100);
    while !pause.reached.load(Ordering::SeqCst) {
        assert!(
            Instant::now() < deadline,
            "original worker never became ready"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    fs::rename(&fixture.socket, fixture.socket.with_extension("old")).unwrap();
    let replacement = UnixListener::bind(&fixture.socket).unwrap();
    stop.store(true, Ordering::SeqCst);
    pause.release.store(true, Ordering::SeqCst);
    assert!(client.join().unwrap().is_err());
    assert!(service.join().unwrap().is_err());
    let db = rusqlite::Connection::open(fixture.root.join("registry.sqlite")).unwrap();
    let counts: (i64,i64) = db.query_row("SELECT COUNT(*),SUM(CASE WHEN state='claimed' THEN 1 ELSE 0 END) FROM entries WHERE state!='completed'", [], |row| Ok((row.get(0)?,row.get(1)?))).unwrap();
    assert_eq!(counts, (1, 1));
    drop(db);
    drop(replacement);
    fs::remove_file(&fixture.socket).unwrap();
    fs::remove_file(identity_path(&fixture.socket)).unwrap();
    assert!(Service::bind(fixture.config.clone()).is_err());
}

#[test]
#[ignore = "internal retained-model prediction child; no fit permission"]
fn synthetic_query_worker() {
    execute_query_worker().unwrap();
}
