#[test]
fn funding_socket_returns_real_verified_start_and_participant_proofs() {
    let fixture = Fixture::new(true);
    let service = fixture.launch();
    fixture.reserve();
    let witness = fixture.original.server(2, false);
    let start = funding_ipc::funding_admission_installed(
        &fixture.installation,
        &fixture.start(),
        request_monotonic_ns().unwrap(),
    )
    .unwrap();
    start.check_current(&fixture.descriptor, None).unwrap();
    // A valid funding proof for this metadata fixture is still not a qualified
    // resident task descriptor and must not become a full GPU execution permit.
    assert!(crate::task_contract::verify_gpu_task_grant(
        &start,
        &fixture.descriptor,
        None,
        "regression_training"
    )
    .is_err());
    assert!(crate::task_contract::verify_gpu_task_grant(
        &start,
        &fixture.descriptor,
        None,
        "unfunded_capability"
    )
    .is_err());
    assert!(fixture.request(&fixture.start()).is_err());
    let rank = funding_ipc::funding_admission_installed(
        &fixture.installation,
        &fixture.participant(0),
        request_monotonic_ns().unwrap(),
    )
    .unwrap();
    rank.check_current(&fixture.descriptor, Some(0)).unwrap();
    assert!(rank.check_current(&fixture.descriptor, Some(1)).is_err());
    assert!(fixture.request(&fixture.participant(0)).is_err());
    assert!(fixture.request(&fixture.participant(1)).is_err());
    assert_eq!(witness.join().unwrap().len(), 2);
    assert_eq!(fixture.count("funding_witness"), 2);
    assert_eq!(fixture.count("funding_participants"), 1);
    let stats = service.finish();
    assert_eq!(stats.worker_spawns, 0);
    assert_eq!(stats.committed, 0);
}

#[test]
fn lost_funding_socket_reply_does_not_renew_original_start() {
    let mut fixture = Fixture::new(true);
    fixture.config.drop_funding_reply = true;
    let service = fixture.launch();
    fixture.reserve();
    let witness = fixture.original.server(1, false);
    assert!(funding_ipc::funding_admission_installed(
        &fixture.installation,
        &fixture.start(),
        request_monotonic_ns().unwrap()
    )
    .is_err());
    assert_eq!(witness.join().unwrap().len(), 1);
    assert_eq!(fixture.count("funding_witness"), 1);
    assert_eq!(fixture.count("funding_reservations"), 1);
    assert_eq!(fixture.count("funding_participants"), 0);
    let error = fixture.request(&fixture.start()).unwrap_err();
    assert!(error.to_string().contains("consumed"), "{error:#}");
    assert_eq!(fixture.count("funding_witness"), 1);
    assert_eq!(service.finish().worker_spawns, 0);
}

#[test]
fn server_rejects_wrong_descriptor_and_actual_runtime_before_start() {
    let fixture = Fixture::new(false);
    let service = fixture.launch();
    fixture.reserve();
    let before = fs::read(fixture.original.authority.join("dispatch.jsonl")).unwrap();
    let mut wrong = fixture.start();
    if let FundingRequest::Start {
        descriptor_json, ..
    } = &mut wrong
    {
        descriptor_json["task"] = json!("substituted task");
    }
    let response = fixture.raw_response(&wrong);
    assert_eq!(response["ok"], false);
    assert!(response["error"]
        .as_str()
        .unwrap()
        .contains("descriptor differs"));
    let response = fixture.raw_response(&fixture.start());
    assert_eq!(response["ok"], false);
    assert!(
        response["error"]
            .as_str()
            .unwrap()
            .contains("execution custody"),
        "{response}"
    );
    assert_eq!(
        fs::read(fixture.original.authority.join("dispatch.jsonl")).unwrap(),
        before
    );
    assert_eq!(fixture.count("funding_witness"), 0);
    assert_eq!(fixture.count("funding_participants"), 0);
    assert_eq!(service.finish().worker_spawns, 0);
}

#[test]
fn bound_service_rejects_replaced_config_without_reloading_trust() {
    let fixture = Fixture::new(true);
    let service = fixture.launch();
    fixture.reserve();
    let before = fs::read(fixture.original.authority.join("dispatch.jsonl")).unwrap();
    // Valid equivalent JSON with changed original bytes must still invalidate
    // the cached installation. A newly loaded file would accept this change.
    let mut bytes = fs::read(&fixture.original.installation).unwrap();
    bytes.push(b'\n');
    fs::write(&fixture.original.installation, &bytes).unwrap();
    InstalledFunding::for_test(&fixture.original.installation).unwrap();
    let response = fixture.raw_response(&fixture.start());
    assert_eq!(response["ok"], false);
    assert!(response["error"]
        .as_str()
        .unwrap()
        .contains("configuration replaced"));
    assert_eq!(
        fs::read(fixture.original.authority.join("dispatch.jsonl")).unwrap(),
        before
    );
    assert_eq!(fixture.count("funding_witness"), 0);
    assert_eq!(service.finish().worker_spawns, 0);
}

#[test]
fn generic_fit_client_still_rejects_an_actual_foreign_executable() {
    let fixture = Fixture::new(true);
    let service = fixture.launch();
    let path = identity_path(&fixture.config.socket);
    let original = fs::read(&path).unwrap();
    let mut identity: ServiceIdentity = serde_json::from_slice(&original).unwrap();
    let mut other = std::process::Command::new("/bin/sleep")
        .arg("30")
        .spawn()
        .unwrap();
    identity.process = ProcessIdentity::read(other.id()).unwrap();
    identity.executable_sha256 = verified_executable(&identity.process).unwrap();
    fs::write(&path, serde_json::to_vec(&identity).unwrap()).unwrap();
    let result = connect_approved_until(
        &fixture.config.socket,
        &fixture.original.authority,
        &fixture.original.watermark,
        request_monotonic_ns().unwrap() + 30_000_000_000,
    );
    fs::write(&path, original).unwrap();
    let _ = other.kill();
    other.wait().unwrap();
    assert!(result
        .err()
        .unwrap()
        .to_string()
        .contains("service executable differs from approved client release"));
    assert_eq!(service.finish().worker_spawns, 0);
}

#[test]
fn funding_concurrency_is_bounded_before_descriptor_read_and_released_on_disconnect() {
    let fixture = Fixture::new(false);
    let service = fixture.launch();
    let mut held = Vec::new();
    for _ in 0..fixture.config.max_workers {
        let deadline = request_monotonic_ns().unwrap() + 10_000_000_000;
        let mut stream =
            connect_bounded(&fixture.config.socket, remaining_request(deadline).unwrap()).unwrap();
        send_json(
            &mut stream,
            &Request::Funding {
                descriptor_sha256: sha(b"not transferred"),
                deadline_monotonic_ns: deadline,
            },
            remaining_request(deadline).unwrap(),
        )
        .unwrap();
        held.push(stream);
    }
    let deadline = Instant::now() + Duration::from_secs(30);
    while service.funding_active() < fixture.config.max_workers.min(16) {
        assert!(
            Instant::now() < deadline,
            "funding slot holders did not reach sealed FD read"
        );
        std::thread::yield_now();
    }
    let response = fixture.raw_response(&fixture.start());
    assert_eq!(response["ok"], false);
    assert!(
        response["error"]
            .as_str()
            .unwrap()
            .contains("funding admission slots exhausted"),
        "funding admission should reject another request while all slots hold at FD read: {response}"
    );
    assert_eq!(fixture.count("funding_authorizations"), 0);
    assert_eq!(fixture.count("funding_witness"), 0);
    drop(held);
    let deadline = Instant::now() + Duration::from_secs(5);
    while service.funding_active() != 0 {
        assert!(
            Instant::now() < deadline,
            "disconnected funding holder retained its slot"
        );
        std::thread::yield_now();
    }
    let response = fixture.raw_response(&fixture.start());
    assert_eq!(response["ok"], false);
    assert!(
        response["error"]
            .as_str()
            .unwrap()
            .contains("execution custody"),
        "disconnect should release funding slot without admitting task: {response}"
    );
    fixture
        .request(&FundingRequest::Register {
            authorization: fixture.original.authorization.clone(),
        })
        .unwrap();
    assert_eq!(fixture.count("funding_authorizations"), 1);
    assert_eq!(service.finish().worker_spawns, 0);
}

#[test]
fn funding_mutex_rejects_expired_uncontended_acquisition() {
    let mutex = std::sync::Mutex::new(());
    let expired = request_monotonic_ns().unwrap().saturating_sub(1);
    assert!(funding_ipc::lock_with_deadline(&mutex, expired).is_err());
    assert!(mutex.try_lock().is_ok());
}

#[test]
fn funding_mutex_contention_expires_without_releasing_holder() {
    let mutex = std::sync::Mutex::new(());
    let held = mutex.lock().unwrap();
    let deadline = request_monotonic_ns().unwrap() + 50_000_000;
    let started = Instant::now();
    assert!(funding_ipc::lock_with_deadline(&mutex, deadline).is_err());
    assert!(request_monotonic_ns().unwrap() >= deadline);
    assert!(started.elapsed() < Duration::from_secs(2));
    drop(held);
    assert!(funding_ipc::lock_with_deadline(
        &mutex,
        request_monotonic_ns().unwrap() + 1_000_000_000
    )
    .is_ok());
}

#[test]
fn funding_mutex_poison_rejects_immediately() {
    let mutex = Arc::new(std::sync::Mutex::new(()));
    let poison = mutex.clone();
    assert!(std::thread::spawn(move || {
        let _held = poison.lock().unwrap();
        panic!("intentional funding mutex poison fixture");
    })
    .join()
    .is_err());
    let started = Instant::now();
    let error =
        funding_ipc::lock_with_deadline(&mutex, request_monotonic_ns().unwrap() + 30_000_000_000)
            .unwrap_err();
    assert!(error.to_string().contains("mutex poisoned"));
    assert!(started.elapsed() < Duration::from_secs(2));
}
