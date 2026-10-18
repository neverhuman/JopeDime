//! Actual native socket, SO_PEERCRED, sealed-descriptor and signed-witness
//! integration. Only loaded CPU runtime metadata is observed; no numerical
//! worker or CUDA context is started. Private installation exists only in tests.
use super::*;
use crate::eval_registry::{
    funding_config::InstalledFunding, witness::funding_tests::Fixture as WitnessFixture,
};
use rusqlite::Connection;

struct Fixture {
    original: WitnessFixture,
    installation: InstalledFunding,
    config: ServiceConfig,
    descriptor: Value,
}
struct Running {
    stop: Arc<AtomicBool>,
    state: Arc<State>,
    thread: Option<std::thread::JoinHandle<Result<ServiceStats>>>,
}
impl Running {
    fn finish(mut self) -> ServiceStats {
        self.stop.store(true, Ordering::SeqCst);
        self.thread.take().unwrap().join().unwrap().unwrap()
    }
}
impl Drop for Running {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
impl Fixture {
    fn new(observed_runtime: bool) -> Self {
        Self::from_witness(WitnessFixture::new(), observed_runtime)
    }
    fn new_resident_training(observed_runtime: bool) -> Self {
        Self::from_witness(WitnessFixture::new_resident_training(), observed_runtime)
    }
    fn from_witness(mut original: WitnessFixture, observed_runtime: bool) -> Self {
        if observed_runtime {
            let actual = crate::runtime_custody::loaded_runtime_descriptor(
                &ProcessIdentity::current().unwrap(),
            )
            .unwrap();
            original.use_observed_runtime(actual["runtime_sha256"].as_str().unwrap());
        }
        let installation = InstalledFunding::for_test(&original.installation).unwrap();
        let installed = installation.revalidate().unwrap();
        let mut config = ServiceConfig::regression(
            original.authority.clone(),
            original.watermark.clone(),
            installed.socket.clone(),
            std::env::current_exe().unwrap(),
            2,
        );
        config.event_witness = Some(original.client.clone());
        config.funding_fixture = Some(installation.clone());
        let descriptor = serde_json::from_slice(&original.descriptor).unwrap();
        Self {
            original,
            installation,
            config,
            descriptor,
        }
    }
    fn launch(&self) -> Running {
        let service = Service::bind(self.config.clone()).unwrap();
        let state = service.state.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let signal = stop.clone();
        Running {
            stop,
            state,
            thread: Some(std::thread::spawn(move || service.run(signal))),
        }
    }
    fn request(&self, request: &FundingRequest) -> Result<Value> {
        funding_ipc::funding_request_installed(&self.installation, request, request_monotonic_ns()?)
    }
    fn start(&self) -> FundingRequest {
        FundingRequest::Start {
            authorization: self.original.authorization.clone(),
            reservation: self.original.reservation.clone(),
            descriptor_json: self.descriptor.clone(),
        }
    }
    fn participant(&self, slot: u32) -> FundingRequest {
        FundingRequest::Participant {
            authorization: self.original.authorization.clone(),
            reservation: self.original.reservation.clone(),
            descriptor_json: self.descriptor.clone(),
            participant_slot: slot,
        }
    }
    fn reserve(&self) {
        self.request(&FundingRequest::Register {
            authorization: self.original.authorization.clone(),
        })
        .unwrap();
        self.request(&FundingRequest::Reserve {
            authorization: self.original.authorization.clone(),
            reservation: self.original.reservation.clone(),
            descriptor_json: self.descriptor.clone(),
        })
        .unwrap();
        let proof = self
            .request(&FundingRequest::Preflight {
                authorization: self.original.authorization.clone(),
                reservation: self.original.reservation.clone(),
                descriptor_json: self.descriptor.clone(),
            })
            .unwrap();
        assert_eq!(proof["numerical_start_released"], false);
    }
    fn raw_response(&self, request: &FundingRequest) -> Value {
        // Deliberately bypass only the client-side validator, never the actual
        // service's peer authentication, sealed FD transport or request parser.
        let deadline = request_monotonic_ns().unwrap() + 30_000_000_000;
        let bytes = serde_json::to_vec(request).unwrap();
        let input = sealed(&bytes).unwrap();
        let mut stream =
            connect_bounded(&self.config.socket, remaining_request(deadline).unwrap()).unwrap();
        send_json(
            &mut stream,
            &Request::Funding {
                descriptor_sha256: sha(&bytes),
                deadline_monotonic_ns: deadline,
            },
            remaining_request(deadline).unwrap(),
        )
        .unwrap();
        let sent = send_fd(
            &stream,
            input.as_raw_fd(),
            remaining_request(deadline).unwrap(),
        );
        let response: Value =
            receive_json(&mut stream, remaining_request(deadline).unwrap()).unwrap();
        // A rejection before descriptor admission may close its receiving end
        // before sendmsg runs. Preserve that original negative response.
        assert!(sent.is_ok() || response["ok"] == false);
        response
    }
    fn count(&self, table: &str) -> u64 {
        assert!([
            "funding_authorizations",
            "funding_reservations",
            "funding_participants",
            "funding_witness"
        ]
        .contains(&table));
        Connection::open(self.original.authority.join("registry.sqlite"))
            .unwrap()
            .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))
            .unwrap()
    }
}

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
    let deadline = Instant::now() + Duration::from_secs(5);
    while service.state.funding_active.load(Ordering::SeqCst) != fixture.config.max_workers {
        assert!(
            Instant::now() < deadline,
            "funding slot holders did not reach sealed FD read"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    let response = fixture.raw_response(&fixture.start());
    assert_eq!(response["ok"], false);
    assert!(response["error"]
        .as_str()
        .unwrap()
        .contains("funding admission slots exhausted"));
    assert_eq!(fixture.count("funding_authorizations"), 0);
    assert_eq!(fixture.count("funding_witness"), 0);
    drop(held);
    let deadline = Instant::now() + Duration::from_secs(5);
    while service.state.funding_active.load(Ordering::SeqCst) != 0 {
        assert!(
            Instant::now() < deadline,
            "disconnected funding holder retained its slot"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
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

fn ensure_numerical_env_or_reexec(test_name: &str) -> bool {
    let raw = fs::read("/proc/self/environ").unwrap_or_default();
    let env_map: std::collections::BTreeMap<String, String> = raw
        .split(|b| *b == 0)
        .filter_map(|v| std::str::from_utf8(v).ok()?.split_once('='))
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect();
    let has_all = [
        "OMP_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]
    .iter()
    .all(|k| env_map.get(*k).map(String::as_str) == Some("1"));
    if has_all {
        return false;
    }
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test_name, "--nocapture"])
        .env("OMP_NUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", "1")
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("MKL_NUM_THREADS", "1")
        .status()
        .expect("re-exec test with numerical environment");
    assert!(
        status.success(),
        "{test_name} failed in numerical env child"
    );
    true
}

#[test]
fn controller_calls_verify_gpu_task_grant() {
    if ensure_numerical_env_or_reexec(
        "eval_registry::service::funding_tests::controller_calls_verify_gpu_task_grant",
    ) {
        return;
    }
    let fixture = Fixture::new_resident_training(true);
    let service = fixture.launch();
    fixture.reserve();
    let witness = fixture.original.server(1, false);
    let start = funding_ipc::funding_admission_installed(
        &fixture.installation,
        &fixture.start(),
        request_monotonic_ns().unwrap(),
    )
    .unwrap();
    crate::controller::verify_controller_gpu_grant(
        &start,
        &fixture.descriptor,
        "regression_training",
    )
    .unwrap();
    assert_eq!(witness.join().unwrap().len(), 1);
    assert_eq!(service.finish().worker_spawns, 0);
}

#[test]
fn participant_per_rank_grant_verified() {
    if ensure_numerical_env_or_reexec(
        "eval_registry::service::funding_tests::participant_per_rank_grant_verified",
    ) {
        return;
    }
    let fixture = Fixture::new_resident_training(true);
    let service = fixture.launch();
    fixture.reserve();
    let witness = fixture.original.server(2, false);
    let _start = funding_ipc::funding_admission_installed(
        &fixture.installation,
        &fixture.start(),
        request_monotonic_ns().unwrap(),
    )
    .unwrap();
    let rank0 = funding_ipc::funding_admission_installed(
        &fixture.installation,
        &fixture.participant(0),
        request_monotonic_ns().unwrap(),
    )
    .unwrap();
    crate::controller::verify_participant_gpu_grant(
        &rank0,
        &fixture.descriptor,
        0,
        "regression_training",
    )
    .unwrap();
    // Same process cannot claim a second participant rank slot
    assert!(fixture.request(&fixture.participant(1)).is_err());
    assert_eq!(witness.join().unwrap().len(), 2);
    assert_eq!(service.finish().worker_spawns, 0);
}

#[test]
fn negative_grant_mismatch_fails_closed() {
    if ensure_numerical_env_or_reexec(
        "eval_registry::service::funding_tests::negative_grant_mismatch_fails_closed",
    ) {
        return;
    }
    let fixture = Fixture::new_resident_training(true);
    let service = fixture.launch();
    fixture.reserve();
    let witness = fixture.original.server(2, false);
    let start = funding_ipc::funding_admission_installed(
        &fixture.installation,
        &fixture.start(),
        request_monotonic_ns().unwrap(),
    )
    .unwrap();
    let rank0 = funding_ipc::funding_admission_installed(
        &fixture.installation,
        &fixture.participant(0),
        request_monotonic_ns().unwrap(),
    )
    .unwrap();

    // Wrong capability
    assert!(crate::controller::verify_controller_gpu_grant(
        &start,
        &fixture.descriptor,
        "regression_proposals"
    )
    .is_err());

    // Controller admission cannot impersonate a participant rank
    assert!(crate::controller::verify_participant_gpu_grant(
        &start,
        &fixture.descriptor,
        0,
        "regression_training"
    )
    .is_err());

    // Participant rank 0 cannot impersonate controller or rank 1
    assert!(crate::controller::verify_controller_gpu_grant(
        &rank0,
        &fixture.descriptor,
        "regression_training"
    )
    .is_err());
    assert!(crate::controller::verify_participant_gpu_grant(
        &rank0,
        &fixture.descriptor,
        1,
        "regression_training"
    )
    .is_err());

    // Tampered descriptor world / deadline
    let mut wrong_world = fixture.descriptor.clone();
    wrong_world["task"]["world"] = json!(8);
    assert!(crate::controller::verify_controller_gpu_grant(
        &start,
        &wrong_world,
        "regression_training"
    )
    .is_err());

    assert_eq!(witness.join().unwrap().len(), 2);
    assert_eq!(service.finish().worker_spawns, 0);
}

#[test]
fn verify_gpu_task_grant_positive_and_negative() {
    if ensure_numerical_env_or_reexec(
        "eval_registry::service::funding_tests::verify_gpu_task_grant_positive_and_negative",
    ) {
        return;
    }
    let fixture = Fixture::new_resident_training(true);
    let service = fixture.launch();
    fixture.reserve();
    let witness = fixture.original.server(2, false);
    let start = funding_ipc::funding_admission_installed(
        &fixture.installation,
        &fixture.start(),
        request_monotonic_ns().unwrap(),
    )
    .unwrap();
    let rank0 = funding_ipc::funding_admission_installed(
        &fixture.installation,
        &fixture.participant(0),
        request_monotonic_ns().unwrap(),
    )
    .unwrap();

    // Positive: controller and participant rank 0 pass
    crate::controller::verify_controller_gpu_grant(
        &start,
        &fixture.descriptor,
        "regression_training",
    )
    .unwrap();
    crate::controller::verify_participant_gpu_grant(
        &rank0,
        &fixture.descriptor,
        0,
        "regression_training",
    )
    .unwrap();

    // Negative: capability mismatch, slot mismatch, second slot from same PID, and descriptor mismatch fail closed
    assert!(fixture.request(&fixture.participant(1)).is_err());
    assert!(crate::controller::verify_controller_gpu_grant(
        &start,
        &fixture.descriptor,
        "unfunded_capability"
    )
    .is_err());
    assert!(crate::controller::verify_participant_gpu_grant(
        &rank0,
        &fixture.descriptor,
        1,
        "regression_training"
    )
    .is_err());
    assert!(crate::controller::verify_controller_gpu_grant(
        &rank0,
        &fixture.descriptor,
        "regression_training"
    )
    .is_err());
    let mut tampered = fixture.descriptor.clone();
    tampered["task"]["world"] = json!(8);
    assert!(crate::controller::verify_controller_gpu_grant(
        &start,
        &tampered,
        "regression_training"
    )
    .is_err());

    assert_eq!(witness.join().unwrap().len(), 2);
    assert_eq!(service.finish().worker_spawns, 0);
}
