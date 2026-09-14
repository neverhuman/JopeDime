//! Synthetic process/socket tests; these do not initialize CUDA or qualify NCCL.
use jope_parallel::manifest;
use jope_parallel::checkpoint_recovery;
use jope_parallel::sealed_storage;
use jope_parallel::task_contract;
#[path = "../src/resident_protocol.rs"]
mod protocol;
use manifest::{read_json, write_json};
use protocol::*;
use serde_json::{json, Value};
use std::{
    fs,
    os::unix::net::{UnixListener, UnixStream},
    process::{Command, Stdio},
    sync::{Arc, Barrier},
    time::{Duration, Instant},
};

fn request() -> Value {
    json!({"rank":0,"task":{"kind":"synthetic","output":"same"}})
}
fn worker() -> WorkerIdentity {
    WorkerIdentity {
        process: process_identity(std::process::id()).unwrap(),
        socket_device: 1,
        socket_inode: 2,
        executable_sha256: "a".repeat(64),
    }
}

fn manifest_request(root: &std::path::Path) -> Value {
    fs::create_dir_all(root.join("corpus")).unwrap();
    fs::create_dir_all(root.join("run")).unwrap();
    write_json(&root.join("run/run.json"), &json!({"schema":"test"})).unwrap();
    write_json(&root.join("corpus/vocabulary.json"), &json!({"tokens":["original"]})).unwrap();
    fs::write(root.join("corpus/records.bin"), b"original records").unwrap();
    write_json(&root.join("corpus/index.json"), &json!({"schema":"parallel.teaching_corpus.v1",
        "vocabulary_sha256":manifest::file_hash(&root.join("corpus/vocabulary.json")).unwrap(),
        "records_sha256":manifest::file_hash(&root.join("corpus/records.bin")).unwrap(),"contexts":[]})).unwrap();
    json!({"root":root,"rank":0,"task":{"kind":"Train","corpus":root.join("corpus"),
        "run":root.join("run"),"resume":false,"initialize":null},"env":{},"executable_sha256":"a".repeat(64)})
}

#[test]
fn frozen_input_manifest_change_rejects_without_authorizing_dispatch() {
    let d = tempfile::tempdir().unwrap();
    let q = manifest_request(d.path());
    let path = d.path().join("run/request.json");
    let deadline = now_ms().unwrap() + 60_000;
    let saved = freeze_request(&path, q.clone(), deadline).unwrap();
    assert_eq!(freeze_request(&path, q.clone(), deadline + 10_000).unwrap(), saved);
    write_json(&d.path().join("corpus/index.json"), &json!({"schema":"changed"})).unwrap();
    assert!(freeze_request(&path, q, deadline).is_err());
    assert!(verify_request_inputs(&saved).is_err());
    assert!(!d.path().join("dispatches").exists());
}

#[test]
fn funding_common_task_is_rank_invariant_and_binds_inputs_and_original_deadline() {
    let d = tempfile::tempdir().unwrap();
    let q = manifest_request(d.path());
    let deadline = now_ms().unwrap() + 60_000;
    let saved = freeze_request(&d.path().join("request.json"), q, deadline).unwrap();
    let first = serde_json::to_value(common_task_descriptor(&saved).unwrap()).unwrap();
    let mut other_rank = saved.clone(); other_rank["rank"] = json!(7);
    assert_eq!(first, serde_json::to_value(common_task_descriptor(&other_rank).unwrap()).unwrap());
    other_rank["deadline_unix_ms"] = json!(deadline + 1);
    assert_ne!(first, serde_json::to_value(common_task_descriptor(&other_rank).unwrap()).unwrap());
    assert!(first.get("rank").is_none() && first.get("reservation").is_none());
    fs::write(d.path().join("corpus/records.bin"), b"changed records").unwrap();
    assert!(common_task_descriptor(&saved).is_err());
    assert!(!d.path().join("dispatches").exists());
}

#[test]
fn unchanged_manifest_cannot_hide_mutated_payload_on_reuse_or_fresh_capture() {
    let d = tempfile::tempdir().unwrap();
    let q = manifest_request(d.path());
    let deadline = now_ms().unwrap() + 60_000;
    let path = d.path().join("request.json");
    let saved = freeze_request(&path, q.clone(), deadline).unwrap();
    fs::write(d.path().join("corpus/records.bin"), b"changed records").unwrap();
    assert!(verify_request_inputs(&saved).is_err());
    assert!(freeze_request(&path, q.clone(), deadline).is_err());
    assert!(freeze_request(&d.path().join("new-request.json"), q, deadline).is_err());
    assert!(!d.path().join("new-request.json").exists());
}

#[test]
fn supplied_subset_cannot_omit_required_input_artifacts() {
    let d = tempfile::tempdir().unwrap();
    let q = manifest_request(d.path());
    let mut saved = freeze_request(&d.path().join("request.json"), q, now_ms().unwrap() + 60_000).unwrap();
    saved["input_artifacts"].as_object_mut().unwrap().remove("corpus/records.bin");
    assert!(verify_request_inputs(&saved).is_err());
}

fn checkpoint_fixture(root: &std::path::Path, name: &str) -> String {
    let cp = root.join(name);
    fs::create_dir_all(&cp).unwrap();
    for filename in ["model.ot", "optimizer.ot", "optimizer.json", "config.json"] {
        fs::write(cp.join(filename), format!("{name}:{filename}")).unwrap();
    }
    write_json(&cp.join("state.json"), &json!({
        "model_sha256":manifest::file_hash(&cp.join("model.ot")).unwrap(),
        "optimizer_sha256":manifest::file_hash(&cp.join("optimizer.ot")).unwrap(),
        "optimizer_metadata_sha256":manifest::file_hash(&cp.join("optimizer.json")).unwrap(),
        "config_sha256":manifest::file_hash(&cp.join("config.json")).unwrap()})).unwrap();
    manifest::file_hash(&cp.join("state.json")).unwrap()
}

#[test]
fn resume_uses_frozen_selection_while_latest_may_advance() {
    let d = tempfile::tempdir().unwrap();
    let mut q = manifest_request(d.path());
    let first_sha = checkpoint_fixture(d.path(), "checkpoint-a");
    let second_sha = checkpoint_fixture(d.path(), "checkpoint-b");
    let first = json!({"checkpoint":"checkpoint-a","state_sha256":first_sha});
    let second = json!({"checkpoint":"checkpoint-b","state_sha256":second_sha});
    fs::create_dir_all(d.path().join("run/launch")).unwrap();
    write_json(&d.path().join("run/initialization.json"), &json!({"kind":"fresh"})).unwrap();
    write_json(&d.path().join("run/launch/resume-input.json"), &first).unwrap();
    write_json(&d.path().join("run/latest.json"), &first).unwrap();
    q["task"]["resume"] = json!(true);
    q["task"]["launch"] = json!(d.path().join("run/launch"));
    let saved = freeze_request(&d.path().join("request.json"), q, now_ms().unwrap() + 60_000).unwrap();
    write_json(&d.path().join("run/latest.json"), &second).unwrap();
    verify_request_inputs(&saved).unwrap();
    assert!(saved["input_artifacts"].get("checkpoint-a/state.json").is_some());
    assert!(saved["input_artifacts"].get("checkpoint-b/state.json").is_none());
    write_json(&d.path().join("run/launch/resume-input.json"), &second).unwrap();
    assert!(verify_request_inputs(&saved).is_err());
}

#[test]
fn proposal_run_seed_champion_and_teacher_labels_are_required_inputs() {
    let d = tempfile::tempdir().unwrap();
    let mut q = manifest_request(d.path());
    checkpoint_fixture(d.path(), "checkpoint");
    fs::write(d.path().join("corpus/labels.jsonl"), b"teacher labels").unwrap();
    write_json(&d.path().join("corpus/provenance.json"), &json!({
        "labels_sha256":manifest::file_hash(&d.path().join("corpus/labels.jsonl")).unwrap()})).unwrap();
    fs::create_dir_all(d.path().join("proposal")).unwrap();
    write_json(&d.path().join("champions.json"), &json!({"metadata":"same","genome":"first"})).unwrap();
    let spec = json!({"proposal_seed":17,
        "champion_snapshot_sha256":manifest::file_hash(&d.path().join("champions.json")).unwrap()});
    write_json(&d.path().join("proposal/run.json"), &spec).unwrap();
    q["task"] = json!({"kind":"Proposal","corpus":d.path().join("corpus"),
        "checkpoint":"checkpoint","output":"proposal"});
    q["env"] = json!({"JOPE_CHAMPION_SNAPSHOT":"champions.json"});
    let saved = freeze_request(&d.path().join("request.json"), q, now_ms().unwrap() + 60_000).unwrap();
    for path in ["proposal/run.json", "champions.json", "corpus/labels.jsonl", "corpus/provenance.json"] {
        assert!(saved["input_artifacts"].get(path).is_some());
    }
    let mut changed = spec.clone(); changed["proposal_seed"] = json!(18);
    write_json(&d.path().join("proposal/run.json"), &changed).unwrap();
    assert!(verify_request_inputs(&saved).is_err());
    write_json(&d.path().join("proposal/run.json"), &spec).unwrap();
    write_json(&d.path().join("champions.json"), &json!({"metadata":"same","genome":"second"})).unwrap();
    assert!(verify_request_inputs(&saved).is_err());
}

#[test]
fn frozen_request_deadline_cannot_be_renewed_or_torn_request_replaced() {
    let d = tempfile::tempdir().unwrap();
    let q = manifest_request(d.path());
    let path = d.path().join("request.json");
    let deadline = now_ms().unwrap() + 60_000;
    let mut saved = freeze_request(&path, q.clone(), deadline).unwrap();
    saved["deadline_unix_ms"] = json!(1);
    write_json(&path, &saved).unwrap();
    assert!(freeze_request(&path, q.clone(), deadline).is_err());
    fs::write(&path, b"{").unwrap();
    assert!(freeze_request(&path, q, deadline).is_err());
    assert_eq!(fs::read(path).unwrap(), b"{");
}

#[test]
fn expired_deadline_rejects_before_request_publication() {
    let d = tempfile::tempdir().unwrap();
    let q = manifest_request(d.path());
    let path = d.path().join("request.json");
    assert!(freeze_request(&path, q, 1).is_err());
    assert!(!path.exists());
}

#[test]
fn partial_ipc_message_does_not_renew_absolute_deadline() {
    use std::io::Write;
    let (mut client, mut server) = UnixStream::pair().unwrap();
    let producer = std::thread::spawn(move || {
        for _ in 0..10 {
            if server.write_all(b"x").is_err() { break; }
            std::thread::sleep(Duration::from_millis(10));
        }
    });
    let started = Instant::now();
    assert!(read_line_until(&mut client, 1024, now_ms().unwrap() + 35).is_err());
    assert!(started.elapsed() < Duration::from_secs(1));
    drop(client);
    producer.join().unwrap();
}

#[test]
fn interrupted_dispatch_is_permanent_even_before_descriptor_transfer() {
    let d = tempfile::tempdir().unwrap();
    let q = request();
    ensure_undispatched(d.path(), &q).unwrap();
    authorize_dispatch(d.path(), &q).unwrap();
    verify_dispatch(d.path(), &q, std::process::id()).unwrap();
    assert!(ensure_undispatched(d.path(), &q).is_err());
    assert!(authorize_dispatch(d.path(), &q).is_err());
    assert!(verify_dispatch(d.path(), &q, 1).is_err());
}

#[test]
fn torn_dispatch_does_not_become_unseen() {
    let d = tempfile::tempdir().unwrap();
    let q = request();
    fs::create_dir(d.path().join("dispatches")).unwrap();
    fs::write(
        d.path().join("dispatches").join(request_id(&q).unwrap()),
        b"{",
    )
    .unwrap();
    assert!(ensure_undispatched(d.path(), &q).is_err());
    assert!(verify_dispatch(d.path(), &q, std::process::id()).is_err());
}

#[test]
fn only_one_concurrent_acceptance_can_start() {
    let d = tempfile::tempdir().unwrap();
    let barrier = Arc::new(Barrier::new(32));
    let jobs = (0..32)
        .map(|_| {
            let barrier = barrier.clone();
            let pool = d.path().to_owned();
            std::thread::spawn(move || {
                barrier.wait();
                AcceptedTask::begin(&pool, &request(), &worker()).is_ok()
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(
        jobs.into_iter()
            .map(|job| job.join().unwrap())
            .filter(|started| *started)
            .count(),
        1
    );
}

#[test]
fn accepted_without_result_never_runs_again() {
    let d = tempfile::tempdir().unwrap();
    let q = request();
    drop(AcceptedTask::begin(d.path(), &q, &worker()).unwrap());
    assert!(committed(d.path(), &q, &d.path().join("output")).is_err());
    assert!(AcceptedTask::begin(d.path(), &q, &worker()).is_err());
}

#[test]
fn interrupted_acceptance_directory_is_also_a_tombstone() {
    let d = tempfile::tempdir().unwrap();
    let q = request();
    fs::create_dir_all(d.path().join("tasks").join(request_id(&q).unwrap())).unwrap();
    assert!(committed(d.path(), &q, &d.path().join("output")).is_err());
    assert!(AcceptedTask::begin(d.path(), &q, &worker()).is_err());
}

#[test]
fn failure_cannot_be_laundered_by_a_valid_looking_output() {
    let d = tempfile::tempdir().unwrap();
    let q = request();
    let output = d.path().join("output");
    fs::write(&output, b"stale success").unwrap();
    let receipt = AcceptedTask::begin(d.path(), &q, &worker())
        .unwrap()
        .finish(&output, Some("injected failure".into()))
        .unwrap();
    assert_eq!(receipt["success"], false);
    assert!(receipt["receipt_sha256"].is_null());
    assert!(committed(d.path(), &q, &output).is_err());
    assert!(AcceptedTask::begin(d.path(), &q, &worker()).is_err());
}

#[test]
fn original_completion_is_adoptable_after_lost_response() {
    let d = tempfile::tempdir().unwrap();
    let q = request();
    let output = d.path().join("output");
    fs::write(&output, b"original output").unwrap();
    let task = AcceptedTask::begin(d.path(), &q, &worker()).unwrap();
    let result = task.finish(&output, None).unwrap();
    assert_eq!(committed(d.path(), &q, &output).unwrap(), Some(result));
    assert!(AcceptedTask::begin(d.path(), &q, &worker()).is_err());
    fs::write(&output, b"changed output").unwrap();
    assert!(committed(d.path(), &q, &output).is_err());
}

#[test]
fn completion_requires_its_original_acceptance_digest() {
    let d = tempfile::tempdir().unwrap();
    let q = request();
    let output = d.path().join("output");
    fs::write(&output, b"original output").unwrap();
    AcceptedTask::begin(d.path(), &q, &worker())
        .unwrap()
        .finish(&output, None)
        .unwrap();
    let accepted = d
        .path()
        .join("tasks")
        .join(request_id(&q).unwrap())
        .join("accepted.json");
    let mut value: Value = read_json(&accepted).unwrap();
    value["worker"]["process"]["start_ticks"] = json!(0);
    write_json(&accepted, &value).unwrap();
    assert!(committed(d.path(), &q, &output).is_err());
}

#[test]
fn socket_and_peer_birth_are_both_verified() {
    let d = tempfile::tempdir().unwrap();
    let socket = d.path().join("worker.sock");
    let listener = UnixListener::bind(&socket).unwrap();
    let identity = WorkerIdentity::capture(
        &socket,
        manifest::file_hash(std::path::Path::new("/proc/self/exe")).unwrap(),
    )
    .unwrap();
    let stream = UnixStream::connect(&socket).unwrap();
    let _server = listener.accept().unwrap();
    identity.verify_connection(&socket, &stream).unwrap();
    let mut stale = identity.clone();
    stale.process.start_ticks += 1;
    assert!(stale.verify_connection(&socket, &stream).is_err());
    stale = identity.clone();
    stale.process.boot_id = "other boot".into();
    assert!(stale.verify_connection(&socket, &stream).is_err());
    stale = identity.clone();
    stale.process.pid += 1;
    assert!(stale.verify_connection(&socket, &stream).is_err());
    fs::remove_file(&socket).unwrap();
    let _replacement = UnixListener::bind(&socket).unwrap();
    assert!(identity.verify_connection(&socket, &stream).is_err());
}

fn publish_worker(pool: &std::path::Path, worker: &WorkerIdentity) {
    write_json(
        &pool.join("pool.json"),
        &json!({"phase":"running","processes":{"0":worker.process}}),
    )
    .unwrap();
    write_json(
        &pool.join("rank-0.json"),
        &json!({"status":"idle","worker":worker}),
    )
    .unwrap();
}

#[test]
fn pretransfer_socket_readiness_race_is_retryable_within_one_deadline() {
    let d = tempfile::tempdir().unwrap();
    let pool = d.path().to_owned();
    let socket = pool.join("rank-0.sock");
    let executable = manifest::file_hash(std::path::Path::new("/proc/self/exe")).unwrap();
    let listener = UnixListener::bind(&socket).unwrap();
    let original = WorkerIdentity::capture(&socket, executable.clone()).unwrap();
    publish_worker(&pool, &original);
    drop(listener);
    fs::remove_file(&socket).unwrap();
    let background_pool = pool.clone();
    let background_executable = executable.clone();
    let (done, finished) = std::sync::mpsc::channel();
    let replacement = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(30));
        let socket = background_pool.join("rank-0.sock");
        let listener = UnixListener::bind(&socket).unwrap();
        let identity = WorkerIdentity::capture(&socket, background_executable).unwrap();
        publish_worker(&background_pool, &identity);
        // Keep the endpoint alive without blocking cleanup if the client fails.
        let _ = finished.recv_timeout(Duration::from_secs(15));
        listener
    });
    let result = connect_rank(&pool, 0, &executable, Duration::from_secs(10));
    let _ = done.send(());
    let _listener = replacement.join().unwrap();
    assert!(result.is_ok(), "{result:?}");
    assert!(
        !pool.join("dispatches").exists(),
        "readiness must not authorize work"
    );
}

#[test]
fn stale_readiness_is_bounded_and_does_not_authorize_dispatch() {
    let d = tempfile::tempdir().unwrap();
    let mut stale = worker();
    stale.process.start_ticks += 1;
    publish_worker(d.path(), &stale);
    let started = Instant::now();
    assert!(connect_rank(
        d.path(),
        0,
        &stale.executable_sha256,
        Duration::from_millis(30)
    )
    .is_err());
    assert!(started.elapsed() < Duration::from_secs(1));
    assert!(!d.path().join("dispatches").exists());
}

#[test]
fn accepted_child_fixture() {
    let Some(pool) = std::env::var_os("JOPE_SYNTHETIC_RESIDENT_TEST_POOL") else {
        return;
    };
    let pool = std::path::PathBuf::from(pool);
    let _task = AcceptedTask::begin(&pool, &request(), &worker()).unwrap();
    fs::write(pool.join("accepted-ready"), b"ready").unwrap();
    std::thread::sleep(Duration::from_secs(10));
}

#[test]
fn worker_death_after_acceptance_does_not_permit_reexecution() {
    let d = tempfile::tempdir().unwrap();
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "accepted_child_fixture"])
        .env("JOPE_SYNTHETIC_RESIDENT_TEST_POOL", d.path())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    let start = Instant::now();
    while !d.path().join("accepted-ready").exists() && start.elapsed() < Duration::from_secs(3) {
        if child.try_wait().unwrap().is_some() {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let ready = d.path().join("accepted-ready").exists();
    let _ = child.kill();
    child.wait().unwrap();
    assert!(ready, "child did not durably accept its synthetic task");
    assert!(committed(d.path(), &request(), &d.path().join("output")).is_err());
    assert!(AcceptedTask::begin(d.path(), &request(), &worker()).is_err());
}
