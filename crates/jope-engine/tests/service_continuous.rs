use jope_parallel::{
    manifest::*,
    service::{GpJob, GpJobs},
};
use serde_json::{json, Value};
use std::{
    fs,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

struct Process(Child);
impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn wait_for(mut condition: impl FnMut() -> bool) {
    let start = Instant::now();
    while !condition() {
        assert!(
            start.elapsed() < Duration::from_secs(12),
            "service condition timed out"
        );
        std::thread::sleep(Duration::from_millis(40));
    }
}
fn manifest() -> GpJobs {
    GpJobs {
        schema: "parallel.gp_jobs.v1".into(),
        qualification: true,
        jobs: vec![],
        vocabulary: None,
    }
}

#[test]
fn empty_waiting_service_wakes_and_preserves_bounded_default() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    fs::create_dir(root.join("runs")).unwrap();
    let path = root.join("jobs.json");
    write_json(&path, &manifest()).unwrap();
    let command = || {
        let mut c = Command::new(env!("CARGO_BIN_EXE_jope-parallel"));
        c.arg("--root")
            .arg(root)
            .arg("gp-service")
            .arg("--manifest")
            .arg(&path)
            .args(["--workers", "1", "--threads", "1"])
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        c
    };
    assert!(
        !command().status().unwrap().success(),
        "legacy empty manifest rejection changed"
    );
    let mut worker = Process(command().arg("--wait-for-work").spawn().unwrap());
    let heartbeat = root.join("runs/gp-service.json");
    let state = || read_json::<Value>(&heartbeat).unwrap_or_default();
    wait_for(|| state()["phase"] == "waiting");
    let first = state()["timestamp"].as_u64().unwrap();
    wait_for(|| state()["timestamp"].as_u64().unwrap_or(0) > first);
    assert!(worker.0.try_wait().unwrap().is_none());
    // A malformed qualification control stops new admission without exiting
    // the supervisor (which would terminate its otherwise healthy children).
    let reservation = root.join("runs/gp-qualification-reserve.json");
    fs::write(&reservation, b"invalid reservation").unwrap();
    wait_for(|| state()["qualification_reservation_error"].is_string());
    assert!(worker.0.try_wait().unwrap().is_none());
    fs::remove_file(reservation).unwrap();
    wait_for(|| state()["qualification_reservation_error"].is_null());
    fs::create_dir_all(root.join("prepared")).unwrap();
    fs::write(root.join("prepared/prepared.json"), b"{}").unwrap();
    fs::create_dir_all(root.join("runs/completed")).unwrap();
    write_json(
        &root.join("runs/completed/search.json"),
        &json!({"schema":"parallel.search.v1","completed_evaluations":7}),
    )
    .unwrap();
    let job = GpJob {
        prepared: "prepared".into(),
        prepared_sha256: file_hash(&root.join("prepared/prepared.json")).unwrap(),
        output: "runs/completed".into(),
        ..Default::default()
    };
    let mut incoming = manifest();
    incoming.jobs.push(job);
    write_json(&root.join("runs/gp-priority-jobs.json"), &incoming).unwrap();
    wait_for(|| state()["completed_views"] == 1);
    assert_eq!(state()["restored_completed_evaluations"], 7);
    assert_eq!(state()["phase"], "waiting");
    assert!(
        !root.join("runs/gp-worker-0.log").exists(),
        "committed search was rerun"
    );
    unsafe {
        libc::kill(worker.0.id() as i32, libc::SIGTERM);
    }
    wait_for(|| worker.0.try_wait().unwrap().is_some());
    assert_eq!(state()["phase"], "paused");
    write_json(&path, &incoming).unwrap();
    assert!(command().status().unwrap().success());
    assert_eq!(state()["phase"], "completed");
    assert_eq!(state()["termination_reason"], "queue exhausted");
}

#[test]
fn generations_reject_sealed_jobs_and_changed_rng_output_identity() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    fs::create_dir_all(root.join("checkpoint")).unwrap();
    fs::create_dir_all(root.join("prepared")).unwrap();
    fs::write(root.join("checkpoint/state.json"), b"{}").unwrap();
    write_json(
        &root.join("prepared/prepared.json"),
        &json!({"meta_partition":"meta_test"}),
    )
    .unwrap();
    let mut job = GpJob {
        prepared: "prepared".into(),
        output: "runs/generation-0/search".into(),
        generation: Some(0),
        rng_seed: Some(42),
        parent_checkpoint: Some("checkpoint".into()),
        parent_checkpoint_sha256: Some(file_hash(&root.join("checkpoint/state.json")).unwrap()),
        seeds: Some("seeds.json".into()),
        seeds_sha256: Some("a".repeat(64)),
        ..Default::default()
    };
    assert!(jope_parallel::service::validate_generation(root, &job).is_err());
    write_json(
        &root.join("prepared/prepared.json"),
        &json!({"meta_partition":"meta_train"}),
    )
    .unwrap();
    jope_parallel::service::validate_generation(root, &job).unwrap();
    let mut jobs = manifest();
    jobs.jobs.push(job.clone());
    jope_parallel::queue::merge(root, &jobs).unwrap();
    let before = file_hash(&root.join("runs/gp-priority-jobs.json")).unwrap();
    job.rng_seed = Some(43);
    jobs.jobs = vec![job];
    assert!(jope_parallel::queue::merge(root, &jobs).is_err());
    assert_eq!(
        file_hash(&root.join("runs/gp-priority-jobs.json")).unwrap(),
        before
    );
    let legacy: GpJob = serde_json::from_value(json!({"prepared":"p","prepared_sha256":"h","cut":0,"output":"o","seeds":null,"seeds_sha256":null})).unwrap();
    assert!(legacy.rng_seed.is_none());
    assert!(serde_json::to_value(legacy)
        .unwrap()
        .get("rng_seed")
        .is_none());
}

#[test]
fn missing_strict_census_withholds_healthy_service_before_worker_launch() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    fs::create_dir(root.join("runs")).unwrap();
    fs::create_dir(root.join("prepared")).unwrap();
    write_json(
        &root.join("prepared/prepared.json"),
        &json!({
        "schema":"parallel.prepared.v1","dataset":"dataset","family":"family",
        "meta_partition":"meta_train","cuts":[{"index":4}]}),
    )
    .unwrap();
    write_json(
        &root.join("runs/hardware.json"),
        &jope_parallel::telemetry::Hardware {
            timestamp: jope_parallel::telemetry::now(),
            available_gb: 200.0,
            disk_free_gb: 500.0,
            ..Default::default()
        },
    )
    .unwrap();
    let mut jobs = manifest();
    jobs.jobs.push(GpJob {
        prepared: "prepared".into(),
        prepared_sha256: file_hash(&root.join("prepared/prepared.json")).unwrap(),
        cut: 4,
        output: "runs/withheld".into(),
        ..Default::default()
    });
    write_json(&root.join("jobs.json"), &jobs).unwrap();
    let mut worker = Process(
        Command::new(env!("CARGO_BIN_EXE_jope-parallel"))
            .arg("--root")
            .arg(root)
            .arg("gp-service")
            .arg("--manifest")
            .arg(root.join("jobs.json"))
            .args(["--workers", "1", "--threads", "1"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );
    wait_for(|| worker.0.try_wait().unwrap().is_some());
    assert!(!worker.0.wait().unwrap().success());
    let state: Value = read_json(&root.join("runs/gp-service.json")).unwrap();
    assert_eq!(state["phase"], "unavailable");
    assert!(state["coverage_error"]
        .as_str()
        .unwrap()
        .contains("census unavailable"));
    assert_eq!(state["coverage"]["attempted"], 0);
    assert!(!root.join("runs/gp-worker-0.log").exists());
    assert!(!root.join("runs/withheld").exists());
}
