use super::*;

#[cfg(target_os = "linux")]
#[test]
fn killing_parent_leaves_no_worker_process() {
    let _guard = process_test_guard();
    let output = fixture_command("worker::tests::parent_death_supervisor")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{} {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("PARENT_DEATH_VERIFIED"));
}

#[test]
fn dropping_the_pool_reaps_its_idle_children() {
    let _guard = process_test_guard();
    let file = matrix();
    let pool = pool(1);
    let response = pool.evaluate(&request(file.path(), 0, 1, "ok"));
    assert!(response.failure.is_none());
    let pid = response.scores[0] as u32;
    assert!(crate::sampler::sample(pid).is_ok());
    drop(pool);
    assert!(crate::sampler::sample(pid).is_err());
}

#[test]
fn wrong_reply_fence_is_rejected_before_publication() {
    let _guard = process_test_guard();
    let mut config = PoolConfig::new(std::env::current_exe().unwrap(), 1);
    config.fixture = true;
    config.wrong_fence = true;
    let pool = WorkerPool::new(config).unwrap();
    let file = matrix();
    assert_eq!(
        pool.evaluate(&request(file.path(), 0, 1, "ok")).failure,
        Some(WorkerFailure::Protocol)
    );
}

#[test]
fn queued_supersession_does_not_consume_a_worker_or_publish_a_score() {
    let _guard = process_test_guard();
    let file = matrix();
    let pool = pool(1);
    let occupying = request(file.path(), 0, 1, "loop");
    let waiting = request(file.path(), 1, 1, "ok");
    let mut replacement = waiting.clone();
    replacement.attempt.attempt = 2;
    thread::scope(|scope| {
        let busy = scope.spawn(|| pool.evaluate(&occupying));
        thread::sleep(Duration::from_millis(100));
        let old = scope.spawn(|| pool.evaluate(&waiting));
        thread::sleep(Duration::from_millis(100));
        let new = scope.spawn(|| pool.evaluate(&replacement));
        assert_eq!(
            old.join().unwrap().failure,
            Some(WorkerFailure::StaleAttempt)
        );
        pool.revoke(&occupying.attempt);
        assert_eq!(
            busy.join().unwrap().failure,
            Some(WorkerFailure::StaleAttempt)
        );
        assert!(new.join().unwrap().failure.is_none());
    });
}

#[cfg(any(windows, target_os = "macos"))]
#[test]
fn killing_parent_leaves_no_worker_process() {
    let _guard = process_test_guard();
    let file = matrix();
    let dir = tempfile::tempdir().unwrap();
    let pid_path = dir.path().join("worker.pid");
    let mut command = fixture_command("worker::tests::parent_helper");
    command
        .env("JOPE_WORKER_PARENT_FIXTURE", "1")
        .env("JOPE_WORKER_MATRIX", file.path())
        .env("JOPE_WORKER_PID_FILE", &pid_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let mut parent = ChildGuard(command.spawn().unwrap());
    let started = Instant::now();
    while !pid_path.exists() {
        assert!(started.elapsed() < Duration::from_secs(5));
        thread::sleep(Duration::from_millis(10));
    }
    let worker: u32 = loop {
        if let Ok(pid) = fs::read_to_string(&pid_path).unwrap().parse() {
            break pid;
        }
        assert!(started.elapsed() < Duration::from_secs(5));
        thread::sleep(Duration::from_millis(10));
    };
    thread::sleep(Duration::from_millis(150));
    parent.0.kill().unwrap();
    parent.0.wait().unwrap();
    let killed = Instant::now();
    while crate::sampler::sample(worker).is_ok() {
        assert!(
            killed.elapsed() < Duration::from_secs(2),
            "worker survived parent death"
        );
        thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn crash_is_not_reexecuted_under_the_same_controller_attempt() {
    let _guard = process_test_guard();
    let file = matrix();
    let pool = pool(1);
    let dir = tempfile::tempdir().unwrap();
    for mode in ["crash", "exit_zero"] {
        assert!(pool
            .evaluate(&request(file.path(), 0, 1, "ok"))
            .failure
            .is_none());
        let marker = dir.path().join(mode);
        let mut value = request(file.path(), 1, 1, mode);
        value.fold_spec = serde_json::json!({"execution_marker": marker});
        assert_eq!(
            pool.evaluate(&value).failure,
            Some(WorkerFailure::ProcessExited)
        );
        assert_eq!(
            fs::read(&marker).unwrap(),
            b"x",
            "a crashed evaluation must not be retried under its old attempt: {mode}"
        );
    }
}
