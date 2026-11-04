use super::*;

#[test]
fn timeout_and_memory_breach_are_bounded_and_the_run_continues() {
    let _guard = process_test_guard();
    let file = matrix();
    let pool = pool(1);
    assert!(pool
        .evaluate(&request(file.path(), 0, 1, "ok"))
        .failure
        .is_none());
    let mut hang = request(file.path(), 1, 1, "loop");
    hang.limits.wall_ms = 200;
    let started = Instant::now();
    assert_eq!(pool.evaluate(&hang).failure, Some(WorkerFailure::Timeout));
    assert!(started.elapsed() < Duration::from_millis(2200));
    let before_breach = pool.evaluate(&request(file.path(), 2, 1, "ok"));
    assert!(before_breach.failure.is_none(), "{before_breach:?}");
    let mut bomb = request(file.path(), 3, 1, "memory");
    bomb.limits.rss_bytes = 96 * 1024 * 1024;
    let started = Instant::now();
    assert_eq!(
        pool.evaluate(&bomb).failure,
        Some(WorkerFailure::MemoryLimit)
    );
    assert!(started.elapsed() < Duration::from_secs(3));
    let after_breach = pool.evaluate(&request(file.path(), 4, 1, "ok"));
    assert!(after_breach.failure.is_none(), "{after_breach:?}");
    assert_ne!(
        before_breach.scores[0], after_breach.scores[0],
        "the worker must be replaced after exceeding its RSS limit"
    );
    eprintln!(
        "WRK01_R2 Linux timeout=200ms memorycap=96MiB workerBefore={} workerAfter={}",
        before_breach.scores[0], after_breach.scores[0]
    );
}

#[test]
fn mmap_is_reused_and_the_501st_evaluation_gets_a_new_process() {
    let _guard = process_test_guard();
    let file = matrix();
    let pool = pool(1);
    let first = pool.evaluate(&request(file.path(), 0, 1, "ok"));
    assert!(first.failure.is_none(), "{first:?}");
    for evaluation in 2..=500 {
        let response = pool.evaluate(&request(file.path(), evaluation, 1, "ok"));
        assert!(response.failure.is_none(), "{evaluation}: {response:?}");
        assert_eq!(response.scores[0], first.scores[0]);
        assert_eq!(response.scores[2], f64::from(evaluation));
        assert_eq!(response.scores[3], first.scores[3]);
    }
    let next = pool.evaluate(&request(file.path(), 501, 1, "ok"));
    assert!(next.failure.is_none(), "{next:?}");
    assert_ne!(next.scores[0], first.scores[0]);
    assert_eq!(next.scores[2], 1.0);
}

#[test]
fn rss_growth_recycles_and_matrix_path_changes_replace_the_worker() {
    let _guard = process_test_guard();
    let file = matrix();
    let another = matrix();
    let pool = pool(1);
    let first = pool.evaluate(&request(file.path(), 0, 1, "ok"));
    assert!(first.failure.is_none());
    let growth = pool.evaluate(&request(file.path(), 1, 1, "grow"));
    assert!(growth.failure.is_none(), "{growth:?}");
    let next = pool.evaluate(&request(file.path(), 2, 1, "ok"));
    assert!(next.failure.is_none(), "{next:?}");
    assert_ne!(next.scores[0], first.scores[0]);
    let changed = pool.evaluate(&request(another.path(), 3, 1, "ok"));
    assert!(changed.failure.is_none());
    assert_ne!(changed.scores[0], next.scores[0]);
}

#[test]
fn newer_and_revoked_attempts_fence_running_and_queued_work() {
    let _guard = process_test_guard();
    let file = matrix();
    let pool = Arc::new(pool(2));
    let old = request(file.path(), 0, 1, "loop");
    let mut new = old.clone();
    new.attempt.attempt = 2;
    new.dsl = "ok".into();
    thread::scope(|scope| {
        let old_result = scope.spawn(|| pool.evaluate(&old));
        thread::sleep(Duration::from_millis(150));
        assert!(pool.evaluate(&new).failure.is_none());
        assert_eq!(
            old_result.join().unwrap().failure,
            Some(WorkerFailure::StaleAttempt)
        );
    });
    assert_eq!(
        pool.evaluate(&new).failure,
        Some(WorkerFailure::StaleAttempt)
    );
    let revoked = request(file.path(), 1, 1, "loop");
    thread::scope(|scope| {
        let result = scope.spawn(|| pool.evaluate(&revoked));
        thread::sleep(Duration::from_millis(150));
        pool.revoke(&revoked.attempt);
        assert_eq!(
            result.join().unwrap().failure,
            Some(WorkerFailure::StaleAttempt)
        );
    });
    assert!(pool
        .evaluate(&request(file.path(), 2, 1, "ok"))
        .failure
        .is_none());
}

#[test]
fn panic_nonfinite_and_bad_matrix_fail_only_the_candidate() {
    let _guard = process_test_guard();
    let file = matrix();
    let pool = pool(1);
    assert_eq!(
        pool.evaluate(&request(file.path(), 1, 1, "panic")).failure,
        Some(WorkerFailure::Panicked)
    );
    assert_eq!(
        pool.evaluate(&request(file.path(), 2, 1, "nan")).failure,
        Some(WorkerFailure::NonFinite)
    );
    let missing = request(&file.path().with_extension("absent"), 3, 1, "ok");
    assert_eq!(
        pool.evaluate(&missing).failure,
        Some(WorkerFailure::InvalidRequest)
    );
    assert!(pool
        .evaluate(&request(file.path(), 4, 1, "ok"))
        .failure
        .is_none());
}

#[test]
fn closing_stdin_terminates_even_a_looping_candidate() {
    let _guard = process_test_guard();
    let file = matrix();
    let mut command = fixture_command("worker::tests::child_helper");
    command
        .env("JOPE_WORKER_FIXTURE", "worker")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    super::process::prepare(&mut command);
    let mut child = ChildGuard(command.spawn().unwrap());
    let mut output = BufReader::new(child.0.stdout.take().unwrap());
    marker(&mut output, "WORKER_FIXTURE_READY");
    let bytes =
        frame::encode_frame(&serde_json::to_value(request(file.path(), 1, 1, "loop")).unwrap())
            .unwrap();
    let mut input = child.0.stdin.take().unwrap();
    input.write_all(&bytes).unwrap();
    input.flush().unwrap();
    thread::sleep(Duration::from_millis(100));
    drop(input);
    let started = Instant::now();
    while child.0.try_wait().unwrap().is_none() {
        assert!(started.elapsed() < Duration::from_secs(2));
        thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn rejects_invalid_plan_and_zero_limits() {
    assert!(WorkerPool::new(PoolConfig::new("jopedime".into(), 0)).is_err());
    let pool = pool(1);
    let file = matrix();
    let mut value = request(file.path(), 0, 1, "ok");
    value.limits.wall_ms = 0;
    assert_eq!(
        pool.evaluate(&value).failure,
        Some(WorkerFailure::InvalidRequest)
    );
}
