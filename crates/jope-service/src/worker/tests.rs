use super::*;
use std::{
    fs,
    io::{BufRead, BufReader, Write},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

// Serialize process fixtures; other service tests remain independently runnable. This mutex
// protects no state, so a prior fixture panic must not poison later fixtures.
static PROCESS_TESTS: Mutex<()> = Mutex::new(());

fn process_test_guard() -> std::sync::MutexGuard<'static, ()> {
    PROCESS_TESTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[test]
fn process_test_guard_recovers_after_a_fixture_panics() {
    let panic = std::panic::catch_unwind(|| {
        let _guard = PROCESS_TESTS.lock().unwrap();
        panic!("simulated worker fixture panic");
    });
    assert!(panic.is_err());
    drop(process_test_guard());
}

fn matrix() -> tempfile::NamedTempFile {
    matrix_rows(1)
}
fn matrix_rows(rows: usize) -> tempfile::NamedTempFile {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    let table = jope_runtime::table::CsvTable {
        headers: vec!["x".into()],
        rows: vec![vec![Some("7".into())]; rows],
        bytes: 4,
        sha256: "fixture".into(),
    };
    file.write_all(&crate::dataset::jmat::encode(&table).unwrap())
        .unwrap();
    file
}
fn request(path: &std::path::Path, slot: u32, attempt: u64, dsl: &str) -> Request {
    Request {
        attempt: Attempt {
            run: jope_api::ids::RunId::new(),
            round: 1,
            slot,
            attempt,
        },
        matrix_path: path.to_path_buf(),
        fold_spec: serde_json::json!({}),
        dsl: dsl.into(),
        objective: serde_json::json!("fixture"),
        seed: 7,
        limits: Limits {
            wall_ms: 4000,
            rss_bytes: 256 * 1024 * 1024,
        },
    }
}
fn pool(workers: usize) -> WorkerPool {
    let mut config = PoolConfig::new(std::env::current_exe().unwrap(), workers);
    config.fixture = true;
    WorkerPool::new(config).unwrap()
}
fn fixture_command(test: &str) -> Command {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command.args(["--exact", test, "--ignored", "--nocapture", "--quiet"]);
    command
}
fn marker(output: &mut impl BufRead, expected: &str) {
    let mut line = String::new();
    loop {
        line.clear();
        assert_ne!(
            output.read_line(&mut line).unwrap(),
            0,
            "fixture exited before {expected}"
        );
        if line.trim() == expected {
            return;
        }
    }
}

#[test]
#[ignore = "subprocess protocol target; invoked by process supervisor tests"]
fn child_helper() {
    assert_eq!(std::env::var("JOPE_WORKER_FIXTURE").unwrap(), "worker");
    if let Some(path) = std::env::var_os("JOPE_WORKER_PID_FILE") {
        fs::write(path, std::process::id().to_string()).unwrap();
    }
    println!("WORKER_FIXTURE_READY");
    std::io::stdout().flush().unwrap();
    if std::env::var_os("JOPE_WORKER_WRONG_FENCE").is_some() {
        let value = frame::decode_frame(&mut std::io::stdin().lock()).unwrap();
        let request: Request = serde_json::from_value(value).unwrap();
        let mut attempt = request.attempt;
        attempt.attempt += 1;
        let response = Response::failed(attempt, WorkerFailure::Unavailable, 0);
        std::io::stdout()
            .write_all(&frame::encode_frame(&serde_json::to_value(response).unwrap()).unwrap())
            .unwrap();
        std::process::exit(0);
    }
    let mut retained = Vec::<u8>::new();
    let mut count = 0;
    serve(1, |request, bytes| {
        count += 1;
        match request.dsl.as_str() {
            "loop" => loop {
                std::hint::black_box(1_u64.wrapping_add(1));
            },
            "memory" => loop {
                let start = retained.len();
                retained.resize(start + 16 * 1024 * 1024, 1);
                for page in retained[start..].chunks_mut(4096) {
                    page[0] = 2;
                }
                std::hint::black_box(&retained);
                thread::sleep(Duration::from_millis(20));
            },
            "grow" => {
                retained.resize(96 * 1024 * 1024, 1);
                for page in retained.chunks_mut(4096) {
                    page[0] = 2;
                }
            }
            "crash" | "exit_zero" => {
                let path = request.fold_spec["execution_marker"].as_str().unwrap();
                let mut marker = fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .unwrap();
                marker.write_all(b"x").unwrap();
                std::process::exit(if request.dsl == "exit_zero" { 0 } else { 70 });
            }
            "panic" => panic!("candidate fixture panic"),
            "nan" => {
                return Ok(Scores {
                    values: vec![f64::NAN],
                    fold_scores: vec![],
                    predictions_digest: "0".repeat(64),
                })
            }
            _ => {}
        }
        Ok(Scores {
            values: vec![
                std::process::id() as f64,
                bytes.rows() as f64,
                count as f64,
                bytes.bytes().as_ptr() as usize as f64,
            ],
            fold_scores: vec![],
            predictions_digest: "0".repeat(64),
        })
    })
    .unwrap();
    std::process::exit(0);
}

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

struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        super::process::kill(&mut self.0);
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

#[test]
#[ignore = "parent-death subprocess target; invoked by the supervisor"]
fn parent_helper() {
    assert_eq!(std::env::var("JOPE_WORKER_PARENT_FIXTURE").unwrap(), "1");
    let path = std::env::var_os("JOPE_WORKER_MATRIX").unwrap();
    let pool = pool(1);
    let mut value = request(std::path::Path::new(&path), 0, 1, "loop");
    value.limits.wall_ms = 60_000;
    let _ = pool.evaluate(&value);
    panic!("parent fixture should be killed by its supervisor");
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "isolated Linux subreaper fixture; invoked by parent-death test"]
#[allow(unsafe_code)]
fn parent_death_supervisor() {
    // Isolate subreaper state in this process; do not alter the main test runner or other jobs.
    // SAFETY: prctl has integer arguments only. waitpid is restricted to our recorded child.
    assert_eq!(unsafe { libc::prctl(libc::PR_SET_CHILD_SUBREAPER, 1) }, 0);
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
    let worker: i32 = loop {
        if let Ok(pid) = fs::read_to_string(&pid_path).unwrap().parse() {
            break pid;
        }
        assert!(started.elapsed() < Duration::from_secs(5));
        thread::sleep(Duration::from_millis(10));
    };
    // Allow the adversarial callback to enter its non-cooperative loop, then kill the parent.
    thread::sleep(Duration::from_millis(150));
    parent.0.kill().unwrap();
    parent.0.wait().unwrap();
    let killed = Instant::now();
    let mut status = 0;
    loop {
        let result = unsafe { libc::waitpid(worker, &mut status, libc::WNOHANG) };
        if result == worker {
            break;
        }
        assert!(
            result == 0,
            "waitpid error: {}",
            std::io::Error::last_os_error()
        );
        assert!(
            killed.elapsed() < Duration::from_secs(2),
            "worker survived parent death"
        );
        thread::sleep(Duration::from_millis(10));
    }
    assert!(libc::WIFSIGNALED(status));
    assert_eq!(libc::WTERMSIG(status), libc::SIGKILL);
    assert!(
        crate::sampler::sample(worker as u32).is_err(),
        "owned worker was not reaped"
    );
    println!("PARENT_DEATH_VERIFIED");
}

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

#[test]
#[cfg(unix)]
fn original_mapping_survives_replacement_of_its_fixture_path() {
    let _guard = process_test_guard();
    let file = matrix();
    let replacement = matrix_rows(2);
    let pool = pool(1);
    let first = pool.evaluate(&request(file.path(), 0, 1, "ok"));
    assert!(first.failure.is_none());
    // Rename replaces the directory entry, leaving the original mapped inode unmodified.
    // Production CAS paths are immutable; this fixture demonstrates retention of the mapping.
    fs::rename(replacement.path(), file.path()).unwrap();
    let second = pool.evaluate(&request(file.path(), 1, 1, "ok"));
    assert!(second.failure.is_none());
    assert_eq!(second.scores[0], first.scores[0]);
    assert_eq!(second.scores[1], 1.0);
    drop(pool);
    let reloaded = super::tests::pool(1).evaluate(&request(file.path(), 2, 1, "ok"));
    assert!(reloaded.failure.is_none());
    assert_eq!(reloaded.scores[1], 2.0);
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
