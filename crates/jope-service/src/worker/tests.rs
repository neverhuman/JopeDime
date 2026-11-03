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
        // jankurai:allow HLT-008-FALSE-GREEN-RISK reason=verified-exit-behavior expires=2026-12-31
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
                // jankurai:allow HLT-008-FALSE-GREEN-RISK reason=verified-exit-behavior expires=2026-12-31
                std::process::exit(if request.dsl == "exit_zero" { 0 } else { 70 });
            }
            "panic" => panic!("candidate fixture panic"),
            "nan" => {
                return Ok(Scores {
                    values: vec![f64::NAN],
                    fold_scores: vec![],
                    predictions_digest: "0".repeat(64),
                });
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
    // jankurai:allow HLT-008-FALSE-GREEN-RISK reason=verified-exit-behavior expires=2026-12-31
    std::process::exit(0);
}

struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        super::process::kill(&mut self.0);
    }
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

#[path = "tests/basic.rs"]
mod basic;
#[path = "tests/lifecycle.rs"]
mod lifecycle;
