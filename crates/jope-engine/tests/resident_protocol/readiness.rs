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

/// Counts completed readiness polls through the readiness file's access time.
/// A future modification time makes relatime update access time on every client
/// read, without allocating a process-wide inotify instance.
struct ReadinessPolls {
    path: std::path::PathBuf,
    last_access: std::time::SystemTime,
}
impl ReadinessPolls {
    fn watch(pool: &std::path::Path, rank: usize) -> Self {
        let path = pool.join(format!("rank-{rank}.json"));
        Self::advance_modified_time(&path);
        let last_access = fs::metadata(&path).unwrap().accessed().unwrap();
        Self { path, last_access }
    }

    fn advance_modified_time(path: &std::path::Path) {
        let file = fs::OpenOptions::new().write(true).open(path).unwrap();
        let future = std::time::SystemTime::now() + Duration::from_secs(60 * 60);
        file.set_times(std::fs::FileTimes::new().set_modified(future))
            .unwrap();
    }

    /// Reset after publishing readiness, discarding any reads before publication settles.
    fn pending(&mut self) {
        Self::advance_modified_time(&self.path);
        self.last_access = fs::metadata(&self.path).unwrap().accessed().unwrap();
    }

    /// Blocks until `count` further polls complete; a client that stops polling
    /// within `bound` fails the test instead of hanging it.
    fn await_polls(&mut self, count: usize, bound: Duration) {
        let deadline = Instant::now() + bound;
        let mut seen = 0;
        while seen < count {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(
                !remaining.is_zero(),
                "readiness client completed {seen} of {count} polls within {bound:?}"
            );
            let accessed = fs::metadata(&self.path).unwrap().accessed().unwrap();
            if accessed > self.last_access {
                self.last_access = accessed;
                seen += 1;
            } else {
                std::thread::sleep(Duration::from_millis(2).min(remaining));
            }
        }
    }
}

/// A worker is replaced before any descriptor transfer. The client keeps
/// polling within its one original bound while the readiness it reads is
/// stale or names an endpoint that does not exist yet, and connects once the
/// replacement's endpoint appears. Every step waits on observed client polls;
/// nothing depends on timing.
#[test]
fn pretransfer_socket_readiness_race_is_retryable_within_one_deadline() {
    let d = tempfile::tempdir().unwrap();
    let pool = d.path().to_owned();
    let socket = pool.join("rank-0.sock");
    let executable = manifest::file_hash(std::path::Path::new("/proc/self/exe")).unwrap();
    // The original worker exited before transfer: its readiness names a process
    // birth that no longer exists and its endpoint is gone.
    let listener = UnixListener::bind(&socket).unwrap();
    let mut original = WorkerIdentity::capture(&socket, executable.clone()).unwrap();
    original.process.start_ticks += 1;
    publish_worker(&pool, &original);
    drop(listener);
    fs::remove_file(&socket).unwrap();
    let mut polls = ReadinessPolls::watch(&pool, 0);
    let bound = Duration::from_secs(30);
    let client_pool = pool.clone();
    let client_executable = executable.clone();
    let client = std::thread::spawn(move || {
        let started = Instant::now();
        let result = connect_rank(&client_pool, 0, &client_executable, bound);
        (result, started.elapsed())
    });
    // The client has read the stale readiness at least once.
    polls.await_polls(1, bound);
    // The replacement is bound under a staging name and published before its
    // endpoint is moved into place, so the client also meets readiness whose
    // endpoint does not exist yet. (This test's original and replacement share
    // one live process; a real replacement's predecessor has exited, which the
    // process-birth check above models.)
    let staging = pool.join("rank-0.staging.sock");
    let replacement = UnixListener::bind(&staging).unwrap();
    let identity = WorkerIdentity::capture(&staging, executable).unwrap();
    publish_worker(&pool, &identity);
    polls.pending();
    // At most one poll can have opened the stale file before publication; the
    // next two opened the replacement's readiness, so the earlier of them made
    // a complete attempt against the absent endpoint and was retried.
    polls.await_polls(3, bound);
    fs::rename(&staging, &socket).unwrap();
    let (result, elapsed) = client.join().unwrap();
    drop(replacement);
    assert!(result.is_ok(), "{result:?}");
    assert!(elapsed < bound, "{elapsed:?}");
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
