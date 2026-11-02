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

/// Counts completed readiness polls. Every `connect_rank` attempt reads
/// `rank-<n>.json` and closes it without writing (inotify `IN_CLOSE_NOWRITE`),
/// so a test can act after a known number of client attempts instead of after
/// a guessed delay.
struct ReadinessPolls {
    inotify: std::os::fd::OwnedFd,
    name: std::ffi::OsString,
}
impl ReadinessPolls {
    fn watch(pool: &std::path::Path, rank: usize) -> Self {
        use std::os::{fd::FromRawFd, unix::ffi::OsStrExt};
        let fd = unsafe { libc::inotify_init1(libc::IN_CLOEXEC | libc::IN_NONBLOCK) };
        assert!(fd >= 0, "inotify: {}", std::io::Error::last_os_error());
        let inotify = unsafe { std::os::fd::OwnedFd::from_raw_fd(fd) };
        let directory = std::ffi::CString::new(pool.as_os_str().as_bytes()).unwrap();
        let watch =
            unsafe { libc::inotify_add_watch(fd, directory.as_ptr(), libc::IN_CLOSE_NOWRITE) };
        assert!(
            watch >= 0,
            "inotify watch: {}",
            std::io::Error::last_os_error()
        );
        Self {
            inotify,
            name: format!("rank-{rank}.json").into(),
        }
    }
    /// Read-only closes of the readiness file among the pending events.
    fn pending(&self) -> usize {
        use std::os::{fd::AsRawFd, unix::ffi::OsStrExt};
        let header = std::mem::size_of::<libc::inotify_event>();
        let mut buffer = [0u8; 4096];
        let mut polls = 0;
        loop {
            let read = unsafe {
                libc::read(
                    self.inotify.as_raw_fd(),
                    buffer.as_mut_ptr().cast(),
                    buffer.len(),
                )
            };
            if read < 0 {
                let error = std::io::Error::last_os_error();
                assert_eq!(error.kind(), std::io::ErrorKind::WouldBlock, "{error}");
                return polls;
            }
            let mut offset = 0;
            while offset + header <= read as usize {
                let event: libc::inotify_event =
                    unsafe { std::ptr::read_unaligned(buffer.as_ptr().add(offset).cast()) };
                let name = &buffer[offset + header..offset + header + event.len as usize];
                let name = &name[..name.iter().position(|b| *b == 0).unwrap_or(name.len())];
                if event.mask & libc::IN_CLOSE_NOWRITE != 0 && name == self.name.as_bytes() {
                    polls += 1;
                }
                offset += header + event.len as usize;
            }
        }
    }
    /// Blocks until `count` further polls complete; a client that stops polling
    /// within `bound` fails the test instead of hanging it.
    fn await_polls(&self, count: usize, bound: Duration) {
        use std::os::fd::AsRawFd;
        let deadline = Instant::now() + bound;
        let mut seen = 0;
        while seen < count {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(
                !remaining.is_zero(),
                "readiness client completed {seen} of {count} polls within {bound:?}"
            );
            let mut ready = libc::pollfd {
                fd: self.inotify.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            let timeout = remaining.as_millis().clamp(1, i32::MAX as u128) as i32;
            if unsafe { libc::poll(&mut ready, 1, timeout) } < 0 {
                let error = std::io::Error::last_os_error();
                assert_eq!(error.kind(), std::io::ErrorKind::Interrupted, "{error}");
            }
            seen += self.pending();
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
    let polls = ReadinessPolls::watch(&pool, 0);
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
