use super::*;
#[test]
fn daemon_exchange_uses_authenticated_connection_even_if_path_replaced() {
    let dir = std::env::temp_dir().join(format!("jope-native-peer-{}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    let socket = dir.join("daemon.sock");
    let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
    let identity = ProcessIdentity::current().unwrap();
    let digest = crate::manifest::file_hash(Path::new("/proc/self/exe")).unwrap();
    std::thread::scope(|scope| {
        let path = socket.clone();
        let server = scope.spawn(move || {
            let (connection, _) = listener.accept().unwrap();
            fs::remove_file(&path).unwrap();
            let _replacement = std::os::unix::net::UnixListener::bind(&path).unwrap();
            connection.set_nonblocking(true).unwrap();
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async {
                let mut connection = tokio::net::UnixStream::from_std(connection).unwrap();
                let codec = FrameCodec::default();
                let Some(ClientFrame::Request(request)) = codec
                    .read_frame::<ClientFrame, _>(&mut connection)
                    .await
                    .unwrap()
                else {
                    panic!("request absent")
                };
                codec
                    .write_frame(
                        &mut connection,
                        &ServerFrame::Response(Box::new(jain_smartcluster::ResponseEnvelope::ok(
                            request.request_id,
                            Response::Job(None),
                        ))),
                    )
                    .await
                    .unwrap();
            });
        });
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let result = runtime.block_on(async {
            authenticated_exchange(
                &socket,
                // SAFETY: geteuid has no preconditions and returns a scalar.
                unsafe { libc::geteuid() },
                &digest,
                Some(&identity),
                Request::Get(GetRequest {
                    job_id: JobId::new(),
                }),
            )
            .unwrap()
        });
        assert!(matches!(result, Response::Job(None)));
        server.join().unwrap();
    });
    fs::remove_dir_all(dir).unwrap();
}
#[test]
fn daemon_exchange_rejects_wrong_executable_before_sending_request() {
    let dir = std::env::temp_dir().join(format!("jope-native-wrong-peer-{}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    let socket = dir.join("daemon.sock");
    let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
    let (finished_tx, finished_rx) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        let server = scope.spawn(move || {
            use std::io::Read;
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                .unwrap();
            // Hashing the real debug image can exceed five seconds under I/O load.
            // Begin the no-request assertion only after authentication returned.
            finished_rx.recv().unwrap();
            let mut byte = [0];
            assert_eq!(stream.read(&mut byte).unwrap(), 0);
        });
        assert!(authenticated_exchange(
            &socket,
            // SAFETY: geteuid has no preconditions and returns a scalar.
            unsafe { libc::geteuid() },
            &"0".repeat(64),
            None,
            Request::Get(GetRequest {
                job_id: JobId::new()
            })
        )
        .is_err());
        finished_tx.send(()).unwrap();
        server.join().unwrap();
    });
    fs::remove_dir_all(dir).unwrap();
}
#[test]
fn requester_ancestry_binds_original_leader_birth_and_rejects_sibling() {
    let leader = ProcessIdentity::current().unwrap();
    let mut child = std::process::Command::new("/bin/sleep")
        .arg("5")
        .spawn()
        .unwrap();
    let mut sibling = std::process::Command::new("/bin/sleep")
        .arg("5")
        .spawn()
        .unwrap();
    let requester = ProcessIdentity::read(child.id()).unwrap();
    let other = ProcessIdentity::read(sibling.id()).unwrap();
    verify_ancestry(&requester, &leader).unwrap();
    assert!(verify_ancestry(&requester, &other).is_err());
    let mut stale = leader.clone();
    stale.birth_ticks += 1;
    assert!(verify_ancestry(&requester, &stale).is_err());
    let mut stale = requester.clone();
    stale.boot_id = "foreign-boot".into();
    assert!(verify_ancestry(&stale, &leader).is_err());
    child.kill().unwrap();
    child.wait().unwrap();
    assert!(verify_ancestry(&requester, &leader).is_err());
    sibling.kill().unwrap();
    sibling.wait().unwrap();
}
#[test]
fn inspected_finite_controller_preserves_original_leader_and_rejects_changed_fences() {
    use jain_smartcluster::{
        AttemptInspection, AttemptLease, AttemptNumber, AttemptRecord, FenceToken,
        NativeProcessIdentity, Timestamp,
    };
    let own = ProcessIdentity::current().unwrap();
    let attempt = FencedAttempt::new(
        JobId::new(),
        AttemptNumber::new(1).unwrap(),
        FenceToken::new(1).unwrap(),
    );
    let record = AttemptRecord {
        attempt,
        state: JobState::Running,
        lease: AttemptLease {
            attempt,
            acquired_at: Timestamp::from(1),
            expires_at: Timestamp::from(2),
        },
        pid: Some(own.pid),
        cgroup_path: Some(cgroup(own.pid).unwrap().to_string_lossy().into_owned()),
        started_at: Some(Timestamp::from(1)),
        finished_at: None,
        failure: None,
    };
    let inspection = AttemptInspection {
        record,
        active_process: Some(NativeProcessIdentity {
            pid: own.pid,
            birth_ticks: own.birth_ticks,
            boot_id: own.boot_id.clone(),
        }),
    };
    assert_eq!(inspected_process(inspection.clone(), attempt).unwrap(), own);
    // A replacement daemon may adopt this same finite controller; only the
    // unchanged original process/fence is accepted, never a replacement.
    assert_eq!(inspected_process(inspection.clone(), attempt).unwrap(), own);
    let stale = FencedAttempt::new(attempt.job_id, attempt.number, FenceToken::new(2).unwrap());
    assert!(inspected_process(inspection.clone(), stale).is_err());
    let mut changed = inspection.clone();
    changed.active_process.as_mut().unwrap().birth_ticks += 1;
    assert!(inspected_process(changed, attempt).is_err());
    let mut changed = inspection.clone();
    changed.record.state = JobState::Paused;
    assert!(inspected_process(changed, attempt).is_err());
    let mut changed = inspection.clone();
    changed.active_process = None;
    assert!(inspected_process(changed, attempt).is_err());
    let mut changed = inspection;
    changed.record.pid = Some(own.pid + 1);
    assert!(inspected_process(changed, attempt).is_err());
}
