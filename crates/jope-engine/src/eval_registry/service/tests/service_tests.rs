use super::*;

#[test]
fn sealed_fd_and_absolute_partial_frame_validation() {
    let file = sealed(b"verified bytes").unwrap();
    assert_eq!(read_sealed(&file, 100).unwrap(), b"verified bytes");
    assert_eq!(unsafe { libc::ftruncate(file.as_raw_fd(), 0) }, -1);
    let temp = tempfile::tempfile().unwrap();
    assert!(read_sealed(&temp, 100).is_err());
    let (mut a, mut b) = UnixStream::pair().unwrap();
    b.write_all(&[0, 0]).unwrap();
    let began = Instant::now();
    assert!(receive_json::<Value>(&mut a, Duration::from_millis(20)).is_err());
    assert!(began.elapsed() < Duration::from_secs(1));
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn service_reuses_one_connection_and_104_clients_start_once() {
    let fixture = Fixture::new();
    let (stop, service) = fixture.launch();
    // Warm the immutable executable authentication cache before contention.
    assert_eq!(
        stats(&fixture.socket, &fixture.root, &fixture.witness)
            .unwrap()
            .authority_opens,
        1
    );
    let barrier = Arc::new(Barrier::new(104));
    let clients: Vec<_> = (0..104)
        .map(|_| {
            let socket = fixture.socket.clone();
            let root = fixture.root.clone();
            let witness = fixture.witness.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                submit_to(&socket, &root, &witness, &inputs(1), provenance())
            })
        })
        .collect();
    let mut completed = Vec::new();
    for client in clients {
        if let Ok(result) = client.join().unwrap() {
            completed.push(result);
        }
    }
    assert!(!completed.is_empty());
    let original = &completed[0];
    assert_eq!(original.receipt.outcome, ExecutionOutcome::Success);
    assert!(original.model_bytes().is_ok());
    let reused = submit_to(
        &fixture.socket,
        &fixture.root,
        &fixture.witness,
        &inputs(1),
        provenance(),
    )
    .unwrap();
    assert!(reused.reused);
    assert_eq!(reused.completion_sha256, original.completion_sha256);
    let verified = verify_original(
        &fixture.socket,
        &fixture.root,
        &fixture.witness,
        &original.completion_sha256,
        "synthetic",
    )
    .unwrap();
    assert_eq!(verified.fit_key, original.fit_key);
    assert!(verify_original(
        &fixture.socket,
        &fixture.root,
        &fixture.witness,
        &original.completion_sha256,
        "sealed_test"
    )
    .is_err());
    stop.store(true, Ordering::SeqCst);
    let stats = service.join().unwrap().unwrap();
    assert_eq!(stats.authority_opens, 1);
    assert_eq!(stats.worker_spawns, 1);
    assert_eq!(stats.committed, 1);
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn lost_submit_response_still_commits_original_without_replay() {
    let fixture = Fixture::new();
    let (stop, service) = fixture.launch();
    let (mut client, _) =
        connect_approved(&fixture.socket, &fixture.root, &fixture.witness).unwrap();
    let bytes = serde_json::to_vec(&inputs(1)).unwrap();
    let fd = sealed(&bytes).unwrap();
    send_json(
        &mut client,
        &Request::Submit {
            provenance: provenance(),
            descriptor_sha256: sha(&bytes),
        },
        Duration::from_secs(5),
    )
    .unwrap();
    send_fd(&client, fd.as_raw_fd(), Duration::from_secs(5)).unwrap();
    drop(client);
    let deadline = Instant::now() + Duration::from_secs(100);
    loop {
        let stats = stats(&fixture.socket, &fixture.root, &fixture.witness).unwrap();
        if stats.committed == 1 {
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(20));
    }
    let reused = submit_to(
        &fixture.socket,
        &fixture.root,
        &fixture.witness,
        &inputs(1),
        provenance(),
    )
    .unwrap();
    assert!(reused.reused);
    stop.store(true, Ordering::SeqCst);
    let stats = service.join().unwrap().unwrap();
    assert_eq!(stats.worker_spawns, 1);
    assert_eq!(stats.committed, 1);
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn failed_signaled_timed_out_and_replaced_custody_are_terminal() {
    let mut fixture = Fixture::new();
    // Leave bounded room for debug-binary authentication and custody transfer:
    // the start token must still be live when the service consumes it, which a
    // two-second cap missed on a loaded host. The timeout worker deliberately
    // sleeps 30 seconds, so a ten-second cap still exercises numerical timeout
    // rather than a pre-consumption refusal.
    fixture.config.fit_deadline = Duration::from_secs(10);
    // The replaced-custody child deliberately rewrites its own custody file.
    fixture.config.sandbox.mutable_dirs = vec![fixture.root.join("custody")];
    let (stop, service) = fixture.launch();
    for alpha in [2, 3, 4, 5] {
        let original = submit_to(
            &fixture.socket,
            &fixture.root,
            &fixture.witness,
            &inputs(alpha),
            provenance(),
        )
        .unwrap();
        assert_ne!(original.receipt.outcome, ExecutionOutcome::Success);
        if alpha == 4 {
            assert_eq!(original.receipt.outcome, ExecutionOutcome::Timeout);
        }
        let reused = submit_to(
            &fixture.socket,
            &fixture.root,
            &fixture.witness,
            &inputs(alpha),
            provenance(),
        )
        .unwrap();
        assert!(reused.reused);
        assert_eq!(reused.completion_sha256, original.completion_sha256);
    }
    stop.store(true, Ordering::SeqCst);
    let stats = service.join().unwrap().unwrap();
    assert_eq!(stats.worker_spawns, 4);
    assert_eq!(stats.committed, 4);
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn original_fit_child_is_confined_while_fitting() {
    let fixture = Fixture::new();
    let (stop, service) = fixture.launch();
    let original = submit_to(
        &fixture.socket,
        &fixture.root,
        &fixture.witness,
        &inputs(6),
        provenance(),
    )
    .unwrap();
    assert_eq!(original.receipt.outcome, ExecutionOutcome::Success);
    stop.store(true, Ordering::SeqCst);
    assert_eq!(service.join().unwrap().unwrap().worker_spawns, 1);
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn lost_start_permission_remains_consumed_and_terminal() {
    let mut fixture = Fixture::new();
    fixture.config.drop_start_reply = true;
    let (stop, service) = fixture.launch();
    let result = submit_to(
        &fixture.socket,
        &fixture.root,
        &fixture.witness,
        &inputs(1),
        provenance(),
    )
    .unwrap();
    assert_ne!(result.receipt.outcome, ExecutionOutcome::Success);
    let reused = submit_to(
        &fixture.socket,
        &fixture.root,
        &fixture.witness,
        &inputs(1),
        provenance(),
    )
    .unwrap();
    assert!(reused.reused);
    stop.store(true, Ordering::SeqCst);
    assert_eq!(service.join().unwrap().unwrap().worker_spawns, 1);
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn missing_authority_and_replaced_socket_cannot_start_service() {
    let fixture = Fixture::new();
    let (stop, service) = fixture.launch();
    assert!(Service::bind(fixture.config.clone()).is_err());
    let old = fixture.socket.with_extension("old");
    fs::rename(&fixture.socket, &old).unwrap();
    let _replacement = UnixListener::bind(&fixture.socket).unwrap();
    assert!(stats(&fixture.socket, &fixture.root, &fixture.witness).is_err());
    stop.store(true, Ordering::SeqCst);
    assert!(service.join().unwrap().is_err());
    let absent = fixture.temp.path().join("missing-root");
    let mut config = fixture.config.clone();
    config.authority_root = absent.clone();
    assert!(Service::bind(config).is_err());
    assert!(!absent.exists());
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn forged_executor_ticket_does_not_consume_any_fit() {
    let fixture = Fixture::new();
    let (stop, service) = fixture.launch();
    let (mut client, _) =
        connect_approved(&fixture.socket, &fixture.root, &fixture.witness).unwrap();
    send_json(
        &mut client,
        &Request::Consume {
            ticket: "forged".into(),
            fit_key: "made-up".into(),
        },
        Duration::from_secs(5),
    )
    .unwrap();
    let response: Value = receive_json(&mut client, Duration::from_secs(5)).unwrap();
    assert_eq!(response["ok"], false);
    stop.store(true, Ordering::SeqCst);
    let stats = service.join().unwrap().unwrap();
    assert_eq!(stats.worker_spawns, 0);
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn backlog_connect_and_invalid_request_deadlines_are_bounded() {
    let fixture = Fixture::new();
    let busy_path = fixture.temp.path().join("busy.sock");
    let busy = UnixListener::bind(&busy_path).unwrap();
    assert_eq!(unsafe { libc::listen(busy.as_raw_fd(), 0) }, 0);
    let _queued = connect_bounded(&busy_path, Duration::from_millis(100)).unwrap();
    let began = Instant::now();
    assert!(connect_bounded(&busy_path, Duration::from_millis(25)).is_err());
    assert!(began.elapsed() < Duration::from_secs(1));
    let (stop, service) = fixture.launch();
    for deadline in [Value::Null, json!(0), json!("123"), json!(-1)] {
        let mut request = provenance();
        request.details["deadline_monotonic_ns"] = deadline;
        assert!(submit_to(
            &fixture.socket,
            &fixture.root,
            &fixture.witness,
            &inputs(1),
            request
        )
        .is_err());
    }
    stop.store(true, Ordering::SeqCst);
    let result = service.join().unwrap().unwrap();
    assert_eq!(result.worker_spawns, 0);
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn socket_loss_during_worker_readiness_fences_authorization_and_restart() {
    let mut fixture = Fixture::new();
    let pause = Arc::new(TestPause::default());
    fixture.config.before_authorize = Some(pause.clone());
    let (stop, service) = fixture.launch();
    let socket = fixture.socket.clone();
    let root = fixture.root.clone();
    let witness = fixture.witness.clone();
    let client =
        std::thread::spawn(move || submit_to(&socket, &root, &witness, &inputs(1), provenance()));
    let deadline = Instant::now() + Duration::from_secs(100);
    while !pause.reached.load(Ordering::SeqCst) {
        assert!(
            Instant::now() < deadline,
            "original worker never became ready"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    fs::rename(&fixture.socket, fixture.socket.with_extension("old")).unwrap();
    let replacement = UnixListener::bind(&fixture.socket).unwrap();
    stop.store(true, Ordering::SeqCst);
    pause.release.store(true, Ordering::SeqCst);
    assert!(client.join().unwrap().is_err());
    assert!(service.join().unwrap().is_err());
    let db = rusqlite::Connection::open(fixture.root.join("registry.sqlite")).unwrap();
    let counts: (i64,i64) = db.query_row("SELECT COUNT(*),SUM(CASE WHEN state='claimed' THEN 1 ELSE 0 END) FROM entries WHERE state!='completed'", [], |row| Ok((row.get(0)?,row.get(1)?))).unwrap();
    assert_eq!(counts, (1, 1));
    drop(db);
    drop(replacement);
    fs::remove_file(&fixture.socket).unwrap();
    fs::remove_file(identity_path(&fixture.socket)).unwrap();
    assert!(Service::bind(fixture.config.clone()).is_err());
}
