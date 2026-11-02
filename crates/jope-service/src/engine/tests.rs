use super::*;
use std::sync::Barrier;

/// Utility for verifying that a run remains active until its owner drops the guard.
#[cfg(test)]
fn _active_run(activity: &Arc<EngineActivity>) -> ActivityGuard {
    activity.begin_run()
}

#[cfg(feature = "research")]
#[test]
fn q1_release_refusal_precedes_home_creation() {
    let directory = tempfile::tempdir().unwrap();
    let home = directory.path().join("engine");
    let release = directory.path().join("untrusted");
    for present in [false, true] {
        if present {
            std::fs::create_dir(&release).unwrap();
            std::fs::write(release.join("release-catalog.json"), b"untrusted").unwrap();
        }
        let error = match open_q1_engine(&home, &release) {
            Ok(_) => panic!("untrusted r782 release admitted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("frozen r782 integrity"));
        assert!(!home.exists());
    }
}

#[test]
fn only_one_process_can_hold_the_home_lock_and_busy_reports_its_owner() {
    let home = tempfile::tempdir().unwrap();
    let owner = EngineLock::acquire(home.path()).unwrap();
    let busy = EngineLock::acquire(home.path()).unwrap_err();
    assert!(matches!(busy, EngineError::Busy { pid: Some(pid), .. } if pid == std::process::id()));
    assert!(owner.is_held());
}

#[cfg(unix)]
#[test]
fn lock_refuses_a_symlink_instead_of_following_it() {
    use std::os::unix::fs::symlink;

    let home = tempfile::tempdir().unwrap();
    let other = tempfile::tempdir().unwrap();
    let target = other.path().join("target");
    fs::write(&target, "untouched").unwrap();
    symlink(&target, home.path().join("lock")).unwrap();
    assert!(EngineLock::acquire(home.path()).is_err());
    assert_eq!(fs::read_to_string(target).unwrap(), "untouched");
}

#[test]
fn in_process_engine_takes_the_same_lock_and_opens_recovered_store_without_daemon_file() {
    let home = tempfile::tempdir().unwrap();
    let engine = InProcessEngine::open(home.path()).unwrap();
    assert!(engine._lock.is_held());
    assert!(!home.path().join("daemon.json").exists());
    assert!(EngineLock::acquire(home.path()).is_err());
}

#[test]
fn active_runs_and_open_tcp_clients_prevent_idle_exit() {
    let activity = Arc::new(EngineActivity::default());
    assert!(!activity.idle_for(Duration::from_secs(1)));
    let run = _active_run(&activity);
    assert!(!activity.idle_for(Duration::from_secs(1)));
    drop(run);
    assert!(!activity.idle_for(Duration::from_secs(1)));
    let client = activity.connected_client();
    assert!(!activity.idle_for(Duration::ZERO));
    drop(client);
    assert!(!activity.idle_for(Duration::from_secs(1)));
}

#[test]
fn idle_exit_duration_parser_is_bounded_and_requires_units() {
    assert_eq!(parse_duration("2s").unwrap(), Duration::from_secs(2));
    assert_eq!(parse_duration("30m").unwrap(), Duration::from_secs(1800));
    assert!(parse_duration("0s").is_err());
    assert!(parse_duration("2").is_err());
    assert!(parse_duration("-1m").is_err());
}

#[tokio::test]
async fn bound_engine_publishes_then_authenticated_stop_shuts_it_down() {
    let home = tempfile::tempdir().unwrap();
    let engine = InProcessEngine::open(home.path()).unwrap();
    let server_home = home.path().to_path_buf();
    let server = tokio::spawn(serve_locked(
        server_home.clone(),
        engine,
        Duration::from_secs(60),
    ));
    let record = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if let Ok(DaemonFileStatus::Running(record)) =
                daemon_file::read_daemon_file(&server_home)
            {
                if request_daemon_async(&record, "GET", "/v1/system")
                    .await
                    .is_ok()
                {
                    break record;
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
    request_daemon_async(&record, "POST", STOP_PATH)
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(5), server)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(!server_home.join("daemon.json").exists());
}

#[tokio::test]
async fn unauthorized_stop_cannot_signal_shutdown() {
    let home = tempfile::tempdir().unwrap();
    let engine = InProcessEngine::open(home.path()).unwrap();
    let server_home = home.path().to_path_buf();
    let server = tokio::spawn(serve_locked(
        server_home.clone(),
        engine,
        Duration::from_secs(60),
    ));
    let record = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if let Ok(DaemonFileStatus::Running(record)) =
                daemon_file::read_daemon_file(&server_home)
            {
                break record;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
    let authority = record.url.strip_prefix("http://").unwrap();
    let port = authority
        .rsplit(':')
        .next()
        .unwrap()
        .parse::<u16>()
        .unwrap();
    let mut stream = TcpStream::connect((std::net::Ipv4Addr::LOCALHOST, port))
        .await
        .unwrap();
    stream.write_all(format!("POST {STOP_PATH} HTTP/1.1\r\nHost: {authority}\r\nAuthorization: Bearer {}\r\nConnection: close\r\n\r\n", "00".repeat(32)).as_bytes()).await.unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).await.unwrap();
    assert!(String::from_utf8_lossy(&response).starts_with("HTTP/1.1 401"));
    assert!(!server.is_finished());
    let record = daemon_file::read_daemon_file(&server_home).unwrap();
    let DaemonFileStatus::Running(record) = record else {
        panic!("daemon stopped unexpectedly")
    };
    request_daemon_async(&record, "POST", STOP_PATH)
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(5), server)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn two_second_idle_exit_waits_for_active_runs_and_connected_clients() {
    let home = tempfile::tempdir().unwrap();
    let engine = InProcessEngine::open(home.path()).unwrap();
    let activity = Arc::new(EngineActivity::default());
    let server_home = home.path().to_path_buf();
    let server_activity = Arc::clone(&activity);
    let server = tokio::spawn(serve_with_activity(
        server_home.clone(),
        engine,
        Duration::from_secs(2),
        server_activity,
    ));
    let record = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if let Ok(DaemonFileStatus::Running(record)) =
                daemon_file::read_daemon_file(&server_home)
            {
                break record;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();

    let active_run = activity.begin_run();
    tokio::time::sleep(Duration::from_millis(2200)).await;
    assert!(!server.is_finished(), "active run must suppress idle exit");
    drop(active_run);

    let authority = record.url.strip_prefix("http://").unwrap();
    let port = authority
        .rsplit(':')
        .next()
        .unwrap()
        .parse::<u16>()
        .unwrap();
    let client = TcpStream::connect((std::net::Ipv4Addr::LOCALHOST, port))
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        !server.is_finished(),
        "connected client must suppress idle exit"
    );
    drop(client);

    tokio::time::timeout(Duration::from_secs(3), server)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(!server_home.join("daemon.json").exists());
}

#[test]
fn simultaneous_lock_attempts_have_one_winner() {
    let home = tempfile::tempdir().unwrap();
    let barrier = Arc::new(Barrier::new(3));
    let mut threads = Vec::new();
    for _ in 0..2 {
        let path = home.path().to_path_buf();
        let barrier = Arc::clone(&barrier);
        threads.push(std::thread::spawn(move || {
            barrier.wait();
            EngineLock::acquire(path)
        }));
    }
    barrier.wait();
    let results = threads
        .into_iter()
        .map(|thread| thread.join().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
    assert_eq!(
        results
            .iter()
            .filter(|result| matches!(result, Err(EngineError::Busy { .. })))
            .count(),
        1
    );
}

#[test]
fn in_process_handoff_retains_ownership_through_store_close() {
    let home = tempfile::tempdir().unwrap();
    drop(InProcessEngine::open(home.path()).unwrap());
    let barrier = Arc::new(Barrier::new(3));
    let mut threads = Vec::new();
    for _ in 0..2 {
        let home = home.path().to_path_buf();
        let barrier = Arc::clone(&barrier);
        threads.push(std::thread::spawn(move || {
            barrier.wait();
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            let mut starts = 0;
            let mut unexpected = Vec::new();
            while std::time::Instant::now() < deadline {
                match InProcessEngine::open(&home) {
                    Ok(engine) => {
                        starts += 1;
                        drop(engine);
                    }
                    Err(EngineError::Busy { .. }) => {}
                    Err(error) => unexpected.push(error.to_string()),
                }
            }
            (starts, unexpected)
        }));
    }
    barrier.wait();
    let mut starts = 0;
    let mut unexpected = Vec::new();
    for thread in threads {
        let (count, errors) = thread.join().unwrap();
        starts += count;
        unexpected.extend(errors);
    }
    assert!(starts > 0, "the probe must open an engine");
    assert!(
        unexpected.is_empty(),
        "advisory ownership ended before Store closed: {unexpected:?}"
    );
}
