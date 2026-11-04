use super::*;

#[tokio::test(flavor = "current_thread")]
async fn running_assignment_quarantines_and_stops_runtime_recovery() {
    let journal_root = tempfile::tempdir().unwrap();
    let cancellation_root = tempfile::tempdir().unwrap();
    let outbox_root = tempfile::tempdir().unwrap();
    fs::set_permissions(cancellation_root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    fs::set_permissions(outbox_root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let journal = AssignmentJournal::open(journal_root.path().to_path_buf()).unwrap();
    let cancellations = Arc::new(
        DurableCancellationRegistry::open(cancellation_root.path().to_path_buf()).unwrap(),
    );
    let outbox =
        Arc::new(DurableControlOutbox::open(outbox_root.path().to_path_buf(), identity()).unwrap());
    let executor = test_executor(
        Arc::new(DockerMock::default()),
        true,
        Arc::clone(&cancellations),
        Arc::clone(&outbox),
        journal.clone(),
    );
    let accepted = executor
        .accept_assignment(work(), placement(), 1, 1)
        .unwrap()
        .0;
    journal.mark_running(&accepted).unwrap();
    drop(accepted);
    drop(executor);

    let executor = test_executor(
        Arc::new(DockerMock::default()),
        true,
        Arc::clone(&cancellations),
        Arc::clone(&outbox),
        journal,
    );
    assert!(matches!(
        ConcurrentExecutorRuntime::recover(executor, cancellations, outbox, 2),
        Err(ExecutionError::StaleAuthority)
    ));
    assert!(fs::read_dir(journal_root.path()).unwrap().any(|entry| entry
        .unwrap()
        .file_name()
        .to_string_lossy()
        .ends_with(".quarantine")));
}

#[tokio::test(flavor = "current_thread")]
async fn uncommitted_prepared_assignment_is_adopted_by_exact_controller_retry() {
    let journal_root = tempfile::tempdir().unwrap();
    let cancellation_root = tempfile::tempdir().unwrap();
    let outbox_root = tempfile::tempdir().unwrap();
    let protocol_root = tempfile::tempdir().unwrap();
    fs::set_permissions(cancellation_root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    fs::set_permissions(outbox_root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let journal = AssignmentJournal::open(journal_root.path().to_path_buf()).unwrap();
    let cancellations = Arc::new(
        DurableCancellationRegistry::open(cancellation_root.path().to_path_buf()).unwrap(),
    );
    let outbox =
        Arc::new(DurableControlOutbox::open(outbox_root.path().to_path_buf(), identity()).unwrap());
    let docker = Arc::new(DockerMock {
        exits: Mutex::new(vec![
            ExitStatus {
                code: 0,
                oom_killed: false,
            },
            ExitStatus {
                code: 0,
                oom_killed: false,
            },
        ]),
        ..DockerMock::default()
    });
    let executor = test_executor(
        Arc::clone(&docker),
        true,
        Arc::clone(&cancellations),
        Arc::clone(&outbox),
        journal.clone(),
    );
    executor
        .accept_assignment(work(), placement(), 1, 1)
        .unwrap();
    drop(executor);
    drop(outbox);
    drop(cancellations);

    let cancellations = Arc::new(
        DurableCancellationRegistry::open(cancellation_root.path().to_path_buf()).unwrap(),
    );
    let outbox =
        Arc::new(DurableControlOutbox::open(outbox_root.path().to_path_buf(), identity()).unwrap());
    let executor = test_executor(
        docker,
        true,
        Arc::clone(&cancellations),
        Arc::clone(&outbox),
        journal,
    );
    let runtime = Arc::new(
        ConcurrentExecutorRuntime::recover(executor, cancellations, Arc::clone(&outbox), 2)
            .unwrap(),
    );
    assert_eq!(runtime.active_count().unwrap(), 0);
    assert_eq!(outbox.pending_count().unwrap(), 0);
    let mut handler = ProductionControlHandler::new(
        protocol(protocol_root.path(), 1),
        Arc::clone(&runtime),
        Arc::new(FixedClock(2)),
    )
    .unwrap();

    assert!(matches!(
        handler.handle(infer_frame(2)).await,
        Err(ControlError::Execution(ExecutionError::StaleAuthority))
    ));
    assert_eq!(runtime.active_count().unwrap(), 0);
    assert_eq!(outbox.pending_count().unwrap(), 0);
    assert_eq!(
        handler.handle(infer_frame(1)).await.unwrap(),
        ControlFlow::Continue
    );
    assert_eq!(runtime.active_count().unwrap(), 1);
    assert_eq!(outbox.pending_count().unwrap(), 1);
    wait_for_completion(&runtime).await;
    let frames = drain_frames(&outbox);
    assert_eq!(
        frames
            .iter()
            .filter(|frame| matches!(frame, NodeControlFrame::AssignmentAccepted { .. }))
            .count(),
        1
    );
    assert!(frames.iter().any(|frame| matches!(
        frame,
        NodeControlFrame::Terminal {
            outcome,
            ..
        } if matches!(outcome.as_ref(), NodeTerminalOutcome::Success(_))
    )));
}

#[tokio::test(flavor = "current_thread")]
async fn failed_controller_position_commit_terminalizes_prepared_without_ack_or_launch() {
    let journal_root = tempfile::tempdir().unwrap();
    let cancellation_root = tempfile::tempdir().unwrap();
    let outbox_root = tempfile::tempdir().unwrap();
    let protocol_root = tempfile::tempdir().unwrap();
    fs::set_permissions(cancellation_root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    fs::set_permissions(outbox_root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let journal = AssignmentJournal::open(journal_root.path().to_path_buf()).unwrap();
    let cancellations = Arc::new(
        DurableCancellationRegistry::open(cancellation_root.path().to_path_buf()).unwrap(),
    );
    let outbox =
        Arc::new(DurableControlOutbox::open(outbox_root.path().to_path_buf(), identity()).unwrap());
    let runtime = Arc::new(ConcurrentExecutorRuntime::new(
        test_executor(
            Arc::new(DockerMock::default()),
            true,
            Arc::clone(&cancellations),
            Arc::clone(&outbox),
            journal.clone(),
        ),
        cancellations,
        Arc::clone(&outbox),
    ));
    let mut handler = ProductionControlHandler::new(
        protocol(protocol_root.path(), 1),
        Arc::clone(&runtime),
        Arc::new(FixedClock(2)),
    )
    .unwrap();
    fs::remove_dir_all(protocol_root.path()).unwrap();

    assert!(matches!(
        handler.handle(infer_frame(1)).await,
        Err(ControlError::Protocol(ProtocolError::Io(_)))
    ));
    assert_eq!(runtime.active_count().unwrap(), 0);
    assert_eq!(outbox.pending_count().unwrap(), 0);
    let recovery = journal.recover(&identity(), 3).unwrap();
    assert!(recovery.replayable.is_empty());
    assert!(recovery.quarantined.is_empty());
}
