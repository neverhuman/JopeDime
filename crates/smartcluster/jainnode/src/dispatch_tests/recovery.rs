use super::*;

#[tokio::test(flavor = "current_thread")]
async fn revocation_rejects_prepared_journal_before_capacity_can_return() {
    let harness = runtime_harness(DockerMock::default(), true);
    harness.runtime.prepare(admitted(), 1).await.unwrap();
    assert_eq!(harness.runtime.active_count().unwrap(), 0);
    assert_eq!(harness.outbox.pending_count().unwrap(), 0);
    assert!(!harness.runtime.ready_for_rotation().unwrap());

    harness.runtime.revoke_all().await.unwrap();

    assert!(harness.runtime.ready_for_rotation().unwrap());
    let recovery = harness.journal.recover(&identity(), 2).unwrap();
    assert!(recovery.replayable.is_empty());
    assert!(recovery.quarantined.is_empty());
}

#[tokio::test(flavor = "current_thread")]
async fn accepted_assignment_recovers_without_duplicate_ack_and_restores_capacity() {
    let journal_root = tempfile::tempdir().unwrap();
    let cancellation_root = tempfile::tempdir().unwrap();
    let outbox_root = tempfile::tempdir().unwrap();
    let protocol_root = tempfile::tempdir().unwrap();
    fs::set_permissions(cancellation_root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    fs::set_permissions(outbox_root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let mut durable_protocol = protocol(protocol_root.path(), 1);
    durable_protocol.admit_infer(infer_frame(1), 2).unwrap();
    drop(durable_protocol);
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
    let (accepted, acknowledgement) = executor
        .accept_assignment(work(), placement(), 1, 1)
        .unwrap();
    outbox
        .enqueue_acknowledgement(acknowledgement.clone())
        .unwrap();
    assert!(outbox.contains_acknowledgement(&acknowledgement).unwrap());
    drop(accepted);
    drop(executor);
    drop(outbox);
    drop(cancellations);

    let cancellations = Arc::new(
        DurableCancellationRegistry::open(cancellation_root.path().to_path_buf()).unwrap(),
    );
    let outbox =
        Arc::new(DurableControlOutbox::open(outbox_root.path().to_path_buf(), identity()).unwrap());
    let executor = test_executor(
        Arc::clone(&docker),
        true,
        Arc::clone(&cancellations),
        Arc::clone(&outbox),
        journal.clone(),
    );
    let runtime = Arc::new(
        ConcurrentExecutorRuntime::recover(executor, cancellations, Arc::clone(&outbox), 2)
            .unwrap(),
    );
    assert_eq!(runtime.active_count().unwrap(), 0);
    assert_eq!(outbox.pending_count().unwrap(), 1);

    let mut handler = ProductionControlHandler::new(
        protocol(protocol_root.path(), 1),
        Arc::clone(&runtime),
        Arc::new(FixedClock(2)),
    )
    .unwrap();
    assert_eq!(runtime.active_count().unwrap(), 1);
    assert!(matches!(
        handler.handle(infer_frame(2)).await,
        Err(ControlError::Protocol(ProtocolError::AdmissionDenied))
    ));

    wait_for_completion(&runtime).await;
    let frames = drain_frames(&outbox);
    let acknowledgements = frames
        .iter()
        .filter_map(|frame| match frame {
            NodeControlFrame::AssignmentAccepted {
                acknowledgement, ..
            } => Some(acknowledgement.as_ref()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(acknowledgements, [&acknowledgement]);
    assert!(frames.iter().any(|frame| matches!(
        frame,
        NodeControlFrame::Terminal {
            outcome,
            ..
        } if matches!(outcome.as_ref(), NodeTerminalOutcome::Success(_))
    )));
}
