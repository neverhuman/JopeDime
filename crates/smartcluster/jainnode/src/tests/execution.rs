#[tokio::test]
async fn smoke_precedes_real_run_and_success_waits_for_durable_outputs() {
    let journal_root = tempfile::tempdir().unwrap();
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
    let shard = Arc::new(ShardMock {
        durable: true,
        ..ShardMock::default()
    });
    let network = Arc::new(NetworkMock::default());
    let events = Arc::new(EventsMock::default());
    let executor = JainNodeExecutor::new(
        ExecutorPorts {
            docker: docker.clone(),
            shard: shard.clone(),
            network: network.clone(),
            cancellation: Arc::new(NeverCancel),
            metrics: Arc::new(MetricsMock),
            events: events.clone(),
            isolation_enforcer: Arc::new(OrderingIsolationMock {
                docker: docker.clone(),
            }),
            clock: Arc::new(FixedClock(2)),
        },
        isolation(),
        identity(),
        AssignmentJournal::open(journal_root.path().to_path_buf()).unwrap(),
    )
    .unwrap();

    let (accepted, acknowledgement) = executor
        .accept_assignment(work(), placement(), 1, 1)
        .unwrap();
    acknowledgement
        .verify_for(
            accepted.work(),
            accepted.placement(),
            SigningKey::from_bytes(&[12; 32]).verifying_key().to_bytes(),
        )
        .unwrap();
    let receipt = executor.execute(&accepted).await.unwrap();
    assert_eq!(receipt.outputs.len(), 1);
    assert_eq!(receipt.logs.len(), 2);
    let specs = docker.specs.lock().unwrap();
    assert_eq!(specs.len(), 2);
    assert_eq!(specs[0].argv[1], "--smoke");
    assert_eq!(specs[1].argv[1], "--real");
    assert_eq!(specs[1].gpu_uuids, ["GPU-exact"]);
    assert_eq!(*network.revocations.lock().unwrap(), 2);
    assert_eq!(*shard.output_seals.lock().unwrap(), 1);
    let recorded = events.0.lock().unwrap();
    assert!(recorded
        .iter()
        .any(|event| matches!(event, NodeEvent::SmokePassed(_))));
    assert!(recorded
        .iter()
        .any(|event| matches!(event, NodeEvent::StdoutChunk { .. })));
    assert!(recorded
        .iter()
        .any(|event| matches!(event, NodeEvent::StderrChunk { .. })));
    assert!(recorded
        .iter()
        .any(|event| matches!(event, NodeEvent::Metrics { .. })));
    assert!(recorded
        .iter()
        .any(|event| matches!(event, NodeEvent::ArtifactPublished { .. })));
    let cursors = recorded.iter().map(event_cursor).collect::<Vec<_>>();
    assert!(cursors.windows(2).all(|pair| pair[0] < pair[1]));
}

#[tokio::test]
async fn failed_smoke_never_creates_the_real_container() {
    let journal_root = tempfile::tempdir().unwrap();
    let docker = Arc::new(DockerMock {
        exits: Mutex::new(vec![ExitStatus {
            code: 7,
            oom_killed: false,
        }]),
        ..DockerMock::default()
    });
    let shard = Arc::new(ShardMock {
        durable: true,
        ..ShardMock::default()
    });
    let executor = JainNodeExecutor::new(
        ExecutorPorts {
            docker: docker.clone(),
            shard: shard.clone(),
            network: Arc::new(NetworkMock::default()),
            cancellation: Arc::new(NeverCancel),
            metrics: Arc::new(MetricsMock),
            events: Arc::new(EventsMock::default()),
            isolation_enforcer: Arc::new(IsolationMock),
            clock: Arc::new(FixedClock(2)),
        },
        isolation(),
        identity(),
        AssignmentJournal::open(journal_root.path().to_path_buf()).unwrap(),
    )
    .unwrap();

    let accepted = executor
        .accept_assignment(work(), placement(), 1, 1)
        .unwrap()
        .0;
    let error = executor
        .execute(&accepted)
        .await
        .expect_err("smoke status 7 must fail");
    assert_eq!(error.to_string(), "process exited with status 7");
    assert_eq!(docker.specs.lock().unwrap().len(), 1);
    assert_eq!(*shard.output_seals.lock().unwrap(), 0);
}

#[tokio::test]
async fn cancellation_is_fenced_into_container_kill_and_no_publication() {
    let journal_root = tempfile::tempdir().unwrap();
    let docker = Arc::new(DockerMock {
        block_wait: true,
        exits: Mutex::new(vec![ExitStatus {
            code: 0,
            oom_killed: false,
        }]),
        ..DockerMock::default()
    });
    let shard = Arc::new(ShardMock {
        durable: true,
        ..ShardMock::default()
    });
    let executor = JainNodeExecutor::new(
        ExecutorPorts {
            docker: docker.clone(),
            shard: shard.clone(),
            network: Arc::new(NetworkMock::default()),
            cancellation: Arc::new(ImmediateCancel),
            metrics: Arc::new(MetricsMock),
            events: Arc::new(EventsMock::default()),
            isolation_enforcer: Arc::new(IsolationMock),
            clock: Arc::new(FixedClock(2)),
        },
        isolation(),
        identity(),
        AssignmentJournal::open(journal_root.path().to_path_buf()).unwrap(),
    )
    .unwrap();

    let accepted = executor
        .accept_assignment(work(), placement(), 1, 1)
        .unwrap()
        .0;
    let error = executor
        .execute(&accepted)
        .await
        .expect_err("fenced cancellation must fail the attempt");
    assert_eq!(error.to_string(), "cancelled");
    assert_eq!(*docker.killed.lock().unwrap(), 1);
    assert_eq!(*shard.output_seals.lock().unwrap(), 0);
}

#[tokio::test]
async fn missing_shard_quorum_prevents_terminal_success() {
    let journal_root = tempfile::tempdir().unwrap();
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
    let shard = Arc::new(ShardMock::default());
    let executor = JainNodeExecutor::new(
        ExecutorPorts {
            docker,
            shard: shard.clone(),
            network: Arc::new(NetworkMock::default()),
            cancellation: Arc::new(NeverCancel),
            metrics: Arc::new(MetricsMock),
            events: Arc::new(EventsMock::default()),
            isolation_enforcer: Arc::new(IsolationMock),
            clock: Arc::new(FixedClock(2)),
        },
        isolation(),
        identity(),
        AssignmentJournal::open(journal_root.path().to_path_buf()).unwrap(),
    )
    .unwrap();

    let accepted = executor
        .accept_assignment(work(), placement(), 1, 1)
        .unwrap()
        .0;
    let error = executor
        .execute(&accepted)
        .await
        .expect_err("missing log quorum must fail the attempt");
    assert_eq!(error.to_string(), "durability_unavailable");
    assert_eq!(*shard.output_seals.lock().unwrap(), 0);
    assert_eq!(*shard.cleanups.lock().unwrap(), 1);
}

#[test]
fn executor_fails_closed_without_complete_isolation_evidence() {
    let journal_root = tempfile::tempdir().unwrap();
    let mut evidence = isolation();
    evidence.network_isolation = false;
    let result = JainNodeExecutor::new(
        ExecutorPorts {
            docker: Arc::new(DockerMock::default()),
            shard: Arc::new(ShardMock::default()),
            network: Arc::new(NetworkMock::default()),
            cancellation: Arc::new(NeverCancel),
            metrics: Arc::new(MetricsMock),
            events: Arc::new(EventsMock::default()),
            isolation_enforcer: Arc::new(IsolationMock),
            clock: Arc::new(FixedClock(2)),
        },
        evidence,
        identity(),
        AssignmentJournal::open(journal_root.path().to_path_buf()).unwrap(),
    );
    let error = result
        .err()
        .expect("incomplete isolation evidence must reject node startup");
    assert_eq!(error.to_string(), "unsupported_host");
}

#[tokio::test]
async fn assignment_is_durable_before_ack_and_missing_gpu_is_rejected() {
    let journal_root = tempfile::tempdir().unwrap();
    let journal = AssignmentJournal::open(journal_root.path().to_path_buf()).unwrap();
    let executor = JainNodeExecutor::new(
        ExecutorPorts {
            docker: Arc::new(DockerMock::default()),
            shard: Arc::new(ShardMock::default()),
            network: Arc::new(NetworkMock::default()),
            cancellation: Arc::new(NeverCancel),
            metrics: Arc::new(MetricsMock),
            events: Arc::new(EventsMock::default()),
            isolation_enforcer: Arc::new(IsolationMock),
            clock: Arc::new(FixedClock(2)),
        },
        isolation(),
        identity(),
        journal.clone(),
    )
    .unwrap();

    let error = executor
        .accept_assignment(work(), placement_with_gpus(Vec::new()), 1, 1)
        .expect_err("GPU work must not accept a CPU-only placement");
    assert!(error.to_string().contains("invalid_placement"));
    assert!(journal.recover(&identity(), 1).unwrap().pending.is_empty());

    let (accepted, acknowledgement) = executor
        .accept_assignment(work(), placement(), 1, 1)
        .unwrap();
    assert_ne!(accepted.journal_digest(), [0; 32]);
    let initial_recovery = journal.recover(&identity(), 1).unwrap();
    assert_eq!(initial_recovery.pending, [work().run_id.to_string()]);
    assert_eq!(initial_recovery.replayable.len(), 1);
    assert_eq!(initial_recovery.replayable[0].work(), accepted.work());
    assert_eq!(
        initial_recovery.replayable[0].placement(),
        accepted.placement()
    );
    acknowledgement
        .verify_for(
            accepted.work(),
            accepted.placement(),
            SigningKey::from_bytes(&[12; 32]).verifying_key().to_bytes(),
        )
        .unwrap();

    let assignment_path = std::fs::read_dir(journal_root.path())
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| path.extension().and_then(|value| value.to_str()) == Some("assignment"))
        .unwrap();
    let interrupted_path = assignment_path.with_extension("pending");
    std::fs::rename(&assignment_path, &interrupted_path).unwrap();
    let recovered_interrupted = journal.recover(&identity(), 1).unwrap();
    assert_eq!(recovered_interrupted.replayable.len(), 1);
    assert!(assignment_path.is_file());
    assert!(!interrupted_path.exists());

    std::fs::write(journal_root.path().join("corrupt.assignment"), b"corrupt").unwrap();
    let recovery = journal.recover(&identity(), 1).unwrap();
    assert_eq!(recovery.pending, [work().run_id.to_string()]);
    assert_eq!(recovery.quarantined.len(), 1);
}

#[tokio::test]
async fn placement_expiring_after_ack_is_rejected_before_container_start() {
    let journal_root = tempfile::tempdir().unwrap();
    let docker = Arc::new(DockerMock::default());
    let executor = JainNodeExecutor::new(
        ExecutorPorts {
            docker: docker.clone(),
            shard: Arc::new(ShardMock::default()),
            network: Arc::new(NetworkMock::default()),
            cancellation: Arc::new(NeverCancel),
            metrics: Arc::new(MetricsMock),
            events: Arc::new(EventsMock::default()),
            isolation_enforcer: Arc::new(IsolationMock),
            clock: Arc::new(FixedClock(2)),
        },
        isolation(),
        identity(),
        AssignmentJournal::open(journal_root.path().to_path_buf()).unwrap(),
    )
    .unwrap();
    let mut expiring = placement();
    expiring.lease_expires_at_ms = 2;
    expiring.signature = DomainSignature::sign(
        SignatureAuthority::PlacementOffer,
        "hub-issuer".to_owned(),
        "hub-key".to_owned(),
        &hub_key(),
        &expiring.signed_bytes().unwrap(),
    );
    let accepted = executor
        .accept_assignment(work(), expiring, 1, 1)
        .expect("placement is current at ACK")
        .0;

    let error = executor
        .execute(&accepted)
        .await
        .expect_err("placement expiring before start must fail closed");
    assert_eq!(error.to_string(), "stale_authority");
    assert!(docker.started.lock().unwrap().is_empty());
    assert_eq!(*docker.removed.lock().unwrap(), 1);
}

#[tokio::test]
async fn cancellation_accepted_after_ack_is_rejected_before_container_start() {
    let journal_root = tempfile::tempdir().unwrap();
    let docker = Arc::new(DockerMock::default());
    let executor = JainNodeExecutor::new(
        ExecutorPorts {
            docker: docker.clone(),
            shard: Arc::new(ShardMock::default()),
            network: Arc::new(NetworkMock::default()),
            cancellation: Arc::new(AlreadyCancelled),
            metrics: Arc::new(MetricsMock),
            events: Arc::new(EventsMock::default()),
            isolation_enforcer: Arc::new(IsolationMock),
            clock: Arc::new(FixedClock(2)),
        },
        isolation(),
        identity(),
        AssignmentJournal::open(journal_root.path().to_path_buf()).unwrap(),
    )
    .unwrap();
    let accepted = executor
        .accept_assignment(work(), placement(), 1, 1)
        .expect("placement is current at ACK")
        .0;

    let error = executor
        .execute(&accepted)
        .await
        .expect_err("a previously accepted cancellation must prevent start");
    assert_eq!(error.to_string(), "cancelled");
    assert!(docker.started.lock().unwrap().is_empty());
    assert_eq!(*docker.killed.lock().unwrap(), 0);
    assert_eq!(*docker.removed.lock().unwrap(), 1);
}

#[tokio::test]
async fn docker_log_failure_or_unmarked_eof_is_terminal() {
    for (log_error, incomplete_log) in [(true, false), (false, true)] {
        let journal_root = tempfile::tempdir().unwrap();
        let docker = Arc::new(DockerMock {
            exits: Mutex::new(vec![ExitStatus {
                code: 0,
                oom_killed: false,
            }]),
            log_error,
            incomplete_log,
            ..DockerMock::default()
        });
        let shard = Arc::new(ShardMock {
            durable: true,
            ..ShardMock::default()
        });
        let executor = JainNodeExecutor::new(
            ExecutorPorts {
                docker,
                shard: shard.clone(),
                network: Arc::new(NetworkMock::default()),
                cancellation: Arc::new(NeverCancel),
                metrics: Arc::new(MetricsMock),
                events: Arc::new(EventsMock::default()),
                isolation_enforcer: Arc::new(IsolationMock),
                clock: Arc::new(FixedClock(2)),
            },
            isolation(),
            identity(),
            AssignmentJournal::open(journal_root.path().to_path_buf()).unwrap(),
        )
        .unwrap();
        let accepted = executor
            .accept_assignment(work(), placement(), 1, 1)
            .unwrap()
            .0;
        assert!(executor.execute(&accepted).await.is_err());
        assert_eq!(*shard.output_seals.lock().unwrap(), 0);
    }
}

fn event_cursor(event: &NodeEvent) -> u64 {
    match event {
        NodeEvent::SmokeStarted(identity)
        | NodeEvent::SmokePassed(identity)
        | NodeEvent::StdoutChunk { identity, .. }
        | NodeEvent::StderrChunk { identity, .. }
        | NodeEvent::Progress { identity, .. }
        | NodeEvent::Metrics { identity, .. }
        | NodeEvent::ArtifactPublished { identity, .. }
        | NodeEvent::StreamGap { identity, .. }
        | NodeEvent::SmokeFailed { identity, .. } => identity.cursor,
    }
}
