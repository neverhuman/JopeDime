use std::{fs, os::unix::fs::PermissionsExt, sync::Arc, time::Duration};

use super::*;
use crate::{
    AdmittedInfer, ConcurrentExecutorRuntime, ControlError, ControlFlow, ControlFrameHandler,
    DurableCancellationRegistry, DurableControlOutbox, InferRequest, NodeControlFrame,
    NodeExecutionPort, NodeFailureCode, NodeProfile, NodeProtocolState, NodeTerminalOutcome,
    OutboundConfirmation, ProductionControlHandler, ProtocolError,
};

type TestRuntime = ConcurrentExecutorRuntime<
    DockerMock,
    ShardMock,
    NetworkMock,
    MetricsMock,
    IsolationMock,
    FixedClock,
>;
type TestExecutor = JainNodeExecutor<
    DockerMock,
    ShardMock,
    NetworkMock,
    DurableCancellationRegistry,
    MetricsMock,
    DurableControlOutbox,
    IsolationMock,
    FixedClock,
>;

struct RuntimeHarness {
    _journal_root: tempfile::TempDir,
    _cancellation_root: tempfile::TempDir,
    _outbox_root: tempfile::TempDir,
    runtime: Arc<TestRuntime>,
    executor: Arc<TestExecutor>,
    outbox: Arc<DurableControlOutbox>,
    journal: AssignmentJournal,
    docker: Arc<DockerMock>,
}

fn runtime_harness(docker: DockerMock, durable_shard: bool) -> RuntimeHarness {
    let journal_root = tempfile::tempdir().unwrap();
    let cancellation_root = tempfile::tempdir().unwrap();
    let outbox_root = tempfile::tempdir().unwrap();
    fs::set_permissions(cancellation_root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    fs::set_permissions(outbox_root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let docker = Arc::new(docker);
    let cancellations = Arc::new(
        DurableCancellationRegistry::open(cancellation_root.path().to_path_buf()).unwrap(),
    );
    let outbox =
        Arc::new(DurableControlOutbox::open(outbox_root.path().to_path_buf(), identity()).unwrap());
    let journal = AssignmentJournal::open(journal_root.path().to_path_buf()).unwrap();
    let executor = test_executor(
        Arc::clone(&docker),
        durable_shard,
        Arc::clone(&cancellations),
        Arc::clone(&outbox),
        journal.clone(),
    );
    let runtime = Arc::new(ConcurrentExecutorRuntime::new(
        Arc::clone(&executor),
        cancellations,
        Arc::clone(&outbox),
    ));
    RuntimeHarness {
        _journal_root: journal_root,
        _cancellation_root: cancellation_root,
        _outbox_root: outbox_root,
        runtime,
        executor,
        outbox,
        journal,
        docker,
    }
}

fn test_executor(
    docker: Arc<DockerMock>,
    durable_shard: bool,
    cancellations: Arc<DurableCancellationRegistry>,
    outbox: Arc<DurableControlOutbox>,
    journal: AssignmentJournal,
) -> Arc<TestExecutor> {
    Arc::new(
        JainNodeExecutor::new(
            ExecutorPorts {
                docker,
                shard: Arc::new(ShardMock {
                    durable: durable_shard,
                    ..ShardMock::default()
                }),
                network: Arc::new(NetworkMock::default()),
                cancellation: cancellations,
                metrics: Arc::new(MetricsMock),
                events: outbox,
                isolation_enforcer: Arc::new(IsolationMock),
                clock: Arc::new(FixedClock(2)),
            },
            isolation(),
            identity(),
            journal,
        )
        .unwrap(),
    )
}

fn admitted() -> AdmittedInfer {
    AdmittedInfer {
        work: work(),
        placement: placement(),
        sequence: 1,
        deadline_ms: 10_000,
    }
}

fn infer_frame(sequence: u64) -> NodeControlFrame {
    let admitted = admitted();
    NodeControlFrame::Infer {
        request: Box::new(InferRequest {
            session_id: "session-1".to_owned(),
            sequence,
            deadline_ms: admitted.deadline_ms,
            work: admitted.work.signed_projection().clone(),
            placement: admitted.placement,
        }),
    }
}

fn protocol(root: &std::path::Path, max_inflight: u32) -> NodeProtocolState {
    protocol_with_provision(
        root,
        max_inflight,
        crate::WorkerProvision::SmartclusterCompute,
    )
}

fn protocol_with_provision(
    root: &std::path::Path,
    max_inflight: u32,
    provision: crate::WorkerProvision,
) -> NodeProtocolState {
    NodeProtocolState::open(
        root.to_path_buf(),
        "node-1".to_owned(),
        "session-1".to_owned(),
        NodeProfile::Gpu,
        provision,
        "hub-issuer".to_owned(),
        hub_key().verifying_key().to_bytes(),
        "hub-issuer".to_owned(),
        hub_key().verifying_key().to_bytes(),
        max_inflight,
    )
    .unwrap()
}

#[test]
fn worker_provision_rejects_a_foreign_signed_workload_before_admission() {
    let root = tempfile::tempdir().unwrap();
    let code_runner = protocol_with_provision(root.path(), 1, crate::WorkerProvision::CodeRunner);
    assert!(matches!(
        code_runner.verify_infer(infer_frame(1), 2),
        Err(ProtocolError::AdmissionDenied)
    ));
}

async fn wait_for_completion(runtime: &TestRuntime) {
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let work = work();
            if runtime
                .active_work(&work.run_id.to_string(), work.attempt, work.attempt_fence)
                .unwrap()
                .is_none()
            {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("governed execution must reach a durable terminal frame");
}

fn drain_frames(outbox: &DurableControlOutbox) -> Vec<NodeControlFrame> {
    let mut frames = Vec::new();
    while let Some(pending) = outbox.oldest().unwrap() {
        frames.push(pending.frame.clone());
        let confirmation = NodeControlFrame::outbound_confirmation(
            "session-1".to_owned(),
            pending.sequence,
            pending.digest,
            50_000,
            "hub-issuer".to_owned(),
            "hub-key".to_owned(),
            &hub_key(),
        )
        .unwrap();
        assert_eq!(
            outbox
                .confirm_controller(&confirmation, 50_000, 100)
                .unwrap(),
            OutboundConfirmation::Retired
        );
    }
    frames
}

#[tokio::test(flavor = "current_thread")]
async fn concurrent_runtime_persists_ack_before_spawning_and_terminalizes_success() {
    let harness = runtime_harness(
        DockerMock {
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
        },
        true,
    );
    let admitted = admitted();
    harness.runtime.admit(admitted.clone(), 1).await.unwrap();
    assert!(matches!(
        harness.runtime.admit(admitted, 1).await,
        Err(ExecutionError::StaleAuthority)
    ));

    assert!(matches!(
        harness.outbox.oldest().unwrap().unwrap().frame,
        NodeControlFrame::AssignmentAccepted { .. }
    ));
    assert!(harness.docker.specs.lock().unwrap().is_empty());

    wait_for_completion(&harness.runtime).await;
    assert_eq!(harness.runtime.drain_completions().unwrap(), 1);
    assert_eq!(harness.runtime.drain_completions().unwrap(), 0);
    assert_eq!(harness.docker.specs.lock().unwrap().len(), 2);

    let frames = drain_frames(&harness.outbox);
    assert!(matches!(
        frames.first(),
        Some(NodeControlFrame::AssignmentAccepted { .. })
    ));
    assert!(
        frames.iter().any(|frame| matches!(
            frame,
            NodeControlFrame::Terminal {
                outcome,
                ..
            } if matches!(outcome.as_ref(), NodeTerminalOutcome::Success(_))
        )),
        "expected terminal success, got {frames:#?}"
    );
    assert!(harness.runtime.ready_for_rotation().unwrap());
    assert!(harness
        .journal
        .recover(&identity(), 3)
        .unwrap()
        .replayable
        .is_empty());
}

#[tokio::test(flavor = "current_thread")]
async fn concurrent_runtime_terminalizes_failed_execution_before_failure_frame() {
    let harness = runtime_harness(
        DockerMock {
            exits: Mutex::new(vec![ExitStatus {
                code: 7,
                oom_killed: false,
            }]),
            ..DockerMock::default()
        },
        true,
    );
    harness.runtime.admit(admitted(), 1).await.unwrap();
    wait_for_completion(&harness.runtime).await;

    assert_eq!(harness.runtime.drain_completions().unwrap(), 1);
    let frames = drain_frames(&harness.outbox);
    assert!(frames.iter().any(|frame| matches!(
        frame,
        NodeControlFrame::Terminal {
            outcome,
            ..
        } if matches!(outcome.as_ref(), NodeTerminalOutcome::Failure { .. })
    )));
    let recovery = harness.journal.recover(&identity(), 3).unwrap();
    assert!(recovery.replayable.is_empty());
    assert!(recovery.quarantined.is_empty());
}

// This fixture orders cancellation against durable I/O, not elapsed host time.
// Keep signed deadlines and all cancellation/durability assertions intact.
#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn concurrent_runtime_accepts_durable_cancellation_while_execution_runs() {
    let harness = runtime_harness(
        DockerMock {
            block_wait: true,
            ..DockerMock::default()
        },
        true,
    );
    let admitted = admitted();
    let run_id = admitted.work.run_id.to_string();
    let attempt = admitted.work.attempt;
    let fence = admitted.work.attempt_fence;
    harness.runtime.admit(admitted, 1).await.unwrap();
    tokio::time::timeout(Duration::from_secs(2), async {
        while harness.docker.started.lock().unwrap().is_empty() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();

    harness
        .runtime
        .cancel(&run_id, attempt, fence)
        .await
        .unwrap();
    wait_for_completion(&harness.runtime).await;

    assert_eq!(*harness.docker.killed.lock().unwrap(), 1);
    let frames = drain_frames(&harness.outbox);
    assert!(frames.iter().any(|frame| matches!(
        frame,
        NodeControlFrame::Terminal {
            outcome,
            ..
        } if matches!(
            outcome.as_ref(),
            NodeTerminalOutcome::Failure {
                code: NodeFailureCode::Cancelled,
                ..
            }
        )
    )));
}

// Paused time in the cancellation-order fixture must not hide the independent
// smoke/run deadline branches. These controls advance the original signed limits.
async fn observe_wait_or_kill(harness: &RuntimeHarness, container: &str, previous: usize) {
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let polls = harness
                .docker
                .waiting
                .lock()
                .unwrap()
                .iter()
                .filter(|id| id.as_str() == container)
                .count();
            if polls > previous || *harness.docker.killed.lock().unwrap() > 0 {
                break;
            }
            harness.docker.wait_observed.notified().await;
        }
    })
    .await
    .unwrap();
}

async fn repoll_wait_or_kill(harness: &RuntimeHarness, container: &str) {
    let previous = harness
        .docker
        .waiting
        .lock()
        .unwrap()
        .iter()
        .filter(|id| id.as_str() == container)
        .count();
    harness.docker.poll_wait.notify_one();
    observe_wait_or_kill(harness, container, previous).await;
}

async fn assert_durable_execution_deadline(run_phase: bool) {
    let harness = runtime_harness(
        DockerMock {
            block_wait: !run_phase,
            block_run_wait: run_phase,
            exits: Mutex::new(vec![ExitStatus {
                code: 0,
                oom_killed: false,
            }]),
            ..DockerMock::default()
        },
        true,
    );
    let admitted = admitted();
    assert_eq!(admitted.work.smoke_timeout_ms, 1_000);
    assert_eq!(admitted.work.limits.wall_time_ms, 10_000);
    let limit = if run_phase {
        admitted.work.limits.wall_time_ms
    } else {
        admitted.work.smoke_timeout_ms
    };
    let container = if run_phase {
        "container-2"
    } else {
        "container-1"
    };
    harness.runtime.admit(admitted, 1).await.unwrap();
    observe_wait_or_kill(&harness, container, 0).await;

    tokio::time::advance(Duration::from_millis(limit - 1)).await;
    // advance alone does not wait for expired timers to run. Re-poll the
    // executor's blocked wait and acknowledge its poll before observing state.
    // This current-thread fixture resumes only after that executor poll returns.
    repoll_wait_or_kill(&harness, container).await;
    assert_eq!(*harness.docker.killed.lock().unwrap(), 0);
    assert_eq!(harness.runtime.active_count().unwrap(), 1);
    assert_eq!(harness.runtime.drain_completions().unwrap(), 0);
    tokio::time::advance(Duration::from_millis(1)).await;
    repoll_wait_or_kill(&harness, container).await;

    assert_eq!(*harness.docker.killed.lock().unwrap(), 1);
    wait_for_completion(&harness.runtime).await;
    assert_eq!(*harness.executor.shard.output_seals.lock().unwrap(), 0);
    assert_eq!(harness.runtime.drain_completions().unwrap(), 1);
    let frames = drain_frames(&harness.outbox);
    assert!(
        frames.iter().any(|frame| matches!(
            frame,
            NodeControlFrame::Terminal { outcome, .. }
                if matches!(outcome.as_ref(), NodeTerminalOutcome::Failure {
                    code: NodeFailureCode::DeadlineExceeded, ..
                })
        )),
        "expected durable deadline failure: {frames:#?}"
    );
    assert!(!frames.iter().any(|frame| matches!(
        frame,
        NodeControlFrame::Terminal { outcome, .. }
            if matches!(outcome.as_ref(), NodeTerminalOutcome::Success(_))
    )));
    let recovered = harness.journal.recover(&identity(), 3).unwrap();
    assert!(recovered.replayable.is_empty());
    assert!(recovered.quarantined.is_empty());
    assert!(harness.runtime.ready_for_rotation().unwrap());
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn concurrent_runtime_enforces_the_signed_smoke_deadline() {
    assert_durable_execution_deadline(false).await;
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn concurrent_runtime_enforces_the_signed_run_deadline() {
    assert_durable_execution_deadline(true).await;
}

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
    fs::set_permissions(protocol_root.path(), fs::Permissions::from_mode(0o500)).unwrap();

    assert!(matches!(
        handler.handle(infer_frame(1)).await,
        Err(ControlError::Protocol(ProtocolError::Io(_)))
    ));
    assert_eq!(runtime.active_count().unwrap(), 0);
    assert_eq!(outbox.pending_count().unwrap(), 0);
    let recovery = journal.recover(&identity(), 3).unwrap();
    assert!(recovery.replayable.is_empty());
    assert!(recovery.quarantined.is_empty());
    fs::set_permissions(protocol_root.path(), fs::Permissions::from_mode(0o700)).unwrap();
}
