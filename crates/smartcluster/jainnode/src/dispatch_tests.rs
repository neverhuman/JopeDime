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

#[path = "dispatch_tests/concurrent.rs"]
mod concurrent;
#[path = "dispatch_tests/deadlines.rs"]
mod deadlines;
#[path = "dispatch_tests/recovery.rs"]
mod recovery;
#[path = "dispatch_tests/recovery_failures.rs"]
mod recovery_failures;
