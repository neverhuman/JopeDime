use std::{collections::BTreeMap, future::pending, path::PathBuf, sync::Mutex};

use ed25519_dalek::SigningKey;
use jain_smartcluster_fabric::{
    image_signed_bytes, projection_signed_bytes, ArtifactMount, DomainSignature, GpuRequirement,
    OutputDeclaration, ResourceLimits, SignatureAuthority, SignedWorkUnitProjection,
    VerifiedWorkUnit, WorkUnitProjection,
};
use tokio::sync::mpsc;
use ulid::Ulid;

use super::*;
use crate::docker::{DockerError, DockerLogChunk, DockerLogEvent, DockerRunSpec, ExitStatus};

#[path = "dispatch_tests.rs"]
mod dispatch_tests;

#[derive(Default)]
struct DockerMock {
    specs: Mutex<Vec<DockerRunSpec>>,
    exits: Mutex<Vec<ExitStatus>>,
    killed: Mutex<usize>,
    removed: Mutex<usize>,
    started: Mutex<Vec<String>>,
    block_wait: bool,
    log_error: bool,
    incomplete_log: bool,
}

#[async_trait]
impl DockerEnginePort for DockerMock {
    async fn create(&self, spec: &DockerRunSpec) -> Result<String, DockerError> {
        self.specs.lock().unwrap().push(spec.clone());
        Ok(format!("container-{}", self.specs.lock().unwrap().len()))
    }

    async fn start(&self, container_id: &str) -> Result<(), DockerError> {
        self.started.lock().unwrap().push(container_id.to_owned());
        Ok(())
    }

    async fn logs(
        &self,
        _container_id: &str,
    ) -> Result<mpsc::Receiver<Result<DockerLogEvent, DockerError>>, DockerError> {
        let (sender, receiver) = mpsc::channel(4);
        if self.log_error {
            sender
                .send(Err(DockerError::Transport(
                    "injected log failure".to_owned(),
                )))
                .await
                .unwrap();
            return Ok(receiver);
        }
        sender
            .send(Ok(DockerLogEvent::Chunk(DockerLogChunk {
                stderr: false,
                bytes: b"stdout\n".to_vec(),
            })))
            .await
            .unwrap();
        sender
            .send(Ok(DockerLogEvent::Chunk(DockerLogChunk {
                stderr: true,
                bytes: b"stderr\n".to_vec(),
            })))
            .await
            .unwrap();
        if !self.incomplete_log {
            sender.send(Ok(DockerLogEvent::Complete)).await.unwrap();
        }
        drop(sender);
        Ok(receiver)
    }

    async fn wait(&self, _container_id: &str) -> Result<ExitStatus, DockerError> {
        if self.block_wait {
            pending::<()>().await;
        }
        Ok(self.exits.lock().unwrap().remove(0))
    }

    async fn kill(&self, _container_id: &str) -> Result<(), DockerError> {
        *self.killed.lock().unwrap() += 1;
        Ok(())
    }

    async fn remove(&self, _container_id: &str) -> Result<(), DockerError> {
        *self.removed.lock().unwrap() += 1;
        Ok(())
    }
}

#[derive(Default)]
struct ShardMock {
    durable: bool,
    output_seals: Mutex<usize>,
    cleanups: Mutex<usize>,
}

#[async_trait]
impl ShardPort for ShardMock {
    async fn stage_inputs(
        &self,
        _project_id: &str,
        _run_id: &str,
        inputs: &[ArtifactMount],
    ) -> Result<Vec<PreparedInput>, ExecutionError> {
        Ok(inputs
            .iter()
            .map(|input| PreparedInput {
                artifact_id: input.artifact_id.clone(),
                host_path: PathBuf::from(format!("/var/lib/jainnode/inputs/{}", input.artifact_id)),
                container_path: input.mount_path.clone(),
                read_only: true,
            })
            .collect())
    }

    async fn allocate_output_staging(
        &self,
        _project_id: &str,
        run_id: &str,
        phase: ExecutionPhase,
    ) -> Result<PathBuf, ExecutionError> {
        Ok(PathBuf::from(format!(
            "/var/lib/jainnode/outputs/{run_id}/{}",
            phase_name(phase)
        )))
    }

    async fn seal_outputs(
        &self,
        _project_id: &str,
        _run_id: &str,
        _staging: &Path,
        outputs: &[ArtifactPublication],
        minimum_writable_domains: u16,
    ) -> Result<Vec<PublishedArtifact>, ExecutionError> {
        *self.output_seals.lock().unwrap() += 1;
        Ok(outputs
            .iter()
            .map(|output| PublishedArtifact {
                artifact_id: format!("output-{}", output.relative_path),
                relative_path: output.relative_path.clone(),
                digest: [20; 32],
                durable_receipt_digest: if self.durable { [21; 32] } else { [0; 32] },
                writable_domains: if self.durable {
                    minimum_writable_domains
                } else {
                    0
                },
            })
            .collect())
    }

    async fn seal_log(
        &self,
        _project_id: &str,
        _run_id: &str,
        phase: ExecutionPhase,
        _stdout: &[u8],
        _stderr: &[u8],
        minimum_writable_domains: u16,
    ) -> Result<Vec<PublishedArtifact>, ExecutionError> {
        Ok(vec![PublishedArtifact {
            artifact_id: format!("{}-log", phase_name(phase)),
            relative_path: format!("{}.log", phase_name(phase)),
            digest: [22; 32],
            durable_receipt_digest: if self.durable { [23; 32] } else { [0; 32] },
            writable_domains: if self.durable {
                minimum_writable_domains
            } else {
                0
            },
        }])
    }

    async fn cleanup_staging(&self, _run_id: &str) -> Result<(), ExecutionError> {
        *self.cleanups.lock().unwrap() += 1;
        Ok(())
    }
}

#[derive(Default)]
struct NetworkMock {
    allocations: Mutex<usize>,
    revocations: Mutex<usize>,
}

#[async_trait]
impl NetworkPolicyPort for NetworkMock {
    async fn allocate(
        &self,
        run_id: &str,
        _open_outbound: bool,
        _byte_limit: u64,
        _connection_limit: u32,
        _deadline_ms: u64,
    ) -> Result<NetworkLease, ExecutionError> {
        *self.allocations.lock().unwrap() += 1;
        Ok(NetworkLease {
            run_id: run_id.to_owned(),
            name: "jain-dedicated-network".to_owned(),
            network_id: "network-id-1".to_owned(),
            policy_digest: [24; 32],
        })
    }

    async fn revoke(&self, _lease: &NetworkLease) -> Result<(), ExecutionError> {
        *self.revocations.lock().unwrap() += 1;
        Ok(())
    }
}

struct NeverCancel;

#[async_trait]
impl CancellationPort for NeverCancel {
    async fn start_unless_cancelled(
        &self,
        docker: &dyn DockerEnginePort,
        container_id: &str,
        _run_id: &str,
        _attempt: u32,
        _fence: u64,
    ) -> Result<(), ExecutionError> {
        docker.start(container_id).await.map_err(Into::into)
    }

    async fn cancelled(&self, _run_id: &str, _attempt: u32, _fence: u64) {
        pending::<()>().await;
    }
}

struct ImmediateCancel;

#[async_trait]
impl CancellationPort for ImmediateCancel {
    async fn start_unless_cancelled(
        &self,
        docker: &dyn DockerEnginePort,
        container_id: &str,
        _run_id: &str,
        _attempt: u32,
        _fence: u64,
    ) -> Result<(), ExecutionError> {
        docker.start(container_id).await.map_err(Into::into)
    }

    async fn cancelled(&self, _run_id: &str, _attempt: u32, _fence: u64) {}
}

struct AlreadyCancelled;

#[async_trait]
impl CancellationPort for AlreadyCancelled {
    async fn start_unless_cancelled(
        &self,
        _docker: &dyn DockerEnginePort,
        _container_id: &str,
        _run_id: &str,
        _attempt: u32,
        _fence: u64,
    ) -> Result<(), ExecutionError> {
        Err(ExecutionError::Cancelled)
    }

    async fn cancelled(&self, _run_id: &str, _attempt: u32, _fence: u64) {}
}

struct MetricsMock;

#[async_trait]
impl MetricsPort for MetricsMock {
    async fn sample(
        &self,
        _container_id: &str,
        _gpu_uuids: &[String],
    ) -> Result<MetricsSample, ExecutionError> {
        Ok(MetricsSample {
            cpu_millis: 2,
            memory_bytes: 3,
            gpu_millis: 4,
        })
    }
}

struct FixedClock(u64);

impl ClockPort for FixedClock {
    fn now_ms(&self) -> Result<u64, ExecutionError> {
        Ok(self.0)
    }
}

struct IsolationMock;

#[async_trait]
impl ContainerIsolationPort for IsolationMock {
    async fn enforce_stopped(
        &self,
        container_id: &str,
        _spec: &DockerRunSpec,
        qualification_digest: Digest32,
        work_unit_digest: Digest32,
        placement_digest: Digest32,
    ) -> Result<ContainerIsolationReceipt, ExecutionError> {
        Ok(ContainerIsolationReceipt {
            container_id: container_id.to_owned(),
            qualification_digest,
            work_unit_digest,
            placement_digest,
            gpu_uuids: _spec.gpu_uuids.clone(),
            cgroup_path: PathBuf::from(format!("/sys/fs/cgroup/jain/{container_id}")),
            stopped_before_executable_bytes: true,
            cgroup_owned: true,
            cgroup_membership_verified: true,
            descendant_fencing: true,
            device_filter_enforced: true,
            io_limits_enforced: true,
        })
    }
}

struct OrderingIsolationMock {
    docker: Arc<DockerMock>,
}

#[async_trait]
impl ContainerIsolationPort for OrderingIsolationMock {
    async fn enforce_stopped(
        &self,
        container_id: &str,
        spec: &DockerRunSpec,
        qualification_digest: Digest32,
        work_unit_digest: Digest32,
        placement_digest: Digest32,
    ) -> Result<ContainerIsolationReceipt, ExecutionError> {
        assert!(!self
            .docker
            .started
            .lock()
            .unwrap()
            .iter()
            .any(|started| started == container_id));
        IsolationMock
            .enforce_stopped(
                container_id,
                spec,
                qualification_digest,
                work_unit_digest,
                placement_digest,
            )
            .await
    }
}

#[derive(Default)]
struct EventsMock(Mutex<Vec<NodeEvent>>);

#[async_trait]
impl EventSink for EventsMock {
    async fn emit(&self, event: NodeEvent) -> Result<(), ExecutionError> {
        self.0.lock().unwrap().push(event);
        Ok(())
    }
}

fn isolation() -> IsolationEvidence {
    IsolationEvidence {
        digest: [7; 32],
        namespaces: true,
        device_filter: true,
        seccomp: true,
        apparmor_or_landlock: true,
        descendant_fencing: true,
        network_isolation: true,
        metadata_and_control_plane_denied: true,
        io_limits: true,
        seccomp_profile: "/etc/jain/seccomp.json".to_owned(),
        apparmor_profile: "jain-governed-rd".to_owned(),
    }
}

fn work() -> VerifiedWorkUnit {
    let key = hub_key();
    let image_digest = [2; 32];
    let image_hex = image_digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let projection = WorkUnitProjection {
        run_id: Ulid::from(1_u128),
        project_id: "project-1".to_owned(),
        workload_class: crate::WorkloadClass::Training,
        image_reference: format!("registry.invalid/research@sha256:{image_hex}"),
        image_digest,
        argv: vec!["/app/run".to_owned(), "--real".to_owned()],
        smoke_argv: vec!["/app/run".to_owned(), "--smoke".to_owned()],
        smoke_timeout_ms: 1_000,
        smoke_only: false,
        working_directory: "work".to_owned(),
        caller_environment: BTreeMap::from([("TRIAL".to_owned(), "7".to_owned())]),
        inputs: vec![ArtifactMount {
            artifact_id: "input".to_owned(),
            artifact_digest: [4; 32],
            mount_path: "/inputs/data".to_owned(),
            size_bytes: 100,
            durable_receipt_digest: [5; 32],
        }],
        outputs: vec![OutputDeclaration {
            relative_path: "result.json".to_owned(),
            media_type: "application/json".to_owned(),
            maximum_bytes: 1_000,
            required: true,
        }],
        limits: ResourceLimits {
            cpu_millicores: 1_000,
            memory_bytes: 1_000,
            scratch_bytes: 1_000,
            pids: 32,
            wall_time_ms: 10_000,
            read_bytes: 1_000,
            write_bytes: 1_000,
            io_operations: 100,
        },
        gpu: Some(GpuRequirement {
            model: Some("RTX 3090".to_owned()),
            count: 1,
            minimum_vram_bytes: 500,
        }),
        open_outbound: true,
        network_bytes: 1_000,
        network_connections: 10,
        network_deadline_ms: 5_000,
        stdout_bytes: 1_000,
        stderr_bytes: 1_000,
        minimum_writable_domains: 2,
        attempt: 1,
        attempt_fence: 9,
        hub_epoch: 2,
        control_position: 3,
        replay_nonce: [6; 32],
    };
    let issuer = "hub-issuer".to_owned();
    VerifiedWorkUnit::verify(
        SignedWorkUnitProjection {
            image_signature: DomainSignature::sign(
                SignatureAuthority::OciImage,
                issuer.clone(),
                "hub-key".to_owned(),
                &key,
                &image_signed_bytes(&projection.image_reference, projection.image_digest).unwrap(),
            ),
            work_unit_signature: DomainSignature::sign(
                SignatureAuthority::WorkUnitIssuer,
                issuer.clone(),
                "hub-key".to_owned(),
                &key,
                &projection_signed_bytes(&projection).unwrap(),
            ),
            projection,
            issuer,
        },
        "hub-issuer",
        key.verifying_key().to_bytes(),
        "hub-issuer",
        key.verifying_key().to_bytes(),
    )
    .unwrap()
}

fn placement() -> Placement {
    placement_with_gpus(vec!["GPU-exact".to_owned()])
}

fn placement_with_gpus(gpu_uuids: Vec<String>) -> Placement {
    let work = work();
    let key = hub_key();
    let mut placement = Placement {
        node_id: "node-1".to_owned(),
        session_id: "session-1".to_owned(),
        lease_id: "lease-1".to_owned(),
        lease_expires_at_ms: u64::MAX,
        work_unit_digest: work.work_unit_digest(),
        command_digest: work.command_digest(),
        attempt: work.attempt,
        attempt_fence: work.attempt_fence,
        hub_epoch: work.hub_epoch,
        control_position: work.control_position,
        capability_advertisement_digest: [8; 32],
        gpu_uuids,
        cached_input_bytes: 100,
        isolation_evidence_digest: [7; 32],
        signature: DomainSignature {
            authority: SignatureAuthority::PlacementOffer,
            issuer: "hub-issuer".to_owned(),
            key_id: "hub-key".to_owned(),
            public_key: key.verifying_key().to_bytes(),
            signature: Vec::new(),
        },
    };
    placement.signature = DomainSignature::sign(
        SignatureAuthority::PlacementOffer,
        "hub-issuer".to_owned(),
        "hub-key".to_owned(),
        &key,
        &placement.signed_bytes().unwrap(),
    );
    placement
}

fn identity() -> NodeRuntimeIdentity {
    NodeRuntimeIdentity::new(
        "node-1".to_owned(),
        "session-1".to_owned(),
        "node-key-1".to_owned(),
        "hub-issuer".to_owned(),
        hub_key().verifying_key().to_bytes(),
        "hub-issuer".to_owned(),
        hub_key().verifying_key().to_bytes(),
        SigningKey::from_bytes(&[12; 32]),
    )
    .unwrap()
}

fn hub_key() -> SigningKey {
    SigningKey::from_bytes(&[11; 32])
}

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
