use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use async_trait::async_trait;
use jain_smartcluster_fabric::{Digest32, Placement, VerifiedWorkUnit};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::docker::{
    DockerEnginePort, DockerError, DockerLogChunk, DockerLogEvent, DockerRunSpec, ExitStatus,
};
use crate::{AcceptedAssignment, AssignmentJournal, NodeRuntimeIdentity, RecoveryReport};

pub const MAX_STREAM_CHUNK_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct IsolationEvidence {
    pub(crate) digest: Digest32,
    pub(crate) namespaces: bool,
    pub(crate) device_filter: bool,
    pub(crate) seccomp: bool,
    pub(crate) apparmor_or_landlock: bool,
    pub(crate) descendant_fencing: bool,
    pub(crate) network_isolation: bool,
    pub(crate) metadata_and_control_plane_denied: bool,
    pub(crate) io_limits: bool,
    pub(crate) seccomp_profile: String,
    pub(crate) apparmor_profile: String,
}

impl IsolationEvidence {
    fn qualified(&self) -> bool {
        self.digest != [0; 32]
            && self.namespaces
            && self.device_filter
            && self.seccomp
            && self.apparmor_or_landlock
            && self.descendant_fencing
            && self.network_isolation
            && self.metadata_and_control_plane_denied
            && self.io_limits
            && self.seccomp_profile.starts_with('/')
            && !self.seccomp_profile.contains("..")
            && !self.apparmor_profile.is_empty()
    }

    pub const fn digest(&self) -> Digest32 {
        self.digest
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ContainerIsolationReceipt {
    pub container_id: String,
    pub qualification_digest: Digest32,
    pub work_unit_digest: Digest32,
    pub placement_digest: Digest32,
    pub gpu_uuids: Vec<String>,
    pub cgroup_path: PathBuf,
    pub stopped_before_executable_bytes: bool,
    pub cgroup_owned: bool,
    pub cgroup_membership_verified: bool,
    pub descendant_fencing: bool,
    pub device_filter_enforced: bool,
    pub io_limits_enforced: bool,
}

#[async_trait]
pub trait ContainerIsolationPort: Send + Sync {
    async fn enforce_stopped(
        &self,
        container_id: &str,
        spec: &DockerRunSpec,
        qualification_digest: Digest32,
        work_unit_digest: Digest32,
        placement_digest: Digest32,
    ) -> Result<ContainerIsolationReceipt, ExecutionError>;
}

pub trait ClockPort: Send + Sync {
    fn now_ms(&self) -> Result<u64, ExecutionError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SystemClock;

impl ClockPort for SystemClock {
    fn now_ms(&self) -> Result<u64, ExecutionError> {
        use std::time::{SystemTime, UNIX_EPOCH};

        let elapsed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| ExecutionError::ClockUnavailable)?;
        u64::try_from(elapsed.as_millis()).map_err(|_| ExecutionError::ClockUnavailable)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedInput {
    pub artifact_id: String,
    pub host_path: PathBuf,
    pub container_path: String,
    pub read_only: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactPublication {
    pub relative_path: String,
    pub media_type: String,
    pub maximum_bytes: u64,
    pub required: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PublishedArtifact {
    pub artifact_id: String,
    pub relative_path: String,
    pub digest: Digest32,
    pub durable_receipt_digest: Digest32,
    pub writable_domains: u16,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NetworkLease {
    pub run_id: String,
    pub name: String,
    pub network_id: String,
    pub policy_digest: Digest32,
}

#[async_trait]
pub trait ShardPort: Send + Sync {
    async fn stage_inputs(
        &self,
        project_id: &str,
        run_id: &str,
        inputs: &[jain_smartcluster_fabric::ArtifactMount],
    ) -> Result<Vec<PreparedInput>, ExecutionError>;

    async fn allocate_output_staging(
        &self,
        project_id: &str,
        run_id: &str,
        phase: ExecutionPhase,
    ) -> Result<PathBuf, ExecutionError>;

    async fn seal_outputs(
        &self,
        project_id: &str,
        run_id: &str,
        staging: &Path,
        outputs: &[ArtifactPublication],
        minimum_writable_domains: u16,
    ) -> Result<Vec<PublishedArtifact>, ExecutionError>;

    async fn seal_log(
        &self,
        project_id: &str,
        run_id: &str,
        phase: ExecutionPhase,
        stdout: &[u8],
        stderr: &[u8],
        minimum_writable_domains: u16,
    ) -> Result<Vec<PublishedArtifact>, ExecutionError>;

    async fn cleanup_staging(&self, run_id: &str) -> Result<(), ExecutionError>;
}

#[async_trait]
pub trait NetworkPolicyPort: Send + Sync {
    async fn allocate(
        &self,
        run_id: &str,
        open_outbound: bool,
        byte_limit: u64,
        connection_limit: u32,
        deadline_ms: u64,
    ) -> Result<NetworkLease, ExecutionError>;

    async fn revoke(&self, lease: &NetworkLease) -> Result<(), ExecutionError>;
}

#[async_trait]
pub trait CancellationPort: Send + Sync {
    async fn start_unless_cancelled(
        &self,
        docker: &dyn DockerEnginePort,
        container_id: &str,
        run_id: &str,
        attempt: u32,
        fence: u64,
    ) -> Result<(), ExecutionError>;

    async fn cancelled(&self, run_id: &str, attempt: u32, fence: u64);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MetricsSample {
    pub cpu_millis: u64,
    pub memory_bytes: u64,
    pub gpu_millis: u64,
}

#[async_trait]
pub trait MetricsPort: Send + Sync {
    async fn sample(
        &self,
        container_id: &str,
        gpu_uuids: &[String],
    ) -> Result<MetricsSample, ExecutionError>;
}

#[async_trait]
pub trait EventSink: Send + Sync {
    async fn emit(&self, event: NodeEvent) -> Result<(), ExecutionError>;
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionPhase {
    Smoke,
    Run,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum NodeEvent {
    SmokeStarted(EventIdentity),
    SmokePassed(EventIdentity),
    SmokeFailed {
        identity: EventIdentity,
        reason: String,
    },
    StdoutChunk {
        identity: EventIdentity,
        bytes: Vec<u8>,
    },
    StderrChunk {
        identity: EventIdentity,
        bytes: Vec<u8>,
    },
    Progress {
        identity: EventIdentity,
        completed: u64,
        total: Option<u64>,
        message: String,
    },
    Metrics {
        identity: EventIdentity,
        cpu_millis: u64,
        memory_bytes: u64,
        gpu_millis: u64,
    },
    ArtifactPublished {
        identity: EventIdentity,
        artifact: PublishedArtifact,
    },
    StreamGap {
        identity: EventIdentity,
        first_available_cursor: u64,
        durable_log_artifact_ids: Vec<String>,
    },
}

impl NodeEvent {
    pub(crate) fn identity(&self) -> &EventIdentity {
        match self {
            Self::SmokeStarted(identity)
            | Self::SmokePassed(identity)
            | Self::SmokeFailed { identity, .. }
            | Self::StdoutChunk { identity, .. }
            | Self::StderrChunk { identity, .. }
            | Self::Progress { identity, .. }
            | Self::Metrics { identity, .. }
            | Self::ArtifactPublished { identity, .. }
            | Self::StreamGap { identity, .. } => identity,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct EventIdentity {
    pub run_id: String,
    pub attempt: u32,
    pub fence: u64,
    pub cursor: u64,
    pub stream_offset: u64,
    pub control_position: u64,
    pub phase: ExecutionPhase,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ExecutionReceipt {
    pub outputs: Vec<PublishedArtifact>,
    pub logs: Vec<PublishedArtifact>,
    pub smoke_only: bool,
}

pub struct JainNodeExecutor<D, S, N, C, M, E, I, K>
where
    D: DockerEnginePort,
    S: ShardPort,
    N: NetworkPolicyPort,
    C: CancellationPort,
    M: MetricsPort,
    E: EventSink,
    I: ContainerIsolationPort,
    K: ClockPort,
{
    docker: Arc<D>,
    shard: Arc<S>,
    network: Arc<N>,
    cancellation: Arc<C>,
    metrics: Arc<M>,
    events: Arc<E>,
    isolation: IsolationEvidence,
    identity: NodeRuntimeIdentity,
    assignment_journal: AssignmentJournal,
    isolation_enforcer: Arc<I>,
    clock: Arc<K>,
}

pub struct ExecutorPorts<D, S, N, C, M, E, I, K> {
    pub docker: Arc<D>,
    pub shard: Arc<S>,
    pub network: Arc<N>,
    pub cancellation: Arc<C>,
    pub metrics: Arc<M>,
    pub events: Arc<E>,
    pub isolation_enforcer: Arc<I>,
    pub clock: Arc<K>,
}

impl<D, S, N, C, M, E, I, K> JainNodeExecutor<D, S, N, C, M, E, I, K>
where
    D: DockerEnginePort + 'static,
    S: ShardPort,
    N: NetworkPolicyPort,
    C: CancellationPort,
    M: MetricsPort,
    E: EventSink,
    I: ContainerIsolationPort,
    K: ClockPort,
{
    pub fn new(
        ports: ExecutorPorts<D, S, N, C, M, E, I, K>,
        isolation: IsolationEvidence,
        identity: NodeRuntimeIdentity,
        assignment_journal: AssignmentJournal,
    ) -> Result<Self, ExecutionError> {
        if !isolation.qualified() {
            return Err(ExecutionError::UnsupportedHost);
        }
        Ok(Self {
            docker: ports.docker,
            shard: ports.shard,
            network: ports.network,
            cancellation: ports.cancellation,
            metrics: ports.metrics,
            events: ports.events,
            isolation,
            identity,
            assignment_journal,
            isolation_enforcer: ports.isolation_enforcer,
            clock: ports.clock,
        })
    }

    pub fn accept_assignment(
        &self,
        work: VerifiedWorkUnit,
        placement: Placement,
        session_sequence: u64,
        now_ms: u64,
    ) -> Result<
        (
            AcceptedAssignment,
            jain_smartcluster_fabric::WorkerAcknowledgement,
        ),
        ExecutionError,
    > {
        if placement.isolation_evidence_digest != self.isolation.digest {
            return Err(ExecutionError::UnsupportedHost);
        }
        self.assignment_journal
            .accept_before_ack(&self.identity, work, placement, session_sequence, now_ms)
            .map_err(ExecutionError::Assignment)
    }

    pub fn recover_assignments(&self, now_ms: u64) -> Result<RecoveryReport, ExecutionError> {
        self.assignment_journal
            .recover(&self.identity, now_ms)
            .map_err(ExecutionError::Assignment)
    }

    pub(crate) fn recovered_acknowledgement(
        &self,
        accepted: &AcceptedAssignment,
    ) -> Result<jain_smartcluster_fabric::WorkerAcknowledgement, ExecutionError> {
        accepted
            .acknowledgement(&self.identity)
            .map_err(ExecutionError::Assignment)
    }

    pub(crate) fn reject_prepared(
        &self,
        accepted: &AcceptedAssignment,
    ) -> Result<(), ExecutionError> {
        self.assignment_journal
            .mark_rejected(accepted)
            .map_err(ExecutionError::Assignment)
    }

    pub async fn execute(
        &self,
        accepted: &AcceptedAssignment,
    ) -> Result<ExecutionReceipt, ExecutionError> {
        let work = accepted.work();
        let placement = accepted.placement();
        work.validate()
            .map_err(|_| ExecutionError::InvalidWorkUnit)?;
        if placement.isolation_evidence_digest != self.isolation.digest {
            return Err(ExecutionError::UnsupportedHost);
        }
        self.assignment_journal
            .mark_running(accepted)
            .map_err(ExecutionError::Assignment)?;
        let run_id = work.run_id.to_string();
        let mut emitter = Emitter::new(work, ExecutionPhase::Smoke);
        let inputs = self
            .shard
            .stage_inputs(&work.project_id, &run_id, &work.inputs)
            .await?;
        if inputs.len() != work.inputs.len()
            || inputs.iter().any(|input| !input.read_only)
            || inputs.iter().any(|input| {
                !input.host_path.is_absolute()
                    || !input.container_path.starts_with("/inputs/")
                    || input.container_path.contains("..")
            })
        {
            let _ = self.shard.cleanup_staging(&run_id).await;
            return Err(ExecutionError::InvalidStaging);
        }

        let smoke_result = self
            .execute_phase(
                work,
                placement,
                &inputs,
                ExecutionPhase::Smoke,
                &mut emitter,
            )
            .await;
        let (smoke_logs, _) = match smoke_result {
            Ok(result) => result,
            Err(error) => {
                let _ = self.shard.cleanup_staging(&run_id).await;
                return Err(error);
            }
        };
        if work.smoke_only {
            self.shard.cleanup_staging(&run_id).await?;
            self.assignment_journal
                .mark_terminal(accepted)
                .map_err(ExecutionError::Assignment)?;
            return Ok(ExecutionReceipt {
                outputs: Vec::new(),
                logs: smoke_logs,
                smoke_only: true,
            });
        }

        let run_result = self
            .execute_phase(work, placement, &inputs, ExecutionPhase::Run, &mut emitter)
            .await;
        let (run_logs, output_staging) = match run_result {
            Ok(result) => result,
            Err(error) => {
                let _ = self.shard.cleanup_staging(&run_id).await;
                return Err(error);
            }
        };
        let declarations = work
            .outputs
            .iter()
            .map(|output| ArtifactPublication {
                relative_path: output.relative_path.clone(),
                media_type: output.media_type.clone(),
                maximum_bytes: output.maximum_bytes,
                required: output.required,
            })
            .collect::<Vec<_>>();
        let outputs_result = self
            .shard
            .seal_outputs(
                &work.project_id,
                &run_id,
                &output_staging,
                &declarations,
                work.minimum_writable_domains,
            )
            .await;
        let outputs = match outputs_result {
            Ok(outputs) => outputs,
            Err(error) => {
                let _ = self.shard.cleanup_staging(&run_id).await;
                return Err(error);
            }
        };
        let published_paths = outputs
            .iter()
            .map(|output| output.relative_path.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        if declarations
            .iter()
            .filter(|output| output.required)
            .any(|output| !published_paths.contains(output.relative_path.as_str()))
            || outputs.iter().any(|artifact| {
                artifact.durable_receipt_digest == [0; 32]
                    || artifact.writable_domains < work.minimum_writable_domains
            })
        {
            let _ = self.shard.cleanup_staging(&run_id).await;
            return Err(ExecutionError::DurabilityUnavailable);
        }
        for artifact in &outputs {
            self.events
                .emit(NodeEvent::ArtifactPublished {
                    identity: emitter.next(0),
                    artifact: artifact.clone(),
                })
                .await?;
        }
        let mut logs = smoke_logs;
        logs.extend(run_logs);
        self.shard.cleanup_staging(&run_id).await?;
        self.assignment_journal
            .mark_terminal(accepted)
            .map_err(ExecutionError::Assignment)?;
        Ok(ExecutionReceipt {
            outputs,
            logs,
            smoke_only: false,
        })
    }

    pub(crate) fn terminalize_failure(
        &self,
        accepted: &AcceptedAssignment,
    ) -> Result<(), ExecutionError> {
        self.assignment_journal
            .mark_terminal(accepted)
            .map_err(ExecutionError::Assignment)
    }

    async fn execute_phase(
        &self,
        work: &VerifiedWorkUnit,
        placement: &Placement,
        inputs: &[PreparedInput],
        phase: ExecutionPhase,
        emitter: &mut Emitter,
    ) -> Result<(Vec<PublishedArtifact>, PathBuf), ExecutionError> {
        let run_id = work.run_id.to_string();
        let output_staging = self
            .shard
            .allocate_output_staging(&work.project_id, &run_id, phase)
            .await?;
        if !output_staging.is_absolute() {
            return Err(ExecutionError::InvalidStaging);
        }
        let network = self
            .network
            .allocate(
                &run_id,
                work.open_outbound,
                work.network_bytes,
                work.network_connections,
                work.network_deadline_ms,
            )
            .await?;
        if network.run_id != run_id
            || network.name.is_empty()
            || matches!(
                network.name.as_str(),
                "host" | "bridge" | "default" | "none"
            )
            || network.network_id.is_empty()
            || network.policy_digest == [0; 32]
        {
            let _ = self.network.revoke(&network).await;
            return Err(ExecutionError::NetworkPolicyUnavailable);
        }

        let argv = match phase {
            ExecutionPhase::Smoke => work.smoke_argv.clone(),
            ExecutionPhase::Run => work.argv.clone(),
        };
        let mut environment = work.caller_environment.clone();
        environment.extend(platform_environment(work, placement, phase));
        let spec = DockerRunSpec {
            name: format!(
                "jain-{}-{}-{}",
                work.run_id,
                work.attempt,
                phase_name(phase)
            ),
            image: work.image_reference.clone(),
            argv,
            working_directory: work.working_directory.clone(),
            environment,
            readonly_binds: inputs
                .iter()
                .map(|input| (input.host_path.clone(), input.container_path.clone()))
                .collect(),
            output_staging: output_staging.clone(),
            scratch_bytes: work.limits.scratch_bytes,
            cpu_millicores: work.limits.cpu_millicores,
            memory_bytes: work.limits.memory_bytes,
            pids: work.limits.pids,
            read_bytes: work.limits.read_bytes,
            write_bytes: work.limits.write_bytes,
            io_operations: work.limits.io_operations,
            gpu_uuids: placement.gpu_uuids.clone(),
            network_name: network.name.clone(),
            seccomp_profile: self.isolation.seccomp_profile.clone(),
            apparmor_profile: self.isolation.apparmor_profile.clone(),
        };
        emitter.phase = phase;
        let result = self
            .run_container(work, placement, phase, &spec, emitter)
            .await;
        let revoke_result = self.network.revoke(&network).await;
        match (result, revoke_result) {
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
            (Ok((stdout, stderr)), Ok(())) => {
                let logs = self
                    .shard
                    .seal_log(
                        &work.project_id,
                        &run_id,
                        phase,
                        &stdout,
                        &stderr,
                        work.minimum_writable_domains,
                    )
                    .await?;
                if logs.iter().any(|artifact| {
                    artifact.durable_receipt_digest == [0; 32]
                        || artifact.writable_domains < work.minimum_writable_domains
                }) {
                    return Err(ExecutionError::DurabilityUnavailable);
                }
                Ok((logs, output_staging))
            }
        }
    }

    async fn run_container(
        &self,
        work: &VerifiedWorkUnit,
        placement: &Placement,
        phase: ExecutionPhase,
        spec: &DockerRunSpec,
        emitter: &mut Emitter,
    ) -> Result<(Vec<u8>, Vec<u8>), ExecutionError> {
        let container_id = self.docker.create(spec).await?;
        let outcome = async {
            let isolation_receipt = self
                .isolation_enforcer
                .enforce_stopped(
                    &container_id,
                    spec,
                    self.isolation.digest,
                    work.work_unit_digest(),
                    placement
                        .digest()
                        .map_err(|_| ExecutionError::InvalidWorkUnit)?,
                )
                .await?;
            let placement_digest = placement
                .digest()
                .map_err(|_| ExecutionError::InvalidWorkUnit)?;
            if isolation_receipt.container_id != container_id
                || isolation_receipt.qualification_digest != self.isolation.digest
                || isolation_receipt.work_unit_digest != work.work_unit_digest()
                || isolation_receipt.placement_digest != placement_digest
                || isolation_receipt.gpu_uuids != placement.gpu_uuids
                || !isolation_receipt.cgroup_path.is_absolute()
                || !isolation_receipt.stopped_before_executable_bytes
                || !isolation_receipt.cgroup_owned
                || !isolation_receipt.cgroup_membership_verified
                || !isolation_receipt.descendant_fencing
                || !isolation_receipt.device_filter_enforced
                || !isolation_receipt.io_limits_enforced
            {
                return Err(ExecutionError::UnsupportedHost);
            }
            self.identity
                .reverify_work(work)
                .map_err(ExecutionError::Assignment)?;
            self.identity
                .reverify_placement(work, placement, self.clock.now_ms()?)
                .map_err(|_| ExecutionError::StaleAuthority)?;
            if phase == ExecutionPhase::Smoke {
                self.events
                    .emit(NodeEvent::SmokeStarted(emitter.next(0)))
                    .await?;
            }
            let mut logs = self.docker.logs(&container_id).await?;
            self.cancellation
                .start_unless_cancelled(
                    &*self.docker,
                    &container_id,
                    &work.run_id.to_string(),
                    work.attempt,
                    work.attempt_fence,
                )
                .await?;
            let initial_metrics = self.metrics.sample(&container_id, &spec.gpu_uuids).await?;
            self.events
                .emit(NodeEvent::Metrics {
                    identity: emitter.next(0),
                    cpu_millis: initial_metrics.cpu_millis,
                    memory_bytes: initial_metrics.memory_bytes,
                    gpu_millis: initial_metrics.gpu_millis,
                })
                .await?;
            let timeout_ms = match phase {
                ExecutionPhase::Smoke => work.smoke_timeout_ms,
                ExecutionPhase::Run => work.limits.wall_time_ms,
            };
            let mut wait = Box::pin(self.docker.wait(&container_id));
            let cancellation_run_id = work.run_id.to_string();
            let mut cancel = Box::pin(self.cancellation.cancelled(
                &cancellation_run_id,
                work.attempt,
                work.attempt_fence,
            ));
            let deadline = tokio::time::sleep(Duration::from_millis(timeout_ms));
            tokio::pin!(deadline);
            let metrics_tick = tokio::time::interval(Duration::from_secs(1));
            tokio::pin!(metrics_tick);
            let mut stdout = Vec::new();
            let mut stderr = Vec::new();
            let mut logs_complete = false;
            let mut status = None;
            loop {
                if logs_complete && status.is_some() {
                    break;
                }
                tokio::select! {
                    result = &mut wait, if status.is_none() => status = Some(result?),
                    () = &mut cancel => {
                        self.docker.kill(&container_id).await?;
                        return Err(ExecutionError::Cancelled);
                    }
                    () = &mut deadline => {
                        self.docker.kill(&container_id).await?;
                        return Err(ExecutionError::DeadlineExceeded);
                    }
                    event = logs.recv(), if !logs_complete => {
                        match event {
                            Some(Ok(DockerLogEvent::Chunk(chunk))) => {
                                record_and_emit_chunk(
                                    &*self.events,
                                    emitter,
                                    chunk,
                                    &mut stdout,
                                    &mut stderr,
                                    work.stdout_bytes,
                                    work.stderr_bytes,
                                ).await?;
                            }
                            Some(Ok(DockerLogEvent::Complete)) => logs_complete = true,
                            Some(Err(error)) => return Err(error.into()),
                            None => return Err(ExecutionError::StreamProtocolViolation),
                        }
                    }
                    _ = metrics_tick.tick() => {
                        let sample = self.metrics.sample(&container_id, &spec.gpu_uuids).await?;
                        self.events.emit(NodeEvent::Metrics {
                            identity: emitter.next(0),
                            cpu_millis: sample.cpu_millis,
                            memory_bytes: sample.memory_bytes,
                            gpu_millis: sample.gpu_millis,
                        }).await?;
                    }
                }
            }
            ensure_success(status.ok_or(ExecutionError::StreamProtocolViolation)?)?;
            if phase == ExecutionPhase::Smoke {
                self.events
                    .emit(NodeEvent::SmokePassed(emitter.next(0)))
                    .await?;
            }
            Ok((stdout, stderr))
        }
        .await;
        let remove = self.docker.remove(&container_id).await;
        match (outcome, remove) {
            (Err(error), _) => {
                if phase == ExecutionPhase::Smoke {
                    let _ = self
                        .events
                        .emit(NodeEvent::SmokeFailed {
                            identity: emitter.next(0),
                            reason: error.to_string(),
                        })
                        .await;
                }
                Err(error)
            }
            (Ok(_), Err(error)) => Err(error.into()),
            (Ok(output), Ok(())) => Ok(output),
        }
    }
}

async fn record_and_emit_chunk<E: EventSink>(
    events: &E,
    emitter: &mut Emitter,
    chunk: DockerLogChunk,
    stdout: &mut Vec<u8>,
    stderr: &mut Vec<u8>,
    stdout_budget: u64,
    stderr_budget: u64,
) -> Result<(), ExecutionError> {
    if chunk.bytes.len() > MAX_STREAM_CHUNK_BYTES {
        return Err(ExecutionError::StreamProtocolViolation);
    }
    let (recorded, budget) = if chunk.stderr {
        (stderr, stderr_budget)
    } else {
        (stdout, stdout_budget)
    };
    let new_length = recorded
        .len()
        .checked_add(chunk.bytes.len())
        .ok_or(ExecutionError::LogBudgetExceeded)?;
    if u64::try_from(new_length).map_err(|_| ExecutionError::LogBudgetExceeded)? > budget {
        return Err(ExecutionError::LogBudgetExceeded);
    }
    let identity = emitter.next(chunk.bytes.len() as u64);
    recorded.extend_from_slice(&chunk.bytes);
    events
        .emit(if chunk.stderr {
            NodeEvent::StderrChunk {
                identity,
                bytes: chunk.bytes,
            }
        } else {
            NodeEvent::StdoutChunk {
                identity,
                bytes: chunk.bytes,
            }
        })
        .await
}

fn ensure_success(status: ExitStatus) -> Result<(), ExecutionError> {
    if status.oom_killed {
        Err(ExecutionError::MemoryLimitExceeded)
    } else if status.code != 0 {
        Err(ExecutionError::NonZeroExit(status.code))
    } else {
        Ok(())
    }
}

fn platform_environment(
    work: &VerifiedWorkUnit,
    placement: &Placement,
    phase: ExecutionPhase,
) -> BTreeMap<String, String> {
    BTreeMap::from([
        ("JAIN_RUN_ID".to_owned(), work.run_id.to_string()),
        ("JAIN_ATTEMPT".to_owned(), work.attempt.to_string()),
        (
            "JAIN_ATTEMPT_FENCE".to_owned(),
            work.attempt_fence.to_string(),
        ),
        (
            "JAIN_CONTROL_POSITION".to_owned(),
            work.control_position.to_string(),
        ),
        ("JAIN_NODE_ID".to_owned(), placement.node_id.clone()),
        ("JAIN_PHASE".to_owned(), phase_name(phase).to_owned()),
        ("JAIN_OUTPUT_DIR".to_owned(), "/outputs".to_owned()),
    ])
}

fn phase_name(phase: ExecutionPhase) -> &'static str {
    match phase {
        ExecutionPhase::Smoke => "smoke",
        ExecutionPhase::Run => "run",
    }
}

struct Emitter {
    run_id: String,
    attempt: u32,
    fence: u64,
    cursor: u64,
    stream_offset: u64,
    control_position: u64,
    phase: ExecutionPhase,
}

impl Emitter {
    fn new(work: &VerifiedWorkUnit, phase: ExecutionPhase) -> Self {
        Self {
            run_id: work.run_id.to_string(),
            attempt: work.attempt,
            fence: work.attempt_fence,
            cursor: 0,
            stream_offset: 0,
            control_position: work.control_position,
            phase,
        }
    }

    fn next(&mut self, stream_bytes: u64) -> EventIdentity {
        self.cursor = self.cursor.saturating_add(1);
        let identity = EventIdentity {
            run_id: self.run_id.clone(),
            attempt: self.attempt,
            fence: self.fence,
            cursor: self.cursor,
            stream_offset: self.stream_offset,
            control_position: self.control_position,
            phase: self.phase,
        };
        self.stream_offset = self.stream_offset.saturating_add(stream_bytes);
        identity
    }
}

#[derive(Debug, Error)]
pub enum ExecutionError {
    #[error("unsupported_host")]
    UnsupportedHost,
    #[error("invalid_work_unit")]
    InvalidWorkUnit,
    #[error("invalid_staging")]
    InvalidStaging,
    #[error("network_policy_unavailable")]
    NetworkPolicyUnavailable,
    #[error("durability_unavailable")]
    DurabilityUnavailable,
    #[error("cancelled")]
    Cancelled,
    #[error("cancellation_state_invalid")]
    CancellationState,
    #[error("deadline_exceeded")]
    DeadlineExceeded,
    #[error("memory_limit_exceeded")]
    MemoryLimitExceeded,
    #[error("process exited with status {0}")]
    NonZeroExit(i32),
    #[error("log_budget_exceeded")]
    LogBudgetExceeded,
    #[error("stream_protocol_violation")]
    StreamProtocolViolation,
    #[error("clock_unavailable")]
    ClockUnavailable,
    #[error("stale_authority")]
    StaleAuthority,
    #[error("outbound_control_unavailable")]
    OutboundControlUnavailable,
    #[error("assignment journal error: {0}")]
    Assignment(crate::AssignmentError),
    #[error("isolation broker error: {0}")]
    IsolationBroker(String),
    #[error("storage error: {0}")]
    Storage(String),
    #[error("network error: {0}")]
    Network(String),
    #[error(transparent)]
    Docker(#[from] DockerError),
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
