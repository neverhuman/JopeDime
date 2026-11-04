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

mod helpers;
mod runner;
pub(super) use helpers::{
    ensure_success, phase_name, platform_environment, record_and_emit_chunk, Emitter,
};

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
