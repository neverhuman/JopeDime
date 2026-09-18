#![deny(unsafe_code)]

mod assignment;
mod cancellation;
mod clock;
mod confirmation;
mod control;
mod decimal_bytes;
mod dispatch;
mod docker;
mod executor;
mod guidance;
mod handler;
mod isolation;
mod network;
mod outbound;
mod outbox;
mod protocol;
mod replay;
mod role;
mod session_state;
mod storage_config;
mod storage_control;
mod storage_facts;
mod storage_topology;

#[cfg(test)]
mod storage_config_tests;
#[cfg(test)]
mod storage_contract_tests;

pub use jain_smartcluster_fabric::{WorkerProvision, WorkloadClass};

#[cfg(test)]
mod role_tests;

pub use assignment::{
    AcceptedAssignment, AssignmentError, AssignmentJournal, NodeRuntimeIdentity, RecoveryReport,
};
pub use cancellation::DurableCancellationRegistry;
pub use control::{
    supervise_outbound_control, ControlConnection, ControlConnector, ControlError, ControlFlow,
    ControlFrameHandler, MtlsClientConfig, MtlsControlClient, MtlsControlConnector,
    ReconnectPolicy,
};
pub use dispatch::{ConcurrentExecutorRuntime, NodeExecutionPort};
pub use docker::{
    DockerEnginePort, DockerLogChunk, DockerLogEvent, DockerRunSpec, DockerUnixEngine, ExitStatus,
};
pub use executor::{
    ArtifactPublication, CancellationPort, ClockPort, ContainerIsolationPort,
    ContainerIsolationReceipt, EventIdentity, EventSink, ExecutionError, ExecutionPhase,
    ExecutionReceipt, ExecutorPorts, IsolationEvidence, JainNodeExecutor, MetricsPort,
    MetricsSample, NetworkLease, NetworkPolicyPort, NodeEvent, PreparedInput, PublishedArtifact,
    ShardPort, SystemClock, MAX_STREAM_CHUNK_BYTES,
};
pub use guidance::ExecutionGuidance;
pub use handler::ProductionControlHandler;
pub use isolation::{LinuxIsolationReceipt, RootBrokerIsolation};
pub use network::{DockerNetworkEnginePort, DockerNetworkPolicy, DockerNetworkReceipt};
pub use outbound::{NodeFailureCode, NodeTerminalOutcome};
pub use outbox::{DurableControlOutbox, OutboundConfirmation, OutboundPending, OutboxError};
pub use protocol::{
    AdmittedInfer, EnrollmentTicket, InferRequest, NodeControlFrame, NodeProfile,
    NodeProtocolState, ProtocolError, SessionControlAction, VerifiedSessionControl,
};
pub use replay::supervise_durable_outbound_control;
pub use role::{ApplianceRole, NodeRoleEnvelope, RoleError, RoleV2Bindings};
pub use storage_config::{
    ProjectKeyConfig, ProjectKeyMaterial, StorageConfig, StorageConfigError, StorageVolumeConfig,
    VersionedKeyPath,
};
pub use storage_control::{
    BeginFragmentWrite, CommitFragmentWrite, FragmentWriteChunk, ReadFragment,
    SignedStorageControlRequest, StorageControlBody, MAX_STORAGE_FRAGMENT_CHUNK_BYTES,
};
pub use storage_facts::{
    ComputeStateTelemetry, FragmentHealthTelemetry, SignedStorageFragmentReceipt,
    SignedStorageTelemetry, StorageFragmentKind, StorageFragmentReceiptFacts, StorageIoCounters,
    StorageTelemetryEmitter, StorageTelemetryFacts, StorageTelemetryObservation,
    StorageVolumeTelemetry, STORAGE_TELEMETRY_INTERVAL_MS,
};
pub use storage_topology::HostVolumeIdentity;
