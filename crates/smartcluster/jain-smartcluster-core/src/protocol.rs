use std::{collections::BTreeMap, collections::BTreeSet, fmt, str::FromStr};

use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::{
    EventOffset, EventRecord, FailureClass, JobId, JobRecord, JobSpec, JobState, OwnerId,
    ProjectId, TenantId, Timestamp, Validate, ValidationError,
};

pub const PROTOCOL_MAJOR: u16 = 1;
pub const PROTOCOL_MINOR: u16 = 1;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ProtocolVersion {
    pub major: u16,
    pub minor: u16,
}

impl ProtocolVersion {
    pub const CURRENT: Self = Self {
        major: PROTOCOL_MAJOR,
        minor: PROTOCOL_MINOR,
    };

    /// Postcard encodes these contracts positionally, so peers must use the exact
    /// same schema version until explicit feature negotiation is implemented.
    pub const fn is_compatible_with(self, peer: Self) -> bool {
        self.major == peer.major && self.minor == peer.minor
    }
}

impl fmt::Display for ProtocolVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}.{}", self.major, self.minor)
    }
}

macro_rules! ulid_id {
    ($name:ident) => {
        #[derive(
            Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
        )]
        #[serde(transparent)]
        pub struct $name(Ulid);

        impl $name {
            pub fn new() -> Self {
                Self(Ulid::new())
            }

            pub const fn from_ulid(value: Ulid) -> Self {
                Self(value)
            }

            pub const fn as_ulid(self) -> Ulid {
                self.0
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(formatter)
            }
        }

        impl FromStr for $name {
            type Err = ulid::DecodeError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                value.parse().map(Self)
            }
        }
    };
}

ulid_id!(RequestId);
ulid_id!(StreamId);

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SubmitRequest {
    pub spec: JobSpec,
    /// Requested effective identity. The daemon accepts this only from its registered web peer.
    pub effective_principal: Option<crate::Principal>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GetRequest {
    pub job_id: JobId,
}

/// Read only an exact authorized launch attempt, including retained completed attempts.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InspectAttemptRequest {
    pub attempt: crate::FencedAttempt,
}
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeProcessIdentity {
    pub pid: u32,
    pub birth_ticks: u64,
    pub boot_id: String,
}
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AttemptInspection {
    pub record: crate::AttemptRecord,
    /// Present only for the original currently live leader, revalidated against the kernel.
    pub active_process: Option<NativeProcessIdentity>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ListFilter {
    pub tenant: Option<TenantId>,
    pub project: Option<ProjectId>,
    pub owner: Option<OwnerId>,
    #[serde(default)]
    pub states: BTreeSet<JobState>,
    #[serde(default)]
    pub labels: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ListRequest {
    #[serde(default)]
    pub filter: ListFilter,
    pub cursor: Option<String>,
    pub limit: u32,
}

impl Default for ListRequest {
    fn default() -> Self {
        Self {
            filter: ListFilter::default(),
            cursor: None,
            limit: 100,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WaitRequest {
    pub job_id: JobId,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CancelMode {
    Graceful { grace_ms: u64 },
    CheckpointThenCancel { checkpoint_deadline_ms: u64 },
}

impl Default for CancelMode {
    fn default() -> Self {
        Self::Graceful { grace_ms: 30_000 }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CancelRequest {
    pub job_id: JobId,
    pub mode: CancelMode,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PauseRequest {
    pub job_id: JobId,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ResumeRequest {
    pub job_id: JobId,
}

/// Typed control-FD commands. Application commands are named and bounded, never shell text.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlMessage {
    Checkpoint { deadline_ms: u64 },
    Advance,
    Application { name: String, payload: Vec<u8> },
}

impl Validate for ControlMessage {
    fn validate(&self) -> Result<(), ValidationError> {
        match self {
            Self::Checkpoint { deadline_ms } => {
                if *deadline_ms == 0 {
                    return Err(ValidationError::new(
                        "control.deadline_ms",
                        "must be nonzero",
                    ));
                }
            }
            Self::Advance => {}
            Self::Application { name, payload } => {
                crate::validation::validate_identifier("control.name", name, 128)?;
                if payload.len() > 64 * 1024 {
                    return Err(ValidationError::new(
                        "control.payload",
                        "must be at most 65536 bytes",
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SendControlRequest {
    pub job_id: JobId,
    pub control: ControlMessage,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct EventFilter {
    pub job_id: Option<JobId>,
    pub tenant: Option<TenantId>,
    pub project: Option<ProjectId>,
    pub owner: Option<OwnerId>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SubscribeRequest {
    pub offset: EventOffset,
    #[serde(default)]
    pub filter: EventFilter,
}

impl Default for SubscribeRequest {
    fn default() -> Self {
        Self {
            offset: EventOffset::ZERO,
            filter: EventFilter::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AttachRequest {
    pub job_id: JobId,
    pub offset: EventOffset,
    pub read_only: bool,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct StatusRequest;

/// Narrow placement request; no client-supplied cgroup or executable path.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AttachTrustedFitChildRequest {
    pub authority_attempt: crate::FencedAttempt,
    pub target_attempt: crate::FencedAttempt,
    pub child_pid: u32,
    pub child_birth_ticks: u64,
    pub child_boot_id: String,
    pub task_manifest_sha256: String,
    pub descriptor_sha256: String,
    pub candidate_index: u32,
    pub idempotency_key: crate::IdempotencyKey,
    pub child_executable_sha256: String,
}
impl Validate for AttachTrustedFitChildRequest {
    fn validate(&self) -> Result<(), ValidationError> {
        if !(2..=i32::MAX as u32).contains(&self.child_pid)
            || self.child_birth_ticks == 0
            || self.authority_attempt.job_id == self.target_attempt.job_id
        {
            return Err(ValidationError::new(
                "fit_child.identity",
                "invalid distinct attempts or child identity",
            ));
        }
        self.idempotency_key.validate()?;
        if self.child_boot_id.len() != 36
            || !self.child_boot_id.bytes().enumerate().all(|(i, b)| {
                if [8, 13, 18, 23].contains(&i) {
                    b == b'-'
                } else {
                    b.is_ascii_digit() || (b'a'..=b'f').contains(&b)
                }
            })
        {
            return Err(ValidationError::new(
                "fit_child.boot_id",
                "expected lowercase boot UUID",
            ));
        }
        for sha in [
            &self.task_manifest_sha256,
            &self.descriptor_sha256,
            &self.child_executable_sha256,
        ] {
            if sha.len() != 64
                || !sha
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            {
                return Err(ValidationError::new(
                    "fit_child.sha256",
                    "expected lowercase SHA256",
                ));
            }
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrustedFitChildPlacement {
    pub schema: String,
    pub authority_attempt: crate::FencedAttempt,
    pub target_attempt: crate::FencedAttempt,
    pub child_pid: u32,
    pub child_birth_ticks: u64,
    pub child_boot_id: String,
    pub task_manifest_sha256: String,
    pub descriptor_sha256: String,
    pub candidate_index: u32,
    pub child_executable_sha256: String,
    pub cgroup_path: std::path::PathBuf,
    pub receipt_sha256: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DaemonMode {
    Full,
    SafeSingleAllocation,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct SchedulerCapabilities {
    pub cgroup_v2: bool,
    pub cgroup_delegated: bool,
    pub clone_into_cgroup: bool,
    pub pidfd: bool,
    pub freezer: bool,
    pub psi: bool,
    pub nvml: bool,
    pub mps: bool,
    pub mps_unavailable_reason: Option<String>,
    pub mig_inventory: bool,
    pub pty: bool,
    /// Scratch capacity is reported for observability, but v1 does not enforce quotas.
    pub scratch_quota: bool,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct EffectiveCapacity {
    pub cpu_millicores: u32,
    pub memory_bytes: u64,
    pub swap_bytes: Option<u64>,
    pub scratch_bytes: u64,
    #[serde(default)]
    pub logical_cpus: Vec<u32>,
    pub numa_nodes: u16,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PressureLevel {
    #[default]
    Normal,
    Warning,
    Critical,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PressureStatus {
    pub cpu: PressureLevel,
    pub memory: PressureLevel,
    pub io: PressureLevel,
    pub admissions_clamped: bool,
}

impl Default for PressureStatus {
    fn default() -> Self {
        Self {
            cpu: PressureLevel::Normal,
            memory: PressureLevel::Normal,
            io: PressureLevel::Normal,
            admissions_clamped: false,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GpuDeviceStatus {
    pub id: String,
    pub uuid: String,
    pub name: String,
    pub total_vram_bytes: u64,
    pub allocatable_vram_bytes: u64,
    pub healthy: bool,
    pub mig_instance: Option<String>,
    pub temperature_millidegrees_celsius: Option<u32>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DaemonStatus {
    pub daemon_version: String,
    pub protocol_version: ProtocolVersion,
    pub mode: DaemonMode,
    pub started_at: Timestamp,
    pub paused: bool,
    pub capabilities: SchedulerCapabilities,
    pub effective_capacity: EffectiveCapacity,
    pub pressure: PressureStatus,
    #[serde(default)]
    pub gpus: Vec<GpuDeviceStatus>,
    pub queued_jobs: u64,
    pub running_jobs: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Request {
    Status(StatusRequest),
    Submit(Box<SubmitRequest>),
    Get(GetRequest),
    List(ListRequest),
    Wait(WaitRequest),
    Cancel(CancelRequest),
    Pause(PauseRequest),
    Resume(ResumeRequest),
    SendControl(SendControlRequest),
    Subscribe(SubscribeRequest),
    Attach(AttachRequest),
    AttachTrustedFitChild(Box<AttachTrustedFitChildRequest>),
    InspectAttempt(InspectAttemptRequest),
}

impl Validate for Request {
    fn validate(&self) -> Result<(), ValidationError> {
        match self {
            Self::Status(_) => Ok(()),
            Self::Submit(request) => {
                request.spec.validate()?;
                if let Some(principal) = &request.effective_principal {
                    principal.validate()?;
                    request.spec.workload.validate_for(principal)?;
                }
                Ok(())
            }
            Self::List(request) => {
                if !(1..=1000).contains(&request.limit) {
                    return Err(ValidationError::new(
                        "list.limit",
                        "must be between 1 and 1000",
                    ));
                }
                if let Some(cursor) = &request.cursor {
                    crate::validation::validate_text("list.cursor", cursor, 4096, false)?;
                }
                Ok(())
            }
            Self::Cancel(request) => match request.mode {
                CancelMode::Graceful { grace_ms: 0 } => {
                    Err(ValidationError::new("cancel.grace_ms", "must be nonzero"))
                }
                CancelMode::CheckpointThenCancel {
                    checkpoint_deadline_ms: 0,
                } => Err(ValidationError::new(
                    "cancel.checkpoint_deadline_ms",
                    "must be nonzero",
                )),
                _ => Ok(()),
            },
            Self::SendControl(request) => request.control.validate(),
            Self::AttachTrustedFitChild(request) => request.validate(),
            Self::InspectAttempt(_)
            | Self::Get(_)
            | Self::Wait(_)
            | Self::Pause(_)
            | Self::Resume(_)
            | Self::Subscribe(_)
            | Self::Attach(_) => Ok(()),
        }
    }
}

impl Request {
    pub fn submit(request: SubmitRequest) -> Self {
        Self::Submit(Box::new(request))
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct JobPage {
    pub jobs: Vec<JobRecord>,
    pub next_cursor: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct OperationAck {
    pub accepted_at: Timestamp,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct StreamOpened {
    pub stream_id: StreamId,
    pub next_offset: EventOffset,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Response {
    Status(DaemonStatus),
    Submitted(JobRecord),
    Job(Option<JobRecord>),
    Jobs(JobPage),
    Waited(JobRecord),
    Mutated(JobRecord),
    ControlAccepted(OperationAck),
    Subscribed(StreamOpened),
    Attached(StreamOpened),
    TrustedFitChildPlaced(TrustedFitChildPlacement),
    AttemptInspected(Option<AttemptInspection>),
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    InvalidRequest,
    Unauthorized,
    Forbidden,
    NotFound,
    Conflict,
    SupersededAttempt,
    ResourceExhausted,
    Unavailable,
    DeadlineExceeded,
    Internal,
    ProtocolMismatch,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ServerError {
    pub code: ErrorCode,
    pub message: String,
    pub retry_after_ms: Option<u64>,
    #[serde(default)]
    pub details: BTreeMap<String, String>,
}

/// Local, agent-readable repair guidance derived from a typed protocol error.
/// This is not part of the wire envelope and can evolve without changing frames.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RepairGuidance {
    pub purpose: String,
    pub reason: String,
    pub common_fixes: Vec<String>,
    pub docs_url: String,
    pub repair_hint: String,
}

impl ServerError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            retry_after_ms: None,
            details: BTreeMap::new(),
        }
    }

    pub fn unavailable(message: impl Into<String>, retry_after_ms: u64) -> Self {
        Self {
            code: ErrorCode::Unavailable,
            message: message.into(),
            retry_after_ms: Some(retry_after_ms),
            details: BTreeMap::new(),
        }
    }

    pub fn repair_guidance(&self) -> RepairGuidance {
        let common_fixes = match self.code {
            ErrorCode::Unauthorized | ErrorCode::Forbidden => vec![
                "verify the Unix peer identity and delegated owner".to_owned(),
                "do not retry as a broader principal".to_owned(),
            ],
            ErrorCode::Conflict | ErrorCode::SupersededAttempt => vec![
                "fetch the current job and active attempt fence".to_owned(),
                "retry only against the current state".to_owned(),
            ],
            ErrorCode::Unavailable | ErrorCode::ResourceExhausted => vec![
                "honor retry_after_ms when present".to_owned(),
                "inspect daemon status and scheduler capabilities".to_owned(),
            ],
            ErrorCode::InvalidRequest | ErrorCode::ProtocolMismatch => vec![
                "validate the request with the exact protocol version".to_owned(),
                "correct the named field before resubmitting".to_owned(),
            ],
            _ => vec![
                "inspect the job event stream and daemon health".to_owned(),
                "rerun the narrow proof lane named in agent/test-map.json".to_owned(),
            ],
        };
        RepairGuidance {
            purpose: "make an SCQ failure locally diagnosable without hidden daemon state"
                .to_owned(),
            reason: self.message.clone(),
            repair_hint: common_fixes.join("; "),
            common_fixes,
            docs_url: "docs/agent-native-standard.md".to_owned(),
        }
    }
}

impl fmt::Display for ServerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?}: {}", self.code, self.message)
    }
}

impl std::error::Error for ServerError {}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RequestEnvelope {
    pub version: ProtocolVersion,
    pub request_id: RequestId,
    pub request: Request,
}

impl RequestEnvelope {
    pub fn new(request: Request) -> Self {
        Self {
            version: ProtocolVersion::CURRENT,
            request_id: RequestId::new(),
            request,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ResponseEnvelope {
    pub version: ProtocolVersion,
    pub request_id: RequestId,
    pub result: Result<Response, ServerError>,
}

impl ResponseEnvelope {
    pub fn ok(request_id: RequestId, response: Response) -> Self {
        Self {
            version: ProtocolVersion::CURRENT,
            request_id,
            result: Ok(response),
        }
    }

    pub fn error(request_id: RequestId, error: ServerError) -> Self {
        Self {
            version: ProtocolVersion::CURRENT,
            request_id,
            result: Err(error),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AttachInput {
    Stdin(Vec<u8>),
    Resize { rows: u16, columns: u16 },
    Control(ControlMessage),
    CloseStdin,
}

impl Validate for AttachInput {
    fn validate(&self) -> Result<(), ValidationError> {
        match self {
            Self::Stdin(data) => {
                if data.len() > 64 * 1024 {
                    Err(ValidationError::new(
                        "attach.stdin",
                        "must be at most 65536 bytes",
                    ))
                } else {
                    Ok(())
                }
            }
            Self::Resize { rows, columns } if *rows == 0 || *columns == 0 => Err(
                ValidationError::new("attach.resize", "rows and columns must be nonzero"),
            ),
            Self::Control(control) => control.validate(),
            Self::Resize { .. } | Self::CloseStdin => Ok(()),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct StreamInputFrame {
    pub stream_id: StreamId,
    pub input: AttachInput,
}

/// Daemon-to-worker frame written on the typed control FD.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WorkerControlFrame {
    pub version: ProtocolVersion,
    pub attempt: crate::FencedAttempt,
    pub sequence: u64,
    pub control: ControlMessage,
}

/// Worker-to-daemon reports written on the typed control FD.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerReport {
    Ready,
    CheckpointAcknowledged {
        checkpoint_id: String,
    },
    CheckpointCommitted(crate::CheckpointEvent),
    Progress(crate::StructuredProgress),
    Metrics(crate::MetricsSnapshot),
    Failure {
        class: FailureClass,
        message: String,
    },
    Application {
        name: String,
        payload: Vec<u8>,
    },
}

impl Validate for WorkerReport {
    fn validate(&self) -> Result<(), ValidationError> {
        match self {
            Self::Ready => Ok(()),
            Self::CheckpointAcknowledged { checkpoint_id } => {
                crate::validation::validate_identifier(
                    "worker_report.checkpoint_id",
                    checkpoint_id,
                    256,
                )
            }
            Self::CheckpointCommitted(checkpoint) => checkpoint.validate(),
            Self::Progress(progress) => progress.validate(),
            Self::Metrics(metrics) => metrics.validate(),
            Self::Failure { message, .. } => crate::validation::validate_text(
                "worker_report.failure.message",
                message,
                4096,
                false,
            ),
            Self::Application { name, payload } => {
                crate::validation::validate_identifier("worker_report.name", name, 128)?;
                if payload.len() > 64 * 1024 {
                    return Err(ValidationError::new(
                        "worker_report.payload",
                        "must be at most 65536 bytes",
                    ));
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WorkerReportFrame {
    pub version: ProtocolVersion,
    pub attempt: crate::FencedAttempt,
    pub sequence: u64,
    pub report: WorkerReport,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ClientFrame {
    Request(Box<RequestEnvelope>),
    StreamInput(StreamInputFrame),
    CloseStream { stream_id: StreamId },
}

impl ClientFrame {
    pub fn request(envelope: RequestEnvelope) -> Self {
        Self::Request(Box::new(envelope))
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct EventEnvelope {
    pub stream_id: StreamId,
    pub event: EventRecord,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ServerFrame {
    Response(Box<ResponseEnvelope>),
    Event(EventEnvelope),
    StreamClosed {
        stream_id: StreamId,
        error: Option<ServerError>,
    },
    /// A global stream marker, deliberately outside [`EventRecord`] because a
    /// compacted range is not owned by any one job.
    StreamGap {
        stream_id: StreamId,
        gap: crate::StreamGap,
    },
}

impl ServerFrame {
    pub fn response(envelope: ResponseEnvelope) -> Self {
        Self::Response(Box::new(envelope))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_compatibility_requires_an_exact_version() {
        assert!(ProtocolVersion::CURRENT.is_compatible_with(ProtocolVersion::CURRENT));
        assert!(
            !ProtocolVersion::CURRENT.is_compatible_with(ProtocolVersion {
                major: PROTOCOL_MAJOR,
                minor: PROTOCOL_MINOR + 1,
            })
        );
        assert!(
            !ProtocolVersion::CURRENT.is_compatible_with(ProtocolVersion {
                major: PROTOCOL_MAJOR + 1,
                minor: 0,
            })
        );
    }

    #[test]
    fn request_validation_bounds_list_pages() {
        let request = Request::List(ListRequest {
            limit: 0,
            ..ListRequest::default()
        });
        assert!(request.validate().is_err());
    }

    #[test]
    fn application_controls_are_bounded() {
        let control = ControlMessage::Application {
            name: "advance-phase".into(),
            payload: vec![0; 65_537],
        };
        assert!(control.validate().is_err());
    }

    #[test]
    fn typed_worker_failures_require_a_bounded_message() {
        let empty = WorkerReport::Failure {
            class: FailureClass::Input,
            message: String::new(),
        };
        let oversized = WorkerReport::Failure {
            class: FailureClass::Network,
            message: "x".repeat(4097),
        };
        let valid = WorkerReport::Failure {
            class: FailureClass::Dependency,
            message: "required model artifact is unavailable".to_owned(),
        };
        assert!(empty.validate().is_err());
        assert!(oversized.validate().is_err());
        assert!(valid.validate().is_ok());
    }

    #[test]
    fn server_errors_provide_local_repair_guidance() {
        let error = ServerError::unavailable("scheduler is starting", 1_000);
        let guidance = error.repair_guidance();
        assert_eq!(guidance.reason, "scheduler is starting");
        assert!(guidance.repair_hint.contains("retry_after_ms"));
        assert_eq!(guidance.docs_url, "docs/agent-native-standard.md");
        assert!(!guidance.common_fixes.is_empty());
    }
}
