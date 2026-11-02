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

mod messages;
pub use messages::*;
