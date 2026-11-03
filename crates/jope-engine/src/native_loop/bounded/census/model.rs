use serde::{Deserialize, Serialize};
use super::*;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeIdentity {
    pub runtime_sha256: String,
    pub compiler_sha256: String,
    pub cuda_sha256: Option<String>,
    pub nccl_sha256: Option<String>,
    pub driver_version: Option<String>,
}
/// Everything that must be byte-identical between the frozen plan and admission time.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HostIdentity {
    /// Operator-facing host name used by grants, ownership and the census.
    pub host: String,
    /// Stable host identity, `/etc/machine-id` (32 lowercase hex). The trainer
    /// assignment's `host_id` is this value.
    pub machine_id: String,
    pub boot_id: String,
    pub address: String,
    pub certificate_sha256: String,
    pub source_commit: String,
    pub executable_sha256: String,
    pub installed_policy_sha256: String,
    pub runtime: RuntimeIdentity,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GpuObservation {
    pub uuid: String,
    pub model: String,
    pub local_index: u32,
    pub total_vram_bytes: u64,
    pub available_vram_bytes: u64,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OwnerKind {
    Ci,
    User,
    Managed,
}
/// A reservation or device owner observed on a host. `run`/`attempt` identify a managed
/// bounded-run owner so that re-running preflight for the *same* attempt is not a conflict.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Ownership {
    pub owner: String,
    pub kind: OwnerKind,
    pub device_uuid: Option<String>,
    pub cpu_millicores: u32,
    pub ram_bytes: u64,
    pub run: Option<String>,
    pub attempt: Option<u32>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HostObservation {
    pub identity: HostIdentity,
    pub observed_unix_ms: u64,
    /// Schedulable capacity; reservations in `ownership` are subtracted from it.
    pub cpu_millicores_allocatable: u32,
    pub ram_bytes_allocatable: u64,
    pub gpus: Vec<GpuObservation>,
    pub ownership: Vec<Ownership>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HostCensus {
    pub schema: String,
    pub captured_unix_ms: u64,
    pub hosts: Vec<HostObservation>,
}

/// Every reason a preflight refuses. The receipt is issued only when this list is empty.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(tag = "reason", rename_all = "snake_case", deny_unknown_fields)]
pub enum RefusalReason {
    StaleTelemetry {
        host: String,
        age_ms: u64,
    },
    FutureTelemetry {
        host: String,
        ahead_ms: u64,
    },
    MissingHost {
        host: String,
    },
    MachineMismatch {
        host: String,
    },
    BootMismatch {
        host: String,
    },
    AddressMismatch {
        host: String,
    },
    CertificateMismatch {
        host: String,
    },
    SourceMismatch {
        host: String,
    },
    ExecutableMismatch {
        host: String,
    },
    RuntimeMismatch {
        host: String,
    },
    PolicyMismatch {
        host: String,
    },
    GpuMissing {
        host: String,
        uuid: String,
    },
    GpuIndexMismatch {
        host: String,
        uuid: String,
    },
    ConflictingOwnership {
        host: String,
        uuid: String,
        owner: String,
    },
    InsufficientCpu {
        host: String,
        free: u64,
        required: u64,
    },
    InsufficientRam {
        host: String,
        free: u64,
        required: u64,
    },
    InsufficientVram {
        host: String,
        uuid: String,
        free: u64,
        required: u64,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PreflightRefusal {
    pub reasons: Vec<RefusalReason>,
}
impl std::fmt::Display for PreflightRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "bounded-run preflight refused: {}",
            serde_json::to_string(&self.reasons).unwrap_or_default()
        )
    }
}
impl std::error::Error for PreflightRefusal {}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HostReceipt {
    pub boot_id: String,
    pub observed_unix_ms: u64,
    pub free_cpu_millicores: u64,
    pub free_ram_bytes: u64,
    pub devices: BTreeSet<String>,
}
/// Issued only for a complete, fresh, matching and funded census. It records what was
/// checked; it is not an allocation and holds no device.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PreflightReceipt {
    pub schema: String,
    pub run: String,
    pub attempt: u32,
    pub plan_sha256: String,
    pub census_sha256: String,
    pub checked_unix_ms: u64,
    pub hosts: BTreeMap<String, HostReceipt>,
    pub allocation_held: bool,
}
