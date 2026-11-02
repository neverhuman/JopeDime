use super::*;

pub const CONTRACT_SCHEMA: &str = "jope.native_bounded_run.v1";
/// The fixed two-block run contract this schema sits beside. It is never reinterpreted.
pub const LEGACY_CONTRACT_SCHEMA: &str = "jope.native_loop.v1";
pub const GRANT_SCHEMA: &str = "jope.native_participant_grant.v1";
pub const GRANT_DOMAIN: &[u8] = b"jope.native_participant_grant.signature.v1\0";
pub const MAX_CONTRACT_BYTES: usize = 1024 * 1024;
pub const MAX_PARTICIPANTS: usize = 16;
pub const MAX_WINDOW_MS: u64 = 7 * 24 * 3600 * 1000;
pub const MAX_CLOSURE_ENTRIES: usize = 4096;
pub const MAX_ARTIFACT_BYTES: u64 = 1 << 30;
pub const MAX_RESULT_BYTES: u64 = 256 << 20;
pub const MAX_WORK_ITEMS: usize = 10_000;
/// Hard bounds on the frozen message policy (T04).
pub const MIN_MESSAGE_BYTES: u32 = 1024;
pub const MAX_MESSAGE_BYTES: u32 = 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TelemetryPolicy {
    pub max_age_ms: u64,
    pub clock_skew_ms: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RendezvousPolicy {
    /// The whole synchronous group must be admitted within this long of opening.
    pub join_timeout_ms: u64,
    /// Frames longer than this are refused before parsing.
    pub max_message_bytes: u32,
    /// A connected participant silent for longer than this has lost control.
    pub heartbeat_timeout_ms: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ClosureEntry {
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DeviceAssignment {
    pub uuid: String,
    pub local_index: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Ceilings {
    pub cpu_millicores: u32,
    pub ram_bytes: u64,
    pub gpu_vram_bytes: Option<u64>,
    pub wall_ms: u64,
}
/// Independently signed per-participant grant. The planner only freezes these bytes; it
/// holds no grant-signing capability.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GrantBody {
    pub schema: String,
    pub run: String,
    pub attempt: u32,
    pub global_rank: u32,
    pub host: String,
    pub boot_id: String,
    pub device_uuid: Option<String>,
    pub participant_public_key: String,
    pub cpu_millicores: u32,
    pub ram_bytes: u64,
    pub gpu_vram_bytes: Option<u64>,
    pub wall_ms: u64,
    pub not_before_unix_ms: u64,
    pub expires_unix_ms: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SignedGrant {
    pub body: GrantBody,
    pub signature: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ResultDestination {
    /// Relative to the controller's `results/` directory.
    pub path: PathBuf,
    pub max_bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Participant {
    pub global_rank: u32,
    pub local_rank: u32,
    pub host: HostIdentity,
    pub device: Option<DeviceAssignment>,
    pub ceilings: Ceilings,
    /// Authenticates this participant's rendezvous messages and results.
    pub public_key: String,
    pub grant: SignedGrant,
    pub result_destination: ResultDestination,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum WorkKind {
    Numerical,
    Verification,
    ArtifactMovement,
    Cleanup,
    FailureAllowance,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct WorkItem {
    pub id: String,
    pub global_rank: u32,
    pub kind: WorkKind,
    pub cpu_core_ms: u64,
    pub gpu_ms: u64,
    pub wall_ms: u64,
    pub peak_ram_bytes: u64,
    pub peak_vram_bytes: u64,
    pub updates: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Counters {
    /// Committed global updates before this attempt; preserved across restarts/attempts.
    pub committed_updates_at_start: u64,
    /// Global synchronous updates this attempt may commit (identical on every rank).
    pub planned_updates: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Predecessor {
    pub plan_sha256: String,
    pub attempt: u32,
    pub journal_tip_sha256: String,
    pub committed_updates: u64,
}
/// Microbatches and global batches the trainer's partition gate accepts
/// (`training::partition::execution`, outside batch qualification).
pub const TRAINING_MICROBATCHES: [u32; 3] = [64, 128, 256];
pub const TRAINING_GLOBAL_BATCHES: [u32; 4] = [2048, 4096, 8192, 16384];

/// The training a plan selects (W7-T06, W7-D2): the execution-partition version and batch
/// shape its ranks train under, and what they train from. The world size is the number of
/// participants, so the partition record is derived for exactly this plan
/// ([`BoundedRunContract::execution_partition`]). A plan that selects none omits the field,
/// and its canonical bytes and digest are unchanged.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TrainingSelection {
    /// `parallel.training_partition.v1` or `parallel.training_partition.v2`.
    pub execution_partition: String,
    pub microbatch: u32,
    pub global_batch: u32,
    /// The corpus directory, relative to each host's input root; its `index.json` is in the
    /// plan's input closure.
    pub corpus: PathBuf,
    /// Optimizer steps between committed checkpoints, `1..=planned_updates`.
    pub checkpoint_steps: u64,
    /// The parent checkpoint directory, relative to the input root; its `state.json` is in
    /// the input closure and is at the plan's committed start. Absent for a fresh start at 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initialize: Option<PathBuf>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BoundedRunContract {
    pub schema: String,
    pub run: String,
    pub attempt: u32,
    pub coordinator_host: String,
    pub coordinator_public_key: String,
    pub grant_authority_public_key: String,
    pub not_before_unix_ms: u64,
    /// The one original deadline; retries, restarts and later attempts never renew it.
    pub original_deadline_unix_ms: u64,
    pub telemetry: TelemetryPolicy,
    pub rendezvous: RendezvousPolicy,
    pub artifact_closure: Vec<ClosureEntry>,
    pub participants: Vec<Participant>,
    pub work_census: Vec<WorkItem>,
    pub counters: Counters,
    pub predecessor: Option<Predecessor>,
    /// The training partition this plan selects, if any. Never serialized when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training: Option<TrainingSelection>,
}

/// Why a plan cannot be frozen. Every variant is raised before any dispatch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanRefusal {
    /// A `jope.native_loop.v1` manifest keeps its old Bind/Tick semantics.
    Legacy(String),
    Invalid(String),
    Underfunded(String),
}
impl std::fmt::Display for PlanRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Legacy(m) => write!(f, "legacy fixed run contract refused as bounded plan: {m}"),
            Self::Invalid(m) => write!(f, "invalid bounded-run plan: {m}"),
            Self::Underfunded(m) => write!(f, "underfunded bounded-run plan: {m}"),
        }
    }
}
impl std::error::Error for PlanRefusal {}
/// What a completed predecessor attempt left behind, read from its verified journal.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AttemptOutcome {
    pub plan_sha256: String,
    pub journal_tip_sha256: String,
    pub terminal: bool,
    pub committed_updates_total: u64,
}
