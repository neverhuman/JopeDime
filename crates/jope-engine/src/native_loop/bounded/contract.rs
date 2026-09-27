//! `jope.native_bounded_run.v1`: a versioned bounded-run contract (W7-T02).
//!
//! The fixed `jope.native_loop.v1` contract ([`crate::native_loop::Config`]) keeps its exact
//! semantics; a legacy manifest offered here is refused as legacy, and a bounded-run
//! contract offered to the old loader fails its `deny_unknown_fields` parse. Nothing is
//! inferred from mutable flags: the frozen plan bytes are the only execution input.
//!
//! **Trainer join point (W7-T03).** A participant freezes `global_rank`, `world_size`
//! (= number of participants), `local_rank`, the host's `machine_id` and `boot_id`, and the
//! device `uuid`. `training::topology::Assignment` is derived from exactly these fields by
//! the pure `bounded::assignment::trainer_assignment` (`neural` builds), never from process
//! flags or `rank` alone. `DeviceAssignment::local_index` is the host-wide index the census
//! observed, not a CUDA ordinal: each worker sees only its own UUID, at ordinal 0.
//!
//! **Training partition (W7-T06).** A plan may select the execution partition its ranks train
//! under ([`TrainingSelection`]): partition v1, or the lossless v2 that three ranks need. The
//! record is derived for exactly the plan's world size, and the selection is admitted only
//! when the trainer's partition gate accepts it. A plan that selects none omits the field, so
//! every earlier plan keeps its exact bytes and digest. A later attempt cannot change it.
use super::{
    census::{validate_identity, HostIdentity},
    is_digest, is_name, is_relative, sign, verify, verifying_key,
};
use ed25519_dalek::SigningKey;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};

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
type Check = std::result::Result<(), PlanRefusal>;
macro_rules! need {
    ($cond:expr, $kind:ident, $($msg:tt)+) => {
        if !$cond {
            return Err(PlanRefusal::$kind(format!($($msg)+)));
        }
    };
}
fn invalid(e: impl std::fmt::Display) -> PlanRefusal {
    PlanRefusal::Invalid(format!("{e:#}"))
}

/// What a completed predecessor attempt left behind, read from its verified journal.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AttemptOutcome {
    pub plan_sha256: String,
    pub journal_tip_sha256: String,
    pub terminal: bool,
    pub committed_updates_total: u64,
}

pub fn sign_grant(authority: &SigningKey, body: GrantBody) -> anyhow::Result<SignedGrant> {
    Ok(SignedGrant {
        signature: sign(authority, GRANT_DOMAIN, &body)?,
        body,
    })
}

impl BoundedRunContract {
    /// Bounded parse. A legacy fixed manifest is reported as legacy, never coerced.
    pub fn parse(bytes: &[u8]) -> std::result::Result<Self, PlanRefusal> {
        need!(
            bytes.len() <= MAX_CONTRACT_BYTES,
            Invalid,
            "contract exceeds {MAX_CONTRACT_BYTES} bytes"
        );
        let value: Value = serde_json::from_slice(bytes).map_err(invalid)?;
        if value["schema"] == LEGACY_CONTRACT_SCHEMA {
            return Err(PlanRefusal::Legacy(format!(
                "`{LEGACY_CONTRACT_SCHEMA}` runs only through Bind/Tick/Serve with its original semantics"
            )));
        }
        let contract: Self = serde_json::from_value(value).map_err(invalid)?;
        contract.validate_structure()?;
        Ok(contract)
    }
    pub fn canonical_bytes(&self) -> anyhow::Result<Vec<u8>> {
        Ok(serde_json::to_vec(self)?)
    }
    pub fn world_size(&self) -> usize {
        self.participants.len()
    }
    pub fn participant(&self, global_rank: u32) -> Option<&Participant> {
        self.participants.get(global_rank as usize)
    }
    /// Time-independent validation: identity, uniqueness, grants and funding arithmetic.
    pub fn validate_structure(&self) -> Check {
        need!(
            self.schema == CONTRACT_SCHEMA,
            Invalid,
            "unsupported contract schema `{}`",
            self.schema
        );
        need!(is_name(&self.run), Invalid, "run identity");
        need!(self.attempt >= 1, Invalid, "attempt numbers start at 1");
        need!(is_name(&self.coordinator_host), Invalid, "coordinator host");
        verifying_key(&self.coordinator_public_key).map_err(invalid)?;
        verifying_key(&self.grant_authority_public_key).map_err(invalid)?;
        need!(
            self.coordinator_public_key != self.grant_authority_public_key,
            Invalid,
            "coordinator cannot hold the grant authority key"
        );
        need!(
            self.not_before_unix_ms < self.original_deadline_unix_ms,
            Invalid,
            "empty run window"
        );
        let window = self.original_deadline_unix_ms - self.not_before_unix_ms;
        need!(window <= MAX_WINDOW_MS, Invalid, "run window exceeds bound");
        need!(
            (1..=600_000).contains(&self.telemetry.max_age_ms)
                && self.telemetry.clock_skew_ms <= 10_000,
            Invalid,
            "telemetry freshness policy"
        );
        let r = &self.rendezvous;
        need!(
            (1..=window).contains(&r.join_timeout_ms)
                && (1..=window).contains(&r.heartbeat_timeout_ms)
                && (MIN_MESSAGE_BYTES..=MAX_MESSAGE_BYTES).contains(&r.max_message_bytes),
            Invalid,
            "rendezvous timeout/message policy"
        );
        self.validate_closure()?;
        self.validate_participants(window)?;
        self.validate_work_census()?;
        need!(
            self.counters.planned_updates > 0
                && self
                    .counters
                    .committed_updates_at_start
                    .checked_add(self.counters.planned_updates)
                    .is_some(),
            Invalid,
            "update counters"
        );
        match (&self.predecessor, self.attempt) {
            (None, 1) => {}
            (Some(p), a) if a > 1 => {
                need!(
                    p.attempt + 1 == a
                        && is_digest(&p.plan_sha256)
                        && is_digest(&p.journal_tip_sha256)
                        && p.committed_updates == self.counters.committed_updates_at_start,
                    Invalid,
                    "predecessor attempt/counters do not continue exactly"
                );
            }
            _ => {
                return Err(PlanRefusal::Invalid(
                    "attempt 1 has no predecessor; later attempts require one".into(),
                ))
            }
        }
        self.validate_training()
    }
    /// A selected partition must be one the trainer's gate accepts for this world size.
    fn validate_training(&self) -> Check {
        let Some(t) = &self.training else {
            return Ok(());
        };
        let world = self.participants.len();
        need!(
            TRAINING_MICROBATCHES.contains(&t.microbatch)
                && TRAINING_GLOBAL_BATCHES.contains(&t.global_batch),
            Invalid,
            "training batch shape: microbatch {} and global batch {}",
            t.microbatch,
            t.global_batch
        );
        match t.execution_partition.as_str() {
            crate::task_contract::PARTITION_V1_SCHEMA => need!(
                [1, 2, 4, 8].contains(&world) && t.global_batch as usize % (world * t.microbatch as usize) == 0,
                Invalid,
                "partition v1 cannot divide the global batch over {world} ranks; select partition v2"
            ),
            crate::task_contract::PARTITION_V2_SCHEMA => need!(
                (1..=8).contains(&world),
                Invalid,
                "partition v2 supports one to eight ranks, not {world}"
            ),
            other => {
                return Err(PlanRefusal::Invalid(format!(
                    "unknown training execution partition `{other}`"
                )))
            }
        }
        self.execution_partition().map_err(invalid)?;
        // W7-D2: what the ranks train from is pinned by the plan.
        let in_closure =
            |path: &std::path::Path| self.artifact_closure.iter().any(|e| e.path == path);
        need!(
            is_relative(&t.corpus) && in_closure(&t.corpus.join("index.json")),
            Invalid,
            "the training corpus index `{}/index.json` is not in the plan's input closure",
            t.corpus.display()
        );
        need!(
            (1..=self.counters.planned_updates).contains(&t.checkpoint_steps),
            Invalid,
            "checkpoint cadence {} is outside 1..={} planned updates",
            t.checkpoint_steps,
            self.counters.planned_updates
        );
        match &t.initialize {
            Some(parent) => need!(
                is_relative(parent) && in_closure(&parent.join("state.json")),
                Invalid,
                "the parent checkpoint state `{}/state.json` is not in the plan's input closure",
                parent.display()
            ),
            None => need!(
                self.counters.committed_updates_at_start == 0,
                Invalid,
                "a fresh training run starts at 0 committed updates, not {}; name the parent checkpoint it continues",
                self.counters.committed_updates_at_start
            ),
        }
        for p in &self.participants {
            need!(
                p.device.is_some(),
                Invalid,
                "rank {} has no GPU; every rank of a training plan trains on a GPU",
                p.global_rank
            );
        }
        Ok(())
    }
    /// The execution-partition record this plan's ranks train under, derived from its
    /// selection for exactly its world size; `None` when the plan selects no training.
    pub fn execution_partition(&self) -> anyhow::Result<Option<Value>> {
        self.training
            .as_ref()
            .map(|t| {
                crate::task_contract::training_partition_of(
                    &t.execution_partition,
                    self.world_size(),
                    t.microbatch as usize,
                    t.global_batch as usize,
                )
            })
            .transpose()
    }
    fn validate_closure(&self) -> Check {
        need!(
            (1..=MAX_CLOSURE_ENTRIES).contains(&self.artifact_closure.len()),
            Invalid,
            "artifact closure size"
        );
        for pair in self.artifact_closure.windows(2) {
            need!(
                pair[0].path < pair[1].path,
                Invalid,
                "artifact closure must be sorted and unique"
            );
        }
        for e in &self.artifact_closure {
            need!(
                is_relative(&e.path)
                    && is_digest(&e.sha256)
                    && (1..=MAX_ARTIFACT_BYTES).contains(&e.bytes),
                Invalid,
                "artifact closure entry `{}`",
                e.path.display()
            );
        }
        Ok(())
    }
    fn validate_participants(&self, window: u64) -> Check {
        let n = self.participants.len();
        need!(
            (1..=MAX_PARTICIPANTS).contains(&n),
            Invalid,
            "participant count {n}"
        );
        let closure: BTreeSet<_> = self.artifact_closure.iter().map(|e| &e.path).collect();
        let mut identities: BTreeMap<&str, &HostIdentity> = BTreeMap::new();
        let mut local_ranks: BTreeMap<&str, BTreeSet<u32>> = BTreeMap::new();
        let mut devices = BTreeSet::new();
        let mut indices = BTreeSet::new();
        let mut keys = BTreeSet::from([
            self.coordinator_public_key.as_str(),
            self.grant_authority_public_key.as_str(),
        ]);
        let mut destinations = BTreeSet::new();
        for (i, p) in self.participants.iter().enumerate() {
            need!(
                p.global_rank as usize == i,
                Invalid,
                "global ranks must be exactly 0..{n} in order"
            );
            validate_identity(&p.host).map_err(invalid)?;
            let host = p.host.host.as_str();
            if let Some(first) = identities.insert(host, &p.host) {
                need!(
                    *first == p.host,
                    Invalid,
                    "participants on host `{host}` disagree on its identity"
                );
            }
            // One machine and one boot are one host: two host names cannot share them.
            for (name, other) in &identities {
                need!(
                    *name == host
                        || (other.machine_id != p.host.machine_id
                            && other.boot_id != p.host.boot_id),
                    Invalid,
                    "hosts `{name}` and `{host}` share a machine or boot identity"
                );
            }
            need!(
                local_ranks.entry(host).or_default().insert(p.local_rank),
                Invalid,
                "duplicate local rank on host `{host}`"
            );
            match &p.device {
                Some(d) => {
                    need!(is_name(&d.uuid), Invalid, "device UUID");
                    need!(
                        devices.insert(d.uuid.clone()),
                        Invalid,
                        "device `{}` assigned to more than one worker",
                        d.uuid
                    );
                    need!(
                        indices.insert((host, d.local_index)),
                        Invalid,
                        "local device index reused on host `{host}`"
                    );
                    need!(
                        p.ceilings.gpu_vram_bytes.is_some_and(|v| v > 0),
                        Invalid,
                        "GPU participant {i} lacks a VRAM ceiling"
                    );
                }
                None => need!(
                    p.ceilings.gpu_vram_bytes.is_none(),
                    Invalid,
                    "CPU participant {i} declares a VRAM ceiling"
                ),
            }
            need!(
                p.ceilings.cpu_millicores > 0
                    && p.ceilings.ram_bytes > 0
                    && (1..=window).contains(&p.ceilings.wall_ms),
                Invalid,
                "participant {i} ceilings"
            );
            verifying_key(&p.public_key).map_err(invalid)?;
            need!(
                keys.insert(p.public_key.as_str()),
                Invalid,
                "participant {i} key is not unique"
            );
            let d = &p.result_destination;
            need!(
                is_relative(&d.path)
                    && (1..=MAX_RESULT_BYTES).contains(&d.max_bytes)
                    && !closure.contains(&d.path)
                    && destinations.insert(d.path.clone()),
                Invalid,
                "participant {i} result destination"
            );
            self.validate_grant(p)?;
        }
        for (host, ranks) in local_ranks {
            need!(
                ranks.iter().copied().eq(0..ranks.len() as u32),
                Invalid,
                "local ranks on host `{host}` must be 0..{}",
                ranks.len()
            );
        }
        Ok(())
    }
    fn validate_grant(&self, p: &Participant) -> Check {
        let g = &p.grant.body;
        need!(
            g.schema == GRANT_SCHEMA
                && g.run == self.run
                && g.attempt == self.attempt
                && g.global_rank == p.global_rank
                && g.host == p.host.host
                && g.boot_id == p.host.boot_id
                && g.device_uuid == p.device.as_ref().map(|d| d.uuid.clone())
                && g.participant_public_key == p.public_key
                && g.not_before_unix_ms < g.expires_unix_ms,
            Invalid,
            "grant for rank {} does not bind this run/attempt/participant",
            p.global_rank
        );
        verify(
            &self.grant_authority_public_key,
            GRANT_DOMAIN,
            g,
            &p.grant.signature,
        )
        .map_err(|e| {
            PlanRefusal::Invalid(format!("grant signature for rank {}: {e:#}", p.global_rank))
        })?;
        need!(
            g.expires_unix_ms <= self.original_deadline_unix_ms,
            Invalid,
            "grant for rank {} outlives the original deadline",
            p.global_rank
        );
        let c = &p.ceilings;
        need!(
            g.cpu_millicores >= c.cpu_millicores
                && g.ram_bytes >= c.ram_bytes
                && g.gpu_vram_bytes.unwrap_or(0) >= c.gpu_vram_bytes.unwrap_or(0)
                && g.wall_ms >= c.wall_ms,
            Underfunded,
            "grant for rank {} does not fund its ceilings",
            p.global_rank
        );
        let start = g.not_before_unix_ms.max(self.not_before_unix_ms);
        need!(
            g.expires_unix_ms.saturating_sub(start) >= c.wall_ms,
            Underfunded,
            "grant window for rank {} is shorter than its wall ceiling",
            p.global_rank
        );
        Ok(())
    }
    fn validate_work_census(&self) -> Check {
        let n = self.participants.len() as u32;
        need!(
            (1..=MAX_WORK_ITEMS).contains(&self.work_census.len()),
            Invalid,
            "work census size"
        );
        let mut ids = BTreeSet::new();
        #[derive(Default)]
        struct Sum {
            cpu: u128,
            gpu: u64,
            wall: u64,
            ram: u64,
            vram: u64,
            updates: u64,
            kinds: BTreeSet<WorkKind>,
        }
        let mut sums: BTreeMap<u32, Sum> = BTreeMap::new();
        for w in &self.work_census {
            need!(
                is_name(&w.id) && ids.insert(w.id.as_str()),
                Invalid,
                "work item id `{}`",
                w.id
            );
            need!(
                w.global_rank < n,
                Invalid,
                "work item `{}` names no participant",
                w.id
            );
            let p = &self.participants[w.global_rank as usize];
            need!(
                p.device.is_some() || (w.gpu_ms == 0 && w.peak_vram_bytes == 0),
                Invalid,
                "GPU work `{}` on CPU participant",
                w.id
            );
            need!(
                w.kind == WorkKind::Numerical || w.updates == 0,
                Invalid,
                "non-numerical work `{}` claims updates",
                w.id
            );
            let s = sums.entry(w.global_rank).or_default();
            s.cpu += u128::from(w.cpu_core_ms);
            s.gpu = s.gpu.saturating_add(w.gpu_ms);
            s.wall = s.wall.saturating_add(w.wall_ms);
            s.ram = s.ram.max(w.peak_ram_bytes);
            s.vram = s.vram.max(w.peak_vram_bytes);
            s.updates = s.updates.saturating_add(w.updates);
            s.kinds.insert(w.kind);
        }
        for p in &self.participants {
            let r = p.global_rank;
            let Some(s) = sums.get(&r) else {
                return Err(PlanRefusal::Invalid(format!(
                    "work census omits participant {r}"
                )));
            };
            need!(
                s.kinds.contains(&WorkKind::Verification) && s.kinds.contains(&WorkKind::Cleanup),
                Invalid,
                "work census for rank {r} omits verification or cleanup"
            );
            need!(
                s.updates == self.counters.planned_updates,
                Invalid,
                "rank {r} plans {} updates; the synchronous group plans {}",
                s.updates,
                self.counters.planned_updates
            );
            let c = &p.ceilings;
            let funded_cpu = u128::from(c.cpu_millicores) * u128::from(c.wall_ms) / 1000;
            need!(
                s.cpu <= funded_cpu,
                Underfunded,
                "rank {r} needs {} CPU core-ms; ceilings fund {funded_cpu}",
                s.cpu
            );
            need!(
                s.wall <= c.wall_ms && s.gpu <= c.wall_ms,
                Underfunded,
                "rank {r} work census exceeds its wall ceiling"
            );
            need!(
                s.ram <= c.ram_bytes && s.vram <= c.gpu_vram_bytes.unwrap_or(0),
                Underfunded,
                "rank {r} peak memory exceeds its ceilings"
            );
        }
        Ok(())
    }
    /// Plan-time validation: structure plus the wall clock. Refuses a plan whose remaining
    /// original window or grants cannot fund the frozen ceilings.
    pub fn validate_at(&self, now_unix_ms: u64) -> Check {
        self.validate_structure()?;
        need!(
            now_unix_ms < self.original_deadline_unix_ms,
            Invalid,
            "original deadline already elapsed"
        );
        let start = now_unix_ms.max(self.not_before_unix_ms);
        let longest = self
            .participants
            .iter()
            .map(|p| p.ceilings.wall_ms)
            .max()
            .unwrap_or(0);
        need!(
            self.original_deadline_unix_ms - start >= longest,
            Underfunded,
            "remaining original window cannot fund the longest wall ceiling"
        );
        for p in &self.participants {
            let g = &p.grant.body;
            need!(
                now_unix_ms < g.expires_unix_ms,
                Invalid,
                "grant for rank {} already expired",
                p.global_rank
            );
            need!(
                g.expires_unix_ms
                    .saturating_sub(start.max(g.not_before_unix_ms))
                    >= p.ceilings.wall_ms,
                Underfunded,
                "remaining grant window for rank {} cannot fund its wall ceiling",
                p.global_rank
            );
        }
        Ok(())
    }
    /// A later attempt continues the same run, deadline, topology and counters.
    pub fn validate_successor(&self, previous: &Self, outcome: &AttemptOutcome) -> Check {
        let Some(p) = &self.predecessor else {
            return Err(PlanRefusal::Invalid(
                "successor names no predecessor".into(),
            ));
        };
        need!(
            outcome.terminal,
            Invalid,
            "predecessor attempt is not terminal; reconcile it first"
        );
        need!(
            self.run == previous.run
                && self.attempt == previous.attempt + 1
                && p.plan_sha256 == outcome.plan_sha256
                && p.journal_tip_sha256 == outcome.journal_tip_sha256
                && p.committed_updates == outcome.committed_updates_total
                && self.counters.committed_updates_at_start == outcome.committed_updates_total,
            Invalid,
            "successor does not continue the exact predecessor journal and counters"
        );
        need!(
            self.original_deadline_unix_ms == previous.original_deadline_unix_ms,
            Invalid,
            "a later attempt cannot renew the original deadline"
        );
        let topology = |c: &Self| -> Vec<(u32, u32, String, Option<String>)> {
            c.participants
                .iter()
                .map(|p| {
                    (
                        p.global_rank,
                        p.local_rank,
                        p.host.host.clone(),
                        p.device.as_ref().map(|d| d.uuid.clone()),
                    )
                })
                .collect()
        };
        // The selected training partition is topology too: a changed rank count, microbatch
        // or partition version is never an exact continuation (W7-T06).
        need!(
            topology(self) == topology(previous) && self.training == previous.training,
            Invalid,
            "a topology change requires an explicit transition and requalification"
        );
        Ok(())
    }
    /// Reviewable summary printed by Plan; it is derived, not a second authority.
    pub fn review(&self, plan_sha256: &str) -> Value {
        let participants: Vec<Value> = self
            .participants
            .iter()
            .map(|p| {
                json!({
                    "global_rank": p.global_rank,
                    "local_rank": p.local_rank,
                    "host": p.host.host,
                    "boot_id": p.host.boot_id,
                    "device": p.device,
                    "ceilings": p.ceilings,
                    "grant_expires_unix_ms": p.grant.body.expires_unix_ms,
                    "result_destination": p.result_destination,
                })
            })
            .collect();
        let mut kinds: BTreeMap<String, u64> = BTreeMap::new();
        for w in &self.work_census {
            // The kind's serde name (`numerical`), never its JSON rendering (`"numerical"`).
            let kind = serde_json::to_value(w.kind)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_default();
            *kinds.entry(kind).or_default() += w.wall_ms;
        }
        let mut review = json!({
            "schema": "jope.native_bounded_plan_review.v1",
            "plan_sha256": plan_sha256,
            "run": self.run,
            "attempt": self.attempt,
            "world_size": self.world_size(),
            "coordinator_host": self.coordinator_host,
            "not_before_unix_ms": self.not_before_unix_ms,
            "original_deadline_unix_ms": self.original_deadline_unix_ms,
            "participants": participants,
            "artifact_closure_entries": self.artifact_closure.len(),
            "work_census_wall_ms_by_kind": kinds,
            "counters": self.counters,
            "predecessor": self.predecessor,
            "dispatch_started": false,
        });
        // Only a plan that selects training gains the key; every other review is unchanged.
        if let Some(t) = &self.training {
            review["training"] = json!({
                "selection": t,
                "execution_partition": self.execution_partition().ok().flatten(),
            });
        }
        review
    }
}
