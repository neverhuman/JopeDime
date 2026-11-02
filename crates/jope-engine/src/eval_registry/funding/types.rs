use super::*;

pub const SQL_SCHEMA: &str = "
CREATE TABLE funding_authorizations(id TEXT PRIMARY KEY,domain_id TEXT NOT NULL UNIQUE,evidence_sha TEXT NOT NULL,json TEXT NOT NULL) WITHOUT ROWID;
CREATE TABLE funding_reservations(id TEXT PRIMARY KEY,authorization_id TEXT NOT NULL,task_sha256 TEXT NOT NULL,json TEXT NOT NULL,state TEXT NOT NULL,start_json TEXT,UNIQUE(authorization_id,task_sha256)) WITHOUT ROWID;
CREATE TABLE funding_participants(id TEXT PRIMARY KEY,reservation_id TEXT NOT NULL,slot TEXT NOT NULL,json TEXT NOT NULL,UNIQUE(reservation_id,slot)) WITHOUT ROWID;
CREATE TABLE funding_witness(event_key TEXT PRIMARY KEY,event_sha TEXT NOT NULL,request_sha TEXT NOT NULL,ack_sha TEXT NOT NULL,policy_sha TEXT NOT NULL) WITHOUT ROWID;";

pub(super) const AUTH_TAG: &str = "parallel.funding-authorization.signature.v1";
pub(super) const RESERVATION_TAG: &str = "parallel.funding-reservation.signature.v1";
// Funding witness proofs retain both the original event and signed request.
// Bound the signed source before hashing/signature work or any ledger mutation.
pub const MAX_AUTHORIZATION_BYTES: usize = 128 * 1024;
pub const MAX_APPROVED_TASKS: usize = 1024;

pub fn digest(value: &impl Serialize) -> Result<String> {
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(value)?)))
}

pub(super) fn hash_valid(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
        && value.bytes().any(|c| c != b'0')
}

pub(super) fn text_valid(value: &str) -> bool {
    !value.is_empty() && value.len() <= 256 && !value.chars().any(char::is_control)
}

pub(super) fn signed_bytes(tag: &str, body: &impl Serialize) -> Result<Vec<u8>> {
    let mut bytes = tag.as_bytes().to_vec();
    bytes.push(0);
    bytes.extend(serde_json::to_vec(body)?);
    Ok(bytes)
}

pub(super) fn verify_signature(
    public: &str,
    tag: &str,
    body: &impl Serialize,
    signature: &str,
) -> Result<()> {
    ensure!(hash_valid(public), "invalid installed funding public key");
    let raw: [u8; 32] = hex::decode(public)?
        .try_into()
        .map_err(|_| anyhow::anyhow!("funding key length"))?;
    let key = VerifyingKey::from_bytes(&raw)?;
    ensure!(!key.is_weak(), "weak funding public key");
    let raw = hex::decode(signature)?;
    let signature = Signature::from_slice(&raw)?;
    key.verify_strict(&signed_bytes(tag, body)?, &signature)?;
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FundingTrust {
    pub schema: String,
    pub domain_id: String,
    pub authority_instance: String,
    pub authority_epoch: String,
    pub authority_manifest_sha256: String,
    pub authority_executable_sha256: String,
    pub authorization_public_key: String,
    /// Exact independently approved canonical authorization body; not supplied
    /// by the client that asks to reserve or execute work.
    pub authorization_sha256: String,
    pub reservation_public_key: String,
    pub host_id: String,
    pub boot_id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FundingAuthorization {
    pub schema: String,
    pub domain_id: String,
    pub scope: String,
    pub period_start_unix_ms: u64,
    pub period_end_unix_ms: u64,
    pub authority_evidence_sha256: String,
    /// Reviewed original charge/unknown-ceiling reconciliation, whose accepted
    /// totals are part of the independently signed and pinned authorization.
    pub historical_reconciliation_sha256: String,
    pub release_prerequisites_sha256: String,
    pub approved_task_sha256: BTreeSet<String>,
    pub historical_core_ms: u64,
    pub historical_node_ms: u64,
    pub core_ms: u64,
    pub node_ms: u64,
    pub max_concurrent_cpu_searches: u32,
    pub allocation: FundingAllocation,
    pub capabilities: BTreeSet<String>,
    pub reservation_public_key: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SignedAuthorization {
    pub body: FundingAuthorization,
    pub signature: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ResourceCaps {
    pub cpu_cores: u32,
    pub gpus: u32,
    pub world: u32,
    pub host_memory_bytes: u64,
    pub gpu_memory_bytes_per_rank: u64,
    pub cpu_searches: u32,
    pub gpu_devices: Vec<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FundingAllocation {
    pub cpu_cores: u32,
    pub gpu_devices: Vec<u32>,
    pub host_memory_bytes: u64,
    pub gpu_memory_bytes_per_device: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskDemand {
    pub task_sha256: String,
    pub capability: String,
    pub host_id: String,
    pub boot_id: String,
    pub executable_sha256: String,
    pub runtime_sha256: String,
    pub deadline_unix_ms: u64,
    pub deadline_monotonic_ns: u64,
    pub resources: ResourceCaps,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskReservation {
    pub schema: String,
    pub reservation_id: String,
    pub authorization_sha256: String,
    pub authority_instance: String,
    pub authority_epoch: String,
    pub authority_manifest_sha256: String,
    pub authority_executable_sha256: String,
    pub domain_id: String,
    pub not_before_unix_ms: u64,
    pub demand: TaskDemand,
    pub core_ms: u64,
    pub node_ms: u64,
    /// Every rank may claim its original slot once. A controller preflight is
    /// read-only and does not occupy a numerical rank slot.
    pub participant_slots: Vec<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SignedReservation {
    pub body: TaskReservation,
    pub signature: String,
}

#[derive(Clone, Copy, Debug)]
pub struct FundingClock {
    pub unix_ms: u64,
    pub monotonic_ns: u64,
}

impl FundingClock {
    pub fn current() -> Result<Self> {
        let unix_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_millis()
            .try_into()?;
        Ok(Self {
            unix_ms,
            monotonic_ns: crate::host_os::monotonic_ns()
                .context("funding monotonic clock unavailable")?,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExecutionClaim {
    /// Obtained by the authority from authenticated peer/original child custody.
    pub process: ProcessIdentity,
    pub executable_sha256: String,
    pub runtime_sha256: String,
    /// Authority-computed digest of the sealed common task descriptor. This
    /// must equal the signed task digest; a per-rank transport request digest
    /// is additional execution evidence, never a replacement for this binding.
    pub request_sha256: String,
}

#[derive(Clone, Debug)]
pub struct VerifiedReservation {
    authorization: SignedAuthorization,
    reservation: SignedReservation,
}

impl VerifiedReservation {
    pub fn authorization(&self) -> &SignedAuthorization {
        &self.authorization
    }
    pub fn reservation(&self) -> &SignedReservation {
        &self.reservation
    }
    pub fn demand(&self) -> &TaskDemand {
        &self.reservation.body.demand
    }
    pub fn authorization_id(&self) -> Result<String> {
        digest(&self.authorization.body)
    }
    pub fn reservation_id(&self) -> &str {
        &self.reservation.body.reservation_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FundingEvent {
    pub schema: String,
    pub event_key: String,
    pub authorization: SignedAuthorization,
    pub reservation: SignedReservation,
    pub demand: TaskDemand,
    pub execution: ExecutionClaim,
    pub participant_slot: Option<u32>,
    pub claimed_unix_ms: u64,
    pub claimed_monotonic_ns: u64,
}

impl FundingEvent {
    pub fn reservation_id(&self) -> &str {
        &self.reservation.body.reservation_id
    }
    pub fn participant_id(&self) -> Result<String> {
        participant_id(
            self.reservation_id(),
            self.participant_slot.context("rank slot absent")?,
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FundingReceipt {
    pub event: FundingEvent,
    pub event_sha256: String,
    pub request_sha256: String,
    pub ack_sha256: String,
    pub policy_sha256: String,
}
