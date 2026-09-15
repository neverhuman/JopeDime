//! Signed funding and single-use native task admission.
//!
//! This module never chooses production trust, opens a ledger, creates a grant
//! from historical observations, or releases a numerical worker. The existing
//! authority service supplies installed trust and authenticated process evidence,
//! runs these SQL operations in its journaled transaction, and obtains the
//! independent event-witness ACK before returning a start/participant receipt.
use anyhow::{ensure, Context, Result};
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

use super::ProcessIdentity;

pub const SQL_SCHEMA: &str = "
CREATE TABLE funding_authorizations(id TEXT PRIMARY KEY,domain_id TEXT NOT NULL UNIQUE,evidence_sha TEXT NOT NULL,json TEXT NOT NULL) WITHOUT ROWID;
CREATE TABLE funding_reservations(id TEXT PRIMARY KEY,authorization_id TEXT NOT NULL,task_sha256 TEXT NOT NULL,json TEXT NOT NULL,state TEXT NOT NULL,start_json TEXT,UNIQUE(authorization_id,task_sha256)) WITHOUT ROWID;
CREATE TABLE funding_participants(id TEXT PRIMARY KEY,reservation_id TEXT NOT NULL,slot TEXT NOT NULL,json TEXT NOT NULL,UNIQUE(reservation_id,slot)) WITHOUT ROWID;
CREATE TABLE funding_witness(event_key TEXT PRIMARY KEY,event_sha TEXT NOT NULL,request_sha TEXT NOT NULL,ack_sha TEXT NOT NULL,policy_sha TEXT NOT NULL) WITHOUT ROWID;";

const AUTH_TAG: &str = "parallel.funding-authorization.signature.v1";
const RESERVATION_TAG: &str = "parallel.funding-reservation.signature.v1";
// Funding witness proofs retain both the original event and signed request.
// Bound the signed source before hashing/signature work or any ledger mutation.
pub const MAX_AUTHORIZATION_BYTES: usize = 128 * 1024;
pub const MAX_APPROVED_TASKS: usize = 1024;

pub fn digest(value: &impl Serialize) -> Result<String> {
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(value)?)))
}

fn hash_valid(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
        && value.bytes().any(|c| c != b'0')
}

fn text_valid(value: &str) -> bool {
    !value.is_empty() && value.len() <= 256 && !value.chars().any(char::is_control)
}

fn signed_bytes(tag: &str, body: &impl Serialize) -> Result<Vec<u8>> {
    let mut bytes = tag.as_bytes().to_vec();
    bytes.push(0);
    bytes.extend(serde_json::to_vec(body)?);
    Ok(bytes)
}

fn verify_signature(public: &str, tag: &str, body: &impl Serialize, signature: &str) -> Result<()> {
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
        let mut value: libc::timespec = unsafe { std::mem::zeroed() };
        ensure!(
            unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut value) } == 0,
            "funding monotonic clock unavailable"
        );
        let seconds: u64 = value.tv_sec.try_into()?;
        let nanos: u64 = value.tv_nsec.try_into()?;
        Ok(Self {
            unix_ms,
            monotonic_ns: seconds
                .checked_mul(1_000_000_000)
                .and_then(|n| n.checked_add(nanos))
                .context("funding clock overflow")?,
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

pub fn verify_authorization(
    trust: &FundingTrust,
    authorization: &SignedAuthorization,
    clock: FundingClock,
) -> Result<String> {
    ensure!(
        trust.schema == "parallel.funding_trust.v1"
            && text_valid(&trust.domain_id)
            && text_valid(&trust.authority_instance)
            && text_valid(&trust.authority_epoch)
            && hash_valid(&trust.authority_manifest_sha256)
            && hash_valid(&trust.authority_executable_sha256)
            && text_valid(&trust.host_id)
            && text_valid(&trust.boot_id),
        "installed funding trust invalid"
    );
    let body = &authorization.body;
    ensure!(
        body.approved_task_sha256.len() <= MAX_APPROVED_TASKS
            && serde_json::to_vec(body)?.len() <= MAX_AUTHORIZATION_BYTES,
        "funding authorization exceeds bounded witness input limits"
    );
    ensure!(
        body.schema == "parallel.funding_authorization.v1"
            && body.domain_id == trust.domain_id
            && text_valid(&body.scope),
        "funding authorization domain/scope mismatch"
    );
    let id = digest(body)?;
    ensure!(
        hash_valid(&trust.authorization_sha256) && id == trust.authorization_sha256,
        "funding authorization is not the independently pinned decision"
    );
    verify_signature(
        &trust.authorization_public_key,
        AUTH_TAG,
        body,
        &authorization.signature,
    )?;
    ensure!(
        body.reservation_public_key == trust.reservation_public_key
            && hash_valid(&body.reservation_public_key),
        "funding reservation delegation mismatch"
    );
    ensure!(
        hash_valid(&body.authority_evidence_sha256)
            && hash_valid(&body.historical_reconciliation_sha256)
            && hash_valid(&body.release_prerequisites_sha256)
            && !body.approved_task_sha256.is_empty()
            && body.approved_task_sha256.iter().all(|id| hash_valid(id)),
        "original authority/accounting/prerequisite evidence or frozen task list missing"
    );
    ensure!(
        body.period_start_unix_ms <= clock.unix_ms && clock.unix_ms < body.period_end_unix_ms,
        "funding accounting period unavailable or elapsed"
    );
    ensure!(
        body.period_start_unix_ms < body.period_end_unix_ms
            && body.historical_core_ms <= body.core_ms
            && body.historical_node_ms <= body.node_ms
            && (body.core_ms > 0 || body.node_ms > 0)
            && body.max_concurrent_cpu_searches <= 104,
        "invalid funding ceilings/history"
    );
    ensure!(
        body.allocation.cpu_cores > 0
            && body.allocation.host_memory_bytes > 0
            && body.allocation.gpu_memory_bytes_per_device > 0
            && !body.allocation.gpu_devices.is_empty()
            && body.allocation.gpu_devices.len() <= 8
            && body.allocation.gpu_devices.iter().all(|&device| device < 8)
            && body
                .allocation
                .gpu_devices
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .len()
                == body.allocation.gpu_devices.len(),
        "invalid independently authorized host allocation"
    );
    ensure!(
        !body.capabilities.is_empty() && body.capabilities.iter().all(|s| text_valid(s)),
        "funding capabilities missing"
    );
    Ok(id)
}

pub fn verify_reservation(
    trust: &FundingTrust,
    authorization: &SignedAuthorization,
    reservation: &SignedReservation,
    demand: &TaskDemand,
    clock: FundingClock,
) -> Result<VerifiedReservation> {
    let authorization_id = verify_authorization(trust, authorization, clock)?;
    let body = &reservation.body;
    ensure!(
        body.schema == "parallel.task_reservation.v1"
            && hash_valid(&body.reservation_id)
            && body.authorization_sha256 == authorization_id
            && body.domain_id == trust.domain_id
            && body.authority_instance == trust.authority_instance
            && body.authority_epoch == trust.authority_epoch
            && body.authority_manifest_sha256 == trust.authority_manifest_sha256
            && body.authority_executable_sha256 == trust.authority_executable_sha256,
        "reservation original domain/authority/epoch mismatch"
    );
    verify_signature(
        &trust.reservation_public_key,
        RESERVATION_TAG,
        body,
        &reservation.signature,
    )?;
    ensure!(
        &body.demand == demand,
        "actual task/deadline/runtime/resource demand differs from signed reservation"
    );
    ensure!(
        hash_valid(&demand.task_sha256)
            && hash_valid(&demand.executable_sha256)
            && hash_valid(&demand.runtime_sha256)
            && demand.host_id == trust.host_id
            && demand.boot_id == trust.boot_id
            && authorization.body.capabilities.contains(&demand.capability)
            && authorization
                .body
                .approved_task_sha256
                .contains(&demand.task_sha256),
        "task identity/host/boot/capability not funded"
    );
    ensure!(
        body.not_before_unix_ms >= authorization.body.period_start_unix_ms
            && body.not_before_unix_ms <= clock.unix_ms
            && clock.unix_ms < demand.deadline_unix_ms
            && demand.deadline_unix_ms <= authorization.body.period_end_unix_ms
            && clock.monotonic_ns < demand.deadline_monotonic_ns,
        "original funding task window elapsed or changed"
    );
    let window = demand
        .deadline_unix_ms
        .checked_sub(body.not_before_unix_ms)
        .context("invalid task window")?;
    let remaining_ns = demand.deadline_monotonic_ns - clock.monotonic_ns;
    ensure!(
        remaining_ns
            <= window
                .checked_mul(1_000_000)
                .context("task window overflow")?,
        "monotonic task deadline exceeds original reserved window"
    );
    let caps = &demand.resources;
    ensure!(
        [1, 2, 4, 8].contains(&caps.world)
            && caps.gpus == caps.world
            && caps.cpu_cores > 0
            && caps.host_memory_bytes > 0
            && caps.gpu_memory_bytes_per_rank > 0
            && caps.cpu_searches <= authorization.body.max_concurrent_cpu_searches
            && caps.gpu_devices.len() == caps.world as usize
            && caps
                .gpu_devices
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .len()
                == caps.gpu_devices.len()
            && caps.gpu_devices.iter().all(|device| authorization
                .body
                .allocation
                .gpu_devices
                .contains(device))
            && caps.cpu_cores <= authorization.body.allocation.cpu_cores
            && caps.host_memory_bytes <= authorization.body.allocation.host_memory_bytes
            && caps.gpu_memory_bytes_per_rank
                <= authorization.body.allocation.gpu_memory_bytes_per_device,
        "invalid native task resource caps"
    );
    ensure!(
        body.participant_slots == (0..caps.world).collect::<Vec<_>>(),
        "original participant slots do not exactly cover the declared world"
    );
    let core_ceiling = window
        .checked_mul(u64::from(caps.cpu_cores))
        .context("core reservation overflow")?;
    ensure!(
        body.core_ms >= core_ceiling
            && body.node_ms >= window
            && body.core_ms <= authorization.body.core_ms
            && body.node_ms <= authorization.body.node_ms,
        "task reservation does not cover its full original resource/window ceiling"
    );
    Ok(VerifiedReservation {
        authorization: authorization.clone(),
        reservation: reservation.clone(),
    })
}

/// Authority signing only: the caller must possess the installed delegated key;
/// successful signing does not register, charge or start the reservation.
pub fn sign_reservation(body: TaskReservation, key: &SigningKey) -> Result<SignedReservation> {
    let signature = hex::encode(key.sign(&signed_bytes(RESERVATION_TAG, &body)?).to_bytes());
    Ok(SignedReservation { body, signature })
}

fn transaction_required(db: &Connection) -> Result<()> {
    ensure!(
        !db.is_autocommit(),
        "funding mutation requires the original authority transaction"
    );
    Ok(())
}

pub fn register_authorization(
    db: &Connection,
    trust: &FundingTrust,
    authorization: &SignedAuthorization,
    clock: FundingClock,
) -> Result<String> {
    transaction_required(db)?;
    let id = verify_authorization(trust, authorization, clock)?;
    let existing: Option<String> = db
        .query_row(
            "SELECT id FROM funding_authorizations WHERE domain_id=?",
            [&authorization.body.domain_id],
            |row| row.get(0),
        )
        .optional()?;
    ensure!(
        existing.is_none(),
        "funding domain already registered; reset/reissue forbidden"
    );
    db.execute(
        "INSERT INTO funding_authorizations VALUES(?,?,?,?)",
        params![
            id,
            authorization.body.domain_id,
            authorization.body.authority_evidence_sha256,
            serde_json::to_string(authorization)?
        ],
    )?;
    Ok(id)
}

fn stored_authorization(db: &Connection, verified: &VerifiedReservation) -> Result<()> {
    let bytes: String = db.query_row(
        "SELECT json FROM funding_authorizations WHERE id=?",
        [verified.authorization_id()?],
        |row| row.get(0),
    )?;
    ensure!(
        serde_json::from_str::<SignedAuthorization>(&bytes)? == verified.authorization,
        "original registered funding authorization differs"
    );
    Ok(())
}

pub fn reserve(db: &Connection, verified: &VerifiedReservation) -> Result<String> {
    transaction_required(db)?;
    stored_authorization(db, verified)?;
    let auth = &verified.authorization.body;
    let mut core = auth.historical_core_ms;
    let mut node = auth.historical_node_ms;
    let mut statement =
        db.prepare("SELECT json FROM funding_reservations WHERE authorization_id=?")?;
    let rows = statement.query_map([verified.authorization_id()?], |row| {
        row.get::<_, String>(0)
    })?;
    for row in rows {
        let old: SignedReservation = serde_json::from_str(&row?)?;
        core = core
            .checked_add(old.body.core_ms)
            .context("funding core charge overflow")?;
        node = node
            .checked_add(old.body.node_ms)
            .context("funding node charge overflow")?;
    }
    let body = &verified.reservation.body;
    ensure!(
        core.checked_add(body.core_ms)
            .is_some_and(|sum| sum <= auth.core_ms)
            && node
                .checked_add(body.node_ms)
                .is_some_and(|sum| sum <= auth.node_ms),
        "original funding ceiling exhausted"
    );
    db.execute(
        "INSERT INTO funding_reservations VALUES(?,?,?,?,?,NULL)",
        params![
            body.reservation_id,
            verified.authorization_id()?,
            body.demand.task_sha256,
            serde_json::to_string(&verified.reservation)?,
            "reserved"
        ],
    )?;
    Ok(body.reservation_id.clone())
}

fn original_reservation(
    db: &Connection,
    verified: &VerifiedReservation,
) -> Result<(String, Option<String>)> {
    stored_authorization(db, verified)?;
    let (json, state, start): (String, String, Option<String>) = db.query_row(
        "SELECT json,state,start_json FROM funding_reservations WHERE id=?",
        [verified.reservation_id()],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )?;
    ensure!(
        serde_json::from_str::<SignedReservation>(&json)? == verified.reservation,
        "original registered reservation differs"
    );
    Ok((state, start))
}

pub fn preflight(db: &Connection, verified: &VerifiedReservation) -> Result<()> {
    let (state, _) = original_reservation(db, verified)?;
    ensure!(
        state == "reserved",
        "original task start already consumed or unavailable"
    );
    Ok(())
}

fn execution_valid(
    verified: &VerifiedReservation,
    execution: &ExecutionClaim,
    clock: FundingClock,
) -> Result<()> {
    let demand = verified.demand();
    ensure!(
        execution.process.boot_id == demand.boot_id
            && execution.process.pid > 0
            && execution.process.birth_ticks > 0
            && execution.executable_sha256 == demand.executable_sha256
            && execution.runtime_sha256 == demand.runtime_sha256
            && execution.request_sha256 == demand.task_sha256,
        "authenticated execution custody does not match the original funded task"
    );
    ensure!(
        clock.unix_ms >= verified.reservation.body.not_before_unix_ms
            && clock.unix_ms < demand.deadline_unix_ms
            && clock.monotonic_ns < demand.deadline_monotonic_ns,
        "original task deadline elapsed before funding consumption"
    );
    Ok(())
}

pub fn participant_id(reservation_id: &str, slot: u32) -> Result<String> {
    digest(&("parallel.funding_participant.v1", reservation_id, slot))
}

pub fn event(
    verified: &VerifiedReservation,
    execution: &ExecutionClaim,
    participant_slot: Option<u32>,
    clock: FundingClock,
) -> Result<FundingEvent> {
    let event_key = match participant_slot {
        Some(slot) => participant_id(verified.reservation_id(), slot)?,
        None => digest(&("parallel.funding_start.v1", verified.reservation_id()))?,
    };
    Ok(FundingEvent {
        schema: "parallel.funding_event.v1".into(),
        event_key,
        authorization: verified.authorization.clone(),
        reservation: verified.reservation.clone(),
        demand: verified.demand().clone(),
        execution: execution.clone(),
        participant_slot,
        claimed_unix_ms: clock.unix_ms,
        claimed_monotonic_ns: clock.monotonic_ns,
    })
}

pub fn consume_start(
    db: &Connection,
    verified: &VerifiedReservation,
    execution: &ExecutionClaim,
    clock: FundingClock,
) -> Result<FundingEvent> {
    transaction_required(db)?;
    execution_valid(verified, execution, clock)?;
    preflight(db, verified)?;
    concurrent_admission(db, verified)?;
    let event = event(verified, execution, None, clock)?;
    ensure!(db.execute("UPDATE funding_reservations SET state='started',start_json=? WHERE id=? AND state='reserved'",
        params![serde_json::to_string(&event)?, verified.reservation_id()])? == 1, "funding start consumed concurrently");
    Ok(event)
}

/// Check every still-started reservation atomically, across funding domains.
/// An unknown or incomplete termination keeps its original allocation occupied;
/// successful task completion never refunds the charged funding ceilings.
fn concurrent_admission(db: &Connection, verified: &VerifiedReservation) -> Result<()> {
    let caps = &verified.demand().resources;
    let allocation = &verified.authorization.body.allocation;
    let (mut cores, mut memory, mut searches) =
        (caps.cpu_cores, caps.host_memory_bytes, caps.cpu_searches);
    let requested: BTreeSet<u32> = caps.gpu_devices.iter().copied().collect();
    let mut statement =
        db.prepare("SELECT json FROM funding_reservations WHERE state='started'")?;
    let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
    for row in rows {
        let old: SignedReservation = serde_json::from_str(&row?)?;
        if old.body.demand.host_id != verified.demand().host_id {
            continue;
        }
        // A host reboot is not evidence that the original allocation/attempt
        // was independently reconciled. It cannot silently release custody.
        ensure!(
            old.body.demand.boot_id == verified.demand().boot_id,
            "unreconciled funding allocation from an earlier host boot"
        );
        let old_caps = &old.body.demand.resources;
        ensure!(
            old_caps
                .gpu_devices
                .iter()
                .all(|id| !requested.contains(id)),
            "funded GPU allocation already occupied"
        );
        cores = cores
            .checked_add(old_caps.cpu_cores)
            .context("concurrent CPU allocation overflow")?;
        memory = memory
            .checked_add(old_caps.host_memory_bytes)
            .context("concurrent host memory overflow")?;
        searches = searches
            .checked_add(old_caps.cpu_searches)
            .context("concurrent search allocation overflow")?;
    }
    ensure!(
        cores <= allocation.cpu_cores
            && memory <= allocation.host_memory_bytes
            && searches <= verified.authorization.body.max_concurrent_cpu_searches,
        "funded concurrent host allocation exhausted"
    );
    Ok(())
}

pub fn claim_participant(
    db: &Connection,
    verified: &VerifiedReservation,
    slot: u32,
    execution: &ExecutionClaim,
    clock: FundingClock,
) -> Result<FundingEvent> {
    transaction_required(db)?;
    execution_valid(verified, execution, clock)?;
    ensure!(
        verified.reservation.body.participant_slots.contains(&slot),
        "undeclared funding participant slot"
    );
    let (state, start) = original_reservation(db, verified)?;
    ensure!(
        state == "started" && start.is_some(),
        "participant cannot precede the original task start"
    );
    let start: FundingEvent =
        serde_json::from_str(start.as_deref().context("original task start absent")?)?;
    ensure!(
        start.reservation == verified.reservation && start.participant_slot.is_none(),
        "participant original supervisor start mismatch"
    );
    // A start whose witness response was lost cannot release any rank.
    ensure!(
        db.query_row(
            "SELECT EXISTS(SELECT 1 FROM funding_witness WHERE event_key=?)",
            [&start.event_key],
            |row| row.get::<_, bool>(0)
        )?,
        "original task start has no retained independent witness ACK"
    );
    let mut statement =
        db.prepare("SELECT json FROM funding_participants WHERE reservation_id=?")?;
    let originals =
        statement.query_map([verified.reservation_id()], |row| row.get::<_, String>(0))?;
    for original in originals {
        let original: FundingEvent = serde_json::from_str(&original?)?;
        ensure!(
            original.execution.process != execution.process,
            "original participant process already consumed a rank slot"
        );
    }
    let event = event(verified, execution, Some(slot), clock)?;
    db.execute(
        "INSERT INTO funding_participants VALUES(?,?,?,?)",
        params![
            event.participant_id()?,
            verified.reservation_id(),
            slot.to_string(),
            serde_json::to_string(&event)?
        ],
    )?;
    Ok(event)
}

/// Validate the full signed event against installed trust and the independently
/// observed execution identity. This is suitable for the existing witness's
/// strict funding-event branch; it does not accept arbitrary journal payloads.
pub fn verify_event(trust: &FundingTrust, value: &FundingEvent, clock: FundingClock) -> Result<()> {
    validate_witness_event(
        trust,
        value,
        &trust.authority_instance,
        &trust.authority_epoch,
    )?;
    let verified = verify_reservation(
        trust,
        &value.authorization,
        &value.reservation,
        &value.demand,
        clock,
    )?;
    execution_valid(&verified, &value.execution, clock)
}

/// Verify original event bytes at their recorded claim time. This supports
/// retrieval of a previously committed ACK after the original deadline without
/// restamping it or authorizing another start. New witness writes must separately
/// enforce the returned original monotonic deadline on their transport path.
pub fn validate_witness_event(
    trust: &FundingTrust,
    value: &FundingEvent,
    authority_instance: &str,
    authority_epoch: &str,
) -> Result<(String, u64)> {
    ensure!(
        trust.authority_instance == authority_instance && trust.authority_epoch == authority_epoch,
        "funding witness authority instance/epoch mismatch"
    );
    ensure!(
        value.schema == "parallel.funding_event.v1",
        "unsupported funding event"
    );
    let clock = FundingClock {
        unix_ms: value.claimed_unix_ms,
        monotonic_ns: value.claimed_monotonic_ns,
    };
    let verified = verify_reservation(
        trust,
        &value.authorization,
        &value.reservation,
        &value.demand,
        clock,
    )?;
    execution_valid(&verified, &value.execution, clock)?;
    if let Some(slot) = value.participant_slot {
        ensure!(
            value.reservation.body.participant_slots.contains(&slot),
            "funding event participant outside reservation"
        );
    }
    ensure!(
        event(&verified, &value.execution, value.participant_slot, clock)? == *value,
        "funding event original identity changed"
    );
    Ok((value.event_key.clone(), value.demand.deadline_monotonic_ns))
}

#[cfg(test)]
mod tests;
