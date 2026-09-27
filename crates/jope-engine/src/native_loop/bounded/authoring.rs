//! Contract authoring and grant signing for bounded runs (W7-D6), so nothing is hand-assembled.
//!
//! * [`author`] turns a small declarative request (`jope.native_bounded_contract_request.v1`)
//!   and a merged host census into an **unsigned** `jope.native_bounded_run.v1` contract. Host
//!   identities and device indices come only from the census, closure digests only from the
//!   files themselves, and every grant body is derived from its participant. Every structural
//!   check except the grant signatures runs before anything is written.
//! * [`sign_grants`] signs every grant body with the grant-authority key and returns the
//!   signed contract, which then passes the full validation `plan` applies. It refuses a key
//!   that is not the contract's grant authority and a grant whose window has already expired.
//! * [`verify_grants`] checks a signed contract as `plan` will: structure, every grant
//!   signature and binding, and that no grant has expired.
//!
//! The authority key is only ever read from a file (PKCS#8 Ed25519, owner-only mode, not a
//! symlink: [`super::participant::load_signing_key`]); it is never an argument or an
//! environment value and is never printed.
use super::{
    census::HostCensus,
    contract::{
        BoundedRunContract, Ceilings, ClosureEntry, Counters, DeviceAssignment, GrantBody,
        Participant, Predecessor, RendezvousPolicy, ResultDestination, SignedGrant,
        TelemetryPolicy, TrainingSelection, WorkItem, WorkKind, CONTRACT_SCHEMA, GRANT_SCHEMA,
        MAX_ARTIFACT_BYTES,
    },
    is_relative,
};
use anyhow::{anyhow, bail, ensure, Context, Result};
use ed25519_dalek::SigningKey;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

pub const REQUEST_SCHEMA: &str = "jope.native_bounded_contract_request.v1";
const DEFAULT_RESULT_BYTES: u64 = 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ParticipantRequest {
    /// A host named by the census.
    pub host: String,
    /// The rank's Ed25519 public key, 64 hex.
    pub public_key: String,
    /// The GPU this rank uses, by UUID; it must be in the census for this host. None for a
    /// CPU rank.
    #[serde(default)]
    pub device_uuid: Option<String>,
    pub ceilings: Ceilings,
}
/// One work item every rank plans; `numerical` work carries the plan's updates.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct WorkRequest {
    pub kind: WorkKind,
    pub cpu_core_ms: u64,
    pub gpu_ms: u64,
    pub wall_ms: u64,
    pub peak_ram_bytes: u64,
    pub peak_vram_bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ContractRequest {
    pub schema: String,
    pub run: String,
    pub attempt: u32,
    pub coordinator_host: String,
    pub coordinator_public_key: String,
    pub grant_authority_public_key: String,
    /// The window's start; authoring time when absent.
    #[serde(default)]
    pub not_before_unix_ms: Option<u64>,
    /// The original deadline is `not_before + window_ms`.
    pub window_ms: u64,
    /// Every grant expires `grant_ms` after `not_before`, within the window.
    pub grant_ms: u64,
    pub telemetry: TelemetryPolicy,
    pub rendezvous: RendezvousPolicy,
    /// Paths relative to the input root; each file's digest and size are read from it.
    pub artifact_closure: Vec<PathBuf>,
    /// In rank order.
    pub participants: Vec<ParticipantRequest>,
    pub work_per_rank: Vec<WorkRequest>,
    pub counters: Counters,
    #[serde(default)]
    pub predecessor: Option<Predecessor>,
    #[serde(default)]
    pub training: Option<TrainingSelection>,
    #[serde(default)]
    pub result_max_bytes: Option<u64>,
}

/// The unsigned contract a request and a census describe. `input_root` holds the closure.
pub fn author(
    request: &ContractRequest,
    census: &HostCensus,
    input_root: &Path,
    now_unix_ms: u64,
) -> Result<BoundedRunContract> {
    ensure!(
        request.schema == REQUEST_SCHEMA,
        "unsupported contract request schema `{}`",
        request.schema
    );
    census.validate()?;
    let not_before = request.not_before_unix_ms.unwrap_or(now_unix_ms);
    let deadline = not_before
        .checked_add(request.window_ms)
        .context("run window overflow")?;
    let expires = not_before
        .checked_add(request.grant_ms)
        .context("grant window overflow")?;
    ensure!(
        request.grant_ms > 0 && expires <= deadline,
        "grants must expire within the run window"
    );
    let mut closure = Vec::new();
    let mut seen = BTreeSet::new();
    for path in &request.artifact_closure {
        ensure!(
            is_relative(path) && seen.insert(path.clone()),
            "closure path `{}` must be relative, normal and listed once",
            path.display()
        );
        let bytes = super::launch::read_bounded(&input_root.join(path), MAX_ARTIFACT_BYTES)
            .with_context(|| format!("closure file `{}`", path.display()))?;
        closure.push(ClosureEntry {
            path: path.clone(),
            sha256: crate::manifest::hash(&bytes),
            bytes: bytes.len() as u64,
        });
    }
    closure.sort_by(|a, b| a.path.cmp(&b.path));
    let hosts: BTreeMap<&str, _> = census
        .hosts
        .iter()
        .map(|h| (h.identity.host.as_str(), h))
        .collect();
    let mut local = BTreeMap::<&str, u32>::new();
    let mut participants = Vec::new();
    for (rank, p) in request.participants.iter().enumerate() {
        let observed = hosts
            .get(p.host.as_str())
            .with_context(|| format!("rank {rank}'s host `{}` is not in the census", p.host))?;
        let local_rank = local.entry(p.host.as_str()).or_default();
        let device = p
            .device_uuid
            .as_ref()
            .map(|uuid| {
                let gpu = observed
                    .gpus
                    .iter()
                    .find(|g| &g.uuid == uuid)
                    .with_context(|| {
                        format!(
                            "rank {rank}'s GPU {uuid} is not in host `{}`'s census",
                            p.host
                        )
                    })?;
                Ok::<_, anyhow::Error>(DeviceAssignment {
                    uuid: uuid.clone(),
                    local_index: gpu.local_index,
                })
            })
            .transpose()?;
        let global_rank = u32::try_from(rank)?;
        let body = GrantBody {
            schema: GRANT_SCHEMA.into(),
            run: request.run.clone(),
            attempt: request.attempt,
            global_rank,
            host: p.host.clone(),
            boot_id: observed.identity.boot_id.clone(),
            device_uuid: p.device_uuid.clone(),
            participant_public_key: p.public_key.clone(),
            cpu_millicores: p.ceilings.cpu_millicores,
            ram_bytes: p.ceilings.ram_bytes,
            gpu_vram_bytes: p.ceilings.gpu_vram_bytes,
            wall_ms: p.ceilings.wall_ms,
            not_before_unix_ms: not_before,
            expires_unix_ms: expires,
        };
        participants.push(Participant {
            global_rank,
            local_rank: *local_rank,
            host: observed.identity.clone(),
            device,
            ceilings: p.ceilings.clone(),
            public_key: p.public_key.clone(),
            grant: SignedGrant {
                body,
                signature: String::new(),
            },
            result_destination: ResultDestination {
                path: PathBuf::from(format!("rank-{rank}/result.bin")),
                max_bytes: request.result_max_bytes.unwrap_or(DEFAULT_RESULT_BYTES),
            },
        });
        *local_rank += 1;
    }
    let mut work_census = Vec::new();
    for rank in 0..participants.len() {
        for w in &request.work_per_rank {
            let kind = serde_json::to_value(w.kind)?
                .as_str()
                .map(str::to_string)
                .unwrap_or_default();
            work_census.push(WorkItem {
                id: format!("rank-{rank}-{kind}"),
                global_rank: rank as u32,
                kind: w.kind,
                cpu_core_ms: w.cpu_core_ms,
                gpu_ms: w.gpu_ms,
                wall_ms: w.wall_ms,
                peak_ram_bytes: w.peak_ram_bytes,
                peak_vram_bytes: w.peak_vram_bytes,
                updates: if w.kind == WorkKind::Numerical {
                    request.counters.planned_updates
                } else {
                    0
                },
            });
        }
    }
    let contract = BoundedRunContract {
        schema: CONTRACT_SCHEMA.into(),
        run: request.run.clone(),
        attempt: request.attempt,
        coordinator_host: request.coordinator_host.clone(),
        coordinator_public_key: request.coordinator_public_key.clone(),
        grant_authority_public_key: request.grant_authority_public_key.clone(),
        not_before_unix_ms: not_before,
        original_deadline_unix_ms: deadline,
        telemetry: request.telemetry.clone(),
        rendezvous: request.rendezvous.clone(),
        artifact_closure: closure,
        participants,
        work_census,
        counters: request.counters.clone(),
        predecessor: request.predecessor.clone(),
        training: request.training.clone(),
    };
    contract.validate_unsigned().map_err(|e| anyhow!("{e}"))?;
    Ok(contract)
}

/// Sign every grant of an unsigned contract with the grant-authority key.
pub fn sign_grants(
    unsigned: &BoundedRunContract,
    authority: &SigningKey,
    now_unix_ms: u64,
) -> Result<BoundedRunContract> {
    ensure!(
        hex::encode(authority.verifying_key().to_bytes()) == unsigned.grant_authority_public_key,
        "this key is not the contract's grant authority"
    );
    unsigned.validate_unsigned().map_err(|e| anyhow!("{e}"))?;
    let mut signed = unsigned.clone();
    for p in &mut signed.participants {
        ensure!(
            now_unix_ms < p.grant.body.expires_unix_ms,
            "the grant window for rank {} has already expired; author a new contract",
            p.global_rank
        );
        p.grant = super::contract::sign_grant(authority, p.grant.body.clone())?;
    }
    signed
        .validate_at(now_unix_ms)
        .map_err(|e| anyhow!("{e}"))?;
    Ok(signed)
}

/// What a verified contract grants, per rank.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GrantReport {
    pub schema: String,
    /// The digest `plan` will freeze: the SHA-256 of the signed contract's canonical bytes.
    /// Absent in the review of an unsigned contract, which is not a plan.
    pub plan_sha256: Option<String>,
    pub run: String,
    pub attempt: u32,
    pub grant_authority_public_key: String,
    pub grants: Vec<GrantSummary>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GrantSummary {
    pub global_rank: u32,
    pub host: String,
    pub device_uuid: Option<String>,
    pub not_before_unix_ms: u64,
    pub expires_unix_ms: u64,
    pub body_sha256: String,
}
pub const GRANT_REPORT_SCHEMA: &str = "jope.native_bounded_grants.v1";

fn report(contract: &BoundedRunContract, signed: bool) -> Result<GrantReport> {
    Ok(GrantReport {
        schema: GRANT_REPORT_SCHEMA.into(),
        plan_sha256: signed
            .then(|| {
                contract
                    .canonical_bytes()
                    .map(|b| crate::manifest::hash(&b))
            })
            .transpose()?,
        run: contract.run.clone(),
        attempt: contract.attempt,
        grant_authority_public_key: contract.grant_authority_public_key.clone(),
        grants: contract
            .participants
            .iter()
            .map(|p| {
                Ok(GrantSummary {
                    global_rank: p.global_rank,
                    host: p.host.host.clone(),
                    device_uuid: p.grant.body.device_uuid.clone(),
                    not_before_unix_ms: p.grant.body.not_before_unix_ms,
                    expires_unix_ms: p.grant.body.expires_unix_ms,
                    body_sha256: super::digest(&p.grant.body)?,
                })
            })
            .collect::<Result<_>>()?,
    })
}

/// A signed contract's grants, checked exactly as `plan` checks them at `now_unix_ms`.
pub fn verify_grants(contract_bytes: &[u8], now_unix_ms: u64) -> Result<GrantReport> {
    let contract = BoundedRunContract::parse(contract_bytes).map_err(|e| anyhow!("{e}"))?;
    contract
        .validate_at(now_unix_ms)
        .map_err(|e| anyhow!("{e}"))?;
    report(&contract, true)
}

/// The report of an unsigned contract, for the owner's review before signing.
pub fn unsigned_report(contract: &BoundedRunContract) -> Result<GrantReport> {
    contract.validate_unsigned().map_err(|e| anyhow!("{e}"))?;
    ensure!(
        contract
            .participants
            .iter()
            .all(|p| p.grant.signature.is_empty()),
        "an authored contract carries no signature"
    );
    report(contract, false)
}

/// Parse an unsigned contract (its grants carry empty signatures).
pub fn parse_unsigned(bytes: &[u8]) -> Result<BoundedRunContract> {
    ensure!(
        bytes.len() <= super::contract::MAX_CONTRACT_BYTES,
        "contract exceeds {} bytes",
        super::contract::MAX_CONTRACT_BYTES
    );
    let contract: BoundedRunContract = serde_json::from_slice(bytes)?;
    if contract
        .participants
        .iter()
        .any(|p| !p.grant.signature.is_empty())
    {
        bail!("the contract is already signed");
    }
    contract.validate_unsigned().map_err(|e| anyhow!("{e}"))?;
    Ok(contract)
}
