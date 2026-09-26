//! Authenticated rendezvous and verified artifact/result exchange (W7-T04).
//!
//! Replaces the shared-filesystem assumption of the legacy same-allocation NCCL path
//! (`neural::Nccl::init` polls an unauthenticated `nccl.id` file). Here every message is
//! a bounded length-prefixed frame carrying a domain-separated ed25519 signature from a key
//! frozen in the plan, bound to run, attempt, plan digest, a per-open rendezvous identity
//! (which includes the coordinator boot), the participant's boot, grant and input closure.
//! The whole synchronous group is admitted, or refused, before any rank may start; a
//! refusal is terminal for the rendezvous and commits nothing.
//!
//! Transport-independent verification lives in [`Rendezvous`], [`verify_challenge`],
//! [`verify_admission`] and [`verify_result`]. [`gather_unix`] / [`join_unix`] bind them to
//! a same-host Unix socket for fixtures; the admission is delivered only by the controller
//! after it is journaled. A cross-host transport must reuse the same verification and is
//! not provided or qualified here (W7-T07).
use super::{
    census::RuntimeIdentity,
    contract::{BoundedRunContract, ClosureEntry},
    digest, is_digest, sign, verify,
};
use anyhow::{ensure, Context, Result};
use ed25519_dalek::SigningKey;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    os::unix::{
        fs::OpenOptionsExt,
        net::{UnixListener, UnixStream},
    },
    path::Path,
    time::Duration,
};

pub const CHALLENGE_SCHEMA: &str = "jope.native_rendezvous_challenge.v1";
pub const HELLO_SCHEMA: &str = "jope.native_rendezvous_hello.v1";
pub const ADMISSION_SCHEMA: &str = "jope.native_group_admission.v1";
pub const RESULT_SCHEMA: &str = "jope.native_rank_result.v1";
const CHALLENGE_DOMAIN: &[u8] = b"jope.native_rendezvous_challenge.signature.v1\0";
const HELLO_DOMAIN: &[u8] = b"jope.native_rendezvous_hello.signature.v1\0";
const ADMISSION_DOMAIN: &[u8] = b"jope.native_group_admission.signature.v1\0";
const RESULT_DOMAIN: &[u8] = b"jope.native_rank_result.signature.v1\0";
/// Opaque communicator bootstrap (for example an NCCL unique id) carried only by rank 0's
/// hello and redistributed only inside a signed full-group admission.
pub const MAX_BOOTSTRAP_BYTES: usize = 256;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ChallengeBody {
    pub schema: String,
    pub run: String,
    pub attempt: u32,
    pub plan_sha256: String,
    pub rendezvous_id: String,
    pub coordinator_boot_id: String,
    pub opened_unix_ms: u64,
    pub join_deadline_unix_ms: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Challenge {
    pub body: ChallengeBody,
    pub signature: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HelloBody {
    pub schema: String,
    pub run: String,
    pub attempt: u32,
    pub plan_sha256: String,
    pub rendezvous_id: String,
    pub global_rank: u32,
    pub delivery_id: u64,
    pub sent_unix_ms: u64,
    /// Observed on the participant's own host, never copied from the plan.
    pub host: String,
    pub machine_id: String,
    pub boot_id: String,
    pub device_uuid: Option<String>,
    pub source_commit: String,
    pub executable_sha256: String,
    pub runtime: RuntimeIdentity,
    pub installed_policy_sha256: String,
    /// [`observed_closure_digest`] over the participant's local input copy.
    pub artifact_closure_sha256: String,
    pub grant_sha256: String,
    pub communicator_bootstrap: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Hello {
    pub body: HelloBody,
    pub signature: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Member {
    pub global_rank: u32,
    pub hello_sha256: String,
    pub host: String,
    pub boot_id: String,
    pub device_uuid: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdmissionBody {
    pub schema: String,
    pub run: String,
    pub attempt: u32,
    pub plan_sha256: String,
    pub rendezvous_id: String,
    pub admitted_unix_ms: u64,
    pub original_deadline_unix_ms: u64,
    pub grants_sha256: String,
    pub members: Vec<Member>,
    pub communicator_bootstrap: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GroupAdmission {
    pub body: AdmissionBody,
    pub signature: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ResultBody {
    pub schema: String,
    pub run: String,
    pub attempt: u32,
    pub plan_sha256: String,
    pub admission_sha256: String,
    pub global_rank: u32,
    pub delivery_id: u64,
    pub result_sha256: String,
    pub result_bytes: u64,
    pub updates: u64,
    pub completed_unix_ms: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SignedResult {
    pub body: ResultBody,
    pub signature: String,
}
/// Post-admission control. Heartbeats carry strictly increasing delivery ids.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ControlFrame {
    Heartbeat { delivery_id: u64 },
    Stop { reason: String },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "cause", rename_all = "snake_case", deny_unknown_fields)]
pub enum RefusalCause {
    Oversized { bytes: u64, limit: u64 },
    Malformed { detail: String },
    UnknownRank { rank: u32 },
    ForgedPeer { claimed_rank: u32 },
    ForgedCoordinator,
    WrongRun,
    ReplayedAttempt { attempt: u32 },
    WrongPlan,
    StaleRendezvous,
    ReplayedBoot { rank: u32 },
    WrongParticipant { rank: u32 },
    SubstitutedSource { rank: u32 },
    SubstitutedInput { rank: u32 },
    WrongGrant { rank: u32 },
    GrantExpired { rank: u32 },
    DeadlineElapsed,
    NotYetOpen,
    StaleMessage { rank: u32 },
    JoinTimeout { missing: Vec<u32> },
    Equivocation { rank: u32 },
    IncompleteGroup,
}
/// Terminal refusal of the whole synchronous group. No rank may start and no update is
/// committed; `committed_updates` is structurally zero.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GroupRefusal {
    pub run: String,
    pub attempt: u32,
    pub rendezvous_id: String,
    pub cause: RefusalCause,
    pub started_ranks: u32,
    pub committed_updates: u64,
}
impl std::fmt::Display for GroupRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "bounded-run group refused: {}",
            serde_json::to_string(&self.cause).unwrap_or_default()
        )
    }
}
impl std::error::Error for GroupRefusal {}

/// Deterministic digest of the planned (path, sha256, bytes) closure.
pub fn closure_digest(closure: &[ClosureEntry]) -> Result<String> {
    digest(&closure)
}
/// Hash the participant's own local copy of every closure entry (bounded by the planned
/// length). A substituted byte changes this digest, which the coordinator refuses.
pub fn observed_closure_digest(root: &Path, closure: &[ClosureEntry]) -> Result<String> {
    let mut observed = Vec::with_capacity(closure.len());
    for e in closure {
        let path = crate::manifest::resolve(root, &e.path)?;
        let file = fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(&path)?;
        ensure!(
            file.metadata()?.is_file(),
            "closure entry is not a regular file"
        );
        let mut bytes = Vec::new();
        file.take(e.bytes + 1).read_to_end(&mut bytes)?;
        observed.push(ClosureEntry {
            path: e.path.clone(),
            sha256: crate::manifest::hash(&bytes),
            bytes: bytes.len() as u64,
        });
    }
    closure_digest(&observed)
}
/// Immutable verified artifact receipt: the bytes must match the planned entry exactly and
/// are published without replacement. An identical existing file is idempotent; any
/// conflicting existing file is refused and retained.
pub fn receive_artifact(root: &Path, entry: &ClosureEntry, bytes: &[u8]) -> Result<()> {
    ensure!(
        bytes.len() as u64 == entry.bytes && crate::manifest::hash(bytes) == entry.sha256,
        "exchanged artifact `{}` differs from the planned closure",
        entry.path.display()
    );
    publish_immutable(root, &entry.path, bytes)
}
pub(crate) fn publish_immutable(root: &Path, relative: &Path, bytes: &[u8]) -> Result<()> {
    ensure!(super::is_relative(relative), "unsafe artifact path");
    let path = root.join(relative);
    let parent = path.parent().context("artifact parent")?;
    fs::create_dir_all(parent)?;
    let (mut f, staged) = tempfile::Builder::new()
        .prefix(".exchange-pending-")
        .tempfile_in(parent)?
        .keep()?;
    f.write_all(bytes)?;
    f.sync_all()?;
    match crate::native_persistence::publish_staged(&staged, &path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            crate::native_persistence::discard_staging(&staged)?;
            let existing = fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
                .open(&path)?;
            let mut old = Vec::new();
            existing
                .take(bytes.len() as u64 + 1)
                .read_to_end(&mut old)?;
            ensure!(
                old == bytes,
                "conflicting immutable artifact `{}`",
                relative.display()
            );
        }
        Err(e) => return Err(e.into()),
    }
    fs::File::open(parent)?.sync_all()?;
    Ok(())
}

/// Length-prefixed frame. The limit is enforced before the body is read or allocated.
pub fn write_frame(w: &mut impl Write, bytes: &[u8], limit: u32) -> Result<()> {
    ensure!(
        bytes.len() as u64 <= u64::from(limit),
        "frame exceeds message limit"
    );
    w.write_all(&(bytes.len() as u32).to_be_bytes())?;
    w.write_all(bytes)?;
    w.flush()?;
    Ok(())
}
pub fn read_frame(r: &mut impl Read, limit: u32) -> Result<Vec<u8>> {
    let mut len = [0u8; 4];
    r.read_exact(&mut len)?;
    let len = u32::from_be_bytes(len);
    ensure!(
        len <= limit,
        "frame of {len} bytes exceeds message limit {limit}"
    );
    let mut body = vec![0u8; len as usize];
    r.read_exact(&mut body)?;
    Ok(body)
}

pub fn sign_hello(key: &SigningKey, body: HelloBody) -> Result<Hello> {
    Ok(Hello {
        signature: sign(key, HELLO_DOMAIN, &body)?,
        body,
    })
}
pub fn sign_result(key: &SigningKey, body: ResultBody) -> Result<SignedResult> {
    Ok(SignedResult {
        signature: sign(key, RESULT_DOMAIN, &body)?,
        body,
    })
}

/// Coordinator-side rendezvous for one plan attempt. Fail-closed: the first invalid,
/// forged, stale or substituted message refuses the whole group.
pub struct Rendezvous {
    contract: BoundedRunContract,
    plan_sha256: String,
    closure_sha256: String,
    challenge: Challenge,
    hellos: BTreeMap<u32, (Vec<u8>, Hello)>,
    refused: Option<GroupRefusal>,
    admitted: Option<GroupAdmission>,
}
impl Rendezvous {
    /// `nonce` must be fresh per open; the id also binds the coordinator boot, so a hello
    /// captured before a coordinator restart can never join the new rendezvous.
    pub fn open(
        contract: &BoundedRunContract,
        plan_sha256: &str,
        coordinator: &SigningKey,
        coordinator_boot_id: &str,
        nonce: &[u8; 16],
        now_unix_ms: u64,
    ) -> Result<Self> {
        ensure!(
            hex::encode(coordinator.verifying_key().to_bytes()) == contract.coordinator_public_key,
            "coordinator key is not the planned coordinator"
        );
        ensure!(
            now_unix_ms >= contract.not_before_unix_ms
                && now_unix_ms < contract.original_deadline_unix_ms,
            "rendezvous outside the original run window"
        );
        let rendezvous_id = digest(&(
            "jope.native_rendezvous_id.v1",
            plan_sha256,
            contract.attempt,
            coordinator_boot_id,
            now_unix_ms,
            hex::encode(nonce),
        ))?;
        let grants_expire = contract
            .participants
            .iter()
            .map(|p| p.grant.body.expires_unix_ms)
            .min()
            .unwrap_or(0);
        let body = ChallengeBody {
            schema: CHALLENGE_SCHEMA.into(),
            run: contract.run.clone(),
            attempt: contract.attempt,
            plan_sha256: plan_sha256.into(),
            rendezvous_id,
            coordinator_boot_id: coordinator_boot_id.into(),
            opened_unix_ms: now_unix_ms,
            join_deadline_unix_ms: now_unix_ms
                .saturating_add(contract.rendezvous.join_timeout_ms)
                .min(contract.original_deadline_unix_ms)
                .min(grants_expire),
        };
        Ok(Self {
            closure_sha256: closure_digest(&contract.artifact_closure)?,
            contract: contract.clone(),
            plan_sha256: plan_sha256.into(),
            challenge: Challenge {
                signature: sign(coordinator, CHALLENGE_DOMAIN, &body)?,
                body,
            },
            hellos: BTreeMap::new(),
            refused: None,
            admitted: None,
        })
    }
    pub fn challenge(&self) -> &Challenge {
        &self.challenge
    }
    pub fn rendezvous_id(&self) -> &str {
        &self.challenge.body.rendezvous_id
    }
    pub fn refusal(&self) -> Option<&GroupRefusal> {
        self.refused.as_ref()
    }
    pub fn joined(&self) -> Vec<u32> {
        self.hellos.keys().copied().collect()
    }
    fn refuse(&mut self, cause: RefusalCause) -> GroupRefusal {
        let refusal = self.refused.get_or_insert_with(|| GroupRefusal {
            run: self.contract.run.clone(),
            attempt: self.contract.attempt,
            rendezvous_id: self.challenge.body.rendezvous_id.clone(),
            cause,
            started_ranks: 0,
            committed_updates: 0,
        });
        refusal.clone()
    }
    /// Offer one untrusted hello frame. Returns the rank it authenticated as.
    pub fn offer(&mut self, frame: &[u8], now_unix_ms: u64) -> Result<u32, GroupRefusal> {
        if let Some(r) = &self.refused {
            return Err(r.clone());
        }
        if self.admitted.is_some() {
            return Err(self.refuse(RefusalCause::Malformed {
                detail: "hello after admission".into(),
            }));
        }
        let limit = u64::from(self.contract.rendezvous.max_message_bytes);
        if frame.len() as u64 > limit {
            return Err(self.refuse(RefusalCause::Oversized {
                bytes: frame.len() as u64,
                limit,
            }));
        }
        let hello: Hello = match serde_json::from_slice(frame) {
            Ok(h) => h,
            Err(e) => {
                return Err(self.refuse(RefusalCause::Malformed {
                    detail: e.to_string(),
                }))
            }
        };
        match self.check_hello(&hello, now_unix_ms) {
            Ok(()) => {}
            Err(cause) => return Err(self.refuse(cause)),
        }
        let rank = hello.body.global_rank;
        if let Some((old, _)) = self.hellos.get(&rank) {
            if old.as_slice() == frame {
                return Ok(rank); // identical redelivery is idempotent
            }
            return Err(self.refuse(RefusalCause::Equivocation { rank }));
        }
        self.hellos.insert(rank, (frame.to_vec(), hello));
        Ok(rank)
    }
    fn check_hello(&self, hello: &Hello, now: u64) -> std::result::Result<(), RefusalCause> {
        let b = &hello.body;
        let c = &self.contract;
        let rank = b.global_rank;
        let Some(p) = c.participant(rank) else {
            return Err(RefusalCause::UnknownRank { rank });
        };
        // Authenticate before trusting any other field.
        if verify(&p.public_key, HELLO_DOMAIN, b, &hello.signature).is_err() {
            return Err(RefusalCause::ForgedPeer { claimed_rank: rank });
        }
        if b.schema != HELLO_SCHEMA || b.delivery_id == 0 {
            return Err(RefusalCause::Malformed {
                detail: "hello schema/delivery id".into(),
            });
        }
        if b.run != c.run {
            return Err(RefusalCause::WrongRun);
        }
        if b.attempt != c.attempt {
            return Err(RefusalCause::ReplayedAttempt { attempt: b.attempt });
        }
        if b.plan_sha256 != self.plan_sha256 {
            return Err(RefusalCause::WrongPlan);
        }
        if b.rendezvous_id != self.challenge.body.rendezvous_id {
            return Err(RefusalCause::StaleRendezvous);
        }
        if b.boot_id != p.host.boot_id {
            return Err(RefusalCause::ReplayedBoot { rank });
        }
        if b.host != p.host.host
            || b.machine_id != p.host.machine_id
            || b.device_uuid != p.device.as_ref().map(|d| d.uuid.clone())
        {
            return Err(RefusalCause::WrongParticipant { rank });
        }
        if b.source_commit != p.host.source_commit
            || b.executable_sha256 != p.host.executable_sha256
            || b.runtime != p.host.runtime
            || b.installed_policy_sha256 != p.host.installed_policy_sha256
        {
            return Err(RefusalCause::SubstitutedSource { rank });
        }
        if b.artifact_closure_sha256 != self.closure_sha256 {
            return Err(RefusalCause::SubstitutedInput { rank });
        }
        if digest(&p.grant).ok().as_deref() != Some(b.grant_sha256.as_str()) {
            return Err(RefusalCause::WrongGrant { rank });
        }
        let skew = c.telemetry.clock_skew_ms;
        if b.sent_unix_ms.saturating_add(skew) < self.challenge.body.opened_unix_ms
            || b.sent_unix_ms > now.saturating_add(skew)
        {
            return Err(RefusalCause::StaleMessage { rank });
        }
        if now >= c.original_deadline_unix_ms {
            return Err(RefusalCause::DeadlineElapsed);
        }
        if now >= p.grant.body.expires_unix_ms || now < p.grant.body.not_before_unix_ms {
            return Err(RefusalCause::GrantExpired { rank });
        }
        if now > self.challenge.body.join_deadline_unix_ms {
            return Err(RefusalCause::JoinTimeout {
                missing: self.missing(),
            });
        }
        let bootstrap_ok = match (&b.communicator_bootstrap, rank) {
            (Some(x), 0) => {
                x.len() <= 2 * MAX_BOOTSTRAP_BYTES && !x.is_empty() && hex::decode(x).is_ok()
            }
            (None, r) => r != 0,
            _ => false,
        };
        if !bootstrap_ok {
            return Err(RefusalCause::Malformed {
                detail: "communicator bootstrap is carried by rank 0 only".into(),
            });
        }
        Ok(())
    }
    fn missing(&self) -> Vec<u32> {
        (0..self.contract.world_size() as u32)
            .filter(|r| !self.hellos.contains_key(r))
            .collect()
    }
    /// `Ok(None)` while waiting inside the join window. Admission requires every planned
    /// rank, unexpired grants and the original deadline; after the join deadline a
    /// partial group is refused as a whole.
    pub fn admit(
        &mut self,
        coordinator: &SigningKey,
        now_unix_ms: u64,
    ) -> Result<Option<GroupAdmission>, GroupRefusal> {
        if let Some(r) = &self.refused {
            return Err(r.clone());
        }
        if let Some(a) = &self.admitted {
            return Ok(Some(a.clone()));
        }
        let missing = self.missing();
        if !missing.is_empty() {
            if now_unix_ms > self.challenge.body.join_deadline_unix_ms {
                return Err(self.refuse(RefusalCause::JoinTimeout { missing }));
            }
            return Ok(None);
        }
        if now_unix_ms >= self.contract.original_deadline_unix_ms {
            return Err(self.refuse(RefusalCause::DeadlineElapsed));
        }
        for p in &self.contract.participants {
            if now_unix_ms >= p.grant.body.expires_unix_ms {
                return Err(self.refuse(RefusalCause::GrantExpired {
                    rank: p.global_rank,
                }));
            }
        }
        let grants_sha256 = {
            let grants: Vec<_> = self
                .contract
                .participants
                .iter()
                .map(|p| &p.grant)
                .collect();
            digest(&grants)
        };
        let grants_sha256 = match grants_sha256 {
            Ok(g) => g,
            Err(e) => {
                return Err(self.refuse(RefusalCause::Malformed {
                    detail: e.to_string(),
                }))
            }
        };
        let members = self
            .hellos
            .iter()
            .map(|(rank, (bytes, h))| Member {
                global_rank: *rank,
                hello_sha256: crate::manifest::hash(bytes),
                host: h.body.host.clone(),
                boot_id: h.body.boot_id.clone(),
                device_uuid: h.body.device_uuid.clone(),
            })
            .collect();
        let body = AdmissionBody {
            schema: ADMISSION_SCHEMA.into(),
            run: self.contract.run.clone(),
            attempt: self.contract.attempt,
            plan_sha256: self.plan_sha256.clone(),
            rendezvous_id: self.challenge.body.rendezvous_id.clone(),
            admitted_unix_ms: now_unix_ms,
            original_deadline_unix_ms: self.contract.original_deadline_unix_ms,
            grants_sha256,
            members,
            communicator_bootstrap: self.hellos[&0]
                .1
                .body
                .communicator_bootstrap
                .clone()
                .unwrap_or_default(),
        };
        let signature = sign(coordinator, ADMISSION_DOMAIN, &body).map_err(|e| {
            self.refuse(RefusalCause::Malformed {
                detail: e.to_string(),
            })
        })?;
        let admission = GroupAdmission { body, signature };
        self.admitted = Some(admission.clone());
        Ok(Some(admission))
    }
}

/// Participant-side check of the coordinator's signed challenge.
pub fn verify_challenge(
    contract: &BoundedRunContract,
    plan_sha256: &str,
    frame: &[u8],
    now_unix_ms: u64,
) -> Result<Challenge, RefusalCause> {
    if frame.len() as u64 > u64::from(contract.rendezvous.max_message_bytes) {
        return Err(RefusalCause::Oversized {
            bytes: frame.len() as u64,
            limit: u64::from(contract.rendezvous.max_message_bytes),
        });
    }
    let c: Challenge = serde_json::from_slice(frame).map_err(|e| RefusalCause::Malformed {
        detail: e.to_string(),
    })?;
    verify(
        &contract.coordinator_public_key,
        CHALLENGE_DOMAIN,
        &c.body,
        &c.signature,
    )
    .map_err(|_| RefusalCause::ForgedCoordinator)?;
    let b = &c.body;
    if b.schema != CHALLENGE_SCHEMA || !is_digest(&b.rendezvous_id) {
        return Err(RefusalCause::Malformed {
            detail: "challenge schema".into(),
        });
    }
    if b.run != contract.run {
        return Err(RefusalCause::WrongRun);
    }
    if b.attempt != contract.attempt {
        return Err(RefusalCause::ReplayedAttempt { attempt: b.attempt });
    }
    if b.plan_sha256 != plan_sha256 {
        return Err(RefusalCause::WrongPlan);
    }
    if now_unix_ms > b.join_deadline_unix_ms
        || b.join_deadline_unix_ms > contract.original_deadline_unix_ms
        || b.opened_unix_ms > now_unix_ms.saturating_add(contract.telemetry.clock_skew_ms)
    {
        return Err(RefusalCause::StaleRendezvous);
    }
    Ok(c)
}

/// Proof, held by one participant, that the whole planned group was admitted. Only
/// [`verify_admission`] constructs it; a rank must hold one before starting any work.
#[derive(Clone, Debug)]
pub struct AdmittedMember {
    admission: GroupAdmission,
    admission_sha256: String,
    global_rank: u32,
    world_size: u32,
    bootstrap: Vec<u8>,
}
impl AdmittedMember {
    pub fn admission(&self) -> &GroupAdmission {
        &self.admission
    }
    pub fn admission_sha256(&self) -> &str {
        &self.admission_sha256
    }
    pub fn global_rank(&self) -> u32 {
        self.global_rank
    }
    pub fn world_size(&self) -> u32 {
        self.world_size
    }
    /// The communicator bootstrap (join point for the trainer's communicator creation).
    pub fn communicator_bootstrap(&self) -> &[u8] {
        &self.bootstrap
    }
}
pub fn admission_sha256(admission: &GroupAdmission) -> Result<String> {
    digest(admission)
}
/// Identity under which a rank's work is dispatched (one per plan attempt and rank); also
/// how a restarted coordinator finds work whose launch it never journaled.
pub fn dispatch_id(plan_sha256: &str, attempt: u32, global_rank: u32) -> Result<String> {
    digest(&(
        "jope.native_dispatch_id.v1",
        plan_sha256,
        attempt,
        global_rank,
    ))
}
/// Fresh rendezvous nonce from the kernel.
pub fn fresh_nonce() -> Result<[u8; 16]> {
    let mut nonce = [0u8; 16];
    fs::File::open("/dev/urandom")?.read_exact(&mut nonce)?;
    Ok(nonce)
}
/// Controller-side check that a group admission is the coordinator's own full-group
/// admission for this plan (used when persisting and when verifying the journal).
pub fn verify_group_admission(
    contract: &BoundedRunContract,
    plan_sha256: &str,
    admission: &GroupAdmission,
) -> Result<()> {
    verify(
        &contract.coordinator_public_key,
        ADMISSION_DOMAIN,
        &admission.body,
        &admission.signature,
    )
    .context("group admission is not signed by the planned coordinator")?;
    let b = &admission.body;
    let grants: Vec<_> = contract.participants.iter().map(|p| &p.grant).collect();
    ensure!(
        b.schema == ADMISSION_SCHEMA
            && b.run == contract.run
            && b.attempt == contract.attempt
            && b.plan_sha256 == plan_sha256
            && b.original_deadline_unix_ms == contract.original_deadline_unix_ms
            && b.grants_sha256 == digest(&grants)?
            && b.members.len() == contract.world_size()
            && b.members
                .iter()
                .zip(&contract.participants)
                .all(|(m, p)| m.global_rank == p.global_rank
                    && m.host == p.host.host
                    && m.boot_id == p.host.boot_id
                    && m.device_uuid == p.device.as_ref().map(|d| d.uuid.clone())
                    && is_digest(&m.hello_sha256)),
        "group admission does not bind the full planned group"
    );
    Ok(())
}

/// Participant-side verification of a full-group admission against its own hello.
pub fn verify_admission(
    contract: &BoundedRunContract,
    plan_sha256: &str,
    own_hello: &Hello,
    frame: &[u8],
    now_unix_ms: u64,
) -> Result<AdmittedMember, RefusalCause> {
    if frame.len() as u64 > u64::from(contract.rendezvous.max_message_bytes) {
        return Err(RefusalCause::Oversized {
            bytes: frame.len() as u64,
            limit: u64::from(contract.rendezvous.max_message_bytes),
        });
    }
    let a: GroupAdmission = serde_json::from_slice(frame).map_err(|e| RefusalCause::Malformed {
        detail: e.to_string(),
    })?;
    verify(
        &contract.coordinator_public_key,
        ADMISSION_DOMAIN,
        &a.body,
        &a.signature,
    )
    .map_err(|_| RefusalCause::ForgedCoordinator)?;
    let b = &a.body;
    let rank = own_hello.body.global_rank;
    if b.schema != ADMISSION_SCHEMA || b.run != contract.run {
        return Err(RefusalCause::WrongRun);
    }
    if b.attempt != contract.attempt {
        return Err(RefusalCause::ReplayedAttempt { attempt: b.attempt });
    }
    if b.plan_sha256 != plan_sha256 {
        return Err(RefusalCause::WrongPlan);
    }
    if b.rendezvous_id != own_hello.body.rendezvous_id {
        return Err(RefusalCause::StaleRendezvous);
    }
    if b.original_deadline_unix_ms != contract.original_deadline_unix_ms
        || now_unix_ms >= contract.original_deadline_unix_ms
    {
        return Err(RefusalCause::DeadlineElapsed);
    }
    let grants: Vec<_> = contract.participants.iter().map(|p| &p.grant).collect();
    if digest(&grants).ok().as_deref() != Some(b.grants_sha256.as_str()) {
        return Err(RefusalCause::WrongGrant { rank });
    }
    let Some(p) = contract.participant(rank) else {
        return Err(RefusalCause::UnknownRank { rank });
    };
    if now_unix_ms >= p.grant.body.expires_unix_ms {
        return Err(RefusalCause::GrantExpired { rank });
    }
    let ranks: Vec<u32> = b.members.iter().map(|m| m.global_rank).collect();
    if !ranks.iter().copied().eq(0..contract.world_size() as u32) {
        return Err(RefusalCause::IncompleteGroup);
    }
    for (m, planned) in b.members.iter().zip(&contract.participants) {
        if m.host != planned.host.host
            || m.boot_id != planned.host.boot_id
            || m.device_uuid != planned.device.as_ref().map(|d| d.uuid.clone())
        {
            return Err(RefusalCause::WrongParticipant {
                rank: m.global_rank,
            });
        }
    }
    let own = serde_json::to_vec(own_hello).map_err(|e| RefusalCause::Malformed {
        detail: e.to_string(),
    })?;
    if b.members[rank as usize].hello_sha256 != crate::manifest::hash(&own) {
        return Err(RefusalCause::Equivocation { rank });
    }
    let bootstrap =
        hex::decode(&b.communicator_bootstrap).map_err(|e| RefusalCause::Malformed {
            detail: e.to_string(),
        })?;
    if bootstrap.is_empty() || bootstrap.len() > MAX_BOOTSTRAP_BYTES {
        return Err(RefusalCause::Malformed {
            detail: "communicator bootstrap".into(),
        });
    }
    Ok(AdmittedMember {
        admission_sha256: admission_sha256(&a).map_err(|e| RefusalCause::Malformed {
            detail: e.to_string(),
        })?,
        admission: a,
        global_rank: rank,
        world_size: contract.world_size() as u32,
        bootstrap,
    })
}

/// A result whose envelope and bytes authenticated against the admitted plan.
/// Only [`verify_result`] constructs it.
#[derive(Clone, Debug)]
pub struct VerifiedResult {
    envelope: Vec<u8>,
    body: ResultBody,
    bytes: Vec<u8>,
}
impl VerifiedResult {
    pub fn body(&self) -> &ResultBody {
        &self.body
    }
    pub fn envelope(&self) -> &[u8] {
        &self.envelope
    }
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}
/// Authenticate one rank's retained result. Refuses oversize envelopes, forged or
/// wrong-key signatures, other runs/attempts/plans/admissions, bytes that differ from the
/// signed digest, results larger than the planned destination and any update count
/// other than the synchronous group's planned count.
pub fn verify_result(
    contract: &BoundedRunContract,
    plan_sha256: &str,
    admission_sha256: &str,
    envelope: &[u8],
    bytes: &[u8],
) -> Result<VerifiedResult> {
    ensure!(
        envelope.len() as u64 <= u64::from(contract.rendezvous.max_message_bytes),
        "result envelope exceeds message limit"
    );
    let signed: SignedResult = serde_json::from_slice(envelope)?;
    let b = &signed.body;
    let p = contract
        .participant(b.global_rank)
        .context("result names no planned participant")?;
    verify(&p.public_key, RESULT_DOMAIN, b, &signed.signature)
        .context("result is not signed by the planned participant key")?;
    ensure!(
        b.schema == RESULT_SCHEMA
            && b.run == contract.run
            && b.attempt == contract.attempt
            && b.plan_sha256 == plan_sha256
            && b.admission_sha256 == admission_sha256
            && b.delivery_id > 0,
        "result is bound to a different run, attempt, plan or admission"
    );
    ensure!(
        b.result_bytes <= p.result_destination.max_bytes
            && bytes.len() as u64 == b.result_bytes
            && crate::manifest::hash(bytes) == b.result_sha256,
        "result bytes differ from the authenticated digest or planned destination bound"
    );
    ensure!(
        b.updates == contract.counters.planned_updates,
        "result update count differs from the synchronous plan"
    );
    Ok(VerifiedResult {
        envelope: envelope.to_vec(),
        body: signed.body,
        bytes: bytes.to_vec(),
    })
}

/// A connected, authenticated participant control stream (same-host fixture transport).
pub struct Connected {
    pub global_rank: u32,
    pub stream: UnixStream,
}
/// Accept and authenticate participants on a Unix socket until the group is admitted or
/// refused. Each handshake is bounded by the join deadline and the frozen message limit.
/// Admission is returned *undelivered*: only the controller delivers it, after journaling.
pub fn gather_unix(
    listener: &UnixListener,
    rendezvous: &mut Rendezvous,
    coordinator: &SigningKey,
    now: &dyn Fn() -> u64,
) -> Result<(GroupAdmission, Vec<Connected>), GroupRefusal> {
    let limit = rendezvous.contract.rendezvous.max_message_bytes;
    let challenge = serde_json::to_vec(rendezvous.challenge()).unwrap_or_default();
    let mut connected = Vec::new();
    let _ = listener.set_nonblocking(true);
    loop {
        if let Some(admission) = rendezvous.admit(coordinator, now())? {
            return Ok((admission, connected));
        }
        match listener.accept() {
            Ok((mut stream, _)) => {
                let remaining = rendezvous
                    .challenge
                    .body
                    .join_deadline_unix_ms
                    .saturating_sub(now())
                    .max(1);
                let _ = stream.set_nonblocking(false);
                let _ = stream.set_read_timeout(Some(Duration::from_millis(remaining)));
                let _ = stream.set_write_timeout(Some(Duration::from_millis(remaining)));
                if write_frame(&mut stream, &challenge, limit).is_err() {
                    continue; // the peer went away before authenticating; nothing trusted
                }
                let frame = match read_frame(&mut stream, limit) {
                    Ok(f) => f,
                    Err(e) => {
                        let cause = if e.to_string().contains("exceeds message limit") {
                            RefusalCause::Oversized {
                                bytes: u64::from(limit) + 1,
                                limit: u64::from(limit),
                            }
                        } else {
                            // An unauthenticated peer that timed out or closed early
                            // proves nothing; the join deadline still governs the group.
                            continue;
                        };
                        return Err(rendezvous.refuse(cause));
                    }
                };
                let rank = rendezvous.offer(&frame, now())?;
                connected.retain(|c: &Connected| c.global_rank != rank);
                connected.push(Connected {
                    global_rank: rank,
                    stream,
                });
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(_) => std::thread::sleep(Duration::from_millis(10)),
        }
    }
}
/// Participant side of the same-host fixture transport: read and verify the challenge,
/// send the signed hello built by `hello`, then verify the full-group admission.
pub fn join_unix(
    stream: &mut UnixStream,
    contract: &BoundedRunContract,
    plan_sha256: &str,
    key: &SigningKey,
    hello: impl FnOnce(&Challenge) -> HelloBody,
    now: &dyn Fn() -> u64,
) -> Result<AdmittedMember> {
    let limit = contract.rendezvous.max_message_bytes;
    let remaining = contract
        .rendezvous
        .join_timeout_ms
        .min(contract.original_deadline_unix_ms.saturating_sub(now()))
        .max(1);
    stream.set_read_timeout(Some(Duration::from_millis(remaining)))?;
    let frame = read_frame(stream, limit)?;
    let challenge = verify_challenge(contract, plan_sha256, &frame, now())
        .map_err(|c| anyhow::anyhow!("challenge refused: {c:?}"))?;
    let signed = sign_hello(key, hello(&challenge))?;
    write_frame(stream, &serde_json::to_vec(&signed)?, limit)?;
    let frame = read_frame(stream, limit)?;
    verify_admission(contract, plan_sha256, &signed, &frame, now())
        .map_err(|c| anyhow::anyhow!("admission refused: {c:?}"))
}
