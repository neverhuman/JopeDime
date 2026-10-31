//! Independent, signed custody of full original fit-authorization events.
//!
//! The ACK key is held only by the witness. The authority key cannot produce an
//! ACK. Signed immutable segments are the durable record; the index is rebuilt
//! from verified segments. Deployment and namespace activation are separate.
pub use super::native_scq::policy::{NativeGrant, NativeWitnessTrust, SignedNativeGrant};
mod epochs;
mod native_grants;
use anyhow::{ensure, Context, Result};
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
pub use epochs::{
    serve_stdio_epoch_transition, validate_native_grant_request_for_epoch, verify_epoch_chain,
    EpochTransitionMode, HistoricalBudgetAccounting, SignedWitnessEpochTransition,
    SignedWitnessEpochTransitionAck, WitnessEpochProof, WitnessEpochTransitionAckBody,
    WitnessEpochTransitionBody,
};
pub use native_grants::{
    serve_stdio_add_native_grant, verify_native_grant_proof, AddNativeGrantAckBody,
    AddNativeGrantBody, NativeGrantArtifact, NativeGrantDelegation, NativeGrantProof,
    NativeGrantRole, SignedAddNativeGrant, SignedAddNativeGrantAck, ADD_NATIVE_GRANT_DOMAIN,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::unix::{
        fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
        process::CommandExt,
    },
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::Duration,
};

const ZERO: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const MAX_FRAME: usize = 4 * 1024 * 1024;
fn sha(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
fn bytes(value: &impl Serialize) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(value)?)
}
fn valid_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn domain(domain: &str, value: &impl Serialize) -> Result<Vec<u8>> {
    let mut out = domain.as_bytes().to_vec();
    out.push(0);
    out.extend(bytes(value)?);
    Ok(out)
}
fn key(value: &str) -> Result<VerifyingKey> {
    let raw: [u8; 32] = hex::decode(value)?
        .try_into()
        .map_err(|_| anyhow::anyhow!("invalid public key width"))?;
    let key = VerifyingKey::from_bytes(&raw)?;
    ensure!(!key.is_weak(), "weak witness public key");
    Ok(key)
}
fn verify(public: &str, tag: &str, value: &impl Serialize, signature: &str) -> Result<()> {
    let signature = Signature::from_slice(&hex::decode(signature)?)?;
    key(public)?.verify_strict(&domain(tag, value)?, &signature)?;
    Ok(())
}
fn sign(key: &SigningKey, tag: &str, value: &impl Serialize) -> Result<String> {
    Ok(hex::encode(key.sign(&domain(tag, value)?).to_bytes()))
}
fn checked_file(path: &Path) -> Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK)
        .open(path)?;
    ensure!(file.metadata()?.is_file(), "regular witness file required");
    Ok(file)
}
fn private_key(path: &Path) -> Result<SigningKey> {
    let mut file = checked_file(path)?;
    let meta = file.metadata()?;
    ensure!(
        meta.is_file() && meta.uid() == crate::host_os::effective_uid() && meta.mode() & 0o077 == 0,
        "witness private key must be owned machine-local mode 0600"
    );
    let mut value = Vec::new();
    Read::by_ref(&mut file).take(33).read_to_end(&mut value)?;
    let value: [u8; 32] = value
        .try_into()
        .map_err(|_| anyhow::anyhow!("private key must contain 32 raw bytes"))?;
    Ok(SigningKey::from_bytes(&value))
}
fn read(path: &Path) -> Result<Vec<u8>> {
    let mut file = checked_file(path)?;
    ensure!(file.metadata()?.is_file(), "regular witness file required");
    let mut result = Vec::new();
    Read::by_ref(&mut file)
        .take((MAX_FRAME + 1) as u64)
        .read_to_end(&mut result)?;
    ensure!(result.len() <= MAX_FRAME, "witness file exceeds bound");
    Ok(result)
}
fn sync_dir(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}
fn publish(path: &Path, data: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)?;
    file.write_all(data)?;
    file.sync_all()?;
    sync_dir(path.parent().context("witness parent absent")?)
}
pub fn source_sha256() -> String {
    source_set_digest(&[
        ("witness.rs", include_bytes!("witness.rs")),
        ("witness/epochs.rs", include_bytes!("witness/epochs.rs")),
        ("native_review.rs", include_bytes!("../native_review.rs")),
        (
            "witness/native_grants.rs",
            include_bytes!("witness/native_grants.rs"),
        ),
        (
            "native_persistence.rs",
            include_bytes!("../native_persistence.rs"),
        ),
        ("native_scq.rs", include_bytes!("native_scq.rs")),
        (
            "native_scq/policy.rs",
            include_bytes!("native_scq/policy.rs"),
        ),
        ("funding.rs", include_bytes!("funding.rs")),
    ])
}
pub fn authority_source_sha256() -> String {
    source_set_digest(&[
        ("witness.rs", include_bytes!("witness.rs")),
        ("witness/epochs.rs", include_bytes!("witness/epochs.rs")),
        (
            "witness/native_grants.rs",
            include_bytes!("witness/native_grants.rs"),
        ),
        (
            "native_persistence.rs",
            include_bytes!("../native_persistence.rs"),
        ),
        ("authority.rs", include_bytes!("authority.rs")),
        (
            "authority/history_aggregate.rs",
            include_bytes!("authority/history_aggregate.rs"),
        ),
        (
            "authority/native_scopes.rs",
            include_bytes!("authority/native_scopes.rs"),
        ),
        (
            "authority/native_recovery.rs",
            include_bytes!("authority/native_recovery.rs"),
        ),
        ("native_review.rs", include_bytes!("../native_review.rs")),
        ("native_scq.rs", include_bytes!("native_scq.rs")),
        (
            "native_scq/policy.rs",
            include_bytes!("native_scq/policy.rs"),
        ),
        (
            "service/native_scq.rs",
            include_bytes!("service/native_scq.rs"),
        ),
        (
            "authority/funding.rs",
            include_bytes!("authority/funding.rs"),
        ),
        ("funding.rs", include_bytes!("funding.rs")),
        ("funding_config.rs", include_bytes!("funding_config.rs")),
        ("service.rs", include_bytes!("service.rs")),
        (
            "service/funding_ipc.rs",
            include_bytes!("service/funding_ipc.rs"),
        ),
        ("query.rs", include_bytes!("query.rs")),
        (
            "service/query_worker.rs",
            include_bytes!("service/query_worker.rs"),
        ),
        (
            "runtime_custody.rs",
            include_bytes!("../runtime_custody.rs"),
        ),
    ])
}
fn source_set_digest(sources: &[(&str, &[u8])]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"parallel.authority_component_sources.v2\0");
    for (path, bytes) in sources {
        digest.update((path.len() as u64).to_le_bytes());
        digest.update(path.as_bytes());
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
    }
    hex::encode(digest.finalize())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WitnessPolicy {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_grant_delegation: Option<NativeGrantDelegation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_fit_trust: Option<NativeWitnessTrust>,
    pub schema: String,
    pub witness_instance: String,
    pub authority_instance: String,
    pub authority_epoch: String,
    pub authority_public_key: String,
    pub witness_public_key: String,
    pub authority_executable_sha256: String,
    pub authority_source_sha256: String,
    pub witness_executable_sha256: String,
    pub witness_source_sha256: String,
    /// Hash of the exact dedicated OpenSSH known-hosts file. No TOFU fallback.
    pub ssh_known_hosts_sha256: String,
    pub ssh_host_alias: String,
    pub ssh_destination: String,
    /// Absent in retained fit-only policies. A funding event cannot choose its
    /// own trust: these keys and exact authorization are pinned in the policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub funding_trust: Option<super::funding::FundingTrust>,
}
impl WitnessPolicy {
    pub fn sha256(&self) -> Result<String> {
        Ok(sha(&bytes(self)?))
    }
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == "parallel.fit_event_witness_policy.v1",
            "unsupported witness policy"
        );
        ensure!(
            [
                &self.witness_instance,
                &self.authority_instance,
                &self.authority_epoch,
                &self.authority_executable_sha256,
                &self.authority_source_sha256,
                &self.witness_executable_sha256,
                &self.witness_source_sha256,
                &self.ssh_known_hosts_sha256
            ]
            .into_iter()
            .all(|v| valid_hash(v)),
            "invalid witness policy identity"
        );
        key(&self.authority_public_key)?;
        key(&self.witness_public_key)?;
        ensure!(
            self.authority_public_key != self.witness_public_key,
            "authority and ACK keys must be independent"
        );
        if let Some(trust) = &self.funding_trust {
            ensure!(
                trust.authority_instance == self.authority_instance
                    && trust.authority_epoch == self.authority_epoch
                    && trust.authority_executable_sha256 == self.authority_executable_sha256,
                "funding witness trust differs from pinned authority"
            );
        }
        ensure!(
            !self.ssh_host_alias.is_empty()
                && self
                    .ssh_host_alias
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c))
                && !self.ssh_destination.is_empty()
                && !self.ssh_destination.starts_with('-')
                && self
                    .ssh_destination
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"@._-".contains(&c)),
            "invalid pinned SSH host binding"
        );
        if let Some(delegation) = &self.native_grant_delegation {
            delegation.validate(self)?;
        }
        Ok(())
    }
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestBody {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    epoch_proofs: Vec<WitnessEpochProof>,
    schema: String,
    policy_sha256: String,
    event: Vec<u8>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedRequest {
    body: RequestBody,
    signature: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AckBody {
    pub schema: String,
    pub policy_sha256: String,
    pub witness_instance: String,
    pub authority_instance: String,
    pub authority_epoch: String,
    pub authority_sequence: u64,
    pub event_sha256: String,
    pub request_sha256: String,
    pub witness_sequence: u64,
    pub previous_record_sha256: String,
    pub committed_record_core_sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SignedAck {
    pub body: AckBody,
    pub signature: String,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RecordCore {
    schema: String,
    sequence: u64,
    previous: String,
    request: SignedRequest,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    core: RecordCore,
    ack: SignedAck,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Head {
    sequence: u64,
    sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OriginalEvent {
    instance: String,
    seq: u64,
    previous: String,
    kind: String,
    payload: Value,
    effects: Vec<OriginalEffect>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OriginalEffect {
    table: String,
    key: String,
    row: Value,
}
struct EventIdentity {
    seq: u64,
    deadline: u64,
}
#[cfg(test)]
fn event_identity(policy: &WitnessPolicy, event: &[u8]) -> Result<EventIdentity> {
    event_identity_effective(policy, policy, event)
}
fn event_identity_effective(
    base: &WitnessPolicy,
    policy: &WitnessPolicy,
    event: &[u8],
) -> Result<EventIdentity> {
    ensure!(
        !event.is_empty() && event.len() < MAX_FRAME / 2,
        "invalid authorization event size"
    );
    let value: OriginalEvent = serde_json::from_slice(event)?;
    ensure!(
        value.instance == policy.authority_instance
            && value.seq > 0
            && valid_hash(&value.previous)
            && matches!(
                value.kind.as_str(),
                "authorize" | "funding_start" | "funding_participant"
            )
            && value.effects.len() == 1,
        "not an original authorization event"
    );
    let effect = &value.effects[0];
    let row = &effect.row;
    if value.kind != "authorize" {
        let trust = policy
            .funding_trust
            .as_ref()
            .context("funding witness trust not installed")?;
        let original: super::funding::FundingEvent =
            serde_json::from_value(value.payload["original"].clone())?;
        let (event_key, deadline) = super::funding::validate_witness_event(
            trust,
            &original,
            &policy.authority_instance,
            &policy.authority_epoch,
        )?;
        let admission_deadline = value.payload["admission_deadline_monotonic_ns"]
            .as_u64()
            .context("original funding admission deadline absent")?;
        ensure!(
            original.claimed_monotonic_ns < admission_deadline && admission_deadline <= deadline,
            "funding admission deadline is outside original task window"
        );
        ensure!(
            value.payload["event_key"] == event_key
                && value.payload["reservation_id"] == original.reservation_id(),
            "funding journal key mismatch"
        );
        match original.participant_slot {
            None => {
                let reservation: super::funding::SignedReservation = serde_json::from_str(
                    row["json"].as_str().context("signed reservation absent")?,
                )?;
                let stored: super::funding::FundingEvent = serde_json::from_str(
                    row["start_json"]
                        .as_str()
                        .context("original funding start absent")?,
                )?;
                ensure!(
                    value.kind == "funding_start"
                        && effect.table == "funding_reservations"
                        && effect.key == original.reservation_id()
                        && row["id"] == effect.key
                        && row["state"] == "started"
                        && row["task_sha256"] == original.demand.task_sha256
                        && reservation == original.reservation
                        && row["authorization_id"]
                            == super::funding::digest(&original.authorization.body)?
                        && stored == original,
                    "funding start postimage differs from original"
                );
            }
            Some(slot) => {
                let id = original.participant_id()?;
                let stored: super::funding::FundingEvent = serde_json::from_str(
                    row["json"]
                        .as_str()
                        .context("original funding participant absent")?,
                )?;
                ensure!(
                    value.kind == "funding_participant"
                        && effect.table == "funding_participants"
                        && effect.key == id
                        && row["id"] == id
                        && value.payload["participant_id"] == id
                        && row["reservation_id"] == original.reservation_id()
                        && row["slot"].as_str() == Some(slot.to_string().as_str())
                        && stored == original,
                    "funding participant postimage differs from original"
                );
            }
        }
        return Ok(EventIdentity {
            seq: value.seq,
            deadline: admission_deadline,
        });
    }
    ensure!(
        effect.table == "entries"
            && row["key"] == effect.key
            && row["state"] == "authorized"
            && row["epoch"] == policy.authority_epoch,
        "authorization epoch/state mismatch"
    );
    let start: super::StartToken = serde_json::from_str(
        row["start_json"]
            .as_str()
            .context("original full start absent")?,
    )?;
    let identity: super::FitIdentity = serde_json::from_str(
        row["identity_json"]
            .as_str()
            .context("original fit identity absent")?,
    )?;
    if let Some(native) = value.payload.get("native_fit") {
        let delegated;
        let trust = if let Some(value) = native.get("native_grant_proof").filter(|v| !v.is_null()) {
            let proof: NativeGrantProof = serde_json::from_value(value.clone())?;
            delegated = verify_native_grant_proof(base, &proof)?;
            ensure!(
                verify_epoch_chain(base, &proof.epoch_proofs)?.sha256()? == policy.sha256()?,
                "native grant epoch differs from event epoch"
            );
            ensure!(
                native["signed_grant"] == serde_json::to_value(&proof.request.body.signed_grant)?,
                "native event grant differs from witnessed delegation"
            );
            &delegated
        } else {
            policy
                .native_fit_trust
                .as_ref()
                .context("native witness trust unavailable")?
        };
        super::native_scq::verify_witness_event(trust, native, &identity, &start, policy)?;
    }
    ensure!(
        start.fit_key == effect.key
            && sha(&bytes(&identity)?) == effect.key
            && identity.executable_sha256 == policy.authority_executable_sha256
            && value.payload["fit_key"] == effect.key
            && value.payload["token"] == start.token
            && value.payload["executor"] == serde_json::to_value(&start.executor)?
            && value.payload["claim_token"] == row["claim_token"]
            && start.task_id == row["task_id"]
            && start.deadline_monotonic_ns > start.authorized_monotonic_ns,
        "authorization original identity/token/custody mismatch"
    );
    Ok(EventIdentity {
        seq: value.seq,
        deadline: start.deadline_monotonic_ns,
    })
}
fn verify_request(base: &WitnessPolicy, request: &SignedRequest) -> Result<EventIdentity> {
    let effective = verify_epoch_chain(base, &request.body.epoch_proofs)?;
    let policy = &effective;
    ensure!(
        request.body.schema == "parallel.fit_event_witness_request.v1"
            && request.body.policy_sha256 == policy.sha256()?,
        "witness request policy mismatch"
    );
    verify(
        &policy.authority_public_key,
        "parallel.fit_event_witness.request.v1",
        &request.body,
        &request.signature,
    )?;
    event_identity_effective(base, policy, &request.body.event)
}
impl SignedAck {
    pub fn verify(&self, policy: &WitnessPolicy, request: &SignedRequest) -> Result<()> {
        let event = verify_request(policy, request)?;
        let base = policy;
        let effective = verify_epoch_chain(base, &request.body.epoch_proofs)?;
        let policy = &effective;
        ensure!(
            self.body.schema == "parallel.fit_event_witness_ack.v1"
                && self.body.policy_sha256 == policy.sha256()?
                && self.body.witness_instance == policy.witness_instance
                && self.body.authority_instance == policy.authority_instance
                && self.body.authority_epoch == policy.authority_epoch
                && self.body.authority_sequence == event.seq
                && self.body.event_sha256 == sha(&request.body.event)
                && self.body.request_sha256 == sha(&bytes(request)?)
                && self.body.witness_sequence > 0
                && valid_hash(&self.body.previous_record_sha256)
                && valid_hash(&self.body.committed_record_core_sha256),
            "witness ACK does not bind exact original authorization"
        );
        let original: OriginalEvent = serde_json::from_slice(&request.body.event)?;
        if let Some(value) = original
            .payload
            .get("native_fit")
            .and_then(|v| v.get("native_grant_proof"))
            .filter(|v| !v.is_null())
        {
            let proof: NativeGrantProof = serde_json::from_value(value.clone())?;
            verify_native_grant_proof(base, &proof)?;
            ensure!(
                proof.ack.body.witness_sequence < self.body.witness_sequence,
                "native grant update does not precede authorization ACK"
            );
        }
        verify(
            &policy.witness_public_key,
            "parallel.fit_event_witness.ack.v1",
            &self.body,
            &self.signature,
        )
    }
}

/// Storage stays outside the authority database restore domain. No startup
/// creates storage, repairs missing committed tails, or erases unknown files.
pub struct WitnessStore {
    root: PathBuf,
    policy: WitnessPolicy,
    signing: Option<SigningKey>,
    _lease: File,
    records: BTreeMap<u64, (String, SignedAck, String)>,
    native_grants: native_grants::NativeGrantState,
    epoch_proofs: Vec<WitnessEpochProof>,
    sequence: u64,
    tip: String,
    root_identity: (u64, u64),
    segment_identity: (u64, u64),
    lease_identity: (u64, u64),
    read_only: bool,
}
impl WitnessStore {
    /// Verify a frozen public copy without access to an ACK key or the original executable.
    /// Original signed records, exact head and immutable policy remain mandatory.
    pub fn verify_public_history(root: &Path, policy: &WitnessPolicy) -> Result<Value> {
        Self::verify_public_history_with_acks(root, policy, &[])
    }
    /// Exact membership; extra committed witness-before-local-ACK events remain.
    pub fn verify_public_history_with_acks(
        root: &Path,
        policy: &WitnessPolicy,
        required: &[SignedAck],
    ) -> Result<Value> {
        ensure!(
            required.len() <= 100_000 && bytes(&required)?.len() <= 64 * 1024 * 1024,
            "public witness membership bound"
        );
        policy.validate()?;
        ensure!(
            root.is_absolute() && fs::canonicalize(root)? == root,
            "public witness snapshot must be physical"
        );
        ensure!(
            read(&root.join("policy.json"))? == bytes(policy)?,
            "public witness original policy changed"
        );
        let lease = checked_file(&root.join("lease"))?;
        ensure!(
            lease.try_lock_shared().is_ok(),
            "public witness snapshot still has an active writer"
        );
        let identity = |p: &Path| -> Result<(u64, u64)> {
            let m = fs::symlink_metadata(p)?;
            ensure!(
                !m.file_type().is_symlink(),
                "public witness snapshot symlink"
            );
            Ok((m.dev(), m.ino()))
        };
        let head = read(&root.join("head.json"))?;
        let mut store = Self {
            root: root.into(),
            policy: policy.clone(),
            signing: None,
            _lease: lease,
            records: BTreeMap::new(),
            native_grants: native_grants::NativeGrantState::default(),
            epoch_proofs: Vec::new(),
            sequence: 0,
            tip: ZERO.into(),
            root_identity: identity(root)?,
            segment_identity: identity(&root.join("segments"))?,
            lease_identity: identity(&root.join("lease"))?,
            read_only: true,
        };
        store.replay()?;
        store.verify_custody()?;
        let expected: Head = serde_json::from_slice(&head)?;
        ensure!(
            expected.sequence == store.sequence
                && expected.sha256 == store.tip
                && read(&root.join("head.json"))? == head,
            "public witness snapshot is incomplete or changed during verification"
        );
        let mut required_sequences = std::collections::BTreeSet::new();
        for ack in required {
            ensure!(
                required_sequences.insert(ack.body.authority_sequence),
                "duplicate required original ACK"
            );
            let (_, original, _) = store
                .records
                .get(&ack.body.authority_sequence)
                .context("public witness snapshot omits original authority ACK")?;
            ensure!(
                original == ack,
                "public witness ACK differs from original authority receipt"
            );
        }
        Ok(
            serde_json::json!({"schema":"parallel.public_witness_history_verification.v1",
            "policy_sha256":policy.sha256()?,"authority_instance":policy.authority_instance,
            "authority_epoch":policy.authority_epoch,"witness_instance":policy.witness_instance,
            "sequence":store.sequence,"tip_sha256":store.tip,"head_sha256":sha(&head),
            "original_authorization_events":store.records.len(),
            "required_ack_count":required.len(),"required_acks_sha256":sha(&bytes(&required)?),
            "effective_policy_sha256":store.effective_policy().sha256()?,"epoch_proofs":store.epoch_proofs,"epoch_status":store.epoch_status()?,"private_key_accessed":false,
            "new_acknowledgement_issued":false}),
        )
    }
    pub fn initialize(root: &Path, policy: &WitnessPolicy) -> Result<()> {
        policy.validate()?;
        ensure!(!root.exists(), "witness already exists");
        fs::create_dir(root)?;
        fs::set_permissions(root, fs::Permissions::from_mode(0o700))?;
        fs::create_dir(root.join("segments"))?;
        publish(&root.join("policy.json"), &bytes(policy)?)?;
        publish(&root.join("lease"), b"")?;
        publish(
            &root.join("head.json"),
            &bytes(&Head {
                sequence: 0,
                sha256: ZERO.into(),
            })?,
        )?;
        sync_dir(root)?;
        sync_dir(root.parent().context("witness root parent absent")?)
    }
    pub fn open(root: &Path, policy: &WitnessPolicy, key_path: &Path) -> Result<Self> {
        Self::open_mode(root, policy, key_path, false)
    }
    pub fn open_read_only(root: &Path, policy: &WitnessPolicy, key_path: &Path) -> Result<Self> {
        Self::open_mode(root, policy, key_path, true)
    }
    fn open_mode(
        root: &Path,
        policy: &WitnessPolicy,
        key_path: &Path,
        read_only: bool,
    ) -> Result<Self> {
        Self::open_mode_transition(root, policy, key_path, read_only, None)
    }
    fn open_mode_transition(
        root: &Path,
        policy: &WitnessPolicy,
        key_path: &Path,
        read_only: bool,
        transition: Option<&SignedWitnessEpochTransition>,
    ) -> Result<Self> {
        policy.validate()?;
        if let Some(request) = transition {
            crate::native_review::require_signer(&request.authorization_public_key)?;
            verify(
                &request.authorization_public_key,
                "parallel.witness_epoch_transition.v1",
                &request.body,
                &request.signature,
            )?;
            ensure!(
                request.body.immutable_base_policy_sha256 == policy.sha256()?,
                "transition immutable base differs"
            );
        }
        ensure!(
            !fs::symlink_metadata(root)?.file_type().is_symlink(),
            "witness root symlink"
        );
        let root = fs::canonicalize(root)?;
        let key_path = fs::canonicalize(key_path)?;
        ensure!(
            !key_path.starts_with(&root),
            "ACK private key must remain outside recovery copy"
        );
        ensure!(
            read(&root.join("policy.json"))? == bytes(policy)?,
            "witness policy changed"
        );

        let signing = private_key(&key_path)?;
        ensure!(
            hex::encode(signing.verifying_key().to_bytes()) == policy.witness_public_key,
            "wrong witness ACK key"
        );
        let lease = checked_file(&root.join("lease"))?;
        ensure!(lease.try_lock().is_ok(), "witness already owned");
        let identity = |p: &Path| -> Result<(u64, u64)> {
            let m = fs::symlink_metadata(p)?;
            Ok((m.dev(), m.ino()))
        };
        let root_identity = identity(&root)?;
        let segment_identity = identity(&root.join("segments"))?;
        let lease_identity = identity(&root.join("lease"))?;
        let mut store = Self {
            root,
            policy: policy.clone(),
            signing: Some(signing),
            _lease: lease,
            records: BTreeMap::new(),
            native_grants: native_grants::NativeGrantState::default(),
            epoch_proofs: Vec::new(),
            sequence: 0,
            tip: ZERO.into(),
            root_identity,
            segment_identity,
            lease_identity,
            read_only: true,
        };
        store.replay()?;
        let runtime = if let Some(request) = transition {
            if let Some(original) = store.epoch_proofs.iter().find(|p| {
                p.epoch_transition.body.next_policy.authority_epoch
                    == request.body.next_policy.authority_epoch
            }) {
                ensure!(
                    bytes(&original.epoch_transition)? == bytes(request)?,
                    "conflicting transition runtime retry"
                );
                store.effective_policy()
            } else {
                store.check_epoch_next(request, super::native_scq::policy::now_ms()?)?;
                &request.body.next_policy
            }
        } else {
            store.effective_policy()
        };
        ensure!(
            sha(&fs::read("/proc/self/exe")?) == runtime.witness_executable_sha256
                && source_sha256() == runtime.witness_source_sha256,
            "unapproved effective witness executable/source"
        );
        store.read_only = read_only;
        let head: Head = serde_json::from_slice(&read(&store.root.join("head.json"))?)?;
        if !read_only && head.sequence != store.sequence {
            store.publish_head()?;
        }
        Ok(store)
    }
    fn replay(&mut self) -> Result<()> {
        let directory = self.root.join("segments");
        ensure!(
            fs::symlink_metadata(&directory)?.is_dir()
                && !fs::symlink_metadata(&directory)?.file_type().is_symlink(),
            "witness segment custody changed"
        );
        let head: Head = serde_json::from_slice(&read(&self.root.join("head.json"))?)?;
        let mut paths = Vec::new();
        for entry in fs::read_dir(&directory)? {
            ensure!(
                paths.len() < 1_000_000,
                "witness history segment count bound"
            );
            paths.push(entry?.path());
        }
        paths.sort();
        let mut matched = head.sequence == 0 && head.sha256 == ZERO;
        for path in paths {
            let expected = format!("{:020}.json", self.sequence + 1);
            ensure!(
                path.file_name().and_then(|v| v.to_str()) == Some(expected.as_str()),
                "unresolved witness segment/gap"
            );
            let data = read(&path)?;
            let shape: Value = serde_json::from_slice(&data)?;
            if shape.get("epoch_transition").is_some() {
                self.replay_epoch_transition(serde_json::from_slice(&data)?, sha(&data))?;
                if head.sequence == self.sequence {
                    ensure!(
                        head.sha256 == self.tip,
                        "witness committed epoch head changed"
                    );
                    matched = true;
                }
                continue;
            }
            if shape.get("request").is_some() {
                self.replay_native_grant(serde_json::from_slice(&data)?, sha(&data))?;
                if head.sequence == self.sequence {
                    ensure!(
                        head.sha256 == self.tip,
                        "witness committed grant head changed"
                    );
                    matched = true;
                }
                continue;
            }
            let record: Record = serde_json::from_slice(&data)?;
            ensure!(
                bytes(&record.core.request.body.epoch_proofs)? == bytes(&self.epoch_proofs)?,
                "authorization epoch not active at original journal position"
            );
            let event = verify_request(&self.policy, &record.core.request)?;
            record.ack.verify(&self.policy, &record.core.request)?;
            ensure!(
                record.core.schema == "parallel.fit_event_witness_record.v1"
                    && record.core.sequence == self.sequence + 1
                    && record.core.previous == self.tip
                    && record.ack.body.witness_sequence == record.core.sequence
                    && record.ack.body.previous_record_sha256 == record.core.previous
                    && record.ack.body.committed_record_core_sha256 == sha(&bytes(&record.core)?),
                "witness record chain mismatch"
            );
            ensure!(
                self.records
                    .last_key_value()
                    .is_none_or(|(seq, _)| event.seq > *seq),
                "replayed/conflicting authority sequence"
            );
            let work =
                self.native_grants
                    .check_work(&self.policy, &record.core.request, self.sequence)?;
            self.native_grants.commit_work(work, event.seq);
            self.sequence = record.core.sequence;
            self.tip = sha(&data);
            self.records.insert(
                event.seq,
                (sha(&record.core.request.body.event), record.ack, sha(&data)),
            );
            if head.sequence == self.sequence {
                ensure!(head.sha256 == self.tip, "witness committed head changed");
                matched = true;
            }
        }
        ensure!(matched, "witness committed tail missing");
        // A valid durable segment may precede a lost head publication. Adopt the
        // original signed record; no authorization is generated by this replay.
        if head.sequence != self.sequence && !self.read_only {
            self.publish_head()?;
        }
        Ok(())
    }
    fn publish_head(&self) -> Result<()> {
        let temporary = self.root.join("head.pending");
        // An interrupted head update remains explicit until it is proven to be
        // the exact derived head, then it is safe to finish its rename.
        let expected = bytes(&Head {
            sequence: self.sequence,
            sha256: self.tip.clone(),
        })?;
        if temporary.exists() {
            ensure!(
                read(&temporary)? == expected,
                "conflicting pending witness head"
            );
        } else {
            publish(&temporary, &expected)?;
        }
        fs::rename(temporary, self.root.join("head.json"))?;
        sync_dir(&self.root)
    }
    fn verify_custody(&self) -> Result<()> {
        for (path, expected) in [
            (&self.root, self.root_identity),
            (&self.root.join("segments"), self.segment_identity),
            (&self.root.join("lease"), self.lease_identity),
        ] {
            let m = fs::symlink_metadata(path)?;
            ensure!(
                !m.file_type().is_symlink() && (m.dev(), m.ino()) == expected,
                "witness owned storage replaced"
            );
        }
        ensure!(
            read(&self.root.join("policy.json"))? == bytes(&self.policy)?,
            "witness policy replaced"
        );
        Ok(())
    }
    fn verify_retained(&self, ack: &SignedAck, digest: &str) -> Result<()> {
        self.verify_custody()?;
        ensure!(
            sha(&read(
                &self
                    .root
                    .join("segments")
                    .join(format!("{:020}.json", ack.body.witness_sequence))
            )?) == digest,
            "original witness segment changed"
        );
        Ok(())
    }
    pub fn record_authorization(&mut self, request: &SignedRequest) -> Result<SignedAck> {
        ensure!(
            !self.read_only,
            "read-only witness cannot record an authorization"
        );
        self.verify_custody()?;
        let event = verify_request(&self.policy, request)?;
        ensure!(
            read(&self.root.join("policy.json"))? == bytes(&self.policy)?,
            "witness policy replaced"
        );
        if let Some((original, ack, record_sha)) = self.records.get(&event.seq) {
            self.verify_retained(ack, record_sha)?;
            ensure!(
                *original == sha(&request.body.event)
                    && ack.body.request_sha256 == sha(&bytes(request)?),
                "conflicting authorization retransmission"
            );
            return Ok(ack.clone());
        }
        ensure!(
            bytes(&request.body.epoch_proofs)? == bytes(&self.epoch_proofs)?,
            "new authorization uses stale epoch"
        );
        ensure!(
            self.records
                .last_key_value()
                .is_none_or(|(seq, _)| event.seq > *seq),
            "out-of-order original authorization"
        );
        let work = self
            .native_grants
            .check_work(&self.policy, request, self.sequence)?;
        let core = RecordCore {
            schema: "parallel.fit_event_witness_record.v1".into(),
            sequence: self.sequence + 1,
            previous: self.tip.clone(),
            request: request.clone(),
        };
        let effective = self.effective_policy();
        let body = AckBody {
            schema: "parallel.fit_event_witness_ack.v1".into(),
            policy_sha256: effective.sha256()?,
            witness_instance: effective.witness_instance.clone(),
            authority_instance: effective.authority_instance.clone(),
            authority_epoch: effective.authority_epoch.clone(),
            authority_sequence: event.seq,
            event_sha256: sha(&request.body.event),
            request_sha256: sha(&bytes(request)?),
            witness_sequence: core.sequence,
            previous_record_sha256: core.previous.clone(),
            committed_record_core_sha256: sha(&bytes(&core)?),
        };
        let ack = SignedAck {
            signature: sign(
                self.signing
                    .as_ref()
                    .context("public witness verifier cannot sign")?,
                "parallel.fit_event_witness.ack.v1",
                &body,
            )?,
            body,
        };
        let record = Record {
            core,
            ack: ack.clone(),
        };
        let data = bytes(&record)?;
        publish(
            &self
                .root
                .join("segments")
                .join(format!("{:020}.json", record.core.sequence)),
            &data,
        )?;
        self.native_grants.commit_work(work, event.seq);
        self.sequence = record.core.sequence;
        self.tip = sha(&data);
        self.records.insert(
            event.seq,
            (sha(&request.body.event), ack.clone(), sha(&data)),
        );
        self.publish_head()?;
        Ok(ack)
    }
    /// An ACK query is an exact signed retransmission, never a request to create
    /// a different event or restamp its original authorization.
    pub fn get_ack(&self, request: &SignedRequest) -> Result<Option<SignedAck>> {
        let event = verify_request(&self.policy, request)?;
        match self.records.get(&event.seq) {
            Some((digest, ack, record_sha)) => {
                self.verify_retained(ack, record_sha)?;
                ensure!(
                    *digest == sha(&request.body.event)
                        && ack.body.request_sha256 == sha(&bytes(request)?),
                    "conflicting ACK query"
                );
                Ok(Some(ack.clone()))
            }
            None => Ok(None),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PinnedSshIdentity {
    pub path: PathBuf,
    pub sha256: String,
}
impl PinnedSshIdentity {
    fn open(&self) -> Result<File> {
        ensure!(
            self.path.is_absolute() && valid_hash(&self.sha256),
            "invalid pinned SSH identity"
        );
        for ancestor in self.path.ancestors() {
            ensure!(
                !fs::symlink_metadata(ancestor)?.file_type().is_symlink(),
                "SSH identity ancestry is a symlink"
            );
        }
        let mut file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
            .open(&self.path)?;
        let metadata = file.metadata()?;
        ensure!(
            metadata.is_file()
                && metadata.uid() == crate::host_os::effective_uid()
                && metadata.mode() & 0o077 == 0
                && metadata.len() <= 16384,
            "SSH identity must be bounded private owner-only regular file"
        );
        let mut contents = Vec::new();
        Read::by_ref(&mut file)
            .take(16385)
            .read_to_end(&mut contents)?;
        let after = file.metadata()?;
        ensure!(
            contents.len() as u64 == metadata.len()
                && sha(&contents) == self.sha256
                && metadata.mtime() == after.mtime()
                && metadata.mtime_nsec() == after.mtime_nsec()
                && metadata.ctime() == after.ctime()
                && metadata.ctime_nsec() == after.ctime_nsec(),
            "pinned SSH identity bytes changed"
        );
        Ok(file)
    }
}
fn bind_ssh_identity(command: &mut Command, identity: &PinnedSshIdentity) -> Result<File> {
    use std::os::fd::AsRawFd;
    let file = identity.open()?;
    // SSH reopens this authority-owned descriptor, not the replaceable configured pathname.
    // The caller retains it until SSH exits. CLOEXEC prevents descriptor inheritance.
    command
        .arg("-i")
        .arg(format!(
            "/proc/{}/fd/{}",
            std::process::id(),
            file.as_raw_fd()
        ))
        .args(["-o", "IdentitiesOnly=yes", "-o", "IdentityAgent=none"]);
    Ok(file)
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WitnessClientConfig {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub epoch_proofs: Vec<WitnessEpochProof>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ssh_identity: Option<PinnedSshIdentity>,
    pub policy: WitnessPolicy,
    pub authority_key: PathBuf,
    pub known_hosts: PathBuf,
    pub remote_executable: PathBuf,
    pub remote_root: PathBuf,
    pub remote_policy: PathBuf,
    pub remote_ack_key: PathBuf,
    #[cfg(test)]
    #[serde(skip)]
    local_socket: Option<PathBuf>,
}
impl WitnessClientConfig {
    /// Load the exact reviewed public transport configuration. Merely knowing a
    /// policy digest cannot replace the actual pinned keys and executable checks.
    pub fn from_file(path: &Path, expected_sha256: &str) -> Result<Self> {
        let config = Self::from_transport_file(path, expected_sha256)?;
        WitnessClient::new(config.clone())?;
        Ok(config)
    }
    /// Root provisioners use transport custody without impersonating the authority.
    pub fn from_transport_file(path: &Path, expected_sha256: &str) -> Result<Self> {
        ensure!(
            path.is_absolute(),
            "event witness configuration must be absolute"
        );
        ensure!(
            valid_hash(expected_sha256),
            "event witness configuration digest invalid"
        );
        for ancestor in path.ancestors() {
            ensure!(
                !fs::symlink_metadata(ancestor)?.file_type().is_symlink(),
                "event witness configuration symlink requires reconciliation"
            );
        }
        let data = read(path)?;
        ensure!(
            sha(&data) == expected_sha256,
            "event witness configuration changed"
        );
        let config: Self = serde_json::from_slice(&data)?;
        validate_transport_config(&config)?;
        Ok(config)
    }
}
fn validate_transport_config(config: &WitnessClientConfig) -> Result<()> {
    let effective = verify_epoch_chain(&config.policy, &config.epoch_proofs)?;
    ensure!(
        (effective.native_fit_trust.is_none() && effective.native_grant_delegation.is_none())
            || config.ssh_identity.is_some(),
        "native witness requires an explicitly pinned SSH identity"
    );
    if let Some(identity) = &config.ssh_identity {
        let _ = identity.open()?;
    }
    ensure!(
        sha(&read(&config.known_hosts)?) == config.policy.ssh_known_hosts_sha256,
        "pinned SSH host file changed"
    );
    for path in [
        &config.remote_executable,
        &config.remote_root,
        &config.remote_policy,
        &config.remote_ack_key,
    ] {
        let value = path.to_str().context("non-UTF8 remote custody path")?;
        ensure!(
            value.starts_with('/')
                && value
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"/._-".contains(&c)),
            "unsafe remote custody path"
        );
    }
    Ok(())
}
/// Validate the complete signed request before any durable root provisioner intent.
pub fn validate_native_grant_request(
    policy: &WitnessPolicy,
    request: &SignedAddNativeGrant,
) -> Result<NativeWitnessTrust> {
    native_grants::validate_update(policy, request)
}
/// Sends an independently signed update using the same pinned deadline-bounded
/// transport, without opening the ordinary authority signing key.
pub fn add_native_grant_remote(
    config: &WitnessClientConfig,
    request: &SignedAddNativeGrant,
    deadline_monotonic_ns: u64,
) -> Result<NativeGrantProof> {
    validate_transport_config(config)?;
    validate_native_grant_request_for_epoch(&config.policy, &config.epoch_proofs, request)?;
    let proof: NativeGrantProof = WitnessClient::exchange_config(
        config,
        request,
        deadline_monotonic_ns,
        "registry-witness-add-native-grant",
    )?;
    verify_native_grant_proof(&config.policy, &proof)?;
    ensure!(
        bytes(&proof.epoch_proofs)? == bytes(&config.epoch_proofs)?,
        "witness grant epoch chain differs"
    );
    ensure!(
        bytes(&proof.request)? == bytes(request)?,
        "witness returned a different native grant request"
    );
    remaining(deadline_monotonic_ns)?;
    Ok(proof)
}
/// Deliver one independently signed transition; this transport owns no signing key.
pub fn epoch_transition_remote(
    config: &WitnessClientConfig,
    request: &SignedWitnessEpochTransition,
    deadline_monotonic_ns: u64,
) -> Result<WitnessEpochProof> {
    validate_transport_config(config)?;
    let previous = verify_epoch_chain(&config.policy, &config.epoch_proofs)?;
    epochs::validate_transition(
        &config.policy,
        &previous,
        request,
        request.body.not_before_unix_ms,
    )?;
    let proof: WitnessEpochProof = WitnessClient::exchange_config(
        config,
        request,
        deadline_monotonic_ns,
        "registry-witness-epoch-transition",
    )?;
    ensure!(
        bytes(&proof.epoch_transition)? == bytes(request)?,
        "witness returned different original epoch transition"
    );
    let mut chain = config.epoch_proofs.clone();
    chain.push(proof.clone());
    verify_epoch_chain(&config.policy, &chain)?;
    remaining(deadline_monotonic_ns)?;
    Ok(proof)
}
pub struct WitnessClient {
    effective: WitnessPolicy,
    config: WitnessClientConfig,
    signing: SigningKey,
}
impl WitnessClient {
    pub fn new(config: WitnessClientConfig) -> Result<Self> {
        validate_transport_config(&config)?;
        let effective = verify_epoch_chain(&config.policy, &config.epoch_proofs)?;
        ensure!(
            sha(&fs::read("/proc/self/exe")?) == effective.authority_executable_sha256
                && authority_source_sha256() == effective.authority_source_sha256,
            "unapproved loaded authority witness producer"
        );
        let signing = private_key(&config.authority_key)?;
        ensure!(
            hex::encode(signing.verifying_key().to_bytes()) == config.policy.authority_public_key,
            "wrong authority witness signing key"
        );
        Ok(Self {
            config,
            signing,
            effective,
        })
    }
    pub fn policy(&self) -> &WitnessPolicy {
        &self.effective
    }
    pub fn base_policy(&self) -> &WitnessPolicy {
        &self.config.policy
    }
    pub fn request(&self, event: &[u8]) -> Result<SignedRequest> {
        event_identity_effective(&self.config.policy, &self.effective, event)?;
        let body = RequestBody {
            epoch_proofs: self.config.epoch_proofs.clone(),
            schema: "parallel.fit_event_witness_request.v1".into(),
            policy_sha256: self.effective.sha256()?,
            event: event.to_vec(),
        };
        let request = SignedRequest {
            signature: sign(
                &self.signing,
                "parallel.fit_event_witness.request.v1",
                &body,
            )?,
            body,
        };
        ensure!(
            bytes(&request)?.len() <= MAX_FRAME,
            "original witness request exceeds transport bound"
        );
        Ok(request)
    }
    /// No reconnect or implicit retry. A lost response leaves original authority
    /// evidence unknown. An operator may retrieve the exact original ACK later.
    pub fn record(&self, event: &[u8]) -> Result<SignedAck> {
        let identity = event_identity_effective(&self.config.policy, &self.effective, event)?;
        self.record_until(event, identity.deadline)
    }
    /// A shorter caller deadline bounds the original acknowledgment operation;
    /// it never rewrites the signed task deadline or permits a later retry.
    pub fn record_until(&self, event: &[u8], deadline: u64) -> Result<SignedAck> {
        let identity = event_identity_effective(&self.config.policy, &self.effective, event)?;
        ensure!(
            deadline <= identity.deadline,
            "witness transport exceeds original admission deadline"
        );
        let request = self.request(event)?;
        let ack: SignedAck = self.exchange(&request, deadline, "registry-witness-request")?;
        ack.verify(&self.config.policy, &request)?;
        Ok(ack)
    }
    /// Fetch an already committed ACK using a separate bounded transport
    /// deadline. An absent record never becomes a new authorization.
    pub fn get_ack(&self, event: &[u8], deadline_monotonic_ns: u64) -> Result<Option<SignedAck>> {
        let request = self.request(event)?;
        let ack: Option<SignedAck> =
            self.exchange(&request, deadline_monotonic_ns, "registry-witness-get-ack")?;
        if let Some(ack) = &ack {
            ack.verify(&self.config.policy, &request)?;
        }
        Ok(ack)
    }
    fn exchange<T: serde::de::DeserializeOwned + Send + 'static>(
        &self,
        request: &SignedRequest,
        deadline: u64,
        operation: &str,
    ) -> Result<T> {
        Self::exchange_config(&self.config, request, deadline, operation)
    }
    fn exchange_config<
        T: serde::de::DeserializeOwned + Send + 'static,
        R: Serialize + Clone + Send + 'static,
    >(
        config: &WitnessClientConfig,
        request: &R,
        deadline: u64,
        operation: &str,
    ) -> Result<T> {
        ensure!(
            sha(&read(&config.known_hosts)?) == config.policy.ssh_known_hosts_sha256,
            "pinned SSH custody changed"
        );
        #[cfg(test)]
        if let Some(socket) = &config.local_socket {
            let mut stream = std::os::unix::net::UnixStream::connect(socket)?;
            stream.set_read_timeout(Some(remaining(deadline)?))?;
            stream.set_write_timeout(Some(remaining(deadline)?))?;
            write_frame(&mut stream, &request)?;
            let reply: T = read_frame(&mut stream)?;
            remaining(deadline)?;
            return Ok(reply);
        }
        let mut command = Command::new("/usr/bin/ssh");
        let _ssh_identity = config
            .ssh_identity
            .as_ref()
            .map(|identity| bind_ssh_identity(&mut command, identity))
            .transpose()?;
        command
            .args([
                "-F",
                "/dev/null",
                "-T",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                "GlobalKnownHostsFile=/dev/null",
                "-o",
                "UpdateHostKeys=no",
                "-o",
                "ControlMaster=no",
                "-o",
                "ControlPath=none",
            ])
            .arg("-o")
            .arg(format!(
                "UserKnownHostsFile={}",
                config.known_hosts.display()
            ))
            .arg("-o")
            .arg(format!("HostKeyAlias={}", config.policy.ssh_host_alias))
            .arg("--")
            .arg(&config.policy.ssh_destination)
            .arg(format!(
                "{} {} --root {} --policy {} --key {}",
                config.remote_executable.display(),
                operation,
                config.remote_root.display(),
                config.remote_policy.display(),
                config.remote_ack_key.display()
            ))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        // SAFETY: the pre-exec callback only sets the child's process group
        // with an integer-only syscall; it does not touch shared Rust state.
        unsafe {
            command.pre_exec(|| {
                if libc::setpgid(0, 0) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        remaining(deadline)?;
        let mut child = command.spawn()?;
        let pid = child.id();
        let input = child
            .stdin
            .take()
            .context("witness transport stdin absent")?;
        let output = child
            .stdout
            .take()
            .context("witness transport stdout absent")?;
        let errors = child
            .stderr
            .take()
            .context("witness transport stderr absent")?;
        let request = (*request).clone();
        let send = std::thread::spawn(move || write_frame(input, &request));
        let receive = std::thread::spawn(move || read_frame::<T>(output));
        let stderr = std::thread::spawn(move || {
            let mut data = Vec::new();
            errors.take(65536).read_to_end(&mut data).map(|_| data)
        });
        let exit = loop {
            if let Some(status) = child.try_wait()? {
                break Ok(status);
            }
            if remaining(deadline).is_err() {
                // SAFETY: `pid` is this live child's pid and pre_exec made it
                // the leader of its own process group before exec.
                unsafe {
                    libc::kill(-(pid as i32), libc::SIGKILL);
                }
                let _ = child.wait();
                break Err(anyhow::anyhow!(
                    "original authorization deadline expired waiting for witness"
                ));
            }
            std::thread::sleep(Duration::from_millis(2));
        };
        let sent = send
            .join()
            .map_err(|_| anyhow::anyhow!("witness writer panicked"))?;
        let reply = receive
            .join()
            .map_err(|_| anyhow::anyhow!("witness reader panicked"))?;
        let _stderr = stderr.join();
        ensure!(
            exit?.success(),
            "witness transport failed; original authorization remains unavailable"
        );
        sent?;
        let reply = reply?;
        remaining(deadline)?;
        Ok(reply)
    }
}
fn remaining(deadline: u64) -> Result<Duration> {
    let now = crate::host_os::monotonic_ns().context("monotonic clock absent")?;
    Ok(Duration::from_nanos(
        deadline
            .checked_sub(now)
            .filter(|v| *v > 0)
            .context("original authorization deadline expired")?,
    ))
}
fn write_frame(mut out: impl Write, value: &impl Serialize) -> Result<()> {
    let data = bytes(value)?;
    ensure!(data.len() <= MAX_FRAME, "oversized witness frame");
    out.write_all(&(data.len() as u32).to_be_bytes())?;
    out.write_all(&data)?;
    out.flush()?;
    Ok(())
}
fn read_frame<T: serde::de::DeserializeOwned>(mut input: impl Read) -> Result<T> {
    let mut width = [0; 4];
    input.read_exact(&mut width)?;
    let width = u32::from_be_bytes(width) as usize;
    ensure!(
        width > 0 && width <= MAX_FRAME,
        "invalid witness frame length"
    );
    let mut data = vec![0; width];
    input.read_exact(&mut data)?;
    Ok(serde_json::from_slice(&data)?)
}
/// One framed request over pinned SSH stdio. The remote executable verifies its
/// actual loaded identity and owns the private ACK key locally.
pub fn serve_stdio(root: &Path, policy_path: &Path, key_path: &Path) -> Result<()> {
    let policy: WitnessPolicy = serde_json::from_slice(&read(policy_path)?)?;
    let request: SignedRequest = read_frame(std::io::stdin().lock())?;
    let mut store = WitnessStore::open(root, &policy, key_path)?;
    let ack = store.record_authorization(&request)?;
    write_frame(std::io::stdout().lock(), &ack)
}
/// Recovery reads cannot append events or repair a witness head. They return
/// the exact signed ACK, including when its original start deadline has expired.
pub fn serve_stdio_get_ack(root: &Path, policy_path: &Path, key_path: &Path) -> Result<()> {
    let policy: WitnessPolicy = serde_json::from_slice(&read(policy_path)?)?;
    let request: SignedRequest = read_frame(std::io::stdin().lock())?;
    let store = WitnessStore::open_read_only(root, &policy, key_path)?;
    write_frame(std::io::stdout().lock(), &store.get_ack(&request)?)
}
#[cfg(test)]
pub(crate) mod funding_tests;
#[cfg(test)]
mod tests;
