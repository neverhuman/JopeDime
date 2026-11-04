use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
};

use ed25519_dalek::SigningKey;
use jain_smartcluster_fabric::{
    sha256, valid_identifier, Digest32, Placement, SignedWorkUnitProjection, VerifiedWorkUnit,
    WorkerAcknowledgement,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

const MAX_ASSIGNMENT_BYTES: usize = 1024 * 1024;
const MAX_RECOVERY_RECORDS: usize = 10_000;

#[derive(Clone)]
pub struct NodeRuntimeIdentity {
    node_id: String,
    session_id: String,
    key_id: String,
    hub_issuer: String,
    hub_public_key: Digest32,
    image_issuer: String,
    image_public_key: Digest32,
    signing_key: SigningKey,
}

impl NodeRuntimeIdentity {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        node_id: String,
        session_id: String,
        key_id: String,
        hub_issuer: String,
        hub_public_key: Digest32,
        image_issuer: String,
        image_public_key: Digest32,
        signing_key: SigningKey,
    ) -> Result<Self, AssignmentError> {
        if !valid_identifier(&node_id)
            || !valid_identifier(&session_id)
            || !valid_identifier(&key_id)
            || !valid_identifier(&hub_issuer)
            || !valid_identifier(&image_issuer)
            || hub_public_key == [0; 32]
            || image_public_key == [0; 32]
            || signing_key.verifying_key().to_bytes() == [0; 32]
        {
            return Err(AssignmentError::InvalidIdentity);
        }
        Ok(Self {
            node_id,
            session_id,
            key_id,
            hub_issuer,
            hub_public_key,
            image_issuer,
            image_public_key,
            signing_key,
        })
    }

    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn public_key(&self) -> Digest32 {
        self.signing_key.verifying_key().to_bytes()
    }

    pub(crate) fn key_id(&self) -> &str {
        &self.key_id
    }

    pub(crate) fn signing_key(&self) -> &SigningKey {
        &self.signing_key
    }

    pub(crate) fn hub_issuer(&self) -> &str {
        &self.hub_issuer
    }

    pub(crate) const fn hub_public_key(&self) -> Digest32 {
        self.hub_public_key
    }

    pub fn session_hello(
        &self,
        profile: crate::NodeProfile,
        provision: crate::WorkerProvision,
        qualification_digest: Digest32,
        sequence: u64,
        at_ms: u64,
    ) -> Result<crate::NodeControlFrame, crate::ProtocolError> {
        crate::NodeControlFrame::session_hello(
            self.node_id.clone(),
            self.session_id.clone(),
            profile,
            provision,
            qualification_digest,
            sequence,
            at_ms,
            self.key_id.clone(),
            &self.signing_key,
        )
    }

    pub(crate) fn reverify_work(&self, work: &VerifiedWorkUnit) -> Result<(), AssignmentError> {
        let verified = VerifiedWorkUnit::verify(
            work.signed_projection().clone(),
            &self.hub_issuer,
            self.hub_public_key,
            &self.image_issuer,
            self.image_public_key,
        )
        .map_err(AssignmentError::Fabric)?;
        if &verified != work {
            return Err(AssignmentError::InvalidRecord);
        }
        Ok(())
    }

    pub(crate) fn reverify_placement(
        &self,
        work: &VerifiedWorkUnit,
        placement: &Placement,
        now_ms: u64,
    ) -> Result<(), AssignmentError> {
        placement
            .verify_for(
                work,
                &self.node_id,
                &self.session_id,
                self.hub_public_key,
                now_ms,
            )
            .map_err(AssignmentError::Fabric)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum AssignmentState {
    Accepted,
    Running,
    Terminal,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct AssignmentRecord {
    signed_work: SignedWorkUnitProjection,
    placement: Placement,
    run_id: String,
    work_unit_digest: Digest32,
    command_digest: Digest32,
    placement_digest: Digest32,
    node_id: String,
    session_id: String,
    lease_id: String,
    lease_expires_at_ms: u64,
    attempt: u32,
    attempt_fence: u64,
    hub_epoch: u64,
    control_position: u64,
    session_sequence: u64,
    acknowledged_at_ms: u64,
    state: AssignmentState,
}

impl AssignmentRecord {
    fn new(
        work: &VerifiedWorkUnit,
        placement: &Placement,
        session_sequence: u64,
        acknowledged_at_ms: u64,
    ) -> Result<Self, AssignmentError> {
        if session_sequence == 0 || acknowledged_at_ms == 0 {
            return Err(AssignmentError::InvalidRecord);
        }
        Ok(Self {
            signed_work: work.signed_projection().clone(),
            placement: placement.clone(),
            run_id: work.run_id.to_string(),
            work_unit_digest: work.work_unit_digest(),
            command_digest: work.command_digest(),
            placement_digest: placement.digest().map_err(AssignmentError::Fabric)?,
            node_id: placement.node_id.clone(),
            session_id: placement.session_id.clone(),
            lease_id: placement.lease_id.clone(),
            lease_expires_at_ms: placement.lease_expires_at_ms,
            attempt: work.attempt,
            attempt_fence: work.attempt_fence,
            hub_epoch: work.hub_epoch,
            control_position: work.control_position,
            session_sequence,
            acknowledged_at_ms,
            state: AssignmentState::Accepted,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AcceptedAssignment {
    work: VerifiedWorkUnit,
    placement: Placement,
    journal_digest: Digest32,
    session_sequence: u64,
    acknowledged_at_ms: u64,
    record_path: PathBuf,
}

impl AcceptedAssignment {
    pub fn work(&self) -> &VerifiedWorkUnit {
        &self.work
    }

    pub fn placement(&self) -> &Placement {
        &self.placement
    }

    pub const fn journal_digest(&self) -> Digest32 {
        self.journal_digest
    }

    pub const fn session_sequence(&self) -> u64 {
        self.session_sequence
    }

    pub(crate) fn acknowledgement(
        &self,
        identity: &NodeRuntimeIdentity,
    ) -> Result<WorkerAcknowledgement, AssignmentError> {
        WorkerAcknowledgement::sign(
            identity.node_id.clone(),
            identity.session_id.clone(),
            &self.work,
            &self.placement,
            self.journal_digest,
            self.acknowledged_at_ms,
            &identity.signing_key,
        )
        .map_err(AssignmentError::Fabric)
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RecoveryReport {
    pub pending: Vec<String>,
    pub replayable: Vec<AcceptedAssignment>,
    pub quarantined: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct AssignmentJournal {
    root: PathBuf,
}

mod journal;
fn atomic_replace(root: &Path, destination: &Path, bytes: &[u8]) -> Result<(), AssignmentError> {
    let pending = destination.with_extension("transition");
    write_new_synced(&pending, bytes)?;
    fs::rename(&pending, destination).map_err(AssignmentError::Io)?;
    sync_directory(root)
}

fn write_new_synced(path: &Path, bytes: &[u8]) -> Result<(), AssignmentError> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(nix::libc::O_NOFOLLOW)
        .open(path)
        .map_err(AssignmentError::Io)?;
    file.write_all(bytes).map_err(AssignmentError::Io)?;
    file.sync_all().map_err(AssignmentError::Io)
}

fn sync_directory(path: &Path) -> Result<(), AssignmentError> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(AssignmentError::Io)
}

fn read_record(path: &Path) -> Result<AssignmentRecord, AssignmentError> {
    let bytes = read_bounded(path)?;
    postcard::from_bytes(&bytes).map_err(|_| AssignmentError::InvalidRecord)
}

fn read_bounded(path: &Path) -> Result<Vec<u8>, AssignmentError> {
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(nix::libc::O_NOFOLLOW)
        .open(path)
        .map_err(AssignmentError::Io)?;
    let metadata = file.metadata().map_err(AssignmentError::Io)?;
    if !metadata.is_file() || metadata.nlink() != 1 || metadata.mode() & 0o022 != 0 {
        return Err(AssignmentError::UnsafeJournalPath);
    }
    let mut bytes = Vec::new();
    std::io::Read::by_ref(&mut file)
        .take((MAX_ASSIGNMENT_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(AssignmentError::Io)?;
    if bytes.len() > MAX_ASSIGNMENT_BYTES {
        return Err(AssignmentError::InvalidRecord);
    }
    Ok(bytes)
}

impl AssignmentRecord {
    fn validate_exact(
        &self,
        identity: &NodeRuntimeIdentity,
        work: &VerifiedWorkUnit,
        now_ms: u64,
    ) -> Result<(), AssignmentError> {
        self.placement
            .verify_for(
                work,
                identity.node_id(),
                identity.session_id(),
                identity.hub_public_key,
                now_ms,
            )
            .map_err(AssignmentError::Fabric)?;
        let expected = Self::new(
            work,
            &self.placement,
            self.session_sequence,
            self.acknowledged_at_ms,
        )?;
        if self.run_id != expected.run_id
            || self.work_unit_digest != expected.work_unit_digest
            || self.command_digest != expected.command_digest
            || self.placement_digest != expected.placement_digest
            || self.node_id != expected.node_id
            || self.session_id != expected.session_id
            || self.lease_id != expected.lease_id
            || self.lease_expires_at_ms != expected.lease_expires_at_ms
            || self.attempt != expected.attempt
            || self.attempt_fence != expected.attempt_fence
            || self.hub_epoch != expected.hub_epoch
            || self.control_position != expected.control_position
            || self.session_sequence != expected.session_sequence
            || self.acknowledged_at_ms != expected.acknowledged_at_ms
        {
            return Err(AssignmentError::InvalidRecord);
        }
        Ok(())
    }
}

fn valid_transition(current: &AssignmentRecord, next: &AssignmentRecord) -> bool {
    let mut expected = current.clone();
    expected.state = match (current.state, next.state) {
        (AssignmentState::Accepted, AssignmentState::Running | AssignmentState::Terminal)
        | (AssignmentState::Running, AssignmentState::Terminal) => next.state,
        _ => return false,
    };
    expected == *next
}

fn quarantine_file(root: &Path, path: &Path) -> Result<PathBuf, AssignmentError> {
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or(AssignmentError::UnsafeJournalPath)?;
    let quarantine = root.join(format!("{file_name}.quarantine"));
    if quarantine.exists() {
        return Err(AssignmentError::ConflictingRecord);
    }
    fs::rename(path, &quarantine).map_err(AssignmentError::Io)?;
    sync_directory(root)?;
    Ok(quarantine)
}

fn hex_digest(digest: Digest32) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[derive(Debug, Error)]
pub enum AssignmentError {
    #[error("invalid node runtime identity")]
    InvalidIdentity,
    #[error("assignment journal path failed physical-directory validation")]
    UnsafeJournalPath,
    #[error("invalid assignment record")]
    InvalidRecord,
    #[error("conflicting assignment record")]
    ConflictingRecord,
    #[error("assignment recovery bound exceeded")]
    RecoveryBoundExceeded,
    #[error("assignment journal I/O failed: {0}")]
    Io(std::io::Error),
    #[error("Fabric assignment validation failed: {0}")]
    Fabric(jain_smartcluster_fabric::FabricError),
}
