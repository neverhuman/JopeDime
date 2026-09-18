use std::{
    collections::BTreeMap,
    fs::{self, DirBuilder, File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
    sync::Mutex,
};

use async_trait::async_trait;
use jain_smartcluster_fabric::{sha256, Digest32, WorkerAcknowledgement};
use nix::unistd::Uid;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    EventSink, ExecutionError, ExecutionReceipt, NodeControlFrame, NodeEvent, NodeRuntimeIdentity,
    NodeTerminalOutcome, ProtocolError, SignedStorageFragmentReceipt, SignedStorageTelemetry,
};

const OUTBOX_SCHEMA: &str = "jain.node.control-outbox/v1";
const POSITION_SCHEMA: &str = "jain.node.control-outbox-position/v1";
const POSITION_CHECKSUM_DOMAIN: &[u8] = b"jain.node.control-outbox-position.v1\0";
const POSITION_FILE: &str = "outbox.position";
const POSITION_PENDING_FILE: &str = "outbox-position.pending";
const ENQUEUE_PENDING_FILE: &str = "outbox-enqueue.pending";
const MAX_OUTBOX_RECORDS: usize = 10_000;
const MAX_OUTBOX_RECORD_BYTES: u64 = 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OutboundPending {
    pub sequence: u64,
    pub digest: Digest32,
    pub frame: NodeControlFrame,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutboundConfirmation {
    Retired,
    AlreadyConfirmed,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct StoredOutbound {
    schema: String,
    sequence: u64,
    digest: Digest32,
    frame_bytes: Vec<u8>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct OutboxPosition {
    schema: String,
    confirmed_sequence: u64,
    confirmed_digest: Digest32,
    checksum: Digest32,
}

impl Default for OutboxPosition {
    fn default() -> Self {
        let mut position = Self {
            schema: POSITION_SCHEMA.to_owned(),
            confirmed_sequence: 0,
            confirmed_digest: [0; 32],
            checksum: [0; 32],
        };
        position.checksum = position_checksum(0, [0; 32]);
        position
    }
}

#[derive(Clone, Debug)]
struct PendingRecord {
    pending: OutboundPending,
    path: PathBuf,
}

#[derive(Debug)]
struct OutboxState {
    position: OutboxPosition,
    pending: BTreeMap<u64, PendingRecord>,
}

pub struct DurableControlOutbox {
    root: PathBuf,
    identity: NodeRuntimeIdentity,
    state: Mutex<OutboxState>,
}

impl DurableControlOutbox {
    pub fn open(root: PathBuf, identity: NodeRuntimeIdentity) -> Result<Self, OutboxError> {
        validate_or_create_root(&root)?;
        if root.join(POSITION_PENDING_FILE).exists() || root.join(ENQUEUE_PENDING_FILE).exists() {
            return Err(OutboxError::InvalidState);
        }
        let position = read_position(&root)?;
        let mut pending = BTreeMap::new();
        let mut confirmed_residue = None;
        let mut entries = fs::read_dir(&root)
            .map_err(OutboxError::Io)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(OutboxError::Io)?;
        entries.sort_by_key(std::fs::DirEntry::file_name);
        if entries.len() > MAX_OUTBOX_RECORDS.saturating_add(1) {
            return Err(OutboxError::CapacityExceeded);
        }
        for entry in entries {
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| OutboxError::InvalidState)?;
            if name == POSITION_FILE {
                continue;
            }
            let (sequence, filename_digest) = parse_record_name(&name)?;
            let path = entry.path();
            let bytes = read_physical_file(&path, MAX_OUTBOX_RECORD_BYTES)?;
            let stored: StoredOutbound =
                postcard::from_bytes(&bytes).map_err(|_| OutboxError::InvalidState)?;
            let pending_record =
                validate_record(stored, sequence, filename_digest, &identity, path.clone())?;
            if sequence < position.confirmed_sequence {
                return Err(OutboxError::InvalidState);
            }
            if sequence == position.confirmed_sequence {
                if sequence == 0
                    || pending_record.pending.digest != position.confirmed_digest
                    || confirmed_residue.replace(path).is_some()
                {
                    return Err(OutboxError::InvalidState);
                }
                continue;
            }
            if pending.insert(sequence, pending_record).is_some() {
                return Err(OutboxError::Conflict);
            }
        }
        if let Some(path) = confirmed_residue {
            fs::remove_file(path).map_err(OutboxError::Io)?;
            sync_directory(&root)?;
        }
        validate_sequence_continuity(&position, &pending)?;
        if position.confirmed_sequence == 0 && position.confirmed_digest != [0; 32] {
            return Err(OutboxError::InvalidState);
        }
        if position.confirmed_sequence != 0 && position.confirmed_digest == [0; 32] {
            return Err(OutboxError::InvalidState);
        }
        Ok(Self {
            root,
            identity,
            state: Mutex::new(OutboxState { position, pending }),
        })
    }

    pub fn enqueue_acknowledgement(
        &self,
        acknowledgement: WorkerAcknowledgement,
    ) -> Result<OutboundPending, OutboxError> {
        self.enqueue(|sequence| {
            NodeControlFrame::assignment_accepted(&self.identity, sequence, acknowledgement)
        })
    }

    pub fn contains_acknowledgement(
        &self,
        acknowledgement: &WorkerAcknowledgement,
    ) -> Result<bool, OutboxError> {
        let state = self.state.lock().map_err(|_| OutboxError::InvalidState)?;
        if state.pending.values().any(|record| {
            matches!(
                &record.pending.frame,
                NodeControlFrame::AssignmentAccepted {
                    acknowledgement: pending,
                    ..
                } if pending.as_ref() == acknowledgement
            )
        }) {
            return Ok(true);
        }
        if state.position.confirmed_sequence == 0 {
            return Ok(false);
        }
        let frame = NodeControlFrame::assignment_accepted(
            &self.identity,
            state.position.confirmed_sequence,
            acknowledgement.clone(),
        )
        .map_err(OutboxError::Protocol)?;
        let digest = sha256(&frame.encode().map_err(OutboxError::Protocol)?);
        Ok(digest == state.position.confirmed_digest)
    }

    pub fn enqueue_event(&self, event: NodeEvent) -> Result<OutboundPending, OutboxError> {
        self.enqueue(|sequence| NodeControlFrame::outbound_event(&self.identity, sequence, event))
    }

    pub fn enqueue_storage_telemetry(
        &self,
        telemetry: SignedStorageTelemetry,
    ) -> Result<OutboundPending, OutboxError> {
        self.enqueue(|sequence| {
            NodeControlFrame::outbound_storage_telemetry(&self.identity, sequence, telemetry)
        })
    }

    pub fn enqueue_storage_fragment_receipt(
        &self,
        receipt: SignedStorageFragmentReceipt,
    ) -> Result<OutboundPending, OutboxError> {
        self.enqueue(|sequence| {
            NodeControlFrame::outbound_storage_fragment_receipt(&self.identity, sequence, receipt)
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_terminal(
        &self,
        run_id: String,
        attempt: u32,
        attempt_fence: u64,
        control_position: u64,
        at_ms: u64,
        outcome: NodeTerminalOutcome,
    ) -> Result<OutboundPending, OutboxError> {
        self.enqueue(|sequence| {
            NodeControlFrame::terminal(
                &self.identity,
                sequence,
                run_id,
                attempt,
                attempt_fence,
                control_position,
                at_ms,
                outcome,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_terminal_success(
        &self,
        run_id: String,
        attempt: u32,
        attempt_fence: u64,
        control_position: u64,
        at_ms: u64,
        receipt: ExecutionReceipt,
    ) -> Result<OutboundPending, OutboxError> {
        self.enqueue_terminal(
            run_id,
            attempt,
            attempt_fence,
            control_position,
            at_ms,
            NodeTerminalOutcome::Success(receipt),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_terminal_failure(
        &self,
        run_id: String,
        attempt: u32,
        attempt_fence: u64,
        control_position: u64,
        at_ms: u64,
        error: &ExecutionError,
    ) -> Result<OutboundPending, OutboxError> {
        self.enqueue_terminal(
            run_id,
            attempt,
            attempt_fence,
            control_position,
            at_ms,
            NodeTerminalOutcome::failure(error),
        )
    }

    pub fn oldest(&self) -> Result<Option<OutboundPending>, OutboxError> {
        let state = self.state.lock().map_err(|_| OutboxError::InvalidState)?;
        Ok(state
            .pending
            .first_key_value()
            .map(|(_, record)| record.pending.clone()))
    }

    pub fn pending_count(&self) -> Result<usize, OutboxError> {
        let state = self.state.lock().map_err(|_| OutboxError::InvalidState)?;
        Ok(state.pending.len())
    }

    pub(crate) fn verify_session_hello_identity(
        &self,
        frame: &NodeControlFrame,
    ) -> Result<(), OutboxError> {
        let NodeControlFrame::SessionHello {
            node_id,
            session_id,
            signature,
            ..
        } = frame
        else {
            return Err(OutboxError::Conflict);
        };
        if node_id != self.identity.node_id()
            || session_id != self.identity.session_id()
            || signature.key_id != self.identity.key_id()
            || signature.public_key != self.identity.public_key()
        {
            return Err(OutboxError::Conflict);
        }
        Ok(())
    }

    pub fn confirm_controller(
        &self,
        frame: &NodeControlFrame,
        now_ms: u64,
        maximum_clock_skew_ms: u64,
    ) -> Result<OutboundConfirmation, OutboxError> {
        let (sequence, digest) = frame
            .verify_outbound_confirmation(
                self.identity.session_id(),
                self.identity.hub_issuer(),
                self.identity.hub_public_key(),
                now_ms,
                maximum_clock_skew_ms,
            )
            .map_err(OutboxError::Protocol)?;
        self.retire_exact(sequence, digest)
    }

    fn retire_exact(
        &self,
        sequence: u64,
        digest: Digest32,
    ) -> Result<OutboundConfirmation, OutboxError> {
        let mut state = self.state.lock().map_err(|_| OutboxError::InvalidState)?;
        if sequence != 0
            && digest != [0; 32]
            && sequence == state.position.confirmed_sequence
            && digest == state.position.confirmed_digest
        {
            return Ok(OutboundConfirmation::AlreadyConfirmed);
        }
        let Some((&oldest_sequence, record)) = state.pending.first_key_value() else {
            return Err(OutboxError::Conflict);
        };
        if sequence != oldest_sequence || digest != record.pending.digest || digest == [0; 32] {
            return Err(OutboxError::Conflict);
        }
        let path = record.path.clone();
        let next_position = OutboxPosition {
            schema: POSITION_SCHEMA.to_owned(),
            confirmed_sequence: sequence,
            confirmed_digest: digest,
            checksum: position_checksum(sequence, digest),
        };
        persist_position(&self.root, &next_position)?;
        fs::remove_file(path).map_err(OutboxError::Io)?;
        sync_directory(&self.root)?;
        state.position = next_position;
        state.pending.remove(&sequence);
        Ok(OutboundConfirmation::Retired)
    }

    fn enqueue<F>(&self, build: F) -> Result<OutboundPending, OutboxError>
    where
        F: FnOnce(u64) -> Result<NodeControlFrame, ProtocolError>,
    {
        let mut state = self.state.lock().map_err(|_| OutboxError::InvalidState)?;
        if state.pending.len() >= MAX_OUTBOX_RECORDS {
            return Err(OutboxError::CapacityExceeded);
        }
        let sequence = state
            .position
            .confirmed_sequence
            .checked_add(
                u64::try_from(state.pending.len()).map_err(|_| OutboxError::CapacityExceeded)?,
            )
            .and_then(|sequence| sequence.checked_add(1))
            .ok_or(OutboxError::CapacityExceeded)?;
        let frame = build(sequence).map_err(OutboxError::Protocol)?;
        frame
            .verify_outbound(
                self.identity.node_id(),
                self.identity.session_id(),
                self.identity.key_id(),
                self.identity.public_key(),
            )
            .map_err(OutboxError::Protocol)?;
        if frame.outbound_sequence() != Some(sequence) {
            return Err(OutboxError::InvalidState);
        }
        let frame_bytes = frame.encode().map_err(OutboxError::Protocol)?;
        let digest = sha256(&frame_bytes);
        let stored = StoredOutbound {
            schema: OUTBOX_SCHEMA.to_owned(),
            sequence,
            digest,
            frame_bytes,
        };
        let bytes = postcard::to_stdvec(&stored).map_err(|_| OutboxError::InvalidState)?;
        if bytes.is_empty() || bytes.len() as u64 > MAX_OUTBOX_RECORD_BYTES {
            return Err(OutboxError::CapacityExceeded);
        }
        let path = self.root.join(record_name(sequence, digest));
        let pending_path = self.root.join(ENQUEUE_PENDING_FILE);
        write_new_synced(&pending_path, &bytes)?;
        fs::rename(&pending_path, &path).map_err(OutboxError::Io)?;
        sync_directory(&self.root)?;
        let pending = OutboundPending {
            sequence,
            digest,
            frame,
        };
        let record = PendingRecord {
            pending: pending.clone(),
            path,
        };
        if state.pending.insert(sequence, record).is_some() {
            return Err(OutboxError::Conflict);
        }
        Ok(pending)
    }
}

#[async_trait]
impl EventSink for DurableControlOutbox {
    async fn emit(&self, event: NodeEvent) -> Result<(), ExecutionError> {
        self.enqueue_event(event)
            .map(|_| ())
            .map_err(|_| ExecutionError::OutboundControlUnavailable)
    }
}

fn validate_or_create_root(root: &Path) -> Result<(), OutboxError> {
    if !root.is_absolute() {
        return Err(OutboxError::UnsafeRoot);
    }
    if !root.exists() {
        DirBuilder::new()
            .mode(0o700)
            .create(root)
            .map_err(OutboxError::Io)?;
        sync_directory(root.parent().ok_or(OutboxError::UnsafeRoot)?)?;
    }
    let metadata = fs::symlink_metadata(root).map_err(OutboxError::Io)?;
    if !metadata.is_dir()
        || metadata.file_type().is_symlink()
        || metadata.uid() != Uid::current().as_raw()
        || metadata.permissions().mode() & 0o777 != 0o700
        || root.canonicalize().map_err(OutboxError::Io)? != root
    {
        return Err(OutboxError::UnsafeRoot);
    }
    Ok(())
}

fn read_position(root: &Path) -> Result<OutboxPosition, OutboxError> {
    let path = root.join(POSITION_FILE);
    if !path.exists() {
        return Ok(OutboxPosition::default());
    }
    let bytes = read_physical_file(&path, 4096)?;
    let position: OutboxPosition =
        postcard::from_bytes(&bytes).map_err(|_| OutboxError::InvalidState)?;
    if position.schema != POSITION_SCHEMA {
        return Err(OutboxError::InvalidState);
    }
    if position.checksum
        != position_checksum(position.confirmed_sequence, position.confirmed_digest)
    {
        return Err(OutboxError::InvalidState);
    }
    Ok(position)
}

fn persist_position(root: &Path, position: &OutboxPosition) -> Result<(), OutboxError> {
    let bytes = postcard::to_stdvec(position).map_err(|_| OutboxError::InvalidState)?;
    let pending = root.join(POSITION_PENDING_FILE);
    write_new_synced(&pending, &bytes)?;
    fs::rename(&pending, root.join(POSITION_FILE)).map_err(OutboxError::Io)?;
    sync_directory(root)
}

fn validate_record(
    stored: StoredOutbound,
    expected_sequence: u64,
    expected_digest: Digest32,
    identity: &NodeRuntimeIdentity,
    path: PathBuf,
) -> Result<PendingRecord, OutboxError> {
    if stored.schema != OUTBOX_SCHEMA
        || stored.sequence != expected_sequence
        || stored.sequence == 0
        || stored.digest != expected_digest
        || stored.digest == [0; 32]
        || sha256(&stored.frame_bytes) != stored.digest
    {
        return Err(OutboxError::InvalidState);
    }
    let frame = NodeControlFrame::decode(&stored.frame_bytes).map_err(OutboxError::Protocol)?;
    if frame.outbound_sequence() != Some(stored.sequence) {
        return Err(OutboxError::InvalidState);
    }
    frame
        .verify_outbound(
            identity.node_id(),
            identity.session_id(),
            identity.key_id(),
            identity.public_key(),
        )
        .map_err(OutboxError::Protocol)?;
    Ok(PendingRecord {
        pending: OutboundPending {
            sequence: stored.sequence,
            digest: stored.digest,
            frame,
        },
        path,
    })
}

fn validate_sequence_continuity(
    position: &OutboxPosition,
    pending: &BTreeMap<u64, PendingRecord>,
) -> Result<(), OutboxError> {
    let mut expected = position
        .confirmed_sequence
        .checked_add(1)
        .ok_or(OutboxError::InvalidState)?;
    for sequence in pending.keys() {
        if *sequence != expected {
            return Err(OutboxError::InvalidState);
        }
        expected = expected.checked_add(1).ok_or(OutboxError::InvalidState)?;
    }
    Ok(())
}

fn record_name(sequence: u64, digest: Digest32) -> String {
    format!("{sequence:020}-{}.outbound", hex_digest(digest))
}

fn parse_record_name(name: &str) -> Result<(u64, Digest32), OutboxError> {
    let base = name
        .strip_suffix(".outbound")
        .ok_or(OutboxError::InvalidState)?;
    let (sequence, digest) = base.split_once('-').ok_or(OutboxError::InvalidState)?;
    if sequence.len() != 20 || digest.len() != 64 {
        return Err(OutboxError::InvalidState);
    }
    let sequence = sequence
        .parse::<u64>()
        .map_err(|_| OutboxError::InvalidState)?;
    let digest = parse_hex_digest(digest)?;
    if record_name(sequence, digest) != name {
        return Err(OutboxError::InvalidState);
    }
    Ok((sequence, digest))
}

fn read_physical_file(path: &Path, maximum_bytes: u64) -> Result<Vec<u8>, OutboxError> {
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_CLOEXEC)
        .open(path)
        .map_err(OutboxError::Io)?;
    let opened = file.metadata().map_err(OutboxError::Io)?;
    let named = fs::symlink_metadata(path).map_err(OutboxError::Io)?;
    if !opened.is_file()
        || named.file_type().is_symlink()
        || opened.dev() != named.dev()
        || opened.ino() != named.ino()
        || opened.uid() != Uid::current().as_raw()
        || opened.nlink() != 1
        || opened.permissions().mode() & 0o777 != 0o600
        || opened.len() == 0
        || opened.len() > maximum_bytes
    {
        return Err(OutboxError::InvalidState);
    }
    let mut bytes = Vec::with_capacity(opened.len() as usize);
    std::io::Read::by_ref(&mut file)
        .take(maximum_bytes + 1)
        .read_to_end(&mut bytes)
        .map_err(OutboxError::Io)?;
    if bytes.len() as u64 != opened.len() {
        return Err(OutboxError::InvalidState);
    }
    Ok(bytes)
}

fn write_new_synced(path: &Path, bytes: &[u8]) -> Result<(), OutboxError> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_CLOEXEC)
        .open(path)
        .map_err(OutboxError::Io)?;
    file.write_all(bytes).map_err(OutboxError::Io)?;
    file.sync_all().map_err(OutboxError::Io)
}

fn sync_directory(path: &Path) -> Result<(), OutboxError> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(OutboxError::Io)
}

fn hex_digest(digest: Digest32) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn position_checksum(sequence: u64, digest: Digest32) -> Digest32 {
    let mut bytes = Vec::with_capacity(POSITION_CHECKSUM_DOMAIN.len() + 8 + digest.len());
    bytes.extend_from_slice(POSITION_CHECKSUM_DOMAIN);
    bytes.extend_from_slice(&sequence.to_le_bytes());
    bytes.extend_from_slice(&digest);
    sha256(&bytes)
}

fn parse_hex_digest(value: &str) -> Result<Digest32, OutboxError> {
    let mut digest = [0_u8; 32];
    for (index, part) in value.as_bytes().chunks_exact(2).enumerate() {
        let part = std::str::from_utf8(part).map_err(|_| OutboxError::InvalidState)?;
        digest[index] = u8::from_str_radix(part, 16).map_err(|_| OutboxError::InvalidState)?;
    }
    Ok(digest)
}

#[derive(Debug, Error)]
pub enum OutboxError {
    #[error("outbox root failed physical custody validation")]
    UnsafeRoot,
    #[error("outbox state is incomplete, corrupt, or non-contiguous")]
    InvalidState,
    #[error("outbox reached its bounded record capacity")]
    CapacityExceeded,
    #[error("outbox confirmation or sequence conflicts with durable state")]
    Conflict,
    #[error("outbound protocol rejected a frame: {0}")]
    Protocol(ProtocolError),
    #[error("outbox I/O failed: {0}")]
    Io(std::io::Error),
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::{symlink, PermissionsExt};

    use ed25519_dalek::SigningKey;

    use super::*;
    use crate::{EventIdentity, ExecutionPhase};

    fn hub_key() -> SigningKey {
        SigningKey::from_bytes(&[73; 32])
    }

    fn identity() -> NodeRuntimeIdentity {
        NodeRuntimeIdentity::new(
            "node-1".to_owned(),
            "session-1".to_owned(),
            "node-key-1".to_owned(),
            "hub-issuer".to_owned(),
            hub_key().verifying_key().to_bytes(),
            "image-issuer".to_owned(),
            [4; 32],
            SigningKey::from_bytes(&[67; 32]),
        )
        .unwrap()
    }

    fn confirmation(pending: &OutboundPending, at_ms: u64) -> NodeControlFrame {
        NodeControlFrame::outbound_confirmation(
            "session-1".to_owned(),
            pending.sequence,
            pending.digest,
            at_ms,
            "hub-issuer".to_owned(),
            "hub-key-1".to_owned(),
            &hub_key(),
        )
        .unwrap()
    }

    fn event(cursor: u64) -> NodeEvent {
        NodeEvent::Progress {
            identity: EventIdentity {
                run_id: "run-1".to_owned(),
                attempt: 1,
                fence: 2,
                cursor,
                stream_offset: 0,
                control_position: 4,
                phase: ExecutionPhase::Run,
            },
            completed: cursor,
            total: Some(10),
            message: "running".to_owned(),
        }
    }

    #[test]
    fn event_is_durable_before_return_and_replays_until_exact_confirmation() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("outbox");
        let identity = identity();
        let outbox = DurableControlOutbox::open(path.clone(), identity.clone()).unwrap();
        let first = outbox.enqueue_event(event(1)).unwrap();
        let second = outbox.enqueue_event(event(2)).unwrap();
        assert_eq!(first.sequence, 1);
        assert_eq!(second.sequence, 2);
        assert_eq!(outbox.oldest().unwrap(), Some(first.clone()));
        drop(outbox);

        let outbox = DurableControlOutbox::open(path.clone(), identity.clone()).unwrap();
        assert_eq!(outbox.pending_count().unwrap(), 2);
        assert!(matches!(
            outbox.retire_exact(first.sequence, [9; 32]),
            Err(OutboxError::Conflict)
        ));
        assert!(matches!(
            outbox.retire_exact(second.sequence, second.digest),
            Err(OutboxError::Conflict)
        ));
        assert_eq!(
            outbox.retire_exact(first.sequence, first.digest).unwrap(),
            OutboundConfirmation::Retired
        );
        assert_eq!(outbox.oldest().unwrap(), Some(second.clone()));
        drop(outbox);

        let outbox = DurableControlOutbox::open(path, identity).unwrap();
        assert_eq!(outbox.oldest().unwrap(), Some(second));
    }

    #[tokio::test]
    async fn event_sink_only_returns_after_a_signed_record_is_pending() {
        let root = tempfile::tempdir().unwrap();
        let outbox = DurableControlOutbox::open(root.path().join("outbox"), identity()).unwrap();
        outbox.emit(event(1)).await.unwrap();
        let pending = outbox.oldest().unwrap().unwrap();
        assert_eq!(pending.sequence, 1);
        pending
            .frame
            .verify_outbound("node-1", "session-1", "node-key-1", identity().public_key())
            .unwrap();
    }

    #[test]
    fn confirmed_position_recovers_an_exact_unremoved_record() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("outbox");
        let identity = identity();
        let outbox = DurableControlOutbox::open(path.clone(), identity.clone()).unwrap();
        let pending = outbox.enqueue_event(event(1)).unwrap();
        persist_position(
            &path,
            &OutboxPosition {
                schema: POSITION_SCHEMA.to_owned(),
                confirmed_sequence: pending.sequence,
                confirmed_digest: pending.digest,
                checksum: position_checksum(pending.sequence, pending.digest),
            },
        )
        .unwrap();
        drop(outbox);

        let reopened = DurableControlOutbox::open(path, identity).unwrap();
        assert_eq!(reopened.pending_count().unwrap(), 0);
    }

    #[test]
    fn controller_confirmation_is_verified_durable_and_exactly_idempotent() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("outbox");
        let identity = identity();
        let outbox = DurableControlOutbox::open(path.clone(), identity.clone()).unwrap();
        let first = outbox.enqueue_event(event(1)).unwrap();
        let second = outbox.enqueue_event(event(2)).unwrap();
        let signed = confirmation(&first, 50_000);
        assert_eq!(
            outbox.confirm_controller(&signed, 50_010, 100).unwrap(),
            OutboundConfirmation::Retired
        );
        drop(outbox);

        let reopened = DurableControlOutbox::open(path, identity).unwrap();
        assert_eq!(
            reopened.confirm_controller(&signed, 50_020, 100).unwrap(),
            OutboundConfirmation::AlreadyConfirmed
        );
        assert_eq!(reopened.oldest().unwrap(), Some(second.clone()));
        assert!(matches!(
            reopened.confirm_controller(&confirmation(&second, 49_899), 50_000, 100),
            Err(OutboxError::Protocol(ProtocolError::InvalidOutbound))
        ));

        let wrong = NodeControlFrame::outbound_confirmation(
            "session-1".to_owned(),
            second.sequence,
            [8; 32],
            50_000,
            "hub-issuer".to_owned(),
            "hub-key-1".to_owned(),
            &hub_key(),
        )
        .unwrap();
        assert!(matches!(
            reopened.confirm_controller(&wrong, 50_000, 100),
            Err(OutboxError::Conflict)
        ));
        assert_eq!(reopened.oldest().unwrap(), Some(second));
    }

    #[test]
    fn gaps_corruption_pending_files_and_links_fail_closed() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("outbox");
        let identity = identity();
        let outbox = DurableControlOutbox::open(path.clone(), identity.clone()).unwrap();
        let first = outbox.enqueue_event(event(1)).unwrap();
        let _second = outbox.enqueue_event(event(2)).unwrap();
        let first_path = path.join(record_name(first.sequence, first.digest));
        fs::remove_file(&first_path).unwrap();
        drop(outbox);
        assert!(DurableControlOutbox::open(path.clone(), identity.clone()).is_err());

        let corrupt_root = root.path().join("corrupt-outbox");
        let outbox = DurableControlOutbox::open(corrupt_root.clone(), identity.clone()).unwrap();
        let pending = outbox.enqueue_event(event(1)).unwrap();
        let corrupt_record = corrupt_root.join(record_name(pending.sequence, pending.digest));
        fs::write(&corrupt_record, b"corrupt").unwrap();
        fs::set_permissions(&corrupt_record, fs::Permissions::from_mode(0o600)).unwrap();
        drop(outbox);
        assert!(DurableControlOutbox::open(corrupt_root, identity.clone()).is_err());

        let separate = root.path().join("linked-outbox");
        let outbox = DurableControlOutbox::open(separate.clone(), identity.clone()).unwrap();
        let pending = outbox.enqueue_event(event(1)).unwrap();
        let record = separate.join(record_name(pending.sequence, pending.digest));
        fs::hard_link(&record, separate.join("linked.outbound")).unwrap();
        drop(outbox);
        assert!(DurableControlOutbox::open(separate, identity.clone()).is_err());

        let pending_root = root.path().join("pending-outbox");
        let outbox = DurableControlOutbox::open(pending_root.clone(), identity.clone()).unwrap();
        drop(outbox);
        fs::write(pending_root.join(ENQUEUE_PENDING_FILE), b"partial").unwrap();
        fs::set_permissions(
            pending_root.join(ENQUEUE_PENDING_FILE),
            fs::Permissions::from_mode(0o600),
        )
        .unwrap();
        assert!(DurableControlOutbox::open(pending_root, identity.clone()).is_err());

        let bad_mode_root = root.path().join("bad-mode-outbox");
        fs::create_dir(&bad_mode_root).unwrap();
        fs::set_permissions(&bad_mode_root, fs::Permissions::from_mode(0o755)).unwrap();
        assert!(DurableControlOutbox::open(bad_mode_root, identity.clone()).is_err());

        let symlink_root = root.path().join("symlink-outbox");
        symlink(&path, &symlink_root).unwrap();
        assert!(DurableControlOutbox::open(symlink_root, identity).is_err());
    }
}
