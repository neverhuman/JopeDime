use std::{
    collections::BTreeMap,
    fs::{self, DirBuilder, File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
    sync::Mutex,
};

use jain_smartcluster_fabric::{sha256, Digest32, WorkerAcknowledgement};
use nix::unistd::Uid;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    NodeControlFrame, NodeEvent, NodeRuntimeIdentity, NodeTerminalOutcome, ProtocolError,
    SignedStorageFragmentReceipt, SignedStorageTelemetry,
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

mod operations;

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
mod tests;
