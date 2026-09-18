use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::Path,
};

use jain_smartcluster_fabric::{sha256, valid_identifier, Digest32};
use nix::unistd::Uid;
use serde::{Deserialize, Serialize};

use crate::ProtocolError;

const SESSION_STATE_SCHEMA: &str = "jain.node.session-state/v1";
const SESSION_STATE_FILE: &str = "session-control.state";
const SESSION_STATE_PENDING: &str = "session-control.pending";
const SESSION_STATE_DOMAIN: &[u8] = b"jain.node.session-state.v1\0";
const MAX_SESSION_STATE_BYTES: u64 = 4096;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct DurableSessionState {
    schema: String,
    pub(crate) node_id: String,
    pub(crate) initial_session_id: String,
    pub(crate) current_session_id: String,
    pub(crate) control_position: u64,
    pub(crate) revoked: bool,
    checksum: Digest32,
}

impl DurableSessionState {
    fn initial(node_id: &str, session_id: &str) -> Result<Self, ProtocolError> {
        if !valid_identifier(node_id) || !valid_identifier(session_id) {
            return Err(ProtocolError::InvalidConfiguration);
        }
        let mut state = Self {
            schema: SESSION_STATE_SCHEMA.to_owned(),
            node_id: node_id.to_owned(),
            initial_session_id: session_id.to_owned(),
            current_session_id: session_id.to_owned(),
            control_position: 0,
            revoked: false,
            checksum: [0; 32],
        };
        state.checksum = state_checksum(&state)?;
        Ok(state)
    }

    pub(crate) fn successor(
        &self,
        sequence: u64,
        next_session_id: String,
        revoked: bool,
    ) -> Result<Self, ProtocolError> {
        if sequence == 0
            || sequence <= self.control_position
            || !valid_identifier(&next_session_id)
            || (self.revoked && !revoked)
        {
            return Err(ProtocolError::ReplayedPosition);
        }
        let mut next = self.clone();
        next.current_session_id = next_session_id;
        next.control_position = sequence;
        next.revoked = revoked;
        next.checksum = state_checksum(&next)?;
        Ok(next)
    }

    fn validate(&self, node_id: &str, initial_session_id: &str) -> Result<(), ProtocolError> {
        if self.schema != SESSION_STATE_SCHEMA
            || self.node_id != node_id
            || self.initial_session_id != initial_session_id
            || !valid_identifier(&self.node_id)
            || !valid_identifier(&self.initial_session_id)
            || !valid_identifier(&self.current_session_id)
            || self.checksum != state_checksum(self)?
        {
            return Err(ProtocolError::InvalidState);
        }
        Ok(())
    }
}

pub(crate) fn load_or_create(
    root: &Path,
    node_id: &str,
    initial_session_id: &str,
) -> Result<DurableSessionState, ProtocolError> {
    let path = root.join(SESSION_STATE_FILE);
    let pending = root.join(SESSION_STATE_PENDING);
    if pending.exists() {
        return Err(ProtocolError::InvalidState);
    }
    if !path.exists() {
        let state = DurableSessionState::initial(node_id, initial_session_id)?;
        persist(root, &state)?;
        return Ok(state);
    }
    let bytes = read_physical(&path)?;
    let state: DurableSessionState =
        postcard::from_bytes(&bytes).map_err(|_| ProtocolError::InvalidState)?;
    state.validate(node_id, initial_session_id)?;
    Ok(state)
}

pub(crate) fn persist(root: &Path, state: &DurableSessionState) -> Result<(), ProtocolError> {
    state.validate(&state.node_id, &state.initial_session_id)?;
    let bytes = postcard::to_stdvec(state).map_err(|_| ProtocolError::InvalidState)?;
    if bytes.is_empty() || bytes.len() as u64 > MAX_SESSION_STATE_BYTES {
        return Err(ProtocolError::InvalidState);
    }
    let pending = root.join(SESSION_STATE_PENDING);
    let destination = root.join(SESSION_STATE_FILE);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_CLOEXEC)
        .open(&pending)
        .map_err(ProtocolError::Io)?;
    file.write_all(&bytes).map_err(ProtocolError::Io)?;
    file.sync_all().map_err(ProtocolError::Io)?;
    fs::rename(&pending, &destination).map_err(ProtocolError::Io)?;
    File::open(root)
        .and_then(|directory| directory.sync_all())
        .map_err(ProtocolError::Io)
}

fn read_physical(path: &Path) -> Result<Vec<u8>, ProtocolError> {
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_CLOEXEC)
        .open(path)
        .map_err(ProtocolError::Io)?;
    let opened = file.metadata().map_err(ProtocolError::Io)?;
    let named = fs::symlink_metadata(path).map_err(ProtocolError::Io)?;
    if !opened.is_file()
        || named.file_type().is_symlink()
        || opened.dev() != named.dev()
        || opened.ino() != named.ino()
        || opened.uid() != Uid::effective().as_raw()
        || opened.nlink() != 1
        || opened.mode() & 0o777 != 0o600
        || opened.len() == 0
        || opened.len() > MAX_SESSION_STATE_BYTES
    {
        return Err(ProtocolError::InvalidState);
    }
    let mut bytes = Vec::new();
    Read::by_ref(&mut file)
        .take(MAX_SESSION_STATE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(ProtocolError::Io)?;
    if bytes.len() as u64 != opened.len() || bytes.len() as u64 > MAX_SESSION_STATE_BYTES {
        return Err(ProtocolError::InvalidState);
    }
    Ok(bytes)
}

fn state_checksum(state: &DurableSessionState) -> Result<Digest32, ProtocolError> {
    let payload = postcard::to_stdvec(&(
        state.schema.as_str(),
        state.node_id.as_str(),
        state.initial_session_id.as_str(),
        state.current_session_id.as_str(),
        state.control_position,
        state.revoked,
    ))
    .map_err(|_| ProtocolError::InvalidState)?;
    let mut bytes = Vec::with_capacity(SESSION_STATE_DOMAIN.len() + payload.len());
    bytes.extend_from_slice(SESSION_STATE_DOMAIN);
    bytes.extend_from_slice(&payload);
    Ok(sha256(&bytes))
}

#[cfg(test)]
mod tests {
    use std::{fs, os::unix::fs::PermissionsExt};

    use ed25519_dalek::SigningKey;
    use jain_smartcluster_fabric::{DomainSignature, SignatureAuthority};

    use super::*;
    use crate::{NodeControlFrame, NodeProfile, NodeProtocolState};

    #[test]
    fn rotation_and_revocation_survive_restart_as_one_position() {
        let root = tempfile::tempdir().unwrap();
        let state_root = root.path().join("protocol");
        let key = SigningKey::from_bytes(&[83; 32]);
        let open = || {
            NodeProtocolState::open(
                state_root.clone(),
                "node-1".to_owned(),
                "session-1".to_owned(),
                NodeProfile::Cpu,
                crate::WorkerProvision::CodeRunner,
                "hub-issuer".to_owned(),
                key.verifying_key().to_bytes(),
                "image-issuer".to_owned(),
                [2; 32],
                2,
            )
        };
        let mut state = open().unwrap();
        let mut rotate = NodeControlFrame::Rotate {
            session_id: "session-1".to_owned(),
            sequence: 5,
            next_session_id: "session-2".to_owned(),
            expires_at_ms: 100,
            signature: DomainSignature {
                authority: SignatureAuthority::NodeSession,
                issuer: "hub-issuer".to_owned(),
                key_id: "hub-key".to_owned(),
                public_key: key.verifying_key().to_bytes(),
                signature: Vec::new(),
            },
        };
        sign_control(&mut rotate, &key);
        state.accept_session_control(&rotate, 10).unwrap();
        assert_eq!(state.session_id(), "session-2");
        drop(state);

        let mut state = open().unwrap();
        assert_eq!(state.session_id(), "session-2");
        assert!(!state.revoked());
        assert!(state.accept_session_control(&rotate, 10).is_err());
        let mut revoke = NodeControlFrame::Revoke {
            session_id: "session-2".to_owned(),
            sequence: 6,
            reason: "owner revocation".to_owned(),
            signature: DomainSignature {
                authority: SignatureAuthority::NodeSession,
                issuer: "hub-issuer".to_owned(),
                key_id: "hub-key".to_owned(),
                public_key: key.verifying_key().to_bytes(),
                signature: Vec::new(),
            },
        };
        sign_control(&mut revoke, &key);
        state.accept_session_control(&revoke, 10).unwrap();
        assert!(state.revoked());
        drop(state);

        let state = open().unwrap();
        assert_eq!(state.session_id(), "session-2");
        assert!(state.revoked());
    }

    #[test]
    fn corrupt_pending_linked_and_wrong_mode_state_fail_closed() {
        let root = tempfile::tempdir().unwrap();
        let state_root = root.path().join("protocol");
        let open = || load_or_create(&state_root, "node-1", "session-1");
        fs::create_dir(&state_root).unwrap();
        fs::set_permissions(&state_root, fs::Permissions::from_mode(0o700)).unwrap();
        open().unwrap();
        let state_file = state_root.join(SESSION_STATE_FILE);
        fs::hard_link(&state_file, state_root.join("state-copy")).unwrap();
        assert!(open().is_err());
        fs::remove_file(state_root.join("state-copy")).unwrap();

        fs::set_permissions(&state_file, fs::Permissions::from_mode(0o644)).unwrap();
        assert!(open().is_err());
        fs::set_permissions(&state_file, fs::Permissions::from_mode(0o600)).unwrap();
        fs::write(&state_file, b"corrupt").unwrap();
        assert!(open().is_err());

        let pending_root = root.path().join("pending-protocol");
        fs::create_dir(&pending_root).unwrap();
        fs::set_permissions(&pending_root, fs::Permissions::from_mode(0o700)).unwrap();
        fs::write(pending_root.join(SESSION_STATE_PENDING), b"partial").unwrap();
        fs::set_permissions(
            pending_root.join(SESSION_STATE_PENDING),
            fs::Permissions::from_mode(0o600),
        )
        .unwrap();
        assert!(load_or_create(&pending_root, "node-1", "session-1").is_err());
    }

    fn sign_control(frame: &mut NodeControlFrame, key: &SigningKey) {
        let signed = frame.signed_control_bytes().unwrap();
        let signature = DomainSignature::sign(
            SignatureAuthority::NodeSession,
            "hub-issuer".to_owned(),
            "hub-key".to_owned(),
            key,
            &signed,
        );
        match frame {
            NodeControlFrame::Rotate {
                signature: target, ..
            }
            | NodeControlFrame::Revoke {
                signature: target, ..
            } => *target = signature,
            _ => unreachable!(),
        }
    }
}
