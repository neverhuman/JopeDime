use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
};

use jain_smartcluster_fabric::{
    valid_identifier, Digest32, DomainSignature, Placement, SignatureAuthority,
    SignedWorkUnitProjection, VerifiedWorkUnit, WorkerAcknowledgement, WorkerProvision,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

const MAX_CONTROL_FRAME_BYTES: usize = 1024 * 1024;
const ENROLLMENT_DOMAIN: &[u8] = b"jain.node.enrollment-ticket.v1\0";
const CANCELLATION_DOMAIN: &[u8] = b"jain.node.cancellation.v1\0";
const SESSION_ROTATION_DOMAIN: &[u8] = b"jain.node.session-rotation.v1\0";
const SESSION_REVOCATION_DOMAIN: &[u8] = b"jain.node.session-revocation.v1\0";
const SESSION_HELLO_DOMAIN: &[u8] = b"jain.node.session-hello.v1\0";

#[path = "protocol_session.rs"]
mod session;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeProfile {
    Cpu,
    Gpu,
    StorageOnly,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct EnrollmentTicket {
    pub node_id: String,
    pub nonce: Digest32,
    pub issued_at_ms: u64,
    pub expires_at_ms: u64,
    pub signature: DomainSignature,
}

impl EnrollmentTicket {
    pub fn signed_bytes(&self) -> Result<Vec<u8>, ProtocolError> {
        let mut unsigned = self.clone();
        unsigned.signature.signature.clear();
        unsigned.signature.issuer.clear();
        unsigned.signature.key_id.clear();
        unsigned.signature.public_key = [0; 32];
        let payload = postcard::to_stdvec(&unsigned).map_err(|_| ProtocolError::MalformedFrame)?;
        let mut bytes = Vec::with_capacity(ENROLLMENT_DOMAIN.len() + payload.len());
        bytes.extend_from_slice(ENROLLMENT_DOMAIN);
        bytes.extend_from_slice(&payload);
        Ok(bytes)
    }

    fn verify(
        &self,
        expected_node_id: &str,
        issuer: &str,
        hub_public_key: Digest32,
        now_ms: u64,
    ) -> Result<(), ProtocolError> {
        if self.node_id != expected_node_id
            || self.nonce == [0; 32]
            || self.issued_at_ms == 0
            || self.expires_at_ms <= now_ms
            || self.issued_at_ms >= self.expires_at_ms
        {
            return Err(ProtocolError::InvalidEnrollment);
        }
        self.signature
            .verify_exact(
                SignatureAuthority::NodeEnrollment,
                issuer,
                hub_public_key,
                &self.signed_bytes()?,
            )
            .map_err(|_| ProtocolError::InvalidEnrollment)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeControlFrame {
    Enroll {
        ticket: EnrollmentTicket,
    },
    SessionHello {
        node_id: String,
        session_id: String,
        profile: NodeProfile,
        provision: WorkerProvision,
        qualification_digest: Digest32,
        sequence: u64,
        at_ms: u64,
        signature: DomainSignature,
    },
    Infer {
        request: Box<InferRequest>,
    },
    Cancel {
        session_id: String,
        sequence: u64,
        run_id: String,
        attempt: u32,
        attempt_fence: u64,
        signature: DomainSignature,
    },
    Rotate {
        session_id: String,
        sequence: u64,
        next_session_id: String,
        expires_at_ms: u64,
        signature: DomainSignature,
    },
    Revoke {
        session_id: String,
        sequence: u64,
        reason: String,
        signature: DomainSignature,
    },
    AssignmentAccepted {
        node_id: String,
        session_id: String,
        outbound_sequence: u64,
        acknowledgement: Box<WorkerAcknowledgement>,
        signature: DomainSignature,
    },
    Event {
        node_id: String,
        session_id: String,
        outbound_sequence: u64,
        event: Box<crate::NodeEvent>,
        signature: DomainSignature,
    },
    Terminal {
        node_id: String,
        session_id: String,
        outbound_sequence: u64,
        run_id: String,
        attempt: u32,
        attempt_fence: u64,
        control_position: u64,
        at_ms: u64,
        outcome: Box<crate::NodeTerminalOutcome>,
        signature: DomainSignature,
    },
    ConfirmOutbound {
        session_id: String,
        outbound_sequence: u64,
        frame_digest: Digest32,
        at_ms: u64,
        signature: DomainSignature,
    },
    StorageControl {
        request: Box<crate::SignedStorageControlRequest>,
    },
    StorageTelemetry {
        node_id: String,
        session_id: String,
        outbound_sequence: u64,
        telemetry: Box<crate::SignedStorageTelemetry>,
        signature: DomainSignature,
    },
    StorageFragmentReceipt {
        node_id: String,
        session_id: String,
        outbound_sequence: u64,
        receipt: Box<crate::SignedStorageFragmentReceipt>,
        signature: DomainSignature,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct InferRequest {
    pub session_id: String,
    pub sequence: u64,
    pub deadline_ms: u64,
    pub work: SignedWorkUnitProjection,
    pub placement: Placement,
}

impl NodeControlFrame {
    pub fn decode(bytes: &[u8]) -> Result<Self, ProtocolError> {
        if bytes.is_empty() || bytes.len() > MAX_CONTROL_FRAME_BYTES {
            return Err(ProtocolError::MalformedFrame);
        }
        postcard::from_bytes(bytes).map_err(|_| ProtocolError::MalformedFrame)
    }

    pub fn encode(&self) -> Result<Vec<u8>, ProtocolError> {
        let bytes = postcard::to_stdvec(self).map_err(|_| ProtocolError::MalformedFrame)?;
        if bytes.len() > MAX_CONTROL_FRAME_BYTES {
            return Err(ProtocolError::MalformedFrame);
        }
        Ok(bytes)
    }

    pub fn signed_control_bytes(&self) -> Result<Vec<u8>, ProtocolError> {
        let (domain, mut unsigned) = match self {
            Self::Cancel { .. } => (CANCELLATION_DOMAIN, self.clone()),
            Self::Rotate { .. } => (SESSION_ROTATION_DOMAIN, self.clone()),
            Self::Revoke { .. } => (SESSION_REVOCATION_DOMAIN, self.clone()),
            Self::ConfirmOutbound { .. } => (
                crate::confirmation::OUTBOUND_CONFIRMATION_DOMAIN,
                self.clone(),
            ),
            _ => return Err(ProtocolError::MalformedFrame),
        };
        match &mut unsigned {
            Self::Cancel { signature, .. }
            | Self::Rotate { signature, .. }
            | Self::Revoke { signature, .. }
            | Self::ConfirmOutbound { signature, .. } => {
                signature.issuer.clear();
                signature.key_id.clear();
                signature.public_key = [0; 32];
                signature.signature.clear();
            }
            _ => return Err(ProtocolError::MalformedFrame),
        }
        let payload = postcard::to_stdvec(&unsigned).map_err(|_| ProtocolError::MalformedFrame)?;
        let mut bytes = Vec::with_capacity(domain.len() + payload.len());
        bytes.extend_from_slice(domain);
        bytes.extend_from_slice(&payload);
        Ok(bytes)
    }
}

#[derive(Clone, Debug)]
pub struct AdmittedInfer {
    pub work: VerifiedWorkUnit,
    pub placement: Placement,
    pub sequence: u64,
    pub deadline_ms: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SessionControlAction {
    Rotate { next_session_id: String },
    Revoke,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedSessionControl {
    pub sequence: u64,
    pub current_session_id: String,
    pub action: SessionControlAction,
}

#[derive(Debug)]
pub struct NodeProtocolState {
    root: PathBuf,
    node_id: String,
    session_id: String,
    profile: NodeProfile,
    provision: WorkerProvision,
    issuer: String,
    hub_public_key: Digest32,
    image_issuer: String,
    image_public_key: Digest32,
    max_inflight: u32,
    inflight: u32,
    read_only_fuse_enabled: bool,
    session_state: crate::session_state::DurableSessionState,
    revoked: bool,
}

impl NodeProtocolState {
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        root: PathBuf,
        node_id: String,
        session_id: String,
        profile: NodeProfile,
        provision: WorkerProvision,
        issuer: String,
        hub_public_key: Digest32,
        image_issuer: String,
        image_public_key: Digest32,
        max_inflight: u32,
    ) -> Result<Self, ProtocolError> {
        if !root.is_absolute()
            || profile == NodeProfile::StorageOnly
            || !valid_identifier(&node_id)
            || !valid_identifier(&session_id)
            || !valid_identifier(&issuer)
            || !valid_identifier(&image_issuer)
            || hub_public_key == [0; 32]
            || image_public_key == [0; 32]
            || max_inflight == 0
        {
            return Err(ProtocolError::InvalidConfiguration);
        }
        fs::create_dir_all(&root).map_err(ProtocolError::Io)?;
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).map_err(ProtocolError::Io)?;
        if root.canonicalize().map_err(ProtocolError::Io)? != root || {
            let metadata = fs::symlink_metadata(&root).map_err(ProtocolError::Io)?;
            metadata.file_type().is_symlink() || !metadata.is_dir() || metadata.mode() & 0o022 != 0
        } {
            return Err(ProtocolError::InvalidConfiguration);
        }
        let session_state = crate::session_state::load_or_create(&root, &node_id, &session_id)?;
        let session_id = session_state.current_session_id.clone();
        let revoked = session_state.revoked;
        Ok(Self {
            root,
            node_id,
            session_id,
            profile,
            provision,
            issuer,
            hub_public_key,
            image_issuer,
            image_public_key,
            max_inflight,
            inflight: 0,
            read_only_fuse_enabled: false,
            session_state,
            revoked,
        })
    }

    pub fn read_only_fuse_enabled(&self) -> bool {
        self.read_only_fuse_enabled
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub const fn revoked(&self) -> bool {
        self.revoked
    }

    pub const fn control_position(&self) -> u64 {
        self.session_state.control_position
    }

    pub fn consume_enrollment(
        &mut self,
        ticket: &EnrollmentTicket,
        now_ms: u64,
    ) -> Result<(), ProtocolError> {
        ticket.verify(&self.node_id, &self.issuer, self.hub_public_key, now_ms)?;
        let consumed = self
            .root
            .join(format!("enrollment-{}.consumed", hex_digest(ticket.nonce)));
        write_new_synced(&self.root, &consumed, &ticket.signed_bytes()?).map_err(
            |error| match error {
                ProtocolError::Io(io) if io.kind() == std::io::ErrorKind::AlreadyExists => {
                    ProtocolError::EnrollmentReplayed
                }
                other => other,
            },
        )
    }

    pub fn verify_infer(
        &self,
        frame: NodeControlFrame,
        now_ms: u64,
    ) -> Result<AdmittedInfer, ProtocolError> {
        if self.revoked || self.profile == NodeProfile::StorageOnly {
            return Err(ProtocolError::SessionRevoked);
        }
        let NodeControlFrame::Infer { request } = frame else {
            return Err(ProtocolError::MalformedFrame);
        };
        let InferRequest {
            session_id,
            sequence,
            deadline_ms,
            work,
            placement,
        } = *request;
        if session_id != self.session_id
            || deadline_ms <= now_ms
            || deadline_ms > placement.lease_expires_at_ms
            || self.inflight >= self.max_inflight
        {
            return Err(ProtocolError::AdmissionDenied);
        }
        let work = VerifiedWorkUnit::verify(
            work,
            &self.issuer,
            self.hub_public_key,
            &self.image_issuer,
            self.image_public_key,
        )
        .map_err(|_| ProtocolError::InvalidWorkUnit)?;
        if !self.provision.admits(work.workload_class) {
            return Err(ProtocolError::AdmissionDenied);
        }
        if matches!(
            (self.profile, work.gpu.is_some()),
            (NodeProfile::Cpu, true) | (NodeProfile::Gpu, false)
        ) {
            return Err(ProtocolError::AdmissionDenied);
        }
        placement
            .verify_for(
                &work,
                &self.node_id,
                &self.session_id,
                self.hub_public_key,
                now_ms,
            )
            .map_err(|_| ProtocolError::InvalidWorkUnit)?;
        Ok(AdmittedInfer {
            work,
            placement,
            sequence,
            deadline_ms,
        })
    }

    pub fn commit_verified_infer(&mut self, admitted: &AdmittedInfer) -> Result<(), ProtocolError> {
        if self.revoked
            || self.profile == NodeProfile::StorageOnly
            || self.inflight >= self.max_inflight
        {
            return Err(ProtocolError::AdmissionDenied);
        }
        self.accept_sequence(admitted.sequence)?;
        self.inflight += 1;
        Ok(())
    }

    pub fn admit_infer(
        &mut self,
        frame: NodeControlFrame,
        now_ms: u64,
    ) -> Result<AdmittedInfer, ProtocolError> {
        let admitted = self.verify_infer(frame, now_ms)?;
        self.commit_verified_infer(&admitted)?;
        Ok(admitted)
    }

    pub fn complete_infer(&mut self) -> Result<(), ProtocolError> {
        self.inflight = self
            .inflight
            .checked_sub(1)
            .ok_or(ProtocolError::AdmissionDenied)?;
        Ok(())
    }

    pub fn restore_inflight(&mut self, count: usize) -> Result<(), ProtocolError> {
        let count = u32::try_from(count).map_err(|_| ProtocolError::AdmissionDenied)?;
        if self.inflight != 0 || count > self.max_inflight {
            return Err(ProtocolError::AdmissionDenied);
        }
        self.inflight = count;
        Ok(())
    }

    pub fn accept_cancel(
        &mut self,
        frame: &NodeControlFrame,
        work: &VerifiedWorkUnit,
    ) -> Result<(), ProtocolError> {
        let sequence = self.verify_cancel(frame, work)?;
        self.accept_sequence(sequence)
    }

    pub fn verify_cancel(
        &self,
        frame: &NodeControlFrame,
        work: &VerifiedWorkUnit,
    ) -> Result<u64, ProtocolError> {
        if self.revoked {
            return Err(ProtocolError::SessionRevoked);
        }
        let NodeControlFrame::Cancel {
            session_id,
            sequence,
            run_id,
            attempt,
            attempt_fence,
            signature,
        } = frame
        else {
            return Err(ProtocolError::MalformedFrame);
        };
        if session_id != &self.session_id
            || run_id != &work.run_id.to_string()
            || *attempt != work.attempt
            || *attempt_fence != work.attempt_fence
        {
            return Err(ProtocolError::FenceRejected);
        }
        signature
            .verify_exact(
                SignatureAuthority::NodeSession,
                &self.issuer,
                self.hub_public_key,
                &frame.signed_control_bytes()?,
            )
            .map_err(|_| ProtocolError::SessionRevoked)?;
        if *sequence == 0 || *sequence <= self.session_state.control_position {
            return Err(ProtocolError::ReplayedPosition);
        }
        Ok(*sequence)
    }

    pub fn accept_session_control(
        &mut self,
        frame: &NodeControlFrame,
        now_ms: u64,
    ) -> Result<(), ProtocolError> {
        let verified = self.verify_session_control(frame, now_ms)?;
        self.apply_verified_session_control(&verified)
    }

    pub fn verify_session_control(
        &self,
        frame: &NodeControlFrame,
        now_ms: u64,
    ) -> Result<VerifiedSessionControl, ProtocolError> {
        if self.revoked {
            return Err(ProtocolError::SessionRevoked);
        }
        let (session_id, sequence, signature, action) = match frame {
            NodeControlFrame::Rotate {
                session_id,
                sequence,
                next_session_id,
                expires_at_ms,
                signature,
            } => {
                if !valid_identifier(next_session_id) || *expires_at_ms <= now_ms {
                    return Err(ProtocolError::AdmissionDenied);
                }
                (
                    session_id,
                    *sequence,
                    signature,
                    SessionControlAction::Rotate {
                        next_session_id: next_session_id.clone(),
                    },
                )
            }
            NodeControlFrame::Revoke {
                session_id,
                sequence,
                reason,
                signature,
            } => {
                if reason.is_empty() || reason.len() > 1024 {
                    return Err(ProtocolError::AdmissionDenied);
                }
                (
                    session_id,
                    *sequence,
                    signature,
                    SessionControlAction::Revoke,
                )
            }
            _ => return Err(ProtocolError::MalformedFrame),
        };
        if session_id != &self.session_id {
            return Err(ProtocolError::SessionRevoked);
        }
        signature
            .verify_exact(
                SignatureAuthority::NodeSession,
                &self.issuer,
                self.hub_public_key,
                &frame.signed_control_bytes()?,
            )
            .map_err(|_| ProtocolError::SessionRevoked)?;
        if sequence == 0 || sequence <= self.session_state.control_position {
            return Err(ProtocolError::ReplayedPosition);
        }
        Ok(VerifiedSessionControl {
            sequence,
            current_session_id: self.session_id.clone(),
            action,
        })
    }

    pub fn apply_verified_session_control(
        &mut self,
        verified: &VerifiedSessionControl,
    ) -> Result<(), ProtocolError> {
        if verified.current_session_id != self.session_id {
            return Err(ProtocolError::SessionRevoked);
        }
        let (next_session_id, revoked) = match &verified.action {
            SessionControlAction::Rotate { next_session_id } => {
                (next_session_id.clone(), self.revoked)
            }
            SessionControlAction::Revoke => (self.session_id.clone(), true),
        };
        let next =
            self.session_state
                .successor(verified.sequence, next_session_id.clone(), revoked)?;
        crate::session_state::persist(&self.root, &next)?;
        self.session_state = next;
        self.session_id = next_session_id;
        self.revoked = revoked;
        Ok(())
    }

    pub fn commit_verified_sequence(&mut self, sequence: u64) -> Result<(), ProtocolError> {
        self.accept_sequence(sequence)
    }

    pub fn verify_storage_control(
        &self,
        frame: &NodeControlFrame,
        now_ms: u64,
    ) -> Result<u64, ProtocolError> {
        let NodeControlFrame::StorageControl { request } = frame else {
            return Err(ProtocolError::MalformedFrame);
        };
        request.verify(&self.session_id, &self.issuer, self.hub_public_key, now_ms)?;
        if self.revoked || request.sequence <= self.session_state.control_position {
            return Err(ProtocolError::ReplayedStorageRevision);
        }
        Ok(request.sequence)
    }

    fn accept_sequence(&mut self, sequence: u64) -> Result<(), ProtocolError> {
        if sequence == 0 || sequence <= self.session_state.control_position {
            return Err(ProtocolError::ReplayedPosition);
        }
        let next = self
            .session_state
            .successor(sequence, self.session_id.clone(), self.revoked)?;
        crate::session_state::persist(&self.root, &next)?;
        self.session_state = next;
        Ok(())
    }
}

fn write_new_synced(root: &Path, path: &Path, bytes: &[u8]) -> Result<(), ProtocolError> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(nix::libc::O_NOFOLLOW)
        .open(path)
        .map_err(ProtocolError::Io)?;
    file.write_all(bytes).map_err(ProtocolError::Io)?;
    file.sync_all().map_err(ProtocolError::Io)?;
    File::open(root)
        .and_then(|directory| directory.sync_all())
        .map_err(ProtocolError::Io)
}

fn hex_digest(digest: Digest32) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("invalid node protocol configuration")]
    InvalidConfiguration,
    #[error("malformed bounded control frame")]
    MalformedFrame,
    #[error("invalid enrollment ticket")]
    InvalidEnrollment,
    #[error("enrollment ticket was already consumed")]
    EnrollmentReplayed,
    #[error("session was revoked or cannot execute")]
    SessionRevoked,
    #[error("bounded admission denied")]
    AdmissionDenied,
    #[error("invalid signed work unit")]
    InvalidWorkUnit,
    #[error("replayed session position")]
    ReplayedPosition,
    #[error("cancellation fence was rejected")]
    FenceRejected,
    #[error("invalid durable protocol state")]
    InvalidState,
    #[error("invalid signed outbound frame")]
    InvalidOutbound,
    #[error("invalid signed storage telemetry or fragment receipt")]
    InvalidStorageFacts,
    #[error("invalid signed storage control request")]
    InvalidStorageControl,
    #[error("replayed storage revision or counter")]
    ReplayedStorageRevision,
    #[error("node protocol I/O failed: {0}")]
    Io(std::io::Error),
}

#[cfg(test)]
mod tests {
    use ed25519_dalek::SigningKey;

    use super::*;

    #[test]
    fn enrollment_is_signed_one_use_and_fuse_defaults_off() {
        let root = tempfile::tempdir().unwrap();
        let key = SigningKey::from_bytes(&[31; 32]);
        let mut ticket = EnrollmentTicket {
            node_id: "node-1".to_owned(),
            nonce: [7; 32],
            issued_at_ms: 1,
            expires_at_ms: 100,
            signature: DomainSignature {
                authority: SignatureAuthority::NodeEnrollment,
                issuer: "hub-issuer".to_owned(),
                key_id: "hub-key".to_owned(),
                public_key: key.verifying_key().to_bytes(),
                signature: Vec::new(),
            },
        };
        ticket.signature = DomainSignature::sign(
            SignatureAuthority::NodeEnrollment,
            "hub-issuer".to_owned(),
            "hub-key".to_owned(),
            &key,
            &ticket.signed_bytes().unwrap(),
        );
        let mut state = NodeProtocolState::open(
            root.path().to_path_buf(),
            "node-1".to_owned(),
            "session-1".to_owned(),
            NodeProfile::Cpu,
            WorkerProvision::CodeRunner,
            "hub-issuer".to_owned(),
            key.verifying_key().to_bytes(),
            "hub-issuer".to_owned(),
            key.verifying_key().to_bytes(),
            2,
        )
        .unwrap();
        assert!(!state.read_only_fuse_enabled());
        state.consume_enrollment(&ticket, 2).unwrap();
        assert!(matches!(
            state.consume_enrollment(&ticket, 2),
            Err(ProtocolError::EnrollmentReplayed)
        ));
    }

    #[test]
    fn control_frames_are_bounded_and_unknown_bytes_fail_closed() {
        assert!(matches!(
            NodeControlFrame::decode(&vec![0; MAX_CONTROL_FRAME_BYTES + 1]),
            Err(ProtocolError::MalformedFrame)
        ));
        assert!(matches!(
            NodeControlFrame::decode(&[255, 255, 255]),
            Err(ProtocolError::MalformedFrame)
        ));

        let root = tempfile::tempdir().unwrap();
        assert!(matches!(
            NodeProtocolState::open(
                root.path().to_path_buf(),
                "node-1".to_owned(),
                "../escape".to_owned(),
                NodeProfile::Cpu,
                WorkerProvision::CodeRunner,
                "hub-issuer".to_owned(),
                [1; 32],
                "image-issuer".to_owned(),
                [2; 32],
                1,
            ),
            Err(ProtocolError::InvalidConfiguration)
        ));
    }

    #[test]
    fn session_hello_is_domain_key_time_and_position_bound() {
        let key = SigningKey::from_bytes(&[43; 32]);
        let frame = NodeControlFrame::session_hello(
            "node-1".to_owned(),
            "session-1".to_owned(),
            NodeProfile::Gpu,
            WorkerProvision::DedicatedInference,
            [9; 32],
            4,
            50_000,
            "node-key-1".to_owned(),
            &key,
        )
        .unwrap();
        frame
            .verify_session_hello(
                "node-1",
                "session-1",
                NodeProfile::Gpu,
                WorkerProvision::DedicatedInference,
                [9; 32],
                "node-key-1",
                key.verifying_key().to_bytes(),
                50_010,
                100,
            )
            .unwrap();
        assert!(frame
            .verify_session_hello(
                "node-1",
                "session-1",
                NodeProfile::Gpu,
                WorkerProvision::CodeRunner,
                [9; 32],
                "node-key-1",
                key.verifying_key().to_bytes(),
                50_010,
                100,
            )
            .is_err());

        for (now_ms, key_id, public_key) in [
            (50_101, "node-key-1", key.verifying_key().to_bytes()),
            (50_010, "wrong-key", key.verifying_key().to_bytes()),
            (50_010, "node-key-1", [8; 32]),
        ] {
            assert!(frame
                .verify_session_hello(
                    "node-1",
                    "session-1",
                    NodeProfile::Gpu,
                    WorkerProvision::DedicatedInference,
                    [9; 32],
                    key_id,
                    public_key,
                    now_ms,
                    100,
                )
                .is_err());
        }

        let mut wrong_domain = frame.clone();
        let NodeControlFrame::SessionHello { signature, .. } = &mut wrong_domain else {
            unreachable!();
        };
        let payload_offset = SESSION_HELLO_DOMAIN.len();
        let canonical = frame.signed_session_hello_bytes().unwrap();
        let mut alternate = b"jain.node.wrong-session-hello.v1\0".to_vec();
        alternate.extend_from_slice(&canonical[payload_offset..]);
        *signature = DomainSignature::sign(
            SignatureAuthority::NodeSession,
            "node-1".to_owned(),
            "node-key-1".to_owned(),
            &key,
            &alternate,
        );
        assert!(wrong_domain
            .verify_session_hello(
                "node-1",
                "session-1",
                NodeProfile::Gpu,
                WorkerProvision::DedicatedInference,
                [9; 32],
                "node-key-1",
                key.verifying_key().to_bytes(),
                50_010,
                100,
            )
            .is_err());

        for (qualification, sequence, at_ms) in [([0; 32], 1, 1), ([1; 32], 0, 1), ([1; 32], 1, 0)]
        {
            assert!(NodeControlFrame::session_hello(
                "node-1".to_owned(),
                "session-1".to_owned(),
                NodeProfile::Cpu,
                WorkerProvision::CodeRunner,
                qualification,
                sequence,
                at_ms,
                "node-key-1".to_owned(),
                &key,
            )
            .is_err());
        }
    }

    #[test]
    fn cancellation_bytes_bind_the_fence_and_signature() {
        let key = SigningKey::from_bytes(&[41; 32]);
        let mut frame = NodeControlFrame::Cancel {
            session_id: "session-1".to_owned(),
            sequence: 9,
            run_id: "run-1".to_owned(),
            attempt: 2,
            attempt_fence: 7,
            signature: DomainSignature {
                authority: SignatureAuthority::NodeSession,
                issuer: "hub-issuer".to_owned(),
                key_id: "hub-key".to_owned(),
                public_key: key.verifying_key().to_bytes(),
                signature: Vec::new(),
            },
        };
        let signed_bytes = frame.signed_control_bytes().unwrap();
        let signature = DomainSignature::sign(
            SignatureAuthority::NodeSession,
            "hub-issuer".to_owned(),
            "hub-key".to_owned(),
            &key,
            &signed_bytes,
        );
        let NodeControlFrame::Cancel {
            signature: frame_signature,
            ..
        } = &mut frame
        else {
            unreachable!();
        };
        *frame_signature = signature.clone();
        signature
            .verify_exact(
                SignatureAuthority::NodeSession,
                "hub-issuer",
                key.verifying_key().to_bytes(),
                &frame.signed_control_bytes().unwrap(),
            )
            .unwrap();

        let NodeControlFrame::Cancel { attempt_fence, .. } = &mut frame else {
            unreachable!();
        };
        *attempt_fence += 1;
        assert!(signature
            .verify_exact(
                SignatureAuthority::NodeSession,
                "hub-issuer",
                key.verifying_key().to_bytes(),
                &frame.signed_control_bytes().unwrap(),
            )
            .is_err());
    }
}
