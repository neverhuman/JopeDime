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

#[path = "protocol/state.rs"]
mod state;
pub use state::NodeProtocolState;

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
#[path = "protocol/tests.rs"]
mod tests;
