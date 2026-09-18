#![deny(unsafe_code)]

use std::collections::BTreeMap;

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

mod placement;
mod replay;
mod work;

pub use placement::{FabricScheduler, GpuDevice, NodeLease, Placement, WorkerAcknowledgement};
pub use replay::{AttemptFence, ReplayJournal};
pub use work::{
    ArtifactMount, GpuRequirement, OutputDeclaration, ResourceLimits, SignedWorkUnitProjection,
    VerifiedWorkUnit, WorkUnitProjection, WorkerProvision, WorkloadClass,
};

pub const MAX_ARGV: usize = 256;
pub const MAX_ARG_BYTES: usize = 16 * 1024;
pub const MAX_ENVIRONMENT: usize = 128;
pub const MAX_INPUTS: usize = 256;
pub const MAX_OUTPUTS: usize = 128;

pub type Digest32 = [u8; 32];

const WORK_UNIT_PROJECTION_DOMAIN: &[u8] = b"jain.fabric.work-unit-projection.v1\0";
const OCI_IMAGE_DOMAIN: &[u8] = b"jain.fabric.oci-image.v1\0";
const COMMAND_DOMAIN: &[u8] = b"jain.fabric.command.v1\0";
pub(crate) const NODE_LEASE_DOMAIN: &[u8] = b"jain.fabric.node-lease.v1\0";
pub(crate) const PLACEMENT_OFFER_DOMAIN: &[u8] = b"jain.fabric.placement-offer.v1\0";
pub(crate) const WORKER_ACK_DOMAIN: &[u8] = b"jain.fabric.worker-acknowledgement.v1\0";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SignatureAuthority {
    WorkUnitIssuer,
    OciImage,
    NodeLease,
    PlacementOffer,
    WorkerAcknowledgement,
    NodeEnrollment,
    NodeSession,
    NodeStorageTelemetry,
    StorageFragmentReceipt,
    StorageControl,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DomainSignature {
    pub authority: SignatureAuthority,
    pub issuer: String,
    pub key_id: String,
    pub public_key: Digest32,
    pub signature: Vec<u8>,
}

impl DomainSignature {
    pub fn sign(
        authority: SignatureAuthority,
        issuer: String,
        key_id: String,
        key: &SigningKey,
        signed_bytes: &[u8],
    ) -> Self {
        Self {
            authority,
            issuer,
            key_id,
            public_key: key.verifying_key().to_bytes(),
            signature: key.sign(signed_bytes).to_bytes().to_vec(),
        }
    }

    pub fn verify_exact(
        &self,
        expected_authority: SignatureAuthority,
        expected_issuer: &str,
        expected_public_key: Digest32,
        signed_bytes: &[u8],
    ) -> Result<(), FabricError> {
        if self.authority != expected_authority
            || !valid_identifier(expected_issuer)
            || self.issuer != expected_issuer
            || !valid_identifier(&self.key_id)
            || self.public_key != expected_public_key
        {
            return Err(FabricError::InvalidSignature);
        }
        let signature_bytes: [u8; 64] = self
            .signature
            .as_slice()
            .try_into()
            .map_err(|_| FabricError::InvalidSignature)?;
        let public_key = VerifyingKey::from_bytes(&self.public_key)
            .map_err(|_| FabricError::InvalidSignature)?;
        public_key
            .verify(signed_bytes, &Signature::from_bytes(&signature_bytes))
            .map_err(|_| FabricError::InvalidSignature)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize, Error)]
pub enum FabricError {
    #[error("invalid_work_unit")]
    InvalidWorkUnit,
    #[error("unpinned_image")]
    UnpinnedImage,
    #[error("reserved_environment")]
    ReservedEnvironment,
    #[error("invalid_artifact")]
    InvalidArtifact,
    #[error("invalid_gpu_request")]
    InvalidGpuRequest,
    #[error("invalid_network_policy")]
    InvalidNetworkPolicy,
    #[error("invalid_signature")]
    InvalidSignature,
    #[error("invalid_placement")]
    InvalidPlacement,
    #[error("unsupported_host")]
    UnsupportedHost,
    #[error("capacity_exhausted")]
    CapacityExhausted,
    #[error("replayed_position")]
    ReplayedPosition,
    #[error("fence_rejected")]
    FenceRejected,
    #[error("invalid_receipt")]
    InvalidReceipt,
    #[error("conflicting_worker_report")]
    ConflictingWorkerReport,
    #[error("journal_io: {0}")]
    JournalIo(String),
    #[error("journal_locked")]
    JournalLocked,
}

pub(crate) fn journal_io(error: std::io::Error) -> FabricError {
    FabricError::JournalIo(error.to_string())
}

pub fn sha256(bytes: &[u8]) -> Digest32 {
    Sha256::digest(bytes).into()
}

pub fn valid_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 256
        && !matches!(value, "." | "..")
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
}

pub fn projection_signed_bytes(projection: &WorkUnitProjection) -> Result<Vec<u8>, FabricError> {
    domain_postcard(WORK_UNIT_PROJECTION_DOMAIN, projection)
}

pub fn image_signed_bytes(
    image_reference: &str,
    image_digest: Digest32,
) -> Result<Vec<u8>, FabricError> {
    domain_postcard(OCI_IMAGE_DOMAIN, &(image_reference, image_digest))
}

fn command_digest(projection: &WorkUnitProjection) -> Result<Digest32, FabricError> {
    #[derive(Serialize)]
    struct CommandBinding<'a> {
        image_digest: Digest32,
        argv: &'a [String],
        smoke_argv: &'a [String],
        working_directory: &'a str,
        caller_environment: &'a BTreeMap<String, String>,
    }

    domain_postcard(
        COMMAND_DOMAIN,
        &CommandBinding {
            image_digest: projection.image_digest,
            argv: &projection.argv,
            smoke_argv: &projection.smoke_argv,
            working_directory: &projection.working_directory,
            caller_environment: &projection.caller_environment,
        },
    )
    .map(|bytes| sha256(&bytes))
}

pub(crate) fn domain_postcard<T: Serialize>(
    domain: &[u8],
    value: &T,
) -> Result<Vec<u8>, FabricError> {
    let payload = postcard::to_stdvec(value).map_err(|_| FabricError::InvalidWorkUnit)?;
    let mut bytes = Vec::with_capacity(domain.len() + payload.len());
    bytes.extend_from_slice(domain);
    bytes.extend_from_slice(&payload);
    Ok(bytes)
}

pub(crate) fn empty_signature(authority: SignatureAuthority) -> DomainSignature {
    DomainSignature {
        authority,
        issuer: String::new(),
        key_id: String::new(),
        public_key: [0; 32],
        signature: Vec::new(),
    }
}

fn validate_argv(argv: &[String]) -> Result<(), FabricError> {
    if argv.is_empty()
        || argv.len() > MAX_ARGV
        || argv
            .iter()
            .any(|arg| arg.is_empty() || arg.len() > MAX_ARG_BYTES || arg.contains('\0'))
    {
        return Err(FabricError::InvalidWorkUnit);
    }
    let launcher = argv[0].rsplit('/').next().unwrap_or(&argv[0]);
    if matches!(
        launcher,
        "sh" | "bash" | "dash" | "zsh" | "fish" | "cmd" | "powershell"
    ) && argv
        .get(1)
        .is_some_and(|arg| matches!(arg.as_str(), "-c" | "/c" | "-Command"))
    {
        return Err(FabricError::InvalidWorkUnit);
    }
    Ok(())
}

fn safe_relative_directory(value: &str) -> bool {
    !value.is_empty()
        && !value.starts_with('/')
        && !value.contains('\0')
        && !value
            .split('/')
            .any(|part| part.is_empty() || matches!(part, ".."))
}

fn safe_container_path(value: &str) -> bool {
    value.starts_with("/inputs/")
        && !value.contains('\0')
        && !value.split('/').any(|part| part == "..")
        && !matches!(
            value,
            "/var/run/docker.sock" | "/run/docker.sock" | "/proc" | "/sys" | "/dev"
        )
}

fn valid_environment_key(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.bytes().enumerate().all(|(index, byte)| {
            byte == b'_' || byte.is_ascii_uppercase() || (index > 0 && byte.is_ascii_digit())
        })
}

fn reserved_environment(value: &str) -> bool {
    value.starts_with("JAIN_")
        || matches!(
            value,
            "PATH" | "HOME" | "USER" | "LOGNAME" | "SHELL" | "DOCKER_HOST" | "LD_PRELOAD"
        )
}

fn hex_digest(digest: Digest32) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in digest {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

#[cfg(test)]
mod tests;
