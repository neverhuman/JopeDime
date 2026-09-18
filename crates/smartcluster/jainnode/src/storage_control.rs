use ed25519_dalek::SigningKey;
use jain_smartcluster_fabric::{
    sha256, valid_identifier, Digest32, DomainSignature, SignatureAuthority,
};
use serde::{Deserialize, Serialize};

use crate::{NodeControlFrame, ProtocolError, StorageFragmentKind};

pub const MAX_STORAGE_FRAGMENT_CHUNK_BYTES: usize = 256 * 1024 + 16;
const STORAGE_CONTROL_DOMAIN: &[u8] = b"jain.node.storage-control/v1\0";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BeginFragmentWrite {
    pub transfer_id: String,
    pub project_id: String,
    pub object_id: Digest32,
    pub manifest_digest: Digest32,
    pub fragment_index: u16,
    pub fragment_kind: StorageFragmentKind,
    pub failure_domain: String,
    pub topology_digest: Digest32,
    #[serde(with = "crate::decimal_bytes")]
    pub expected_bytes: u64,
    pub expected_sha256: Digest32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FragmentWriteChunk {
    pub transfer_id: String,
    #[serde(with = "crate::decimal_bytes")]
    pub offset: u64,
    pub bytes: Vec<u8>,
    pub sha256: Digest32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CommitFragmentWrite {
    pub transfer_id: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReadFragment {
    pub transfer_id: String,
    pub project_id: String,
    pub object_id: Digest32,
    pub manifest_digest: Digest32,
    pub fragment_index: u16,
    pub fragment_kind: StorageFragmentKind,
    pub expected_sha256: Digest32,
    #[serde(with = "crate::decimal_bytes")]
    pub expected_bytes: u64,
    #[serde(with = "crate::decimal_bytes")]
    pub offset: u64,
    pub maximum_bytes: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageControlBody {
    BeginWrite(BeginFragmentWrite),
    WriteChunk(FragmentWriteChunk),
    CommitWrite(CommitFragmentWrite),
    Read(ReadFragment),
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SignedStorageControlRequest {
    pub session_id: String,
    pub sequence: u64,
    pub deadline_ms: u64,
    pub body: StorageControlBody,
    pub signature: DomainSignature,
}

impl SignedStorageControlRequest {
    #[allow(clippy::too_many_arguments)]
    pub fn sign(
        issuer: String,
        key_id: String,
        signing_key: &SigningKey,
        session_id: String,
        sequence: u64,
        deadline_ms: u64,
        body: StorageControlBody,
    ) -> Result<Self, ProtocolError> {
        let mut request = Self {
            session_id,
            sequence,
            deadline_ms,
            body,
            signature: empty_signature(),
        };
        request.validate_shape(0)?;
        request.signature = DomainSignature::sign(
            SignatureAuthority::StorageControl,
            issuer,
            key_id,
            signing_key,
            &request.signed_bytes()?,
        );
        Ok(request)
    }

    pub fn signed_bytes(&self) -> Result<Vec<u8>, ProtocolError> {
        let mut unsigned = self.clone();
        unsigned.signature = empty_signature();
        let payload =
            postcard::to_stdvec(&unsigned).map_err(|_| ProtocolError::InvalidStorageControl)?;
        let mut bytes = Vec::with_capacity(STORAGE_CONTROL_DOMAIN.len() + payload.len());
        bytes.extend_from_slice(STORAGE_CONTROL_DOMAIN);
        bytes.extend_from_slice(&payload);
        Ok(bytes)
    }

    pub fn verify(
        &self,
        expected_session_id: &str,
        expected_issuer: &str,
        expected_public_key: Digest32,
        now_ms: u64,
    ) -> Result<(), ProtocolError> {
        self.validate_shape(now_ms)?;
        if self.session_id != expected_session_id {
            return Err(ProtocolError::InvalidStorageControl);
        }
        self.signature
            .verify_exact(
                SignatureAuthority::StorageControl,
                expected_issuer,
                expected_public_key,
                &self.signed_bytes()?,
            )
            .map_err(|_| ProtocolError::InvalidStorageControl)
    }

    fn validate_shape(&self, now_ms: u64) -> Result<(), ProtocolError> {
        if !valid_identifier(&self.session_id) || self.sequence == 0 || self.deadline_ms <= now_ms {
            return Err(ProtocolError::InvalidStorageControl);
        }
        match &self.body {
            StorageControlBody::BeginWrite(begin) => validate_begin(begin),
            StorageControlBody::WriteChunk(chunk) => {
                if !valid_identifier(&chunk.transfer_id)
                    || chunk.bytes.is_empty()
                    || chunk.bytes.len() > MAX_STORAGE_FRAGMENT_CHUNK_BYTES
                    || chunk.sha256 == [0; 32]
                    || sha256(&chunk.bytes) != chunk.sha256
                {
                    return Err(ProtocolError::InvalidStorageControl);
                }
                Ok(())
            }
            StorageControlBody::CommitWrite(commit) => {
                if !valid_identifier(&commit.transfer_id) {
                    return Err(ProtocolError::InvalidStorageControl);
                }
                Ok(())
            }
            StorageControlBody::Read(read) => {
                if !valid_identifier(&read.transfer_id)
                    || !valid_identifier(&read.project_id)
                    || read.object_id == [0; 32]
                    || read.manifest_digest == [0; 32]
                    || read.expected_sha256 == [0; 32]
                    || read.expected_bytes == 0
                    || read.offset >= read.expected_bytes
                    || read.maximum_bytes == 0
                    || read.maximum_bytes as usize > MAX_STORAGE_FRAGMENT_CHUNK_BYTES
                    || !fragment_kind_matches(read.fragment_index, read.fragment_kind)
                {
                    return Err(ProtocolError::InvalidStorageControl);
                }
                Ok(())
            }
        }
    }
}

impl NodeControlFrame {
    pub fn storage_control(request: SignedStorageControlRequest) -> Result<Self, ProtocolError> {
        request.validate_shape(0)?;
        let frame = Self::StorageControl {
            request: Box::new(request),
        };
        frame.encode()?;
        Ok(frame)
    }
}

fn validate_begin(begin: &BeginFragmentWrite) -> Result<(), ProtocolError> {
    if !valid_identifier(&begin.transfer_id)
        || !valid_identifier(&begin.project_id)
        || !valid_identifier(&begin.failure_domain)
        || begin.object_id == [0; 32]
        || begin.manifest_digest == [0; 32]
        || begin.topology_digest == [0; 32]
        || begin.expected_bytes == 0
        || begin.expected_sha256 == [0; 32]
        || !fragment_kind_matches(begin.fragment_index, begin.fragment_kind)
    {
        return Err(ProtocolError::InvalidStorageControl);
    }
    Ok(())
}

fn fragment_kind_matches(index: u16, kind: StorageFragmentKind) -> bool {
    matches!(
        (index, kind),
        (0, StorageFragmentKind::Replica(0))
            | (1, StorageFragmentKind::Replica(1))
            | (0, StorageFragmentKind::Data(0))
            | (1, StorageFragmentKind::Data(1))
            | (2, StorageFragmentKind::Data(2))
            | (3, StorageFragmentKind::Data(3))
            | (4, StorageFragmentKind::Recovery(0))
    )
}

fn empty_signature() -> DomainSignature {
    DomainSignature {
        authority: SignatureAuthority::StorageControl,
        issuer: String::new(),
        key_id: String::new(),
        public_key: [0; 32],
        signature: Vec::new(),
    }
}
