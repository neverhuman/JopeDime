use std::collections::BTreeSet;

use jain_smartcluster_fabric::{
    sha256, valid_identifier, Digest32, DomainSignature, SignatureAuthority,
};
use serde::{Deserialize, Serialize};

use crate::{NodeRuntimeIdentity, ProtocolError};

pub const STORAGE_TELEMETRY_INTERVAL_MS: u64 = 2_000;
const STORAGE_TELEMETRY_VERSION: u16 = 1;
const STORAGE_FRAGMENT_RECEIPT_VERSION: u16 = 1;
const MAX_STORAGE_VOLUMES: usize = 64;
const STORAGE_TELEMETRY_DOMAIN: &[u8] = b"jain.node.storage-telemetry/v1\0";
const STORAGE_FRAGMENT_RECEIPT_DOMAIN: &[u8] = b"jain.storage-fragment-receipt/v1\0";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StorageVolumeTelemetry {
    pub failure_domain: String,
    #[serde(with = "crate::decimal_bytes")]
    pub capacity_bytes: u64,
    #[serde(with = "crate::decimal_bytes")]
    pub used_bytes: u64,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StorageIoCounters {
    #[serde(with = "crate::decimal_bytes")]
    pub read_bytes_total: u64,
    #[serde(with = "crate::decimal_bytes")]
    pub write_bytes_total: u64,
    pub read_operations_total: u64,
    pub write_operations_total: u64,
}

impl StorageIoCounters {
    fn dominates(&self, previous: &Self) -> bool {
        self.read_bytes_total >= previous.read_bytes_total
            && self.write_bytes_total >= previous.write_bytes_total
            && self.read_operations_total >= previous.read_operations_total
            && self.write_operations_total >= previous.write_operations_total
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FragmentHealthTelemetry {
    pub healthy_fragments: u64,
    pub missing_fragments: u64,
    pub corrupt_fragments: u64,
    pub underprotected_objects: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ComputeStateTelemetry {
    pub slots_total: u32,
    pub slots_busy: u32,
    pub completed_execution_receipt_digest: Option<Digest32>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StorageTelemetryObservation {
    pub lease_id: String,
    pub lease_expires_at_ms: u64,
    pub observed_at_ms: u64,
    pub topology_digest: Digest32,
    pub storage_capable: bool,
    pub compute_capable: bool,
    pub volumes: Vec<StorageVolumeTelemetry>,
    pub io: StorageIoCounters,
    pub fragment_health: FragmentHealthTelemetry,
    pub compute: ComputeStateTelemetry,
}

impl StorageTelemetryObservation {
    fn validate(&self) -> Result<(), ProtocolError> {
        let domains = self
            .volumes
            .iter()
            .map(|volume| volume.failure_domain.as_str())
            .collect::<BTreeSet<_>>();
        if !valid_identifier(&self.lease_id)
            || self.observed_at_ms == 0
            || self.lease_expires_at_ms <= self.observed_at_ms
            || self.topology_digest == [0; 32]
            || !self.storage_capable
            || self.volumes.is_empty()
            || self.volumes.len() > MAX_STORAGE_VOLUMES
            || domains.len() != self.volumes.len()
            || self.volumes.iter().any(|volume| {
                !valid_identifier(&volume.failure_domain)
                    || volume.capacity_bytes == 0
                    || volume.used_bytes > volume.capacity_bytes
            })
            || self.compute.slots_busy > self.compute.slots_total
            || self.compute_capable != (self.compute.slots_total > 0)
            || self
                .compute
                .completed_execution_receipt_digest
                .is_some_and(|digest| digest == [0; 32])
        {
            return Err(ProtocolError::InvalidStorageFacts);
        }
        Ok(())
    }

    fn transition_digest(&self) -> Result<Digest32, ProtocolError> {
        let payload = postcard::to_stdvec(&(
            self.topology_digest,
            &self.volumes,
            &self.fragment_health,
            &self.compute,
        ))
        .map_err(|_| ProtocolError::InvalidStorageFacts)?;
        Ok(sha256(&payload))
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StorageTelemetryFacts {
    pub version: u16,
    pub node_id: String,
    pub session_id: String,
    pub revision: u64,
    pub observation: StorageTelemetryObservation,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SignedStorageTelemetry {
    pub facts: StorageTelemetryFacts,
    pub signature: DomainSignature,
}

impl SignedStorageTelemetry {
    fn sign(
        identity: &NodeRuntimeIdentity,
        revision: u64,
        observation: StorageTelemetryObservation,
    ) -> Result<Self, ProtocolError> {
        let mut telemetry = Self {
            facts: StorageTelemetryFacts {
                version: STORAGE_TELEMETRY_VERSION,
                node_id: identity.node_id().to_owned(),
                session_id: identity.session_id().to_owned(),
                revision,
                observation,
            },
            signature: empty_signature(SignatureAuthority::NodeStorageTelemetry),
        };
        telemetry.validate_shape(identity.node_id(), identity.session_id())?;
        telemetry.signature = DomainSignature::sign(
            SignatureAuthority::NodeStorageTelemetry,
            identity.node_id().to_owned(),
            identity.key_id().to_owned(),
            identity.signing_key(),
            &telemetry.signed_bytes()?,
        );
        Ok(telemetry)
    }

    pub fn signed_bytes(&self) -> Result<Vec<u8>, ProtocolError> {
        let mut unsigned = self.clone();
        unsigned.signature = empty_signature(SignatureAuthority::NodeStorageTelemetry);
        domain_postcard(STORAGE_TELEMETRY_DOMAIN, &unsigned)
    }

    pub fn verify(
        &self,
        expected_node_id: &str,
        expected_session_id: &str,
        expected_key_id: &str,
        expected_public_key: Digest32,
    ) -> Result<(), ProtocolError> {
        self.validate_shape(expected_node_id, expected_session_id)?;
        if self.signature.key_id != expected_key_id {
            return Err(ProtocolError::InvalidStorageFacts);
        }
        self.signature
            .verify_exact(
                SignatureAuthority::NodeStorageTelemetry,
                expected_node_id,
                expected_public_key,
                &self.signed_bytes()?,
            )
            .map_err(|_| ProtocolError::InvalidStorageFacts)
    }

    pub(crate) fn validate_shape(
        &self,
        expected_node_id: &str,
        expected_session_id: &str,
    ) -> Result<(), ProtocolError> {
        if self.facts.version != STORAGE_TELEMETRY_VERSION
            || self.facts.node_id != expected_node_id
            || self.facts.session_id != expected_session_id
            || self.facts.revision == 0
            || !valid_identifier(expected_node_id)
            || !valid_identifier(expected_session_id)
        {
            return Err(ProtocolError::InvalidStorageFacts);
        }
        self.facts.observation.validate()
    }
}

#[derive(Clone, Debug, Default)]
pub struct StorageTelemetryEmitter {
    revision: u64,
    last_emitted_at_ms: Option<u64>,
    last_transition_digest: Option<Digest32>,
    last_counters: Option<StorageIoCounters>,
}

impl StorageTelemetryEmitter {
    pub fn observe(
        &mut self,
        identity: &NodeRuntimeIdentity,
        observation: StorageTelemetryObservation,
    ) -> Result<Option<SignedStorageTelemetry>, ProtocolError> {
        observation.validate()?;
        if self
            .last_counters
            .as_ref()
            .is_some_and(|previous| !observation.io.dominates(previous))
        {
            return Err(ProtocolError::ReplayedStorageRevision);
        }
        let transition_digest = observation.transition_digest()?;
        let immediate = self.last_transition_digest != Some(transition_digest);
        let periodic = self.last_emitted_at_ms.is_none_or(|last| {
            observation.observed_at_ms.saturating_sub(last) >= STORAGE_TELEMETRY_INTERVAL_MS
        });
        self.last_counters = Some(observation.io.clone());
        if !immediate && !periodic {
            return Ok(None);
        }
        self.revision = self
            .revision
            .checked_add(1)
            .ok_or(ProtocolError::ReplayedStorageRevision)?;
        self.last_emitted_at_ms = Some(observation.observed_at_ms);
        self.last_transition_digest = Some(transition_digest);
        SignedStorageTelemetry::sign(identity, self.revision, observation).map(Some)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageFragmentKind {
    Replica(u16),
    Data(u16),
    Recovery(u16),
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StorageFragmentReceiptFacts {
    pub version: u16,
    pub project_id: String,
    pub object_id: Digest32,
    pub manifest_digest: Digest32,
    pub fragment_index: u16,
    pub fragment_kind: StorageFragmentKind,
    pub node_id: String,
    pub session_id: String,
    pub failure_domain: String,
    pub topology_digest: Digest32,
    pub fragment_sha256: Digest32,
    #[serde(with = "crate::decimal_bytes")]
    pub fragment_bytes: u64,
    pub fsync_completed_at_ms: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SignedStorageFragmentReceipt {
    pub facts: StorageFragmentReceiptFacts,
    pub signature: DomainSignature,
}

impl SignedStorageFragmentReceipt {
    #[allow(clippy::too_many_arguments)]
    pub fn sign_fsynced(
        identity: &NodeRuntimeIdentity,
        project_id: String,
        object_id: Digest32,
        manifest_digest: Digest32,
        fragment_index: u16,
        fragment_kind: StorageFragmentKind,
        failure_domain: String,
        topology_digest: Digest32,
        fragment_sha256: Digest32,
        fragment_bytes: u64,
        fsync_completed_at_ms: u64,
    ) -> Result<Self, ProtocolError> {
        let mut receipt = Self {
            facts: StorageFragmentReceiptFacts {
                version: STORAGE_FRAGMENT_RECEIPT_VERSION,
                project_id,
                object_id,
                manifest_digest,
                fragment_index,
                fragment_kind,
                node_id: identity.node_id().to_owned(),
                session_id: identity.session_id().to_owned(),
                failure_domain,
                topology_digest,
                fragment_sha256,
                fragment_bytes,
                fsync_completed_at_ms,
            },
            signature: empty_signature(SignatureAuthority::StorageFragmentReceipt),
        };
        receipt.validate_shape(identity.node_id(), identity.session_id())?;
        receipt.signature = DomainSignature::sign(
            SignatureAuthority::StorageFragmentReceipt,
            identity.node_id().to_owned(),
            identity.key_id().to_owned(),
            identity.signing_key(),
            &receipt.signed_bytes()?,
        );
        Ok(receipt)
    }

    pub fn signed_bytes(&self) -> Result<Vec<u8>, ProtocolError> {
        let mut unsigned = self.clone();
        unsigned.signature = empty_signature(SignatureAuthority::StorageFragmentReceipt);
        domain_postcard(STORAGE_FRAGMENT_RECEIPT_DOMAIN, &unsigned)
    }

    pub fn verify(
        &self,
        expected_node_id: &str,
        expected_session_id: &str,
        expected_key_id: &str,
        expected_public_key: Digest32,
    ) -> Result<(), ProtocolError> {
        self.validate_shape(expected_node_id, expected_session_id)?;
        if self.signature.key_id != expected_key_id {
            return Err(ProtocolError::InvalidStorageFacts);
        }
        self.signature
            .verify_exact(
                SignatureAuthority::StorageFragmentReceipt,
                expected_node_id,
                expected_public_key,
                &self.signed_bytes()?,
            )
            .map_err(|_| ProtocolError::InvalidStorageFacts)
    }

    pub(crate) fn validate_shape(
        &self,
        expected_node_id: &str,
        expected_session_id: &str,
    ) -> Result<(), ProtocolError> {
        if self.facts.version != STORAGE_FRAGMENT_RECEIPT_VERSION
            || self.facts.node_id != expected_node_id
            || self.facts.session_id != expected_session_id
            || !valid_identifier(&self.facts.project_id)
            || !valid_identifier(&self.facts.failure_domain)
            || self.facts.object_id == [0; 32]
            || self.facts.manifest_digest == [0; 32]
            || self.facts.topology_digest == [0; 32]
            || self.facts.fragment_sha256 == [0; 32]
            || self.facts.fragment_bytes == 0
            || self.facts.fsync_completed_at_ms == 0
            || !fragment_kind_matches(self.facts.fragment_index, self.facts.fragment_kind)
        {
            return Err(ProtocolError::InvalidStorageFacts);
        }
        Ok(())
    }
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

fn empty_signature(authority: SignatureAuthority) -> DomainSignature {
    DomainSignature {
        authority,
        issuer: String::new(),
        key_id: String::new(),
        public_key: [0; 32],
        signature: Vec::new(),
    }
}

fn domain_postcard<T: Serialize>(domain: &[u8], value: &T) -> Result<Vec<u8>, ProtocolError> {
    let payload = postcard::to_stdvec(value).map_err(|_| ProtocolError::InvalidStorageFacts)?;
    let mut bytes = Vec::with_capacity(domain.len() + payload.len());
    bytes.extend_from_slice(domain);
    bytes.extend_from_slice(&payload);
    Ok(bytes)
}
