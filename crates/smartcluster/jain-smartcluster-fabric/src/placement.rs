use std::collections::{BTreeMap, BTreeSet};

use ed25519_dalek::SigningKey;
use serde::{Deserialize, Serialize};

use crate::{
    domain_postcard, empty_signature, sha256, valid_identifier, Digest32, DomainSignature,
    FabricError, GpuRequirement, SignatureAuthority, VerifiedWorkUnit, WorkerProvision,
    NODE_LEASE_DOMAIN, PLACEMENT_OFFER_DOMAIN, WORKER_ACK_DOMAIN,
};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GpuDevice {
    pub uuid: String,
    pub model: String,
    pub free_vram_bytes: u64,
    pub exclusively_available: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct NodeLease {
    pub node_id: String,
    pub session_id: String,
    pub provision: WorkerProvision,
    pub lease_id: String,
    pub lease_expires_at_ms: u64,
    pub hub_epoch: u64,
    pub lease_revision: u64,
    pub last_control_position: u64,
    pub isolated: bool,
    pub isolation_evidence_digest: Digest32,
    pub open_outbound_enforced: bool,
    pub cpu_millicores_available: u32,
    pub memory_bytes_available: u64,
    pub scratch_bytes_available: u64,
    pub pids_available: u32,
    pub storage_domains: BTreeSet<String>,
    pub cached_artifacts: BTreeSet<String>,
    pub gpus: Vec<GpuDevice>,
    pub advertisement_digest: Digest32,
    pub signature: DomainSignature,
}

impl NodeLease {
    pub fn signed_bytes(&self) -> Result<Vec<u8>, FabricError> {
        let mut unsigned = self.clone();
        unsigned.signature = empty_signature(SignatureAuthority::NodeLease);
        domain_postcard(NODE_LEASE_DOMAIN, &unsigned)
    }

    fn verify(&self, trusted_public_key: Digest32) -> Result<(), FabricError> {
        if self.advertisement_digest == [0; 32] {
            return Err(FabricError::InvalidSignature);
        }
        self.signature.verify_exact(
            SignatureAuthority::NodeLease,
            &self.node_id,
            trusted_public_key,
            &self.signed_bytes()?,
        )
    }

    fn eligible(&self, work: &VerifiedWorkUnit, now_ms: u64) -> bool {
        self.provision.admits(work.workload_class)
            && self.lease_expires_at_ms > now_ms
            && self.hub_epoch == work.hub_epoch
            && self.last_control_position < work.control_position
            && self.isolated
            && self.isolation_evidence_digest != [0; 32]
            && (!work.open_outbound || self.open_outbound_enforced)
            && self.cpu_millicores_available >= work.limits.cpu_millicores
            && self.memory_bytes_available >= work.limits.memory_bytes
            && self.scratch_bytes_available >= work.limits.scratch_bytes
            && self.pids_available >= work.limits.pids
            && self.storage_domains.len() >= usize::from(work.minimum_writable_domains)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Placement {
    pub node_id: String,
    pub session_id: String,
    pub lease_id: String,
    pub lease_expires_at_ms: u64,
    pub work_unit_digest: Digest32,
    pub command_digest: Digest32,
    pub attempt: u32,
    pub attempt_fence: u64,
    pub hub_epoch: u64,
    pub control_position: u64,
    pub capability_advertisement_digest: Digest32,
    pub gpu_uuids: Vec<String>,
    pub cached_input_bytes: u64,
    pub isolation_evidence_digest: Digest32,
    pub signature: DomainSignature,
}

impl Placement {
    pub fn signed_bytes(&self) -> Result<Vec<u8>, FabricError> {
        let mut unsigned = self.clone();
        unsigned.signature = empty_signature(SignatureAuthority::PlacementOffer);
        domain_postcard(PLACEMENT_OFFER_DOMAIN, &unsigned)
    }

    pub fn verify_for(
        &self,
        work: &VerifiedWorkUnit,
        expected_node_id: &str,
        expected_session_id: &str,
        trusted_hub_public_key: Digest32,
        now_ms: u64,
    ) -> Result<(), FabricError> {
        work.validate()?;
        let unique_gpus = self.gpu_uuids.iter().collect::<BTreeSet<_>>();
        if !valid_identifier(&self.node_id)
            || !valid_identifier(&self.session_id)
            || !valid_identifier(&self.lease_id)
            || self.node_id != expected_node_id
            || self.session_id != expected_session_id
            || self.lease_id.is_empty()
            || self.lease_expires_at_ms <= now_ms
            || self.work_unit_digest != work.work_unit_digest()
            || self.command_digest != work.command_digest()
            || self.attempt != work.attempt
            || self.attempt_fence != work.attempt_fence
            || self.hub_epoch != work.hub_epoch
            || self.control_position != work.control_position
            || self.capability_advertisement_digest == [0; 32]
            || self.isolation_evidence_digest == [0; 32]
            || self.gpu_uuids.len() != work.gpu.as_ref().map_or(0, |gpu| usize::from(gpu.count))
            || unique_gpus.len() != self.gpu_uuids.len()
            || self.gpu_uuids.iter().any(|uuid| !valid_identifier(uuid))
        {
            return Err(FabricError::InvalidPlacement);
        }
        self.signature.verify_exact(
            SignatureAuthority::PlacementOffer,
            work.issuer(),
            trusted_hub_public_key,
            &self.signed_bytes()?,
        )
    }

    pub fn digest(&self) -> Result<Digest32, FabricError> {
        self.signed_bytes().map(|bytes| sha256(&bytes))
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WorkerAcknowledgement {
    pub node_id: String,
    pub session_id: String,
    pub work_unit_digest: Digest32,
    pub placement_digest: Digest32,
    pub assignment_journal_digest: Digest32,
    pub attempt: u32,
    pub attempt_fence: u64,
    pub hub_epoch: u64,
    pub control_position: u64,
    pub acknowledged_at_ms: u64,
    pub signature: DomainSignature,
}

impl WorkerAcknowledgement {
    pub fn signed_bytes(&self) -> Result<Vec<u8>, FabricError> {
        let mut unsigned = self.clone();
        unsigned.signature = empty_signature(SignatureAuthority::WorkerAcknowledgement);
        domain_postcard(WORKER_ACK_DOMAIN, &unsigned)
    }

    pub fn sign(
        node_id: String,
        session_id: String,
        work: &VerifiedWorkUnit,
        placement: &Placement,
        assignment_journal_digest: Digest32,
        acknowledged_at_ms: u64,
        key: &SigningKey,
    ) -> Result<Self, FabricError> {
        if assignment_journal_digest == [0; 32] || acknowledged_at_ms == 0 {
            return Err(FabricError::InvalidReceipt);
        }
        let mut acknowledgement = Self {
            node_id,
            session_id,
            work_unit_digest: work.work_unit_digest(),
            placement_digest: placement.digest()?,
            assignment_journal_digest,
            attempt: work.attempt,
            attempt_fence: work.attempt_fence,
            hub_epoch: work.hub_epoch,
            control_position: work.control_position,
            acknowledged_at_ms,
            signature: empty_signature(SignatureAuthority::WorkerAcknowledgement),
        };
        acknowledgement.signature = DomainSignature::sign(
            SignatureAuthority::WorkerAcknowledgement,
            acknowledgement.node_id.clone(),
            "jainnode-session".to_owned(),
            key,
            &acknowledgement.signed_bytes()?,
        );
        Ok(acknowledgement)
    }

    pub fn verify_for(
        &self,
        work: &VerifiedWorkUnit,
        placement: &Placement,
        trusted_node_public_key: Digest32,
    ) -> Result<(), FabricError> {
        if self.node_id != placement.node_id
            || self.session_id != placement.session_id
            || self.work_unit_digest != work.work_unit_digest()
            || self.placement_digest != placement.digest()?
            || self.assignment_journal_digest == [0; 32]
            || self.attempt != work.attempt
            || self.attempt_fence != work.attempt_fence
            || self.hub_epoch != work.hub_epoch
            || self.control_position != work.control_position
            || self.acknowledged_at_ms == 0
        {
            return Err(FabricError::InvalidReceipt);
        }
        self.signature.verify_exact(
            SignatureAuthority::WorkerAcknowledgement,
            &self.node_id,
            trusted_node_public_key,
            &self.signed_bytes()?,
        )
    }
}

pub struct FabricScheduler {
    nodes: BTreeMap<String, NodeLease>,
    trusted_node_keys: BTreeMap<String, Digest32>,
    issuer: String,
    signing_key: SigningKey,
}

impl FabricScheduler {
    pub fn new(issuer: String, signing_key: SigningKey) -> Result<Self, FabricError> {
        if issuer.is_empty() {
            return Err(FabricError::InvalidSignature);
        }
        Ok(Self {
            nodes: BTreeMap::new(),
            trusted_node_keys: BTreeMap::new(),
            issuer,
            signing_key,
        })
    }

    pub fn trust_node_key(
        &mut self,
        node_id: String,
        public_key: Digest32,
    ) -> Result<(), FabricError> {
        if !valid_identifier(&node_id) || public_key == [0; 32] {
            return Err(FabricError::InvalidSignature);
        }
        self.trusted_node_keys.insert(node_id, public_key);
        Ok(())
    }

    pub fn upsert(&mut self, lease: NodeLease) -> Result<(), FabricError> {
        if !valid_identifier(&lease.node_id)
            || !valid_identifier(&lease.session_id)
            || !valid_identifier(&lease.lease_id)
            || lease.hub_epoch == 0
            || lease.lease_revision == 0
            || lease.gpus.iter().any(|gpu| !valid_identifier(&gpu.uuid))
            || (lease.isolated && lease.isolation_evidence_digest == [0; 32])
            || (!lease.isolated && lease.isolation_evidence_digest != [0; 32])
        {
            return Err(FabricError::UnsupportedHost);
        }
        let unique_gpu_count = lease
            .gpus
            .iter()
            .map(|gpu| gpu.uuid.as_str())
            .collect::<BTreeSet<_>>()
            .len();
        if unique_gpu_count != lease.gpus.len() {
            return Err(FabricError::InvalidGpuRequest);
        }
        let trusted_public_key = self
            .trusted_node_keys
            .get(&lease.node_id)
            .copied()
            .ok_or(FabricError::InvalidSignature)?;
        lease.verify(trusted_public_key)?;
        if let Some(current) = self.nodes.get(&lease.node_id) {
            let current_revision = (current.hub_epoch, current.lease_revision);
            let next_revision = (lease.hub_epoch, lease.lease_revision);
            if next_revision < current_revision
                || (lease.hub_epoch == current.hub_epoch
                    && lease.last_control_position < current.last_control_position)
                || (next_revision == current_revision && &lease != current)
            {
                return Err(FabricError::ReplayedPosition);
            }
            if &lease == current {
                return Ok(());
            }
        }
        self.nodes.insert(lease.node_id.clone(), lease);
        Ok(())
    }

    pub fn place_and_reserve(
        &mut self,
        work: &VerifiedWorkUnit,
        now_ms: u64,
    ) -> Result<Placement, FabricError> {
        work.validate()?;
        if work.issuer() != self.issuer {
            return Err(FabricError::InvalidSignature);
        }
        let mut candidates = Vec::new();
        for node in self
            .nodes
            .values()
            .filter(|node| node.eligible(work, now_ms))
        {
            let Some(gpu_uuids) = select_gpus(node, work.gpu.as_ref()) else {
                continue;
            };
            let cached_input_bytes = work
                .inputs
                .iter()
                .filter(|input| node.cached_artifacts.contains(&input.artifact_id))
                .try_fold(0_u64, |total, input| {
                    total
                        .checked_add(input.size_bytes)
                        .ok_or(FabricError::InvalidArtifact)
                })?;
            candidates.push((node.node_id.clone(), gpu_uuids, cached_input_bytes));
        }
        candidates.sort_by(|left, right| right.2.cmp(&left.2).then_with(|| left.0.cmp(&right.0)));
        let (node_id, gpu_uuids, cached_input_bytes) = candidates
            .into_iter()
            .next()
            .ok_or(FabricError::CapacityExhausted)?;
        let node = self
            .nodes
            .get_mut(&node_id)
            .ok_or(FabricError::CapacityExhausted)?;
        node.cpu_millicores_available -= work.limits.cpu_millicores;
        node.memory_bytes_available -= work.limits.memory_bytes;
        node.scratch_bytes_available -= work.limits.scratch_bytes;
        node.pids_available -= work.limits.pids;
        for uuid in &gpu_uuids {
            let gpu = node
                .gpus
                .iter_mut()
                .find(|gpu| &gpu.uuid == uuid)
                .ok_or(FabricError::InvalidGpuRequest)?;
            gpu.exclusively_available = false;
        }
        node.last_control_position = work.control_position;
        let mut placement = Placement {
            node_id: node.node_id.clone(),
            session_id: node.session_id.clone(),
            lease_id: node.lease_id.clone(),
            lease_expires_at_ms: node.lease_expires_at_ms,
            work_unit_digest: work.work_unit_digest(),
            command_digest: work.command_digest(),
            attempt: work.attempt,
            attempt_fence: work.attempt_fence,
            hub_epoch: work.hub_epoch,
            control_position: work.control_position,
            capability_advertisement_digest: node.advertisement_digest,
            gpu_uuids,
            cached_input_bytes,
            isolation_evidence_digest: node.isolation_evidence_digest,
            signature: DomainSignature {
                authority: SignatureAuthority::PlacementOffer,
                issuer: self.issuer.clone(),
                key_id: "pending".to_owned(),
                public_key: self.signing_key.verifying_key().to_bytes(),
                signature: Vec::new(),
            },
        };
        placement.signature = DomainSignature::sign(
            SignatureAuthority::PlacementOffer,
            self.issuer.clone(),
            "scheduler-v1".to_owned(),
            &self.signing_key,
            &placement.signed_bytes()?,
        );
        Ok(placement)
    }
}

fn select_gpus(node: &NodeLease, requirement: Option<&GpuRequirement>) -> Option<Vec<String>> {
    let Some(requirement) = requirement else {
        return Some(Vec::new());
    };
    let selected = node
        .gpus
        .iter()
        .filter(|gpu| {
            gpu.exclusively_available
                && gpu.free_vram_bytes >= requirement.minimum_vram_bytes
                && requirement
                    .model
                    .as_ref()
                    .is_none_or(|model| &gpu.model == model)
        })
        .take(usize::from(requirement.count))
        .map(|gpu| gpu.uuid.clone())
        .collect::<Vec<_>>();
    (selected.len() == usize::from(requirement.count)).then_some(selected)
}
