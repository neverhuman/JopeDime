use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Deref,
};

use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::{
    command_digest, hex_digest, image_signed_bytes, projection_signed_bytes, reserved_environment,
    safe_container_path, safe_relative_directory, sha256, valid_environment_key, valid_identifier,
    validate_argv, Digest32, DomainSignature, FabricError, SignatureAuthority, MAX_ENVIRONMENT,
    MAX_INPUTS, MAX_OUTPUTS,
};

/// The closed purpose carried by an exact signed WorkUnit.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkloadClass {
    ContinuousIntegration,
    Coding,
    InteractiveResearch,
    BatchResearch,
    Training,
    Evaluation,
    OneShotInference,
    ReservedInference,
    Repair,
    Maintenance,
}

/// One Jane Worker (`jainnode`) provisioned for one closed workload family.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerProvision {
    CodeRunner,
    SmartclusterCompute,
    DedicatedInference,
}

impl WorkerProvision {
    #[must_use]
    pub const fn admits(self, workload: WorkloadClass) -> bool {
        match self {
            Self::CodeRunner => matches!(
                workload,
                WorkloadClass::ContinuousIntegration
                    | WorkloadClass::Coding
                    | WorkloadClass::Repair
                    | WorkloadClass::Maintenance
            ),
            Self::SmartclusterCompute => matches!(
                workload,
                WorkloadClass::InteractiveResearch
                    | WorkloadClass::BatchResearch
                    | WorkloadClass::Training
                    | WorkloadClass::Evaluation
            ),
            Self::DedicatedInference => matches!(
                workload,
                WorkloadClass::OneShotInference | WorkloadClass::ReservedInference
            ),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ResourceLimits {
    pub cpu_millicores: u32,
    pub memory_bytes: u64,
    pub scratch_bytes: u64,
    pub pids: u32,
    pub wall_time_ms: u64,
    pub read_bytes: u64,
    pub write_bytes: u64,
    pub io_operations: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GpuRequirement {
    pub model: Option<String>,
    pub count: u16,
    pub minimum_vram_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ArtifactMount {
    pub artifact_id: String,
    pub artifact_digest: Digest32,
    pub mount_path: String,
    pub size_bytes: u64,
    pub durable_receipt_digest: Digest32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct OutputDeclaration {
    pub relative_path: String,
    pub media_type: String,
    pub maximum_bytes: u64,
    pub required: bool,
}

/// The closed scheduler projection signed by the Hub's Fabric adapter.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WorkUnitProjection {
    pub run_id: Ulid,
    pub project_id: String,
    pub workload_class: WorkloadClass,
    pub image_reference: String,
    pub image_digest: Digest32,
    pub argv: Vec<String>,
    pub smoke_argv: Vec<String>,
    pub smoke_timeout_ms: u64,
    pub smoke_only: bool,
    pub working_directory: String,
    pub caller_environment: BTreeMap<String, String>,
    pub inputs: Vec<ArtifactMount>,
    pub outputs: Vec<OutputDeclaration>,
    pub limits: ResourceLimits,
    pub gpu: Option<GpuRequirement>,
    pub open_outbound: bool,
    pub network_bytes: u64,
    pub network_connections: u32,
    pub network_deadline_ms: u64,
    pub stdout_bytes: u64,
    pub stderr_bytes: u64,
    pub minimum_writable_domains: u16,
    pub attempt: u32,
    pub attempt_fence: u64,
    pub hub_epoch: u64,
    pub control_position: u64,
    pub replay_nonce: Digest32,
}

impl WorkUnitProjection {
    fn validate(&self) -> Result<(), FabricError> {
        if self.run_id == Ulid::from(0_u128)
            || !valid_identifier(&self.project_id)
            || self.image_digest == [0; 32]
            || self.attempt == 0
            || self.attempt_fence == 0
            || self.hub_epoch == 0
            || self.control_position == 0
            || self.replay_nonce == [0; 32]
            || self.minimum_writable_domains == 0
            || self.stdout_bytes == 0
            || self.stderr_bytes == 0
        {
            return Err(FabricError::InvalidWorkUnit);
        }
        let expected_image = format!("@sha256:{}", hex_digest(self.image_digest));
        if !self.image_reference.ends_with(&expected_image) {
            return Err(FabricError::UnpinnedImage);
        }
        validate_argv(&self.argv)?;
        validate_argv(&self.smoke_argv)?;
        if self.smoke_timeout_ms == 0 || self.smoke_timeout_ms > self.limits.wall_time_ms {
            return Err(FabricError::InvalidWorkUnit);
        }
        if !safe_relative_directory(&self.working_directory)
            || self.caller_environment.len() > MAX_ENVIRONMENT
            || self.inputs.len() > MAX_INPUTS
            || self.outputs.len() > MAX_OUTPUTS
            || self.outputs.is_empty()
        {
            return Err(FabricError::InvalidWorkUnit);
        }
        for (key, value) in &self.caller_environment {
            if reserved_environment(key)
                || !valid_environment_key(key)
                || value.len() > 16 * 1024
                || value.contains('\0')
            {
                return Err(FabricError::ReservedEnvironment);
            }
        }
        let mut mount_paths = BTreeSet::new();
        let mut total_input_bytes = 0_u64;
        for input in &self.inputs {
            if input.artifact_id.is_empty()
                || input.artifact_digest == [0; 32]
                || input.durable_receipt_digest == [0; 32]
                || input.size_bytes == 0
                || !safe_container_path(&input.mount_path)
                || !mount_paths.insert(input.mount_path.as_str())
            {
                return Err(FabricError::InvalidArtifact);
            }
            total_input_bytes = total_input_bytes
                .checked_add(input.size_bytes)
                .ok_or(FabricError::InvalidArtifact)?;
        }
        let mut output_paths = BTreeSet::new();
        for output in &self.outputs {
            if !safe_relative_directory(&output.relative_path)
                || output.relative_path == "."
                || output.media_type.is_empty()
                || output.maximum_bytes == 0
                || !output_paths.insert(output.relative_path.as_str())
            {
                return Err(FabricError::InvalidArtifact);
            }
        }
        if self.limits.cpu_millicores == 0
            || self.limits.memory_bytes == 0
            || self.limits.scratch_bytes == 0
            || self.limits.pids == 0
            || self.limits.wall_time_ms == 0
            || self.limits.read_bytes == 0
            || self.limits.write_bytes == 0
            || self.limits.io_operations == 0
        {
            return Err(FabricError::InvalidWorkUnit);
        }
        if let Some(gpu) = &self.gpu {
            if gpu.count == 0 || gpu.minimum_vram_bytes == 0 {
                return Err(FabricError::InvalidGpuRequest);
            }
        }
        if self.open_outbound {
            if self.network_bytes == 0
                || self.network_connections == 0
                || self.network_deadline_ms == 0
                || self.network_deadline_ms > self.limits.wall_time_ms
            {
                return Err(FabricError::InvalidNetworkPolicy);
            }
        } else if self.network_bytes != 0
            || self.network_connections != 0
            || self.network_deadline_ms != 0
        {
            return Err(FabricError::InvalidNetworkPolicy);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SignedWorkUnitProjection {
    pub projection: WorkUnitProjection,
    pub issuer: String,
    pub image_signature: DomainSignature,
    pub work_unit_signature: DomainSignature,
}

/// A projection whose exact bytes and image digest were cryptographically verified.
/// Its fields are intentionally not publicly constructible.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedWorkUnit {
    signed: SignedWorkUnitProjection,
    work_unit_digest: Digest32,
    command_digest: Digest32,
}

impl VerifiedWorkUnit {
    pub fn verify(
        signed: SignedWorkUnitProjection,
        trusted_work_issuer: &str,
        trusted_work_public_key: Digest32,
        trusted_image_issuer: &str,
        trusted_image_public_key: Digest32,
    ) -> Result<Self, FabricError> {
        signed.projection.validate()?;
        if signed.issuer != trusted_work_issuer {
            return Err(FabricError::InvalidSignature);
        }
        let work_bytes = projection_signed_bytes(&signed.projection)?;
        signed.work_unit_signature.verify_exact(
            SignatureAuthority::WorkUnitIssuer,
            trusted_work_issuer,
            trusted_work_public_key,
            &work_bytes,
        )?;
        let image_bytes = image_signed_bytes(
            &signed.projection.image_reference,
            signed.projection.image_digest,
        )?;
        signed.image_signature.verify_exact(
            SignatureAuthority::OciImage,
            trusted_image_issuer,
            trusted_image_public_key,
            &image_bytes,
        )?;
        let command_digest = command_digest(&signed.projection)?;
        Ok(Self {
            signed,
            work_unit_digest: sha256(&work_bytes),
            command_digest,
        })
    }

    pub fn validate(&self) -> Result<(), FabricError> {
        self.signed.projection.validate()?;
        if sha256(&projection_signed_bytes(&self.signed.projection)?) != self.work_unit_digest
            || command_digest(&self.signed.projection)? != self.command_digest
        {
            return Err(FabricError::InvalidWorkUnit);
        }
        Ok(())
    }

    pub const fn work_unit_digest(&self) -> Digest32 {
        self.work_unit_digest
    }

    pub const fn command_digest(&self) -> Digest32 {
        self.command_digest
    }

    pub fn issuer(&self) -> &str {
        &self.signed.issuer
    }

    /// Return the exact signed envelope that produced this verified projection.
    /// Callers must re-run `verify` after loading these bytes from durable storage.
    pub fn signed_projection(&self) -> &SignedWorkUnitProjection {
        &self.signed
    }
}

impl Deref for VerifiedWorkUnit {
    type Target = WorkUnitProjection;

    fn deref(&self) -> &Self::Target {
        &self.signed.projection
    }
}
