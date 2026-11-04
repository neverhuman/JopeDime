use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HyperionEncoderIdentityV1 {
    pub binary: FrozenArtifactV1,
    pub weights: FrozenArtifactV1,
    pub manifest: FrozenArtifactV1,
    pub seed: u64,
    pub sweeps: u32,
    pub scales: Vec<u32>,
    pub batch_size: u32,
    pub device_type: String,
    pub device_ordinal: u32,
    pub runtime_versions: BTreeMap<String, String>,
    pub source_hashes: BTreeMap<String, String>,
}

impl HyperionEncoderIdentityV1 {
    pub fn validate(&self) -> Result<(), String> {
        self.binary.validate()?;
        self.weights.validate()?;
        self.manifest.validate()?;
        if self.seed == 0
            || self.sweeps == 0
            || self.scales.is_empty()
            || self.scales.contains(&0)
            || self.scales.iter().collect::<BTreeSet<_>>().len() != self.scales.len()
            || self.batch_size == 0
            || self.device_type != "cuda"
            || ["rustc", "cuda_driver", "gpu_name", "gpu_uuid"]
                .iter()
                .any(|key| !self.runtime_versions.contains_key(*key))
            || self.source_hashes.is_empty()
            || self
                .runtime_versions
                .iter()
                .any(|(key, value)| key.trim().is_empty() || value.trim().is_empty())
        {
            return Err("encoder configuration is incomplete or is not CUDA-pinned".into());
        }
        for digest in self.source_hashes.values() {
            validate_sha256("encoder source", digest)?;
        }
        Ok(())
    }

    pub fn verify_artifacts(&self) -> Result<(), String> {
        self.validate()?;
        self.binary.verify()?;
        self.weights.verify()?;
        self.manifest.verify()?;
        verify_runtime_identity(self)
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }

    /// Derive the node-oblivious authority used by V2 receipts.  The V1
    /// registry still carries physical placement for admission, but that
    /// proof is intentionally removed from the portable identity before a
    /// cache or policy is frozen.
    pub fn portable_identity_v2(
        &self,
        runtime_class: impl Into<String>,
    ) -> Result<HyperionEncoderIdentityV2, String> {
        self.validate()?;
        let mut runtime_versions = self.runtime_versions.clone();
        runtime_versions.remove("gpu_uuid");
        runtime_versions.remove("gpu_ordinal");
        let identity = HyperionEncoderIdentityV2 {
            schema: HYPERION_ENCODER_IDENTITY_V2_SCHEMA.into(),
            binary: self.binary.clone(),
            weights: self.weights.clone(),
            manifest: self.manifest.clone(),
            seed: self.seed,
            sweeps: self.sweeps,
            scales: self.scales.clone(),
            batch_size: self.batch_size,
            device_type: self.device_type.clone(),
            runtime_class: runtime_class.into(),
            runtime_versions,
            source_hashes: self.source_hashes.clone(),
            argv_schema: hyperion_argv_schema_v2(),
            environment: hyperion_environment_v2(),
        };
        identity.validate()?;
        Ok(identity)
    }
}

pub const HYPERION_ENCODER_IDENTITY_V2_SCHEMA: &str = "HyperionEncoderIdentity.v2";
pub const HYPERION_ENCODER_PROOF_V2_SCHEMA: &str = "HyperionEncoderExecutionProof.v2";

pub fn hyperion_argv_schema_v2() -> Vec<String> {
    [
        "--train",
        "--test",
        "--out-cell-train",
        "--out-cell-test",
        "--out-row-train",
        "--out-row-test",
        "--weights",
        "--manifest",
        "--sweeps",
        "--scales",
        "--seed",
        "--batch",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

pub fn hyperion_environment_v2() -> BTreeMap<String, String> {
    [
        ("JAIN_HYPERION_CUDA_MAX_LOAD_RATIO", "1000000"),
        ("OMP_NUM_THREADS", "2"),
        ("OPENBLAS_NUM_THREADS", "2"),
        ("MKL_NUM_THREADS", "2"),
        ("BLIS_NUM_THREADS", "2"),
        ("VECLIB_MAXIMUM_THREADS", "2"),
        ("NUMEXPR_NUM_THREADS", "2"),
    ]
    .into_iter()
    .map(|(key, value)| (key.into(), value.into()))
    .collect()
}

/// Portable encoder authority. It names software, weights, runtime class,
/// argv, and fixed environment, but deliberately carries no node, GPU UUID,
/// or physical ordinal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HyperionEncoderIdentityV2 {
    pub schema: String,
    pub binary: FrozenArtifactV1,
    pub weights: FrozenArtifactV1,
    pub manifest: FrozenArtifactV1,
    pub seed: u64,
    pub sweeps: u32,
    pub scales: Vec<u32>,
    pub batch_size: u32,
    pub device_type: String,
    pub runtime_class: String,
    pub runtime_versions: BTreeMap<String, String>,
    pub source_hashes: BTreeMap<String, String>,
    pub argv_schema: Vec<String>,
    pub environment: BTreeMap<String, String>,
}

impl HyperionEncoderIdentityV2 {
    pub fn validate(&self) -> Result<(), String> {
        self.binary.validate()?;
        self.weights.validate()?;
        self.manifest.validate()?;
        if self.schema != HYPERION_ENCODER_IDENTITY_V2_SCHEMA
            || self.seed == 0
            || self.sweeps == 0
            || self.scales.is_empty()
            || self.scales.contains(&0)
            || self.scales.iter().collect::<BTreeSet<_>>().len() != self.scales.len()
            || self.batch_size == 0
            || self.device_type != "cuda"
            || self.runtime_class.trim().is_empty()
            || self.argv_schema != hyperion_argv_schema_v2()
            || self.environment != hyperion_environment_v2()
            || self.runtime_versions.contains_key("gpu_uuid")
            || self.runtime_versions.contains_key("gpu_ordinal")
            || ["rustc", "cuda_driver", "gpu_name"]
                .iter()
                .any(|key| self.runtime_versions.get(*key).is_none_or(String::is_empty))
            || self.source_hashes.is_empty()
        {
            return Err("Hyperion V2 portable identity is incomplete or drifted".into());
        }
        for digest in self.source_hashes.values() {
            validate_sha256("Hyperion V2 source", digest)?;
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HyperionEncoderExecutionProofV2 {
    pub schema: String,
    pub encoder_identity_sha256: String,
    pub node: String,
    pub gpu_uuid: String,
    pub gpu_name: String,
    pub cuda_driver: String,
    pub cuda_visible_devices: String,
    pub observed_device: String,
    pub argv_sha256: String,
    pub environment_sha256: String,
    pub stdout: String,
    pub stdout_sha256: String,
    pub stderr: String,
    pub stderr_sha256: String,
}

impl HyperionEncoderExecutionProofV2 {
    pub fn validate(&self, identity: &HyperionEncoderIdentityV2) -> Result<(), String> {
        identity.validate()?;
        if self.schema != HYPERION_ENCODER_PROOF_V2_SCHEMA
            || self.node.trim().is_empty()
            || self.gpu_uuid.trim().is_empty()
            || self.gpu_name != identity.runtime_versions["gpu_name"]
            || self.cuda_driver != identity.runtime_versions["cuda_driver"]
            || self.cuda_visible_devices != self.gpu_uuid
            || self.observed_device != "cuda:0"
            || !encoder_output_proves_device(&self.stdout, &self.stderr, "cuda:0")
            || self.argv_sha256 != canonical_sha256(&identity.argv_schema)?
            || self.environment_sha256 != canonical_sha256(&identity.environment)?
        {
            return Err("Hyperion V2 execution proof does not match portable authority".into());
        }
        if self.encoder_identity_sha256 != identity.identity()? {
            return Err("Hyperion V2 execution proof names a different encoder".into());
        }
        for (name, actual, expected) in [
            (
                "stdout",
                &self.stdout_sha256,
                raw_sha256(self.stdout.as_bytes()),
            ),
            (
                "stderr",
                &self.stderr_sha256,
                raw_sha256(self.stderr.as_bytes()),
            ),
        ] {
            validate_sha256(name, actual)?;
            if actual != &expected {
                return Err(format!("Hyperion V2 {name} hash mismatch"));
            }
        }
        Ok(())
    }
}
