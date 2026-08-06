//! Encoder V2 execution and capability attestations.

use crate::encoder_v2_identity::{
    cache_key, ArtifactIdentity, FoundationDatasetInputV2, FoundationEncoderIdentityV2,
    OUTPUT_BYTES, OUTPUT_FEATURES,
};
use crate::selfplay_data::canonical_json_sha256;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const CAPABILITY_SCHEMA: &str = "JopeEncoderCapabilityReceipt.v1";
pub const EXECUTION_SCHEMA: &str = "JopeEncoderExecutionReceipt.v2";
pub const ATTEMPT_SCHEMA: &str = "JopeEncoderAttemptReceipt.v2";

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_sha256(value: &str, field: &str) -> Result<()> {
    ensure!(is_sha256(value), "{field} is not a lowercase SHA-256");
    Ok(())
}

fn validate_freshness(generated: u64, budget_seconds: u64, now: u64) -> Result<()> {
    ensure!(budget_seconds > 0, "freshness budget is zero");
    ensure!(now >= generated, "receipt is future-dated");
    ensure!(
        now - generated <= budget_seconds.saturating_mul(1_000),
        "receipt is stale"
    );
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderCapabilityReceiptV1 {
    pub schema: String,
    pub generated_at_unix_ms: u64,
    pub freshness_budget_seconds: u64,
    pub node: String,
    pub runtime_bundle_sha256: String,
    pub encoder_identity_sha256: String,
    pub python_runtime_sha256: String,
    pub landlock_abi: u32,
    pub landlock_enforced: bool,
    pub device_class: String,
    pub gpu_compute_capability: Option<String>,
    pub total_memory_bytes: u64,
    pub package_tree_sha256: BTreeMap<String, String>,
    pub eligible: bool,
    pub blockers: Vec<String>,
}

impl EncoderCapabilityReceiptV1 {
    pub fn validate_for(
        &self,
        encoder: &FoundationEncoderIdentityV2,
        now_unix_ms: u64,
    ) -> Result<()> {
        ensure!(
            self.schema == CAPABILITY_SCHEMA,
            "invalid encoder capability schema"
        );
        validate_freshness(
            self.generated_at_unix_ms,
            self.freshness_budget_seconds,
            now_unix_ms,
        )?;
        ensure!(!self.node.is_empty(), "capability node is absent");
        require_sha256(
            &self.runtime_bundle_sha256,
            "capability runtime bundle SHA-256",
        )?;
        ensure!(
            self.runtime_bundle_sha256 == encoder.bundle_sha256,
            "capability runtime bundle does not match the encoder capsule"
        );
        ensure!(
            self.encoder_identity_sha256 == encoder.digest()?,
            "capability encoder identity mismatch"
        );
        require_sha256(&self.python_runtime_sha256, "Python runtime SHA-256")?;
        ensure!(self.landlock_abi > 0, "Landlock ABI is unavailable");
        ensure!(self.landlock_enforced, "Landlock sandbox is not enforced");
        ensure!(
            self.device_class == encoder.runtime.device_class,
            "capability device class mismatch"
        );
        ensure!(
            self.gpu_compute_capability == encoder.runtime.gpu_compute_capability,
            "capability compute class mismatch"
        );
        ensure!(self.total_memory_bytes > 0, "capability memory is absent");
        let expected = encoder
            .runtime
            .packages
            .iter()
            .map(|(name, package)| (name.clone(), package.package_tree_sha256.clone()))
            .collect::<BTreeMap<_, _>>();
        ensure!(
            self.package_tree_sha256 == expected,
            "capability package-tree identity mismatch"
        );
        ensure!(self.eligible, "encoder capability is ineligible");
        ensure!(self.blockers.is_empty(), "encoder capability has blockers");
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ObservedEncoderRuntime {
    pub actual_backend: String,
    pub selected_device: String,
    pub device_class: String,
    pub gpu_compute_capability: Option<String>,
    pub cuda_version: Option<String>,
    pub driver_version: Option<String>,
    pub pytorch_version: String,
    pub package_tree_sha256: BTreeMap<String, String>,
    pub input_dtype: String,
    pub accumulation_dtype: String,
    pub deterministic_flags: BTreeMap<String, bool>,
    pub loaded_weights: Vec<ArtifactIdentity>,
    pub peak_rss_bytes: u64,
    pub peak_vram_bytes: Option<u64>,
}

impl ObservedEncoderRuntime {
    fn validate_for(&self, encoder: &FoundationEncoderIdentityV2) -> Result<()> {
        ensure!(
            self.actual_backend == encoder.runtime.required_actual_backend,
            "observed backend does not match required backend"
        );
        ensure!(
            !self.selected_device.is_empty(),
            "selected device is absent"
        );
        ensure!(
            self.device_class == encoder.runtime.device_class
                && self.gpu_compute_capability == encoder.runtime.gpu_compute_capability,
            "observed device identity mismatch"
        );
        ensure!(
            self.cuda_version == encoder.runtime.cuda_version
                && self.driver_version == encoder.runtime.driver_version,
            "observed CUDA/driver identity mismatch"
        );
        let pytorch = encoder
            .runtime
            .packages
            .get("pytorch")
            .context("encoder identity does not bind PyTorch")?;
        ensure!(
            self.pytorch_version == pytorch.version,
            "observed PyTorch version mismatch"
        );
        let packages = encoder
            .runtime
            .packages
            .iter()
            .map(|(name, package)| (name.clone(), package.package_tree_sha256.clone()))
            .collect::<BTreeMap<_, _>>();
        ensure!(
            self.package_tree_sha256 == packages,
            "observed package-tree identity mismatch"
        );
        ensure!(
            self.input_dtype == encoder.runtime.input_dtype
                && self.accumulation_dtype == encoder.runtime.accumulation_dtype,
            "observed dtype mismatch"
        );
        ensure!(
            self.deterministic_flags == encoder.runtime.deterministic_flags
                && self.deterministic_flags.values().all(|enabled| *enabled),
            "observed deterministic flags mismatch"
        );
        ensure!(
            self.loaded_weights == encoder.model_weights,
            "loaded model-weight inventory mismatch"
        );
        ensure!(self.peak_rss_bytes > 0, "peak RSS is absent");
        match self.actual_backend.as_str() {
            "cuda" => ensure!(
                self.peak_vram_bytes.is_some_and(|bytes| bytes > 0),
                "CUDA execution has no VRAM observation"
            ),
            "cpu" => ensure!(
                self.peak_vram_bytes.is_none(),
                "CPU execution reports ambiguous GPU memory"
            ),
            _ => ensure!(false, "unknown observed backend"),
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderStageTiming {
    pub input_hash_ns: u64,
    pub preprocess_ns: u64,
    pub expert_load_ns: u64,
    pub kernel_ns: u64,
    pub output_hash_ns: u64,
    pub verification_ns: u64,
    pub total_ns: u64,
}

impl EncoderStageTiming {
    fn validate(&self) -> Result<()> {
        ensure!(self.total_ns > 0, "execution timing is absent");
        let accounted = self
            .input_hash_ns
            .saturating_add(self.preprocess_ns)
            .saturating_add(self.expert_load_ns)
            .saturating_add(self.kernel_ns)
            .saturating_add(self.output_hash_ns)
            .saturating_add(self.verification_ns);
        ensure!(
            accounted <= self.total_ns,
            "stage timing exceeds total time"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderExecutionReceiptV2 {
    pub schema: String,
    pub generated_at_unix_ms: u64,
    pub freshness_budget_seconds: u64,
    pub scq_job_identity_sha256: String,
    pub encoder_identity_sha256: String,
    pub input_identity_sha256: String,
    pub cache_key_sha256: String,
    pub declared_inputs: Vec<ArtifactIdentity>,
    pub sandbox_policy_sha256: String,
    pub child_receipt_sha256: String,
    pub runtime_artifacts_sha256: String,
    pub observed: ObservedEncoderRuntime,
    pub output_sha256: String,
    pub output_vector_count: u32,
    pub output_feature_count: u32,
    pub output_size_bytes: u64,
    pub output_finite: bool,
    pub extra_output_count: u32,
    pub timing: EncoderStageTiming,
}

impl EncoderExecutionReceiptV2 {
    pub fn validate_for(
        &self,
        encoder: &FoundationEncoderIdentityV2,
        input: &FoundationDatasetInputV2,
        now_unix_ms: u64,
    ) -> Result<()> {
        ensure!(
            self.schema == EXECUTION_SCHEMA,
            "invalid encoder execution schema"
        );
        validate_freshness(
            self.generated_at_unix_ms,
            self.freshness_budget_seconds,
            now_unix_ms,
        )?;
        for (value, field) in [
            (&self.scq_job_identity_sha256, "SCQ job identity SHA-256"),
            (&self.encoder_identity_sha256, "encoder identity SHA-256"),
            (&self.input_identity_sha256, "input identity SHA-256"),
            (&self.cache_key_sha256, "cache key SHA-256"),
            (&self.sandbox_policy_sha256, "sandbox policy SHA-256"),
            (&self.child_receipt_sha256, "child receipt SHA-256"),
            (&self.runtime_artifacts_sha256, "runtime artifacts SHA-256"),
            (&self.output_sha256, "encoder output SHA-256"),
        ] {
            require_sha256(value, field)?;
        }
        ensure!(
            self.encoder_identity_sha256 == encoder.digest()?
                && self.input_identity_sha256 == input.digest()?
                && self.cache_key_sha256 == cache_key(encoder, input)?,
            "execution receipt identity mismatch"
        );
        let expected_inputs = input
            .files_read
            .iter()
            .map(|read| read.artifact.clone())
            .collect::<Vec<_>>();
        ensure!(
            self.declared_inputs == expected_inputs,
            "execution receipt declared-input inventory mismatch"
        );
        let mut paths = BTreeSet::new();
        ensure!(
            self.declared_inputs
                .iter()
                .all(|artifact| paths.insert(artifact.path.as_str())),
            "execution receipt repeats an input path"
        );
        self.observed.validate_for(encoder)?;
        ensure!(
            self.output_vector_count == 1
                && self.output_feature_count == OUTPUT_FEATURES
                && self.output_size_bytes == OUTPUT_BYTES,
            "execution receipt output shape is invalid"
        );
        ensure!(
            self.output_finite,
            "execution receipt reports nonfinite output"
        );
        ensure!(
            self.extra_output_count == 0,
            "execution produced undeclared output"
        );
        self.timing.validate()
    }

    pub fn digest(&self) -> Result<String> {
        canonical_json_sha256(&serde_json::to_value(self)?)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderAttemptReceiptV2 {
    pub schema: String,
    pub generated_at_unix_ms: u64,
    pub freshness_budget_seconds: u64,
    pub node: String,
    pub scq_job_identity_sha256: String,
    pub cache_key_sha256: String,
    pub status: String,
    pub failure_class: Option<String>,
    pub execution_receipt_sha256: Option<String>,
    pub output_sha256: Option<String>,
    pub queue_wait_ns: u64,
    pub total_ns: u64,
}

impl EncoderAttemptReceiptV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == ATTEMPT_SCHEMA,
            "invalid encoder attempt schema"
        );
        ensure!(
            self.freshness_budget_seconds > 0,
            "attempt freshness budget is zero"
        );
        ensure!(!self.node.is_empty(), "attempt node is absent");
        require_sha256(
            &self.scq_job_identity_sha256,
            "attempt SCQ identity SHA-256",
        )?;
        require_sha256(&self.cache_key_sha256, "attempt cache key SHA-256")?;
        ensure!(self.total_ns > 0, "attempt timing is absent");
        match self.status.as_str() {
            "succeeded" => {
                ensure!(
                    self.failure_class.is_none(),
                    "successful attempt has a failure class"
                );
                require_sha256(
                    self.execution_receipt_sha256
                        .as_deref()
                        .context("successful attempt lacks execution receipt")?,
                    "attempt execution receipt SHA-256",
                )?;
                require_sha256(
                    self.output_sha256
                        .as_deref()
                        .context("successful attempt lacks output SHA-256")?,
                    "attempt output SHA-256",
                )?;
            }
            "failed" => {
                ensure!(
                    self.failure_class.as_deref().is_some_and(|value| {
                        matches!(
                            value,
                            "capability"
                                | "sandbox"
                                | "input_drift"
                                | "backend_mismatch"
                                | "package_drift"
                                | "malformed_receipt"
                                | "output_shape"
                                | "nonfinite_output"
                                | "extra_output"
                                | "oom"
                                | "child_exit"
                                | "publication"
                                | "nondeterminism"
                        )
                    }),
                    "failed attempt has an unknown failure class"
                );
                ensure!(
                    self.execution_receipt_sha256.is_none() && self.output_sha256.is_none(),
                    "failed attempt claims binding-capable output"
                );
            }
            _ => ensure!(false, "unknown encoder attempt status"),
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        canonical_json_sha256(&serde_json::to_value(self)?)
    }
}

pub fn validate_output_bytes(bytes: &[u8]) -> Result<String> {
    ensure!(
        bytes.len() == OUTPUT_BYTES as usize,
        "encoder output is not exactly {OUTPUT_BYTES} bytes"
    );
    for (index, chunk) in bytes.chunks_exact(4).enumerate() {
        let value = f32::from_le_bytes(chunk.try_into().expect("four-byte chunk"));
        ensure!(
            value.is_finite(),
            "encoder output feature {index} is nonfinite"
        );
    }
    Ok(hex::encode(Sha256::digest(bytes)))
}

pub fn runtime_artifacts_digest(artifacts: &[ArtifactIdentity]) -> Result<String> {
    ensure!(
        !artifacts.is_empty(),
        "runtime artifact inventory is absent"
    );
    for artifact in artifacts {
        artifact.validate()?;
    }
    canonical_json_sha256(&json!({
        "domain": "jope-encoder-runtime-artifacts-v2",
        "artifacts": artifacts,
    }))
}
