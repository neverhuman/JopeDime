//! Policy configuration and the checkpoint export-receipt binding.
//!
//! [`PolicyConfig`] is jope-dime `neural::Config` (`neural.rs:15-70`) with the same field names
//! and `deny_unknown_fields`, so a step-60000 `config.json` or the `checkpoint.config` object of
//! a `jope.checkpoint_export.v1` receipt parses unchanged and anything else is refused.
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{
    is_sha256_hex, refuse_rejected_checkpoint, sha256_hex, ACCEPTED_CONFIG_SHA256,
    ACCEPTED_MODEL_SHA256, ACCEPTED_PARAMETER_SHA256, ACCEPTED_STATE_SHA256, ACCEPTED_STEP,
    CHECKPOINT_VOCABULARY_SHA256, REJECTED_STEP,
};

/// Schema tag written by jain-datasets `tools/ot-export` (plan v2 W4 PR-A).
pub const EXPORT_RECEIPT_SCHEMA: &str = "jope.checkpoint_export.v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyConfig {
    pub dataset_dim: usize,
    pub target_dim: usize,
    pub target_keep_dim: usize,
    pub width: usize,
    pub layers: usize,
    pub heads: usize,
    pub ffn: usize,
    pub vocab_size: usize,
    pub embedding_rows: usize,
    pub max_seq: usize,
    pub dropout: f64,
    pub dataset_rms: f64,
    pub target_rms: f64,
}

impl PolicyConfig {
    /// jope-dime `Config::regression` (`neural.rs:33-49`).
    pub fn regression(vocab_size: usize) -> Self {
        Self {
            dataset_dim: 1536,
            target_dim: 512,
            target_keep_dim: 512,
            width: 512,
            layers: 6,
            heads: 4,
            ffn: 2048,
            vocab_size,
            embedding_rows: vocab_size + 256,
            max_seq: 128,
            dropout: 0.1,
            dataset_rms: 1.0,
            target_rms: 1.0,
        }
    }

    /// Parse a JSON config, refusing unknown fields, then validate it.
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let config: Self = serde_json::from_slice(bytes).context("policy config")?;
        config.validate()?;
        Ok(config)
    }

    /// jope-dime `Config::validate` plus the context width this port assumes.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.width > 0
                && self.heads > 0
                && self.width.is_multiple_of(self.heads)
                && self.layers > 0
                && self.vocab_size > 2
                && self.embedding_rows >= self.vocab_size
                && self.target_keep_dim > 0
                && self.target_keep_dim <= self.target_dim
                && self.max_seq > 1
                && self.dataset_rms.is_finite()
                && self.dataset_rms > 0.0
                && self.target_rms.is_finite()
                && self.target_rms > 0.0
                && self.dropout >= 0.0
                && self.dropout < 1.0,
            "invalid policy configuration"
        );
        Ok(())
    }

    /// sha256 of this configuration as a `config.json` file: `serde_json` pretty printing
    /// (two-space indent), fields in declaration order, no trailing newline. That is the byte
    /// layout of step-60000's `config.json`, whose file sha256 `tools/ot-export` records as
    /// `checkpoint.config_sha256`, so the accepted configuration hashes to
    /// [`super::ACCEPTED_CONFIG_SHA256`]. The receipt re-emits the object with sorted keys, so its
    /// bytes cannot be hashed directly; parsing into this struct and re-serializing restores the
    /// file layout.
    pub fn sha256(&self) -> Result<String> {
        Ok(sha256_hex(
            &serde_json::to_vec_pretty(self).context("serializing policy config")?,
        ))
    }

    /// Length of one saved context vector (dataset block + target block).
    pub fn context_len(&self) -> usize {
        self.dataset_dim + self.target_dim
    }
}

/// One row of the receipt's `policy.tensors` inventory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorEntry {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub numel: usize,
    pub bytes: usize,
    pub sha256: String,
}

/// `policy.safetensors` sha256 of the accepted step-60000 export
/// (`models/bundles/policy-step-60000.json:41-44`): the release decision for the weights file,
/// which [`CheckpointBinding::require_accepted_identity`] and `require_accepted_policy` compare
/// with instead of the receipt's own `policy.output.sha256`. Crate-private: no public
/// `ACCEPTED_*` item changes, and the existing literals (`producer.rs`, jope-bench
/// `optimize.rs`) stay as they are.
pub(crate) const ACCEPTED_WEIGHTS_SHA256: &str =
    "4b4b5a40f03c9226b4b15856fd35b90127c8b6a4d1435b815484cdd050744f1e";

/// A preregistered native challenger checkpoint: pins only. Deserializing one proves nothing;
/// [`CheckpointBinding::from_challenger_receipt_bytes`] and `PolicyModel::load_bound` prove the
/// bytes. It never names the accepted identity, so a challenger never satisfies
/// `require_accepted_policy`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChallengerIdentity {
    /// `state.json` sha256 of the challenger checkpoint.
    pub state_sha256: String,
    /// Parameter hash its `state.json` records (sorted names + little-endian f32 values).
    pub parameter_sha256: String,
    /// `config.json` sha256 its `state.json` records.
    pub config_sha256: String,
    /// sha256 of the exported `policy.safetensors`.
    pub policy_safetensors_sha256: String,
}

impl ChallengerIdentity {
    /// Structural checks only, before any file is read: four lowercase sha256 pins, never the
    /// rejected step-69000 state, and never the accepted state, parameters or weights.
    pub fn validate(&self) -> Result<()> {
        for pin in [
            &self.state_sha256,
            &self.parameter_sha256,
            &self.config_sha256,
            &self.policy_safetensors_sha256,
        ] {
            // is_sha256_hex accepts uppercase; identities compare exactly.
            ensure!(
                is_sha256_hex(pin) && !pin.bytes().any(|b| b.is_ascii_uppercase()),
                "challenger identity pins must be lowercase sha256"
            );
        }
        refuse_rejected_checkpoint(&self.state_sha256, "")?;
        ensure!(
            !super::is_accepted_state(&self.state_sha256)
                && self.parameter_sha256 != ACCEPTED_PARAMETER_SHA256
                && self.policy_safetensors_sha256 != ACCEPTED_WEIGHTS_SHA256,
            "refused: challenger names the accepted identity"
        );
        Ok(())
    }
}

/// The checkpoint identity carried by a `jope.checkpoint_export.v1` receipt, checked.
///
/// The fields are private: outside this crate's test support, the only constructors are
/// [`CheckpointBinding::from_receipt_bytes`], which always names the accepted state, and
/// [`CheckpointBinding::from_challenger_receipt_bytes`], which names exactly a pinned
/// non-accepted challenger.
#[derive(Debug, Clone)]
pub struct CheckpointBinding {
    receipt_sha256: String,
    step: u64,
    state_sha256: String,
    model_sha256: String,
    config_sha256: String,
    parameter_sha256: String,
    vocabulary_sha256: String,
    corpus_sha256: String,
    policy_safetensors_sha256: String,
    config: PolicyConfig,
    tensors: Vec<TensorEntry>,
}

fn text<'a>(value: &'a Value, path: &[&str]) -> Result<&'a str> {
    let mut at = value;
    for key in path {
        at = at
            .get(key)
            .with_context(|| format!("export receipt lacks {}", path.join(".")))?;
    }
    at.as_str()
        .with_context(|| format!("export receipt {} is not text", path.join(".")))
}

fn digest(value: &Value, path: &[&str]) -> Result<String> {
    let found = text(value, path)?;
    ensure!(
        is_sha256_hex(found),
        "export receipt {} is not a sha256",
        path.join(".")
    );
    Ok(found.to_ascii_lowercase())
}

impl CheckpointBinding {
    /// Parse and check an export receipt. Refuses the rejected step-69000 checkpoint, any
    /// identity other than the accepted step-60000 state, unverified bindings, and a
    /// configuration with unknown fields.
    pub fn from_receipt_bytes(bytes: &[u8]) -> Result<Self> {
        Self::parse(bytes, None)
    }

    /// Parse a receipt written for a pinned native challenger. Refuses the rejected checkpoint
    /// first, the accepted identity, and any pin the receipt does not reproduce.
    pub fn from_challenger_receipt_bytes(
        bytes: &[u8],
        identity: &ChallengerIdentity,
    ) -> Result<Self> {
        Self::parse(bytes, Some(identity))
    }

    /// The binding a proposal request names: the complete accepted identity, or exactly its
    /// pinned native challenger. Every policy-worker entry point parses through here.
    pub fn for_proposal(bytes: &[u8], challenger: Option<&ChallengerIdentity>) -> Result<Self> {
        match challenger {
            None => {
                let binding = Self::from_receipt_bytes(bytes)?;
                binding.require_accepted_identity()?;
                Ok(binding)
            }
            Some(identity) => Self::from_challenger_receipt_bytes(bytes, identity),
        }
    }

    fn parse(bytes: &[u8], challenger: Option<&ChallengerIdentity>) -> Result<Self> {
        let receipt: Value = serde_json::from_slice(bytes).context("export receipt")?;
        ensure!(
            receipt.get("schema").and_then(Value::as_str) == Some(EXPORT_RECEIPT_SCHEMA),
            "export receipt schema is not {EXPORT_RECEIPT_SCHEMA}"
        );
        let checkpoint = receipt
            .get("checkpoint")
            .context("export receipt lacks checkpoint")?;
        let state_sha256 = digest(checkpoint, &["state_sha256"])?;
        let model_sha256 = digest(checkpoint, &["model_sha256"])?;
        let step = checkpoint
            .get("step")
            .and_then(Value::as_u64)
            .context("export receipt lacks checkpoint.step")?;
        // Refusal comes first so a rejected checkpoint never reaches any other check.
        refuse_rejected_checkpoint(&state_sha256, &model_sha256)?;
        ensure!(
            step != REJECTED_STEP,
            "refused: rejected step-{REJECTED_STEP} checkpoint"
        );
        match challenger {
            None => {
                ensure!(
                    checkpoint.get("accepted_identity").and_then(Value::as_bool) == Some(true)
                        && digest(checkpoint, &["expected_state_sha256"])? == state_sha256,
                    "export receipt does not carry a verified accepted identity"
                );
                if state_sha256 != ACCEPTED_STATE_SHA256 {
                    bail!("checkpoint {state_sha256} is not the accepted step-60000 state");
                }
            }
            Some(identity) => {
                identity.validate()?;
                ensure!(
                    checkpoint.get("accepted_identity").and_then(Value::as_bool) == Some(false)
                        && digest(checkpoint, &["expected_state_sha256"])? == state_sha256
                        && state_sha256 == identity.state_sha256,
                    "export receipt does not carry the pinned challenger identity"
                );
            }
        }
        let parameter_sha256 = digest(checkpoint, &["parameter_sha256"])?;
        ensure!(
            digest(checkpoint, &["parameter_sha256_recomputed"])? == parameter_sha256,
            "export receipt parameter hash was not reproduced"
        );
        let verified = checkpoint
            .get("verified_bindings")
            .and_then(Value::as_array)
            .context("export receipt lacks verified_bindings")?;
        for required in ["model_sha256", "config_sha256", "parameter_sha256"] {
            ensure!(
                verified.iter().any(|v| v.as_str() == Some(required)),
                "export receipt did not verify {required}"
            );
        }
        let config: PolicyConfig = serde_json::from_value(
            checkpoint
                .get("config")
                .cloned()
                .context("export receipt lacks checkpoint.config")?,
        )
        .context("export receipt checkpoint.config")?;
        config.validate()?;
        let policy = receipt
            .get("policy")
            .context("export receipt lacks policy")?;
        let tensors: Vec<TensorEntry> = serde_json::from_value(
            policy
                .get("tensors")
                .cloned()
                .context("export receipt lacks policy.tensors")?,
        )
        .context("export receipt policy.tensors")?;
        ensure!(
            policy.get("tensor_count").and_then(Value::as_u64) == Some(tensors.len() as u64),
            "export receipt tensor_count disagrees with its inventory"
        );
        let binding = Self {
            receipt_sha256: sha256_hex(bytes),
            step,
            state_sha256,
            model_sha256,
            config_sha256: digest(checkpoint, &["config_sha256"])?,
            parameter_sha256,
            vocabulary_sha256: digest(checkpoint, &["carried_bindings", "vocabulary_sha256"])?,
            corpus_sha256: digest(checkpoint, &["carried_bindings", "corpus_sha256"])?,
            policy_safetensors_sha256: digest(policy, &["output", "sha256"])?,
            config,
            tensors,
        };
        if let Some(identity) = challenger {
            binding.require_challenger_identity(identity)?;
        }
        Ok(binding)
    }

    /// Refuse unless this receipt names exactly the pinned challenger: its state, parameters,
    /// claimed and recomputed configuration, exported weights, the step-60000 vocabulary and a
    /// step after the accepted one; never the accepted or rejected model.
    pub fn require_challenger_identity(&self, identity: &ChallengerIdentity) -> Result<()> {
        identity.validate()?;
        refuse_rejected_checkpoint(&self.state_sha256, &self.model_sha256)?;
        ensure!(
            self.step > ACCEPTED_STEP && self.step != REJECTED_STEP,
            "refused: challenger step {} does not continue the accepted step-{ACCEPTED_STEP} checkpoint",
            self.step
        );
        ensure!(
            self.state_sha256 == identity.state_sha256
                && self.model_sha256 != ACCEPTED_MODEL_SHA256,
            "refused: checkpoint is not the pinned challenger"
        );
        ensure!(
            self.parameter_sha256 == identity.parameter_sha256,
            "refused: checkpoint parameters are not the pinned challenger parameters"
        );
        ensure!(
            self.config_sha256 == identity.config_sha256
                && self.config.sha256()? == identity.config_sha256,
            "refused: checkpoint config is not the pinned challenger config"
        );
        ensure!(
            self.policy_safetensors_sha256 == identity.policy_safetensors_sha256,
            "refused: export receipt names other challenger weights"
        );
        ensure!(
            self.vocabulary_sha256 == CHECKPOINT_VOCABULARY_SHA256,
            "refused: checkpoint vocabulary {} is not the step-60000 vocabulary",
            self.vocabulary_sha256
        );
        Ok(())
    }
}

impl CheckpointBinding {
    pub fn receipt_sha256(&self) -> &str {
        &self.receipt_sha256
    }

    pub fn step(&self) -> u64 {
        self.step
    }

    /// `state.json` sha256: the accepted state, or the pinned challenger's.
    pub fn state_sha256(&self) -> &str {
        &self.state_sha256
    }

    pub fn model_sha256(&self) -> &str {
        &self.model_sha256
    }

    pub fn config_sha256(&self) -> &str {
        &self.config_sha256
    }

    /// Parameter hash as claimed by the receipt (and reproduced by the exporter).
    pub fn parameter_sha256(&self) -> &str {
        &self.parameter_sha256
    }

    pub fn vocabulary_sha256(&self) -> &str {
        &self.vocabulary_sha256
    }

    pub fn corpus_sha256(&self) -> &str {
        &self.corpus_sha256
    }

    pub fn policy_safetensors_sha256(&self) -> &str {
        &self.policy_safetensors_sha256
    }

    pub fn config(&self) -> &PolicyConfig {
        &self.config
    }

    /// Refuse unless this receipt names the complete accepted step-60000 checkpoint identity:
    /// step, state, source model, parameters, configuration, vocabulary, and the released
    /// `policy.safetensors` file (the receipt's own weights hash is only a claim).
    pub fn require_accepted_identity(&self) -> Result<()> {
        ensure!(
            self.step == ACCEPTED_STEP,
            "refused: checkpoint step {} is not the accepted step-{ACCEPTED_STEP}",
            self.step
        );
        ensure!(
            self.state_sha256 == ACCEPTED_STATE_SHA256,
            "refused: checkpoint state {} is not the accepted step-{ACCEPTED_STEP} state",
            self.state_sha256
        );
        ensure!(
            self.model_sha256 == ACCEPTED_MODEL_SHA256,
            "refused: checkpoint model_sha256 {} is not the accepted step-{ACCEPTED_STEP} model",
            self.model_sha256
        );
        ensure!(
            self.parameter_sha256 == ACCEPTED_PARAMETER_SHA256,
            "refused: checkpoint parameters are not the accepted step-{ACCEPTED_STEP} parameters"
        );
        ensure!(
            self.config_sha256 == ACCEPTED_CONFIG_SHA256,
            "refused: checkpoint config_sha256 {} is not the accepted step-60000 config",
            self.config_sha256
        );
        let recomputed = self.config.sha256()?;
        ensure!(
            recomputed == ACCEPTED_CONFIG_SHA256,
            "refused: checkpoint config hashes to {recomputed}, not the accepted step-60000 config"
        );
        ensure!(
            self.vocabulary_sha256 == CHECKPOINT_VOCABULARY_SHA256,
            "refused: checkpoint vocabulary {} is not the step-60000 vocabulary",
            self.vocabulary_sha256
        );
        ensure!(
            self.policy_safetensors_sha256 == ACCEPTED_WEIGHTS_SHA256,
            "refused: export receipt names weights {} that are not the released step-{ACCEPTED_STEP} policy.safetensors",
            self.policy_safetensors_sha256
        );
        // W2-T02: the accepted semantic manifest is the release decision; the constants above
        // are its values (`semantic::tests::every_compiled_identity_is_the_manifests`), so their
        // refusal wording is kept and this check binds the whole identity to the manifest.
        let release = crate::semantic::accepted()?;
        ensure!(
            self.step == release.policy.step
                && self.state_sha256 == release.policy.state_sha256
                && self.model_sha256 == release.policy.model_sha256
                && self.parameter_sha256 == release.policy.parameter_sha256
                && self.config_sha256 == release.policy.config_sha256
                && self.vocabulary_sha256 == release.vocabulary.checkpoint_sha256
                && self.policy_safetensors_sha256 == release.policy.weights_sha256,
            "refused: checkpoint is not the accepted semantic release {}",
            crate::semantic::ACCEPTED_SEMANTIC_MANIFEST_SHA256
        );
        Ok(())
    }

    pub fn tensors(&self) -> &[TensorEntry] {
        &self.tensors
    }

    /// Test support only (`testing`): a binding for untrained weights. Its state is
    /// [`super::testing::UNBOUND_TEST_STATE_SHA256`], which the production `propose` refuses,
    /// and its other digests are labels, not file hashes.
    #[cfg(any(test, feature = "testing"))]
    pub fn unbound_for_testing(config: PolicyConfig, vocabulary_sha256: &str) -> Self {
        let label = |name: &str| sha256_hex(format!("jope-dime unbound test {name}").as_bytes());
        Self {
            receipt_sha256: label("receipt"),
            step: 0,
            state_sha256: super::testing::UNBOUND_TEST_STATE_SHA256.into(),
            model_sha256: label("model"),
            config_sha256: label("config"),
            parameter_sha256: label("parameters"),
            vocabulary_sha256: vocabulary_sha256.to_ascii_lowercase(),
            corpus_sha256: label("corpus"),
            policy_safetensors_sha256: label("policy"),
            config,
            tensors: Vec::new(),
        }
    }
}
