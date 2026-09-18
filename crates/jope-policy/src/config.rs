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
    ACCEPTED_STATE_SHA256, CHECKPOINT_VOCABULARY_SHA256, REJECTED_STEP,
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

/// The checkpoint identity carried by a `jope.checkpoint_export.v1` receipt, checked.
///
/// The fields are private: outside this crate's test support, the only constructor is
/// [`CheckpointBinding::from_receipt_bytes`], so a binding always names the accepted state.
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
        ensure!(
            checkpoint.get("accepted_identity").and_then(Value::as_bool) == Some(true)
                && digest(checkpoint, &["expected_state_sha256"])? == state_sha256,
            "export receipt does not carry a verified accepted identity"
        );
        if state_sha256 != ACCEPTED_STATE_SHA256 {
            bail!("checkpoint {state_sha256} is not the accepted step-60000 state");
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
        Ok(Self {
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
        })
    }
}

impl CheckpointBinding {
    pub fn receipt_sha256(&self) -> &str {
        &self.receipt_sha256
    }

    pub fn step(&self) -> u64 {
        self.step
    }

    /// `state.json` sha256; always [`ACCEPTED_STATE_SHA256`] for a parsed receipt.
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

    /// Refuse unless this receipt names the accepted step-60000 configuration and vocabulary:
    /// the claimed `config_sha256` and the hash recomputed from the parsed `checkpoint.config`
    /// ([`PolicyConfig::sha256`]) must both be [`super::ACCEPTED_CONFIG_SHA256`], and the carried
    /// vocabulary must be [`super::CHECKPOINT_VOCABULARY_SHA256`].
    pub fn require_accepted_identity(&self) -> Result<()> {
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
