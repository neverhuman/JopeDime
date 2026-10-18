//! Pure preflight for the I11 allocated training-intent identity.
//!
//! I11's typed DTO is not present on this base, so this adapter consumes its
//! JSON projection without allocating a competing serialized Rust type.
use crate::manifest::hash;
use anyhow::{ensure, Context, Result};
use serde_json::Value;
use std::collections::BTreeMap;

pub const INTENT_SCHEMA: &str = "jope.training-intent/v1";
pub const FROZEN_INPUTS_SCHEMA: &str = "jope.training_intent_inputs/v1";

/// Read-only state observed before any device/context allocation or optimizer
/// operation. Shape maps are keyed by the stable parameter names in the parent.
#[derive(Clone, Debug)]
pub struct ParentState<'a> {
    pub checkpoint_sha256: &'a str,
    pub optimizer_epoch_sha256: &'a str,
    pub optimizer_step: u64,
    pub parameter_shapes: &'a BTreeMap<String, Vec<i64>>,
    pub first_moment_shapes: &'a BTreeMap<String, Vec<i64>>,
    pub second_moment_shapes: &'a BTreeMap<String, Vec<i64>>,
}

/// Result is a local validation receipt, not a grant and not a launch token.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatedIntent {
    pub intent_sha256: String,
    pub input_closure_sha256: String,
    pub initial_step: u64,
    pub target_step: u64,
}

/// Validate an I11 intent against its frozen input closure and the accepted
/// parent's optimizer state. This function is deliberately pure: it has no
/// device, filesystem, journal, optimizer, or grant access.
pub fn validate(
    intent: &Value,
    frozen_inputs: &Value,
    parent: &ParentState<'_>,
) -> Result<ValidatedIntent> {
    ensure!(
        intent["schema"].as_str() == Some(INTENT_SCHEMA),
        "unsupported training intent schema"
    );
    ensure!(
        frozen_inputs["schema"].as_str() == Some(FROZEN_INPUTS_SCHEMA),
        "unsupported frozen intent inputs schema"
    );
    let parent_sha = digest_field(intent, "parent_checkpoint_sha256")?;
    ensure!(
        parent_sha == parent.checkpoint_sha256,
        "training intent parent differs from accepted checkpoint"
    );
    let epoch_sha = digest_field(intent, "optimizer_epoch_sha256")?;
    ensure!(
        epoch_sha == parent.optimizer_epoch_sha256,
        "optimizer epoch mismatch"
    );
    let initial = intent["initial_step"]
        .as_u64()
        .context("intent initial_step missing")?;
    let target = intent["target_step"]
        .as_u64()
        .context("intent target_step missing")?;
    ensure!(
        initial == parent.optimizer_step,
        "intent epoch step differs from optimizer step"
    );
    ensure!(
        target > initial,
        "intent target step must advance the accepted parent"
    );

    let protocol = intent["protocol"]
        .as_object()
        .context("intent protocol missing")?;
    for name in [
        "objective_sha256",
        "recipe_sha256",
        "architecture_sha256",
        "vocabulary_sha256",
        "support_sha256",
    ] {
        require_digest(protocol.get(name).and_then(Value::as_str), name)?;
    }
    for name in [
        "corpus_sha256",
        "execution_partition_sha256",
        "sampler_sha256",
    ] {
        digest_field(intent, name)?;
    }
    let corpus_sha = intent["corpus_sha256"].as_str().unwrap();
    ensure!(
        frozen_inputs["base_inputs"]["corpus"]["index_sha256"].as_str() == Some(corpus_sha),
        "intent corpus differs from frozen input snapshot"
    );

    // Exact key and shape agreement catches absent, extra, or mismatched Adam
    // moments before the native path can allocate CUDA state or mutate AdamW.
    ensure!(
        !parent.parameter_shapes.is_empty(),
        "accepted parent has no parameter shapes"
    );
    ensure!(
        parent.first_moment_shapes == parent.parameter_shapes,
        "first optimizer moment shape/keys differ from accepted parameters"
    );
    ensure!(
        parent.second_moment_shapes == parent.parameter_shapes,
        "second optimizer moment shape/keys differ from accepted parameters"
    );
    let intent_sha256 = hash(&serde_json::to_vec(intent)?);
    let input_closure_sha256 = hash(&serde_json::to_vec(frozen_inputs)?);
    Ok(ValidatedIntent {
        intent_sha256,
        input_closure_sha256,
        initial_step: initial,
        target_step: target,
    })
}

fn digest_field<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    let field = value.get(name).and_then(Value::as_str);
    require_digest(field, name)
}

fn require_digest<'a>(value: Option<&'a str>, name: &str) -> Result<&'a str> {
    let value = value.with_context(|| format!("{name} missing"))?;
    ensure!(
        value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit()),
        "{name} must be a SHA256 digest"
    );
    Ok(value)
}
