//! Additive checkpoint export v2 lineage validation for non-serving challengers.
//!
//! This module defines the worker draft wire shape until I11/I13 land their canonical types.
//! It is deliberately separate from strict v1 parsing and never constructs an accepted binding.

use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use serde_json::Value;

use super::{ACCEPTED_PARAMETER_SHA256, ACCEPTED_STATE_SHA256};
use crate::REJECTED_STATE_SHA256;

pub const EXPORT_RECEIPT_V2_SCHEMA: &str = "jope.checkpoint_export.v2";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReceiptV2 {
    schema: String,
    checkpoint: CheckpointV2,
    lineage: LineageV2,
    delta: DeltaV2,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckpointV2 {
    step: u64,
    state_sha256: String,
    model_sha256: String,
    config_sha256: String,
    parameter_sha256: String,
    tensor_inventory_sha256: String,
    tensors: Vec<TensorV2>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorV2 {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    finite: bool,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LineageV2 {
    accepted_anchor_state_sha256: String,
    parent_state_sha256: String,
    ancestors: Vec<AncestorV2>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AncestorV2 {
    state_sha256: String,
    parameter_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DeltaV2 {
    committed_updates: u64,
    optimizer_epoch_sha256: String,
    parent_parameter_sha256: String,
    child_parameter_sha256: String,
}

/// Validate the provisional v2 receipt shape, ancestry, committed update and tensor inventory.
/// The return value is only a challenger claim; callers must never route it to serving identity.
pub fn validate_receipt_v2(bytes: &[u8]) -> Result<()> {
    let receipt: ReceiptV2 = serde_json::from_slice(bytes).context("checkpoint export v2")?;
    ensure!(
        receipt.schema == EXPORT_RECEIPT_V2_SCHEMA,
        "wrong checkpoint export v2 schema"
    );
    let checkpoint = receipt.checkpoint;
    let lineage = receipt.lineage;
    let delta = receipt.delta;

    // The round count is relative to the warm-start epoch. A step count alone never establishes
    // ancestry or authorizes a checkpoint.
    ensure!(
        matches!(checkpoint.step, 3000 | 6000),
        "refused: unsupported challenger round"
    );
    for digest in [
        &checkpoint.state_sha256,
        &checkpoint.model_sha256,
        &checkpoint.config_sha256,
        &checkpoint.parameter_sha256,
        &checkpoint.tensor_inventory_sha256,
        &lineage.accepted_anchor_state_sha256,
        &lineage.parent_state_sha256,
        &delta.optimizer_epoch_sha256,
        &delta.parent_parameter_sha256,
        &delta.child_parameter_sha256,
    ] {
        ensure!(
            is_lower_digest(digest),
            "v2 identity must be lowercase sha256"
        );
    }
    ensure!(
        lineage.accepted_anchor_state_sha256 == ACCEPTED_STATE_SHA256,
        "refused: challenger has a different accepted anchor"
    );
    ensure!(
        checkpoint.state_sha256 != ACCEPTED_STATE_SHA256
            && checkpoint.state_sha256 != REJECTED_STATE_SHA256,
        "refused: candidate names an accepted or rejected checkpoint"
    );
    ensure!(
        checkpoint.parameter_sha256 != ACCEPTED_PARAMETER_SHA256,
        "refused: candidate aliases accepted parameters"
    );
    ensure!(
        delta.committed_updates == checkpoint.step,
        "committed update count disagrees with round"
    );
    ensure!(
        delta.parent_parameter_sha256 == ACCEPTED_PARAMETER_SHA256
            && delta.child_parameter_sha256 == checkpoint.parameter_sha256,
        "committed delta does not bind the accepted parent and candidate parameters"
    );
    ensure!(!lineage.ancestors.is_empty(), "refused: ancestry is empty");
    ensure!(
        lineage.ancestors.first().map(|a| a.state_sha256.as_str()) == Some(ACCEPTED_STATE_SHA256),
        "refused: ancestry does not begin at the accepted anchor"
    );
    ensure!(
        lineage.ancestors.last().map(|a| a.state_sha256.as_str())
            == Some(lineage.parent_state_sha256.as_str()),
        "refused: declared parent is not the terminal verified ancestor"
    );
    for ancestor in &lineage.ancestors {
        ensure!(
            is_lower_digest(&ancestor.state_sha256) && is_lower_digest(&ancestor.parameter_sha256),
            "ancestor identity must be lowercase sha256"
        );
        ensure!(
            ancestor.state_sha256 != REJECTED_STATE_SHA256,
            "refused: rejected checkpoint in ancestry"
        );
    }
    ensure!(
        !checkpoint.tensors.is_empty(),
        "refused: tensor inventory is empty"
    );
    let mut previous: Option<&str> = None;
    for tensor in &checkpoint.tensors {
        ensure!(!tensor.name.is_empty(), "tensor name is empty");
        ensure!(
            previous.is_none_or(|name| name < tensor.name.as_str()),
            "tensor names must be unique and sorted"
        );
        previous = Some(&tensor.name);
        ensure!(tensor.dtype == "F32", "unsupported policy tensor dtype");
        ensure!(
            !tensor.shape.is_empty() && tensor.shape.iter().all(|d| *d > 0),
            "invalid tensor shape"
        );
        ensure!(tensor.finite, "refused: tensor contains a non-finite value");
        ensure!(
            is_lower_digest(&tensor.sha256),
            "tensor digest must be lowercase sha256"
        );
    }
    Ok(())
}

fn is_lower_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// A checked view for callers that need to keep v2 identity visibly separate from v1.
pub fn is_v2_receipt(value: &Value) -> bool {
    value.get("schema").and_then(Value::as_str) == Some(EXPORT_RECEIPT_V2_SCHEMA)
}
