//! Rust-owned, content-addressed family split and train-only view contracts.
//!
//! These contracts decide which rows and labels belong to training. Their
//! identities are deterministic and their serialized form matches the existing
//! `SplitViewManifest.v1` and `BinaryTargetSummary.v1` wire contracts.

use crate::catalog::Task;
use crate::hash::sha256_hex;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::Digest;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// Derive the stable source-family identity used by fused binary-corpus admission.
/// The algorithm intentionally excludes per-dataset seeds, labels, acceptance scores,
/// and held-out measurements so related generated datasets remain in one split family.
fn row_set_hash(row_hashes: &[String]) -> Result<String> {
    let mut sorted = row_hashes.to_vec();
    sorted.sort();
    canonical_hash(&serde_json::json!(sorted))
}

fn canonical_hash(value: &Value) -> Result<String> {
    let mut bytes = Vec::new();
    write_canonical(value, &mut bytes)?;
    Ok(sha256_hex(&bytes))
}

fn write_canonical(value: &Value, output: &mut Vec<u8>) -> Result<()> {
    match value {
        Value::Null => output.extend_from_slice(b"null"),
        Value::Bool(true) => output.extend_from_slice(b"true"),
        Value::Bool(false) => output.extend_from_slice(b"false"),
        Value::Number(number) => output.extend_from_slice(number.to_string().as_bytes()),
        Value::String(string) => {
            output.extend_from_slice(serde_json::to_string(string)?.as_bytes())
        }
        Value::Array(items) => {
            output.push(b'[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                write_canonical(item, output)?;
            }
            output.push(b']');
        }
        Value::Object(map) => {
            output.push(b'{');
            for (index, (key, item)) in map.iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                output.extend_from_slice(serde_json::to_string(key)?.as_bytes());
                output.push(b':');
                write_canonical(item, output)?;
            }
            output.push(b'}');
        }
    }
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

mod binary_summary;
mod family;
mod split_view;

pub use binary_summary::{
    binary_target_summary, build_binary_target_summaries, BinaryTargetSummaryV1,
};
pub use family::{
    build_training_family_partition, derive_binary_source_family, freeze_source_family_split,
    source_family_campaign_split,
};
pub use split_view::{build_split_view_manifest, OuterSplit, SplitViewManifestV1, SplitViewV1};

#[cfg(test)]
mod tests;
