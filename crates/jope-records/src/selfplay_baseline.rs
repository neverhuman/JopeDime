//! Verify the `I01` executable baseline (`docs/plans/algorithmic-selfplay-20261017/BASELINE.md`)
//! and its file-map successor (`docs/plans/algorithmic-selfplay-20261017/FILE_MAP.i01.json`)
//! against the pinned receiving `main` commit (`22939ca1c67bdcc4beb54c5a4fbdb9e24073db2d`),
//! Cargo workspace manifests, committed JSON schemas, and the immutable v4 unified workplan packet
//! (`PACKET_MANIFEST.v4.json`).
//!
//! Covers the three `I01` acceptance criteria:
//! 1. `reconcile_benchmark_and_product_profile_identities`: reconciles the 8-entry `SearchProfile`
//!    registry (`crates/feat-math/src/kernel/profile.rs:727-738`), the 2-entry
//!    `shipped_kernel2_profiles()` registry (`:748-753`), the product (`jope-pipeline`) and
//!    benchmark (`jope-bench`) profile mappings (`legacy`, `kernel2-baseline`,
//!    `kernel2-experimental`), the planned orthogonal `SearchStrategy` axis (`I17`–`I19`, `D08`),
//!    and the three model/tokenizer lineages (`lineage_b_native_accepted_step60000` 401-token
//!    native PolicyLM vs `clover_rlfw_prefix_trie_282` 282-token CLOVER prefix trie vs
//!    `native_loop_v2_weight_warm_start`).
//! 2. `distinguish_current_ci_historical_runs_and_proposed_protocols`: classifies every finding as
//!    `confirmed`, `rejected`, or `unknown` with explicit evidence domain (`current_ci`,
//!    `historical_run`, or `proposed_protocol`) and per-finding 64-hex `evidence_sha256`,
//!    rejecting missing 40-hex commit / repo-relative witness bindings on `confirmed` findings,
//!    coerced `unknown` -> `confirmed` transitions across any category, or historical/skipped/proposed
//!    runs promoted to live qualification.
//! 3. `record_all_skipped_and_unavailable_runtime_combinations`: inventories skipped/unavailable
//!    runtime combinations (`RT01`–`RT07`) dynamically against the workspace manifests,
//!    `crates/jope-engine/src/` GPU guard call sites, `docs/parity/test-inventory.json` (`72`
//!    ignored tests), and `deploy/ci/coverage.json` (`22` non-default features), rejecting
//!    0-collected test filters or unverified GPU grant routes treated as passed or active.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::path::{Component, Path};
use std::process::{Command, Stdio};

use sha2::{Digest, Sha256};

use crate::pyjson::{self, Dict, Value};
use crate::{require, Error, Result};

/// Relative directory of the unified algorithmic self-play workplan packet.
pub const PLAN_DIR: &str = "docs/plans/algorithmic-selfplay-20261017";

/// Relative path to the `I01` executable baseline document.
pub const BASELINE_MD_PATH: &str = "docs/plans/algorithmic-selfplay-20261017/BASELINE.md";

/// Relative path to the `I01` machine-readable file map successor.
pub const FILE_MAP_I01_PATH: &str = "docs/plans/algorithmic-selfplay-20261017/FILE_MAP.i01.json";

/// Relative path to the immutable v4 packet file map.
pub const FILE_MAP_V1_PATH: &str = "docs/plans/algorithmic-selfplay-20261017/FILE_MAP.json";

/// Relative path to the immutable v4 packet manifest.
pub const PACKET_MANIFEST_V4_PATH: &str =
    "docs/plans/algorithmic-selfplay-20261017/PACKET_MANIFEST.v4.json";

/// Expected SHA-256 of `PACKET_MANIFEST.v4.json`.
pub const EXPECTED_PACKET_MANIFEST_V4_SHA256: &str =
    "921643bc3ae0b7d75bbab13434f4d8211bc3c8638154c45ac00fc7df184750d5";

/// Expected SHA-256 of immutable `FILE_MAP.json`.
pub const EXPECTED_FILE_MAP_V1_SHA256: &str =
    "7c2eca800309a520aacf38c6fdf0af0929d9a8c497efcc08d03d9d98954cb2c2";

/// Expected receiving `main` commit SHA after PR #356 merge.
pub const RECEIVING_MAIN_HEAD_SHA: &str = "22939ca1c67bdcc4beb54c5a4fbdb9e24073db2d";

/// Expected receiving `main` tree SHA after PR #356 merge.
pub const RECEIVING_MAIN_TREE_SHA: &str = "885189fe077e4a23acd7333370c1e95b57193bc4";

/// Expected receiving `main` parent SHA after PR #356 merge.
pub const RECEIVING_MAIN_PARENT_SHA: &str = "0e063f4ce110dd142f18f8482867c2d3d237598c";

const HEX_SHA1_LEN: usize = 40;
const HEX_SHA256_LEN: usize = 64;
const V4_PAYLOAD_COUNT: usize = 39;
const V4_TOTAL_FILE_COUNT: usize = 40;
const EXPECTED_EXISTING_FILE_COUNT: usize = 134;
const EXPECTED_WRITE_PATH_COUNT: usize = 133;
const EXPECTED_TASK_COUNT: usize = 66;
const EXPECTED_MAPPING_COUNT: usize = 3;
const EXPECTED_LINEAGE_COUNT: usize = 3;
const MIN_HISTORICAL_PIN_COUNT: usize = 4;
const MIN_CAT_FILE_HEADER_PARTS: usize = 3;

/// Summary returned by [`verify_selfplay_baseline`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelfplayBaselineSummary {
    pub schema: String,
    pub task_id: String,
    pub workspace_crate_count: usize,
    pub neural_gated_test_count: usize,
    pub partially_neural_gated_test_count: usize,
    pub schema_file_count: usize,
    pub symbol_count: usize,
    pub held_gpu_route_count: usize,
    pub confirmed_finding_count: usize,
    pub rejected_finding_count: usize,
    pub unknown_finding_count: usize,
    pub runtime_combination_count: usize,
    pub git_object_store_verified: bool,
    pub git_object_store_status: String,
}
fn has_commit_object(repo: &Path, commit_sha: &str) -> bool {
    Command::new("git")
        .arg("-C")
        .arg(repo)
        .arg("cat-file")
        .arg("-e")
        .arg(format!("{commit_sha}^{{commit}}"))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn read_utf8(path: &Path) -> Result<String> {
    let bytes = fs::read(path).map_err(|e| Error::io(path, e))?;
    String::from_utf8(bytes).map_err(|e| Error::parse(path, e.to_string()))
}

fn read_json(path: &Path) -> Result<Value> {
    let text = read_utf8(path)?;
    pyjson::parse_str(&text).map_err(|e| Error::parse(path, e))
}

fn dict_of<'a>(val: &'a Value, ctx: &str) -> Result<&'a Dict> {
    val.as_dict()
        .ok_or_else(|| Error::Invalid(format!("{ctx}: expected JSON object")))
}

fn list_of<'a>(val: &'a Value, ctx: &str) -> Result<&'a Vec<Value>> {
    val.as_list()
        .ok_or_else(|| Error::Invalid(format!("{ctx}: expected JSON array")))
}

fn req_str<'a>(dict: &'a Dict, key: &str, ctx: &str) -> Result<&'a str> {
    dict.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Invalid(format!("{ctx}: missing or non-string `{key}`")))
}

fn req_u64(dict: &Dict, key: &str, ctx: &str) -> Result<u64> {
    let v = dict
        .get(key)
        .and_then(Value::as_int)
        .ok_or_else(|| Error::Invalid(format!("{ctx}: missing or non-integer `{key}`")))?;
    u64::try_from(v).map_err(|_| Error::Invalid(format!("{ctx}: negative integer `{key}` ({v})")))
}

fn req_bool(dict: &Dict, key: &str, ctx: &str) -> Result<bool> {
    match dict.get(key) {
        Some(Value::Bool(b)) => Ok(*b),
        _ => Err(Error::Invalid(format!(
            "{ctx}: missing or non-boolean `{key}`"
        ))),
    }
}

fn req_dict<'a>(dict: &'a Dict, key: &str, ctx: &str) -> Result<&'a Dict> {
    dict.get(key)
        .and_then(Value::as_dict)
        .ok_or_else(|| Error::Invalid(format!("{ctx}: missing or non-object `{key}`")))
}

fn req_list<'a>(dict: &'a Dict, key: &str, ctx: &str) -> Result<&'a Vec<Value>> {
    dict.get(key)
        .and_then(Value::as_list)
        .ok_or_else(|| Error::Invalid(format!("{ctx}: missing or non-array `{key}`")))
}

mod findings;
mod packet;
mod profile;
mod runtime;
mod snapshot;
mod symbols;
mod verify;
mod workspace;

pub use findings::verify_finding_classifications;
pub use packet::verify_immutable_packet_v4;
pub use profile::verify_profile_identity_reconciliation;
pub use runtime::verify_runtime_combinations;
pub use snapshot::{
    is_lower_hex_sha40, is_lower_hex_sha64, validate_repo_relative_path, BaselineSnapshot,
};
pub use symbols::verify_symbol_inventory;
pub use verify::verify_selfplay_baseline;
pub use workspace::verify_workspace_and_schemas;

use findings::verify_finding_classifications_with_snapshot;
use profile::verify_profile_identity_reconciliation_with_snapshot;
use runtime::verify_runtime_combinations_with_snapshot;
use symbols::verify_symbol_inventory_with_snapshot;
use workspace::verify_workspace_and_schemas_with_snapshot;

#[cfg(test)]
mod tests;
