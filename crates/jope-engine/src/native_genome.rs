//! Explicit per-genome evaluation and budgeted continuation. No standalone execution grant.
use crate::{
    corpus::ContextRow,
    manifest::{file_hash, hash, read_verified, resolve},
    native_graph::Artifact,
};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
#[cfg(test)]
use serde_json::json;
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
};

mod execution;
mod validation;
mod verification;

pub(crate) use execution::execute;
#[cfg(test)]
use execution::summarize_outcomes;
pub use execution::target;
#[cfg(test)]
pub(crate) use execution::training_admission;
#[cfg(test)]
use verification::{artifact_closure, prepared_closure};
pub(crate) use verification::{direct_parent, required_inputs, verified_rows, verify_result};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum Arm {
    Model,
    Generic,
    Shuffled,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GenomeEvaluationStage {
    Direct,
    Continuation,
}
pub type Stage = GenomeEvaluationStage;
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub schema: String,
    pub run: String,
    pub generation: u64,
    pub arm: Arm,
    pub matched_slot: u64,
    pub context: ContextRow,
    pub genome: String,
    pub genome_sha256: String,
    pub proposal: Artifact,
    pub parent_checkpoint: Artifact,
    pub sampling_seed: u64,
    pub origin_capture: Artifact,
    pub origin_row: usize,
    pub generic_draw: Option<u64>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenomeTask {
    pub schema: String,
    pub stage: GenomeEvaluationStage,
    pub candidate: Artifact,
    pub output: PathBuf,
    pub vocabulary: Artifact,
    pub admission_policy: Artifact,
    pub qualification: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preparation_overlay: Option<Artifact>,
    pub evaluations: usize,
    pub seconds: u64,
    pub rng_seed: u64,
    /// Original direct completion wrapper, never an unverified score or arbitrary model.
    pub direct_parent: Option<Artifact>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenomeResult {
    pub schema: String,
    pub task: Artifact,
    pub candidate: Artifact,
    pub stage: GenomeEvaluationStage,
    pub search: Artifact,
    pub observations: Artifact,
    pub direct_parent: Option<Artifact>,
    pub artifacts: BTreeMap<PathBuf, String>,
}
/// Reviewed first-window continuation protocol (root preregistration 2026-09-14T19:06:10Z).
pub const CONTINUATION_PROTOCOL: &str = "plain_seeded_continuation_v1";
/// A continuation is ordinary training evidence only with at least this many completed children.
pub const MINIMUM_COMPLETED_CHILDREN: usize = 1;
pub const TRAINING_ADMITTED: &str = "admitted";
pub const REFUSED_NO_COMPLETED_CHILD: &str = "refused_no_completed_child";
/// A different learning target, not a rewritten original evaluator observation.
/// Refused targets keep every original row, cost and best observed value as incomplete evidence.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContinuationTarget {
    pub schema: String,
    pub protocol: String,
    pub candidate: Artifact,
    pub direct: Artifact,
    pub continuation: Artifact,
    pub parent_genome_sha256: String,
    pub winning_original_observation: Option<Artifact>,
    pub winning_row: Option<usize>,
    pub quality_rmse_cut_normalized: Option<f64>,
    pub fit_runtime_seconds: f64,
    pub completed_evaluations: usize,
    pub failed_evaluations: usize,
    /// Continuation children with a null original failure and a measured RMSE; the parent is excluded.
    pub completed_children: usize,
    pub training_admission: String,
    pub evaluation_budget: usize,
    pub search_seconds_budget: u64,
    pub target_kind: String,
}
fn valid_hash(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}
pub(crate) fn checked(root: &Path, a: &Artifact, maximum: u64) -> Result<Vec<u8>> {
    ensure!(valid_hash(&a.sha256), "invalid native genome artifact hash");
    let path = resolve(root, &a.path)?;
    let f = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    ensure!(
        f.metadata()?.is_file() && f.metadata()?.len() <= maximum,
        "bounded native genome regular artifact required"
    );
    let mut b = Vec::new();
    f.take(maximum + 1).read_to_end(&mut b)?;
    ensure!(
        b.len() as u64 <= maximum && hash(&b) == a.sha256,
        "native genome artifact changed"
    );
    Ok(b)
}
pub(crate) fn load<T: serde::de::DeserializeOwned>(root: &Path, a: &Artifact) -> Result<T> {
    Ok(serde_json::from_slice(&checked(
        root,
        a,
        16 * 1024 * 1024,
    )?)?)
}
pub(crate) const DATASET_METADATA_LIMIT: u64 = 256 * 1024 * 1024;
/// Explicit large scientific metadata reader; candidate/task generic bounds stay unchanged.
pub(crate) fn load_corpus_index(root: &Path, a: &Artifact) -> Result<Value> {
    let value: Value = serde_json::from_slice(&checked(root, a, DATASET_METADATA_LIMIT)?)?;
    ensure!(
        matches!(
            value["schema"].as_str(),
            Some("parallel.teaching_corpus.v1" | "parallel.teaching_shards.v1")
        ),
        "unsupported corpus index metadata"
    );
    Ok(value)
}
pub(crate) fn load_dataset_receipt(root: &Path, a: &Artifact) -> Result<Value> {
    let value: Value = serde_json::from_slice(&checked(root, a, DATASET_METADATA_LIMIT)?)?;
    ensure!(
        value["schema"] == "jain.datasets_service_receipt.v1",
        "unsupported dataset receipt metadata"
    );
    Ok(value)
}

#[cfg(test)]
mod tests;
