//! A bounded, resumable sequence of qualification cycles. Original final tests stay sealed.
#[cfg(any(feature = "neural", test))]
use crate::corpus;
#[cfg(feature = "neural")]
use crate::{manifest::*, telemetry::now};
#[cfg(feature = "neural")]
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
#[cfg(feature = "neural")]
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

// Process-scoped: rank children are supervised and reaped by the controller;
// unrelated GP/encoding services keep their own independent resource budgets.
static ACTIVE_DEADLINE: AtomicU64 = AtomicU64::new(0);

pub(crate) fn check_deadline() -> anyhow::Result<()> {
    let deadline = ACTIVE_DEADLINE.load(Ordering::Relaxed);
    anyhow::ensure!(
        deadline == 0 || crate::telemetry::now() < deadline,
        "fixed campaign deadline reached"
    );
    Ok(())
}

#[cfg(any(feature = "neural", test))]
pub(crate) struct DeadlineGuard(u64);
#[cfg(any(feature = "neural", test))]
impl DeadlineGuard {
    pub(crate) fn new(deadline: u64) -> Self {
        Self(ACTIVE_DEADLINE.swap(deadline, Ordering::Relaxed))
    }
}
#[cfg(any(feature = "neural", test))]
impl Drop for DeadlineGuard {
    fn drop(&mut self) {
        ACTIVE_DEADLINE.store(self.0, Ordering::Relaxed);
    }
}
#[cfg(any(feature = "neural", test))]
use std::collections::BTreeSet;
#[cfg(feature = "neural")]
use std::{
    fs::{self, OpenOptions},
    path::Path,
};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CutStrategy {
    #[default]
    Rotation,
    FixedAnchors,
}
#[cfg(any(feature = "neural", test))]
impl CutStrategy {
    fn start(&self, starts: &[usize], sequence: usize) -> usize {
        match self {
            Self::Rotation => starts[sequence % starts.len()],
            Self::FixedAnchors => starts[0],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub schema: String,
    pub contexts: PathBuf,
    pub initial_corpus: PathBuf,
    pub output: PathBuf,
    pub observations: Vec<PathBuf>,
    pub hours: u64,
    pub steps: usize,
    pub cuts: usize,
    #[serde(default)]
    pub cut_strategy: CutStrategy,
    pub qualification: bool,
    pub required_training_families: usize,
    pub required_validation_families: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Schedule {
    pub spec_sha256: String,
    pub contexts_sha256: String,
    pub initial_corpus_sha256: String,
    pub vocabulary_sha256: String,
    pub executable_sha256: String,
    pub gp_executable_sha256: String,
    pub predictor_executable_sha256: String,
    pub started_at: u64,
    pub stop_admission_at: u64,
    pub deadline: u64,
}
impl Schedule {
    pub fn may_start(&self, timestamp: u64) -> bool {
        timestamp < self.stop_admission_at
    }
    pub fn same_inputs(&self, proposed: &Self) -> bool {
        let mut identity = self.clone();
        identity.started_at = proposed.started_at;
        identity.stop_admission_at = proposed.stop_admission_at;
        identity.deadline = proposed.deadline;
        identity == *proposed
    }
}
#[cfg(any(feature = "neural", test))]
fn cut_starts(index: &corpus::Index, cuts: usize) -> Vec<usize> {
    index
        .contexts
        .iter()
        .filter(|c| c.partition != "meta_test")
        .map(|c| c.cut / cuts * cuts)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}
#[cfg(feature = "neural")]
fn path(root: &Path, p: &Path) -> Result<PathBuf> {
    resolve(root, p)
}
#[cfg(feature = "neural")]
fn event(root: &Path, message: &str) -> Result<()> {
    use std::io::Write;
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(root.join("runs/events.log"))?;
    writeln!(f, "{} · {}", now(), message)?;
    f.sync_data()?;
    Ok(())
}
#[cfg(feature = "neural")]
fn family_sets(index: &corpus::Index) -> Result<(BTreeSet<String>, BTreeSet<String>)> {
    ensure!(
        index
            .groups
            .iter()
            .all(|g| matches!(g.partition.as_str(), "meta_train" | "meta_validation")),
        "non-training/validation supervision in campaign corpus"
    );
    Ok((
        index
            .groups
            .iter()
            .filter(|g| g.partition == "meta_train")
            .map(|g| g.family.clone())
            .collect(),
        index
            .groups
            .iter()
            .filter(|g| g.partition == "meta_validation")
            .map(|g| g.family.clone())
            .collect(),
    ))
}

#[cfg(feature = "neural")]
fn evaluator_hashes(root: &Path) -> Result<(String, String)> {
    let service: Value = read_json(&root.join("runs/gp-service.json"))?;
    let gp = Path::new(
        service["executable"]
            .as_str()
            .context("GP service executable absent")?,
    );
    let gp = path(
        root,
        gp.strip_prefix(root)
            .context("GP executable outside workspace")?,
    )?;
    let gp_sha = file_hash(&gp)?;
    ensure!(
        service["executable_sha256"] == gp_sha,
        "GP service executable receipt changed"
    );
    let predictor = PathBuf::from(
        std::env::var_os("JOPE_GP_EXECUTABLE").context("pinned predictor executable absent")?,
    );
    let predictor = path(
        root,
        predictor
            .strip_prefix(root)
            .context("predictor executable outside workspace")?,
    )?;
    Ok((gp_sha, file_hash(&predictor)?))
}

#[cfg(feature = "neural")]
mod run;
#[cfg(feature = "neural")]
pub use run::run;
#[cfg(feature = "neural")]
#[cfg(test)]
use run::run_admitted;

#[cfg(test)]
mod tests;
