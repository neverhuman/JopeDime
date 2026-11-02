//! The Elo reporter's data directory (`SOTARENA_REPORTER_ROOT`): `config.json`,
//! `data/<task>/<hash>/dataset.json`, `data/<task>/<hash>/results/*.json`,
//! `data/<task>/analysis-blacklist.json`, `data/global/results/*.json`.
//!
//! Ported from the SOTArena reporter's `src/main.rs` (config validation, dataset catalogue, result
//! records, score documents, analysis blacklists). The verified-ASPFM archive loader
//! (`submission/ASPFM/release.json`, an age-encrypted tarball replayed by `aspfm.rs`) is **not**
//! ported: those results enter the benchmark clone as ordinary `scores.csv` rows instead.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::fs;
use std::path::{Component, Path, PathBuf};

use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use serde_json::Value;

use crate::elo::{metric_better, Direction, EloConfig, TaskResults};
use crate::io::read_json;

/// `config.json`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// `sotarena.config`.
    pub schema: String,
    /// Tasks in report order.
    pub tasks: Vec<TaskConfig>,
    /// Fit constants.
    pub elo: EloConfig,
    /// Method families (prefix patterns).
    pub families: Vec<FamilyConfig>,
}

/// One task of the reporter.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskConfig {
    /// `binary`, `regression`, `multiclass`, `timeseries`.
    pub id: String,
    /// `roc_auc`, `rmse`, `macro_f1`, `mase`.
    pub metric: String,
    /// Which way is better.
    pub direction: Direction,
    /// Weight in the global family table (weights sum to 1).
    pub weight: f64,
    /// Relative path of the task's analysis blacklist.
    pub analysis_blacklist: PathBuf,
}

/// One family: an id, a display name and exact-or-prefix method patterns.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FamilyConfig {
    /// Family id (`aspfm`, `tabicl`, …).
    pub id: String,
    /// Table label.
    pub display_name: String,
    /// `exact` or `prefix*` patterns; no two families may overlap.
    pub patterns: Vec<String>,
}

/// `data/<task>/<hash>/dataset.json`.
#[derive(Clone, Debug, Deserialize)]
pub struct Dataset {
    /// `sotarena.dataset`.
    pub schema: String,
    /// 16 or 64 lower-case hex.
    pub dataset_hash: String,
    /// Human id (often equal to the hash).
    pub dataset_id: String,
    /// Task id.
    pub task: String,
    /// Metric id.
    pub metric: String,
    /// Training split.
    pub train: DataFile,
    /// Test split.
    pub test: DataFile,
}

/// A downloadable (or metadata-only) split.
#[derive(Clone, Debug, Deserialize)]
pub struct DataFile {
    /// HTTPS CSV URL; `None` for metadata-only datasets.
    pub url: Option<String>,
    /// SHA-256 of the file.
    pub sha256: String,
    /// Exact byte count.
    pub bytes: u64,
    /// Row count.
    pub rows: u64,
}

/// Task → hash → dataset.
pub type DatasetCatalog = BTreeMap<String, BTreeMap<String, Dataset>>;
/// Task → method → dataset → best score.
pub type ResultCatalog = BTreeMap<String, TaskResults>;
/// Task → blacklisted dataset hashes.
pub type AnalysisBlacklists = BTreeMap<String, BTreeSet<String>>;

/// Everything `load_results` produced.
#[derive(Debug)]
pub struct LoadedResults {
    /// Best score per (task, method, dataset) over every record.
    pub catalog: ResultCatalog,
    /// Same, restricted to `benchmark_valid` records.
    pub valid_catalog: ResultCatalog,
    /// Result JSON files per task.
    pub result_jsons: BTreeMap<String, usize>,
    /// Result JSON files overall.
    pub result_jsons_total: usize,
}

/// `sotarena.result(.v1)` record.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResultRecord {
    /// Schema id.
    pub schema: String,
    /// Task id.
    pub task: String,
    /// Metric id.
    pub metric: String,
    /// Method id.
    pub method_id: String,
    /// Display name.
    pub display_name: String,
    /// Provenance scope label.
    pub claim_scope: String,
    /// Dataset id.
    pub dataset_id: String,
    /// Dataset hash.
    pub dataset_hash: String,
    /// Metric value.
    pub value: f64,
    /// SHA-256 of the test split.
    pub test_sha256: String,
    /// SHA-256 of the predictions.
    pub predictions_sha256: Option<String>,
    /// Run id.
    pub run_id: String,
    /// Explicit benchmark validity.
    #[serde(default)]
    pub benchmark_valid: bool,
}

/// `sotarena.scores` document (many methods for one dataset).
#[derive(Debug, Deserialize)]
pub struct ScoreDocument {
    /// Schema id.
    pub schema: String,
    /// Task id.
    pub task: String,
    /// Dataset hash.
    pub dataset_hash: String,
    /// Metric id.
    pub metric: String,
    /// Rows.
    pub rows: Vec<ScoreDocumentRow>,
}

/// One row of a [`ScoreDocument`].
#[derive(Debug, Deserialize)]
pub struct ScoreDocumentRow {
    /// Method id.
    pub method_id: String,
    /// Value, when scored.
    pub value: Option<f64>,
    /// `ok` counts; everything else is skipped.
    pub status: String,
    /// Explicit benchmark validity.
    pub benchmark_valid: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AnalysisBlacklist {
    schema: String,
    task: String,
    datasets: Vec<String>,
}

/// Task/family/method ids: non-empty `[A-Za-z0-9_-]+`.
pub fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
}

/// 16 or 64 lower-case hex characters.
pub fn valid_hash(value: &str) -> bool {
    (value.len() == 16 || value.len() == 64)
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// `exact` or `prefix*` (one trailing star, at least one character before it).
pub fn validate_pattern(pattern: &str) -> Result<()> {
    let stars = pattern.bytes().filter(|byte| *byte == b'*').count();
    ensure!(
        !pattern.is_empty()
            && (stars == 0 || (stars == 1 && pattern.ends_with('*') && pattern.len() > 1)),
        "invalid method pattern {pattern}"
    );
    Ok(())
}

fn pattern_prefix(pattern: &str) -> Option<&str> {
    pattern.strip_suffix('*')
}

/// Whether two patterns can match the same method id.
pub fn patterns_overlap(left: &str, right: &str) -> bool {
    match (pattern_prefix(left), pattern_prefix(right)) {
        (None, None) => left == right,
        (Some(prefix), None) => right.starts_with(prefix),
        (None, Some(prefix)) => left.starts_with(prefix),
        (Some(a), Some(b)) => a.starts_with(b) || b.starts_with(a),
    }
}

/// Whether `pattern` matches `method`.
pub fn pattern_matches(pattern: &str, method: &str) -> bool {
    pattern_prefix(pattern).map_or(method == pattern, |prefix| method.starts_with(prefix))
}

/// The task with id `task`.
pub fn task_config<'a>(config: &'a Config, task: &str) -> Result<&'a TaskConfig> {
    config
        .tasks
        .iter()
        .find(|candidate| candidate.id == task)
        .with_context(|| format!("unknown task {task}"))
}

fn valid_https_csv_url(url: &str) -> bool {
    let Some(rest) = url.strip_prefix("https://") else {
        return false;
    };
    let Some((authority, path)) = rest.split_once('/') else {
        return false;
    };
    !authority.is_empty()
        && !authority.bytes().any(|byte| byte.is_ascii_whitespace())
        && path.ends_with(".csv")
        && !path.contains(['?', '#'])
}

fn check_data_file(file: &DataFile) -> Result<()> {
    if let Some(url) = &file.url {
        ensure!(valid_https_csv_url(url), "invalid CSV URL {url}");
        ensure!(file.bytes > 0, "invalid downloadable file size");
    }
    ensure!(
        file.sha256.len() == 64 && valid_hash(&file.sha256),
        "invalid data SHA-256"
    );
    ensure!(file.rows > 0, "invalid data row count");
    Ok(())
}

fn check_dataset_rows(train: &DataFile, test: &DataFile) -> Result<()> {
    let rows = train
        .rows
        .checked_add(test.rows)
        .context("dataset row count overflow")?;
    ensure!(
        rows >= 300,
        "dataset has {rows} combined train/test rows; minimum is 300"
    );
    Ok(())
}

/// Read and validate `<root>/config.json`.
pub fn load_config(root: &Path) -> Result<Config> {
    let config: Config = read_json(&root.join("config.json"))?;
    validate_config(&config)?;
    Ok(config)
}

/// The reporter's `validate_config`.
pub fn validate_config(config: &Config) -> Result<()> {
    ensure!(config.schema == "sotarena.config", "invalid config schema");
    ensure!(!config.tasks.is_empty(), "config has no tasks");
    ensure!(!config.families.is_empty(), "config has no families");
    ensure!(
        config.elo.initial_rating.is_finite(),
        "invalid initial rating"
    );
    ensure!(
        config.elo.k_factor.is_finite() && config.elo.k_factor > 0.0,
        "invalid K factor"
    );
    ensure!(
        config.elo.tie_tolerance.is_finite() && config.elo.tie_tolerance >= 0.0,
        "invalid tie tolerance"
    );
    ensure!(config.elo.iterations > 0, "invalid Elo iteration count");
    ensure!(
        config.elo.regularization.is_finite() && config.elo.regularization > 0.0,
        "invalid Elo regularization"
    );
    ensure!(
        config.elo.convergence_tolerance.is_finite() && config.elo.convergence_tolerance > 0.0,
        "invalid Elo convergence tolerance"
    );
    let mut task_ids = BTreeSet::new();
    let mut blacklist_paths = BTreeSet::new();
    let mut weights = 0.0;
    for task in &config.tasks {
        ensure!(valid_id(&task.id), "invalid task id {}", task.id);
        ensure!(task_ids.insert(&task.id), "duplicate task {}", task.id);
        ensure!(
            !task.metric.trim().is_empty(),
            "empty metric for {}",
            task.id
        );
        ensure!(
            task.weight.is_finite() && task.weight >= 0.0,
            "invalid weight for {}",
            task.id
        );
        ensure!(
            !task.analysis_blacklist.as_os_str().is_empty()
                && task.analysis_blacklist.is_relative()
                && task
                    .analysis_blacklist
                    .components()
                    .all(|component| matches!(component, Component::Normal(_)))
                && task.analysis_blacklist.extension().and_then(OsStr::to_str) == Some("json"),
            "invalid analysis blacklist reference for {}",
            task.id
        );
        ensure!(
            blacklist_paths.insert(&task.analysis_blacklist),
            "duplicate analysis blacklist reference {}",
            task.analysis_blacklist.display()
        );
        weights += task.weight;
    }
    ensure!(
        (weights - 1.0).abs() <= 1e-12,
        "task weights must sum to 1.0"
    );

    let mut family_ids = BTreeSet::new();
    let mut patterns = Vec::new();
    for family in &config.families {
        ensure!(valid_id(&family.id), "invalid family id {}", family.id);
        ensure!(
            family_ids.insert(&family.id),
            "duplicate family {}",
            family.id
        );
        ensure!(
            !family.display_name.trim().is_empty(),
            "empty family display name"
        );
        ensure!(
            !family.patterns.is_empty(),
            "family {} has no patterns",
            family.id
        );
        for pattern in &family.patterns {
            validate_pattern(pattern)?;
            patterns.push((family.id.as_str(), pattern.as_str()));
        }
    }
    for left in 0..patterns.len() {
        for right in (left + 1)..patterns.len() {
            ensure!(
                !patterns_overlap(patterns[left].1, patterns[right].1),
                "overlapping patterns {}:{} and {}:{}",
                patterns[left].0,
                patterns[left].1,
                patterns[right].0,
                patterns[right].1
            );
        }
    }
    Ok(())
}

mod results;
pub use results::*;

#[cfg(test)]
pub(crate) mod fixtures;
#[cfg(test)]
mod tests;
