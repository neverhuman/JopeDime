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

/// Read every `data/<task>/<hash>/dataset.json`.
pub fn load_datasets(root: &Path, config: &Config) -> Result<DatasetCatalog> {
    let mut catalog: DatasetCatalog = config
        .tasks
        .iter()
        .map(|task| (task.id.clone(), BTreeMap::new()))
        .collect();
    for task in &config.tasks {
        let directory = root.join("data").join(&task.id);
        let entries = fs::read_dir(&directory)
            .with_context(|| format!("cannot read task directory {}", directory.display()))?;
        for entry in entries {
            let entry = entry?;
            if !entry.file_type()?.is_dir() || entry.file_name() == OsStr::new("global") {
                continue;
            }
            let path = entry.path().join("dataset.json");
            let dataset: Dataset = read_json(&path)?;
            ensure!(
                dataset.schema == "sotarena.dataset",
                "invalid dataset schema"
            );
            ensure!(
                dataset.task == task.id && dataset.metric == task.metric,
                "dataset task/metric mismatch"
            );
            ensure!(valid_hash(&dataset.dataset_hash), "invalid dataset hash");
            ensure!(
                entry.file_name() == OsStr::new(&dataset.dataset_hash),
                "dataset directory/hash mismatch"
            );
            check_data_file(&dataset.train)?;
            check_data_file(&dataset.test)?;
            ensure!(
                dataset.train.url.is_some() == dataset.test.url.is_some(),
                "dataset train/test download availability mismatch"
            );
            check_dataset_rows(&dataset.train, &dataset.test)?;
            ensure!(
                catalog
                    .get_mut(&task.id)
                    .expect("task inserted above")
                    .insert(dataset.dataset_hash.clone(), dataset)
                    .is_none(),
                "duplicate dataset hash"
            );
        }
        ensure!(
            !catalog[&task.id].is_empty(),
            "task {} has no datasets",
            task.id
        );
    }
    Ok(catalog)
}

/// Read and validate every task's analysis blacklist.
pub fn load_analysis_blacklists(
    root: &Path,
    config: &Config,
    datasets: &DatasetCatalog,
) -> Result<AnalysisBlacklists> {
    let mut blacklists = BTreeMap::new();
    for task in &config.tasks {
        let path = root.join(&task.analysis_blacklist);
        let blacklist: AnalysisBlacklist = read_json(&path)?;
        ensure!(
            blacklist.schema == "sotarena.analysis-blacklist",
            "invalid analysis blacklist schema for {}",
            task.id
        );
        ensure!(
            blacklist.task == task.id,
            "analysis blacklist task mismatch for {}",
            task.id
        );
        ensure!(
            blacklist.datasets.windows(2).all(|pair| pair[0] < pair[1]),
            "analysis blacklist datasets for {} must be sorted and unique",
            task.id
        );
        for hash in &blacklist.datasets {
            ensure!(
                datasets[&task.id].contains_key(hash),
                "unknown dataset {hash} in analysis blacklist for {}",
                task.id
            );
        }
        blacklists.insert(task.id.clone(), blacklist.datasets.into_iter().collect());
    }
    Ok(blacklists)
}

fn dataset_hash<'a>(
    datasets: &'a BTreeMap<String, Dataset>,
    reference: &'a str,
) -> Result<&'a str> {
    if datasets.contains_key(reference) {
        return Ok(reference);
    }
    datasets
        .iter()
        .find(|(_, dataset)| dataset.dataset_id == reference)
        .map(|(hash, _)| hash.as_str())
        .with_context(|| format!("unknown dataset {reference}"))
}

fn json_files(directory: &Path) -> Result<Vec<PathBuf>> {
    if !directory.is_dir() {
        return Ok(Vec::new());
    }
    Ok(fs::read_dir(directory)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_file() && path.extension().and_then(OsStr::to_str) == Some("json"))
        .collect())
}

fn result_paths(
    root: &Path,
    task: &str,
    datasets: &BTreeMap<String, Dataset>,
) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for hash in datasets.keys() {
        paths.extend(json_files(
            &root.join("data").join(task).join(hash).join("results"),
        )?);
    }
    paths.sort();
    Ok(paths)
}

fn global_result_paths(root: &Path, config: &Config) -> Result<Vec<PathBuf>> {
    let mut paths = json_files(&root.join("data/global/results"))?;
    for task in &config.tasks {
        paths.extend(json_files(
            &root.join("data").join(&task.id).join("global/results"),
        )?);
    }
    paths.sort();
    Ok(paths)
}

/// Non-empty, no control characters.
pub fn valid_method_id(method: &str) -> bool {
    !method.trim().is_empty() && !method.chars().any(char::is_control)
}

/// Finite and inside the metric's range.
pub fn valid_metric_value(metric: &str, value: f64) -> bool {
    value.is_finite()
        && match metric {
            "roc_auc" | "macro_f1" => (0.0..=1.0).contains(&value),
            "rmse" | "mase" => value >= 0.0,
            _ => false,
        }
}

fn empty_result_catalog(config: &Config) -> ResultCatalog {
    config
        .tasks
        .iter()
        .map(|task| (task.id.clone(), BTreeMap::new()))
        .collect()
}

/// Keep the best score per (task, method, dataset).
pub fn insert_best_score(
    catalog: &mut ResultCatalog,
    task: &TaskConfig,
    method: String,
    dataset: String,
    value: f64,
) {
    let scores = catalog
        .get_mut(&task.id)
        .expect("validated task")
        .entry(method)
        .or_default();
    match scores.get_mut(&dataset) {
        Some(current) if metric_better(value, *current, task.direction) => *current = value,
        Some(_) => {}
        None => {
            scores.insert(dataset, value);
        }
    }
}

fn add_score_document(
    config: &Config,
    datasets: &DatasetCatalog,
    catalog: &mut ResultCatalog,
    valid_catalog: &mut ResultCatalog,
    document: ScoreDocument,
) -> Result<()> {
    ensure!(document.schema == "sotarena.scores", "invalid score schema");
    let task = task_config(config, &document.task)?;
    ensure!(document.metric == task.metric, "score metric mismatch");
    let task_datasets = &datasets[&task.id];
    let hash = dataset_hash(task_datasets, &document.dataset_hash)?;
    ensure!(hash == document.dataset_hash, "score dataset mismatch");
    for row in document.rows {
        let Some(value) = row.value else { continue };
        if row.status != "ok" {
            continue;
        }
        ensure!(valid_method_id(&row.method_id), "invalid method id");
        ensure!(
            valid_metric_value(&task.metric, value),
            "invalid {} score for {}",
            task.metric,
            row.method_id
        );
        insert_best_score(catalog, task, row.method_id.clone(), hash.to_owned(), value);
        if row.benchmark_valid == Some(true) {
            insert_best_score(valid_catalog, task, row.method_id, hash.to_owned(), value);
        }
    }
    Ok(())
}

fn add_result_record(
    config: &Config,
    datasets: &DatasetCatalog,
    catalog: &mut ResultCatalog,
    valid_catalog: &mut ResultCatalog,
    record: ResultRecord,
) -> Result<()> {
    ensure!(
        matches!(
            record.schema.as_str(),
            "sotarena.result" | "sotarena.result.v1"
        ),
        "invalid result schema"
    );
    let task = task_config(config, &record.task)?;
    ensure!(record.metric == task.metric, "result metric mismatch");
    let task_datasets = &datasets[&task.id];
    let hash = dataset_hash(task_datasets, &record.dataset_hash)?;
    ensure!(hash == record.dataset_hash, "result dataset mismatch");
    let dataset = &task_datasets[hash];
    ensure!(
        record.dataset_id == dataset.dataset_id || record.dataset_id == dataset.dataset_hash,
        "result dataset id mismatch"
    );
    ensure!(
        record.test_sha256 == dataset.test.sha256,
        "result test split mismatch for {}",
        record.method_id
    );
    ensure!(valid_method_id(&record.method_id), "invalid method id");
    ensure!(
        !record.display_name.trim().is_empty(),
        "empty display name for {}",
        record.method_id
    );
    ensure!(
        valid_metric_value(&task.metric, record.value),
        "invalid {} score for {}",
        task.metric,
        record.method_id
    );
    ensure!(
        !record.run_id.trim().is_empty() && !record.claim_scope.trim().is_empty(),
        "invalid result provenance"
    );
    insert_best_score(
        catalog,
        task,
        record.method_id.clone(),
        hash.to_owned(),
        record.value,
    );
    if record.benchmark_valid {
        insert_best_score(
            valid_catalog,
            task,
            record.method_id,
            hash.to_owned(),
            record.value,
        );
    }
    Ok(())
}

fn collect_dataset_references(value: &Value, references: &mut BTreeSet<String>) -> Result<()> {
    match value {
        Value::Object(object) => {
            for (key, child) in object {
                if key == "dataset_hash" {
                    let hash = child
                        .as_str()
                        .context("dataset_hash reference must be a string")?;
                    references.insert(hash.to_owned());
                }
                collect_dataset_references(child, references)?;
            }
        }
        Value::Array(values) => {
            for child in values {
                collect_dataset_references(child, references)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn validate_dataset_references(
    path: &Path,
    value: &Value,
    datasets: &DatasetCatalog,
) -> Result<()> {
    let mut references = BTreeSet::new();
    collect_dataset_references(value, &mut references)
        .with_context(|| format!("invalid dataset reference in {}", path.display()))?;
    if references.is_empty() {
        return Ok(());
    }
    let task = value
        .get("task")
        .or_else(|| value.get("task_type"))
        .and_then(Value::as_str);
    if let Some(task) = task {
        let task_datasets = datasets
            .get(task)
            .with_context(|| format!("unknown task {task} in {}", path.display()))?;
        for hash in references {
            ensure!(
                task_datasets.contains_key(&hash),
                "unknown dataset {hash} referenced by {}",
                path.display()
            );
        }
    } else {
        for hash in references {
            let matches = datasets
                .values()
                .filter(|task_datasets| task_datasets.contains_key(&hash))
                .count();
            ensure!(
                matches == 1,
                "dataset {hash} referenced by {} resolves to {matches} tasks",
                path.display()
            );
        }
    }
    Ok(())
}

/// Read every result record and score document under `data/`.
pub fn load_results(
    root: &Path,
    config: &Config,
    datasets: &DatasetCatalog,
) -> Result<LoadedResults> {
    let mut catalog = empty_result_catalog(config);
    let mut valid_catalog = empty_result_catalog(config);
    let mut result_jsons: BTreeMap<_, _> = config
        .tasks
        .iter()
        .map(|task| (task.id.clone(), 0))
        .collect();
    let mut paths = Vec::new();
    for task in &config.tasks {
        let task_paths = result_paths(root, &task.id, &datasets[&task.id])?;
        result_jsons.insert(task.id.clone(), task_paths.len());
        paths.extend(task_paths);
    }
    paths.extend(global_result_paths(root, config)?);
    paths.sort();
    paths.dedup();
    let result_jsons_total = paths.len();
    for path in paths {
        let value: Value = read_json(&path)?;
        validate_dataset_references(&path, &value, datasets)?;
        match value["schema"].as_str() {
            Some("sotarena.scores") => {
                let document: ScoreDocument = serde_json::from_value(value)?;
                add_score_document(config, datasets, &mut catalog, &mut valid_catalog, document)
                    .with_context(|| path.display().to_string())?;
            }
            Some("sotarena.result" | "sotarena.result.v1") => {
                let record: ResultRecord = serde_json::from_value(value)?;
                add_result_record(config, datasets, &mut catalog, &mut valid_catalog, record)
                    .with_context(|| path.display().to_string())?;
            }
            _ => {}
        }
    }
    Ok(LoadedResults {
        catalog,
        valid_catalog,
        result_jsons,
        result_jsons_total,
    })
}

#[cfg(test)]
pub(crate) mod fixtures {
    use super::*;
    use crate::elo::EloConfig;

    /// The reporter's `test_config`.
    pub fn test_config(direction: Direction) -> Config {
        Config {
            schema: "sotarena.config".to_owned(),
            tasks: vec![TaskConfig {
                id: "binary".to_owned(),
                metric: "roc_auc".to_owned(),
                direction,
                weight: 1.0,
                analysis_blacklist: "data/binary/analysis-blacklist.json".into(),
            }],
            elo: EloConfig::default(),
            families: vec![FamilyConfig {
                id: "family".to_owned(),
                display_name: "Family".to_owned(),
                patterns: vec!["family-*".to_owned()],
            }],
        }
    }

    pub fn data_file(url: Option<&str>) -> DataFile {
        DataFile {
            url: url.map(str::to_owned),
            sha256: "a".repeat(64),
            bytes: 1,
            rows: 200,
        }
    }

    /// Write a reporter data dir with one binary dataset and the given result records.
    pub fn write_reporter_root(root: &Path, records: &[(&str, &str, f64, bool)]) {
        fs::create_dir_all(root.join("data/binary/0123456789abcdef/results")).unwrap();
        fs::write(
            root.join("config.json"),
            serde_json::json!({
                "schema": "sotarena.config",
                "tasks": [{"id": "binary", "metric": "roc_auc", "direction": "maximize", "weight": 1.0, "analysis_blacklist": "data/binary/analysis-blacklist.json"}],
                "elo": {"initial_rating": 1500.0, "k_factor": 40.0, "tie_tolerance": 1e-12, "iterations": 250, "regularization": 1e-4, "convergence_tolerance": 1e-8},
                "families": [{"id": "family", "display_name": "Family", "patterns": ["family-*"]}]
            })
            .to_string(),
        )
        .unwrap();
        fs::write(
            root.join("data/binary/analysis-blacklist.json"),
            r#"{"schema":"sotarena.analysis-blacklist","task":"binary","datasets":[]}"#,
        )
        .unwrap();
        fs::write(
            root.join("data/binary/0123456789abcdef/dataset.json"),
            serde_json::json!({
                "schema": "sotarena.dataset", "dataset_hash": "0123456789abcdef", "dataset_id": "0123456789abcdef",
                "task": "binary", "metric": "roc_auc",
                "train": {"url": "https://example.test/train.csv", "sha256": "a".repeat(64), "bytes": 1, "rows": 200},
                "test": {"url": "https://example.test/test.csv", "sha256": "b".repeat(64), "bytes": 1, "rows": 200}
            })
            .to_string(),
        )
        .unwrap();
        for (index, (method, dataset, value, valid)) in records.iter().enumerate() {
            fs::write(
                root.join(format!("data/binary/0123456789abcdef/results/{index}.json")),
                serde_json::json!({
                    "schema": "sotarena.result.v1", "task": "binary", "metric": "roc_auc",
                    "method_id": method, "display_name": method, "claim_scope": "test",
                    "dataset_id": dataset, "dataset_hash": dataset, "value": value,
                    "test_sha256": "b".repeat(64), "predictions_sha256": null, "run_id": format!("run-{index}"),
                    "benchmark_valid": valid
                })
                .to_string(),
            )
            .unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reporter's `patterns_are_exact_or_one_trailing_prefix_wildcard`.
    #[test]
    fn patterns_are_exact_or_one_trailing_prefix_wildcard() {
        assert!(validate_pattern("jope-*").is_ok());
        assert!(validate_pattern("tabfm").is_ok());
        assert!(validate_pattern("*").is_err());
        assert!(validate_pattern("a*b").is_err());
        assert!(validate_pattern("").is_err());
        assert!(pattern_matches("jope-*", "jope-dime"));
        assert!(!pattern_matches("jope-*", "jopedime"));
        assert!(pattern_matches("tabfm", "tabfm"));
        assert!(!pattern_matches("tabfm", "tabfm_ensemble"));
        assert!(patterns_overlap("tabfm*", "tabfm_ensemble*"));
        assert!(!patterns_overlap("tabfm", "tabfm_ensemble*"));
    }

    /// The reporter's `hashes_and_urls_fail_closed`.
    #[test]
    fn hashes_and_urls_fail_closed() {
        assert!(valid_hash("0123456789abcdef"));
        assert!(!valid_hash("0123456789ABCDEF"));
        assert!(!valid_hash("0123456789abcde"));
        assert!(valid_https_csv_url(
            "https://data.example/regression/x/train.csv"
        ));
        assert!(!valid_https_csv_url("http://data.example/train.csv"));
        assert!(!valid_https_csv_url("https://data.example/train.csv?x=1"));
        assert!(valid_metric_value("rmse", 0.0));
        assert!(!valid_metric_value("roc_auc", 1.5));
        assert!(!valid_metric_value("rmse", f64::NAN));
    }

    #[test]
    fn config_validation_rejects_overlaps_and_bad_weights() {
        let mut config = fixtures::test_config(Direction::Maximize);
        assert!(validate_config(&config).is_ok());
        config.families.push(FamilyConfig {
            id: "other".to_owned(),
            display_name: "Other".to_owned(),
            patterns: vec!["family-x*".to_owned()],
        });
        assert!(validate_config(&config)
            .unwrap_err()
            .to_string()
            .contains("overlapping patterns"));
        config.families.pop();
        config.tasks[0].weight = 0.5;
        assert!(validate_config(&config)
            .unwrap_err()
            .to_string()
            .contains("sum to 1.0"));
    }

    #[test]
    fn results_keep_the_best_score_and_track_validity() {
        let dir = tempfile::tempdir().unwrap();
        fixtures::write_reporter_root(
            dir.path(),
            &[
                ("family-a", "0123456789abcdef", 0.7, false),
                ("family-a", "0123456789abcdef", 0.9, true),
                ("other", "0123456789abcdef", 0.8, false),
            ],
        );
        let config = load_config(dir.path()).unwrap();
        let datasets = load_datasets(dir.path(), &config).unwrap();
        assert_eq!(datasets["binary"].len(), 1);
        let loaded = load_results(dir.path(), &config, &datasets).unwrap();
        assert_eq!(loaded.result_jsons_total, 3);
        assert_eq!(
            loaded.catalog["binary"]["family-a"]["0123456789abcdef"],
            0.9
        );
        assert_eq!(loaded.catalog["binary"]["other"]["0123456789abcdef"], 0.8);
        assert_eq!(loaded.valid_catalog["binary"].len(), 1);
        let blacklists = load_analysis_blacklists(dir.path(), &config, &datasets).unwrap();
        assert!(blacklists["binary"].is_empty());
    }
}
