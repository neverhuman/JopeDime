//! `jope pipeline score --split test`: a genome's look at its dataset's sealed test split.
//!
//! Everything before this point — search, selection, the HWM ledger, the champion bank, the
//! lineage — measures on the training rows only (`dataset::resolve` hashes `test.csv` to verify
//! it, and nothing else reads its rows). This refits a *verified* receipt's selected genome on
//! all of `train.csv` and scores it once on the sha-verified `test.csv`, writing a separate
//! `jope.pipeline-test-score/v1` receipt. Nothing in the pipeline reads these receipts back and
//! the run's own receipt is not touched, so the tools never feed a test score to selection.
//!
//! What is enforced, and what is not. "Once" is per `(dataset, genome)` in one registry: the
//! score receipt's path is `<registry>/<dataset_hash>/<program_sha256>.json`, created exclusively
//! (`O_EXCL`), and checked before anything is fitted, so a second request for the same pair — from
//! the same receipt or any other run that selected the same genome — is refused. It is not once
//! per dataset: every other genome of the dataset can still be scored, and an operator who scores
//! several (every round's champion, the control) and keeps the test-best has made the test split
//! a selection input. The registry is a local record the caller chooses (`--registry`, default
//! `$JOPE_STATE/pipeline/test-scores`), not access control: another registry starts empty. What
//! the tool can do is not hide it: every score records the registry it was written to and
//! `earlier_looks`, the genomes already scored on the same dataset's test split there, and the
//! command warns when there are any.
//!
//! Concurrency. The repeat check, the `earlier_looks` listing, the fit and the publication are
//! one transaction per (registry, dataset). `score` takes an exclusive advisory lock
//! (`std::fs::File::lock`, `flock` on Linux) on `<registry>/<dataset_hash>/.lock` before the
//! repeat check and holds it until the score receipt is written and synced, with its directory
//! entry. A second scorer of the same dataset in the same registry waits for the lock (and says
//! so), then sees the first score: it is refused before it fits if the first scored the same
//! genome, and otherwise lists the first among its `earlier_looks`, so the command warns. The
//! lock file is not a look, and it stays (removing it would let a waiter lock an unlinked file).
//! The lock orders only scorers that take it, on a file system that honours `flock` between
//! them (a local one); like the registry, it is not access control.

use anyhow::{bail, ensure, Context, Result};
use feat_math::{
    genome::GenomeIr,
    metrics::{r2, rmse},
    Rows,
};
use serde::{Deserialize, Serialize};
use std::{
    io::Write,
    path::{Path, PathBuf},
};

use crate::{clock::Clock, dataset, evidence, TOOL_VERSION};

pub const TEST_SCORE_SCHEMA_ID: &str = "jope.pipeline-test-score/v1";

/// The per-dataset lock in `<registry>/<dataset_hash>/`. Not a look: `earlier_looks` lists only
/// `<program_sha256>.json` entries.
pub const LOCK_FILE: &str = ".lock";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoredDataset {
    pub id: String,
    pub dataset_hash: String,
    pub train_sha256: String,
    pub test_sha256: String,
    pub train_rows: u64,
    pub test_rows: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TestScore {
    pub schema: String,
    pub tool_version: String,
    pub created_at: String,
    /// Always `test`.
    pub split: String,
    /// The verified pipeline receipt whose selected genome this scores.
    pub receipt_sha256: String,
    pub dataset: ScoredDataset,
    pub genome: String,
    pub program_sha256: String,
    /// On the test rows, the dataset's own target units; fitted on every training row.
    pub test_rmse: f64,
    pub test_r2: f64,
    /// The receipt's development estimate (mean held-out RMSE over its cuts), for reference.
    pub validation_rmse: f64,
    /// When the run ranked programs by a regression objective: that objective on the test rows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective: Option<ObjectiveScore>,
    /// Where the once-per-(dataset, genome) record lives.
    pub registry: String,
    /// The genomes (program sha256s, sorted) already scored on this dataset's test split in this
    /// registry when this score was written. Non-empty means this is not the split's first
    /// look: choosing among several test-scored genomes makes the test split a selection input.
    /// Absent from scores written before it was recorded.
    #[serde(default)]
    pub earlier_looks: Vec<String>,
    /// Always false: nothing in the tools reads a test score back into selection.
    pub feeds_selection: bool,
    pub seconds: f64,
}

/// The receipt's objective scored on the sealed test rows.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectiveScore {
    pub name: String,
    pub fitness_sha256: String,
    /// The objective's natural value on the test rows; absent when it is undefined there.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub test_value: Option<f64>,
    /// Its minimization loss (the value negated for a maximized objective).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub test_loss: Option<f64>,
    /// Why the objective is undefined on the test rows (a domain or nonfinite failure). The
    /// look is recorded and spent all the same, so a failing objective cannot probe test labels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub undefined: Option<String>,
    /// The receipt's mean held-out loss over its cuts, for reference; absent when the run
    /// ranked by RMSE or by another objective.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validation_loss: Option<f64>,
}

pub struct ScoreOptions {
    pub receipt: PathBuf,
    /// Defaults to the receipt's recorded dataset path.
    pub dataset: Option<String>,
    pub registry: PathBuf,
    pub clock: Clock,
    /// Also report this objective on the test rows (`--fitness`); defaults to the receipt's own.
    /// It is a report of the same one look, never an input to selection.
    pub objective: Option<jope_runtime::fitness::Fitness>,
}

/// `$JOPE_STATE/pipeline/test-scores`.
pub fn default_registry() -> PathBuf {
    jope_paths::jope_state()
        .join("pipeline")
        .join("test-scores")
}

fn plain_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// The genomes already scored in `directory` (one dataset's registry directory), sorted.
fn earlier_looks(directory: &Path) -> Result<Vec<String>> {
    let entries = match std::fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(error).with_context(|| format!("list {}", directory.display()));
        }
    };
    let mut looks = Vec::new();
    for entry in entries {
        let name = entry
            .with_context(|| format!("list {}", directory.display()))?
            .file_name();
        if let Some(stem) = name.to_str().and_then(|n| n.strip_suffix(".json")) {
            if plain_hex(stem, 64) {
                looks.push(stem.to_owned());
            }
        }
    }
    looks.sort();
    Ok(looks)
}

/// What [`score_observed`] reports while it runs; tests use it to order concurrent scorers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    /// Another scorer holds the dataset's lock in this registry; this one waits for it.
    Waiting,
    /// The lock is held, the repeat check passed and the earlier looks are read: the fit is next.
    Admitted,
}

/// Take the exclusive lock on `directory`'s [`LOCK_FILE`], waiting while another scorer holds
/// it. The lock is released when the returned file is dropped.
fn lock_dataset(directory: &Path, observe: &dyn Fn(Stage)) -> Result<std::fs::File> {
    std::fs::create_dir_all(directory)
        .with_context(|| format!("create {}", directory.display()))?;
    let path = directory.join(LOCK_FILE);
    let lock = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&path)
        .with_context(|| format!("open {}", path.display()))?;
    match lock.try_lock() {
        Ok(()) => {}
        Err(std::fs::TryLockError::WouldBlock) => {
            observe(Stage::Waiting);
            eprintln!(
                "jope pipeline score: another score of this dataset holds {}; waiting for it \
                 (the scores of one dataset in one registry run one at a time)",
                path.display()
            );
            lock.lock()
                .with_context(|| format!("lock {}", path.display()))?;
        }
        Err(std::fs::TryLockError::Error(error)) => {
            return Err(error).with_context(|| format!("lock {}", path.display()));
        }
    }
    Ok(lock)
}

/// fsync a directory, so the entries just created in it are durable.
fn sync_directory(directory: &Path) -> Result<()> {
    std::fs::File::open(directory)
        .and_then(|d| d.sync_all())
        .with_context(|| format!("sync {}", directory.display()))
}

fn split_rows(rows: &[Vec<f64>]) -> (Rows, Vec<f64>) {
    rows.iter()
        .map(|row| {
            let (features, target) = row.split_at(row.len() - 1);
            (features.to_vec(), target[0])
        })
        .unzip()
}

/// Score a verified receipt's champion on the test split, once.
pub fn score(options: &ScoreOptions) -> Result<(TestScore, PathBuf)> {
    score_observed(options, &|_| {})
}

fn score_observed(options: &ScoreOptions, observe: &dyn Fn(Stage)) -> Result<(TestScore, PathBuf)> {
    let run = evidence::verify_champion_run(&options.receipt)?;
    let receipt = &run.receipt;
    let selector = options
        .dataset
        .clone()
        .unwrap_or_else(|| receipt.dataset.path.clone());
    let resolved = dataset::resolve(&selector)?;
    ensure!(
        resolved.row.dataset_hash == receipt.dataset.dataset_hash,
        "{selector} is dataset {}, the receipt measured {}",
        resolved.row.dataset_hash,
        receipt.dataset.dataset_hash
    );
    let digest = |name: &str| {
        resolved
            .files
            .iter()
            .find(|f| f.name == name)
            .map(|f| f.sha256.clone())
            .with_context(|| format!("{selector}: the catalog names no {name}"))
    };
    let (train_sha256, test_sha256) = (digest("train.csv")?, digest("test.csv")?);
    let recorded_train = receipt
        .dataset
        .files
        .iter()
        .find(|f| f.name == "train.csv")
        .map(|f| f.sha256.as_str());
    ensure!(
        recorded_train == Some(train_sha256.as_str()),
        "{selector}: train.csv is not the file the receipt's search fitted on"
    );
    ensure!(
        plain_hex(&receipt.dataset.dataset_hash, 16)
            && plain_hex(&receipt.search.selected_program_sha256, 64),
        "the receipt's dataset hash or program digest is not a plain registry key"
    );
    let directory = options.registry.join(&receipt.dataset.dataset_hash);
    let path = directory.join(format!("{}.json", receipt.search.selected_program_sha256));
    // From here to the synced publication this scorer holds the dataset's lock, so no other
    // score of it in this registry is admitted, reads the looks or publishes in between.
    let lock = lock_dataset(&directory, observe)?;
    ensure!(
        !path.exists(),
        "the test split of dataset {} was already scored for genome {} ({}); a test split is \
         scored once per (dataset, genome) and never again",
        receipt.dataset.dataset_hash,
        receipt.search.selected_program_sha256,
        path.display()
    );
    let earlier_looks = earlier_looks(&directory)?;
    observe(Stage::Admitted);

    let started = options.clock.start();
    let train = dataset::training_rows(&resolved)?;
    let test = dataset::test_rows(&resolved)?;
    let (train_x, train_y) = split_rows(&train);
    let (test_x, test_y) = split_rows(&test);
    let parsed = GenomeIr::parse(&receipt.search.selected_genome)?;
    let recorded = receipt
        .search
        .objective
        .as_ref()
        .map(crate::receipt::Objective::fitness)
        .transpose()?;
    // An objective that reads resources is scored with readings of this refit and these test
    // rows, measured the way the search measured its candidates.
    let wanted = options
        .objective
        .as_ref()
        .or(recorded.as_ref())
        .map(jope_runtime::fitness::Fitness::resources)
        .unwrap_or_default();
    // On the stack the search fitted this genome with, not whatever the caller's thread has.
    let (predictions, resources) = crate::parallel::on_evaluation_stack(|| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if wanted.is_empty() {
                let model = feat_math::fit(&parsed, &train_x, &train_y)?;
                return anyhow::Ok((model.predict(&test_x)?, Default::default()));
            }
            let cut = crate::prepare::Cut {
                index: 0,
                train_x: train_x.clone(),
                train_y: train_y.clone(),
                hold_x: test_x.clone(),
                hold_y: test_y.clone(),
                hold_indices: (0..test_y.len()).collect(),
            };
            let measured = crate::resource_measure::fit_and_measure(
                &parsed,
                &cut,
                &wanted,
                options.clock,
                options.clock.start(),
            )?;
            anyhow::Ok((measured.hold, measured.resources))
        }))
        .map_err(|_| anyhow::anyhow!("the fit panicked"))
    })???;
    ensure!(
        predictions.len() == test_y.len(),
        "predicted {} values for {} test rows",
        predictions.len(),
        test_y.len()
    );
    let (test_rmse, test_r2) = (rmse(&test_y, &predictions), r2(&test_y, &predictions));
    if !test_rmse.is_finite() || !test_r2.is_finite() {
        bail!("nonfinite test score: rmse {test_rmse}, r2 {test_r2}");
    }
    let objective = match options.objective.clone().or_else(|| recorded.clone()) {
        None => None,
        Some(fitness) => {
            jope_runtime::fitness::require_no_columns(&fitness, "a test score")?;
            let fitness_sha256 = fitness.identity()?;
            // Whatever the objective makes of the test rows, the look below is recorded.
            let scored = fitness.evaluate_with_resources(
                &test_y,
                &predictions,
                &Default::default(),
                &resources,
            );
            let (test_value, test_loss, undefined) = match scored {
                Ok(score) => (Some(score.value), Some(score.loss), None),
                Err(error) => (None, None, Some(format!("{error:#}"))),
            };
            let ranked_by_it = recorded
                .as_ref()
                .is_some_and(|r| r.identity().ok().as_ref() == Some(&fitness_sha256));
            Some(ObjectiveScore {
                name: fitness.name(),
                fitness_sha256,
                test_value,
                test_loss,
                undefined,
                validation_loss: if ranked_by_it {
                    Some(
                        receipt
                            .search
                            .best
                            .as_ref()
                            .and_then(|b| b.validation_objective)
                            .context("the receipt's champion has no objective loss")?,
                    )
                } else {
                    None
                },
            })
        }
    };
    let record = TestScore {
        schema: TEST_SCORE_SCHEMA_ID.to_owned(),
        tool_version: TOOL_VERSION.to_owned(),
        created_at: options.clock.now_rfc3339(),
        split: "test".to_owned(),
        receipt_sha256: run.receipt_sha256.clone(),
        dataset: ScoredDataset {
            id: receipt.dataset.id.clone(),
            dataset_hash: receipt.dataset.dataset_hash.clone(),
            train_sha256,
            test_sha256,
            train_rows: train.len() as u64,
            test_rows: test.len() as u64,
        },
        genome: receipt.search.selected_genome.clone(),
        program_sha256: receipt.search.selected_program_sha256.clone(),
        test_rmse,
        test_r2,
        validation_rmse: receipt
            .search
            .best
            .as_ref()
            .context("the receipt names no measured champion")?
            .validation_rmse,
        objective,
        registry: options.registry.display().to_string(),
        earlier_looks,
        feeds_selection: false,
        seconds: started.secs(),
    };
    let mut bytes = serde_json::to_vec_pretty(&record)?;
    bytes.push(b'\n');
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .with_context(|| {
            format!(
                "{}: the test split was scored concurrently; it is scored once",
                path.display()
            )
        })?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    sync_directory(&directory)?;
    sync_directory(&options.registry)?;
    drop(lock);
    Ok((record, path))
}

pub fn read(path: &Path) -> Result<TestScore> {
    let score: TestScore = serde_json::from_slice(&std::fs::read(path)?)
        .with_context(|| format!("parse {}", path.display()))?;
    ensure!(score.schema == TEST_SCORE_SCHEMA_ID, "not a test score");
    Ok(score)
}

#[cfg(test)]
mod tests;
