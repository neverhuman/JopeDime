//! `jope pipeline score --split test`: the one look at a dataset's sealed test split.
//!
//! Everything before this point — search, selection, the HWM ledger, the champion bank, the
//! lineage — measures on the training rows only (`dataset::resolve` hashes `test.csv` to verify
//! it, and nothing else reads its rows). This refits a *verified* receipt's selected genome on
//! all of `train.csv` and
//! scores it once on the sha-verified `test.csv`, writing a separate
//! `jope.pipeline-test-score/v1` receipt. It never feeds selection: nothing in the pipeline reads
//! these receipts back, and the run's own receipt is not touched.
//!
//! "Once" is enforced per `(dataset, genome)` by the registry: the score receipt's path is
//! `<registry>/<dataset_hash>/<program_sha256>.json`, created exclusively (`O_EXCL`), and checked
//! before anything is fitted. A second request for the same pair — from the same receipt or any
//! other run that selected the same genome — is refused. The default registry is
//! `$JOPE_STATE/pipeline/test-scores`, so one host scores a pair once unless an operator points
//! elsewhere on purpose; the receipt records which registry it was written to.

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
    /// Where the once-per-(dataset, genome) record lives.
    pub registry: String,
    /// Always false: a test score is never an input to selection.
    pub feeds_selection: bool,
    pub seconds: f64,
}

pub struct ScoreOptions {
    pub receipt: PathBuf,
    /// Defaults to the receipt's recorded dataset path.
    pub dataset: Option<String>,
    pub registry: PathBuf,
    pub clock: Clock,
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
    ensure!(
        !path.exists(),
        "the test split of dataset {} was already scored for genome {} ({}); a test split is \
         scored once per (dataset, genome) and never again",
        receipt.dataset.dataset_hash,
        receipt.search.selected_program_sha256,
        path.display()
    );

    let started = options.clock.start();
    let train = dataset::training_rows(&resolved)?;
    let test = dataset::test_rows(&resolved)?;
    let (train_x, train_y) = split_rows(&train);
    let (test_x, test_y) = split_rows(&test);
    let parsed = GenomeIr::parse(&receipt.search.selected_genome)?;
    // On the stack the search fitted this genome with, not whatever the caller's thread has.
    let predictions = crate::parallel::on_evaluation_stack(|| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let model = feat_math::fit(&parsed, &train_x, &train_y)?;
            anyhow::Ok(model.predict(&test_x))
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
        registry: options.registry.display().to_string(),
        feeds_selection: false,
        seconds: started.secs(),
    };
    std::fs::create_dir_all(&directory)
        .with_context(|| format!("create {}", directory.display()))?;
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
    Ok((record, path))
}

pub fn read(path: &Path) -> Result<TestScore> {
    let score: TestScore = serde_json::from_slice(&std::fs::read(path)?)
        .with_context(|| format!("parse {}", path.display()))?;
    ensure!(score.schema == TEST_SCORE_SCHEMA_ID, "not a test score");
    Ok(score)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        emit::Language,
        run::{run, Policy, RunOptions},
        search::Budget,
    };

    fn dataset(slug: &str) -> String {
        format!("{}/../../datasets/test/{slug}", env!("CARGO_MANIFEST_DIR"))
    }

    fn a_run(root: &Path, name: &str) -> PathBuf {
        let output = root.join(name);
        run(&RunOptions {
            budget: Budget {
                evaluations: 6,
                seconds: 600,
            },
            rng_seed: 7,
            languages: vec![Language::Rust],
            policy: Policy::Skip,
            ..RunOptions::new(dataset("pmlb-229-pwlinear"), &output)
        })
        .unwrap();
        output.join("receipt.json")
    }

    fn options(receipt: &Path, registry: &Path) -> ScoreOptions {
        ScoreOptions {
            receipt: receipt.to_path_buf(),
            dataset: None,
            registry: registry.to_path_buf(),
            clock: Clock::Wall,
        }
    }

    #[test]
    fn the_test_split_is_scored_once_per_dataset_and_genome() {
        let temp = tempfile::tempdir().unwrap();
        let receipt = a_run(temp.path(), "run");
        let before = std::fs::read(&receipt).unwrap();
        let registry = temp.path().join("registry");
        let (scored, path) = score(&options(&receipt, &registry)).unwrap();
        assert_eq!(scored.schema, TEST_SCORE_SCHEMA_ID);
        assert_eq!(scored.split, "test");
        assert!(!scored.feeds_selection);
        assert_eq!(scored.dataset.test_rows, 34);
        assert!(scored.test_rmse.is_finite() && scored.test_rmse >= 0.0);
        assert_eq!(scored.receipt_sha256, crate::sha256_hex(&before));
        assert_eq!(read(&path).unwrap(), scored);
        assert!(path.ends_with(format!("1cb97b1d7314d43d/{}.json", scored.program_sha256)));
        // The run's own receipt is untouched: the test score lives beside it, never in it.
        assert_eq!(std::fs::read(&receipt).unwrap(), before);

        // A repeat is refused, and so is any other run that selected the same genome.
        let again = format!("{:#}", score(&options(&receipt, &registry)).unwrap_err());
        assert!(again.contains("already scored"), "{again}");
        let twin = a_run(temp.path(), "twin");
        let error = format!("{:#}", score(&options(&twin, &registry)).unwrap_err());
        assert!(error.contains("already scored"), "{error}");
        assert_eq!(read(&path).unwrap(), scored);
    }

    #[test]
    fn a_receipt_that_does_not_verify_or_another_dataset_scores_nothing() {
        let temp = tempfile::tempdir().unwrap();
        let receipt = a_run(temp.path(), "run");
        let registry = temp.path().join("registry");

        let mismatch = ScoreOptions {
            dataset: Some(dataset("pmlb-695-chatfield-4")),
            ..options(&receipt, &registry)
        };
        let error = format!("{:#}", score(&mismatch).unwrap_err());
        assert!(error.contains("the receipt measured"), "{error}");

        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&receipt).unwrap()).unwrap();
        let rmse = value["search"]["best"]["validation_rmse"].as_f64().unwrap();
        value["search"]["best"]["validation_rmse"] = (rmse / 2.0).into();
        std::fs::write(&receipt, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
        assert!(score(&options(&receipt, &registry)).is_err());
        assert!(!registry.exists(), "a refusal wrote to the registry");
    }

    #[test]
    fn test_rows_are_the_sealed_split_parsed_like_the_training_rows() {
        let resolved = dataset::resolve(&dataset("pmlb-229-pwlinear")).unwrap();
        let train = dataset::training_rows(&resolved).unwrap();
        let test = dataset::test_rows(&resolved).unwrap();
        assert_eq!(train.len() as u64, resolved.row.split.train_rows);
        assert_eq!(test.len() as u64, resolved.row.split.test_rows);
        assert!(test
            .iter()
            .all(|row| row.len() == resolved.row.shape.columns as usize));
    }
}
