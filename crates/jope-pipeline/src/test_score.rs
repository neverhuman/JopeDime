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
    // On the stack the search fitted this genome with, not whatever the caller's thread has.
    let predictions = crate::parallel::on_evaluation_stack(|| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let model = feat_math::fit(&parsed, &train_x, &train_y)?;
            anyhow::Ok(model.predict(&test_x)?)
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
    let recorded = receipt
        .search
        .objective
        .as_ref()
        .map(crate::receipt::Objective::fitness)
        .transpose()?;
    let objective = match options.objective.clone().or_else(|| recorded.clone()) {
        None => None,
        Some(fitness) => {
            jope_runtime::fitness::require_no_columns(&fitness, "a test score")?;
            let fitness_sha256 = fitness.identity()?;
            // Whatever the objective makes of the test rows, the look below is recorded.
            let (test_value, test_loss, undefined) = match fitness.evaluate(&test_y, &predictions) {
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
        a_run_of(root, name, 6)
    }

    fn a_run_of(root: &Path, name: &str, evaluations: usize) -> PathBuf {
        let output = root.join(name);
        run(&RunOptions {
            budget: Budget {
                evaluations,
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
            objective: None,
        }
    }

    #[test]
    fn a_test_score_reports_the_run_objective_or_a_requested_one() {
        let temp = tempfile::tempdir().unwrap();
        let registry = temp.path().join("registry");
        let objective = jope_runtime::fitness::Fitness::parse("pinball:quantile=0.9").unwrap();
        let ranked = temp.path().join("ranked");
        run(&RunOptions {
            budget: Budget {
                evaluations: 6,
                seconds: 600,
            },
            rng_seed: 7,
            languages: vec![Language::Rust],
            policy: Policy::Skip,
            objective: Some(objective.clone()),
            ..RunOptions::new(dataset("pmlb-229-pwlinear"), &ranked)
        })
        .unwrap();
        let (score_ranked, _) = score(&options(&ranked.join("receipt.json"), &registry)).unwrap();
        let reported = score_ranked.objective.unwrap();
        assert_eq!(reported.name, "pinball");
        assert_eq!(reported.fitness_sha256, objective.identity().unwrap());
        assert!(reported.validation_loss.is_some());
        assert!(reported
            .test_loss
            .is_some_and(|l| l.is_finite() && l >= 0.0));
        assert!(reported.undefined.is_none());
        // An RMSE run's single look can also report the objective, with no development value.
        let plain = a_run_of(temp.path(), "plain", 8);
        let (score_plain, _) = score(&ScoreOptions {
            objective: Some(objective),
            ..options(&plain, &temp.path().join("another-registry"))
        })
        .unwrap_or_else(|e| panic!("{e:#}"));
        let reported = score_plain.objective.unwrap();
        assert!(reported.validation_loss.is_none());
        // An objective that is undefined on the test rows still spends the one look: the score
        // and its RMSE are recorded, and a second probe of the same genome is refused.
        let probe: jope_runtime::fitness::Fitness = serde_json::from_value(serde_json::json!({
            "metric": "custom", "name": "probe", "direction": "minimize",
            "expression": {"op": "ln", "arg": {"op": "sub", "left": {"op": "y"},
                "right": {"op": "constant", "value": 1e300}}},
            "aggregation": {"kind": "mean"}}))
        .unwrap();
        let probed = a_run_of(temp.path(), "probed", 9);
        let probe_registry = temp.path().join("probe-registry");
        let (scored, path) = score(&ScoreOptions {
            objective: Some(probe.clone()),
            ..options(&probed, &probe_registry)
        })
        .unwrap_or_else(|e| panic!("{e:#}"));
        let reported = scored.objective.as_ref().unwrap();
        assert!(reported.test_value.is_none() && reported.test_loss.is_none());
        assert!(reported.undefined.is_some());
        assert!(scored.test_rmse.is_finite());
        assert_eq!(read(&path).unwrap(), scored);
        assert!(score(&ScoreOptions {
            objective: Some(probe),
            ..options(&probed, &probe_registry)
        })
        .is_err());
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

    /// The registry enforces once per (dataset, genome), not once per dataset: every other genome
    /// of the same dataset can still be scored. Each score therefore names the looks at that
    /// dataset's test split the registry already holds, so choosing among several test-scored
    /// genomes is visible in the receipts rather than hidden.
    #[test]
    fn a_score_names_the_earlier_looks_at_its_datasets_test_split() {
        let temp = tempfile::tempdir().unwrap();
        let registry = temp.path().join("registry");
        let run_with = |name: &str, evaluations: usize| {
            let output = temp.path().join(name);
            run(&RunOptions {
                budget: Budget {
                    evaluations,
                    seconds: 600,
                },
                rng_seed: 7,
                languages: vec![Language::Rust],
                policy: Policy::Skip,
                ..RunOptions::new(dataset("pmlb-229-pwlinear"), &output)
            })
            .unwrap();
            output.join("receipt.json")
        };
        let bootstrap = run_with("bootstrap", 1);
        let searched = run_with("searched", 24);
        let (first, first_path) = score(&options(&bootstrap, &registry)).unwrap();
        let (second, second_path) = score(&options(&searched, &registry)).unwrap();
        assert_ne!(first.program_sha256, second.program_sha256);
        let looks = |path: &Path| -> serde_json::Value {
            serde_json::from_slice::<serde_json::Value>(&std::fs::read(path).unwrap()).unwrap()
                ["earlier_looks"]
                .clone()
        };
        assert_eq!(looks(&first_path), serde_json::json!([]));
        assert_eq!(
            looks(&second_path),
            serde_json::json!([first.program_sha256]),
            "the second genome's test score hides the first look"
        );
        // Another dataset's looks are not this one's.
        let other = temp.path().join("other");
        run(&RunOptions {
            budget: Budget {
                evaluations: 1,
                seconds: 600,
            },
            languages: vec![Language::Rust],
            policy: Policy::Skip,
            ..RunOptions::new(dataset("pmlb-695-chatfield-4"), &other)
        })
        .unwrap();
        let (_, other_path) = score(&options(&other.join("receipt.json"), &registry)).unwrap();
        assert_eq!(looks(&other_path), serde_json::json!([]));
    }

    type Scored = Result<(TestScore, PathBuf)>;

    /// Scores `first` and `second` into their registries at once, deterministically interleaved:
    /// `first` runs until it is admitted (it holds the dataset's lock and has read the earlier
    /// looks), then `second` starts, and `first` goes on to fit and publish only once `second`
    /// has found the lock held. Without the lock `second` would read the looks and publish while
    /// `first` is still paused, however quick either fit is. Returns both results and the stages
    /// `second` reported. Threads with their own lock files contend exactly as processes do:
    /// `flock` locks belong to the open file, not to the process.
    fn race(first: &ScoreOptions, second: &ScoreOptions) -> (Scored, Scored, Vec<Stage>) {
        use std::{sync::mpsc, time::Duration};
        let patience = Duration::from_secs(600);
        let (admitted_tx, admitted_rx) = mpsc::channel::<()>();
        let (stage_tx, stage_rx) = mpsc::channel::<Stage>();
        let seen = std::sync::Mutex::new(Vec::new());
        let seen = &seen;
        std::thread::scope(|scope| {
            let holder = scope.spawn(move || {
                score_observed(first, &|stage| {
                    if stage == Stage::Admitted {
                        admitted_tx.send(()).unwrap();
                        // Every stage `second` reports is sent here, and its sender is dropped
                        // when it returns, so a `second` that never waits fails this at once.
                        match stage_rx.recv_timeout(patience) {
                            Ok(Stage::Waiting) => {}
                            other => {
                                panic!("the second scorer did not wait for the lock: {other:?}")
                            }
                        }
                    }
                })
            });
            admitted_rx
                .recv_timeout(patience)
                .expect("the first scorer was not admitted");
            let waiter = scope.spawn(move || {
                let scored = score_observed(second, &|stage| {
                    seen.lock().unwrap().push(stage);
                    let _ = stage_tx.send(stage);
                });
                drop(stage_tx);
                scored
            });
            let first = holder.join().expect("the first scorer panicked");
            let second = waiter.join().expect("the second scorer panicked");
            (first, second, seen.lock().unwrap().clone())
        })
    }

    /// Review 5334625316 on PR #118: two genomes of one dataset scored into one registry at once
    /// both recorded `earlier_looks: []`. The reviewer's pair: the slower fit starts first
    /// (24 evaluations select `(pipe X (rank append=true) (gpr m=400 steps=60))`), the quicker
    /// one (1 evaluation, `(pipe X (ridge))`) while it fits. The second now waits for the first
    /// and names it.
    #[test]
    fn a_concurrent_score_waits_and_names_the_look_it_waited_for() {
        let temp = tempfile::tempdir().unwrap();
        let registry = temp.path().join("registry");
        let slow = a_run_of(temp.path(), "slow", 24);
        let quick = a_run_of(temp.path(), "quick", 1);
        let (first, second, stages) = race(&options(&slow, &registry), &options(&quick, &registry));
        let (first, first_path) = first.unwrap();
        let (second, second_path) = second.unwrap();
        assert_ne!(first.program_sha256, second.program_sha256);
        assert_eq!(stages, [Stage::Waiting, Stage::Admitted]);
        assert!(first.earlier_looks.is_empty(), "{:?}", first.earlier_looks);
        assert_eq!(
            second.earlier_looks,
            std::slice::from_ref(&first.program_sha256),
            "the second score hides the look it waited for"
        );
        assert_eq!(read(&first_path).unwrap(), first);
        assert_eq!(read(&second_path).unwrap(), second);
        // The lock file stays beside the scores and is not a look.
        let directory = first_path.parent().unwrap();
        assert_eq!(second_path.parent().unwrap(), directory);
        assert!(directory.join(LOCK_FILE).is_file());
        let mut both = vec![first.program_sha256.clone(), second.program_sha256.clone()];
        both.sort();
        assert_eq!(earlier_looks(directory).unwrap(), both);
    }

    /// The same genome scored twice at once (two runs that selected it): the second waits, then
    /// is refused by the repeat check before it fits (it is never admitted), and the first
    /// score is untouched.
    #[test]
    fn a_concurrent_repeat_is_refused_before_it_fits() {
        let temp = tempfile::tempdir().unwrap();
        let registry = temp.path().join("registry");
        let receipt = a_run(temp.path(), "run");
        let twin = a_run(temp.path(), "twin");
        let (first, second, stages) =
            race(&options(&receipt, &registry), &options(&twin, &registry));
        let (first, path) = first.unwrap();
        let error = format!("{:#}", second.unwrap_err());
        assert!(error.contains("already scored"), "{error}");
        assert_eq!(stages, [Stage::Waiting], "the repeat was admitted to a fit");
        assert!(first.earlier_looks.is_empty(), "{:?}", first.earlier_looks);
        assert_eq!(read(&path).unwrap(), first);
        assert_eq!(
            earlier_looks(path.parent().unwrap()).unwrap(),
            std::slice::from_ref(&first.program_sha256)
        );
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
