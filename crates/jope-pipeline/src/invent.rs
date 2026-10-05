//! `jope pipeline invent`: the pipeline's GP search as a candidate generator for another engine.
//!
//! jain-core's Phase-6 invention (the `Jope` engine) hands this command a header CSV of the
//! champion feature frame and reads back ranked genomes. It re-fits, cross-validates, reports and
//! emits every genome itself, so what crosses the boundary is the canonical DSL string and the
//! measurements that ranked it here, not predictions.
//!
//! The search is [`crate::run`]'s, minus everything a caller with its own data does not need: no
//! catalog row, no receipt, no emitted sources. Rows are partitioned by [`prepare::partition`]
//! (keyed on the CSV's sha256), searched by [`search::run_cuts_with`] and ranked by
//! [`select::ranking`], unchanged.
//!
//! A ranked genome is offered only when every stage op is executable in the jain-math revision
//! the caller links ([`JAIN_EXECUTABLE_OPS`], the same snapshot `jope-engine` restricts its
//! vocabulary with). The search grammar is wider (its heads include `ridge_gcv`), so programs
//! outside that set are counted and left out rather than offered for the caller to reject.

use anyhow::{ensure, Context, Result};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::{clock::Clock, prepare, search, select, sha256_hex};

/// The stdout document's schema.
pub const SCHEMA: &str = "jope.candidates.v1";

/// The jain-math registry snapshot that decides which genomes are offered.
pub const JAIN_EXECUTABLE_OPS: &str =
    include_str!("../../jope-engine/assets/grammar/jain-math-executable-9fa594e2.json");

/// Exit status when the search measured nothing the caller can execute.
pub const NO_CANDIDATES_EXIT: u8 = 3;

/// Most genomes one call may return.
pub const MAX_TOP_K: usize = 64;

#[derive(Clone, Debug)]
pub struct InventOptions {
    /// Header CSV: every column numeric and finite, one of them the target.
    pub train: PathBuf,
    pub target: String,
    /// A new or empty directory the search publishes its per-cut evidence under.
    pub work_dir: PathBuf,
    pub budget: search::Budget,
    pub seed: u64,
    pub rng_seed: u64,
    pub cuts: usize,
    pub top_k: usize,
    pub threads: usize,
    pub islands: usize,
    pub migration_interval: usize,
    pub cpu_ceiling_millicores: Option<u64>,
    pub source_commit: Option<String>,
    pub cancel: Option<Arc<AtomicBool>>,
    pub clock: Clock,
}

impl InventOptions {
    pub fn new(
        train: impl Into<PathBuf>,
        target: impl Into<String>,
        work_dir: impl Into<PathBuf>,
    ) -> Self {
        InventOptions {
            train: train.into(),
            target: target.into(),
            work_dir: work_dir.into(),
            budget: search::Budget::default(),
            seed: crate::DEFAULT_SEED,
            rng_seed: crate::DEFAULT_SEED,
            cuts: 3,
            top_k: 8,
            threads: 1,
            islands: 1,
            migration_interval: search::DEFAULT_MIGRATION_INTERVAL,
            cpu_ceiling_millicores: None,
            source_commit: None,
            cancel: None,
            clock: Clock::Wall,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Candidates {
    pub schema: &'static str,
    pub ok: bool,
    pub tool_version: &'static str,
    /// `--source-commit` when given; `git` is never spawned.
    pub git_sha: Option<String>,
    pub dataset_sha256: String,
    pub rows: usize,
    pub target: String,
    pub feature_names: Vec<String>,
    pub cuts: usize,
    pub seed: u64,
    pub rng_seed: u64,
    pub budget: BudgetRecord,
    pub vocabulary: VocabularyRecord,
    pub searches: Vec<CutRecord>,
    /// Programs measured on every cut, before the jain-math restriction.
    pub ranked: usize,
    /// Of those, programs with a stage jain-math cannot execute.
    pub excluded_not_executable: usize,
    pub elapsed_seconds: f64,
    pub candidates: Vec<Candidate>,
}

#[derive(Clone, Debug, Serialize)]
pub struct BudgetRecord {
    pub evaluations: usize,
    pub seconds: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct VocabularyRecord {
    pub engine: String,
    pub jain_math_rev: String,
    pub executable_ops: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct CutRecord {
    pub cut: usize,
    pub train_rows: usize,
    pub hold_rows: usize,
    pub completed_evaluations: u64,
    pub successful_measurements: u64,
    pub stopped_by: &'static str,
    pub seconds: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Candidate {
    pub rank: usize,
    /// Canonical pipeline DSL, e.g. `(pipe X (standardize) (ridge))`.
    pub genome: String,
    pub program_sha256: String,
    /// The search operator that first produced the program (`bootstrap`, `mutation`, ...).
    pub origin: String,
    pub mean_hold_rmse: f64,
    pub mean_hold_r2: f64,
    pub train_r2: f64,
    pub cuts_measured: usize,
    pub seconds: f64,
}

/// The op names jain-math executes, and the revision they were read from.
pub struct Executable {
    pub names: BTreeSet<String>,
    pub engine: String,
    pub rev: String,
}

impl Executable {
    pub fn load() -> Result<Self> {
        let doc: serde_json::Value =
            serde_json::from_str(JAIN_EXECUTABLE_OPS).context("parse the jain-math op snapshot")?;
        let names: BTreeSet<String> = doc["executable"]
            .as_array()
            .context("op snapshot: executable is not a list")?
            .iter()
            .filter_map(|op| op["name"].as_str().map(str::to_owned))
            .collect();
        ensure!(!names.is_empty(), "op snapshot lists no executable op");
        Ok(Executable {
            names,
            engine: doc["engine"].as_str().unwrap_or_default().to_owned(),
            rev: doc["provenance"]["rev"]
                .as_str()
                .unwrap_or_default()
                .to_owned(),
        })
    }

    /// A pipeline genome whose every stage op jain-math executes.
    pub fn admits(&self, genome: &str) -> bool {
        feat_math::dsl::parse_dsl(genome).is_ok_and(|expr| {
            expr.stages
                .iter()
                .all(|stage| self.names.contains(&stage.op))
        })
    }
}

/// The header CSV as `features..., target` rows, plus the feature names in column order.
pub fn load_csv(path: &Path, target: &str) -> Result<(Vec<String>, Vec<Vec<f64>>, String)> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let digest = sha256_hex(&bytes);
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(bytes.as_slice());
    let header: Vec<String> = reader
        .headers()
        .context("read the CSV header")?
        .iter()
        .map(str::to_owned)
        .collect();
    let target_index = header
        .iter()
        .position(|name| name == target)
        .with_context(|| format!("--target {target:?} is not a column of {}", path.display()))?;
    ensure!(
        header.len() >= 2,
        "{}: {} column(s); features and a target are required",
        path.display(),
        header.len()
    );
    let mut rows = Vec::new();
    for (line, record) in reader.records().enumerate() {
        let record = record.with_context(|| format!("CSV data row {}", line + 1))?;
        ensure!(
            record.len() == header.len(),
            "CSV data row {}: {} field(s), the header has {}",
            line + 1,
            record.len(),
            header.len()
        );
        let mut features = Vec::with_capacity(header.len());
        let mut value_of_target = 0.0;
        for (column, field) in record.iter().enumerate() {
            let value: f64 = field.trim().parse().with_context(|| {
                format!(
                    "CSV data row {}, column {:?}: {field:?} is not a number",
                    line + 1,
                    header[column]
                )
            })?;
            ensure!(
                value.is_finite(),
                "CSV data row {}, column {:?}: {value} is not finite",
                line + 1,
                header[column]
            );
            if column == target_index {
                value_of_target = value;
            } else {
                features.push(value);
            }
        }
        features.push(value_of_target);
        rows.push(features);
    }
    let features = header
        .into_iter()
        .enumerate()
        .filter(|&(column, _)| column != target_index)
        .map(|(_, name)| name)
        .collect();
    Ok((features, rows, digest))
}

/// Every refusal that needs no data and writes nothing; returns the worker count.
pub fn preflight(options: &InventOptions) -> Result<usize> {
    prepare::check_cuts(options.cuts)?;
    options.budget.check()?;
    ensure!(
        (1..=MAX_TOP_K).contains(&options.top_k),
        "--top-k must be 1..={MAX_TOP_K}, got {}",
        options.top_k
    );
    ensure!(
        options.islands <= 1 || options.migration_interval >= 1,
        "--migration-interval must be at least 1 with more than one island"
    );
    if let Some(commit) = &options.source_commit {
        ensure!(
            crate::run::is_commit(commit),
            "--source-commit {commit:?} is not a full commit id (40 or 64 lowercase hex digits)"
        );
    }
    crate::parallel::admit(
        options.threads,
        options.islands,
        options.cpu_ceiling_millicores,
    )
}

/// Search the CSV and return the ranked genomes jain-math can execute.
pub fn invent(options: &InventOptions) -> Result<Candidates> {
    preflight(options)?;
    let executable = Executable::load()?;
    let started = options.clock.start();

    let (feature_names, rows, dataset_sha256) = load_csv(&options.train, &options.target)?;
    // Refusals (too few rows in a cut, a constant held-out target) come before anything is written.
    let partition = prepare::partition(&rows, &dataset_sha256, options.seed, options.cuts)?;

    let search_root = options.work_dir.join("search");
    ensure!(
        std::fs::symlink_metadata(&search_root).is_err(),
        "{}: already holds a search; choose a new --work-dir",
        options.work_dir.display()
    );
    std::fs::create_dir_all(&options.work_dir)
        .with_context(|| format!("create {}", options.work_dir.display()))?;
    // The search rejects traversal and symlink components beneath this root.
    let root = options
        .work_dir
        .canonicalize()
        .with_context(|| format!("resolve {}", options.work_dir.display()))?;
    let preparation = partition.write(&root.join("prepared"))?;

    let search_options = search::SearchOptions {
        threads: options.threads,
        islands: options.islands,
        migration_interval: options.migration_interval,
        clock: options.clock,
        cancel: options.cancel.clone(),
        ..search::SearchOptions::new(options.budget, options.rng_seed)
    };
    let searched = search::run_cuts_with(&preparation.cuts, &search_options, &root.join("search"))?;
    if options
        .cancel
        .as_ref()
        .is_some_and(|flag| flag.load(Ordering::SeqCst))
    {
        return Err(crate::parallel::Cancelled.into());
    }

    let per_cut: Vec<(usize, &[search::Observation])> = searched
        .iter()
        .map(|c| (c.cut, c.observations.as_slice()))
        .collect();
    let ranking = select::ranking(&per_cut)?;
    // The operator that produced each program, in search order. The comparison phase re-measures
    // programs other cuts produced, so its label stands only when nothing else measured them.
    let mut origin: BTreeMap<&str, &'static str> = BTreeMap::new();
    for search in &searched {
        for observation in &search.observations {
            let entry = origin
                .entry(observation.genome.as_str())
                .or_insert(observation.origin);
            if *entry == search::REMEASURE {
                *entry = observation.origin;
            }
        }
    }
    let ranked = ranking.len();
    let admitted: Vec<&select::Best> = ranking
        .iter()
        .filter(|best| executable.admits(&best.genome))
        .collect();
    let excluded_not_executable = ranked - admitted.len();
    let candidates = admitted
        .into_iter()
        .take(options.top_k)
        .enumerate()
        .map(|(index, best)| Candidate {
            rank: index + 1,
            genome: best.genome.clone(),
            program_sha256: best.program_sha256.clone(),
            origin: origin
                .get(best.genome.as_str())
                .copied()
                .unwrap_or("unknown")
                .to_owned(),
            mean_hold_rmse: best.validation_rmse,
            mean_hold_r2: best.validation_r2,
            train_r2: best.train_r2,
            cuts_measured: best.cuts_measured,
            seconds: best.seconds,
        })
        .collect::<Vec<_>>();

    Ok(Candidates {
        schema: SCHEMA,
        ok: !candidates.is_empty(),
        tool_version: env!("CARGO_PKG_VERSION"),
        git_sha: options.source_commit.clone(),
        dataset_sha256,
        rows: rows.len(),
        target: options.target.clone(),
        feature_names,
        cuts: options.cuts,
        seed: options.seed,
        rng_seed: options.rng_seed,
        budget: BudgetRecord {
            evaluations: options.budget.evaluations,
            seconds: options.budget.seconds,
        },
        vocabulary: VocabularyRecord {
            engine: executable.engine,
            jain_math_rev: executable.rev,
            executable_ops: executable.names.len(),
        },
        searches: searched
            .iter()
            .map(|c| CutRecord {
                cut: c.cut,
                train_rows: c.train_rows,
                hold_rows: c.hold_rows,
                completed_evaluations: c.completed_evaluations,
                successful_measurements: c.successful_measurements,
                stopped_by: c.stopped_by,
                seconds: c.seconds,
            })
            .collect(),
        ranked,
        excluded_not_executable,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        candidates,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_csv(dir: &Path, rows: usize) -> PathBuf {
        let path = dir.join("train.csv");
        let mut text = String::from("a,y,b\n");
        for i in 0..rows {
            let a = i as f64 * 0.37 % 5.0;
            let b = (i as f64 * 1.3).sin();
            let y = 2.0 * a - 3.0 * b + 0.01 * (i % 7) as f64;
            text.push_str(&format!("{a},{y},{b}\n"));
        }
        std::fs::write(&path, text).unwrap();
        path
    }

    #[test]
    fn csv_moves_the_named_target_last_and_keeps_feature_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(dir.path(), 4);
        let (features, rows, digest) = load_csv(&path, "y").unwrap();
        assert_eq!(features, ["a", "b"]);
        assert_eq!(rows.len(), 4);
        let (a, b) = (rows[1][0], rows[1][1]);
        assert!((rows[1][2] - (2.0 * a - 3.0 * b + 0.01)).abs() < 1e-12);
        assert_eq!(digest.len(), 64);
    }

    #[test]
    fn csv_refuses_missing_target_text_and_non_finite_values() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(dir.path(), 4);
        let error = load_csv(&path, "z").unwrap_err().to_string();
        assert!(error.contains("--target \"z\""), "{error}");
        for (body, needle) in [
            ("a,y\n1,NA\n", "not a number"),
            ("a,y\n1,inf\n", "not finite"),
            ("a,y\n1\n", "field"),
        ] {
            let bad = dir.path().join("bad.csv");
            std::fs::write(&bad, body).unwrap();
            let error = format!("{:#}", load_csv(&bad, "y").unwrap_err());
            assert!(error.contains(needle), "{body:?}: {error}");
        }
    }

    #[test]
    fn only_jain_executable_pipelines_are_admitted() {
        let executable = Executable::load().unwrap();
        assert_eq!(executable.engine, "featmath");
        assert_eq!(executable.rev.len(), 40);
        assert!(executable.admits("(pipe X (standardize) (ridge))"));
        // A search head jain-math 9fa594e does not implement.
        assert!(!executable.admits("(pipe X (standardize) (ridge_gcv))"));
        // Quarantined in jain-math.
        assert!(!executable.admits("(pipe X (rff_orthogonal) (ridge))"));
        assert!(!executable.admits("not a genome"));
    }

    #[test]
    fn preflight_refuses_bad_counts_before_reading_anything() {
        let mut options = InventOptions::new("/nonexistent.csv", "y", "/nonexistent");
        options.top_k = 0;
        assert!(preflight(&options)
            .unwrap_err()
            .to_string()
            .contains("--top-k"));
        options.top_k = 8;
        options.cuts = prepare::FOLDS + 1;
        assert!(preflight(&options).is_err());
        options.cuts = 3;
        options.source_commit = Some("abc".into());
        assert!(preflight(&options)
            .unwrap_err()
            .to_string()
            .contains("--source-commit"));
    }

    #[test]
    fn invent_ranks_executable_genomes_and_is_deterministic_under_an_evaluation_budget() {
        let dir = tempfile::tempdir().unwrap();
        let train = write_csv(dir.path(), 120);
        let run = |work: &str| {
            let mut options = InventOptions::new(&train, "y", dir.path().join(work));
            options.budget = search::Budget {
                evaluations: 24,
                seconds: 600,
            };
            options.cuts = 2;
            options.top_k = 4;
            invent(&options).unwrap()
        };
        let first = run("one");
        assert!(first.ok, "{first:?}");
        assert_eq!(first.schema, SCHEMA);
        assert_eq!(first.rows, 120);
        assert_eq!(first.feature_names, ["a", "b"]);
        assert!(!first.candidates.is_empty() && first.candidates.len() <= 4);
        let executable = Executable::load().unwrap();
        for (index, candidate) in first.candidates.iter().enumerate() {
            assert_eq!(candidate.rank, index + 1);
            assert_ne!(candidate.origin, search::REMEASURE, "{}", candidate.genome);
            assert!(executable.admits(&candidate.genome), "{}", candidate.genome);
            assert_eq!(candidate.cuts_measured, 2);
            assert!(candidate.mean_hold_rmse.is_finite());
        }
        assert!(first
            .candidates
            .windows(2)
            .all(|w| w[0].mean_hold_rmse <= w[1].mean_hold_rmse));
        assert!(first.ranked - first.excluded_not_executable >= first.candidates.len());
        let second = run("two");
        let genomes = |c: &Candidates| {
            c.candidates
                .iter()
                .map(|c| c.genome.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(genomes(&first), genomes(&second));
        // A used work directory is refused.
        let mut again = InventOptions::new(&train, "y", dir.path().join("one"));
        again.budget = search::Budget {
            evaluations: 24,
            seconds: 600,
        };
        assert!(invent(&again)
            .unwrap_err()
            .to_string()
            .contains("--work-dir"));
    }
}
