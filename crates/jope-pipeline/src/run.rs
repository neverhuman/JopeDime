//! The run: `dataset -> prepare -> (propose) -> search -> select -> emit -> receipt`.
//!
//! Every stage is timed and recorded. The propose stage — encode the dataset and
//! let the policy propose a seed population — is not built in this work package;
//! it is recorded as a skip with its reason and printed, and `--policy require`
//! fails closed rather than letting a caller believe it ran. The S3 publication
//! and router-training stages are likewise out of scope and are not pretended to.

use anyhow::{bail, ensure, Context, Result};
use jope_datasets::paths::now_rfc3339;
use std::{path::PathBuf, time::Instant};

use crate::{
    dataset, emit,
    emit::Language,
    prepare, receipt,
    receipt::Receipt,
    search::{self, Budget},
    select, sha256_hex, TOOL_VERSION,
};

/// What to do about the proposal stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    /// Use it when it is available; on this SHA it is not, so it is skipped.
    Auto,
    /// Explicitly run without it.
    Skip,
    /// Refuse to run without it.
    Require,
}

impl Policy {
    pub fn as_str(self) -> &'static str {
        match self {
            Policy::Auto => "auto",
            Policy::Skip => "skip",
            Policy::Require => "require",
        }
    }

    pub fn parse(value: &str) -> Option<Policy> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Policy::Auto),
            "skip" => Some(Policy::Skip),
            "require" => Some(Policy::Require),
            _ => None,
        }
    }
}

/// Why the propose stage is not run here.
const PROPOSE_SKIP_REASON: &str = "the encode/propose stage is not implemented in this work \
     package: no policy checkpoint is bound and no seed population is proposed, so the search \
     starts from its own bootstrap population";

#[derive(Clone, Debug)]
pub struct RunOptions {
    /// A 16-hex catalog id or a dataset directory.
    pub dataset: String,
    pub output: PathBuf,
    pub budget: Budget,
    /// Fold seed for the cut preparation.
    pub seed: u64,
    /// Search random-stream seed.
    pub rng_seed: u64,
    pub cuts: usize,
    pub languages: Vec<Language>,
    pub policy: Policy,
}

struct Stages {
    stages: Vec<receipt::Stage>,
}

impl Stages {
    fn new() -> Self {
        Stages { stages: Vec::new() }
    }

    fn record(&mut self, name: &str, status: &str, reason: Option<String>, started: Instant) {
        self.stages.push(receipt::Stage {
            name: name.to_owned(),
            status: status.to_owned(),
            reason,
            seconds: started.elapsed().as_secs_f64(),
        });
    }
}

/// Run the pipeline and write `<output>/receipt.json`.
pub fn run(options: &RunOptions) -> Result<Receipt> {
    ensure!(options.cuts > 0, "--cuts must be at least 1");
    ensure!(!options.languages.is_empty(), "--emit needs a target");
    options.budget.check()?;
    // Fail closed before anything is written: a refused run must not leave a half
    // prepared output directory behind to block the retry.
    ensure!(
        options.policy != Policy::Require,
        "--policy require: {PROPOSE_SKIP_REASON}. Re-run with --policy skip to proceed \
         without it."
    );

    let output = &options.output;
    for existing in ["receipt.json", "prepared", "search", "emit"] {
        ensure!(
            !output.join(existing).exists(),
            "{}: already holds a pipeline run ({existing}); choose a new --output",
            output.display()
        );
    }
    let mut stages = Stages::new();

    let started = Instant::now();
    let resolved = dataset::resolve(&options.dataset)?;
    ensure!(
        resolved.row.task == jope_datasets::catalog::Task::Regression,
        "task {} is unsupported by jope pipeline run: this path fits and scores regression only; \
         use a task-qualified evaluator when available",
        resolved.row.task.as_str()
    );
    let rows = dataset::training_rows(&resolved)?;
    stages.record("dataset", "passed", None, started);

    // Validate the task and inputs before publishing any run artifacts.
    std::fs::create_dir_all(output).with_context(|| format!("create {}", output.display()))?;
    // The search rejects traversal and symlink components beneath this root.
    let root = output
        .canonicalize()
        .with_context(|| format!("resolve {}", output.display()))?;

    let started = Instant::now();
    let prepared_dir = root.join("prepared");
    let preparation = prepare::build(
        &rows,
        &resolved.row.dataset_hash,
        options.seed,
        options.cuts,
        &prepared_dir,
    )?;
    stages.record("prepare", "passed", None, started);

    let started = Instant::now();
    // Never let a skipped stage pass unseen.
    eprintln!("jope pipeline: propose stage SKIPPED - {PROPOSE_SKIP_REASON}");
    stages.record(
        "propose",
        "skipped",
        Some(PROPOSE_SKIP_REASON.to_owned()),
        started,
    );

    let started = Instant::now();
    let searched = search::run_cuts(
        &preparation.cuts,
        options.budget,
        options.rng_seed,
        &root.join("search"),
    )?;
    let successful: u64 = searched.iter().map(|c| c.successful_measurements).sum();
    let failed: u64 = searched
        .iter()
        .map(|c| c.failed_or_unavailable_measurements)
        .sum();
    ensure!(
        successful > 0,
        "the search published no measurement across {} cut(s)",
        searched.len()
    );
    // A pass is still worth a reason when candidates were lost along the way.
    let search_reason = (failed > 0).then(|| {
        format!(
            "{failed} of {} candidate(s) did not measure and were skipped; \
             per-candidate reasons are in each cut's observations.json",
            successful + failed
        )
    });
    if let Some(reason) = &search_reason {
        eprintln!("jope pipeline: search - {reason}");
    }
    stages.record("search", "passed", search_reason, started);

    let started = Instant::now();
    let selection = select::select(&searched)?;
    if let Some(reason) = &selection.reason {
        eprintln!("jope pipeline: select - {reason}");
    }
    stages.record("select", "passed", selection.reason.clone(), started);

    let started = Instant::now();
    let emit_dir = root.join("emit");
    let emitted = emit::emit(&selection.genome, &options.languages, &emit_dir)?;
    // Both fields below are read back out of emit/manifest.json, which transpile builds from the
    // IR it actually lowered -- so these compare the artifacts on disk against the selection.
    // The previous pair derived both sides from emit()'s own argument, which made them hold even
    // when the transpiler emitted a different genome entirely:
    //   ensure!(emitted.genome_sha256 == sha256(emitted.canonical_genome))  is  hash(x) == hash(x)
    //   ensure!(emitted.canonical_genome == selection.genome)               is  parse(g).canonical() == g
    // The second only tested that the selected genome was already canonical, which every genome
    // reaching this point is.
    ensure!(
        emitted.canonical_genome == selection.genome,
        "the emitted manifest describes {}, not the selected genome {}",
        emitted.canonical_genome,
        selection.genome
    );
    ensure!(
        emitted.genome_sha256 == crate::sha256_hex(selection.genome.as_bytes()),
        "the emitted manifest's genome_sha256 {} is not the digest of the selected genome {}",
        emitted.genome_sha256,
        selection.genome
    );
    ensure!(
        emitted.genome_sha256 == selection.program_sha256,
        "the emitted genome is not the selected one"
    );
    for exclusion in &emitted.exclusions {
        eprintln!(
            "jope pipeline: emit - {} not emitted: {}",
            exclusion.language, exclusion.reason
        );
    }
    let emit_reason = (!emitted.exclusions.is_empty()).then(|| {
        emitted
            .exclusions
            .iter()
            .map(|e| format!("{}: {}", e.language, e.reason))
            .collect::<Vec<_>>()
            .join("; ")
    });
    // Every requested language excluded means the run produced no code at all. Recording that as
    // `passed` because the emitter "handled" each one is the false green this pipeline exists to
    // avoid: the caller asked for working source and got none, so the stage failed and so did the
    // run. A partial emission is a pass with its exclusions recorded, which is a different thing.
    let all_excluded = emitted.artifacts.is_empty();
    let emit_status = if all_excluded { "failed" } else { "passed" };
    stages.record("emit", emit_status, emit_reason, started);

    let started = Instant::now();
    stages.record("receipt", "passed", None, started);
    let mut receipt = assemble(
        options,
        &resolved,
        &preparation,
        &searched,
        &selection,
        &emitted,
        stages.stages,
    )?;
    // A receipt that does not satisfy its own contract is a failed run.
    receipt::validate_value(&serde_json::to_value(&receipt)?)
        .map_err(|errors| anyhow::anyhow!("receipt is invalid: {}", errors.join("; ")))?;
    if let Some(stage) = receipt.stages.last_mut() {
        stage.seconds = started.elapsed().as_secs_f64();
    }
    let path = root.join("receipt.json");
    std::fs::write(&path, serde_json::to_vec_pretty(&receipt)?)
        .with_context(|| format!("write {}", path.display()))?;
    // Written first, so the receipt that records the failure survives the failure.
    if all_excluded {
        bail!(
            "no code was emitted: every requested language was excluded ({}); receipt in {}",
            receipt
                .emit_exclusions
                .iter()
                .map(|e| format!("{}: {}", e.language, e.reason))
                .collect::<Vec<_>>()
                .join("; "),
            root.display()
        );
    }
    Ok(receipt)
}

#[allow(clippy::too_many_arguments)]
fn assemble(
    options: &RunOptions,
    resolved: &dataset::Resolved,
    preparation: &prepare::Preparation,
    searched: &[search::CutSearch],
    selection: &select::Selection,
    emitted: &emit::Emitted,
    stages: Vec<receipt::Stage>,
) -> Result<Receipt> {
    let mut observation_digests = String::new();
    let mut cuts = Vec::new();
    for cut in searched {
        observation_digests.push_str(&cut.observations_sha256);
        cuts.push(receipt::CutSearch {
            cut: cut.cut as u64,
            search_sha256: cut.search_sha256.clone(),
            observations_sha256: cut.observations_sha256.clone(),
            train_rows: cut.train_rows as u64,
            hold_rows: cut.hold_rows as u64,
            generations: cut.generations,
            completed_evaluations: cut.completed_evaluations,
            successful_measurements: cut.successful_measurements,
            failed_or_unavailable_measurements: cut.failed_or_unavailable_measurements,
            stopped_by: cut.stopped_by.to_owned(),
            accounting: Some(receipt::Accounting {
                exploration: cut.exploration.clone(),
                comparison: cut.comparison.clone(),
                active_seconds: cut.seconds,
                deadline_overrun: cut.seconds > cut.budget.seconds as f64,
            }),
            best_validation_rmse: cut.best_validation_rmse,
            best_validation_r2: cut.best_validation_r2,
            best_program: cut.best_program.clone(),
        });
    }
    Ok(Receipt {
        schema: receipt::RECEIPT_SCHEMA_ID.to_owned(),
        ok: stages.iter().all(|s| s.status != "failed"),
        tool_version: TOOL_VERSION.to_owned(),
        git_sha: git_sha(),
        created_at: now_rfc3339(),
        policy: options.policy.as_str().to_owned(),
        dataset: receipt::Dataset {
            id: resolved.row.id.clone(),
            dataset_hash: resolved.row.dataset_hash.clone(),
            path: options.dataset.clone(),
            task: resolved.row.task.as_str().to_owned(),
            rows: resolved.row.shape.rows,
            columns: resolved.row.shape.columns,
            training_rows: resolved.row.split.train_rows,
            files: resolved
                .files
                .iter()
                .map(|f| receipt::DatasetFile {
                    name: f.name.clone(),
                    sha256: f.sha256.clone(),
                    bytes: f.bytes,
                })
                .collect(),
        },
        prepared: receipt::Prepared {
            sha256: preparation.sha256.clone(),
            seed: preparation.seed,
            rows: preparation.rows as u64,
            folds: prepare::FOLDS as u64,
            cuts: preparation.cuts.iter().map(|c| c.index as u64).collect(),
        },
        stages,
        search: receipt::Search {
            engine: search::SEARCH_ENGINE.to_owned(),
            rng_seed: options.rng_seed,
            evaluations: options.budget.evaluations as u64,
            seconds: options.budget.seconds,
            population: search::POPULATION as u64,
            max_transforms: search::MAX_TRANSFORMS as u64,
            bootstrap_genome: search::BOOTSTRAP_GENOME.to_owned(),
            cuts,
            selection: selection.method.to_owned(),
            budget_policy: Some(search::BUDGET_POLICY.to_owned()),
            compared_cuts: Some(selection.compared_cuts.clone()),
            coverage: Some(selection.coverage.clone()),
            selected_genome: selection.genome.clone(),
            selected_program_sha256: selection.program_sha256.clone(),
            best: selection.best.as_ref().map(|b| receipt::Best {
                genome: b.genome.clone(),
                program_sha256: b.program_sha256.clone(),
                validation_rmse: b.validation_rmse,
                validation_r2: b.validation_r2,
                train_r2: b.train_r2,
                seconds: b.seconds,
                cuts_measured: b.cuts_measured as u64,
            }),
        },
        artifacts: emitted
            .artifacts
            .iter()
            .map(|a| receipt::Artifact {
                language: a.language.to_owned(),
                path: a.path.clone(),
                sha256: a.sha256.clone(),
                bytes: a.bytes,
                build: a.build.clone(),
            })
            .collect(),
        emit_exclusions: emitted
            .exclusions
            .iter()
            .map(|e| receipt::Exclusion {
                language: e.language.to_owned(),
                reason: e.reason.clone(),
            })
            .collect(),
        observations_sha256: sha256_hex(observation_digests.as_bytes()),
    })
}

/// `JOPE_GIT_SHA`, else `git rev-parse HEAD` in the working directory, else none.
fn git_sha() -> Option<String> {
    if let Some(sha) = std::env::var("JOPE_GIT_SHA").ok().filter(|s| !s.is_empty()) {
        return Some(sha);
    }
    let output = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .filter(|sha| sha.len() == 40 && sha.bytes().all(|b| b.is_ascii_hexdigit()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification_is_refused_before_writing_a_regression_run() {
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("binary");
        std::fs::create_dir(&input).unwrap();
        // An admitted binary catalog and genuine 0/1 labels must never reach
        // this regression-only search, even though both parse as numeric CSV.
        let csv: String = (0..60).map(|i| format!("{},{}\n", i, i % 2)).collect();
        let mut row: serde_json::Value = serde_json::from_str(include_str!(
            "../../../datasets/test/pmlb-229-pwlinear/metadata.json"
        ))
        .unwrap();
        row["task"] = "binary".into();
        row["target"]["type"] = "binary".into();
        row["shape"] = serde_json::json!({
            "rows":120,"features":1,"categorical_features":0,"columns":2
        });
        row["split"]["train_rows"] = 60.into();
        row["split"]["test_rows"] = 60.into();
        for name in ["train", "test"] {
            row["files"][name] = serde_json::json!({
                "sha256": crate::sha256_hex(csv.as_bytes()), "bytes": csv.len(),
                "rows": 60, "columns": 2, "has_header": false
            });
            std::fs::write(input.join(format!("{name}.csv")), &csv).unwrap();
        }
        std::fs::write(input.join("metadata.json"), row.to_string()).unwrap();
        let output = temp.path().join("run");
        let options = RunOptions {
            dataset: input.display().to_string(),
            output: output.clone(),
            budget: Budget {
                evaluations: 1,
                seconds: 1,
            },
            seed: 1,
            rng_seed: 1,
            cuts: 1,
            languages: vec![Language::Rust],
            policy: Policy::Skip,
        };
        let error = format!("{:#}", run(&options).unwrap_err());
        assert!(error.contains("task binary is unsupported"), "{error}");
        assert!(!output.exists(), "refusal left a misleading run directory");
    }

    /// The dataset is unresolvable, so this also proves the policy is checked
    /// before any other work — including anything that writes to `--output`.
    #[test]
    fn require_fails_closed_before_the_output_directory_is_touched() {
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("run");
        let options = RunOptions {
            dataset: "0000000000000000".to_owned(),
            output: output.clone(),
            budget: Budget::default(),
            seed: 1,
            rng_seed: 1,
            cuts: 1,
            languages: vec![Language::Rust],
            policy: Policy::Require,
        };
        let error = format!("{:#}", run(&options).unwrap_err());
        assert!(error.contains("--policy require"), "{error}");
        assert!(error.contains("--policy skip"), "{error}");
        assert!(!output.exists());
    }

    #[test]
    fn policies_parse() {
        assert_eq!(Policy::parse("auto"), Some(Policy::Auto));
        assert_eq!(Policy::parse(" SKIP "), Some(Policy::Skip));
        assert_eq!(Policy::parse("require"), Some(Policy::Require));
        assert_eq!(Policy::parse("maybe"), None);
        for policy in [Policy::Auto, Policy::Skip, Policy::Require] {
            assert_eq!(Policy::parse(policy.as_str()), Some(policy));
        }
    }
}
