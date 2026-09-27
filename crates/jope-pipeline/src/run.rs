//! The run: `dataset -> prepare -> (propose) -> search -> select -> emit -> receipt`.
//!
//! Every stage is timed and recorded. The propose stage — encode the dataset and
//! let the policy propose a seed population — is not built in this work package;
//! it is recorded as a skip with its reason and printed, and `--policy require`
//! fails closed rather than letting a caller believe it ran. The S3 publication
//! and router-training stages are likewise out of scope and are not pretended to.

use anyhow::{bail, ensure, Context, Result};
use std::path::PathBuf;

use crate::{
    clock::{Clock, Stamp},
    dataset, emit,
    emit::Language,
    prepare, receipt,
    receipt::Receipt,
    search::{self, Budget},
    seeds, select, sha256_hex, TOOL_VERSION,
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
    /// A `jope.pipeline-seeds/v1` file whose genomes enter generation 0 after the bootstrap.
    pub seed_genomes: Option<PathBuf>,
    /// A parent receipt: re-derived from its run's evidence, then its selected genome is seeded
    /// first and its sha256 recorded as this run's parent.
    pub incumbent: Option<PathBuf>,
    /// Worker threads evaluating each generation. Never changes a decision or a published byte
    /// other than wall-clock readings; not recorded in the receipt for that reason.
    pub threads: usize,
    /// An explicit CPU admission in millicores that the worker count must fit.
    pub cpu_ceiling_millicores: Option<u64>,
    pub clock: Clock,
}

impl RunOptions {
    /// Defaults matching `jope pipeline run` with only `--dataset` and `--output` given.
    pub fn new(dataset: impl Into<String>, output: impl Into<PathBuf>) -> Self {
        RunOptions {
            dataset: dataset.into(),
            output: output.into(),
            budget: Budget::default(),
            seed: crate::DEFAULT_SEED,
            rng_seed: crate::DEFAULT_SEED,
            cuts: 1,
            languages: Language::parse_list("rust,python,cpp17,cpp20,js")
                .expect("the default emit list parses"),
            policy: Policy::Auto,
            seed_genomes: None,
            incumbent: None,
            threads: 1,
            cpu_ceiling_millicores: None,
            clock: Clock::Wall,
        }
    }
}

/// Every refusal that needs no dataset and writes nothing: the counts, the budget, the policy
/// and the parallelism this host admits. Returns the worker count.
pub fn preflight(options: &RunOptions) -> Result<usize> {
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
    crate::parallel::admit(options.threads, 1, options.cpu_ceiling_millicores)
}

/// Seeds resolved from `--incumbent` and `--seed-genomes`, before anything is written.
fn resolve_seeding(options: &RunOptions) -> Result<(Vec<String>, receipt::Seeding)> {
    let mut offered: Vec<(String, &'static str)> = Vec::new();
    let mut parent_receipt_sha256 = None;
    let mut seeds_file_sha256 = None;
    if let Some(path) = &options.incumbent {
        let parent = crate::evidence::verify_champion_run(path)
            .with_context(|| format!("--incumbent {}", path.display()))?;
        offered.push((
            parent.receipt.search.selected_genome.clone(),
            seeds::FROM_INCUMBENT,
        ));
        parent_receipt_sha256 = Some(parent.receipt_sha256);
    }
    if let Some(path) = &options.seed_genomes {
        let (file, sha) =
            seeds::read(path).with_context(|| format!("--seed-genomes {}", path.display()))?;
        offered.extend(
            file.genomes
                .into_iter()
                .map(|g| (g.genome, seeds::FROM_SEEDS_FILE)),
        );
        seeds_file_sha256 = Some(sha);
    }
    let admitted = seeds::admit(&offered)?;
    let limit = search::exploration_limit(options.budget, options.cuts);
    ensure!(
        admitted.seeds.len() < limit || admitted.seeds.is_empty(),
        "the budget cannot re-measure {} seed(s): generation 0 needs the bootstrap plus every seed \
         on each cut, {} evaluation(s), and exploration allows {limit} per cut",
        admitted.seeds.len(),
        admitted.seeds.len() + 1
    );
    let record = receipt::Seeding {
        offered: admitted.offered as u64,
        seeds: admitted
            .seeds
            .iter()
            .map(|s| receipt::Seed {
                genome: s.genome.clone(),
                program_sha256: s.program_sha256.clone(),
                source: s.source.to_owned(),
            })
            .collect(),
        parent_receipt_sha256,
        seeds_file_sha256,
    };
    Ok((
        admitted.seeds.into_iter().map(|s| s.genome).collect(),
        record,
    ))
}

struct Stages {
    stages: Vec<receipt::Stage>,
}

impl Stages {
    fn new() -> Self {
        Stages { stages: Vec::new() }
    }

    fn record(&mut self, name: &str, status: &str, reason: Option<String>, started: Stamp) {
        self.stages.push(receipt::Stage {
            name: name.to_owned(),
            status: status.to_owned(),
            reason,
            seconds: started.secs(),
        });
    }
}

/// Run the pipeline and write `<output>/receipt.json`.
pub fn run(options: &RunOptions) -> Result<Receipt> {
    preflight(options)?;
    let clock = options.clock;

    let output = &options.output;
    for existing in ["receipt.json", "prepared", "search", "emit"] {
        ensure!(
            !output.join(existing).exists(),
            "{}: already holds a pipeline run ({existing}); choose a new --output",
            output.display()
        );
    }
    // A seed that does not parse, a seeds file that does not verify or a parent receipt its own
    // evidence does not re-derive is refused here, before anything is written.
    let (seed_genomes, seeding) = resolve_seeding(options)?;
    let mut stages = Stages::new();

    let started = clock.start();
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

    let started = clock.start();
    let prepared_dir = root.join("prepared");
    let preparation = prepare::build(
        &rows,
        &resolved.row.dataset_hash,
        options.seed,
        options.cuts,
        &prepared_dir,
    )?;
    stages.record("prepare", "passed", None, started);

    let started = clock.start();
    // Never let a skipped stage pass unseen.
    eprintln!("jope pipeline: propose stage SKIPPED - {PROPOSE_SKIP_REASON}");
    stages.record(
        "propose",
        "skipped",
        Some(PROPOSE_SKIP_REASON.to_owned()),
        started,
    );

    let started = clock.start();
    let search_options = search::SearchOptions {
        seeds: seed_genomes,
        threads: options.threads,
        clock,
        ..search::SearchOptions::new(options.budget, options.rng_seed)
    };
    let searched = search::run_cuts_with(&preparation.cuts, &search_options, &root.join("search"))?;
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

    let started = clock.start();
    let selection = select::select(&searched)?;
    if let Some(reason) = &selection.reason {
        eprintln!("jope pipeline: select - {reason}");
    }
    stages.record("select", "passed", selection.reason.clone(), started);

    let started = clock.start();
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

    let started = clock.start();
    stages.record("receipt", "passed", None, started);
    let mut receipt = assemble(
        options,
        &resolved,
        &preparation,
        &searched,
        &selection,
        &emitted,
        stages.stages,
        seeding,
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
    seeding: receipt::Seeding,
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
        created_at: options.clock.now_rfc3339(),
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
            seeding: Some(seeding),
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
            budget: Budget {
                evaluations: 1,
                seconds: 1,
            },
            seed: 1,
            rng_seed: 1,
            languages: vec![Language::Rust],
            policy: Policy::Skip,
            ..RunOptions::new(input.display().to_string(), output.clone())
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
            seed: 1,
            rng_seed: 1,
            languages: vec![Language::Rust],
            policy: Policy::Require,
            ..RunOptions::new("0000000000000000", output.clone())
        };
        let error = format!("{:#}", run(&options).unwrap_err());
        assert!(error.contains("--policy require"), "{error}");
        assert!(error.contains("--policy skip"), "{error}");
        assert!(!output.exists());
    }

    fn committed_dataset() -> String {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../datasets/test/pmlb-229-pwlinear"
        )
        .to_owned()
    }

    fn small(output: &std::path::Path, evaluations: usize) -> RunOptions {
        RunOptions {
            budget: Budget {
                evaluations,
                seconds: 300,
            },
            rng_seed: 7,
            languages: vec![Language::Rust],
            policy: Policy::Skip,
            ..RunOptions::new(committed_dataset(), output)
        }
    }

    /// What a run decided, without the wall clock: genome, generation, origin, status, scores.
    type Decided = Vec<(
        usize,
        String,
        usize,
        &'static str,
        &'static str,
        Option<f64>,
        Option<f64>,
    )>;

    fn decided(run: &crate::evidence::VerifiedRun) -> Decided {
        run.cuts
            .iter()
            .flat_map(|(cut, observations)| {
                observations.iter().map(move |o| {
                    (
                        *cut,
                        o.genome.clone(),
                        o.generation,
                        o.origin,
                        o.status,
                        o.rmse,
                        o.hold_r2,
                    )
                })
            })
            .collect()
    }

    fn seeds_file(dir: &std::path::Path, genomes: &[&str]) -> std::path::PathBuf {
        let path = dir.join("seeds.json");
        let file = seeds::SeedsFile::new(
            None,
            genomes
                .iter()
                .map(|g| seeds::SeedGenome {
                    genome: (*g).to_owned(),
                    program_sha256: sha256_hex(g.as_bytes()),
                    bank_entry_sha256: None,
                    receipt_sha256: None,
                })
                .collect(),
        );
        seeds::write(&path, &file).unwrap();
        path
    }

    #[test]
    fn a_seed_enters_generation_zero_after_the_bootstrap_as_an_incumbent() {
        let temp = tempfile::tempdir().unwrap();
        let seed = seeds::canonical_pipeline("(pipe X (standardize) (ridge_gcv))").unwrap();
        let file = seeds_file(temp.path(), &[&seed]);
        let output = temp.path().join("run");
        let receipt = run(&RunOptions {
            seed_genomes: Some(file.clone()),
            ..small(&output, 14)
        })
        .unwrap();
        assert_eq!(receipt.schema, receipt::RECEIPT_SCHEMA_ID);
        let seeding = receipt.search.seeding.as_ref().unwrap();
        assert_eq!(seeding.offered, 1);
        assert_eq!(seeding.seeds.len(), 1);
        assert_eq!(seeding.seeds[0].genome, seed);
        assert_eq!(seeding.seeds[0].program_sha256, sha256_hex(seed.as_bytes()));
        assert_eq!(seeding.seeds[0].source, seeds::FROM_SEEDS_FILE);
        assert_eq!(
            seeding.seeds_file_sha256.as_deref(),
            Some(crate::sha256_file(&file).unwrap().as_str())
        );
        assert!(seeding.parent_receipt_sha256.is_none());

        let verified = crate::evidence::verify_run(&output.join("receipt.json")).unwrap();
        let observations = &verified.cuts[0].1;
        assert_eq!(observations[0].origin, search::BOOTSTRAP);
        assert_eq!(observations[1].genome, seed);
        assert_eq!(observations[1].origin, search::INCUMBENT);
        assert_eq!(observations[1].generation, 0);
        // The seed spent budget like any candidate: the evaluation bound still holds.
        assert!(receipt.search.cuts[0].completed_evaluations <= 14);
    }

    #[test]
    fn the_same_seed_and_rng_replay_identically() {
        let temp = tempfile::tempdir().unwrap();
        let seed = seeds::canonical_pipeline("(pipe X (robust_scale) (ridge))").unwrap();
        let file = seeds_file(temp.path(), &[&seed]);
        let mut replays = Vec::new();
        for name in ["a", "b"] {
            let output = temp.path().join(name);
            run(&RunOptions {
                seed_genomes: Some(file.clone()),
                ..small(&output, 16)
            })
            .unwrap();
            replays.push(crate::evidence::verify_run(&output.join("receipt.json")).unwrap());
        }
        assert_eq!(decided(&replays[0]), decided(&replays[1]));
        assert_eq!(
            replays[0].receipt.search.selected_genome,
            replays[1].receipt.search.selected_genome
        );
        assert_eq!(
            replays[0].receipt.search.seeding,
            replays[1].receipt.search.seeding
        );
    }

    /// Seeding from a parent re-measures the parent's champion on this run's cuts, and the
    /// selection over those measurements can only match or beat it.
    #[test]
    fn a_champion_is_never_worse_than_its_seed_on_the_same_cuts() {
        let temp = tempfile::tempdir().unwrap();
        let parent_dir = temp.path().join("parent");
        let parent = run(&RunOptions {
            cuts: 2,
            ..small(&parent_dir, 24)
        })
        .unwrap();
        let parent_best = parent.search.best.clone().unwrap();
        let child_dir = temp.path().join("child");
        let child = run(&RunOptions {
            cuts: 2,
            rng_seed: 8,
            incumbent: Some(parent_dir.join("receipt.json")),
            ..small(&child_dir, 24)
        })
        .unwrap();
        let seeding = child.search.seeding.clone().unwrap();
        assert_eq!(
            seeding.parent_receipt_sha256.as_deref(),
            Some(
                crate::sha256_file(&parent_dir.join("receipt.json"))
                    .unwrap()
                    .as_str()
            )
        );
        let child_best = child.search.best.clone().unwrap();
        assert!(
            child_best.validation_rmse <= parent_best.validation_rmse,
            "child {} ({}) is worse than its seed {} ({})",
            child_best.genome,
            child_best.validation_rmse,
            parent_best.genome,
            parent_best.validation_rmse
        );
        // Same cuts, same fit: the re-measurement reproduces the parent's own numbers.
        assert_eq!(child.prepared.sha256, parent.prepared.sha256);
        let verified = crate::evidence::verify_run(&child_dir.join("receipt.json")).unwrap();
        let ranking = verified.ranking().unwrap();
        if parent_best.genome != search::BOOTSTRAP_GENOME {
            let again = ranking
                .iter()
                .find(|b| b.genome == parent_best.genome)
                .expect("the seed was measured on every cut");
            assert_eq!(
                again.validation_rmse.to_bits(),
                parent_best.validation_rmse.to_bits()
            );
        }
    }

    #[test]
    fn a_tampered_parent_receipt_is_refused_before_anything_is_written() {
        let temp = tempfile::tempdir().unwrap();
        let parent_dir = temp.path().join("parent");
        run(&small(&parent_dir, 12)).unwrap();
        let receipt_path = parent_dir.join("receipt.json");
        let pristine = std::fs::read(&receipt_path).unwrap();

        // A better score than the evidence shows.
        let mut value: serde_json::Value = serde_json::from_slice(&pristine).unwrap();
        let rmse = value["search"]["best"]["validation_rmse"].as_f64().unwrap();
        value["search"]["best"]["validation_rmse"] = (rmse / 2.0).into();
        std::fs::write(&receipt_path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
        let child = temp.path().join("child-score");
        let error = format!(
            "{:#}",
            run(&RunOptions {
                incumbent: Some(receipt_path.clone()),
                ..small(&child, 12)
            })
            .unwrap_err()
        );
        assert!(error.contains("--incumbent"), "{error}");
        assert!(!child.exists(), "a refused run wrote {}", child.display());

        // An observation edited under an untouched receipt.
        std::fs::write(&receipt_path, &pristine).unwrap();
        let observations = parent_dir.join("search/cut-0/observations.json");
        let mut rows: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&observations).unwrap()).unwrap();
        rows[0]["rmse"] = 0.0.into();
        std::fs::write(&observations, serde_json::to_vec_pretty(&rows).unwrap()).unwrap();
        let child = temp.path().join("child-observation");
        let error = format!(
            "{:#}",
            run(&RunOptions {
                incumbent: Some(receipt_path.clone()),
                ..small(&child, 12)
            })
            .unwrap_err()
        );
        assert!(error.contains("hashes to"), "{error}");
        assert!(!child.exists());
    }

    /// Every file under `root`, by relative path.
    fn tree(root: &std::path::Path) -> std::collections::BTreeMap<String, Vec<u8>> {
        fn walk(
            root: &std::path::Path,
            dir: &std::path::Path,
            out: &mut std::collections::BTreeMap<String, Vec<u8>>,
        ) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    walk(root, &path, out);
                } else {
                    out.insert(
                        path.strip_prefix(root).unwrap().display().to_string(),
                        std::fs::read(&path).unwrap(),
                    );
                }
            }
        }
        let mut out = std::collections::BTreeMap::new();
        walk(root, root, &mut out);
        out
    }

    /// The worker count this host admits, up to eight.
    fn workers_up_to_eight() -> usize {
        let available = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
        crate::parallel::affinity_cpus()
            .unwrap()
            .min(available)
            .min(8)
    }

    /// `--threads 1` and `--threads 8` write the same run: the receipt, the prepared cuts, every
    /// cut's search and observations, and the emitted sources, byte for byte. The wall clock is
    /// frozen for the comparison because it is the one thing two runs never share.
    #[test]
    fn one_thread_and_eight_threads_write_byte_identical_runs() {
        let workers = workers_up_to_eight();
        if workers < 2 {
            eprintln!("skipped: this host admits one worker, so no parallel run can be compared");
            return;
        }
        let temp = tempfile::tempdir().unwrap();
        let write = |threads: usize, name: &str| {
            let output = temp.path().join(name);
            let receipt = run(&RunOptions {
                threads,
                clock: Clock::Frozen,
                cuts: 2,
                ..small(&output, 30)
            })
            .unwrap();
            (receipt, tree(&output))
        };
        let (one, one_tree) = write(1, "one");
        let (many, many_tree) = write(workers, "many");
        assert_eq!(
            one_tree.keys().collect::<Vec<_>>(),
            many_tree.keys().collect::<Vec<_>>()
        );
        for (path, bytes) in &one_tree {
            assert!(
                many_tree[path] == *bytes,
                "{path} differs between 1 and {workers} threads"
            );
        }
        assert_eq!(one, many);
        assert!(one_tree.contains_key("receipt.json"));
        assert!(one.search.cuts.iter().all(|c| c
            .accounting
            .as_ref()
            .unwrap()
            .comparison
            .completed_evaluations
            > 0));
    }

    #[test]
    fn workers_above_the_cpu_ceiling_or_the_affinity_mask_are_refused_first() {
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("run");
        let cpus = crate::parallel::affinity_cpus().unwrap();
        let error = format!(
            "{:#}",
            run(&RunOptions {
                threads: cpus + 1,
                ..small(&output, 4)
            })
            .unwrap_err()
        );
        assert!(error.contains("sched_getaffinity"), "{error}");
        let error = format!(
            "{:#}",
            run(&RunOptions {
                cpu_ceiling_millicores: Some(999),
                ..small(&output, 4)
            })
            .unwrap_err()
        );
        assert!(error.contains("--cpu-ceiling 999"), "{error}");
        assert!(!output.exists(), "a refused run wrote {}", output.display());
    }

    #[test]
    fn a_budget_that_cannot_remeasure_the_seeds_is_refused() {
        let temp = tempfile::tempdir().unwrap();
        let file = seeds_file(temp.path(), &["(pipe X (standardize) (ridge))"]);
        let output = temp.path().join("run");
        let error = format!(
            "{:#}",
            run(&RunOptions {
                seed_genomes: Some(file),
                ..small(&output, 1)
            })
            .unwrap_err()
        );
        assert!(error.contains("cannot re-measure"), "{error}");
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
