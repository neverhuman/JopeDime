//! The run: `dataset -> prepare -> (propose) -> search -> select -> emit -> receipt`.
//!
//! Every stage is timed and recorded. The propose stage — encode the dataset and
//! let the policy propose a seed population — is not built in this work package;
//! it is recorded as a skip with its reason and printed, and `--policy require`
//! fails closed rather than letting a caller believe it ran. The S3 publication
//! and router-training stages are likewise out of scope and are not pretended to.

use anyhow::{bail, ensure, Context, Result};
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

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
pub(crate) const PROPOSE_SKIP_REASON: &str =
    // jankurai:allow HLT-001-DEAD-MARKER reason=typed-unsupported-operation-error expires=2026-12-31
    "the encode/propose stage is not implemented in this work \
     package: no policy checkpoint is bound and no seed population is proposed, so the search \
     starts from its own bootstrap population";

#[derive(Clone, Debug)]
pub struct RunOptions {
    pub landmarks: jope_runtime::landmark_args::LandmarkArgs,
    pub encode_device: String,
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
    /// Worker threads evaluating each generation. When the evaluation budget ends every phase it
    /// changes no decision and no published byte other than wall-clock readings. A binding
    /// seconds bound is checked before each fit starts, on every worker, so how many candidates
    /// start before it depends on timing and therefore on this count; the receipt then records
    /// `seconds` as the phase's stop reason. Not recorded in the receipt.
    pub threads: usize,
    /// Independent random streams over the same cuts (recorded, with every migration).
    pub islands: usize,
    pub migration_interval: usize,
    /// An explicit CPU admission in millicores that `islands x threads` must fit.
    pub cpu_ceiling_millicores: Option<u64>,
    /// The commit the caller runs. When given it is the receipt's `git_sha` and `git` is never
    /// spawned (a SmartCluster job's working directory need not be a checkout at all).
    pub source_commit: Option<String>,
    /// The termination flag: once set, the search stops between evaluations and no receipt is
    /// written.
    pub cancel: Option<Arc<AtomicBool>>,
    pub clock: Clock,
    /// Rank programs by this regression objective on held-out rows instead of RMSE.
    pub objective: Option<jope_runtime::fitness::Fitness>,
}

impl RunOptions {
    /// Defaults matching `jope pipeline run` with only `--dataset` and `--output` given.
    pub fn new(dataset: impl Into<String>, output: impl Into<PathBuf>) -> Self {
        RunOptions {
            landmarks: Default::default(),
            encode_device: "cuda:0".into(),
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
            islands: 1,
            migration_interval: search::DEFAULT_MIGRATION_INTERVAL,
            cpu_ceiling_millicores: None,
            source_commit: None,
            cancel: None,
            clock: Clock::Wall,
            objective: None,
        }
    }
}

/// [`crate::parallel::Cancelled`] once the run's termination flag is set.
pub fn check_cancelled(options: &RunOptions) -> Result<()> {
    if options
        .cancel
        .as_ref()
        .is_some_and(|flag| flag.load(Ordering::SeqCst))
    {
        return Err(crate::parallel::Cancelled.into());
    }
    Ok(())
}

/// A full git object id: 40 (SHA-1) or 64 (SHA-256) lowercase hex digits.
pub(crate) fn is_commit(value: &str) -> bool {
    matches!(value.len(), 40 | 64)
        && value
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// Every refusal that needs no dataset and writes nothing: the counts (`--cuts` within the fold
/// partition), the budget, the policy and the parallelism this host admits. Returns the worker
/// count.
pub fn preflight(options: &RunOptions) -> Result<usize> {
    options.landmarks.validate()?;
    crate::landmark_encoding::preflight(options)?;
    // The whole range the preparation accepts, not just a positive count: a count it refuses
    // after `evolve` has created `round-001/` blocks the corrected retry.
    prepare::check_cuts(options.cuts)?;
    ensure!(!options.languages.is_empty(), "--emit needs a target");
    options.budget.check()?;
    // Fail closed before anything is written: a refused run must not leave a half
    // prepared output directory behind to block the retry.
    ensure!(
        options.policy != Policy::Require,
        "--policy require: {PROPOSE_SKIP_REASON}. Re-run with --policy skip to proceed \
         without it."
    );
    ensure!(
        options.islands <= 1 || options.migration_interval >= 1,
        "--migration-interval must be at least 1 with more than one island"
    );
    if let Some(commit) = &options.source_commit {
        ensure!(
            is_commit(commit),
            "--source-commit {commit:?} is not a full commit id (40 or 64 lowercase hex digits)"
        );
    }
    if let Some(objective) = &options.objective {
        check_objective(objective)?;
    }
    let workers = crate::parallel::admit(
        options.threads,
        options.islands,
        options.cpu_ceiling_millicores,
    )?;
    if options.landmarks.enabled()
        && matches!(
            options.landmarks.mode(),
            jope_runtime::landmark_args::LandmarkMode::Full
                | jope_runtime::landmark_args::LandmarkMode::Progressive
        )
    {
        let total = workers
            .checked_add(1)
            .context("encoding worker count overflows")?;
        crate::parallel::admit(total, 1, options.cpu_ceiling_millicores)
            .context("native landmark encoding requires one CPU in addition to search workers")?;
    }
    Ok(workers)
}

/// The objectives a program search can rank by: valid, scored from targets and predictions
/// alone (the prepared cuts carry features only), and independent of row order (the cuts are
/// the dataset's fold partition, not a chronological split).
pub fn check_objective(objective: &jope_runtime::fitness::Fitness) -> Result<()> {
    objective.validate()?;
    jope_runtime::fitness::require_no_columns(objective, "a program search")?;
    ensure!(
        !objective.requires_order(),
        "{} depends on row order; the search's cuts are an unordered fold partition. Use \
         jopedime with --validation forward for order-dependent objectives",
        objective.name()
    );
    Ok(())
}

/// A catalog row this regression-only path can fit, resolved and byte-verified, or the refusal.
pub(crate) fn resolve_regression(selector: &str) -> Result<dataset::Resolved> {
    let resolved = dataset::resolve(selector)?;
    ensure!(
        resolved.row.task == jope_datasets::catalog::Task::Regression,
        "task {} is unsupported by jope pipeline run: this path fits and scores regression only; \
         use a task-qualified evaluator when available",
        resolved.row.task.as_str()
    );
    Ok(resolved)
}

/// Every refusal [`run`] makes before it writes a byte, made without writing anything:
/// [`preflight`], the seeds and the parent receipt, and a dataset that resolves, is a regression
/// task, parses and partitions into the requested cuts ([`prepare::partition`]). `jope pipeline
/// evolve` calls this before it creates its first round, so a refused lineage leaves nothing
/// behind that would block the corrected retry.
pub fn check(options: &RunOptions) -> Result<()> {
    preflight(options)?;
    resolve_seeding(options)?;
    let resolved = resolve_regression(&options.dataset)?;
    let rows = dataset::training_rows(&resolved)?;
    prepare::partition(
        &rows,
        &resolved.row.dataset_hash,
        options.seed,
        options.cuts,
    )?;
    Ok(())
}

/// Seeds resolved from `--incumbent` and `--seed-genomes`, before anything is written.
pub(crate) fn resolve_seeding(options: &RunOptions) -> Result<(Vec<String>, receipt::Seeding)> {
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

pub(crate) struct Stages {
    pub(crate) stages: Vec<receipt::Stage>,
}

impl Stages {
    pub(crate) fn new() -> Self {
        Stages { stages: Vec::new() }
    }

    pub(crate) fn record(
        &mut self,
        name: &str,
        status: &str,
        reason: Option<String>,
        started: Stamp,
    ) {
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
    crate::narration::from_env(
        |tap| run_observed(options, tap),
        || {
            options
                .cancel
                .as_ref()
                .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::SeqCst))
        },
    )
}

/// Run with an explicit observer. Narration failure never changes the pipeline result.
pub fn run_with_narration(
    options: &RunOptions,
    narration: Option<jope_narrator::Tap>,
) -> Result<Receipt> {
    crate::narration::with_tap(
        narration,
        |tap| run_observed(options, tap),
        || {
            options
                .cancel
                .as_ref()
                .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::SeqCst))
        },
    )
}

fn run_observed(options: &RunOptions, narration: Option<&jope_narrator::Tap>) -> Result<Receipt> {
    preflight(options)?;
    let clock = options.clock;

    let output = &options.output;
    // Every entry the run writes, looked up without following symbolic links: `exists()` passes
    // a dangling link, and the run then wrote its receipt through it, outside `--output`, or was
    // refused by it only after `prepared/` and `search/` existed, which blocked the retry. Other
    // entries are left alone (`evolve` hands each round a directory holding its `seeds.json`).
    for existing in ["receipt.json", "prepared", "search", "emit"] {
        ensure!(
            std::fs::symlink_metadata(output.join(existing)).is_err(),
            "{}: already holds a pipeline run ({existing}); choose a new --output",
            output.display()
        );
    }
    // A seed that does not parse, a seeds file that does not verify or a parent receipt its own
    // evidence does not re-derive is refused here, before anything is written.
    let (seed_genomes, seeding) = resolve_seeding(options)?;
    let mut stages = Stages::new();

    let started = clock.start();
    let resolved = resolve_regression(&options.dataset)?;
    let rows = dataset::training_rows(&resolved)?;
    stages.record("dataset", "passed", None, started);

    // Validate the task and inputs before publishing any run artifacts: the partition makes its
    // refusals (too few rows in a cut, a constant held-out target) before `--output` exists.
    let started = clock.start();
    let partition = prepare::partition(
        &rows,
        &resolved.row.dataset_hash,
        options.seed,
        options.cuts,
    )?;
    std::fs::create_dir_all(output).with_context(|| format!("create {}", output.display()))?;
    // The search rejects traversal and symlink components beneath this root.
    let root = output
        .canonicalize()
        .with_context(|| format!("resolve {}", output.display()))?;
    let preparation = partition.write(&root.join("prepared"))?;
    stages.record("prepare", "passed", None, started);
    let encodings = crate::landmark_encoding::start(options, &preparation, &root)?;

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
        islands: options.islands,
        migration_interval: options.migration_interval,
        clock,
        cancel: options.cancel.clone(),
        objective: options.objective.clone(),
        narration: narration.cloned(),
        ..search::SearchOptions::new(options.budget, options.rng_seed)
    };
    let searched =
        search::run_cuts_observed(&preparation.cuts, &search_options, &root.join("search"))?;
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
    if let Some(tap) = narration {
        tap.emit(jope_narrator::LineageFact::Selected {
            program_sha256: selection.program_sha256.clone(),
            rule: selection.method.to_owned(),
        });
    }
    if let Some(reason) = &selection.reason {
        eprintln!("jope pipeline: select - {reason}");
    }
    stages.record("select", "passed", selection.reason.clone(), started);

    check_cancelled(options)?;
    let started = clock.start();
    let emit_dir = root.join("emit");
    // Fitted export refits only on the verified training partition after selection.
    // Legacy recipe requests delegate directly without inspecting these rows.
    let emitted =
        emit::emit_with_training(&selection.genome, &options.languages, &emit_dir, &rows)?;
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
    if let Some(tap) = narration {
        tap.emit(jope_narrator::LineageFact::Emitted {
            program_sha256: emitted.genome_sha256.clone(),
            languages: emitted
                .artifacts
                .iter()
                .map(|artifact| artifact.language.to_owned())
                .collect(),
        });
    }
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
    // The last point a stop request is honoured: after this the receipt exists and the run
    // counts as completed.
    check_cancelled(options)?;
    encodings.finish()?;
    crate::run::check_cancelled(options)?;
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
            best_validation_objective: cut.best_validation_objective,
            best_program: cut.best_program.clone(),
        });
    }
    Ok(Receipt {
        schema: receipt::schema_id(options.objective.is_some()).to_owned(),
        ok: stages.iter().all(|s| s.status != "failed"),
        tool_version: TOOL_VERSION.to_owned(),
        git_sha: match &options.source_commit {
            Some(commit) => Some(commit.clone()),
            None => git_sha(),
        },
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
                validation_objective: b.validation_objective,
                validation_r2: b.validation_r2,
                train_r2: b.train_r2,
                seconds: b.seconds,
                cuts_measured: b.cuts_measured as u64,
            }),
            seeding: Some(seeding),
            islands: Some(receipt::Islands {
                count: options.islands.max(1) as u64,
                migration_interval: options.migration_interval as u64,
                migrations: searched
                    .iter()
                    .flat_map(|cut| cut.migrations.iter().cloned())
                    .collect(),
            }),
            objective: options
                .objective
                .as_ref()
                .map(receipt::Objective::of)
                .transpose()?,
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
pub(crate) fn git_sha() -> Option<String> {
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
pub(crate) mod tests;
