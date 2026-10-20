//! `jope pipeline …`: the same `Cli`/`Cmd`/`exec`/`run` shape the other
//! subcommand crates expose, so the umbrella delegates to it in process.

use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use std::{path::PathBuf, process::ExitCode};

use crate::{
    emit::Language,
    receipt,
    run::{Policy, RunOptions},
    search::Budget,
};

#[derive(Parser)]
#[command(
    name = "jope-pipeline",
    version,
    about = "JOPE-Dime product pipeline: dataset, prepared cuts, CPU GP search, emitted code and a receipt"
)]
pub struct Cli {
    #[command(subcommand)]
    pub cmd: Cmd,
}

/// How many CPUs the search may use. The worker count changes no decision while the evaluation
/// budget ends every phase; when the seconds budget binds, which candidates start depends on
/// timing, and the receipt records that phase as stopped by `seconds`.
#[derive(Args, Clone, Debug)]
pub struct ParallelArgs {
    /// Worker threads evaluating each generation's candidates (1 = sequential). Refused above
    /// the CPUs sched_getaffinity allows, available_parallelism, or --cpu-ceiling.
    #[arg(long, default_value_t = 1)]
    pub threads: usize,
    /// Independent random streams over the same cuts, each on --threads workers; islands x
    /// threads is the worker count the CPU checks bound.
    #[arg(long, default_value_t = 1)]
    pub islands: usize,
    /// With more than one island: every K generations each island's champion migrates into the
    /// others' elites.
    #[arg(long, default_value_t = crate::search::DEFAULT_MIGRATION_INTERVAL)]
    pub migration_interval: usize,
    /// The CPU admission in millicores (e.g. 8000 for eight cores) the workers must fit.
    #[arg(long)]
    pub cpu_ceiling: Option<u64>,
}

/// What a search runs on and with, shared by `run` and `evolve`.
#[derive(Args, Clone, Debug)]
pub struct SearchArgs {
    /// A 16-hex catalog id resolved in the local cache, or a dataset directory.
    #[arg(long)]
    pub dataset: String,
    /// Total per-cut budget across search and comparison (1..=1000 attempts / 1..=600 active seconds).
    /// Multiple cuts reserve half for comparison; an in-flight fit may overrun time.
    #[arg(long, default_value = "evaluations=64,seconds=300")]
    pub budget: String,
    /// Fold seed for the cut preparation.
    #[arg(long, default_value_t = crate::DEFAULT_SEED)]
    pub seed: u64,
    /// Seed of the search's random streams.
    #[arg(long, default_value_t = crate::DEFAULT_SEED)]
    pub rng_seed: u64,
    /// How many of the fold partition's cuts to prepare and search (1..=5).
    #[arg(long, default_value_t = 1)]
    pub cuts: usize,
    /// Languages to emit: rust, python, cpp17, cpp20, js, cpp17-fitted, or all (legacy recipes).
    /// Defaults to rust,python,cpp17,cpp20,js; kernel2 also includes cpp17-fitted.
    #[arg(long)]
    pub emit: Option<String>,
    /// The proposal stage: auto and skip record a skip, require fails closed.
    #[arg(long, default_value = "auto")]
    pub policy: String,
    #[command(flatten)]
    pub parallel: ParallelArgs,
    /// The commit the caller runs (40 or 64 lowercase hex): recorded as the receipt's git_sha,
    /// and git is never spawned.
    #[arg(long)]
    pub source_commit: Option<String>,
}

impl SearchArgs {
    /// Run options for `output`, with the termination flag wired when `signals`.
    pub fn options(&self, output: PathBuf, signals: bool) -> Result<RunOptions> {
        let policy = &self.policy;
        Ok(RunOptions {
            budget: Budget::parse(&self.budget)?,
            seed: self.seed,
            rng_seed: self.rng_seed,
            cuts: self.cuts,
            languages: Language::parse_list(
                self.emit.as_deref().unwrap_or("rust,python,cpp17,cpp20,js"),
            )?,
            policy: Policy::parse(policy).ok_or_else(|| {
                anyhow::anyhow!("--policy must be auto, skip or require, got {policy:?}")
            })?,
            threads: self.parallel.threads,
            islands: self.parallel.islands,
            migration_interval: self.parallel.migration_interval,
            cpu_ceiling_millicores: self.parallel.cpu_ceiling,
            source_commit: self.source_commit.clone(),
            cancel: if signals {
                Some(crate::parallel::termination_flag()?)
            } else {
                None
            },
            ..RunOptions::new(self.dataset.clone(), output)
        })
    }
}

#[derive(Subcommand)]
pub enum Cmd {
    /// Run dataset -> prepare -> search -> select -> emit and write the receipt.
    Run {
        #[command(flatten)]
        search: SearchArgs,
        #[command(flatten)]
        profile: crate::kernel2::cli::ProfileArgs,
        /// Where the preparation, searches, emitted sources and receipt are written.
        #[arg(long)]
        output: PathBuf,
        /// A jope.pipeline-seeds/v1 file: its genomes enter generation 0 after the bootstrap
        /// (canonicalised, deduplicated, at most 4) and are re-measured on this run's cuts.
        #[arg(long)]
        seed_genomes: Option<PathBuf>,
        /// A parent receipt: verified against its run's evidence, its selected genome is seeded
        /// first and its sha256 recorded as this run's parent.
        #[arg(long)]
        incumbent: Option<PathBuf>,
        /// Print the receipt as one JSON document.
        #[arg(long)]
        json: bool,
        /// Model-build switches (`--automl`, `--ensemble`, `--thinking`) and options file.
        #[command(flatten)]
        model: crate::model_build::ModelBuildArgs,
    },
    /// Run the native AutoML bake-off and outer refit from a request file. Native heads only: no
    /// neural route is involved. With `--output`, the admitted ensemble record is written as a new
    /// file and an existing file is refused.
    ModelBuildNative {
        /// A jope.model-build-native-request/v1 JSON file.
        #[arg(long)]
        request: PathBuf,
        /// Where the persisted ensemble record is written. Refused when the file exists.
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Explicit legacy generic GP search: no neural input and no model-build route. Its receipts
    /// are unchanged and remain verifiable. It is never a fallback for `run`.
    RunLegacyGp {
        #[command(flatten)]
        search: SearchArgs,
        #[command(flatten)]
        profile: crate::kernel2::cli::ProfileArgs,
        /// Where the preparation, searches, emitted sources and receipt are written.
        #[arg(long)]
        output: PathBuf,
        /// A jope.pipeline-seeds/v1 file: its genomes enter generation 0 after the bootstrap.
        #[arg(long)]
        seed_genomes: Option<PathBuf>,
        /// A parent receipt: verified against its run's evidence before it is seeded.
        #[arg(long)]
        incumbent: Option<PathBuf>,
        /// Print the receipt as one JSON document.
        #[arg(long)]
        json: bool,
    },
    /// Validate a receipt against the committed jope.pipeline-receipt/v1, v2 or v3 schema.
    VerifyReceipt {
        #[arg(long)]
        receipt: PathBuf,
        /// Also re-derive the receipt from its run directory's evidence (v2 and v3).
        #[arg(long)]
        evidence: bool,
        #[arg(long)]
        json: bool,
    },
    /// Rounds of seeded search on one dataset's fixed cuts, chained in lineage.json. Round r>1 is
    /// seeded from round r-1's champion and ranked runners-up; --control adds one unseeded run
    /// at the final round's configuration (one round's budget: not compute-matched against the
    /// lineage). Never writes to the learning store.
    Evolve {
        #[command(flatten)]
        search: SearchArgs,
        /// A new or empty directory for round-NNN/, control/ and lineage.json; anything already
        /// in it is refused before anything is written.
        #[arg(long)]
        output: PathBuf,
        /// Rounds to run (1..=100).
        #[arg(long)]
        rounds: usize,
        /// After the last round, one unseeded run with its configuration and one round's budget
        /// (the rounds' cumulative search is not matched; lineage verify reports both totals).
        #[arg(long)]
        control: bool,
        /// Round 1's jope.pipeline-seeds/v1 file (for example `jope-learning bank export`).
        #[arg(long)]
        seeds: Option<PathBuf>,
        /// Print the lineage as one JSON document.
        #[arg(long)]
        json: bool,
    },
    /// jope.pipeline-lineage/v1 records.
    Lineage {
        #[command(subcommand)]
        cmd: LineageCmd,
    },
    /// Score a verified receipt's selected genome once on the dataset's sealed test split (refit
    /// on every train.csv row). Writes a separate jope.pipeline-test-score/v1; a repeat for the
    /// same (dataset, genome) in the registry is refused, and the score lists the genomes already
    /// scored on that test split there (choosing among several makes the test split a selection
    /// input). Nothing in the tools reads it back into selection.
    Score {
        #[arg(long)]
        receipt: PathBuf,
        /// The split to score; only `test`.
        #[arg(long)]
        split: String,
        /// The dataset (default: the path the receipt recorded).
        #[arg(long)]
        dataset: Option<String>,
        /// Where the once-per-(dataset, genome) records live (default:
        /// $JOPE_STATE/pipeline/test-scores). A local record, not access control: another
        /// registry starts empty.
        #[arg(long)]
        registry: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Search a header CSV and print ranked candidate programs jain-math can execute
    /// (jope.candidates.v1): the candidate generator behind jain-core's Jope invention engine.
    /// Regression only; every value must be numeric and finite. Exit 3 when nothing executable
    /// was measured.
    Invent {
        /// Header CSV of training rows.
        #[arg(long)]
        train: PathBuf,
        /// The target column's header name.
        #[arg(long)]
        target: String,
        /// A new or empty directory for the per-cut search evidence.
        #[arg(long)]
        work_dir: PathBuf,
        /// Per-cut budget (1..=1000 attempts / 1..=600 active seconds); an in-flight fit may overrun time.
        #[arg(long, default_value = "evaluations=64,seconds=300")]
        budget: String,
        /// Fold seed for the cut preparation.
        #[arg(long, default_value_t = crate::DEFAULT_SEED)]
        seed: u64,
        /// Seed of the search's random streams.
        #[arg(long, default_value_t = crate::DEFAULT_SEED)]
        rng_seed: u64,
        /// Cuts of the five-fold partition to search (1..=5).
        #[arg(long, default_value_t = 3)]
        cuts: usize,
        /// Most candidates to print (1..=64), best first.
        #[arg(long, default_value_t = 8)]
        top_k: usize,
        #[command(flatten)]
        parallel: ParallelArgs,
        /// The commit the caller runs (40 or 64 lowercase hex), reported as git_sha.
        #[arg(long)]
        source_commit: Option<String>,
        /// Print one jope.candidates.v1 document, including on failure.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum LineageCmd {
    /// Re-derive every round and the control from their directories and check the chain:
    /// identical cuts and budget, each parent the previous receipt, each round's seeds the
    /// previous champion and runners-up, and no champion worse than the round before.
    Verify {
        #[arg(long)]
        lineage: PathBuf,
        /// Also require a measured candidate bred in generation 1 or later in every round: the
        /// search ran past its starting population. That is not a discovery; the report's
        /// champion_found_in_round and improved_on_previous_round say what the rounds found.
        #[arg(long)]
        require_bred_candidates: bool,
        #[arg(long)]
        json: bool,
    },
}

pub fn exec(cli: Cli) -> ExitCode {
    match run(cli) {
        Ok(code) => code,
        Err(error) if crate::parallel::is_cancelled(&error) => {
            eprintln!("error: {error:#}");
            ExitCode::from(crate::parallel::CANCELLED_EXIT)
        }
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::from(2)
        }
    }
}

pub fn run(cli: Cli) -> Result<ExitCode> {
    match cli.cmd {
        Cmd::Run { model, .. } => {
            // Every route needs a verified checkpoint. Until P02 supplies a release closure, each
            // route is refused here with its reason. Nothing falls back to the generic GP search.
            let file = match &model.model_build_config {
                Some(path) => Some(crate::model_build::config::load_file(path)?),
                None => None,
            };
            let resolved = crate::model_build::resolve(file.as_ref(), model.flags()?)?;
            // The source expectation comes from the operator, never from the claim. No P02 release
            // closure provider exists yet, so a verified checkpoint cannot be produced here.
            let checkpoint = match (
                file.as_ref().and_then(|f| f.neural_checkpoint.as_ref()),
                model.expected_source_commit.as_deref(),
            ) {
                (Some(claim), Some(expected)) => crate::model_build::NeuralCheckpoint::Refused(
                    crate::model_build::verify_checkpoint(claim, None, expected)
                        .err()
                        .unwrap_or(crate::model_build::CheckpointRefusal::Absent),
                ),
                (Some(_), None) => crate::model_build::NeuralCheckpoint::Refused(
                    crate::model_build::CheckpointRefusal::NoSourceExpectation,
                ),
                (None, _) => crate::model_build::NeuralCheckpoint::Refused(
                    crate::model_build::CheckpointRefusal::Absent,
                ),
            };
            let route = crate::model_build::route(&resolved);
            match crate::model_build::disposition(route, &checkpoint) {
                crate::model_build::Disposition::Refused { reason, .. } => anyhow::bail!(reason),
                crate::model_build::Disposition::Pending { missing, .. } => anyhow::bail!(
                    "{route:?} is not executable yet; missing: {}",
                    missing.join(", ")
                ),
            }
        }
        Cmd::ModelBuildNative { request, output } => {
            let request = crate::model_build::driver::read_native_request(&request)?;
            let outcome =
                crate::model_build::driver::run_native_request(&request, output.as_deref())?;
            let summary = serde_json::json!({
                "selected": outcome.receipt.selected,
                "persisted": outcome.persisted.is_some(),
                "output": output.as_ref().map(|path| path.display().to_string()),
            });
            println!("{}", serde_json::to_string_pretty(&summary)?);
            Ok(ExitCode::SUCCESS)
        }
        Cmd::RunLegacyGp {
            search,
            profile,
            output,
            seed_genomes,
            incumbent,
            json,
        } => {
            if let Some(mut options) = profile.options(&search, output.clone(), true)? {
                options.run.seed_genomes = seed_genomes;
                options.run.incumbent = incumbent;
                return crate::kernel2::cli::print_run(&options, json);
            }
            let options = RunOptions {
                seed_genomes,
                incumbent,
                ..search.options(output, true)?
            };
            let receipt = crate::run::run(&options)?;
            if json {
                println!("{}", serde_json::to_string(&receipt)?);
            } else {
                let measured: u64 = receipt
                    .search
                    .cuts
                    .iter()
                    .map(|c| c.successful_measurements)
                    .sum();
                let score = match &receipt.search.best {
                    Some(best) => format!(
                        "hold RMSE {:.6}, hold R2 {:.6} over {} cut(s)",
                        best.validation_rmse, best.validation_r2, best.cuts_measured
                    ),
                    None => "no measurement".to_owned(),
                };
                println!(
                    "Pipeline ran {} on {} cut(s): {measured} measurement(s); \
                     selected {} by {} ({score}); emitted {}; receipt in {}.",
                    receipt.dataset.id,
                    receipt.prepared.cuts.len(),
                    receipt.search.selected_genome,
                    receipt.search.selection,
                    receipt
                        .artifacts
                        .iter()
                        .map(|a| a.language.as_str())
                        .collect::<Vec<_>>()
                        .join(" "),
                    options.output.display()
                );
            }
            Ok(ExitCode::SUCCESS)
        }
        Cmd::Evolve {
            search,
            output,
            rounds,
            control,
            seeds,
            json,
        } => {
            let options = crate::lineage::EvolveOptions {
                run: search.options(output.clone(), true)?,
                output,
                rounds,
                control,
                seeds,
            };
            let lineage = crate::lineage::evolve(&options)?;
            if json {
                println!("{}", serde_json::to_string(&lineage)?);
            } else {
                for round in lineage.rounds.iter().chain(lineage.control.iter()) {
                    println!(
                        "{} {} ({}): mean held-out RMSE {:.6}, R2 {:.6} over {} cut(s); \
                         {} generation-1+ measurement(s); champion {}: {}",
                        round.kind,
                        round.round,
                        round.directory,
                        round.validation_rmse,
                        round.validation_r2,
                        round.cuts_measured,
                        round.generation_ge1_measured,
                        round.champion_origin(),
                        round.selected_genome
                    );
                }
                if let Some(control) = &lineage.control {
                    println!(
                        "the rounds made {} evaluation(s) in total, the control {} (one round's \
                         budget: not a compute-matched comparison)",
                        lineage
                            .rounds
                            .iter()
                            .map(|r| r.completed_evaluations)
                            .sum::<u64>(),
                        control.completed_evaluations
                    );
                }
                println!(
                    "lineage in {}",
                    options.output.join(crate::lineage::LINEAGE_FILE).display()
                );
            }
            Ok(ExitCode::SUCCESS)
        }
        Cmd::Lineage {
            cmd:
                LineageCmd::Verify {
                    lineage,
                    require_bred_candidates,
                    json,
                },
        } => match crate::lineage::verify(&lineage, require_bred_candidates) {
            Ok(report) => {
                if json {
                    println!("{}", serde_json::to_string(&report)?);
                } else {
                    println!(
                        "ok {} ({} round(s) on prepared cuts {}; mean held-out RMSE by round \
                         {:?}; control {:?}; champion found in the round itself {:?}; \
                         improved on the round before {:?}; evaluations: rounds {} in total, \
                         control {:?})",
                        lineage.display(),
                        report.rounds,
                        report.prepared_sha256,
                        report.validation_rmse,
                        report.control_validation_rmse,
                        report.champion_found_in_round,
                        report.improved_on_previous_round,
                        report.rounds_completed_evaluations,
                        report.control_completed_evaluations
                    );
                }
                Ok(ExitCode::SUCCESS)
            }
            Err(error) => {
                if json {
                    println!(
                        "{}",
                        serde_json::json!({"schema": crate::lineage::LINEAGE_SCHEMA_ID,
                                           "ok": false, "error": format!("{error:#}")})
                    );
                } else {
                    eprintln!("invalid lineage: {error:#}");
                }
                Ok(ExitCode::FAILURE)
            }
        },
        Cmd::Score {
            receipt,
            split,
            dataset,
            registry,
            json,
        } => {
            anyhow::ensure!(split == "test", "--split must be test, got {split:?}");
            let (score, path) = crate::test_score::score(&crate::test_score::ScoreOptions {
                receipt,
                dataset,
                registry: registry.unwrap_or_else(crate::test_score::default_registry),
                clock: crate::clock::Clock::Wall,
            })?;
            if !score.earlier_looks.is_empty() {
                eprintln!(
                    "jope pipeline score: the test split of {} was already scored for {} other \
                     genome(s) in {} ({}). This is look {} at it: keeping the test-best of \
                     several makes the test split a selection input. Report the genome chosen \
                     before any test score was seen, and name every look.",
                    score.dataset.dataset_hash,
                    score.earlier_looks.len(),
                    score.registry,
                    score.earlier_looks.join(", "),
                    score.earlier_looks.len() + 1
                );
            }
            if json {
                println!("{}", serde_json::to_string(&score)?);
            } else {
                println!(
                    "scored {} on {} test row(s) of {}: test RMSE {:.6}, test R2 {:.6} \
                     (development held-out RMSE {:.6}); score receipt {}",
                    score.genome,
                    score.dataset.test_rows,
                    score.dataset.id,
                    score.test_rmse,
                    score.test_r2,
                    score.validation_rmse,
                    path.display()
                );
            }
            Ok(ExitCode::SUCCESS)
        }
        Cmd::VerifyReceipt {
            receipt: path,
            evidence,
            json,
        } => {
            if crate::kernel2::run::is_v4(&path) {
                return crate::kernel2::cli::verify(&path, evidence, json);
            }
            match if evidence {
                crate::evidence::verify_run(&path).map(|run| run.receipt)
            } else {
                receipt::read(&path)
            } {
                Ok(receipt) => {
                    if json {
                        println!("{}", serde_json::to_string(&receipt)?);
                    } else {
                        println!(
                            "ok {} ({} stage(s), {} artifact(s), {})",
                            path.display(),
                            receipt.stages.len(),
                            receipt.artifacts.len(),
                            receipt.schema
                        );
                    }
                    Ok(ExitCode::SUCCESS)
                }
                Err(error) => {
                    if json {
                        println!(
                            "{}",
                            serde_json::json!({"schema": receipt::RECEIPT_SCHEMA_ID, "ok": false,
                                           "error": format!("{error:#}")})
                        );
                    } else {
                        eprintln!("invalid receipt: {error:#}");
                    }
                    Ok(ExitCode::FAILURE)
                }
            }
        }
        Cmd::Invent {
            train,
            target,
            work_dir,
            budget,
            seed,
            rng_seed,
            cuts,
            top_k,
            parallel,
            source_commit,
            json,
        } => {
            let outcome = Budget::parse(&budget).and_then(|budget| {
                let options = crate::invent::InventOptions {
                    budget,
                    seed,
                    rng_seed,
                    cuts,
                    top_k,
                    threads: parallel.threads,
                    islands: parallel.islands,
                    migration_interval: parallel.migration_interval,
                    cpu_ceiling_millicores: parallel.cpu_ceiling,
                    source_commit,
                    cancel: Some(crate::parallel::termination_flag()?),
                    ..crate::invent::InventOptions::new(train, target, work_dir)
                };
                crate::invent::invent(&options)
            });
            match outcome {
                Ok(candidates) => {
                    if json {
                        println!("{}", serde_json::to_string(&candidates)?);
                    } else {
                        for c in &candidates.candidates {
                            println!(
                                "{:>2}. hold RMSE {:.6}, hold R2 {:.6}  {}",
                                c.rank, c.mean_hold_rmse, c.mean_hold_r2, c.genome
                            );
                        }
                        println!(
                            "{} candidate(s) from {} ranked program(s); {} left out as not \
                             executable in jain-math {}",
                            candidates.candidates.len(),
                            candidates.ranked,
                            candidates.excluded_not_executable,
                            candidates.vocabulary.jain_math_rev
                        );
                    }
                    Ok(if candidates.ok {
                        ExitCode::SUCCESS
                    } else {
                        ExitCode::from(crate::invent::NO_CANDIDATES_EXIT)
                    })
                }
                // Cancellation keeps its own exit status (see `exec`).
                Err(error) if crate::parallel::is_cancelled(&error) => Err(error),
                Err(error) if json => {
                    println!(
                        "{}",
                        serde_json::json!({"schema": crate::invent::SCHEMA, "ok": false,
                                           "error": format!("{error:#}")})
                    );
                    Ok(ExitCode::from(2))
                }
                Err(error) => Err(error),
            }
        }
    }
}
