use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

use crate::{
    emit::Language,
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
    #[command(flatten)]
    pub landmarks: jope_runtime::landmark_args::LandmarkArgs,
    /// Device for opt-in Dope encoding. Explicit CUDA never falls back.
    #[arg(long = "device", default_value = "cuda:0")]
    pub encode_device: String,
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
    /// Languages to emit: rust, python, cpp17, cpp20, js, ts, go, c17, cpp17-fitted, or all (legacy recipes).
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
            landmarks: self.landmarks.clone(),
            encode_device: self.encode_device.clone(),
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
        /// How many --seed-genomes (after canonicalising and deduplicating) generation 0 admits,
        /// 1..=32 (default 4). Seeds join generation 0 on top of its random draws, so a larger
        /// limit lets more proposals, such as a neural policy's, compete there.
        #[arg(long, default_value_t = crate::search::ELITES)]
        seed_limit: usize,
        /// A parent receipt: verified against its run's evidence, its selected genome is seeded
        /// first and its sha256 recorded as this run's parent.
        #[arg(long)]
        incumbent: Option<PathBuf>,
        /// Rank programs by this regression objective on held-out rows instead of RMSE: a
        /// catalog name (`mae`), `name:key=value` (`pinball:quantile=0.9`), inline JSON or
        /// @path.json. Legacy search profile only; order-dependent objectives and objectives
        /// that read extra columns are refused (use `jopedime` for those).
        #[arg(long)]
        fitness: Option<String>,
        /// Print the receipt as one JSON document.
        #[arg(long)]
        json: bool,
    },
    /// Validate a receipt against the committed jope.pipeline-receipt/v1, v2 or v3 schema, or
    /// jope.pipeline-objective-receipt/v1 for a search ranked by an objective.
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
        /// Also report this regression objective on the test rows (default: the objective the
        /// run ranked by, if any). The same single look; nothing reads it back into selection.
        #[arg(long)]
        fitness: Option<String>,
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
        /// Write jope.invent.progress.v1 records (NDJSON) to this file while searching; created
        /// or truncated. Must not be the --train file, the --work-dir or inside it.
        #[arg(long)]
        events: Option<PathBuf>,
        /// Rank candidates by this regression objective on held-out rows instead of RMSE (as
        /// `pipeline run --fitness` accepts it).
        #[arg(long)]
        fitness: Option<String>,
        /// Finish early when this file appears: start nothing new, then rank and print what was
        /// measured (exit 0, `finished_early` in --json). SIGUSR1 does the same. SIGTERM still
        /// cancels without output.
        #[arg(long)]
        stop_file: Option<PathBuf>,
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
