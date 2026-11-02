#[derive(Subcommand)]
enum Command {
    /// Report current ASM capabilities and remaining qualification gaps.
    Doctor,
    /// Fit, save, reload and score one bounded demonstration with measured elapsed time.
    Demo(jope_bench::panel::demo::DemoCli),
    /// Verify and plan the private, fixed-panel benchmark.
    Benchmark(jope_bench::panel::cli::BenchmarkCli),
    /// Plan, run, resume and verify bounded experimental policy training.
    #[cfg(target_os = "linux")]
    Train(train::TrainCli),
    /// Generate and optimize executable algorithms from the verified development panel.
    Optimize(jope_bench::panel::optimize_cli::OptimizeCli),
    /// Verify all forty synthetic fixtures against the pinned deterministic generator.
    VerifyInputs {
        /// Explicitly select offline qualification; no network access is performed.
        #[arg(long)]
        offline: bool,
        /// Emit one machine-readable result, including on verification failure.
        #[arg(long)]
        json: bool,
        /// Directory containing catalog.json; otherwise resolve at runtime.
        #[arg(long)]
        root: Option<PathBuf>,
    },
    /// Create the versioned synthetic corpus in a new output directory.
    GenerateFixtures {
        #[arg(long)]
        output: PathBuf,
        /// Check an existing directory for exact deterministic regeneration instead.
        #[arg(long)]
        check: bool,
        #[arg(long)]
        json: bool,
    },
    /// Encode one admitted dataset with CPU Hyperion. Needs HYPERION_V7_WEIGHTS; does not download.
    Encode {
        #[command(flatten)]
        landmarks: jope_runtime::landmark_args::LandmarkArgs,
        #[arg(long, default_value = "cuda:0")]
        device: String,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Fit a GP genome on one validated regression fixture (train R², validation R²).
    Fit {
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value = DEFAULT_GENOME)]
        genome: String,
        #[arg(long)]
        json: bool,
    },
    /// Emit working Python / Rust / C++17 / C++20 / JavaScript / TypeScript / Go / C17 / CUDA C++, or fitted C++17 from numeric training CSV.
    Emit {
        #[arg(long, default_value = DEFAULT_GENOME)]
        genome: String,
        /// rust, python, cpp17, cpp17-recipe-v2, cpp20, javascript (or js), typescript (or ts), go (or golang), c17 (or c), cuda (or cuda-cpp, cu), cpp17-fitted, or all (the four default recipes)
        #[arg(long)]
        lang: String,
        /// Numeric CSV with a final `target` column; required for cpp17-fitted.
        #[arg(long)]
        train: Option<PathBuf>,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Score a tiny ridge-family GP candidate set on a regression fixture's validation split.
    Search {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Search supported regression fixtures and report the rest as unsupported. --pull/--push fail closed (no live S3).
    Champions {
        #[arg(long)]
        local: bool,
        #[arg(long)]
        pull: bool,
        #[arg(long)]
        push: bool,
        #[arg(long)]
        root: Option<PathBuf>,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// One local train step: fit a 128-d centroid baseline; accepted policy weights unchanged. Live --env fails closed.
    TrainStep {
        #[arg(long)]
        local: bool,
        #[arg(long)]
        env: Option<String>,
        #[arg(long)]
        root: Option<PathBuf>,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Route one regression fixture through the local centroid baseline (accepted policy weights unchanged), then fit the chosen genome.
    Route {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        router: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Model bundles: fetch, verify, publish and locate content-addressed weights.
    Models(jope_models::cli::Cli),
    /// Dataset catalog: ingest, fetch, verify and link content-addressed datasets.
    Datasets(jope_datasets::cli::Cli),
    /// Product pipeline: dataset, prepared cuts, GP search, emitted code and a receipt.
    Pipeline(Box<jope_pipeline::cli::Cli>),
    /// Reusable program fragments kept in a private table, with exact expansion checks.
    /// Hidden from the default help until the capability is published (owner decision,
    /// 2026-09-24); `jope motifs --help` still works.
    #[command(hide = true)]
    Motifs(jope_motifs::cli::Cli),
}
