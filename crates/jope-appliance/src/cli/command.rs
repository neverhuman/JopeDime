use super::*;

#[derive(Parser)]
#[command(
    name = "jopedime",
    about = "Fit and predict tabular models with native Rust methods, from one signed executable.",
    disable_version_flag = true,
    after_help = "Exit status: 0 success, 2 invalid input, 3 integrity or compatibility, \
                  4 backend unavailable, 5 time budget exhausted, 6 numerical failure, \
                  7 output failure."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: ApplianceCommand,
}

#[derive(Subcommand)]
pub enum ApplianceCommand {
    /// Connect this CLI to a VEOX account.
    #[cfg(feature = "client")]
    Auth {
        #[command(subcommand)]
        command: crate::cloud::AuthCommand,
    },
    /// Synchronize run metadata and explicitly transfer model objects.
    #[cfg(feature = "client")]
    Cloud {
        #[command(subcommand)]
        command: crate::cloud::CloudCommand,
    },
    /// Internal research training-record tools.
    #[cfg(feature = "research")]
    Research(crate::research::ResearchArgs),
    /// Export a run's selected model and generated sources.
    #[cfg(feature = "mcp")]
    Export {
        run: String,
        #[arg(long)]
        alg: Option<String>,
        #[arg(long, value_enum)]
        kind: ExportKindArg,
        #[arg(long, value_enum)]
        lang: LanguageArg,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        force: bool,
        #[arg(long)]
        idempotency_key: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Manage datasets registered with the local engine.
    #[cfg(feature = "mcp")]
    Dataset {
        #[command(subcommand)]
        command: DatasetCommand,
    },
    /// Validate an objective specification.
    #[cfg(feature = "mcp")]
    Objective {
        #[command(subcommand)]
        command: ObjectiveCommand,
    },
    /// Collect unused engine-store objects.
    #[cfg(feature = "mcp")]
    Gc {
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        idempotency_key: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Internal isolated candidate worker.
    #[cfg(feature = "engine")]
    #[command(name = "__worker", hide = true)]
    Worker {
        #[arg(long)]
        protocol: u32,
    },
    /// Save, inspect, verify and export language-neutral fitted objects; or serve their MCP tools.
    Model(crate::model::ModelArgs),
    /// Start a run from a CSV or an API dataset identifier.
    #[cfg(feature = "mcp")]
    Run(crate::cmd::run::RunArgs),
    /// Read or long-poll a run snapshot.
    #[cfg(feature = "mcp")]
    Status(crate::cmd::status::StatusArgs),
    /// Read or follow run events as NDJSON.
    #[cfg(feature = "mcp")]
    Events(crate::cmd::events::EventsArgs),
    /// Control a run through the local JopeDime service.
    #[cfg(feature = "client")]
    Control {
        run: String,
        #[command(subcommand)]
        action: ControlAction,
        /// Raw JSON request body file, or `-` for stdin.
        #[arg(long)]
        body: Option<String>,
        /// Reuse this key when retrying the same mutation.
        #[arg(long)]
        idempotency_key: Option<String>,
    },
    /// Fork a run with the requested restart options.
    #[cfg(feature = "client")]
    Restart {
        run: String,
        #[arg(long)]
        new_seed: bool,
        #[arg(long)]
        warm: bool,
        /// Raw JSON request body file, or `-` for stdin.
        #[arg(long)]
        body: Option<String>,
        /// Reuse this key when retrying the same mutation.
        #[arg(long)]
        idempotency_key: Option<String>,
    },
    /// List, delete or export feedback from service runs.
    #[cfg(feature = "client")]
    Runs {
        #[command(subcommand)]
        command: RunsCommand,
    },
    /// Read source for an algorithm from a run.
    #[cfg(feature = "client")]
    Code {
        run: String,
        #[arg(long)]
        alg: Option<String>,
        #[arg(long)]
        lang: Option<String>,
    },
    /// Product version, source commit, runtime ABI, target and package identity.
    Version(JsonFlag),
    /// Package integrity, devices, promised methods and limits.
    Doctor(JsonFlag),
    /// Every method this build knows and whether it can run.
    Methods {
        /// Only methods for this task: regression, binary or multiclass.
        #[arg(long)]
        task: Option<String>,
        #[command(flatten)]
        json: JsonFlag,
    },
    /// Fit a model on a CSV table and write it to a model file.
    Fit {
        #[command(flatten)]
        train: TrainArgs,
        /// Write a legacy fitted file instead of the standard content-addressed object.
        #[arg(long)]
        output: Option<PathBuf>,
        /// Object store root when --output is omitted.
        #[arg(long, default_value = "models", conflicts_with = "output")]
        model_root: PathBuf,
        #[command(flatten)]
        common: CommonArgs,
    },
    /// Predict with a local fitted model or a model from a run; never refits.
    Predict {
        /// A local model path or a completed run ID, optionally with `:ALG`.
        #[arg(long)]
        model: String,
        /// CSV rows to predict, with the model's feature columns in any order.
        #[arg(long)]
        input: PathBuf,
        /// Predictions CSV to write (atomically).
        #[arg(long)]
        output: PathBuf,
        #[command(flatten)]
        common: CommonArgs,
    },
    /// Fit and predict in one invocation, without writing a model file.
    FitPredict {
        #[command(flatten)]
        train: TrainArgs,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[command(flatten)]
        common: CommonArgs,
    },
    /// Discover, validate or evaluate regression objectives (`--fitness` specifications).
    Fitness {
        #[command(subcommand)]
        command: FitnessCommand,
    },
    /// Serve JSON-RPC MCP tools on stdin/stdout (one JSON document per line).
    Mcp,
    /// Serve the same fit/evaluate API on stdin/stdout as JSON lines.
    Api,
    /// Start or inspect the shared background engine.
    Serve {
        #[arg(long, conflicts_with_all = ["stop", "idle_exit"])]
        status: bool,
        #[arg(long, conflicts_with = "idle_exit")]
        stop: bool,
        #[arg(long, value_name = "D", conflicts_with_all = ["status", "stop"])]
        idle_exit: Option<String>,
    },
    /// Verify this executable's embedded package completely. A diagnostic: it cannot establish
    /// trust in a binary that may itself be modified; verify downloads with `jope-dist verify`.
    VerifySelf(JsonFlag),
    /// Print the embedded licence and notice inventory.
    Licenses(JsonFlag),
}

#[cfg(all(test, not(feature = "research")))]
mod research_feature_tests {
    use super::Cli;
    use clap::{CommandFactory, Parser};

    #[test]
    fn ordinary_build_has_no_research_help_or_dispatch() {
        assert!(!Cli::command()
            .render_help()
            .to_string()
            .contains("research"));
        assert!(
            Cli::try_parse_from(["jopedime", "research", "record", "--out", "unused.jsonl"])
                .is_err()
        );
    }
}
