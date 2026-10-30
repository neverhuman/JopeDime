//! User-facing commands and flags (SPECIFICATION.md §3.3).
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

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
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Internal isolated candidate worker.
    #[cfg(feature = "engine")]
    #[command(name = "__worker", hide = true)]
    Worker {
        #[arg(long)]
        protocol: u32,
    },
    /// Save, inspect, verify and export language-neutral fitted objects; or serve their MCP tools.
    Model(crate::model::ModelArgs),
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
    /// Predict with a fitted model file; never refits.
    Predict {
        #[arg(long)]
        model: PathBuf,
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

#[derive(Args, Clone, Copy)]
pub struct JsonFlag {
    /// Print machine-readable JSON.
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Clone)]
pub struct TrainArgs {
    #[command(flatten)]
    pub landmarks: jope_runtime::landmark_args::LandmarkArgs,
    /// Training CSV (header row first).
    #[arg(long)]
    pub train: PathBuf,
    /// The target column's name.
    #[arg(long)]
    pub target: String,
    /// `auto`, regression, binary or multiclass.
    #[arg(long, default_value = "auto")]
    pub task: String,
    /// `auto` (bounded selection) or a method id, family or alias from `jopedime methods`.
    #[arg(long, default_value = "auto")]
    pub method: String,
    /// With `--method auto`: comma-separated candidates instead of the whole roster.
    #[arg(long, value_delimiter = ',')]
    pub candidates: Vec<String>,
    /// With `--method auto`: cross-validation folds (reduced for small tables).
    #[arg(long, default_value_t = 5)]
    pub folds: usize,
    /// Regression objective for selection and calibration: a catalog name (`mae`), a name with
    /// numeric overrides (`pinball:quantile=0.9`), inline JSON, or @path.json.
    #[arg(long)]
    pub fitness: Option<String>,
    /// Objective-aware output calibration learned on out-of-fold predictions: auto (affine with
    /// --fitness, otherwise none), none, shift or affine.
    #[arg(long, default_value = "auto", value_parser = ["auto", "none", "shift", "affine"])]
    pub calibration: String,
    /// iid or forward (chronological expanding-window validation).
    #[arg(long, default_value = "iid", value_parser = ["iid", "forward"])]
    pub validation: String,
    /// Rows excluded between each forward training window and its validation block.
    #[arg(long, default_value_t = 0)]
    pub gap: usize,
}

#[derive(Args, Clone)]
pub struct CommonArgs {
    /// auto, cpu or cuda:N. An explicit device never falls back.
    #[arg(long, default_value = "auto")]
    pub device: String,
    /// Worker threads for numerical work; 0 uses the available parallelism.
    #[arg(long, default_value_t = 0)]
    pub threads: usize,
    /// Stop starting new selection folds after this many seconds.
    #[arg(long)]
    pub time_limit_seconds: Option<f64>,
    /// Largest input CSV accepted, in MiB.
    #[arg(long, default_value_t = 1024)]
    pub max_input_mib: u64,
    /// Most data rows accepted per input table.
    #[arg(long, default_value_t = 2_000_000)]
    pub max_rows: usize,
    /// Write the run receipt (JSON) here as well.
    #[arg(long)]
    pub receipt: Option<PathBuf>,
    /// Print the run receipt as JSON on stdout instead of a summary on stderr.
    #[arg(long)]
    pub json: bool,
}

#[derive(Subcommand)]
pub enum FitnessCommand {
    /// List built-in names with their exact default parameters, directions and descriptions.
    List,
    /// Check a specification (name, name:key=value, JSON or @path.json) and print its identity.
    Validate {
        /// The specification, as `--fitness` accepts it.
        spec: String,
    },
    /// Score a JSON file containing {fitness, targets, predictions[, columns]}.
    Evaluate {
        #[arg(long)]
        input: PathBuf,
    },
    /// Print the JSON Schema of objective specifications.
    Schema,
}

#[cfg(test)]
mod serve_tests {
    use super::{Cli, Command};
    use clap::Parser;

    #[test]
    fn serve_accepts_the_specified_lifecycle_flags() {
        assert!(matches!(
            Cli::try_parse_from(["jopedime", "serve"]).unwrap().command,
            Command::Serve {
                status: false,
                stop: false,
                idle_exit: None
            }
        ));
        assert!(matches!(
            Cli::try_parse_from(["jopedime", "serve", "--status"])
                .unwrap()
                .command,
            Command::Serve {
                status: true,
                stop: false,
                idle_exit: None
            }
        ));
        assert!(matches!(
            Cli::try_parse_from(["jopedime", "serve", "--stop"])
                .unwrap()
                .command,
            Command::Serve {
                status: false,
                stop: true,
                idle_exit: None
            }
        ));
        assert!(matches!(
            Cli::try_parse_from(["jopedime", "serve", "--idle-exit", "2s"])
                .unwrap()
                .command,
            Command::Serve { status: false, stop: false, idle_exit: Some(value) } if value == "2s"
        ));
        assert!(Cli::try_parse_from(["jopedime", "serve", "--status", "--stop"]).is_err());
        assert!(Cli::try_parse_from(["jopedime", "serve", "--stop", "--idle-exit", "2s"]).is_err());
    }
}
