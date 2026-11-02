use super::*;

#[cfg(feature = "client")]
#[derive(Subcommand)]
pub enum ControlAction {
    Pause,
    Continue,
    Cancel,
    Effort {
        #[arg(value_parser = ["fast", "medium", "high", "ultra"])]
        level: String,
    },
}

#[cfg(feature = "client")]
#[derive(Subcommand)]
pub enum RunsCommand {
    List,
    Delete {
        run: String,
        #[arg(long)]
        idempotency_key: Option<String>,
    },
    ExportFeedback {
        run: String,
        #[arg(long)]
        out: PathBuf,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum ExportKindArg {
    Model,
    Source,
    Bundle,
}
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum LanguageArg {
    Rust,
    Python,
    Javascript,
    Typescript,
}
#[cfg(feature = "mcp")]
#[derive(Subcommand)]
pub enum DatasetCommand {
    Add {
        path: PathBuf,
        #[arg(long)]
        name: Option<String>,
        #[arg(long)]
        idempotency_key: Option<String>,
        #[arg(long)]
        json: bool,
    },
    List {
        #[arg(long)]
        json: bool,
    },
    Show {
        id: String,
        #[arg(long)]
        json: bool,
    },
    Delete {
        id: String,
        #[arg(long)]
        idempotency_key: Option<String>,
        #[arg(long)]
        json: bool,
    },
}
#[cfg(feature = "mcp")]
#[derive(Subcommand)]
pub enum ObjectiveCommand {
    Validate {
        spec: String,
        #[arg(long)]
        json: bool,
    },
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
