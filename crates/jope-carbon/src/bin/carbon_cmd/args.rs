//! Clap CLI schema (`Cli`, `Command`, `CommonTrainArgs`, `FitnessCommand`) and dynamic shell completions.

use super::*;

const LONG_VERSION: &str = concat!(
    env!("CARBON_SEMVER"),
    " (",
    env!("CARBON_RELEASE_TAG"),
    ", git ",
    env!("CARBON_GIT_SHA"),
    ", ",
    env!("CARBON_BUILD_TARGET"),
    ", ",
    env!("CARBON_BUILD_PROFILE"),
    ")"
);

#[derive(Debug, Parser)]
#[command(
    name = "carbon",
    version = LONG_VERSION,
    subcommand_precedence_over_arg = true,
    about = "Carbon zero-knowledge serverless CLI client, benchmark & red-team harness, Lambda runtime, Enterprise server, and CI emulator"
)]
pub(crate) struct Cli {
    /// Increase diagnostic verbosity on stderr (`-v` = debug, `-vv` = trace; overrides `CARBON_LOG`).
    #[arg(short = 'v', long = "verbose", action = ArgAction::Count, global = true)]
    pub(crate) verbose: u8,

    /// Suppress interactive progress and diagnostic messages on stderr (`-q` / `--quiet`).
    #[arg(short = 'q', long = "quiet", global = true)]
    pub(crate) quiet: bool,

    /// Diagnostic output format on stderr (`text` or `json`).
    #[arg(long = "log-format", default_value = "text", global = true)]
    pub(crate) log_format: String,

    /// ANSI color policy (`auto`, `always`, or `never`; respects `NO_COLOR`).
    #[arg(long = "color", default_value = "auto", global = true)]
    pub(crate) color: String,

    /// Positional shorthand (`carbon data.csv` / `data.csv.gz` / `data.jsonl` / `-`): conditions & encrypts
    /// client-side, runs a `fast` zero-knowledge cloud job, decrypts predictions, and shreds remote blobs.
    #[arg(value_name = "CSV_FILE")]
    pub(crate) csv_file: Option<PathBuf>,

    /// Retain a `0600` local model handle (`$HOME/.config/jopedime/runs/<id>.json`) and remote ciphertext on positional `<CSV_FILE>` shorthand.
    #[arg(long, conflicts_with = "ephemeral")]
    pub(crate) keep: bool,

    /// Cryptographically shred remote S3/CAS ciphertext and skip saving a local model handle upon completion.
    #[arg(long, global = true)]
    pub(crate) ephemeral: bool,

    #[command(subcommand)]
    pub(crate) command: Option<Command>,
}

#[derive(Debug, Subcommand)]
pub(crate) enum FitnessCommand {
    /// List all 45 built-in regression objectives, directions, order requirements, and descriptions.
    List {
        /// Emit pure JSON array instead of formatted table.
        #[arg(long)]
        json: bool,
    },
    /// Parse, canonicalize, and compute the domain-separated SHA-256 identity of a fitness spec or `@file.json`.
    Validate {
        /// Preset (`rmse`, `pinball(0.9)`), inline JSON (`{"metric":...}`), or `@path/to/objective.json`.
        #[arg(value_name = "SPEC_OR_AT_FILE")]
        spec: String,
        /// Emit pure JSON on stdout (default).
        #[arg(long)]
        json: bool,
    },
    /// Emit the committed `jope.regression-fitness.v1` JSON Schema document.
    Schema {
        /// Emit pure JSON on stdout (default).
        #[arg(long)]
        json: bool,
    },
    /// Score aligned `(target, pred)` columns (plus any `col(...)` auxiliary columns) from a CSV/JSONL file.
    Evaluate {
        /// Preset, inline JSON, or `@path/to/objective.json`.
        #[arg(long, visible_alias = "fitness")]
        spec: String,
        /// Input CSV/TSV/JSONL file containing target, prediction, and optional auxiliary columns.
        #[arg(long)]
        csv: PathBuf,
        /// Ground-truth target column name.
        #[arg(long)]
        target: String,
        /// Prediction column name.
        #[arg(long)]
        pred: String,
        /// Emit pure JSON on stdout (default).
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Default, Clone, clap::Args)]
pub(crate) struct HandleTargetArgs {
    /// Optional named Carbon cloud (`dev` or `prod`); accepted for CLI flag symmetry and stateless `--run` operations.
    #[arg(long)]
    pub(crate) cloud: Option<String>,
    /// Optional path or HTTPS URL overriding the public `carbon-clouds.json` manifest.
    #[arg(long)]
    pub(crate) clouds_manifest: Option<String>,
    /// Optional explicit endpoint URL overriding saved-handle or cloud discovery.
    #[arg(long)]
    pub(crate) endpoint: Option<String>,
    /// Optional explicit tenant ID (used for stateless `carbon shred --run <ID>` when no local handle exists).
    #[arg(long)]
    pub(crate) tenant: Option<String>,
    /// Accept `--free` for CLI flag symmetry across subcommands.
    #[arg(long, conflicts_with_all = ["pro", "tier"])]
    pub(crate) free: bool,
    /// Accept `--pro` for CLI flag symmetry across subcommands.
    #[arg(long, conflicts_with_all = ["free", "tier"])]
    pub(crate) pro: bool,
    /// Optional tier hint (`free`, `pro`, or `enterprise`) for CLI flag symmetry across subcommands.
    #[arg(long, conflicts_with_all = ["free", "pro"])]
    pub(crate) tier: Option<String>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct CommonTrainArgs {
    #[arg(long, value_delimiter = ',')]
    pub(crate) endpoint: Vec<String>,
    /// Named public Carbon cloud (`dev` or `prod`; defaults to `prod` when no endpoint/env is given).
    #[arg(long)]
    pub(crate) cloud: Option<String>,
    /// Optional path or HTTPS URL overriding the public `carbon-clouds.json` manifest.
    #[arg(long)]
    pub(crate) clouds_manifest: Option<String>,
    /// Optional stack environment (`dev`, `staging`, `prod`) to auto-resolve endpoint & vault token.
    #[arg(long = "env")]
    pub(crate) environment: Option<String>,
    #[arg(long, default_value = "us-east-2")]
    pub(crate) region: String,
    /// Optional explicit tenant ID (automatically extracted from `ctk1.*` token claims when omitted).
    #[arg(long)]
    pub(crate) tenant: Option<String>,
    /// Choose Free anonymous tier (`<= 5,000` rows, `fast`/`medium`).
    #[arg(long, conflicts_with_all = ["pro", "tier"])]
    pub(crate) free: bool,
    /// Choose authenticated Pro tier (`<= 100,000` rows, `fast`..`ultra` multi-wave + `HybridFanOut`).
    #[arg(long, conflicts_with_all = ["free", "tier"])]
    pub(crate) pro: bool,
    /// Explicit tier (`free`, `pro`, or `enterprise`).
    #[arg(long, conflicts_with_all = ["free", "pro"])]
    pub(crate) tier: Option<String>,
    /// Signed `ctk1.*` capability token for Pro/Enterprise runs.
    #[arg(long, conflicts_with_all = ["token_file", "token_stdin"])]
    pub(crate) token: Option<String>,
    /// File containing a signed `ctk1.*` capability token (`-` reads stdin).
    #[arg(long, conflicts_with_all = ["token", "token_stdin"])]
    pub(crate) token_file: Option<PathBuf>,
    /// Read signed `ctk1.*` capability token from standard input.
    #[arg(long, conflicts_with_all = ["token", "token_file"])]
    pub(crate) token_stdin: bool,
    /// Boot an ephemeral local multi-cell Lambda emulator offline instead of connecting to a remote cloud.
    #[arg(long, conflicts_with_all = ["endpoint", "cloud", "environment"])]
    pub(crate) local: bool,
    #[arg(long, default_value = "fast")]
    pub(crate) effort: String,
    /// Regression objective preset (`rmse`, `mae`, `pinball(0.9)`), inline JSON, or `@file.json` (`--fitness` alias).
    #[arg(long, visible_alias = "fitness", default_value = "rmse")]
    pub(crate) objective: String,
    #[arg(long, default_value_t = 42)]
    pub(crate) seed: u64,
    /// Input CSV has no header row (columns are auto-named `f0..fk`; `--target -1` selects final column).
    #[arg(long)]
    pub(crate) no_header: bool,
    /// Override delimiter character (`,`, `;`, `tab` / `\t`, or `|`; auto-detected when omitted).
    #[arg(long)]
    pub(crate) delimiter: Option<String>,
    /// Holdout split strategy (`random` deterministic hash split, or `forward` chronological tail split).
    #[arg(long)]
    pub(crate) split: Option<String>,
    /// Fold-safe post-hoc prediction calibration (`auto`, `none`, `shift`, or `affine`).
    #[arg(long)]
    pub(crate) calibration: Option<String>,
    /// Additional comma-separated custom missing-value tokens (`--na-values -999,MISSING`).
    #[arg(long, value_delimiter = ',')]
    pub(crate) na_values: Vec<String>,
    /// Coerce `Inf` / `-Inf` cells to missing (`NaN`) instead of failing closed.
    #[arg(long)]
    pub(crate) inf_as_nan: bool,
    /// Embed the active `ctk1.*` capability token inside an explicit `--save` file (default strips token outside `$CARBON_CONFIG_DIR/runs/`).
    #[arg(long)]
    pub(crate) embed_token: bool,
    /// Validate conditioning, objective, admission limits, and execution cost locally with zero network I/O.
    #[arg(long)]
    pub(crate) dry_run: bool,
    /// Permit unencrypted dataset upload to a non-loopback self-hosted `carbon serve` endpoint without `--enable-user-envelope`.
    #[arg(long)]
    pub(crate) allow_unencrypted_enterprise: bool,
    /// Force pure JSON output on stdout (`--format json` shorthand).
    #[arg(long)]
    pub(crate) json: bool,
}

#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    /// Print build, git SHA, release tag, target triple, and compiled role readiness as JSON.
    Version,
    /// Cryptographically verify the installed `carbon` binary SHA-256, `carbon-clouds.json` P-256 KEK pins,
    /// embedded release public key fingerprint, and ECDSA-P256 `SHA256SUMS.txt.sig` signature.
    SelfVerify {
        /// Optional path to `SHA256SUMS.txt` (defaults to `$CARBON_CONFIG_DIR/SHA256SUMS.txt` if present).
        #[arg(long, alias = "sums")]
        manifest: Option<PathBuf>,
        /// Optional path to base64-encoded ECDSA-P256 `SHA256SUMS.txt.sig` (defaults to `<manifest>.sig`).
        #[arg(long, alias = "sig")]
        signature: Option<PathBuf>,
        /// Optional explicit `carbon-clouds.json` path override.
        #[arg(long)]
        clouds_manifest: Option<PathBuf>,
        /// Optional PEM public key override for offline CI/E2E fixture testing only.
        #[arg(long)]
        allow_test_pubkey: Option<PathBuf>,
        /// Require signed `SHA256SUMS.txt` + `SHA256SUMS.txt.sig` verification even when `$CARBON_CONFIG_DIR` is empty.
        #[arg(long, conflicts_with = "binary_only")]
        require_manifest: bool,
        /// Verify only embedded release key fingerprint without requiring `SHA256SUMS.txt`.
        #[arg(long, conflicts_with = "require_manifest")]
        binary_only: bool,
    },
    /// Cryptographically verify `r882`, cross-compile ARM64 `carbon-edge` (`CT3` isolated) + `carbon-worker`,
    /// and emit deterministic reproducible `.zip` bundles + `release-manifest.json`.
    #[cfg(feature = "engine")]
    Package {
        /// Directory containing extracted `r882` release assets (`models/bundles/r882.json` + cosign files).
        #[arg(long)]
        r882_dir: PathBuf,
        /// Output directory for `carbon-edge.zip`, `carbon-worker.zip`, `r882-layer.zip`, and `release-manifest.json`.
        #[arg(long)]
        out_dir: PathBuf,
        /// Optional prebuilt `aarch64-unknown-linux-musl` `carbon-edge` binary path.
        #[arg(long)]
        edge_bin: Option<PathBuf>,
        /// Optional prebuilt `aarch64-unknown-linux-musl` `carbon` binary path.
        #[arg(long)]
        worker_bin: Option<PathBuf>,
        /// Max Cargo build jobs when cross-compiling (`-j`).
        #[arg(long, default_value_t = 8)]
        jobs: usize,
        /// Also cross-compile and package static `x86_64` + `aarch64` musl `carbon` CLI tarballs + `CLI-SHA256SUMS`.
        #[arg(long)]
        include_cli_binaries: bool,
    },
    /// Deploy a packaged Carbon cell via AWS CloudFormation ChangeSet + Secrets Manager + `:live` aliases,
    /// followed by mandatory `/ping`, `/version`, and 16-vector live `carbon redteam` verification, or run
    /// read-only `--check-drift` verification across regions.
    #[cfg(feature = "engine")]
    Deploy {
        /// Path to `release-manifest.json` produced by `carbon package` (optional when `--check-drift` is set).
        #[arg(long, required_unless_present = "check_drift")]
        manifest: Option<PathBuf>,
        /// Run read-only CloudFormation + `:live` alias + `$LATEST` + P-256 KEK pin drift verification without mutating AWS.
        #[arg(long)]
        check_drift: bool,
        /// Optional path to `carbon-clouds.json` used to cross-check public endpoints and KEK pins under `--check-drift`.
        #[arg(long)]
        clouds_manifest: Option<PathBuf>,
        /// Target environment (`dev`, `staging`, or `prod`).
        #[arg(long = "env", default_value = "dev")]
        environment: String,
        /// Comma-separated AWS regions or wave preset (`canary`, `wave1`, `all-17`).
        #[arg(long, value_delimiter = ',', default_value = "us-east-2")]
        regions: Vec<String>,
        /// Optional regional S3 artifacts bucket override.
        #[arg(long)]
        artifacts_bucket: Option<String>,
        /// Initialize CSPRNG cell secrets in AWS Secrets Manager (`jopedime/carbon/<env>/cell-secrets`) if absent.
        #[arg(long)]
        bootstrap_secrets: bool,
        /// Cleanly decommission legacy un-stacked prototype Lambdas/roles before initial stack creation.
        #[arg(long)]
        decommission_unstacked_prototype: bool,
        /// Run a local Lambda Custom Runtime + SigV4 S3/SecretsManager preflight before touching AWS.
        #[arg(long)]
        verify_local_lambda: bool,
        /// Validate template & preview CloudFormation ChangeSet resource diff without executing.
        #[arg(long)]
        dry_run: bool,
        /// Skip post-deploy 16-vector red-team suite (not recommended outside offline diagnostics).
        #[arg(long)]
        skip_redteam: bool,
        /// Optional comma-separated Base64 uncompressed P-256 public keys (`[active, previous]`, or `CLEAR`) synced to regional `edge-secrets` for asymmetric `ctk2.*` verification.
        #[arg(long)]
        app_auth_pubkeys_b64: Option<String>,
        /// Optional path to write the JSON deployment receipt.
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
    /// Atomically roll `{edge,engine,eval}:live` aliases back to prior retained immutable versions (`< 1 s`).
    #[cfg(feature = "engine")]
    Rollback {
        /// Target environment (`dev`, `staging`, or `prod`).
        #[arg(long = "env", default_value = "dev")]
        environment: String,
        /// Comma-separated AWS regions or wave preset (`canary`, `wave1`, `all-17`; `--region` alias supported).
        #[arg(
            long,
            alias = "region",
            value_delimiter = ',',
            default_value = "us-east-2"
        )]
        regions: Vec<String>,
        /// Optional explicit `<edge_ver>,<engine_ver>,<eval_ver>` override.
        #[arg(long, value_delimiter = ',')]
        to_versions: Vec<String>,
    },
    /// Query live CloudFormation stacks + Secrets Manager + `/v1/carbon/ping` across regions and emit
    /// a multi-region pinned `carbon-clouds.json` manifest (`jope-carbon-clouds-v2`).
    #[cfg(feature = "engine")]
    CloudsManifest {
        /// Output file path for `carbon-clouds.json`.
        #[arg(long)]
        out: PathBuf,
        /// Comma-separated `dev` regions or preset (`us-east-2` by default).
        #[arg(long, value_delimiter = ',', default_value = "us-east-2")]
        dev_regions: Vec<String>,
        /// Comma-separated `prod` regions or preset (`wave1` by default).
        #[arg(long, value_delimiter = ',', default_value = "wave1")]
        prod_regions: Vec<String>,
    },
    /// Register a persistent 30-day Free-tier account (`<= 5,000` rows, `fast`/`medium`) and store
    /// its signed `ctk1.*` capability token in `0600` `$HOME/.config/jopedime/carbon-credentials.json`.
    Register {
        /// Named public Carbon cloud (`dev` or `prod`, defaults to `prod`).
        #[arg(long, default_value = "prod")]
        cloud: String,
        /// Explicitly select the Free tier (`default = true`).
        #[arg(long)]
        free: bool,
        /// Optional alphanumeric handle (`[A-Za-z0-9_-]{1,24}`) embedded in `free_<label>_<ulid>`.
        #[arg(long)]
        label: Option<String>,
        /// Optional path or HTTPS URL overriding the public `carbon-clouds.json` manifest.
        #[arg(long)]
        clouds_manifest: Option<String>,
        /// Optional explicit endpoint URL overriding `--cloud`.
        #[arg(long)]
        endpoint: Option<String>,
    },
    /// Store a `ctk1.*` Pro/Enterprise capability token in `0600` `$HOME/.config/jopedime/carbon-credentials.json`.
    Login {
        /// Named public Carbon cloud (`dev` or `prod`, defaults to `prod`).
        #[arg(long, default_value = "prod")]
        cloud: String,
        /// Optional path or HTTPS URL overriding the public `carbon-clouds.json` manifest.
        #[arg(long)]
        clouds_manifest: Option<String>,
        /// Optional explicit endpoint URL overriding `--cloud`.
        #[arg(long)]
        endpoint: Option<String>,
        /// Signed `ctk1.*` capability token string.
        #[arg(long, conflicts_with_all = ["token_file", "token_stdin"])]
        token: Option<String>,
        /// Path to a file containing the `ctk1.*` capability token (`-` reads stdin).
        #[arg(long, conflicts_with_all = ["token", "token_stdin"])]
        token_file: Option<PathBuf>,
        /// Read the `ctk1.*` capability token from standard input.
        #[arg(long, conflicts_with_all = ["token", "token_file"])]
        token_stdin: bool,
    },
    /// Inspect active Carbon cloud, authentication mode (`anonymous_free` vs `capability_token`), `tenant_id`, and TTL.
    Whoami {
        /// Named public Carbon cloud (`dev` or `prod`, defaults to `prod`).
        #[arg(long, default_value = "prod")]
        cloud: String,
        /// Optional path or HTTPS URL overriding the public `carbon-clouds.json` manifest.
        #[arg(long)]
        clouds_manifest: Option<String>,
        /// Optional inline `ctk1.*` capability token to inspect instead of stored credentials.
        #[arg(long, conflicts_with_all = ["token_file", "token_stdin"])]
        token: Option<String>,
        /// Optional file containing a `ctk1.*` capability token to inspect.
        #[arg(long, conflicts_with_all = ["token", "token_stdin"])]
        token_file: Option<PathBuf>,
        /// Read `ctk1.*` capability token from standard input.
        #[arg(long, conflicts_with_all = ["token", "token_file"])]
        token_stdin: bool,
        /// Emit pure JSON on stdout (default).
        #[arg(long)]
        json: bool,
    },
    /// Remove stored capability token(s) from `$HOME/.config/jopedime/carbon-credentials.json`.
    Logout {
        /// Cloud name (`dev` or `prod`) to remove; defaults to `prod` unless `--all` is passed.
        #[arg(long, conflicts_with = "all")]
        cloud: Option<String>,
        /// Remove stored capability tokens for all clouds.
        #[arg(long)]
        all: bool,
    },
    /// Verify local binary roles, `0700`/`0600` config & credential permissions, regional `ECDH P-256`
    /// KEK pins, stored capability token TTLs, and live `/v1/carbon/ping` RTT across clouds.
    Doctor {
        /// Target Carbon cloud (`prod`, `dev`, or `all`; defaults to `all`).
        #[arg(long, default_value = "all")]
        cloud: String,
        /// Optional path or HTTPS URL overriding the public `carbon-clouds.json` manifest.
        #[arg(long)]
        clouds_manifest: Option<String>,
        /// Skip live network `/v1/carbon/ping` probes (validate local config, perms, and KEK pins only).
        #[arg(long)]
        offline: bool,
        /// Override ECDSA-P256 release public key PEM for CI/E2E test fixtures only.
        #[arg(long)]
        allow_test_pubkey: Option<PathBuf>,
    },
    /// Probe `/v1/carbon/ping` RTT, region, and zero-knowledge `engine_kek_pubkey_b64` with zero AWS credentials.
    Ping {
        #[arg(long, value_delimiter = ',')]
        endpoint: Vec<String>,
        /// Named public Carbon cloud (`dev` or `prod`; defaults to `prod` when no endpoint/env is given).
        #[arg(long)]
        cloud: Option<String>,
        /// Optional path or HTTPS URL overriding the public `carbon-clouds.json` manifest.
        #[arg(long)]
        clouds_manifest: Option<String>,
        /// Optional AWS CloudFormation stack environment (`dev`, `staging`, `prod`).
        #[arg(long = "env")]
        environment: Option<String>,
        #[arg(long, default_value = "us-east-2")]
        region: String,
    },
    /// Condition & encrypt a local CSV (or synthetic matrix), upload via size-pinned presigned URL,
    /// run `r882`-guided GP waves, and execute E2E encrypted prediction.
    Run {
        #[arg(long)]
        csv: Option<PathBuf>,
        /// Target column name (defaults to 'y', or auto-selects 'target' when 'y' is absent).
        #[arg(long)]
        target: Option<String>,
        #[arg(long, default_value_t = 240)]
        synthetic_rows: usize,
        #[arg(long, default_value_t = 6)]
        features: usize,
        #[arg(long)]
        stop_after_ms: Option<u64>,
        #[arg(long)]
        resume: bool,
        #[arg(long, default_value_t = 5)]
        predict_rows: usize,
        /// Persist a `0600` zero-knowledge model handle (`model.carbon.json`) for subsequent `carbon predict`.
        #[arg(long)]
        save: Option<PathBuf>,
        /// Output format on stdout (`auto`, `json`, or `pretty`).
        #[arg(long, default_value = "auto")]
        format: String,
        /// Cryptographically shred (`DELETE /v1/carbon/runs/{id}`) blobs and state upon completion.
        #[arg(long)]
        shred: bool,
        /// Force live wave progress and human summary card on stderr even when stderr is not a TTY.
        #[arg(long)]
        progress: bool,
        #[command(flatten)]
        common: CommonTrainArgs,
    },
    /// Train a persistent zero-knowledge Carbon model on a CSV/TSV/JSONL file and save a `0600` local handle
    /// (`--save model.carbon.json` and `$HOME/.config/jopedime/runs/<run_id>.json`) for `carbon predict`.
    Fit {
        /// Path to training CSV/TSV/JSONL/compressed file (or `-` for stdin).
        #[arg(value_name = "CSV_FILE", required_unless_present = "csv_flag")]
        csv: Option<PathBuf>,
        /// Optional `--csv <PATH>` flag alias for positional `CSV_FILE`.
        #[arg(long = "csv", value_name = "CSV_FILE", conflicts_with = "csv")]
        csv_flag: Option<PathBuf>,
        /// Target column name (or `-1` / `last` when `--no-header` is set).
        #[arg(long, short = 't')]
        target: Option<String>,
        /// Output path for the `0600` zero-knowledge model handle (e.g. `model.carbon.json`).
        #[arg(long, short = 's')]
        save: Option<PathBuf>,
        /// Output format on stdout (`auto`, `pretty`, or `json`).
        #[arg(long, default_value = "auto")]
        format: String,
        #[command(flatten)]
        common: CommonTrainArgs,
    },
    /// Score an unlabeled (or labeled) CSV/TSV file in zero-knowledge mode using a saved model handle.
    Predict {
        /// Path to CSV/TSV file to score.
        #[arg(value_name = "CSV_FILE", required_unless_present = "csv_flag")]
        csv: Option<PathBuf>,
        /// Optional `--csv <PATH>` flag alias for positional `CSV_FILE`.
        #[arg(long = "csv", value_name = "CSV_FILE", conflicts_with = "csv")]
        csv_flag: Option<PathBuf>,
        /// Path to a saved `0600` model handle (`--save model.carbon.json` from `carbon fit`).
        #[arg(long, short = 'm', conflicts_with = "run")]
        model: Option<PathBuf>,
        /// Saved `run_id` (or `latest`) from `$HOME/.config/jopedime/runs/` when `--model` is omitted.
        #[arg(long, visible_alias = "run-id", conflicts_with = "model")]
        run: Option<String>,
        /// Optional output CSV path (`prediction` column, or `<id>,prediction` / `<inputs...>,prediction`).
        #[arg(long, short = 'o')]
        out: Option<PathBuf>,
        /// Optional ID column from the input CSV to carry into `--out`.
        #[arg(long)]
        id_column: Option<String>,
        /// Include all original input columns alongside `prediction` in `--out`.
        #[arg(long)]
        include_inputs: bool,
        /// Input CSV has no header row.
        #[arg(long)]
        no_header: bool,
        /// Override delimiter character (`,`, `;`, `tab` / `\t`, or `|`).
        #[arg(long)]
        delimiter: Option<String>,
        /// Optional override `ctk1.*` capability token.
        #[arg(long, conflicts_with_all = ["token_file", "token_stdin"])]
        token: Option<String>,
        #[arg(long, conflicts_with_all = ["token", "token_stdin"])]
        token_file: Option<PathBuf>,
        /// Read override `ctk1.*` capability token from standard input.
        #[arg(long, conflicts_with_all = ["token", "token_file"])]
        token_stdin: bool,
        /// Output format on stdout (`auto`, `pretty`, or `json`).
        #[arg(long, default_value = "auto")]
        format: String,
        /// Emit pure JSON on stdout.
        #[arg(long)]
        json: bool,
        #[command(flatten)]
        target_opts: HandleTargetArgs,
    },
    /// Instant zero-network local dataset preflight: schema types, missingness, target stats, and AWS cost/time preview.
    Inspect {
        /// Path to CSV/TSV file to inspect locally (no bytes leave your machine).
        #[arg(value_name = "CSV_FILE", required_unless_present = "csv_flag")]
        csv: Option<PathBuf>,
        /// Optional `--csv <PATH>` flag alias for positional `CSV_FILE`.
        #[arg(long = "csv", value_name = "CSV_FILE", conflicts_with = "csv")]
        csv_flag: Option<PathBuf>,
        /// Optional target column name (auto-detected when omitted).
        #[arg(long, short = 't')]
        target: Option<String>,
        /// Input CSV has no header row.
        #[arg(long)]
        no_header: bool,
        /// Override delimiter character (`,`, `;`, `tab` / `\t`, or `|`).
        #[arg(long)]
        delimiter: Option<String>,
        /// Product tier for admission & cost estimation (`free`, `pro`, or `enterprise`).
        #[arg(long, default_value = "pro")]
        tier: String,
        /// Output format on stdout (`auto`, `pretty`, or `json`).
        #[arg(long, default_value = "auto")]
        format: String,
        /// Emit pure JSON on stdout.
        #[arg(long)]
        json: bool,
    },
    /// List locally saved `0600` zero-knowledge Carbon model handles (`$HOME/.config/jopedime/runs/`).
    Runs {
        /// Emit pure JSON even on interactive terminals.
        #[arg(long)]
        json: bool,
    },
    /// Query live remote cloud status (`/v1/carbon/runs/{id}`) for a saved zero-knowledge model handle.
    Status {
        #[arg(long, short = 'm', conflicts_with = "run")]
        model: Option<PathBuf>,
        #[arg(long, visible_alias = "run-id", conflicts_with = "model")]
        run: Option<String>,
        #[arg(long, conflicts_with_all = ["token_file", "token_stdin"])]
        token: Option<String>,
        #[arg(long, conflicts_with_all = ["token", "token_stdin"])]
        token_file: Option<PathBuf>,
        #[arg(long, conflicts_with_all = ["token", "token_file"])]
        token_stdin: bool,
        #[arg(long)]
        json: bool,
        #[command(flatten)]
        target_opts: HandleTargetArgs,
    },
    /// Cryptographically shred remote S3/CAS objects (`DELETE /v1/carbon/runs/{id}`) AND zero-overwrite local key handles.
    Shred {
        #[arg(long, short = 'm', conflicts_with = "run")]
        model: Option<PathBuf>,
        #[arg(long, visible_alias = "run-id", conflicts_with = "model")]
        run: Option<String>,
        #[arg(long, conflicts_with_all = ["token_file", "token_stdin"])]
        token: Option<String>,
        #[arg(long, conflicts_with_all = ["token", "token_stdin"])]
        token_file: Option<PathBuf>,
        #[arg(long, conflicts_with_all = ["token", "token_file"])]
        token_stdin: bool,
        #[arg(long)]
        json: bool,
        #[command(flatten)]
        target_opts: HandleTargetArgs,
    },
    /// Inspect, validate, emit schema, or locally evaluate regression objectives (`--fitness`).
    Fitness {
        #[command(subcommand)]
        action: FitnessCommand,
    },
    /// Compute zero-knowledge holdout permutation feature importances (`POST /v1/carbon/runs/{id}/explain`).
    Explain {
        /// Path to a saved `0600` model handle (`--save model.carbon.json` from `carbon fit`).
        #[arg(long, short = 'm', conflicts_with = "run")]
        model: Option<PathBuf>,
        /// Saved `run_id` (or `latest`) from `$HOME/.config/jopedime/runs/` when `--model` is omitted.
        #[arg(long, visible_alias = "run-id", conflicts_with = "model")]
        run: Option<String>,
        /// Maximum number of top features to return (`1..=64`, defaults to `12`).
        #[arg(long)]
        top_k: Option<u32>,
        #[arg(long, conflicts_with_all = ["token_file", "token_stdin"])]
        token: Option<String>,
        #[arg(long, conflicts_with_all = ["token", "token_stdin"])]
        token_file: Option<PathBuf>,
        #[arg(long, conflicts_with_all = ["token", "token_file"])]
        token_stdin: bool,
        /// Output format on stdout (`auto`, `pretty`, or `json`).
        #[arg(long, default_value = "auto")]
        format: String,
        /// Emit pure JSON on stdout.
        #[arg(long)]
        json: bool,
        #[command(flatten)]
        target_opts: HandleTargetArgs,
    },
    /// Export the winning symbolic GP champion to standalone zero-dependency predictor source code (`POST /v1/carbon/runs/{id}/export`).
    Export {
        /// Path to a saved `0600` model handle (`--save model.carbon.json` from `carbon fit`).
        #[arg(long, short = 'm', conflicts_with = "run")]
        model: Option<PathBuf>,
        /// Saved `run_id` (or `latest`) from `$HOME/.config/jopedime/runs/` when `--model` is omitted.
        #[arg(long, visible_alias = "run-id", conflicts_with = "model")]
        run: Option<String>,
        /// Target standalone language (`python`, `rust`, `rust-refittable`, `js`, `c`, `cpp`, or `model-object`).
        #[arg(long, default_value = "python")]
        lang: String,
        /// Optional output file path (or `-` to write source code to stdout).
        #[arg(long, short = 'o')]
        out: Option<PathBuf>,
        #[arg(long, conflicts_with_all = ["token_file", "token_stdin"])]
        token: Option<String>,
        #[arg(long, conflicts_with_all = ["token", "token_stdin"])]
        token_file: Option<PathBuf>,
        #[arg(long, conflicts_with_all = ["token", "token_file"])]
        token_stdin: bool,
        /// Emit structured JSON receipt on stdout instead of raw source text.
        #[arg(long)]
        json: bool,
        #[command(flatten)]
        target_opts: HandleTargetArgs,
    },
    /// Resume a paused multi-wave zero-knowledge Carbon run (`POST /v1/carbon/runs/{id}/resume`).
    Resume {
        /// Path to a saved `0600` model handle (`--save model.carbon.json` from `carbon fit`).
        #[arg(long, short = 'm', conflicts_with = "run")]
        model: Option<PathBuf>,
        /// Saved `run_id` (or `latest`) from `$HOME/.config/jopedime/runs/` when `--model` is omitted.
        #[arg(long, visible_alias = "run-id", conflicts_with = "model")]
        run: Option<String>,
        /// Optional updated search effort (`fast`, `medium`, `high`, or `ultra`).
        #[arg(long)]
        effort: Option<String>,
        /// Optional additional waves to run.
        #[arg(long)]
        additional_waves: Option<u32>,
        #[arg(long, conflicts_with_all = ["token_file", "token_stdin"])]
        token: Option<String>,
        #[arg(long, conflicts_with_all = ["token", "token_stdin"])]
        token_file: Option<PathBuf>,
        #[arg(long, conflicts_with_all = ["token", "token_file"])]
        token_stdin: bool,
        /// Emit pure JSON on stdout.
        #[arg(long)]
        json: bool,
        #[command(flatten)]
        target_opts: HandleTargetArgs,
    },
    /// Check or install ECDSA-P256 cryptographically signed Carbon CLI updates (`SHA256SUMS.txt.sig`).
    #[command(visible_aliases = ["self-update", "update"])]
    Upgrade {
        /// Verify signature and check whether a newer binary is available without modifying disk.
        #[arg(long)]
        check: bool,
        /// Force replacement even when running an operator/engine build (by default refuses downgrading operator builds to thin CLI).
        #[arg(long)]
        force: bool,
        /// Optional release tag to target (defaults to `latest`).
        #[arg(long)]
        version: Option<String>,
        /// Optional HTTPS/HTTP base URL or local directory override for release assets.
        #[arg(long)]
        base_url: Option<String>,
        /// Optional PEM public key override for offline CI/E2E testing only.
        #[arg(long)]
        allow_test_pubkey: Option<PathBuf>,
    },
    /// Remove the installed `carbon` binary and cached release manifests (preserving saved runs & credentials unless `--purge-all`).
    Uninstall {
        /// Keep `$CARBON_CONFIG_DIR` (`carbon-clouds.json`, `SHA256SUMS.txt`, credentials, and saved runs) intact; remove only the binary.
        #[arg(long, conflicts_with = "purge_all")]
        keep_data: bool,
        /// Permanently zero-overwrite and delete saved model handles (`runs/`) and `carbon-credentials.json` in addition to the binary.
        #[arg(long, conflicts_with = "keep_data")]
        purge_all: bool,
        /// Confirm destructive `--purge-all` without an interactive TTY prompt.
        #[arg(long, short = 'y')]
        yes: bool,
        /// Optional target binary path override (defaults to `CARBON_UNINSTALL_BIN_PATH` or `current_exe()`, refusing Cargo `target/` paths unless overridden).
        #[arg(long)]
        bin_path: Option<PathBuf>,
    },
    /// Emit deterministic shell tab-completion script (`bash`, `zsh`, `fish`, or `powershell`).
    Completions {
        /// Target shell (`bash`, `zsh`, `fish`, or `powershell`).
        #[arg(value_name = "SHELL")]
        shell: String,
    },
    /// Run the Free + Pro `r882` zero-knowledge live Carbon Lambda / Enterprise server benchmark suite
    /// (`ops` 5-scenario cloud suite or `smoke` / `standard` / `comprehensive` 11-comparator scientific suite).
    Bench {
        #[arg(long, value_delimiter = ',')]
        endpoint: Vec<String>,
        /// Named public Carbon cloud (`dev` or `prod`; defaults to `prod` when no endpoint/env is given).
        #[arg(long)]
        cloud: Option<String>,
        /// Optional path or HTTPS URL overriding the public `carbon-clouds.json` manifest.
        #[arg(long)]
        clouds_manifest: Option<String>,
        /// Optional stack environment (`dev`, `staging`, `prod`) to auto-resolve endpoint & vault token.
        #[arg(long = "env")]
        environment: Option<String>,
        #[arg(long, default_value = "us-east-2")]
        region: String,
        /// Benchmark suite: `ops` (default 5-scenario cloud lifecycle) or `smoke` | `standard` | `comprehensive` (head-to-head vs 11 tabular comparators).
        #[arg(long)]
        suite: Option<String>,
        /// Optional directory of real `<task>/{train.csv,test.csv,meta.json}` tasks (e.g. `datasets/test`) for Panel C.
        #[arg(long)]
        datasets_dir: Option<PathBuf>,
        /// Optional directory of user `*.csv` files to benchmark Carbon vs all 11 comparators on identical splits.
        #[arg(long)]
        csv_dir: Option<PathBuf>,
        /// Carbon search effort for scientific benchmark suites (`fast`, `medium`, `high`, or `ultra`).
        #[arg(long, default_value = "fast")]
        effort: String,
        /// Deterministic RNG seed for scientific benchmark suites.
        #[arg(long, default_value_t = 20261114)]
        seed: u64,
        /// Run only Free-tier compatible scenarios.
        #[arg(long, conflicts_with_all = ["pro", "tier"])]
        free: bool,
        /// Run full Free + Pro multi-wave & `HybridFanOut` benchmark suite.
        #[arg(long, conflicts_with_all = ["free", "tier"])]
        pro: bool,
        #[arg(long, conflicts_with_all = ["free", "pro"])]
        tier: Option<String>,
        /// Signed `ctk1.*` Pro capability token.
        #[arg(long, conflicts_with_all = ["token_file", "token_stdin"])]
        token: Option<String>,
        /// File containing a signed `ctk1.*` Pro capability token (`-` reads stdin).
        #[arg(long, conflicts_with_all = ["token", "token_stdin"])]
        token_file: Option<PathBuf>,
        /// Read signed `ctk1.*` Pro capability token from standard input.
        #[arg(long, conflicts_with_all = ["token", "token_file"])]
        token_stdin: bool,
        /// Optional explicit tenant ID (automatically extracted from `ctk1.*` token claims when omitted).
        #[arg(long)]
        tenant: Option<String>,
        #[arg(long)]
        json_out: Option<PathBuf>,
        #[arg(long)]
        markdown_out: Option<PathBuf>,
    },
    /// Run the 16-vector (`V-01`..`V-16`) adversarial zero-knowledge & security red-team suite.
    Redteam {
        /// Target Carbon endpoint URL (`https://...lambda-url...` or `http://host:port`).
        #[arg(long)]
        endpoint: Option<String>,
        /// Named public Carbon cloud (`dev` or `prod`) resolved without AWS credentials.
        #[arg(long)]
        cloud: Option<String>,
        /// Optional path or HTTPS URL overriding the public `carbon-clouds.json` manifest.
        #[arg(long)]
        clouds_manifest: Option<String>,
        /// Optional stack environment (`dev`, `staging`, `prod`) to auto-resolve endpoint.
        #[arg(long = "env")]
        environment: Option<String>,
        #[arg(long, default_value = "us-east-2")]
        region: String,
        #[arg(long)]
        json_out: Option<PathBuf>,
        #[arg(long)]
        markdown_out: Option<PathBuf>,
    },
    /// Mint a signed `ctk1.*` (Secrets Manager / HMAC) or asymmetric `ctk2.*` (`--signing-key-file`) capability token.
    TokenMint {
        #[arg(long)]
        tenant: String,
        #[arg(long, default_value = "pro")]
        tier: String,
        #[arg(long, default_value_t = 86_400)]
        ttl_secs: u64,
        #[arg(long)]
        admin: bool,
        #[arg(long = "env")]
        environment: Option<String>,
        #[arg(long, default_value = "us-east-2")]
        region: String,
        /// Optional PKCS#8 PEM or 64-char hex P-256 private key file to mint an asymmetric `ctk2.*` token offline (`ADR-0007`).
        #[arg(long, conflicts_with = "environment")]
        signing_key_file: Option<PathBuf>,
        /// Optional issuer claim (`iss`, defaults to `app.veox.ai`) when minting `ctk2.*` tokens.
        #[arg(long)]
        issuer: Option<String>,
        /// Optional key identifier (`kid`) embedded in `ctk2.*` claims.
        #[arg(long)]
        kid: Option<String>,
    },
    /// Generate an asymmetric NIST P-256 signing keypair for `app.veox.ai` per-user `ctk2.*` capability tokens (`ADR-0007`).
    TokenKeygen {
        /// Output path (`0600` PKCS#8 PEM) for the generated P-256 private key.
        #[arg(long, short = 'o')]
        out: PathBuf,
        /// Optional key identifier (`kid`) echoed in the public key JSON receipt.
        #[arg(long)]
        kid: Option<String>,
    },
    /// Compute the deterministic Solo vs Hybrid Fan-Out execution plan for a dataset shape.
    Plan {
        #[arg(long, default_value = "pro")]
        tier: String,
        #[arg(long, default_value = "high")]
        effort: String,
        #[arg(long)]
        total_rows: usize,
        #[arg(long)]
        fit_rows: usize,
        #[arg(long)]
        features: usize,
        /// Optional override per-candidate 1-vCPU latency in ms (calibrated automatically when omitted).
        #[arg(long)]
        ms_per_candidate: Option<f64>,
    },
    /// Evaluate the 3-Gate Spot-Hybrid break-even advisor on a regional monthly snapshot.
    SpotCheck {
        #[arg(long, default_value = "us-east-2")]
        region: String,
        #[arg(long)]
        monthly_spend_usd: f64,
        #[arg(long)]
        p20_vcpus: f64,
        #[arg(long)]
        mean_vcpus: f64,
        #[arg(long)]
        p95_vcpus: f64,
        #[arg(long, default_value_t = 0.01)]
        cold_start_rate: f64,
        #[arg(long, default_value_t = 1)]
        consecutive_months: usize,
    },
    /// Run the self-hosted VPC / on-premises Enterprise single-binary server (`no_limit()`).
    #[cfg(feature = "serve")]
    Serve {
        #[arg(long, default_value = "127.0.0.1:8088")]
        listen: SocketAddr,
        #[arg(long)]
        data_dir: PathBuf,
        /// Optional S3 CAS URL (`s3://bucket/prefix`, or `CARBON_CAS_URL` env var) for durable object storage.
        #[arg(long)]
        cas_url: Option<String>,
        /// Permit local filesystem CAS (`--data-dir`) when listening on a non-loopback address without `--cas-url`.
        #[arg(long)]
        allow_local_fs_dev: bool,
        /// Permit ephemeral CSPRNG seal/presign secrets when listening on a non-loopback address without env secrets.
        #[arg(long)]
        dev_ephemeral: bool,
        #[arg(long, default_value = "vpc-local-1")]
        region: String,
        #[arg(long)]
        enable_user_envelope: bool,
    },
    /// Boot the local multi-cell Lambda Runtime API + S3 CAS emulator (`--self-test` or foreground server).
    #[cfg(feature = "emu")]
    Emu {
        /// Optional working directory for local S3 CAS and worker scratch (uses an ephemeral config subdir when omitted).
        #[arg(long)]
        work_dir: Option<PathBuf>,
        /// Optional TCP address to bind the public Function URL gateway (`127.0.0.1:0` ephemeral port by default).
        #[arg(long)]
        listen: Option<SocketAddr>,
        /// Run automated E2E `CUEN\x01` + `r882` + `/predict` + `--shred` self-test and exit.
        #[arg(long)]
        self_test: bool,
        /// Spawn real `carbon-edge` and `carbon-worker` OS subprocesses over `AWS_LAMBDA_RUNTIME_API`.
        #[arg(long)]
        real_workers: bool,
        /// Optional path to `carbon-edge` executable when `--real-workers` is enabled.
        #[arg(long)]
        edge_bin: Option<PathBuf>,
        /// Optional path to `carbon-worker` executable when `--real-workers` is enabled.
        #[arg(long)]
        worker_bin: Option<PathBuf>,
    },
    /// Execute the clean-room `ubuntu:24.04` Fresh-Docker production CLI install + `dev`/`prod` 4-tier E2E qualification.
    #[cfg(feature = "emu")]
    #[command(name = "ci-live-4tier")]
    CiLive4Tier {
        /// Exact 40-hex Git source commit SHA to bind into `live-ci-4tier-docker.json`.
        #[arg(long)]
        source_sha: String,
        /// Output directory for `live-ci-4tier-docker.json` and `live-ci-4tier-docker.md`.
        #[arg(long)]
        out_dir: PathBuf,
        /// Require live AWS Pro `ctk1.*` tokens (`CARBON_CI_DEV_PRO_TOKEN` + `CARBON_CI_PROD_PRO_TOKEN`) for `high`/`ultra` (Option A).
        #[arg(long)]
        require_live_pro: bool,
        /// Repository root path (defaults to `.`).
        #[arg(long, default_value = ".")]
        repo_root: PathBuf,
    },
    /// Generate a sample CSV and/or run a 1-command zero-knowledge E2E live demo (`CUEN\x01` + `r882` + `/predict` + `--shred`).
    Quickstart {
        /// Optional path to write the generated 240-row x 6-feature nonlinear CSV (`feat_1..feat_6,target`).
        #[arg(long)]
        emit_csv: Option<PathBuf>,
        /// Only write `--emit-csv` and exit without running a cloud job.
        #[arg(long, requires = "emit_csv")]
        csv_only: bool,
        /// Named public Carbon cloud (`prod` or `dev`; defaults to `prod` when logged in, otherwise `dev` sandbox).
        #[arg(long)]
        cloud: Option<String>,
        /// Optional path or HTTPS URL overriding the public `carbon-clouds.json` manifest.
        #[arg(long)]
        clouds_manifest: Option<String>,
        /// Optional explicit endpoint URL(s) overriding `--cloud`.
        #[arg(long, value_delimiter = ',')]
        endpoint: Vec<String>,
        /// Boot an ephemeral local multi-cell Lambda emulator offline instead of connecting to a remote cloud.
        #[arg(long, conflicts_with = "endpoint")]
        local: bool,
        /// Search effort (`fast` or `medium` on Free; up to `ultra` when logged into Pro or `--local`).
        #[arg(long, default_value = "fast")]
        effort: String,
        /// Keep remote encrypted run artifacts instead of cryptographically shredding upon completion.
        #[arg(long)]
        no_shred: bool,
    },
    /// Generate the publication-grade Carbon engineering & testing specification HTML + CI quickstart receipts.
    #[cfg(feature = "emu")]
    SpecReport {
        /// Directory holding SHA-256-verified qualification receipts (`docs/carbon/evidence`).
        #[arg(long, default_value = "docs/carbon/evidence")]
        evidence_dir: PathBuf,
        /// Output directory for `carbon-engineering-testing-spec.html`, screenshots, and manifests.
        #[arg(long)]
        out_dir: PathBuf,
        /// Execute the live 8-suite exact-HEAD CI install, multi-region emulator, quickstart & redteam qualification.
        #[arg(long)]
        run_ci_qualification: bool,
        /// Optional pre-existing `ci-quickstart-qualification.json` when `--run-ci-qualification` is false.
        #[arg(long)]
        ci_qualification_in: Option<PathBuf>,
        /// Optional `live-ci-4tier-docker.json` receipt from the `carbon-live-e2e` CI job.
        #[arg(long)]
        live_ci_receipt: Option<PathBuf>,
        /// Optional `llvm-cov export` JSON or `code-coverage-summary.json` for Section 14 coverage matrix.
        #[arg(long)]
        code_coverage_export: Option<PathBuf>,
        /// Optional explicit Git source SHA to bind into the specification receipt.
        #[arg(long)]
        source_sha: Option<String>,
        /// Repository root containing `deploy/carbon/install.sh` (defaults to `.`).
        #[arg(long, default_value = ".")]
        repo_root: PathBuf,
    },
    /// Cryptographically verify `carbon-engineering-testing-spec.pdf`, all 8 PNG screenshots, and receipts.
    #[cfg(feature = "emu")]
    SpecVerify {
        /// Directory containing `spec-pdf-receipt.json`, `carbon-engineering-testing-spec.pdf`, and `screenshots/`.
        #[arg(long)]
        dir: PathBuf,
        /// Optional expected Git source SHA (`JOPE_SOURCE_SHA` in CI `aggregate`).
        #[arg(long)]
        expected_source_sha: Option<String>,
    },
    /// Execute the 28-check + 16-vector dual-cloud (`dev` + `prod`) live product coverage audit with `--shred`.
    #[cfg(feature = "engine")]
    CoverageAudit {
        /// Path to `carbon-clouds.json` manifest.
        #[arg(long, default_value = "docs/carbon/evidence/carbon-clouds.json")]
        clouds_manifest: PathBuf,
        /// Path to `docs/release/signing.pub`.
        #[arg(long, default_value = "docs/release/signing.pub")]
        signing_pub: PathBuf,
        /// Optional output JSON path for the secret-scrubbed dual-cloud receipt.
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
    /// Run the sealed `carbon engine` AWS Lambda Custom Runtime (`2018-06-01`) worker loop.
    #[cfg(feature = "engine")]
    Engine {
        #[arg(long)]
        work_dir: PathBuf,
    },
    /// Run the stateless `carbon eval` AWS Lambda Custom Runtime (`2018-06-01`) shard worker loop.
    #[cfg(feature = "eval")]
    Eval {
        #[arg(long)]
        scratch_dir: PathBuf,
    },
}

pub(crate) fn emit_shell_completions(shell: &str) -> Result<String, String> {
    let (subcommands, long_flags) = collect_dynamic_cli_vocabulary();
    let cmds_joined = subcommands.join(" ");
    let flags_joined = long_flags.join(" ");
    match shell.trim().to_ascii_lowercase().as_str() {
        "bash" => Ok(format!(
            "# bash completion for carbon (dynamically generated from clap Cli)\n_carbon_completions() {{\n    local cur prev cmds flags\n    COMPREPLY=()\n    cur=\"${{COMP_WORDS[COMP_CWORD]}}\"\n    prev=\"${{COMP_WORDS[COMP_CWORD-1]}}\"\n    cmds=\"{cmds}\"\n    flags=\"{flags}\"\n    if [[ ${{COMP_CWORD}} -eq 1 ]]; then\n        COMPREPLY=( $(compgen -W \"${{cmds}} ${{flags}}\" -- \"${{cur}}\") )\n        return 0\n    fi\n    case \"${{prev}}\" in\n        --effort) COMPREPLY=( $(compgen -W \"fast medium high ultra\" -- \"${{cur}}\") ); return 0 ;;\n        --cloud) COMPREPLY=( $(compgen -W \"prod dev\" -- \"${{cur}}\") ); return 0 ;;\n        --format) COMPREPLY=( $(compgen -W \"auto pretty json\" -- \"${{cur}}\") ); return 0 ;;\n        --suite) COMPREPLY=( $(compgen -W \"ops smoke standard comprehensive\" -- \"${{cur}}\") ); return 0 ;;\n        completions) COMPREPLY=( $(compgen -W \"bash zsh fish powershell\" -- \"${{cur}}\") ); return 0 ;;\n    esac\n    if [[ \"${{cur}}\" == -* ]]; then\n        COMPREPLY=( $(compgen -W \"${{flags}}\" -- \"${{cur}}\") )\n        return 0\n    fi\n    COMPREPLY=( $(compgen -f -- \"${{cur}}\") )\n}}\ncomplete -F _carbon_completions carbon\n",
            cmds = cmds_joined,
            flags = flags_joined,
        )),
        "zsh" => Ok(format!(
            "#compdef carbon\n_carbon() {{\n    local -a subcmds flags\n    subcmds=({cmds})\n    flags=({flags})\n    if (( CURRENT == 2 )); then\n        _describe 'carbon command' subcmds\n        _describe 'carbon flag' flags\n        _files -g '*.csv *.tsv'\n    else\n        _arguments \\\n            '--target=[Target column]:column:' \\\n            '--effort=[Search effort]:effort:(fast medium high ultra)' \\\n            '--cloud=[Public cloud]:cloud:(prod dev)' \\\n            '--format=[Output format]:format:(auto pretty json)' \\\n            '--suite=[Benchmark suite]:suite:(ops smoke standard comprehensive)' \\\n            '--save=[Model handle output]:file:_files' \\\n            '--model=[Saved model handle]:file:_files' \\\n            '--out=[Predictions CSV]:file:_files' \\\n            '*:file:_files'\n        _describe 'carbon flag' flags\n    fi\n}}\n_carbon \"$@\"\n",
            cmds = cmds_joined,
            flags = flags_joined,
        )),
        "fish" => {
            let mut s = String::from("# fish completion for carbon (dynamically generated from clap Cli)\n");
            for cmd in &subcommands {
                s.push_str(&format!(
                    "complete -c carbon -n '__fish_use_subcommand' -a '{cmd}'\n"
                ));
            }
            for fl in &long_flags {
                if let Some(stripped) = fl.strip_prefix("--") {
                    if !matches!(stripped, "effort" | "cloud" | "format" | "suite") {
                        s.push_str(&format!("complete -c carbon -l '{stripped}'\n"));
                    }
                }
            }
            s.push_str("complete -c carbon -l effort -xa 'fast medium high ultra'\n");
            s.push_str("complete -c carbon -l cloud -xa 'prod dev'\n");
            s.push_str("complete -c carbon -l format -xa 'auto pretty json'\n");
            s.push_str("complete -c carbon -l suite -xa 'ops smoke standard comprehensive'\n");
            Ok(s)
        }
        "powershell" | "pwsh" => Ok(format!(
            "# PowerShell completion for carbon (dynamically generated from clap Cli)\nRegister-ArgumentCompleter -Native -CommandName carbon -ScriptBlock {{\n    param($wordToComplete, $commandAst, $cursorPosition)\n    $candidates = @({quoted})\n    $candidates | Where-Object {{ $_ -like \"$wordToComplete*\" }} | ForEach-Object {{\n        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)\n    }}\n}}\n",
            quoted = subcommands
                .iter()
                .chain(long_flags.iter())
                .map(|c| format!("'{c}'"))
                .collect::<Vec<_>>()
                .join(", ")
        )),
        other => Err(format!(
            "unsupported shell `{other}` (expected bash, zsh, fish, or powershell)"
        )),
    }
}

fn collect_dynamic_cli_vocabulary() -> (Vec<String>, Vec<String>) {
    let root = Cli::command();
    let mut subcommands = Vec::new();
    let mut seen_cmds = std::collections::BTreeSet::new();
    let mut long_flags = std::collections::BTreeSet::new();
    long_flags.insert("--help".to_owned());
    long_flags.insert("--version".to_owned());

    let record_args = |cmd: &clap::Command, out: &mut std::collections::BTreeSet<String>| {
        for arg in cmd.get_arguments() {
            if arg.is_hide_set() {
                continue;
            }
            if let Some(long) = arg.get_long() {
                out.insert(format!("--{long}"));
            }
            if let Some(aliases) = arg.get_visible_aliases() {
                for alias in aliases {
                    out.insert(format!("--{alias}"));
                }
            }
        }
    };
    record_args(&root, &mut long_flags);

    for sc in root.get_subcommands() {
        if sc.is_hide_set() || sc.get_name() == "help" {
            continue;
        }
        let primary = sc.get_name().to_owned();
        if seen_cmds.insert(primary.clone()) {
            subcommands.push(primary);
        }
        for alias in sc.get_visible_aliases() {
            let alias_owned = alias.to_owned();
            if seen_cmds.insert(alias_owned.clone()) {
                subcommands.push(alias_owned);
            }
        }
        record_args(sc, &mut long_flags);
    }

    (subcommands, long_flags.into_iter().collect())
}
